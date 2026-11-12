"""Sample training memory without changing kernels or synchronizing GPUs."""

import json
from contextlib import contextmanager, ExitStack
from functools import wraps
import importlib
import os
from pathlib import Path
import socket
import sys
import threading
import time

from apertus_amd.artifacts import write_json
from apertus_amd.entrypoint import train
from apertus_amd.config import option


def compilation_snapshot() -> dict:
    """Read version-specific compiler counters without resetting them or synchronizing GPUs."""
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    return {'counters': {str(group): {str(key): value for key, value in items.items()}
                         for group, items in counters.items()},
            'generated_kernel_count': metrics.generated_kernel_count}


@contextmanager
def trace_compilation_steps(module, output: Path, rank: int):
    """Observe completed native train_step calls and restore the entry point on every exit."""
    original = module.train_step
    completed = 0
    with (output / f'compilation-{rank}.jsonl').open('w', buffering=1) as stream:
        def record(status):
            stream.write(json.dumps({'time': time.time(), 'completed_train_steps': completed,
                                     'status': status, **compilation_snapshot()}) + '\n')

        @wraps(original)
        def step(*args, **kwargs):
            nonlocal completed
            result = original(*args, **kwargs)
            completed += 1
            record('step')
            return result

        record('start')
        module.train_step = step
        try:
            yield
        finally:
            module.train_step = original
            record('end')


def read_text(path: Path) -> str:
    """Retain unavailable kernel counters as diagnostic errors rather than aborting training."""
    try:
        return path.read_text()
    except OSError as error:
        return f"unavailable: {error}"


def cgroup_paths() -> list[Path]:
    """Find visible process memory cgroups, retaining the mount root for namespaced mounts."""
    paths = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers and 'memory' not in controllers.split(','):
            continue
        root = Path('/sys/fs/cgroup/memory' if controllers else '/sys/fs/cgroup')
        candidate = root / relative.lstrip('/')
        if '..' not in candidate.parts and candidate.is_dir():
            paths.append(candidate)
        if root.is_dir():
            paths.append(root)
    return list(dict.fromkeys(paths))


def cache_usage(root: Path) -> dict:
    """Count this rank's private cache files without following directory symlinks."""
    result = {'files': 0, 'logical_bytes': 0, 'allocated_bytes': 0}
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            try:
                stat = (Path(directory) / name).lstat()
            except FileNotFoundError:
                continue  # Compiler temporary files may disappear during the walk.
            result['files'] += 1
            result['logical_bytes'] += stat.st_size
            result['allocated_bytes'] += stat.st_blocks * 512
    return result


def snapshot(torch, device: int, groups: list[Path], *, include_cache: bool) -> dict:
    """Read host and allocator counters without initializing CUDA or resetting peaks."""
    value = {'time': time.time(), 'process_status': read_text(Path('/proc/self/status')),
             'process_memory': read_text(Path('/proc/self/smaps_rollup')),
             'cgroups': {}}
    for group in groups:
        names = ('memory.current', 'memory.peak', 'memory.max', 'memory.events', 'memory.stat',
                 'memory.usage_in_bytes', 'memory.max_usage_in_bytes', 'memory.limit_in_bytes',
                 'memory.failcnt', 'memory.oom_control')
        value['cgroups'][str(group)] = {name: read_text(group / name) for name in names
                                      if (group / name).exists()}
    if torch.cuda.is_initialized():
        stats = torch.cuda.memory_stats(device)
        value['gpu'] = {key: item for key, item in stats.items() if key.startswith(
            ('allocated_bytes.', 'reserved_bytes.', 'active_bytes.', 'inactive_split_bytes.',
             'num_alloc_retries', 'num_ooms'))}
    if device == 0:
        value['node_meminfo'] = read_text(Path('/proc/meminfo'))
        value['node_pressure'] = read_text(Path('/proc/pressure/memory'))
        value['numa_meminfo'] = {str(p): read_text(p) for p in
                                 Path('/sys/devices/system/node').glob('node*/meminfo')}
        stat = os.statvfs('/dev/shm')
        value['shm'] = {'total_bytes': stat.f_blocks * stat.f_frsize,
                        'used_bytes': (stat.f_blocks - stat.f_bfree) * stat.f_frsize}
    if include_cache:
        value['private_cache'] = cache_usage(Path(os.environ['TMPDIR']).parent)
    return value


def trace_training(output: Path, source: Path, entrypoint: str, args: list[str]) -> None:
    """Run the ordinary verified entrypoint while one rank-owned thread samples memory."""
    import torch

    rank, device = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    groups = cgroup_paths()
    variables = ('PYTORCH_HIP_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
                 'TORCH_COMPILE_DISABLE', 'TORCHDYNAMO_DISABLE', 'MAX_JOBS', 'OMP_NUM_THREADS',
                  'TMPDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
                  'APERTUS_COMPILE_TRACE', 'TORCH_LOGS')
    write_json(output / f'memory-start-{rank}.json', {
        'rank': rank, 'device': device, 'hostname': socket.gethostname(), 'pid': os.getpid(),
        'environment': {key: os.environ.get(key) for key in variables},
        'cgroup_membership': read_text(Path('/proc/self/cgroup')),
        'cgroup_paths': [str(path) for path in groups], 'torch': torch.__version__,
        'sample_seconds': 10, 'cache_sample_seconds': 60})
    stop = threading.Event()
    failures = []

    def sample():
        try:
            with (output / f'memory-{rank}.jsonl').open('w', buffering=1) as stream:
                next_cache = 0
                while True:
                    now = time.monotonic()
                    value = snapshot(torch, device, groups, include_cache=now >= next_cache)
                    if now >= next_cache:
                        next_cache = now + 60
                    stream.write(json.dumps(value) + '\n')
                    if stop.wait(10):
                        break
        except Exception as error:
            failures.append(repr(error))

    # The sampler never selects a device, synchronizes GPU work, or owns cache cleanup.
    worker = threading.Thread(target=sample, name='memory-trace', daemon=True)
    worker.start()
    completed = False
    try:
        with ExitStack() as observers:
            if os.environ.get('APERTUS_COMPILE_TRACE') == '1':
                def setup():
                    # train() installs the frozen source path before this import and
                    # verifies its identity; no training kernels or counters are replaced.
                    module = importlib.import_module('megatron.training.training')
                    observers.enter_context(trace_compilation_steps(module, output, rank))
                train(output, source, entrypoint, args, before_run=setup)
            else:
                train(output, source, entrypoint, args)
        completed = True
    finally:
        stop.set()
        worker.join(timeout=30)
        if worker.is_alive():
            failures.append('memory sampler did not stop within 30 seconds')
        write_json(output / f'memory-end-{rank}.json', {
            'training_completed': completed, 'sampler_errors': failures,
            'final': snapshot(torch, device, groups, include_cache=False)})
    if failures:
        raise RuntimeError(f'memory telemetry failed: {failures}')


def check(run: Path) -> None:
    """Check telemetry completeness only; do not imply numerical or checkpoint validation."""
    settings = json.loads((run / 'resolved.json').read_text())
    world = settings['resources']['nodes'] * settings['resources']['gpus_per_node']
    records = []
    for rank in range(world):
        output = run / 'output'
        start = json.loads((output / f'memory-start-{rank}.json').read_text())
        end = json.loads((output / f'memory-end-{rank}.json').read_text())
        if not end['training_completed'] or end['sampler_errors'] or not end['final'].get('gpu'):
            raise ValueError(f'incomplete memory telemetry for rank {rank}')
        samples = 0
        with (output / f'memory-{rank}.jsonl').open() as stream:
            for line in stream:
                json.loads(line)
                samples += 1
        if not samples:
            raise ValueError(f'no memory samples for rank {rank}')
        for key in ('PYTORCH_HIP_ALLOC_CONF', 'TORCH_COMPILE_DISABLE'):
            if start['environment'][key] != settings['environment'][key]:
                raise ValueError(f'unexpected effective {key} on rank {rank}')
        if start['environment']['PYTORCH_CUDA_ALLOC_CONF'] is not None or \
                start['environment']['TORCHDYNAMO_DISABLE'] not in (None, '0'):
            raise ValueError(f'confounding allocator/compiler override on rank {rank}')
        if settings['environment'].get('APERTUS_COMPILE_TRACE') == '1':
            rows = [json.loads(line) for line in (output / f'compilation-{rank}.jsonl').read_text().splitlines()]
            steps = int(option(settings['args'], '--train-iters'))
            if [r['status'] for r in rows] != ['start', *(['step'] * steps), 'end'] or \
                    [r['completed_train_steps'] for r in rows] != [0, *range(1, steps + 1), steps]:
                raise ValueError(f'incomplete compilation telemetry for rank {rank}')
        records.append({'rank': rank, 'samples': samples, 'hostname': start['hostname']})
    write_json(run / 'output/checks.json', {
        'status': 'passed', 'scope': 'memory telemetry completeness and explicit settings',
        'numerical_validation': 'not requested', 'checkpoint_validation': 'not requested',
        'ranks': records})


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        trace_training(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif sys.argv[1] == 'check':
        check(Path(sys.argv[2]))
    else:
        raise ValueError(f'unknown operation: {sys.argv[1]}')
