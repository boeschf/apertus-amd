"""Sample training memory, with an optional explicitly requested experimental compiler policy."""

import json
import hashlib
from contextlib import contextmanager, ExitStack
from functools import wraps
import importlib
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from types import MappingProxyType

from apertus_amd.artifacts import write_json
from apertus_amd.entrypoint import train
from apertus_amd.config import option


POLICY_NAME = 'single_candidate_backend'
BACKEND_NAME = 'apertus_single_candidate_backend'
POLICY_FLAGS = MappingProxyType({'triton.autotune_pointwise': False,
                'test_configs.force_filter_reduction_configs': True,
                'dynamic_scale_rblock': False})
OPTIONS_ID = hashlib.sha256(json.dumps(dict(POLICY_FLAGS), sort_keys=True).encode()).hexdigest()
POLICY_SCHEMA = 2
MAX_RESTARTS = 100  # Evidence acceptance bound only; never changes Dynamo's own limits.
RESTART_SOURCE = MappingProxyType({
    '_guards.py': '9e6b04f99ba7fd002bf584fc7eb1ec1ba7252bd88e5c19ad394686b908c98678',
    '_dynamo/exc.py': 'adb173498ae5a917c5eb60352d5e22749c833eb844a63c5bbf731dbf154fa174',
    '_dynamo/convert_frame.py': 'b453c9d88aae46cf94198d4a87acc1c58d3e115949879bd4a32f1b516778024f',
    '_dynamo/backends/common.py': '7072831d5463301cf8d743c0364cf01e55583575fe2119f4c28abcded281bc49',
    '_dynamo/output_graph.py': '022797d682404b0fb09628007feb89fcbe3d1e4f62bd74096c0bdb28a15e02d4',
})
POLICY_GUARDS = {'max_autotune': False, 'max_autotune_pointwise': False, 'max_autotune_gemm': False,
                 'deterministic': False, 'batch_invariant': False,
                 'coordinate_descent_tuning': False, 'combo_kernels': False, 'incremental_autotune': False,
                 'profile_bandwidth': False, 'triton.debug_sync_graph': False, 'triton.debug_sync_kernel': False}


def config_values(config, names) -> dict:
    """Read explicitly audited dotted config names without changing compiler state."""
    values = {}
    for name in names:
        value = config
        for part in name.split('.'):
            value = getattr(value, part)
        values[name] = value
    return values


_policy_owner = None
_registered_delegate = None
_setup_lock = threading.Lock()


def policy_guards(config):
    """Check preconditions in the calling compiler context, not just the setup thread."""
    guards = config_values(config, POLICY_GUARDS)
    compile_time = config.triton.autotune_at_compile_time
    if guards != POLICY_GUARDS or compile_time not in (None, False):
        raise ValueError(f'conflicting Inductor preconditions: {guards}, compile_time={compile_time}')
    return guards, compile_time


def single_candidate_backend(graph, example_inputs, **kwargs):
    """Delegate identical graph/inputs to shipped Inductor with exactly three explicit options."""
    owner = _policy_owner
    if owner is None:
        raise ValueError('registered single-candidate backend has no active owner')
    row = {'thread': threading.get_ident(), 'thread_name': threading.current_thread().name,
           'options_id': OPTIONS_ID, 'status': 'entered', 'observation_errors': []}
    with owner.lock:
        row['sequence'] = len(owner.record['compile_calls'])
        owner.record['compile_calls'].append(row)
    forwarded = dict(POLICY_FLAGS)
    try:
        owner.observe(row, '')
        if kwargs:
            raise ValueError(f'backend does not support explicit modes/options: {sorted(kwargs)}')
        if owner.torch.compiler.get_default_backend() != BACKEND_NAME:
            raise ValueError('default backend changed during policy execution')
        policy_guards(owner.config)
        owner.update(row, options_forwarded=dict(POLICY_FLAGS))
        result = owner.inductor.compile(graph, example_inputs, options=forwarded)
        owner.observe(row, '_after')
        guards, compile_time = policy_guards(owner.config)
        owner.update(row, guards_after=guards, compile_time_after=compile_time,
                     options_exit=dict(forwarded), status='returned')
        return result
    except BaseException as error:
        # Observers must never replace the compiler's original exception, including its args.
        with owner.lock:
            row['status'] = 'failed'
            row['options_exit'] = dict(forwarded)
            owner.observe(row, '_after')
            try:
                row['error'] = {'type': type(error).__name__, 'module': type(error).__module__,
                                'qualname': type(error).__qualname__, 'message': str(error)[:2048],
                                'restart_reason': getattr(error, 'restart_reason', None),
                                'traceback': ''.join(traceback.format_tb(error.__traceback__, limit=16))[:8192]}
                reason = row['error']['restart_reason']
                if reason is not None:
                    row['error']['restart_reason'] = str(reason)[:2048]
                exc = importlib.import_module('torch._dynamo.exc')
                if type(error) is exc.TensorifyScalarRestartAnalysis:
                    owner.audit_restart_source()
                    row['exact_tensorify_restart'] = True
                    row['status'] = 'restart'
            except BaseException as secondary:
                row['observation_errors'].append(f'exception observation: {type(secondary).__name__}')
            try:
                owner.publish()
            except BaseException as secondary:
                row['observation_errors'].append(f'exception publication: {type(secondary).__name__}')
        raise


class BackendPolicy:
    """Own default selection and rank-local evidence; registration itself is process-lived."""

    def __init__(self, torch, config, inductor, record, on_update):
        self.torch, self.config, self.inductor = torch, config, inductor
        self.record, self.on_update = record, on_update
        self.lock = threading.RLock()
        self.closed = False
        # Retain references until finish: Python id()/thread identifiers can otherwise be reused.
        self.contexts, self.threads = [], []

    def token(self, objects, value):
        """Allocate stable owner-local identity without relying on object equality or id reuse."""
        with self.lock:
            for index, item in enumerate(objects):
                if item is value:
                    return index
            objects.append(value)
            return len(objects) - 1

    def compile_identity(self):
        """Read the pinned thread-local CompileContext/TraceId APIs without creating a context."""
        guards = importlib.import_module('torch._guards')
        context = guards.CompileContext.try_get()
        trace = guards.CompileContext.current_trace_id()
        if context is None or trace is None:
            return None
        if trace.compile_id is not context.compile_id or trace.attempt != context.attempt:
            raise ValueError('inconsistent CompileContext/TraceId')
        compile_id = trace.compile_id
        return {'frame_id': compile_id.frame_id, 'frame_compile_id': compile_id.frame_compile_id,
                'compiled_autograd_id': compile_id.compiled_autograd_id,
                'compile_id': str(compile_id), 'attempt': trace.attempt,
                'context': self.token(self.contexts, context),
                'thread_owner': self.token(self.threads, threading.current_thread())}

    def observe(self, row, suffix):
        """Retain exit evidence even on exceptions; incomplete observation always fails checking."""
        with self.lock:
            self.observe_locked(row, suffix)

    def observe_locked(self, row, suffix):
        """Update a complete snapshot while other callback publishers cannot traverse the row."""
        try:
            row.update({'guards' + suffix: config_values(self.config, POLICY_GUARDS),
                        'compile_time' + suffix: self.config.triton.autotune_at_compile_time,
                        'ambient_after' if suffix else 'ambient_before': config_values(self.config, POLICY_FLAGS),
                        'default' + suffix: self.torch.compiler.get_default_backend(),
                        'owned' + suffix: _policy_owner is self and not self.closed})
            try:
                row['identity' + suffix] = self.compile_identity()
            except (ImportError, KeyError):
                # Normal CPU analogs/lazy callbacks may have no Dynamo context; never accept a restart without it.
                row['identity' + suffix] = None
        except BaseException as error:
            row['observation_errors'].append(f'context observation: {type(error).__name__}')

    def audit_restart_source(self):
        """Authorize restart classification only for the retained, byte-identical source/API audit."""
        if self.record.get('restart_source') == dict(RESTART_SOURCE):
            return
        guards = importlib.import_module('torch._guards')
        root = Path(guards.__file__).parent
        actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RESTART_SOURCE}
        if actual != RESTART_SOURCE:
            raise ValueError('unaudited Tensorify restart source')
        self.record['restart_source'] = actual

    def publish(self):
        """Serialize evidence writes without serializing or initiating compiler work."""
        with self.lock:
            if self.on_update is not None:
                self.on_update(json.loads(json.dumps(self.record)))

    def update(self, row, **values):
        """Publish a whole callback transition under the evidence lock."""
        with self.lock:
            row.update(values)
            self.publish()

    def finish(self):
        """Record setup-context end state, then restore only the default this owner selected."""
        global _policy_owner
        if self.closed:
            raise ValueError('backend policy already closed')
        self.record.update(default_end=self.torch.compiler.get_default_backend(),
                           guards_end=config_values(self.config, POLICY_GUARDS),
                           compile_time_end=self.config.triton.autotune_at_compile_time,
                           setup_ambient_end=config_values(self.config, POLICY_FLAGS))
        try:
            if self.record['default_end'] != BACKEND_NAME:
                raise ValueError('cannot restore unowned default backend override')
            self.torch.compiler.set_default_backend(self.record['default_before'])
            self.record['restored_default'] = self.torch.compiler.get_default_backend()
        finally:
            self.closed = True
            _policy_owner = None
            self.publish()
        return self.record


def configure_inductor_policy(torch, on_update=None):
    """Serialize registration/selection; an active owner rejects overlapping setup."""
    with _setup_lock:
        return _configure_inductor_policy(torch, on_update)


def _configure_inductor_policy(torch, on_update):
    """Register/select the distinct option-carrying backend before any native decorations."""
    global _policy_owner, _registered_delegate
    requested = os.environ.get('APERTUS_INDUCTOR_POLICY')
    if requested is None:
        return None, None
    if requested == 'single_candidate':
        raise ValueError('unsafe ambient single_candidate policy is rejected; use single_candidate_backend')
    if requested != POLICY_NAME:
        raise ValueError(f'unknown APERTUS_INDUCTOR_POLICY: {requested!r}')
    if _policy_owner is not None:
        raise ValueError('backend policy already active')
    if any(name == 'megatron.core.jit' or name.startswith('megatron.core.fusions.fused_bias_') for name in tuple(sys.modules)):
        raise ValueError('Inductor policy must precede native JIT imports')
    if os.environ.get('TORCH_COMPILE_DISABLE', '0') != '0' or os.environ.get('TORCHDYNAMO_DISABLE') not in (None, '0'):
        raise ValueError('single_candidate_backend requires native torch.compile enabled')
    config = importlib.import_module('torch._inductor.config')
    guards, compile_time = policy_guards(config)
    dynamo = importlib.import_module('torch._dynamo')
    inductor = importlib.import_module('torch._inductor')
    if torch.compiler.get_default_backend() != 'inductor':
        raise ValueError('pre-existing default backend override')
    if BACKEND_NAME in dynamo.list_backends(exclude_tags=()):
        if _registered_delegate is not single_candidate_backend or dynamo.lookup_backend(BACKEND_NAME) is not single_candidate_backend:
            raise ValueError('backend registration name collision')
    else:
        dynamo.register_backend(single_candidate_backend, name=BACKEND_NAME)
        _registered_delegate = single_candidate_backend
    if dynamo.lookup_backend(BACKEND_NAME) is not single_candidate_backend:
        raise ValueError('backend registration identity mismatch')
    record = {'schema_version': POLICY_SCHEMA, 'requested': requested, 'torch': torch.__version__, 'backend': BACKEND_NAME,
              'options': dict(POLICY_FLAGS), 'options_id': OPTIONS_ID, 'default_before': 'inductor',
              'guards_start': guards, 'compile_time_start': compile_time,
              'setup_ambient_start': config_values(config, POLICY_FLAGS), 'compile_calls': []}
    owner = BackendPolicy(torch, config, inductor, record, on_update)
    _policy_owner = owner
    try:
        torch.compiler.set_default_backend(BACKEND_NAME)
        record['default_start'] = torch.compiler.get_default_backend()
        if record['default_start'] != BACKEND_NAME:
            raise ValueError('default backend selection failed')
        owner.publish()
    except BaseException:
        if torch.compiler.get_default_backend() == BACKEND_NAME:
            owner.finish()
        else:
            owner.closed = True
            _policy_owner = None
        raise
    return owner, record


def finish_inductor_policy(owner, record: dict) -> dict:
    """Finish only this execution's selection after native work has returned."""
    return owner.finish()


def check_inductor_policy(record: dict, requested: str, torch_version: str) -> None:
    """Reject requested/effective policy drift; this is not numerical validation."""
    if requested != POLICY_NAME or record.get('requested') != requested or record.get('torch') != torch_version:
        raise ValueError('requested/effective Inductor policy or Torch version mismatch')
    for phase in ('start', 'end'):
        if record.get(f'default_{phase}') != BACKEND_NAME or record.get(f'guards_{phase}') != POLICY_GUARDS or \
                record.get(f'compile_time_{phase}', 'missing') not in (None, False):
            raise ValueError(f'effective Inductor policy drift at {phase}')
    if record.get('backend') != BACKEND_NAME or record.get('options') != POLICY_FLAGS or record.get('options_id') != OPTIONS_ID or \
            record.get('default_before') != 'inductor' or record.get('restored_default') != 'inductor':
        raise ValueError('backend policy identity or restoration mismatch')
    if not record.get('compile_calls'):
        raise ValueError('backend policy was never invoked')
    schema = record.get('schema_version', 1)
    if schema not in (1, POLICY_SCHEMA):
        raise ValueError('unsupported backend policy schema')
    pending, counts, last_restart = {}, {}, {}
    for sequence, call in enumerate(record['compile_calls']):
        status = call.get('status')
        if (status != 'returned' and not (schema == POLICY_SCHEMA and status == 'restart')) or \
                call.get('options_forwarded') != POLICY_FLAGS or call.get('options_id') != OPTIONS_ID:
            raise ValueError('backend options delegation failed')
        for suffix in ('', '_after'):
            if call.get('guards' + suffix) != POLICY_GUARDS or call.get('compile_time' + suffix, 'missing') not in (None, False):
                raise ValueError('backend invoking-context guard drift')
        if schema == 1:
            continue  # Legacy *normal* records remain valid; failed/name-only restarts never do.
        if call.get('sequence') != sequence or call.get('observation_errors') != [] or \
                call.get('options_exit') != POLICY_FLAGS or 'ambient_before' not in call or \
                call['ambient_before'] != call.get('ambient_after'):
            raise ValueError('backend observation or options drift')
        for suffix in ('', '_after'):
            if call.get('default' + suffix) != BACKEND_NAME or call.get('owned' + suffix) is not True:
                raise ValueError('backend callback ownership drift')
        if 'identity' not in call or call['identity'] != call.get('identity_after', 'missing'):
            raise ValueError('backend compilation identity drift')
        identity = call['identity']
        key = None
        if identity is not None:
            key = compilation_key(identity, call.get('thread'))
        if status == 'restart':
            error = call.get('error', {})
            if record.get('restart_source') != RESTART_SOURCE or call.get('exact_tensorify_restart') is not True or \
                    error.get('module') != 'torch._dynamo.exc' or error.get('qualname') != 'TensorifyScalarRestartAnalysis' or \
                    error.get('type') != 'TensorifyScalarRestartAnalysis' or \
                    not isinstance(error.get('traceback'), str) or not 0 < len(error['traceback']) <= 8192 or \
                    'restart_reason' not in error or \
                    (error['restart_reason'] is not None and (not isinstance(error['restart_reason'], str) or
                                                              len(error['restart_reason']) > 2048)):
                raise ValueError('unaudited or incomplete exact Tensorify restart')
            if key is None:
                raise ValueError('restart missing Dynamo compilation identity')
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > MAX_RESTARTS or identity['attempt'] > MAX_RESTARTS:
                raise ValueError('excessive Tensorify restarts')
            if key in last_restart and identity['attempt'] <= last_restart[key]:
                raise ValueError('restart attempt did not increase')
            last_restart[key] = identity['attempt']
            pending[key] = identity['attempt']
        elif key in pending:
            if identity['attempt'] <= pending[key]:
                raise ValueError('recovery attempt did not increase')
            del pending[key]
    if pending:
        raise ValueError('unrecovered Tensorify restart')


def compilation_key(identity: dict, thread) -> tuple:
    """Validate the pinned CompileId wire schema and include owner-local context/thread identity."""
    names = ('frame_id', 'frame_compile_id', 'compiled_autograd_id', 'compile_id', 'attempt', 'context', 'thread_owner')
    if not isinstance(identity, dict) or set(identity) != set(names):
        raise ValueError('incomplete Dynamo compilation identity')
    for name in ('frame_id', 'frame_compile_id', 'attempt', 'context', 'thread_owner'):
        if type(identity[name]) is not int or identity[name] < 0:
            raise ValueError('invalid Dynamo compilation identity')
    autograd = identity['compiled_autograd_id']
    if autograd is not None and (type(autograd) is not int or autograd < 0):
        raise ValueError('invalid compiled autograd identity')
    expected = f"{identity['frame_id']}/{identity['frame_compile_id']}"
    if autograd is not None:
        expected = f'!{autograd}/' + expected
    if identity['compile_id'] != expected or type(thread) is not int:
        raise ValueError('inconsistent Dynamo compile ID')
    return (expected, identity['context'], identity['thread_owner'], thread)


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
        value['buddyinfo'] = read_text(Path('/proc/buddyinfo'))
        value['base_page_bytes'] = os.sysconf('SC_PAGE_SIZE')
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
    """Run verified native training with passive telemetry and an opt-in compiler policy."""
    import torch

    rank, device = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    groups = cgroup_paths()
    variables = ('PYTORCH_HIP_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
                 'TORCH_COMPILE_DISABLE', 'TORCHDYNAMO_DISABLE', 'MAX_JOBS', 'OMP_NUM_THREADS',
                  'TMPDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
                   'APERTUS_COMPILE_TRACE', 'TORCH_LOGS', 'APERTUS_INDUCTOR_POLICY')
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
    policy_config, policy_record = None, None
    try:
        with ExitStack() as observers:
            if os.environ.get('APERTUS_COMPILE_TRACE') == '1' or 'APERTUS_INDUCTOR_POLICY' in os.environ:
                def setup():
                    nonlocal policy_config, policy_record
                    # Configure before ANY native training import creates lazy JIT wrappers.
                    policy_config, policy_record = configure_inductor_policy(
                        torch, on_update=lambda value: write_json(output / f'inductor-policy-{rank}.json', value))
                    if policy_record is not None:
                        write_json(output / f'inductor-policy-{rank}.json', policy_record)
                    if os.environ.get('APERTUS_COMPILE_TRACE') == '1':
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
        if policy_record is not None:
            write_json(output / f'inductor-policy-{rank}.json', finish_inductor_policy(policy_config, policy_record))
        write_json(output / f'memory-end-{rank}.json', {
            'training_completed': completed, 'sampler_errors': failures,
            'final': snapshot(torch, device, groups, include_cache=False)})
    if failures:
        raise RuntimeError(f'memory telemetry failed: {failures}')


def check(run: Path) -> None:
    """Check telemetry completeness only; do not imply numerical or checkpoint validation."""
    settings = json.loads((run / 'resolved.json').read_text())
    requested = settings['environment'].get('APERTUS_INDUCTOR_POLICY')
    if requested == 'single_candidate':
        raise ValueError('unsafe ambient single_candidate policy is rejected')
    if requested not in (None, POLICY_NAME):
        raise ValueError(f'unknown requested Inductor policy: {requested!r}')
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
        if requested is not None:
            if start['environment'].get('APERTUS_INDUCTOR_POLICY') != requested:
                raise ValueError(f'Inductor policy environment mismatch on rank {rank}')
            policy = json.loads((output / f'inductor-policy-{rank}.json').read_text())
            check_inductor_policy(policy, requested, start['torch'])
        elif start['environment'].get('APERTUS_INDUCTOR_POLICY') is not None:
            raise ValueError(f'unrequested Inductor policy on rank {rank}')
        if settings['environment'].get('APERTUS_COMPILE_TRACE') == '1':
            rows = [json.loads(line) for line in (output / f'compilation-{rank}.jsonl').read_text().splitlines()]
            steps = int(option(settings['args'], '--train-iters'))
            if [r['status'] for r in rows] != ['start', *(['step'] * steps), 'end'] or \
                    [r['completed_train_steps'] for r in rows] != [0, *range(1, steps + 1), steps]:
                raise ValueError(f'incomplete compilation telemetry for rank {rank}')
        records.append({'rank': rank, 'samples': samples, 'hostname': start['hostname']})
    result = {
        'status': 'passed', 'scope': 'memory telemetry completeness and explicit settings',
        'numerical_validation': 'not requested', 'checkpoint_validation': 'not requested',
        'ranks': records}
    if requested is not None:
        result['inductor_policy'] = requested
    write_json(run / 'output/checks.json', result)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        trace_training(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif sys.argv[1] == 'check':
        check(Path(sys.argv[2]))
    else:
        raise ValueError(f'unknown operation: {sys.argv[1]}')
