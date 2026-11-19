"""Run pinned UCCL library tests with their original numerical assertions.

Slurm supplies one process per GPU; only rendezvous and bounded test selection
are adapted. No Megatron comparison, tolerance change or library build occurs.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace

IMAGE = '910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490'
COMMIT = '52d4184c89deb5631be6dac6a9b1c40896c9e26e'
PHASES = ('intranode', 'internode', 'low_latency', 'dual_mode')
KEEPALIVE = []


def digest(path):
    """Hash a frozen source file, never an image or private cache."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    """Atomically publish new coordination/evidence without replacing prior work."""
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        primary = sys.exception()
        try:
            temporary.unlink()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f'temporary evidence cleanup failed: {error}')


def wait_json(paths, timeout=60):
    """Bound filesystem agreement; rank failure never authorizes an automatic retry."""
    deadline = time.monotonic() + timeout
    while not all(path.is_file() for path in paths):
        if time.monotonic() >= deadline:
            raise TimeoutError('upstream-test rendezvous')
        time.sleep(.1)
    return [json.loads(path.read_text()) for path in paths]


def validate_launch(settings, env, phase):
    """Admit exactly the two-node library-test experiment and its fixed schedule."""
    resources, procedure = settings['resources'], settings['procedure']
    expected = dict(nodes=2, gpus_per_node=4, cpus_per_node=192, mpi='pmix',
                    time='00:30:00', reservation='cscs', distribution='block:block',
                    cpu_bind='map_ldom:0,1,2,3')
    if (settings['image']['sha256'] != IMAGE or any(resources.get(k) != v for k, v in expected.items())
            or resources.get('cpus_per_task') is not None or procedure['kind'] != 'commands'
            or procedure['timeout'] != 360 or procedure['commands'] != [
                ['python3', '-u', '/apertus/validation.py', p] for p in PHASES] or phase not in PHASES):
        raise ValueError('library test launch contract')
    rank, local, node, world = (int(env[k]) for k in ('SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NODEID', 'SLURM_NTASKS'))
    if world != 8 or not 0 <= node < 2 or not 0 <= local < 4 or rank != 4*node+local:
        raise ValueError('requires eight block-placed ranks')
    if len(os.sched_getaffinity(0)) != 48:
        raise ValueError('requires allocation-derived 48 CPUs/rank')
    for key in ('UCCL_EP_DISPATCH_CONFIG', 'UCCL_EP_COMBINE_CONFIG'):
        if key in env:
            raise ValueError('library tests must use their own explicit Config values')
    if any(env.get(k) != v for k, v in dict(OMP_NUM_THREADS='1', MAX_JOBS='2',
                                          NCCL_P2P_DISABLE='0', HWLOC_COMPONENTS='-gl').items()):
        raise ValueError('library test environment differs')
    return rank, local, node


def sources(run, settings):
    """Verify immutable copied benchmark sources before adding their import path."""
    manifest = run/'upstream-manifest.json'
    expected = settings['environment']['APERTUS_UCCL_TEST_SOURCE_SHA256']
    if manifest.is_symlink() or digest(manifest) != expected:
        raise ValueError('upstream manifest identity')
    metadata = json.loads(manifest.read_text())
    if metadata['commit'] != COMMIT:
        raise ValueError('upstream commit differs from installed image selection')
    directory = run/'upstream'
    for name, checksum in metadata['files'].items():
        path = directory/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or digest(path) != checksum:
            raise ValueError(f'upstream file identity: {name}')
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*.py')}
    if actual != {name for name in metadata['files'] if name.endswith('.py')}:
        raise ValueError('unexpected/missing benchmark Python source')
    return directory/'ep/bench', metadata


def rendezvous(output, phase, rank, local, node):
    """Provide torchrun-compatible variables without spawning nested GPU workers."""
    scope = f'node-{node}' if phase == 'intranode' else 'world'
    path = output/f'rdzv-{scope}.json'
    leader = local == 0 if phase == 'intranode' else rank == 0
    if leader:
        with socket.socket() as listener:
            listener.bind(('', 0))
            port = listener.getsockname()[1]
        write_new(path, dict(job_id=os.environ['SLURM_JOB_ID'], address=socket.gethostname(), port=port))
    value = wait_json([path])[0]
    if value['job_id'] != os.environ['SLURM_JOB_ID']:
        raise ValueError('stale rendezvous')
    env = dict(MASTER_ADDR=value['address'], MASTER_PORT=str(value['port']), LOCAL_RANK=str(local),
               LOCAL_WORLD_SIZE='4', RANK=str(local if phase == 'intranode' else rank),
               WORLD_SIZE='4' if phase == 'intranode' else '8')
    os.environ.update(env)
    return env


def barrier_ownership(run, output, phase, rank):
    """Refuse preexisting EP barrier files; upstream may remove its own mode files.

    All eight first-step checks finish before either local group starts UCCL.
    Later steps may see files created by this exclusive allocation, never files
    present before its first step. No manual shared-memory cleanup is performed.
    """
    paths = list(Path('/dev/shm').glob('uccl_barrier_*'))
    if any(path.stat().st_uid != os.getuid() for path in paths):
        raise ValueError('foreign EP shared-memory object; refusing library initialization')
    first = run/'output/0'
    if phase == 'intranode':
        write_new(first/f'initial-ep-shm-{rank}.json', dict(rank=rank, job_id=os.environ['SLURM_JOB_ID'],
                  node=socket.gethostname(), objects=[path.name for path in paths]))
    records = wait_json([first/f'initial-ep-shm-{r}.json' for r in range(8)])
    if any(row['job_id'] != os.environ['SLURM_JOB_ID'] or row['rank'] != r or row['objects'] for r, row in enumerate(records)):
        raise ValueError('preexisting EP shared memory; no existing barrier may be deleted')
    return [path.name for path in paths]


def invoke(phase, module, utils, buffer_class, rank, local):
    """Execute unchanged upstream test functions; skip only supported benchmark tails."""
    args = SimpleNamespace(num_processes=4, num_tokens=4096, hidden=7168,
                           num_topk=8, num_experts=256, num_topk_groups=2,
                           pressure_test_mode=0, test_ll_compatibility=False)
    if phase == 'intranode':
        module.test_loop(local, 4, args)
        return dict(entrypoint='test_intranode.test_loop', benchmark='original intranode timing/tuning tail retained')
    if phase == 'dual_mode':
        module.main(local, 4, SimpleNamespace(num_tokens=8, hidden=1024, num_experts=16))
        return dict(entrypoint='test_dual_mode.main', construction_only=True)
    import torch
    rank_seen, world, group = utils.init_dist_under_torchrun(local, 4)
    if rank_seen != rank or world != 8:
        raise ValueError('upstream group differs from allocated rank layout')
    if phase == 'internode':
        nvl, rdma = module.compute_buffer_sizes(64, args.hidden, world)
        buffer = buffer_class(group, nvl, rdma, low_latency_mode=False,
                              num_qps_per_rank=64, explicitly_destroy=True)
    else:
        args = SimpleNamespace(num_tokens=128, hidden=7168, num_experts=288, num_topk=8)
        rdma = buffer_class.get_low_latency_rdma_size_hint(args.num_tokens, args.hidden, world, args.num_experts)
        buffer = buffer_class(group, num_rdma_bytes=rdma, low_latency_mode=True,
                              num_qps_per_rank=args.num_experts//world,
                              allow_nvlink_for_low_latency_mode=True, explicitly_destroy=True,
                              allow_mnnvl=False)
    # On assertion/native failure, preserve Buffer ownership until process exit;
    # do not enter new collective teardown while another rank may have failed.
    KEEPALIVE.append(buffer)
    torch.manual_seed(rank)
    if phase == 'internode':
        value = module.test_main(args, 64, local, 4, world, 2, rank, buffer, group, skip_benchmark=True)
    else:
        value = module.test_main(args.num_tokens, args.hidden, args.num_experts, args.num_topk,
            rank, world, group, buffer, use_logfmt=False, dispatch_use_fp8=True,
            seed=0, skip_benchmark=True, debug_hash=True)
    torch.distributed.barrier()
    buffer.destroy()
    KEEPALIVE.remove(buffer)
    torch.distributed.destroy_process_group()
    return dict(entrypoint=f'test_{phase}.test_main', skip_benchmark=True, seed=0,
                upstream_result=value, buffer_sizes=dict(nvl=0 if phase == 'low_latency' else nvl, rdma=rdma))


def run(phase, settings, root=Path('/apertus')):
    """Retain source/device/environment evidence and propagate upstream assertions."""
    rank, local, node = validate_launch(settings, os.environ, phase)
    output = Path.cwd(); target = output/f'uccl-library-rank-{rank}.json'
    record = dict(status='failed', rank=rank, local_rank=local, node=node, phase=phase,
                  image=IMAGE, commit=COMMIT, pid=os.getpid(), pgid=os.getpgrp(),
                  started_ns=time.monotonic_ns(), numerical_assertions='unchanged upstream')
    try:
        bench, metadata = sources(root, settings)
        record['source_manifest_sha256'] = digest(root/'upstream-manifest.json')
        record['files'] = metadata['files']
        record['barriers_before'] = barrier_ownership(root, output, phase, rank)
        record['rendezvous'] = rendezvous(output, phase, rank, local, node)
        from apertus_amd.config import CACHE_PATHS
        parent = Path(settings['scratch_root']); parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f'uccl-library-{os.environ["SLURM_JOB_ID"]}-{rank}-{phase}-', dir=parent))
        record['scratch'] = str(scratch)
        for key, relative in CACHE_PATHS.items():
            path = scratch/relative; path.mkdir(parents=True, exist_ok=True); os.environ[key] = str(path)
        write_new(output/f'uccl-library-start-{rank}.json', record)
        from apertus_amd.runtime import preserve_affinity, require_affinity
        with preserve_affinity(record.setdefault('affinity', {})) as affinity:
            import torch
            if torch.cuda.device_count() != 4 or not torch.version.hip:
                raise ValueError('all four ROCm devices must remain visible')
            torch.cuda.set_device(local)
            torch.set_num_threads(1)
            sys.path.insert(0, str(bench))
            module = importlib.import_module(f'test_{phase}')
            utils = importlib.import_module('utils')
            buffer_module = importlib.import_module('buffer')
            import uccl.ep as ep
            installed_buffer = importlib.import_module('deep_ep.buffer')
        record['selected_device'] = torch.cuda.current_device()
        if record['selected_device'] != local:
            raise ValueError('upstream import changed local GPU')
        record['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'uccl', 'deep_ep')}
        if record['versions'] != {'torch':'2.14.0+rocm10.1.0', 'uccl':'0.1.1', 'deep_ep':'0.1.0'}:
            raise ValueError('installed package versions differ')
        record['max_nvl_peers'] = ep.num_max_nvl_peers()
        if record['max_nvl_peers'] != 4:
            raise ValueError('installed topology does not support four-GPU nodes')
        record['origins'] = {}
        for name, imported in tuple(sys.modules.items()):
            if name in ('utils','buffer',*[f'test_{p}' for p in PHASES]):
                path = Path(imported.__file__).resolve()
                if not path.is_relative_to(bench) or digest(path) != metadata['files'][str(path.relative_to(root/'upstream'))]:
                    raise ValueError('executed benchmark import differs')
                record['origins'][name] = str(path)
        record['native_module'] = dict(path=ep.__file__,sha256=digest(ep.__file__))
        record['installed_buffer'] = dict(path=installed_buffer.__file__,sha256=digest(installed_buffer.__file__))
        if record['installed_buffer']['sha256'] != metadata['files']['ep/bench/buffer.py']:
            raise ValueError('benchmark wrapper differs from deployed image wrapper')
        require_affinity(affinity, 'before upstream EP test')
        write_new(output/f'uccl-library-ready-{rank}.json', record)
        record['invocation'] = invoke(phase, module, utils, buffer_module.Buffer, rank, local)
        require_affinity(affinity, 'after upstream EP test')
        sources(root, settings)
        record['status'] = 'passed'
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['finished_ns'] = time.monotonic_ns()
        primary = sys.exception()
        try:
            write_new(target, record)
        except BaseException as evidence_error:
            if primary is None:
                raise
            primary.add_note(f'evidence failure: {evidence_error}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=PHASES)
    arguments = parser.parse_args()
    run(arguments.phase, json.loads(Path('/apertus/resolved.json').read_text()))
