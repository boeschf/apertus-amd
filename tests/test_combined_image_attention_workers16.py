"""CPU-only strict variant/ownership-gate contracts for the single workers16 proposal."""

import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from apertus_amd.cli import plan
from apertus_amd.config import resolve
from apertus_amd.runs import edf


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'experiments/validation/combined_image_attention_parallel.py'
NEW = ROOT / 'experiments/validation/combined_image_attention_workers16.py'
EXPERIMENT = ROOT / 'experiments/combined-aiter-attention-workers16-i910bfcf936d3'
PREP = ROOT / 'artifacts/investigations/performance-next/restart-20261113/attention-workers16-preparation'


def load(path):
    spec = importlib.util.spec_from_file_location('workers16_cpu_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings():
    return resolve(EXPERIMENT, site=ROOT / 'config.toml', submit=True, reservation='cscs', time='00:15:00')


def environment(rank=0):
    return {'SLURM_PROCID': str(rank), 'SLURM_LOCALID': str(rank), 'SLURM_NODEID': '0',
            'SLURM_NTASKS': '4', 'SLURM_JOB_ID': '123', 'OMP_NUM_THREADS': '1', 'MAX_JOBS': '16',
            'AITER_LOG_MORE': '1', 'NCCL_P2P_DISABLE': '1', 'OMPI_MCA_memory': '^patcher', 'HWLOC_COMPONENTS': '-gl'}


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'aiter', 'triton', 'megatron'}:
            pytest.fail('CPU test imported vendor code')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    return load(NEW)


def test_only_concurrency_and_docstrings_differ():
    assert hashlib.sha256(OLD.read_bytes()).hexdigest() == '4501d7dca8aed54c1059cbe4eed7eae6aca6a00579340f689e0564382435983c'
    assert NEW.read_text() == OLD.read_text().replace('eight-worker', 'sixteen-worker').replace('"MAX_JOBS": "8"', '"MAX_JOBS": "16"')
    assert (PREP / 'original' / OLD.name).read_bytes() == OLD.read_bytes()


@pytest.mark.parametrize('rank', range(4))
def test_all_local_ranks(probe, rank):
    assert probe.validate_launch(settings(), environment(rank)) == (rank, rank)


@pytest.mark.parametrize('key,value', [('MAX_JOBS', '8'), ('MAX_JOBS', '32'), ('MAX_JOBS', '0'),
    ('OMP_NUM_THREADS', '16'), ('AITER_LOG_MORE', '0'), ('PREBUILD_THREAD_NUM', ''),
    ('PREBUILD_THREAD_NUM', '1'), ('PREBUILD_THREAD_NUM', '2'), ('SLURM_NTASKS', '8'),
    ('SLURM_LOCALID', '3'), ('BEVERIN_AITER_ATTN', '1')])
def test_preimport_environment_rejection(probe, monkeypatch, key, value):
    monkeypatch.setattr(probe.os, 'environ', environment() | {key: value})
    with pytest.raises(ValueError):
        probe.run(settings())


@pytest.mark.parametrize('section,key,value', [('resources', 'cpus_per_node', 96), ('resources', 'gpus_per_node', 1),
    ('resources', 'nodes', 2), ('resources', 'time', '00:20:00'), ('resources', 'cpu_bind', 'none'),
    ('resources', 'distribution', 'block:cyclic'), ('procedure', 'timeout', 900), ('image', 'sha256', '0' * 64)])
def test_budget_layout_identity_rejection(probe, section, key, value):
    config = deepcopy(settings())
    config[section][key] = value
    with pytest.raises(ValueError):
        probe.validate_launch(config, environment())


@pytest.fixture
def inputs(probe, tmp_path, monkeypatch):
    tests, seed = tmp_path / 'tests', tmp_path / 'seed'
    tests.mkdir()
    seed.mkdir()
    for name in probe.PINS:
        (tests / name).write_text('def validate_gradient_error(*args): return {}\n')
        monkeypatch.setitem(probe.PINS, name, probe.digest(tests / name))
    (seed / 'module_aiter_core.so').write_bytes(b'CPU fixture not ELF')
    (seed / 'seed.json').write_text(json.dumps({'gpu_archs': 'gfx942', 'cu_num': 228, 'packages': probe.VERSIONS,
        'modules': {'module_aiter_core.so': probe.digest(seed / 'module_aiter_core.so')}}))
    cpp = tmp_path / 'cpp_extension.py'
    cpp.write_text('# CPU source fixture, never imported\n')
    monkeypatch.setattr(probe, 'TESTS', tests)
    monkeypatch.setattr(probe, 'SEED', seed)
    monkeypatch.setattr(probe, 'SEED_HASH', probe.digest(seed / 'seed.json'))
    monkeypatch.setattr(probe, 'CPP_HASH', probe.digest(cpp))
    monkeypatch.setattr(probe.importlib.metadata, 'distribution', lambda name: SimpleNamespace(locate_file=lambda path: cpp))
    monkeypatch.setattr(probe.importlib.metadata, 'version', lambda name: probe.VERSIONS[name])
    monkeypatch.setattr(probe.os, 'environ', environment())
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    monkeypatch.chdir(tmp_path)
    return tests, seed, cpp


@pytest.mark.parametrize('target', ['script', 'seed', 'core', 'cpp', 'version', 'affinity'])
def test_preimport_identity_failure(probe, inputs, monkeypatch, target):
    tests, seed, cpp = inputs
    if target == 'version':
        monkeypatch.setattr(probe.importlib.metadata, 'version', lambda name: 'wrong')
    elif target == 'affinity':
        monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: {0})
    else:
        {'script': tests / 'attention-accuracy.py', 'seed': seed / 'seed.json',
         'core': seed / 'module_aiter_core.so', 'cpp': cpp}[target].write_bytes(b'changed')
    with pytest.raises(ValueError):
        probe.run(settings())
    value = json.loads(Path('attention-rank-0.json').read_text())
    assert value['status'] == 'failed' and 'scratch' not in value


@pytest.mark.parametrize('rank', range(4))
@pytest.mark.parametrize('failure', [False, True])
def test_private_same_process_gpu_and_cleanup(probe, inputs, tmp_path, monkeypatch, rank, failure):
    config = settings()
    config['scratch_root'] = str(tmp_path / 'scratch')
    monkeypatch.setattr(probe.os, 'environ', environment(rank))
    state = SimpleNamespace(device=None)
    fake_torch = SimpleNamespace(version=SimpleNamespace(hip='CPU fixture'), set_num_threads=lambda n: None,
        cuda=SimpleNamespace(device_count=lambda: 4, set_device=lambda n: setattr(state, 'device', n),
            current_device=lambda: state.device,
            get_device_properties=lambda n: SimpleNamespace(gcnArchName='gfx942', multi_processor_count=228)))
    guarded = builtins.__import__
    operation = object()
    def imported(name, *args, **kwargs):
        if name == 'torch':
            return fake_torch
        if name == 'aiter':
            return SimpleNamespace(flash_attn_func=operation)
        return guarded(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', imported)
    def numeric(torch, function, policy):
        cache = Path(probe.os.environ['AITER_JIT_DIR'])
        assert state.device == rank and function is operation and torch is fake_torch
        assert cache.is_relative_to(Path(config['scratch_root'])) and cache != probe.SEED
        assert len(list(cache.iterdir())) == 1
        assert probe.digest(cache / 'module_aiter_core.so') == probe.digest(probe.SEED / 'module_aiter_core.so')
        if failure:
            raise ValueError('synthetic numerical failure')
        return {'CPU_fixture_only': True}
    monkeypatch.setattr(probe, 'attention', numeric)
    if failure:
        with pytest.raises(ValueError):
            probe.run(config)
    else:
        probe.run(config)
    record = json.loads(Path(f'attention-rank-{rank}.json').read_text())
    assert record['status'] == ('failed' if failure else 'passed')
    assert record['selected_device'] == rank and record['jit_environment']['MAX_JOBS'] == '16'
    assert record['scratch_cleanup'] == 'removed' and not Path(record['scratch']).exists()
    assert record['affinity']['initial'] == record['affinity']['final']


def test_plan_is_pure_and_edf_is_pinned(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail('planning wrote or called subprocess')
    with monkeypatch.context() as guard:
        guard.setattr(subprocess, 'Popen', refuse)
        for name in ('write_text', 'write_bytes', 'mkdir'):
            guard.setattr(Path, name, refuse)
        value = plan(EXPERIMENT, site=ROOT / 'config.toml', submit=True, reservation='cscs', time='00:15:00')
    text = edf(value, '/future-run')
    for fragment in ('MAX_JOBS = "16"', 'OMP_NUM_THREADS = "1"', 'AITER_LOG_MORE = "1"',
                     'HWLOC_COMPONENTS = "-gl"', 'com.hooks.cxi.enabled = "false"', 'com.hooks.aws_ofi_nccl.enabled = "false"'):
        assert fragment in text
    assert value['procedure']['timeout'] == 720 and value['resources']['time'] == '00:15:00'


@pytest.fixture
def transport():
    return load(PREP / 'transport.py')


def test_transport_requires_separate_approval(transport, monkeypatch):
    monkeypatch.setattr(transport.sys, 'executable', transport.PYTHON)
    monkeypatch.delenv('APERTUS_ATTENTION_WORKERS16_APPROVAL', raising=False)
    with pytest.raises(RuntimeError, match='PREPARATION ONLY'):
        transport.main('sbatch')


def test_transport_and_clean_launch_environment(transport):
    command = transport.forwarded('sbatch', ['--parsable', '/future/job.sh'])
    assert command[:3] == ['/usr/bin/sbatch', '--uenv-passthrough=ignore', '--nodelist=nid002456']
    for node in ('nid002412', 'nid002454', 'nid002458', 'nid002744', 'nid002760'):
        assert node in command[3]
    assert 'nid002456' not in command[3]
    assert {'741653', '741609', '738614', '741645', '741650'} <= transport.PROTECTED_JOBS
    for arg in ('--nodelist=nid2', '--exclude=nid2', '-wnid2', '--uenv-passthrough=all'):
        with pytest.raises(ValueError):
            transport.forwarded('sbatch', [arg])
    launch = load(PREP / 'launch.py')
    env = launch.command_environment({'SLURM_JOB_ID': '741609', 'SBATCH_NODELIST': 'bad', 'SRUN_CPUS_PER_TASK': '1', 'PATH': '/usr/bin'})
    assert not any(key.startswith(('SLURM_', 'SBATCH_', 'SRUN_')) for key in env)
    assert env['PATH'].startswith(str(PREP / 'bin'))


@pytest.fixture
def approved_stub(transport, tmp_path, monkeypatch):
    # Synthetic CPU fixture only: scheduler execution is always stubbed.
    monkeypatch.setattr(transport, 'HERE', tmp_path)
    monkeypatch.setattr(transport.sys, 'executable', transport.PYTHON)
    (tmp_path / 'manifest.json').write_text('{}\n')
    evidence = tmp_path / 'execution-fixture'
    evidence.mkdir()
    approval = {'execution_authorized': True, 'maximum_submissions': 1, 'experiment': transport.EXPERIMENT,
        'node': 'nid002456', 'candidate_freshly_verified': True,
        'native_predecessor': {'job_id': '999998', 'terminal_reviewed': True, 'owned_group_quiescence_reviewed': True},
        'preparation_manifest_sha256': hashlib.sha256((tmp_path / 'manifest.json').read_bytes()).hexdigest(),
        'execution_artifacts': str(evidence)}
    path = tmp_path / 'synthetic-approval.json'
    path.write_text(json.dumps(approval))
    monkeypatch.setattr(transport.os, 'environ', {'APERTUS_ATTENTION_WORKERS16_APPROVAL': str(path)})
    calls = []
    monkeypatch.setattr(transport.os, 'execv', lambda executable, args: calls.append((executable, args)))
    monkeypatch.setattr(transport.sys, 'argv', ['sbatch', '--parsable', '/future/job.sh'])
    return approval, path, calls


@pytest.mark.parametrize('missing', ['native_predecessor', 'terminal_reviewed', 'owned_group_quiescence_reviewed', 'candidate_freshly_verified'])
def test_native_priority_gate_blocks_without_parent_review(transport, approved_stub, missing):
    approval, path, calls = approved_stub
    if missing in approval:
        del approval[missing]
    else:
        del approval['native_predecessor'][missing]
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match='native priority'):
        transport.main('sbatch')
    assert not calls


def test_one_shot_submission_stub(transport, approved_stub):
    _, _, calls = approved_stub
    transport.main('sbatch')
    assert len(calls) == 1 and calls[0][0] == '/usr/bin/sbatch'
    with pytest.raises(FileExistsError):
        transport.main('sbatch')
    assert len(calls) == 1


@pytest.mark.parametrize('job,node', [('741653', 'nid002456'), ('741609', 'nid002456'),
                                    ('999998', 'nid002456'), ('999999', 'nid002458')])
def test_protected_or_native_allocation_cannot_be_reused(transport, approved_stub, monkeypatch, job, node):
    _, _, calls = approved_stub
    transport.os.environ.update(SLURM_JOB_ID=job, SLURM_JOB_NODELIST=node)
    monkeypatch.setattr(transport.sys, 'argv', ['srun', f'--jobid={job}'])
    with pytest.raises(ValueError, match='fresh owned'):
        transport.main('srun')
    assert not calls
