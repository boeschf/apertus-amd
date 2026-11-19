"""Pure CPU fixtures: no local Torch/vendor/native import or GPU execution."""
import ast
import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tomllib
from types import SimpleNamespace
import unittest

import pytest
from apertus_amd.cli import plan
from apertus_amd.config import resolve
from apertus_amd.runs import edf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'experiments/validation/kda_attention_shape_probe.py'
EXPERIMENT = ROOT / 'experiments/chonk-kda-attention-shape-i910bfcf936d3'
PREP = ROOT / 'artifacts/investigations/performance-next/restart-20261116/aiter-shape-preparation'


def load(path):
    spec = importlib.util.spec_from_file_location('shape_fixture_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'aiter', 'triton', 'transformer_engine', 'megatron'}:
            pytest.fail('host vendor import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    return load(SCRIPT)


def settings():
    return resolve(EXPERIMENT, site=ROOT / 'config.toml', submit=True, reservation='cscs', time='00:15:00')


def environment(rank=0):
    return {'SLURM_PROCID': str(rank), 'SLURM_LOCALID': str(rank), 'SLURM_NODEID': '0', 'SLURM_NTASKS': '4',
            'SLURM_JOB_ID': 'fixture', 'OMP_NUM_THREADS': '1', 'MAX_JOBS': '16', 'HWLOC_COMPONENTS': '-gl', 'NCCL_P2P_DISABLE': '1',
            'NVTE_FLASH_ATTN': '1', 'NVTE_FUSED_ATTN': '1', 'NVTE_UNFUSED_ATTN': '1'}


@pytest.mark.parametrize('rank', range(4))
def test_launch(probe, monkeypatch, rank):
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    assert probe.validate_launch(settings(), environment(rank)) == (rank, rank)


@pytest.mark.parametrize('key,value', [('MAX_JOBS', '2'), ('OMP_NUM_THREADS', '4'), ('SLURM_NTASKS', '8'),
    ('SLURM_LOCALID', '3'), ('PREBUILD_THREAD_NUM', ''), ('BEVERIN_AITER_ATTN', '1')])
def test_reject_environment(probe, monkeypatch, key, value):
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    with pytest.raises(ValueError):
        probe.validate_launch(settings(), environment() | {key: value})


@pytest.mark.parametrize('flag', ['NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN'])
@pytest.mark.parametrize('value', [None, '0', ''])
def test_require_explicit_baseline_attention_flags(probe, monkeypatch, flag, value):
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    env = environment()
    if value is None:
        env.pop(flag)
    else:
        env[flag] = value
    with pytest.raises(ValueError, match='admission flags'):
        probe.validate_launch(settings(), env)


def test_attention_admission_matches_explicit_workload_not_image_defaults(probe):
    baseline = tomllib.loads((ROOT / 'experiments/moe-chonk-kda-me242f3c3c113-i910bfcf936d3/experiment.toml').read_text())
    proposal = tomllib.loads((EXPERIMENT / 'experiment.toml').read_text())
    assert len(probe.ATTENTION_FLAGS) == 3
    for name, value in probe.ATTENTION_FLAGS.items():
        assert value == baseline['environment'][name] == proposal['environment'][name] == '1'


@pytest.mark.parametrize('section,key,value', [('image', 'sha256', 'bad'), ('resources', 'nodes', 2),
    ('resources', 'time', '00:05:00'), ('procedure', 'timeout', 900), ('resources', 'cpu_bind', 'none')])
def test_reject_resources(probe, monkeypatch, section, key, value):
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    value_settings = deepcopy(settings())
    value_settings[section][key] = value
    with pytest.raises(ValueError):
        probe.validate_launch(value_settings, environment())


def test_getter_pinned_actual_source_and_readonly_assertion(probe, monkeypatch):
    text = (PREP / 'torch-backends-cuda.py').read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'fp16_bf16_reduction_math_sdp_allowed')
    getter = ast.get_source_segment(text, node) + '\n'
    assert hashlib.sha256(getter.encode()).hexdigest() == probe.ACCESSOR_HASH
    assert '_get_math_sdp_allow_fp16_bf16_reduction' in getter and '_set_' not in getter
    monkeypatch.setattr(probe.inspect, 'getsource', lambda fn: getter)
    for allowed in (False, True, None):
        torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(fp16_bf16_reduction_math_sdp_allowed=lambda: allowed)),
            are_deterministic_algorithms_enabled=lambda: False, get_float32_matmul_precision=lambda: 'highest')
        if allowed is False:
            assert probe.precision(torch)['math_reduction_allowed'] is False
        else:
            with pytest.raises(ValueError, match='observed'):
                probe.precision(torch)
    monkeypatch.setattr(probe.inspect, 'getsource', lambda fn: 'changed')
    with pytest.raises(ValueError, match='differs'):
        probe.precision(torch)


def test_original_three_stdlib_equation_policy_checks(probe):
    path = ROOT / 'artifacts/investigations/performance-next/restart-20261116/aiter-reference-feasibility/check_formulas.py'
    module = load(path)
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(module.FormulaTests).run(result)
    assert result.testsRun == 3 and result.wasSuccessful()


@pytest.mark.parametrize('gib', [64, 120, 128])
def test_reported_memory_headroom_not_nominal_physical_class(probe, gib):
    hardware = {'arch': 'gfx942:sramecc+:xnack-', 'cu': 228, 'reported_memory_bytes': gib * 2**30}
    before = deepcopy(hardware)
    assert probe.validate_hardware(hardware) is None
    assert hardware == before
    assert probe.MIN_REPORTED_MEMORY_BYTES == 2 * probe.GPU_PLANNING_BYTES == 64 * 2**30


@pytest.mark.parametrize('gib', [0, 32, 63])
def test_insufficient_reported_memory_fails_clearly(probe, gib):
    with pytest.raises(ValueError, match='below 64GiB planning floor'):
        probe.validate_hardware({'arch': 'gfx942', 'cu': 228, 'reported_memory_bytes': gib * 2**30})


@pytest.mark.parametrize('arch,cu', [('gfx950', 228), ('gfx942', 227), ('gfx942', 304)])
def test_hardware_identity_unchanged(probe, arch, cu):
    with pytest.raises(ValueError, match='hardware identity'):
        probe.validate_hardware({'arch': arch, 'cu': cu, 'reported_memory_bytes': 120 * 2**30})


def test_monolithic_control_and_worker_only_selfcheck_source_contract(probe):
    tree = ast.parse(SCRIPT.read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    qualify = ast.unparse(funcs['qualify'])
    assert qualify.count('scaled_dot_product_attention(') == 1
    assert 'enable_gqa=True' in qualify and 'deterministic=False' in qualify and 'return_lse=True' in qualify
    assert 'repeat_interleave' not in qualify and 'sdpa_kernel(backend.MATH)' in qualify
    helper = ast.unparse(funcs['reference'])
    assert 'torch.arange(start, stop)' in helper and '.sum(1, keepdim=True)' in helper
    check = ast.unparse(funcs['selfcheck'])
    assert 'torch.float32, 1e-05' in check and 'torch.float64, 1e-11' in check
    assert 'torch.autograd.grad' in check
    worker = ast.unparse(funcs['worker'])
    assert worker.index('precision(torch)') < worker.index('torch.cuda.set_device')
    assert worker.index('selfcheck(') < worker.index('torch.cuda.set_device')


@pytest.mark.parametrize('failure', [False, True])
def test_trace_restore_on_failure(probe, failure):
    def original(value):
        if failure:
            raise RuntimeError('primary')
        return value
    module = SimpleNamespace(__name__='fixture', op=original)
    records = []
    try:
        with probe.trace(module, ('op', 'absent'), records):
            module.op(3)
    except RuntimeError:
        assert failure
    assert module.op is original and records[0]['call'] == 'op'


@pytest.mark.parametrize('outcome', ['pass', 'failure', 'timeout', 'unquiescent'])
def test_same_PGID_private_seed_cleanup_and_failure_retention(probe, tmp_path, monkeypatch, outcome):
    seed = tmp_path / 'seed'; seed.mkdir()
    (seed / 'module_aiter_core.so').write_bytes(b'CPU fixture')
    (seed / 'seed.json').write_text('{}')
    monkeypatch.setattr(probe, 'SEED', seed)
    monkeypatch.setattr(probe, 'SEED_HASH', probe.digest(seed / 'seed.json'))
    monkeypatch.setattr(probe, 'provenance', lambda: {'seed': {'modules': {'module_aiter_core.so': probe.digest(seed / 'module_aiter_core.so')}}})
    monkeypatch.setattr(probe, 'quiescent', lambda: outcome != 'unquiescent')
    monkeypatch.setattr(probe.os, 'environ', environment())
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    monkeypatch.chdir(tmp_path)
    def popen(command, **kwargs):
        assert kwargs['start_new_session'] is False and 'worker' in command
        cache = Path(kwargs['env']['AITER_JIT_DIR'])
        assert cache.is_relative_to(tmp_path / 'scratch') and (cache / 'module_aiter_core.so').read_bytes() == b'CPU fixture'
        output = Path(command[-1])
        def wait(timeout):
            assert timeout == 690
            if outcome == 'timeout':
                raise subprocess.TimeoutExpired(command, timeout)
            (output / 'worker.json').write_text(json.dumps({'status': 'failed' if outcome == 'failure' else 'passed', 'pgid': probe.os.getpgrp(), 'error': 'fixture failure'}))
            return int(outcome == 'failure')
        return SimpleNamespace(wait=wait)
    monkeypatch.setattr(probe.subprocess, 'Popen', popen)
    config = settings(); config['scratch_root'] = str(tmp_path / 'scratch')
    if outcome == 'pass':
        probe.run(config)
    else:
        with pytest.raises((ValueError, subprocess.TimeoutExpired)):
            probe.run(config)
    result = json.loads((tmp_path / 'attention-shape-rank-0/summary.json').read_text())
    assert result['status'] == ('passed' if outcome == 'pass' else 'failed')
    assert Path(result['scratch']).exists() == (outcome != 'pass')


def test_secondary_evidence_failure_does_not_mask_primary(probe, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'write_text', lambda *a, **k: (_ for _ in ()).throw(OSError('secondary')))
    with pytest.raises(RuntimeError, match='primary') as caught:
        try:
            raise RuntimeError('primary')
        finally:
            probe.save(tmp_path / 'out.json', {})
    assert 'secondary evidence error' in caught.value.__notes__[0]


def test_pure_plan_and_hook_guards(probe, monkeypatch):
    def refuse(*a, **k):
        pytest.fail('pure plan wrote/called subprocess')
    with monkeypatch.context() as guard:
        guard.setattr(subprocess, 'Popen', refuse)
        for name in ('write_text', 'write_bytes', 'mkdir'):
            guard.setattr(Path, name, refuse)
        value = plan(EXPERIMENT, site=ROOT / 'config.toml', submit=True, reservation='cscs', time='00:15:00')
    text = edf(value, '/future')
    for required in ('MAX_JOBS = "16"', 'OMP_NUM_THREADS = "1"', 'NVTE_FLASH_ATTN = "1"', 'NVTE_FUSED_ATTN = "1"', 'NVTE_UNFUSED_ATTN = "1"', 'HWLOC_COMPONENTS = "-gl"', 'com.hooks.cxi.enabled = "false"', 'com.hooks.aws_ofi_nccl.enabled = "false"'):
        assert required in text
    assert value['procedure']['timeout'] == 720
