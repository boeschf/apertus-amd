"""Pure preparation fixtures: no Torch/Triton/vendor import or GPU execution."""
import ast
import builtins
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'artifacts/investigations/performance-next/restart-20261116/imanol-patch-integration/output-gate-qualification-preparation'
SCRIPT = ROOT / 'experiments/validation/chonk_output_gate_probe.py'


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'triton', 'aiter', 'transformer_engine', 'megatron'):
            pytest.fail('vendor import during CPU preparation')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    spec = importlib.util.spec_from_file_location('output_gate_probe_fixture', SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def settings(probe):
    return {'image': {'sha256': probe.IMAGE}, 'scratch_root': '/unused',
        'resources': {'nodes': 1, 'gpus_per_node': 4, 'cpus_per_node': 192, 'reservation': 'cscs', 'time': '00:15:00',
                      'mpi': 'pmix', 'cpu_bind': 'map_ldom:0,1,2,3', 'distribution': 'block:block'},
        'procedure': {'kind': 'commands', 'timeout': 720, 'commands': [['python3', '-u', '/apertus/validation.py', 'run']]}}


def env(rank=0):
    return {'SLURM_PROCID': str(rank), 'SLURM_LOCALID': str(rank), 'SLURM_NODEID': '0', 'SLURM_NTASKS': '4',
            'OMP_NUM_THREADS': '1', 'MAX_JOBS': '2', 'TORCH_COMPILE_DISABLE': '0', 'SLURM_JOB_ID': 'fixture'}


def test_source_chain_exact_applied_helper_and_AST_delta(probe, tmp_path):
    chain = json.loads((PREP / 'SOURCE-CHAIN.json').read_text())
    for name, checksum in chain['files'].items():
        assert probe.sha((PREP / name).read_bytes()) == checksum
    assert probe.HELPER == (PREP / 'beverin_gate.py.txt').read_text()
    assert probe.METHOD == (PREP / 'reference-method.py.txt').read_text()
    before = ast.parse(probe.METHOD); expected = deepcopy(before)
    expected.body[0].decorator_list = []; expected.body[0].args.args = expected.body[0].args.args[1:]
    assert ast.dump(probe.source_tree()) == ast.dump(expected)
    assert ast.dump(before.body[0], include_attributes=False) == chain['method_adaptation']['before_ast']
    assert ast.dump(probe.source_tree().body[0], include_attributes=False) == chain['method_adaptation']['after_ast']
    base = tmp_path / 'megatron/core/transformer/attention.py'; base.parent.mkdir(parents=True)
    base.write_bytes((PREP / 'base-attention.py.txt').read_bytes())
    subprocess.run(['git', 'apply', str(PREP / 'source.patch')], cwd=tmp_path, check=True, capture_output=True)
    assert (tmp_path / 'megatron/core/extensions/beverin_gate.py').read_text() == probe.HELPER
    jit = (PREP / 'jit.py.txt').read_text()
    assert 'jit_fuser = torch.compile' in jit


@pytest.mark.parametrize('rank', range(4))
def test_resource_contract(probe, monkeypatch, rank):
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    assert probe.validate(settings(probe), env(rank)) == (rank, rank)


@pytest.mark.parametrize('key,value', [('TORCH_COMPILE_DISABLE', '1'), ('TORCHDYNAMO_DISABLE', '1'),
    ('OMP_NUM_THREADS', '4'), ('SLURM_NTASKS', '1'), ('SLURM_LOCALID', '3')])
def test_no_eager_or_resource_substitute(probe, monkeypatch, key, value):
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    with pytest.raises(ValueError):
        probe.validate(settings(probe), env() | {key: value})


def metric(probe, count=17):
    return dict(finite=True, numel=count, mismatched_elements=0, bf16_mismatches=0, raw_bf16_mismatches=0,
                max_bf16_ulp=0, rms_error=0.0, reference_rms=1.0, max_abs=0.0, passed=True)


def test_proposed_constants_match_existing_pointwise_policy(probe):
    assert (probe.ELEMENT_RTOL, probe.ELEMENT_ATOL, probe.RMS_RTOL, probe.RMS_ATOL) == (4*2**-8, 2**-8, 4*2**-8, 1e-5)
    assert probe.metric_pass(metric(probe))


@pytest.mark.parametrize('changes', [{'finite': False}, {'numel': 0}, {'mismatched_elements': 1},
    {'rms_error': 1.0}, {'max_abs': math.nan}, {'reference_rms': math.inf}])
def test_metrics_fail_closed(probe, changes):
    assert not probe.metric_pass(metric(probe) | changes)


def record(probe, rank=0):
    cases = {}
    for name in probe.CASES:
        count = 4096*3584 if name.startswith('actual') else 4097 if name == 'tail-4097' else 17
        checks = {k: metric(probe, count) for k in ('forward', 'dx', 'dgate')}
        cases[name] = {'inputs': probe.expected_inputs(name), 'arms': {'compiled-baseline': deepcopy(checks), 'candidate': deepcopy(checks)}, 'candidate_vs_compiled': deepcopy(checks)}
    return dict(status='passed', image=probe.IMAGE, rank=rank, local_rank=rank, selected_device=rank,
                helper_sha256=probe.HELPER_SHA, method_sha256=probe.METHOD_SHA, cases=cases, probe_sha256='a'*64,
                versions=probe.VERSIONS.copy(), torch_git=probe.GIT, torch_hip=probe.HIP,
                scratch='/tmp/opencode/output-gate-fixture', helper_path='/tmp/opencode/output-gate-fixture/component-source/megatron/core/extensions/beverin_gate.py',
                timings=False, sampling=False, policy='PROPOSED-parent-review-required',
                tolerances=dict(element_rtol=probe.ELEMENT_RTOL, element_atol=probe.ELEMENT_ATOL, rms_rtol=probe.RMS_RTOL, rms_atol=probe.RMS_ATOL),
                affinity=dict(initial=list(range(48)), post_gpu_init=list(range(48)), final=list(range(48)), errors=[]))


@pytest.mark.parametrize('failure', ['status', 'case', 'arm', 'tensor', 'sampling', 'metric'])
def test_incomplete_or_failed_native_records_never_pass(probe, failure):
    r = record(probe)
    if failure == 'status': r['status'] = 'failed'
    elif failure == 'case': r['cases'].pop('short-17')
    elif failure == 'arm': r['cases']['short-17']['arms'].pop('candidate')
    elif failure == 'tensor': r['cases']['short-17']['candidate_vs_compiled'].pop('dx')
    elif failure == 'sampling': r['cases']['short-17']['arms']['candidate']['dx']['numel'] = 1
    else: r['cases']['short-17']['arms']['candidate']['dx']['mismatched_elements'] = 1
    with pytest.raises(ValueError): probe.validate_record(r, 0, 'a'*64)


def test_check_requires_all_four_records(probe, tmp_path):
    output = tmp_path / 'output/0'; output.mkdir(parents=True)
    frozen = b'frozen historical validation, not current live source'
    expected = probe.sha(frozen)
    (tmp_path / 'validation.py').write_bytes(frozen)
    (tmp_path / 'resolved.json').write_text(json.dumps({'validation': {'sha256': expected}}))
    for rank in range(3):
        (output / f'output-gate-{rank}.json').write_text(json.dumps(record(probe, rank) | {'probe_sha256': expected}))
    with pytest.raises(FileNotFoundError): probe.check(tmp_path)
    (output / 'output-gate-3.json').write_text(json.dumps(record(probe, 3) | {'probe_sha256': expected}))
    probe.check(tmp_path)  # Synthetic summary plumbing only, not GPU evidence.
    assert json.loads((tmp_path / 'output/checks.json').read_text())['status'] == 'passed'
    (tmp_path / 'validation.py').write_text('tampered')
    with pytest.raises(ValueError, match='frozen validation'): probe.check(tmp_path)


@pytest.mark.parametrize('key', ['probe_sha256', 'versions', 'torch_git', 'torch_hip', 'helper_path', 'scratch', 'affinity', 'tolerances', 'sampling', 'timings'])
def test_missing_provenance_rejected(probe, key):
    r = record(probe); r.pop(key)
    with pytest.raises((KeyError, ValueError)): probe.validate_record(r, 0, 'a'*64)


@pytest.mark.parametrize('key,value', [('probe_sha256', 'b'*64), ('versions', {}), ('torch_git', 'wrong'), ('torch_hip', 'wrong'),
    ('helper_path', '/outside/beverin_gate.py'), ('scratch', 'relative'), ('timings', 0), ('sampling', True),
    ('policy', 'approved'), ('rank', False), ('method_sha256', 'wrong'), ('helper_sha256', 'wrong'), ('error', None)])
def test_tampered_provenance_rejected(probe, key, value):
    r = record(probe); r[key] = value
    with pytest.raises(ValueError): probe.validate_record(r, 0, 'a'*64)


@pytest.mark.parametrize('field', ['initial', 'post_gpu_init', 'final', 'errors'])
def test_affinity_drift_or_error_rejected(probe, field):
    r = record(probe); r['affinity'][field] = ['failure'] if field == 'errors' else list(range(47))
    with pytest.raises(ValueError): probe.validate_record(r, 0, 'a'*64)


@pytest.mark.parametrize('case', ['actual-contiguous', 'actual-strided-offset', 'tail-4097', 'short-17'])
@pytest.mark.parametrize('field,value', [('shape', [1]), ('stride', [1]), ('offset', 9), ('dtype', 'torch.float32')])
def test_layout_contract_rejects_missing_or_wrong_values(probe, case, field, value):
    r = record(probe); r['cases'][case]['inputs'][0][field] = value
    with pytest.raises(ValueError): probe.validate_record(r, 0, 'a'*64)


@pytest.mark.parametrize('key,value', [('numel', True), ('finite', 1), ('numel', -1), ('numel', 1.5),
    ('mismatched_elements', -1), ('raw_bf16_mismatches', 18), ('bf16_mismatches', True),
    ('max_bf16_ulp', 65536), ('rms_error', -1), ('max_abs', '0'), ('reference_rms', False)])
def test_metric_types_ranges_and_nonfinite_values_rejected(probe, key, value):
    assert not probe.metric_pass(metric(probe) | {key: value})


@pytest.mark.parametrize('key', ['bf16_mismatches', 'raw_bf16_mismatches', 'max_bf16_ulp'])
def test_missing_diagnostics_rejected(probe, key):
    m = metric(probe); m.pop(key); assert not probe.metric_pass(m)


def test_diagnostics_not_a_bitwise_equality_gate(probe):
    assert probe.metric_pass(metric(probe) | {'raw_bf16_mismatches': 1})  # signed zero
    assert probe.metric_pass(metric(probe) | {'raw_bf16_mismatches': 2, 'bf16_mismatches': 2, 'max_bf16_ulp': 3})
    r = record(probe); r['tolerances']['element_atol'] *= 2
    with pytest.raises(ValueError): probe.validate_record(r, 0, 'a'*64)
    minus_zero, plus_zero = -32768, 0
    ordered = lambda word: -32768-word if word < 0 else word
    assert ordered(minus_zero) == ordered(plus_zero) and minus_zero != plus_zero


def test_native_success_cannot_bypass_strict_validation(probe, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path); monkeypatch.setattr(probe.os, 'environ', env())
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    monkeypatch.setattr(probe.importlib.metadata, 'version', lambda name: probe.VERSIONS[name])
    fake = SimpleNamespace(__version__=probe.VERSIONS['torch'], version=SimpleNamespace(git_version=probe.GIT, hip=probe.HIP),
        _dynamo=SimpleNamespace(config=SimpleNamespace(suppress_errors=False)), compile=lambda f: f, set_num_threads=lambda n: None,
        cuda=SimpleNamespace(device_count=lambda: 4, set_device=lambda n: None, current_device=lambda: 0,
            get_device_properties=lambda n: SimpleNamespace(gcnArchName='gfx942', multi_processor_count=228)))
    original = builtins.__import__
    monkeypatch.setattr(builtins, '__import__', lambda name, *a, **k: fake if name == 'torch' else original(name, *a, **k))
    monkeypatch.setattr(probe.importlib.util, 'spec_from_file_location', lambda name, path: SimpleNamespace(path=path, loader=SimpleNamespace(exec_module=lambda module: None)))
    monkeypatch.setattr(probe.importlib.util, 'module_from_spec', lambda spec: SimpleNamespace(__file__=str(spec.path), sigmoid_gate=lambda *a: None))
    def numerical(torch, baseline, candidate, r, device):
        r['cases'] = record(probe)['cases']; r['sampling'] = True
    monkeypatch.setattr(probe, 'numerical', numerical)
    s = settings(probe); s.update(scratch_root=str(tmp_path / 'scratch'), validation={'sha256': probe.sha(SCRIPT.read_bytes())})
    with pytest.raises(ValueError, match='sampling'): probe.run(s)
    assert json.loads((tmp_path / 'output-gate-0.json').read_text())['status'] == 'failed'


def test_explicit_input_pure_plan_and_default_pmix(probe, monkeypatch):
    from apertus_amd.cli import plan
    from apertus_amd.runs import edf
    def refuse(*a, **k): pytest.fail('planning wrote or executed a subprocess')
    with monkeypatch.context() as guard:
        guard.setattr(subprocess, 'Popen', refuse)
        for name in ('write_text', 'write_bytes', 'mkdir'): guard.setattr(Path, name, refuse)
        value = plan(ROOT / 'experiments/chonk-output-gate-qualification-i910bfcf936d3', site=ROOT / 'config.toml', submit=True, reservation='cscs', time='00:15:00')
    assert value['resources']['mpi'] == 'pmix' and value['procedure']['timeout'] == 720
    assert value['environment']['MAX_JOBS'] == '2' and value['environment']['TORCH_COMPILE_DISABLE'] == '0'
    text = edf(value, '/future')
    for item in ('HWLOC_COMPONENTS = "-gl"', 'com.hooks.cxi.enabled = "false"', 'com.hooks.aws_ofi_nccl.enabled = "false"'):
        assert item in text


@pytest.mark.parametrize('key,value', [('mpi', 'none'), ('cpu_bind', 'none'), ('distribution', 'cyclic')])
def test_resource_guard_retains_existing_defaults(probe, monkeypatch, key, value):
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    s = settings(probe); s['resources'][key] = value
    with pytest.raises(ValueError): probe.validate(s, env())


def test_source_contract_no_reimplementation_or_compile_toggles(probe):
    source = SCRIPT.read_text(); tree = ast.parse(source)
    assert 'torch.compile(namespace[\'_apply_output_gate\'])' in source
    assert 'fullgraph=' not in source and 'backend=' not in source and 'mode=' not in source
    assert 'module.sigmoid_gate' in source and 'module.__file__' in source
    assert '1::2' in source and 'storage_offset()' in source
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('Popen', 'run', 'set_float32_matmul_precision') for n in ast.walk(tree))


def test_original_exception_primary_when_evidence_fails(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, 'write_text', lambda *a, **k: (_ for _ in ()).throw(OSError('secondary')))
    with pytest.raises(ValueError, match='primary') as caught:
        try: raise ValueError('primary')
        finally: probe.save(tmp_path / 'record.json', {})
    assert caught.value.__notes__


def test_preimport_version_failure_retained_and_nonzero(probe, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(probe.os, 'environ', env())
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    monkeypatch.setattr(probe.importlib.metadata, 'version', lambda name: 'wrong')
    s = settings(probe); s['validation'] = {'sha256': probe.sha(SCRIPT.read_bytes())}
    with pytest.raises(ValueError, match='version'):
        probe.run(s)
    assert json.loads((tmp_path / 'output-gate-0.json').read_text())['status'] == 'failed'
