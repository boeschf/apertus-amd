"""CPU-only source/protocol/schema fixtures; never import native vendor code."""
import ast
import builtins
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from apertus_amd.cli import plan
from apertus_amd.runs import edf

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'artifacts/investigations/performance-next/restart-20261116/output-gate-performance-implementation'
SCRIPT = ROOT / 'experiments/validation/chonk_output_gate_performance.py'


@pytest.fixture
def probe(monkeypatch):
    old = builtins.__import__
    def guard(name, *a, **k):
        if name.split('.')[0] in ('torch', 'triton', 'aiter', 'transformer_engine', 'megatron'): pytest.fail('vendor import')
        return old(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', guard)
    spec = importlib.util.spec_from_file_location('gate_perf_fixture', SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_exact_source_math_and_canonical_operand_reuse(probe):
    original = ast.parse((PREP / 'accepted-validation.py.txt').read_text()); current = ast.parse(SCRIPT.read_text())
    find = lambda tree, name: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ('analytic', 'compare', 'source_tree', 'metric_pass', 'tensors', 'validate_record'):
        assert ast.dump(find(original, name)) == ast.dump(find(current, name))
    assert probe.sha(probe.HELPER.encode()) == probe.HELPER_SHA
    assert probe.sha(probe.METHOD.encode()) == probe.METHOD_SHA
    numeric = ast.unparse(find(current, 'numerical'))
    assert 'values = provided[case]' in numeric and 'manual_seed' not in numeric and 'tensors(' not in numeric
    run = ast.unparse(find(current, 'run'))
    assert run.count('torch.manual_seed(123)') == 1
    assert 'for case in CASES' in run and 'provided' in run and 'snapshots' in run
    assert 'torch.compile(namespace[\'_apply_output_gate\'])' in run and 'backend=' not in run
    assert (probe.ELEMENT_RTOL, probe.ELEMENT_ATOL, probe.RMS_RTOL, probe.RMS_ATOL) == (4*2**-8, 2**-8, 4*2**-8, 1e-5)


@pytest.mark.parametrize('rank', range(4))
def test_orders_counts_and_rank_pairing(probe, rank):
    sequence = [probe.order(rank, block) for block in range(probe.BLOCKS)]
    assert len(sequence) == 12 and sequence.count('ABBA') == sequence.count('BAAB') == 6
    assert sum(s.count('A') for s in sequence) == sum(s.count('B') for s in sequence) == 24
    assert probe.WARM == 20 and probe.WARM_BLOCKS == 2 and probe.BATCH == 16
    assert probe.case_order(rank) == (probe.PERF_CASES if rank % 2 == 0 else probe.PERF_CASES[::-1])


def rows(probe, rank=0, b=90.0):
    return [dict(case=case, block=block, slot=slot, arm=arm, completion_us=100.0 if arm == 'A' else b)
            for case in probe.case_order(rank) for block in range(12) for slot, arm in enumerate(probe.order(rank, block))]


@pytest.mark.parametrize('b,positive', [(90, True), (98, False), (110, False)])
def test_preregistered_positive_and_negative_decisions(probe, b, positive):
    result = probe.decision(rows(probe, b=b), 0)
    assert all(r['positive_timing'] is positive for r in result.values())
    assert len(result) == 2 and all(len(r['block_gains']) == 12 for r in result.values())


def test_noise_and_tail_regression_prevent_positive(probe):
    samples = rows(probe)
    for r in samples:
        if r['arm'] == 'A': r['completion_us'] = 80 if r['slot'] in (0, 1) else 120
    assert not any(r['positive_timing'] for r in probe.decision(samples, 0).values())


def snapshot(graphs=1):
    return {'dynamo': {'stats': {'unique_graphs': graphs, 'calls_captured': graphs}, 'graph_break': {}},
            'generated_kernel_count': graphs, 'triton_keys': {'_gate_fwd': {'0': ['1'*64]}, '_gate_bwd': {'0': ['1'*64]}}}


def test_compiler_proof_no_reset_or_unproved_fallback(probe):
    proof = {'before_pre': snapshot(0), 'after_pre': snapshot(), 'after_warm': snapshot(), 'after_measure': snapshot()}
    assert probe.proof_ok(proof)
    proof['after_measure']['generated_kernel_count'] += 1
    assert not probe.proof_ok(proof)
    proof['after_measure'] = snapshot(); proof['after_pre'] = snapshot(0)
    assert not probe.proof_ok(proof)
    src = SCRIPT.read_text()
    assert 'du.counters.clear' not in src and 'metrics.reset' not in src and 'empty_cache' not in src
    with pytest.raises(ValueError, match='UNQUALIFIED'):
        probe.counters(SimpleNamespace(counters={}), SimpleNamespace(generated_kernel_count=0), SimpleNamespace(_gate_fwd=SimpleNamespace(), _gate_bwd=SimpleNamespace()))


def test_batch_clocks_units_and_identical_reset_semantics(probe, monkeypatch):
    ticks = iter((1000, 5000, 9000)); monkeypatch.setattr(probe.time, 'perf_counter_ns', lambda: next(ticks))
    leaves = [SimpleNamespace(grad='previous'), SimpleNamespace(grad='previous')]; calls = []
    def operation(x, g):
        calls.append((x, g)); return SimpleNamespace(backward=lambda dy: None)
    event = SimpleNamespace(record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: .004)
    cuda = SimpleNamespace(synchronize=lambda device: None, reset_peak_memory_stats=lambda device: None,
        max_memory_allocated=lambda device: 20, max_memory_reserved=lambda device: 30)
    r = probe.sample(SimpleNamespace(cuda=cuda), operation, leaves, object(), (event, event), 0)
    assert len(calls) == 16 and all(x.grad is None for x in leaves)
    assert r['completion_us'] == .5 and r['host_return_us'] == .25 and r['device_span_us'] == .25


def valid_record(probe):
    r = dict(status='passed', rank=0, image=probe.IMAGE, probe_sha256='a'*64, protocol=[20,2,12,16], operands_unchanged=True,
        phases={p: {'status': 'passed', 'sha256': 'b'*64} for p in ('pre', 'post')},
        scratch='/tmp/opencode/output-gate-perf-fixture',
        component=dict(helper_sha256=probe.HELPER_SHA, method_sha256=probe.METHOD_SHA, versions=probe.VERSIONS, torch_git=probe.GIT, torch_hip=probe.HIP,
            tolerances=dict(element_rtol=probe.ELEMENT_RTOL, element_atol=probe.ELEMENT_ATOL, rms_rtol=probe.RMS_RTOL, rms_atol=probe.RMS_ATOL),
            helper_path='/tmp/opencode/output-gate-perf-fixture/component-source/megatron/core/extensions/beverin_gate.py'),
        counter_sources={name: {'path': '/opt/venv/lib/python3.12/site-packages/'+name, 'sha256': 'c'*64} for name in ('torch/_dynamo/utils.py', 'torch/_inductor/metrics.py', 'triton/runtime/jit.py')},
        compiler=dict(before_pre=snapshot(0), after_pre=snapshot(), after_warm=snapshot(), after_measure=snapshot()), samples=[])
    t0 = 1000000
    for row in rows(probe):
        duration=int(row['completion_us']*1000*16)
        row.update(status='complete',batch=16,t0_ns=t0,return_ns=t0+duration//2,complete_ns=t0+duration,
            event_ms=row['completion_us']*16/1000, host_return_us=row['completion_us']/2, device_span_us=row['completion_us'],peak_allocated=10,peak_reserved=20)
        r['samples'].append(row)
        t0 += duration + 1000
    r['warm_stability'] = {case: [snapshot(), snapshot()] for case in probe.PERF_CASES}
    r['decision']=probe.decision(r['samples'],0)
    return r


@pytest.mark.parametrize('kind', ['missing-sample','order','unit','counter','phase','source','numeric-mutation'])
def test_strict_performance_schema_failures(probe, kind):
    r = valid_record(probe); probe.validate_performance(r,0,'a'*64)
    if kind=='missing-sample': r['samples'].pop()
    elif kind=='order': r['samples'][0]['arm']='B'
    elif kind=='unit': r['samples'][0]['completion_us']*=1000
    elif kind=='counter': r['compiler']['after_measure']['generated_kernel_count']+=1
    elif kind=='phase': r['phases'].pop('post')
    elif kind=='source': r['component']['helper_sha256']='wrong'
    else: r['operands_unchanged']=False
    with pytest.raises((ValueError, KeyError)): probe.validate_performance(r,0,'a'*64)


def test_input_and_edf_remain_default_pmix(probe):
    s=plan(ROOT/'experiments/chonk-output-gate-performance-i910bfcf936d3',site=ROOT/'config.toml',submit=True,reservation='cscs',time='00:15:00')
    assert s['resources']['mpi']=='pmix' and s['resources']['cpu_bind']=='map_ldom:0,1,2,3' and s['resources'].get('cpus_per_task') is None
    assert s['environment']['MAX_JOBS']=='2' and s['procedure']['timeout']==720
    text=edf(s,'/future'); assert 'HWLOC_COMPONENTS = "-gl"' in text and 'com.hooks.cxi.enabled = "false"' in text and 'com.hooks.aws_ofi_nccl.enabled = "false"' in text


def test_preimport_failure_retains_unqualified_record_without_timings(probe, monkeypatch, tmp_path):
    s=plan(ROOT/'experiments/chonk-output-gate-performance-i910bfcf936d3',site=ROOT/'config.toml',submit=True,reservation='cscs',time='00:15:00')
    env=dict(s['environment'], SLURM_PROCID='0', SLURM_LOCALID='0', SLURM_NODEID='0', SLURM_NTASKS='4', SLURM_JOB_ID='fixture')
    monkeypatch.chdir(tmp_path); monkeypatch.setattr(probe.os,'environ',env)
    monkeypatch.setattr(probe.os,'sched_getaffinity',lambda pid:set(range(48)))
    monkeypatch.setattr(probe.importlib.metadata,'version',lambda name:'wrong')
    with pytest.raises(ValueError,match='source/version'):probe.run(s)
    r=json.loads((tmp_path/'output-gate-performance-0.json').read_text())
    assert r['status']=='unqualified' and r['samples']==[] and 'error' in r


@pytest.mark.parametrize('bad', ['empty', 'missing', 'wrong-device', 'nonhex', 'unsorted', 'duplicate', 'zero'])
def test_helper_cache_proof_failures(probe, bad):
    r = valid_record(probe)
    for phase in ('after_pre', 'after_warm', 'after_measure'):
        helpers = r['compiler'][phase]['triton_keys']
        if bad == 'empty': helpers.clear()
        elif bad == 'missing': helpers.pop('_gate_bwd')
        elif bad == 'wrong-device': helpers['_gate_fwd'] = {'3': ['1'*64]}
        else: helpers['_gate_fwd']['0'] = {'nonhex': ['g'*64], 'unsorted': ['2'*64, '1'*64], 'duplicate': ['1'*64]*2, 'zero': ['0'*64]}[bad]
    with pytest.raises(ValueError): probe.validate_performance(r, 0, 'a'*64)


@pytest.mark.parametrize('bad', ['overlap', 'reuse', 'zero-event', 'missing-warm', 'one-warm', 'extra-warm', 'unstable-warm'])
def test_clock_and_fixed_warm_failures(probe, bad):
    r = valid_record(probe)
    if bad in ('overlap', 'reuse'):
        row = r['samples'][1]
        target = r['samples'][0]['t0_ns'] if bad == 'reuse' else r['samples'][0]['complete_ns'] - 1
        delta = target - row['t0_ns']
        for name in ('t0_ns', 'return_ns', 'complete_ns'): row[name] += delta
    elif bad == 'zero-event': r['samples'][0]['event_ms'] = r['samples'][0]['device_span_us'] = 0
    elif bad == 'missing-warm': r['warm_stability'].clear()
    elif bad == 'one-warm': r['warm_stability'][probe.PERF_CASES[0]].pop()
    elif bad == 'extra-warm': r['warm_stability'][probe.PERF_CASES[0]].append(snapshot())
    else: r['warm_stability'][probe.PERF_CASES[0]][1]['generated_kernel_count'] += 1
    with pytest.raises(ValueError): probe.validate_performance(r, 0, 'a'*64)


@pytest.mark.parametrize('unstable', [False, True])
def test_actual_run_orchestration_with_stub_vendor(probe, monkeypatch, tmp_path, unstable):
    """Exercise real orchestration/clocks/validator, not numerical or GPU correctness."""
    from contextlib import contextmanager
    from apertus_amd import runtime
    s = plan(ROOT/'experiments/chonk-output-gate-performance-i910bfcf936d3', site=ROOT/'config.toml', submit=True, reservation='cscs', time='00:15:00')
    s['scratch_root'] = str(tmp_path/'scratch')
    env = dict(s['environment'], SLURM_PROCID='0', SLURM_LOCALID='0', SLURM_NODEID='0', SLURM_NTASKS='4', SLURM_JOB_ID='fixture')
    monkeypatch.chdir(tmp_path); monkeypatch.setattr(probe.os, 'environ', env)
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda _: set(range(48)))
    monkeypatch.setattr(probe.importlib.metadata, 'version', probe.VERSIONS.__getitem__)
    trace, allocated, numeric_inputs, counter_calls = [], [], [], []
    class Tensor:
        grad = None
        def clone(self): return self
        def detach(self): return self
        def requires_grad_(self): return self
    def operation(x, g):
        trace.append('pair')
        return SimpleNamespace(backward=lambda dy: None)
    event = lambda **kw: SimpleNamespace(record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: .004)
    cuda = SimpleNamespace(device_count=lambda: 4, set_device=lambda _: None, current_device=lambda: 0,
        get_device_properties=lambda _: SimpleNamespace(gcnArchName='gfx942', multi_processor_count=228),
        Event=event, synchronize=lambda *a: None, reset_peak_memory_stats=lambda _: None,
        max_memory_allocated=lambda _: 10, max_memory_reserved=lambda _: 20)
    def equal(value, old):
        trace.append('equal'); return value is old
    torch = SimpleNamespace(__version__=probe.VERSIONS['torch'], version=SimpleNamespace(git_version=probe.GIT, hip=probe.HIP),
        _dynamo=SimpleNamespace(config=SimpleNamespace(suppress_errors=False)), cuda=cuda,
        set_num_threads=lambda _: None, manual_seed=lambda seed: trace.append(('seed', seed)), compile=lambda fn: operation, equal=equal)
    original_import = builtins.__import__
    monkeypatch.setattr(builtins, '__import__', lambda name, *a, **kw: torch if name == 'torch' else original_import(name, *a, **kw))
    @contextmanager
    def affinity(record): yield set(range(48))
    monkeypatch.setattr(runtime, 'preserve_affinity', affinity)
    monkeypatch.setattr(runtime, 'require_affinity', lambda *a: None)
    monkeypatch.setattr(probe, 'source_tree', lambda: ast.parse('def _apply_output_gate(x, g): pass'))
    original_spec = probe.importlib.util.spec_from_file_location
    def helper_spec(name, path):
        if name != 'frozen_output_gate_perf': return original_spec(name, path)
        def load(module): module.__file__ = str(path); module.sigmoid_gate = operation
        return SimpleNamespace(name=name, loader=SimpleNamespace(create_module=lambda spec: None, exec_module=load), origin=str(path), has_location=True, cached=None, submodule_search_locations=None)
    monkeypatch.setattr(probe.importlib.util, 'spec_from_file_location', helper_spec)
    def tensors(torch, case, device):
        allocated.append(case); return [Tensor(), Tensor(), Tensor()]
    monkeypatch.setattr(probe, 'tensors', tensors)
    def numerical(torch, baseline, candidate, record, device, provided):
        trace.append('pre' if not numeric_inputs else 'post'); numeric_inputs.append(provided)
        record['stub_numerical'] = True
    monkeypatch.setattr(probe, 'numerical', numerical)
    monkeypatch.setattr(probe, 'validate_record', lambda r, *a: r['stub_numerical'])
    monkeypatch.setattr(probe, 'counter_sources', lambda: (None, None, valid_record(probe)['counter_sources']))
    def counters(*args):
        counter_calls.append(len(trace))
        result = snapshot(0 if len(counter_calls) == 1 else 1)
        if unstable and len(counter_calls) == 4: result['generated_kernel_count'] += 1
        return result
    monkeypatch.setattr(probe, 'counters', counters)
    ticks = iter(range(1000000, 1000000000, 10000))
    monkeypatch.setattr(probe.time, 'perf_counter_ns', lambda: next(ticks))
    # validate_performance, iterate, sample, decision, run and save remain real.
    if unstable:
        with pytest.raises(ValueError, match='warm compiler/cache instability'): probe.run(s)
    else: probe.run(s)
    r = json.loads((tmp_path/'output-gate-performance-0.json').read_text())
    assert allocated == list(probe.CASES) and numeric_inputs[0] is numeric_inputs[1]
    assert trace.index('pre') < trace.index('pair') < trace.index('post')
    if unstable:
        assert r['status'] == 'unqualified' and r['samples'] == [] and len(counter_calls) == 4
    else:
        probe.validate_performance(r, 0, s['validation']['sha256'])
        assert r['status'] == 'passed' and len(r['samples']) == 96 and r['operands_unchanged']
        assert trace.count('pair') == 2*(2*20 + 2*4*16) + 96*16
        assert trace.count('equal') == 12 and trace.index('equal') < trace.index('post')
        assert all(len(v) == 2 for v in r['warm_stability'].values())
    for phase in ('pre', 'post'):
        path = tmp_path/f'output-gate-{phase}-0.json'
        assert json.loads(path.read_text())['status'] == 'passed'
        assert probe.sha(path.read_bytes()) == r['phases'][phase]['sha256']
    if not unstable:
        # Audit synthetic four-rank records through the real frozen offline checker.
        directory = tmp_path/'frozen-run'; output = directory/'output/0'; output.mkdir(parents=True)
        (directory/'validation.py').write_bytes(SCRIPT.read_bytes())
        (directory/'resolved.json').write_text(json.dumps(s))
        for rank in range(4):
            record = deepcopy(r); record['rank'] = rank
            for state in list(record['compiler'].values()) + [state for values in record['warm_stability'].values() for state in values]:
                for devices in state['triton_keys'].values(): devices[str(rank)] = devices.pop('0')
            templates = iter(record['samples'])
            record['samples'] = [dict(next(templates), case=case, block=block, slot=slot, arm=arm)
                for case in probe.case_order(rank) for block in range(12) for slot, arm in enumerate(probe.order(rank, block))]
            record['decision'] = probe.decision(record['samples'], rank)
            for phase in ('pre', 'post'):
                path = output/f'output-gate-{phase}-{rank}.json'
                path.write_bytes((tmp_path/f'output-gate-{phase}-0.json').read_bytes())
                record['phases'][phase].update(path=path.name, sha256=probe.sha(path.read_bytes()))
            (output/f'output-gate-performance-{rank}.json').write_text(json.dumps(record))
        probe.check(directory)
        assert json.loads((directory/'output/performance-checks.json').read_text())['status'] == 'passed'
