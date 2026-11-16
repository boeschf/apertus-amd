"""CPU policy/provenance/reference contracts; these tests do not establish GPU correctness."""

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import tomllib
from types import SimpleNamespace

import pytest
from test_memory_trace import AUDIT, RESTART_AUDIT, ContextConfig, backend_runtime, policy_record, restart_runtime

from apertus_amd.artifacts import sha256
from apertus_amd.cli import plan
from apertus_amd.config import resolve


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / 'experiments/sssglu-compile-probe-m1072583a9632-icb5a384016a9'
FULL = ROOT / 'experiments/moe-chonk-kda-pr73-sc-m1072583a9632-icb5a384016a9'
RESTART_CPU = ROOT / 'experiments/inductor-restart-probe-icb5a384016a9'
EAGER = ROOT / 'experiments/moe-chonk-kda-pr73-eager-m1072583a9632-icb5a384016a9'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'experiments/validation/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe():
    return load('sssglu_compile_probe')


def successful_record(probe):
    return {'status': 'passed', 'source': probe.SOURCE, 'image': probe.IMAGE, 'torch': 'stub',
            'case_isolation': {'method': 'torch._dynamo.reset_between_families',
                               'limits_start': deepcopy(probe.DYNAMO_LIMITS), 'limits_end': deepcopy(probe.DYNAMO_LIMITS),
                               'families': [{'family': family, 'limits_before': deepcopy(probe.DYNAMO_LIMITS),
                                             'limits_after': deepcopy(probe.DYNAMO_LIMITS)}
                                            for family, _, _ in probe.VARIANTS],
                               'resets': [{'after_family': previous, 'before_family': following,
                                           'limits_before': deepcopy(probe.DYNAMO_LIMITS),
                                           'limits_after': deepcopy(probe.DYNAMO_LIMITS)}
                                          for previous, following in (('dense', 'shared'), ('shared', 'expert'))]},
            'initial_compiler_caches': {key: {'files': 0, 'path': '/private/' + key}
                                        for key in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')},
            'checks': [{'label': label, 'passed': True} for label in sorted(probe.required_labels())],
            'native_calls': [{'label': label, 'observed_compiled_launches': 1}
                             for label in sorted(probe.required_native_labels())],
            'policy': policy_record(probe),
            'autotuners': {'run_calls': len(probe.required_native_labels()), 'benchmark_all_configs_calls': 0,
                           'kernels': [{'family': label.split('/')[0], 'native_call': label,
                                        'kernel': label, 'heuristic': 'REDUCTION' if label.startswith('expert/') and label.endswith('backward') else 'POINTWISE',
                                        'config_count': None, 'config_state': 'released', 'launcher_count_before': 1, 'launcher_count': 1,
                                       'policy_meta': {'autotune_pointwise': False,
                                                       'force_filter_reduction_configs': True,
                                                       'dynamic_scale_rblock': False}}
                                       for label in sorted(probe.required_native_labels())]}}


def test_full_experiment_matches_eager_except_compile_and_explicit_policy():
    eager = tomllib.loads((EAGER / 'experiment.toml').read_text())
    selected = tomllib.loads((FULL / 'experiment.toml').read_text())
    eager['environment']['TORCH_COMPILE_DISABLE'] = '0'
    eager['environment']['APERTUS_INDUCTOR_POLICY'] = 'single_candidate_backend'
    assert selected == eager
    assert len(FULL.name) <= 64


def test_component_plan_is_frozen_train_kind_and_requires_cold_at_execution(probe, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('planning must not launch anything'))
    settings = plan(COMPONENT, submit=True, cache_mode='cold')
    probe.validate(settings)
    assert settings['resources']['nodes'] == 1 and settings['resources']['gpus_per_node'] == 4
    assert settings['args'] == [] and settings['prepare'] == []
    assert settings['validation']['sha256'] == sha256(Path(settings['validation']['path']))
    assert settings['environment']['APERTUS_SSSGLU_TIMEOUT_SECONDS'] == '900'
    assert settings['environment']['TORCH_COMPILE_DISABLE'] == '0'
    with pytest.raises(ValueError, match='cold'):
        probe.validate(resolve(COMPONENT, submit=True))


def test_standalone_and_training_policy_dictionaries_and_guards_match(probe):
    trace = load('memory_trace')
    assert probe.POLICY_FLAGS == trace.POLICY_FLAGS
    assert probe.POLICY_GUARDS == trace.POLICY_GUARDS
    assert probe.OPTIONS_ID == trace.OPTIONS_ID and probe.BACKEND_NAME == trace.BACKEND_NAME
    names = {'config_values', 'policy_guards', 'single_candidate_backend', 'BackendPolicy', 'configure_inductor_policy',
             '_configure_inductor_policy', 'finish_inductor_policy', 'check_inductor_policy', 'compilation_key'}
    def definitions(module):
        result = {node.name: ast.dump(node) for node in ast.parse(Path(module.__file__).read_text()).body
                  if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
        assert set(result) == names
        return result
    assert definitions(probe) == definitions(trace)
    constants = {'POLICY_NAME', 'BACKEND_NAME', 'POLICY_FLAGS', 'OPTIONS_ID', 'POLICY_SCHEMA', 'MAX_RESTARTS',
                 'RESTART_SOURCE', 'POLICY_GUARDS', '_policy_owner', '_registered_delegate', '_setup_lock'}
    def declarations(module):
        result = {target.id: ast.dump(node) for node in ast.parse(Path(module.__file__).read_text()).body
                  if isinstance(node, ast.Assign) for target in node.targets
                  if isinstance(target, ast.Name) and target.id in constants}
        assert set(result) == constants
        return result
    assert declarations(probe) == declarations(trace)
    assert probe.POLICY_SCHEMA == trace.POLICY_SCHEMA == 2
    assert probe.MAX_RESTARTS == trace.MAX_RESTARTS == 100
    assert probe.RESTART_SOURCE == trace.RESTART_SOURCE
    for name in ('hashlib', 'importlib', 'json', 'os', 'Path', 'sys', 'threading', 'traceback', 'MappingProxyType'):
        assert getattr(probe, name) is getattr(trace, name)


def test_autotuner_observer_delegates_unchanged_and_restores_on_exception(probe, tmp_path):
    calls = []
    sentinel = object()

    class Autotuner:
        configs = [object()]
        launchers = [object()]
        heuristic_type = 'POINTWISE'
        fn = SimpleNamespace(__name__='native_kernel')
        inductor_meta = {'autotune_pointwise': False, 'force_filter_reduction_configs': True,
                         'dynamic_scale_rblock': False}

        def run(self, *args, **kwargs):
            calls.append(('run', args, kwargs))
            return sentinel

        def benchmark_all_configs(self, *args, **kwargs):
            calls.append(('bench', args, kwargs))
            return sentinel

    before = Autotuner.run, Autotuner.benchmark_all_configs
    evidence = probe.Evidence(tmp_path, 0, {'checks': []})
    try:
        with pytest.raises(RuntimeError, match='injected'):
            with probe.observe_autotuners(Autotuner, evidence) as stats:
                instance = Autotuner()
                assert instance.run(1, stream=3) is sentinel
                assert instance.benchmark_all_configs(2, stream=4) is sentinel
                assert calls == [('run', (1,), {'stream': 3}), ('bench', (2,), {'stream': 4})]
                assert stats['run_calls'] == 1 and stats['benchmark_all_configs_calls'] == 1
                assert stats['kernels'][0]['launcher_count'] == 1
                raise RuntimeError('injected')
        assert (Autotuner.run, Autotuner.benchmark_all_configs) == before
        assert 'benchmark-observed' in (tmp_path / 'sssglu-compile-0.jsonl').read_text()
    finally:
        evidence.close()


@pytest.mark.parametrize('defect', ['coverage', 'numerical', 'benchmark', 'candidates', 'pre-candidates', 'meta', 'metadata-context', 'policy', 'torch', 'reduction', 'eager', 'cache', 'family', 'limit', 'reset'])
def test_component_outcome_rejects_missing_or_failed_evidence(probe, defect):
    record = successful_record(probe)
    probe.validate_outcome(record)
    if defect == 'coverage': record['checks'].pop()
    elif defect == 'numerical': record['checks'][0]['passed'] = False
    elif defect == 'benchmark': record['autotuners']['benchmark_all_configs_calls'] = 1
    elif defect == 'candidates': record['autotuners']['kernels'][0]['launcher_count'] = 2
    elif defect == 'pre-candidates': record['autotuners']['kernels'][0]['launcher_count_before'] = 5
    elif defect == 'meta': record['autotuners']['kernels'][0]['policy_meta']['autotune_pointwise'] = True
    elif defect == 'metadata-context': record['autotuners']['kernels'][0]['native_call'] = 'unknown'
    elif defect == 'policy': record['policy']['options'] = {}
    elif defect == 'torch': record['policy']['torch'] = 'other'
    elif defect == 'reduction':
        record['autotuners']['kernels'] = [row for row in record['autotuners']['kernels'] if row['heuristic'] == 'POINTWISE']
    elif defect == 'eager': record['native_calls'][0]['observed_compiled_launches'] = 0
    elif defect == 'cache': record['initial_compiler_caches']['TRITON_CACHE_DIR']['files'] = 1
    elif defect == 'family': record['autotuners']['kernels'] = record['autotuners']['kernels'][:2]
    elif defect == 'limit': record['case_isolation']['limits_end']['recompile_limit'] = 64
    elif defect == 'reset': record['case_isolation']['resets'].pop()
    with pytest.raises(ValueError):
        probe.validate_outcome(record)


@pytest.mark.parametrize('precompiled', [False, True])
@pytest.mark.parametrize('initial_launchers', [None, [], ['already-created']])
def test_observer_accepts_native_released_configs_and_nullable_launcher_lifecycle(
    probe, tmp_path, precompiled, initial_launchers
):
    sentinel = object()

    class Autotuner:
        heuristic_type = 'POINTWISE'
        # prepare_for_pickle releases fn.fn, not the named wrapper itself.
        fn = SimpleNamespace(__name__='worker_kernel', fn=None)
        inductor_meta = {'autotune_pointwise': False, 'force_filter_reduction_configs': True,
                         'dynamic_scale_rblock': False}

        def __init__(self):
            self.configs = None if precompiled else [object()]
            self.launchers = initial_launchers
            self.calls = 0

        def run(self, *args, **kwargs):
            if self.calls == 0:
                assert self.launchers is initial_launchers  # Observer has not populated anything.
                assert (self.configs is None) is precompiled
            self.calls += 1
            self.configs = None  # Actual _precompile_worker lifecycle, line 835.
            self.launchers = [object()]
            return sentinel

        def precompile(self):
            pytest.fail('observer must never initiate compilation')

        def benchmark_all_configs(self, *args, **kwargs):
            pytest.fail('observer must never initiate benchmarking')

    label = sorted(probe.required_native_labels())[0]
    evidence = probe.Evidence(tmp_path, 0, {'checks': [], 'active_family': 'dense', 'active_native_call': label})
    try:
        instance = Autotuner()
        with probe.observe_autotuners(Autotuner, evidence) as stats:
            assert instance.run(1, stream=3) is sentinel
            assert instance.run(2, stream=4) is sentinel
        first = stats['kernels'][0]
        assert first['config_count_before'] == (None if precompiled else 1)
        assert first['config_count'] is None and first['config_state'] == 'released'
        assert first['launcher_count_before'] == (None if initial_launchers is None else len(initial_launchers))
        assert first['launcher_count'] == 1 and first['launcher_state'] == 'present'
        assert stats['run_calls'] == 2 and stats['benchmark_all_configs_calls'] == 0
        record = successful_record(probe)
        record['autotuners']['kernels'][0] = first
        if initial_launchers:
            probe.validate_outcome(record)
        else:
            with pytest.raises(ValueError, match='pre/post-run launcher'):
                probe.validate_outcome(record)
    finally:
        evidence.close()


@pytest.mark.parametrize('method', ['run', 'benchmark_all_configs'])
def test_observer_delegates_original_exception_and_restores_methods(probe, tmp_path, method):
    error = RuntimeError(f'native {method} failed')

    class Autotuner:
        configs = None
        launchers = None

        def run(self, *args, **kwargs):
            assert method == 'run'
            assert args == (1,) and kwargs == {'stream': 2}
            raise error

        def benchmark_all_configs(self, *args, **kwargs):
            assert method == 'benchmark_all_configs'
            assert args == (1,) and kwargs == {'stream': 2}
            raise error

    original = Autotuner.run, Autotuner.benchmark_all_configs
    evidence = probe.Evidence(tmp_path, 0, {'checks': []})
    try:
        with pytest.raises(RuntimeError) as caught:
            with probe.observe_autotuners(Autotuner, evidence) as stats:
                getattr(Autotuner(), method)(1, stream=2)
        assert caught.value is error
        assert (Autotuner.run, Autotuner.benchmark_all_configs) == original
        assert stats['run_calls'] == 0 and stats['kernels'] == []
        assert stats['benchmark_all_configs_calls'] == (method == 'benchmark_all_configs')
    finally:
        evidence.close()


@pytest.mark.parametrize('missing', ['launchers', 'fn', 'heuristic_type', 'inductor_meta'])
def test_unavailable_required_postrun_metadata_is_reported_then_rejected(probe, tmp_path, missing):
    class Autotuner:
        configs = None
        launchers = [object()]
        fn = SimpleNamespace(__name__='native_kernel', fn=None)
        heuristic_type = 'POINTWISE'
        inductor_meta = {'autotune_pointwise': False, 'force_filter_reduction_configs': True,
                         'dynamic_scale_rblock': False}

        def run(self):
            setattr(self, missing, None)
            return 'native-result'

        def benchmark_all_configs(self):
            pytest.fail('not called')

    evidence = probe.Evidence(tmp_path, 0, {'checks': []})
    try:
        with probe.observe_autotuners(Autotuner, evidence) as stats:
            assert Autotuner().run() == 'native-result'
        row = stats['kernels'][0]
        assert json.loads(evidence.path.read_text())['autotuners']['kernels'][0] == row
        record = successful_record(probe)
        record['autotuners']['kernels'][0] = row
        with pytest.raises(ValueError, match='must be observed'):
            probe.validate_outcome(record)
    finally:
        evidence.close()


def test_case_coverage_and_fixed_precision_budget(probe):
    assert probe.VARIANTS == (('dense', 17280, False), ('shared', 1920, False), ('expert', 1920, True))
    assert probe.ROWS == (4096, 129, 257) and probe.REPEATS == 2
    assert probe.REFERENCE_ROWS < min(probe.ROWS)
    assert len(probe.required_labels()) == 63
    assert len(probe.required_native_labels()) == 54
    metrics = {'finite': True, 'mismatched_elements': 0, 'rms_error': 0.001, 'reference_rms': 1,
               'relative_l2': 0.001, 'max_abs': 0.001}
    assert probe.metric_failures(metrics) == []
    assert probe.metric_failures(metrics | {'mismatched_elements': 1}) == ['elementwise precision gate']
    assert probe.metric_failures(metrics | {'rms_error': 0.1}) == ['RMS precision gate']
    assert probe.metric_failures(metrics | {'rms_error': math.nan}) == ['nonfinite values or metrics']


class Matrix:
    """Tiny eager scalar algebra used only to test the analytic reference formula."""

    def __init__(self, rows): self.rows = rows
    def double(self): return self
    def apply(self, fn): return Matrix([[fn(x) for x in row] for row in self.rows])
    def binary(self, other, fn):
        if not isinstance(other, Matrix): return self.apply(lambda x: fn(x, other))
        return Matrix([[fn(x, other.rows[i][j if len(other.rows[i]) > 1 else 0])
                        for j, x in enumerate(row)] for i, row in enumerate(self.rows)])
    def __add__(self, value): return self.binary(value, lambda a, b: a + b)
    def __radd__(self, value): return self + value
    def __sub__(self, value): return self.binary(value, lambda a, b: a - b)
    def __mul__(self, value): return self.binary(value, lambda a, b: a * b)
    def __truediv__(self, value): return self.binary(value, lambda a, b: a / b)
    def abs(self): return self.apply(abs)
    def reciprocal(self): return self.apply(lambda x: 1 / x)
    def square(self): return self.apply(lambda x: x * x)
    def chunk(self, count, dim):
        assert count == 2 and dim == -1
        half = len(self.rows[0]) // 2
        return Matrix([row[:half] for row in self.rows]), Matrix([row[half:] for row in self.rows])
    def sum(self, dim, keepdim):
        assert dim == -1 and keepdim
        return Matrix([[sum(row)] for row in self.rows])


@pytest.mark.parametrize('weighted', [False, True])
def test_independent_analytic_reference_against_finite_differences(probe, weighted):
    torch = SimpleNamespace(cat=lambda values, dim: Matrix([a + b for a, b in zip(values[0].rows, values[1].rows)]))
    values, dy, probability = [0.25, 1.8, 0.3, -0.6], [0.4, -0.2], 0.7

    def output(y, p):
        return [(0.5 + (a - 1) / (1 + abs(a - 1))) * b * p for a, b in zip(y[:2], y[2:])]

    def loss(y, p): return sum(a * b for a, b in zip(output(y, p), dy))

    refs, scale = probe.analytic_reference(torch, Matrix([values]), Matrix([dy]),
                                          Matrix([[probability]]) if weighted else None)
    p = probability if weighted else 1.0
    assert refs['forward'].rows[0] == pytest.approx(output(values, p))
    epsilon = 1e-6
    for index in range(4):
        plus, minus = values.copy(), values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        derivative = (loss(plus, p) - loss(minus, p)) / (2 * epsilon)
        assert refs['input-gradient'].rows[0][index] == pytest.approx(derivative, abs=1e-9)
    if weighted:
        assert refs['probability-gradient'].rows[0][0] == pytest.approx(
            (loss(values, p + epsilon) - loss(values, p - epsilon)) / (2 * epsilon), abs=1e-9)
        assert scale.rows[0][0] >= abs(refs['probability-gradient'].rows[0][0])


def test_each_native_call_requires_observed_compiled_execution(probe, tmp_path):
    evidence = probe.Evidence(tmp_path, 0, {'checks': [], 'native_calls': [], 'autotuners': {'run_calls': 0}})
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    value, calls = object(), []

    def operation():
        calls.append('native')
        evidence.record['autotuners']['run_calls'] += 2
        return value

    try:
        assert probe.native_call(torch, operation, 'compiled', evidence) is value
        assert calls == ['native']
        assert evidence.record['native_calls'][0]['observed_compiled_launches'] == 2
        with pytest.raises(ValueError, match='possible eager fallback'):
            probe.native_call(torch, lambda: None, 'unobserved', evidence)
        assert json.loads(evidence.path.read_text())['last_event']['observed_compiled_launches'] == 0
    finally:
        evidence.close()


def test_family_resets_are_synchronized_only_between_complete_families(probe, tmp_path, monkeypatch):
    calls = []
    config = SimpleNamespace(**probe.DYNAMO_LIMITS)
    torch = SimpleNamespace(_dynamo=SimpleNamespace(config=config, reset=lambda: calls.append('reset')),
                            cuda=SimpleNamespace(synchronize=lambda: calls.append('synchronize')))

    def case(torch, native, case, evidence):
        family, _, _, rows = case
        assert evidence.record['active_family'] == family
        calls.append((family, rows))

    monkeypatch.setattr(probe, 'run_case', case)
    evidence = probe.Evidence(tmp_path, 0, {'checks': []})
    try:
        probe.run_families(torch, object(), evidence)
        assert calls == [('dense', 4096), ('dense', 129), ('dense', 257), 'synchronize', 'reset',
                         ('shared', 4096), ('shared', 129), ('shared', 257), 'synchronize', 'reset',
                         ('expert', 4096), ('expert', 129), ('expert', 257)]
        assert vars(config) == probe.DYNAMO_LIMITS
        assert evidence.record['case_isolation'] == successful_record(probe)['case_isolation']
        assert 'active_family' not in evidence.record
        events = [json.loads(line) for line in (tmp_path / 'sssglu-compile-0.jsonl').read_text().splitlines()]
        assert len([row for row in events if row['kind'] == 'family-reset']) == 2
    finally:
        evidence.close()


@pytest.mark.parametrize('failure', ['case', 'synchronize', 'reset', 'limit-before', 'limit-during', 'limit-within'])
def test_family_isolation_propagates_failures_without_retry_or_budget_increase(probe, tmp_path, monkeypatch, failure):
    calls = []
    config = SimpleNamespace(**probe.DYNAMO_LIMITS)
    error = RuntimeError('injected lifecycle failure')

    def synchronize():
        calls.append('synchronize')
        if failure == 'synchronize': raise error

    def reset():
        calls.append('reset')
        if failure == 'reset': raise error
        if failure == 'limit-during': config.recompile_limit += 1

    def case(torch, native, case, evidence):
        calls.append(case)
        if failure == 'case': raise error
        if failure == 'limit-within': config.recompile_limit += 1

    if failure == 'limit-before': config.recompile_limit += 1
    torch = SimpleNamespace(_dynamo=SimpleNamespace(config=config, reset=reset),
                            cuda=SimpleNamespace(synchronize=synchronize))
    monkeypatch.setattr(probe, 'run_case', case)
    evidence = probe.Evidence(tmp_path, 0, {'checks': []})
    try:
        with pytest.raises(ValueError if failure.startswith('limit') else RuntimeError) as caught:
            probe.run_families(torch, object(), evidence)
        if not failure.startswith('limit'): assert caught.value is error
        assert all(not isinstance(call, tuple) or call[0] == 'dense' for call in calls)
        assert calls.count('reset') <= 1
        if failure in ('case', 'limit-before', 'limit-within'): assert 'reset' not in calls
    finally:
        evidence.close()


def test_checker_requires_four_verified_rank_reports_and_requested_policy(probe, tmp_path):
    settings = resolve(COMPONENT, submit=True, cache_mode='cold')
    (tmp_path / 'output').mkdir()
    (tmp_path / 'source').mkdir()
    (tmp_path / 'resolved.json').write_text(json.dumps(settings))
    (tmp_path / 'source/manifest.json').write_text(json.dumps({'files': {
        'megatron/core/fusions/fused_bias_sssglu.py': 'native-source-hash'}}))
    for rank in range(4):
        record = successful_record(probe) | {'rank': rank, 'local_rank': rank, 'verified_modules': 4,
                                           'probe_sha256': settings['validation']['sha256'],
                                           'native_file_sha256': 'native-source-hash'}
        (tmp_path / f'output/sssglu-compile-{rank}.json').write_text(json.dumps(record))
    probe.check(tmp_path)
    checks = json.loads((tmp_path / 'output/checks.json').read_text())
    assert len(checks['ranks']) == 4 and 'component' in checks['scope']
    path = tmp_path / 'output/sssglu-compile-3.json'
    record = json.loads(path.read_text())
    record['policy']['options'] = {}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='policy identity'):
        probe.check(tmp_path)


@pytest.mark.parametrize('fails', [False, True])
def test_train_never_executes_gpt_and_verifies_imports_on_failure(probe, tmp_path, monkeypatch, fails):
    source, run, output = tmp_path / 'source', tmp_path / 'run', tmp_path / 'output'
    source.mkdir(); (run / 'source').mkdir(parents=True)
    entry = source / 'pretrain_gpt.py'
    entry.write_text("raise AssertionError('must not execute GPT')\n")
    native_file = source / 'fused_bias_sssglu.py'
    native_file.write_text('# stub import identity only\n')
    settings = resolve(COMPONENT, submit=True, cache_mode='cold')
    (run / 'resolved.json').write_text(json.dumps(settings))
    (run / 'source/manifest.json').write_text(json.dumps({'selection': {'commit': probe.SOURCE},
        'files': {'pretrain_gpt.py': sha256(entry)}}))
    for key in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
        path = tmp_path / key; path.mkdir(); monkeypatch.setenv(key, str(path))
    monkeypatch.setenv('RANK', '0'); monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('APERTUS_SSSGLU_TIMEOUT_SECONDS', '900')
    calls = []
    runtime = backend_runtime(probe, monkeypatch)
    torch = runtime.torch
    torch.manual_seed = lambda seed: None
    torch._dynamo = SimpleNamespace(config=SimpleNamespace(**probe.DYNAMO_LIMITS), reset=lambda: calls.append('reset'))
    torch.cuda = SimpleNamespace(set_device=lambda rank: calls.append(('device', rank)),
                                 get_device_name=lambda: 'stub', synchronize=lambda: calls.append('sync'))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch._inductor.runtime.triton_heuristics', SimpleNamespace(CachingAutotuner=object))
    def compiled(): pass
    compiled._torchdynamo_orig_callable = object()
    native = SimpleNamespace(__file__=str(native_file), **{name: compiled for name in
        ('sssglu', 'sssglu_back', 'weighted_sssglu', 'weighted_sssglu_back')})

    def imported(name):
        if name in runtime.imports: return runtime.imports[name]
        if name == 'torch._guards': raise ModuleNotFoundError('CPU training stub has no Dynamo context API')
        assert runtime.registry.get_default_backend() == probe.BACKEND_NAME
        if name == 'megatron.core.jit': return SimpleNamespace(jit_fuser=torch.compile)
        assert name == 'megatron.core.fusions.fused_bias_sssglu'
        calls.append('native')
        return native

    @contextmanager
    def observed(cls, evidence):
        evidence.record['autotuners'] = successful_record(probe)['autotuners']
        yield evidence.record['autotuners']

    def case(torch, native, case, evidence):
        if fails: raise RuntimeError('injected component failure')
        probe.single_candidate_backend(object(), [])
        variant, _, _, rows = case
        for label in sorted(probe.required_labels()):
            if label.startswith(f'{variant}/rows{rows}/'): evidence.emit('check', label=label, passed=True)
        for label in sorted(probe.required_native_labels()):
            if label.startswith(f'{variant}/rows{rows}/'): evidence.emit('native-call', label=label, observed_compiled_launches=1)

    monkeypatch.setattr(probe.importlib, 'import_module', imported)
    monkeypatch.setattr(probe, 'observe_autotuners', observed)
    monkeypatch.setattr(probe, 'run_case', case)
    monkeypatch.setattr(probe, 'verify_imports', lambda *args: calls.append('verify') or 4)
    monkeypatch.setattr(probe.faulthandler, 'dump_traceback_later', lambda seconds, exit: calls.append(('watchdog', seconds, exit)))
    monkeypatch.setattr(probe.faulthandler, 'cancel_dump_traceback_later', lambda: calls.append('cancel-watchdog'))
    if fails:
        with pytest.raises(RuntimeError, match='injected'):
            probe.train(output, source, 'pretrain_gpt.py', [], run=run)
    else:
        probe.train(output, source, 'pretrain_gpt.py', [], run=run)
    report = json.loads((output / 'sssglu-compile-0.json').read_text())
    assert report['status'] == ('failed' if fails else 'passed')
    assert calls.count('verify') == 3 and calls[-1] == 'cancel-watchdog'
    assert ('watchdog', 900, True) in calls
    assert runtime.registry.get_default_backend() == 'inductor'
    assert report['policy']['default_start'] == probe.BACKEND_NAME
    assert report['verified_modules'] == 4
    if not fails:
        assert report['policy']['schema_version'] == 2
        for row in report['policy']['compile_calls']:
            assert row['identity'] is None and row['identity_after'] is None
            assert row['observation_errors'] == []


def restart_cpu_settings():
    return resolve(RESTART_CPU, submit=True, reservation='cscs', gpus_per_node=1,
                   cpus_per_task=4, time='00:05:00')


def test_restart_cpu_commands_plan_is_cpu_work_not_zero_gpu_allocation(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('plan must not execute a command'))
    settings = plan(RESTART_CPU, submit=True, reservation='cscs', gpus_per_node=1,
                    cpus_per_task=4, time='00:05:00')
    probe.validate_restart_cpu(settings)
    assert settings['procedure']['commands'] == [probe.RESTART_CPU_COMMAND]
    assert settings['procedure']['timeout'] == 180
    assert not settings.get('source') and not settings.get('source_snapshot')
    assert settings['resources']['cpu_bind'] == 'cores'
    assert settings['validation']['sha256'] == sha256(Path(probe.__file__))
    from apertus_amd.runs import edf
    config = tomllib.loads(edf(settings, tmp_path))
    assert any(mount.endswith(':/apertus/validation.py:ro') for mount in config['mounts'])
    assert config['env']['TENSORIFY_PYTHON_SCALARS'] == '1'
    assert config['annotations']['com']['hooks']['cxi']['enabled'] == 'false'
    assert config['annotations']['com']['hooks']['aws_ofi_nccl']['enabled'] == 'false'
    assert config['env']['HWLOC_COMPONENTS'] == '-gl'
    with pytest.raises(ValueError, match='commands do not use'):
        resolve(RESTART_CPU, submit=True, cache_mode='cold')


@pytest.mark.parametrize('defect', ['image', 'command', 'timeout', 'ranks', 'cpus', 'time', 'source',
                                    'cache', 'environment', 'validation', 'comparison', 'reservation', 'binding'])
def test_restart_cpu_protocol_is_not_a_broader_sssglu_validator(probe, defect):
    settings = restart_cpu_settings()
    if defect == 'image': settings['image']['sha256'] = '0' * 64
    elif defect == 'command': settings['procedure']['commands'][0].append('--variant')
    elif defect == 'timeout': settings['procedure']['timeout'] = 181
    elif defect == 'ranks': settings['resources']['gpus_per_node'] = 4
    elif defect == 'cpus': settings['resources']['cpus_per_task'] = 8
    elif defect == 'time': settings['resources']['time'] = '00:06:00'
    elif defect == 'source': settings['source'] = {'commit': probe.SOURCE}
    elif defect == 'cache': settings['cache']['mode'] = 'cold'
    elif defect == 'environment': settings['environment']['TENSORIFY_PYTHON_SCALARS'] = '0'
    elif defect == 'validation': settings['validation']['sha256'] = 'other'
    elif defect == 'comparison': settings['compare_run'] = '/previous'
    elif defect == 'reservation': settings['resources']['reservation'] = 'other'
    elif defect == 'binding': settings['resources']['cpu_bind'] = 'none'
    with pytest.raises(ValueError): probe.validate_restart_cpu(settings)


@pytest.mark.parametrize('mode', ['train', 'check', 'restart-cpu'])
def test_mode_dispatch_has_no_implicit_torch_import(probe, monkeypatch, tmp_path, mode):
    calls = []
    monkeypatch.setattr(probe.importlib, 'import_module', lambda *a: pytest.fail('dispatch imported Torch'))
    monkeypatch.setattr(probe, mode.replace('-', '_'), lambda *args: calls.append(args))
    args = ['train', str(tmp_path), str(tmp_path / 'source'), 'pretrain_gpt.py', '--flag'] if mode == 'train' else [mode, str(tmp_path)]
    probe.main(args)
    assert calls and calls[0][0] == tmp_path
    if mode == 'train': assert calls[0][1:] == (tmp_path / 'source', 'pretrain_gpt.py', ['--flag'])


@pytest.mark.parametrize('tamper', [False, True])
def test_restart_cpu_actual_raise_path_attestation_one_byte_change(probe, tmp_path, tamper):
    extra = ROOT / 'artifacts/investigations/performance-next/restart-20261113/native-restart-preparation/source/torch'
    if not AUDIT.is_dir() or not extra.is_dir(): pytest.skip('optional pinned source snapshots absent')
    target = tmp_path / 'torch'
    for name, expected in probe.RESTART_CPU_SOURCE.items():
        source = extra / name if (extra / name).is_file() else AUDIT / 'torch' / name
        if name == '_dynamo/exc.py': source = RESTART_AUDIT / 'torch-dynamo-exc.py'
        assert sha256(source) == expected
        data = source.read_bytes()
        if tamper and name == 'fx/passes/_tensorify_python_scalars.py': data = bytes([data[0] ^ 1]) + data[1:]
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    torch = SimpleNamespace(__version__=probe.RESTART_CPU_TORCH, __file__=str(target / '__init__.py'),
                            version=SimpleNamespace(git_version=probe.RESTART_CPU_GIT))
    if tamper:
        with pytest.raises(ValueError, match='unaudited'): probe.restart_cpu_sources(torch)
    else:
        assert probe.restart_cpu_sources(torch) == probe.RESTART_CPU_SOURCE
        torch.version.git_version = 'foreign'
        with pytest.raises(ValueError, match='version/git'): probe.restart_cpu_sources(torch)


def cpu_mode_runtime(probe, monkeypatch, tmp_path, *, event=True, primary=None, callback_failure=None):
    """Synthetic CPU lifecycle evidence, not a claim of naturally triggered native branch coverage."""
    from contextvars import ContextVar
    runtime = restart_runtime(probe, monkeypatch)
    if primary == 'subclass':
        primary = type('Child', (runtime.exc.TensorifyScalarRestartAnalysis,), {})('child')
    torch = runtime.torch
    torch.__version__ = probe.RESTART_CPU_TORCH
    torch.version = SimpleNamespace(git_version=probe.RESTART_CPU_GIT)
    torch.cuda = SimpleNamespace(is_initialized=lambda: False,
                                 set_device=lambda *a: pytest.fail('CPU mode selected CUDA'))
    dynamo_values = {**probe.DYNAMO_LIMITS, 'assume_static_by_default': True,
                     'specialize_float': False, 'capture_scalar_outputs': False}
    dynamo = ContextConfig({key: ContextVar('cpu-' + key, default=value) for key, value in dynamo_values.items()})
    runtime.imports.update({'torch': torch, 'torch._dynamo.config': dynamo,
                            'torch._dynamo.utils': SimpleNamespace(_is_tensorify_enabled=lambda: True)})
    monkeypatch.setattr(probe, 'restart_cpu_sources', lambda torch: dict(probe.RESTART_CPU_SOURCE))
    settings = restart_cpu_settings()
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'resolved.json').write_text(json.dumps(settings))
    for key, value in settings['environment'].items(): monkeypatch.setenv(key, value)
    for key, value in {'SLURM_NTASKS': '1', 'SLURM_PROCID': '0', 'SLURM_LOCALID': '0', 'SLURM_JOB_ID': '123456'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('TORCHDYNAMO_DISABLE', raising=False)
    monkeypatch.delenv('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS', raising=False)
    watchdog = []
    monkeypatch.setattr(probe.faulthandler, 'dump_traceback_later', lambda seconds, exit: watchdog.append((seconds, exit)))
    monkeypatch.setattr(probe.faulthandler, 'cancel_dump_traceback_later', lambda: watchdog.append('cancel'))
    cache_root = tmp_path / 'private'
    cache_root.mkdir()
    variables = ('TMPDIR', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR')
    paths = {key: str(cache_root / key) for key in variables}
    for path in paths.values(): Path(path).mkdir()
    monkeypatch.setattr(probe, 'restart_cpu_caches', lambda *a: {'root': str(cache_root), 'environment': paths,
                        'initial_files': {key: 0 for key in variables}, 'mode': 'cold-private'})
    original = runtime.inductor.compile
    restart = runtime.exc.TensorifyScalarRestartAnalysis('CPU stub event')
    def graph(torch, config, record, evidence):
        names = (*probe.RESTART_CPU_DYNAMO, 'capture_scalar_outputs')
        record['dynamo_before'] = probe.config_values(config, names)
        record['fixture_dynamo'] = dict(probe.RESTART_CPU_DYNAMO)
        record['compile_request'] = {'fullgraph': True, 'dynamic': 'unspecified', 'backend': 'selected registered default'}
        try:
            with config.patch(probe.RESTART_CPU_DYNAMO):
                record['dynamo_during'] = probe.config_values(config, names)
                with runtime.guards.compile_context(runtime.context):
                    def failed(*args, **kwargs): raise primary if primary is not None else restart
                    if primary is not None or event:
                        runtime.inductor.compile = failed
                        try: probe.single_candidate_backend(object(), [])
                        except type(restart) as caught:
                            if primary is not None: raise
                            assert caught is restart
                        runtime.context.attempt = 1
                        runtime.inductor.compile = original
                    if callback_failure is not None:
                        def fail_publication(value): raise callback_failure
                        probe._policy_owner.on_update = fail_publication
                    probe.single_candidate_backend(object(), [])
                record['numerics'] = [{'label': label, 'shape': [3, 14, 22], 'device': 'cpu',
                    'dtype': 'torch.float32', 'requires_grad': False, 'finite': True, 'close': True, 'rtol': 1e-5, 'atol': 1e-6}
                    for label in ('first', 'repeat')]
                record['reference'] = {'shape': [3, 14, 22], 'device': 'cpu', 'dtype': 'torch.float32',
                                       'requires_grad': False, 'finite': True}
        finally:
            record['dynamo_after'] = probe.config_values(config, names)
    monkeypatch.setattr(probe, 'run_restart_cpu_graph', graph)
    return SimpleNamespace(runtime=runtime, run=run, output=run / 'output/0', paths=paths,
                           cache_root=cache_root, dynamo=dynamo, watchdog=watchdog, primary=primary)


@pytest.mark.parametrize('event', [False, True])
def test_restart_cpu_stub_success_requires_event_not_just_normal_return(probe, monkeypatch, tmp_path, event):
    stub = cpu_mode_runtime(probe, monkeypatch, tmp_path, event=event)
    before = {key: os.environ.get(key) for key in stub.paths}
    if event:
        record = probe.restart_cpu(stub.output, run=stub.run)
        probe.validate_restart_cpu_outcome(record)
        assert record['status'] == 'passed'
    else:
        with pytest.raises(ValueError, match='not observed'): probe.restart_cpu(stub.output, run=stub.run)
    report = json.loads((stub.output / 'restart-cpu.json').read_text())
    assert report['status'] == ('passed' if event else 'failed')
    assert report['policy']['restored_default'] == 'inductor'
    assert report['dynamo_before'] == report['dynamo_after']
    assert report['environment_restored'] is True
    assert {key: os.environ.get(key) for key in stub.paths} == before
    assert stub.watchdog == [(150, True), 'cancel']
    assert all(Path(path).is_dir() for path in stub.paths.values())
    assert not list(stub.output.glob('sssglu-*'))


@pytest.mark.parametrize('kind', ['ordinary', 'foreign-name', 'subclass', 'callback'])
def test_restart_cpu_serializes_failure_restores_and_never_masks_primary(probe, monkeypatch, tmp_path, kind):
    primary = RuntimeError('original CPU body failure')
    if kind == 'foreign-name': primary = type('TensorifyScalarRestartAnalysis', (RuntimeError,), {})('foreign')
    if kind == 'subclass': primary = 'subclass'
    stub = cpu_mode_runtime(probe, monkeypatch, tmp_path, primary=None if kind == 'callback' else primary,
                            callback_failure=primary if kind == 'callback' else None)
    if kind == 'subclass': primary = stub.primary
    with pytest.raises(type(primary)) as caught: probe.restart_cpu(stub.output, run=stub.run)
    assert caught.value is primary
    report = json.loads((stub.output / 'restart-cpu.json').read_text())
    assert report['status'] == 'failed' and report['error']['message'] == str(primary)
    assert report['policy']['restored_default'] == 'inductor'
    assert report['dynamo_before'] == report['dynamo_after']
    assert stub.runtime.registry.get_default_backend() == 'inductor'
    assert all(Path(path).is_dir() for path in stub.paths.values())


def test_restart_cpu_validation_failure_precedes_torch_import_and_is_exclusive(probe, monkeypatch, tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    settings = restart_cpu_settings()
    settings['image']['sha256'] = '0' * 64
    (run / 'resolved.json').write_text(json.dumps(settings))
    monkeypatch.setattr(probe.importlib, 'import_module', lambda *a: pytest.fail('invalid protocol imported Torch'))
    monkeypatch.setattr(probe.faulthandler, 'cancel_dump_traceback_later', lambda: None)
    output = run / 'output/0'
    with pytest.raises(ValueError, match='pinned'): probe.restart_cpu(output, run=run)
    assert json.loads((output / 'restart-cpu.json').read_text())['status'] == 'failed'
    before = (output / 'restart-cpu.json').read_bytes()
    with pytest.raises(FileExistsError): probe.restart_cpu(output, run=run)
    assert (output / 'restart-cpu.json').read_bytes() == before


def test_restart_cpu_private_caches_survive_a_pending_compiler_worker(probe, tmp_path):
    # Only this test owns the scratch parent; stop/join the CPU analog before test cleanup.
    with tempfile.TemporaryDirectory(prefix='restart-cpu-test-', dir='/dev/shm') as parent:
        cache = probe.restart_cpu_caches(Path(parent), '123456', tmp_path)
        root = Path(cache['root'])
        assert root.stat().st_uid == os.getuid() and root.stat().st_mode & 0o077 == 0
        assert json.loads((root / 'owner.json').read_text())['job_id'] == '123456'
        release, errors = threading.Event(), []
        def worker():
            try:
                assert release.wait(timeout=5)
                (Path(cache['environment']['TORCHINDUCTOR_CACHE_DIR']) / 'late-worker').write_text('complete')
            except BaseException as error: errors.append(error)
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            assert thread.is_alive() and all(Path(value).is_dir() for value in cache['environment'].values())
            assert cache['disposition'].startswith('preserved')
        finally:
            release.set()
            thread.join(timeout=5)
        assert not thread.is_alive() and not errors
        assert (Path(cache['environment']['TORCHINDUCTOR_CACHE_DIR']) / 'late-worker').is_file()


@pytest.mark.parametrize('fail_compile,observation_failure', [(False, False), (True, False), (True, True)])
def test_actual_fixture_body_uses_only_cpu_graph_and_restores_config(probe, monkeypatch, fail_compile, observation_failure):
    from contextvars import ContextVar
    class Tensor:
        def __init__(self, shape):
            self.shape, self.device, self.requires_grad = shape, SimpleNamespace(type='cpu'), False
            self.dtype = 'fp32'
        def reshape(self, *shape): return Tensor(shape)
        def __truediv__(self, value):
            assert value == 100
            return self
        def __getitem__(self, index): return Tensor((1, *self.shape) if index is None else self.shape[1:])
    calls = []
    def arange(count, **kwargs):
        calls.append(('arange', count, kwargs))
        assert count == 288 and kwargs == {'dtype': 'fp32', 'device': 'cpu'}
        return Tensor((count,))
    def interpolate(image, **kwargs):
        assert image.shape == (1, 3, 8, 12)
        assert kwargs == {'size': None, 'scale_factor': 1.873536229133606, 'mode': 'bilinear',
                          'recompute_scale_factor': True, 'align_corners': False}
        return Tensor((1, 3, 14, 22))
    error = RuntimeError('native compiler body analog')
    failed = []
    def compile(fn, **kwargs):
        assert kwargs == {'fullgraph': True}  # No replacement backend/options/mode or dynamic override.
        if fail_compile:
            failed.append(True)
            raise error
        return fn
    torch = SimpleNamespace(arange=arange, float32='fp32', compile=compile,
                            nn=SimpleNamespace(functional=SimpleNamespace(interpolate=interpolate)),
                            isfinite=lambda value: SimpleNamespace(all=lambda: True), allclose=lambda *a, **k: True)
    config = ContextConfig({key: ContextVar(key, default=value) for key, value in
                            {'assume_static_by_default': True, 'specialize_float': False,
                             'capture_scalar_outputs': False}.items()})
    record = {}
    evidence = SimpleNamespace(emit=lambda stage: calls.append(stage))
    values = probe.config_values
    def observed(config, names):
        if observation_failure and failed: raise OSError('exit observation failed')
        return values(config, names)
    monkeypatch.setattr(probe, 'config_values', observed)
    if fail_compile:
        with pytest.raises(RuntimeError) as caught: probe.run_restart_cpu_graph(torch, config, record, evidence)
        assert caught.value is error
    else:
        probe.run_restart_cpu_graph(torch, config, record, evidence)
        assert len(record['numerics']) == 2 and all(row['close'] for row in record['numerics'])
    if observation_failure:
        assert record['secondary_errors'][0]['message'] == 'exit observation failed'
        assert values(config, record['dynamo_before']) == record['dynamo_before']
    else:
        assert record['dynamo_before'] == record['dynamo_after']
    assert record['dynamo_during']['assume_static_by_default'] is False
    assert record['dynamo_during']['capture_scalar_outputs'] is False


@pytest.mark.parametrize('defect', ['source', 'config', 'compile-request', 'limit', 'numerics', 'no-recovery',
                                   'foreign-thread', 'options', 'observation', 'cache', 'cuda'])
def test_restart_cpu_outcome_does_not_waive_missing_evidence(probe, monkeypatch, tmp_path, defect):
    stub = cpu_mode_runtime(probe, monkeypatch, tmp_path)
    record = probe.restart_cpu(stub.output, run=stub.run)
    if defect == 'source': record['sources_end'] = {}
    elif defect == 'config': record['dynamo_after'] = {}
    elif defect == 'compile-request': record['compile_request'] = {}
    elif defect == 'limit': record['limits_after'] = {'recompile_limit': 9}
    elif defect == 'numerics': record['numerics'].pop()
    elif defect == 'no-recovery': record['policy']['compile_calls'].pop()
    elif defect == 'foreign-thread': record['policy']['compile_calls'][-1]['thread'] += 1
    elif defect == 'options': record['policy']['compile_calls'][0]['options_exit'] = {}
    elif defect == 'observation': record['policy']['compile_calls'][0]['observation_errors'].append('lost evidence')
    elif defect == 'cache': record['private_caches']['preserved_at_exit'] = False
    elif defect == 'cuda': record['cuda_initialized_after'] = True
    with pytest.raises(ValueError): probe.validate_restart_cpu_outcome(record)


@pytest.mark.parametrize('secondary_stage', ['watchdog', 'source-exit', 'final-publication', 'close'])
def test_restart_cpu_secondary_finalization_failures_preserve_primary_and_record(probe, monkeypatch, tmp_path, secondary_stage):
    primary, secondary = RuntimeError('primary native analog'), OSError('secondary observer failure')
    stub = cpu_mode_runtime(probe, monkeypatch, tmp_path, primary=primary)
    if secondary_stage == 'watchdog':
        def cancel(): raise secondary
        monkeypatch.setattr(probe.faulthandler, 'cancel_dump_traceback_later', cancel)
    elif secondary_stage == 'source-exit':
        calls = []
        def sources(torch):
            if calls: raise secondary
            calls.append(True)
            return dict(probe.RESTART_CPU_SOURCE)
        monkeypatch.setattr(probe, 'restart_cpu_sources', sources)
    elif secondary_stage == 'final-publication':
        original = probe.RestartCpuEvidence.emit
        def emit(self, stage):
            if stage == 'failed': raise secondary
            return original(self, stage)
        monkeypatch.setattr(probe.RestartCpuEvidence, 'emit', emit)
    elif secondary_stage == 'close':
        original = probe.RestartCpuEvidence.close
        def close(self):
            original(self)
            raise secondary
        monkeypatch.setattr(probe.RestartCpuEvidence, 'close', close)
    with pytest.raises(RuntimeError) as caught: probe.restart_cpu(stub.output, run=stub.run)
    assert caught.value is primary and primary.args == ('primary native analog',)
    report = json.loads((stub.output / 'restart-cpu.json').read_text())
    assert report['status'] == 'failed' and report['error']['message'] == str(primary)
    assert any(row['message'] == str(secondary) for row in report['secondary_errors'])
    assert report['environment_restored'] and report['policy']['restored_default'] == 'inductor'
