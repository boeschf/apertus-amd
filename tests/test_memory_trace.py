"""Check diagnostic collection and lifecycle without requiring GPU libraries."""

import importlib.util
import ast
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import sys
import threading
from typing import NamedTuple
from types import SimpleNamespace
from types import ModuleType

import pytest


@pytest.fixture
def trace(request):
    name = getattr(request, 'param', 'memory_trace')
    path = Path(__file__).parents[1] / f'experiments/validation/{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_does_not_initialize_gpu(trace, monkeypatch, tmp_path):
    monkeypatch.setenv('TMPDIR', str(tmp_path / 'tmp'))
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    value = trace.snapshot(torch, 1, [], include_cache=True)
    assert 'gpu' not in value
    assert 'VmRSS:' in value['process_status']
    assert value['private_cache']['files'] == 0


def test_node_sampler_retains_buddyinfo_without_initializing_gpu(trace, monkeypatch):
    monkeypatch.setattr(trace, 'read_text', lambda path: f'snapshot:{path}')
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    value = trace.snapshot(torch, 0, [], include_cache=False)
    assert value['buddyinfo'] == 'snapshot:/proc/buddyinfo'
    assert value['base_page_bytes'] > 0
    assert 'gpu' not in value
    assert 'buddyinfo' not in trace.snapshot(torch, 1, [], include_cache=False)


def test_cache_scan_does_not_follow_directory_links(trace, tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'large').write_bytes(b'x' * 10000)
    (cache / 'link').symlink_to(outside, target_is_directory=True)
    (cache / 'kernel').write_bytes(b'abc')
    assert trace.cache_usage(cache)['logical_bytes'] == 3


@pytest.mark.parametrize('fails', [False, True])
def test_training_exit_stops_sampler_and_retains_outcome(trace, monkeypatch, tmp_path, fails):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='stub'))
    monkeypatch.setattr(trace, 'cgroup_paths', lambda: [])
    monkeypatch.setattr(trace, 'snapshot', lambda *a, **k: {'gpu': {'reserved_bytes.all.peak': 1}})

    def training(*args):
        if fails:
            raise ValueError('training failure')

    monkeypatch.setattr(trace, 'train', training)
    if fails:
        with pytest.raises(ValueError, match='training failure'):
            trace.trace_training(tmp_path, tmp_path, 'train.py', [])
    else:
        trace.trace_training(tmp_path, tmp_path, 'train.py', [])
    end = json.loads((tmp_path / 'memory-end-0.json').read_text())
    assert end['training_completed'] is not fails
    assert end['sampler_errors'] == []
    assert (tmp_path / 'memory-0.jsonl').read_text()


def test_sampler_error_is_not_reported_as_success(trace, monkeypatch, tmp_path):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='stub'))
    monkeypatch.setattr(trace, 'cgroup_paths', lambda: [])
    monkeypatch.setattr(trace, 'train', lambda *a: None)

    def snapshot(*args, include_cache):
        if include_cache:
            raise OSError('cannot sample')
        return {'gpu': {'reserved_bytes.all.peak': 1}}

    monkeypatch.setattr(trace, 'snapshot', snapshot)
    with pytest.raises(RuntimeError, match='cannot sample'):
        trace.trace_training(tmp_path, tmp_path, 'train.py', [])


def test_check_rejects_missing_rank_and_conflicting_override(trace, tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    env = {'PYTORCH_HIP_ALLOC_CONF': 'expandable_segments:True', 'TORCH_COMPILE_DISABLE': '1',
           'PYTORCH_CUDA_ALLOC_CONF': None, 'TORCHDYNAMO_DISABLE': None}
    (tmp_path / 'resolved.json').write_text(json.dumps({
        'resources': {'nodes': 1, 'gpus_per_node': 2}, 'environment': env}))
    for rank in range(2):
        (output / f'memory-start-{rank}.json').write_text(json.dumps({
            'environment': env, 'hostname': 'node'}))
        (output / f'memory-{rank}.jsonl').write_text('{}\n')
    end = {'training_completed': True, 'sampler_errors': [], 'final': {'gpu': {'peak': 1}}}
    (output / 'memory-end-0.json').write_text(json.dumps(end))
    with pytest.raises(FileNotFoundError):
        trace.check(tmp_path)
    assert not (output / 'checks.json').exists()
    (output / 'memory-end-1.json').write_text(json.dumps(end))
    trace.check(tmp_path)
    assert json.loads((output / 'checks.json').read_text())['numerical_validation'] == 'not requested'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    (output / 'memory-start-1.json').write_text(json.dumps({'environment': env, 'hostname': 'node'}))
    with pytest.raises(ValueError, match='confounding'):
        trace.check(tmp_path)


@pytest.mark.parametrize('fails', [False, True])
def test_compilation_observer_preserves_results_and_restores_on_failure(trace, monkeypatch, tmp_path, fails):
    counters = {'generated_kernel_count': 0, 'counters': {}}
    monkeypatch.setattr(trace, 'compilation_snapshot', lambda: counters)

    def original(value, *, fail=False):
        if fail:
            raise RuntimeError('step failed')
        counters['generated_kernel_count'] += 2
        return value

    module = SimpleNamespace(train_step=original)
    value = object()
    try:
        with trace.trace_compilation_steps(module, tmp_path, 0):
            assert module.train_step(value) is value
            if fails:
                module.train_step(value, fail=True)
    except RuntimeError as error:
        assert fails and str(error) == 'step failed'
    assert module.train_step is original
    rows = [json.loads(line) for line in (tmp_path / 'compilation-0.jsonl').read_text().splitlines()]
    assert [r['status'] for r in rows] == ['start', 'step', 'end']
    assert [r['generated_kernel_count'] for r in rows] == [0, 2, 2]
    assert [r['completed_train_steps'] for r in rows] == [0, 1, 1]


AUDIT = Path(__file__).parents[1] / 'artifacts/investigations/performance-next/autotune-control/backward-propagation/source'


class ContextConfig:
    """Model the pinned per-entry ContextVar storage, not process-global SimpleNamespace flags."""

    def __init__(self, values, prefix=''):
        object.__setattr__(self, '_values', values)
        object.__setattr__(self, '_prefix', prefix)

    def __getattr__(self, name):
        key = self._prefix + name
        if key in self._values: return self._values[key].get()
        return ContextConfig(self._values, key + '.')

    def __setattr__(self, name, value):
        self._values[self._prefix + name].set(value)

    @contextmanager
    def patch(self, options):
        tokens = [(self._values[key], self._values[key].set(value)) for key, value in options.items()]
        try:
            yield
        finally:
            for variable, token in reversed(tokens): variable.reset(token)


def source_definitions(path, names, scope):
    """Execute exact pinned API definitions with CPU dependency stubs, not a fake registry implementation."""
    tree = ast.parse((AUDIT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    code = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    exec(compile('from __future__ import annotations\n' + code, str(AUDIT / path), 'exec'), scope)


def registry_analog(registry, torch):
    """Provide the same small API surface when optional pinned audit snapshots are absent."""
    backends = {}
    default = 'inductor'
    def register_backend(function, *, name):
        assert name not in backends
        backends[name] = function
        return function
    def set_default_backend(value):
        nonlocal default
        default = 'inductor' if value is None else value
    class Wrapper:
        def __init__(self, name, mode, options, dynamic):
            self.compiler_fn = backends[name]
            self.kwargs = {}
            if mode and mode != 'default': self.kwargs['mode'] = mode
            if options: self.kwargs['options'] = options
        def __call__(self, graph, inputs):
            return self.compiler_fn(graph, inputs, **self.kwargs)
    registry.register_backend = register_backend
    registry.lookup_backend = lambda name: backends[name]
    registry.list_backends = lambda **kwargs: list(backends)
    registry.set_default_backend = set_default_backend
    registry.get_default_backend = lambda: default
    registry._is_registered_backend = lambda wrapper: wrapper.compiler_fn in backends.values()
    torch._TorchCompileWrapper = Wrapper


def backend_runtime(module, monkeypatch):
    """Exercise actual registry/default/wrapper definitions against a context-local CPU compiler analog."""
    import functools
    from types import ModuleType
    values = {**module.POLICY_GUARDS, **{key: not value for key, value in module.POLICY_FLAGS.items()},
              'triton.autotune_at_compile_time': None}
    config = ContextConfig({key: ContextVar(key, default=value) for key, value in values.items()})
    registry = ModuleType('torch._dynamo.backends.registry')
    torch = SimpleNamespace(__version__='stub', compile=object(),
                            _TorchCompileInductorWrapper=type('BuiltinWrapper', (), {}))
    scope = registry.__dict__
    scope.update(torch=torch, functools=functools, _BACKENDS={}, _COMPILER_FNS={}, _BACKEND_TAGS={},
                 _default_backend='inductor', _lazy_import=lambda: None)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    if AUDIT.is_dir():
        source_definitions('torch/_dynamo/backends/registry.py',
                           ['register_backend', 'lookup_backend', 'list_backends', '_is_registered_backend',
                            'set_default_backend', 'get_default_backend'], scope)
        wrapper_scope = {}
        source_definitions('torch/__init__.py', ['_TorchCompileWrapper'], wrapper_scope)
        torch._TorchCompileWrapper = wrapper_scope['_TorchCompileWrapper']
    else:
        registry_analog(registry, torch)
    torch.compiler = SimpleNamespace(get_default_backend=registry.get_default_backend, set_default_backend=registry.set_default_backend)
    calls, result = [], object()

    def compile_graph(graph, inputs, *, options):
        with config.patch(options):
            calls.append({'graph': graph, 'inputs': inputs, 'options': options,
                          'effective': module.config_values(config, module.POLICY_FLAGS)})
        return result

    inductor = SimpleNamespace(compile=compile_graph)
    imports = {'torch._inductor.config': config, 'torch._dynamo.backends.registry': registry,
               'torch._dynamo': SimpleNamespace(register_backend=registry.register_backend, list_backends=registry.list_backends,
                                               lookup_backend=registry.lookup_backend),
               'torch._inductor': inductor}
    monkeypatch.setattr(module.importlib, 'import_module', lambda name: imports[name])
    monkeypatch.setenv('APERTUS_INDUCTOR_POLICY', module.POLICY_NAME)
    monkeypatch.setenv('TORCH_COMPILE_DISABLE', '0')
    monkeypatch.delenv('TORCHDYNAMO_DISABLE', raising=False)
    return SimpleNamespace(torch=torch, config=config, registry=registry, imports=imports,
                           inductor=inductor, calls=calls, result=result)


def policy_record(module):
    """Return complete successful evidence without claiming a GPU compilation."""
    guards = dict(module.POLICY_GUARDS)
    return {'requested': module.POLICY_NAME, 'torch': 'stub', 'backend': module.BACKEND_NAME,
            'options': dict(module.POLICY_FLAGS), 'options_id': module.OPTIONS_ID,
            'default_before': 'inductor', 'default_start': module.BACKEND_NAME, 'default_end': module.BACKEND_NAME,
            'restored_default': 'inductor', 'guards_start': guards, 'guards_end': guards,
            'compile_time_start': None, 'compile_time_end': None,
            'compile_calls': [{'status': 'returned', 'options_forwarded': dict(module.POLICY_FLAGS),
                               'options_id': module.OPTIONS_ID, 'guards': guards, 'guards_after': guards,
                               'compile_time': None, 'compile_time_after': None}]}


@pytest.fixture(params=['memory_trace', 'sssglu_compile_probe'])
def policy_module(request):
    path = Path(__file__).parents[1] / f'experiments/validation/{request.param}.py'
    spec = importlib.util.spec_from_file_location(request.param, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_absent_policy_touches_no_compiler(policy_module, monkeypatch):
    monkeypatch.delenv('APERTUS_INDUCTOR_POLICY', raising=False)
    monkeypatch.setattr(policy_module.importlib, 'import_module', lambda name: pytest.fail('baseline compiler import'))
    assert policy_module.configure_inductor_policy(SimpleNamespace()) == (None, None)


def test_cpu_analogs_remain_self_contained_without_audit_snapshots(trace, monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules[__name__], 'AUDIT', tmp_path / 'absent')
    runtime = backend_runtime(trace, monkeypatch)
    before = trace.config_values(runtime.config, trace.POLICY_FLAGS)
    owner, _ = trace.configure_inductor_policy(runtime.torch)
    wrapper = runtime.torch._TorchCompileWrapper(trace.BACKEND_NAME, 'default', None, None)
    assert wrapper(object(), []) is runtime.result
    assert trace.config_values(runtime.config, trace.POLICY_FLAGS) == before
    trace.check_inductor_policy(owner.finish(), trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('requested', ['', 'default', 'single-candidate', 'eager', 'single_candidate'])
def test_unknown_or_unsafe_policy_is_rejected_before_import(policy_module, monkeypatch, requested):
    monkeypatch.setenv('APERTUS_INDUCTOR_POLICY', requested)
    monkeypatch.setattr(policy_module.importlib, 'import_module', lambda name: pytest.fail('invalid policy compiler import'))
    with pytest.raises(ValueError, match='unsafe ambient' if requested == 'single_candidate' else 'unknown APERTUS'):
        policy_module.configure_inductor_policy(SimpleNamespace())


def test_context_isolated_native_backward_analog_receives_explicit_options(policy_module, monkeypatch):
    module = policy_module
    runtime = backend_runtime(module, monkeypatch)
    native_compile = runtime.torch.compile
    owner, record = module.configure_inductor_policy(runtime.torch)
    # An ambient assignment is deliberately lost in the isolated engine analog.
    for key, value in module.POLICY_FLAGS.items():
        runtime.config._values[key].set(value)
    graph, inputs = object(), [object()]
    wrapper = runtime.torch._TorchCompileWrapper(runtime.registry.get_default_backend(), 'default', None, None)
    seen, errors = [], []

    def backward():
        try:
            seen.append(module.config_values(runtime.config, module.POLICY_FLAGS))
            assert wrapper(graph, inputs) is runtime.result
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=backward, name='engine-analog')
    thread.start(); thread.join(timeout=10)
    assert not thread.is_alive() and not errors
    assert seen[0] != module.POLICY_FLAGS
    assert runtime.calls[0]['effective'] == module.POLICY_FLAGS
    assert runtime.calls[0]['graph'] is graph and runtime.calls[0]['inputs'] is inputs
    assert runtime.registry._is_registered_backend(wrapper)
    assert runtime.torch.compile is native_compile
    assert record['compile_calls'][0]['ambient_before'] == seen[0]
    assert record['compile_calls'][0]['thread_name'] == 'engine-analog'
    module.check_inductor_policy(owner.finish(), module.POLICY_NAME, 'stub')
    assert runtime.registry.get_default_backend() == 'inductor'
    with pytest.raises(TypeError): module.POLICY_FLAGS['dynamic_scale_rblock'] = True


def test_pinned_registered_backend_cleanup_and_inductor_options_bridge(trace, monkeypatch):
    if not AUDIT.is_dir(): pytest.skip('optional pinned image source audit is not available')
    runtime = backend_runtime(trace, monkeypatch)
    owner, _ = trace.configure_inductor_policy(runtime.torch)
    wrapper = runtime.torch._TorchCompileWrapper(trace.BACKEND_NAME, 'default', None, None)
    cleared = []
    scope = {'config': SimpleNamespace(invalidate_compile_context_weakrefs=None),
             '_is_registered_backend': runtime.registry._is_registered_backend,
             'innermost_backend': lambda backend: backend,
             '_clear_fake_mode_weakrefs': lambda mode: cleared.append(mode)}
    source_definitions('torch/_dynamo/convert_frame.py', ['clear_compile_context_weakrefs'], scope)
    tensor_context = {'retained': object()}
    output = SimpleNamespace(tracing_context=SimpleNamespace(tensor_to_context=tensor_context, fake_mode=object()))
    scope['clear_compile_context_weakrefs'](SimpleNamespace(output_graph_for_cleanup=output), wrapper)
    assert tensor_context == {} and cleared == [output.tracing_context.fake_mode]

    # Run the exact shipped Inductor options entry point, stubbing only actual compilation.
    from types import ModuleType
    compiler = ModuleType('torch._inductor.compile_fx')
    received = []
    sentinel = object()
    def compile_fx(graph, inputs, *, config_patches):
        received.append((graph, inputs, config_patches))
        return sentinel
    compiler.compile_fx = compile_fx
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    bridge = {'__name__': 'torch._inductor', '__package__': 'torch._inductor'}
    source_definitions('torch/_inductor/__init__.py', ['compile'], bridge)
    runtime.inductor.compile = bridge['compile']
    graph, inputs = object(), [object()]
    assert wrapper(graph, inputs) is sentinel
    assert received[0][0] is graph and received[0][1] is inputs and received[0][2] == trace.POLICY_FLAGS
    # The downstream lazy-backward patch is explicitly part of pinned compile_fx.
    tree = ast.parse((AUDIT / 'torch/_inductor/compile_fx.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compile_fx')
    assert any(isinstance(node, ast.keyword) and node.arg == 'inner_compile' and
               ast.unparse(node.value) == 'config.patch(config_patches)(inner_compile)' for node in ast.walk(function))
    owner.finish()


@pytest.mark.parametrize('failure', ['before-selection', 'after-selection', 'publication'])
def test_setup_failure_restores_only_owned_default(policy_module, monkeypatch, failure):
    module = policy_module; runtime = backend_runtime(module, monkeypatch)
    setter = runtime.torch.compiler.set_default_backend
    error = RuntimeError('setup failed')
    def select(name):
        if name == module.BACKEND_NAME:
            if failure == 'after-selection': setter(name)
            raise error
        setter(name)
    def publish(record):
        if record.get('default_end') is None: raise error
    if failure != 'publication': runtime.torch.compiler.set_default_backend = select
    with pytest.raises(RuntimeError) as caught:
        module.configure_inductor_policy(runtime.torch, on_update=publish if failure == 'publication' else None)
    assert caught.value is error
    assert runtime.registry.get_default_backend() == 'inductor' and module._policy_owner is None


@pytest.mark.parametrize('name', [
    'max_autotune', 'max_autotune_pointwise', 'max_autotune_gemm', 'deterministic', 'batch_invariant',
    'coordinate_descent_tuning', 'combo_kernels', 'incremental_autotune', 'profile_bandwidth',
    'triton.debug_sync_graph', 'triton.debug_sync_kernel', 'triton.autotune_at_compile_time'])
def test_guards_are_checked_in_actual_compiling_context(policy_module, monkeypatch, name):
    module = policy_module; runtime = backend_runtime(module, monkeypatch)
    owner, record = module.configure_inductor_policy(runtime.torch)
    errors = []

    def backward():
        runtime.config._values[name].set(True)
        try: module.single_candidate_backend(object(), [])
        except ValueError as error: errors.append(error)

    thread = threading.Thread(target=backward); thread.start(); thread.join(timeout=10)
    assert not thread.is_alive() and len(errors) == 1 and 'preconditions' in str(errors[0])
    assert runtime.calls == [] and record['compile_calls'][0]['status'] == 'failed'
    assert module.config_values(runtime.config, module.POLICY_GUARDS) == module.POLICY_GUARDS
    owner.finish()


@pytest.mark.parametrize('conflict', ['default', 'collision', 'late', 'eager', 'setup-guard'])
def test_setup_conflicts_do_not_replace_existing_state(policy_module, monkeypatch, conflict):
    module = policy_module; runtime = backend_runtime(module, monkeypatch)
    if conflict == 'default': runtime.registry.set_default_backend('eager')
    elif conflict == 'collision': runtime.registry.register_backend(lambda graph, inputs: None, name=module.BACKEND_NAME)
    elif conflict == 'late': monkeypatch.setitem(sys.modules, 'megatron.core.fusions.fused_bias_sssglu', SimpleNamespace())
    elif conflict == 'eager': monkeypatch.setenv('TORCH_COMPILE_DISABLE', '1')
    elif conflict == 'setup-guard': runtime.config.max_autotune = True
    default = runtime.registry.get_default_backend()
    with pytest.raises(ValueError): module.configure_inductor_policy(runtime.torch)
    assert runtime.registry.get_default_backend() == default and module._policy_owner is None
    assert runtime.calls == []


def test_registration_is_process_lived_and_default_selection_is_owned(policy_module, monkeypatch):
    module = policy_module; runtime = backend_runtime(module, monkeypatch)
    owner, _ = module.configure_inductor_policy(runtime.torch)
    with pytest.raises(ValueError, match='already active'): module.configure_inductor_policy(runtime.torch)
    owner.finish()
    assert runtime.registry.lookup_backend(module.BACKEND_NAME) is module.single_candidate_backend
    with pytest.raises(ValueError, match='no active owner'): module.single_candidate_backend(object(), [])
    with pytest.raises(ValueError, match='already closed'): owner.finish()
    second, _ = module.configure_inductor_policy(runtime.torch)
    runtime.registry.set_default_backend('eager')
    with pytest.raises(ValueError, match='unowned'): second.finish()
    assert runtime.registry.get_default_backend() == 'eager' and module._policy_owner is None


@pytest.mark.parametrize('kwargs', [{'options': {}}, {'mode': 'default'}, {'unexpected': True}])
def test_delegate_rejects_explicit_mode_options_without_compiling(policy_module, monkeypatch, kwargs):
    runtime = backend_runtime(policy_module, monkeypatch)
    owner, record = policy_module.configure_inductor_policy(runtime.torch)
    with pytest.raises(ValueError, match='does not support'): policy_module.single_candidate_backend(object(), [], **kwargs)
    assert runtime.calls == [] and record['compile_calls'][0]['status'] == 'failed'
    owner.finish()


def test_delegate_preserves_original_exception_and_restores_default(policy_module, monkeypatch):
    runtime = backend_runtime(policy_module, monkeypatch)
    error = RuntimeError('native compiler failed')
    def failed(*args, **kwargs): raise error
    runtime.inductor.compile = failed
    owner, record = policy_module.configure_inductor_policy(runtime.torch)
    with pytest.raises(RuntimeError) as caught: policy_module.single_candidate_backend(object(), [])
    assert caught.value is error and record['compile_calls'][0]['error']['message'] == str(error)
    owner.finish()
    assert runtime.registry.get_default_backend() == 'inductor'


@pytest.mark.parametrize('trace_enabled', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_training_backend_precedes_imports_persists_and_restores(trace, monkeypatch, tmp_path, trace_enabled, fails):
    runtime = backend_runtime(trace, monkeypatch)
    monkeypatch.setenv('RANK', '0'); monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('APERTUS_COMPILE_TRACE', '1' if trace_enabled else '0')
    monkeypatch.setitem(sys.modules, 'torch', runtime.torch)
    monkeypatch.setattr(trace, 'cgroup_paths', lambda: [])
    monkeypatch.setattr(trace, 'snapshot', lambda *a, **k: {'gpu': {'peak': 1}})
    monkeypatch.setattr(trace, 'compilation_snapshot', lambda: {'counters': {}, 'generated_kernel_count': 0})
    runtime.imports['megatron.training.training'] = SimpleNamespace(train_step=lambda: None)

    def training(*args, before_run):
        before_run()
        assert runtime.registry.get_default_backend() == trace.BACKEND_NAME
        assert (tmp_path / 'inductor-policy-0.json').exists()
        trace.single_candidate_backend(object(), [])
        if fails: raise ValueError('training failed')

    monkeypatch.setattr(trace, 'train', training)
    if fails:
        with pytest.raises(ValueError, match='training failed'): trace.trace_training(tmp_path, tmp_path, 'train.py', [])
    else: trace.trace_training(tmp_path, tmp_path, 'train.py', [])
    record = json.loads((tmp_path / 'inductor-policy-0.json').read_text())
    trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')
    assert runtime.registry.get_default_backend() == 'inductor'
    assert json.loads((tmp_path / 'memory-end-0.json').read_text())['training_completed'] is not fails


@pytest.mark.parametrize('defect', ['torch', 'options', 'calls', 'guards', 'restored'])
def test_backend_checker_rejects_policy_drift(policy_module, defect):
    module = policy_module; record = policy_record(module)
    module.check_inductor_policy(record, module.POLICY_NAME, 'stub')
    if defect == 'torch': record['torch'] = 'other'
    elif defect == 'options': record['options'] = {}
    elif defect == 'calls': record['compile_calls'] = []
    elif defect == 'guards': record['compile_calls'][0]['guards_after'] = {}
    elif defect == 'restored': record['restored_default'] = 'eager'
    with pytest.raises(ValueError): module.check_inductor_policy(record, module.POLICY_NAME, 'stub')


def test_memory_checker_requires_backend_rank_record(trace, tmp_path):
    output = tmp_path / 'output'; output.mkdir()
    environment = {'APERTUS_INDUCTOR_POLICY': trace.POLICY_NAME, 'PYTORCH_HIP_ALLOC_CONF': 'expandable_segments:True',
                   'TORCH_COMPILE_DISABLE': '0', 'PYTORCH_CUDA_ALLOC_CONF': None, 'TORCHDYNAMO_DISABLE': None}
    (tmp_path / 'resolved.json').write_text(json.dumps({'resources': {'nodes': 1, 'gpus_per_node': 1}, 'environment': environment}))
    (output / 'memory-start-0.json').write_text(json.dumps({'environment': environment, 'torch': 'stub', 'hostname': 'node'}))
    (output / 'memory-end-0.json').write_text(json.dumps({'training_completed': True, 'sampler_errors': [], 'final': {'gpu': {'peak': 1}}}))
    (output / 'memory-0.jsonl').write_text('{}\n')
    with pytest.raises(FileNotFoundError): trace.check(tmp_path)
    record = policy_record(trace)
    (output / 'inductor-policy-0.json').write_text(json.dumps(record))
    trace.check(tmp_path)
    assert json.loads((output / 'checks.json').read_text())['numerical_validation'] == 'not requested'
    record['compile_calls'][0]['options_forwarded'] = {}
    (output / 'inductor-policy-0.json').write_text(json.dumps(record))
    with pytest.raises(ValueError): trace.check(tmp_path)


RESTART_AUDIT = Path(__file__).parents[1] / 'artifacts/investigations/performance-next/restart-20261113/compiled-p2p/restart-audit'


def restart_runtime(trace, monkeypatch):
    """Use pinned CompileContext/CompileId and exception definitions when retained source is available."""
    runtime = backend_runtime(trace, monkeypatch)
    guards = ModuleType('torch._guards')
    exc = ModuleType('torch._dynamo.exc')
    monkeypatch.setitem(sys.modules, guards.__name__, guards)
    monkeypatch.setitem(sys.modules, exc.__name__, exc)
    guards.__dict__.update(dataclass=dataclass, NamedTuple=NamedTuple, threading=threading, contextmanager=contextmanager)
    if AUDIT.is_dir() and (RESTART_AUDIT / 'torch-dynamo-exc.py').is_file():
        guards.__file__ = str(AUDIT / 'torch/_guards.py')
        source_definitions('torch/_guards.py', ['CompileId', 'TraceId', '_TLSStorage', 'CompileContext', 'compile_context'], guards.__dict__)
        guards._TLS = guards._TLSStorage()
        tree = ast.parse((RESTART_AUDIT / 'torch-dynamo-exc.py').read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
                   node.name in ('TorchDynamoException', 'RestartAnalysis', 'TensorifyScalarRestartAnalysis')]
        assert len(classes) == 3
        exec(compile('from __future__ import annotations\n' + ast.unparse(ast.Module(body=classes, type_ignores=[])),
                     str(RESTART_AUDIT / 'torch-dynamo-exc.py'), 'exec'), exc.__dict__)
    else:
        # A self-contained CPU analog is not vendor-source attestation.
        @dataclass
        class CompileId:
            frame_id: int
            frame_compile_id: int
            compiled_autograd_id: int | None = None
            def __str__(self):
                prefix = '' if self.compiled_autograd_id is None else f'!{self.compiled_autograd_id}/'
                return prefix + f'{self.frame_id}/{self.frame_compile_id}'
        tls = threading.local()
        class CompileContext:
            def __init__(self, compile_id): self.compile_id, self.attempt = compile_id, 0
            @staticmethod
            def try_get(): return getattr(tls, 'context', None)
            @staticmethod
            def current_trace_id():
                value = CompileContext.try_get()
                return None if value is None else SimpleNamespace(compile_id=value.compile_id, attempt=value.attempt)
        @contextmanager
        def compile_context(context):
            previous = CompileContext.try_get()
            tls.context = context
            try: yield context
            finally: tls.context = previous
        class TensorifyScalarRestartAnalysis(RuntimeError):
            def __init__(self, *args, restart_reason=None):
                self.restart_reason = restart_reason
                super().__init__(*args)
        TensorifyScalarRestartAnalysis.__module__ = exc.__name__
        TensorifyScalarRestartAnalysis.__qualname__ = 'TensorifyScalarRestartAnalysis'
        guards.CompileId, guards.CompileContext, guards.compile_context = CompileId, CompileContext, compile_context
        exc.TensorifyScalarRestartAnalysis = TensorifyScalarRestartAnalysis
    runtime.imports.update({'torch._guards': guards, 'torch._dynamo.exc': exc})
    runtime.guards, runtime.exc = guards, exc
    runtime.context = guards.CompileContext(guards.CompileId(frame_id=42, frame_compile_id=7))
    # Class/control-flow tests use only CPU definitions; actual file attestation is tested separately below.
    monkeypatch.setattr(trace.BackendPolicy, 'audit_restart_source',
                        lambda owner: owner.record.update(restart_source=dict(trace.RESTART_SOURCE)))
    return runtime


def recovered_record(trace, runtime):
    """Reraise the exact restart once, then simulate the audited loop's next attempt in the same context."""
    owner, record = trace.configure_inductor_policy(runtime.torch)
    original = runtime.inductor.compile
    argument = object()
    error = runtime.exc.TensorifyScalarRestartAnalysis(argument, restart_reason='tensorify scalar')
    graph, inputs = object(), [object()]
    def restart(g, i, *, options):
        assert g is graph and i is inputs and options == trace.POLICY_FLAGS
        raise error
    runtime.inductor.compile = restart
    with runtime.guards.compile_context(runtime.context):
        with pytest.raises(runtime.exc.TensorifyScalarRestartAnalysis) as caught:
            trace.single_candidate_backend(graph, inputs)
        assert caught.value is error and error.args[0] is argument
        runtime.context.attempt = 1
        runtime.inductor.compile = original
        assert trace.single_candidate_backend(graph, inputs) is runtime.result
    return owner, record, error


@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_exact_tensorify_restart_records_and_rethrows_original_then_recovers(trace, monkeypatch):
    runtime = restart_runtime(trace, monkeypatch)
    owner, record, error = recovered_record(trace, runtime)
    first, second = record['compile_calls']
    assert first['status'] == 'restart' and second['status'] == 'returned'
    assert first['error']['module'] == 'torch._dynamo.exc'
    assert first['error']['qualname'] == 'TensorifyScalarRestartAnalysis'
    assert first['error']['restart_reason'] == error.restart_reason
    assert 'raise error' in first['error']['traceback']
    assert first['identity'] == first['identity_after']
    assert first['identity']['compile_id'] == '42/7'
    assert first['identity']['attempt'] == 0 and second['identity']['attempt'] == 1
    trace.check_inductor_policy(json.loads(json.dumps(owner.finish())), trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('kind', ['ordinary', 'foreign-name', 'subclass'])
@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_only_exact_class_is_restart_and_later_return_never_waives_errors(trace, monkeypatch, kind):
    runtime = restart_runtime(trace, monkeypatch)
    cls = {'ordinary': RuntimeError, 'foreign-name': type('TensorifyScalarRestartAnalysis', (RuntimeError,), {}),
           'subclass': type('Child', (runtime.exc.TensorifyScalarRestartAnalysis,), {})}[kind]
    error = cls('original')
    original = runtime.inductor.compile
    def failed(*args, **kwargs): raise error
    runtime.inductor.compile = failed
    owner, record = trace.configure_inductor_policy(runtime.torch)
    with runtime.guards.compile_context(runtime.context):
        with pytest.raises(cls) as caught: trace.single_candidate_backend(object(), [])
        assert caught.value is error and error.args == ('original',)
        runtime.context.attempt = 1
        runtime.inductor.compile = original
        trace.single_candidate_backend(object(), [])
    assert record['compile_calls'][0]['status'] == 'failed'
    with pytest.raises(ValueError): trace.check_inductor_policy(owner.finish(), trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('defect', [
    'missing-id', 'missing-post-id', 'missing-post-guard', 'missing-default', 'foreign-frame',
    'foreign-compilation', 'foreign-autograd', 'foreign-context', 'foreign-thread', 'foreign-thread-owner',
    'repeated-attempt', 'unrecovered', 'guard', 'compile-time', 'options', 'ambient', 'ownership',
    'source', 'exact-class', 'observation', 'sequence', 'excessive-attempt', 'unknown-schema',
])
@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_restart_evidence_fails_closed(trace, monkeypatch, defect):
    runtime = restart_runtime(trace, monkeypatch)
    owner, record, _ = recovered_record(trace, runtime)
    owner.finish()
    first, recovered = record['compile_calls']
    if defect == 'missing-id': first['identity'] = first['identity_after'] = None
    elif defect == 'missing-post-id': del first['identity_after']
    elif defect == 'missing-post-guard': del first['guards_after']
    elif defect == 'missing-default': del first['default_after']
    elif defect.startswith('foreign-'):
        identity = recovered['identity'].copy()
        if defect == 'foreign-frame': identity.update(frame_id=43, compile_id='43/7')
        elif defect == 'foreign-compilation': identity.update(frame_compile_id=8, compile_id='42/8')
        elif defect == 'foreign-autograd': identity.update(compiled_autograd_id=3, compile_id='!3/42/7')
        elif defect == 'foreign-context': identity['context'] += 1
        elif defect == 'foreign-thread': recovered['thread'] += 1
        elif defect == 'foreign-thread-owner': identity['thread_owner'] += 1
        recovered['identity'] = recovered['identity_after'] = identity
    elif defect == 'repeated-attempt': recovered['identity']['attempt'] = 0
    elif defect == 'unrecovered': record['compile_calls'].pop()
    elif defect == 'guard': first['guards_after'] = {**trace.POLICY_GUARDS, 'max_autotune': True}
    elif defect == 'compile-time': first['compile_time_after'] = True
    elif defect == 'options': first['options_exit'] = {}
    elif defect == 'ambient': first['ambient_after'] = {}
    elif defect == 'ownership': first['owned_after'] = False
    elif defect == 'source': record['restart_source'] = {}
    elif defect == 'exact-class': first['exact_tensorify_restart'] = False
    elif defect == 'observation': first['observation_errors'].append('lost observation')
    elif defect == 'sequence': first['sequence'] = 2
    elif defect == 'excessive-attempt': first['identity']['attempt'] = trace.MAX_RESTARTS + 1
    elif defect == 'unknown-schema': record['schema_version'] = 42
    with pytest.raises(ValueError): trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('drift', ['guard', 'options', 'default', 'ambient', 'ownership'])
@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_live_restart_exit_drift_is_recorded_without_masking_original(trace, monkeypatch, drift):
    runtime = restart_runtime(trace, monkeypatch)
    owner, record = trace.configure_inductor_policy(runtime.torch)
    error = runtime.exc.TensorifyScalarRestartAnalysis('original', restart_reason='retry')
    def restart(*args, options):
        if drift == 'guard': runtime.config.max_autotune = True
        elif drift == 'options': options.clear()
        elif drift == 'default': runtime.registry.set_default_backend('eager')
        elif drift == 'ambient': runtime.config.dynamic_scale_rblock = not runtime.config.dynamic_scale_rblock
        elif drift == 'ownership': owner.closed = True
        raise error
    runtime.inductor.compile = restart
    with runtime.guards.compile_context(runtime.context):
        with pytest.raises(type(error)) as caught: trace.single_candidate_backend(object(), [])
    assert caught.value is error and error.args == ('original',)
    row = record['compile_calls'][0]
    assert row['status'] == 'restart' and row['error']['restart_reason'] == 'retry'
    if drift == 'guard': assert row['guards_after']['max_autotune'] is True
    elif drift == 'options': assert row['options_exit'] == {}
    elif drift == 'default': assert row['default_after'] == 'eager'
    elif drift == 'ambient': assert row['ambient_before'] != row['ambient_after']
    elif drift == 'ownership': assert row['owned_after'] is False
    # Only test cleanup is repaired; retained row evidence remains invalid.
    owner.closed = False
    runtime.registry.set_default_backend(trace.BACKEND_NAME)
    owner.finish()
    with pytest.raises(ValueError): trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('failure', ['identity', 'formatting', 'audit', 'publication'])
@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_secondary_observation_failure_cannot_mask_original_restart(trace, monkeypatch, failure):
    runtime = restart_runtime(trace, monkeypatch)
    owner, record = trace.configure_inductor_policy(runtime.torch)
    if failure == 'formatting':
        # Even hostile ordinary exception formatting must not replace the raised object.
        class BadString(RuntimeError):
            def __str__(self): raise OSError('format failed')
        error = BadString('untouched')
    else:
        error = runtime.exc.TensorifyScalarRestartAnalysis('untouched')
    def broken(*args, **kwargs): raise OSError('observer failed')
    def restart(*args, **kwargs):
        if failure == 'identity': owner.compile_identity = broken
        elif failure == 'audit': owner.audit_restart_source = broken
        elif failure == 'publication': owner.on_update = broken
        raise error
    runtime.inductor.compile = restart
    with runtime.guards.compile_context(runtime.context):
        with pytest.raises(type(error)) as caught: trace.single_candidate_backend(object(), [])
    assert caught.value is error and error.args == ('untouched',)
    assert record['compile_calls'][0]['observation_errors']
    owner.on_update = None
    owner.finish()
    with pytest.raises(ValueError): trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_legacy_normal_valid_but_short_name_restart_never_passes(trace):
    record = policy_record(trace)
    trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')
    import copy
    failed = copy.deepcopy(record['compile_calls'][0])
    failed.update(status='failed', error={'type': 'TensorifyScalarRestartAnalysis', 'message': ''})
    record['compile_calls'].insert(0, failed)
    with pytest.raises(ValueError): trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')
    failed['status'] = 'restart'
    with pytest.raises(ValueError): trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_no_context_restart_is_rethrown_but_cannot_be_accepted(trace, monkeypatch):
    runtime = restart_runtime(trace, monkeypatch)
    owner, record = trace.configure_inductor_policy(runtime.torch)
    error = runtime.exc.TensorifyScalarRestartAnalysis()
    def restart(*args, **kwargs): raise error
    runtime.inductor.compile = restart
    with pytest.raises(type(error)) as caught: trace.single_candidate_backend(object(), [])
    assert caught.value is error
    assert record['compile_calls'][0]['identity'] is None
    with pytest.raises(ValueError, match='missing Dynamo'): trace.check_inductor_policy(owner.finish(), trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_missing_context_api_restart_is_never_accepted(trace, monkeypatch):
    runtime = restart_runtime(trace, monkeypatch)
    del runtime.imports['torch._guards']
    owner, record = trace.configure_inductor_policy(runtime.torch)
    error = runtime.exc.TensorifyScalarRestartAnalysis('no context API')
    def restart(*args, **kwargs): raise error
    runtime.inductor.compile = restart
    with pytest.raises(type(error)) as caught: trace.single_candidate_backend(object(), [])
    assert caught.value is error and error.args == ('no context API',)
    assert record['compile_calls'][0]['identity'] is None
    with pytest.raises(ValueError, match='missing Dynamo'): trace.check_inductor_policy(owner.finish(), trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('restarts,valid', [([0, 1], True), ([0, 0], False), ([1, 0], False)])
@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_repeated_restarts_require_increasing_attempts_and_eventual_return(trace, monkeypatch, restarts, valid):
    runtime = restart_runtime(trace, monkeypatch)
    original = runtime.inductor.compile
    owner, record = trace.configure_inductor_policy(runtime.torch)
    error = runtime.exc.TensorifyScalarRestartAnalysis()
    def restart(*args, **kwargs): raise error
    runtime.inductor.compile = restart
    with runtime.guards.compile_context(runtime.context):
        for attempt in restarts:
            runtime.context.attempt = attempt
            with pytest.raises(type(error)): trace.single_candidate_backend(object(), [])
        runtime.context.attempt = max(restarts) + 1
        runtime.inductor.compile = original
        trace.single_candidate_backend(object(), [])
    owner.finish()
    if valid: trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')
    else:
        with pytest.raises(ValueError, match='attempt'): trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_excessive_restart_count_is_a_checker_bound_not_a_compiler_change(trace, monkeypatch):
    runtime = restart_runtime(trace, monkeypatch)
    owner, record, _ = recovered_record(trace, runtime)
    owner.finish()
    import copy
    restart = record['compile_calls'][0]
    rows = []
    for attempt in range(trace.MAX_RESTARTS + 1):
        row = copy.deepcopy(restart)
        row['sequence'] = attempt
        row['identity']['attempt'] = row['identity_after']['attempt'] = attempt
        rows.append(row)
    recovered = record['compile_calls'][1]
    recovered['sequence'] = len(rows)
    recovered['identity']['attempt'] = recovered['identity_after']['attempt'] = len(rows)
    record['compile_calls'] = [*rows, recovered]
    with pytest.raises(ValueError, match='excessive'): trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_autograd_thread_callbacks_overlap_but_published_evidence_is_serialized(trace, monkeypatch):
    runtime = restart_runtime(trace, monkeypatch)
    snapshots, publication_active = [], threading.Lock()
    def publish(value):
        assert publication_active.acquire(blocking=False)
        try: snapshots.append(json.loads(json.dumps(value)))
        finally: publication_active.release()
    owner, record = trace.configure_inductor_policy(runtime.torch, on_update=publish)
    rendezvous = threading.Barrier(2, timeout=10)
    compile_graph = runtime.inductor.compile
    def compile_concurrently(graph, inputs, *, options):
        context = runtime.guards.CompileContext.try_get()
        if context.attempt == 0:
            raise runtime.exc.TensorifyScalarRestartAnalysis(restart_reason='engine retry')
        rendezvous.wait()  # Would time out if the observer serialized actual compilation.
        return compile_graph(graph, inputs, options=options)
    runtime.inductor.compile = compile_concurrently
    errors = []
    def backward(frame):
        try:
            context = runtime.guards.CompileContext(runtime.guards.CompileId(frame_id=frame, frame_compile_id=0))
            with runtime.guards.compile_context(context):
                with pytest.raises(runtime.exc.TensorifyScalarRestartAnalysis):
                    trace.single_candidate_backend(object(), [])
                context.attempt = 1
                assert trace.single_candidate_backend(object(), []) is runtime.result
        except BaseException as error: errors.append(error)
    threads = [threading.Thread(target=backward, args=(frame,), name=f'engine-{frame}') for frame in (1, 2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=15)
    assert not errors and all(not thread.is_alive() for thread in threads)
    owner.finish()
    trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')
    assert len({row['identity']['context'] for row in record['compile_calls']}) == 2
    assert len({row['identity']['thread_owner'] for row in record['compile_calls']}) == 2
    assert [row['status'] for row in record['compile_calls']].count('restart') == 2
    for snapshot in snapshots:
        for row in snapshot['compile_calls']:
            if row['status'] == 'returned':
                assert row['guards_after'] == trace.POLICY_GUARDS
                assert row['identity'] == row['identity_after']
                assert row['options_exit'] == trace.POLICY_FLAGS


@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_same_compile_id_in_new_live_context_cannot_recover_restart(trace, monkeypatch):
    runtime = restart_runtime(trace, monkeypatch)
    original = runtime.inductor.compile
    owner, record = trace.configure_inductor_policy(runtime.torch)
    error = runtime.exc.TensorifyScalarRestartAnalysis()
    def restart(*args, **kwargs): raise error
    runtime.inductor.compile = restart
    with runtime.guards.compile_context(runtime.context):
        with pytest.raises(type(error)): trace.single_candidate_backend(object(), [])
    foreign = runtime.guards.CompileContext(runtime.context.compile_id)
    foreign.attempt = 1
    runtime.inductor.compile = original
    with runtime.guards.compile_context(foreign): trace.single_candidate_backend(object(), [])
    assert record['compile_calls'][0]['identity']['context'] != record['compile_calls'][1]['identity']['context']
    with pytest.raises(ValueError, match='unrecovered'): trace.check_inductor_policy(owner.finish(), trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_recovery_after_default_restoration_collision_remains_rejected(trace, monkeypatch):
    runtime = restart_runtime(trace, monkeypatch)
    owner, record, _ = recovered_record(trace, runtime)
    runtime.registry.set_default_backend('foreign')
    with pytest.raises(ValueError, match='unowned'): owner.finish()
    assert runtime.registry.get_default_backend() == 'foreign'
    assert trace._policy_owner is None
    with pytest.raises(ValueError): trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('tamper', [False, True])
@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_actual_restart_source_attestation_uses_all_retained_hashes(trace, monkeypatch, tmp_path, tamper):
    if not AUDIT.is_dir() or not (RESTART_AUDIT / 'torch-dynamo-exc.py').is_file():
        pytest.skip('optional pinned image source audit is not available')
    attest = trace.BackendPolicy.audit_restart_source
    runtime = restart_runtime(trace, monkeypatch)
    monkeypatch.setattr(trace.BackendPolicy, 'audit_restart_source', attest)
    for name, digest in trace.RESTART_SOURCE.items():
        source = RESTART_AUDIT / 'torch-dynamo-exc.py' if name == '_dynamo/exc.py' else AUDIT / 'torch' / name
        data = source.read_bytes()
        assert trace.hashlib.sha256(data).hexdigest() == digest
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if tamper and name == '_guards.py':
            data = bytes([data[0] ^ 1]) + data[1:]  # Exactly one changed byte, with identical file length.
        target.write_bytes(data)
    runtime.guards.__file__ = str(tmp_path / '_guards.py')
    owner, record, _ = recovered_record(trace, runtime)
    owner.finish()
    if tamper:
        assert record['compile_calls'][0]['status'] == 'failed'
        assert record['compile_calls'][0]['observation_errors']
        with pytest.raises(ValueError): trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')
    else:
        assert record['restart_source'] == trace.RESTART_SOURCE
        trace.check_inductor_policy(record, trace.POLICY_NAME, 'stub')


@pytest.mark.parametrize('trace', ['memory_trace', 'sssglu_compile_probe'], indirect=True)
def test_restart_cpu_analog_without_optional_source_snapshots(trace, monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules[__name__], 'AUDIT', tmp_path / 'absent')
    runtime = restart_runtime(trace, monkeypatch)
    owner, record, _ = recovered_record(trace, runtime)
    trace.check_inductor_policy(owner.finish(), trace.POLICY_NAME, 'stub')
    assert record['schema_version'] == 2
