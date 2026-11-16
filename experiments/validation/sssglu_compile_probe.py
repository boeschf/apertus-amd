"""Check selected-policy native compiled SSSGLU against tiled analytic references.

This train-kind component never executes GPT, initializes process groups, or runs
an optimizer. It checks activation/input/probability gradients, not whole-MoE,
router, DDP, pipeline, optimizer, checkpoint, or performance equivalence.
Families have isolated Dynamo code caches; uninterrupted production compilation
budget behavior is deliberately not established by this component.
"""

from contextlib import contextmanager
from functools import wraps
import faulthandler
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import threading
import traceback
import tempfile
from types import MappingProxyType

from apertus_amd.artifacts import sha256
from apertus_amd.entrypoint import verify_imports


SOURCE = '1072583a9632fd1902544f7eaea562e2578ef1de'
IMAGE = 'cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0'
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
ROWS = (4096, 129, 257)
VARIANTS = (('dense', 17280, False), ('shared', 1920, False), ('expert', 1920, True))
DYNAMO_LIMITS = {'recompile_limit': 8, 'accumulated_recompile_limit': 256}
REPEATS = 2
REFERENCE_ROWS = 64
BF16_UNIT_ROUNDOFF = 2**-8
ELEMENT_RTOL = 4 * BF16_UNIT_ROUNDOFF
ELEMENT_ATOL = BF16_UNIT_ROUNDOFF
RMS_RTOL = 4 * BF16_UNIT_ROUNDOFF
RMS_ATOL = 1e-5
RESTART_CPU_TORCH = '2.14.0+rocm10.1.0'
RESTART_CPU_GIT = '11ef10939935619f2c723ba326e3808a750a924e'
RESTART_CPU_DYNAMO = {'assume_static_by_default': False, 'specialize_float': False}
RESTART_CPU_COMMAND = ['python3', '-u', '/apertus/validation.py', 'restart-cpu', '/apertus/output/0']
RESTART_CPU_SOURCE = MappingProxyType({
    **RESTART_SOURCE,
    '__init__.py': 'bd4a10ff16357ed78f45aaf4b46ce5b771fdc0e8549402206e324d0eb792c672',
    'version.py': '0acbe9df220d88e1375452334c72a056939f1eaa74640f78d036130caff96178',
    '_dynamo/backends/registry.py': 'f25502a2a2dac854484b0d755bd30cc2014f4fd0b7e0c8a25caec2e8d49d5d00',
    '_dynamo/config.py': 'd44a413bf4274ff0ce4384fb0538c3c48a9972af963d8f30c5b16416892ee6b1',
    '_dynamo/symbolic_convert.py': '7f59466c14b65f3ecf1bbfab6c8eaf6897d7af4fb50d4b66277051178d2774be',
    '_dynamo/variables/builder.py': '900e23d3fad5636389796eefe1863e861fd848c69040de0d84decd6e00c52142',
    '_dynamo/utils.py': 'e550c68c924e1c0babaf31fc629e900c1d3b6ecbcf10aa88442d39b405b7102a',
    '_functorch/_aot_autograd/graph_compile.py': 'bf72acad866b95a12cebd75d3010dcfc2a8de533b8b1f83fc7cf53140882022e',
    'fx/passes/_tensorify_python_scalars.py': '18e5ec282015bc698e0631f25c0ae11e527cb8955608004f839b8fe3ff132bc9',
    'utils/_config_module.py': 'a875ea7c7f74cd60fbd797e6fe29df344dde135eb82790613a51d446bd980edc',
})


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


def configure_policy(torch, on_update=None):
    """Require the explicit registered backend for this standalone component."""
    owner, record = configure_inductor_policy(torch, on_update)
    if owner is None:
        raise ValueError('component requires APERTUS_INDUCTOR_POLICY=single_candidate_backend')
    return owner, record


class Evidence:
    """Flush rank-owned phase/check events and keep an atomic valid summary on timeout."""

    def __init__(self, output: Path, rank: int, record: dict):
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / f'sssglu-compile-{rank}.json'
        self.temporary = output / f'sssglu-compile-{rank}.json.tmp'
        self.record = record
        self.started = time.monotonic()
        with self.path.open('x') as stream:
            stream.write(json.dumps(record) + '\n')
        self.stream = (output / f'sssglu-compile-{rank}.jsonl').open('x', buffering=1)

    def emit(self, kind: str, **values):
        """Persist phase boundaries before first-use compilation or GPU work."""
        event = {'kind': kind, 'time': time.time(), 'elapsed_seconds': time.monotonic() - self.started, **values}
        self.stream.write(json.dumps(event, allow_nan=False) + '\n')
        self.record['last_event'] = event
        if kind == 'check':
            self.record['checks'].append(event)
        if kind == 'native-call':
            self.record['native_calls'].append(event)
        self.temporary.write_text(json.dumps(self.record, allow_nan=False) + '\n')
        os.replace(self.temporary, self.path)

    def close(self):
        """Close only the rank's evidence stream; runtime owns cache/process cleanup."""
        self.stream.close()


@contextmanager
def phase(evidence, name):
    """Leave useful evidence if native compilation or GPU synchronization stalls."""
    evidence.emit('phase-begin', phase=name)
    began = time.monotonic()
    try:
        yield
    except BaseException as error:
        evidence.emit('phase-failed', phase=name, error=repr(error))
        raise
    evidence.emit('phase-end', phase=name, seconds=time.monotonic() - began)


@contextmanager
def observe_autotuners(cls, evidence):
    """Observe/delegate unchanged native methods; never suppress tuning or replace a kernel."""
    original_run, original_bench = cls.run, cls.benchmark_all_configs
    stats = {'run_calls': 0, 'benchmark_all_configs_calls': 0, 'kernels': []}
    seen = set()
    evidence.record['autotuners'] = stats

    @wraps(original_run)
    def run(autotuner, *args, **kwargs):
        # Worker precompilation can release configs before the first run, and
        # _precompile_worker releases them during run. Neither is a missing winner.
        configs_before = sequence_observation(getattr(autotuner, 'configs', None), 'released')
        launchers_before = sequence_observation(getattr(autotuner, 'launchers', None), 'unavailable')
        result = original_run(autotuner, *args, **kwargs)
        stats['run_calls'] += 1
        configs_after = sequence_observation(getattr(autotuner, 'configs', None), 'released')
        launchers_after = sequence_observation(getattr(autotuner, 'launchers', None), 'unavailable')
        meta = getattr(autotuner, 'inductor_meta', None)
        heuristic = getattr(autotuner, 'heuristic_type', None)
        row = {'family': evidence.record.get('active_family'),
               'native_call': evidence.record.get('active_native_call'),
               'kernel': getattr(getattr(autotuner, 'fn', None), '__name__', None),
               'heuristic': str(heuristic) if heuristic is not None else None,
               'config_count_before': configs_before['count'], 'config_state_before': configs_before['state'],
               'config_count': configs_after['count'], 'config_state': configs_after['state'],
               'launcher_count_before': launchers_before['count'], 'launcher_state_before': launchers_before['state'],
               'launcher_count': launchers_after['count'], 'launcher_state': launchers_after['state'],
               'policy_meta': {key: meta.get(key) for key in
                               ('autotune_pointwise', 'force_filter_reduction_configs', 'dynamic_scale_rblock')}
                              if isinstance(meta, dict) else None}
        key = json.dumps(row, sort_keys=True)
        if key not in seen:
            seen.add(key)
            stats['kernels'].append(row)
            evidence.emit('autotuner-observed', **row)
        return result

    @wraps(original_bench)
    def bench(autotuner, *args, **kwargs):
        stats['benchmark_all_configs_calls'] += 1
        evidence.emit('benchmark-observed', family=evidence.record.get('active_family'),
                      native_call=evidence.record.get('active_native_call'),
                      kernel=getattr(getattr(autotuner, 'fn', None), '__name__', None))
        return original_bench(autotuner, *args, **kwargs)

    cls.run, cls.benchmark_all_configs = run, bench
    try:
        yield stats
    finally:
        cls.run, cls.benchmark_all_configs = original_run, original_bench


def sequence_observation(value, none_state):
    """Describe nullable runtime-owned sequences without constructing or mutating them."""
    if value is None:
        return {'count': None, 'state': none_state}
    if isinstance(value, (list, tuple)):
        return {'count': len(value), 'state': 'present' if value else 'empty'}
    return {'count': None, 'state': f'unsupported:{type(value).__name__}'}


def required_labels() -> set[str]:
    """Define complete coverage independently of a report's claimed success."""
    labels = set()
    for variant, _, weighted in VARIANTS:
        for rows in ROWS:
            for call in ('direct', 'autograd0', 'autograd1'):
                for value in ('forward', 'input-gradient', *(['probability-gradient'] if weighted else [])):
                    labels.add(f'{variant}/rows{rows}/{call}/{value}')
    return labels


def required_native_labels() -> set[str]:
    """Require observed compiled execution for every direct/autograd forward and backward."""
    return {f'{variant}/rows{rows}/{call}-{direction}'
            for variant, _, _ in VARIANTS for rows in ROWS
            for call in ('direct', 'autograd0', 'autograd1') for direction in ('forward', 'backward')}


def native_call(torch, operation, label, evidence):
    """Call unchanged native code once and reject unobserved/eager fallback execution."""
    before = evidence.record['autotuners']['run_calls']
    evidence.record['active_native_call'] = label
    try:
        with phase(evidence, label):
            result = operation()
            torch.cuda.synchronize()
    finally:
        evidence.record.pop('active_native_call')
    count = evidence.record['autotuners']['run_calls'] - before
    evidence.emit('native-call', label=label, observed_compiled_launches=count)
    if count <= 0:
        raise ValueError(f'{label}: compiled launch was not observed (possible eager fallback)')
    return result


def analytic_reference(torch, y, dy, weights):
    """Evaluate the continuous gate and its analytic derivatives in FP64 on one row tile."""
    first, second = y.double().chunk(2, dim=-1)
    u = first - 1
    denominator = 1 + u.abs()
    gate = 0.5 + u / denominator
    derivative = denominator.reciprocal().square()
    grad = dy.double()
    activation = gate * second
    if weights is None:
        return {'forward': activation,
                'input-gradient': torch.cat((grad * derivative * second, grad * gate), dim=-1)}, None
    probability = weights.double()
    products = activation * grad
    return {'forward': activation * probability,
            'input-gradient': torch.cat((grad * probability * derivative * second,
                                         grad * probability * gate), dim=-1),
            'probability-gradient': products.sum(dim=-1, keepdim=True)}, products.abs().sum(dim=-1, keepdim=True)


def metric_failures(metrics) -> list[str]:
    """Use fixed BF16-arithmetic budgets, including the weighted reduction's conditioning."""
    if not metrics['finite'] or any(not math.isfinite(metrics[key]) for key in
                                    ('max_abs', 'rms_error', 'reference_rms', 'relative_l2')):
        return ['nonfinite values or metrics']
    failures = []
    if metrics['mismatched_elements']:
        failures.append('elementwise precision gate')
    if metrics['rms_error'] > RMS_ATOL + RMS_RTOL * metrics['reference_rms']:
        failures.append('RMS precision gate')
    return failures


def check_values(torch, y, dy, weights, actual, label, evidence):
    """Check all rows without retaining a full FP64 activation or gradient reference."""
    totals = {key: {'numel': 0, 'error_squared': 0.0, 'reference_squared': 0.0,
                    'max_abs': 0.0, 'mismatched_elements': 0, 'finite': True} for key in actual}
    with phase(evidence, label + '/reference'):
        for start in range(0, y.shape[0], REFERENCE_ROWS):
            end = min(start + REFERENCE_ROWS, y.shape[0])
            references, probability_scale = analytic_reference(
                torch, y[start:end], dy[start:end], None if weights is None else weights[start:end])
            for key, value in actual.items():
                reference = references[key]
                observed = value[start:end].detach().double()
                if observed.shape != reference.shape:
                    raise ValueError(f'{label}/{key}: shape mismatch')
                total = totals[key]
                total['numel'] += reference.numel()
                if not bool(torch.isfinite(observed).all() and torch.isfinite(reference).all()):
                    total['finite'] = False
                    continue
                difference = observed - reference
                scale = probability_scale if key == 'probability-gradient' else reference.abs()
                allowance = ELEMENT_ATOL + ELEMENT_RTOL * scale
                total['max_abs'] = max(total['max_abs'], difference.abs().max().item())
                total['error_squared'] += difference.square().sum().item()
                total['reference_squared'] += reference.square().sum().item()
                total['mismatched_elements'] += int((difference.abs() > allowance).sum().item())
            if start == 0 or end == y.shape[0]:
                evidence.emit('reference-progress', label=label, rows_checked=end)
        for key, metrics in totals.items():
            if metrics['finite']:
                metrics.update(rms_error=math.sqrt(metrics['error_squared'] / metrics['numel']),
                               reference_rms=math.sqrt(metrics['reference_squared'] / metrics['numel']),
                               relative_l2=math.sqrt(metrics['error_squared'] / max(metrics['reference_squared'], 1e-60)))
                metrics['finite'] = all(math.isfinite(metrics[name]) for name in
                                        ('rms_error', 'reference_rms', 'relative_l2', 'max_abs'))
            failures = metric_failures(metrics)
            safe = {name: None if isinstance(value, float) and not math.isfinite(value) else value
                    for name, value in metrics.items()}
            evidence.emit('check', label=f'{label}/{key}', passed=not failures, failures=failures, **safe)


def tensor_spec(value):
    """Retain actual native tensor dtype/layout and grad-state assumptions."""
    return {'shape': list(value.shape), 'stride': list(value.stride()), 'dtype': str(value.dtype),
            'requires_grad': value.requires_grad}


def run_case(torch, native, case, evidence):
    """Exercise native direct functions and their unchanged custom-autograd entry points."""
    variant, hidden, weighted, rows = case
    label = f'{variant}/rows{rows}'
    step = 2 if rows == ROWS[-1] else 1
    y = (torch.randn(rows * step, 2 * hidden, device='cuda', dtype=torch.bfloat16) * 0.5)[::step]
    dy = (torch.randn(rows * step, hidden, device='cuda', dtype=torch.bfloat16) * 0.25)[::step]
    weights = (torch.rand(rows * step, 1, device='cuda', dtype=torch.float32) * 0.5)[::step] if weighted else None
    evidence.emit('case', label=label, y=tensor_spec(y), dy=tensor_spec(dy),
                  weights=None if weights is None else tensor_spec(weights))
    with torch.no_grad():
        output = native_call(torch, lambda: native.weighted_sssglu(y, weights) if weighted else native.sssglu(y),
                             label + '/direct-forward', evidence)
        gradients = native_call(torch, lambda: native.weighted_sssglu_back(dy, y, weights) if weighted else native.sssglu_back(dy, y),
                                label + '/direct-backward', evidence)
    input_grad, probability_grad = gradients if weighted else (gradients, None)
    values = {'forward': output, 'input-gradient': input_grad}
    if weighted:
        values['probability-gradient'] = probability_grad
    validate_returns(torch, values)
    check_values(torch, y, dy, weights, values, label + '/direct', evidence)
    del output, input_grad, probability_grad, gradients, values
    y = y.detach().requires_grad_(True)
    if weighted:
        weights = weights.detach().requires_grad_(True)
    evidence.emit('autograd-inputs', label=label, y=tensor_spec(y),
                  weights=None if weights is None else tensor_spec(weights))
    for repeat in range(REPEATS):
        y.grad = None
        if weighted:
            weights.grad = None
        output = native_call(torch, lambda: native.weighted_bias_sssglu_impl(y, None, weights) if weighted else
                             native.bias_sssglu_impl(y, None), label + f'/autograd{repeat}-forward', evidence)
        native_call(torch, lambda: output.backward(dy), label + f'/autograd{repeat}-backward', evidence)
        values = {'forward': output, 'input-gradient': y.grad}
        if weighted:
            values['probability-gradient'] = weights.grad
        validate_returns(torch, values)
        check_values(torch, y.detach(), dy, None if weights is None else weights.detach(), values,
                     label + f'/autograd{repeat}', evidence)
        del output, values


def validate_returns(torch, values):
    """Enforce source-defined BF16 activation/input grads and FP32 probability grads."""
    for name, value in values.items():
        expected = torch.float32 if name == 'probability-gradient' else torch.bfloat16
        if value is None or value.dtype != expected:
            raise ValueError(f'unexpected native {name} dtype or missing gradient')


def run_families(torch, native, evidence):
    """Isolate component families, not rows or production recompilation budgets."""
    dynamo = torch._dynamo
    isolation = {'method': 'torch._dynamo.reset_between_families',
                 'limits_start': config_values(dynamo.config, DYNAMO_LIMITS), 'resets': [], 'families': []}
    evidence.record['case_isolation'] = isolation
    if isolation['limits_start'] != DYNAMO_LIMITS:
        raise ValueError('component requires unchanged pinned Dynamo recompile limits')
    for index, (family, width, weighted) in enumerate(VARIANTS):
        if index:
            # Pinned Dynamo __init__.py:138-218 resets code caches and compile
            # counts, not filesystem caches or config. Existing compile wrappers
            # retain their callback (eval_frame.py:1139-1246); never rewrap native
            # functions or raise limits. Production can still exhaust its budget.
            # All local GPU work must finish before releasing compiled products.
            with phase(evidence, f'family-reset/{VARIANTS[index - 1][0]}-to-{family}'):
                torch.cuda.synchronize()
                before = config_values(dynamo.config, DYNAMO_LIMITS)
                if before != DYNAMO_LIMITS:
                    raise ValueError('Dynamo recompile limits changed before family reset')
                dynamo.reset()
                after = config_values(dynamo.config, DYNAMO_LIMITS)
                reset = {'after_family': VARIANTS[index - 1][0], 'before_family': family,
                         'limits_before': before, 'limits_after': after}
                isolation['resets'].append(reset)
                evidence.emit('family-reset', **reset)
                if after != before:
                    raise ValueError('Dynamo recompile limits changed during family reset')
        evidence.record['active_family'] = family
        family_record = {'family': family, 'limits_before': config_values(dynamo.config, DYNAMO_LIMITS)}
        isolation['families'].append(family_record)
        with phase(evidence, f'family/{family}'):
            # Keep dynamic promotion, row strides, and grad-mode transitions in
            # one uninterrupted Dynamo cache/budget within each family.
            for rows in ROWS:
                run_case(torch, native, (family, width, weighted, rows), evidence)
            family_record['limits_after'] = config_values(dynamo.config, DYNAMO_LIMITS)
            if family_record['limits_after'] != DYNAMO_LIMITS:
                raise ValueError('Dynamo recompile limits changed within family')
    isolation['limits_end'] = config_values(dynamo.config, DYNAMO_LIMITS)
    evidence.record.pop('active_family')
    evidence.emit('families-complete')


def validate(settings):
    """Require the explicit cold component, not a training or cache-reuse comparison."""
    if settings['environment'].get('APERTUS_INDUCTOR_POLICY') == 'single_candidate':
        raise ValueError('unsafe ambient single_candidate policy is rejected')
    if settings['source']['commit'] != SOURCE or settings['image']['sha256'] != IMAGE or settings['args'] or \
            settings['procedure']['kind'] != 'train' or settings['environment'].get('APERTUS_INDUCTOR_POLICY') != POLICY_NAME:
        raise ValueError('component requires pinned source/image and explicit selected policy')
    if settings['cache']['mode'] != 'cold':
        raise ValueError('component requires --cache-mode cold; a cached winner cannot establish first-use coverage')
    if settings['resources']['nodes'] != 1 or settings['resources']['gpus_per_node'] != 4:
        raise ValueError('component requires one node with four visible local-rank GPUs')


def validate_outcome(record):
    """Require complete numerical coverage and observed singleton launches without tuning."""
    checks = record['checks']
    caches = record['initial_compiler_caches']
    if set(caches) != {'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'} or any(
        row['files'] != 0 for row in caches.values()
    ):
        raise ValueError('cold first-use compiler-cache evidence missing')
    isolation = record.get('case_isolation', {})
    families = [family for family, _, _ in VARIANTS]
    if isolation.get('method') != 'torch._dynamo.reset_between_families' or \
            isolation.get('limits_start') != DYNAMO_LIMITS or isolation.get('limits_end') != DYNAMO_LIMITS or \
            isolation.get('families') != [{'family': family, 'limits_before': DYNAMO_LIMITS,
                                           'limits_after': DYNAMO_LIMITS} for family in families] or \
            isolation.get('resets') != [{'after_family': previous, 'before_family': following,
                                         'limits_before': DYNAMO_LIMITS, 'limits_after': DYNAMO_LIMITS}
                                        for previous, following in zip(families, families[1:])]:
        raise ValueError('component family isolation or unchanged recompile-limit evidence missing')
    if len(checks) != len(required_labels()) or {r['label'] for r in checks} != required_labels() or \
            any(not r['passed'] for r in checks):
        raise ValueError('incomplete or failed native SSSGLU numerical coverage')
    calls = record['native_calls']
    if len(calls) != len(required_native_labels()) or {r['label'] for r in calls} != required_native_labels() or \
            any(r['observed_compiled_launches'] <= 0 for r in calls):
        raise ValueError('compiled native forward/backward execution coverage missing')
    check_inductor_policy(record['policy'], POLICY_NAME, record['torch'])
    stats = record['autotuners']
    if stats['run_calls'] != sum(row['observed_compiled_launches'] for row in calls) or \
            stats['benchmark_all_configs_calls'] != 0 or not stats['kernels']:
        raise ValueError('autotuner observation missing or benchmarking occurred')
    expected_meta = {'autotune_pointwise': False, 'force_filter_reduction_configs': True, 'dynamic_scale_rblock': False}
    if any(row.get('launcher_count_before') != 1 or row.get('launcher_count') != 1 or row.get('policy_meta') != expected_meta or
           not isinstance(row.get('kernel'), str) or not row['kernel'] or
           not isinstance(row.get('heuristic'), str) or not row['heuristic']
           for row in stats['kernels']):
        raise ValueError('single pre/post-run launcher, kernel identity, and policy metadata must be observed')
    if not all(any(kind in row['heuristic'] for row in stats['kernels']) for kind in ('POINTWISE', 'REDUCTION')):
        raise ValueError('pointwise and reduction launches must both be observed')
    if {row.get('family') for row in stats['kernels']} != set(families):
        raise ValueError('native policy metadata must be observed independently for every family')
    if {row.get('native_call') for row in stats['kernels']} != required_native_labels() or any(
        row['family'] != row['native_call'].split('/')[0] for row in stats['kernels']
    ):
        raise ValueError('native metadata must cover every forward/backward grad context')


def train(output: Path, source: Path, entrypoint: str, argv: list[str], *, run=Path('/apertus')):
    """Verify frozen source and run only activation components under an own-process watchdog."""
    settings = json.loads((run / 'resolved.json').read_text())
    validate(settings)
    manifest = json.loads((run / 'source/manifest.json').read_text())
    if manifest['selection']['commit'] != SOURCE or entrypoint != settings['entrypoint'] or \
            sha256(source / entrypoint) != manifest['files'][entrypoint]:
        raise ValueError('selected entrypoint/source identity mismatch')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    timeout = int(os.environ['APERTUS_SSSGLU_TIMEOUT_SECONDS'])
    if not 1 <= timeout <= 1800:
        raise ValueError('component timeout must be between 1 and 1800 seconds')
    record = {'status': 'running', 'source': SOURCE, 'image': IMAGE, 'rank': rank, 'local_rank': local,
              'probe_sha256': sha256(Path(__file__)), 'timeout_seconds': timeout, 'checks': [], 'native_calls': [],
              'reference': 'tiled FP64 continuous SSSGLU and analytic derivatives; no eager/native substitution',
              'tolerances': {'element_rtol': ELEMENT_RTOL, 'element_atol': ELEMENT_ATOL,
                             'rms_rtol': RMS_RTOL, 'rms_atol': RMS_ATOL,
                             'probability_element_scale': 'sum(abs(activation * upstream_gradient))'},
              'not_checked': ['default-compiled counterfactual', 'whole MoE/router', 'DDP/optimizer',
                               'pipeline/communication', 'checkpoints', 'performance acceptance',
                               'uninterrupted cross-family/production recompile budget']}
    evidence = Evidence(output, rank, record)
    policy_owner = None
    original_path = sys.path.copy()
    # The C-level watchdog exits THIS probe process even while Python waits in a
    # compiled operator. Runtime still owns reaping and job-private cache cleanup.
    faulthandler.dump_traceback_later(timeout, exit=True)
    try:
        record['initial_compiler_caches'] = {}
        for variable in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
            root = Path(os.environ[variable])
            if not root.is_dir() or any(path.is_file() for path in root.rglob('*')):
                raise ValueError(f'nonempty initial component cache: {variable}')
            record['initial_compiler_caches'][variable] = {'path': str(root), 'files': 0}
        evidence.emit('cold-cache-confirmed')
        with phase(evidence, 'torch-and-policy'):
            import torch
            torch.cuda.set_device(local)
            torch.manual_seed(42)
            def policy_update(value):
                record['policy'] = value
                evidence.emit('backend-policy')
            policy_owner, record['policy'] = configure_policy(torch, on_update=policy_update)
            record.update(torch=torch.__version__, device=torch.cuda.get_device_name(),
                          environment={key: os.environ.get(key) for key in
                                       ('APERTUS_INDUCTOR_POLICY', 'TORCH_COMPILE_DISABLE', 'PYTORCH_HIP_ALLOC_CONF',
                                        'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')})
        sys.path.insert(0, str(source.resolve()))
        try:
            with phase(evidence, 'native-import'):
                verify_imports(source, manifest['files'])
                native = importlib.import_module('megatron.core.fusions.fused_bias_sssglu')
                jit = importlib.import_module('megatron.core.jit')
                if jit.jit_fuser is not torch.compile or not all(
                    hasattr(getattr(native, name), '_torchdynamo_orig_callable') for name in
                    ('sssglu', 'weighted_sssglu', 'sssglu_back', 'weighted_sssglu_back')
                ):
                    raise ValueError('native torch.compile fused SSSGLU wrappers were not selected')
                record['native_file_sha256'] = sha256(Path(native.__file__))
                verify_imports(source, manifest['files'])
            from torch._inductor.runtime.triton_heuristics import CachingAutotuner
            with observe_autotuners(CachingAutotuner, evidence):
                run_families(torch, native, evidence)
            record['policy'] = finish_inductor_policy(policy_owner, record['policy'])
            validate_outcome(record)
        finally:
            record['verified_modules'] = verify_imports(source, manifest['files'])
        record['status'] = 'passed'
        evidence.emit('complete')
    except BaseException as error:
        record.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        evidence.emit('failed')
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        sys.path[:] = original_path
        try:
            if policy_owner is not None and not policy_owner.closed:
                record['policy'] = finish_inductor_policy(policy_owner, record['policy'])
        finally:
            evidence.close()


def check(run: Path):
    """Check all component ranks independently, without claiming GPT/optimizer correctness."""
    settings = json.loads((run / 'resolved.json').read_text())
    validate(settings)
    expected = json.loads((run / 'source/manifest.json').read_text())['files']
    records = []
    for rank in range(4):
        record = json.loads((run / f'output/sssglu-compile-{rank}.json').read_text())
        if record['status'] != 'passed' or record['source'] != SOURCE or record['image'] != IMAGE or \
                record['rank'] != rank or record['local_rank'] != rank or record.get('verified_modules', 0) <= 0 or \
                record['probe_sha256'] != settings['validation']['sha256'] or \
                record['native_file_sha256'] != expected['megatron/core/fusions/fused_bias_sssglu.py']:
            raise ValueError(f'incomplete component rank {rank}')
        validate_outcome(record)
        records.append({'rank': rank, 'checks': len(record['checks']), 'kernels': len(record['autotuners']['kernels'])})
    from apertus_amd.artifacts import write_json
    write_json(run / 'output/checks.json', {'status': 'passed', 'scope': 'native SSSGLU component and selected compiler policy',
                                          'not_checked': 'full training, MoE/router/DDP/optimizer/checkpoint equivalence; uninterrupted production recompile budget',
                                          'ranks': records})


def validate_restart_cpu(settings):
    """Accept only the separate pinned, single-process commands-kind CPU fixture protocol."""
    resource, procedure, env = settings['resources'], settings['procedure'], settings['environment']
    if settings['image']['sha256'] != IMAGE or procedure.get('kind') != 'commands' or \
            procedure.get('commands') != [RESTART_CPU_COMMAND] or procedure.get('timeout') != 180 or \
            any(settings.get(name) for name in ('source', 'source_snapshot', 'args', 'prepare', 'compare_run')):
        raise ValueError('restart-cpu requires its pinned image-only command protocol')
    seconds = sum(int(value) * scale for value, scale in zip(resource['time'].split(':'), (3600, 60, 1)))
    if resource['nodes'] != 1 or resource['gpus_per_node'] != 1 or resource.get('cpus_per_task') != 4 or \
            resource.get('reservation') != 'cscs' or resource.get('cpu_bind') != 'cores' or \
            not 0 < seconds <= 300 or settings['cache']['mode'] != 'auto':
        raise ValueError('restart-cpu requires one planned rank, four CPUs, at most five minutes and no training cache mode')
    required = {'APERTUS_INDUCTOR_POLICY': POLICY_NAME, 'TORCH_COMPILE_DISABLE': '0',
                'TENSORIFY_PYTHON_SCALARS': '1', 'APERTUS_RESTART_CPU_TIMEOUT_SECONDS': '150'}
    if any(env.get(name) != value for name, value in required.items()) or \
            env.get('TORCHDYNAMO_DISABLE', '0') != '0' or env.get('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS', '0') != '0':
        raise ValueError('restart-cpu fixture environment mismatch')
    if not settings.get('validation') or settings['validation']['sha256'] != sha256(Path(__file__)):
        raise ValueError('restart-cpu frozen validator identity mismatch')


def restart_cpu_sources(torch):
    """Attest the actual raise pass and complete audited Python path without compiling or touching CUDA."""
    if torch.__version__ != RESTART_CPU_TORCH or torch.version.git_version != RESTART_CPU_GIT:
        raise ValueError('restart-cpu requires exact pinned Torch version/git identity')
    root = Path(torch.__file__).resolve().parent
    actual = {name: sha256(root / name) for name in RESTART_CPU_SOURCE}
    if actual != RESTART_CPU_SOURCE:
        raise ValueError('restart-cpu unaudited native Python source')
    return actual


def restart_cpu_caches(scratch: Path, job: str, run: Path):
    """Create empty UID/job-owned private caches; intentionally never remove compiler-worker storage."""
    if not job.isdigit() or '..' in scratch.parts or scratch.is_symlink() or \
            scratch.resolve() != scratch or not scratch.is_relative_to('/dev/shm') or scratch == Path('/dev/shm'):
        raise ValueError('restart-cpu requires safe job-owned scratch below /dev/shm')
    parent = scratch / f'restart-cpu-u{os.getuid()}'
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent.is_symlink() or parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o077:
        raise ValueError('restart-cpu cache parent is not private and UID-owned')
    root = Path(tempfile.mkdtemp(prefix=f'j{job}-p{os.getpid()}-', dir=parent))
    owner = {'uid': os.getuid(), 'job_id': job, 'pid': os.getpid(), 'run': str(run)}
    (root / 'owner.json').write_text(json.dumps(owner) + '\n')
    paths = {}
    for variable, name in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                           ('TRITON_CACHE_DIR', 'triton'), ('TORCH_EXTENSIONS_DIR', 'extensions')):
        directory = root / name
        directory.mkdir(mode=0o700)
        paths[variable] = str(directory)
    return {'root': str(root), 'owner': owner, 'environment': paths,
            'initial_files': {name: 0 for name in paths}, 'mode': 'cold-private',
            'disposition': 'preserved; enclosing commands runtime reaps the process group after fixture exit',
            'publication': 'not applicable; no restore/capture/publication of training caches'}


class RestartCpuEvidence:
    """Retain exclusive stage/failure records independently of the SSSGLU report format."""

    def __init__(self, output, record):
        output.mkdir(parents=True, exist_ok=True)
        self.path, self.record = output / 'restart-cpu.json', record
        with self.path.open('x') as stream:
            json.dump(record, stream, allow_nan=False)
        self.stream = (output / 'restart-cpu.jsonl').open('x', buffering=1)
        self.started = time.monotonic()
        self.lock = threading.RLock()

    def emit(self, stage):
        """Flush bounded stage evidence and replace only this fixture's own snapshot."""
        with self.lock:
            event = {'stage': stage, 'time': time.time(), 'elapsed_seconds': time.monotonic() - self.started}
            self.record['last_event'] = event
            self.stream.write(json.dumps(event, allow_nan=False) + '\n')
            self.snapshot()

    def snapshot(self):
        """Persist final state independently of stream closure and callback publication failures."""
        from apertus_amd.artifacts import write_json
        with self.lock:
            write_json(self.path, self.record)

    def close(self):
        """Close evidence only, never compiler-worker caches or child processes."""
        self.stream.close()


def restart_cpu_error(error):
    """Format bounded diagnostics without allowing a hostile error formatter to replace the primary failure."""
    try:
        message = str(error)[:2048]
    except BaseException:
        message = '<exception formatting failed>'
    return {'module': type(error).__module__, 'qualname': type(error).__qualname__, 'message': message}


def run_restart_cpu_graph(torch, dynamo, record, evidence):
    """Run the one source-grounded deterministic CPU graph, never inject or retry an exception."""
    names = (*RESTART_CPU_DYNAMO, 'capture_scalar_outputs')
    record['dynamo_before'] = config_values(dynamo, names)
    record['fixture_dynamo'] = dict(RESTART_CPU_DYNAMO)
    record['compile_request'] = {'fullgraph': True, 'dynamic': 'unspecified', 'backend': 'selected registered default'}

    def resize(image, scale_factor):
        return torch.nn.functional.interpolate(image[None], size=None, scale_factor=scale_factor,
                   mode='bilinear', recompute_scale_factor=True, align_corners=False)[0]

    image = torch.arange(3 * 8 * 12, dtype=torch.float32, device='cpu').reshape(3, 8, 12) / 100
    scale = 1.873536229133606
    reference = resize(image, scale)
    record['reference'] = {'shape': list(reference.shape), 'device': reference.device.type,
                           'dtype': str(reference.dtype), 'requires_grad': reference.requires_grad,
                           'finite': bool(torch.isfinite(reference).all())}
    record['numerics'] = []
    failure = None
    try:
        with dynamo.patch(RESTART_CPU_DYNAMO):
            record['dynamo_during'] = config_values(dynamo, names)
            evidence.emit('compile-and-first-call')
            compiled = torch.compile(resize, fullgraph=True)
            for label in ('first', 'repeat'):
                actual = compiled(image, scale)
                row = {'label': label, 'shape': list(actual.shape), 'device': actual.device.type,
                       'dtype': str(actual.dtype), 'requires_grad': actual.requires_grad,
                       'finite': bool(torch.isfinite(actual).all()),
                       'close': bool(torch.allclose(actual, reference, rtol=1e-5, atol=1e-6)),
                       'rtol': 1e-5, 'atol': 1e-6}
                record['numerics'].append(row)
                if row['shape'] != list(reference.shape) or row['shape'] != [3, 14, 22] or \
                        row['device'] != 'cpu' or actual.dtype != torch.float32 or row['requires_grad'] or \
                        not row['finite'] or not row['close']:
                    raise ValueError('restart-cpu interpolation numerical/device gate failed')
                evidence.emit(label + '-numerical-check')
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            record['dynamo_after'] = config_values(dynamo, names)
        except BaseException as error:
            record.setdefault('secondary_errors', []).append({'stage': 'dynamo_exit', **restart_cpu_error(error)})
            if failure is None:
                raise


def validate_restart_cpu_outcome(record):
    """Require natural exact-class recovery and CPU numerics, never reinterpret a SSSGLU outcome."""
    if record.get('status') != 'passed' or record.get('sources_start') != RESTART_CPU_SOURCE or \
            record.get('sources_end') != RESTART_CPU_SOURCE or record.get('torch') != RESTART_CPU_TORCH or \
            record.get('torch_git') != RESTART_CPU_GIT or record.get('tensorify_enabled') is not True or \
            record.get('cuda_initialized_before') is not False or record.get('cuda_initialized_after') is not False or \
            record.get('environment_restored') is not True or record.get('secondary_errors') != []:
        raise ValueError('restart-cpu provenance, CPU-only or observation evidence incomplete')
    before = record.get('dynamo_before', {})
    if set(before) != {*RESTART_CPU_DYNAMO, 'capture_scalar_outputs'} or record.get('dynamo_after') != before or \
            record.get('fixture_dynamo') != RESTART_CPU_DYNAMO or \
            record.get('dynamo_during') != {**before, **RESTART_CPU_DYNAMO} or \
            record.get('compile_request') != {'fullgraph': True, 'dynamic': 'unspecified',
                                              'backend': 'selected registered default'} or \
            record.get('limits_before') != DYNAMO_LIMITS or record.get('limits_after') != DYNAMO_LIMITS:
        raise ValueError('restart-cpu fixture configuration/restoration drift')
    rows = record.get('numerics', [])
    if record.get('reference') != {'shape': [3, 14, 22], 'device': 'cpu', 'dtype': 'torch.float32',
                                   'requires_grad': False, 'finite': True} or \
            len(rows) != 2 or [row.get('label') for row in rows] != ['first', 'repeat'] or any(
            row != {'label': row['label'], 'shape': [3, 14, 22], 'device': 'cpu', 'dtype': 'torch.float32', 'requires_grad': False,
                    'finite': True, 'close': True, 'rtol': 1e-5, 'atol': 1e-6} for row in rows):
        raise ValueError('restart-cpu numerical coverage missing')
    caches = record.get('private_caches', {})
    if caches.get('mode') != 'cold-private' or caches.get('preserved_at_exit') is not True or \
            set(caches.get('initial_files', {})) != {'TMPDIR', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR',
                                                   'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'} or \
            any(caches['initial_files'].values()):
        raise ValueError('restart-cpu fresh private-cache evidence missing')
    policy = record.get('policy', {})
    check_inductor_policy(policy, POLICY_NAME, RESTART_CPU_TORCH)
    if policy.get('schema_version') != POLICY_SCHEMA or not any(
            call.get('status') == 'restart' for call in policy['compile_calls']):
        raise ValueError('natural native Tensorify restart not observed; no branch-coverage pass')


def restart_cpu(output: Path, *, run=Path('/apertus')):
    """Run only the explicit CPU command fixture; preserve caches until enclosing runtime reaps helpers."""
    record = {'status': 'running', 'mode': 'restart-cpu', 'probe_sha256': sha256(Path(__file__)),
              'secondary_errors': [], 'not_checked': ['GPU execution', 'SSSGLU gates', 'full model',
                  'training/checkpoints', 'GPU memory/performance', 'training cache publication or warm reuse']}
    evidence = RestartCpuEvidence(output, record)
    owner, torch, dynamo, failure, failure_traceback = None, None, None, None, None
    old_environment = {}
    def secondary(stage, error):
        nonlocal failure, failure_traceback
        record['secondary_errors'].append({'stage': stage, **restart_cpu_error(error)})
        if failure is None:
            failure, failure_traceback = error, error.__traceback__
    try:
        settings = json.loads((run / 'resolved.json').read_text())
        validate_restart_cpu(settings)
        if output.resolve() != (run / 'output/0').resolve() or any(
                os.environ.get(name) != value for name, value in
                (('SLURM_NTASKS', '1'), ('SLURM_PROCID', '0'), ('SLURM_LOCALID', '0'))):
            raise ValueError('restart-cpu requires the single planned command rank/output')
        for name, expected in settings['environment'].items():
            if os.environ.get(name) != expected:
                raise ValueError(f'restart-cpu effective environment mismatch: {name}')
        if os.environ.get('TORCHDYNAMO_DISABLE', '0') != '0' or \
                os.environ.get('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS', '0') != '0':
            raise ValueError('restart-cpu inherited Dynamo override')
        faulthandler.dump_traceback_later(150, exit=True)
        record['private_caches'] = restart_cpu_caches(Path(settings['scratch_root']), os.environ.get('SLURM_JOB_ID', ''), run)
        old_environment = {name: os.environ.get(name) for name in record['private_caches']['environment']}
        os.environ.update(record['private_caches']['environment'])
        evidence.emit('cold-private-caches')
        torch = importlib.import_module('torch')
        record.update(torch=torch.__version__, torch_git=torch.version.git_version,
                      cuda_initialized_before=torch.cuda.is_initialized())
        if record['cuda_initialized_before']:
            raise ValueError('restart-cpu requires uninitialized CUDA/HIP')
        record['sources_start'] = restart_cpu_sources(torch)
        dynamo = importlib.import_module('torch._dynamo.config')
        utils = importlib.import_module('torch._dynamo.utils')
        record['tensorify_enabled'] = utils._is_tensorify_enabled()
        record['limits_before'] = config_values(dynamo, DYNAMO_LIMITS)
        if record['tensorify_enabled'] is not True or record['limits_before'] != DYNAMO_LIMITS:
            raise ValueError('restart-cpu enablement/native limits mismatch')
        def policy_update(value):
            record['policy'] = value
            evidence.emit('backend-policy')
        owner, record['policy'] = configure_policy(torch, on_update=policy_update)
        evidence.emit('pinned-policy-selected')
        run_restart_cpu_graph(torch, dynamo, record, evidence)
    except BaseException as error:
        failure, failure_traceback = error, error.__traceback__
    finally:
        # Observe every restoration failure, but never replace an existing compiler/fixture exception.
        actions = []
        if owner is not None and not owner.closed:
            def finish_policy():
                try:
                    owner.finish()
                finally:
                    # A failed callback publication must not hide successful default restoration.
                    record['policy'] = owner.record
            actions.append(('policy_finish', finish_policy))
        if torch is not None:
            actions.append(('source_exit', lambda: record.update(sources_end=restart_cpu_sources(torch),
                               cuda_initialized_after=torch.cuda.is_initialized())))
        if dynamo is not None:
            actions.append(('limits_exit', lambda: record.update(limits_after=config_values(dynamo, DYNAMO_LIMITS))))
        for label, action in actions:
            try:
                action()
            except BaseException as error:
                secondary(label, error)
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        record['environment_restored'] = all(os.environ.get(name) == value for name, value in old_environment.items())
        if 'private_caches' in record:
            record['private_caches']['preserved_at_exit'] = all(
                Path(value).is_dir() for value in record['private_caches']['environment'].values())
        try:
            faulthandler.cancel_dump_traceback_later()
        except BaseException as error:
            secondary('watchdog_cancel', error)
        if failure is None:
            record['status'] = 'passed'
            try:
                validate_restart_cpu_outcome(record)
            except BaseException as error:
                failure, failure_traceback = error, error.__traceback__
        if failure is not None:
            record.update(status='failed', error=restart_cpu_error(failure))
        try:
            evidence.emit('failed' if failure is not None else 'complete')
        except BaseException as error:
            secondary('final_publication', error)
            record.update(status='failed', error=restart_cpu_error(failure))
        finally:
            try:
                evidence.close()
            except BaseException as error:
                secondary('evidence_close', error)
        if failure is not None:
            record.update(status='failed', error=restart_cpu_error(failure))
        try:
            evidence.snapshot()
        except BaseException as error:
            secondary('final_snapshot', error)
    if failure is not None:
        raise failure.with_traceback(failure_traceback)
    return record


def main(argv):
    """Dispatch independent native SSSGLU and CPU restart protocols without importing Torch for selection."""
    if argv[0] == 'train':
        train(Path(argv[1]), Path(argv[2]), argv[3], argv[4:])
    elif argv[0] == 'check':
        check(Path(argv[1]))
    elif argv[0] == 'restart-cpu' and len(argv) == 2:
        restart_cpu(Path(argv[1]))
    else:
        raise ValueError(f'unknown operation: {argv[0]}')


if __name__ == '__main__':
    main(sys.argv[1:])
