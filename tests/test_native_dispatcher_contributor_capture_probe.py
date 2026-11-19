"""CPU-only contracts for a deliberately non-launchable contributor observer."""
import ast
import builtins
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'experiments/validation/native_dispatcher_boundary_capture_probe.py'
NEW = ROOT/'experiments/validation/native_dispatcher_contributor_capture_probe.py'


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guard(name, *a, **kw):
        if name.split('.')[0] in {'torch', 'triton', 'megatron', 'uccl', 'deep_ep'}: pytest.fail('vendor import')
        return original(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', guard)
    spec = importlib.util.spec_from_file_location('contributor_fixture', NEW)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_original_native_body_and_math_unchanged(probe):
    assert hashlib.sha256(OLD.read_bytes()).hexdigest() == '43231093c4db6d78f98fb3e7eab4019ae8c389ff121c3d18208727ec879a0129'
    old, new = ast.parse(OLD.read_text()), ast.parse(NEW.read_text())
    for node in old.body:
        if isinstance(node, ast.FunctionDef) and node.name not in {'observe', 'boundary_keys', 'verify_boundary_artifacts', 'gpu', 'check'}:
            other = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == node.name)
            assert ast.dump(node) == ast.dump(other), node.name
    assert not probe.IMPLEMENTATION_READY
    assert 'class BoundaryCapture' not in NEW.read_text() and 'BOUNDARY_TOKENS' not in NEW.read_text()
    assert probe.CHANNELS == tuple(range(0, 1792, 28))


@pytest.mark.parametrize('raises', [False, True])
def test_delegate_once_exact_tuple_restore_ownership(probe, raises):
    class Dispatcher:
        def dispatch_preprocess(self): return result
        def token_dispatch(self): return result
        def dispatch_postprocess(self): return (None, SimpleNamespace(detach=lambda: SimpleNamespace(clone=lambda: None)), None)
        def combine_preprocess(self): return result
        def token_combine(self): return result
    result = (object(), None); dispatcher = Dispatcher(); calls = []
    def instance_override(): calls.append('native'); return result
    dispatcher.dispatch_preprocess = instance_override
    removed = []
    model = SimpleNamespace(token_dispatcher=dispatcher, router=SimpleNamespace(register_forward_hook=lambda fn: SimpleNamespace(remove=lambda: removed.append(True))))
    fake = SimpleNamespace(is_grad_enabled=lambda: True)
    capture = probe.ContributorCapture(fake, Path('/unused'), 0)
    capture.method = lambda *a, **kw: calls.append('before' if kw.get('before') else 'after')
    with capture.arm('alltoall', 0):
        with pytest.raises(RuntimeError) if raises else __import__('contextlib').nullcontext():
            with probe.observe(fake, model, 'alltoall', [], [], []):
                assert dispatcher.dispatch_preprocess() is result
                if raises: raise RuntimeError('native primary')
    assert dispatcher.dispatch_preprocess is instance_override
    assert set(vars(dispatcher)) == {'dispatch_preprocess'} and removed == [True]
    assert calls == ['before', 'native', 'after'] and probe.CURRENT_CAPTURE is None


def test_snapshot_stream_ownership_labels_and_rank_cap(probe):
    calls = []
    class Tensor:
        is_cuda, dtype, shape, requires_grad = True, 'bf16', (4096, 1, 1792), True
        def detach(self): calls.append('detach'); return self
        def index_select(self, dimension, index): calls.append(('index_select', dimension, index)); return object()
        def stride(self): return (1792, 1792, 1)
        def storage_offset(self): return 0
        def element_size(self): return 2
    fake = SimpleNamespace(Tensor=Tensor, bfloat16='bf16', is_grad_enabled=lambda: True)
    capture = probe.ContributorCapture(fake, Path('/unused'), 2); capture.index = 'private-index'; capture.replays['flex'] = 1
    capture.bindings['flex'] = {'sha256': 'a'*64}
    with capture.arm('flex', 0): capture.latent('source', Tensor(), 4096)
    assert calls == ['detach', ('index_select', 2, 'private-index')]
    assert capture.owned[0] is not None and capture.entries[0]['channels'] == list(probe.CHANNELS)
    assert capture.entries[0]['layer'] == 4 and capture.entries[0]['kind'] == 'forward'
    with capture.arm('flex', 0): capture.latent('source', Tensor(), 4096)
    assert len(capture.owned) == 1 and 'duplicate' in capture.errors[0]
    capture.bytes = probe.RANK_CAP
    with capture.arm('flex', 0): capture.latent('postcombine', Tensor(), 4096)
    assert 'cap' in capture.errors[-1]
    capture.__exit__(RuntimeError, RuntimeError('primary'), None)
    assert capture.summary['native_exception'] == 'RuntimeError: primary' and not capture.owned


def test_real_cardinality_streamed_metadata_and_prewrite_rejection(probe):
    count = 262144; sink = io.BytesIO()
    result = probe.stream_i32(lambda: iter(range(count)), sink, count, lower=0, upper=count-1, byte_cap=4*count)
    assert result['bytes'] == 1048576 and result['sha256'] == hashlib.sha256(sink.getvalue()).hexdigest()
    assert probe.REAL_TOKENS == 32768 and probe.REAL_ASSIGNMENTS == count
    for words, size, cap in ((lambda: iter([True]), 1, 4), (lambda: iter([2**31]), 1, 4), (lambda: iter([0]), 2, 8), (lambda: iter([0]), 1, 3)):
        rejected = io.BytesIO()
        with pytest.raises(ValueError): probe.stream_i32(words, rejected, size, lower=0, upper=2**32, byte_cap=cap)
        assert not rejected.getvalue()


def test_uninitialized_te_tails_never_semantic(probe):
    row = [object() for _ in range(65)]; row[0] = 7; row[32] = 2; row[-1] = 1
    words = list(probe.active_map_words([row], 32, 8))
    assert words == [7]+[0]*31+[2]+[0]*31+[1]
    assert probe.chunk_rows([2, 0, 1], [2, 0, 1]) == [2, 0, 1]
    with pytest.raises(ValueError): probe.chunk_rows([1], [1])


def test_callbacks_have_no_host_transfer_or_collective(probe):
    tree = ast.parse(NEW.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ContributorCapture')
    forbidden = {'cpu', 'item', 'tolist', 'nonzero', 'synchronize', 'all_gather', 'all_reduce', 'barrier', 'open', 'write'}
    for name in ('snapshot', 'latent', 'router', 'method', 'attach', 'attach_checked', 'host_splits', 'expert_tuple'):
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert not {n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)} & forbidden


def fake_contract(probe):
    class Tensor:
        def __init__(self, shape, dtype='bf16', cuda=True): self.shape, self.dtype, self.is_cuda = shape, dtype, cuda
        def detach(self): return self
        def clone(self): return Tensor(self.shape, self.dtype, self.is_cuda)
        def index_select(self, dim, index): return self.clone()
        def stride(self): return (1,)*len(self.shape)
        def storage_offset(self): return 0
        def element_size(self): return 2 if self.dtype == 'bf16' else 8 if self.dtype == 'i64' else 4
        @property
        def T(self): return Tensor(tuple(reversed(self.shape)), self.dtype, self.is_cuda)
        def ravel(self): return Tensor((__import__('math').prod(self.shape),), self.dtype, self.is_cuda)
    class Module:
        def __init__(self): self.before, self.after = [], []
        def register_forward_pre_hook(self, hook): self.before.append(hook); return SimpleNamespace(remove=lambda: None)
        def register_forward_hook(self, hook): self.after.append(hook); return SimpleNamespace(remove=lambda: None)
    fake = SimpleNamespace(Tensor=Tensor, bfloat16='bf16', float32='f32', int64='i64', int32='i32', bool='bool',
        is_grad_enabled=lambda: True, tensor=lambda *a, **k: object(),
        distributed=SimpleNamespace(get_process_group_ranks=lambda group: list(range(8))))
    frozen = dict(validation_sha256='a'*64, source_manifest_sha256='b'*64, tooling={'runtime': 'c'*64}, source_files={'moe.py': 'd'*64})
    capture = probe.ContributorCapture(fake, Path('/unused'), 0, frozen=frozen)
    model = SimpleNamespace(token_dispatcher=SimpleNamespace(ep_group=object(), tp_ep_group=object()),
        num_local_experts=32, local_expert_indices=list(range(32)), fc1_latent_proj=Module(), fc2_latent_proj=Module(), experts=Module())
    capture.attach({'flex': model})
    return Tensor, capture, model


@pytest.mark.parametrize('bad', [None, 'width', 'probs', 'counts', 'tuple', 'output'])
def test_public_expert_hook_contracts(probe, bad):
    tensor, capture, model = fake_contract(probe)
    inputs = (tensor((17,1792)), tensor((32,), 'i64', False), tensor((17,), 'f32'))
    if bad == 'width': inputs = (tensor((17,1920)), *inputs[1:])
    elif bad == 'probs': inputs = (*inputs[:2], tensor((16,), 'f32'))
    elif bad == 'counts': inputs = (inputs[0], tensor((32,), 'f32', False), inputs[2])
    elif bad == 'tuple': inputs = inputs[:2]
    with capture.arm('flex', 0):
        capture.replays['flex'] = 1
        assert model.experts.before[0](model.experts, inputs) is None
        if bad in (None, 'output'):
            result = (tensor((18 if bad == 'output' else 17,1792)), None)
            assert model.experts.after[0](model.experts, inputs, result) is None
    assert bool(capture.errors) is (bad is not None)
    if bad is None:
        assert {e['boundary'] for e in capture.entries} == {'expert_input','expert_probs','expert_counts','expert_output'}
        assert next(e for e in capture.entries if e['boundary']=='expert_counts')['storage'] == 'cpu-native-metadata'


def test_native_method_throw_preserves_primary_and_ownership(probe):
    tensor, capture, model = fake_contract(probe); calls=[]
    primary = RuntimeError('native method primary')
    class Dispatcher:
        def dispatch_preprocess(self): calls.append('delegate'); raise primary
        def token_dispatch(self): pass
        def dispatch_postprocess(self): pass
        def combine_preprocess(self): pass
        def token_combine(self): pass
    dispatcher = Dispatcher(); model.token_dispatcher = dispatcher
    model.router = SimpleNamespace(register_forward_hook=lambda hook: SimpleNamespace(remove=lambda: None))
    with pytest.raises(RuntimeError) as caught:
        with capture, capture.arm('flex', 0), probe.observe(capture.torch, model, 'flex', [], [], []):
            capture.replays['flex'] = 1
            dispatcher.dispatch_preprocess()
    assert caught.value is primary and calls == ['delegate'] and not vars(dispatcher)
    assert capture.summary['native_exception'] == 'RuntimeError: native method primary'


def test_exact_metadata_lifetimes_and_missing_map(probe):
    tensor, capture, model = fake_contract(probe)
    capture.bind_models({'alltoall': model})
    dispatcher = model.token_dispatcher
    dispatcher.input_splits = [4096]*8; dispatcher.output_splits = [1]*8
    dispatcher.num_global_tokens_per_local_expert = tensor((8,32), 'i64')
    dispatcher.sort_input_by_local_experts = tensor((256,), 'i64', False)
    dispatcher.restore_output_by_local_experts = tensor((256,), 'i64', False)
    with capture.arm('alltoall', 0):
        capture.replays['alltoall'] = 1
        capture.method(dispatcher, 'alltoall', 'token_dispatch', (), (tensor((17,1792)),None,tensor((17,),'f32')))
        dispatcher.input_splits[0] = 999
        capture.method(dispatcher, 'alltoall', 'dispatch_postprocess', (), None, before=True)
        capture.method(dispatcher, 'alltoall', 'combine_preprocess', (), None, before=True)
    assert capture.owned[0][0] == 4096
    assert {e['boundary'] for e in capture.entries} == {'input_splits','output_splits','forward_chunk_sizes','forward_chunk_order','restore_chunk_sizes','restore_chunk_order'}
    assert not capture.errors
    with capture.arm('flex', 0):
        capture.replays['flex'] = 1; capture.shapes[('flex','received')] = 8
        dispatcher._comm_manager = SimpleNamespace()
        capture.method(dispatcher, 'flex', 'dispatch_postprocess', (), (tensor((17,1792)), tensor((32,), 'i64'), tensor((17,), 'f32')))
    assert 'reversed_mapping_for_combine' in capture.errors[-1]


def test_router_repeat_received_bound_and_flex_topk_order(probe):
    tensor, capture, model = fake_contract(probe)
    manager = SimpleNamespace(token_indices=tensor((4096,8),'i64'), token_probs=tensor((4096,8),'f32'),
        dispatched_indices=tensor((8,8),'i64'), dispatched_probs=tensor((8,8),'f32'), reversed_mapping_for_combine=tensor((8,65),'i32'))
    model.token_dispatcher._comm_manager = manager
    with capture.arm('flex', 0):
        capture.router((tensor((4096,256),'f32'),tensor((4096,256),'bool')))
        capture.method(model.token_dispatcher,'flex','token_dispatch',(),(tensor((8,1792)),None,manager.dispatched_probs))
        capture.method(model.token_dispatcher,'flex','dispatch_postprocess',(),(tensor((17,1792)),tensor((32,),'i64'),tensor((17,),'f32')))
        assert not capture.errors
        assert ('flex','flex_te_map') in capture.seen
        capture.router((tensor((4096,256),'f32'),tensor((4096,256),'bool')))
        assert 'duplicate' in capture.errors[-1]
    tensor, capture, model = fake_contract(probe)
    model.token_dispatcher._comm_manager = manager
    with capture.arm('flex',0):
        capture.replays['flex'] = 1
        capture.method(model.token_dispatcher,'flex','token_dispatch',(),(tensor((32769,1792)),None,manager.dispatched_probs))
    assert capture.errors and not capture.owned


def test_changed_metadata_replay_is_never_accepted(probe):
    calls=[]
    def words(): calls.append(1); return iter([0,1] if len(calls)==1 else [0,2])
    output=io.BytesIO()
    with pytest.raises(ValueError,match='changed metadata'): probe.stream_i32(words,output,2,lower=0,upper=2,byte_cap=8)
    assert not output.getvalue()


def test_setup_errors_secondary_and_binding_group_order(probe):
    tensor, capture, model = fake_contract(probe)
    assert capture.bindings['flex']['metadata']['group'] == tuple(range(8))
    assert capture.bindings['flex']['metadata']['source_columns'] == list(range(256))
    assert capture.bindings['flex']['metadata']['owners'][7] == list(range(224,256))
    capture.torch.distributed.get_process_group_ranks = lambda group: list(reversed(range(8)))
    capture.attach({'flex':model})
    assert 'group order' in capture.errors[-1]
    capture.frozen = None
    capture.attach({'flex':model})
    assert 'frozen input' in capture.errors[-1]
