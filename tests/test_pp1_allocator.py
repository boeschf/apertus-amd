"""CPU checks for the matched stage-1 allocator diagnostic."""

from pathlib import Path
import json
import runpy
import sys
from types import SimpleNamespace
from weakref import ref

import pytest

from apertus_amd.cli import plan
from apertus_amd.runs import run


ROOT = Path(__file__).resolve().parents[1]
V = runpy.run_path(ROOT / "experiments/validation/chonk_fnuz_oom.py")
TRACE = V["PP1AllocatorTrace"]
FLAG = V["PP1_ALLOCATOR_TRACE"]


def frame(path, name):
    return dict(filename="/frozen/" + path, line=100, name=name)


def fixture():
    return dict(segments=[
        dict(device=0, stream=7, total_size=112, blocks=[
            dict(state="active_allocated", address=100, size=64, requested_size=61,
                 frames=[frame("megatron/core/ssm/kimi_delta_attention.py", "_kda_core")]),
            dict(state="active_allocated", address=164, size=16, requested_size=15),
            dict(state="active_awaiting_free", address=180, size=24, requested_size=20,
                 frames=[frame("megatron/core/transformer/moe/resident_fp8.py", "_backward")]),
            dict(state="inactive", address=204, size=8, requested_size=0),
        ]),
        dict(device=1, stream=10, total_size=64, blocks=[
            dict(state="inactive", address=300, size=64, requested_size=0),
        ]),
    ], device_traces=[[
        dict(action="free_requested", addr=180, size=20, stream=9,
             frames=[frame("megatron/core/pipeline_parallel/schedules.py", "deallocate_output_tensor")])
    ], []])


@pytest.mark.parametrize("pending", ["active_pending_free", "active_awaiting_free"])
def test_snapshot_reconciles_states_and_matches_live_cache_and_free_request(pending):
    snapshot = fixture()
    snapshot["segments"][0]["blocks"][2]["state"] = pending
    sample = V["summarize_allocator"](snapshot, 0, 2, {164: 15, 300: 2})
    assert sample["block_bytes_by_state"] == {
        "active_allocated": 80, "active_awaiting_free": 24, "inactive": 8}
    assert sample["block_bytes_by_raw_state"][pending] == 24
    assert sample["origin_bytes_by_state"]["active_allocated"] == {"kda": 64, "resident_weight_cache": 16}
    assert sample["resident_storage_bytes"] == 17
    assert sample["resident_storage_bytes_matched"] == 15
    assert sample["resident_matched_block_bytes"] == 16
    assert sample["pending_bytes_with_matching_free_request"] == 24
    assert sample["pending_examples"][0]["allocation_stream"] == 7
    assert sample["pending_examples"][0]["free_request_stream"] == 9
    assert not sample["history_at_capacity"]


def test_address_reuse_and_truncated_history_are_not_mistaken_for_pending_free():
    snapshot = fixture()
    snapshot["device_traces"][0].append(dict(action="alloc", addr=180, size=24, stream=7))
    sample = V["summarize_allocator"](snapshot, 0, 2)
    assert sample["history_at_capacity"]
    assert sample["pending_bytes_with_matching_free_request"] == 0
    assert "free_request_stream" not in sample["pending_examples"][0]
    with pytest.raises(ValueError, match="capacity"):
        V["summarize_allocator"](snapshot, 0, 1)
    snapshot["segments"][0]["total_size"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        V["summarize_allocator"](snapshot, 0, 2)


def test_unknown_state_is_reported_not_silently_counted_as_pending():
    snapshot = fixture()
    snapshot["segments"][0]["blocks"][2]["state"] = "unexpected"
    with pytest.raises(ValueError, match="unknown allocator block state: 'unexpected'"):
        V["summarize_allocator"](snapshot, 0, 2)


def test_owner_inventory_deduplicates_storage_shared_by_te_views():
    storage = SimpleNamespace(data_ptr=lambda: 164, nbytes=lambda: 15)
    a = SimpleNamespace(untyped_storage=lambda: storage)
    weight = SimpleNamespace(_rowwise_data=a, _columnwise_data=a)
    module = SimpleNamespace(_resident_fp8=SimpleNamespace(_weights={1: (weight, weight)}))
    chunk = SimpleNamespace(modules=lambda: iter((object(), module)))
    assert V["resident_weight_storages"]([chunk, chunk]) == {164: 15}


def test_observer_uses_only_scoped_cpu_metadata_and_turns_history_off():
    settings, trace = [], []
    stats = dict(zip(V["DrainOwnership"].FIELDS, (80, 104, 112, 0, 0, 0), strict=True))
    snapshot = fixture()
    snapshot["segments"][0]["blocks"][2]["state"] = "active_pending_free"
    mem = SimpleNamespace(_record_memory_history=lambda **kw: settings.append(kw),
                          _snapshot=lambda: snapshot)
    cuda = SimpleNamespace(memory=mem, current_device=lambda: 0, memory_stats=lambda: stats)
    observer = TRACE(cuda, lambda kind, **kw: trace.append(dict(event=kind, **kw)))

    def get_chunk(virtual_microbatch_id, forward):
        chunk = virtual_microbatch_id // 4 % 2
        return chunk if forward else 1 - chunk

    for call in range(1, 266):
        progress = dict(call=call, forward=call - 1 if call < 10 else None,
                        backward=call - 10 if call >= 10 else None)
        for phase in ("enter", "return"):
            observer.observe(phase, progress, get_chunk, lambda: {164: 15})
    assert [v["enabled"] for v in settings] == ["state", None, "all", None]
    assert all(v["max_entries"] == TRACE.LIMIT and v["clear_history"]
               for v in settings if v["enabled"])
    assert len(trace) == 15
    assert [(r["phase"], r["chunk"]) for r in trace if r["call"] == 259] == [
        ("enter", 1), ("return", 1)
    ]
    assert all(r["block_bytes_by_raw_state"]["active_pending_free"] == 24 for r in trace)
    V["verify_pp1_allocator"](trace, 16)
    with pytest.raises(ValueError, match="unexpected"):
        V["verify_pp1_allocator"](trace, 17)
    with pytest.raises(ValueError, match="incomplete"):
        V["verify_pp1_allocator"](trace[:-1], 16)
    trace[0]["allocator"]["active_bytes.all.current"] = 106
    with pytest.raises(ValueError, match="metadata and counters differ"):
        V["verify_pp1_allocator"](trace, 16)


def test_private_api_failure_records_partial_receipt_and_disables_history():
    settings, trace = [], []

    def fail():
        raise RuntimeError("snapshot unsupported")

    mem = SimpleNamespace(_record_memory_history=lambda **kw: settings.append(kw), _snapshot=fail)
    observer = TRACE(SimpleNamespace(memory=mem, current_device=lambda: 0),
                     lambda kind, **kw: trace.append(dict(event=kind, **kw)))
    for _ in range(2):
        observer.observe("enter", dict(call=1, forward=0, backward=None),
                         lambda virtual_microbatch_id, forward: 0, dict)
    assert trace == [dict(event="pp1_allocator_error", phase="enter", call=1, error="snapshot unsupported")]
    assert [v["enabled"] for v in settings] == ["state", None]


def test_ep_labels_distinguish_origin_and_consumer_and_reject_address_reuse():
    snapshot = fixture()
    events = snapshot["device_traces"][0]
    events[:0] = [
        dict(action="alloc", addr=180, size=20, stream=7,
             user_metadata=V["pp1_ep_metadata"](258, "dispatch", "scope", comm_stream=7)),
        dict(action="annotate", addr=180,
             user_metadata=V["pp1_ep_metadata"](258, "combine", "in.tokens", storage_bytes=20)),
    ]
    sample = V["summarize_allocator"](snapshot, 0, 16, lifetime=True)
    assert sample["pending_ep_label_bytes"] == [["resident_experts", "scope", "dispatch", "scope", 24]]
    assert sample["pending_examples"][0]["ep_labels"]["input"]["operation"] == "combine"
    assert sample["pending_examples"][0]["allocation_frames"][0][1:] == [100, "_backward"]
    assert sample["pending_examples"][0]["free_request_frames"][0][2] == "deallocate_output_tensor"
    events.extend([dict(action="free_completed", addr=180, size=24),
                   dict(action="alloc", addr=180, size=20, stream=7),
                   dict(action="free_requested", addr=180, size=20, stream=9)])
    sample = V["summarize_allocator"](snapshot, 0, 16, lifetime=True)
    assert sample["pending_ep_label_bytes"] == []
    assert sample["recent_ep_free_completed_bytes"] == [["scope", "dispatch", "scope", 24]]


def test_ep_hook_keeps_original_result_and_restores_metadata():
    tags = []
    storage = SimpleNamespace(data_ptr=lambda: 42, nbytes=lambda: 20)
    tensor = SimpleNamespace(is_cuda=True, untyped_storage=lambda: storage)
    memory = SimpleNamespace(tag="initial")
    memory._get_memory_metadata = lambda: memory.tag
    memory._set_memory_metadata = lambda tag: setattr(memory, "tag", tag)
    memory._annotate_tensor = lambda tensor, tag: tags.append(tag)
    cuda = SimpleNamespace(memory=memory, current_stream=lambda: SimpleNamespace(cuda_stream=1))
    buffer = SimpleNamespace(get_comm_stream=lambda: SimpleNamespace(cuda_stream=2))
    original_result = ((tensor, tensor), None, tensor, [], (tensor,), object())

    def original(buffer, x, handle=None):
        assert '"role":"scope"' in memory.tag
        assert handle == (tensor,)
        return original_result

    observer = TRACE(cuda, lambda *a, **kw: pytest.fail("unexpected observer error"), lifetime=True)
    observer.window = "actions"
    wrapped = observer.wrap_ep(original, "dispatch", lambda: {"call": 259})
    assert wrapped(buffer, tensor, (tensor,)) is original_result
    assert memory.tag == "initial"
    assert [tag.split('"role":"')[1].split('"')[0] for tag in tags] == [
        "in.tokens", "in.handle.0", "out.tokens.0", "out.tokens.1", "out.topk_weights", "out.handle.0"]


def test_lifetime_keeps_state_history_through_fill_and_records_actions_before_drain():
    settings, trace = [], []
    stats = dict(zip(V["DrainOwnership"].FIELDS, (80, 104, 112, 0, 0, 0), strict=True))
    cuda = SimpleNamespace(memory=SimpleNamespace(_record_memory_history=lambda **kw: settings.append(kw),
                                                  _snapshot=fixture),
                          current_device=lambda: 0, memory_stats=lambda: stats)
    observer = TRACE(cuda, lambda kind, **kw: trace.append(dict(event=kind, **kw)), lifetime=True)
    for call in range(1, 266):
        progress = dict(call=call, forward=call - 1 if call < 257 else None,
                        backward=call - 10 if call >= 257 else None)
        for phase in ("enter", "return"):
            observer.observe(phase, progress, lambda virtual_microbatch_id, forward: int(not forward),
                             lambda: {164: 15})
        if call == 248:
            observer.ep_counts.update(dispatch=1, combine=1)
    assert [v["enabled"] for v in settings] == ["state", "all", None]
    assert [r["call"] for r in trace if r["event"] == "pp1_lifetime_actions_begin"] == [248]
    assert any(r["call"] == 259 and r["phase"] == "return" and r["chunk"] == 1 for r in trace)
    V["verify_pp1_allocator"](trace, 16, lifetime=True)


def test_matched_recipe_and_guard():
    cfg = dict(site=ROOT / "config.toml", submit=True, gpus_per_node=4, cpus_per_task=48,
               cpu_bind="map_ldom:0,1,2,3", time="01:00:00", cache_mode="cold",
               exclude="nid002450", cache_image=True, reservation="cscs")
    a, b = (plan(ROOT / "experiments" / f"chonk-native-fnuz-{x}-i910bfcf936d3", **cfg)
            for x in ("kda-release", "pp1-allocator"))
    assert {k for k in a if a[k] != b[k]} == {"experiment", "environment", "procedure"}
    assert {k: v for k, v in a["procedure"].items() if k != "launch_blocked"} == {
        k: v for k, v in b["procedure"].items() if k != "launch_blocked"}
    lifetime = V["PP1_LIFETIME_TRACE"]
    completion = V["PP1_COMPLETION_TRACE"]
    assert b["environment"] == {**a["environment"], FLAG: "1", lifetime: "1", completion: "1"}
    assert V["pp1_allocator_requested"](b["environment"], True, True)
    assert not V["pp1_allocator_requested"]({}, False, False)
    with pytest.raises(ValueError, match="requires the signed KDA"):
        V["pp1_allocator_requested"](b["environment"], True, False)
    assert V["pp1_lifetime_requested"](b["environment"], True)
    with pytest.raises(ValueError, match="requires the stage-1 allocator"):
        V["pp1_lifetime_requested"](b["environment"], False)
    assert V["pp1_completion_requested"](b["environment"], True)
    with pytest.raises(ValueError, match="requires the PP1 lifetime trace"):
        V["pp1_completion_requested"](b["environment"], False)
    with pytest.raises(ValueError, match="launch blocked"):
        run(b)


def completion_fixture():
    trace, operations, tags = [], [], []
    ctx = dict(call=259, forward=None, backward=249)
    caller, comm = SimpleNamespace(cuda_stream=7), SimpleNamespace(cuda_stream=8)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is False
            self.complete = False

        def record(self, stream):
            operations.append(("record", stream.cuda_stream))

        def query(self):
            operations.append(("query",))
            return self.complete

    def tensor(addr):
        return SimpleNamespace(is_cuda=True,
                               untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: addr, nbytes=lambda: 32))

    cuda = SimpleNamespace(Event=Event, current_stream=lambda: caller,
                           memory=SimpleNamespace(_annotate_tensor=lambda t, tag: tags.append(tag)))
    buffer = SimpleNamespace(get_comm_stream=lambda: comm, runtime=SimpleNamespace(get_comm_stream=lambda: 8))
    probe = V["PP1CompletionTrace"](cuda, lambda event, **kw: trace.append(dict(event=event, **kw)),
                                   lambda: ctx, lambda: True)
    return probe, ctx, buffer, caller, tensor, trace, tags, operations


@pytest.mark.parametrize("operation", ["dispatch", "combine"])
def test_completion_pinned_internode_positions_existing_wait_and_no_owned_wrapper(operation):
    probe, ctx, buffer, caller, tensor, trace, tags, ops = completion_fixture()

    class Overlap:
        event = object()

    def original(*args, **kwargs):
        ops.append(("exchange",))
        overlap = Overlap()
        overlap.extra_tensors = (args[1],)
        return ((tensor(42), None, None, None, None, overlap) if operation == "dispatch"
                else (tensor(42), None, overlap))

    call = probe.wrap_exchange(original, operation)
    result = call(buffer, tensor(41), (tensor(99),), *([None] * (10 if operation == "dispatch" else 4)), True)
    assert ops == [("exchange",), ("record", 8)]
    weak = ref(result[-1])
    wait = probe.wrap_wait(lambda overlap: ops.append(("existing_wait",)) or 80)
    assert wait(result[-1]) == 80
    assert ops[-2:] == [("existing_wait",), ("record", 7)]
    del result
    assert weak() is None and not probe.waits
    probe.sample(259, "return")
    probe.sample(260, "enter")
    for event, _ in probe.markers.values():
        event.complete = True
    probe.sample(260, "return")
    assert not probe.markers
    assert [r["state"] for r in trace if r["event"] == "pp1_completion_query"] == [
        "not_ready", "not_ready", "complete", "complete"]
    assert {json.loads(s.split(":", 1)[1])["role"] for s in tags} == (
        {"dispatch.output"} if operation == "dispatch" else {"combine.input", "combine.output"})
    ctx.update(forward=250)
    call(buffer, tensor(41), (tensor(99),), async_finish=True)
    assert ops[-1] == ("exchange",) and not probe.markers


def test_completion_tracks_kda_keyword_and_wgrad_without_holding_released_grads():
    probe, _, _, stream, tensor, _, tags, ops = completion_fixture()
    kda = probe.wrap_consumer(lambda **kw: ops.append(("kda",)) or kw["do"], "kda")
    grad = tensor(41)
    assert kda(q=None, do=grad) is grad
    assert ops == [("kda",), ("record", 7)]

    class Grad:
        is_cuda = True
        untyped_storage = staticmethod(grad.untyped_storage)

    owner = SimpleNamespace(name="mlp", output_grads=(Grad(),), stream=stream)
    weak = ref(owner.output_grads[0])
    probe.outputs.add(41)

    def original(node):
        node.output_grads = None
        assert weak() is None
        ops.append(("wgrad",))

    probe.wrap_consumer(original, "wgrad")(owner)
    assert ops[-2:] == [("wgrad",), ("record", 7)]
    assert [json.loads(s.split(":", 1)[1])["role"] for s in tags] == ["kda.input", "mlp.wgrad"]


def test_signed_first_backward_leaves_late_kda_comparison_hook_installed(monkeypatch):
    probe, ctx, _, _, tensor, trace, tags, ops = completion_fixture()
    calls = []

    def signed_backward(*, q, do):
        calls.append(ctx["call"])
        return q, do

    chunk = SimpleNamespace(chunk_kda_bwd=signed_backward)
    monkeypatch.setitem(sys.modules, V["KDA_MODULE"], chunk)
    wrapper = V["wrap_pp1_kda_backward"](signed_backward, probe, {"commit": "signed-test"})
    monkeypatch.setattr(chunk, "chunk_kda_bwd", wrapper)
    grad = tensor(41)

    ctx.update(call=10, forward=9, backward=0)
    assert chunk.chunk_kda_bwd(q=None, do=grad) == (None, grad)
    assert chunk.chunk_kda_bwd is wrapper and not tags and not ops
    ctx.update(call=259, forward=None, backward=249)
    assert chunk.chunk_kda_bwd(q=None, do=grad) == (None, grad)
    probe.sample(259, "return")
    assert chunk.chunk_kda_bwd is wrapper
    assert calls == [10, 259]
    assert [r["event"] for r in trace] == [
        "kda_release_backward_enter", "kda_release_backward_return", "pp1_completion_kda_input",
        "pp1_completion_mark", "pp1_completion_query"]
    assert trace[2]["tagged"] is True and trace[3]["kind"] == "kda.kda"
    assert json.loads(tags[0].split(":", 1)[1])["role"] == "kda.input"
    assert ops == [("record", 7), ("query",)]
    with pytest.raises(ValueError, match="signed KDA backward"):
        V["wrap_pp1_kda_backward"](lambda: None, probe, {})


def test_late_kda_receipt_distinguishes_missing_input_from_missing_hook(monkeypatch):
    probe, _, _, _, _, trace, tags, ops = completion_fixture()

    def signed_backward(*, do):
        assert do is None
        return "original result"

    monkeypatch.setitem(sys.modules, V["KDA_MODULE"], SimpleNamespace(chunk_kda_bwd=signed_backward))
    hook = V["wrap_pp1_kda_backward"](signed_backward, probe, {})
    assert hook(do=None) == "original result"
    assert trace == [
        dict(event="pp1_completion_kda_input", call=259, ticket=1, tagged=False),
        dict(event="kda_release_backward_enter"),
        dict(event="kda_release_backward_return"),
    ]
    assert not tags and not ops


def test_completion_fails_open_on_missing_event_and_does_not_repeat_the_wait():
    probe, _, buffer, _, tensor, trace, _, ops = completion_fixture()
    result = (tensor(42), None, SimpleNamespace(event=None))
    wrapped = probe.wrap_exchange(lambda *a, **kw: result, "combine")
    assert wrapped(buffer, tensor(41), handle=(), async_finish=True) is result
    assert probe.failed and not probe.markers and not ops
    assert trace[-1]["boundary"] == "exchange_return"
    waiter = probe.wrap_wait(lambda ev: ops.append(("wait",)))
    waiter(result[-1])
    assert ops == [("wait",)]


def test_completion_query_error_is_not_reported_as_pending_or_complete():
    probe, _, _, stream, _, trace, _, _ = completion_fixture()
    probe.mark(259, 1, "mlp.node", stream)
    event = probe.markers[1, "mlp.node"][0]
    def bad():
        raise RuntimeError("query failure")
    event.query = bad
    probe.sample(259, "return")
    probe.close()
    assert [r["event"] for r in trace] == [
        "pp1_completion_mark", "pp1_completion_error", "pp1_completion_unresolved"]
    assert trace[-1]["markers"] == [[1, "mlp.node"]]


def test_completion_original_wait_failure_is_propagated_without_an_extra_marker():
    probe, _, _, _, _, trace, _, ops = completion_fixture()

    class Overlap:
        pass

    overlap = Overlap()
    probe.waits[id(overlap)] = (ref(overlap), 259, 1)

    def fail(event):
        assert event is overlap
        raise RuntimeError("original wait failure")

    with pytest.raises(RuntimeError, match="original wait failure"):
        probe.wrap_wait(fail)(overlap)
    assert not ops and not probe.markers
    assert [r["event"] for r in trace] == ["pp1_completion_wait_enter"]


def test_completion_generation_join_distinguishes_pending_completed_reuse_and_truncation():
    summarize = V["pp1_completion_lifetimes"]

    def event(action, t, role=None, ticket=0):
        data = dict(action=action, addr=100, size=32, stream=7, time_us=t)
        if role:
            data["user_metadata"] = V["PP1_COMPLETION_LABEL"] + json.dumps(
                dict(call=259, ticket=ticket, role=role, storage_bytes=32))
        return data

    old = [event("alloc", 1), event("annotate", 2, "dispatch.output", 1),
           event("annotate", 3, "mlp.node", 2), event("free_requested", 4), event("free_completed", 5)]
    new = [event("alloc", 6), event("annotate", 7, "kda.input", 3), event("free_requested", 8)]
    blocks = [dict(address=100, requested_size=32, size=48, state="active_pending_free")]
    snapshot = dict(device_traces=[old + new], segments=[dict(device=0, blocks=blocks)])
    ended, pending = summarize(snapshot, 0, {100})
    assert (ended["state"], ended["free_completed"], [u["ticket"] for u in ended["uses"]]) == (
        "free_completed", 5, [1, 2])
    assert (pending["state"], pending["free_requested"], pending["block_bytes"]) == ("awaiting_free", 8, 48)
    assert pending["cohorts"] == ["kda.input"]
    snapshot["device_traces"][0] = new[1:]
    assert summarize(snapshot, 0, {100}) == []
    snapshot["device_traces"][0] = old + [new[0], event("annotate", 7, "mlp.node", 3), new[-1]]
    assert summarize(snapshot, 0, {100}) == [ended]


def test_completion_receipts_allow_unresolved_gpu_work_but_not_missing_wait():
    rows = []
    for ticket, kind in enumerate(("dispatch", "combine"), 1):
        for name in ("exchange_enter", "exchange_return", "wait_enter", "wait_enqueued"):
            rows.append(dict(event="pp1_completion_" + name, ticket=ticket, kind=kind))
        for marker in ("producer_after_native_record", "caller_after_existing_wait"):
            rows.append(dict(event="pp1_completion_mark", ticket=ticket, kind=marker))
    rows += [
        dict(event="pp1_completion_kda_input", ticket=3, tagged=True),
        dict(event="pp1_completion_mark", ticket=3, kind="kda.kda"),
        dict(event="pp1_completion_query", ticket=3, kind="kda.kda", state="not_ready"),
        *(dict(event="pp1_completion_lifetimes", call=call, phase=phase, lifetimes=[
            dict(cohorts=["dispatch.output", "combine.output", "te_backward_unpermute", "kda.input"])])
          for call, phase in TRACE.DRAIN),
    ]
    V["verify_pp1_completion"](rows, 16)
    with pytest.raises(ValueError, match="inconsistent"):
        V["verify_pp1_completion"]([r for r in rows if r["event"] != "pp1_completion_wait_enqueued"], 16)
    with pytest.raises(ValueError, match="inconsistent"):
        V["verify_pp1_completion"]([r for r in rows if r["event"] != "pp1_completion_kda_input"], 16)
