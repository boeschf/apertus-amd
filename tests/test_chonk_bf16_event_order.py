"""CPU checks for the BF16 producer-gate probe and its guarded launch."""

from contextlib import contextmanager
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from apertus_amd import runs
from apertus_amd.cli import plan


ROOT = Path(__file__).resolve().parents[1]
PROBE = runpy.run_path(ROOT / "experiments/validation/chonk_bf16_event_order.py")


def test_exact_one_gpu_plan_stays_guarded():
    settings = plan(ROOT / "experiments/chonk-bf16-event-order-i910bfcf936d3", site=ROOT / "config.toml",
                    submit=True, reservation="cscs", gpus_per_node=1, cpus_per_task=48, cpu_bind="cores",
                    time="00:10:00", exclude="nid002450", cache_image=True)
    env = dict(SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="1", SLURM_NODEID="0")
    PROBE["verify_launch"](settings, env)
    assert ":/apertus/cache" not in runs.edf(settings, Path("/run"))
    assert "source" not in settings
    with pytest.raises(ValueError, match="unexpected BF16 event-order launch"):
        PROBE["verify_launch"](settings, {**env, "SLURM_NTASKS": "4"})
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(settings)


@pytest.mark.parametrize("arm", PROBE["ARMS"])
@pytest.mark.parametrize("defect", [None, "producer-not-held", "consumer-query-error", "wrong-values"])
def test_arms_release_before_read_or_propagate_failure(monkeypatch, arm, defect):
    execute = PROBE["run_arm"]
    g = execute.__globals__
    tick = iter(range(0, 100))
    monkeypatch.setitem(g, "monotonic", lambda: next(tick) * 0.5)
    monkeypatch.setitem(g, "sleep", lambda _: None)
    signal = 0
    operations = []
    held = False
    current = None
    producer, consumer, release = (SimpleNamespace(tail=lambda: True) for _ in range(3))

    def enqueue(stream, predicate):
        prior = stream.tail
        stream.tail = lambda: prior() and predicate()

    @contextmanager
    def stream(where):
        nonlocal current
        before, current = current, where
        try:
            yield
        finally:
            current = before

    class Event:
        def record(self, where):
            self.where, self.fence = where, where.tail

        def query(self):
            if defect == "consumer-query-error" and held and not signal and self.where is consumer:
                raise RuntimeError("query failed")
            return self.fence()

    class Overlap:
        def __init__(self, fence):
            self.fence = fence

        def current_stream_wait(self):
            enqueue(current, self.fence)

    class Tensor:
        dtype = "bf16"
        shape = PROBE["SHAPE"]

        def __init__(self, contiguous):
            self.dense = contiguous

        def is_contiguous(self):
            return self.dense

        def data_ptr(self):
            return id(self)

        def contiguous(self):
            return self if self.dense else Tensor(True)

        def cpu(self):
            assert signal == 1 and producer.tail() and consumer.tail()
            operations.append("read")
            return self

    def hold(where):
        nonlocal held
        assert where is producer
        held = True
        enqueue(where, lambda: signal == 1 or defect == "producer-not-held")

    def write(where, value):
        nonlocal signal
        assert where is release
        operations.append(f"write:{value}")
        signal = value

    torch = SimpleNamespace(cuda=SimpleNamespace(stream=stream, Event=lambda **_: Event()),
                            bfloat16="bf16", equal=lambda *_: defect != "wrong-values")
    utils = SimpleNamespace(EventOverlap=Overlap, EventHandle=lambda: current.tail)
    record = {}

    def run():
        execute(torch, utils, SimpleNamespace(hold=hold, write=write), (producer, consumer, release), arm,
                Tensor(arm == "contiguous-control"), object(), record)

    if defect:
        with pytest.raises((RuntimeError, ValueError)):
            run()
    else:
        run()
        assert record["samples"][-1] == dict(consumer_done=arm != "after-copy", producer_done=False,
                                            consumer_prefix=True)
        assert record["copy"] is (arm != "contiguous-control") and record["exact_bf16"]
    assert record["drained"]
    assert operations[:2] == ["write:0", "write:1"]
    assert ("read" in operations) == (defect in (None, "wrong-values"))


def test_drain_polls_all_markers_until_each_completes(monkeypatch):
    await_events = PROBE["await_events"]
    monkeypatch.setitem(await_events.__globals__, "sleep", lambda _: None)
    first, second = iter([False, True, True]), iter([False, True])
    await_events((SimpleNamespace(query=lambda values=values: next(values))
                  for values in (first, second)), 1, "drain")
    assert list(first) == list(second) == []


def test_ep16_native_plan_and_rank_placement():
    settings = plan(ROOT / "experiments/chonk-bf16-event-order-ep16-i910bfcf936d3", site=ROOT / "config.toml",
                    submit=True, reservation="cscs", gpus_per_node=4, cpus_per_task=48,
                    cpu_bind="map_ldom:0,1,2,3", time="00:20:00", exclude="nid002450", cache_image=True)
    verify = PROBE["verify_ep16_launch"]
    for rank in range(16):
        env = {**settings["environment"], "SLURM_PROCID": str(rank), "SLURM_LOCALID": str(rank % 4),
               "SLURM_NTASKS": "16", "SLURM_NODEID": str(rank // 4)}
        assert verify(settings, env) == (rank, rank % 4)
    for bad in ({"SLURM_LOCALID": "1"}, {"UCCL_EP_TRANSPORT": "ib"}, {"UCCL_EP_DISPATCH_CONFIG": ""}):
        with pytest.raises(ValueError, match="unexpected native EP16"):
            verify(settings, {**env, **bad})
    assert "--ntasks=16" in settings["steps"][0]
    assert ":/apertus/cache" not in runs.edf(settings, Path("/run"))
    assert "source" not in settings
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(settings)


@pytest.mark.parametrize("changed", [None, "prior_comm", "producer_done", "after_comm", "consumer_done", "empty"])
def test_ep16_held_marker_gate(changed):
    verify = PROBE["verify_cached_held"]
    expected = dict(prior_comm=True, producer_done=False, after_comm=False, consumer_done=False)
    if changed is None:
        verify([expected, expected])
    else:
        rows = [] if changed == "empty" else [expected, {**expected, changed: not expected[changed]}]
        with pytest.raises(ValueError, match="did not remain gated"):
            verify(rows)


@pytest.mark.parametrize("defect", [None, "metadata", "no-event"])
def test_cached_dispatch_preserves_native_options_and_handle(defect):
    source, handle, ready, output = (object() for _ in range(4))
    done = SimpleNamespace(event=None if defect == "no-event" else object())

    def dispatch(x, **kwargs):
        assert x is source
        assert kwargs == dict(handle=handle, previous_event=ready, async_finish=True, allocate_on_comm_stream=True)
        return output, None, None, None, object() if defect == "metadata" else None, done

    submit = PROBE["cached_dispatch"]
    if defect:
        with pytest.raises(ValueError, match="unexpected metadata"):
            submit(SimpleNamespace(dispatch=dispatch), source, handle, ready)
    else:
        assert submit(SimpleNamespace(dispatch=dispatch), source, handle, ready) == (output, done)


def test_ep16_routing_fixture_balanced_and_two_epochs_distinct():
    ep, tokens, experts, topk = (PROBE[k] for k in ("EP", "TOKENS", "LOCAL_EXPERTS", "TOPK"))
    route = PROBE["route_index"]
    epochs = []
    for epoch in (0, 1):
        inbound = [set() for _ in range(ep)]
        counts = [0] * (ep * experts)
        for key in range(ep * tokens):
            ids = [route(key, s, epoch) for s in range(topk)]
            dests = {i // experts for i in ids}
            assert len(dests) == topk
            assert sum(d + 1 + epoch * 16 for d in dests) <= 256  # exact BF16 integer sum
            for i in ids:
                counts[i] += 1
                inbound[i // experts].add(key)
        assert all(v == tokens * topk for v in map(len, inbound))
        assert all(v == ep * tokens * topk // (ep * experts) for v in counts)
        epochs.append(inbound)
    assert all(a != b for a, b in zip(*epochs))
