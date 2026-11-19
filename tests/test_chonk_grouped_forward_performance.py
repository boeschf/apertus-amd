"""Vendor-free contracts for the actual grouped timing phase/sample state machine."""
import ast
import builtins
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/validation/chonk_grouped_forward_performance.py"
ARTIFACTS = ROOT / "artifacts/investigations/performance-next/restart-20261116"


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guard(name, *a, **k):
        if name.split(".")[0] in {"torch", "triton", "transformer_engine", "aiter", "megatron"}:
            pytest.fail("host vendor import")
        return original(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guard)
    spec = importlib.util.spec_from_file_location("grouped_perf_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_all_installed_pins_and_qualified_operand_identity(probe):
    source = ROOT / "experiments/validation/chonk_grouped_forward_probe.py"
    spec = importlib.util.spec_from_file_location("grouped_accuracy_fixture", source)
    accuracy = importlib.util.module_from_spec(spec); spec.loader.exec_module(accuracy)
    assert {k: probe.PINS[k] for k in accuracy.PINS} == accuracy.PINS
    for name, filename in (("torch/_dynamo/utils.py", "output-gate-counter-source-review"),
                           ("torch/_inductor/metrics.py", "output-gate-counter-source-review"),
                           ("triton/runtime/jit.py", "output-gate-counter-source-review"),
                           ("triton/runtime/autotuner.py", "grouped-forward-cache-abi-review")):
        data = (ARTIFACTS / filename / name).read_bytes()
        assert probe.PINS[name] == (len(data), hashlib.sha256(data).hexdigest())
    run = ROOT / "artifacts/experiment-data/runs/chonk-grouped-forward-selector-i910bfcf936d3/20261116T143844Z-9d9ee6ae"
    row = json.loads((run / "output/0/grouped-forward-0/0/worker.json").read_text())
    assert probe.object_digest(row["operands"]) == probe.OPERANDS_SHA


def test_exact_reference_gate_and_no_timed_observers(probe):
    old = ast.parse((ROOT / "experiments/validation/chonk_grouped_forward_probe.py").read_text())
    new = ast.parse(SCRIPT.read_text())
    def function(tree, name):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        node.body = node.body[1:]  # Contract docstrings differ, arithmetic does not.
        return ast.dump(node)
    assert function(old, "metrics_pass") == function(new, "metrics_pass")
    calls = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "calls")
    assert [n.func.id for n in ast.walk(calls) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)] == ["range", "operation"]
    assert probe.IMPLEMENTATION_READY is True


def fake_torch():
    events = []
    class Event:
        def __init__(self, **kwargs): self.records = 0; events.append(self)
        def record(self): self.records += 1
        def synchronize(self): pass
        def elapsed_time(self, other): return .16
    cuda = SimpleNamespace(Event=Event, synchronize=lambda *a: None, reset_peak_memory_stats=lambda *a: None,
                           max_memory_allocated=lambda *a: 100, max_memory_reserved=lambda *a: 200)
    return SimpleNamespace(cuda=cuda), events


@pytest.mark.parametrize("rank", range(4))
def test_actual_phase_warm_sample_orchestration(probe, tmp_path, monkeypatch, rank):
    torch, events = fake_torch(); calls = Counter(); phases = []
    operations = {case: lambda case=case: calls.update([case]) for case in probe.CASES}
    snap = dict(dynamo={k: {} for k in ("frames", "stats", "inductor", "graph_break")}, generated_kernel_count=0,
                gmm_keys={str(rank): ["a"*64]})
    record = dict(rank=rank, local_device=rank, selector="1", samples=[])
    def phase(name): phases.append((name, dict(calls)))
    with (tmp_path / "native.log").open("w") as log:
        probe.measure_protocol(torch, operations, phase, lambda: deepcopy(snap), record, log.fileno())
    assert [p[0] for p in phases] == ["pre", "post"]
    assert phases[0][1] == {} and calls == Counter({case: 20+12*16 for case in probe.CASES})
    assert len(record["samples"]) == 48 and len(events) == 8
    assert all(event.records == 13 for event in events)  # Initialization plus12batches.
    assert [(r["block"], r["case"]) for r in record["samples"]] == [(b, c) for b in range(12) for c in probe.rotate(rank, b)]
    assert all(row["log_before"] == row["log_after"] == 0 for row in record["samples"])
    assert record["warm_calls_per_case"] == [10, 10]
    assert set(record["phase_memory"]) == {"pre", "warm", "measure", "post"}
    assert all(record["cache"][k] == snap for k in record["cache"])


def test_cache_instability_stops_before_timing_without_adaptive_warm(probe, tmp_path):
    torch, _ = fake_torch(); counts = Counter(); observed = []
    operations = {case: lambda case=case: counts.update([case]) for case in probe.CASES}
    def snapshot():
        observed.append(True)
        return dict(dynamo={}, generated_kernel_count=len(observed), gmm_keys={"0": ["a"*64]})
    record = dict(rank=0, local_device=0, selector="1", samples=[])
    with (tmp_path / "native.log").open("w") as log:
        with pytest.raises(ValueError, match="warm compiler/cache"):
            probe.measure_protocol(torch, operations, lambda _: None, snapshot, record, log.fileno())
    assert counts == Counter({case: 20 for case in probe.CASES}) and not record["samples"]


def test_sample_exact_endpoints_and_release_count(probe, tmp_path):
    torch, _ = fake_torch(); calls = []; ticks = iter((1000, 17000, 33000))
    with (tmp_path / "native.log").open("w") as log:
        row = probe.sample(torch, lambda: calls.append(True), (torch.cuda.Event(), torch.cuda.Event()), 0, log.fileno(), lambda: next(ticks))
    assert len(calls) == 16
    assert row["host_return_us"] == 1 and row["completion_us"] == 2 and row["device_span_us"] == 10


def test_resolver_admits_actual_ten_minute_contract(probe, monkeypatch):
    from apertus_amd.config import resolve
    settings = resolve(ROOT / "experiments/chonk-grouped-forward-performance-i910bfcf936d3", site=ROOT/"config.toml",
                       submit=True, reservation="cscs", time="00:10:00")
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _: set(range(48)))
    for rank in range(4):
        env = dict(SLURM_PROCID=str(rank), SLURM_LOCALID=str(rank), SLURM_NTASKS="4", SLURM_NODEID="0")
        assert probe.validate_launch(settings, env) == (rank, rank)
    settings["procedure"]["timeout"] = 511
    with pytest.raises(ValueError): probe.validate_launch(settings, env)
