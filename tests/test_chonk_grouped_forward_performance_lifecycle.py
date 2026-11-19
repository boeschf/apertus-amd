"""CPU child/timeout tests; no processes, scheduler, vendor modules or GPU access."""
from copy import deepcopy
import builtins
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/investigations/performance-next/restart-20261116"
SCRIPT = ROOT / "experiments/validation/chonk_grouped_forward_performance.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from apertus_amd.config import resolve
    p = load("grouped_lifecycle_probe", SCRIPT)
    fixtures = load("grouped_closed_record_fixtures", ARTIFACTS / "grouped-forward-performance-record-validation/test_validator.py")
    settings = resolve(ROOT / "experiments/chonk-grouped-forward-performance-i910bfcf936d3", site=ROOT / "config.toml",
                       submit=True, reservation="cscs", time="00:10:00")
    settings["scratch_root"] = str(tmp_path / "scratch")
    settings["validation"]["sha256"] = p.digest(SCRIPT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(p.os, "sched_getaffinity", lambda _: set(range(48)))
    for key, value in dict(SLURM_JOB_ID="999999", SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="4", SLURM_NODEID="0", **p.BASELINE).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(p.SELECTOR, raising=False)
    tick = [100_000_000]
    def clock(): tick[0] += 100_000_000; return tick[0]
    monkeypatch.setattr(p, "time", SimpleNamespace(monotonic=lambda: 1., monotonic_ns=clock))
    seen, waits, quiescence = [], [], []
    actual = json.loads((ROOT / "artifacts/experiment-data/runs/chonk-grouped-forward-selector-i910bfcf936d3/20261116T143844Z-9d9ee6ae/output/0/grouped-forward-0/0/worker.json").read_text())
    mode = {"failure": None}
    class Child:
        def __init__(self, argv, *, env, stdout, stderr, start_new_session):
            assert not start_new_session and stderr == p.subprocess.STDOUT
            slot = int(argv[argv.index("--slot")+1]); self.slot = slot; self.pid = 1000+slot; self.waits = 0
            directory = Path(argv[argv.index("--directory")+1]); selector = argv[argv.index("--selector")+1]
            assert env[p.SELECTOR] == selector
            row = fixtures.worker(0, slot); old_start = row["started_ns"]; start = clock()
            row.update(pid=self.pid, pgid=p.os.getpgrp(), driver_sha256=p.digest(SCRIPT), provenance=p.expected_provenance(),
                       operands=deepcopy(actual["operands"]), operands_post=deepcopy(actual["operands"]),
                       operands_device_pre=deepcopy(actual["operands"]),
                       affinity=deepcopy(actual["affinity"]),
                       started_ns=start, finished_ns=start+2_000_000)
            for sample in row["samples"]:
                for key in ("t0_ns", "return_ns", "complete_ns"): sample[key] += start-old_start
            # Simulate a closed native worker record; actual phase execution is
            # tested separately against measure_protocol, not claimed by this stub.
            (directory / "worker.json").write_text(json.dumps(row))
            seen.append(dict(slot=slot, selector=selector, env=env, directory=directory, row=row))
        def wait(self, timeout):
            self.waits += 1
            waits.append((self.slot, timeout))
            if self.slot == 1 and mode["failure"] == "timeout" and self.waits == 1: raise p.subprocess.TimeoutExpired("fake worker", timeout)
            return 1 if self.slot == 1 and mode["failure"] == "exit" else 0
        def terminate(self): pass
        def kill(self): pytest.fail("cooperative fixture child should not need kill")
    monkeypatch.setattr(p.subprocess, "Popen", Child)
    def quiescent(): quiescence.append(True); return mode["failure"] != "quiescence"
    monkeypatch.setattr(p, "quiescent", quiescent)
    if hasattr(p, "validate_rank"):
        real_validate = p.validate_rank
        def validate_temporary_rank(record, **kwargs):
            # CPU fixture caches live under pytest's private directory, not the
            # native /dev/shm namespace; all other strict gates remain active.
            assert Path(record["scratch"]).is_relative_to(tmp_path / "scratch")
            copy = deepcopy(record); copy["scratch"] = "/dev/shm/apertus-amd/grouped-perf-999999-0-test"
            kwargs["expected_scratch"] = copy["scratch"]
            return real_validate(copy, **kwargs)
        monkeypatch.setattr(p, "validate_rank", validate_temporary_rank)
    return p, settings, mode, seen, waits, quiescence, tmp_path


def test_actual_supervisor_reaps_four_ordered_workers_and_retains_caches(harness):
    p, settings, _, seen, waits, quiescence, tmp = harness
    p.run(settings)
    summary = json.loads((tmp / "grouped-performance-0/summary.json").read_text())
    assert summary["status"] == "passed" and summary["cleanup"] == "retained"
    assert [s["selector"] for s in seen] == ["0", "1", "1", "0"]
    assert len(waits) == 4 and all(t == 470 for _, t in waits) and quiescence == [True]
    assert summary["workers"] == [s["row"] for s in seen]
    for cache_var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        paths = [s["env"][cache_var] for s in seen]
        assert len(set(paths)) == 4 and all(Path(path).is_dir() for path in paths)
    assert Path(summary["scratch"]).is_dir()


@pytest.mark.parametrize("failure", ["exit", "timeout", "quiescence"])
def test_supervisor_failures_retain_evidence_without_retry_or_cleanup(harness, failure):
    p, settings, mode, seen, waits, quiescence, tmp = harness
    mode["failure"] = failure
    with pytest.raises((ValueError, p.subprocess.TimeoutExpired)): p.run(settings)
    summary = json.loads((tmp / "grouped-performance-0/summary.json").read_text())
    assert summary["status"] == "unqualified" and summary["cleanup"] == "retained" and "error" in summary
    assert Path(summary["scratch"]).is_dir()
    assert len(seen) == (4 if failure == "quiescence" else 2)
    assert quiescence == ([True] if failure == "quiescence" else [])
    assert len(summary["workers"]) == (4 if failure == "quiescence" else 1)
    assert len(waits) == len(seen)+(failure == "timeout")  # Bounded reap, never retry/extend.


def test_direct_timeout_escalation_reaps_within_reserved_budget(monkeypatch):
    p = load("grouped_timeout_fixture", SCRIPT); order = []
    monkeypatch.setattr(p, "time", SimpleNamespace(monotonic=lambda: 0.))
    primary = p.subprocess.TimeoutExpired("original", 470)
    class Child:
        waits = 0
        def wait(self, timeout):
            order.append(("wait", timeout)); self.waits += 1
            if self.waits == 1: raise primary
            if self.waits == 2: raise p.subprocess.TimeoutExpired("terminating", timeout)
            return -9
        def terminate(self): order.append("terminate")
        def kill(self): order.append("kill")
    with pytest.raises(p.subprocess.TimeoutExpired) as caught: p.wait_worker(Child(), 480)
    assert caught.value is primary
    assert order == [("wait", 470), "terminate", ("wait", 5), "kill", ("wait", 5)]


def test_failed_direct_teardown_does_not_mask_primary(monkeypatch):
    p = load("grouped_teardown_failure_fixture", SCRIPT)
    monkeypatch.setattr(p, "time", SimpleNamespace(monotonic=lambda: 0.))
    primary = KeyboardInterrupt("primary")
    class Child:
        def wait(self, timeout): raise primary
        def terminate(self): raise OSError("stop failed")
    with pytest.raises(KeyboardInterrupt) as caught: p.wait_worker(Child(), 480)
    assert caught.value is primary and any("stop failed" in note for note in primary.__notes__)


@pytest.mark.parametrize("outcome", ["measured_failure", "post_failure", "success"])
def test_actual_worker_complete_or_emergency_post_path(tmp_path, monkeypatch, outcome):
    p = load("grouped_worker_failure_probe", SCRIPT)
    state = {"grad": True, "device": 0}; phases = []; calls = {}
    @contextmanager
    def no_grad():
        previous = state["grad"]; state["grad"] = False
        try: yield
        finally: state["grad"] = previous
    class Event:
        def __init__(self, **kwargs): pass
        def record(self): pass
        def synchronize(self): pass
        def elapsed_time(self, other): return .16
    cuda = SimpleNamespace(Event=Event, device_count=lambda: 4, current_device=lambda: state["device"],
        set_device=lambda rank: state.update(device=rank), get_device_properties=lambda _: SimpleNamespace(gcnArchName="gfx942"),
        synchronize=lambda *a: None, reset_peak_memory_stats=lambda *a: None,
        max_memory_allocated=lambda *a: 100, max_memory_reserved=lambda *a: 200)
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip="fixture"), no_grad=no_grad,
        get_float32_matmul_precision=lambda: "highest", backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
    linear, native, candidate = object(), object(), object()
    te = SimpleNamespace(GroupedLinear=linear)
    aliases = SimpleNamespace(general_grouped_gemm=native, general_grouped_gemm_triton=candidate)
    modules = {"transformer_engine.pytorch.module.grouped_linear": aliases,
               "transformer_engine.pytorch.cpp_extensions.gemm": SimpleNamespace(general_grouped_gemm=native),
               "transformer_engine.pytorch.triton_kernels.grouped_gemm": SimpleNamespace(general_grouped_gemm_triton=candidate),
               "transformer_engine.pytorch.triton_kernels.gmm.gmm_kernels": object()}
    original_import = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == "torch": return torch
        if name == "transformer_engine.pytorch": return SimpleNamespace(pytorch=te)
        if name.split(".")[0] in {"transformer_engine", "triton", "aiter", "megatron"}: pytest.fail("unexpected host vendor import")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.setattr(p.importlib, "import_module", lambda name: modules[name])
    original_file = p.inspect.getfile
    def getfile(value):
        for obj, relative in ((linear, "module/grouped_linear.py"), (native, "cpp_extensions/gemm.py"), (candidate, "triton_kernels/grouped_gemm.py")):
            if value is obj: return str(p.PACKAGE / (p.PREFIX+relative))
        return original_file(value)
    monkeypatch.setattr(p.inspect, "getfile", getfile)
    original_read = Path.read_text
    def read_text(path, *a, **k):
        if str(path) == "/proc/self/maps":
            return "\n".join("0 0 0 0 0 "+str(p.PACKAGE/name) for name in p.PINS if name.endswith(".so"))
        return original_read(path, *a, **k)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(p, "provenance", p.expected_provenance)
    monkeypatch.setattr(p.os, "sched_getaffinity", lambda _: set(range(48)))
    for key, value in {**p.BASELINE, p.SELECTOR: "0"}.items(): monkeypatch.setenv(key, value)
    snapshot = dict(dynamo={k: {} for k in ("frames", "stats", "inductor", "graph_break")}, generated_kernel_count=0, gmm_keys={})
    monkeypatch.setattr(p, "cache_reader", lambda *a: lambda: deepcopy(snapshot))
    class Layer:
        def __call__(self, key, *args, **kwargs):
            assert not state["grad"]
            calls[key] = calls.get(key, 0)+1
            if calls[key] > 20 and outcome != "success": raise RuntimeError("primary measured operation failed")
    anchor = ROOT / "artifacts/experiment-data/runs/chonk-grouped-forward-selector-i910bfcf936d3/20261116T143844Z-9d9ee6ae/output/0/grouped-forward-0/0/worker.json"
    operands = json.loads(anchor.read_text())["operands"]
    monkeypatch.setattr(p, "operand_hashes", lambda _: deepcopy(operands))
    def prepare(_torch, _te, record):
        record["operands"] = operands
        return {key: dict(layer=Layer(), input=key, splits=[], sizes=None) for key in p.CASES}
    monkeypatch.setattr(p, "prepare_cases", prepare)
    fixtures = load("worker_phase_record_fixture", ARTIFACTS / "grouped-forward-performance-record-validation/test_validator.py")
    valid = fixtures.worker()
    def phase(_torch, _cases, _module, _aliases, record, name):
        phases.append((name, state["grad"]))
        if name == "post" and outcome == "post_failure": raise ValueError("secondary postcheck failed")
        record["branch"][name] = dict(zip(p.ALIASES, (4, 0)))
        record["checks"][name] = deepcopy(valid["checks"][name])
        record["output_hashes"][name] = deepcopy(valid["output_hashes"][name])
        record["layouts"] = deepcopy(valid["layouts"])
        record["aliases_restored"] = True
    monkeypatch.setattr(p, "numerical_phase", phase)
    with (tmp_path / "native.log").open("w") as log:
        monkeypatch.setattr(p.sys, "stdout", log)
        if outcome == "success":
            p.worker(tmp_path, 0, 0, "0")
        else:
            with pytest.raises(RuntimeError, match="primary measured operation") as caught:
                p.worker(tmp_path, 0, 0, "0")
    assert phases == [("pre", False), ("post", False)]
    assert state["grad"] is True
    if outcome == "post_failure": assert any("secondary postcheck" in note for note in caught.value.__notes__)
    record = json.loads((tmp_path / "worker.json").read_text())
    if outcome == "success":
        assert record["status"] == "passed" and len(record["samples"]) == 48
        assert calls == dict.fromkeys(p.CASES, 212)
        p.validate_worker(record, rank=0, slot=0, expected_sha=p.digest(SCRIPT), expected_operands=operands, expected_provenance=p.expected_provenance())
    else:
        assert record["status"] == "unqualified" and "primary measured operation" in record["error"]
