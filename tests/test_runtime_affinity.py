"""Calling-thread startup affinity contracts, using fake masks/Torch and CPU-only workers."""

import builtins
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import pytest

from apertus_amd import runtime


BOUND = frozenset({3, 11})
WIDE = frozenset({0, 1, 2, 3, 8, 9, 10, 11})


@pytest.fixture
def affinity(monkeypatch):
    state = SimpleNamespace(mask=BOUND, gets=0, sets=[], read_errors={}, set_error=None, ignore_set=False)

    def get(pid):
        assert pid == 0
        state.gets += 1
        if state.gets in state.read_errors:
            raise state.read_errors[state.gets]
        return set(state.mask)

    def put(pid, mask):
        assert pid == 0
        state.sets.append(frozenset(mask))
        if state.set_error:
            raise state.set_error
        if not state.ignore_set:
            state.mask = frozenset(mask)

    monkeypatch.setattr(runtime.os, "sched_getaffinity", get)
    monkeypatch.setattr(runtime.os, "sched_setaffinity", put)
    return state


def test_widening_restores_exact_captured_mask(affinity):
    record = {}
    with runtime.preserve_affinity(record) as initial:
        assert initial == BOUND and isinstance(initial, frozenset)
        affinity.mask = WIDE
    assert affinity.mask == BOUND
    assert affinity.sets == [BOUND]
    assert record == {"initial": sorted(BOUND), "post_gpu_init": sorted(WIDE), "final": sorted(BOUND),
                      "restoration_attempted": True, "restoration_applied": True, "errors": []}


def test_unchanged_mask_verified_without_setting(affinity):
    record = {}
    with runtime.preserve_affinity(record):
        pass
    assert affinity.gets == 3
    assert affinity.sets == []
    assert record["final"] == sorted(BOUND)
    assert not record["restoration_applied"]


@pytest.mark.parametrize("error", [ValueError("GPU init failed"), KeyboardInterrupt(), InterruptedError("signal 15")])
def test_startup_exception_restores_and_preserves_error(affinity, error):
    record = {}
    with pytest.raises(type(error)) as caught:
        with runtime.preserve_affinity(record):
            affinity.mask = WIDE
            raise error
    assert caught.value is error
    assert affinity.mask == BOUND
    assert record["initialization_error"] == f"{type(error).__name__}: {error}"


@pytest.mark.parametrize("read_number", [1, 2, 3])
def test_any_affinity_read_failure_fails_closed(affinity, read_number):
    affinity.read_errors[read_number] = OSError("get denied")
    record = {}
    launched = []
    with pytest.raises((OSError, RuntimeError), match="get denied"):
        with runtime.preserve_affinity(record):
            affinity.mask = WIDE
        launched.append(True)
    assert not launched
    assert record["errors"]
    if read_number != 1:
        assert affinity.sets == [BOUND]
        assert affinity.mask == BOUND
    else:
        assert record["initial"] is None
        assert affinity.sets == []


@pytest.mark.parametrize("stage", ["initial", "post-init", "final"])
def test_empty_mask_fails_closed(affinity, monkeypatch, stage):
    if stage == "initial":
        affinity.mask = frozenset()
    if stage == "final":
        def empty_restore(pid, mask):
            affinity.sets.append(frozenset(mask))
            affinity.mask = frozenset()
        monkeypatch.setattr(runtime.os, "sched_setaffinity", empty_restore)
    record = {}
    with pytest.raises(RuntimeError, match="empty calling-thread"):
        with runtime.preserve_affinity(record):
            affinity.mask = frozenset() if stage == "post-init" else WIDE
    if stage == "post-init":
        assert affinity.mask == BOUND
        assert record["final"] == sorted(BOUND)


@pytest.mark.parametrize("failure", ["set", "wrong-final"])
def test_failed_restoration_retains_both_errors_and_prevents_launch(affinity, failure):
    affinity.set_error = OSError("set denied") if failure == "set" else None
    affinity.ignore_set = failure == "wrong-final"
    record = {}
    launched = []
    with pytest.raises(RuntimeError, match="affinity preservation failed") as caught:
        with runtime.preserve_affinity(record):
            affinity.mask = WIDE
            raise ValueError("original startup failure")
        launched.append(True)
    assert not launched
    assert isinstance(caught.value.__context__, ValueError)
    assert record["initialization_error"] == "ValueError: original startup failure"
    assert record["final"] == sorted(WIDE)
    assert record["restoration_attempted"]
    assert not record["restoration_applied"]


@pytest.mark.parametrize("mask", [WIDE, frozenset({2, 10}), frozenset()])
def test_child_boundary_rejects_drift_without_repair(affinity, mask):
    affinity.mask = mask
    with pytest.raises(RuntimeError, match="affinity"):
        runtime.require_affinity(BOUND, "training child")
    assert affinity.sets == []


class WorkloadReached(Exception):
    """Stop the fake worker at the training launch, before unrelated cache collection."""


@pytest.fixture
def worker(tmp_path, monkeypatch, affinity):
    run = tmp_path / "run"
    output = run / "output"
    output.mkdir(parents=True)
    state = SimpleNamespace(hook="properties", startup_error=None, drift=None, children=[], statuses=[],
                            events=[], overrides={}, workload_env=None, startup_only=False, final_result=None,
                            identity_timeout=None, startup_trace=False)
    # Only the owned scratch tree is real; no GPU, compiler, or workload process runs.
    with tempfile.TemporaryDirectory(prefix="affinity-cpu-test-", dir="/dev/shm") as scratch:
        settings = {"environment": {}, "procedure": {"kind": "training"},
                    "resources": {"nodes": 1, "gpus_per_node": 1}, "launch": {"job_id": "123"},
                    "cache": {"timeout": 0.1, "max_gb": 1, "mode": "cold"}, "id": "cpu-affinity",
                    "scratch_root": scratch, "image": {"sha256": "image"},
                    "source_snapshot": {"sha256": "source"}, "prepare": [["prepare"]]}
        for key in ("OMP_NUM_THREADS", "MAX_JOBS", "AITER_JIT_DIR", "APERTUS_INDUCTOR_POLICY"):
            monkeypatch.delenv(key, raising=False)
        for key, value in {"SLURM_PROCID": "0", "SLURM_NTASKS": "1", "SLURM_LOCALID": "0",
                           "SLURM_NODEID": "0", "SLURM_JOB_ID": "123", "HIP_VISIBLE_DEVICES": "0,1,2,3"}.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(sys, "argv", ["runtime.py"])
        monkeypatch.setattr(runtime, "Path", lambda p: run if str(p) == "/apertus" else
                            output if str(p) == "/apertus/output" else Path(p))
        monkeypatch.setattr(runtime, "verify_tooling", lambda *args: None)
        monkeypatch.setattr(runtime, "operation_timeout", lambda *args: 1)
        monkeypatch.setattr(runtime, "sha256", lambda *args: "source")
        monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: None)
        monkeypatch.setattr(runtime.ctypes, "CDLL", lambda *args: SimpleNamespace(probe=lambda: 42))
        monkeypatch.setattr(runtime, "seed_aiter", lambda *args: None)
        monkeypatch.setattr(runtime, "phases", lambda *args: [{"name": "train", "output": str(output), "extra_args": []}])

        def hook(stage):
            state.events.append(stage)
            if state.hook == stage:
                affinity.mask = WIDE
                if state.startup_error:
                    raise state.startup_error

        def count():
            hook("count")
            return 4

        def select(local):
            assert local == 0
            hook("selection")

        def properties(local):
            assert local == 0
            hook("properties")
            return SimpleNamespace(name="fake GPU", gcnArchName="fake arch")

        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=count, set_device=select,
                                                     get_device_properties=properties),
                                __version__="fake", version=SimpleNamespace(hip="fake"))
        original_import = builtins.__import__

        def import_module(name, *args, **kwargs):
            if name == "torch":
                early = json.loads((output / "environment-0.json").read_text())
                assert early["cpu_affinity"] == sorted(BOUND)
                hook("import")
                return torch
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_module)

        def extract(*args):
            if state.drift == "prepare":
                affinity.mask = WIDE

        def command(*args):
            if state.drift == "training":
                affinity.mask = WIDE
            return ["training"]

        def child(argv, *args, env, **kwargs):
            assert affinity.mask == BOUND
            state.children.append(argv)
            if argv == ["training"]:
                state.workload_env = env
                raise WorkloadReached()

        def finalize(*args, status, result, **kwargs):
            state.statuses.append(dict(status))
            return result if state.final_result is None else state.final_result

        monkeypatch.setattr(runtime, "extract", extract)
        monkeypatch.setattr(runtime, "training_command", command)
        monkeypatch.setattr(runtime, "child", child)
        monkeypatch.setattr(runtime, "finalize", finalize)

        def execute():
            settings["environment"] = state.overrides
            if state.startup_only:
                settings["procedure"] = {"kind": "probe", "startup_only": True, "timeout": 60}
                monkeypatch.setattr(sys, "argv", ["runtime.py", "--startup-trace"])
            if state.identity_timeout is not None:
                settings["procedure"] = {"kind": "probe", "timeout": 60, "gpu_identity_timeout": state.identity_timeout}
            if state.startup_trace:
                settings["procedure"]["startup_trace"] = True
                monkeypatch.setattr(sys, "argv", ["runtime.py", "--startup-trace"])
            (run / "resolved.json").write_text(json.dumps(settings))
            runtime.main()

        state.run = execute
        state.output = output
        yield state


@pytest.mark.parametrize("hook", ["import", "count", "selection", "properties"])
def test_worker_restores_each_gpu_init_boundary_before_workload(worker, affinity, hook):
    worker.hook = hook
    with pytest.raises(WorkloadReached):
        worker.run()
    assert worker.children == [["prepare"], ["training"]]
    assert affinity.sets == [BOUND]
    record = json.loads((worker.output / "rank-0.json").read_text())
    assert record["cpu_affinity"] == sorted(BOUND)
    assert record["affinity"]["post_gpu_init"] == sorted(WIDE)
    assert record["affinity"]["final"] == sorted(BOUND)
    assert worker.workload_env["HIP_VISIBLE_DEVICES"] == "0,1,2,3"
    assert worker.workload_env["LOCAL_RANK"] == "0"
    assert "APERTUS_INDUCTOR_POLICY" not in worker.workload_env


def test_worker_unchanged_path_never_sets_affinity(worker, affinity):
    worker.hook = None
    with pytest.raises(WorkloadReached):
        worker.run()
    assert affinity.sets == []
    record = json.loads((worker.output / "rank-0.json").read_text())
    assert record["affinity"]["initial"] == record["affinity"]["final"] == sorted(BOUND)
    assert not record["affinity"]["restoration_attempted"]


@pytest.mark.parametrize("trace", [False, True])
def test_worker_only_extends_gpu_identity_exchange(worker, monkeypatch, capsys, trace):
    worker.identity_timeout = 600
    worker.startup_trace = trace
    original = runtime.exchange
    exchanges = []

    def exchange(session, name, rank, world, value, timeout, **kwargs):
        exchanges.append((name, timeout))
        return original(session, name, rank, world, value, timeout, **kwargs)

    monkeypatch.setattr(runtime, "exchange", exchange)
    with pytest.raises(WorkloadReached):
        worker.run()
    assert exchanges == [("identity", 600), ("ready", 0.1), ("train-ready", 1)]
    assert worker.children == [["prepare"], ["training"]]
    rows = [json.loads(line.removeprefix("APERTUS_STARTUP "))
            for line in capsys.readouterr().err.splitlines() if line.startswith("APERTUS_STARTUP ")]
    if trace:
        assert [r["event"] for r in rows[-3:]] == [
            "identity_wait_begin", "identity_wait_end", "startup_identity_validated"]
        assert rows[-3]["timeout_seconds"] == 600
    else:
        assert rows == []


def test_startup_probe_exits_before_any_training_or_staging(worker, capsys):
    worker.startup_only = True
    worker.run()
    assert worker.children == []
    assert worker.statuses[0]["status"] == "passed"
    assert worker.statuses[0]["target"] == "fake arch"
    session = worker.output / ".runtime"
    assert {path.name for path in session.iterdir()} == {"identity-0.json"}
    rows = [json.loads(line.removeprefix("APERTUS_STARTUP "))
            for line in capsys.readouterr().err.splitlines() if line.startswith("APERTUS_STARTUP ")]
    assert [row["event"] for row in rows] == [
        "python_settings_begin", "python_settings_end", "tooling_verify_begin",
        "source_manifest_verify_begin", "verification_end", "environment_write_begin",
        "environment_write_end", "torch_import_begin", "device_count_begin", "device_count_end",
        "device_select_begin", "device_select_end", "device_properties_begin", "device_properties_end",
        "affinity_first_write_begin", "affinity_first_write_end", "affinity_final_write_begin",
        "affinity_final_write_end", "identity_write_begin", "identity_wait_begin", "identity_wait_end",
        "startup_identity_validated",
    ]
    assert all(row["rank"] == row["node"] == 0 and row["unix_ns"] > 0 for row in rows)
    assert [row["elapsed_ns"] for row in rows] == sorted(row["elapsed_ns"] for row in rows)
    assert rows[-3]["timeout_seconds"] == 0.1


def test_startup_probe_preserves_rendezvous_failure_without_training(worker, capsys, monkeypatch):
    worker.startup_only = True
    original = runtime.exchange

    def delayed_peer(session, name, rank, world, value, timeout, **kwargs):
        return original(session, name, rank, world + 1, value, timeout, **kwargs)

    # Keep the fake allocation geometry valid while withholding one identity record.
    monkeypatch.setattr(runtime, "exchange", delayed_peer)
    with pytest.raises(TimeoutError, match="identity-1.json"):
        worker.run()
    assert worker.children == [] and worker.statuses[0]["status"] == "failed"
    stderr = capsys.readouterr().err
    assert "identity_wait_begin" in stderr and "identity_wait_end" not in stderr
    assert "startup_error" in stderr and "identity-1.json" in stderr


def test_startup_probe_propagates_a_peer_finalization_failure(worker):
    worker.startup_only = True
    worker.final_result = {"status": "failed"}
    with pytest.raises(RuntimeError, match="startup finalization failed"):
        worker.run()
    assert worker.children == []


@pytest.mark.parametrize("overrides,expected", [({}, ("2", "2")), ({"MAX_JOBS": "128"}, ("2", "128")),
                                                ({"OMP_NUM_THREADS": "7"}, ("7", "2")),
                                                ({"OMP_NUM_THREADS": "7", "MAX_JOBS": "128"}, ("7", "128"))])
def test_worker_defaults_use_captured_mask_and_keep_explicit_overrides(worker, overrides, expected, monkeypatch):
    # Image overrides arrive in os.environ; experiment overrides also remain explicit.
    if "MAX_JOBS" in overrides:
        monkeypatch.setenv("MAX_JOBS", overrides["MAX_JOBS"])
    worker.overrides = {k: v for k, v in overrides.items() if k != "MAX_JOBS"}
    with pytest.raises(WorkloadReached):
        worker.run()
    assert tuple(worker.workload_env[k] for k in ("OMP_NUM_THREADS", "MAX_JOBS")) == expected
    record = json.loads((worker.output / "affinity-0.json").read_text())
    assert record["thread_defaults"]["captured_cpu_count"] == 2
    assert record["thread_defaults"]["explicit"] == overrides


@pytest.mark.parametrize("hook", ["import", "count", "selection", "properties"])
def test_worker_startup_failure_retains_evidence_and_never_launches(worker, affinity, hook):
    worker.hook = hook
    worker.startup_error = ValueError("fake GPU failure")
    with pytest.raises(ValueError, match="fake GPU failure"):
        worker.run()
    assert affinity.mask == BOUND
    assert worker.children == []
    record = json.loads((worker.output / "affinity-0.json").read_text())
    assert record["initialization_error"] == "ValueError: fake GPU failure"
    assert record["final"] == sorted(BOUND)
    assert worker.statuses[0]["affinity"] == record
    assert not (worker.output / "rank-0.json").exists()


@pytest.mark.parametrize("failure", ["set", "get", "wrong-final"])
def test_worker_restoration_failure_blocks_prepare_and_training(worker, affinity, failure):
    if failure == "set":
        affinity.set_error = OSError("restore denied")
    elif failure == "get":
        # First read is the preserved early environment record, then initial/post/final.
        affinity.read_errors[4] = OSError("verify denied")
    else:
        affinity.ignore_set = True
    with pytest.raises(RuntimeError, match="affinity preservation failed"):
        worker.run()
    assert worker.children == []
    record = json.loads((worker.output / "affinity-0.json").read_text())
    assert record["errors"]
    assert worker.statuses[0]["status"] == "failed"
    assert not (worker.output / "rank-0.json").exists()


@pytest.mark.parametrize("stage,children", [("prepare", []), ("training", [["prepare"]])])
def test_worker_later_mask_drift_prevents_corresponding_child(worker, affinity, stage, children):
    worker.drift = stage
    with pytest.raises(RuntimeError, match=f"affinity changed before {stage} child"):
        worker.run()
    assert worker.children == children
    assert affinity.mask == WIDE  # Later drift is rejected, not silently repaired.
    assert affinity.sets == [BOUND]  # Only GPU-init restoration was attempted.
    assert worker.statuses[0]["status"] == "failed"


@pytest.mark.parametrize("stage,children", [("prepare", []), ("training", [["prepare"]])])
def test_worker_boundary_read_failure_prevents_child(worker, affinity, monkeypatch, stage, children):
    target = "extract" if stage == "prepare" else "training_command"
    original = getattr(runtime, target)

    def fail_next_read(*args):
        value = original(*args)
        affinity.read_errors[affinity.gets + 1] = OSError("boundary read denied")
        return value

    monkeypatch.setattr(runtime, target, fail_next_read)
    with pytest.raises(OSError, match="boundary read denied"):
        worker.run()
    assert worker.children == children
    assert worker.statuses[0]["status"] == "failed"
