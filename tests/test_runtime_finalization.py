"""Real CPU rank processes exercise scheduler-like exit and signal races."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import pytest

from apertus_amd import runtime
from apertus_amd.artifacts import write_json


RANK = r'''
import json, os, signal, sys, threading, time
from pathlib import Path
from apertus_amd import runtime

config = json.loads(sys.argv[1])
rank = config["rank"]
run = Path(config["run"])
session = run / "output/.runtime"
node = rank // 2
scratch = Path(config["scratch"]) / str(node)
owner = {"uid": os.getuid(), "node": node, "run_id": "cpu-finalization"}
runtime.CHILD_GRACE = 0.05
def interrupted(signum, frame):
    raise InterruptedError(f"signal {signum}")
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
if config.get("threaded"):
    ready = threading.Event()
    def native_worker():
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
        ready.set()
        threading.Event().wait()
    threading.Thread(target=native_worker, daemon=True).start()
    assert ready.wait(2)
    original_signal = signal.signal
    def install(signum, handler):
        if callable(handler) and handler.__name__ == "deferred":
            def observed(signum, frame):
                handler(signum, frame)
                (run / f"deferred-signal-{rank}.json").write_text("{}")
            return original_signal(signum, observed)
        return original_signal(signum, handler)
    signal.signal = install
original_child = runtime.child
def child(command, *args, **kwargs):
    if "--remove-session" in command and config.get("session_cleanup_hang"):
        command = [sys.executable, "-c", "import time; time.sleep(60)"]
    if "--cleanup" in command:
        runtime.write_json(run / f"cleanup-start-{node}.json", {})
        time.sleep(config.get("cleanup_delay", 0))
        if config.get("cleanup_hold"):
            deadline = time.monotonic() + 3
            while not (run / "release-cleanup.json").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("test cleanup release missing")
                time.sleep(0.01)
        if config.get("cleanup_failure"):
            raise OSError("injected cleanup failure")
        if config.get("cleanup_hang"):
            command = [sys.executable, "-c", "import time; time.sleep(60)"]
    return original_child(command, *args, **kwargs)
runtime.child = child
if config.get("reap_hold"):
    original_stop = runtime.stop_child
    def stop(process):
        runtime.write_json(run / f"reap-start-{rank}.json", {})
        deadline = time.monotonic() + 3
        while not (run / "release-reap.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("test reap release missing")
            time.sleep(0.01)
        original_stop(process)
    runtime.stop_child = stop
original_write = runtime.write_json
def write(path, value):
    if path == run / "result.json":
        original_write(run / "result-start.json", {})
        time.sleep(config.get("result_delay", 0))
    original_write(path, value)
runtime.write_json = write
original_wait = runtime.wait
def wait(paths, *args, **kwargs):
    if paths == [run / "result.json"] and rank:
        runtime.write_json(run / f"gate-start-{rank}.json", {})
        time.sleep(config.get("gate_delay", 0))
    return original_wait(paths, *args, **kwargs)
runtime.wait = wait
status = {"rank": rank, "node": node, "status": "failed"}
result = {"status": "passed", "execution": "passed", "checks": "not requested", "cache": {}}
try:
    runtime.child([sys.executable, "-c", "pass"], run / f"child-{rank}.log", cwd=run, env=dict(os.environ))
    if rank == config.get("unreaped_rank"):
        raise runtime.UnreapedChildError("cannot establish workload quiescence")
    if rank == config.get("failed_rank"):
        raise ValueError("original checker failure")
    status["status"] = "passed"
except BaseException as error:
    status["error"] = str(error)
    if isinstance(error, runtime.UnreapedChildError):
        status["children_reaped"] = False
    raise
finally:
    result = runtime.finalize(run, run / "output", session, scratch, owner,
        rank=rank, world=config.get("world", 4), local=rank % 2, ranks_per_node=2,
        timeout=config.get("timeout", 1), status=status, result=result)
if result["status"] != "passed":
    raise RuntimeError("rank finalization failed; see result.json")
'''


@contextmanager
def ranks(tmp_path, *, active=(0, 1, 2, 3), **options):
    """Own only these CPU processes and temporary node scratch directories."""
    (tmp_path / "output/.runtime").mkdir(parents=True)
    if options.get("controller_result"):
        write_json(tmp_path / "result.json", {"status": "failed", "error": "controller disconnected"})
    with tempfile.TemporaryDirectory(prefix="apertus-finalization-", dir="/dev/shm") as scratch:
        for node in (0, 1):
            directory = Path(scratch) / str(node)
            directory.mkdir()
            run_id = "wrong-owner" if options.get("owner_mismatch") and node == 1 else "cpu-finalization"
            write_json(directory / "owner.json", {"uid": os.getuid(), "node": node, "run_id": run_id})
            (directory / "private-cache").write_text("owned data")
        processes = []
        streams = []
        try:
            for rank in active:
                config = {"run": str(tmp_path), "scratch": scratch, "rank": rank, **options}
                stream = (tmp_path / f"rank-{rank}.log").open("w")
                streams.append(stream)
                processes.append(subprocess.Popen([sys.executable, "-c", RANK, json.dumps(config)],
                    env=dict(os.environ, PYTHONPATH=str(Path(runtime.__file__).resolve().parents[1])),
                    stdout=stream, stderr=subprocess.STDOUT))
            yield processes, Path(scratch)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            for stream in streams:
                stream.close()


def await_file(path, processes):
    deadline = time.monotonic() + 10
    while not path.is_file():
        assert time.monotonic() < deadline, f"missing {path}"
        assert all(process.poll() is None for process in processes)
        time.sleep(0.01)


def supervise(processes, run, scratch, *, healthy_nodes=(0, 1)):
    """Simulate Slurm kill-on-bad-exit, checking the state at the first bad exit."""
    deadline = time.monotonic() + 10
    first_bad = None
    while any(process.poll() is None for process in processes):
        assert time.monotonic() < deadline
        if first_bad is None and any(process.poll() not in (None, 0) for process in processes):
            first_bad = json.loads((run / "result.json").read_text())
            assert first_bad["finalized_by"] == "rank-0"
            assert all(not (scratch / str(node)).exists() for node in healthy_nodes)
            for process in processes:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
        time.sleep(0.01)
    if first_bad is None and any(process.returncode for process in processes):
        first_bad = json.loads((run / "result.json").read_text())
        assert first_bad["finalized_by"] == "rank-0"
        assert all(not (scratch / str(node)).exists() for node in healthy_nodes)
    return first_bad


def test_first_nonzero_exit_waits_for_slowest_owner_and_authoritative_result(tmp_path):
    with ranks(tmp_path, failed_rank=1, cleanup_delay=0.3, result_delay=0.2) as (processes, scratch):
        result = supervise(processes, tmp_path, scratch)
    assert result["status"] == "failed"
    assert len(result["ranks"]) == 4
    assert result["ranks"][1]["error"] == "original checker failure"
    assert all(row["cleanup"] == "completed" for row in result["ranks"] if row["rank"] % 2 == 0)
    assert "original checker failure" in (tmp_path / "rank-1.log").read_text()


def test_controller_failure_stub_cannot_release_ranks_before_rank_zero_finalizes(tmp_path):
    with ranks(tmp_path, failed_rank=1, controller_result=True, cleanup_delay=0.3) as (processes, scratch):
        result = supervise(processes, tmp_path, scratch)
    assert result["finalized_by"] == "rank-0" and len(result["ranks"]) == 4


def test_repeated_pending_term_and_int_do_not_interrupt_cleanup_or_result(tmp_path):
    with ranks(tmp_path, failed_rank=1, cleanup_delay=0.4, result_delay=0.4) as (processes, scratch):
        for node in (0, 1):
            await_file(tmp_path / f"cleanup-start-{node}.json", processes)
        for _ in range(3):
            for process in processes:
                process.send_signal(signal.SIGTERM)
                process.send_signal(signal.SIGINT)
            time.sleep(0.02)
        await_file(tmp_path / "result-start.json", processes)
        assert not (scratch / "0").exists() and not (scratch / "1").exists()
        for process in processes:
            process.send_signal(signal.SIGTERM)
            process.send_signal(signal.SIGINT)
        result = supervise(processes, tmp_path, scratch)
    assert result["status"] == "failed" and len(result["ranks"]) == 4
    assert result["ranks"][1]["error"] == "original checker failure"
    assert any("InterruptedError: signal" in (tmp_path / f"rank-{rank}.log").read_text() for rank in range(4))


def test_unblocked_worker_thread_cannot_dispatch_raising_handler_during_finalization(tmp_path):
    with ranks(tmp_path, threaded=True, failed_rank=1, cleanup_hold=True, result_delay=0.3) as (processes, scratch):
        for node in (0, 1):
            await_file(tmp_path / f"cleanup-start-{node}.json", processes)
        for process in processes:
            process.send_signal(signal.SIGTERM)
            process.send_signal(signal.SIGINT)
        # The handlers actually dispatched despite the main thread's blocked mask:
        # each rank has an already-existing worker with TERM/INT unblocked.
        for rank in range(4):
            await_file(tmp_path / f"deferred-signal-{rank}.json", processes)
        assert not (tmp_path / "result.json").exists()
        assert all((scratch / str(node)).exists() for node in (0, 1))
        write_json(tmp_path / "release-cleanup.json", {})
        result = supervise(processes, tmp_path, scratch)
    assert result["status"] == "failed" and len(result["ranks"]) == 4
    assert result["ranks"][1]["error"] == "original checker failure"
    assert not any(row.get("children_reaped") is False for row in result["ranks"])
    assert any("InterruptedError: signal" in (tmp_path / f"rank-{rank}.log").read_text() for rank in range(4))


def test_success_session_removal_cannot_race_peer_exit_gate(tmp_path):
    with ranks(tmp_path, gate_delay=0.4) as (processes, scratch):
        assert supervise(processes, tmp_path, scratch) is None
        assert all(process.returncode == 0 for process in processes)
    assert not (tmp_path / "output/.runtime").exists()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "passed" and len(result["ranks"]) == 4


def test_thread_delivered_signal_after_reaping_does_not_claim_unreaped_children(tmp_path):
    with ranks(tmp_path, threaded=True, reap_hold=True) as (processes, scratch):
        for rank in range(4):
            await_file(tmp_path / f"reap-start-{rank}.json", processes)
        for process in processes:
            process.send_signal(signal.SIGTERM)
            process.send_signal(signal.SIGINT)
        for rank in range(4):
            await_file(tmp_path / f"deferred-signal-{rank}.json", processes)
        write_json(tmp_path / "release-reap.json", {})
        result = supervise(processes, tmp_path, scratch)
    assert result["status"] == "failed"
    assert all("signal" in row["error"] and row.get("children_reaped", True) for row in result["ranks"])
    assert len(list((tmp_path / "output/.runtime").glob("exited-*.json"))) == 4


def test_missing_local_rank_is_bounded_and_preserves_partial_result_and_scratch(tmp_path):
    started = time.monotonic()
    with ranks(tmp_path, active=(0, 1, 2), timeout=0.15) as (processes, scratch):
        result = supervise(processes, tmp_path, scratch, healthy_nodes=(0,))
        assert (scratch / "1/private-cache").is_file()
    assert time.monotonic() - started < 5
    assert result["status"] == "failed" and result["missing_ranks"] == [3]
    assert len(result["ranks"]) == 3
    owner = next(row for row in result["ranks"] if row["rank"] == 2)
    assert "exited-3.json" in owner["cleanup"]


def test_missing_coordinator_has_bounded_external_exit_gate(tmp_path):
    started = time.monotonic()
    with ranks(tmp_path, active=(1,), timeout=0.1) as (processes, scratch):
        assert processes[0].wait(timeout=5) != 0
        assert (scratch / "0/private-cache").is_file()
    assert time.monotonic() - started < 5
    assert not (tmp_path / "result.json").exists()
    assert "rendezvous deadline: ['result.json']" in (tmp_path / "rank-1.log").read_text()


def test_signal_replay_after_missing_coordinator_keeps_primary_failure_record(tmp_path):
    with ranks(tmp_path, active=(1,), threaded=True, failed_rank=1, timeout=0.1) as (processes, scratch):
        await_file(tmp_path / "gate-start-1.json", processes)
        processes[0].send_signal(signal.SIGTERM)
        processes[0].send_signal(signal.SIGINT)
        assert processes[0].wait(timeout=5) != 0
        assert (scratch / "0/private-cache").is_file()
    failed = json.loads((tmp_path / "output/.runtime/failed-1.json").read_text())
    assert failed["error"] == "original checker failure"
    log = (tmp_path / "rank-1.log").read_text()
    assert "rendezvous deadline: ['result.json']" in log and "InterruptedError: signal" in log


@pytest.mark.parametrize("failure", ["cleanup_failure", "cleanup_hang"])
def test_failed_or_timed_out_cleanup_is_not_certified(tmp_path, failure):
    with ranks(tmp_path, timeout=0.3, **{failure: True}) as (processes, scratch):
        result = supervise(processes, tmp_path, scratch, healthy_nodes=())
        assert all((scratch / str(node) / "private-cache").is_file() for node in (0, 1))
    assert result["status"] == "failed"
    owners = [row for row in result["ranks"] if row["rank"] % 2 == 0]
    assert all(row["status"] == "failed" and row["cleanup"] != "completed" for row in owners)


def test_unreaped_child_has_no_exit_marker_and_prevents_local_deletion(tmp_path):
    with ranks(tmp_path, unreaped_rank=3, timeout=0.2) as (processes, scratch):
        result = supervise(processes, tmp_path, scratch, healthy_nodes=(0,))
        assert (scratch / "1/private-cache").is_file()
    assert not (tmp_path / "output/.runtime/exited-3.json").exists()
    assert result["status"] == "failed"
    assert result["ranks"][3]["children_reaped"] is False
    assert "exited-3.json" in result["ranks"][2]["cleanup"]


def test_actual_cleanup_worker_rejects_scratch_owner_mismatch(tmp_path):
    with ranks(tmp_path, owner_mismatch=True) as (processes, scratch):
        result = supervise(processes, tmp_path, scratch, healthy_nodes=(0,))
        assert (scratch / "1/private-cache").is_file()
    assert result["status"] == "failed"
    assert result["ranks"][2]["cleanup"] != "completed"
    assert "scratch owner mismatch" in (tmp_path / "output/cleanup-1.log").read_text()


def test_session_removal_timeout_keeps_collected_ranks_and_failed_result(tmp_path):
    with ranks(tmp_path, session_cleanup_hang=True, timeout=0.3) as (processes, scratch):
        result = supervise(processes, tmp_path, scratch)
    assert result["status"] == "failed" and len(result["ranks"]) == 4
    assert all(row["status"] == "passed" for row in result["ranks"])
    assert "timed out" in result["collection_error"]
    assert (tmp_path / "output/.runtime").is_dir()


def test_nested_signal_deferral_restores_handlers_and_mask_on_success_and_error():
    script = r'''
import json, os, signal, threading, time
from apertus_amd.runtime import defer_termination
def interrupted(signum, frame):
    raise InterruptedError(f"signal {signum}")
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
ready = threading.Event()
def worker():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    ready.set()
    threading.Event().wait()
threading.Thread(target=worker, daemon=True).start()
assert ready.wait(2)
previous = signal.pthread_sigmask(signal.SIG_BLOCK, set())
events = []
try:
    with defer_termination():
        with defer_termination():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
            events.append("inner completed")
        events.append("outer completed")
except InterruptedError:
    events.append("replayed")
try:
    with defer_termination():
        raise ValueError("body error")
except ValueError:
    events.append("body error retained")
assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == previous
assert all(signal.getsignal(sig) is interrupted for sig in (signal.SIGTERM, signal.SIGINT))
try:
    signal.raise_signal(signal.SIGINT)
except InterruptedError:
    events.append("later signal not ignored")
print(json.dumps(events))
'''
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True,
        timeout=5, env=dict(os.environ, PYTHONPATH=str(Path(runtime.__file__).resolve().parents[1])))
    assert json.loads(result.stdout) == ["inner completed", "outer completed", "replayed",
                                       "body error retained", "later signal not ignored"]
