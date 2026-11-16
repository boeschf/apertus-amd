"""CPU-only process, publication-stage, and image-seed lifecycle regressions."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from apertus_amd import cache, runtime
from apertus_amd.artifacts import write_json


@pytest.fixture
def completion(tmp_path):
    session = tmp_path / ".runtime"
    session.mkdir()
    payload = tmp_path / "payload"
    (payload / "aiter").mkdir(parents=True)
    (payload / "aiter/module.so").write_bytes(b"compiled module")
    bundle = tmp_path / "bundle"
    cache.capture(payload, bundle, max_bytes=1024)
    settings = {"cache": {"timeout": 0.1, "max_gb": 1}, "resources": {"time": "00:00:02"},
                "validation": None}
    env = dict(os.environ, PYTHONPATH=str(Path(runtime.__file__).parents[1]))
    result = {"status": "passed", "execution": "passed", "checks": "not requested",
              "cache": {"descriptor": {"image": "pinned"}}}
    args = {"settings": settings, "run": tmp_path, "output": tmp_path, "session": session,
            "source": tmp_path, "env": env,
            "gathered": [{"bundles": [str(bundle)], "errors": []}], "result": result}
    return args


def delayed_publication(monkeypatch, delay):
    original = runtime.child
    def child(command, *args, **kwargs):
        if "--publish" in command:
            request = command[-1]
            command = [sys.executable, "-c", "import time; from pathlib import Path; "
                       "from apertus_amd.runtime import publication_worker; "
                       f"time.sleep({delay}); publication_worker(Path({request!r}))"]
        return original(command, *args, **kwargs)
    monkeypatch.setattr(runtime, "child", child)


def test_slow_publication_does_not_exhaust_peer_rendezvous(completion, monkeypatch):
    delayed_publication(monkeypatch, 0.4)
    with ThreadPoolExecutor(max_workers=2) as pool:
        peers = [pool.submit(runtime.complete, **completion, rank=rank) for rank in (1, 0)]
        results = [peer.result(timeout=5) for peer in peers]
    assert all(result["cache"]["publication"] == "published" for result in results)
    assert results[0] == results[1]
    assert (completion["session"] / "validated.json").is_file()
    namespace = completion["run"] / "cache" / cache.identity(completion["result"]["cache"]["descriptor"])
    assert (namespace / "CURRENT").read_text().strip() == results[0]["cache"]["published_generation"]


def test_publication_error_retains_checked_execution_and_existing_generation(completion):
    first = runtime.complete(**completion, rank=0)
    namespace = completion["run"] / "cache" / cache.identity(completion["result"]["cache"]["descriptor"])
    current = (namespace / "CURRENT").read_bytes()
    manifest = (namespace / "generations" / current.decode().strip() / "manifest.json").read_bytes()
    bundle = Path(completion["gathered"][0]["bundles"][0])
    (bundle / "payload.tar").write_bytes(b"corrupt")
    result = runtime.complete(**completion, rank=0)
    assert first["cache"]["publication"] == "published"
    assert result["status"] == "passed" and result["execution"] == "passed"
    assert result["cache"]["publication"] == "failed"
    assert "cache-publication.log" in result["cache"]["error"]
    assert "checksum/schema mismatch" in (completion["output"] / "cache-publication.log").read_text()
    assert (namespace / "CURRENT").read_bytes() == current
    assert (namespace / "generations" / current.decode().strip() / "manifest.json").read_bytes() == manifest


def test_publication_deadline_stops_worker_and_notifies_peers(completion, monkeypatch):
    delayed_publication(monkeypatch, 60)
    monkeypatch.setattr(runtime, "operation_timeout", lambda settings: 0.2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        peers = [pool.submit(runtime.complete, **completion, rank=rank) for rank in (1, 0)]
        results = [peer.result(timeout=5) for peer in peers]
    assert all(result["cache"]["publication"] == "failed" for result in results)
    assert "timed out" in results[0]["cache"]["error"]
    assert not list((completion["run"] / "cache").glob("*/CURRENT"))


def test_interrupted_generation_keeps_previous_pointer_and_immutable_bytes(completion, monkeypatch):
    result = runtime.complete(**completion, rank=0)
    namespace = completion["run"] / "cache" / cache.identity(completion["result"]["cache"]["descriptor"])
    current = (namespace / "CURRENT").read_bytes()
    generation = namespace / "generations" / result["cache"]["published_generation"]
    original = {path.name: path.read_bytes() for path in generation.iterdir()}
    payload = completion["run"] / "extra"
    (payload / "aiter").mkdir(parents=True)
    (payload / "aiter/extra.so").write_bytes(b"new compiled module")
    bundle = completion["run"] / "extra-bundle"
    cache.capture(payload, bundle, max_bytes=1024)
    completion["gathered"] = [{"bundles": [str(bundle)], "errors": []}]
    marker = completion["run"] / "sync-started"
    original_child = runtime.child
    def child(command, *args, **kwargs):
        if "--publish" in command:
            script = ("import time\nfrom pathlib import Path\nfrom apertus_amd import cache, runtime\n"
                      "original = cache._sync\n"
                      "def delayed(path):\n"
                      "    if path.name == 'payload.tar':\n"
                      f"        Path({str(marker)!r}).write_text('sync started')\n"
                      "        time.sleep(60)\n"
                      "    original(path)\n"
                      "cache._sync = delayed\n"
                      # Use main() to install the production worker's SIGTERM handler.
                      "runtime.main()\n")
            command = [sys.executable, "-c", script, "--publish", command[-1]]
        return original_child(command, *args, **kwargs)
    monkeypatch.setattr(runtime, "child", child)
    monkeypatch.setattr(runtime, "operation_timeout", lambda settings: 1)
    result = runtime.complete(**completion, rank=0)
    assert marker.is_file()
    assert result["cache"]["publication"] == "failed"
    assert (namespace / "CURRENT").read_bytes() == current
    assert {path.name: path.read_bytes() for path in generation.iterdir()} == original
    assert list((namespace / "generations").iterdir()) == [generation]


def test_missing_coordinator_has_finite_stage_deadline(completion, monkeypatch):
    monkeypatch.setattr(runtime, "operation_timeout", lambda settings: 0.1)
    monkeypatch.setattr(runtime, "CHILD_GRACE", 0.01)
    with pytest.raises(TimeoutError, match="validated.json"):
        runtime.complete(**completion, rank=1)


def test_disappeared_publisher_has_finite_deadline_after_checks(completion, monkeypatch):
    monkeypatch.setattr(runtime, "operation_timeout", lambda settings: 0.1)
    monkeypatch.setattr(runtime, "CHILD_GRACE", 0.01)
    write_json(completion["session"] / "validated.json", completion["result"])
    with pytest.raises(TimeoutError, match="checked.json"):
        runtime.complete(**completion, rank=1)


def test_peer_failure_aborts_publication_instead_of_becoming_cache_warning(completion, monkeypatch):
    delayed_publication(monkeypatch, 60)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(runtime.complete, **completion, rank=0)
        request = completion["session"] / "publication-request.json"
        deadline = time.monotonic() + 5
        while not request.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        write_json(completion["session"] / "failed-1.json", {"error": "peer lost during publication"})
        with pytest.raises(RuntimeError, match="peer lost during publication"):
            task.result(timeout=5)
    assert not (completion["session"] / "checked.json").exists()


def test_validation_failure_prevents_publication_and_interrupts_peer(completion, monkeypatch):
    completion["settings"]["validation"] = {"sha256": "fixture"}
    (completion["run"] / "validation.py").write_text("raise ValueError('numerical check failed')\n")
    def finish(rank):
        try:
            return runtime.complete(**completion, rank=rank)
        except Exception as error:
            write_json(completion["session"] / f"failed-{rank}.json", {"error": str(error)})
            raise
    with ThreadPoolExecutor(max_workers=2) as pool:
        peer = pool.submit(finish, 1)
        coordinator = pool.submit(finish, 0)
        with pytest.raises(ValueError, match="child failed"):
            coordinator.result(timeout=5)
        with pytest.raises(RuntimeError, match="peer failed"):
            peer.result(timeout=5)
    assert not (completion["session"] / "publication-request.json").exists()
    assert completion["result"]["checks"] == "failed; output/comparison.log"
    assert completion["result"]["execution"] == "passed"
    assert "child failed" in completion["result"]["check_error"]


def test_failed_check_is_retained_in_authoritative_final_result(completion):
    completion["settings"]["validation"] = {"sha256": "fixture"}
    (completion["run"] / "validation.py").write_text("raise ValueError('numerical check failed')\n")
    with pytest.raises(ValueError, match="child failed") as failed:
        runtime.complete(**completion, rank=0)
    result = runtime.finalize(completion["run"], completion["output"], completion["session"],
        completion["run"] / "absent-scratch", {"node": 0}, rank=0, world=1, local=0,
        ranks_per_node=1, timeout=0.1,
        status={"rank": 0, "node": 0, "status": "failed", "error": str(failed.value)},
        result=completion["result"])
    assert result["status"] == "failed" and result["execution"] == "passed"
    assert result["checks"] == "failed; output/comparison.log"
    assert result["check_error"] == str(failed.value)
    assert result["finalized_by"] == "rank-0"
    assert not (completion["session"] / "publication-request.json").exists()
    assert not list((completion["run"] / "cache").glob("*/CURRENT"))


def test_peer_failure_stops_running_child_before_returning(tmp_path):
    pidfile = tmp_path / "pid"
    command = [sys.executable, "-c", f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(60)"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(runtime.child, command, tmp_path / "log", cwd=tmp_path,
                           env=dict(os.environ), session=tmp_path)
        deadline = time.monotonic() + 5
        while not pidfile.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        pid = int(pidfile.read_text())
        write_json(tmp_path / "failed-1.json", {"error": "peer lost"})
        with pytest.raises(RuntimeError, match="peer lost"):
            task.result(timeout=5)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not runtime.group_running(pid)


def test_successful_leader_cannot_leave_cache_writing_descendants(tmp_path):
    pidfile = tmp_path / "group"
    script = ("import os,subprocess,sys; "
              f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); "
              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])")
    runtime.child([sys.executable, "-c", script], tmp_path / "log", cwd=tmp_path, env=dict(os.environ))
    assert not runtime.group_running(int(pidfile.read_text()))


def test_stubborn_child_is_killed_before_timeout_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "CHILD_GRACE", 0.1)
    pidfile = tmp_path / "pid"
    script = ("import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(60)")
    with pytest.raises(subprocess.TimeoutExpired):
        runtime.child([sys.executable, "-c", script], tmp_path / "log", cwd=tmp_path,
                      env=dict(os.environ), timeout=0.3)
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_unreaped_child_is_distinguished_from_safe_child_failure(tmp_path, monkeypatch):
    def stop(process):
        process.wait()
        raise RuntimeError("cannot establish group quiescence")
    monkeypatch.setattr(runtime, "stop_child", stop)
    with pytest.raises(runtime.UnreapedChildError, match="could not stop process group"):
        runtime.child([sys.executable, "-c", "pass"], tmp_path / "log", cwd=tmp_path, env=dict(os.environ))


def test_image_aiter_seed_is_private_and_does_not_overwrite_warm_modules(tmp_path):
    seed, payload = tmp_path / "image", tmp_path / "payload"
    seed.mkdir()
    (payload / "aiter").mkdir(parents=True)
    (seed / "module_core.so").write_bytes(b"image core")
    (seed / "module_warm.so").write_bytes(b"old image version")
    (seed / "build").mkdir()
    (seed / "build/mutable.ninja").write_text("not portable")
    (payload / "aiter/module_warm.so").write_bytes(b"restored version")
    runtime.seed_aiter(str(seed), payload)
    assert (payload / "aiter/module_core.so").read_bytes() == b"image core"
    assert (payload / "aiter/module_warm.so").read_bytes() == b"restored version"
    assert not (payload / "aiter/build").exists()
    assert (payload / "aiter/module_core.so").stat().st_ino != (seed / "module_core.so").stat().st_ino
    (payload / "aiter/module_core.so").write_bytes(b"private change")
    assert (seed / "module_core.so").read_bytes() == b"image core"


def test_image_aiter_seed_rejects_links_and_is_optional(tmp_path):
    (tmp_path / "aiter").mkdir()
    runtime.seed_aiter(None, tmp_path)
    runtime.seed_aiter(str(tmp_path / "missing"), tmp_path)
    seed = tmp_path / "image"
    seed.mkdir()
    (seed / "module.so").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="nonregular"):
        runtime.seed_aiter(str(seed), tmp_path)


def test_cache_selection_does_not_treat_termination_as_auto_miss(tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise InterruptedError("signal 15")
    monkeypatch.setattr(Path, "read_text", interrupted)
    with pytest.raises(InterruptedError, match="signal 15"):
        runtime.choose_generation(tmp_path, {}, "auto")


def test_publication_does_not_swallow_scheduler_termination(completion, monkeypatch):
    def interrupted(*args, **kwargs):
        raise InterruptedError("signal 15")
    monkeypatch.setattr(runtime, "child", interrupted)
    with pytest.raises(InterruptedError, match="signal 15"):
        runtime.complete(**completion, rank=0)
    assert not (completion["session"] / "checked.json").exists()
