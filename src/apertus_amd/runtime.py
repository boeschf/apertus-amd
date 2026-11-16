"""Slurm rank lifecycle: node preparation, private caches, explicit trials, and collection."""

import ctypes
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from . import cache
from .artifacts import sha256, write_json
from .config import CACHE_PATHS, phases
from .sources import extract


CHILD_GRACE = 10


class UnreapedChildError(RuntimeError):
    """A process group is still live; its scratch must not be removed."""


def calling_thread_affinity() -> frozenset[int]:
    """Read a nonempty calling-thread CPU mask, failing closed on read errors."""
    mask = frozenset(os.sched_getaffinity(0))
    if not mask:
        raise RuntimeError("empty calling-thread CPU affinity")
    return mask


@contextmanager
def preserve_affinity(record: dict):
    """Restore/verify the caller's exact mask after startup, not helper-thread masks."""
    record.update(initial=None, post_gpu_init=None, final=None,
                  restoration_attempted=False, restoration_applied=False, errors=[])
    try:
        initial = calling_thread_affinity()
    except BaseException as error:
        record["errors"].append(f"initial affinity: {error}")
        raise
    record["initial"] = sorted(initial)
    try:
        yield initial
    except BaseException as error:
        record["initialization_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        observed = None
        try:
            observed = calling_thread_affinity()
            record["post_gpu_init"] = sorted(observed)
        except BaseException as error:
            record["errors"].append(f"post-init affinity: {error}")
        # An unreadable mask also requires a restoration attempt, but cannot pass.
        restored = False
        if observed != initial:
            record["restoration_attempted"] = True
            try:
                os.sched_setaffinity(0, initial)
                restored = True
            except BaseException as error:
                record["errors"].append(f"restore affinity: {error}")
        try:
            final = calling_thread_affinity()
            record["final"] = sorted(final)
            if final != initial:
                record["errors"].append(f"final affinity differs from initial: {sorted(final)}")
            else:
                record["restoration_applied"] = restored
        except BaseException as error:
            record["errors"].append(f"verify affinity: {error}")
        if record["errors"]:
            raise RuntimeError("calling-thread affinity preservation failed: " + "; ".join(record["errors"]))


def require_affinity(expected: frozenset[int], stage: str) -> None:
    """Reject later caller-mask drift before launching workload children; do not repair it."""
    observed = calling_thread_affinity()
    if observed != expected:
        raise RuntimeError(f"calling-thread affinity changed before {stage}: "
                           f"expected {sorted(expected)}, observed {sorted(observed)}")


@contextmanager
def defer_termination():
    """Replay TERM/INT after teardown, including signals delivered to other threads."""
    signals = {signal.SIGTERM, signal.SIGINT}
    handlers = {}
    pending = []
    def deferred(signum, frame):
        if not pending:
            pending.append(signum)
    # Python dispatches handlers on the main thread even when a native worker
    # receives the process signal.  Masking only the main thread is insufficient.
    # Production lifecycle calls are on main; CPU helpers may run in worker threads.
    if threading.current_thread() is threading.main_thread():
        for signum in signals:
            handlers[signum] = signal.signal(signum, deferred)
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    try:
        yield
    finally:
        try:
            # Keep deferring while unmasking: this can dispatch pending handlers.
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        if pending:
            # Repeated signals coalesce; replay one original interruption, never
            # permanently ignore termination or let it preempt durable evidence.
            signal.raise_signal(pending[0])


def peer_failure(session: Path) -> None:
    """Raise on a published rank failure, including while local children run."""
    failed = sorted(session.glob("failed-*.json"))
    if failed:
        raise RuntimeError(f"peer failed: {failed[0].read_text()}")


def group_running(group: int) -> bool:
    """Find live Linux process-group members without waiting on orphaned zombies."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group and fields[0] not in {"Z", "X"}:
            return True
    return False


def stop_child(process: subprocess.Popen) -> None:
    """Reap the direct child and stop surviving descendants before cache access."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + CHILD_GRACE
        while True:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
            if process.poll() is not None and not group_running(process.pid):
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    raise UnreapedChildError(f"process group {process.pid} did not exit")


def child(command: list[str], log: Path, *, cwd: Path, env: dict, timeout: float | None = None,
          session: Path | None = None) -> None:
    """Monitor a bounded child and peer failures; stop its group before returning."""
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout if timeout is not None else None
        try:
            while True:
                if session is not None:
                    peer_failure(session)
                remaining = deadline - time.monotonic() if deadline is not None else 0.1
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    status = process.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            # A successful leader can also leave helpers alive and writing caches.
            # Defer repeated scheduler signals until teardown has finished.
            stopped = False
            try:
                with defer_termination():
                    stop_child(process)
                    stopped = True
            except BaseException as error:
                if not stopped:
                    raise UnreapedChildError(f"could not stop process group {process.pid}") from error
                raise
    if status:
        raise ValueError(f"child failed ({status}): {log}")


def wait(paths: list[Path], session: Path, timeout: float, *, errors: bool = True) -> list[dict]:
    """Bounded filesystem rendezvous; a rank failure interrupts peer preparation."""
    deadline = time.monotonic() + timeout
    while True:
        if errors:
            peer_failure(session)
        if all(path.is_file() for path in paths):
            return [json.loads(path.read_text()) for path in paths]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"rendezvous deadline: {[p.name for p in paths if not p.exists()]}")
        time.sleep(0.1)


def exchange(session: Path, name: str, rank: int, world: int, value: dict, timeout: float) -> list[dict]:
    """Publish one rank record and require contributions from every expected rank."""
    write_json(session / f"{name}-{rank}.json", value)
    return wait([session / f"{name}-{r}.json" for r in range(world)], session, timeout)


def rank_environment(environ: dict, visible_gpus: int) -> dict:
    """Preserve Slurm local rank as the selector into the allocation's visible GPUs."""
    local_rank = int(environ["SLURM_LOCALID"])
    if not 0 <= local_rank < visible_gpus:
        raise ValueError(f"local rank {local_rank} cannot select from {visible_gpus} visible GPUs")
    return {"RANK": environ["SLURM_PROCID"], "WORLD_SIZE": environ["SLURM_NTASKS"],
            "LOCAL_RANK": str(local_rank)}


def cleanup(directory: Path, owner: dict) -> None:
    """Remove only owned node scratch after all local children have exited."""
    if directory.is_symlink() or not directory.resolve().is_relative_to("/dev/shm"):
        raise ValueError("refusing unsafe scratch cleanup")
    if json.loads((directory / "owner.json").read_text()) != owner:
        raise ValueError("scratch owner mismatch")
    shutil.rmtree(directory)


def choose_generation(root: Path, descriptor: dict, mode: str) -> dict:
    """Select once; auto misses become coordinated cold starts, warm misses fail."""
    if mode == "cold":
        return {"generation": None, "reason": "cold requested"}
    try:
        generation = (root / cache.identity(descriptor) / "CURRENT").read_text().strip()
        if not cache.DIGEST.fullmatch(generation):
            raise ValueError("invalid cache generation pointer")
        return {"generation": generation, "reason": "compatible namespace selected"}
    except InterruptedError:
        raise
    except (OSError, ValueError) as error:
        if mode == "warm":
            raise
        return {"generation": None, "reason": str(error)}


def record_environment(output: Path, rank: int, node: int, env: dict) -> None:
    """Retain effective transport settings after the image's startup hooks."""
    write_json(output / f"environment-{rank}.json", {
        "rank": rank, "node": node, "hostname": socket.gethostname(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {k: v for k, v in env.items() if k.startswith(
            ("FI_", "NCCL_", "OFI_", "RCCL_", "OMPI_", "PMIX_", "HSA_", "HWLOC_", "ALPS_")) or k in {
                "BASH_ENV", "PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "SLURM_NETWORK",
                "SLURM_LOCALID", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}}})


def native_command(settings: dict, output: Path, index: int, rank: int, env: dict) -> None:
    """Run one image executable; the host collects evidence even if MPI aborts this step."""
    command = settings["procedure"]["commands"][index]
    record = {"rank": rank, "status": "running", "argv": command}
    try:
        executable = shutil.which(command[0], path=env.get("PATH"))
        if executable is None:
            raise FileNotFoundError(command[0])
        record.update(executable=str(Path(executable).resolve()), sha256=sha256(Path(executable)))
        # MPI_Abort can kill the parent without running finally; retain identity first.
        write_json(output / f"rank-{rank}.json", record)
        child(command, output / f"rank-{rank}.log", cwd=output, env=env,
              timeout=settings["procedure"]["timeout"])
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / f"rank-{rank}.json", record)


def verify_tooling(run: Path, settings: dict) -> None:
    """Reject modified frozen runtime modules before executing workload commands."""
    for name, expected in settings["tooling"].items():
        if sha256(run / "tooling" / name) != expected:
            raise ValueError(f"frozen tooling changed: {name}")
    if settings["validation"] and sha256(run / "validation.py") != settings["validation"]["sha256"]:
        raise ValueError("frozen validation script changed")


def training_command(settings: dict, output: Path, source: Path, extra_args: list[str]) -> list[str]:
    """Select ordinary source execution or the experiment's explicit validation wrapper."""
    wrapper = ["/apertus/validation.py", "train"] if settings["validation"] else ["-m", "apertus_amd.entrypoint"]
    return [sys.executable, *wrapper, str(output), str(source), settings["entrypoint"],
            *settings["args"], *extra_args]


def operation_timeout(settings: dict) -> float:
    """Budget bulk cache/check work by planned walltime, not rendezvous latency."""
    hours, minutes, seconds = map(int, settings["resources"]["time"].split(":"))
    return max(settings["cache"]["timeout"], hours * 3600 + minutes * 60 + seconds)


def seed_aiter(seed: str | None, payload: Path) -> None:
    """Copy image-baked modules to a private cache; restored whole objects win."""
    if not seed or not Path(seed).exists():
        return
    directory = Path(seed)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("invalid image AITER seed directory")
    for module in sorted(directory.glob("*.so")):
        cache.object_key(f"aiter/{module.name}")
        if module.is_symlink() or not module.is_file():
            raise ValueError(f"nonregular image AITER seed: {module}")
        target = payload / "aiter" / module.name
        if not target.exists():
            shutil.copy2(module, target)


def publication_worker(request: Path) -> None:
    """Publish in an interruptible process; cache.py's timeout bounds only its lock."""
    value = json.loads(request.read_text())
    published = cache.publish(Path(value["root"]), value["descriptor"],
                              [Path(path) for path in value["bundles"]],
                              max_bytes=value["max_bytes"], timeout=value["lock_timeout"])
    write_json(request.with_name("publication.json"), {
        "publication": "published", "published_generation": published["generation"],
        "differing_contributions": published["differing_contributions"]})


def complete(settings: dict, run: Path, output: Path, session: Path, *, rank: int,
             source: Path, env: dict, gathered: list[dict], result: dict) -> dict:
    """Check before publication and coordinate bounded, separately budgeted stages."""
    timeout = settings["cache"]["timeout"]
    budget = operation_timeout(settings)
    stage_wait = budget + timeout + 2 * CHILD_GRACE
    if rank == 0:
        if settings["validation"]:
            try:
                child([sys.executable, str(run / "validation.py"), "check", str(run)], output / "comparison.log",
                      cwd=source, env=env, timeout=budget, session=session)
            except BaseException as error:
                result.update(checks="failed; output/comparison.log", check_error=str(error))
                raise
            result["checks"] = "passed; output/checks.json"
        write_json(session / "validated.json", result)
    result = wait([session / "validated.json"], session, stage_wait)[0]
    if rank == 0:
        try:
            if any(item["errors"] for item in gathered):
                raise ValueError(f"rank cache capture failed: {[item['errors'] for item in gathered]}")
            bundles = [path for item in gathered for path in item["bundles"]]
            if bundles:
                request = session / "publication-request.json"
                write_json(request, {"root": str(run / "cache"), "descriptor": result["cache"]["descriptor"],
                                     "bundles": bundles, "max_bytes": settings["cache"]["max_gb"] * 2**30,
                                     "lock_timeout": timeout})
                # The subprocess deadline covers lock wait AND verification/merge/fsync.
                # Peer waits include rendezvous slack and bounded group teardown.
                child([sys.executable, "-m", "apertus_amd.runtime", "--publish", str(request)],
                      output / "cache-publication.log", cwd=run,
                      env=dict(env, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
                      timeout=budget, session=session)
                result["cache"].update(json.loads((session / "publication.json").read_text()))
            else:
                result["cache"]["publication"] = "no portable artifacts"
        except InterruptedError:
            raise
        except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired) as error:
            result["cache"].update(publication="failed", error=str(error))
        write_json(session / "checked.json", result)
    return wait([session / "checked.json"], session, stage_wait)[0]


def wait_result(run: Path, session: Path, timeout: float) -> dict:
    """Accept only rank zero's terminal result, not a disconnected controller's stub."""
    deadline = time.monotonic() + timeout
    while True:
        result = wait([run / "result.json"], session, max(0, deadline - time.monotonic()), errors=False)[0]
        if result.get("finalized_by") == "rank-0":
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError("authoritative rank-zero result deadline")
        time.sleep(0.1)


def finalize(run: Path, output: Path, session: Path, scratch: Path, owner: dict, *,
             rank: int, world: int, local: int, ranks_per_node: int, timeout: float,
             status: dict, result: dict) -> dict:
    """Finish owned cleanup and persist the all-rank result before any rank exits."""
    # SIGKILL / a hard scheduler deadline cannot guarantee cleanup.  TERM/INT
    # are deferred only through bounded teardown and rendezvous, never discarded.
    with defer_termination():
        worker_env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        if status["status"] != "passed":
            write_json(session / f"failed-{rank}.json", status)
        # Uncertain child quiescence must never certify a local exit or permit deletion.
        if status.get("children_reaped", True):
            write_json(session / f"exited-{rank}.json", status)
        if local == 0 and (scratch / "owner.json").exists():
            try:
                local_ranks = range(rank, rank + ranks_per_node)
                wait([session / f"exited-{r}.json" for r in local_ranks], session, timeout, errors=False)
                # Bound removal as well as rendezvous; the worker rechecks ownership.
                # Its process group is stopped before this rank publishes finished.
                child([sys.executable, "-m", "apertus_amd.runtime", "--cleanup", str(scratch), json.dumps(owner)],
                      output / f"cleanup-{owner['node']}.log", cwd=run,
                      env=worker_env, timeout=timeout)
                status["cleanup"] = "completed"
            except Exception as error:
                status.update(status="failed", cleanup=str(error))
        write_json(session / f"finished-{rank}.json", status)
        # Peers may still be stopping workload children when this rank arrives;
        # allow that teardown AND the cleanup worker's teardown, plus poll slack.
        collection_timeout = 3 * timeout + 4 * CHILD_GRACE + 0.5
        if rank == 0:
            paths = [session / f"finished-{r}.json" for r in range(world)]
            try:
                result["ranks"] = wait(paths, session, collection_timeout, errors=False)
                if any(item["status"] != "passed" or item.get("cleanup", "completed") != "completed"
                       for item in result["ranks"]):
                    result["status"] = "failed"
                if result["status"] == "passed" and result["cache"].get("publication") != "failed":
                    child([sys.executable, "-m", "apertus_amd.runtime", "--remove-session", str(session)],
                          output / "session-cleanup.log", cwd=run, env=worker_env, timeout=timeout)
            except Exception as error:
                result.update(status="failed", collection_error=str(error))
                if "ranks" not in result:
                    result["ranks"] = [json.loads(path.read_text()) for path in paths if path.is_file()]
                    result["missing_ranks"] = [r for r, path in enumerate(paths) if not path.is_file()]
            result["finalized_by"] = "rank-0"
            write_json(run / "result.json", result)
        # Outside .runtime: rank zero may remove a successful session before peers
        # observe it.  Ignore failure records here, including on mixed pass/fail paths.
        # Also allow rank zero's own local wait/removal and session removal before
        # its collection deadline.  An early non-owner must not time out first.
        return wait_result(run, session, 2 * collection_timeout + timeout)


def main() -> None:
    """Execute one Slurm task; rank zero collects checks and publishes all-rank additions."""
    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if sys.argv[1:2] in (["--cleanup"], ["--remove-session"]):
        # The finalizing parent blocks these signals; the bounded worker must not.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
        if sys.argv[1] == "--cleanup":
            cleanup(Path(sys.argv[2]), json.loads(sys.argv[3]))
        else:
            session = Path(sys.argv[2])
            if session.name != ".runtime" or session.is_symlink():
                raise ValueError("refusing unsafe session cleanup")
            shutil.rmtree(session)
        return
    if sys.argv[1:2] == ["--publish"]:
        publication_worker(Path(sys.argv[2]))
        return
    run, output = Path("/apertus"), Path("/apertus/output")
    settings = json.loads((run / "resolved.json").read_text())
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local, node = int(os.environ["SLURM_LOCALID"]), int(os.environ["SLURM_NODEID"])
    env = dict(os.environ, **settings["environment"])
    if settings["procedure"]["kind"] == "commands":
        verify_tooling(run, settings)
        index = int(sys.argv[1])
        output = output / str(index)
        output.mkdir(exist_ok=True)
        record_environment(output, rank, node, env)
        if world != settings["resources"]["nodes"] * settings["resources"]["gpus_per_node"]:
            raise ValueError("Slurm task count differs from planned layout")
        if settings["launch"]["job_id"] and os.environ["SLURM_JOB_ID"] != settings["launch"]["job_id"]:
            raise ValueError("allocation differs from selected job")
        native_command(settings, output, index, rank, env)
        return
    ranks_per_node = settings["resources"]["gpus_per_node"]
    session = output / ".runtime"
    session.mkdir(exist_ok=True)
    timeout = settings["cache"]["timeout"]
    bulk_timeout = operation_timeout(settings)
    budget = settings["cache"]["max_gb"] * 2**30
    owner = {"uid": os.getuid(), "job_id": os.environ["SLURM_JOB_ID"], "run_id": settings["id"], "node": node}
    scratch = Path(settings["scratch_root"]) / f"{owner['uid']}-{owner['job_id']}-{settings['id']}-{node}"
    status = {"status": "failed", "rank": rank, "node": node}
    result = {"status": "failed"}
    affinity = {}
    try:
        verify_tooling(run, settings)
        if world != settings["resources"]["nodes"] * ranks_per_node:
            raise ValueError("Slurm task count differs from planned layout")
        if settings["launch"]["job_id"] and owner["job_id"] != settings["launch"]["job_id"]:
            raise ValueError("allocation differs from selected job")
        record_environment(output, rank, node, env)
        try:
            with preserve_affinity(affinity) as initial_affinity:
                import torch
                env = dict(os.environ, **settings["environment"], **rank_environment(os.environ, torch.cuda.device_count()))
                torch.cuda.set_device(local)
                gpu = torch.cuda.get_device_properties(local)
        finally:
            # Keep startup/restoration failures even when no GPU rank record is possible.
            write_json(output / f"affinity-{rank}.json", affinity)
        cpus = str(len(initial_affinity))
        affinity["thread_defaults"] = {"captured_cpu_count": len(initial_affinity),
                                      "explicit": {k: env[k] for k in ("OMP_NUM_THREADS", "MAX_JOBS") if k in env}}
        env.setdefault("OMP_NUM_THREADS", cpus)
        env.setdefault("MAX_JOBS", cpus)
        affinity["thread_defaults"]["effective"] = {k: env[k] for k in ("OMP_NUM_THREADS", "MAX_JOBS")}
        write_json(output / f"affinity-{rank}.json", affinity)
        descriptor = {"format": cache.SCHEMA, "adapter": 3, "image": settings["image"]["sha256"],
                      "source": settings["source_snapshot"]["sha256"], "gpu": gpu.gcnArchName,
                      "environment": settings["environment"], "prepare": settings["prepare"]}
        identities = exchange(session, "identity", rank, world, {"key": cache.identity(descriptor)}, timeout)
        if len({item["key"] for item in identities}) != 1:
            raise ValueError("ranks disagree on cache compatibility")
        if rank == 0:
            selection = choose_generation(run / "cache", descriptor, settings["cache"]["mode"])
            with socket.socket() as rendezvous:
                rendezvous.bind(("", 0))
                selection.update(address=socket.gethostname(), port=rendezvous.getsockname()[1])
            write_json(session / "selection.json", selection)
        selected = wait([session / "selection.json"], session, timeout)[0]
        env.update(MASTER_ADDR=selected["address"], MASTER_PORT=str(selected["port"]))
        if local == 0:
            scratch.mkdir(parents=True)
            write_json(scratch / "owner.json", owner)
            if not scratch.resolve().is_relative_to("/dev/shm"):
                raise ValueError("scratch escaped /dev/shm")
            if sha256(run / "source/source.tar") != settings["source_snapshot"]["sha256"]:
                raise ValueError("retained source snapshot changed after preparation")
            # Test executable mappings in the actual container scratch mount.
            (scratch / "probe.c").write_text("int probe(void) { return 42; }\n")
            subprocess.run(["cc", "-shared", "-fPIC", str(scratch / "probe.c"), "-o", str(scratch / "probe.so")], check=True)
            if ctypes.CDLL(str(scratch / "probe.so")).probe() != 42:
                raise ValueError("scratch executable mapping failed")
            staged = {"ok": True}
            if selected["generation"]:
                try:
                    record = cache.stage(run / "cache", descriptor, scratch / "seed", max_bytes=budget,
                                         generation=selected["generation"])
                    needed = sum(item["size"] for item in record["files"].values()) * ranks_per_node
                    if shutil.disk_usage(scratch).free < needed:
                        raise ValueError("insufficient aggregate cache hydration space")
                except InterruptedError:
                    raise
                except (OSError, ValueError) as error:
                    staged = {"ok": False, "error": str(error)}
            write_json(session / f"node-{node}.json", staged)
        nodes = wait([session / f"node-{n}.json" for n in range(settings["resources"]["nodes"])], session, bulk_timeout)
        if not all(item["ok"] for item in nodes):
            if settings["cache"]["mode"] == "warm":
                raise ValueError(f"warm staging failed: {nodes}")
            selected.update(generation=None, reason=f"coordinated cold fallback: {nodes}")
        actual = "warm" if selected["generation"] else "cold"
        exchange(session, "ready", rank, world, {"generation": selected["generation"]}, timeout)
        write_json(output / f"rank-{rank}.json", {"rank": rank, "node": node, "local_task": local,
                   "visible_gpu": local, "gpu": gpu.name, "target": gpu.gcnArchName,
                   "cpu_affinity": sorted(os.sched_getaffinity(0)), "affinity": affinity, "python": sys.version,
                   "torch": torch.__version__, "hip": torch.version.hip, "cache": actual})
        contributions = []
        capture_errors = []
        for phase in phases(settings, output):
            name, directory = phase["name"], Path(phase["output"])
            directory.mkdir(exist_ok=True)
            work, payload = scratch / name, scratch / name / f"cache-{rank}"
            work.mkdir(exist_ok=True)
            source = work / "source"
            if actual == "warm":
                cache.restore(scratch / "seed", payload, max_bytes=budget)
            else:
                payload.mkdir()
            phase_env = dict(env, PYTHONPATH=f"{source}:/apertus/tooling")
            for variable, relative in CACHE_PATHS.items():
                (payload / relative).mkdir(exist_ok=True)
                phase_env[variable] = str(payload / relative)
            seed_aiter(os.environ.get("AITER_JIT_DIR"), payload)
            if local == 0:
                extract(run / "source/source.tar", source)
                for binary in (payload / "helpers").rglob("*.so"):
                    target = source / binary.relative_to(payload / "helpers")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(binary, target)
                for index, command in enumerate(settings["prepare"]):
                    require_affinity(initial_affinity, "prepare child")
                    child(command, directory / f"prepare-{node}-{index}.log", cwd=source, env=phase_env,
                          timeout=bulk_timeout, session=session)
                write_json(session / f"{name}-node-{node}.json", {"ready": True})
            wait([session / f"{name}-node-{node}.json"], session, bulk_timeout)
            exchange(session, name + "-ready", rank, world, {"ready": True}, bulk_timeout)
            command = training_command(settings, directory, source, phase["extra_args"])
            if rank == 0:
                write_json(directory / "command.json", {"argv": command, "environment": {
                    k: phase_env[k] for k in {*settings['environment'], *CACHE_PATHS, "OMP_NUM_THREADS", "MAX_JOBS"}}})
            require_affinity(initial_affinity, "training child")
            child(command, directory / f"rank-{rank}.log", cwd=source, env=phase_env, session=session)
            exchange(session, name + "-exited", rank, world, {"exited": True}, timeout)
            try:
                helpers = payload / "helpers"
                helpers.mkdir(exist_ok=True)
                for binary in source.rglob("*.so"):
                    target = helpers / binary.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(binary, target)
                shutil.rmtree(payload / "tmp")
                record = cache.capture(payload, work / f"bundle-{rank}", max_bytes=budget)
                if record["files"]:
                    destination = session / f"contribution-{name}-{rank}"
                    shutil.copytree(work / f"bundle-{rank}", destination)
                    contributions.append(str(destination))
            except InterruptedError:
                raise
            except (OSError, ValueError) as error:
                capture_errors.append(str(error))
        gathered = exchange(session, "captured", rank, world,
                            {"bundles": contributions, "errors": capture_errors}, bulk_timeout)
        result = {"status": "passed", "execution": "passed", "checks": "not requested",
                  "cache": {"requested": settings["cache"]["mode"], "actual": actual,
                            "key": cache.identity(descriptor), "descriptor": descriptor, **selected}}
        result = complete(settings, run, output, session, rank=rank, source=source,
                          env=phase_env, gathered=gathered, result=result)
        status["status"] = "passed"
    except BaseException as error:
        status["error"] = str(error)
        if affinity:
            status["affinity"] = affinity
        if isinstance(error, UnreapedChildError):
            status["children_reaped"] = False
        raise
    finally:
        result = finalize(run, output, session, scratch, owner, rank=rank, world=world, local=local,
                          ranks_per_node=ranks_per_node, timeout=timeout, status=status, result=result)
    if result["status"] != "passed":
        raise RuntimeError("rank finalization failed; see result.json")


if __name__ == "__main__":
    main()
