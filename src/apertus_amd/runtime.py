"""Slurm rank lifecycle: node preparation, private caches, explicit trials, and collection."""

import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from . import cache
from .artifacts import sha256, write_json
from .config import CACHE_PATHS, phases
from .sources import extract


def child(command: list[str], log: Path, *, cwd: Path, env: dict, timeout: float | None = None) -> None:
    """Wait for a child and terminate its process group before returning on interruption."""
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if status:
        raise ValueError(f"child failed ({status}): {log}")


def wait(paths: list[Path], session: Path, timeout: float, *, errors: bool = True) -> list[dict]:
    """Bounded filesystem rendezvous; a rank failure interrupts peer preparation."""
    deadline = time.monotonic() + timeout
    while True:
        failed = sorted(session.glob("failed-*.json")) if errors else []
        if failed:
            raise RuntimeError(f"peer failed: {failed[0].read_text()}")
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


def main() -> None:
    """Execute one Slurm task; rank zero collects checks and publishes all-rank additions."""
    run, output = Path("/apertus"), Path("/apertus/output")
    settings = json.loads((run / "resolved.json").read_text())
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local, node = int(os.environ["SLURM_LOCALID"]), int(os.environ["SLURM_NODEID"])
    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
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
    budget = settings["cache"]["max_gb"] * 2**30
    owner = {"uid": os.getuid(), "job_id": os.environ["SLURM_JOB_ID"], "run_id": settings["id"], "node": node}
    scratch = Path(settings["scratch_root"]) / f"{owner['uid']}-{owner['job_id']}-{settings['id']}-{node}"
    status = {"status": "failed", "rank": rank, "node": node}
    result = {"status": "failed"}
    try:
        verify_tooling(run, settings)
        if world != settings["resources"]["nodes"] * ranks_per_node:
            raise ValueError("Slurm task count differs from planned layout")
        if settings["launch"]["job_id"] and owner["job_id"] != settings["launch"]["job_id"]:
            raise ValueError("allocation differs from selected job")
        record_environment(output, rank, node, env)
        import torch
        env = dict(os.environ, **settings["environment"], **rank_environment(os.environ, torch.cuda.device_count()))
        torch.cuda.set_device(local)
        gpu = torch.cuda.get_device_properties(local)
        cpus = str(len(os.sched_getaffinity(0)))
        env.setdefault("OMP_NUM_THREADS", cpus)
        env.setdefault("MAX_JOBS", cpus)
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
                except (OSError, ValueError) as error:
                    staged = {"ok": False, "error": str(error)}
            write_json(session / f"node-{node}.json", staged)
        nodes = wait([session / f"node-{n}.json" for n in range(settings["resources"]["nodes"])], session, timeout)
        if not all(item["ok"] for item in nodes):
            if settings["cache"]["mode"] == "warm":
                raise ValueError(f"warm staging failed: {nodes}")
            selected.update(generation=None, reason=f"coordinated cold fallback: {nodes}")
        actual = "warm" if selected["generation"] else "cold"
        exchange(session, "ready", rank, world, {"generation": selected["generation"]}, timeout)
        write_json(output / f"rank-{rank}.json", {"rank": rank, "node": node, "local_task": local,
                   "visible_gpu": local, "gpu": gpu.name, "target": gpu.gcnArchName,
                   "cpu_affinity": sorted(os.sched_getaffinity(0)), "python": sys.version,
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
            if local == 0:
                extract(run / "source/source.tar", source)
                for binary in (payload / "helpers").rglob("*.so"):
                    target = source / binary.relative_to(payload / "helpers")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(binary, target)
                for index, command in enumerate(settings["prepare"]):
                    child(command, directory / f"prepare-{node}-{index}.log", cwd=source, env=phase_env)
                write_json(session / f"{name}-node-{node}.json", {"ready": True})
            wait([session / f"{name}-node-{node}.json"], session, timeout)
            exchange(session, name + "-ready", rank, world, {"ready": True}, timeout)
            command = training_command(settings, directory, source, phase["extra_args"])
            if rank == 0:
                write_json(directory / "command.json", {"argv": command, "environment": {
                    k: phase_env[k] for k in {*settings['environment'], *CACHE_PATHS, "OMP_NUM_THREADS", "MAX_JOBS"}}})
            child(command, directory / f"rank-{rank}.log", cwd=source, env=phase_env)
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
            except (OSError, ValueError) as error:
                capture_errors.append(str(error))
        gathered = exchange(session, "captured", rank, world,
                            {"bundles": contributions, "errors": capture_errors}, timeout)
        if rank == 0:
            result = {"status": "passed", "execution": "passed", "checks": "not requested",
                      "cache": {"requested": settings["cache"]["mode"], "actual": actual,
                                "key": cache.identity(descriptor), **selected}}
            if settings["validation"]:
                child([sys.executable, str(run / "validation.py"), "check", str(run)], output / "comparison.log",
                      cwd=source, env=phase_env)
                result["checks"] = "passed; output/checks.json"
            try:
                if any(item["errors"] for item in gathered):
                    raise ValueError(f"rank cache capture failed: {[item['errors'] for item in gathered]}")
                bundles = [Path(path) for item in gathered for path in item["bundles"]]
                if bundles:
                    published = cache.publish(run / "cache", descriptor, bundles, max_bytes=budget,
                                              timeout=timeout)
                    result["cache"].update(publication="published", published_generation=published["generation"],
                                          differing_contributions=published["differing_contributions"])
                else:
                    result["cache"]["publication"] = "no portable artifacts"
            except (OSError, ValueError, TimeoutError) as error:
                result["cache"].update(publication="failed", error=str(error))
            write_json(session / "checked.json", result)
        wait([session / "checked.json"], session, timeout)
        status["status"] = "passed"
    except BaseException as error:
        status["error"] = str(error)
        write_json(session / f"failed-{rank}.json", status)
        raise
    finally:
        # This marker is written only after child() has reaped the process group.
        write_json(session / f"exited-{rank}.json", status)
        if local == 0 and (scratch / "owner.json").exists():
            try:
                local_ranks = range(rank, rank + ranks_per_node)
                wait([session / f"exited-{r}.json" for r in local_ranks], session, timeout, errors=False)
                cleanup(scratch, owner)
                status["cleanup"] = "completed"
            except Exception as error:
                status["cleanup"] = str(error)
        write_json(session / f"finished-{rank}.json", status)
        if rank == 0:
            try:
                completed = wait([session / f"finished-{r}.json" for r in range(world)], session, timeout, errors=False)
                result["ranks"] = completed
                if any(item["status"] != "passed" or item.get("cleanup", "completed") != "completed" for item in completed):
                    result["status"] = "failed"
                if result["status"] == "passed" and result["cache"].get("publication") != "failed":
                    shutil.rmtree(session)
            except Exception as error:
                result.update(status="failed", collection_error=str(error))
            write_json(run / "result.json", result)


if __name__ == "__main__":
    main()
