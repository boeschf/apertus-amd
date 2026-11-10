"""Freeze one run and launch the same Slurm step directly or through sbatch."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

from . import sources
from .artifacts import image_manifest, sha256, write_json


def step_command(value: dict, directory: Path, *, cpus=None, job_id=None) -> list[str]:
    """Render one task per GPU; allocation-derived CPU counts can remain symbolic in plans."""
    resource = value["resources"]
    cpus = cpus or resource.get("cpus_per_task") or "<allocated CPUs per node / ranks per node>"
    command = ["srun", f"--jobid={job_id or value['launch']['job_id'] or '<assigned-job-id>'}",
               "--overlap", "--exact", "--kill-on-bad-exit=1", "--unbuffered",
               f"--distribution={resource['distribution']}", f"--mpi={resource['mpi']}", f"--network={resource['network']}",
               f"--nodes={resource['nodes']}", f"--ntasks={resource['nodes'] * resource['gpus_per_node']}",
               f"--ntasks-per-node={resource['gpus_per_node']}", "--gpus-per-task=1", "--gpu-bind=none",
               f"--cpus-per-task={cpus}", f"--cpu-bind={resource['cpu_bind']}",
               f"--mem={resource['memory_gb']}G", f"--time={resource['time']}",
               f"--environment={directory / 'container.toml'}",
               # ROCR carries the allocation's physical GPU set. Applying the
               # CUDA/HIP physical indices again can hide remapped devices.
               "env", "-u", "CUDA_VISIBLE_DEVICES", "-u", "HIP_VISIBLE_DEVICES",
                # The image's BASH_ENV initializes its runtime defaults even
                # when its entrypoint simply execs the supplied command.
                "bash", "-c", 'exec "$@"', "apertus",
                "python3", "-u", "-m", "apertus_amd.runtime"]
    return command


def edf(value: dict, directory: Path) -> str:
    """Mount frozen inputs and writable outputs; retain the image's built-in network stack."""
    values = {"image": value["image_path"], "workdir": "/apertus/output", "entrypoint": True,
              "mounts": [f"{directory}:/apertus", f"{directory}/tooling:/apertus/tooling:ro",
                         f"{directory}/resolved.json:/apertus/resolved.json:ro",
                          "/dev/shm:/dev/shm"]}
    if "source_snapshot" in value:
        values["mounts"] += [f"{value['source_snapshot']['directory']}:/apertus/source:ro",
                              f"{value['data_root']}/cache:/apertus/cache"]
    if value["validation"]:
        values["mounts"].append(f"{directory}/validation.py:/apertus/validation.py:ro")
    # GL display discovery can block in XOpenDisplay during OFI topology loading.
    env = {**value["environment"], "HWLOC_COMPONENTS": "-gl", "PYTHONPATH": "/apertus/tooling",
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    if any("$" in str(v) for v in [*values.values(), *env.values()]):
        raise ValueError("EDF settings must not contain interpolation")
    # Alps images contain their complete network stack. Site injection would
    # replace those pinned libraries with a different host-provided build.
    return "\n".join(f"{k} = {json.dumps(v)}" for k, v in values.items()) + "\n\n[env]\n" + \
        "\n".join(f"{k} = {json.dumps(v)}" for k, v in env.items()) + \
        '\n\n[annotations]\ncom.hooks.cxi.enabled = "false"\ncom.hooks.aws_ofi_nccl.enabled = "false"\n'


def batch_script(value: dict, directory: Path) -> str:
    """Thin batch launcher; no workload or runtime lifecycle logic in shell."""
    resource = value["resources"]
    directives = {"job-name": "apertus-" + value["id"], "account": resource["account"],
                  "partition": resource["partition"], "nodes": resource["nodes"],
                  "ntasks-per-node": resource["gpus_per_node"], "distribution": resource["distribution"],
                  "gpus-per-node": resource["gpus_per_node"], "mem": f"{resource['memory_gb']}G",
                  "time": resource["time"], "output": str(directory / "output/slurm.log")}
    if resource.get("reservation"):
        directives["reservation"] = resource["reservation"]
    if resource.get("dependency"):
        directives["dependency"] = resource["dependency"]
    if resource.get("uenv_passthrough"):
        directives["uenv-passthrough"] = resource["uenv_passthrough"]
    lines = ["#!/bin/bash", "#SBATCH --exclusive"]
    for key, item in directives.items():
        if any(c in str(item) for c in "\n\r"):
            raise ValueError("newline in scheduler setting")
        lines.append(f"#SBATCH --{key}={shlex.quote(str(item))}")
    lines += [f"export PYTHONPATH={shlex.quote(str(directory / 'tooling'))}",
               "exec " + shlex.join([sys.executable, "-m", "apertus_amd.runs", str(directory)])]
    return "\n".join(lines) + "\n"


def launch(directory: Path, job_id: str) -> None:
    """Wait for the step; ranks own runtime evidence even if this controller disconnects."""
    value = json.loads((directory / "resolved.json").read_text())
    record = json.loads((directory / "launch.json").read_text())
    try:
        cpus = value["resources"].get("cpus_per_task")
        if cpus is None:
            info = subprocess.run(["scontrol", "show", "job", job_id, "-o"], check=True, capture_output=True, text=True).stdout
            fields = dict(re.findall(r"(\w+)=([^ ]+)", info))
            available = int(fields["NumCPUs"]) // int(fields["NumNodes"])
            cpus = available // value["resources"]["gpus_per_node"]
            if cpus < 1:
                raise ValueError("allocation has fewer CPUs than ranks")
        command = step_command(value, directory, cpus=cpus, job_id=job_id)
        launch_env = dict(os.environ, SLURM_NETWORK=value["resources"]["network"])
        record.update(status="running", job_id=job_id, step=command,
                      environment={"SLURM_NETWORK": launch_env["SLURM_NETWORK"]})
        write_json(directory / "launch.json", record)
        if value["procedure"]["kind"] == "commands":
            native_steps(value, directory, command, launch_env, record)
            return
        with (directory / "output/launcher.log").open("w") as stream:
            status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=launch_env).returncode
        record.update(status="finished", returncode=status)
        write_json(directory / "launch.json", record)
        result = directory / "result.json"
        if status or not result.exists():
            detail = json.loads(result.read_text()) if result.exists() else {}
            write_json(result, {**detail, "status": "failed", "error": f"Slurm step returned {status}; see output/launcher.log"})
            raise ValueError(f"step failed; retained run: {directory}")
        if json.loads(result.read_text())["status"] != "passed":
            raise ValueError(f"runtime checks failed; retained run: {directory}")
    except BaseException as error:
        if not (directory / "result.json").exists():
            write_json(directory / "result.json", {"status": "failed", "error": str(error)})
        raise


def native_steps(value: dict, directory: Path, command: list[str], env: dict, launch: dict) -> None:
    """Use independent steps so MPI_Abort cannot suppress later diagnostic commands."""
    results = []
    world = value["resources"]["nodes"] * value["resources"]["gpus_per_node"]
    for index, argv in enumerate(value["procedure"]["commands"]):
        output = directory / "output" / str(index)
        output.mkdir(exist_ok=True)
        step = [*command, str(index)]
        launch.setdefault("steps", []).append(step)
        write_json(directory / "launch.json", launch)
        with (output / "launcher.log").open("w") as stream:
            status = subprocess.run(step, stdout=stream, stderr=subprocess.STDOUT, env=env).returncode
        ranks = []
        for rank in range(world):
            path = output / f"rank-{rank}.json"
            item = json.loads(path.read_text()) if path.exists() else {"rank": rank, "status": "missing"}
            if item["status"] in {"running", "missing"}:
                item.update(status="failed", error=f"step exited {status} without a final rank record")
            ranks.append(item)
        results.append({"argv": argv, "returncode": status, "ranks": ranks})
        passed = all(result["returncode"] == 0 and all(r["status"] == "passed" for r in result["ranks"])
                     for result in results)
        # Persist each completed step, including ranks killed before their finally block.
        write_json(directory / "result.json", {
            "status": ("passed" if passed else "failed") if len(results) == len(value["procedure"]["commands"]) else "running",
            "execution": "passed" if passed else "failed", "commands": results,
            "checks": "native command exit status; inspect benchmark numerical reports",
            "cache": {"publication": "not applicable"}})
    launch.update(status="finished", returncode=0 if passed else 1)
    write_json(directory / "launch.json", launch)
    if not passed:
        raise ValueError(f"native command failed; retained run: {directory}")


def run(value: dict, checkout: Path | None = None) -> Path:
    """Prepare once, then submit once or execute in an explicitly selected allocation."""
    root = Path(value["data_root"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = root / "runs" / value["experiment"] / run_id
    directory.mkdir(parents=True)
    (directory / "output").mkdir()
    print(f"Run: {directory}", file=sys.stderr, flush=True)
    try:
        image_manifest(Path(value["image_path"]), value["image"]["sha256"])
        if value["procedure"]["kind"] != "commands":
            (root / "cache").mkdir(exist_ok=True)
            source = sources.prepare(value["source"], root / "sources", local=checkout)
            if value["entrypoint"] not in source["files"]:
                raise ValueError("selected source does not contain entrypoint")
            value = {**value, "source_snapshot": {k: source[k] for k in ("directory", "sha256", "tree")}}
        package = Path(__file__).resolve().parent
        shutil.copytree(package, directory / "tooling/apertus_amd", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if value["validation"]:
            shutil.copyfile(value["validation"]["path"], directory / "validation.py")
            if sha256(directory / "validation.py") != value["validation"]["sha256"]:
                raise ValueError("validation script changed after resolution")
        value = {**value, "id": run_id,
                 "tooling": {str(p.relative_to(directory / 'tooling')): sha256(p)
                             for p in (directory / "tooling").rglob("*.py")}}
        if value["compare_run"]:
            reference = Path(value["compare_run"])
            previous = json.loads((reference / "resolved.json").read_text())
            for key in ("args", "entrypoint", "procedure", "environment", "image"):
                if previous[key] != value[key]:
                    raise ValueError(f"comparison input differs: {key}")
            if previous["validation"]["sha256"] != value["validation"]["sha256"]:
                raise ValueError("comparison validation script differs")
            if previous["source_snapshot"]["sha256"] != value["source_snapshot"]["sha256"]:
                raise ValueError("comparison source differs")
            if previous["resources"]["nodes"] * previous["resources"]["gpus_per_node"] != \
                    value["resources"]["nodes"] * value["resources"]["gpus_per_node"]:
                raise ValueError("comparison world size differs")
            if json.loads((reference / "result.json").read_text())["status"] != "passed":
                raise ValueError("comparison reference has not passed")
            shutil.copyfile(reference / "output/checks.json", directory / "reference.json")
        write_json(directory / "resolved.json", value)
        (directory / "container.toml").write_text(edf(value, directory))
        write_json(directory / "launch.json", {"status": "prepared", **value["launch"]})
        if value["launch"]["submit"]:
            script = directory / "job.sh"
            script.write_text(batch_script(value, directory))
            command = ["sbatch", "--parsable", str(script)]
            # Keep intent on ambiguous interruption; never retry sbatch automatically.
            write_json(directory / "launch.json", {"status": "submitting", "command": command, "run_id": run_id})
            reply = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
            job_id = reply.split(";", 1)[0]
            if not job_id.isdigit():
                raise ValueError(f"ambiguous sbatch response: {reply!r}")
            # The job may already be running: submission receipt is separate from
            # the runtime-owned launch/result records, avoiding concurrent overwrite.
            write_json(directory / "submission.json", {"job_id": job_id, "response": reply})
            print(f"Submitted job {job_id}", file=sys.stderr)
        else:
            launch(directory, value["launch"]["job_id"])
    except BaseException as error:
        intent = directory / "launch.json"
        # Batch execution may advance launch.status before sbatch returns (for
        # example with SBATCH_WAIT). Submission intent remains in command.
        submission_intent = intent.exists() and json.loads(intent.read_text()).get("command", [None])[0] == "sbatch"
        if submission_intent:
            receipt = {"status": "unknown", "error": str(error)}
            if isinstance(error, subprocess.CalledProcessError):
                receipt.update(returncode=error.returncode, stdout=error.stdout, stderr=error.stderr)
            write_json(directory / "submission.json", receipt)
        elif not (directory / "result.json").exists():
            write_json(directory / "result.json", {"status": "failed", "error": str(error)})
        raise
    return directory


if __name__ == "__main__":
    launch(Path(sys.argv[1]), os.environ["SLURM_JOB_ID"])
