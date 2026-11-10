"""Scheduler stubs and CPU failure/coordination checks for the common rank lifecycle."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from apertus_amd import cache, config, runs, runtime, sources
from apertus_amd.artifacts import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]


def test_rank_mapping_preserves_slurm_local_rank_with_all_gpus_visible():
    env = {"SLURM_PROCID": "7", "SLURM_NTASKS": "8", "SLURM_LOCALID": "3"}
    assert runtime.rank_environment(env, 4) == {"RANK": "7", "WORLD_SIZE": "8", "LOCAL_RANK": "3"}
    with pytest.raises(ValueError, match="visible GPU"):
        runtime.rank_environment(env, 1)


def test_all_rank_agreement_and_failure_propagation(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda rank: runtime.exchange(tmp_path, "ready", rank, 8,
                                  {"generation": "selected", "rank": rank}, 5), range(8)))
    assert all([item["rank"] for item in result] == list(range(8)) for result in results)
    write_json(tmp_path / "failed-3.json", {"error": "preparation failed"})
    with pytest.raises(RuntimeError, match="preparation failed"):
        runtime.wait([tmp_path / "missing"], tmp_path, 5)
    with pytest.raises(TimeoutError):
        runtime.wait([tmp_path / "missing"], tmp_path, 0, errors=False)


def test_auto_cold_warm_selection(tmp_path):
    assert runtime.choose_generation(tmp_path, {}, "auto")["generation"] is None
    with pytest.raises(FileNotFoundError):
        runtime.choose_generation(tmp_path, {}, "warm")
    namespace = tmp_path / cache.identity({})
    namespace.mkdir()
    (namespace / "CURRENT").write_text("a" * 64)
    assert runtime.choose_generation(tmp_path, {}, "auto")["generation"] == "a" * 64
    assert runtime.choose_generation(tmp_path, {}, "cold")["generation"] is None
    (namespace / "CURRENT").write_text("../../invalid")
    assert runtime.choose_generation(tmp_path, {}, "auto")["generation"] is None
    with pytest.raises(ValueError):
        runtime.choose_generation(tmp_path, {}, "warm")


def test_interruption_waits_for_child_before_returning(tmp_path, monkeypatch):
    events = []
    class Process:
        pid = 42
        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: events.append(("kill", pid)))
    with pytest.raises(KeyboardInterrupt):
        runtime.child(["fixture"], tmp_path / "log", cwd=tmp_path, env={})
    assert events == [("wait", None), ("kill", 42), ("wait", 10)]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    value = config.resolve(ROOT / "experiments/dense-baseline-mc56351143a8e-ie84d7ba62309", data_root=tmp_path / "data", job_id="123")
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.tar").write_bytes(b"fixture")
    monkeypatch.setattr(sources, "prepare", lambda *a, **k: {
        "directory": str(source), "sha256": "source-sha", "tree": "git-tree", "files": {"pretrain_gpt.py": "file-sha"}})
    monkeypatch.setattr(runs, "image_manifest", lambda *a, **k: {})
    return value


def test_step_failure_retains_inputs_without_source_copies(prepared, tmp_path, monkeypatch):
    launches = []
    def failed(command, **kwargs):
        launches.append(command)
        if command[0] == "scontrol":
            return subprocess.CompletedProcess(command, 0, stdout="NumCPUs=192 NumNodes=1 ")
        assert kwargs["env"]["SLURM_NETWORK"] == "disable_rdzv_get"
        return subprocess.CompletedProcess(command, 17)
    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(ValueError, match="step failed"):
        runs.run(prepared, tmp_path)
    directory, = (Path(prepared["data_root"]) / "runs" / prepared["experiment"]).iterdir()
    assert json.loads((directory / "result.json").read_text())["status"] == "failed"
    assert (directory / "tooling/apertus_amd/runtime.py").is_file()
    assert not list(directory.rglob("*.tar")) and not list(directory.rglob("*.bundle"))
    assert "--cpus-per-task=48" in launches[-1] and "--ntasks=4" in launches[-1]
    assert "--gpu-bind=none" in launches[-1]
    assert {"--distribution=block:block", "--mpi=pmix", "--network=disable_rdzv_get"}.issubset(launches[-1])
    assert launches[-1][-4:] == ["python3", "-u", "-m", "apertus_amd.runtime"]
    assert "torchrun" not in str(launches[-1])
    assert launches[-1][-8:-4] == ["bash", "-c", 'exec "$@"', "apertus"]
    edf = tomllib.loads((directory / "container.toml").read_text())
    assert edf["entrypoint"] is True
    assert edf["env"]["HWLOC_COMPONENTS"] == "-gl"
    assert edf["annotations"]["com"]["hooks"]["cxi"]["enabled"] == "false"
    assert edf["annotations"]["com"]["hooks"]["aws_ofi_nccl"]["enabled"] == "false"


@pytest.mark.parametrize("reservation", [None, "cscs"])
def test_batch_is_thin_and_submitted_once(prepared, tmp_path, monkeypatch, reservation):
    prepared["launch"] = {"submit": True, "job_id": None}
    prepared["resources"].update(nodes=2, reservation=reservation)
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="12345;cluster\n")
    monkeypatch.setattr(subprocess, "run", submit)
    directory = runs.run(prepared, tmp_path)
    assert len(calls) == 1 and calls[0][:2] == ["sbatch", "--parsable"]
    assert json.loads((directory / "submission.json").read_text())["job_id"] == "12345"
    script = (directory / "job.sh").read_text()
    assert "#SBATCH --exclusive" in script and "#SBATCH --mem=0G" in script
    assert "#SBATCH --distribution=block:block" in script and "#SBATCH --ntasks-per-node=4" in script
    assert "#SBATCH --nodes=2" in script
    assert ("#SBATCH --reservation=cscs" in script) == bool(reservation)
    assert "#SBATCH --uenv-passthrough=ignore" in script
    assert "apertus_amd.runs" in script and "--num-layers" not in script


def test_ambiguous_submission_is_retained_and_never_retried(prepared, tmp_path, monkeypatch):
    prepared["launch"] = {"submit": True, "job_id": None}
    calls = []
    def interrupted(command, **kwargs):
        calls.append(command)
        raise KeyboardInterrupt
    monkeypatch.setattr(subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runs.run(prepared, tmp_path)
    directory, = (Path(prepared["data_root"]) / "runs" / prepared["experiment"]).iterdir()
    assert len(calls) == 1
    assert json.loads((directory / "launch.json").read_text())["status"] == "submitting"


@pytest.mark.parametrize("job_started", [False, True])
def test_submission_error_retains_scheduler_diagnostics_without_retry(prepared, tmp_path, monkeypatch, job_started):
    prepared["launch"] = {"submit": True, "job_id": None}
    calls = []
    def rejected(command, **kwargs):
        calls.append(command)
        if job_started:
            directory = Path(command[-1]).parent
            launch = json.loads((directory / "launch.json").read_text())
            write_json(directory / "launch.json", {**launch, "status": "finished", "job_id": "42"})
            write_json(directory / "result.json", {"status": "failed", "error": "runtime failed"})
        raise subprocess.CalledProcessError(1, command, output="", stderr="site plugin rejected submission")
    monkeypatch.setattr(subprocess, "run", rejected)
    with pytest.raises(subprocess.CalledProcessError):
        runs.run(prepared, tmp_path)
    directory, = (Path(prepared["data_root"]) / "runs" / prepared["experiment"]).iterdir()
    receipt = json.loads((directory / "submission.json").read_text())
    assert len(calls) == 1
    assert receipt["returncode"] == 1 and receipt["stderr"] == "site plugin rejected submission"
    if job_started:
        assert json.loads((directory / "result.json").read_text())["error"] == "runtime failed"


def test_native_commands_submission_needs_no_source_or_cache(tmp_path, monkeypatch):
    value = config.resolve(ROOT / "experiments/distributed-collectives-i0662f3a37ef3", submit=True,
                           nodes=2, data_root=tmp_path)
    monkeypatch.setattr(runs, "image_manifest", lambda *a, **k: {})
    monkeypatch.setattr(sources, "prepare", lambda *a, **k: pytest.fail("native image workload has no source"))
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout="12345\n"))
    directory = runs.run(value)
    edf = tomllib.loads((directory / "container.toml").read_text())
    assert edf["entrypoint"] is True
    assert edf["env"]["HWLOC_COMPONENTS"] == "-gl"
    assert edf["annotations"]["com"]["hooks"]["cxi"]["enabled"] == "false"
    assert edf["annotations"]["com"]["hooks"]["aws_ofi_nccl"]["enabled"] == "false"
    assert not any(":/apertus/source" in mount or ":/apertus/cache" in mount for mount in edf["mounts"])
    assert not (tmp_path / "sources").exists() and not (tmp_path / "cache").exists()


def test_native_command_retains_executable_identity_and_failure(tmp_path):
    settings = {"procedure": {"kind": "commands", "timeout": 5, "commands": [
        [sys.executable, "-c", "raise SystemExit(3)"]]}}
    with pytest.raises(ValueError, match="child failed"):
        runtime.native_command(settings, tmp_path, 0, 0, {})
    result = json.loads((tmp_path / "rank-0.json").read_text())
    assert result["status"] == "failed" and result["sha256"]


def test_native_steps_continue_after_mpi_abort(tmp_path, monkeypatch):
    value = config.resolve(ROOT / "experiments/distributed-collectives-i0662f3a37ef3", job_id="123")
    (tmp_path / "output").mkdir()
    calls = []
    def step(command, **kwargs):
        calls.append(command)
        index = command[-1]
        if index == "0":
            write_json(tmp_path / "output/0/rank-0.json", {"rank": 0, "status": "running", "sha256": "retained"})
            return subprocess.CompletedProcess(command, 137)
        for rank in range(value["resources"]["nodes"] * value["resources"]["gpus_per_node"]):
            write_json(tmp_path / f"output/{index}/rank-{rank}.json", {"rank": rank, "status": "passed"})
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", step)
    with pytest.raises(ValueError, match="native command failed"):
        runs.native_steps(value, tmp_path, ["srun", "runtime"], {}, {})
    assert [c[-1] for c in calls] == ["0", "1"]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["commands"][0]["ranks"][0]["sha256"] == "retained"
    assert all(r["status"] == "failed" for r in result["commands"][0]["ranks"])
    assert all(r["status"] == "passed" for r in result["commands"][1]["ranks"])


def test_child_timeout_reaps_process(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        runtime.child([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "log",
                      cwd=tmp_path, env={}, timeout=0.1)


def test_validation_is_frozen_and_modification_is_rejected(prepared, tmp_path, monkeypatch):
    validator = tmp_path / "validator.py"
    validator.write_text("# original validation\n")
    prepared["validation"] = {"path": str(validator), "sha256": sha256(validator)}
    prepared["launch"] = {"submit": True, "job_id": None}
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout="12345\n"))
    directory = runs.run(prepared, tmp_path)
    value = json.loads((directory / "resolved.json").read_text())
    validator.write_text("# changed working tree\n")
    runtime.verify_tooling(directory, value)
    assert (directory / "validation.py").read_text() == "# original validation\n"
    edf = tomllib.loads((directory / "container.toml").read_text())
    assert f"{directory}/validation.py:/apertus/validation.py:ro" in edf["mounts"]
    (directory / "validation.py").write_text("# changed frozen copy\n")
    with pytest.raises(ValueError, match="frozen validation"):
        runtime.verify_tooling(directory, value)


def test_validation_change_between_plan_and_run_is_rejected(prepared, tmp_path):
    validator = tmp_path / "validator.py"
    validator.write_text("# original\n")
    prepared["validation"] = {"path": str(validator), "sha256": sha256(validator)}
    validator.write_text("# changed\n")
    with pytest.raises(ValueError, match="changed after resolution"):
        runs.run(prepared, tmp_path)


def test_ordinary_training_has_no_validation_wrapper_or_payload(prepared, tmp_path, monkeypatch):
    prepared["procedure"] = {"kind": "train"}
    prepared["validation"] = None
    prepared["launch"] = {"submit": True, "job_id": None}
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout="12345\n"))
    directory = runs.run(prepared, tmp_path)
    assert not (directory / "validation.py").exists()
    command = runtime.training_command(prepared, Path("/output"), Path("/source"), ["--save", "/output/checkpoints"])
    assert command[1:3] == ["-m", "apertus_amd.entrypoint"]
    assert command[-2:] == ["--save", "/output/checkpoints"]
    prepared["validation"] = {"path": "unused", "sha256": "unused"}
    validated = runtime.training_command(prepared, Path("/output"), Path("/source"), ["--save", "/output/checkpoints"])
    assert validated[1:3] == ["/apertus/validation.py", "train"]
    assert command[3:] == validated[3:]
