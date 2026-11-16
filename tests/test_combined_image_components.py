"""CPU-only contracts for same-process embedded component tests; never import real Torch."""

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from apertus_amd.cli import plan
from apertus_amd.config import resolve
from apertus_amd.runs import batch_script, edf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/combined-uccl-aiter-components-i910bfcf936d3"
SCRIPT = ROOT / "experiments/validation/combined_image_components.py"


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "aiter", "deep_ep", "triton"}:
            pytest.fail("driver import must not load vendor code")
        return original(name, *args, **kwargs)
    spec = importlib.util.spec_from_file_location("combined_image_components", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded)
        spec.loader.exec_module(module)
    return module


def settings():
    return resolve(EXPERIMENT, site=ROOT / "config.toml", submit=True, time="00:30:00", reservation="cscs")


@pytest.fixture
def fake_runtime(probe, tmp_path, monkeypatch):
    scripts = {phase: tmp_path / f"{phase}-embedded.py" for phase in ("aiter", "uccl")}
    for phase, path in scripts.items():
        path.write_text(f"# CPU-only {phase} identity fixture, never executed\n")
    seed = tmp_path / "seed.json"
    seed.write_text('{"cpu_fixture": true}\n')
    monkeypatch.setattr(probe, "SCRIPTS", scripts)
    monkeypatch.setattr(probe, "SCRIPT_SHA256", {
        phase: hashlib.sha256(path.read_bytes()).hexdigest() for phase, path in scripts.items()})
    monkeypatch.setattr(probe, "SEED", seed)
    monkeypatch.setattr(probe, "SEED_SHA256", hashlib.sha256(seed.read_bytes()).hexdigest())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda pid: {16, 17})
    for name, value in {**settings()["environment"], "HWLOC_COMPONENTS": "-gl",
                        "SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NODEID": "0",
                        "SLURM_NTASKS": "8", "ROCR_VISIBLE_DEVICES": "0,1,2,3"}.items():
        monkeypatch.setenv(name, value)
    events = []
    cuda = SimpleNamespace(device_count=lambda: 4, current_device=lambda: 0,
                           get_device_properties=lambda index: SimpleNamespace(
                               name=f"fake GPU {index}", gcnArchName="gfx942", multi_processor_count=228))
    def select(local):
        events.append(("set_device", local))
        cuda.current_device = lambda: local
    cuda.set_device = select
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip="fake HIP"),
                            set_num_threads=lambda count: events.append(("threads", count)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return probe, cuda, events, scripts, tmp_path


@pytest.mark.parametrize("phase", ["aiter", "uccl"])
@pytest.mark.parametrize("rank", range(8))
def test_every_rank_selects_local_gpu_before_same_process_main(fake_runtime, monkeypatch, phase, rank):
    probe, cuda, events, scripts, directory = fake_runtime
    local = rank % 4
    for name, value in {"SLURM_PROCID": str(rank), "SLURM_LOCALID": str(local),
                        "SLURM_NODEID": str(rank // 4)}.items():
        monkeypatch.setenv(name, value)
    previous = sys.argv
    def embedded(path, *, run_name):
        events.append(("runpy", path))
        assert run_name == "__main__" and path == str(scripts[phase])
        assert sys.argv == [path]  # No phase argument or --metadata-only leaks into embedded argparse.
        assert events[:2] == [("set_device", local), ("threads", 1)]
        before = json.loads((directory / f"component-rank-{rank}.json").read_text())
        assert before["pid"] == probe.os.getpid() and before["selected_device"] == local
        assert before["visible_device_count"] == 4 and len(before["embedded_scripts"]) == 2
        assert before["seed_manifest"]["sha256"] == probe.SEED_SHA256
        assert before["environment_before"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"
    monkeypatch.setattr(probe.runpy, "run_path", embedded)
    probe.run(phase, settings())
    record = json.loads((directory / f"component-rank-{rank}.json").read_text())
    assert record["status"] == "passed" and record["selected_device"] == local
    assert record["cpu_affinity"] == [16, 17] and len(record["devices"]) == 4
    assert record["seed_manifest_after_sha256"] == record["seed_manifest"]["sha256"]
    assert probe.os.environ["ROCR_VISIBLE_DEVICES"] == "0,1,2,3" and sys.argv is previous


@pytest.mark.parametrize("count", [1, 3, 8])
def test_rejects_hidden_or_wrong_gpu_sets_before_embedded_work(fake_runtime, monkeypatch, count):
    probe, cuda, events, _, directory = fake_runtime
    cuda.device_count = lambda: count
    monkeypatch.setattr(probe.runpy, "run_path", lambda *a, **k: pytest.fail("embedded work"))
    with pytest.raises(ValueError, match="all four ROCm"):
        probe.run("aiter", settings())
    assert not events
    assert json.loads((directory / "component-rank-0.json").read_text())["status"] == "failed"


def test_script_hash_failure_precedes_vendor_import_and_execution(fake_runtime, monkeypatch):
    probe, _, events, scripts, directory = fake_runtime
    scripts["aiter"].write_text("changed fixture\n")
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch":
            pytest.fail("hash refusal must precede Torch import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(probe.runpy, "run_path", lambda *a, **k: pytest.fail("embedded work"))
    with pytest.raises(ValueError, match="script hash mismatch"):
        probe.run("aiter", settings())
    assert not events
    assert json.loads((directory / "component-rank-0.json").read_text())["status"] == "failed"


def test_embedded_failure_is_retained_and_argv_restored(fake_runtime, monkeypatch):
    probe, _, _, _, directory = fake_runtime
    previous = sys.argv
    def failed(*args, **kwargs):
        raise RuntimeError("CPU-only embedded failure")
    monkeypatch.setattr(probe.runpy, "run_path", failed)
    with pytest.raises(RuntimeError, match="CPU-only"):
        probe.run("uccl", settings())
    record = json.loads((directory / "component-rank-0.json").read_text())
    assert record["status"] == "failed" and "embedded failure" in record["error"]
    assert sys.argv is previous


def test_plan_is_pure_two_serial_bounded_phases_with_protected_edf(probe, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("plan subprocess"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("plan subprocess"))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: pytest.fail("plan subprocess"))
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: pytest.fail("plan write"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("plan mkdir"))
    value = plan(EXPERIMENT, submit=True, time="00:30:00", reservation="cscs")
    assert value["image"]["sha256"] == probe.IMAGE and value["resources"]["nodes"] == 2
    assert value["resources"]["gpus_per_node"] == 4 and value["resources"]["memory_gb"] == 0
    assert value["resources"]["cpu_bind"] == "map_ldom:0,1,2,3"
    assert value["environment"] == {"OMPI_MCA_memory": "^patcher", "NCCL_P2P_DISABLE": "1",
                                    "PMIX_MCA_psec": "native", "OMP_NUM_THREADS": "1", "MAX_JOBS": "2"}
    assert value["procedure"]["timeout"] == 660 and [p["argv"][-1] for p in value["phases"]] == ["aiter", "uccl"]
    assert "source" not in value and value["prepare"] == []
    for step in value["steps"]:
        assert "--ntasks=8" in step and "--ntasks-per-node=4" in step and "--gpu-bind=none" in step
        assert "--distribution=block:block" in step and "--time=00:30:00" in step
    container = edf(value, Path("/run"))
    assert 'com.hooks.cxi.enabled = "false"' in container and 'com.hooks.aws_ofi_nccl.enabled = "false"' in container
    assert 'HWLOC_COMPONENTS = "-gl"' in container and "/apertus/validation.py:ro" in container
    batch = batch_script({**value, "id": "component-cpu-contract"}, Path("/run"))
    assert "#SBATCH --exclusive" in batch and "--reservation=cscs" in batch and "--time=00:30:00" in batch
    assert "--uenv-passthrough=ignore" in batch and "reference.env" not in str(value)


@pytest.mark.parametrize("damage", ["image", "world", "placement", "threads"])
def test_launch_mismatches_fail_before_vendor_work(probe, damage):
    value = settings()
    env = {**value["environment"], "HWLOC_COMPONENTS": "-gl", "SLURM_PROCID": "5",
           "SLURM_LOCALID": "1", "SLURM_NODEID": "1", "SLURM_NTASKS": "8"}
    if damage == "image":
        value["image"]["sha256"] = "0" * 64
    elif damage == "world":
        env["SLURM_NTASKS"] = "4"
    elif damage == "placement":
        env["SLURM_LOCALID"] = "0"
    else:
        env["OMP_NUM_THREADS"] = "192"
    with pytest.raises(ValueError):
        probe.validate_launch(value, env)
