"""Check Turbo component admission without importing vendor code or invoking Slurm."""

import importlib.util
from pathlib import Path
import shutil
import tomllib

import pytest

from apertus_amd import config, runs


ROOT = Path(__file__).resolve().parents[1]
TURBO = ROOT / "artifacts/investigations/performance-next/next-wave-20261119/turbo"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path):
    driver = load("turbo_admission", ROOT / "experiments/validation/chonk_turbo_ck_launch.py")
    proposal = load("turbo_proposal", TURBO / "prepare_launch.py")
    settings = config.resolve(proposal.EXPERIMENT, site=ROOT / "config.toml", **proposal.OPTIONS)
    env = {"SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1", "SLURM_PROCID": "0",
           "SLURM_LOCALID": "0", "SLURM_NODEID": "0", "ROCR_VISIBLE_DEVICES": "3"}
    for name in driver.PINS:
        shutil.copyfile(TURBO / name, tmp_path / name)
    return driver, proposal, settings, env, tmp_path


def test_exact_inputs_are_admitted(fixture):
    driver, proposal, settings, env, path = fixture
    driver.admit(settings, env, set(range(48)), path)
    edf = tomllib.loads(proposal.component_edf(settings, path))
    canonical = tomllib.loads(runs.edf(settings, path))
    assert edf["mounts"].pop() == f"{path}/turbo-input:/apertus/turbo-input:ro"
    assert edf == canonical
    hooks = edf["annotations"]["com"]["hooks"]
    assert hooks["cxi"]["enabled"] == "false"
    assert hooks["aws_ofi_nccl"]["enabled"] == "false"


@pytest.mark.parametrize("key,bad", [("SLURM_NTASKS", "4"), ("SLURM_LOCALID", "1"),
                                     ("ROCR_VISIBLE_DEVICES", ""), ("ROCR_VISIBLE_DEVICES", "0,1")])
def test_wrong_task_and_gpu_layout_are_rejected(fixture, key, bad):
    driver, _, settings, env, path = fixture
    with pytest.raises(ValueError, match="requires"):
        driver.admit(settings, env | {key: bad}, set(range(48)), path)


def test_wrong_cpu_mask_or_timeout_is_rejected(fixture):
    driver, _, settings, env, path = fixture
    with pytest.raises(ValueError, match="48 allocated CPUs"):
        driver.admit(settings, env, set(range(47)), path)
    settings["procedure"]["timeout"] = 301
    with pytest.raises(ValueError, match="timeout"):
        driver.admit(settings, env, set(range(48)), path)


@pytest.mark.parametrize("kind", ["modified", "symlink"])
def test_workload_pin_changes_are_rejected(fixture, kind):
    driver, _, settings, env, path = fixture
    pins = path / "pins.json"
    if kind == "modified":
        pins.write_text("{}")
    else:
        pins.unlink()
        pins.symlink_to(TURBO / "pins.json")
    with pytest.raises(ValueError, match="pinned input differs"):
        driver.admit(settings, env, set(range(48)), path)
