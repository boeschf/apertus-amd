"""Test Q/K admission without Slurm or vendor imports."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/validation/chonk_qknorm_native_preflight.py"
SPEC = importlib.util.spec_from_file_location("qk_preflight", SCRIPT)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)
ENV = dict(SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NODEID="0", SLURM_NTASKS="1")


@pytest.fixture
def frozen(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest = assets / "bundle.json"
    manifest.write_text("{}")
    value = dict(
        image={"sha256": DRIVER.IMAGE},
        procedure={"timeout": 450},
        resources=dict(nodes=1, gpus_per_node=1, cpus_per_task=8, reservation="cscs",
                       memory_gb=16, mpi="none", time="00:08:00", cpu_bind="cores"),
        environment={"APERTUS_QK_BUNDLE_SHA256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
    )
    (tmp_path / "resolved.json").write_text(json.dumps(value))
    return tmp_path, assets


def test_exact_bundle_command(frozen):
    root, assets = frozen
    argv = DRIVER.command(root, ENV, range(8), assets)
    assert argv == ["/bin/bash", "--noprofile", "--norc", "/qk-input/native-task.sh",
                    hashlib.sha256(b"{}").hexdigest(), "--allow-cold-fla-autotune"]


def test_rejects_bundle_drift(frozen):
    root, assets = frozen
    (assets / "bundle.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="input manifest differs"):
        DRIVER.command(root, ENV, range(8), assets)


def test_rejects_affinity_and_resource_drift(frozen):
    root, assets = frozen
    with pytest.raises(ValueError, match="eight bound CPUs"):
        DRIVER.command(root, ENV, range(7), assets)
    settings = root / "resolved.json"
    value = json.loads(settings.read_text())
    value["resources"]["nodes"] = 2
    settings.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="resource settings"):
        DRIVER.command(root, ENV, range(8), assets)
