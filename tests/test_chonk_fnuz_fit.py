"""CPU coverage of the bounded full-layout Chonk fit and checkpoint observer."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from apertus_amd import config, runs
from apertus_amd.artifacts import write_json
from apertus_amd.cli import plan


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/chonk-native-fnuz-fit-i910bfcf936d3"
FULL = ROOT / "experiments/chonk-swissai-resident-fnuz-i910bfcf936d3"
SCRIPT = ROOT / "experiments/validation/chonk_fnuz_fit.py"
SPEC = importlib.util.spec_from_file_location("chonk_fnuz_fit", SCRIPT)
fit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit)


def test_plan_and_unchanged_recipe(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected scheduler call"))
    site = ROOT / "config.toml"
    opts = dict(site=site, submit=True, cpus_per_task=48, cpu_bind="map_ldom:0,1,2,3",
                time="01:00:00", cache_mode="cold", data_root=tmp_path / "absent")
    planned = plan(EXP, **opts)
    full = config.resolve(FULL, site=site, submit=True)
    for key in ("source", "image", "environment", "entrypoint", "prepare"):
        assert planned[key] == full[key]
    expected = list(full["args"])
    expected.remove("--log-device-memory-used")
    for name in ("--train-iters", "--save-interval"):
        expected[expected.index(name) + 1] = "2"
    assert planned["args"] == expected
    assert planned["phases"] == [{"name": "probe", "output": "/apertus/output",
                                  "extra_args": ["--save", "/apertus/output/checkpoints"]}]
    assert planned["procedure"]["timeout"] == 2700
    assert planned["resources"]["nodes"] == 16 and "reservation" not in planned["resources"]
    assert {"--ntasks=64", "--cpus-per-task=48", "--mem=0G"} <= set(planned["step"])
    assert "/apertus/cache" not in runs.edf(planned, tmp_path)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(planned)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("line, replacement", [
    ("checkpoint = true", "checkpoint = false"),
    ('validation = "../validation/chonk_fnuz_fit.py"', ""),
    ('"--save-interval", "2"', '"--save-interval", "1"'),
    ('"--ckpt-format", "torch_dist"', '"--ckpt-format", "torch"'),
    ('"--save-interval", "2"', '"--save-interval", "2", "--async-save"'),
    ('"--save-interval", "2"', '"--save-interval", "2", "--load=/other"'),
    ('"--save-interval", "2"', '"--save-interval", "2", "--no-save-optim"'),
])
def test_probe_save_guards(tmp_path, line, replacement):
    source = (EXP / "experiment.toml").read_text()
    path = tmp_path / "fit-fixture" / "experiment.toml"
    path.parent.mkdir()
    path.write_text(source.replace(line, replacement).replace("../validation/chonk_fnuz_fit.py", str(SCRIPT)))
    with pytest.raises(ValueError, match="checkpoint probe"):
        config.resolve(path, site=ROOT / "config.toml", submit=True)


def test_fresh_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "64")
    monkeypatch.setattr(fit, "train", lambda *a, **kw: pytest.fail("unexpected training"))
    ckpt = tmp_path / "checkpoints"
    argv = ["--save", str(ckpt)]
    with pytest.raises(ValueError, match="fresh synchronous"):
        fit.run_training(tmp_path, tmp_path, "pretrain_gpt.py", ["--load=/other", *argv])
    ckpt.mkdir()
    with pytest.raises(ValueError, match="fresh synchronous"):
        fit.run_training(tmp_path, tmp_path, "pretrain_gpt.py", argv)


@pytest.fixture
def evidence(tmp_path):
    output = tmp_path / "output"
    root = output / "checkpoints"
    target = root / "iter_0000002"
    target.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("2\n")
    (target / "metadata.json").write_text('{"sharded_backend":"torch_dist","common_backend":"torch"}')
    for name in ("common.pt", ".metadata", "__0_0.distcp"):
        (target / name).write_text("inventory-only")
    mem = dict(allocated=5, reserved=9, peak_allocated=8, peak_reserved=11, capacity=12,
               hip_snapshot={"free": 1, "total": 12})
    for rank in range(fit.WORLD):
        records = [
            dict(update=n, consumed_before=(n - 1) * fit.BATCH, microbatches=fit.MICRO,
                 skip=0, reruns=0, discarded=0, forced_save=False, early_exit=False,
                 grad_norm=2.1, losses={"lm loss": 10} if rank in fit.LOSS_RANKS else {},
                 seconds=10.0 + rank / 100, schedule_seconds=9.0,
                 after_schedule=mem, after_optimizer=mem)
            for n in (1, 2)
        ]
        write_json(output / f"fit-rank-{rank}.json",
                   dict(rank=rank, completed=True, geometry=fit.GEOMETRY, config=fit.CONFIG_FIELDS,
                        effective=fit.SETTINGS, transport=fit.TRANSPORT, updates=records,
                        startup_seconds=9, child_seconds=35, before_training=mem, terminal_memory=mem,
                        checkpoint=dict(iteration=2, seconds=3.0, completed=True, before=mem, after=mem)))
        write_json(output / f"imports-{rank}.json", {"verified_modules": 20})
        (output / f"rank-{rank}.log").write_text("UCCL_EP_TRANSPORT=cxi initialized\n")
    return output


def test_fit_inventory_and_timing_are_not_restore(evidence):
    fit.check(evidence.parent)
    report = json.loads((evidence / "checks.json").read_text())
    assert report["status"] == "passed" and report["ranks"] == 64
    assert report["max_update_seconds"] == [10.63, 10.63]
    assert report["max_checkpoint_seconds"] == 3 and report["peak_reserved_bytes"] == 11
    assert report["checkpoint"]["tensor_integrity"] == report["checkpoint"]["restoration"] == "not assessed"


@pytest.mark.parametrize("defect", ["missing_rank", "rerun", "memory", "save", "empty_shard", "early_save", "no_cxi"])
def test_failed_fit_is_not_accepted(evidence, defect):
    row_path = evidence / "fit-rank-63.json"
    row = json.loads(row_path.read_text())
    if defect == "missing_rank":
        row_path.unlink()
    elif defect == "no_cxi":
        (evidence / "rank-63.log").write_text("")
    elif defect == "empty_shard":
        (evidence / "checkpoints/iter_0000002/__0_0.distcp").write_text("")
    elif defect == "early_save":
        (evidence / "checkpoints/iter_0000001").mkdir()
    else:
        if defect == "rerun":
            row["updates"][1]["reruns"] = 1
        elif defect == "memory":
            row["terminal_memory"]["peak_allocated"] = 13
        else:
            row["checkpoint"]["completed"] = False
        write_json(row_path, row)
    with pytest.raises(ValueError):
        fit.check(evidence.parent)
    assert json.loads((evidence / "checks.json").read_text())["status"] == "failed"
