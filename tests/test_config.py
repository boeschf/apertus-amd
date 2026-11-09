"""Read-only plans, explicit workload procedures, and configuration validation."""

from pathlib import Path
import subprocess

import pytest

from apertus_amd import config
from apertus_amd.cli import plan


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/0001-dense-baseline"


def test_plan_uses_full_node_defaults_without_subprocesses(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("planning must be read-only"))
    result = plan(EXPERIMENT, job_id="123")
    assert result["resources"]["gpus_per_node"] == 4
    assert result["resources"]["cpus_per_node"] == 192
    assert result["resources"]["memory_gb"] == 0
    assert result["cache"]["mode"] == "auto"
    assert result["step"][0] == "srun" and "torchrun" not in str(result)
    assert result["environment"]["NVTE_FLASH_ATTN"] == "1"
    assert result["phases"][1]["extra_args"][-2:] == ["--exit-interval", "2"]


def test_resource_overrides_and_procedure_preserve_explicit_arguments():
    value = config.resolve(EXPERIMENT, job_id="123", gpus_per_node=1, cpus_per_task=16, cache_mode="cold")
    assert value["resources"]["cpu_bind"] == "cores"
    trials = config.phases(value, Path("/output"))
    for trial in trials:
        assert "--train-iters" not in trial["extra_args"]
        assert config.option(value["args"] + trial["extra_args"], "--train-iters") == "4"
    assert config.option(trials[2]["extra_args"], "--save") != config.option(trials[2]["extra_args"], "--load")
    value["procedure"] = {"kind": "train"}
    assert len(config.phases(value, Path("/output"))) == 1


def test_node_count_precedence_in_plans(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("planning must be read-only"))
    experiment = tmp_path / "fixture"
    experiment.mkdir()
    text = (ROOT / "experiments/0003-distributed-collectives/experiment.toml").read_text()
    path = experiment / "experiment.toml"
    path.write_text(text)
    for launch in ({"submit": True}, {"job_id": "123"}):
        result = plan(experiment, **launch)
        assert result["resources"]["nodes"] == 2
        assert "--nodes=2" in result["steps"][0]
        result = plan(experiment, nodes=4, **launch)
        assert result["resources"]["nodes"] == 4
        assert "--nodes=4" in result["steps"][0]
    path.write_text(text.replace("nodes = 2\n", ""))
    assert plan(experiment, submit=True)["resources"]["nodes"] == 1


def test_native_plan_is_read_only_and_rejects_training_options(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("planning must be read-only"))
    experiment = ROOT / "experiments/0003-distributed-collectives"
    value = plan(experiment, submit=True, nodes=2)
    assert len(value["phases"]) == 2
    assert value["phases"][0]["argv"][0] == "all_reduce_perf"
    assert [step[-1] for step in value["steps"]] == ["0", "1"]
    assert "source" not in value and "runtime_environment" not in value
    for options in ({"cache_mode": "warm"}, {"compare_run": Path("reference")}):
        with pytest.raises(ValueError, match="commands do not use"):
            config.resolve(experiment, submit=True, **options)


@pytest.mark.parametrize("old,new", [
    ('nodes = 1', 'nodes = 0'),
    ('nodes = 1', 'nodes = true'),
    ('nodes = 1', 'nodes = "2"'),
    ('commit = "c56351143a8e15899ef3d7d7f97867772f8e0e5c"', 'commit = "main"'),
    ('split = 2', 'split = 4'),
    ('entrypoint = "pretrain_gpt.py"', 'entrypoint = "../escape.py"'),
    ('NVTE_FLASH_ATTN = "1"', 'RANK = "1"'),
])
def test_reject_invalid_inputs(tmp_path, old, new):
    experiment = tmp_path / "fixture"
    experiment.mkdir()
    text = (EXPERIMENT / "experiment.toml").read_text().replace(
        '../validation/dense_restart.py', str(ROOT / 'experiments/validation/dense_restart.py'))
    (experiment / "experiment.toml").write_text(text.replace(old, new))
    with pytest.raises(ValueError):
        config.resolve(experiment, job_id="123")


def test_validation_is_explicit_and_not_executed_during_planning(tmp_path):
    experiment = tmp_path / "fixture"
    experiment.mkdir()
    validator = experiment / "validate.py"
    validator.write_text("raise RuntimeError('planning must not execute validation')\n")
    text = (EXPERIMENT / "experiment.toml").read_text().replace('../validation/dense_restart.py', 'validate.py')
    path = experiment / "experiment.toml"
    path.write_text(text.replace('"--ckpt-format", "torch"', '"--ckpt-format", "torch_dist"'))
    value = config.resolve(experiment, job_id="123")
    assert value["validation"]["path"] == str(validator)
    path.write_text(text.replace('validation = "validate.py"\n', ''))
    with pytest.raises(ValueError, match="explicit validation"):
        config.resolve(experiment, job_id="123")
    path.write_text(text.replace('validation = "validate.py"\n', '').replace('kind = "restart"', 'kind = "train"'))
    assert config.resolve(experiment, job_id="123")["validation"] is None
    with pytest.raises(ValueError, match="explicit validation"):
        config.resolve(experiment, job_id="123", compare_run=tmp_path / "previous")
