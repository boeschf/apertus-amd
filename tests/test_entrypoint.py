"""Execute source wrappers and experiment instrumentation without a GPU runtime."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from apertus_amd.artifacts import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dense_restart", ROOT / "experiments/validation/dense_restart.py")
dense = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dense)


@pytest.mark.parametrize("failure", [False, True])
def test_source_execution_records_imports_even_on_failure(tmp_path, failure):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "megatron").mkdir()
    (source / "megatron/__init__.py").write_text("value = 'selected source'\n")
    script = "import megatron, sys\nassert megatron.value == 'selected source'\nassert sys.argv[1:] == ['--native', 'value']\n"
    if failure:
        script += "raise RuntimeError('workload failed')\n"
    (source / "train.py").write_text(script)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"files": {str(p.relative_to(source)): sha256(p) for p in source.rglob("*.py")}})
    command = [sys.executable, "-c", "from pathlib import Path; from apertus_amd.entrypoint import train; "
               f"train(Path({str(output)!r}), Path({str(source)!r}), 'train.py', ['--native', 'value'], manifest=Path({str(manifest)!r}))"]
    result = subprocess.run(command, env={**os.environ, "RANK": "3", "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    assert (result.returncode != 0) == failure
    if failure:
        assert "workload failed" in result.stderr
    assert json.loads((output / "imports-3.json").read_text())["verified_modules"] == 1
    assert not list(output.glob("steps-*")) and not list(output.glob("model-*"))


def test_source_wrapper_rejects_shadowed_import_before_execution(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    marker = tmp_path / "executed"
    (source / "train.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"files": {"train.py": sha256(source / "train.py")}})
    command = [sys.executable, "-c", "import sys, types; from pathlib import Path; from apertus_amd.entrypoint import train; "
               f"m = types.ModuleType('megatron'); m.__file__ = {str(tmp_path / 'image/megatron.py')!r}; sys.modules['megatron'] = m; "
               f"train(Path({str(tmp_path)!r}), Path({str(source)!r}), 'train.py', [], manifest=Path({str(manifest)!r}))"]
    result = subprocess.run(command, env={**os.environ, "RANK": "0"}, capture_output=True, text=True)
    assert result.returncode != 0 and "not in the subpath" in result.stderr
    assert not marker.exists()


@pytest.fixture
def training_stub(monkeypatch):
    package = ModuleType("megatron")
    training = ModuleType("megatron.training")
    module = ModuleType("megatron.training.training")
    package.training = training
    training.training = module
    training.get_args = lambda: SimpleNamespace(iteration=0)
    attention = SimpleNamespace()
    model = SimpleNamespace(core_attention=attention)
    model.modules = lambda: [model]
    module.setup_model_and_optimizer = lambda: ([model], object(), object())
    module.train_step = lambda iteration: ({"lm loss": 1.25}, 0, None, None, None, 0.5)
    for name, value in (("megatron", package), ("megatron.training", training), ("megatron.training.training", module)):
        monkeypatch.setitem(sys.modules, name, value)
    return module


def test_dense_instrumentation_preserves_returns_and_records_metrics(tmp_path, training_stub):
    expected = training_stub.train_step(0)
    dense.instrument(tmp_path, {"attention": "types.SimpleNamespace"}, 2)
    training_stub.setup_model_and_optimizer()
    assert training_stub.train_step(0) == expected
    assert json.loads((tmp_path / "steps-2.jsonl").read_text()) == {
        "iteration": 1, "losses": {"lm loss": 1.25}, "skipped": 0, "grad_norm": 0.5}
    assert json.loads((tmp_path / "model-2.json").read_text())["attention"] == ["types.SimpleNamespace"]


def test_dense_instrumentation_rejects_wrong_attention_and_invalid_updates(tmp_path, training_stub):
    training_stub.train_step = lambda iteration: ({"lm loss": float("nan")}, 0, None, None, None, 0.5)
    dense.instrument(tmp_path, {"attention": "expected.Attention"}, 0)
    with pytest.raises(ValueError, match="unexpected attention"):
        training_stub.setup_model_and_optimizer()
    with pytest.raises(ValueError, match="invalid optimizer update"):
        training_stub.train_step(0)


def test_dense_specific_constraints_and_failure_record_are_experiment_owned(tmp_path):
    settings = {"procedure": {"kind": "restart"}, "args": ["--ckpt-format", "torch_dist"]}
    (tmp_path / "output").mkdir()
    write_json(tmp_path / "resolved.json", settings)
    with pytest.raises(ValueError, match="unsharded torch"):
        dense.check_run(tmp_path)
    result = json.loads((tmp_path / "output/checks.json").read_text())
    assert result["status"] == "failed" and "unsharded torch" in result["error"]


def test_dense_wrapper_executes_source_and_instruments_only_explicit_validation(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "megatron/training").mkdir(parents=True)
    output.mkdir()
    (source / "megatron/__init__.py").write_text("")
    (source / "megatron/training/__init__.py").write_text(
        "from types import SimpleNamespace\ndef get_args(): return SimpleNamespace(iteration=0)\n")
    (source / "megatron/training/training.py").write_text(
        "def setup_model_and_optimizer(): return [], object(), object()\n"
        "def train_step(iteration): return {'lm loss': 2.0}, 0, None, None, None, 1.0\n")
    (source / "train.py").write_text(
        "from megatron.training import training\ntraining.setup_model_and_optimizer()\ntraining.train_step(iteration=0)\n")
    write_json(source / "manifest.json", {"files": {str(p.relative_to(source)): sha256(p) for p in source.rglob("*.py")}})
    write_json(tmp_path / "resolved.json", {"procedure": {"kind": "restart"}, "args": ["--ckpt-format", "torch"]})
    wrapper = ROOT / "experiments/validation/dense_restart.py"
    command = [sys.executable, "-c", "import runpy; from pathlib import Path; "
               f"runpy.run_path({str(wrapper)!r})['main'](['train', {str(output)!r}, {str(source)!r}, 'train.py'], run=Path({str(tmp_path)!r}))"]
    result = subprocess.run(command, env={**os.environ, "RANK": "0", "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "steps-0.jsonl").read_text())["iteration"] == 1
    assert json.loads((output / "model-0.json").read_text())["initial_iteration"] == 0
    assert json.loads((output / "imports-0.json").read_text())["verified_modules"] == 3
