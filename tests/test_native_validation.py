"""Explicit native probes use the same frozen validation-script integrity contract."""

from pathlib import Path

import pytest

from apertus_amd.artifacts import sha256
from apertus_amd.config import resolve
from apertus_amd.runs import edf
from apertus_amd import runtime


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/chonk-grouped-linear-probe-icb5a384016a9"


def test_native_validation_is_explicit_hashed_and_mounted():
    settings = resolve(EXPERIMENT, submit=True)
    assert settings["procedure"]["kind"] == "commands"
    assert settings["validation"]["sha256"] == sha256(Path(settings["validation"]["path"]))
    assert "/apertus/validation.py:ro" in edf(settings, Path("/run"))
    assert "/apertus/validation.py" in settings["procedure"]["commands"][0][-1]
    compile(settings["procedure"]["commands"][0][-1], "native-probe-wrapper", "exec")
    assert "source" not in settings
    assert settings["resources"]["nodes"] == 1


def test_native_validation_rejects_mutated_frozen_probe(tmp_path):
    script = tmp_path / "validation.py"
    script.write_text("original\n")
    settings = {"tooling": {}, "validation": {"sha256": sha256(script)}}
    runtime.verify_tooling(tmp_path, settings)
    script.write_text("changed\n")
    with pytest.raises(ValueError, match="validation script changed"):
        runtime.verify_tooling(tmp_path, settings)


def test_native_validation_keeps_compare_run_unsupported():
    with pytest.raises(ValueError, match="commands do not use"):
        resolve(EXPERIMENT, submit=True, compare_run=Path("previous"))
