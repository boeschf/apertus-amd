"""CPU checks for isolated Chonk comparison inputs, not GPU numerical validation."""

import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tomllib
from types import ModuleType, SimpleNamespace

import pytest

from apertus_amd.cli import plan
from apertus_amd.config import option


ROOT = Path(__file__).resolve().parents[1]
SUFFIX = "m1072583a9632-icb5a384016a9"
BASE = ROOT / "experiments" / f"moe-chonk-kda-{SUFFIX}"
VARIANTS = ("gaf", "moeact")


def load_probe():
    """Import probe utilities without importing image PyTorch or touching a GPU."""
    spec = importlib.util.spec_from_file_location(
        "grouped_linear_probe", ROOT / "experiments/validation/grouped_linear_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    return probe


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_changes_only_the_named_native_control(variant):
    baseline = tomllib.loads((BASE / "experiment.toml").read_text())
    path = ROOT / "experiments" / f"moe-chonk-kda-{variant}-{SUFFIX}"
    candidate = tomllib.loads((path / "experiment.toml").read_text())
    expected = baseline.copy()
    expected["args"] = list(baseline["args"])
    if variant == "gaf":
        expected["args"].remove("--no-gradient-accumulation-fusion")
    else:
        expected["args"][expected["args"].index("moe")] = "moe_act"
    assert candidate == expected
    assert len(path.name) <= 64
    assert option(candidate["args"], "--train-iters") == "60"
    assert option(candidate["args"], "--save-interval") == "60"
    assert "--no-overlap-p2p-communication" in candidate["args"]
    assert candidate["environment"]["NCCL_P2P_DISABLE"] == "1"
    assert candidate["environment"]["TORCH_COMPILE_DISABLE"] == "0"
    assert candidate["environment"]["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:True"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("launch", [{"submit": True}, {"job_id": "123"}])
def test_variant_plans_64_ranks_without_launching(monkeypatch, variant, launch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("planning launched a process"))
    candidate = plan(ROOT / "experiments" / f"moe-chonk-kda-{variant}-{SUFFIX}", **launch)
    baseline = plan(BASE, **launch)
    assert candidate["resources"]["nodes"] == 16
    assert candidate["resources"]["gpus_per_node"] == 4
    # Current runtime compatibility includes these configured inputs, but NOT args/name.
    for key in ("image", "source", "environment", "prepare"):
        assert candidate[key] == baseline[key]
    assert candidate["validation"]["path"].endswith("experiments/validation/memory_trace.py")


def test_probe_uses_real_expert_projection_widths_and_multiple_accumulation_windows():
    probe = load_probe()
    assert probe.PROJECTIONS == (("fc1", 1792, 3840), ("fc2", 1920, 1792))
    assert probe.LOCAL_EXPERTS == 256 // 16
    assert probe.MICROBATCHES >= 2 and probe.UPDATES >= 2
    assert probe.token_splits(0, 0) == [4096 * 16 * 8 // 256] * 16
    for update in range(probe.UPDATES):
        for microbatch in range(probe.MICROBATCHES):
            splits = probe.token_splits(update, microbatch)
            assert len(splits) == probe.LOCAL_EXPERTS
            assert min(splits) >= 0 and sum(splits) > 0
    assert 0 in probe.token_splits(1, 0)  # Empty expert on first overwrite, not just add.
    assert 1 in probe.token_splits(0, 1)


class ScalarReference:
    """Model only reference accumulation/casting; this is not a TE/GPU simulation."""

    def __init__(self, value=0):
        self.value = value

    def add_(self, other):
        self.value += other.value

    def to(self, dtype):
        assert dtype == "bf16"
        bits = struct.unpack("I", struct.pack("f", self.value))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return ScalarReference(struct.unpack("f", struct.pack("I", bits))[0])

    def float(self):
        return self

    def abs(self):
        return ScalarReference(abs(self.value))


@pytest.mark.parametrize("first", [16.03, 16.08])
def test_plain_reference_rounds_each_microbatch_before_fp32_accumulation(first):
    probe = load_probe()
    exact, rounded, magnitude = ScalarReference(), ScalarReference(), ScalarReference()
    contributions = [ScalarReference(first), ScalarReference(-16)]
    for contribution in contributions:
        probe.accumulate_wgrad_reference(exact, rounded, magnitude, contribution, "bf16")
    expected_plain = sum(value.to("bf16").value for value in contributions)
    assert rounded.value == expected_plain
    assert exact.value == pytest.approx(first - 16)
    assert rounded.value != ScalarReference(exact.value).to("bf16").value
    assert magnitude.value == sum(abs(value.to("bf16").value) for value in contributions)
    # This correctly rounded plain control can fail the OLD exact-sum gate near cancellation.
    assert abs(rounded.value - exact.value) > probe.ATOL + probe.RTOL * abs(exact.value)


def valid_metrics(**changes):
    result = {"finite": True, "max_abs": 0.001, "relative_l2": 0.001,
              "rms_error": 0.001, "reference_rms": 1, "max_tolerance_ratio": 0.1}
    return result | changes


def test_probe_has_independent_elementwise_and_precision_specific_rms_gates():
    probe = load_probe()
    assert probe.RTOL == 0.02 and probe.ATOL == 0.02  # Not a blanket relaxation.
    assert probe.error_failures(valid_metrics(), probe.BF16_RMS_RTOL) == []
    assert probe.error_failures(valid_metrics(), probe.FP32_RMS_RTOL) == [
        "RMS error exceeds precision-specific tolerance"]
    assert probe.error_failures(valid_metrics(max_tolerance_ratio=1.01), probe.BF16_RMS_RTOL) == [
        "elementwise error exceeds propagated tolerance"]
    assert probe.error_failures(valid_metrics(rms_error=0.1), probe.BF16_RMS_RTOL) == [
        "RMS error exceeds precision-specific tolerance"]
    assert probe.error_failures(valid_metrics(finite=False), probe.BF16_RMS_RTOL) == [
        "nonfinite input or reference"]
    for value in (math.inf, math.nan):
        assert probe.error_failures(valid_metrics(rms_error=value), probe.BF16_RMS_RTOL) == [
            "nonfinite error metrics"]


def test_probe_diagnostics_cannot_mask_failed_checks():
    probe = load_probe()
    diagnostic = {"label": "rounding-only", "required": False, "status": "diagnostic"}
    failure = {"label": "fused/wgrad", "required": True, "status": "failed",
               "max_abs": 0.5, "relative_l2": 0.1, "failures": ["RMS error"]}
    probe.require_passed([diagnostic, {"required": True, "status": "passed"}])
    with pytest.raises(AssertionError, match="fused/wgrad.*max_abs.*0.5"):
        probe.require_passed([diagnostic, failure])


@pytest.mark.parametrize("unexpected", [False, True])
def test_probe_retains_incremental_records_on_numerical_or_execution_failure(tmp_path, monkeypatch, unexpected):
    probe = load_probe()
    torch = SimpleNamespace(
        __version__="CPU-test-stub", manual_seed=lambda seed: None,
        cuda=SimpleNamespace(set_device=lambda device: None, get_device_name=lambda: "stub",
                             synchronize=lambda: None),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
    te = ModuleType("transformer_engine")
    te.pytorch = SimpleNamespace(GroupedLinear=ScalarReference)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformer_engine", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te.pytorch)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    emitted = []

    def projection(torch, te, name, inputs, outputs, emit):
        for mode in ("plain", "fused"):
            check = {"label": f"{name}/{mode}", "required": True,
                     "status": "failed" if mode == "plain" else "passed", "max_abs": 0.03}
            emit(check)
            emitted.append(check)
            # Buffered output must already contain this record, before returning/raising.
            lines = (tmp_path / "grouped-linear-probe-0.jsonl").read_text().splitlines()
            assert json.loads(lines[-1]) == check
        if unexpected:
            raise RuntimeError("injected execution failure after emitted checks")

    monkeypatch.setattr(probe, "projection_probe", projection)
    with pytest.raises((AssertionError, RuntimeError)):
        probe.main([str(tmp_path)])
    report = json.loads((tmp_path / "grouped-linear-probe-0.json").read_text())
    assert report["status"] == "failed"
    assert report["checks"] == emitted
    assert any(row["label"].endswith("fused") for row in report["checks"])
    assert report["error"]["type"] == ("RuntimeError" if unexpected else "AssertionError")
    assert report["error"]["message"]
    evidence = (tmp_path / "grouped-linear-probe-0.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.main([str(tmp_path)])
    assert (tmp_path / "grouped-linear-probe-0.json").read_bytes() == evidence
