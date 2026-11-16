"""CPU protocol/failure tests; these do not execute the native GPU component."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tomllib
from types import SimpleNamespace

import pytest

from apertus_amd.artifacts import sha256
from apertus_amd.config import resolve
from apertus_amd.runs import edf, batch_script


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "router-histogram-probe-m"
EXPERIMENTS = [ROOT / f"experiments/{PREFIX}{source}-icb5a384016a9"
               for source in ("1072583a9632", "e242f3c3c113")]


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("router_tensor_probe", ROOT / "experiments/validation/router_tensor_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=EXPERIMENTS)
def settings(request):
    return resolve(request.param, submit=True, reservation="cscs", time="00:15:00", cache_mode="cold")


def test_embedded_row_is_full_f32_checksum_verified_without_external_reads(probe, monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda path: pytest.fail("row must not read external artifacts"))
    row = probe.captured_row()
    assert len(row) == 256
    assert probe.hashlib.sha256(struct.pack("<256f", *row)).hexdigest() == probe.ROW_SHA
    assert probe.ROW_PROVENANCE["job"] == 738351 and probe.ROW_PROVENANCE["token"] == 1168
    assert row[44] != row[228]  # No CPU assertion about actual GPU sigmoid/tie order.
    monkeypatch.setattr(probe, "CAPTURED_ROW", "AAAA")
    with pytest.raises(ValueError, match="checksum"):
        probe.captured_row()


def test_paired_experiments_change_only_source_and_use_frozen_standard_protocol(probe, settings):
    probe.validate(settings)
    values = [tomllib.loads((path / "experiment.toml").read_text()) for path in EXPERIMENTS]
    assert [value["source"].pop("commit") for value in values] == [probe.BASE, probe.FIX]
    assert values[0] == values[1]
    assert all(len(path.name) <= 64 for path in EXPERIMENTS)
    container = edf(settings, Path("/run"))
    assert "/apertus/validation.py:ro" in container
    assert 'com.hooks.cxi.enabled = "false"' in container
    assert 'com.hooks.aws_ofi_nccl.enabled = "false"' in container
    assert 'HWLOC_COMPONENTS = "-gl"' in container
    batch = batch_script({**settings, "id": "cpu-protocol"}, Path("/run"))
    assert "#SBATCH --exclusive" in batch and "--ntasks-per-node=4" in batch
    assert "--nodes=1" in batch and "--time=00:15:00" in batch and "--reservation=cscs" in batch
    assert settings["cache"]["mode"] == "cold"


@pytest.mark.parametrize("mutation", [
    lambda s: s["source"].update(commit="0" * 40),
    lambda s: s["image"].update(sha256="0" * 64),
    lambda s: s["procedure"].update(kind="restart"),
    lambda s: s.update(compare_run="other-run"),
    lambda s: s["resources"].update(nodes=4),
    lambda s: s["resources"].update(gpus_per_node=1),
    lambda s: s["args"].extend(["--hidden-size", "1024"]),
    lambda s: s["args"].extend(["--expert-model-parallel-size", "4"]),
    lambda s: s["args"].extend(["--global-batch-size", "4"]),
    lambda s: s["args"].append("--moe-router-fusion"),
    lambda s: s["args"].extend(["--moe-expert-capacity-factor", "1"]),
    lambda s: s["environment"].update(TORCH_COMPILE_DISABLE="1"),
])
def test_scope_changes_fail_closed(probe, settings, mutation):
    mutation(settings)
    with pytest.raises(ValueError):
        probe.validate(settings)


def fake_evidence(probe, settings, run):
    (run / "output").mkdir()
    (run / "resolved.json").write_text(json.dumps(settings))
    primary, compatibility = probe.cases()
    for rank in range(4):
        record = {"status": "passed", "scope": probe.SCOPE, "rank": rank,
                  "source": settings["source"]["commit"], "image": probe.IMAGE,
                  "probe_sha256": settings["validation"]["sha256"], "row_provenance": probe.ROW_PROVENANCE,
                  "checks": [{"label": label, "passed": True} for label in sorted(probe.required_labels())],
                  "primary": {name: {"calls": [{"microbatch": mb, "metric_count": mb + 1, "grad_mode_equal": True}
                                                for mb in range(2)], "captured_boundary_gap": 0,
                                     "non_tied_boundary_gap": 0.1} for name, *_ in primary},
                  "compatibility": {f"{name}/grad{int(grad)}": {} for name, *_, training in compatibility
                                    for grad in ((False, True) if training else (False,))}}
        (run / f"output/router-tensor-{rank}.json").write_text(json.dumps(record))
        (run / f"output/imports-{rank}.json").write_text(json.dumps({"verified_modules": 100}))


def test_all_rank_complete_gate_distinguishes_baseline_diagnostic(probe, settings, tmp_path):
    fake_evidence(probe, settings, tmp_path)
    probe.check(tmp_path)
    result = json.loads((tmp_path / "output/checks.json").read_text())
    assert result["status"] == "passed" and result["ranks"] == 4
    expected = "fixed-source regression" if settings["source"]["commit"] == probe.FIX else "baseline diagnostic control"
    assert result["acceptance_kind"] == expected
    assert result["cross_source_compatibility"] == "requires separate paired host review"


@pytest.mark.parametrize("damage", ["missing-rank", "failed", "missing-check", "duplicate-check", "imports",
                                     "source", "validator", "row", "matrix", "microbatch", "tie", "count"])
def test_incomplete_or_failed_native_evidence_cannot_publish(probe, settings, tmp_path, damage):
    fake_evidence(probe, settings, tmp_path)
    path = tmp_path / "output/router-tensor-3.json"
    value = json.loads(path.read_text())
    if damage == "missing-rank":
        path.unlink()
    elif damage == "imports":
        (tmp_path / "output/imports-3.json").write_text('{"verified_modules": 0}')
    else:
        case = value["primary"]["histogram/sigmoid/pad0"]
        if damage == "failed":
            value["checks"][0]["passed"] = False
        elif damage == "missing-check":
            value["checks"].pop()
        elif damage == "duplicate-check":
            value["checks"].append(value["checks"][0])
        elif damage == "source":
            value["source"] = "0" * 40
        elif damage == "validator":
            value["probe_sha256"] = "0" * 64
        elif damage == "row":
            value["row_provenance"]["row_sha256"] = "0" * 64
        elif damage == "matrix":
            value["compatibility"].pop(next(iter(value["compatibility"])))
        elif damage == "microbatch":
            case["calls"].pop()
        elif damage == "tie":
            case["captured_boundary_gap"] = 0.001
        elif damage == "count":
            case["calls"][1]["metric_count"] = 3
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        probe.check(tmp_path)
    assert json.loads((tmp_path / "output/checks.json").read_text())["status"] == "failed"


def test_fixed_source_cannot_claim_pass_with_failed_grad_mode_equality(probe, tmp_path):
    settings = resolve(EXPERIMENTS[1], submit=True)
    fake_evidence(probe, settings, tmp_path)
    path = tmp_path / "output/router-tensor-0.json"
    value = json.loads(path.read_text())
    value["primary"]["histogram/sigmoid/pad0"]["calls"][0]["grad_mode_equal"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="captured-row"):
        probe.check(tmp_path)


@pytest.mark.parametrize("failure", [None, "initialize", "compute", "partial", "final-imports"])
def test_train_freezes_source_verifies_before_compute_and_retains_failures(
        probe, settings, tmp_path, monkeypatch, failure):
    source = tmp_path / "native"
    (source / Path(probe.ROUTER).parent).mkdir(parents=True)
    (source / "pretrain_gpt.py").write_text("raise AssertionError('model entrypoint must not execute')\n")
    (source / probe.ROUTER).write_text("# native stub for CPU protocol only\n")
    monkeypatch.setitem(probe.ROUTER_SHA, settings["source"]["commit"], sha256(source / probe.ROUTER))
    run = tmp_path / "run"
    (run / "source").mkdir(parents=True)
    output = run / "output"
    output.mkdir()
    (run / "resolved.json").write_text(json.dumps(settings))
    (run / "source/manifest.json").write_text(json.dumps({"selection": {"commit": settings["source"]["commit"]},
        "files": {"pretrain_gpt.py": sha256(source / "pretrain_gpt.py")}}))
    calls = []

    def verify(*args):
        calls.append("verify")
        if failure == "final-imports" and calls.count("verify") == 3:
            raise ValueError("bad final imports")
        return 4

    def initialize():
        calls.append("initialize")
        if failure == "initialize":
            raise ValueError("bad initialization")

    def compute(emit, record, **kwargs):
        calls.append("compute")
        if failure == "compute":
            raise ValueError("bad native compute")
        for label in sorted(probe.required_labels()):
            if failure != "partial":
                emit(label, True)

    monkeypatch.setattr(probe, "verify_imports", verify)
    monkeypatch.setattr(probe, "run_probe", compute)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", sys.argv.copy())
    monkeypatch.setitem(sys.modules, "megatron", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "megatron.training", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "megatron.training.initialize", SimpleNamespace(initialize_megatron=initialize))
    if failure:
        with pytest.raises((ValueError, AssertionError)):
            probe.train(output, source, "pretrain_gpt.py", [], run=run)
    else:
        probe.train(output, source, "pretrain_gpt.py", [], run=run)
    assert calls[0] == calls[-1] == "verify"
    if "compute" in calls:
        assert calls[:4] == ["verify", "initialize", "verify", "compute"]
    record = json.loads((output / "router-tensor-0.json").read_text())
    assert record["status"] == ("failed" if failure else "passed")


def test_source_identity_rejected_before_native_imports(probe, settings, tmp_path, monkeypatch):
    (tmp_path / "resolved.json").write_text(json.dumps(settings))
    (tmp_path / "source").mkdir()
    (tmp_path / "source/manifest.json").write_text(json.dumps({"selection": {"commit": "0" * 40}}))
    monkeypatch.setattr(probe, "verify_imports", lambda *args: pytest.fail("identity must fail before imports"))
    with pytest.raises(ValueError, match="identity mismatch"):
        probe.train(tmp_path, tmp_path, "pretrain_gpt.py", [], run=tmp_path)
