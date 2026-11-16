"""CPU-only contracts for the installed-file inventory, never vendor execution."""

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from apertus_amd.config import resolve
from apertus_amd.runs import edf, batch_script


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/chonk-te-source-inventory-icb5a384016a9"
SCRIPT = ROOT / "experiments/validation/te_source_inventory.py"


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformer_engine", "triton"}:
            pytest.fail("inventory imported vendor code")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("te_source_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_is_read_only_bounded_and_uses_normal_protected_edf(probe, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("subprocess"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("subprocess"))
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: pytest.fail("plan write"))
    value = resolve(EXPERIMENT, submit=True, memory_gb=2, time="00:05:00", reservation="cscs")
    probe.validate_launch(value, dict(SLURM_NTASKS="4", SLURM_PROCID="2", SLURM_LOCALID="2"))
    assert value["procedure"]["timeout"] == 120
    assert value["validation"]["sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    container = edf(value, Path("/run"))
    assert 'com.hooks.cxi.enabled = "false"' in container and 'com.hooks.aws_ofi_nccl.enabled = "false"' in container
    assert 'HWLOC_COMPONENTS = "-gl"' in container
    batch = batch_script({**value, "id": "inventory-cpu-contract"}, Path("/run"))
    assert "--time=00:05:00" in batch and "--mem=2G" in batch and "--uenv-passthrough=ignore" in batch
    value["resources"]["memory_gb"] = 0
    with pytest.raises(ValueError):
        probe.validate_launch(value, dict(SLURM_NTASKS="4", SLURM_PROCID="2", SLURM_LOCALID="2"))


def test_missing_native_files_versions_record_and_build_info_are_observations(probe, tmp_path):
    package = tmp_path / "package"
    distribution = package / "transformer_engine-2.17.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text("Name: transformer-engine\nVersion: different\n")
    (distribution / "RECORD").write_text("transformer_engine/build_info.txt,,\n")
    te = package / "transformer_engine"
    te.mkdir()
    (te / "build_info.txt").write_text(f"ROCM_VERSION: 10.1\nGPU_TARGETS: gfx942\nCOMMIT_ID: {probe.COMMIT}\n")
    (te / "libtransformer_engine.so").write_bytes(b"\x7fELFfixture-not-executable")
    result = probe.inventory(package, tmp_path / "absent-workspace", max_hash_bytes=1024 * 1024, seconds=10)
    assert len(result["sources"]) == 9 and all(v["observation"] == "absent" for v in result["sources"])
    assert result["distributions"][0]["matches_expected_version"] is False
    assert result["build_info"][0]["matches_expected_COMMIT_ID"] is True
    assert result["elf_candidates"][0]["elf"] is True
    assert result["binary_attestation"] is False and result["runtime_loaded_selection"].startswith("not observed")
    assert not any(result["distributions"][0]["expected_sources_listed_in_RECORD"].values())


def test_hash_mismatch_budget_and_external_symlink_do_not_read_outside_root(probe, tmp_path):
    root = tmp_path / "selected"
    root.mkdir()
    file = root / "file"
    file.write_bytes(b"pinned")
    state = {"remaining": 1024, "deadline": probe.time.monotonic() + 10}
    value = probe.inspect_file(file, root, state, expected="0" * 64)
    assert value["matches_expected"] is False
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not read")
    (root / "link").symlink_to(outside)
    before = state["remaining"]
    assert probe.inspect_file(root / "link", root, state)["observation"] == "external-symlink-not-followed"
    assert state["remaining"] == before
    assert probe.inspect_file(file, root, {"remaining": 1, "deadline": probe.time.monotonic() + 10})["hash"] == "skipped-total-byte-budget"


def test_changed_metadata_and_io_errors_fail_instead_of_becoming_absence(probe, tmp_path, monkeypatch):
    package = tmp_path / "package"
    distribution = package / "transformer_engine-2.17.dist-info"
    distribution.mkdir(parents=True)
    metadata = distribution / "METADATA"
    metadata.write_text("Name: transformer-engine\nVersion: initial\n")
    (distribution / "RECORD").write_text("")
    original = probe.PathDistribution
    def changed(path):
        metadata.write_text("Name: transformer-engine\nVersion: changed\n")
        return original(path)
    monkeypatch.setattr(probe, "PathDistribution", changed)
    # Same-size content changes must be caught even if filesystem stats are stale.
    monkeypatch.setattr(probe, "fingerprint", lambda info: (0, 0, 0, 0, 0))
    partial = {}
    with pytest.raises(ValueError, match="metadata changed"):
        probe.inventory(package, tmp_path / "workspace", max_hash_bytes=1024 * 1024, seconds=10, result=partial)
    assert len(partial["sources"]) == 9 and partial["distributions"][0]["record_count_complete"] is True
    monkeypatch.setattr(Path, "lstat", lambda *a: (_ for _ in ()).throw(PermissionError("CPU fixture")))
    with pytest.raises(PermissionError):
        probe.selected_path(package / "denied", package)


def test_large_wellformed_record_counts_all_rows_with_bounded_relevant_matches(probe, tmp_path):
    package = tmp_path / "package"
    distribution = package / "transformer_engine-2.17.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text(f"Name: transformer-engine\nVersion: {probe.VERSION}\n")
    names = [*probe.EXPECTED, "transformer_engine/build_info.txt",
             *(f"transformer_engine/libpart{i}.so" for i in range(100)), "transformer_engine/module.source"]
    names += [f"transformer_engine/plain{i}.py" for i in range(9000 - len(names))]
    (distribution / "RECORD").write_text("".join(name + ",,\n" for name in names))
    result = probe.inventory(package, tmp_path / "workspace", max_hash_bytes=16 * 1024 * 1024, seconds=10)
    entry = result["distributions"][0]
    assert entry["record_entry_count"] == 9000 and entry["record_count_complete"] is True
    assert entry["record_schema_valid"] is True and entry["record_bad_row_width_count"] == 0
    assert all(entry["expected_sources_listed_in_RECORD"].values())
    assert len(entry["relevant_record_entries"]) == 64
    assert entry["relevant_record_entry_match_count"] == 107
    assert entry["relevant_record_entries_truncated_count"] == 43
    assert result["build_info"][0]["observation"] == "absent"


def test_bad_widths_fail_after_build_info_and_elf_and_main_retains_partial_json(probe, tmp_path, monkeypatch):
    package = tmp_path / "package"
    distribution = package / "transformer_engine-2.17.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text(f"Name: transformer-engine\nVersion: {probe.VERSION}\n")
    name = next(iter(probe.EXPECTED))
    source = package / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"CPU source fixture")
    monkeypatch.setitem(probe.EXPECTED, name, hashlib.sha256(source.read_bytes()).hexdigest())
    (distribution / "RECORD").write_text(name + ",,\n" + "".join(f"bad{i}\n" for i in range(10)))
    te = package / "transformer_engine"
    (te / "build_info.txt").write_text(f"COMMIT_ID: {probe.COMMIT}\n")
    binary = te / "libtransformer_engine.so"
    binary.write_bytes(b"\x7fELFCPU-fixture-not-executed")
    monkeypatch.setattr(probe, "PACKAGE", package)
    monkeypatch.setattr(probe, "WORKSPACE", tmp_path / "workspace")
    value = resolve(EXPERIMENT, submit=True, memory_gb=2, time="00:05:00", reservation="cscs")
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        return json.dumps(value) if str(path) == "/apertus/resolved.json" else original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    for key, val in dict(SLURM_NTASKS="4", SLURM_PROCID="0", SLURM_LOCALID="0").items():
        monkeypatch.setenv(key, val)
    with pytest.raises(ValueError, match="invalid TE RECORD schema"):
        probe.main([str(tmp_path), "--max-hash-bytes", "1073741824", "--deadline-seconds", "110"])
    result = json.loads((tmp_path / "te-source-inventory-0.json").read_text())
    assert result["status"] == "failed" and len(result["sources"]) == 9
    assert result["sources"][0]["matches_expected"] is True
    entry = result["distributions"][0]
    assert entry["record_entry_count"] == 11 and entry["record_count_complete"] is True
    assert entry["record_schema_valid"] is False and entry["record_bad_row_width_count"] == 10
    assert len(entry["record_bad_row_width_diagnostics"]) == 8
    assert entry["record_bad_row_width_diagnostics_truncated_count"] == 2
    assert entry["record_bad_row_width_diagnostics"][0] == {"row_number": 2, "width": 1, "path_prefix": "bad0"}
    assert result["build_info"][0]["matches_expected_COMMIT_ID"] is True
    assert result["elf_candidates"][0]["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert result["hashed_bytes"] > 0 and result["binary_attestation"] is False
