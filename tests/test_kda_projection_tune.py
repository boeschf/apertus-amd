"""CPU-only fixed-component contracts; no Torch import or GPU correctness claim."""

from copy import deepcopy
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from apertus_amd import config, runtime
from apertus_amd.artifacts import sha256
from apertus_amd.runs import batch_script, edf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/validation/kda_projection_tune.py"
EXPERIMENT = ROOT / "experiments/chonk-kda-projection-tune-icb5a384016a9"


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("kda_projection_tune", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(probe, output):
    argv = ["run", str(output)]
    for key, value in probe.BOUNDS.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    return probe.parser().parse_args(argv)


def settings():
    return config.resolve(EXPERIMENT, submit=True, time="00:20:00", reservation="cscs")


def test_resolver_has_no_write_or_subprocess_and_explicit_frozen_native_command(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("resolver subprocess"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("resolver subprocess"))
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: pytest.fail("resolver write"))
    value = settings()
    assert value["validation"]["sha256"] == sha256(SCRIPT)
    assert value["procedure"]["timeout"] == 940
    assert "source" not in value
    argv = value["procedure"]["commands"][0]
    assert argv[:5] == ["python3", "-u", "/apertus/validation.py", "run", "."]
    assert argv[argv.index("--outputs") + 1] == "11036"
    assert argv[argv.index("--phase-timeout") + 1] == "120"
    assert argv[argv.index("--rank-timeout") + 1] == "900"
    container = edf(value, Path("/run"))
    assert 'com.hooks.cxi.enabled = "false"' in container
    assert 'com.hooks.aws_ofi_nccl.enabled = "false"' in container
    assert 'HWLOC_COMPONENTS = "-gl"' in container
    batch = batch_script({**value, "id": "cpu-contract"}, Path("/run"))
    assert "#SBATCH --exclusive" in batch and "--ntasks-per-node=4" in batch
    assert "--time=00:20:00" in batch and "--reservation=cscs" in batch


def test_fixed_geometry_layout_bounds_and_fresh_process_sequence(probe):
    args = arguments(probe, Path("/output"))
    probe.validate_args(args)
    assert probe.PHASES == ("default-0", "tune", "replay-0", "replay-1", "default-1")
    for phase in probe.PHASES:
        command = probe.child_command(args, phase, Path("/frozen/validation.py"))
        child = probe.parser().parse_args(command[3:])
        probe.validate_args(child)
        assert child.mode == "gpu" and child.phase == phase
    for key, value in probe.BOUNDS.items():
        changed = deepcopy(args)
        setattr(changed, key, value + 1)
        with pytest.raises(ValueError, match="fixed unpadded"):
            probe.validate_args(changed)
    assert [(c["layout"], c["split"]) for c in probe.CASES.values()] == [("TN", False), ("NN", True), ("NT", True)]
    assert probe.remaining_budget(100, 120, 99) == 1
    with pytest.raises(TimeoutError):
        probe.remaining_budget(100, 120, 100)


def test_environment_is_private_te_only_without_visibility_changes(probe):
    original = {"ROCR_VISIBLE_DEVICES": "0,1,2,3", "HIP_VISIBLE_DEVICES": "keep",
                "CUDA_VISIBLE_DEVICES": "also-keep", "SLURM_LOCALID": "2",
                "PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_FILENAME": "old",
                "HIPBLASLT_TUNING_FILE": "old", "HIPBLASLT_TUNING_OVERRIDE_FILE": "old",
                "TE_HIPBLASLT_ALGO_SAVE": "old", "TE_HIPBLASLT_ALGO_LOAD": "old",
                "TE_HIPBLASLT_ALGO_SELECTION": "42"}
    for phase in probe.PHASES:
        env = probe.phase_environment(original, phase, Path("/private/table.csv"))
        assert not any(k.startswith("PYTORCH_TUNABLEOP_") or k.startswith("HIPBLASLT_TUNING") for k in env)
        assert env["TE_HIPBLASLT_ALGO_SELECTION"] == "0"
        assert env.get("TE_HIPBLASLT_ALGO_SAVE") == ("/private/table.csv" if phase == "tune" else None)
        assert env.get("TE_HIPBLASLT_ALGO_LOAD") == ("/private/table.csv" if phase.startswith("replay") else None)
        assert env["TE_HIPBLASLT_TUNING_RUN_COUNT"] == ("5" if phase == "tune" else "0")
        assert all(env[key] == original[key] for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "SLURM_LOCALID"))
    assert original["TE_HIPBLASLT_ALGO_SELECTION"] == "42"


def table_fixture(probe):
    rows = []
    for index, case in enumerate(probe.CASES.values()):
        m, n, k = case["mnk"]
        lda, ldb, ldd = case["ld"]
        rows.append(dict(dev_cap="904", m=str(m), n=str(n), k=str(k), trans_a=case["layout"][0],
                         trans_b=case["layout"][1], type_a="bfloat16", type_b="bfloat16", type_d=case["dtype"],
                         bias_type="-", aux_type="-", lda=str(lda), ldb=str(ldb), ldd=str(ldd), scale_mode="0",
                         epi="-", comp="f32", scale_type="float32", fp4_alpha="0", ws_min="0",
                         ws_max=str(probe.WORKSPACE), algo_id=str(123 + index), aidx=str(index)))
    return rows


def table_text(probe, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=probe.CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def test_cache_consumption_requires_all_exact_keys_and_native_selected_ids(probe):
    rows = probe.parse_table(table_text(probe, table_fixture(probe)))
    log = "Loading autotune results from /private/table.csv\n"
    for name, case in probe.CASES.items():
        m, n, k = case["mnk"]
        row = rows[name]
        log += (f"KDA_PROOF_BEGIN {name}\nm={m} k={k} n={n} transa={case['layout'][0]} transb={case['layout'][1]} A_type=5\n"
                f"[INFO] Use hipBLASLt algo [{row['aidx']}] {row['algo_id']}\nKDA_PROOF_END {name}\n")
    assert set(probe.consumption_proof(log, rows, "/private/table.csv")) == set(probe.CASES)
    for changed in ("Loading autotune results from /private/table.csv\n", log.replace(
            "algo [0] 123", "algo [0] 999"), log + "[WARNING] Cannot get corresponding solution\n",
            log.replace("m=11036 k=3584", "m=11040 k=3584"), log.replace("KDA_PROOF_END wgrad", "missing")):
        with pytest.raises(ValueError):
            probe.consumption_proof(changed, rows, "/private/table.csv")


@pytest.mark.parametrize("field,value", [("dev_cap", "950"), ("type_d", "float16"), ("lda", "1"),
                                         ("ws_max", "1"), ("aidx", "16"), ("epi", "bias")])
def test_table_rejects_mismatched_keys_or_workspace(probe, field, value):
    rows = table_fixture(probe)
    rows[0][field] = value
    with pytest.raises(ValueError):
        probe.parse_table(table_text(probe, rows))


def test_source_absence_or_drift_and_numerical_failure_are_not_success(probe, tmp_path):
    file = tmp_path / "source.py"
    file.write_text("pinned")
    expected = {"source.py": sha256(file)}
    assert probe.verify_sources([tmp_path], expected)
    file.write_text("drift")
    with pytest.raises(ValueError, match="mismatch"):
        probe.verify_sources([tmp_path], expected)
    with pytest.raises(ValueError, match="unavailable"):
        probe.verify_sources([tmp_path], {"missing.cu": "0" * 64})
    metric = dict(finite=True, mismatched_elements=0, max_abs=0.0, rms_error=0.0, reference_rms=1.0, relative_l2=0.0)
    assert probe.metric_failures(metric, True) == []
    assert probe.metric_failures(metric | {"mismatched_elements": 1}, True)
    assert probe.metric_failures(metric | {"rms_error": 0.001}, True)
    assert not probe.metric_failures(metric | {"rms_error": 0.001}, False)
    assert probe.metric_failures(metric | {"relative_l2": math.nan}, False)


def test_native_call_uses_te_operands_and_explicit_precision_not_torch_matmul(probe):
    class Tensor:
        dtype = "bf16"
        def add_(self, value):
            self.added = value
        def copy_(self, value):
            self.copied = value
    tensors = {key: Tensor() for key in ("X", "W", "dY", "Y", "dX", "main_grad", "temporary")}
    tensors["main_grad"].dtype = "fp32"
    calls = []
    def gemm(*args, **kwargs):
        calls.append((args, kwargs))
    for name, case in probe.CASES.items():
        probe.invoke(gemm, tensors, name, int(name == "wgrad"))
        args, kwargs = calls[-1]
        assert args == (tensors[case["A"]], tensors[case["B"]])
        assert kwargs["layout"] == case["layout"] and kwargs["use_split_accumulator"] == case["split"]
        assert kwargs["out"] is tensors[case["out"]]
    probe.invoke(gemm, tensors, "wgrad", 1, control=True)
    assert calls[-1][1]["out"] is tensors["temporary"]
    assert calls[-1][1]["beta"] == 0 and calls[-1][1]["accumulate"] is False
    assert tensors["main_grad"].added is tensors["temporary"]
    probe.invoke(gemm, tensors, "wgrad", 0, control=True)
    assert tensors["main_grad"].copied is tensors["temporary"]


def test_missing_installed_sources_fail_before_any_torch_import(probe, tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformer_engine"):
            pytest.fail("vendor imported before source preflight")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(probe, "PACKAGE", tmp_path / "absent-packages")
    args = arguments(probe, tmp_path)
    args.mode, args.phase = "gpu", "tune"
    with pytest.raises(ValueError, match="source unavailable"):
        probe.gpu_phase(args)
    result = json.loads((tmp_path / "tune.json").read_text())
    assert result["status"] == "failed" and "timings" in result


def test_unreaped_group_keeps_private_scratch(probe, tmp_path, monkeypatch):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    for key, val in dict(SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="4", SLURM_JOB_ID="123").items():
        monkeypatch.setenv(key, val)
    def unreaped(*args, **kwargs):
        raise runtime.UnreapedChildError("CPU fixture: quiescence unconfirmed")
    monkeypatch.setattr(runtime, "child", unreaped)
    out = tmp_path / "output"
    with pytest.raises(runtime.UnreapedChildError):
        probe.orchestrate(arguments(probe, out), value, out)
    result = json.loads((out / "summary.json").read_text())
    assert result["scratch_cleanup"] == "retained-unreaped" and Path(result["scratch"]).is_dir()


def test_phase_timeout_uses_existing_child_and_cleans_only_after_it_returns(probe, tmp_path, monkeypatch):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    for key, val in dict(SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="4", SLURM_JOB_ID="123").items():
        monkeypatch.setenv(key, val)
    seen = []
    def timeout(command, log, **kwargs):
        seen.append(kwargs)
        assert all(Path(kwargs["env"][key]).is_dir() for key in config.CACHE_PATHS)
        assert kwargs["timeout"] <= 120
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(runtime, "child", timeout)
    out = tmp_path / "output"
    with pytest.raises(subprocess.TimeoutExpired):
        probe.orchestrate(arguments(probe, out), value, out)
    result = json.loads((out / "summary.json").read_text())
    assert len(seen) == 1 and result["status"] == "failed"
    assert result["scratch_cleanup"] == "removed" and not Path(result["scratch"]).exists()


def test_term_reaps_nested_cpu_group_before_cache_cleanup(tmp_path):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    pidfile = tmp_path / "child-pid"
    out = tmp_path / "output"
    child_code = f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(60)"
    code = f'''
import importlib.util, os, sys
from pathlib import Path
from apertus_amd import runtime
spec = importlib.util.spec_from_file_location("probe", {str(SCRIPT)!r})
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.install_interrupts()
real_child = runtime.child
def cpu_child(command, log, **kwargs):
    return real_child([sys.executable, "-c", {child_code!r}], log, **kwargs)
runtime.child = cpu_child
argv = ["run", {str(out)!r}]
for key, value in probe.BOUNDS.items():
    argv += ["--" + key.replace("_", "-"), str(value)]
try:
    probe.orchestrate(probe.parser().parse_args(argv), {value!r}, Path({str(out)!r}))
except InterruptedError:
    sys.exit(7)
'''
    env = dict(os.environ, SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="4", SLURM_JOB_ID="123")
    with subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            deadline = time.monotonic() + 10
            while not pidfile.exists():
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.01)
            pid = int(pidfile.read_text())
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 7, (stdout, stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not runtime.group_running(pid)
    result = json.loads((out / "summary.json").read_text())
    assert "InterruptedError" in result["error"] and result["scratch_cleanup"] == "removed"
    assert not Path(result["scratch"]).exists()


@pytest.fixture
def pinned_files(probe, tmp_path, monkeypatch):
    """Use tiny CPU fixture binaries, never load or execute an ELF."""
    package = tmp_path / "packages"
    for relative in probe.PY_SOURCES:
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + " CPU fixture").encode())
        monkeypatch.setitem(probe.PY_SOURCES, relative, sha256(path))
    info = package / "transformer_engine/build_info.txt"
    info.write_text(probe.BUILD_INFO_TEXT)
    for relative in probe.TE_BINARY_PINS:
        path = package / relative
        path.write_bytes((relative + " CPU fixture, not an executable").encode())
        monkeypatch.setitem(probe.TE_BINARY_PINS, relative, {"bytes": path.stat().st_size, "sha256": sha256(path)})
    return package


def forbid_vendor_imports(monkeypatch):
    """Keep provenance fixtures on stdlib-only paths, including failure handling."""
    import builtins

    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformer_engine"):
            pytest.fail("provenance check imported vendor code")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


def test_named_provenance_selection_is_explicit_and_propagates_to_every_child(probe, tmp_path):
    strict = arguments(probe, tmp_path)
    assert strict.native_provenance == "installed-sources"
    argv = settings()["procedure"]["commands"][0]
    selected = probe.parser().parse_args(argv[3:])
    assert selected.native_provenance == "pinned-image-build-provenance"
    for args in (strict, selected):
        for phase in probe.PHASES:
            command = probe.child_command(args, phase, Path("/frozen/validation.py"))
            child = probe.parser().parse_args(command[3:])
            assert child.native_provenance == args.native_provenance
    assert probe.BUILD_INFO_PIN == {"bytes": 98, "sha256": "891ae1c35e89a9d833eb624f09ac017dcf1256870df5c692795c5bfb21de4395"}
    assert probe.TE_BINARY_PINS["transformer_engine/libtransformer_engine.so"] == {
        "bytes": 401104592, "sha256": "140dc626463ae6072121ca66a988c71954268f7b2c27b4d447894a97f2cba69a"}
    assert probe.TE_BINARY_PINS["transformer_engine/transformer_engine_torch.cpython-312-x86_64-linux-gnu.so"] == {
        "bytes": 3438144, "sha256": "1bdeb20270594556bc705e2b9842439925cf70ab03611145d388a92d4cc9f620"}


def test_build_contract_records_absence_without_attestation_and_strict_still_fails(
        probe, pinned_files, tmp_path, monkeypatch):
    forbid_vendor_imports(monkeypatch)
    result = probe.verify_build_provenance(pinned_files, probe.TE_VERSION, workspace=tmp_path / "absent-workspace")
    assert result["contract"] == probe.BUILD_PROVENANCE and result["binary_attestation"] is False
    assert result["source_to_binary_correspondence"] == "not established"
    assert len(result["installed_native_source_observations"]) == 6
    assert all(item["observation"] == "absent" for item in result["installed_native_source_observations"])
    assert result["audited_upstream_native_sources"] == probe.NATIVE_SOURCES
    assert result["build_info"]["sha256"] == probe.BUILD_INFO_PIN["sha256"]
    monkeypatch.setattr(probe, "PACKAGE", pinned_files)
    args = arguments(probe, tmp_path)
    args.mode, args.phase = "gpu", "tune"
    with pytest.raises(ValueError, match="pinned source unavailable"):
        probe.gpu_phase(args)
    record = json.loads((tmp_path / "tune.json").read_text())
    assert record["native_provenance_contract"] == probe.STRICT_PROVENANCE
    assert record["status"] == "failed" and record["binary_attestation"] is False


@pytest.mark.parametrize("damage", ["build_info", "binary_hash", "TE_version"])
def test_build_contract_mismatches_fail_before_vendor_imports(
        probe, pinned_files, tmp_path, monkeypatch, damage):
    forbid_vendor_imports(monkeypatch)
    monkeypatch.setattr(probe, "PACKAGE", pinned_files)
    versions = {"torch": "2.14.0+rocm10.1.0", "transformer-engine": probe.TE_VERSION,
                "triton": "3.8.0+git669b31ac.rocm10.1.0"}
    monkeypatch.setattr(probe.importlib.metadata, "version", versions.__getitem__)
    if damage == "build_info":
        (pinned_files / "transformer_engine/build_info.txt").write_text(probe.BUILD_INFO_TEXT.replace("130099", "000000"))
    elif damage == "binary_hash":
        binary = pinned_files / next(iter(probe.TE_BINARY_PINS))
        original = binary.read_bytes()
        binary.write_bytes(b"!" + original[1:])  # Same length: hash, not size, rejects it.
    else:
        versions["transformer-engine"] = "2.17.1+wrong-commit"
    args = arguments(probe, tmp_path)
    args.mode, args.phase, args.native_provenance = "gpu", "tune", probe.BUILD_PROVENANCE
    with pytest.raises(ValueError, match="mismatch"):
        probe.gpu_phase(args)
    record = json.loads((tmp_path / "tune.json").read_text())
    assert record["native_provenance_contract"] == probe.BUILD_PROVENANCE
    assert record["status"] == "failed" and record["binary_attestation"] is False


def test_actual_mapped_te_paths_sizes_hashes_must_match_not_just_candidates(
        probe, pinned_files, tmp_path, monkeypatch):
    forbid_vendor_imports(monkeypatch)
    result = probe.verify_build_provenance(pinned_files, probe.TE_VERSION, workspace=tmp_path / "absent-workspace")
    libraries = result["te_binary_files"] + [{"path": "/observed/libhipblaslt.so", "bytes": 7, "sha256": "0" * 64}]
    mapped = probe.verify_mapped_te_libraries(libraries, pinned_files)
    assert len(mapped["verified_te_libraries"]) == 2 and mapped["binary_attestation"] is False
    assert "not independently pinned" in mapped["hipblaslt_identity"]
    for key, value in (("path", "/wrong/libtransformer_engine.so"), ("bytes", 1), ("sha256", "0" * 64)):
        changed = deepcopy(libraries)
        changed[0][key] = value
        with pytest.raises(ValueError, match="mapped TE .* mismatch"):
            probe.verify_mapped_te_libraries(changed, pinned_files)


def test_phase_affinity_comparison_rejects_within_phase_and_paired_binding_changes(probe):
    affinity = {"initial": [2, 4], "final": [2, 4], "errors": []}
    assert probe.require_phase_affinity(affinity) == [2, 4]
    assert probe.require_phase_affinity(affinity, [2, 4]) == [2, 4]
    for changed, previous in ((affinity | {"final": [2, 4, 6]}, None),
                              (affinity, [1, 3]), (affinity | {"initial": []}, None),
                              (affinity | {"errors": ["restore failed"]}, None), ({}, None)):
        with pytest.raises(ValueError, match="phase affinity mismatch"):
            probe.require_phase_affinity(changed, previous)


def test_orchestrator_rejects_passed_children_with_different_masks_before_table_use(
        probe, tmp_path, monkeypatch):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    for key, val in dict(SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="4", SLURM_JOB_ID="123").items():
        monkeypatch.setenv(key, val)
    seen = []
    def child(command, log, **kwargs):
        phase = command[command.index("--phase") + 1]
        seen.append(phase)
        mask = [2, 4] if phase == "default-0" else [1, 3]
        probe.write(kwargs["cwd"] / f"{phase}.json", {"status": "passed", "affinity": {
            "initial": mask, "final": mask, "errors": []}})
    monkeypatch.setattr(runtime, "child", child)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="phase affinity mismatch"):
        probe.orchestrate(arguments(probe, output), value, output)
    record = json.loads((output / "summary.json").read_text())
    assert seen == ["default-0", "tune"]
    assert record["status"] == "failed" and record["phases"][-1]["status"] == "failed"
    assert record["scratch_cleanup"] == "removed" and not Path(record["scratch"]).exists()
