"""CPU contracts for the isolated Apex probe; no GPU arithmetic correctness claim."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace

import pytest

from apertus_amd import config
from apertus_amd.artifacts import sha256
from apertus_amd.cli import plan
from apertus_amd.runs import edf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/chonk-apex-wgrad-probe-icb5a384016a9"


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location(
        "apex_wgrad_probe", ROOT / "experiments/validation/apex_wgrad_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_experiment_is_read_only_hashed_and_uses_four_local_gpu_ranks(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("plan launched a process"))
    value = plan(EXPERIMENT, submit=True)
    assert value["resources"]["nodes"] == 1
    assert value["resources"]["gpus_per_node"] == 4
    assert value["procedure"]["timeout"] == 600
    assert value["procedure"]["kind"] == "commands"
    assert value["validation"]["sha256"] == sha256(Path(value["validation"]["path"]))
    assert value["validation"]["path"].endswith("experiments/validation/apex_wgrad_probe.py")
    assert value["image"]["sha256"] == "cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0"
    assert "source" not in value
    container = edf(value, Path("/run"))
    assert "/apertus/validation.py:ro" in container
    assert 'com.hooks.cxi.enabled = "false"' in container
    assert 'com.hooks.aws_ofi_nccl.enabled = "false"' in container
    assert 'HWLOC_COMPONENTS = "-gl"' in container
    assert value["environment"]["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:True"


def test_wrapper_owns_private_caches_without_replacing_gpu_visibility(tmp_path, monkeypatch):
    import os

    settings = tomllib.loads((EXPERIMENT / "experiment.toml").read_text())
    wrapper = settings["procedure"]["commands"][0][-1]
    original_temporary = tempfile.TemporaryDirectory
    roots = []

    def temporary(*, prefix, dir):
        assert dir == "/dev/shm" and prefix == "apertus-apex-probe-123-2-"
        result = original_temporary(prefix=prefix, dir=tmp_path)
        roots.append(Path(result.name))
        return result

    monkeypatch.setattr(tempfile, "TemporaryDirectory", temporary)
    monkeypatch.setattr(sys, "argv", ["test-wrapper"])
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_LOCALID", "2")
    visibility = {"ROCR_VISIBLE_DEVICES": "0,1,2,3", "CUDA_VISIBLE_DEVICES": "keep-cuda",
                  "HIP_VISIBLE_DEVICES": "keep-hip"}
    for key, value in visibility.items():
        monkeypatch.setenv(key, value)
    for key in config.CACHE_PATHS:
        monkeypatch.setenv(key, "old-value")

    def execute(path, run_name):
        assert path == "/apertus/validation.py" and run_name == "__main__"
        assert sys.argv == [path, "."]
        for key in config.CACHE_PATHS:
            assert Path(os.environ[key]).is_dir()
            assert Path(os.environ[key]).parent == roots[0]
        assert {key: os.environ[key] for key in visibility} == visibility

    monkeypatch.setattr(runpy, "run_path", execute)
    exec(compile(wrapper, "apex-probe-wrapper", "exec"), {})
    assert roots and not roots[0].exists()


def test_exact_shape_call_limit_and_reference_tile_bound(probe):
    assert probe.SHAPES == (("small-diagnostic", 128, 3584, 512),
                            ("full-lmhead", 4096, 3584, 200064))
    assert probe.SHAPES[1][3] == ((200063 + 127) // 128) * 128
    assert len(probe.COEFFICIENTS) == 3
    assert probe.INITIAL_ACCUMULATOR != 0
    assert probe.COEFFICIENTS == (1.0, -1.0, 0.25)
    ranges = list(probe.tile_ranges(200064))
    assert ranges[0] == (0, 4096) and ranges[-1][1] == 200064
    assert all(end - start <= probe.VOCAB_TILE for start, end in ranges)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert sum(end - start for start, end in ranges) == 200064


def test_source_hashes_are_pinned_and_drift_is_rejected(probe, tmp_path):
    assert len(probe.EXPECTED_SOURCES) == 4
    assert all(re.fullmatch("[0-9a-f]{64}", value) for value in probe.EXPECTED_SOURCES.values())
    path = tmp_path / "source.hip"
    path.write_text("original")
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    assert probe.verify_sources(tmp_path, expected) == expected
    path.write_text("changed")
    with pytest.raises(ValueError, match="source mismatch"):
        probe.verify_sources(tmp_path, expected)


class NativeOperation:
    """Supply only the pybind-style API contract for CPU identity/timing tests."""

    __doc__ = ("wgrad_gemm_accum_fp32(arg0: torch.Tensor, arg1: torch.Tensor, "
               "arg2: torch.Tensor) -> None\n\nwgrad gemm accum in fp32")

    def __call__(self, *args):
        return None


def test_actual_signature_contract_is_not_guessed(probe):
    assert probe.verify_signature(NativeOperation()).endswith("-> None")
    for doc in (None, "wgrad_gemm_accum_fp32(arg0: torch.Tensor) -> None",
                NativeOperation.__doc__.replace("-> None", "-> torch.Tensor")):
        operation = NativeOperation()
        operation.__doc__ = doc
        with pytest.raises(ValueError, match="unexpected installed Apex signature"):
            probe.verify_signature(operation)


def test_lazy_load_records_source_and_selected_private_binary(probe, tmp_path, monkeypatch):
    private = tmp_path / "cache"
    private.mkdir()
    binary = private / "fused_weight_gradient_mlp_cuda.so"
    binary.write_bytes(b"test-binary-not-a-real-extension")
    ninja = private / "build.ninja"
    ninja.write_text(str(probe.SITE_PACKAGES / "apex/csrc/megatron/fused_weight_gradient_dense_hip.hip"))
    operation = NativeOperation()
    native = SimpleNamespace(__file__=str(binary), wgrad_gemm_accum_fp32=operation)
    evidence = probe.Evidence(tmp_path / "output", 0)

    class Shim:
        def __init__(self):
            self._loaded_module = native

        @property
        def wgrad_gemm_accum_fp32(self):
            assert evidence.record["last_event"]["phase"] == "apex-native-jit-load"
            assert evidence.record["last_event"]["kind"] == "phase-begin"
            return operation

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(private))
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: Shim())
    try:
        assert probe.load_operation(evidence) is operation
        assert evidence.record["api"]["binary_sha256"] == sha256(binary)
        assert evidence.record["api"]["build_ninja_sha256"] == sha256(ninja)
    finally:
        evidence.close()


class ScalarMatrix:
    """Test reference accumulation order only; not a substitute for GPU GEMM testing."""

    shape = (1, 1)

    def __init__(self, value):
        self.value = value

    def float(self):
        return self

    def __truediv__(self, value):
        return ScalarMatrix(self.value / value)

    @property
    def T(self):
        return self

    def addmm_(self, left, right, *, alpha):
        self.value += alpha * left.value * right.value


@pytest.mark.parametrize("count", [1, 2, 3])
def test_tiled_reference_retains_nonzero_initial_value_and_additive_cancellation(probe, count):
    coefficients = probe.COEFFICIENTS[:count]
    torch = SimpleNamespace(float32="fp32", full=lambda shape, value, **kwargs: ScalarMatrix(value))
    x, base_dy = 2.0, 3.0
    result = probe.reference_tile(torch, ScalarMatrix(x), ScalarMatrix(base_dy * coefficients[-1]), coefficients)
    assert result.value == pytest.approx(probe.INITIAL_ACCUMULATOR + x * base_dy * sum(coefficients))
    if count == 2:
        assert result.value == probe.INITIAL_ACCUMULATOR  # Overwrite or missing beta=1 fails this case.


def test_fixed_fp32_error_gates_reject_localized_errors_and_nonfinite_norms(probe):
    metrics = {"finite": True, "max_abs": 1e-5, "relative_l2": 1e-6,
               "rms_error": 1e-6, "reference_rms": 1, "mismatched_elements": 0}
    assert probe.metric_failures(metrics) == []
    assert probe.metric_failures(metrics | {"mismatched_elements": 1}) == ["elementwise FP32 error gate"]
    assert probe.metric_failures(metrics | {"rms_error": 0.01}) == ["RMS FP32 error gate"]
    for field in ("max_abs", "relative_l2", "rms_error", "reference_rms"):
        assert probe.metric_failures(metrics | {field: math.nan}) == ["nonfinite tensor or error metric"]
    assert probe.metric_failures({"finite": False}) == ["nonfinite tensor or error metric"]


def test_call_evidence_precedes_native_api_and_separates_gpu_wait(probe, tmp_path):
    evidence = probe.Evidence(tmp_path, 2)
    order = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            order.append("event-record")

        def synchronize(self):
            assert evidence.record["last_event"]["kind"] == "gpu-wait-begin"
            order.append("event-sync")

        def elapsed_time(self, end):
            return 12.5

    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: order.append("pre-sync")))
    inputs = (object(), object(), object())

    def operation(*actual):
        assert actual == inputs
        assert json.loads(evidence.path.read_text())["last_event"]["kind"] == "call-begin"
        order.append("native-call")

    try:
        probe.timed_call(torch, operation, *inputs, "full-lmhead/microbatch0", evidence)
        assert order == ["pre-sync", "event-record", "native-call", "event-record", "event-sync"]
        timing = evidence.record["timings"][0]
        assert timing["gpu_event_ms"] == 12.5
        assert timing["host_call_seconds"] >= 0
        assert timing["synchronized_wall_seconds"] >= timing["host_call_seconds"]
    finally:
        evidence.close()


def test_shape_probe_uses_only_three_full_calls_and_one_full_fp32_accumulator(probe, tmp_path, monkeypatch):
    allocations, calls, references = [], [], []

    class Tensor:
        device = "cuda:3"

        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype
            self.scale = 1.0

        def mul_(self, value):
            self.scale *= value

        def is_contiguous(self):
            return True

        def stride(self):
            return (self.shape[1], 1)

    def allocate(shape, dtype):
        allocations.append((shape, dtype))
        return Tensor(shape, dtype)

    torch = SimpleNamespace(
        bfloat16="bf16", float32="fp32", cuda=SimpleNamespace(synchronize=lambda: None),
        randn=lambda *shape, dtype, **kwargs: allocate(shape, dtype),
        full=lambda shape, initial, dtype, **kwargs: allocate(shape, dtype))

    def call(torch, operation, x, dy, accumulator, label, evidence):
        calls.append((x.shape, dy.shape, accumulator.shape, dy.scale, label))

    monkeypatch.setattr(probe, "timed_call", call)
    monkeypatch.setattr(probe, "check_accumulator", lambda torch, x, dy, accumulator, coefficients,
                        label, evidence: references.append(coefficients))
    evidence = probe.Evidence(tmp_path, 3)
    try:
        probe.shape_probe(torch, None, probe.SHAPES[1], evidence)
        assert len(calls) == 3
        assert [call[3] for call in calls] == [0.25, -0.25, 0.0625]
        assert all(call[:3] == ((4096, 3584), (4096, 200064), (200064, 3584)) for call in calls)
        assert references == [probe.COEFFICIENTS[:count] for count in (1, 2, 3)]
        assert allocations == [((4096, 3584), "bf16"), ((4096, 200064), "bf16"),
                               ((200064, 3584), "fp32")]
    finally:
        evidence.close()


def test_main_preserves_slurm_local_rank_and_keeps_diagnostic_shape_separate(probe, tmp_path, monkeypatch):
    selected, cases = [], []
    torch = SimpleNamespace(
        __version__="CPU-contract-stub", manual_seed=lambda seed: None,
        set_float32_matmul_precision=lambda precision: None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
        cuda=SimpleNamespace(set_device=lambda device: selected.append(device), get_device_name=lambda: "stub"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("SLURM_PROCID", "15")
    monkeypatch.setenv("SLURM_LOCALID", "3")
    monkeypatch.setattr(probe, "verify_sources", lambda: {})
    monkeypatch.setattr(probe, "load_operation", lambda evidence: None)
    monkeypatch.setattr(probe, "shape_probe", lambda torch, operation, case, evidence: cases.append(case))
    probe.main([str(tmp_path)])
    assert selected == [3]  # Never choose rank 15, device 0, or rewrite visibility.
    assert cases == list(probe.SHAPES)
    report = json.loads((tmp_path / "apex-wgrad-probe-15.json").read_text())
    assert report["local_rank"] == 3
    assert "small-diagnostic" in report["shapes"][0]


def test_exception_records_incremental_phase_and_does_not_overwrite_old_run(probe, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_LOCALID", "3")

    def failure():
        assert json.loads((tmp_path / "apex-wgrad-probe-3.json").read_text())["last_event"]["phase"] == "source-verification"
        raise RuntimeError("injected pre-JIT failure")

    monkeypatch.setattr(probe, "verify_sources", failure)
    with pytest.raises(RuntimeError, match="injected"):
        probe.main([str(tmp_path)])
    report_path = tmp_path / "apex-wgrad-probe-3.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["local_rank"] == 3
    assert report["error"]["type"] == "RuntimeError"
    events = [json.loads(line) for line in (tmp_path / "apex-wgrad-probe-3.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events] == ["phase-begin", "phase-failed", "failed"]
    evidence = report_path.read_bytes()
    with pytest.raises(FileExistsError):
        probe.main([str(tmp_path)])
    assert report_path.read_bytes() == evidence
