"""CPU fixtures for capture contracts; no Torch/native import or GPU correctness claim."""

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import signal
import struct
import subprocess
import sys
import time

import pytest

from apertus_amd import config, runtime
from apertus_amd.runs import batch_script, edf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/validation/kda_projection_failure_capture.py"
EXPERIMENT = ROOT / "experiments/chonk-kda-projection-failure-capture-icb5a384016a9"
PREPARATION = ROOT / "artifacts/investigations/performance-next/restart-20261113/te-control-capture-preparation"


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("projection_failure_capture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(probe, output):
    argv = ["run", str(output), "--native-provenance", probe.BUILD_PROVENANCE]
    for key, value in probe.BOUNDS.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    return probe.parser().parse_args(argv)


def settings():
    return config.resolve(EXPERIMENT, submit=True, time="00:05:00", reservation="cscs")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16))[0]


def mapped(values, operation):
    return [mapped(v, operation) for v in values] if isinstance(values, list) else operation(values)


class Tensor:
    """Tiny list-backed fixture for the exact operations exercised by this driver."""

    def __init__(self, values, dtype="fp32"):
        self.values, self.dtype = deepcopy(values), dtype
        self.device = "cpu-fixture"

    @property
    def shape(self):
        value, shape = self.values, []
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else 0
        return tuple(shape)

    def numel(self):
        return math.prod(self.shape)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            value = [r[column] for r in self.values[row]] if isinstance(row, slice) else self.values[row][column]
        else:
            value = self.values[key]
        return Tensor(value, self.dtype)

    def op(self, other, operation, dtype=None):
        other = other.values if isinstance(other, Tensor) else other
        def apply(a, b):
            if isinstance(a, list):
                return [apply(v, b[i] if isinstance(b, list) else b) for i, v in enumerate(a)]
            return operation(a, b)
        return Tensor(apply(self.values, other), dtype or self.dtype)

    def __add__(self, other):
        return self.op(other, lambda a, b: f32(a + b))

    __radd__ = __add__

    def __sub__(self, other):
        return self.op(other, lambda a, b: f32(a - b))

    def __mul__(self, other):
        return self.op(other, lambda a, b: f32(a * b))

    __rmul__ = __mul__

    def __gt__(self, other):
        return self.op(other, lambda a, b: a > b, "bool")

    def __ne__(self, other):
        return self.op(other, lambda a, b: a != b, "bool")

    def __matmul__(self, other):
        return Tensor([[f32(sum(a * b for a, b in zip(row, column))) for column in zip(*other.values)]
                       for row in self.values])

    @property
    def T(self):
        return Tensor([list(v) for v in zip(*self.values)], self.dtype)

    def abs(self):
        return Tensor(mapped(self.values, abs), self.dtype)

    def square(self):
        return self * self

    def flatten(self):
        def flatten(value):
            return [x for part in value for x in flatten(part)] if isinstance(value, list) else [value]
        return Tensor(flatten(self.values), self.dtype)

    def sum(self, dtype=None):
        return Tensor(sum(self.flatten().values))

    def max(self):
        return Tensor(max(self.flatten().values))

    def all(self):
        return all(self.flatten().values)

    def any(self, dim):
        assert dim == 1
        return Tensor([any(row) for row in self.values], "bool")

    def nonzero(self):
        return Tensor([[index] for index, value in enumerate(self.values) if value], "int")

    def view(self, dtype):
        if dtype == "i16":
            operation = lambda value: struct.unpack("<I", struct.pack("<f", value))[0] >> 16
        else:
            assert dtype == "i32"
            operation = lambda value: struct.unpack("<I", struct.pack("<f", value))[0]
        return Tensor(mapped(self.values, operation), dtype)

    def to(self, dtype):
        assert dtype == "bf16"
        return Tensor(mapped(self.values, bf16), dtype)

    def float(self):
        return Tensor(mapped(self.values, float))

    def item(self):
        return self.values

    def tolist(self):
        return self.values

    def clone(self):
        return Tensor(self.values, self.dtype)

    def contiguous(self):
        return self

    cpu = contiguous

    def fill_(self, value):
        self.values = mapped(self.values, lambda _: value)

    def add_(self, other):
        self.values = (self + other).values

    def copy_(self, other):
        self.values = deepcopy(other.values)


class Torch:
    """Minimal numerical fixture, not a substitute for vendor numerical validation."""

    bfloat16, float32, float64, int16, int32 = "bf16", "fp32", "fp64", "i16", "i32"

    def __init__(self):
        self.random_calls = []

    def randn_like(self, value):
        self.random_calls.append(value.shape)
        return Tensor(mapped(value.values, lambda _: 0.25), value.dtype)

    def full(self, shape, value, **kwargs):
        return Tensor([[value] * shape[1] for _ in range(shape[0])])

    def isfinite(self, value):
        return Tensor(mapped(value.values, math.isfinite), "bool")


def tensors():
    return {key: Tensor([[0.0625, 0.0625], [0.0625, 0.0625]],
                        "fp32" if key == "main_grad" else "bf16")
            for key in ("X", "W", "dY", "Y", "dX", "main_grad", "temporary")}


def native_fixture(calls, *, target_failure=False):
    def gemm(a, b, **kwargs):
        calls.append(kwargs.copy())
        value = b @ a.T if kwargs["layout"] == "TN" else b @ a if kwargs["layout"] == "NN" else b.T @ a
        if kwargs["out_dtype"] == "bf16":
            value = value.to("bf16")
        if target_failure and len(calls) == 13:
            value.values[0][0] += 0.0625
        if kwargs["beta"]:
            value = kwargs["out"] + value
        kwargs["out"].copy_(value)
    return gemm


def test_fixed_input_environment_and_pure_plan(probe, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("plan subprocess"))
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: pytest.fail("plan write"))
    value = settings()
    assert value["procedure"]["timeout"] == 200
    assert value["image"]["sha256"] == probe.IMAGE
    args = probe.parser().parse_args(value["procedure"]["commands"][0][3:])
    probe.validate_args(args)
    assert args.rank_timeout == 180 and args.phase_timeout == 120 and args.max_coordinates == 512
    assert probe.PHASES == ("default-0",)
    batch = batch_script({**value, "id": "cpu-capture"}, Path("/run"))
    assert "--time=00:05:00" in batch and "--ntasks-per-node=4" in batch and "--exclusive" in batch
    container = edf(value, Path("/run"))
    for expected in ('com.hooks.cxi.enabled = "false"', 'com.hooks.aws_ofi_nccl.enabled = "false"', 'HWLOC_COMPONENTS = "-gl"'):
        assert expected in container
    env = probe.phase_environment({"ROCR_VISIBLE_DEVICES": "0,1,2,3", "SLURM_LOCALID": "3",
                                   "TE_HIPBLASLT_ALGO_LOAD": "old", "TE_HIPBLASLT_ALGO_SAVE": "old",
                                   "PYTORCH_TUNABLEOP_ENABLED": "1", "HIPBLASLT_TUNING_FILE": "old",
                                   "HIPBLASLT_TUNING_OVERRIDE_FILE": "old"})
    assert env["ROCR_VISIBLE_DEVICES"] == "0,1,2,3" and env["SLURM_LOCALID"] == "3"
    assert env["TE_HIPBLASLT_TUNING_RUN_COUNT"] == "0"
    assert not any(k.endswith(("ALGO_LOAD", "ALGO_SAVE")) or k.startswith(("PYTORCH_TUNABLEOP_", "HIPBLASLT_TUNING")) for k in env)


def test_original_pins_gates_native_call_and_rng_expressions_are_unchanged(probe):
    old_text = (PREPARATION / "old-driver.py").read_text()
    new_text = SCRIPT.read_text()
    assert hashlib.sha256(old_text.encode()).hexdigest() == probe.ORIGINAL_DRIVER_SHA256
    old, new = ast.parse(old_text), ast.parse(new_text)
    def function(tree, name):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ("invoke", "metric_failures", "verify_sources", "verify_pinned_file", "verify_build_provenance", "verify_mapped_te_libraries"):
        assert ast.dump(function(old, name)) == ast.dump(function(new, name))
    for attribute in ("manual_seed", "randn", "randn_like", "set_float32_matmul_precision"):
        def calls(tree):
            return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == attribute]
        assert calls(old) == calls(new)
    def matmuls(tree):
        return [ast.dump(n) for n in ast.walk(function(tree, "validate_numerics"))
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult)]
    assert matmuls(old) == matmuls(new)
    for constant in ("IMAGE", "TE_VERSION", "TE_COMMIT", "PY_SOURCES", "NATIVE_SOURCES", "BUILD_INFO_TEXT", "BUILD_INFO_PIN", "TE_BINARY_PINS", "CASES", "WORKSPACE", "TILE"):
        def assignment(tree):
            return next(ast.dump(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == constant for t in n.targets))
        assert assignment(old) == assignment(new)
    assert not any(isinstance(n, ast.FunctionDef) and n.name in ("measure", "parse_table", "consumption_proof") for n in new.body)
    assert "cuda.Event" not in new_text and "replay-" not in new_text


def test_coordinate_mapping_is_lexicographic_bounded_and_signed_word_endian(probe):
    mask = Tensor([[False, True, True], [True, False, True]], "bool")
    assert probe.failure_coordinates(mask, 512, 3) == [[512, 1], [512, 2], [513, 0]]
    assert probe.failure_coordinates(mask, 512, 0) == []
    assert probe.encode_bf16_words([0x3F80, -32768, 0xFFFF]) == b"\x80\x3f\x00\x80\xff\xff"


@pytest.mark.parametrize("failure", [False, True])
def test_original_22_check_prefix_and_target_capture_or_nonreproduction(probe, tmp_path, failure):
    torch, calls, checks = Torch(), [], []
    record = {"capture": {"status": "not-reached"}, "copy_add": {}}
    invoke = lambda: probe.validate_numerics(torch, native_fixture(calls, target_failure=failure),
                                            tensors(), checks, tmp_path, record)
    if failure:
        with pytest.raises(ValueError, match=r"^numerical failure: unfused/independent/1/cumulative:"):
            invoke()
    else:
        invoke()
        assert record["outcome"] == "nonreproduced"
    assert len(checks) == 23 and all(c["status"] == "passed" for c in checks[:22])
    assert sum(c["fp32_gate"] for c in checks[:22]) == 12
    assert checks[-1]["label"] == probe.TARGET
    assert len(calls) == 13 and len(torch.random_calls) == 6
    assert record["copy_add"]["first_copy"]["mismatched_elements"] == 0
    assert record["copy_add"]["second_add"]["mismatched_elements"] == 0
    index = json.loads((tmp_path / "capture-index.json").read_text())
    if failure:
        point = index["coordinates"][0]
        assert point["coordinate"] == [0, 0] and len(point["reference_fp32_bits"]) == 2
        data = (tmp_path / index["binary"]["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == index["binary"]["sha256"]
        assert len(data) == index["binary"]["bytes"] == 16
        assert struct.unpack("<8H", data) == (0x3D80,) * 8
        assert index["status"] == "captured" and point["native_bf16_bits"] != point["reference_bf16_bits"]
        exact = runpy.run_path(str(PREPARATION / "exact_bf16.py"))
        for pair in range(2):
            vectors = [v for v in point["vectors"] if v["pair"] == pair]
            words = [struct.unpack(f"<{v['words']}H", data[v["offset"]:v["offset"] + v["bytes"]]) for v in vectors]
            result = exact["round_bf16"](exact["dot_units"](*words))
            assert result == point["reference_bf16_bits"][pair]
        assert point["native_bf16_bits"][1] != point["reference_bf16_bits"][1]
        assert exact["add_native_bf16"](*point["native_bf16_bits"]) == point["actual_fp32_bits"]
    else:
        assert index["status"] == "nonreproduced" and index["coordinates"] == []
        assert not (tmp_path / "capture-inputs.bin").exists()


def test_cap_overflow_completes_all_tiles_and_is_incomplete(probe, tmp_path, monkeypatch):
    monkeypatch.setitem(probe.BOUNDS, "max_coordinates", 2)
    monkeypatch.setattr(probe, "TILE", 1)
    torch = Torch()
    actual, native = Tensor([[1.0, 1.0], [1.0, 1.0]]), Tensor([[1.0, 1.0], [1.0, 1.0]], "bf16")
    capture = probe.FailureCapture(torch, tmp_path, [(native, native)] * 2, actual, native, actual, {})
    visited, checks = [], []
    def reference(a, b):
        visited.append(a)
        unrounded, rounded = Tensor([[0.0, 0.0]]), Tensor([[0.0, 0.0]], "bf16")
        capture.reference_tiles[:] = [(unrounded, rounded)] * 2
        return unrounded
    with pytest.raises(ValueError, match="numerical failure"):
        probe.numerical_check(torch, actual, reference, probe.TARGET, checks, fp32=False, capture=capture)
    assert visited == [0, 1] and checks[0]["mismatched_elements"] == 4
    index = json.loads((tmp_path / "capture-index.json").read_text())
    assert index["coordinate_cap_exceeded"] and index["status"] == "incomplete"
    assert [p["coordinate"] for p in index["coordinates"]] == [[0, 0], [0, 1]]


@pytest.mark.parametrize("method", ["tile", "finish"])
def test_observer_errors_cannot_replace_original_numerical_failure(probe, tmp_path, monkeypatch, method):
    def broken(*args):
        raise OSError("CPU fixture observer unavailable")
    monkeypatch.setattr(probe.FailureCapture, method, broken)
    record, checks = {"capture": {}, "copy_add": {}}, []
    with pytest.raises(ValueError, match=r"^numerical failure: unfused/independent/1/cumulative"):
        probe.validate_numerics(Torch(), native_fixture([], target_failure=True), tensors(), checks, tmp_path, record)
    assert len(checks) == 23 and record["capture"]["status"] == "incomplete"
    assert "observer unavailable" in record["capture"]["capture_error"]


def test_missing_original_tiles_is_capture_error_not_a_new_oracle(probe, tmp_path):
    capture = probe.FailureCapture(Torch(), tmp_path, [], Tensor([[1.0]]), Tensor([[1.0]], "bf16"), None, {})
    checks = []
    with pytest.raises(ValueError, match="numerical failure"):
        probe.numerical_check(Torch(), Tensor([[1.0]]), lambda a, b: Tensor([[0.0]]),
                              probe.TARGET, checks, fp32=False, capture=capture)
    assert "missing original contribution tiles" in capture.record["capture_error"]


def test_original_metric_values_and_error_text_survive_observer_failure(probe, tmp_path, monkeypatch):
    old = runpy.run_path(str(PREPARATION / "old-driver.py"))
    original, changed = [], []
    with pytest.raises(ValueError) as before:
        old["numerical_check"](Torch(), Tensor([[1.0]]), lambda a, b: Tensor([[0.0]]),
                               probe.TARGET, original, fp32=False)
    capture = probe.FailureCapture(Torch(), tmp_path, [], Tensor([[1.0]]), Tensor([[1.0]], "bf16"), None, {})
    with pytest.raises(ValueError) as after:
        probe.numerical_check(Torch(), Tensor([[1.0]]), lambda a, b: Tensor([[0.0]]),
                              probe.TARGET, changed, fp32=False, capture=capture)
    assert str(before.value) == str(after.value) and original == changed


def test_observer_error_on_nonreproduction_is_not_silently_successful(probe, tmp_path, monkeypatch):
    def broken(*args):
        raise OSError("CPU capture error on passing target")
    monkeypatch.setattr(probe.FailureCapture, "tile", broken)
    record, checks = {"capture": {}, "copy_add": {}}, []
    with pytest.raises(OSError, match="passing target"):
        probe.validate_numerics(Torch(), native_fixture([]), tensors(), checks, tmp_path, record)
    assert len(checks) == 23 and checks[-1]["status"] == "passed"
    assert record["capture"]["status"] == "incomplete" and "outcome" not in record


def test_add_observer_error_preserves_target_numerical_failure(probe, tmp_path, monkeypatch):
    original = probe.copy_add_check
    def broken_add(*args, **kwargs):
        if kwargs.get("previous") is not None:
            raise OSError("CPU add observer error")
        return original(*args, **kwargs)
    monkeypatch.setattr(probe, "copy_add_check", broken_add)
    record, checks = {"capture": {}, "copy_add": {}}, []
    with pytest.raises(ValueError, match="numerical failure: unfused/independent/1/cumulative"):
        probe.validate_numerics(Torch(), native_fixture([], target_failure=True), tensors(), checks, tmp_path, record)
    assert record["capture"]["status"] == "incomplete"
    assert "CPU add observer error" in record["copy_add"]["second_add"]["diagnostic_error"]


def test_copy_invariant_failure_stops_before_second_update_and_captures(probe, tmp_path, monkeypatch):
    original = Tensor.copy_
    native_calls = []
    def damaged_copy(self, other):
        original(self, other)
        if self.dtype == "fp32" and other.dtype == "bf16":
            bits = struct.unpack("<I", struct.pack("<f", self.values[0][0]))[0]
            self.values[0][0] = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
    monkeypatch.setattr(Tensor, "copy_", damaged_copy)
    record, checks = {"capture": {}, "copy_add": {}}, []
    with pytest.raises(ValueError, match="bitwise copy diagnostic failure"):
        probe.validate_numerics(Torch(), native_fixture(native_calls), tensors(), checks, tmp_path, record)
    assert len(checks) == 22 and all(c["status"] == "passed" for c in checks)
    assert len(native_calls) == 12 and record["copy_add"]["first_copy"]["mismatched_elements"] == 1
    index = json.loads((tmp_path / "capture-index.json").read_text())
    assert index["original_check"]["label"] == "bitwise-copy"
    assert len(index["coordinates"][0]["vectors"]) == 2
    assert index["status"] == "copy-add-failure"


def test_add_invariant_failure_is_not_labelled_nonreproduced(probe, tmp_path, monkeypatch):
    values, calls = tensors(), []
    original, additions = Tensor.add_, []
    def damaged_add(self, other):
        original(self, other)
        if self is values["main_grad"]:
            additions.append(1)
            if len(additions) == 4:
                bits = struct.unpack("<I", struct.pack("<f", self.values[0][0]))[0]
                self.values[0][0] = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
    monkeypatch.setattr(Tensor, "add_", damaged_add)
    record, checks = {"capture": {}, "copy_add": {}}, []
    with pytest.raises(ValueError, match="bitwise add diagnostic failure"):
        probe.validate_numerics(Torch(), native_fixture(calls), values, checks, tmp_path, record)
    assert len(checks) == 23 and all(c["status"] == "passed" for c in checks)
    assert record["copy_add"]["second_add"]["mismatched_elements"] == 1
    assert json.loads((tmp_path / "capture-index.json").read_text())["status"] == "copy-add-failure"


def test_byte_bound_and_missing_binary_are_incomplete_not_numerical_pass(probe, tmp_path, monkeypatch):
    monkeypatch.setitem(probe.BOUNDS, "capture_bytes", 1024 * 1024)
    record, checks = {"capture": {}, "copy_add": {}}, []
    with pytest.raises(ValueError, match="numerical failure: unfused/independent/1/cumulative"):
        probe.validate_numerics(Torch(), native_fixture([], target_failure=True), tensors(), checks, tmp_path, record)
    assert record["capture"]["status"] == "incomplete"
    assert "byte bound exceeded" in record["capture"]["capture_error"]


def test_copy_and_add_bitwise_failures_retain_actual_scalar_bits(probe):
    torch = Torch()
    actual, native = Tensor([[1.0000001192092896]]), Tensor([[1.0]], "bf16")
    copy = probe.copy_add_check(torch, actual, native)
    assert copy["mismatched_elements"] == 1
    assert copy["selected"][0]["actual_fp32_bits"] == 0x3F800001
    assert copy["selected"][0]["native_bf16_bits"] == 0x3F80
    add = probe.copy_add_check(torch, Tensor([[3.0]]), native, previous=Tensor([[1.0]]))
    assert add["selected"][0]["expected_fp32_bits"] == 0x40000000
    assert add["selected"][0]["previous_fp32_bits"] == 0x3F800000


def test_earlier_original_failure_stops_before_capture(probe, tmp_path):
    record, checks = {"capture": {"status": "not-reached"}, "copy_add": {}}, []
    def bad_gemm(*args, **kwargs):
        kwargs["out"].fill_(100)
    with pytest.raises(ValueError, match="numerical failure: forward"):
        probe.validate_numerics(Torch(), bad_gemm, tensors(), checks, tmp_path, record)
    assert len(checks) == 1 and record["capture"]["status"] == "not-reached"
    assert not (tmp_path / "capture-index.json").exists()


def test_preflight_source_failure_precedes_vendor_import(probe, tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformer_engine"):
            pytest.fail("vendor import before preflight")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(probe, "PACKAGE", tmp_path / "absent")
    args = arguments(probe, tmp_path)
    args.mode, args.phase = "gpu", "default-0"
    with pytest.raises(ValueError, match="source unavailable"):
        probe.gpu_phase(args)
    result = json.loads((tmp_path / "default-0.json").read_text())
    assert result["status"] == "failed" and result["capture"]["status"] == "not-reached"
    assert result["numerical_acceptance"] is False


def test_nonreproduced_child_completes_only_one_diagnostic_phase(probe, tmp_path, monkeypatch):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    for key, val in dict(SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="4", SLURM_JOB_ID="123").items():
        monkeypatch.setenv(key, val)
    seen = []
    def completed(command, log, **kwargs):
        seen.append(command)
        probe.write(kwargs["cwd"] / "default-0.json", {"status": "diagnostic-complete", "outcome": "nonreproduced"})
    monkeypatch.setattr(runtime, "child", completed)
    output = tmp_path / "output"
    probe.orchestrate(arguments(probe, output), value, output)
    result = json.loads((output / "summary.json").read_text())
    assert len(seen) == 1 and result["status"] == "diagnostic-complete"
    assert result["outcome"] == "nonreproduced" and result["numerical_acceptance"] is False


@pytest.mark.parametrize("unreaped", [False, True])
def test_existing_child_timeout_and_unreaped_cleanup_contract(probe, tmp_path, monkeypatch, unreaped):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    for key, val in dict(SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="4", SLURM_JOB_ID="123").items():
        monkeypatch.setenv(key, val)
    seen = []
    def failed_child(command, log, **kwargs):
        seen.append(command)
        assert kwargs["timeout"] <= 120 and "session" not in kwargs
        assert all(Path(kwargs["env"][key]).is_dir() for key in config.CACHE_PATHS)
        probe.write(kwargs["cwd"] / "default-0.json", {"status": "failed", "error": "original numerical fixture",
                                                     "capture": {"status": "captured"}})
        if unreaped:
            raise runtime.UnreapedChildError("CPU fixture")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(runtime, "child", failed_child)
    output = tmp_path / "output"
    with pytest.raises(runtime.UnreapedChildError if unreaped else subprocess.TimeoutExpired):
        probe.orchestrate(arguments(probe, output), value, output)
    summary = json.loads((output / "summary.json").read_text())
    assert len(seen) == 1 and summary["numerical_acceptance"] is False
    assert summary["gpu_result"]["capture"]["status"] == "captured"
    assert summary["scratch_cleanup"] == ("retained-unreaped" if unreaped else "removed")
    assert Path(summary["scratch"]).exists() == unreaped


def test_term_reaps_existing_child_before_private_cleanup(tmp_path):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    pidfile, output = tmp_path / "pid", tmp_path / "output"
    sleeper = f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(60)"
    code = f'''
import importlib.util, sys
from pathlib import Path
from apertus_amd import runtime
spec = importlib.util.spec_from_file_location("probe", {str(SCRIPT)!r})
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.install_interrupts()
real_child = runtime.child
def cpu_child(command, log, **kwargs):
    return real_child([sys.executable, "-c", {sleeper!r}], log, **kwargs)
runtime.child = cpu_child
argv = ["run", {str(output)!r}, "--native-provenance", probe.BUILD_PROVENANCE]
for key, value in probe.BOUNDS.items():
    argv += ["--" + key.replace("_", "-"), str(value)]
try:
    probe.orchestrate(probe.parser().parse_args(argv), {value!r}, Path({str(output)!r}))
except InterruptedError:
    sys.exit(7)
'''
    env = dict(os.environ, SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="4", SLURM_JOB_ID="123")
    with subprocess.Popen([sys.executable, "-B", "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            deadline = time.monotonic() + 10
            while not pidfile.exists():
                assert process.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
            pid = int(pidfile.read_text())
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 7, (stdout, stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert not runtime.group_running(pid)
    result = json.loads((output / "summary.json").read_text())
    assert result["scratch_cleanup"] == "removed" and not Path(result["scratch"]).exists()
