"""Probe the installed native Apex LM-head wgrad, not distributed training correctness.

The small diagnostic shape is not Chonk geometry. Only the full-lmhead case uses
the exact TP=1 padded-vocabulary shape. References are tiled, and there are only
three native calls per shape. No vendor source, allocator, or workspace fix is
installed; the audited Apex workspace-lifetime hazard remains present.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import time


SHAPES = (("small-diagnostic", 128, 3584, 512), ("full-lmhead", 4096, 3584, 200064))
COEFFICIENTS = (1.0, -1.0, 0.25)
INITIAL_ACCUMULATOR = 0.125
VOCAB_TILE = 4096
ELEMENT_RTOL = 1e-4
ELEMENT_ATOL = 1e-4
RMS_RTOL = 1e-4
RMS_ATOL = 1e-6
SITE_PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")
EXPECTED_SOURCES = {
    "fused_weight_gradient_mlp_cuda.py": "f3a49f8a76f9eaa86536215c4ae1b2ff3919d30a82df851e59214c86eee45e5d",
    "apex/op_builder/fused_weight_gradient_mlp.py": "913e10bc7a0e4fb20c5a737a44f5dec5917107217aa001579a48095c3a5719e4",
    "apex/csrc/megatron/fused_weight_gradient_dense.cpp": "ac2b9a4fee5eb4544fadb9ae9a5075dd93f7f6e18a864ddc225ac30b442a86a6",
    "apex/csrc/megatron/fused_weight_gradient_dense_hip.hip": "8638d44a7cb2625a3f1855e9d4fee9974aa1256e1f838c7deb9a6b1afb1fc856",
}


def checksum(path: Path) -> str:
    """Hash an explicitly selected small source, probe, or extension file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Evidence:
    """Retain flushed phase events and an atomic live summary without replacing old runs."""

    def __init__(self, output: Path, rank: int):
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / f"apex-wgrad-probe-{rank}.json"
        self.temporary = output / f"apex-wgrad-probe-{rank}.json.tmp"
        self.started = time.monotonic()
        self.record = {"status": "running", "rank": rank, "timings": [], "checks": [],
                       "scope": "installed native Apex additive wgrad component only"}
        with self.path.open("x") as stream:
            stream.write(json.dumps(self.record) + "\n")
        self.stream = (output / f"apex-wgrad-probe-{rank}.jsonl").open("x", buffering=1)

    def emit(self, kind: str, **values) -> None:
        """Flush an event before work starts; keep the last phase valid even on hard timeout."""
        event = {"kind": kind, "unix_time": time.time(),
                 "elapsed_seconds": time.monotonic() - self.started, **values}
        self.stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.record["last_event"] = event
        if kind in ("timing", "check-summary"):
            self.record["timings" if kind == "timing" else "checks"].append(event)
        self.temporary.write_text(json.dumps(self.record, allow_nan=False) + "\n")
        os.replace(self.temporary, self.path)

    def close(self) -> None:
        """Close only this process's evidence stream, not its caller-owned caches."""
        self.stream.close()


@contextmanager
def phase(evidence: Evidence, name: str, **values):
    """Record both successful and exceptional phase duration without hiding original errors."""
    evidence.emit("phase-begin", phase=name, **values)
    started = time.monotonic()
    try:
        yield
    except BaseException as error:
        evidence.emit("phase-failed", phase=name, wall_seconds=time.monotonic() - started,
                      error_type=type(error).__name__, error=str(error))
        raise
    evidence.emit("phase-end", phase=name, wall_seconds=time.monotonic() - started)


def verify_sources(root=SITE_PACKAGES, expected=EXPECTED_SOURCES) -> dict:
    """Reject source/loader drift from the audited cb5a image before loading the API."""
    result = {}
    for relative, wanted in expected.items():
        path = root / relative
        actual = checksum(path)
        if actual != wanted:
            raise ValueError(f"Apex source mismatch: {path}: {actual} != {wanted}")
        result[relative] = actual
    return result


def verify_signature(operation) -> str:
    """Check the actual pybind11 API's three Tensor arguments and void return signature."""
    signature = (getattr(operation, "__doc__", None) or "").splitlines()
    first = signature[0] if signature else ""
    wanted = (r"wgrad_gemm_accum_fp32\(arg0: torch\.Tensor, arg1: torch\.Tensor, "
              r"arg2: torch\.Tensor\) -> None")
    if not callable(operation) or re.fullmatch(wanted, first) is None:
        raise ValueError(f"unexpected installed Apex signature: {first!r}")
    return first


def load_operation(evidence: Evidence):
    """Separate shim import from lazy JIT/load and verify the selected private binary."""
    with phase(evidence, "apex-shim-import"):
        shim = importlib.import_module("fused_weight_gradient_mlp_cuda")
    with phase(evidence, "apex-native-jit-load"):
        operation = shim.wgrad_gemm_accum_fp32
    with phase(evidence, "apex-api-verification"):
        signature = verify_signature(operation)
        module = vars(shim)["_loaded_module"]
        if module.wgrad_gemm_accum_fp32 is not operation:
            raise ValueError("Apex loader and native module select different operations")
        binary = Path(module.__file__).resolve()
        private = Path(os.environ["TORCH_EXTENSIONS_DIR"]).resolve()
        if not binary.is_relative_to(private):
            raise ValueError(f"Apex extension is outside the private cache: {binary}")
        ninja = binary.parent / "build.ninja"
        if str(SITE_PACKAGES / "apex/csrc/megatron/fused_weight_gradient_dense_hip.hip") not in ninja.read_text():
            raise ValueError("Apex build does not name the audited HIP source")
        evidence.record["api"] = {"signature": signature, "signature_source": "pybind11 docstring",
                                  "module": operation.__module__, "binary": str(binary),
                                  "binary_sha256": checksum(binary), "build_ninja_sha256": checksum(ninja)}
    return operation


def tile_ranges(vocabulary: int):
    """Visit every output row once without constructing a full reference weight gradient."""
    for start in range(0, vocabulary, VOCAB_TILE):
        yield start, min(start + VOCAB_TILE, vocabulary)


def reference_tile(torch, x_float, dy_slice, coefficients):
    """Independently accumulate FP32 GEMMs from exactly power-of-two-related BF16 inputs."""
    # The actual dy is changed in place between calls. These coefficients are exact
    # powers of two, so recovering the base tile does not add BF16 rounding here.
    base = dy_slice.float() / coefficients[-1]
    reference = torch.full((dy_slice.shape[1], x_float.shape[1]), INITIAL_ACCUMULATOR,
                           device="cuda", dtype=torch.float32)
    for coefficient in coefficients:
        reference.addmm_(base.T, x_float, alpha=coefficient)
    return reference


def metric_failures(metrics: dict) -> list[str]:
    """Apply fixed elementwise and FP32 normwise gates, not BF16-output tolerances."""
    if not metrics["finite"] or any(not math.isfinite(metrics[key]) for key in
                                    ("max_abs", "rms_error", "reference_rms", "relative_l2")):
        return ["nonfinite tensor or error metric"]
    result = []
    if metrics["mismatched_elements"]:
        result.append("elementwise FP32 error gate")
    if metrics["rms_error"] > RMS_ATOL + RMS_RTOL * metrics["reference_rms"]:
        result.append("RMS FP32 error gate")
    return result


def check_accumulator(torch, x, dy, accumulator, coefficients, label: str, evidence: Evidence) -> None:
    """Check the entire accumulated gradient using bounded FP32 vocabulary tiles."""
    totals = {"numel": 0, "error_squared": 0.0, "reference_squared": 0.0,
              "max_abs": 0.0, "mismatched_elements": 0, "finite": True}
    with phase(evidence, label + "/reference-check", coefficients=coefficients):
        x_float = x.float()
        for start, end in tile_ranges(dy.shape[1]):
            reference = reference_tile(torch, x_float, dy[:, start:end], coefficients)
            actual = accumulator[start:end]
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
            tile = {"label": label, "rows": [start, end], "finite": finite, "numel": actual.numel()}
            if finite:
                difference = actual - reference
                tile.update(max_abs=difference.abs().max().item(),
                            error_squared=difference.square().sum(dtype=torch.float64).item(),
                            reference_squared=reference.square().sum(dtype=torch.float64).item(),
                            mismatched_elements=int((difference.abs() > ELEMENT_ATOL +
                                                     ELEMENT_RTOL * reference.abs()).sum().item()))
                finite = all(math.isfinite(tile[key]) for key in
                             ("max_abs", "error_squared", "reference_squared"))
                tile["finite"] = finite
            totals["finite"] &= finite
            totals["numel"] += tile["numel"]
            if finite:
                for key in ("error_squared", "reference_squared", "mismatched_elements"):
                    totals[key] += tile[key]
                totals["max_abs"] = max(totals["max_abs"], tile["max_abs"])
            # JSON remains valid on reduction overflow as well as nonfinite operands.
            tile = {key: None if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in tile.items()}
            evidence.emit("reference-tile", **tile)
            del reference, actual
        if totals["finite"]:
            totals.update(rms_error=math.sqrt(totals["error_squared"] / totals["numel"]),
                          reference_rms=math.sqrt(totals["reference_squared"] / totals["numel"]),
                          relative_l2=math.sqrt(totals["error_squared"] /
                                               max(totals["reference_squared"], 1e-60)))
        failures = metric_failures(totals)
        totals = {key: None if isinstance(value, float) and not math.isfinite(value) else value
                  for key, value in totals.items()}
        evidence.emit("check-summary", label=label, status="failed" if failures else "passed",
                      failures=failures, **totals)
        if failures:
            raise AssertionError(f"{label}: {failures}: {totals}")


def timed_call(torch, operation, x, dy, accumulator, label: str, evidence: Evidence) -> None:
    """Distinguish host-call return from eventual GPU completion; never time reference work."""
    with phase(evidence, label + "/pre-call-sync"):
        torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    evidence.emit("call-begin", label=label)
    start.record()
    began = time.monotonic()
    result = operation(x, dy, accumulator)
    host_seconds = time.monotonic() - began
    end.record()
    evidence.emit("call-returned", label=label, host_call_seconds=host_seconds)
    if result is not None:
        raise ValueError("native Apex wgrad unexpectedly returned a value")
    evidence.emit("gpu-wait-begin", label=label)
    end.synchronize()
    evidence.emit("timing", label=label, host_call_seconds=host_seconds,
                  synchronized_wall_seconds=time.monotonic() - began,
                  gpu_event_ms=start.elapsed_time(end))


def shape_probe(torch, operation, case, evidence: Evidence) -> None:
    """Run three additive contributions with one full accumulator and BF16 dy allocation."""
    name, tokens, hidden, vocabulary = case
    with phase(evidence, name + "/allocate", x_shape=[tokens, hidden],
               dy_shape=[tokens, vocabulary], main_grad_shape=[vocabulary, hidden]):
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        x.mul_(0.25)
        dy = torch.randn(tokens, vocabulary, device="cuda", dtype=torch.bfloat16)
        dy.mul_(0.25)
        accumulator = torch.full((vocabulary, hidden), INITIAL_ACCUMULATOR,
                                 device="cuda", dtype=torch.float32)
        if not x.is_contiguous() or not dy.is_contiguous() or not accumulator.is_contiguous():
            raise ValueError("native LM-head probe requires contiguous tensors")
        evidence.emit("tensor-spec", shape=name,
                      tensors={key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                     "stride": list(value.stride()), "device": str(value.device)}
                               for key, value in (("x", x), ("dy", dy), ("main_grad", accumulator))})
    for index, coefficient in enumerate(COEFFICIENTS):
        if index:
            dy.mul_(coefficient / COEFFICIENTS[index - 1])
        label = f"{name}/microbatch{index}"
        timed_call(torch, operation, x, dy, accumulator, label, evidence)
        check_accumulator(torch, x, dy, accumulator, COEFFICIENTS[:index + 1], label, evidence)
    del accumulator, dy, x
    torch.cuda.synchronize()
    evidence.emit("shape-complete", shape=name)


def main(argv=None) -> None:
    """Use the caller's private caches and Slurm-local device; emit before importing PyTorch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    evidence = Evidence(args.output, rank)
    try:
        evidence.record.update(probe_sha256=checksum(Path(__file__)), local_rank=local,
                               shapes=SHAPES, coefficients=COEFFICIENTS,
                               initial_accumulator=INITIAL_ACCUMULATOR, vocab_tile=VOCAB_TILE,
                               tolerances={"element_rtol": ELEMENT_RTOL, "element_atol": ELEMENT_ATOL,
                                           "rms_rtol": RMS_RTOL, "rms_atol": RMS_ATOL},
                               limitations=["No DDP, optimizer, router, recompute, or distributed validation",
                                            "Correlated power-of-two contributions, not arbitrary training inputs",
                                            "GPU-event interval includes possible host launch/heuristic gaps",
                                            "Original Apex workspace-lifetime hazard is not patched or worked around"])
        with phase(evidence, "source-verification"):
            evidence.record["sources"] = verify_sources()
        with phase(evidence, "torch-import"):
            import torch
        with phase(evidence, "device-initialization"):
            torch.cuda.set_device(local)
            torch.manual_seed(42)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            evidence.record.update(torch=torch.__version__, device=torch.cuda.get_device_name(), seed=42,
                                   environment={key: os.environ.get(key) for key in
                                                ("TORCH_EXTENSIONS_DIR", "PYTORCH_HIP_ALLOC_CONF",
                                                 "TORCH_COMPILE_DISABLE", "ROCR_VISIBLE_DEVICES",
                                                 "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")})
        operation = load_operation(evidence)
        for case in SHAPES:
            shape_probe(torch, operation, case, evidence)
        evidence.record["status"] = "passed"
        evidence.emit("complete")
    except BaseException as error:
        evidence.record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        evidence.emit("failed")
        raise
    finally:
        evidence.close()


if __name__ == "__main__":
    main()
