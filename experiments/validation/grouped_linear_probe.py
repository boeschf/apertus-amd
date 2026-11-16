"""Standalone GPU numerical probe for Chonk's TE expert projections, not a train validator.

Run with the pinned image's Python inside a parent-owned allocation, passing an
output directory. No source imports, training hooks, process groups, or optimizer
updates are installed. This checks the underlying TE operation, not complete MoE
recompute, router state, DDP hooks, KDA, compilation, or distributed equivalence.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path


PROJECTIONS = (("fc1", 1792, 3840), ("fc2", 1920, 1792))
LOCAL_EXPERTS = 16
MICROBATCHES = 3
UPDATES = 2
RTOL = 0.02
ATOL = 0.02
# BF16 outputs incur at most 2^-8 relative rounding error away from underflow.
# Two units allow independent GEMM reductions to straddle a BF16 rounding tie.
BF16_RMS_RTOL = 2 * 2**-8
# Fused BF16-input/FP32-output GEMMs have no BF16 output rounding. This separate,
# deliberately stricter budget is a probe gate, not a backend acceptance claim.
FP32_RMS_RTOL = 1e-4
RMS_ATOL = 1e-6


def token_splits(update: int, microbatch: int) -> list[int]:
    """Include balanced Chonk volume and empty/one-token/unaligned expert groups."""
    # EP=16 gathers 16 x 4096 tokens, with top-8/256 experts: 2048 rows/expert.
    if (update + microbatch) % 2 == 0:
        return [2048] * LOCAL_EXPERTS
    return [0, 1, 127, 128, 129, 511, 1023, 2047, 2048, 2049, 3071, 3072, 4095, 4096, 17, 33]


def error_failures(metrics: dict, rms_rtol: float) -> list[str]:
    """Apply elementwise and normwise gates independently, rejecting nonfinite metrics."""
    if not metrics["finite"]:
        return ["nonfinite input or reference"]
    if any(metrics[key] is None or not math.isfinite(metrics[key]) for key in
           ("max_abs", "relative_l2", "rms_error", "reference_rms", "max_tolerance_ratio")):
        return ["nonfinite error metrics"]
    failures = []
    if metrics["max_tolerance_ratio"] > 1:
        failures.append("elementwise error exceeds propagated tolerance")
    if metrics["rms_error"] > RMS_ATOL + rms_rtol * metrics["reference_rms"]:
        failures.append("RMS error exceeds precision-specific tolerance")
    return failures


def compare(torch, actual, expected, label: str, emit, *, rms_rtol=BF16_RMS_RTOL,
            required=True, absolute_sum=None, contributions=1) -> None:
    """Emit diagnostics before judging; numerical failures do not hide subsequent checks."""
    actual_dtype, reference_dtype = str(actual.dtype), str(expected.dtype)
    actual = actual.detach().float()
    expected = expected.detach().float()
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape {actual.shape} != {expected.shape}")
    actual_nonfinite = int((~torch.isfinite(actual)).sum().item())
    reference_nonfinite = int((~torch.isfinite(expected)).sum().item())
    finite = actual_nonfinite == reference_nonfinite == 0
    metrics = {"finite": finite, "max_abs": None, "relative_l2": None,
               "rms_error": None, "reference_rms": None, "max_tolerance_ratio": None,
               "actual_nonfinite": actual_nonfinite, "reference_nonfinite": reference_nonfinite}
    worst = None
    if finite:
        difference = actual - expected
        # An accumulated BF16 reference may nearly cancel. Propagate the SAME
        # per-contribution tolerance, instead of scaling it by the cancelled sum.
        scale = expected.abs() if absolute_sum is None else absolute_sum
        allowance = ATOL * contributions + RTOL * scale
        ratio = difference.abs() / allowance
        metrics.update(max_abs=difference.abs().max().item(),
                       relative_l2=(difference.norm() / expected.norm().clamp_min(1e-30)).item(),
                       rms_error=(difference.norm() / math.sqrt(difference.numel())).item(),
                       reference_rms=(expected.norm() / math.sqrt(expected.numel())).item(),
                       max_tolerance_ratio=ratio.max().item(),
                       mismatched_elements=int((ratio > 1).sum().item()))
        index = int(ratio.reshape(-1).argmax().item())
        worst = {"flat_index": index, "actual": actual.reshape(-1)[index].item(),
                 "reference": expected.reshape(-1)[index].item(),
                 "allowed_abs": allowance.reshape(-1)[index].item()}
    failures = error_failures(metrics, rms_rtol)
    # Keep failure reports serializable even if finite operands overflow reductions.
    metrics = {key: None if isinstance(value, float) and not math.isfinite(value) else value
               for key, value in metrics.items()}
    if worst is not None:
        worst = {key: None if isinstance(value, float) and not math.isfinite(value) else value
                 for key, value in worst.items()}
    emit({"label": label, "required": required,
          "status": ("failed" if failures else "passed") if required else "diagnostic",
          "actual_dtype": actual_dtype, "reference_dtype": reference_dtype,
          "shape": list(actual.shape), "numel": actual.numel(), **metrics, "worst_tolerance": worst,
          "policy": {"rtol": RTOL, "atol_per_contribution": ATOL,
                     "contributions": contributions, "scale": "sum_abs_contributions" if
                     absolute_sum is not None else "abs_reference", "rms_rtol": rms_rtol,
                     "rms_atol": RMS_ATOL}, "failures": failures})


def require_passed(records: list[dict]) -> None:
    """Fail after collecting both modes; diagnostics alone never count as acceptance."""
    failures = [row for row in records if row["required"] and row["status"] != "passed"]
    if failures:
        raise AssertionError(f"{len(failures)} numerical checks failed; first: " +
                             json.dumps(failures[0], allow_nan=False))


def accumulate_wgrad_reference(exact, rounded, absolute_sum, contribution, dtype) -> None:
    """Round each independent GEMM before the plain FP32 sum, never the final sum."""
    exact.add_(contribution)
    quantized = contribution.to(dtype).float()
    rounded.add_(quantized)
    absolute_sum.add_(quantized.abs())


def projection_probe(torch, te, name: str, inputs: int, outputs: int, emit) -> None:
    """Check forward, dgrad, and FP32 accumulated wgrad against independent FP32 GEMMs."""
    plain = te.GroupedLinear(LOCAL_EXPERTS, inputs, outputs, bias=False,
                             params_dtype=torch.bfloat16, fuse_wgrad_accumulation=False)
    fused = te.GroupedLinear(LOCAL_EXPERTS, inputs, outputs, bias=False,
                             params_dtype=torch.bfloat16, fuse_wgrad_accumulation=True)
    plain_weights = [getattr(plain, f"weight{i}") for i in range(LOCAL_EXPERTS)]
    fused_weights = [getattr(fused, f"weight{i}") for i in range(LOCAL_EXPERTS)]
    with torch.no_grad():
        for original, candidate in zip(plain_weights, fused_weights, strict=True):
            original.normal_(std=0.02)
            candidate.copy_(original)
            candidate.main_grad = torch.empty_like(candidate, dtype=torch.float32)
    for update in range(UPDATES):
        expected_wgrads = [torch.zeros_like(w, dtype=torch.float32) for w in plain_weights]
        rounded_wgrads = [torch.zeros_like(w, dtype=torch.float32) for w in plain_weights]
        rounded_absolute_sums = [torch.zeros_like(w, dtype=torch.float32) for w in plain_weights]
        plain_wgrads = [torch.zeros_like(w, dtype=torch.float32) for w in plain_weights]
        # A sentinel detects missing first-microbatch overwrite, including empty groups.
        for weight in fused_weights:
            weight.main_grad.fill_(17)
        for microbatch in range(MICROBATCHES):
            splits = token_splits(update, microbatch)
            x = torch.randn(sum(splits), inputs, device="cuda", dtype=torch.bfloat16) * 0.25
            dy = torch.randn(sum(splits), outputs, device="cuda", dtype=torch.bfloat16) * 0.25
            reference_y = torch.empty_like(dy, dtype=torch.float32)
            reference_dx = torch.empty_like(x, dtype=torch.float32)
            reference_microbatch = []
            start = 0
            with torch.no_grad():
                for expert, (count, weight) in enumerate(zip(splits, plain_weights, strict=True)):
                    end = start + count
                    xf, gyf, wf = x[start:end].float(), dy[start:end].float(), weight.float()
                    reference_y[start:end] = xf @ wf.T
                    reference_dx[start:end] = gyf @ wf
                    dw = gyf.T @ xf
                    reference_microbatch.append(dw)
                    accumulate_wgrad_reference(expected_wgrads[expert], rounded_wgrads[expert],
                                               rounded_absolute_sums[expert], dw, weight.dtype)
                    start = end
            label = f"{name}/update{update}/microbatch{microbatch}"
            for mode, module, weights in (("plain", plain, plain_weights),
                                           ("fused", fused, fused_weights)):
                module.zero_grad(set_to_none=True)
                previous_fused = ([w.main_grad.clone() for w in weights]
                                  if mode == "fused" and microbatch else None)
                value = x.detach().clone().requires_grad_(True)
                result = module(value, splits, is_first_microbatch=microbatch == 0)
                compare(torch, result, reference_y, f"{label}/{mode}/forward", emit)
                result.backward(dy)
                compare(torch, value.grad, reference_dx, f"{label}/{mode}/dgrad", emit)
                if mode == "plain":
                    # Match Megatron's FP32 DDP accumulation, not PyTorch BF16 .grad summation.
                    for expert, (total, weight) in enumerate(zip(plain_wgrads, weights, strict=True)):
                        if weight.grad is None:
                            raise AssertionError(f"{label}: missing plain weight gradient")
                        compare(torch, weight.grad, reference_microbatch[expert].to(weight.dtype).float(),
                                f"{label}/{mode}/expert{expert}/microbatch-wgrad-rounded", emit)
                        total.add_(weight.grad.float())
                totals = plain_wgrads if mode == "plain" else [w.main_grad for w in weights]
                for expert, total in enumerate(totals):
                    prefix = f"{label}/{mode}/expert{expert}"
                    if mode == "plain":
                        compare(torch, total, rounded_wgrads[expert],
                                f"{prefix}/accumulated-wgrad-rounded", emit,
                                absolute_sum=rounded_absolute_sums[expert], contributions=microbatch + 1)
                        compare(torch, total, expected_wgrads[expert],
                                f"{prefix}/actual-minus-fp32", emit, required=False)
                        compare(torch, rounded_wgrads[expert], expected_wgrads[expert],
                                f"{prefix}/rounding-only-minus-fp32", emit, required=False)
                    else:
                        delta = total if previous_fused is None else total - previous_fused[expert]
                        compare(torch, delta, reference_microbatch[expert],
                                f"{prefix}/microbatch-wgrad-fp32", emit, rms_rtol=FP32_RMS_RTOL)
                        compare(torch, total, expected_wgrads[expert],
                                f"{prefix}/accumulated-wgrad-fp32", emit, rms_rtol=FP32_RMS_RTOL)
                del value, result, previous_fused
            del x, dy, reference_y, reference_dx, reference_microbatch
        del expected_wgrads, rounded_wgrads, rounded_absolute_sums, plain_wgrads


def main(argv=None) -> None:
    """Run only when explicitly invoked with image PyTorch and an allocated GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    import torch
    import transformer_engine.pytorch as te

    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    implementation = Path(inspect.getfile(te.GroupedLinear))
    record = {"status": "failed", "scope": "TE grouped-linear numerical probe only",
              "rank": rank, "local_rank": local_rank, "seed": 42, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "rtol": RTOL, "atol": ATOL,
              "bf16_rms_rtol": BF16_RMS_RTOL, "fp32_rms_rtol": FP32_RMS_RTOL,
              "rms_atol": RMS_ATOL, "reference_policy": {
                  "plain": "FP32 sum of individually BF16-rounded independent FP32 GEMMs",
                  "fused": "FP32 sum of independent FP32 GEMMs, without BF16 output rounding",
                  "plain_accumulated_elementwise": "sum of per-microbatch tolerance budgets"},
              "updates": UPDATES, "microbatches": MICROBATCHES, "local_experts": LOCAL_EXPERTS,
              "projections": PROJECTIONS, "implementation": str(implementation),
              "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": {key: os.environ.get(key) for key in
                              ("NVTE_USE_GROUPED_GEMM_TRITON", "NVTE_GROUPED_LINEAR_SINGLE_PARAM",
                               "PYTORCH_HIP_ALLOC_CONF", "TORCH_COMPILE_DISABLE")},
              "not_checked": ["MoE activation/recompute", "router state", "DDP/optimizer",
                              "streams", "compilation", "distributed training"]}
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an accidental rerun from replacing prior evidence.
    with (args.output / f"grouped-linear-probe-{rank}.json").open("x") as summary, \
            (args.output / f"grouped-linear-probe-{rank}.jsonl").open("x", buffering=1) as stream:
        record["checks"] = []

        def emit(check):
            record["checks"].append(check)
            stream.write(json.dumps(check, allow_nan=False) + "\n")

        try:
            for name, inputs, outputs in PROJECTIONS:
                projection_probe(torch, te, name, inputs, outputs, emit)
            torch.cuda.synchronize()
            require_passed(record["checks"])
            record["status"] = "passed"
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error), "repr": repr(error)}
            raise
        finally:
            summary.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
