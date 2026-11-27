"""Isolated ABBA comparison of local post-combine sum retention and replay."""

from collections import deque
from datetime import timedelta
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

from apertus_amd.artifacts import sha256, write_json
from apertus_amd.entrypoint import verify_imports


COMMIT = "041c1d753028e4190efb342accb98a6f362ee7dd"
IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
ENTRYPOINT = "megatron/core/transformer/moe/moe_layer.py"
BLOBS = {
    ENTRYPOINT: "be864a03f21e997df6590ae9294c77f1849953af",
    "megatron/core/transformer/moe/token_dispatcher.py": "52b11896f18711b1532e6e0cef789aed38f049c1",
    "megatron/core/transformer/transformer_layer.py": "6f53e0625c10645a3e90722949ed34af1ec08576",
    "megatron/core/extensions/transformer_engine.py": "24b4b15208af6816cb4a2ba3e755b490cc5c1f79",
    "megatron/core/tensor_parallel/random.py": "92d39ba92efbccf22c71d48173b3ddf49a5197df",
    "megatron/core/model_parallel_config.py": "d5cd5397d562d8b9652578e4548e222c889a624d",
    "megatron/core/fusions/fused_bias_dropout.py": "2eb4007f75c538b672f628a011ec6eeb3742a9c4",
}
ARMS = ("A", "B", "B", "A")
IN_FLIGHT, WARMUPS, SAMPLES = 4, 3, 5
LATENT = (4096, 2, 1792)
HIDDEN = (4096, 2, 3584)
SUM_BYTES = math.prod(HIDDEN) * 2
STATS = ("allocated", "reserved")
FIXTURES = ("latent", "shared", "residual", "cotangent")
DERIVATIVES = ("output", "dlatent", "dshared", "dresidual", "projection_wgrad", "norm_wgrad")


def verify_source(source):
    """Check source provenance for the actual projection, normalization and checkpoint path."""
    for path, blob in BLOBS.items():
        module = importlib.import_module(path.removesuffix(".py").replace("/", "."))
        file = Path(module.__file__).resolve()
        data = file.read_bytes()
        if file != source / path or hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != blob:
            raise ValueError(f"unexpected Megatron import: {path}")
    manifest = json.loads(Path("/apertus/source/manifest.json").read_text())
    if not verify_imports(source, manifest["files"]):
        raise ValueError("no verified Megatron imports")
    wheel = Path("/opt/venv/lib/python3.12/site-packages/transformer_engine")
    for name in ("module.linear", "module.rmsnorm", "ops.basic.rmsnorm"):
        module = importlib.import_module(f"transformer_engine.pytorch.{name}")
        if not Path(module.__file__).resolve().is_relative_to(wheel):
            raise ValueError(f"unexpected TE import: {name}")


def build(torch):
    """Use the signed Megatron layer methods with pre-supplied combine/shared/residual outputs."""
    from megatron.core import tensor_parallel
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
    from megatron.core.transformer.transformer_layer import TransformerLayer

    torch.manual_seed(42)
    tensor_parallel.model_parallel_cuda_manual_seed(42)
    # This fixture has no EP group. Carry only the local module's effective settings.
    local = ModelParallelConfig(
        bf16=True, params_dtype=torch.bfloat16, gradient_accumulation_fusion=False, delay_wgrad_compute=True)
    config = SimpleNamespace(**vars(local), moe_latent_size=LATENT[-1], normalization="RMSNorm",
                             layernorm_zero_centered_gamma=False, fused_residual_rmsnorm=False,
                             disable_parameter_transpose_cache=False, init_model_with_meta_device=False)
    proj = TELinear(LATENT[-1], HIDDEN[-1], parallel_mode="duplicated", config=config,
                    init_method=lambda w: torch.nn.init.normal_(w, std=0.0167038),
                    bias=False, skip_bias_add=False, skip_weight_param_allocation=False, is_expert=False).train()
    norm = TENorm(config, HIDDEN[-1], eps=1e-5).train()
    if proj.weight.dtype != torch.bfloat16 or norm.weight.dtype != torch.bfloat16:
        raise ValueError("expected the BF16 projection and RMSNorm path")
    dispatcher = SimpleNamespace(hidden_shape=LATENT)
    dispatcher.combine_postprocess = MoEFlexTokenDispatcher.combine_postprocess.__get__(dispatcher)
    moe = SimpleNamespace(config=config, fc2_latent_proj=proj, token_dispatcher=dispatcher)
    layer = SimpleNamespace(residual_output_scale=1.0 / math.sqrt(2 * 42))
    return SimpleNamespace(
        projection=proj, norm=norm, checkpoint=CheckpointWithoutOutput,
        region=MoELayer.postprocess.__get__(moe),
        post_norm=TransformerLayer._apply_post_norm.__get__(layer),
        bda=get_bias_dropout_add(training=True, fused=True))


def post_combine(model, arm, latent, shared, residual, *, region=None):
    """Replay only the projection and BF16 add; keep the same downstream norm and residual."""
    region = model.region if region is None else region
    checkpoint = model.checkpoint(fp8=None) if arm == "B" else None
    summed = (region(latent, shared) if checkpoint is None else checkpoint.checkpoint(region, latent, shared))
    normalized = model.post_norm((summed, None), model.norm)
    if checkpoint is not None:
        checkpoint.discard_output_and_register_recompute(normalized[0])
        if summed.untyped_storage().nbytes() != 0:
            raise ValueError("sum storage survived the output-discard checkpoint")
    return model.bda(normalized, residual, 0.0)


def fingerprint(torch, tensor):
    """Check finiteness and compare exact tensor bytes outside measured trials."""
    if tensor is None:
        raise ValueError("missing component tensor")
    cpu = tensor.detach().cpu().contiguous()
    if not torch.isfinite(cpu).all().item():
        raise ValueError("nonfinite component tensor")
    return [list(cpu.shape), str(cpu.dtype),
            hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy()).hexdigest()]


def expected_calls(arm):
    """Require the original path in fill and exactly one local replay per backward in B."""
    return {"fill_grad": IN_FLIGHT if arm == "A" else 0, "fill_no_grad": IN_FLIGHT if arm == "B" else 0,
            "backward_grad": IN_FLIGHT if arm == "B" else 0, "backward_no_grad": 0}


def schedule(torch, model, arm, fixture, *, capture=False):
    """Hold four graphs, then drain each dgrad and its single queued projection wgrad."""
    cuda = torch.cuda
    proj, norm = model.projection, model.norm
    leaves = [tuple(t.detach().requires_grad_() for t in values[:3]) for values in fixture]
    queue = proj.wgrad_store.context
    if not proj.wgrad_store.delay_wgrad_compute() or queue.qsize():
        raise ValueError("expected an enabled, empty delayed projection wgrad queue")
    live, wgrad, values = deque(), [], {}
    calls = dict.fromkeys(expected_calls(arm), 0)
    phase = "fill"

    def region(*inputs):
        calls[f"{phase}_{'grad' if torch.is_grad_enabled() else 'no_grad'}"] += 1
        return model.region(*inputs)

    def memory():
        return [cuda.memory_allocated(), cuda.memory_reserved()]

    def peak():
        return [cuda.max_memory_allocated(), cuda.max_memory_reserved()]

    cuda.synchronize()
    entry = memory()
    cuda.reset_peak_memory_stats()
    start = perf_counter()
    for inputs in leaves:
        y = post_combine(model, arm, *inputs, region=region if capture else None)
        if tuple(y.shape) != HIDDEN or y.dtype != torch.bfloat16 or y.grad_fn is None:
            raise ValueError("unexpected component output")
        live.append(y)
    del y
    cuda.synchronize()
    forward_seconds = perf_counter() - start
    retained, forward_peak = memory(), peak()
    if len({t.data_ptr() for t in live}) != IN_FLIGHT or queue.qsize():
        raise ValueError("expected four independent forward graphs and no queued wgrad")

    phase = "backward"
    cuda.reset_peak_memory_stats()
    start = perf_counter()
    for i, (xs, data) in enumerate(zip(leaves, fixture, strict=True)):
        proj.weight.grad = norm.weight.grad = None
        y = live.popleft()
        y.backward(data[-1])
        queued = queue.qsize()
        if queued != 1 or proj.weight.grad is not None:
            raise ValueError("expected exactly one deferred projection wgrad and no eager contribution")
        proj.backward_dw()
        drained = queue.qsize()
        if drained or proj.weight.grad is None:
            raise ValueError("projection wgrad did not drain exactly once")
        wgrad.append([queued, drained])
        if capture:
            tensors = (y, *(x.grad for x in xs), proj.weight.grad, norm.weight.grad)
            values.update({f"{k}:{i}": fingerprint(torch, v) for k, v in zip(DERIVATIVES, tensors, strict=True)})
            del tensors
        for x in xs:
            x.grad = None
        proj.weight.grad = norm.weight.grad = None
        del y
    cuda.synchronize()
    return (dict(entry=entry, retained=retained, forward_peak=forward_peak,
                 backward_peak=peak(), forward_seconds=forward_seconds,
                 backward_seconds=perf_counter() - start, wgrad=wgrad), calls, values)


def worker(output, source, index):
    """Collect an untimed parity pass and five warm memory samples in a fresh process."""
    import torch
    from megatron.core import parallel_state

    if ((torch.__version__, torch.version.hip, torch.cuda.device_count()) != ("2.14.0+rocm10.1.0", "7.16.26385", 1)
            or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx942"):
        raise ValueError("expected one image-910 MI300A")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    arm = ARMS[index]
    try:
        model = build(torch)
        verify_source(source)
        rng = torch.Generator(device="cuda").manual_seed(1701)
        fixture = [tuple(torch.randn(s, generator=rng, device="cuda", dtype=torch.bfloat16)
                         for s in (LATENT, HIDDEN, HIDDEN, HIDDEN)) for _ in range(IN_FLIGHT)]
        inputs = {f"{name}:{i}": fingerprint(torch, t) for i, row in enumerate(fixture)
                  for name, t in zip(FIXTURES, row, strict=True)}
        inputs.update(projection=fingerprint(torch, model.projection.weight), norm=fingerprint(torch, model.norm.weight))
        print(f"arm {index} {arm}: signed projection and RMSNorm verified", flush=True)
        parity_sample, calls, values = schedule(torch, model, arm, fixture, capture=True)
        write_json(output / f"evidence-{index}.json",
                   dict(arm=arm, calls=calls, fixtures=inputs, derivatives=values, wgrad=parity_sample["wgrad"]))
        if calls != expected_calls(arm):
            raise ValueError(f"unexpected local replay counts: {calls}")
        for _ in range(WARMUPS):
            schedule(torch, model, arm, fixture)
        samples = []
        for _ in range(SAMPLES):
            sample, _, _ = schedule(torch, model, arm, fixture)
            samples.append(sample)
            write_json(output / f"samples-{index}.json", samples)
        print(f"arm {index} {arm}: completed {SAMPLES} warm samples", flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def measurements(sample):
    """Validate lifecycle and return absolute peaks, not just changes during backward."""
    entry, retained, fwd, bwd = (sample[k] for k in ("entry", "retained", "forward_peak", "backward_peak"))
    times = {k: sample[k] for k in ("forward_seconds", "backward_seconds")}
    if (any(len(v) != 2 or not 0 < v[0] <= v[1] for v in (entry, retained, fwd, bwd))
            or not all(e <= r <= f and r <= b for e, r, f, b in zip(entry, retained, fwd, bwd, strict=True))
            or not all(math.isfinite(t) and t > 0 for t in times.values())
            or sample["wgrad"] != [[1, 0]] * IN_FLIGHT):
        raise ValueError("invalid memory or delayed-wgrad evidence")
    return {**times, **{f"{phase}_{kind}": v[i] for phase, v in (
        ("entry", entry), ("retained", retained), ("forward_peak", fwd), ("backward_peak", bwd),
        ("absolute_peak", [max(a, b) for a, b in zip(fwd, bwd, strict=True)]))
        for i, kind in enumerate(STATS)}}


def verdict(reports, samples):
    """Require bitwise parity and lower fill and absolute backward peaks in both pairings."""
    if tuple(x["arm"] for x in reports) != ARMS or len(samples) != len(ARMS):
        raise ValueError("unexpected ABBA order")
    medians, differences, entries = [], [], set()
    reference = reports[0]
    keys = lambda names: {f"{k}:{i}" for k in names for i in range(IN_FLIGHT)}
    for arm, e, trials in zip(ARMS, reports, samples, strict=True):
        if (e["calls"] != expected_calls(arm) or e["wgrad"] != [[1, 0]] * IN_FLIGHT
                or set(e["fixtures"]) != keys(FIXTURES) | {"projection", "norm"}
                or set(e["derivatives"]) != keys(DERIVATIVES) or len(trials) != SAMPLES):
            raise ValueError("incomplete component parity or replay evidence")
        differences.append({k: sorted(n for n, v in e[k].items() if v != reference[k][n])
                            for k in ("fixtures", "derivatives")})
        rows = [measurements(s) for s in trials]
        entries.update(x["entry_allocated"] for x in rows)
        medians.append({k: median(x[k] for x in rows) for k in rows[0]})
    pairs = [{k: medians[b][k] - medians[a][k] for k in medians[a]} for a, b in ((0, 1), (3, 2))]
    equal = not any(v for diff in differences for v in diff.values())
    lower = all(p["retained_allocated"] < 0 and p["backward_peak_allocated"] < 0
                and p["absolute_peak_allocated"] <= 0 for p in pairs)
    return dict(status="passed" if equal and lower and len(entries) == 1 else "failed",
                scope="single-GPU post-combine component; no distributed fit or throughput verdict",
                matched_entry_allocated=len(entries) == 1, bitwise_equal=equal, memory_gate=lower,
                expected_sum_bytes=SUM_BYTES * IN_FLIGHT, differing_tensors=differences,
                median_by_arm=medians, paired_B_minus_A=pairs)


def train(output, source, entrypoint, args):
    """Run only the single-rank component, with private caches and no training publication."""
    frozen = json.loads((output.parent / "resolved.json").read_text())
    if (entrypoint != ENTRYPOINT or args or os.environ.get("WORLD_SIZE") != "1" or os.environ.get("RANK") != "0"
            or frozen["image"]["sha256"] != IMAGE or frozen["source"]["commit"] != COMMIT
            or frozen["procedure"].get("auxiliary_source") is not None):
        raise ValueError("unexpected frozen component inputs")
    for index in range(len(ARMS)):
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                        str(output), str(source.resolve()), str(index)], cwd=source, check=True)


def check(run):
    """Preserve measured deltas even if the parity or memory gate rejects the candidate."""
    out = run / "output"
    reports, samples = ([json.loads((out / f"{name}-{i}.json").read_text()) for i in range(len(ARMS))]
                        for name in ("evidence", "samples"))
    result = verdict(reports, samples)
    write_json(out / "checks.json", result)
    if result["status"] != "passed":
        raise ValueError("post-combine parity, matched-entry or memory gate failed")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["train", output, source, entrypoint, *args]:
            train(Path(output), Path(source), entrypoint, args)
        case ["worker", output, source, index] if index in ("0", "1", "2", "3"):
            worker(Path(output), Path(source), int(index))
        case ["check", path]:
            check(Path(path))
        case _:
            raise SystemExit("unexpected post-combine probe invocation")
