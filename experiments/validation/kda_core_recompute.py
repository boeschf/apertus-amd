"""Measure the outer KDA core checkpoint with four simultaneously live Megatron forwards."""

from collections import deque
from datetime import timedelta
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

from apertus_amd.artifacts import sha256, write_json


COMMIT = "041c1d753028e4190efb342accb98a6f362ee7dd"
IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
ENTRYPOINT = "megatron/core/ssm/kimi_delta_attention.py"
BLOBS = {
    ENTRYPOINT: "dec38d64b0c14641eb668d770b6f10313fee8891",
    "megatron/core/ssm/gated_delta_net.py": "16f9384df055627c0c4a39d3275fdd12b7a86a0c",
    "megatron/core/tensor_parallel/random.py": "92d39ba92efbccf22c71d48173b3ddf49a5197df",
    "megatron/core/transformer/transformer_config.py": "be4d74b7aecf67a6cafe30540a4ca950c4782d98",
    "megatron/core/models/gpt/experimental_attention_variant_module_specs.py": "8ea070246d5aaf5aaf9d0e618aef39d5133215fb",
    "megatron/core/extensions/transformer_engine.py": "24b4b15208af6816cb4a2ba3e755b490cc5c1f79",
}
RELEASE = {
    "commit": "6c70d55a71e7fb4fca7de990015f6a3046193762",
    "manifest_sha256": "6c383647a70e2c6f0b0fc13f14cd5e0ef6e95ca6d3b34d3e7ecdfdf7e1ba6c1b",
    "path": "fla/ops/kda/chunk_bwd.py",
    "sha256": "79d34bd5711a71e455a6b0346085c9431f5b224a5766ed839468a8902ead0e21",
}
WHEEL = Path("/opt/venv/lib/python3.12/site-packages")
AUXILIARY = Path("/apertus/auxiliary.py")
SHIPPED_SHA = "069395c9232647f5a4b82fa2dfe7ca077423016d4b641f2696f9d6328dc2e9be"
MODULE = "fla.ops.kda.chunk_bwd"
ARMS = ("A", "B", "B", "A")
SHAPE = (4096, 2, 3584)
HEADS, HEAD_DIM = 28, 128
IN_FLIGHT, WARMUPS, SAMPLES = 4, 3, 5


def modules(arm):
    """Keep the training recompute list identical except for the KDA consumer."""
    return ["layernorm", *(["linear_attn"] if arm == "A" else []), "qkv", "shared_experts"]


def expected_calls(arm):
    """Count the original forwards, producer/core replays and signed backward returns."""
    return {"forward": {"qkv": IN_FLIGHT, "chunk_kda": IN_FLIGHT, "signed_backward": 0},
            "backward": {"qkv": IN_FLIGHT, "chunk_kda": IN_FLIGHT if arm == "A" else 0,
                         "signed_backward": IN_FLIGHT}}


class SignedBackwardFinder(importlib.abc.MetaPathFinder):
    """Select only the intact signed KDA backward module, never a package overlay."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == MODULE:
            return importlib.util.spec_from_file_location(fullname, AUXILIARY)
        return None


def select_kda():
    """Validate the image baseline and signed override before any FLA import."""
    if any(name == "fla" or name.startswith("fla.") for name in sys.modules):
        raise ValueError("FLA already resolved before signed module selection")
    if (AUXILIARY.is_symlink() or sha256(AUXILIARY) != RELEASE["sha256"]
            or sha256(WHEEL / RELEASE["path"]) != SHIPPED_SHA
            or any(importlib.metadata.version(p) != "0.5.2" for p in ("flash-linear-attention", "fla-core"))):
        raise ValueError("unexpected signed KDA or image FLA bytes")
    sys.meta_path.insert(0, SignedBackwardFinder())
    return importlib.import_module("fla.ops.kda.chunk")


def verify_imports(source, chunk):
    """Verify the executed Megatron sources and the image/signed FLA boundary."""
    bwd = importlib.import_module(MODULE)
    if (Path(bwd.__file__).resolve() != AUXILIARY or chunk.chunk_kda_bwd is not bwd.chunk_kda_bwd
            or any(not Path(mod.__file__).resolve().is_relative_to(WHEEL)
                   for name, mod in tuple(sys.modules.items())
                   if (name == "fla" or name.startswith("fla.")) and name != MODULE
                   and getattr(mod, "__file__", None))):
        raise ValueError("unexpected FLA module resolution")
    for path, blob in BLOBS.items():
        module = importlib.import_module(path.removesuffix(".py").replace("/", "."))
        file = Path(module.__file__).resolve()
        data = file.read_bytes()
        if (file != source / path
                or hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != blob):
            raise ValueError(f"unexpected Megatron import: {path}")


def kda_config(config_class, dtype, arm):
    """Declare the matched sublayer settings without requiring GPU imports in the test."""
    return config_class(
        num_layers=42, hidden_size=SHAPE[-1], ffn_hidden_size=8960, num_attention_heads=HEADS,
        num_query_groups=7, kv_channels=HEAD_DIM, normalization="RMSNorm", layernorm_epsilon=1e-5,
        add_bias_linear=False, attention_dropout=0.0, hidden_dropout=0.0, init_method_std=0.0167038,
        sandwich_norm=True, residual_output_scaling=True, qk_layernorm=True,
        bf16=True, params_dtype=dtype, fp8=None, gradient_accumulation_fusion=False,
        cuda_graph_impl="none", delay_wgrad_compute=False, transformer_impl="transformer_engine",
        experimental_attention_variant="kda", linear_attention_freq=[1, 1, 1, 0] * 10 + [1, 0],
        linear_attention_use_output_gate=True,
        linear_attention_safe_output_gate=True, linear_attention_safe_output_gate_lower_bound=-5.0,
        kda_legacy_gate_out_proj_bias=True, linear_conv_kernel_dim=4,
        linear_key_head_dim=HEAD_DIM, linear_value_head_dim=HEAD_DIM,
        linear_num_key_heads=HEADS, linear_num_value_heads=HEADS,
        recompute_granularity="selective", recompute_modules=modules(arm),
    )


def build_kda(torch, arm):
    """Build the actual TE-backed KDA spec, with immediate wgrad in this isolated fixture."""
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear, TELayerNormColumnParallelLinear, TERowParallelLinear,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.kimi_delta_attention import get_kimi_delta_attention_module_spec
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    torch.manual_seed(42)
    tensor_parallel.model_parallel_cuda_manual_seed(42)
    config = kda_config(TransformerConfig, torch.bfloat16, arm)
    groups = ProcessGroupCollection.use_mpu_process_groups(["tp", "cp"])
    model = build_module(get_kimi_delta_attention_module_spec(config), config=config,
                         layer_number=1, pg_collection=groups).train()
    if (model.tp_size != 1 or model.cp_size != 1 or not model.recompute_qkv or model.recompute_qkv_fine
            or model._recompute_core != (arm == "A") or config.recompute_modules != modules(arm)
            or not model._use_fused_output_norm_gate or model._carry_enabled
            or model.initial_state_param is not None
            or not isinstance(model.in_proj, TELayerNormColumnParallelLinear)
            or not isinstance(model.out_proj, TERowParallelLinear)
            or any(not isinstance(p, TEColumnParallelLinear)
                   for p in (model.decay_out_proj, model.gate_out_proj))):
        raise ValueError("unexpected effective Megatron KDA path")
    return parallel_state, model


def tensor_receipt(torch, value, name):
    """Require finite tensors and fingerprint their exact contents outside timed windows."""
    if value is None:
        raise ValueError(f"missing {name}")
    cpu = value.detach().cpu().contiguous()
    if not torch.isfinite(cpu).all():
        raise ValueError(f"nonfinite {name}")
    data = cpu.reshape(-1).view(torch.uint8).numpy()
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "sha256": hashlib.sha256(data).hexdigest()}


def schedule(torch, model, inputs, cotangents, *, advance=None, receipt=None):
    """Hold four graphs before the oldest-first drain, measuring the two phases separately."""
    cuda = torch.cuda
    leaves = [x.detach().requires_grad_() for x in inputs]
    live = deque()

    def memory():
        return [cuda.memory_allocated(), cuda.memory_reserved()]

    def peak():
        return [cuda.max_memory_allocated(), cuda.max_memory_reserved()]

    model.zero_grad(set_to_none=True)
    cuda.synchronize()
    entry = memory()
    cuda.reset_peak_memory_stats()
    start = perf_counter()
    for x in leaves:
        y, bias = model(x, attention_mask=None)
        if tuple(y.shape) != SHAPE or y.dtype != torch.bfloat16 or bias is not None or y.grad_fn is None:
            raise ValueError("unexpected KDA output contract")
        live.append(y)
    del y
    cuda.synchronize()
    forward_seconds = perf_counter() - start
    retained, forward_peak = memory(), peak()
    if len(live) != IN_FLIGHT or len({y.data_ptr() for y in live}) != IN_FLIGHT:
        raise ValueError("expected distinct simultaneous forward graphs")
    if advance is not None:
        advance()
    cuda.reset_peak_memory_stats()
    start = perf_counter()
    for index, (x, dy) in enumerate(zip(leaves, cotangents, strict=True)):
        y = live.popleft()
        # CheckpointWithoutOutput requires .backward(), not autograd.grad().
        y.backward(dy)
        if receipt is not None:
            receipt(f"output:{index}", y)
            receipt(f"input_grad:{index}", x.grad)
        x.grad = None
        del y
    cuda.synchronize()
    backward_seconds = perf_counter() - start
    result = dict(entry=entry, retained=retained, forward_peak=forward_peak, backward_peak=peak(),
                  forward_seconds=forward_seconds, backward_seconds=backward_seconds)
    if receipt is not None:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                receipt(f"param_grad:{name}", parameter.grad)
    return result


def exercise(torch, model, chunk, arm, inputs, cotangents):
    """Verify actual replay and signed backward calls in an untimed parity pass."""
    qkv, kda, backward = model._in_proj_to_attn_inputs, model.gated_delta_rule, chunk.chunk_kda_bwd
    if kda is not chunk.chunk_kda or inspect.signature(kda).parameters["disable_recompute"].default is not False:
        raise ValueError("unexpected KDA implementation or internal recompute default")
    counts = {phase: dict.fromkeys(expected_calls(arm)[phase], 0) for phase in ("forward", "backward")}
    phase = "forward"
    numerics = {}

    def advance():
        nonlocal phase
        phase = "backward"

    def producer(*args, **kwargs):
        counts[phase]["qkv"] += 1
        return qkv(*args, **kwargs)

    def core(q, k, v, **kwargs):
        shape = (SHAPE[1], SHAPE[0], HEADS, HEAD_DIM)
        if (any(tuple(x.shape) != shape or x.dtype != torch.bfloat16 for x in (q, k, v, kwargs["g"]))
                or tuple(kwargs["beta"].shape) != shape[:-1]
                or any(kwargs.get(key) is not True for key in (
                    "use_gate_in_kernel", "use_beta_sigmoid_in_kernel", "use_qk_l2norm_in_kernel", "safe_gate"))
                or kwargs.get("lower_bound") != -5.0 or kwargs.get("initial_state") is not None
                or kwargs.get("output_final_state") is not False or kwargs.get("disable_recompute", False)):
            raise ValueError("unexpected native chunk_kda kernel inputs")
        counts[phase]["chunk_kda"] += 1
        return kda(q, k, v, **kwargs)

    def signed_backward(*args, **kwargs):
        result = backward(*args, **kwargs)
        counts[phase]["signed_backward"] += 1
        return result

    with (patch.object(model, "_in_proj_to_attn_inputs", producer),
          patch.object(model, "gated_delta_rule", core),
          patch.object(chunk, "chunk_kda_bwd", signed_backward)):
        schedule(torch, model, inputs, cotangents, advance=advance,
                 receipt=lambda name, x: numerics.update({name: tensor_receipt(torch, x, name)}))
    if counts != expected_calls(arm):
        raise ValueError(f"unexpected forward/backward call counts: {counts}")
    return counts, numerics


def worker(output, source, index):
    """Run one fresh process on the shared GPU and private probe JIT cache."""
    import torch

    arm = ARMS[index]
    if (torch.__version__, torch.version.hip, torch.cuda.device_count()) != ("2.14.0+rocm10.1.0", "7.16.26385", 1):
        raise ValueError("expected one image-910 ROCm GPU")
    target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if target != "gfx942":
        raise ValueError(f"expected gfx942, got {target}")
    torch.cuda.set_device(0)
    chunk = select_kda()
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120))
    state, model = build_kda(torch, arm)
    try:
        verify_imports(source, chunk)
        rng = torch.Generator(device="cuda").manual_seed(1701)
        inputs, cotangents = [[torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16, generator=rng)
                              for _ in range(IN_FLIGHT)] for _ in range(2)]
        fixtures = {f"{kind}:{i}": tensor_receipt(torch, value, kind)
                    for kind, tensors in (("input", inputs), ("cotangent", cotangents))
                    for i, value in enumerate(tensors)}
        fixtures.update({f"param:{name}": tensor_receipt(torch, value, name)
                         for name, value in model.named_parameters() if value.requires_grad})
        print(f"arm {index} {arm}: verified imports and built KDA, tracing {IN_FLIGHT} forwards", flush=True)
        calls, numerics = exercise(torch, model, chunk, arm, inputs, cotangents)
        write_json(output / f"evidence-{index}.json", dict(
            arm=arm, target=target, shape=list(SHAPE), modules=modules(arm), in_flight=IN_FLIGHT,
            delay_wgrad_compute=False, module_sha256=RELEASE["sha256"], calls=calls,
            conv_backend=model._conv1d_backend, fixtures=fixtures, numerics=numerics))
        for _ in range(WARMUPS):
            schedule(torch, model, inputs, cotangents)
        print(f"arm {index} {arm}: verified replay and warmed, collecting {SAMPLES} samples", flush=True)
        samples = []
        for _ in range(SAMPLES):
            samples.append(schedule(torch, model, inputs, cotangents))
            write_json(output / f"samples-{index}.json", samples)
    finally:
        state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def measurements(sample):
    """Convert one phase-split sample to absolute and entry-relative metrics."""
    entry, retained, forward, backward = (sample[key] for key in (
        "entry", "retained", "forward_peak", "backward_peak"))
    if (any(len(p) != 2 or not 0 < p[0] <= p[1] for p in (entry, retained, forward, backward))
            or not all(e <= r <= f and r <= b for e, r, f, b in zip(entry, retained, forward, backward, strict=True))
            or not all(math.isfinite(sample[k]) and sample[k] > 0 for k in ("forward_seconds", "backward_seconds"))):
        raise ValueError("invalid phase measurement")
    result = {key: sample[key] for key in ("forward_seconds", "backward_seconds")}
    result["total_seconds"] = result["forward_seconds"] + result["backward_seconds"]
    for i, kind in enumerate(("allocated", "reserved")):
        result.update({
            f"entry_{kind}": entry[i], f"forward_end_{kind}": retained[i],
            f"forward_retained_{kind}": retained[i] - entry[i],
            f"forward_peak_{kind}": forward[i], f"forward_peak_above_entry_{kind}": forward[i] - entry[i],
            f"backward_peak_{kind}": backward[i], f"backward_peak_above_entry_{kind}": backward[i] - retained[i],
            f"combined_peak_{kind}": max(forward[i], backward[i]),
        })
    return result


def verdict(evidence, samples):
    """Require comparable evidence; do not require a speedup or memory saving to pass."""
    if tuple(r["arm"] for r in evidence) != ARMS or len(samples) != len(ARMS):
        raise ValueError("unexpected ABBA sequence")
    ref = evidence[0]
    ordinary = {f"{k}:{i}" for k in ("input", "cotangent") for i in range(IN_FLIGHT)}
    derivatives = {f"{k}:{i}" for k in ("output", "input_grad") for i in range(IN_FLIGHT)}
    medians, parity = [], []
    entries = set()
    for arm, report, trials in zip(ARMS, evidence, samples, strict=True):
        fixtures, numerics = report["fixtures"], report["numerics"]
        parameters = set(fixtures) - ordinary
        if (report["modules"] != modules(arm) or report["calls"] != expected_calls(arm)
                or report["shape"] != list(SHAPE) or report["in_flight"] != IN_FLIGHT
                or report["target"] != "gfx942" or report["module_sha256"] != RELEASE["sha256"]
                or report["delay_wgrad_compute"] is not False or report["conv_backend"] not in ("cuda", "triton")
                or report["conv_backend"] != ref["conv_backend"] or len(trials) != SAMPLES
                or not ordinary <= fixtures.keys() or not parameters
                or not all(name.startswith("param:") for name in parameters)
                or set(numerics) != derivatives | {f"param_grad:{name[6:]}" for name in parameters}):
            raise ValueError("incomplete or incomparable KDA evidence")
        rows = [measurements(sample) for sample in trials]
        entries.update(row["entry_allocated"] for row in rows)
        medians.append({key: median(row[key] for row in rows) for key in rows[0]})
        parity.append({label: sorted(k for k in current.keys() | baseline.keys()
                                     if current.get(k) != baseline.get(k))
                       for label, current, baseline in (("fixtures", fixtures, ref["fixtures"]),
                                                        ("numerics", numerics, ref["numerics"]))})
    equal = not any(names for report in parity for names in report.values())
    matched = len(entries) == 1
    differences = [{key: medians[b][key] - medians[a][key] for key in medians[a]}
                   for a, b in ((0, 1), (3, 2))]
    return dict(status="passed" if equal and matched else "failed",
                scope="isolated Megatron KDA; no distributed fit or throughput claim",
                bitwise_equal=equal, differences_from_first_arm=parity, matched_entry_allocated=matched,
                median_by_arm=medians, paired_B_minus_A=differences)


def train(output, source, entrypoint, args):
    """Execute the fixed single-rank comparison without training, checkpointing or publication."""
    frozen = json.loads((output.parent / "resolved.json").read_text())
    if (entrypoint != ENTRYPOINT or args or os.environ.get("WORLD_SIZE") != "1"
            or os.environ.get("RANK") != "0" or frozen["image"]["sha256"] != IMAGE
            or frozen["source"]["commit"] != COMMIT or frozen["procedure"].get("auxiliary_source") != RELEASE):
        raise ValueError("unexpected frozen KDA experiment inputs")
    for index in range(len(ARMS)):
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                        str(output), str(source.resolve()), str(index)], cwd=source, check=True)


def check(run):
    """Retain paired deltas even if the bitwise parity or entry comparison fails."""
    output = run / "output"
    evidence, samples = ([json.loads((output / f"{kind}-{i}.json").read_text()) for i in range(len(ARMS))]
                         for kind in ("evidence", "samples"))
    result = verdict(evidence, samples)
    write_json(output / "checks.json", result)
    if result["status"] != "passed":
        raise ValueError("KDA recompute parity or matched-entry check failed")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["train", output, source, entrypoint, *args]:
            train(Path(output), Path(source), entrypoint, args)
        case ["worker", output, source, index] if index in ("0", "1", "2", "3"):
            worker(Path(output), Path(source), int(index))
        case ["check", run]:
            check(Path(run))
        case _:
            raise SystemExit("unexpected KDA recompute validator invocation")
