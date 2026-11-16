"""Native single-MoELayer EP16 component proxy, NOT full Chonk training.

Compare full ``moe`` checkpointing with ``moe_act`` using identical weights,
inputs, upstream gradients, and RNG states for two forward/backward microbatches.
No optimizer, DDP reduction, beta-finalization, restart, or reference kernels run.
The unfused native BF16 leaf gradients are summed into explicit FP32 buffers;
this checks local accumulation, not Megatron DDP hooks or fused FP32 GEMMs.
Router hooks and a restored public-routing call-through observe native compute.
Owned snapshots, hashing, and detached top-k controls synchronize/slow this probe;
it is a numerical diagnostic, never a performance measurement.
The train/check protocol provides committed-source staging and private caches.
"""

from copy import deepcopy
from contextlib import contextmanager, ExitStack
from functools import wraps
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import random
import sys

from apertus_amd.artifacts import sha256, write_json
from apertus_amd.config import option
from apertus_amd.entrypoint import verify_imports


SOURCE = "1072583a9632fd1902544f7eaea562e2578ef1de"
ROUTING_FIX_SOURCE = "e242f3c3c1136aeae498529bfde5e06bda9a7cca"
IMAGE = "cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0"
SCOPE = "single native MoELayer EP16 component proxy; not full Chonk"
MODES = ("moe", "moe_act")
MICROBATCHES = 2
RTOL, ATOL = 0.02, 1e-5
RMS_RTOL, RMS_ATOL = 2**-7, 1e-7
MAX_MISMATCH_ROWS = 16
REQUIRED = {
    "--num-layers": "42", "--hidden-size": "3584", "--ffn-hidden-size": "17280",
    "--num-attention-heads": "28", "--seq-length": "4096",
    "--moe-latent-size": "1792", "--moe-ffn-hidden-size": "1920",
    "--moe-shared-expert-intermediate-size": "1920", "--num-experts": "256",
    "--moe-router-topk": "8", "--expert-model-parallel-size": "16",
    "--tensor-model-parallel-size": "1", "--pipeline-model-parallel-size": "1",
    "--context-parallel-size": "1", "--expert-tensor-parallel-size": "1",
    "--micro-batch-size": "1", "--global-batch-size": "32", "--main-grads-dtype": "fp32",
    "--moe-router-load-balancing-type": "quantile_balancing",
    "--moe-router-quantile-balancing-method": "histogram",
    "--moe-router-quantile-balancing-num-bins": "1000", "--moe-router-violation-metrics": "mbs",
    "--moe-router-quantile-balancing-ema": "0", "--moe-router-score-function": "sigmoid",
    "--moe-router-dtype": "fp32", "--moe-router-topk-scaling-factor": "2.5",
    "--moe-token-dispatcher-type": "alltoall", "--transformer-impl": "transformer_engine",
    "--moe-aux-loss-coeff": "0.0", "--attention-dropout": "0", "--hidden-dropout": "0",
    "--recompute-granularity": "selective", "--seed": "42",
    "--init-method-std": "0.016703827619526525",
    "--train-iters": "1", "--save-interval": "1",
}


def validate(settings: dict) -> None:
    """Fail closed on settings outside this source/image/topology-specific component."""
    if settings["source"]["commit"] not in (SOURCE, ROUTING_FIX_SOURCE) or settings["image"]["sha256"] != IMAGE:
        raise ValueError("probe requires the selected source and image pins")
    if settings["procedure"]["kind"] != "train" or settings.get("compare_run"):
        raise ValueError("probe supports train/check only, not restart or compare-run")
    resources = settings["resources"]
    if (resources["nodes"], resources["gpus_per_node"]) != (4, 4):
        raise ValueError("probe requires four nodes with four GPUs each (EP16, PP1)")
    args = settings["args"]
    for flag, value in REQUIRED.items():
        if option(args, flag) != value:
            raise ValueError(f"probe requires {flag} {value}")
    for flag in ("--bf16", "--sssglu", "--disable-bias-linear", "--moe-grouped-gemm",
                 "--moe-permute-fusion", "--no-gradient-accumulation-fusion"):
        if flag not in args:
            raise ValueError(f"probe requires {flag}")
    if settings["environment"].get("TORCH_COMPILE_DISABLE") != "0":
        raise ValueError("probe retains native compilation, without a separate eager mode")
    if any(a.split("=", 1)[0] in {
        "--cpu-offloading", "--fine-grained-activation-offloading", "--moe-use-offloading-experts",
        "--moe-offload-main-grad", "--moe-offload-activations", "--use-te-op-fuser", "--fp8", "--fp4",
        "--moe-shared-expert-overlap", "--use-distributed-optimizer", "--use-layer-wise-distributed-optimizer",
    } for a in args):
        raise ValueError("probe does not support offload, alternate precision, fusion, or distributed optimizer")


def accepted(metrics: dict) -> bool:
    """Require finite elementwise AND RMS gates, including for zero references."""
    keys = ("max_tolerance_ratio", "rms_error", "reference_rms")
    return metrics["finite"] and all(
        metrics[k] is not None and math.isfinite(metrics[k]) for k in keys
    ) and metrics["max_tolerance_ratio"] <= 1 and (
        metrics["rms_error"] <= RMS_ATOL + RMS_RTOL * metrics["reference_rms"]
    )


def compare(torch, actual, expected, label: str, emit, *, contributions=1, scale=None) -> None:
    """Record pairwise native-mode errors; accumulation propagates per-MB budgets."""
    if actual is None or expected is None or actual.shape != expected.shape:
        emit({"label": label, "passed": False, "error": "missing tensor or shape mismatch"})
        return
    a, b = actual.detach().float(), expected.detach().float()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    metrics = {"finite": finite, "max_tolerance_ratio": None, "rms_error": None,
               "reference_rms": None, "max_abs": None}
    if finite:
        error = (a - b).abs()
        allowance = ATOL * contributions + RTOL * (b.abs() if scale is None else scale)
        metrics.update(max_tolerance_ratio=(error / allowance).max().item(),
                       max_abs=error.max().item(), rms_error=error.square().mean().sqrt().item(),
                       reference_rms=b.square().mean().sqrt().item())
    passed = accepted(metrics)
    metrics = {k: None if isinstance(v, float) and not math.isfinite(v) else v
               for k, v in metrics.items()}
    emit({"label": label, "passed": passed, "shape": list(a.shape), **metrics})


def exact(torch, actual, expected, label: str, emit) -> None:
    """Router/RNG/state comparisons have no floating-point tolerance."""
    emit({"label": label, "passed": bool(torch.equal(actual, expected))})


def tensor_hash(tensor) -> str:
    """Hash tensor identity for evidence without retaining large tensor artifacts."""
    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}:".encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_snapshot(torch, tracker) -> dict:
    """Clone all RNG families so one mode cannot consume the other's initial state."""
    import numpy as np

    return {"python": random.getstate(), "numpy": deepcopy(np.random.get_state()),
            "cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state().clone(),
            "tracker": {k: v.clone() for k, v in tracker.get_states().items()}}


def rng_restore(torch, tracker, state: dict) -> None:
    """Restore the component's identical per-microbatch RNG start, including TP/EP."""
    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(deepcopy(state["numpy"]))
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])
    tracker.set_states({k: v.clone() for k, v in state["tracker"].items()})


def rng_fingerprints(state: dict) -> dict:
    """Fingerprint observed RNG snapshots without retaining their serialized contents."""
    return {"python": hashlib.sha256(pickle.dumps(state["python"], protocol=5)).hexdigest(),
            "numpy": hashlib.sha256(pickle.dumps(state["numpy"], protocol=5)).hexdigest(),
            "cpu": tensor_hash(state["cpu"]), "cuda": tensor_hash(state["cuda"]),
            "tracker": {k: tensor_hash(v) for k, v in state["tracker"].items()}}


def expected_topk_branch(source: str, *, training: bool, grad_enabled: bool,
                         method: str, drop_priority: bool) -> int:
    """Describe the selected source's branch for diagnostics, never native routing."""
    if source not in (SOURCE, ROUTING_FIX_SOURCE):
        raise ValueError("unknown routing source")
    update = training and grad_enabled
    histogram = training and method == "histogram" and (grad_enabled or source == ROUTING_FIX_SOURCE)
    return 9 if histogram or (update and method in ("marin_histogram", "marin_exact")) or drop_priority else 8


@contextmanager
def capture_router_calls(torch, router, calls: list, tracker, *, source=SOURCE):
    """Observe real routing logits and inputs without changing native args/results/grad mode.

    Clones are queued on the caller's current stream before native consumers run.
    No hashing, top-k, or replacement routing occurs in the call-through wrapper.
    Restore the exact instance attribute/descriptor ownership even on exceptions.
    """
    original = router.routing
    had_override = "routing" in vars(router)
    previous_override = vars(router).get("routing")
    handle = None

    def before(module, inputs):
        calls.append({"input": inputs[0].detach().clone(), "weight": module.weight.detach().clone(),
                      "beta": module.qb_beta.detach().clone(), "rng": rng_snapshot(torch, tracker),
                      "grad_enabled": torch.is_grad_enabled(), "training": module.training})

    @wraps(original)
    def routing(*args, **kwargs):
        call = calls[-1]
        logits = args[0] if args else kwargs["logits"]
        call["logits"] = logits.detach().clone()
        call["routing_grad_enabled"] = torch.is_grad_enabled()
        config = router.config
        drop_priority = config.moe_expert_capacity_factor is not None and config.moe_token_drop_policy == "probs"
        call["topk_branch"] = expected_topk_branch(
            source, training=router.training, grad_enabled=call["routing_grad_enabled"],
            method=config.moe_router_quantile_balancing_method, drop_priority=drop_priority)
        return original(*args, **kwargs)

    try:
        router.routing = routing
        handle = router.register_forward_pre_hook(before)
        yield
    finally:
        try:
            if handle is not None:
                handle.remove()
        finally:
            if had_override:
                router.routing = previous_override
            elif "routing" in vars(router):
                delattr(router, "routing")


def tensor_delta(torch, actual, expected) -> dict:
    """Summarize exact equality and maximum delta without an acceptance tolerance."""
    a, b = actual.detach(), expected.detach()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    maximum = (a.float() - b.float()).abs().max().item() if finite else None
    return {"exact_equal": bool(torch.equal(a, b)), "finite": finite,
            "max_abs": maximum if maximum is None or math.isfinite(maximum) else None,
            "different_elements": int((a != b).sum().item())}


def mismatch_summary(torch, mask) -> dict:
    """Count/hash all differing token indices; do not serialize an unbounded row list."""
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    return {"different_rows": indices.numel(), "token_indices_sha256": tensor_hash(indices)}


def topk_controls(torch, logits, beta) -> dict:
    """Run repeated native Torch top-k on detached copies, never in actual MoE compute."""
    logits = logits.detach().reshape(-1, logits.shape[-1]).float()
    sigmoid = torch.sigmoid(logits)
    biased = sigmoid - beta.detach().float()
    values9, indices9 = biased.topk(9, dim=1)
    indices8 = biased.topk(8, dim=1).indices
    repeat8 = biased.topk(8, dim=1).indices
    repeat9 = biased.topk(9, dim=1).indices

    def mask(indices):
        return torch.zeros_like(biased, dtype=torch.bool).scatter(1, indices, True)

    map8, map9 = mask(indices8), mask(indices9[:, :8])
    return {"logits": logits, "sigmoid": sigmoid, "biased": biased, "values9": values9,
            "indices8": indices8, "indices9": indices9, "map8": map8, "map9": map9,
            "gap": values9[:, 7] - values9[:, 8],
            "repeat8_exact": bool(torch.equal(indices8, repeat8)),
            "repeat9_exact": bool(torch.equal(indices9, repeat9)),
            "repeat8_set_equal": bool(torch.equal(map8, mask(repeat8))),
            "repeat9prefix_set_equal": bool(torch.equal(map9, mask(repeat9[:, :8])))}


def routing_diagnosis(calls: dict, comparisons: dict, *, all_changed_rows_tied: bool) -> str:
    """Attribute only fully observed, repeatable boundary-tie branch differences."""
    observed = all(pair["logits"]["exact_equal"] and pair["logits"]["bitwise_equal"] and pair["logits"]["finite"] and
                   pair["same_observed_inputs"] for pair in comparisons.values())
    stable = all(call["repeat8_exact"] and call["repeat9_exact"] and
                 call["actual_matches_branch"] for call in calls.values())
    changed = comparisons["original_vs_replay"]["actual_expert_sets"]["different_rows"] > 0
    if observed and stable and changed and all_changed_rows_tied and (
        calls["original"]["topk_branch"], calls["replay"]["topk_branch"], calls["candidate"]["topk_branch"]
    ) == (8, 9, 9):
        return "observed_deterministic_topk_boundary_branch_difference"
    if observed and stable and all(pair["actual_expert_sets"]["different_rows"] == 0 for pair in comparisons.values()):
        return "no_expert_set_difference_observed"
    return "unresolved_same_mode_repeat_required"


def routing_row(raw_logits: list, sigmoid: list, biased: list, selected: list, top8: list, top9: list) -> dict:
    """Format one detached mismatch row, preserving complete scores and all boundary ties."""
    eighth, ninth = biased[top9[7]], biased[top9[8]]

    def finite(values):
        return [value if math.isfinite(value) else None for value in values]

    return {"raw_logits": finite(raw_logits), "sigmoid_scores": finite(sigmoid),
            "sigmoid_biased_scores": finite(biased), "actual_selected8": selected,
            "top8": top8, "top9": top9, "top9_prefix8": top9[:8],
            "eighth_ninth_scores": finite([eighth, ninth]),
            "eighth_ninth_gap": eighth - ninth if math.isfinite(eighth) and math.isfinite(ninth) else None,
            "tied_at_eighth": [i for i, score in enumerate(biased) if score == eighth],
            "tied_at_ninth": [i for i, score in enumerate(biased) if score == ninth]}


def routing_diagnostics(torch, calls: dict, observations: dict) -> dict:
    """Summarize all original/replay/candidate calls and bound raw rows to sixteen."""
    if len(calls["moe"]) != 2 or len(calls["moe_act"]) != 1 or len(observations["moe"]) != 2 or len(observations["moe_act"]) != 1:
        raise ValueError("diagnostics require original/replay/candidate router captures")
    entries = {"original": calls["moe"][0], "replay": calls["moe"][1], "candidate": calls["moe_act"][0]}
    actual = {"original": observations["moe"][0], "replay": observations["moe"][1],
              "candidate": observations["moe_act"][0]}
    controls, summaries = {}, {}
    for name, call in entries.items():
        control = controls[name] = topk_controls(torch, call["logits"], call["beta"])
        selected = actual[name][1]
        branch = control["map8"] if call["topk_branch"] == 8 else control["map9"]
        summaries[name] = {key: call[key] for key in ("grad_enabled", "routing_grad_enabled", "training", "topk_branch")}
        summaries[name].update(
            logits_shape=list(call["logits"].shape), logits_dtype=str(call["logits"].dtype),
            input_sha256=tensor_hash(call["input"]), weight_sha256=tensor_hash(call["weight"]),
            beta_sha256=tensor_hash(call["beta"]), logits_sha256=tensor_hash(call["logits"]),
            actual_routing_sha256=tensor_hash(selected), rng=rng_fingerprints(call["rng"]),
            actual_matches_branch=bool(torch.equal(selected, branch)),
            top8_vs_top9prefix=mismatch_summary(torch, (control["map8"] != control["map9"]).any(dim=1)),
            tied_boundary=mismatch_summary(torch, control["gap"] == 0),
            **{key: control[key] for key in ("repeat8_exact", "repeat9_exact", "repeat8_set_equal", "repeat9prefix_set_equal")})

    mismatch = torch.zeros(controls["original"]["logits"].shape[0], device=entries["original"]["logits"].device, dtype=torch.bool)
    comparisons = {}
    for left, right in (("original", "replay"), ("original", "candidate"), ("replay", "candidate")):
        a, b = controls[left], controls[right]
        set_mask = (actual[left][1] != actual[right][1]).any(dim=1)
        probability_mask = (actual[left][0] != actual[right][0]).any(dim=1)
        logit_mask = (a["logits"] != b["logits"]).any(dim=1)
        mismatch |= set_mask | probability_mask | logit_mask
        comparisons[f"{left}_vs_{right}"] = {
            "logits": {**tensor_delta(torch, a["logits"], b["logits"]),
                       "bitwise_equal": summaries[left]["logits_sha256"] == summaries[right]["logits_sha256"]},
            "probabilities": tensor_delta(torch, actual[left][0], actual[right][0]),
            "actual_expert_sets": {**mismatch_summary(torch, set_mask),
                "different_assignments": int((actual[left][1] != actual[right][1]).sum().item())},
            "observed_input_equality": {key: summaries[left][key] == summaries[right][key]
                for key in ("input_sha256", "weight_sha256", "beta_sha256", "rng")}}
        comparisons[f"{left}_vs_{right}"]["same_observed_inputs"] = all(
            comparisons[f"{left}_vs_{right}"]["observed_input_equality"].values())
    for name, control in controls.items():
        branch = control["map8"] if entries[name]["topk_branch"] == 8 else control["map9"]
        mismatch |= (actual[name][1] != branch).any(dim=1)
    changed = (actual["original"][1] != actual["replay"][1]).any(dim=1)
    all_tied = bool((controls["original"]["gap"][changed] == 0).all())
    indices = torch.nonzero(mismatch, as_tuple=False).flatten()
    selected_rows = indices[:MAX_MISMATCH_ROWS].tolist()

    rows = []
    for index in selected_rows:
        row = {"token_index": index, "calls": {}}
        for name, control in controls.items():
            row["calls"][name] = routing_row(
                control["logits"][index].tolist(), control["sigmoid"][index].tolist(),
                control["biased"][index].tolist(),
                torch.nonzero(actual[name][1][index], as_tuple=False).flatten().tolist(),
                control["indices8"][index].tolist(), control["indices9"][index].tolist())
        rows.append(row)
    diagnosis = routing_diagnosis(summaries, comparisons, all_changed_rows_tied=all_tied)
    return {"calls": summaries, "comparisons": comparisons, "diagnosis": diagnosis,
            "score_provenance": "raw logits/beta are owned native-call snapshots; sigmoid/biased scores and top-k controls are reconstructed ONLY on detached copies; actual_selected8 comes from native routing-map snapshots",
            "attribution_scope": "expert-set routing only; unexplained output-only differences still need same-mode repeats",
            "same_mode_repeat": "not performed; if unresolved or output-only differences remain, repeat the SAME mode twice, resetting identical initialized weights, ALL router buffers (including nonpersistent), input/dy, and Python/NumPy/CPU/CUDA/tracker RNG before each pass",
            "all_original_replay_changed_rows_tied": all_tied,
            "mismatch_rows": rows, "row_limit": MAX_MISMATCH_ROWS,
            "mismatch_row_count": indices.numel(), "mismatch_indices_sha256": tensor_hash(indices),
            "omitted_row_count": max(0, indices.numel() - MAX_MISMATCH_ROWS),
            "omitted_indices_sha256": tensor_hash(indices[MAX_MISMATCH_ROWS:])}


def router_snapshot(router) -> dict:
    """Include nonpersistent state, excluding uninitialized future metric sample slots."""
    count = int(router.expert_load_sample_count.item())
    return {k: (v[:count] if k in {"mbs_expert_load_samples", "seq_expert_load_samples"} else v
                ).detach().clone() for k, v in router.named_buffers()}


def required_labels(parameters) -> set[str]:
    """Define acceptance coverage independently of the report's claimed status."""
    labels = {"initial-weights-identical", "moe/no-weight-updates", "moe_act/no-weight-updates"}
    for mb in range(MICROBATCHES):
        prefix = f"mb{mb}"
        labels.update(f"{prefix}/{key}" for key in ("forward", "input-gradient", "probabilities", "routing-map",
                                                   "rng/cpu", "rng/cuda", "rng/host", "forward-pre-backward"))
        for state in ("qb_beta", "qb_histogram", "local_tokens_per_expert", "expert_load_sample_count"):
            labels.add(f"{prefix}/router-state/{state}")
        for mode in MODES:
            labels.update(f"{prefix}/{mode}/{key}" for key in (
                "router-once", "fixed-beta", "top8", "metric-once", "local-load-once", "output-unmutated"))
        labels.update(f"{prefix}/moe/replay-{key}" for key in ("probabilities", "routing-map"))
        for name in parameters:
            labels.update(f"{prefix}/{key}/{name}" for key in ("weight-gradient", "accumulated-gradient"))
    return labels


def check_router_trace(trace: list[dict], mode: str, microbatch: int) -> bool:
    """Full MoE counts in replay; activation-only counts in the original forward."""
    enabled = [False, True] if mode == "moe" else [True]
    totals = [4096 * microbatch, 4096 * (microbatch + 1)] if mode == "moe" else [4096 * (microbatch + 1)]
    return [r["grad_enabled"] for r in trace] == enabled and [
        r["histogram_rows_per_expert"] for r in trace
    ] == [[count] * 256 for count in totals]


def state_update_routing(observations: list, trace: list[dict]):
    """Use this mode's single grad-enabled map for native local-load accounting.

    The pinned router updates local_tokens_per_expert only with grad enabled.
    Full-MoE's original no-grad map is an independent equivalence gate, not the
    counter oracle: replay can disagree with it, which must still fail routing.
    """
    if len(observations) != len(trace):
        raise ValueError("router observations and trace lengths differ")
    enabled = [i for i, row in enumerate(trace) if row["grad_enabled"]]
    if len(enabled) != 1:
        raise ValueError("local-load accounting requires exactly one grad-enabled router call")
    return observations[enabled[0]][1]


def validate_native_config(config, activation, microbatches: int) -> None:
    """Reject effective native modes that would change the selected computation."""
    if microbatches != MICROBATCHES or config.activation_func is not activation or any(
        getattr(config, key) for key in ("gradient_accumulation_fusion", "cpu_offloading",
                                        "fine_grained_activation_offloading", "moe_use_offloading_experts",
                                        "moe_shared_expert_overlap", "fp8", "fp4")
    ):
        raise ValueError("effective native config has an unsupported activation/offload/fusion mode")
    if not config.bias_activation_fusion or config.use_te_activation_func or config.cuda_graph_impl != "none":
        raise ValueError("probe requires the native fused SSSGLU path without TE activation substitution or CUDA graphs")


def run_probe(emit, record: dict, *, source: Path, expected: dict) -> None:
    """Construct only two native MoELayers; execute paired local gradients with EP collectives."""
    import torch
    import transformer_engine
    from megatron.core import tensor_parallel
    from megatron.core.activations import sssglu_act
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import build_module
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    verify_imports(source, expected)
    args = get_args()
    config = core_transformer_config_from_args(args)
    validate_native_config(config, sssglu_act, get_num_microbatches())
    record.update(torch=torch.__version__, transformer_engine=transformer_engine.__version__,
                  device=torch.cuda.get_device_name(), probe_sha256=sha256(Path(__file__)),
                  environment={k: os.environ.get(k) for k in (
                      "NVTE_USE_GROUPED_GEMM_TRITON", "NVTE_GROUPED_LINEAR_SINGLE_PARAM",
                      "TORCH_COMPILE_DISABLE", "PYTORCH_HIP_ALLOC_CONF", "BEVERIN_COMM_WARMUP")})
    spec = get_moe_module_spec(use_te=True, num_experts=256, moe_grouped_gemm=True)
    models = {}
    record["native_selection"] = {}
    for mode in MODES:
        selected = deepcopy(config)
        selected.recompute_modules = [mode]
        model = build_module(spec, config=selected, layer_number=4).cuda().train()
        model.set_layer_number(4)
        if type(model) is not MoELayer or type(model.experts) is not TEGroupedMLP or model.num_local_experts != 16:
            raise ValueError("native spec did not select MoELayer/TEGroupedMLP with sixteen local experts")
        if model.moe_layer_recompute != (mode == "moe") or model.experts.activation_recompute != (mode == "moe_act"):
            raise ValueError("native recomputation selection differs from requested mode")
        record["native_selection"][mode] = {
            "modules": {name: f"{type(module).__module__}.{type(module).__qualname__}" for name, module in (
                ("layer", model), ("experts", model.experts), ("shared_experts", model.shared_experts),
                ("dispatcher", model.token_dispatcher), ("router", model.router),
                ("expert_fc1", model.experts.linear_fc1), ("expert_fc2", model.experts.linear_fc2))},
            "activation": f"{model.experts.activation_func.__module__}.{model.experts.activation_func.__name__}",
            "recompute_modules": selected.recompute_modules,
            "moe_layer_recompute": model.moe_layer_recompute,
            "activation_recompute": model.experts.activation_recompute,
            "local_experts": model.num_local_experts,
        }
        models[mode] = model
    baseline, candidate = (models[mode] for mode in MODES)
    params = {mode: dict(model.named_parameters()) for mode, model in models.items()}
    if params["moe"].keys() != params["moe_act"].keys():
        raise ValueError("parameter names differ across native modes")
    if any(p.dtype != torch.bfloat16 or not p.is_cuda for values in params.values() for p in values.values()):
        raise ValueError("component requires GPU-resident BF16 parameters")
    with torch.no_grad():
        for name, original in params["moe"].items():
            params["moe_act"][name].copy_(original)
        original_buffers = dict(baseline.named_buffers())
        for name, target in candidate.named_buffers():
            target.copy_(original_buffers[name])
    initial = {mode: {name: tensor_hash(p) for name, p in values.items()} for mode, values in params.items()}
    emit({"label": "initial-weights-identical", "passed": initial["moe"] == initial["moe_act"]})
    record["initial_weights"] = initial["moe"]
    record["parameters"] = {name: {"shape": list(p.shape), "dtype": str(p.dtype)}
                            for name, p in params["moe"].items()}
    totals = {mode: {name: torch.zeros_like(p, dtype=torch.float32) for name, p in values.items()}
              for mode, values in params.items()}
    absolute_sum = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in params["moe"].items()}
    expected_local_load = {mode: torch.zeros(256, device="cuda", dtype=torch.float32) for mode in MODES}
    verify_imports(source, expected)
    tracker = tensor_parallel.get_cuda_rng_tracker()
    observations = {mode: [] for mode in MODES}
    traces = {mode: [] for mode in MODES}
    captured_calls = {mode: [] for mode in MODES}
    observers = ExitStack()

    def observe(mode):
        def hook(router, inputs, result):
            probabilities, routing = result
            counts = router.qb_histogram.sum(dim=-1).tolist()
            traces[mode].append({"grad_enabled": torch.is_grad_enabled(),
                                 "histogram_rows_per_expert": counts})
            # Clones detach evidence without altering the values used by dispatch/checkpoint.
            observations[mode].append((probabilities.detach().clone(), routing.detach().clone()))
        return hook

    record["microbatches"] = []
    record["observer_notice"] = "Owned clones, fingerprints and detached top-k controls may synchronize/slow this component diagnostic; not performance data."
    try:
        for mode, model in models.items():
            observers.callback(model.router.register_forward_hook(observe(mode)).remove)
            observers.enter_context(capture_router_calls(
                torch, model.router, captured_calls[mode], tracker, source=record["source"]))
        for microbatch in range(MICROBATCHES):
            generator = torch.Generator(device="cuda").manual_seed(1000 + 16 * microbatch + args.rank)
            x = torch.randn(4096, 1, 3584, device="cuda", dtype=torch.bfloat16, generator=generator)
            dy = torch.randn(x.shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
            start = rng_snapshot(torch, tracker)
            reference = None
            reference_pre = None
            states, ends = {}, {}
            evidence = {"index": microbatch, "input_sha256": tensor_hash(x),
                        "upstream_sha256": tensor_hash(dy), "router_trace": {},
                        "expected_pre_forward_rng": rng_fingerprints(start), "pre_forward": {}, "outputs": {}}
            for mode, model in models.items():
                observations[mode].clear()
                traces[mode].clear()
                captured_calls[mode].clear()
                rng_restore(torch, tracker, start)
                value = x.detach().clone().requires_grad_(True)
                evidence["pre_forward"][mode] = {"rng": rng_fingerprints(rng_snapshot(torch, tracker)),
                    "input_sha256": tensor_hash(value), "upstream_sha256": tensor_hash(dy)}
                before = evidence["pre_forward"][mode]
                before["matches_expected"] = {"rng": before["rng"] == evidence["expected_pre_forward_rng"],
                    "input": before["input_sha256"] == evidence["input_sha256"],
                    "upstream": before["upstream_sha256"] == evidence["upstream_sha256"]}
                result, bias = model(value)
                # Capture the FINAL layer output before checkpoint backward can reuse any storage.
                pre_backward = result.detach().clone()
                if bias is not None:
                    raise ValueError("unexpected MoELayer output bias")
                result.backward(dy)
                post_backward = result.detach().clone()
                prefix = f"mb{microbatch}/{mode}"
                exact(torch, post_backward, pre_backward, f"{prefix}/output-unmutated", emit)
                evidence["outputs"][mode] = {"pre_backward_sha256": tensor_hash(pre_backward),
                    "post_backward_sha256": tensor_hash(post_backward),
                    "pre_vs_post": tensor_delta(torch, post_backward, pre_backward)}
                states[mode] = router_snapshot(model.router)
                ends[mode] = rng_snapshot(torch, tracker)
                evidence["router_trace"][mode] = deepcopy(traces[mode])
                emit({"label": f"{prefix}/router-once", "passed": check_router_trace(traces[mode], mode, microbatch)})
                emit({"label": f"{prefix}/metric-once", "passed":
                      int(model.router.expert_load_sample_count.item()) == microbatch + 1})
                exact(torch, model.router.qb_beta, torch.zeros_like(model.router.qb_beta), f"{prefix}/fixed-beta", emit)
                emit({"label": f"{prefix}/top8", "passed": all(bool((r.sum(-1) == 8).all())
                      for _, r in observations[mode])})
                if mode == "moe":
                    reference = (result.detach().clone(), value.grad.detach().clone())
                    reference_pre = pre_backward
                    for i in range(1, len(observations[mode])):
                        for j, name in enumerate(("probabilities", "routing-map")):
                            exact(torch, observations[mode][i][j], observations[mode][0][j],
                                  f"{prefix}/replay-{name}", emit)
                else:
                    compare(torch, pre_backward, reference_pre, f"mb{microbatch}/forward-pre-backward", emit)
                    # Preserve the ORIGINAL post-backward forward gate as well as the new pre-backward gate.
                    compare(torch, result, reference[0], f"mb{microbatch}/forward", emit)
                    compare(torch, value.grad, reference[1], f"mb{microbatch}/input-gradient", emit)
                    for j, name in enumerate(("probabilities", "routing-map")):
                        exact(torch, observations[mode][0][j], observations["moe"][0][j],
                              f"mb{microbatch}/{name}", emit)
                counted_map = state_update_routing(observations[mode], traces[mode])
                expected_local_load[mode].add_(counted_map.sum(dim=0).float())
                exact(torch, model.router.local_tokens_per_expert, expected_local_load[mode],
                      f"{prefix}/local-load-once", emit)
                for name, p in params[mode].items():
                    if p.grad is None:
                        raise ValueError(f"missing native gradient: {mode}/{name}")
                    totals[mode][name].add_(p.grad.float())
                    if mode == "moe":
                        absolute_sum[name].add_(p.grad.float().abs())
                    else:
                        compare(torch, p.grad, params["moe"][name].grad,
                                f"mb{microbatch}/weight-gradient/{name}", emit)
                        compare(torch, totals[mode][name], totals["moe"][name],
                                f"mb{microbatch}/accumulated-gradient/{name}", emit,
                                contributions=microbatch + 1, scale=absolute_sum[name])
                del value, result, pre_backward, post_backward
            for name, state in states["moe"].items():
                exact(torch, states["moe_act"][name], state, f"mb{microbatch}/router-state/{name}", emit)
            for family in ("cpu", "cuda"):
                exact(torch, ends["moe_act"][family], ends["moe"][family], f"mb{microbatch}/rng/{family}", emit)
            for name, state in ends["moe"]["tracker"].items():
                exact(torch, ends["moe_act"]["tracker"][name], state, f"mb{microbatch}/rng/{name}", emit)
            left, right = ends["moe"]["numpy"], ends["moe_act"]["numpy"]
            emit({"label": f"mb{microbatch}/rng/host", "passed":
                  left[0] == right[0] and left[1].tobytes() == right[1].tobytes() and left[2:] == right[2:] and
                  ends["moe"]["python"] == ends["moe_act"]["python"]})
            evidence["router_state_sha256"] = {mode: {k: tensor_hash(v) for k, v in state.items()}
                                                 for mode, state in states.items()}
            # All detached alternate top-k work occurs AFTER both modes' native forward/backward.
            evidence["router_diagnostics"] = routing_diagnostics(torch, captured_calls, observations)
            record["microbatches"].append(evidence)
            for model in models.values():
                model.zero_grad(set_to_none=True)
        for mode, values in params.items():
            final = {name: tensor_hash(p) for name, p in values.items()}
            emit({"label": f"{mode}/no-weight-updates", "passed": final == initial[mode]})
        torch.cuda.synchronize()
        record["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    finally:
        observers.close()


def train(output: Path, source: Path, entrypoint: str, argv: list[str], *, run=Path("/apertus")) -> None:
    """Use the retained source/import verifier without running a GPT model or optimizer."""
    settings = json.loads((run / "resolved.json").read_text())
    validate(settings)
    selected_source = settings["source"]["commit"]
    manifest = json.loads((run / "source/manifest.json").read_text())
    if manifest["selection"]["commit"] != selected_source or entrypoint != settings["entrypoint"] or \
            sha256(source / entrypoint) != manifest["files"][entrypoint]:
        raise ValueError("selected native entrypoint/source identity mismatch")
    source = source.resolve()
    sys.path.insert(0, str(source))
    sys.argv = [str(source / entrypoint), *argv]
    rank = int(os.environ["RANK"])
    record = {"status": "failed", "scope": SCOPE, "rank": rank, "source": selected_source, "image": IMAGE,
              "policy": {"rtol": RTOL, "atol": ATOL, "rms_rtol": RMS_RTOL, "rms_atol": RMS_ATOL},
              "accumulation": "local FP32 sum of native unfused BF16 leaf gradients; no DDP reduction",
              "not_checked": ["full Chonk", "KDA", "PP4/VPP", "optimizer/hypersphere initialization",
                              "DDP/finalize_model_grads", "QB beta batch-boundary update", "restart", "cache reuse"],
              "checks": []}
    # Exclusive creation avoids replacing evidence if manually invoked twice.
    with (output / f"moe-recompute-{rank}.json").open("x") as summary, \
            (output / f"moe-recompute-{rank}.jsonl").open("x", buffering=1) as stream:
        def emit(check):
            record["checks"].append(check)
            stream.write(json.dumps(check, allow_nan=False) + "\n")

        try:
            from megatron.training.initialize import initialize_megatron
            # Verify image-vs-source imports BEFORE native initialization or probe compute.
            verify_imports(source, manifest["files"])
            initialize_megatron()
            verify_imports(source, manifest["files"])
            run_probe(emit, record, source=source, expected=manifest["files"])
            if not record["checks"] or not all(row["passed"] for row in record["checks"]):
                raise AssertionError("native MoE recompute checks failed; inspect per-rank JSONL")
            record["status"] = "passed"
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            try:
                count = verify_imports(source, manifest["files"])
                write_json(output / f"imports-{rank}.json", {"verified_modules": count, "source": str(source)})
            except BaseException as error:
                record.update(status="failed", import_error=str(error))
                raise
            finally:
                summary.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


def check(run: Path) -> None:
    """Require complete all-rank numerical evidence; missing/partial runs cannot pass."""
    try:
        settings = json.loads((run / "resolved.json").read_text())
        validate(settings)
        for rank in range(16):
            value = json.loads((run / f"output/moe-recompute-{rank}.json").read_text())
            imports = json.loads((run / f"output/imports-{rank}.json").read_text())
            labels = [row["label"] for row in value["checks"]]
            if value["status"] != "passed" or value["scope"] != SCOPE or value["rank"] != rank or \
                    value["source"] != settings["source"]["commit"] or value["image"] != IMAGE or \
                    [mb["index"] for mb in value["microbatches"]] != [0, 1] or \
                    not value["checks"] or not all(row["passed"] for row in value["checks"]) or \
                    not value["parameters"] or not required_labels(value["parameters"]).issubset(labels) or \
                    len(labels) != len(set(labels)) or \
                    imports["verified_modules"] <= 0:
                raise ValueError(f"incomplete or failed component evidence: rank {rank}")
        result = {"status": "passed", "scope": SCOPE, "ranks": 16, "microbatches": MICROBATCHES,
                  "source": settings["source"]["commit"], "image": IMAGE,
                  "numerical_equivalence": "native moe vs moe_act within recorded tolerances",
                  "optimizer_validation": "not performed", "restart_validation": "not performed"}
    except Exception as error:
        write_json(run / "output/checks.json", {"status": "failed", "scope": SCOPE, "error": str(error)})
        raise
    write_json(run / "output/checks.json", result)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "train":
        train(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        check(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: moe_recompute_probe.py train OUTPUT SOURCE ENTRYPOINT [ARGS...] | check RUN")
