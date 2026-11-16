"""Native router-only GPU regression; no model, optimizer, or EP16 repeat.

The parent source is a diagnostic control, not a routing-equivalence acceptance.
Frozen captured logits are injected at the public native routing input, never by
replacing native routing, score functions, top-k, statistics, or vendor Torch.
"""

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

from apertus_amd.artifacts import sha256, write_json
from apertus_amd.config import option
from apertus_amd.entrypoint import verify_imports


BASE = "1072583a9632fd1902544f7eaea562e2578ef1de"
FIX = "e242f3c3c1136aeae498529bfde5e06bda9a7cca"
IMAGE = "cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0"
ROUTER = "megatron/core/transformer/moe/router.py"
ROUTER_SHA = {BASE: "41a9f702a138176ee9e199210756a94c894161d80f20f168c448d8e432da64a6",
              FIX: "facf8a03dff9fba668f2da4d1e617174854aadad5e5475006622a379e2de6c3b"}
SCOPE = "native TopKRouter.routing GPU component; not model/EP16/training equivalence"
ROWS, EXPERTS, TOPK, CAPTURED_INDEX = 4096, 256, 8, 1168
ROW_SHA = "521bfdb1a335545644b068e5796acff268cccd9402b7cbeed3ecbaa7198873e1"
ROW_PROVENANCE = {
    "job": 738351, "rank": 0, "microbatch": 1, "token": 1168,
    "report": "20261112T173406Z-375e6973/output/moe-recompute-0.json",
    "report_sha256": "12222043fcb795587bebd158cdc1f02ed5974b12fef927b33b1db27aa2fd324a",
    "source": BASE, "encoding": "base64 of 256 little-endian IEEE float32 raw logits",
    "row_sha256": ROW_SHA,
}
CAPTURED_ROW = (
    "Wlg/PyKibD/xuLu+H4kZva4FUz83l0q/oLdZP2t9n7+oYCC+SpewP+fiIL+KLvO+ufdBv936+D/cEsK/ClYQv/k8"
    "1b9WVji+rZonPhDR/L0N0Qg+nfY+v3pboj325og/Wkkdv2Q6iD8eA5c+D7VPv9K/9j33Wf6+atbbPhoK3T5aMXi/"
    "truNPt2eZb5yek+/F93cP/SQAD4ch9w++bHTvjwhnj/62sa/b0B7P9WVpb90WOg/PD/XPyqhUL60T+U+XVjHvyrI"
    "eb+XQxpAMLlQwASR/D6oUpK+96iIP8oWiD593uo+RL+dv6TYTL+dpSi/EIrqvnDCML+NVq++hZnCvwo0z740EwK/"
    "6E7tPrrMJ78gfai+tduhvid3S7+bguC/zqXdPRSmMr/AgAW/3rBWP4iIVz4WQTc98h4YPyXc9D0Avx++D3UMwEiD"
    "Yb5XgZw+OBsuvsL64T9Yg668DuQbwHFEoL6k5my8Y2jcP0IxXD4UBNs/7/jivwYq9L7wa1g/8JAdPbbRAL5dnoe/"
    "FD6WPxm+PD+6uQRASroDQC1umT3gJwk/ptsqP3puH7/URac+Z51nPq6fA7+QDJM/vDS2Ps4SBz5ElQi/W6kzP55F"
    "iD+bA7Y/4EXrPh7jhj8XQS0/FFycP+FrCL/K2tC+cHV+PK9vsD9iCXi/dma2vG8JhD9E2QI+ZJA/v3wnVD/ldNQ/"
    "JJHHPIgOeb6ns4S/Uyfsv2CA6r5cdqs/FHiEvr5MI0DODtm+sP9vPnU/6L/kEyE/JXHkP8mhKL9hc8g+8E/ePSd3"
    "KD9bBsw/Q3I+vyB2mb5rNP4+AOHDvwR8ML94nE8/03DGvy5Zkj7W4ZA+NsilvQ4w2z4echS+C4Wvv/0IEECo3IK/"
    "Mrq7PlLKOz/gplE8el6APzSK8T4vA52/Mt25vgRScD8WB+A/+ENSP/qpEcCS4oi/uOGXP1CuhbsE7nw/kJTQvHAJ"
    "DTwnbgbAHQeKvT195D7yxBc+RS42v+Ac7b74j1U/XT1rPw04fj+7I8s+xfKqvyzB0b+pD/u+bGDTP+T9uL5EcfA9"
    "5tVUP3jAjD6nlrc+vDQfPccdnT/rb66+YFs6Pfs48T+OAo+/1S6uP40h37/2ZYW/sPCfP/agX78tq7k/UhUsv5ID"
    "5T73Mie/H2PKv5EeDD6nNhE/18h6P7g2Hz7Kyq4+SkBAvzJ1vT87Nlc+6qwLP/a10r+CuDe/cljoP7Zihb8iaoQ/"
    "/2AJv9MNHj9TyJe+XxAGP3Sn4T7UNU+/sEvQvlnYYD6YdaO+miR0vyRIjz8lw4K/E+/qv4vmpj9o1xvANAD7ve2A"
    "ED9+mxQ+seuLP0b3r74zJnI/uDQnvFay7D6kyq69Fk0Ivw=="
)
REQUIRED = {
    "--num-layers": "42", "--hidden-size": "3584", "--num-experts": "256",
    "--moe-router-topk": "8", "--seq-length": "4096", "--micro-batch-size": "1",
    "--global-batch-size": "8", "--expert-model-parallel-size": "1",
    "--tensor-model-parallel-size": "1", "--pipeline-model-parallel-size": "1",
    "--context-parallel-size": "1", "--expert-tensor-parallel-size": "1",
    "--moe-router-load-balancing-type": "quantile_balancing",
    "--moe-router-quantile-balancing-method": "histogram",
    "--moe-router-quantile-balancing-num-bins": "1000", "--moe-router-quantile-balancing-ema": "0",
    "--moe-router-score-function": "sigmoid", "--moe-router-dtype": "fp32",
    "--moe-router-topk-scaling-factor": "2.5", "--moe-router-violation-metrics": "mbs",
    "--moe-aux-loss-coeff": "0.0", "--train-iters": "1", "--save-interval": "1",
}


def captured_row() -> tuple:
    """Decode and verify the immutable full captured row without external reads."""
    raw = base64.b64decode(CAPTURED_ROW, validate=True)
    if len(raw) != 4 * EXPERTS or hashlib.sha256(raw).hexdigest() != ROW_SHA:
        raise ValueError("captured raw-logit row checksum mismatch")
    return struct.unpack("<256f", raw)


def validate(settings: dict) -> None:
    """Fail closed outside the explicitly authorized one-node component protocol."""
    if settings["source"]["commit"] not in ROUTER_SHA or settings["image"]["sha256"] != IMAGE:
        raise ValueError("router probe requires the two exact source pins and selected image")
    if settings["procedure"]["kind"] != "train" or settings.get("compare_run"):
        raise ValueError("router probe supports train/check only")
    if (settings["resources"]["nodes"], settings["resources"]["gpus_per_node"]) != (1, 4):
        raise ValueError("router probe requires one node / four GPUs")
    for flag, expected in REQUIRED.items():
        if option(settings["args"], flag) != expected:
            raise ValueError(f"router probe requires {flag} {expected}")
    if settings["environment"].get("TORCH_COMPILE_DISABLE") != "0":
        raise ValueError("router probe retains native compilation")
    if any(a.split("=", 1)[0] in {"--moe-router-fusion", "--moe-expert-capacity-factor",
                                 "--fp8", "--fp4", "--enable-cuda-graph", "--use-distributed-optimizer"}
           for a in settings["args"]):
        raise ValueError("router probe rejects unselected native modes")
    captured_row()


def cases() -> tuple[list, list]:
    """Enumerate the bounded native matrix identically on every rank/source."""
    primary = [(f"histogram/{score}/pad{int(pad)}", score, pad)
               for score in ("sigmoid", "softmax") for pad in (False, True)]
    compatibility = []
    for score in ("sigmoid", "softmax"):
        for pad in (False, True):
            for capacity, policy in ((None, "probs"), (0.75, "probs"), (0.75, "position")):
                compatibility.append((f"eval/{score}/pad{int(pad)}/{capacity}/{policy}",
                                      score, "histogram", pad, capacity, policy, False))
            compatibility.append((f"train/histogram/{score}/pad{int(pad)}/drop-probs",
                                  score, "histogram", pad, 0.75, "probs", True))
        for method in ("average", "legacy_average", "marin_histogram", "marin_exact"):
            for capacity in (None, 0.75):
                compatibility.append((f"train/{method}/{score}/pad1/{capacity}/probs",
                                      score, method, True, capacity, "probs", True))
    return primary, compatibility


def required_labels() -> set:
    """Require complete matrix and accounting coverage independently of reported status."""
    labels = {"captured-row-identity", "native-geometry", "native-groups"}
    primary, compatibility = cases()
    for name, score, _ in primary:
        labels.update(f"{name}/{key}" for key in ("same-k-repeat", "non-tied-control"))
        if score == "sigmoid":
            labels.add(f"{name}/captured-sigmoid-tie")
        for mb in range(2):
            labels.update(f"{name}/mb{mb}/{key}" for key in (
                "no-grad-state", "no-grad-repeat", "branch-maps", "source-routing-contract",
                "non-tied-row-invariant", "histogram-once", "local-load-once", "metric-once",
                "fixed-beta", "finite", "top8"))
    for name, *_, training in compatibility:
        labels.add(f"{name}/no-grad-state")
        for grad in ((False, True) if training else (False,)):
            labels.update(f"{name}/grad{int(grad)}/{key}" for key in ("finite", "fixed-beta", "metrics"))
    return labels


def tensor_hash(tensor) -> str:
    """Fingerprint dtype, shape, and exact owned tensor bytes."""
    import torch

    value = tensor.detach().cpu().contiguous()
    result = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}:".encode())
    result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def snapshot(router) -> dict:
    """Clone initialized native state, including nonpersistent and Marin exact buffers."""
    count = int(router.expert_load_sample_count.item())
    result = {name: (value[:count] if name in {"mbs_expert_load_samples", "seq_expert_load_samples"}
                     else value).detach().clone() for name, value in router.named_buffers()}
    result["exact_rows"] = router._qb_exact_rows
    if router._qb_exact_margins is not None:
        result["exact_margins"] = router._qb_exact_margins[:router._qb_exact_rows].detach().clone()
        result["exact_valid"] = router._qb_exact_valid[:router._qb_exact_rows].detach().clone()
    for name in ("inference_mbs_expert_load_samples", "inference_seq_expert_load_samples"):
        result[name] = [value.detach().clone() for value in getattr(router, name)]
    return result


def state_hash(state: dict) -> dict:
    """Hash only initialized state; do not include unused metric/buffer capacity."""
    return {key: ([tensor_hash(item) for item in value] if isinstance(value, list)
                  else value if isinstance(value, int) else tensor_hash(value)) for key, value in state.items()}


def call(torch, router, logits, padding, grad: bool) -> tuple:
    """Call the unmodified native public method in the requested real grad mode."""
    with torch.set_grad_enabled(grad):
        probabilities, routing = router.routing(logits.detach().clone().requires_grad_(grad), padding_mask=padding)
    torch.cuda.synchronize()  # Also complete native asynchronous host margin copies.
    return probabilities.detach().clone(), routing.detach().clone()


def run_probe(emit, record: dict, *, source: Path, expected: dict) -> None:
    """Exercise the native router only, with initialized native groups and real GPU state."""
    import torch
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    verify_imports(source, expected)
    config = core_transformer_config_from_args(get_args())
    if config.moe_router_fusion or config.enable_cuda_graph or config.cuda_graph_impl != "none" or get_num_microbatches() != 2:
        raise ValueError("unexpected effective fused/graph/microbatch mode")
    groups = ProcessGroupCollection.use_mpu_process_groups(["tp", "cp", "tp_cp", "tp_dp_cp", "ep"])
    emit("native-groups", all(getattr(groups, key).size() == size for key, size in (
        ("tp", 1), ("cp", 1), ("tp_cp", 1), ("ep", 1), ("tp_dp_cp", 4))))
    row = torch.tensor(captured_row(), device="cuda", dtype=torch.float32)
    emit("captured-row-identity", hashlib.sha256(row.cpu().numpy().tobytes()).hexdigest() == ROW_SHA)
    logits = torch.linspace(-1, 1, EXPERTS, device="cuda", dtype=torch.float32).repeat(ROWS, 1)
    logits[CAPTURED_INDEX].copy_(row)
    padding = torch.zeros(ROWS, device="cuda", dtype=torch.bool)
    padding[16::17] = True
    record.update(torch=torch.__version__, device=torch.cuda.get_device_name(), row_provenance=ROW_PROVENANCE,
                  logits_sha256=tensor_hash(logits), primary={}, compatibility={},
                  observer_notice="detached controls and hashing synchronize; not a performance test")

    def make_router(score, method="histogram", capacity=None, policy="probs", training=True):
        selected = deepcopy(config)
        selected.moe_router_score_function = score
        selected.moe_router_quantile_balancing_method = method
        selected.moe_expert_capacity_factor = capacity
        selected.moe_token_drop_policy = policy
        selected.moe_router_inference_violation_metrics = ["mbs"]
        router = TopKRouter(config=selected, pg_collection=groups).cuda().train(training)
        router.set_layer_number(4)
        if type(router) is not TopKRouter or "routing" in vars(router) or "quantile_balancing" in vars(router):
            raise ValueError("unexpected native router method ownership")
        return router

    geometry = make_router("sigmoid")
    emit("native-geometry", tuple(geometry.weight.shape) == (256, 3584) and geometry.topk == TOPK)
    record["native_router"] = f"{type(geometry).__module__}.{type(geometry).__qualname__}"
    del geometry
    for name, score, pad in cases()[0]:
        router = make_router(score)
        mask = padding if pad else None
        valid = ROWS - int(padding.sum().item()) if pad else ROWS
        scores = torch.sigmoid(logits) if score == "sigmoid" else torch.softmax(logits, dim=-1)
        control8, control9 = scores.topk(8, dim=1), scores.topk(9, dim=1)
        map8 = torch.zeros_like(scores, dtype=torch.bool).scatter(1, control8.indices, True)
        map9 = torch.zeros_like(scores, dtype=torch.bool).scatter(1, control9.indices[:, :8], True)
        emit(f"{name}/same-k-repeat", torch.equal(control8.indices, scores.topk(8, dim=1).indices)
             and torch.equal(control9.indices, scores.topk(9, dim=1).indices))
        gap = control9.values[:, 7] - control9.values[:, 8]
        emit(f"{name}/non-tied-control", bool(gap[0] > 0))
        if score == "sigmoid":
            emit(f"{name}/captured-sigmoid-tie", bool(gap[CAPTURED_INDEX] == 0))
        observations = record["primary"][name] = {
            "captured_boundary_gap": gap[CAPTURED_INDEX].item(), "non_tied_boundary_gap": gap[0].item(),
            "control8_captured_indices": control8.indices[CAPTURED_INDEX].tolist(),
            "control9_captured_indices": control9.indices[CAPTURED_INDEX].tolist(),
            "control8_vs_9_different_rows": int((map8 != map9).any(dim=1).sum().item()), "calls": []}
        total_load = torch.zeros(EXPERTS, device="cuda", dtype=torch.float32)
        metric_samples = []
        first_histogram = None
        for mb in range(2):
            prefix = f"{name}/mb{mb}"
            before = snapshot(router)
            original = call(torch, router, logits, mask, False)
            repeated = call(torch, router, logits, mask, False)
            emit(f"{prefix}/no-grad-state", state_hash(snapshot(router)) == state_hash(before))
            emit(f"{prefix}/no-grad-repeat", all(torch.equal(a, b) for a, b in zip(original, repeated)))
            replay = call(torch, router, logits, mask, True)
            after = snapshot(router)
            old_map = map8 if record["source"] == BASE else map9
            emit(f"{prefix}/branch-maps", torch.equal(original[1], old_map) and torch.equal(replay[1], map9))
            same = all(torch.equal(a, b) for a, b in zip(original, replay))
            # Baseline diagnoses the existing branch; it is never labelled an equivalence pass.
            contract = same if record["source"] == FIX else torch.equal(original[1] != replay[1], map8 != map9)
            emit(f"{prefix}/source-routing-contract", contract)
            emit(f"{prefix}/non-tied-row-invariant", all(torch.equal(a[0], b[0]) for a, b in zip(original, replay)))
            increment = after["qb_histogram"] - before["qb_histogram"]
            if first_histogram is None:
                first_histogram = increment.clone()
            emit(f"{prefix}/histogram-once", torch.equal(increment, first_histogram)
                 and bool((increment.sum(-1) == valid).all())
                 and torch.equal(after["qb_histogram"], first_histogram * (mb + 1)))
            counted = replay[1] if mask is None else replay[1] & (~mask.unsqueeze(-1))
            loads = counted.sum(0).float()
            total_load.add_(loads)
            metric_samples.append(torch.cat((loads, loads.new_tensor([valid]))))
            emit(f"{prefix}/local-load-once", torch.equal(after["local_tokens_per_expert"], total_load))
            emit(f"{prefix}/metric-once", int(router.expert_load_sample_count.item()) == mb + 1
                 and torch.equal(router.mbs_expert_load_samples[:mb + 1], torch.stack(metric_samples)))
            emit(f"{prefix}/fixed-beta", bool((router.qb_beta == 0).all()))
            emit(f"{prefix}/finite", all(bool(torch.isfinite(value).all()) for value in (original[0], replay[0])))
            emit(f"{prefix}/top8", all(bool((value[1].sum(-1) == 8).all()) for value in (original, replay)))
            observations["calls"].append({"microbatch": mb, "grad_mode_equal": same,
                "different_routing_rows": int((original[1] != replay[1]).any(dim=1).sum().item()),
                "original_captured_selected": torch.nonzero(original[1][CAPTURED_INDEX]).flatten().tolist(),
                "replay_captured_selected": torch.nonzero(replay[1][CAPTURED_INDEX]).flatten().tolist(),
                "original": [tensor_hash(value) for value in original],
                "replay": [tensor_hash(value) for value in replay],
                "histogram_rows_per_expert": after["qb_histogram"].sum(-1).tolist(),
                "metric_count": int(router.expert_load_sample_count.item()), "state": state_hash(after)})
        del router

    for name, score, method, pad, capacity, policy, training in cases()[1]:
        router = make_router(score, method, capacity, policy, training)
        mask = padding if pad else None
        before = state_hash(snapshot(router))
        for grad in ((False, True) if training else (False,)):
            probs, routing = call(torch, router, logits, mask, grad)
            state = snapshot(router)
            prefix = f"{name}/grad{int(grad)}"
            if not grad:
                # Eval intentionally collects inference samples; every other state must remain unchanged.
                actual, reference = state_hash(state), dict(before)
                if not training:
                    actual.pop("inference_mbs_expert_load_samples")
                    reference.pop("inference_mbs_expert_load_samples")
                emit(f"{name}/no-grad-state", actual == reference)
            emit(f"{prefix}/finite", bool(torch.isfinite(probs).all()) and all(
                not hasattr(value, "is_floating_point") or not value.is_floating_point()
                or bool(torch.isfinite(value).all()) for value in state.values()))
            emit(f"{prefix}/fixed-beta", bool((router.qb_beta == 0).all()))
            emit(f"{prefix}/metrics", int(router.expert_load_sample_count.item()) == int(training and grad)
                 and len(router.inference_mbs_expert_load_samples) == int(not training))
            record["compatibility"][prefix] = {"probabilities": tensor_hash(probs),
                "routing_map": tensor_hash(routing), "state": state_hash(state)}
        del router
    verify_imports(source, expected)
    torch.cuda.synchronize()
    record["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()


def train(output: Path, source: Path, entrypoint: str, argv: list, *, run=Path("/apertus")) -> None:
    """Initialize native groups through the frozen train/check/import-verification protocol."""
    settings = json.loads((run / "resolved.json").read_text())
    validate(settings)
    commit = settings["source"]["commit"]
    manifest = json.loads((run / "source/manifest.json").read_text())
    if manifest["selection"]["commit"] != commit or entrypoint != settings["entrypoint"] or \
            sha256(source / entrypoint) != manifest["files"][entrypoint] or sha256(source / ROUTER) != ROUTER_SHA[commit]:
        raise ValueError("native entrypoint/router/source identity mismatch")
    source = source.resolve()
    sys.path.insert(0, str(source))
    sys.argv = [str(source / entrypoint), *argv]
    rank = int(os.environ["RANK"])
    record = {"status": "failed", "scope": SCOPE, "rank": rank, "source": commit, "image": IMAGE,
              "acceptance_kind": "fixed-source regression" if commit == FIX else "baseline diagnostic control",
              "native_equivalence": "router grad-mode equality only" if commit == FIX else "not established; baseline diagnostic",
              "checks": [],
              "probe_sha256": sha256(Path(__file__)), "row_provenance": ROW_PROVENANCE}
    with (output / f"router-tensor-{rank}.json").open("x") as summary, \
            (output / f"router-tensor-{rank}.jsonl").open("x", buffering=1) as stream:
        def emit(label, passed):
            row = {"label": label, "passed": bool(passed)}
            record["checks"].append(row)
            stream.write(json.dumps(row) + "\n")

        try:
            from megatron.training.initialize import initialize_megatron
            verify_imports(source, manifest["files"])
            initialize_megatron()
            verify_imports(source, manifest["files"])
            run_probe(emit, record, source=source, expected=manifest["files"])
            labels = [row["label"] for row in record["checks"]]
            if set(labels) != required_labels() or len(labels) != len(set(labels)) or \
                    not all(row["passed"] for row in record["checks"]):
                raise AssertionError("native router checks failed; no retry without review")
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
    """Require all four complete rank reports before the runtime can publish caches."""
    try:
        settings = json.loads((run / "resolved.json").read_text())
        validate(settings)
        for rank in range(4):
            value = json.loads((run / f"output/router-tensor-{rank}.json").read_text())
            imports = json.loads((run / f"output/imports-{rank}.json").read_text())
            labels = [row["label"] for row in value["checks"]]
            if value["status"] != "passed" or value["scope"] != SCOPE or value["rank"] != rank or \
                    value["source"] != settings["source"]["commit"] or value["image"] != IMAGE or \
                    value["probe_sha256"] != settings["validation"]["sha256"] or \
                    value["row_provenance"] != ROW_PROVENANCE or set(labels) != required_labels() or \
                    len(labels) != len(set(labels)) or not all(row["passed"] for row in value["checks"]) or \
                    imports["verified_modules"] <= 0:
                raise ValueError(f"incomplete or failed native router evidence: rank {rank}")
            primary, compatibility = cases()
            expected_compatibility = {f"{name}/grad{int(grad)}" for name, *_, training in compatibility
                                      for grad in ((False, True) if training else (False,))}
            if set(value["primary"]) != {name for name, *_ in primary} or \
                    set(value["compatibility"]) != expected_compatibility:
                raise ValueError(f"incomplete native router matrix: rank {rank}")
            for name, score, _ in primary:
                evidence = value["primary"][name]
                if [row["microbatch"] for row in evidence["calls"]] != [0, 1] or \
                        [row["metric_count"] for row in evidence["calls"]] != [1, 2] or \
                        (score == "sigmoid" and evidence["captured_boundary_gap"] != 0) or \
                        evidence["non_tied_boundary_gap"] <= 0 or \
                        (value["source"] == FIX and not all(row["grad_mode_equal"] for row in evidence["calls"])):
                    raise ValueError(f"invalid captured-row/accounting evidence: rank {rank}, {name}")
        result = {"status": "passed", "scope": SCOPE, "ranks": 4,
                  "source": settings["source"]["commit"], "image": IMAGE,
                  "acceptance_kind": "fixed-source regression" if settings["source"]["commit"] == FIX else "baseline diagnostic control",
                  "cross_source_compatibility": "requires separate paired host review",
                  "full_model_equivalence": "not tested"}
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
        raise SystemExit("usage: router_tensor_probe.py train OUTPUT SOURCE ENTRYPOINT [ARGS...] | check RUN")
