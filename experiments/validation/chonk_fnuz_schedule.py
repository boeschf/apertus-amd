"""Validate two integrated resident FNUZ training updates on the reduced schedule."""

from functools import wraps
import json
import math
import os
from pathlib import Path
import sys

from apertus_amd.artifacts import write_json
from apertus_amd.entrypoint import train


WORLD, UPDATES, MICRO, BATCH, LOCAL_EXPERTS = 32, 2, 8, 256, 16
LAYOUT = "Et|t|t|tL"
GEOMETRY = dict(tp=1, pp=2, vpp=2, ep=16, etp=1, cp=1, dp=16, edp=1)
LOSS_RANKS = range(WORLD // GEOMETRY["pp"], WORLD)
SETTINGS = {
    "num_layers": 4, "hidden_size": 3584, "ffn_hidden_size": 8960,
    "linear_attention_freq": [1, 1, 1, 0], "moe_layer_freq": 1,
    "kda_legacy_gate_out_proj_bias": True, "add_bias_linear": False,
    "num_experts": 256, "moe_router_topk": 8,
    "moe_latent_size": 1792, "moe_ffn_hidden_size": 1920, "padded_vocab_size": 200064,
    "seq_length": 4096, "micro_batch_size": 2, "global_batch_size": BATCH, "train_iters": UPDATES,
    "bf16": True, "fp8": None, "optimizer": "md_decoupling",
    "check_grad_norm": True, "seed": 42, "mock_data": True, "use_tp_pp_dp_mapping": False,
    "use_layer_wise_distributed_optimizer": True, "overlap_param_gather": False,
    "overlap_grad_reduce": True, "align_grad_reduce": True, "overlap_p2p_comm": True,
    "overlap_moe_expert_parallel_comm": True, "delay_wgrad_compute": True,
    "gradient_accumulation_fusion": False, "cuda_graph_impl": "none",
    "moe_use_fp8_dispatch": True, "moe_fp8_dispatch_fnuz": True, "moe_resident_fp8_fnuz": True,
    "moe_use_offloading_experts": False, "moe_shared_expert_overlap": False,
    "moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": "deepep",
    "recompute_granularity": "selective",
    "recompute_modules": ["layernorm", "linear_attn", "qkv", "shared_experts"],
    "pipeline_model_parallel_layout": LAYOUT, "load": None, "save": None,
}
EXPECTED_CHUNK = {"parameters": 2 * LOCAL_EXPERTS, "ready_events": 2 * LOCAL_EXPERTS * MICRO}
EXPERT_PARAMS = EXPECTED_CHUNK["parameters"] * GEOMETRY["vpp"]
TRANSPORT = {
    "dispatch": [20, 36, 288, 20, 140], "combine": [20, 4, 288, 12, 132],
    "nvl_bytes": 64_974_592, "rdma_bytes": 63_350_400,
}


def transport_state():
    """Read the same effective EP16 config and live buffer as the component gate."""
    from megatron.core.transformer.moe import fused_a2a

    buffer = fused_a2a._buffer
    if buffer is None or buffer.group.size() != GEOMETRY["ep"]:
        raise ValueError("missing native EP16 transport buffer")
    fields = ("num_sms", "num_max_nvl_chunked_send_tokens", "num_max_nvl_chunked_recv_tokens",
              "num_max_rdma_chunked_send_tokens", "num_max_rdma_chunked_recv_tokens")
    dispatch, combine = (getter(GEOMETRY["ep"])
                         for getter in (buffer.get_dispatch_config, buffer.get_combine_config))
    return dict(dispatch=[int(getattr(dispatch, k)) for k in fields],
                combine=[int(getattr(combine, k)) for k in fields],
                nvl_bytes=buffer.num_nvl_bytes, rdma_bytes=buffer.num_rdma_bytes)


def verify_chunks(model, torch):
    """Require two single-expert-layer DDP chunks with resident FP32 buffers."""
    chunks = []
    for chunk in model:
        adapters = [m._resident_fp8 for m in chunk.modules() if getattr(m, "_resident_fp8", None)]
        if len(adapters) != 1:
            raise ValueError("expected one resident expert adapter per schedule chunk")
        expert, = adapters
        params = tuple(expert.params)
        if [tuple(p.shape) for p in params] != (
            [(3840, 1792)] * LOCAL_EXPERTS + [(1792, 1920)] * LOCAL_EXPERTS
        ) or {id(p) for p in params} != {
            id(p) for p in chunk._resident_fp8_delayed_params
        }:
            raise ValueError("resident expert parameters are not registered with delayed DDP")
        for p in params:
            grad = p.main_grad
            if (not p.is_cuda or p.dtype != torch.bfloat16 or grad.dtype != torch.float32
                    or grad.device != p.device or grad.shape != p.shape):
                raise ValueError("non-resident or incorrectly typed expert parameters/gradients")
        chunks.append((chunk, expert, params))
    if len(chunks) != 2:
        raise ValueError("expected VPP2")
    return chunks


def verify_optimizer_grads(optimizer, chunks, torch):
    """Check the resident MD master weights receive the exact finalized FP32 views."""
    from megatron.core.optimizer.md_decoupling import MDDecoupling
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    pending = {id(p) for _, _, params in chunks for p in params}
    for chain in optimizer.chained_optimizers:
        if not isinstance(chain, Float16OptimizerWithFloat16Params):
            continue
        for models, mains in zip(chain.float16_groups, chain.fp32_from_float16_groups, strict=True):
            for p, main in zip(models, mains, strict=True):
                if not getattr(p, "resident_fp8_delayed_wgrad", False):
                    continue
                expected, actual = p.main_grad, main.grad
                if (id(p) not in pending or not isinstance(chain.optimizer, MDDecoupling)
                        or main.dtype != torch.float32 or main.device != p.device
                        or actual is None or actual.dtype != expected.dtype or actual.device != expected.device
                        or actual.size() != expected.size() or actual.stride() != expected.stride()
                        or actual.data_ptr() != expected.data_ptr()):
                    raise ValueError("optimizer did not receive the resident FP32 main_grad view exactly once")
                pending.remove(id(p))
    if pending:
        raise ValueError(f"optimizer missing {len(pending)} resident FP32 gradients")
    return sum(len(params) for _, _, params in chunks)


def install_observer(record):
    """Observe the actual schedule and optimizer boundaries without replacing their operations."""
    import torch
    from megatron.core import parallel_state as mpu
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving
    from megatron.core.rerun_state_machine import get_rerun_state_machine
    from megatron.training import training

    original = training.train_step

    @wraps(original)
    def observe(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config,
                forward_backward_func, iteration=None):
        args = training.get_args()
        if not record.get("effective"):
            geometry = dict(tp=mpu.get_tensor_model_parallel_world_size(),
                            pp=mpu.get_pipeline_model_parallel_world_size(),
                            vpp=mpu.get_virtual_pipeline_model_parallel_world_size(),
                            ep=mpu.get_expert_model_parallel_world_size(),
                            etp=mpu.get_expert_tensor_parallel_world_size(),
                            cp=mpu.get_context_parallel_world_size(),
                            dp=mpu.get_data_parallel_world_size(),
                            edp=mpu.get_expert_data_parallel_world_size())
            effective = {key: getattr(args, key) for key in SETTINGS}
            record.update(effective=effective, geometry=geometry)
            if (geometry != GEOMETRY or effective != SETTINGS or args.consumed_train_samples
                    or args.main_grads_dtype != torch.float32 or args.params_dtype != torch.bfloat16
                    or not isinstance(optimizer, LayerWiseDistributedOptimizer) or optimizer.async_allgather):
                raise ValueError("unexpected combined-schedule topology or effective recipe")
        if (get_num_microbatches() != MICRO or iteration != len(record["updates"])
                or args.consumed_train_samples != iteration * BATCH
                or forward_backward_func is not forward_backward_pipelining_with_interleaving
                or config.microbatch_group_size_per_vp_stage != GEOMETRY["pp"]
                or not config.overlap_p2p_comm or not config.overlap_moe_expert_parallel_comm
                or not config.delay_wgrad_compute or not config.finalize_model_grads_func
                or len(config.grad_sync_func or []) != GEOMETRY["vpp"]):
            raise ValueError("unexpected schedule microbatch or iteration count")
        last_stage = bool(mpu.is_pipeline_last_stage(ignore_virtual=True))
        if last_stage != (record["rank"] in LOSS_RANKS):
            raise ValueError("unexpected physical loss stage")
        chunks = verify_chunks(model, torch)
        before = [expert._weight_version() for _, expert, _ in chunks]
        rerun = get_rerun_state_machine()
        reruns = rerun.rerun_count
        completed = []
        prepared = []

        @wraps(forward_backward_func)
        def scheduled(*a, **kw):
            if completed:
                raise ValueError("unexpected rerun in combined-schedule rehearsal")
            result = forward_backward_func(*a, **kw)
            for (chunk, expert, params), version in zip(chunks, before, strict=True):
                counts = chunk._resident_fp8_grad_counts
                if (expert._outstanding or expert._pending or expert._version != version
                        or len(counts) != len(params) or any(counts.get(p) != MICRO for p in params)
                        or not chunk._resident_fp8_grad_streams or any(p.grad is not None for p in params)
                        or any(bg.grad_reduce_handle is not None
                               for bg in chunk.bucket_groups + chunk.expert_parallel_bucket_groups)):
                    raise ValueError("resident expert generation, delayed drain, or DDP readiness mismatch")
                completed.append({"parameters": len(params), "ready_events": sum(counts.values())})
            return result

        prepare_grad_norm = optimizer.prepare_grad_norm

        def observe_prepare():
            if prepared or completed != [EXPECTED_CHUNK] * GEOMETRY["vpp"]:
                raise ValueError("optimizer preparation did not follow exactly one finalized schedule")
            inf, norm = prepare_grad_norm()
            # Check before train_step can repair or rerun this gradient set.
            if inf or norm is None or not math.isfinite(norm) or norm <= 0:
                raise ValueError(f"invalid optimizer-input gradient norm: {norm} (found_inf={inf})")
            prepared.append(verify_optimizer_grads(optimizer, chunks, torch))
            return inf, norm

        optimizer.prepare_grad_norm = observe_prepare
        try:
            result = original(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler,
                              config, scheduled, iteration)
        finally:
            del optimizer.prepare_grad_norm
        loss, skipped, checkpoint, exit_now, _, grad_norm, _, _ = result
        versions_advanced = all(
            all(new_id == p_id and new_version > old_version and new_ptr == old_ptr
                for (p_id, old_version, old_ptr), (new_id, new_version, new_ptr)
                in zip(version, expert._weight_version(), strict=True))
            for (_, expert, _), version in zip(chunks, before, strict=True)
        )
        step = dict(update=iteration + 1, skipped=int(skipped), reruns=rerun.rerun_count - reruns,
                    discarded=rerun.discarded_batches, checkpoint=bool(checkpoint), early_exit=bool(exit_now),
                    grad_norm=float(grad_norm) if grad_norm is not None else None,
                    losses={k: float(v) for k, v in loss.items()},
                    versions_advanced=versions_advanced, chunks=completed, optimizer_grad_views=prepared,
                    finite_resident_weights=bool(torch.stack([
                        torch.isfinite(p).all() for _, _, params in chunks for p in params
                    ]).all()))
        record["updates"].append(step)
        verify_step(step, iteration + 1, last_stage)
        return result

    training.train_step = observe


def verify_step(step, update, last_stage):
    """Reject missing/non-finite observations, skips, and lifecycle mismatches."""
    norm = step["grad_norm"]
    if (step["update"] != update or any(step[k] for k in (
            "skipped", "reruns", "discarded", "checkpoint", "early_exit"))
            or norm is None or not math.isfinite(norm) or norm <= 0
            or not step["versions_advanced"] or step["chunks"] != [EXPECTED_CHUNK] * GEOMETRY["vpp"]
            or step["optimizer_grad_views"] != [EXPERT_PARAMS] or not step["finite_resident_weights"]):
        raise ValueError(f"unqualified update: {step}")
    if set(step["losses"]) != ({"lm loss"} if last_stage else set()) or any(
        not math.isfinite(v) for v in step["losses"].values()
    ):
        raise ValueError(f"unqualified loss on update {update}: {step['losses']}")


def run_training(output, source, entrypoint, argv):
    """Run the signed entrypoint and retain local evidence even on failure."""
    if int(os.environ["WORLD_SIZE"]) != WORLD or any(
        opt.split("=", 1)[0] in {"--save", "--load", "--pretrained-checkpoint", "--finetune"}
        for opt in argv
    ):
        raise ValueError("schedule probe requires 32 ranks and no checkpoint I/O")
    rank = int(os.environ["RANK"])
    record = dict(rank=rank, completed=False, distributed_teardown=False, updates=[])
    try:
        train(output, source, entrypoint, argv, before_run=lambda: install_observer(record))
        import torch

        if len(record["updates"]) != UPDATES or not torch.distributed.is_initialized():
            raise ValueError("expected two completed updates before distributed teardown")
        record["transport"] = transport_state()
        if record["transport"] != TRANSPORT:
            raise ValueError(f"unexpected EP16 transport config: {record['transport']}")
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()
        record["distributed_teardown"] = True
        record["completed"] = True
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / f"schedule-rank-{rank}.json", record)


def check(run):
    """Accept only a complete 32-rank set with matching settings and two clean updates."""
    output = run / "output"
    result = {"status": "failed", "scope": "reduced schedule only; not fit, restart, or throughput"}
    try:
        expected_ranks = {f"schedule-rank-{rank}.json" for rank in range(WORLD)}
        if {p.name for p in output.glob("schedule-rank-*.json")} != expected_ranks:
            raise ValueError("expected exactly 32 schedule rank records")
        for rank in range(WORLD):
            record = json.loads((output / f"schedule-rank-{rank}.json").read_text())
            imports = json.loads((output / f"imports-{rank}.json").read_text())
            if (record["rank"] != rank or not record["completed"] or imports["verified_modules"] < 1
                    or not record["distributed_teardown"]
                    or record["effective"] != SETTINGS or record["geometry"] != GEOMETRY
                    or record["transport"] != TRANSPORT
                    or len(record["updates"]) != UPDATES):
                raise ValueError(f"incomplete or wrong schedule evidence from rank {rank}")
            for step, expected in zip(record["updates"], range(1, UPDATES + 1), strict=True):
                verify_step(step, expected, rank in LOSS_RANKS)
            with (output / f"rank-{rank}.log").open() as log:
                if not any("UCCL_EP_TRANSPORT=cxi initialized" in line for line in log):
                    raise ValueError(f"missing native CXI initialization from rank {rank}")
        if (output / "checkpoints").exists():
            raise ValueError("schedule probe must not write checkpoints")
        result.update(status="passed", ranks=WORLD, updates=UPDATES, loss_ranks=list(LOSS_RANKS))
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        write_json(output / "checks.json", result)


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"] and len(sys.argv) >= 6:
        run_training(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif sys.argv[1:2] == ["check"] and len(sys.argv) == 3:
        check(Path(sys.argv[2]))
    else:
        raise SystemExit("expected train or check")
