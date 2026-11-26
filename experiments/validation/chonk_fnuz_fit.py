"""Observe two cold full-layout Chonk updates and one synchronous save."""

from functools import wraps
import json
import math
import os
from pathlib import Path
import sys
from time import monotonic

from apertus_amd.artifacts import write_json
from apertus_amd.entrypoint import train


WORLD, STEPS, BATCH, MICRO = 64, 2, 4096, 128
LAYOUT = "Ettt|(tttttt|)*2,(ttttt|)*2,(tttttt|)*2,tttttL"
GEOMETRY = dict(tp=1, pp=4, vpp=2, ep=16, etp=1, cp=1, dp=16, edp=1)
SETTINGS = dict(
    num_layers=42, hidden_size=3584, ffn_hidden_size=8960, padded_vocab_size=200064,
    num_experts=256, moe_router_topk=8, moe_latent_size=1792, moe_ffn_hidden_size=1920,
    moe_layer_freq=[0] * 3 + [1] * 39, linear_attention_freq=[1, 1, 1, 0] * 10 + [1, 0],
    pipeline_model_parallel_layout=LAYOUT, seq_length=4096, micro_batch_size=2, global_batch_size=BATCH,
    train_iters=STEPS, save_interval=STEPS, ckpt_format="torch_dist", async_save=False, load=None,
    log_device_memory_used=False, lr_warmup_iters=100, lr_decay_iters=59604, seed=42, mock_data=True,
    bf16=True, fp8=None, optimizer="md_decoupling", use_layer_wise_distributed_optimizer=True,
    optimizer_cpu_offload=False, overlap_param_gather=False, overlap_p2p_comm=True,
    overlap_grad_reduce=True, align_grad_reduce=True, overlap_moe_expert_parallel_comm=True,
    delay_wgrad_compute=True, gradient_accumulation_fusion=False, cuda_graph_impl="none",
    moe_use_fp8_dispatch=True, moe_fp8_dispatch_fnuz=True, moe_resident_fp8_fnuz=True,
    moe_use_offloading_experts=False, moe_shared_expert_overlap=False,
    moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="deepep",
    recompute_granularity="selective", recompute_modules=["layernorm", "linear_attn", "qkv", "shared_experts"],
)
CONFIG_FIELDS = dict(microbatch_group_size_per_vp_stage=4, overlap_p2p_comm=True,
                     overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True,
                     moe_fp8_dispatch_fnuz=True, moe_resident_fp8_fnuz=True)
TRANSPORT = dict(dispatch=[20, 36, 288, 20, 140], combine=[20, 4, 288, 12, 132],
                 nvl_bytes=64_974_592, rdma_bytes=63_350_400)
LOSS_RANKS = range(48, WORLD)


def memory(torch):
    """Report process allocator peaks and separate point-in-time HIP device usage."""
    cuda = torch.cuda
    record = dict(allocated=cuda.memory_allocated(), peak_allocated=cuda.max_memory_allocated(),
                  reserved=cuda.memory_reserved(), peak_reserved=cuda.max_memory_reserved(),
                  capacity=cuda.get_device_properties(cuda.current_device()).total_memory)
    try:
        free, total = cuda.mem_get_info()
        if not 0 <= free <= total or total <= 0:
            raise ValueError(f"invalid HIP memory info: {free}/{total}")
        record["hip_snapshot"] = {"free": free, "total": total}
    except (RuntimeError, OSError, ValueError) as exc:
        record["hip_snapshot_error"] = str(exc)
    return record


def transport():
    """Read live EP16 buffer sizes and config using the qualified backend API."""
    from megatron.core.transformer.moe import fused_a2a

    buf = fused_a2a._buffer
    if buf is None or buf.group.size() != GEOMETRY["ep"]:
        raise ValueError("no live EP16 transport buffer")
    fields = ("num_sms", "num_max_nvl_chunked_send_tokens", "num_max_nvl_chunked_recv_tokens",
              "num_max_rdma_chunked_send_tokens", "num_max_rdma_chunked_recv_tokens")
    dispatch, combine = buf.get_dispatch_config(16), buf.get_combine_config(16)
    return dict(dispatch=[int(getattr(dispatch, key)) for key in fields],
                combine=[int(getattr(combine, key)) for key in fields],
                nvl_bytes=buf.num_nvl_bytes, rdma_bytes=buf.num_rdma_bytes)


def verify_update(row, update, rank):
    """Reject invalid numerics, reruns, or a partial measured update."""
    if (row["update"] != update or row["consumed_before"] != (update - 1) * BATCH
            or row["microbatches"] != MICRO or any(row[k] for k in
                ("skip", "reruns", "discarded", "forced_save", "early_exit"))
            or not all(math.isfinite(row[k]) and row[k] > 0 for k in
                       ("grad_norm", "seconds", "schedule_seconds"))):
        raise ValueError(f"rank {rank}: failed full-layout update {update}")
    losses = row["losses"]
    if set(losses) != ({"lm loss"} if rank in LOSS_RANKS else set()) or not all(
        math.isfinite(v) for v in losses.values()
    ):
        raise ValueError(f"rank {rank}: invalid loss evidence")


def observe(record, checkpoint, started):
    """Time the pinned train-step and checkpoint boundaries without changing their work."""
    import torch
    from torch._dynamo.utils import counters
    from torch._inductor import config as inductor_config, metrics
    from megatron.core import parallel_state as mpu
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving
    from megatron.core.rerun_state_machine import get_rerun_state_machine
    from megatron.training import training

    original, original_save = training.train_step, training.save_checkpoint_and_time

    @wraps(original)
    def step(forward, data, model, optimizer, scheduler, config, schedule, iteration=None):
        args = training.get_args()
        if not record["updates"]:
            names = dict(tp="tensor_model", pp="pipeline_model", vpp="virtual_pipeline_model",
                         ep="expert_model", etp="expert_tensor", cp="context", dp="data", edp="expert_data")
            geometry = {k: getattr(mpu, f"get_{name}_parallel_world_size")() for k, name in names.items()}
            effective = {k: getattr(args, k) for k in SETTINGS}
            config_values = {k: getattr(config, k) for k in CONFIG_FIELDS}
            if (geometry != GEOMETRY or effective != SETTINGS or config_values != CONFIG_FIELDS
                    or args.save != str(checkpoint) or args.main_grads_dtype != torch.float32
                    or args.params_dtype != torch.bfloat16 or not isinstance(optimizer, LayerWiseDistributedOptimizer)
                    or optimizer.async_allgather or len(model) != 2):
                raise ValueError("unexpected effective full-layout fit recipe")
            record.update(geometry=geometry, effective=effective, config=config_values,
                          startup_seconds=monotonic() - started, before_training=memory(torch))
            print(f"CHONK_FIT_FIRST_STEP rank={record['rank']} startup_seconds={record['startup_seconds']:.3f}",
                  flush=True)
        count = get_num_microbatches()
        if (iteration != len(record["updates"]) or iteration >= STEPS or count != MICRO
                or args.consumed_train_samples != iteration * BATCH
                or schedule is not forward_backward_pipelining_with_interleaving):
            raise ValueError("unexpected fit schedule or microbatch count")
        rerun = get_rerun_state_machine()
        previous = rerun.rerun_count
        observed = {}

        @wraps(schedule)
        def scheduled(*a, **kw):
            if observed:
                raise ValueError("rerun during fit")
            begin = monotonic()
            result = schedule(*a, **kw)
            observed.update(schedule_seconds=monotonic() - begin, after_schedule=memory(torch))
            return result

        begin = monotonic()
        result = original(forward, data, model, optimizer, scheduler, config, scheduled, iteration)
        loss, skip, save, exit_now, _, norm, _, _ = result
        row = dict(update=iteration + 1, consumed_before=args.consumed_train_samples, microbatches=count,
                   skip=int(skip), reruns=rerun.rerun_count - previous, discarded=rerun.discarded_batches,
                   forced_save=bool(save), early_exit=bool(exit_now), losses={k: float(v) for k, v in loss.items()},
                   grad_norm=float(norm) if norm is not None else math.nan, seconds=monotonic() - begin,
                   **observed)
        row["after_optimizer"] = memory(torch)
        row["compiler"] = dict(compile_threads=inductor_config.compile_threads,
                               generated_kernels=metrics.generated_kernel_count,
                               counters={k: dict(v) for k, v in counters.items()})
        record["updates"].append(row)
        verify_update(row, iteration + 1, record["rank"])
        print(f"CHONK_FIT_STEP rank={record['rank']} update={iteration + 1} seconds={row['seconds']:.3f}",
              flush=True)
        return result

    @wraps(original_save)
    def save(iteration, *args, **kwargs):
        if iteration != STEPS or len(record["updates"]) != STEPS or "checkpoint" in record:
            raise ValueError("unexpected checkpoint boundary")
        info = record["checkpoint"] = dict(iteration=iteration, before=memory(torch), completed=False)
        begin = monotonic()
        result = original_save(iteration, *args, **kwargs)
        info.update(seconds=monotonic() - begin, after=memory(torch), completed=True)
        return result

    training.train_step, training.save_checkpoint_and_time = step, save


def run_training(output, source, entrypoint, argv):
    """Run one fresh checkpoint probe and retain per-rank observations."""
    started = monotonic()
    checkpoint = output / "checkpoints"
    options = [arg.split("=", 1)[0] for arg in argv if arg.startswith("--")]
    rejected = {"--load", "--finetune", "--pretrained-checkpoint", "--async-save",
                "--log-device-memory-used", "--no-save-optim", "--save-iters"}
    if (int(os.environ["WORLD_SIZE"]) != WORLD or checkpoint.exists() or checkpoint.is_symlink()
            or argv[-2:] != ["--save", str(checkpoint)] or options.count("--save") != 1
            or rejected.intersection(options)):
        raise ValueError("fit requires 64 ranks and one fresh synchronous runner-owned checkpoint")
    rank = int(os.environ["RANK"])
    record = dict(rank=rank, completed=False, updates=[])
    try:
        train(output, source, entrypoint, argv, before_run=lambda: observe(record, checkpoint, started))
        import torch

        if len(record["updates"]) != STEPS or not record.get("checkpoint", {}).get("completed"):
            raise ValueError("missing full-layout updates or synchronous save")
        record["transport"] = transport()
        torch.cuda.synchronize()
        record["terminal_memory"] = memory(torch)
        torch.distributed.destroy_process_group()
        record.update(completed=True, child_seconds=monotonic() - started)
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / f"fit-rank-{rank}.json", record)


def verify_memory(row):
    """Validate allocator peaks; never substitute HIP snapshots for allocator evidence."""
    if not (0 < row["allocated"] <= row["peak_allocated"] <= row["peak_reserved"] <= row["capacity"]
            and row["allocated"] <= row["reserved"] <= row["peak_reserved"]):
        raise ValueError("invalid allocator evidence")
    if "hip_snapshot" in row:
        info = row["hip_snapshot"]
        if not 0 <= info["free"] <= info["total"] or info["total"] <= 0:
            raise ValueError("invalid HIP memory snapshot")
    elif not row.get("hip_snapshot_error"):
        raise ValueError("missing HIP memory snapshot result")


def inventory(root):
    """Inventory one synchronous torch_dist save without claiming tensor integrity."""
    tracker = root / "latest_checkpointed_iteration.txt"
    target = root / f"iter_{STEPS:07d}"
    if (root.is_symlink() or tracker.is_symlink() or tracker.read_text().strip() != str(STEPS)
            or target.is_symlink() or not target.is_dir()
            or {p.name for p in root.glob("iter_*")} != {target.name}):
        raise ValueError("expected only the terminal fit checkpoint")
    files = {}
    for p in target.iterdir():
        if p.is_symlink() or not p.is_file() or not (size := p.stat().st_size):
            raise ValueError(f"invalid checkpoint member {p.name}")
        files[p.name] = size
    shards = sorted(k for k in files if k.endswith(".distcp"))
    if not shards or not {"metadata.json", ".metadata", "common.pt"} <= files.keys():
        raise ValueError("incomplete torch_dist inventory")
    meta = json.loads((target / "metadata.json").read_text())
    if meta.get("sharded_backend") != "torch_dist" or meta.get("common_backend") != "torch":
        raise ValueError("unexpected checkpoint backend")
    return dict(status="inventoried", bytes=sum(files.values()), files=files, iteration=STEPS,
                tensor_integrity="not assessed", optimizer_integrity="not assessed", restoration="not assessed")


def check(run):
    """Accept only all-rank fit and save evidence; derive timing from observed boundaries."""
    output = run / "output"
    result = dict(status="failed", scope="cold full-layout fit; no restart, cache reuse or throughput verdict")
    try:
        if {p.name for p in output.glob("fit-rank-*.json")} != {f"fit-rank-{r}.json" for r in range(WORLD)}:
            raise ValueError("incomplete fit rank set")
        records = []
        for rank in range(WORLD):
            row = json.loads((output / f"fit-rank-{rank}.json").read_text())
            imports = json.loads((output / f"imports-{rank}.json").read_text())
            if (row["rank"] != rank or not row["completed"] or imports["verified_modules"] < 1
                    or row["geometry"] != GEOMETRY or row["config"] != CONFIG_FIELDS
                    or row["effective"] != SETTINGS or row["transport"] != TRANSPORT
                    or len(row["updates"]) != STEPS):
                raise ValueError(f"rank {rank}: incomplete fit evidence")
            with (output / f"rank-{rank}.log").open() as log:
                if not any("UCCL_EP_TRANSPORT=cxi initialized" in line for line in log):
                    raise ValueError(f"rank {rank}: no native CXI initialization")
            save = row["checkpoint"]
            if (save["iteration"] != STEPS or not save["completed"]
                    or not all(math.isfinite(x) and x > 0
                               for x in (row["startup_seconds"], row["child_seconds"], save["seconds"]))):
                raise ValueError(f"rank {rank}: missing checkpoint or timing")
            for mem in (row["before_training"], row["terminal_memory"], save["before"], save["after"]):
                verify_memory(mem)
            for n, update in enumerate(row["updates"], 1):
                verify_update(update, n, rank)
                verify_memory(update["after_schedule"])
                verify_memory(update["after_optimizer"])
            records.append(row)
        result.update(
            status="passed", ranks=WORLD, updates=STEPS, checkpoint=inventory(output / "checkpoints"),
            max_startup_seconds=max(r["startup_seconds"] for r in records),
            max_update_seconds=[max(r["updates"][i]["seconds"] for r in records) for i in range(STEPS)],
            max_schedule_seconds=[max(r["updates"][i]["schedule_seconds"] for r in records) for i in range(STEPS)],
            max_checkpoint_seconds=max(r["checkpoint"]["seconds"] for r in records),
            max_child_seconds=max(r["child_seconds"] for r in records),
            peak_allocated_bytes=max(r["terminal_memory"]["peak_allocated"] for r in records),
            peak_reserved_bytes=max(r["terminal_memory"]["peak_reserved"] for r in records),
            terminal_hip_snapshot_ranks=sum("hip_snapshot" in r["terminal_memory"] for r in records),
        )
    except Exception as exc:
        result.update(status="failed", error=str(exc))
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
