"""Observe the pinned Swiss-AI preparation without changing its training schedule."""

from datetime import datetime
from functools import wraps
import json
import math
import os
from pathlib import Path
import re
import resource
import statistics
import sys

from apertus_amd.artifacts import write_json
from apertus_amd.entrypoint import train


WORLD, STEPS, BATCH = 64, 60, 4096
WINDOWS = ((11, 26), (45, 60))
LAYOUT = "Ettt|(tttttt|)*2,(ttttt|)*2,(tttttt|)*2,tttttL"
EFFECTIVE_ARGS = {
    "padded_vocab_size": 200064, "ffn_hidden_size": 8960,
    "kda_legacy_gate_out_proj_bias": True, "add_bias_linear": False,
    "pipeline_model_parallel_layout": LAYOUT, "sequence_parallel": False,
    "micro_batch_size": 2, "global_batch_size": BATCH, "seq_length": 4096,
    "train_iters": STEPS, "lr_warmup_iters": 100, "lr_decay_iters": 59604,
    "ckpt_format": "torch_dist", "async_save": False, "save_interval": STEPS,
    "load": None, "recompute_granularity": "selective",
    "recompute_modules": ["layernorm", "linear_attn", "qkv", "shared_experts"],
    "optimizer": "md_decoupling", "use_layer_wise_distributed_optimizer": True,
    "use_distributed_optimizer": False, "overlap_param_gather": False,
    "moe_shared_expert_overlap": False, "moe_router_load_balancing_type": "quantile_balancing",
    "moe_router_quantile_balancing_method": "histogram", "moe_router_quantile_balancing_num_bins": 1000,
    "moe_router_quantile_balancing_ema": 0.0, "moe_aux_loss_coeff": 0.001,
    "overlap_grad_reduce": True, "overlap_moe_expert_parallel_comm": True,
    "delay_wgrad_compute": True, "overlap_p2p_comm": True,
    "gradient_accumulation_fusion": False, "cuda_graph_impl": "none",
    "moe_use_fp8_dispatch": False, "moe_use_offloading_experts": False,
    "moe_use_inplace_fp8_param": False, "cross_entropy_loss_fusion": True,
    "cross_entropy_fusion_impl": "native", "check_for_nan_in_loss_and_grad": False,
    "check_grad_norm": True, "check_grad_norm_threshold": 5.0,
    "rerun_mode": "validate_results", "rerun_strategy": "rerun_in_place",
}
EFFECTIVE_CONFIG = {"batch_p2p_comm": False, "overlap_p2p_comm": True,
                    "overlap_moe_expert_parallel_comm": True, "delay_wgrad_compute": True,
                    "gradient_accumulation_fusion": False}
GEOMETRY = {"tp": 1, "pp": 4, "vpp": 2, "ep": 16, "etp": 1, "cp": 1,
            "dp": 16, "edp": 1, "microbatches": 128}
UPDATE = re.compile(r"\[(?P<utc>[0-9 .:-]+)\]\s+iteration\s+(\d+)\s*/\s*(\d+)\s*\|")
CONFLICTING_ENV = {
    "APERTUS_INDUCTOR_POLICY", "BEVERIN_QKNORM_FLA", "BEVERIN_FUSED_SSSGLU",
    "AITER_CONFIG_GEMM_BF16", "UCCL_EP_DISPATCH_CONFIG", "UCCL_EP_COMBINE_CONFIG",
    "UCCL_EP_CPU_TIMEOUT_SECS", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTORCH_CUDA_ALLOC_CONF",
    "NCCL_GRAPH_REGISTER", "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "TE_HIPBLASLT_ALGO_LOAD",
}


def install_observer(output: Path, records: list, state: dict, compiler: list):
    """Observe the public train-step boundary, not autograd or kernel internals."""
    import torch
    from torch._dynamo.utils import counters
    from torch._inductor import config as inductor_config, metrics
    from megatron.core import parallel_state as mpu
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.rerun_state_machine import get_rerun_state_machine
    from megatron.training import training

    original = training.train_step

    @wraps(original)
    def observe(*args, **kwargs):
        runtime = training.get_args()
        if not state:
            config = kwargs["config"] if "config" in kwargs else args[5]
            effective = {name: getattr(runtime, name) for name in EFFECTIVE_ARGS}
            effective_config = {name: getattr(config, name) for name in EFFECTIVE_CONFIG}
            effective_geo = dict(
                tp=mpu.get_tensor_model_parallel_world_size(),
                pp=mpu.get_pipeline_model_parallel_world_size(),
                vpp=mpu.get_virtual_pipeline_model_parallel_world_size(),
                ep=mpu.get_expert_model_parallel_world_size(),
                etp=mpu.get_expert_tensor_parallel_world_size(),
                cp=mpu.get_context_parallel_world_size(),
                dp=mpu.get_data_parallel_world_size(),
                edp=mpu.get_expert_data_parallel_world_size(),
                microbatches=get_num_microbatches())
            state.update(args=effective, config=effective_config, geometry=effective_geo,
                         initial_samples=runtime.consumed_train_samples,
                         max_jobs=os.environ.get("MAX_JOBS"))
            if (effective != EFFECTIVE_ARGS or effective_config != EFFECTIVE_CONFIG or effective_geo != GEOMETRY
                    or runtime.consumed_train_samples != 0 or state["max_jobs"] != "8"
                    or Path(runtime.save) != output / "checkpoints"):
                raise ValueError("unexpected effective Chonk settings at first training step")
        rerun = get_rerun_state_machine()
        previous = rerun.rerun_count
        result = original(*args, **kwargs)
        records.append(dict(update=runtime.curr_iteration + 1, skipped=int(result[1]),
                            reruns=rerun.rerun_count - previous,
                            discarded_batches=rerun.discarded_batches,
                            early_exit=bool(result[3]), forced_checkpoint=bool(result[2]),
                            microbatches=get_num_microbatches()))
        if os.environ.get("APERTUS_COMPILE_TRACE") == "1":
            compiler.append(dict(update=runtime.curr_iteration + 1,
                                 configured_compile_threads=inductor_config.compile_threads,
                                 generated_kernel_count=metrics.generated_kernel_count,
                                 counters={str(group): dict(values) for group, values in counters.items()}))
        return result

    training.train_step = observe
    device = int(os.environ["LOCAL_RANK"])
    def memory():
        if not torch.cuda.is_initialized():
            return None
        return dict(total_bytes=torch.cuda.get_device_properties(device).total_memory,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    return memory, lambda: setattr(training, "train_step", original)


def run_training(output: Path, source: Path, entrypoint: str, argv: list[str]) -> None:
    """Write rank-local evidence even if native training fails after startup."""
    checkpoint = output / "checkpoints"
    rejected = {"--load", "--finetune", "--pretrained-checkpoint", "--async-save"}
    if (checkpoint.exists() or checkpoint.is_symlink() or argv[-2:] != ["--save", str(checkpoint)]
            or any(arg.split("=", 1)[0] in rejected for arg in argv)):
        raise ValueError("fresh training and runner-owned synchronous save required")
    if conflicts := CONFLICTING_ENV.intersection(os.environ):
        raise ValueError(f"unreviewed execution overrides: {sorted(conflicts)}")
    rank = int(os.environ["RANK"])
    records, state, observer, compiler = [], {}, {}, []

    def setup():
        observer["memory"], observer["restore"] = install_observer(output, records, state, compiler)

    completed = False
    try:
        train(output, source, entrypoint, argv, before_run=setup)
        completed = True
    finally:
        primary = sys.exception()
        try:
            if "restore" in observer:
                observer["restore"]()
            write_json(output / f"swissai-rank-{rank}.json", {
                "rank": rank, "completed": completed, "state": state, "updates": records,
                "compiler": compiler, "memory": observer["memory"]() if "memory" in observer else None})
        except Exception as exc:
            if primary is None:
                raise
            primary.add_note(f"Swiss-AI evidence write failed: {exc}")


def parse_updates(lines) -> dict:
    """Require the full fixed log sequence and finite loss, norm and timing."""
    rows = {}
    for line in lines:
        found = UPDATE.search(line)
        if not found:
            continue
        n, total = map(int, found.group(2, 3))
        if total != STEPS or n != len(rows) + 1 or n > STEPS:
            raise ValueError("unexpected update order or count")
        fields = {}
        for field in line[found.end():].split("|"):
            key, colon, value = field.partition(":")
            if colon:
                fields[key.strip()] = value.strip()
        try:
            loss, norm, ms = (float(fields[k]) for k in
                              ("lm loss", "grad norm", "elapsed time per iteration (ms)"))
            skip, nan, samples = (int(fields[k]) for k in
                                  ("number of skipped iterations", "number of nan iterations", "consumed samples"))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"missing or unreadable update metrics at {n}") from exc
        if (not all(map(math.isfinite, (loss, norm, ms))) or ms <= 0 or norm < 0
                or skip != 0 or nan != 0 or samples != BATCH * n):
            raise ValueError(f"non-finite, skipped, or inconsistent update {n}")
        rows[n] = dict(loss=loss, grad_norm=norm, seconds=ms / 1000, consumed_samples=samples,
                       skipped=skip, nan=nan, timestamp=datetime.fromisoformat(found["utc"]).isoformat())
    if len(rows) != STEPS:
        raise ValueError(f"missing updates: got {len(rows)}, expected {STEPS}")
    return rows


def verify_rank_records(output: Path) -> dict:
    """Require all ranks and every attempted update; do not infer zero reruns from logs."""
    peaks = {}
    clean = [dict(update=n, skipped=0, reruns=0, discarded_batches=0,
                  early_exit=False, forced_checkpoint=False, microbatches=128)
             for n in range(1, STEPS + 1)]
    for rank in range(WORLD):
        result = json.loads((output / f"swissai-rank-{rank}.json").read_text())
        if (result["rank"] != rank or result["completed"] is not True
                or result["state"] != dict(args=EFFECTIVE_ARGS, config=EFFECTIVE_CONFIG, geometry=GEOMETRY,
                                           initial_samples=0, max_jobs="8")):
            raise ValueError(f"rank {rank}: incomplete or mismatched effective settings")
        imports = json.loads((output / f"imports-{rank}.json").read_text())
        if type(imports["verified_modules"]) is not int or imports["verified_modules"] < 1:
            raise ValueError(f"rank {rank}: no verified Megatron imports")
        if result["updates"] != clean or [row["update"] for row in result["compiler"]] != list(range(1, STEPS + 1)):
            raise ValueError(f"rank {rank}: missing, skipped, rerun or checkpoint-triggered step")
        memory = result["memory"]
        if not memory or not 0 < memory["peak_allocated_bytes"] <= memory["peak_reserved_bytes"] <= memory["total_bytes"]:
            raise ValueError(f"rank {rank}: incomplete GPU allocator peak evidence")
        peaks[rank] = memory
    return peaks


def inventory_checkpoint(root: Path) -> dict:
    """Inventory a synchronous torch_dist artifact, without deserializing tensors."""
    tracker = root / "latest_checkpointed_iteration.txt"
    iteration = root / f"iter_{STEPS:07d}"
    if root.is_symlink() or tracker.is_symlink() or tracker.read_text().strip() != str(STEPS):
        raise ValueError("wrong terminal checkpoint tracker")
    if iteration.is_symlink() or not iteration.is_dir():
        raise ValueError("incorrect final distributed-checkpoint directory")
    if {p.name for p in root.iterdir() if p.name.startswith("iter_")} != {iteration.name}:
        raise ValueError("unexpected early or extra checkpoint")
    files = {}
    for p in iteration.iterdir():
        if p.is_symlink() or not p.is_file() or not p.stat().st_size:
            raise ValueError(f"invalid checkpoint member {p.name}")
        files[p.name] = p.stat().st_size
    shards = sorted(p for p in files if p.endswith(".distcp"))
    if not {"metadata.json", "common.pt", ".metadata"} <= files.keys() or not shards:
        raise ValueError("incomplete torch_dist file inventory")
    metadata = json.loads((iteration / "metadata.json").read_text())
    if metadata.get("sharded_backend") != "torch_dist" or metadata.get("common_backend") != "torch":
        raise ValueError("wrong distributed-checkpoint backend metadata")
    return dict(status="inventoried", iteration=STEPS, bytes=sum(files.values()), files=files, shards=shards,
                tensor_integrity="not assessed", optimizer_integrity="not assessed", restoration="not assessed")


def check(run: Path) -> None:
    """Report independent results; only accepted updates and inventory pass this gate."""
    output = run / "output"
    result = dict(status="failed", numerical_agreement="not assessed", memory_feasibility="not assessed",
                  cache_reuse="see runner result", runtime_path="requires separate dispatch proof",
                  finite_updates="not assessed", steady_state_throughput="not qualified")
    try:
        with (output / f"rank-{WORLD - 1}.log").open() as stream:
            updates = parse_updates(stream)
        result.update(updates=updates, finite_updates="logged loss and grad norm passed")
        result["per_rank_memory"] = verify_rank_records(output)
        result["rerun_and_skip_gate"] = "pass on all ranks"
        with (output / "rank-0.log").open() as stream:
            saves = [int(m[1]) for line in stream if (m := re.search(
                r"successfully saved checkpoint from iteration\s+(\d+)\s+to ", line))]
        if saves != [STEPS]:
            raise ValueError("expected exactly one completed checkpoint save")
        result["checkpoint"] = inventory_checkpoint(output / "checkpoints")
        result["update_windows"] = {}
        for lo, hi in WINDOWS:
            times = [updates[n]["seconds"] for n in range(lo, hi + 1)]
            median = statistics.median(times)
            result["update_windows"][f"{lo}-{hi}"] = dict(
                count=len(times), median_seconds=median, mean_seconds=statistics.mean(times),
                min_seconds=min(times), max_seconds=max(times), tokens_per_second_per_gpu=BATCH * 4096 / WORLD / median)
        result["status"] = "passed"
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
