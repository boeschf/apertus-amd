"""Instrument and validate the dense AdamW restart examples (unsharded Torch, TP/PP=1)."""

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

from apertus_amd.artifacts import sha256, write_json
from apertus_amd.config import option
from apertus_amd.entrypoint import train


def validate(settings: dict) -> None:
    """Reject workloads outside this experiment validator's supported checkpoint format."""
    args = settings["args"]
    if settings["procedure"]["kind"] != "restart" or option(args, "--ckpt-format") != "torch" or \
            option(args, "--optimizer", "adam") != "adam" or "--use-distributed-optimizer" in args or any(
                int(option(args, flag, 1)) != 1 for flag in ("--tensor-model-parallel-size", "--pipeline-model-parallel-size")
            ):
        raise ValueError("dense restart validation requires AdamW, unsharded torch checkpoints and TP/PP size one")


def instrument(output: Path, procedure: dict, rank: int) -> None:
    """Record dense model identity and optimizer updates without replacing compute functions."""
    import megatron.training.training as training
    from megatron.training import get_args
    original_setup = training.setup_model_and_optimizer
    def setup(*args, **kwargs):
        value = original_setup(*args, **kwargs)
        models, optimizer, scheduler = value
        types = sorted({f"{type(m).__module__}.{type(m).__qualname__}" for model in models for m in model.modules()})
        attention_types = {f"{type(m.core_attention).__module__}.{type(m.core_attention).__qualname__}"
                           for model in models for m in model.modules() if hasattr(m, "core_attention")}
        if procedure.get("attention") and attention_types != {procedure["attention"]}:
            raise ValueError(f"dense baseline selected unexpected attention: {attention_types}")
        optimizers = []
        current = optimizer
        while current is not None:
            optimizers.append(f"{type(current).__module__}.{type(current).__qualname__}")
            current = getattr(current, "optimizer", None)
        write_json(output / f"model-{rank}.json", {"modules": types, "optimizers": optimizers,
                                            "attention": sorted(attention_types),
                                            "scheduler": type(scheduler).__name__,
                                            "initial_iteration": get_args().iteration})
        return value
    training.setup_model_and_optimizer = setup
    original_step = training.train_step
    signature = inspect.signature(original_step)
    def step(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        value = original_step(*args, **kwargs)
        metrics = {"iteration": bound.arguments["iteration"] + 1,
                   "losses": {k: float(v) for k, v in value[0].items()},
                   "skipped": int(value[1]), "grad_norm": float(value[5])}
        if metrics["skipped"] or not all(math.isfinite(v) for v in [*metrics["losses"].values(), metrics["grad_norm"]]):
            raise ValueError(f"invalid optimizer update: {metrics}")
        with (output / f"steps-{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        return value
    training.train_step = step


def fingerprint(value) -> tuple[str, int]:
    """Hash typed nested state deterministically and reject all nonfinite values."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    tensors = 0
    def visit(item):
        nonlocal tensors
        if isinstance(item, torch.Tensor):
            tensors += 1
            item = item.detach().cpu().contiguous()
            if not torch.isfinite(item).all():
                raise ValueError("nonfinite checkpoint tensor")
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            if not np.isfinite(item).all():
                raise ValueError("nonfinite checkpoint array")
            digest.update(f"array:{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}[".encode())
            for child in item:
                visit(child)
            digest.update(b"]")
        elif item is None or isinstance(item, (str, bytes, bool, int, float, np.generic)):
            if isinstance(item, (float, np.floating)) and not math.isfinite(item):
                raise ValueError("nonfinite checkpoint scalar")
            digest.update(f"{type(item).__name__}:{repr(item)};".encode())
        else:
            raise ValueError(f"unsupported checkpoint state type: {type(item)}")
    visit(value)
    return digest.hexdigest(), tensors


def check(output: Path, settings: dict) -> dict:
    """Check independent checkpoint integrity, updates, and restart equality gates."""
    validate(settings)
    import torch
    final = int(option(settings["args"], "--train-iters"))
    split = settings["procedure"]["split"]
    batch = int(option(settings["args"], "--global-batch-size"))
    world = settings["resources"]["nodes"] * settings["resources"]["gpus_per_node"]
    checkpoints = {}
    details = {}
    for phase, steps in (("continuous", (split, final)), ("prefix", (split,)), ("resume", (final,))):
        base = output / phase
        latest = (base / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip()
        if latest != str(steps[-1]):
            raise ValueError(f"incorrect checkpoint tracker: {phase}")
        metrics = {}
        for rank in range(world):
            model = json.loads((base / f"model-{rank}.json").read_text())
            if model["initial_iteration"] != (split if phase == "resume" else 0):
                raise ValueError(f"incorrect loaded iteration: {phase}/{rank}")
            rows = [json.loads(line) for line in (base / f"steps-{rank}.jsonl").read_text().splitlines()]
            wanted = list(range(split + 1, final + 1)) if phase == "resume" else list(range(1, steps[-1] + 1))
            if [m["iteration"] for m in rows] != wanted:
                raise ValueError(f"incorrect completed updates: {phase}/{rank}")
            for metric in rows:
                if metric["skipped"] or not metric["losses"] or not all(
                    math.isfinite(x) for x in [*metric["losses"].values(), metric["grad_norm"]]
                ):
                    raise ValueError(f"invalid metrics: {phase}/{rank}")
            metrics[str(rank)] = rows
        details[phase] = {"metrics": metrics, "checkpoints": {}}
        for iteration in steps:
            path = base / f"checkpoints/iter_{iteration:07d}/mp_rank_00/model_optim_rng.pt"
            # These files were generated by this run, not supplied externally.
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["iteration"] != iteration or state["args"].train_iters != final or \
                    state["args"].consumed_train_samples != iteration * batch:
                raise ValueError(f"checkpoint iteration/sample accounting mismatch: {path}")
            selected = {key: state[key] for key in ("model", "optimizer", "opt_param_scheduler", "rng_state")}
            hashes = {key: fingerprint(value) for key, value in selected.items()}
            if not hashes["model"][1] or not hashes["optimizer"][1] or not hashes["rng_state"][1]:
                raise ValueError("checkpoint is missing tensor state")
            checkpoints[phase, iteration] = hashes
            details[phase]["checkpoints"][str(iteration)] = {"sha256": sha256(path), "state": hashes}
    if checkpoints["continuous", split] != checkpoints["prefix", split]:
        raise ValueError("independent initial trajectories differ at the restart split")
    if checkpoints["continuous", final] != checkpoints["resume", final]:
        raise ValueError("continuous and resumed checkpoint states differ")
    if checkpoints["continuous", split]["model"] == checkpoints["continuous", final]["model"]:
        raise ValueError("model did not update between the restart split and final iteration")
    for rank in map(str, range(world)):
        if details["continuous"]["metrics"][rank] != details["prefix"]["metrics"][rank] + details["resume"]["metrics"][rank]:
            raise ValueError(f"continuous and resumed step metrics differ: rank {rank}")
    return {"status": "passed", "checkpoint_integrity": "passed", "finite_updates": "passed",
            "restart_equivalence": "exact", "details": details,
            "state_sha256": hashlib.sha256(json.dumps(checkpoints["continuous", final], sort_keys=True).encode()).hexdigest()}


def check_run(run: Path) -> None:
    """Write check outcomes, including failures, before returning or raising."""
    output = run / "output"
    try:
        result = check(output, json.loads((run / "resolved.json").read_text()))
        if (run / "reference.json").exists():
            reference = json.loads((run / "reference.json").read_text())
            if result["state_sha256"] != reference["state_sha256"] or any(
                value["metrics"] != reference["details"][phase]["metrics"] for phase, value in result["details"].items()
            ):
                raise ValueError("state or metrics differ from selected comparison run")
            result["reference_equivalence"] = "exact"
    except Exception as error:
        write_json(output / "checks.json", {"status": "failed", "error": str(error)})
        raise
    write_json(output / "checks.json", result)


def main(argv: list[str], *, run: Path = Path("/apertus")) -> None:
    """Implement the experiment validation script's train/check command contract."""
    if argv and argv[0] == "train" and len(argv) >= 4:
        settings = json.loads((run / "resolved.json").read_text())
        validate(settings)
        output = Path(argv[1])
        train(output, Path(argv[2]), argv[3], argv[4:], manifest=run / "source/manifest.json",
              before_run=lambda: instrument(output, settings["procedure"], int(os.environ["RANK"])))
    elif len(argv) == 2 and argv[0] == "check":
        check_run(Path(argv[1]))
    else:
        raise SystemExit("usage: dense_restart.py train OUTPUT SOURCE ENTRYPOINT [ARGS...] | check RUN")


if __name__ == "__main__":
    main(sys.argv[1:])
