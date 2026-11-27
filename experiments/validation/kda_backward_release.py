"""Measure warm KDA backward on the frozen image and the signed complete replacement module."""

import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from apertus_amd.artifacts import sha256, write_json


MODULE = "fla.ops.kda.chunk_bwd"
FILE = Path("fla/ops/kda/chunk_bwd.py")
WHEEL = Path("/opt/venv/lib/python3.12/site-packages")
COMMIT = "6c70d55a71e7fb4fca7de990015f6a3046193762"
DIGEST = {
    "baseline": "069395c9232647f5a4b82fa2dfe7ca077423016d4b641f2696f9d6328dc2e9be",
    "release": "79d34bd5711a71e455a6b0346085c9431f5b224a5766ed839468a8902ead0e21",
}
ARMS = ("baseline", "release", "release", "baseline")
SHAPE = (2, 4096, 28, 128)
WARMUPS, SAMPLES = 3, 5
MIN_DROP = 32 << 20
OUTPUTS = {"output", "q", "k", "v", "gate", "beta", "a_log", "dt_bias"}


class CandidateFinder(importlib.abc.MetaPathFinder):
    """Select the intact signed module before FLA loads; leave all other imports alone."""

    def __init__(self, path):
        self.path = path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == MODULE:
            return importlib.util.spec_from_file_location(fullname, self.path)
        return None


def selected_kda(source, arm):
    """Verify that the selected backward and the rest of FLA have the intended origins."""
    if any(name == "fla" or name.startswith("fla.") for name in sys.modules):
        raise ValueError("FLA already imported before arm selection")
    if any(Path(p).resolve().is_relative_to(source) for p in sys.path):
        raise ValueError("candidate source tree on the package import path")
    shipped, candidate = WHEEL / FILE, source / FILE
    for path, name in ((shipped, "baseline"), (candidate, "release")):
        if path.is_symlink() or sha256(path) != DIGEST[name]:
            raise ValueError(f"unexpected FLA {name} module bytes")
    for package in ("flash-linear-attention", "fla-core"):
        if importlib.metadata.version(package) != "0.5.2":
            raise ValueError(f"unexpected {package} version")
    path = shipped if arm == "baseline" else candidate
    if arm == "release":
        sys.meta_path.insert(0, CandidateFinder(path))
    chunk = importlib.import_module("fla.ops.kda.chunk")
    bwd = importlib.import_module(MODULE)
    if (Path(bwd.__file__).resolve() != path or sha256(path) != DIGEST[arm]
            or chunk.chunk_kda_bwd is not bwd.chunk_kda_bwd
            or any(not Path(mod.__file__).resolve().is_relative_to(WHEEL)
                   for name, mod in sys.modules.items()
                   if (name == "fla" or name.startswith("fla.")) and name != MODULE
                   and getattr(mod, "__file__", None))):
        raise ValueError("unexpected FLA module resolution")
    return chunk, bwd.chunk_kda_bwd


def worker(output, source, index):
    """Exercise one arm in a fresh process and keep all warm samples."""
    import torch

    arm = ARMS[index]
    if (torch.__version__, torch.version.hip, torch.cuda.device_count()) != ("2.14.0+rocm10.1.0", "7.16.26385", 1):
        raise ValueError("expected one image-910 ROCm GPU")
    target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if target != "gfx942":
        raise ValueError(f"expected gfx942, got {target}")
    chunk, backward = selected_kda(source, arm)
    print(f"KDA arm {index}: {arm}, verified imports", flush=True)
    b, s, h, d = SHAPE
    rng = torch.Generator(device="cuda").manual_seed(42)

    def normal(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device="cuda", generator=rng)

    x = {key: normal(SHAPE).requires_grad_() for key in ("q", "k", "v", "gate")}
    x.update(beta=normal((b, s, h)).requires_grad_(),
             a_log=torch.zeros(h, dtype=torch.float32, device="cuda", requires_grad=True),
             dt_bias=torch.zeros(h * d, dtype=torch.float32, device="cuda", requires_grad=True))
    do = normal(SHAPE)

    def forward():
        y, state = chunk.chunk_kda(x["q"], x["k"], x["v"], x["gate"], x["beta"],
                                   A_log=x["a_log"], dt_bias=x["dt_bias"], use_qk_l2norm_in_kernel=True,
                                   use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                                   safe_gate=True, lower_bound=-5.0, disable_recompute=False)
        if tuple(y.shape) != SHAPE or y.dtype != torch.bfloat16 or state is not None:
            raise ValueError("unexpected KDA forward contract")
        return y

    def grad(y):
        return torch.autograd.grad(y, tuple(x.values()), do)

    count = 0

    def observe_backward(*args, **kwargs):
        nonlocal count
        count += 1
        return backward(*args, **kwargs)

    # Prove the native module was called, without profiling measured samples.
    y = forward()
    chunk.chunk_kda_bwd = observe_backward
    try:
        grad(y)
    finally:
        chunk.chunk_kda_bwd = backward
    del y
    torch.cuda.synchronize()
    if count != 1:
        raise ValueError(f"selected KDA backward executed {count} times")
    for _ in range(WARMUPS):
        grad(forward())
        torch.cuda.synchronize()
    print(f"KDA arm {index}: warmed, collecting {SAMPLES} backward samples", flush=True)

    def fingerprints(values):
        from hashlib import sha256 as digest

        result = {}
        for name, tensor in values.items():
            cpu = tensor.detach().cpu().contiguous()
            if not torch.isfinite(cpu).all():
                raise ValueError(f"nonfinite {name}")
            result[name] = dict(dtype=str(cpu.dtype), shape=list(cpu.shape),
                                sha256=digest(cpu.reshape(-1).view(torch.uint8).numpy()).hexdigest())
        return result

    trials, outputs = [], []
    for _ in range(SAMPLES):
        torch.cuda.empty_cache()
        y = forward()
        torch.cuda.synchronize()
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        derivatives = grad(y)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        ap, rp = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        trials.append(dict(seconds=seconds, allocated=allocated, reserved=reserved,
                           peak_allocated=ap, peak_reserved=rp))
        outputs.append(fingerprints({"output": y, **dict(zip(x, derivatives, strict=True))}))
        del y, derivatives
    write_json(output / f"arm-{index}.json", dict(arm=arm, shape=list(SHAPE), target=target,
               backward_calls_verified=count, module_sha256=DIGEST[arm], trials=trials, outputs=outputs))


def verdict(reports):
    """Require complete warm evidence; preserve parity and memory as separate results."""
    if tuple(r["arm"] for r in reports) != ARMS:
        raise ValueError("unexpected arm sequence")
    ref = reports[0]["outputs"][0]
    if set(ref) != OUTPUTS:
        raise ValueError("missing gradient or output fingerprints")
    medians = []
    equal = True
    for r in reports:
        if (r["shape"] != list(SHAPE) or r["target"] != "gfx942" or r["backward_calls_verified"] != 1
                or r["module_sha256"] != DIGEST[r["arm"]] or len(r["trials"]) != SAMPLES
                or len(r["outputs"]) != SAMPLES):
            raise ValueError("incomplete arm evidence")
        for s in r["trials"]:
            if (not math.isfinite(s["seconds"]) or s["seconds"] <= 0
                    or not 0 < s["allocated"] <= s["peak_allocated"] <= s["peak_reserved"]
                    or not s["allocated"] <= s["reserved"] <= s["peak_reserved"]):
                raise ValueError("invalid warm memory sample")
        equal &= all(out == ref for out in r["outputs"])
        medians.append({key: statistics.median(s[key] for s in r["trials"])
                        for key in ("seconds", "allocated", "reserved", "peak_allocated", "peak_reserved")})
    drop = [medians[a]["peak_allocated"] - medians[b]["peak_allocated"] for a, b in ((0, 1), (3, 2))]
    match_baseline = len({s["allocated"] for r in reports for s in r["trials"]}) == 1
    return dict(status="passed" if equal and match_baseline else "failed",
                scope="isolated KDA backward, not full Chonk", bitwise_equal=equal,
                matched_backward_entry=match_baseline, median_by_arm=medians, paired_peak_drop_bytes=drop,
                repeatable_saving=equal and match_baseline and min(drop) >= MIN_DROP)


def train(output, source, entrypoint, argv):
    """Run the predeclared ABBA sequence using only the frozen signed source file."""
    if entrypoint != str(FILE) or argv or os.environ["WORLD_SIZE"] != "1":
        raise ValueError("unexpected KDA probe inputs")
    frozen = json.loads((output.parent / "resolved.json").read_text())
    if frozen["source"]["commit"] != COMMIT or sha256(source / FILE) != DIGEST["release"]:
        raise ValueError("unexpected frozen FLA source")
    env = dict(os.environ, PYTHONPATH="/apertus/tooling", PYTHONDONTWRITEBYTECODE="1")
    for i in range(len(ARMS)):
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                        str(output), str(source.resolve()), str(i)], env=env, cwd=output, check=True)


def check(run):
    """Persist a validated delta even when the component shows no memory improvement."""
    output = run / "output"
    result = verdict([json.loads((output / f"arm-{i}.json").read_text()) for i in range(len(ARMS))])
    write_json(output / "checks.json", result)
    if result["status"] != "passed":
        raise ValueError("KDA parity or matched-entry check failed")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["train", output, source, entrypoint, *argv]:
            train(Path(output), Path(source), entrypoint, argv)
        case ["worker", output, source, index] if index in ("0", "1", "2", "3"):
            worker(Path(output), Path(source), int(index))
        case ["check", run]:
            check(Path(run))
        case _:
            raise SystemExit("unexpected KDA validator invocation")
