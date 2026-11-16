"""Canonical projection forward only, not measured training layout or gradient acceptance.

The commands runtime owns the outer PGID/deadline. A stdlib supervisor starts two
inherited-PGID worker; failed/unquiescent workers leave private scratch for review.
Only the worker imports vendor modules. No tuning, replay, dgrad or wgrad exists.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
SOURCE = "e242f3c3c1136aeae498529bfde5e06bda9a7cca"
CONTRACT = "pinned-image-build-provenance"
PACKAGE = Path("/opt/venv/lib/python3.12/site-packages")
VERSIONS = {"torch": "2.14.0+rocm10.1.0", "transformer-engine": "2.17.1+130099ce4",
            "triton": "3.8.0+git669b31ac.rocm10.1.0"}
PY_PINS = {
    "transformer_engine/pytorch/cpp_extensions/gemm.py": "957974b26d32ab74f1521780d96e2d9e56a30e3a8420d7bb066c69cb4c6a5e9d",
    "transformer_engine/pytorch/module/base.py": "2b3ed9049450cd7002185584a9e4a0ee6176b40d3f2e33f1f30636622ab3691f",
    "transformer_engine/pytorch/module/layernorm_linear.py": "d5f177055fa489f7b922dccbb7f21c16f36b806116d6cab7bd0d5ca3b7400c73",
}
FILE_PINS = {
    "transformer_engine/build_info.txt": (98, "891ae1c35e89a9d833eb624f09ac017dcf1256870df5c692795c5bfb21de4395"),
    "transformer_engine/libtransformer_engine.so": (401104592, "140dc626463ae6072121ca66a988c71954268f7b2c27b4d447894a97f2cba69a"),
    "transformer_engine/transformer_engine_torch.cpython-312-x86_64-linux-gnu.so": (3438144, "1bdeb20270594556bc705e2b9842439925cf70ab03611145d388a92d4cc9f620"),
}
WORKSPACE = 33554432
DEFAULTS = {"TE_HIPBLASLT_ALGO_SELECTION": "0", "TE_HIPBLASLT_TUNING_RUN_COUNT": "0",
            "TE_HIPBLASLT_LOG_TUNING": "1", "NVTE_LOG_GEMM_CONFIG": "1"}
COMMAND = ["python3", "-u", "/apertus/validation.py", "run", "--native-provenance", CONTRACT]


def digest(path):
    """Hash only explicitly selected files, not image/cache payloads."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    """Atomically retain evidence in an exclusively owned output directory."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def retain(path, value):
    """Keep an active workload error primary if final evidence writing also fails."""
    primary = sys.exception()
    try:
        write(path, value)
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note(f"secondary evidence error: {error}")


def validate_environment(env, stage="proof"):
    """Reject inherited table/search/TunableOp settings rather than hiding them."""
    for name, value in env.items():
        if name.startswith(("TE_HIPBLASLT_", "PYTORCH_TUNABLEOP_", "HIPBLASLT_TUNING")):
            if name not in DEFAULTS or value != DEFAULTS[name]:
                raise ValueError(f"non-default selection environment: {name}")
    expected = DEFAULTS | {"NVTE_LOG_GEMM_CONFIG": "1" if stage == "proof" else "0"}
    if any(env.get(k) != v for k, v in expected.items()):
        raise ValueError("explicit default logging/selection required")


def validate_launch(settings, env, stage="proof"):
    """Require the proposed one-node/four-rank/48-CPU bounded command contract."""
    r, p = settings["resources"], settings["procedure"]
    if (settings["image"]["sha256"] != IMAGE or r["nodes"] != 1 or r["gpus_per_node"] != 4
            or r["time"] != "00:05:00" or r.get("reservation") != "cscs"
            or (r.get("cpus_per_task") or r["cpus_per_node"] // r["gpus_per_node"]) != 48
            or r["distribution"].split(":")[0] != "block"
            or p["kind"] != "commands" or p["timeout"] != 180 or p["commands"] != [COMMAND]):
        raise ValueError("forward-only image/resource/procedure mismatch")
    rank, local, node, world = (int(env[k]) for k in
                               ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID", "SLURM_NTASKS"))
    if not (world == 4 and node == 0 and 0 <= rank == local < 4):
        raise ValueError("four independent block ranks required")
    if any(env.get(k) != v for k, v in {"OMP_NUM_THREADS": "1", "MAX_JOBS": "2",
                                       "HWLOC_COMPONENTS": "-gl", "NCCL_P2P_DISABLE": "1"}.items()):
        raise ValueError("thread/network environment mismatch")
    if len(os.sched_getaffinity(0)) != 48:
        raise ValueError("48 CPU caller affinity required")
    validate_environment(env, stage)
    return rank, local


def provenance():
    """Pin immutable-image Python/build/ELF identities, never compiled-source attestation."""
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError("compute version mismatch")
    result = {}
    for relative, wanted in {**{k: (None, v) for k, v in PY_PINS.items()}, **FILE_PINS}.items():
        path = PACKAGE / relative
        if path.resolve(strict=True) != path:
            raise ValueError("pinned installed path redirected")
        actual = (path.stat().st_size, digest(path))
        if actual[1] != wanted[1] or (wanted[0] is not None and actual[0] != wanted[0]):
            raise ValueError(f"installed pin mismatch: {relative}")
        result[str(path)] = {"bytes": actual[0], "sha256": actual[1]}
    return {"contract": CONTRACT, "image": IMAGE, "versions": versions, "files": result,
            "binary_attestation": False, "native_sources": "absent in retained inventory; no source-to-binary proof"}


def mapped_libraries():
    """Require both actual mapped TE files, recording linked hipBLASLt identity separately."""
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and re.search(
                r"(?:libtransformer_engine|transformer_engine_torch|libhipblaslt).*\.so", fields[5]):
            paths.add(Path(fields[5]).resolve(strict=True))
    result = {str(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in paths}
    expected = {str(PACKAGE / k): {"bytes": v[0], "sha256": v[1]}
                for k, v in FILE_PINS.items() if k.endswith(".so")}
    actual_te = {k: v for k, v in result.items() if "libhipblaslt" not in Path(k).name}
    if actual_te != expected or not any("libhipblaslt" in p.name for p in paths):
        raise ValueError("mapped TE/hipBLASLt identity mismatch")
    return result


def spec(tensor):
    """Record canonical component storage without claiming a training-layout capture."""
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
                storage_offset=tensor.storage_offset(), pointer_mod_16=tensor.data_ptr() % 16,
                pointer_mod_256=tensor.data_ptr() % 256)


def invoke(gemm, x, w, out):
    """Execute only the source-derived forward TN BF16 beta-zero contract."""
    for tensor, shape in ((x, (4096, 3584)), (w, (11036, 3584)), (out, (4096, 11036))):
        if (tuple(tensor.shape) != shape or tuple(tensor.stride()) != (shape[1], 1)
                or tensor.storage_offset() or str(tensor.dtype) != "torch.bfloat16"):
            raise ValueError("canonical forward tensor shape/stride/dtype mismatch")
    gemm(w, x, out_dtype=out.dtype, out=out, layout="TN", alpha=1.0, beta=0.0,
         accumulate=False, grad=False, use_split_accumulator=False, bias=None, gelu=False)


def numerical_check(torch, actual, reference):
    """Reuse old tiled FP32-then-BF16 reference and unchanged all-element/RMS gates."""
    totals = dict(finite=True, mismatched_elements=0, max_abs=0.0, squared_error=0.0, squared_reference=0.0)
    for start in range(0, actual.shape[0], 512):
        end = min(actual.shape[0], start + 512)
        expected, value = reference(start, end), actual[start:end].float()
        finite = bool(torch.isfinite(value).all() and torch.isfinite(expected).all())
        totals["finite"] &= finite
        if not finite:
            continue
        diff = value - expected
        totals["mismatched_elements"] += int((diff.abs() > 0.02 + 0.02 * expected.abs()).sum().item())
        totals["max_abs"] = max(totals["max_abs"], diff.abs().max().item())
        totals["squared_error"] += diff.square().sum(dtype=torch.float64).item()
        totals["squared_reference"] += expected.square().sum(dtype=torch.float64).item()
    totals.update(rms_error=math.sqrt(totals["squared_error"] / actual.numel()),
                  reference_rms=math.sqrt(totals["squared_reference"] / actual.numel()))
    totals["passed"] = metrics_pass(totals)
    return totals


def metrics_pass(metrics):
    """Never allow finite/RMS success to override a localized element failure."""
    return (metrics["finite"] and not metrics["mismatched_elements"]
            and all(math.isfinite(metrics[k]) for k in ("max_abs", "rms_error", "reference_rms"))
            and metrics["rms_error"] <= 1e-6 + 2**-7 * metrics["reference_rms"])


def algorithm_metadata(log, stage="proof"):
    """Allow only the retained import warning and expected native first-key output."""
    ids = re.findall(r"\[INFO\] Use hipBLASLt algo \[(\d+)\] (-?\d+)", log)
    descriptor = ("m=11036 k=3584 n=4096 transa=T transb=N A_type=6 B_type=6 D_type=6 "
                  "bias_type=4 grad=0 bias=0 gelu=0 use_fp8=0 scale_mode=tensor alpha=1 beta=0")
    descriptors = re.findall(r"^m=.*$", log, re.MULTILINE)
    if len(ids) != 1 or descriptors != ([descriptor] if stage == "proof" else []):
        raise ValueError("native default algorithm/descriptor metadata absent or ambiguous")
    rest = re.sub(r"^\[INFO\] Use hipBLASLt algo \[\d+\] -?\d+\s*$|^m=.*$", "", log, flags=re.MULTILINE)
    rest = re.sub(r"^/opt/venv/lib/python3.12/site-packages/torch/jit/_script.py:\d+: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`\.\s*$|^  warnings.warn\(\s*$", "", rest, flags=re.MULTILINE)
    if rest.strip():
        raise ValueError("unexpected native/import output or selection warning")
    return {"aidx": int(ids[0][0]), "algo_id": int(ids[0][1]), "native_mnk": [11036, 4096, 3584],
            "ld": [3584, 3584, 11036]}


def compare_workers(proof, measurement):
    """Require matched fresh DEFAULT identities/signature, never a tuning/replay arm."""
    for key in ("rank", "seed", "consumer_source", "driver_sha256", "provenance", "tensors", "workspace",
                "mapped_libraries", "parameters", "local_device", "algorithm"):
        if proof[key] != measurement[key]:
            raise ValueError(f"fresh default workers differ: {key}")


def measure(torch, call, clock=time.perf_counter):
    """Two warmups and ten samples; no reference, allocation or reset inside timed calls."""
    samples = []
    for index in range(12):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        began = clock()
        call()
        returned = clock()
        end.record()
        end.synchronize()
        completed = clock()
        if index >= 2:
            samples.append(dict(host_return_seconds=returned - began, completion_seconds=completed - began,
                                device_event_ms=start.elapsed_time(end)))
    return samples


def gpu_worker(directory, rank, local, stage="proof"):
    """Prove descriptors OR freshly qualify/measure, with immutable startup logging."""
    record = {"status": "running", "rank": rank, "seed": 42 + rank, "pid": os.getpid(), "pgid": os.getpgrp(),
              "consumer_source": SOURCE, "driver_sha256": digest(__file__), "stage": stage,
              "scope": "canonicalprojectionforwardonly—not measuredtraininglayout/fullprojection/gradient/optimizer acceptance"}
    output = directory / "worker.json"
    try:
        validate_environment(os.environ, stage)
        record["startup_logging"] = os.environ["NVTE_LOG_GEMM_CONFIG"]
        record["provenance"] = provenance()  # Before any vendor import.
        write(output, record)
        from apertus_amd.runtime import preserve_affinity, require_affinity
        affinity = record["affinity"] = {}
        with preserve_affinity(affinity) as initial:
            import torch
            import transformer_engine.pytorch.cpp_extensions.gemm as module
            if Path(module.__file__).resolve() != PACKAGE / "transformer_engine/pytorch/cpp_extensions/gemm.py":
                raise ValueError("loaded TE Python path mismatch")
            if torch.cuda.device_count() != 4 or not torch.version.hip:
                raise ValueError("all four ROCm GPUs must remain visible")
            torch.cuda.set_device(local)
            if (not torch.cuda.get_device_properties(local).gcnArchName.startswith("gfx942")
                    or torch.cuda.current_device() != local):
                raise ValueError("gfx942/local GPU required")
        require_affinity(initial, "forward")
        record["local_device"] = local
        torch.manual_seed(42 + rank)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        x = torch.randn(4096, 3584, device="cuda", dtype=torch.bfloat16) * 0.25
        w = torch.randn(11036, 3584, device="cuda", dtype=torch.bfloat16) * 0.02
        y = torch.empty(4096, 11036, device="cuda", dtype=torch.bfloat16)
        disposable = torch.empty_like(y)
        workspace = module.get_cublas_workspace(local, False, False)
        tensors = {"X": x, "W": w, "Y": y, "measurement_output": disposable}
        if workspace.numel() != WORKSPACE or any(not t.is_contiguous() or t.storage_offset() for t in tensors.values()):
            raise ValueError("canonical storage/workspace mismatch")
        record.update(tensors={k: spec(v) for k, v in tensors.items()}, workspace=spec(workspace),
                      mapped_libraries=mapped_libraries(), parameters=dict(layout="TN", alpha=1, beta=0,
                      split_accumulator=False, output_dtype="bfloat16", workspace_bytes=WORKSPACE))
        write(output, record)
        invoke(module.general_gemm, x, w, y)
        torch.cuda.synchronize()
        record["algorithm"] = algorithm_metadata((directory / "native.log").read_text(), stage)
        if stage == "measurement":
            compare_workers(json.loads((directory.parent / "proof/worker.json").read_text()), record)
        reference = lambda a, b: (x[a:b].float() @ w.float().T).to(torch.bfloat16).float()
        record["before"] = numerical_check(torch, y, reference)
        write(output, record)
        if not record["before"]["passed"]:
            raise ValueError("forward numerical gate failed")
        if stage == "measurement":
            saved, saved_x, saved_w = y.clone(), x.clone(), w.clone()
            if algorithm_metadata((directory / "native.log").read_text(), stage) != record["algorithm"]:
                raise ValueError("native output changed before timing")
            record["samples"] = measure(torch, lambda: invoke(module.general_gemm, x, w, disposable))
            if algorithm_metadata((directory / "native.log").read_text(), stage) != record["algorithm"]:
                raise ValueError("native output changed during timing")
            record["after"] = numerical_check(torch, disposable, reference)
            record["qualification_output_unchanged"] = bool(torch.equal(saved, y) and torch.equal(saved_x, x) and torch.equal(saved_w, w))
            if not record["after"]["passed"] or not record["qualification_output_unchanged"]:
                raise ValueError("post-measurement output/gate mismatch")
        if algorithm_metadata((directory / "native.log").read_text(), stage) != record["algorithm"]:
            raise ValueError("native output changed")
        require_affinity(initial, "forward completion")
        validate_environment(os.environ, stage)
        if provenance() != record["provenance"] or mapped_libraries() != record["mapped_libraries"]:
            raise ValueError("immutable pins changed")
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        retain(output, record)


def quiescent():
    """Only remove scratch when our inherited outer PGID has no other live processes."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == os.getpgrp() and fields[0] not in {"Z", "X"}:
            return False
    return True


def run(settings):
    """Supervise two sequential default workers under ONE deadline and outer PGID."""
    from apertus_amd.config import CACHE_PATHS
    rank, local = validate_launch(settings, os.environ)
    directory = Path(f"forward-rank-{rank}").resolve()
    directory.mkdir()  # Never replace old evidence.
    root = Path(settings["scratch_root"])
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"forward-{os.environ['SLURM_JOB_ID']}-{rank}-", dir=root))
    record = {"status": "failed", "scratch": str(scratch), "cleanup": "retained", "pgid": os.getpgrp()}
    deadline = time.monotonic() + 160
    try:
        env = dict(os.environ)
        for key, relative in CACHE_PATHS.items():
            cache = scratch / relative
            cache.mkdir()
            env[key] = str(cache)
        workers = record["workers"] = {}
        for stage, logging in (("proof", "1"), ("measurement", "0")):
            output = directory / stage
            output.mkdir()
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--native-provenance", CONTRACT,
                       "--stage", stage, "--directory", str(output)]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("total two-worker deadline")
            with (output / "native.log").open("x") as log:
                process = subprocess.Popen(command, env=env | {"NVTE_LOG_GEMM_CONFIG": logging}, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=False)
                status = process.wait(timeout=remaining)
            worker = json.loads((output / "worker.json").read_text()) if (output / "worker.json").exists() else {}
            if status or worker.get("status") != "passed" or worker.get("pgid") != os.getpgrp():
                raise ValueError(f"{stage} worker failed: {worker.get('error', status)}")
            if algorithm_metadata((output / "native.log").read_text(), stage) != worker["algorithm"]:
                raise ValueError("completed native log differs")
            workers[stage] = worker
        compare_workers(workers["proof"], workers["measurement"])
        if time.monotonic() > deadline:
            raise TimeoutError("total two-worker deadline")
        if not quiescent():
            raise ValueError("worker descendants not quiescent; scratch retained")
        shutil.rmtree(scratch)
        record.update(status="passed", cleanup="removed")
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        retain(directory / "summary.json", record)


def main():
    """Expose only default qualification and its private worker, no configurable workloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker"))
    parser.add_argument("--native-provenance", choices=(CONTRACT,), required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--stage", choices=("proof", "measurement"))
    args = parser.parse_args()
    if (args.mode == "worker") != (args.directory is not None and args.stage is not None):
        parser.error("only a worker requires its private directory and stage")
    settings = json.loads(Path("/apertus/resolved.json").read_text())
    rank, local = validate_launch(settings, os.environ, args.stage or "proof")
    if args.mode == "run":
        run(settings)
    else:
        gpu_worker(args.directory, rank, local, args.stage)


if __name__ == "__main__":
    main()
