"""Bounded TE KDA-projection component; canonical source-derived layout, not training acceptance.

Only GPU children import Torch/TE. The parent uses the frozen runtime's child
process-group lifecycle and owns private job/rank scratch. Installed native sources
are required by default; the explicit alternative pins image/build/library identity,
not source-to-binary correspondence. Missing cache-consumption proof fails replay.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import sys
import tempfile
import time


IMAGE = "cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0"
TE_VERSION = "2.17.1+130099ce4"
TE_COMMIT = "130099ce48dc806398611de9960f8e0c486ad5f1"
PACKAGE = Path("/opt/venv/lib/python3.12/site-packages")
PY_SOURCES = {
    "transformer_engine/pytorch/cpp_extensions/gemm.py": "957974b26d32ab74f1521780d96e2d9e56a30e3a8420d7bb066c69cb4c6a5e9d",
    "transformer_engine/pytorch/module/base.py": "2b3ed9049450cd7002185584a9e4a0ee6176b40d3f2e33f1f30636622ab3691f",
    "transformer_engine/pytorch/module/layernorm_linear.py": "d5f177055fa489f7b922dccbb7f21c16f36b806116d6cab7bd0d5ca3b7400c73",
}
NATIVE_SOURCES = {
    "transformer_engine/common/gemm/rocm_gemm.cu": "66451919925d0206853597fac8e15e806254c08a1bdb77710f21ba45269f0016",
    "transformer_engine/common/gemm/cublaslt_gemm.cu": "d21c3828938fac30fd171ecd1b384514debec894beebc02aa82c5f36f6dae668",
    "transformer_engine/pytorch/csrc/extensions/gemm.cpp": "f68a35ccf3238f25cee096a7149d24121a2ae764f62800f26c6cddafffb51dd1",
}
STRICT_PROVENANCE = "installed-sources"
BUILD_PROVENANCE = "pinned-image-build-provenance"
BUILD_INFO_TEXT = ("ROCM_VERSION: 10.1\nGPU_TARGETS: gfx942;gfx950\n"
                   "COMMIT_ID: 130099ce48dc806398611de9960f8e0c486ad5f1\n")
BUILD_INFO_PIN = {"bytes": 98, "sha256": "891ae1c35e89a9d833eb624f09ac017dcf1256870df5c692795c5bfb21de4395"}
TE_BINARY_PINS = {
    "transformer_engine/libtransformer_engine.so": {
        "bytes": 401104592, "sha256": "140dc626463ae6072121ca66a988c71954268f7b2c27b4d447894a97f2cba69a"},
    "transformer_engine/transformer_engine_torch.cpython-312-x86_64-linux-gnu.so": {
        "bytes": 3438144, "sha256": "1bdeb20270594556bc705e2b9842439925cf70ab03611145d388a92d4cc9f620"},
}
BOUNDS = dict(tokens=4096, inputs=3584, outputs=11036, phase_timeout=120,
              rank_timeout=900, warmups=3, samples=20, candidates=16, trials=5)
PHASES = ("default-0", "tune", "replay-0", "replay-1", "default-1")
WORKSPACE = 33554432
TILE = 512
CSV_FIELDS = ("dev_cap,m,n,k,trans_a,trans_b,type_a,type_b,type_d,bias_type,aux_type,"
              "lda,ldb,ldd,scale_mode,epi,comp,scale_type,fp4_alpha,ws_min,ws_max,algo_id,aidx").split(",")
# Column-major native descriptors: A/B are TE API operands, not row-major notation.
CASES = {
    "forward": dict(A="W", B="X", out="Y", layout="TN", grad=False, split=False,
                    mnk=(11036, 4096, 3584), ld=(3584, 3584, 11036), dtype="bfloat16"),
    "dgrad": dict(A="W", B="dY", out="dX", layout="NN", grad=True, split=True,
                  mnk=(3584, 4096, 11036), ld=(3584, 11036, 3584), dtype="bfloat16"),
    "wgrad": dict(A="X", B="dY", out="main_grad", layout="NT", grad=True, split=True,
                  mnk=(3584, 11036, 4096), ld=(3584, 11036, 3584), dtype="float32"),
}


def digest(path):
    """Hash selected source/library files without importing them."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Atomically retain this invocation's evidence; caller owns the fresh directory."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def parser():
    """Expose fixed workload arguments identically to the parent and GPU children."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "gpu"))
    p.add_argument("output", type=Path)
    p.add_argument("--phase", choices=PHASES)
    p.add_argument("--native-provenance", choices=(STRICT_PROVENANCE, BUILD_PROVENANCE), default=STRICT_PROVENANCE)
    for name in BOUNDS:
        p.add_argument("--" + name.replace("_", "-"), type=int, required=True)
    return p


def validate_args(args):
    """Reject shape, precision-work bound or phase changes outside this component."""
    if any(getattr(args, name) != value for name, value in BOUNDS.items()):
        raise ValueError("only fixed unpadded KDA geometry and reviewed bounds are supported")
    if (args.mode == "gpu") != (args.phase is not None):
        raise ValueError("GPU children require a phase; the orchestrator must not select one")


def phase_environment(environ, phase, table):
    """Remove competing consumers and set fresh-process TE-only selection state."""
    env = {k: v for k, v in environ.items() if not k.startswith(("PYTORCH_TUNABLEOP_", "TE_HIPBLASLT_"))
           and k not in {"HIPBLASLT_TUNING_FILE", "HIPBLASLT_TUNING_OVERRIDE_FILE"}}
    env.update(TE_HIPBLASLT_ALGO_SELECTION="0", TE_HIPBLASLT_LOG_TUNING="1",
               TE_HIPBLASLT_TUNING_RUN_COUNT="5" if phase == "tune" else "0",
               TE_HIPBLASLT_TUNING_ALGO_COUNT="16", NVTE_LOG_GEMM_CONFIG="1")
    if phase == "tune":
        env["TE_HIPBLASLT_ALGO_SAVE"] = str(table)
    elif phase.startswith("replay"):
        env["TE_HIPBLASLT_ALGO_LOAD"] = str(table)
    return env


def child_command(args, phase, script):
    """Run exactly the frozen selected file, with no sibling workload dependencies."""
    argv = [sys.executable, "-u", str(script), "gpu", str(args.output), "--phase", phase]
    for name in BOUNDS:
        argv += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    argv += ["--native-provenance", args.native_provenance]
    return argv


def remaining_budget(deadline, phase_timeout, now):
    """Cap each child by both the phase bound and the rank's remaining deadline."""
    remaining = deadline - now
    if remaining <= 0:
        raise TimeoutError("rank total diagnostic deadline")
    return min(phase_timeout, remaining)


def verify_sources(roots, expected):
    """Fail closed on missing/drifted installed source; never fetch substitutes at runtime."""
    result = {}
    for relative, wanted in expected.items():
        candidates = [root / relative for root in roots if (root / relative).is_file()]
        if not candidates:
            raise ValueError(f"pinned source unavailable: {relative}; no tuning authorized by this gate")
        actual = [{"path": str(path), "sha256": digest(path)} for path in candidates]
        if any(item["sha256"] != wanted for item in actual):
            raise ValueError(f"pinned source mismatch: {relative}: {actual}")
        result[relative] = actual
    return result


def verify_pinned_file(path, expected, label):
    """Check an exact installed path, byte count and SHA without importing its code."""
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"pinned {label} path mismatch: {path} -> {resolved}")
    actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
        raise ValueError(f"pinned {label} mismatch: {actual}")
    return actual


def verify_build_provenance(package, version, *, workspace=Path("/workspace/TransformerEngine")):
    """Verify the explicit image/build identity contract, never native-source attestation."""
    if version != TE_VERSION:
        raise ValueError("pinned TE version mismatch")
    info_path = package / "transformer_engine/build_info.txt"
    info = verify_pinned_file(info_path, BUILD_INFO_PIN, "build_info")
    info["text"] = info_path.read_text(encoding="utf-8")
    if info["text"] != BUILD_INFO_TEXT:
        raise ValueError("pinned build_info text/commit mismatch")
    binaries = [verify_pinned_file(package / name, pin, "TE binary") for name, pin in TE_BINARY_PINS.items()]
    observations = [{"path": str(root / name), "observation": "present" if (root / name).is_file() else "absent"}
                    for root in (package, workspace) for name in NATIVE_SOURCES]
    return {"contract": BUILD_PROVENANCE, "image": IMAGE, "TE_version": version, "TE_commit": TE_COMMIT,
            "build_info": info, "te_binary_files": binaries, "installed_native_source_observations": observations,
            "audited_upstream_native_sources": dict(NATIVE_SOURCES), "binary_attestation": False,
            "source_to_binary_correspondence": "not established",
            "identity_basis": "trusted immutable image/build provenance and metadata inventory740614, not compiled-source attestation",
            "build_history": "artifacts/image-builds/primus268-20261110T084213Z/primus268-base/evidence/inspect.json:638"}


def verify_mapped_te_libraries(libraries, package):
    """Require actual mapped TE paths/sizes/hashes to match the separately pinned files."""
    expected = {str(package / name): pin for name, pin in TE_BINARY_PINS.items()}
    mapped = [item for item in libraries if Path(item["path"]).name.startswith(
        ("libtransformer_engine", "transformer_engine_torch"))]
    if len(mapped) != len(expected) or {item["path"] for item in mapped} != set(expected):
        raise ValueError(f"mapped TE path mismatch: {mapped}")
    for item in mapped:
        if any(item[key] != expected[item["path"]][key] for key in ("bytes", "sha256")):
            raise ValueError(f"mapped TE binary mismatch: {item}")
    return {"contract": BUILD_PROVENANCE, "verified_te_libraries": mapped, "binary_attestation": False,
            "hipblaslt_identity": "mapped path/size/hash recorded only; not independently pinned or attested"}


def parse_table(text):
    """Require exactly the three native TE cache keys and supported workspace/schema."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != CSV_FIELDS:
        raise ValueError("TE cache schema mismatch")
    found = {}
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("malformed TE cache row")
        names = [name for name, case in CASES.items()
                 if tuple(int(row[k]) for k in ("m", "n", "k")) == case["mnk"]
                 and row["trans_a"] + row["trans_b"] == case["layout"]
                 and row["type_d"] == case["dtype"]]
        if len(names) != 1 or names[0] in found:
            raise ValueError("unexpected/duplicate TE cache key")
        name = names[0]
        exact = dict(dev_cap="904", type_a="bfloat16", type_b="bfloat16", bias_type="-",
                     aux_type="-", scale_mode="0", epi="-", comp="f32", scale_type="float32", fp4_alpha="0")
        if any(row[key] != value for key, value in exact.items()):
            raise ValueError("TE cache precision/device/epilogue mismatch")
        if tuple(int(row[k]) for k in ("lda", "ldb", "ldd")) != CASES[name]["ld"]:
            raise ValueError("TE cache leading-dimension mismatch")
        if not 0 <= int(row["ws_min"]) <= WORKSPACE == int(row["ws_max"]) or not 0 <= int(row["aidx"]) < 16:
            raise ValueError("TE cache workspace/candidate bound mismatch")
        int(row["algo_id"])
        found[name] = row
    if set(found) != set(CASES):
        raise ValueError("incomplete TE cache")
    return found


def consumption_proof(log, rows, table):
    """Match native first-call IDs to every loaded exact-key row, rejecting fallback."""
    if f"Loading autotune results from {table}\n" not in log:
        raise ValueError("native TE load acknowledgment absent")
    if any(word in log for word in ("[WARNING]", "Incorrect algo storage", "Parsing CSV line failed",
                                    "Could not load", "Cannot get corresponding", "Perform hipBLASLt algo selection")):
        raise ValueError("TE replay load/fallback/tuning warning")
    result = {}
    for name, case in CASES.items():
        begin, end = f"KDA_PROOF_BEGIN {name}\n", f"KDA_PROOF_END {name}\n"
        if log.count(begin) != 1 or log.count(end) != 1:
            raise ValueError("missing/duplicate native proof scope")
        section = log.split(begin)[1].split(end)[0]
        row = rows[name]
        ids = re.findall(r"\[INFO\] Use hipBLASLt algo \[(\d+)\] (-?\d+)", section)
        if ids != [(row["aidx"], row["algo_id"])]:
            raise ValueError(f"loaded native algorithm not proven: {name}")
        m, n, k = case["mnk"]
        if f"m={m} k={k} n={n} transa={case['layout'][0]} transb={case['layout'][1]} " not in section:
            raise ValueError(f"native descriptor log mismatch: {name}")
        result[name] = {"aidx": int(row["aidx"]), "algo_id": int(row["algo_id"]),
                        "proof": "pinned exact-key parser plus native per-call Use-algo log, no fallback"}
    return result


def metric_failures(metrics, fp32):
    """Reject nonfinite/localized errors and fixed precision-specific RMS violations."""
    if not metrics["finite"] or any(not math.isfinite(metrics[k]) for k in
                                    ("max_abs", "rms_error", "reference_rms", "relative_l2")):
        return ["nonfinite tensor/error metric"]
    failures = []
    if metrics["mismatched_elements"]:
        failures.append("elementwise gate")
    if metrics["rms_error"] > 1e-6 + (1e-4 if fp32 else 0.0078125) * metrics["reference_rms"]:
        failures.append("RMS gate")
    return failures


def install_interrupts():
    """Turn TERM/INT into exceptions so runtime.child reaps nested process groups."""
    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)


def require_phase_affinity(affinity, previous=None):
    """Reject drift within a passed phase or any change to the paired phase binding."""
    initial, final = affinity.get("initial"), affinity.get("final")
    if not initial or initial != final or affinity.get("errors") or (previous is not None and initial != previous):
        raise ValueError(f"phase affinity mismatch: previous={previous}, current={affinity}")
    return list(initial)


def orchestrate(args, settings, output):
    """Own rank scratch and sequence fresh processes using the existing bounded lifecycle."""
    from apertus_amd.config import CACHE_PATHS
    from apertus_amd.runtime import child, UnreapedChildError, defer_termination

    rank, local = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_LOCALID"])
    if (settings["image"]["sha256"] != IMAGE or settings["resources"]["nodes"] != 1
            or settings["resources"]["gpus_per_node"] != 4 or settings["resources"]["time"] != "00:20:00"
            or settings["resources"].get("reservation") != "cscs" or int(os.environ["SLURM_NTASKS"]) != 4
            or not 0 <= rank == local < 4):
        raise ValueError("requires reviewed image and explicit one-node/four-rank/20min cscs launch")
    started = time.monotonic()
    deadline = started + args.rank_timeout
    output.mkdir()  # Exclusive per-rank evidence; never replace an old run.
    record = {"status": "running", "rank": rank, "scope": "canonical source-derived KDA TE GEMM boundary",
              "image": IMAGE, "script_sha256": digest(Path(__file__)), "phases": [],
              "native_provenance_contract": args.native_provenance, "binary_attestation": False}
    path = output / "summary.json"
    root = Path(settings["scratch_root"])
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"kda-{os.environ['SLURM_JOB_ID']}-{rank}-", dir=root))
    safe_cleanup = True
    try:
        env = dict(os.environ)
        for key, relative in CACHE_PATHS.items():
            directory = scratch / relative
            directory.mkdir()
            env[key] = str(directory)
        table = scratch / "te-algos.csv"
        rows, table_hash = None, None
        phase_affinity = None
        for phase in PHASES:
            phase_args = argparse.Namespace(**vars(args))
            phase_args.output = output
            log = output / f"{phase}.log"
            item = {"phase": phase, "status": "running"}
            record["phases"].append(item)
            write(path, record)
            before = time.monotonic()
            child(child_command(phase_args, phase, Path(__file__).resolve()), log, cwd=output,
                  env=phase_environment(env, phase, table),
                  timeout=remaining_budget(deadline, args.phase_timeout, before))
            result = json.loads((output / f"{phase}.json").read_text())
            if result["status"] != "passed":
                raise ValueError(f"incomplete GPU child: {phase}")
            item["affinity"] = result.get("affinity", {})
            phase_affinity = require_phase_affinity(item["affinity"], phase_affinity)
            if phase == "tune":
                rows = parse_table(table.read_text())
                table_hash = digest(table)
                (output / "tuned.csv").write_bytes(table.read_bytes())
                table.chmod(0o400)  # Replay LOAD is private and read-only, never SAVE.
            elif phase.startswith("replay"):
                if digest(table) != table_hash:
                    raise ValueError("private tuning table changed")
                item["consumption"] = consumption_proof(log.read_text(), rows, table)
            item.update(status="passed", wall_seconds=time.monotonic() - before)
            write(path, record)
        remaining_budget(deadline, args.phase_timeout, time.monotonic())
        record.update(status="passed", table_sha256=table_hash, matched_phase_affinity=phase_affinity,
                      acceptance="component numerics and native TE cache replay only; no training-layout/performance acceptance")
    except BaseException as error:
        if isinstance(error, UnreapedChildError):
            safe_cleanup = False
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        if record["phases"] and record["phases"][-1]["status"] == "running":
            record["phases"][-1]["status"] = "failed"
        raise
    finally:
        # InterruptedError reaches child()'s own finally before this point. Never
        # remove a cache if that helper could not establish group quiescence.
        with defer_termination():
            record.update(scratch=str(scratch), elapsed_seconds=time.monotonic() - started)
            try:
                if safe_cleanup:
                    shutil.rmtree(scratch)
                    record["scratch_cleanup"] = "removed"
                else:
                    record["scratch_cleanup"] = "retained-unreaped"
            except OSError as error:
                record.update(status="failed", scratch_cleanup="failed", cleanup_error=str(error))
                raise
            finally:
                write(path, record)


def tensor_spec(tensor):
    """Describe actual component storage without relabeling it as training evidence."""
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
                storage_offset=tensor.storage_offset(), pointer_mod_16=tensor.data_ptr() % 16,
                pointer_mod_256=tensor.data_ptr() % 256, contiguous=tensor.is_contiguous())


def loaded_libraries():
    """Hash the actual mapped TE/hipBLASLt libraries, not a guessed search-path file."""
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and re.search(
                r"(?:libtransformer_engine|transformer_engine_torch|libhipblaslt|libamdhip64|libtorch_hip|libtorch_cuda).*\.so", fields[5]):
            paths.add(Path(fields[5]).resolve(strict=True))
    for required in ("libtransformer_engine", "transformer_engine_torch", "libhipblaslt"):
        if not any(required in p.name for p in paths):
            raise ValueError(f"native mapped library absent: {required}")
    return [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(paths)]


def numerical_check(torch, actual, reference, label, checks, *, fp32, previous=None):
    """Check every element against tiled FP32 references with fixed gates."""
    totals = dict(finite=True, mismatched_elements=0, max_abs=0.0, squared_error=0.0, squared_reference=0.0)
    count = actual.numel()
    rtol = atol = 1e-4 if fp32 else 0.02
    for start in range(0, actual.shape[0], TILE):
        end = min(actual.shape[0], start + TILE)
        expected = reference(start, end)
        value = actual[start:end].float()
        if previous is not None:
            value = value - previous[start:end]
        finite = bool(torch.isfinite(value).all() and torch.isfinite(expected).all())
        totals["finite"] &= finite
        if not finite:
            continue
        diff = value - expected
        totals["mismatched_elements"] += int((diff.abs() > atol + rtol * expected.abs()).sum().item())
        totals["max_abs"] = max(totals["max_abs"], diff.abs().max().item())
        totals["squared_error"] += diff.square().sum(dtype=torch.float64).item()
        totals["squared_reference"] += expected.square().sum(dtype=torch.float64).item()
    totals.update(rms_error=math.sqrt(totals["squared_error"] / count),
                  reference_rms=math.sqrt(totals["squared_reference"] / count),
                  relative_l2=math.sqrt(totals["squared_error"] / max(totals["squared_reference"], 1e-60)))
    failures = metric_failures(totals, fp32)
    checks.append(dict(label=label, status="failed" if failures else "passed", fp32_gate=fp32,
                       failures=failures, **{k: v if not isinstance(v, float) or math.isfinite(v) else None
                                            for k, v in totals.items()}))
    if failures:
        raise ValueError(f"numerical failure: {label}: {failures}")


def invoke(gemm, tensors, name, beta=0, *, control=False):
    """Use native TE general_gemm, with the pinned precision/layout/accumulator contract."""
    case = CASES[name]
    out = tensors["temporary"] if control else tensors[case["out"]]
    gemm_beta = 0 if control else beta
    gemm(tensors[case["A"]], tensors[case["B"]], out_dtype=out.dtype, out=out,
         layout=case["layout"], alpha=1.0, beta=float(gemm_beta), accumulate=bool(gemm_beta),
         grad=case["grad"], use_split_accumulator=case["split"], bias=None, gelu=False)
    if control:
        if beta:
            tensors["main_grad"].add_(out)
        else:
            tensors["main_grad"].copy_(out)


def validate_numerics(torch, gemm, tensors, checks):
    """Separate BF16 outputs, FP32 deltas/cumulative sums and unfused rounded gradients."""
    x, w, dy = (tensors[k] for k in ("X", "W", "dY"))
    for name in ("forward", "dgrad"):
        invoke(gemm, tensors, name)
        def reference(a, b):
            result = x[a:b].float() @ w.float().T if name == "forward" else dy[a:b].float() @ w.float()
            return result.to(torch.bfloat16).float()
        numerical_check(torch, tensors[CASES[name]["out"]], reference, name, checks, fp32=False)
    # Both independent BF16 inputs and a power-of-two exact cancellation sequence.
    for independent in (False, True):
        pairs = []
        base_x, base_dy = x.clone(), dy.clone()
        for coefficient in (1.0, -1.0, 0.25):
            pairs.append((torch.randn_like(x) * 0.25, torch.randn_like(dy) * 0.25) if independent
                         else (base_x, base_dy * coefficient))
        for control in (False, True):
            accumulator = tensors["main_grad"]
            accumulator.fill_(17 if independent else 0.125)
            initial = 0.0 if independent else 0.125
            for index, (current_x, current_dy) in enumerate(pairs):
                tensors.update(X=current_x, dY=current_dy)
                beta = int(not independent or index > 0)
                previous = accumulator.clone() if beta else None
                invoke(gemm, tensors, "wgrad", beta, control=control)
                def contribution(a, b, pair):
                    value = pair[1][:, a:b].float().T @ pair[0].float()
                    return value.to(torch.bfloat16).float() if control else value
                def cumulative(a, b):
                    value = torch.full((b - a, x.shape[1]), initial, dtype=torch.float32, device=x.device)
                    for pair in pairs[:index + 1]:
                        value.add_(contribution(a, b, pair))
                    return value
                label = f"{'unfused' if control else 'fp32'}/{'independent' if independent else 'cancellation'}/{index}"
                numerical_check(torch, accumulator, cumulative, label + "/cumulative", checks, fp32=not control)
                numerical_check(torch, accumulator, lambda a, b: contribution(a, b, pairs[index]),
                                label + "/delta", checks, fp32=not control, previous=previous)
                del previous
        del pairs, base_x, base_dy
    tensors.update(X=x, dY=dy)


def measure(torch, gemm, tensors, args):
    """Keep host return, synchronized completion and event interval distinct."""
    result = {}
    for name in (*CASES, "unfused-wgrad"):
        samples = []
        for index in range(args.warmups + args.samples):
            tensors["main_grad"].fill_(0.125)  # Reset outside timing, including beta1.
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            began = time.perf_counter()
            invoke(gemm, tensors, "wgrad" if name == "unfused-wgrad" else name,
                   int(name in {"wgrad", "unfused-wgrad"}), control=name == "unfused-wgrad")
            host = time.perf_counter() - began
            end.record()
            end.synchronize()
            if index >= args.warmups:
                samples.append(dict(host_seconds=host, completion_seconds=time.perf_counter() - began,
                                    event_ms=start.elapsed_time(end)))
        result[name] = {"samples": samples, "summary": {
            key: dict(median=statistics.median(v[key] for v in samples),
                      mean=statistics.mean(v[key] for v in samples),
                      minimum=min(v[key] for v in samples), maximum=max(v[key] for v in samples))
            for key in samples[0]}}
    return result


def gpu_phase(args):
    """Verify pinned installed consumers before importing/executing GPU dependencies."""
    path = args.output / f"{args.phase}.json"
    record = dict(status="running", phase=args.phase, checks=[], timings={}, script_sha256=digest(__file__),
                  native_provenance_contract=args.native_provenance, binary_attestation=False)
    started = time.monotonic()
    write(path, record)
    try:
        record["sources"] = verify_sources([PACKAGE], PY_SOURCES)
        if args.native_provenance == STRICT_PROVENANCE:
            record["native_sources"] = verify_sources([PACKAGE, Path("/workspace/TransformerEngine")], NATIVE_SOURCES)
            record["provenance"] = {"contract": STRICT_PROVENANCE, "binary_attestation": False}
        else:
            record["provenance"] = verify_build_provenance(PACKAGE, importlib.metadata.version("transformer-engine"))
        record["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformer-engine", "triton")}
        if record["versions"] != {"torch": "2.14.0+rocm10.1.0", "transformer-engine": TE_VERSION,
                                   "triton": "3.8.0+git669b31ac.rocm10.1.0"}:
            raise ValueError("pinned compute version mismatch")
        write(path, record)
        from apertus_amd.runtime import preserve_affinity, require_affinity
        affinity = record["affinity"] = {"thread_environment": {
            "before_imports": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MAX_JOBS")}}}
        with preserve_affinity(affinity) as initial_affinity:
            import torch
            import transformer_engine.pytorch.cpp_extensions.gemm as gemm_module

            if Path(gemm_module.__file__).resolve() != PACKAGE / "transformer_engine/pytorch/cpp_extensions/gemm.py":
                raise ValueError("loaded TE GEMM source path differs from verified source")
            local = int(os.environ["SLURM_LOCALID"])
            torch.cuda.set_device(local)
            properties = torch.cuda.get_device_properties(local)
            if not properties.gcnArchName.startswith("gfx942") or torch.cuda.get_device_capability(local) != (9, 4):
                raise ValueError("requires gfx942 MI300A")
        affinity["cpu_counts"] = {key: len(affinity[key]) for key in ("initial", "post_gpu_init", "final")}
        affinity["thread_environment"]["after_init"] = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MAX_JOBS")}
        torch.manual_seed(42)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        t, i, o = args.tokens, args.inputs, args.outputs
        tensors = {"X": torch.randn(t, i, device="cuda", dtype=torch.bfloat16) * 0.25,
                   "W": torch.randn(o, i, device="cuda", dtype=torch.bfloat16) * 0.02,
                   "dY": torch.randn(t, o, device="cuda", dtype=torch.bfloat16) * 0.25}
        for key, shape, dtype in (("Y", (t, o), torch.bfloat16), ("dX", (t, i), torch.bfloat16),
                                  ("main_grad", (o, i), torch.float32), ("temporary", (o, i), torch.bfloat16)):
            tensors[key] = torch.empty(shape, device="cuda", dtype=dtype)
        workspace = gemm_module.get_cublas_workspace(local, False, False)
        if workspace.numel() != WORKSPACE or not all(v.is_contiguous() and v.storage_offset() == 0 for v in tensors.values()):
            raise ValueError("canonical storage/workspace mismatch")
        record["libraries"] = loaded_libraries()
        if args.native_provenance == BUILD_PROVENANCE:
            record["mapped_te_provenance"] = verify_mapped_te_libraries(record["libraries"], PACKAGE)
        record.update(tensors={k: tensor_spec(v) for k, v in tensors.items()},
                      workspace=tensor_spec(workspace),
                      device=dict(name=properties.name, arch=properties.gcnArchName, local_rank=local),
                      torch_git=torch.version.git_version, torch_hip=torch.version.hip, te_commit=TE_COMMIT,
                      cases=CASES,
                      proof_parameters={name: dict(layout=case["layout"], alpha=1.0,
                          beta=int(name == "wgrad"), accumulate=name == "wgrad", grad=case["grad"],
                          use_split_accumulator=case["split"], output_dtype=case["dtype"],
                          bias=False, gelu=False, workspace_bytes=WORKSPACE) for name, case in CASES.items()},
                      environment={k: v for k, v in os.environ.items() if k.startswith(
                          ("TE_HIPBLASLT_", "NVTE_LOG", "PYTORCH_TUNABLEOP_", "HIPBLASLT_TUNING", "SLURM_LOCAL", "ROCR_VISIBLE", "HIP_VISIBLE", "CUDA_VISIBLE"))})
        record["import_setup_seconds"] = time.monotonic() - started
        affinity["before_gemm"] = sorted(os.sched_getaffinity(0))
        require_affinity(initial_affinity, "projection GEMMs")
        write(path, record)  # Actual layout/library evidence precedes the first GEMM.
        for name in CASES:
            tensors["main_grad"].fill_(0.125)
            print(f"KDA_PROOF_BEGIN {name}", flush=True)
            invoke(gemm_module.general_gemm, tensors, name, int(name == "wgrad"))
            torch.cuda.synchronize()
            print(f"KDA_PROOF_END {name}", flush=True)
        if args.phase != "tune":
            os.environ["NVTE_LOG_GEMM_CONFIG"] = "0"  # Don't time per-call stdout logging.
            try:
                validate_numerics(torch, gemm_module.general_gemm, tensors, record["checks"])
            finally:
                write(path, record)
            record["timings"] = measure(torch, gemm_module.general_gemm, tensors, args)
        affinity["final"] = sorted(os.sched_getaffinity(0))
        affinity["cpu_counts"]["final"] = len(affinity["final"])
        affinity["thread_environment"]["after_work"] = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MAX_JOBS")}
        require_affinity(initial_affinity, "projection phase completion")
        record.update(status="passed", tuning_outputs="sacrificial-discarded" if args.phase == "tune" else "not used")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        write(path, record)


def main(argv=None):
    """Select only this fixed component; GPU execution requires the future parent-owned job."""
    args = parser().parse_args(argv)
    validate_args(args)
    install_interrupts()
    args.output = args.output.resolve()
    if args.mode == "gpu":
        gpu_phase(args)
    else:
        settings = json.loads(Path("/apertus/resolved.json").read_text())
        orchestrate(args, settings, args.output / f"kda-projection-{os.environ['SLURM_PROCID']}")


if __name__ == "__main__":
    main()
