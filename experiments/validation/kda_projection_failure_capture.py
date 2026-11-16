"""Capture the failed KDA BF16 control without accepting it or reaching a tuner.

This standalone variant preserves the f11e993 driver's default-prefix GEMMs,
RNG sequence and numerical gates. Only GPU children import Torch/TE. Captures
are authoritative output, not private scratch; numerical failure remains fatal.
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
import signal
import struct
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
              rank_timeout=180, max_coordinates=512, capture_bytes=33554432)
PHASES = ("default-0",)
TARGET = "unfused/independent/1/cumulative"
ORIGINAL_DRIVER_SHA256 = "f11e993b37f8bbc4f8dc6de3327e88602c2a92d9415dcf2e31985d45f34b2566"
WORKSPACE = 33554432
TILE = 512
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
    p.add_argument("--native-provenance", choices=(BUILD_PROVENANCE,), required=True)
    for name in BOUNDS:
        p.add_argument("--" + name.replace("_", "-"), type=int, required=True)
    return p


def validate_args(args):
    """Reject shape, precision-work bound or phase changes outside this component."""
    if any(getattr(args, name) != value for name, value in BOUNDS.items()):
        raise ValueError("only fixed unpadded KDA geometry and reviewed bounds are supported")
    if (args.mode == "gpu") != (args.phase is not None):
        raise ValueError("GPU children require a phase; the orchestrator must not select one")


def phase_environment(environ):
    """Select the original untuned default without changing device visibility."""
    env = {k: v for k, v in environ.items() if not k.startswith(("PYTORCH_TUNABLEOP_", "TE_HIPBLASLT_"))
           and k not in {"HIPBLASLT_TUNING_FILE", "HIPBLASLT_TUNING_OVERRIDE_FILE"}}
    env.update(TE_HIPBLASLT_ALGO_SELECTION="0", TE_HIPBLASLT_LOG_TUNING="1",
                TE_HIPBLASLT_TUNING_RUN_COUNT="0",
                TE_HIPBLASLT_TUNING_ALGO_COUNT="16", NVTE_LOG_GEMM_CONFIG="1")
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


def orchestrate(args, settings, output):
    """Own private rank scratch and one bounded child; never wait for sibling captures."""
    from apertus_amd.config import CACHE_PATHS
    from apertus_amd.runtime import child, UnreapedChildError, defer_termination

    rank, local = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_LOCALID"])
    if (settings["image"]["sha256"] != IMAGE or settings["resources"]["nodes"] != 1
            or settings["resources"]["gpus_per_node"] != 4 or settings["resources"]["time"] != "00:05:00"
            or settings["resources"].get("reservation") != "cscs" or int(os.environ["SLURM_NTASKS"]) != 4
            or not 0 <= rank == local < 4):
        raise ValueError("requires reviewed image and explicit one-node/four-rank/5min cscs launch")
    started = time.monotonic()
    deadline = started + args.rank_timeout
    output.mkdir()  # Exclusive per-rank evidence; never replace an old run.
    record = {"status": "running", "rank": rank, "scope": "canonical source-derived KDA TE GEMM boundary",
              "image": IMAGE, "script_sha256": digest(Path(__file__)), "phases": [],
              "native_provenance_contract": args.native_provenance, "binary_attestation": False,
              "original_driver_sha256": ORIGINAL_DRIVER_SHA256, "numerical_acceptance": False}
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
        for phase in PHASES:
            phase_args = argparse.Namespace(**vars(args))
            phase_args.output = output
            log = output / f"{phase}.log"
            item = {"phase": phase, "status": "running"}
            record["phases"].append(item)
            write(path, record)
            before = time.monotonic()
            child(child_command(phase_args, phase, Path(__file__).resolve()), log, cwd=output,
                  env=phase_environment(env),
                  timeout=remaining_budget(deadline, args.phase_timeout, before))
            result = json.loads((output / f"{phase}.json").read_text())
            if result["status"] != "diagnostic-complete" or result.get("outcome") != "nonreproduced":
                raise ValueError(f"incomplete GPU child: {phase}")
            item.update(status="diagnostic-complete", wall_seconds=time.monotonic() - before)
            write(path, record)
        remaining_budget(deadline, args.phase_timeout, time.monotonic())
        record.update(status="diagnostic-complete", outcome="nonreproduced")
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
                # The GPU child flushes its original failure/capture before exiting.
                # Collect after local reaping; never wait for sibling ranks.
                phase_path = output / "default-0.json"
                if phase_path.is_file():
                    try:
                        phase_result = json.loads(phase_path.read_text())
                        record["gpu_result"] = {key: phase_result.get(key) for key in
                                                ("status", "error", "outcome", "capture", "copy_add")}
                    except Exception as error:
                        record["collection_error"] = f"{type(error).__name__}: {error}"
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


def failure_coordinates(mask, start, limit):
    """Select bounded lexicographic coordinates without materializing a dense nonzero list."""
    coordinates = []
    if limit <= 0:
        return coordinates
    for row in mask.any(dim=1).nonzero().flatten().cpu().tolist():
        columns = mask[row].nonzero().flatten()[:limit - len(coordinates)].cpu().tolist()
        coordinates.extend([[start + row, column] for column in columns])
        if len(coordinates) == limit:
            break
    return coordinates


def encode_bf16_words(words):
    """Serialize signed or unsigned raw BF16 words explicitly as little-endian uint16."""
    return struct.pack(f"<{len(words)}H", *(word & 0xFFFF for word in words))


def tensor_word(torch, value, *, bf16=False):
    """Read a scalar's raw bits, never a decimal approximation of its value."""
    return int(value.view(torch.int16 if bf16 else torch.int32).item()) & (0xFFFF if bf16 else 0xFFFFFFFF)


def copy_add_check(torch, actual, native, *, previous=None):
    """Retain all-element bitwise copy/add diagnostics separately from original gates."""
    result = {"operation": "copy" if previous is None else "add", "mismatched_elements": 0, "selected": []}
    for start in range(0, actual.shape[0], TILE):
        end = min(actual.shape[0], start + TILE)
        expected = native[start:end].float()
        if previous is not None:
            expected = previous[start:end] + expected
        mask = actual[start:end].view(torch.int32) != expected.view(torch.int32)
        result["mismatched_elements"] += int(mask.sum().item())
        for row, column in failure_coordinates(mask, start, BOUNDS["max_coordinates"] - len(result["selected"])):
            point = {"coordinate": [row, column],
                     "actual_fp32_bits": tensor_word(torch, actual[row, column]),
                     "expected_fp32_bits": tensor_word(torch, expected[row - start, column]),
                     "native_bf16_bits": tensor_word(torch, native[row, column], bf16=True)}
            if previous is not None:
                point["previous_fp32_bits"] = tensor_word(torch, previous[row, column])
            result["selected"].append(point)
    result["coordinate_cap_exceeded"] = result["mismatched_elements"] > BOUNDS["max_coordinates"]
    return result


class FailureCapture:
    """Own one bounded authoritative capture while the original reference tiles are alive."""

    def __init__(self, torch, output, pairs, actual, native, previous, record):
        self.torch, self.output, self.pairs = torch, output, pairs
        self.actual, self.native, self.previous = actual, native, previous
        self.record = record
        self.record.update(status="collecting", coordinates=[], byte_order="little",
                           vector_dtype="raw BF16 uint16", vector_order=["dY", "X"],
                           original_driver_sha256=ORIGINAL_DRIVER_SHA256)
        self.reference_tiles = []
        self.error = None
        self.stream = None
        self.binary = output / "capture-inputs.bin"

    def remember_error(self, error):
        """Retain the first observer error without replacing a numerical exception."""
        if self.error is None:
            self.error = error
            self.record.update(status="incomplete", capture_error=f"{type(error).__name__}: {error}")

    def point(self, row, column):
        """Persist actual input vectors and native scalar bits for one coordinate."""
        torch = self.torch
        if self.stream is None:
            self.stream = self.binary.with_suffix(".bin.tmp").open("xb")
        point = {"coordinate": [row, column], "vectors": [],
                 "actual_fp32_bits": tensor_word(torch, self.actual[row, column]),
                 "native_bf16_bits": []}
        if self.previous is not None:
            previous = tensor_word(torch, self.previous[row, column])
            point["previous_fp32_bits"] = previous
            point["native_bf16_bits"].append(previous >> 16)
        point["native_bf16_bits"].append(tensor_word(torch, self.native[row, column], bf16=True))
        for index, (x, dy) in enumerate(self.pairs):
            for name, vector in (("dY", dy[:, row]), ("X", x[:, column])):
                words = vector.contiguous().view(torch.int16).cpu().flatten().tolist()
                data = encode_bf16_words(words)
                if self.stream.tell() + len(data) > BOUNDS["capture_bytes"] - 1024 * 1024:
                    raise ValueError("capture byte bound exceeded")
                offset = self.stream.tell()
                self.stream.write(data)
                point["vectors"].append({"pair": index, "name": name, "offset": offset,
                                         "words": len(words), "bytes": len(data)})
        self.record["coordinates"].append(point)
        return point

    def tile(self, start, expected, diff, mask, threshold):
        """Observe only the original failing tile, without another GEMM or random draw."""
        torch = self.torch
        if len(self.reference_tiles) != 2:
            raise ValueError("missing original contribution tiles")
        for row, column in failure_coordinates(mask, start, BOUNDS["max_coordinates"] - len(self.record["coordinates"])):
            point = self.point(row, column)
            local_row = row - start
            point["reference_fp32_bits"] = [tensor_word(torch, unrounded[local_row, column])
                                             for unrounded, rounded in self.reference_tiles]
            point["reference_bf16_bits"] = [tensor_word(torch, rounded[local_row, column], bf16=True)
                                             for unrounded, rounded in self.reference_tiles]
            point["cumulative_reference_fp32_bits"] = tensor_word(torch, expected[local_row, column])
            point["difference_fp32_bits"] = tensor_word(torch, diff[local_row, column])
            point["threshold_fp32_bits"] = tensor_word(torch, threshold[local_row, column])

    def finish(self, check):
        """Publish binary then its bounded index before the original failure is raised."""
        self.reference_tiles.clear()
        self.record["original_check"] = check
        count = check["mismatched_elements"]
        self.record["coordinate_cap_exceeded"] = count > BOUNDS["max_coordinates"]
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            os.replace(self.binary.with_suffix(".bin.tmp"), self.binary)
            self.record["binary"] = {"path": self.binary.name, "bytes": self.binary.stat().st_size,
                                      "sha256": digest(self.binary)}
        complete = (not self.error and check.get("finite", False) and not self.record["coordinate_cap_exceeded"]
                    and len(self.record["coordinates"]) == count
                    and not any("diagnostic_error" in item for item in self.record.get("copy_add", {}).values()))
        invariant_failure = any(item.get("mismatched_elements", 0) for item in self.record.get("copy_add", {}).values())
        self.record["status"] = ("copy-add-failure" if complete and invariant_failure else "captured"
                                 if complete and count else "nonreproduced"
                                 if complete and not check.get("failures") else "incomplete")
        metadata = json.dumps(self.record, indent=2, allow_nan=False).encode() + b"\n"
        if len(metadata) + self.record.get("binary", {}).get("bytes", 0) > BOUNDS["capture_bytes"] - 65536:
            raise ValueError("capture index plus binary exceeds output bound")
        write(self.output / "capture-index.json", self.record)


def observe(capture, method, *args):
    """Isolate observer errors until the original all-element numerical scan completes."""
    if capture.error is None or method == "finish":
        try:
            getattr(capture, method)(*args)
        except BaseException as error:
            capture.remember_error(error)
    if method == "tile":
        capture.reference_tiles.clear()


def numerical_check(torch, actual, reference, label, checks, *, fp32, previous=None, capture=None):
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
        threshold = atol + rtol * expected.abs()
        mask = diff.abs() > threshold
        totals["mismatched_elements"] += int(mask.sum().item())
        totals["max_abs"] = max(totals["max_abs"], diff.abs().max().item())
        totals["squared_error"] += diff.square().sum(dtype=torch.float64).item()
        totals["squared_reference"] += expected.square().sum(dtype=torch.float64).item()
        if capture is not None:
            observe(capture, "tile", start, expected, diff, mask, threshold)
    totals.update(rms_error=math.sqrt(totals["squared_error"] / count),
                  reference_rms=math.sqrt(totals["squared_reference"] / count),
                  relative_l2=math.sqrt(totals["squared_error"] / max(totals["squared_reference"], 1e-60)))
    failures = metric_failures(totals, fp32)
    checks.append(dict(label=label, status="failed" if failures else "passed", fp32_gate=fp32,
                       failures=failures, **{k: v if not isinstance(v, float) or math.isfinite(v) else None
                                             for k, v in totals.items()}))
    if capture is not None:
        observe(capture, "finish", checks[-1])
    if failures:
        raise ValueError(f"numerical failure: {label}: {failures}")
    if capture is not None and capture.error is not None:
        raise capture.error


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


def validate_numerics(torch, gemm, tensors, checks, output, record):
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
                if control and index == 0 and not independent:
                    print("KDA_CAPTURE_BEGIN first-bf16-wgrad", flush=True)
                if control and independent and index == 1:
                    print("KDA_CAPTURE_BEGIN target-update", flush=True)
                invoke(gemm, tensors, "wgrad", beta, control=control)
                if control and index == 0 and not independent:
                    print("KDA_CAPTURE_END first-bf16-wgrad", flush=True)
                if control and independent and index == 1:
                    print("KDA_CAPTURE_END target-update", flush=True)
                capture = None
                add_error = None
                if control and independent and index == 1:
                    try:
                        record["copy_add"]["second_add"] = copy_add_check(
                            torch, accumulator, tensors["temporary"], previous=previous)
                    except BaseException as error:
                        add_error = error
                        record["copy_add"]["second_add"] = {"diagnostic_error": f"{type(error).__name__}: {error}"}
                    capture = FailureCapture(torch, output, pairs[:2], accumulator,
                                             tensors["temporary"], previous, record["capture"])
                    capture.record["copy_add"] = record["copy_add"]
                def contribution(a, b, pair):
                    value = pair[1][:, a:b].float().T @ pair[0].float()
                    if control:
                        rounded = value.to(torch.bfloat16)
                        if capture is not None:
                            capture.reference_tiles.append((value, rounded))
                        return rounded.float()
                    return value
                def cumulative(a, b):
                    if capture is not None:
                        capture.reference_tiles.clear()
                    value = torch.full((b - a, x.shape[1]), initial, dtype=torch.float32, device=x.device)
                    for pair in pairs[:index + 1]:
                        value.add_(contribution(a, b, pair))
                    return value
                label = f"{'unfused' if control else 'fp32'}/{'independent' if independent else 'cancellation'}/{index}"
                numerical_check(torch, accumulator, cumulative, label + "/cumulative", checks,
                                fp32=not control, capture=capture)
                if capture is not None:
                    if add_error is not None:
                        raise add_error
                    if record["copy_add"]["second_add"]["mismatched_elements"]:
                        raise ValueError("bitwise add diagnostic failure")
                    record["outcome"] = "nonreproduced"
                    return  # No second-control delta, third update or performance path.
                numerical_check(torch, accumulator, lambda a, b: contribution(a, b, pairs[index]),
                                 label + "/delta", checks, fp32=not control, previous=previous)
                if control and independent and index == 0:
                    check = copy_add_check(torch, accumulator, tensors["temporary"])
                    record["copy_add"]["first_copy"] = check
                    if check["mismatched_elements"]:
                        failed_copy = FailureCapture(torch, output, pairs[:1], accumulator,
                                                     tensors["temporary"], None, record["capture"])
                        failed_copy.record["copy_add"] = record["copy_add"]
                        for point in check["selected"]:
                            observe(failed_copy, "point", *point["coordinate"])
                        observe(failed_copy, "finish", dict(label="bitwise-copy", finite=True,
                                                           mismatched_elements=check["mismatched_elements"]))
                        raise ValueError("bitwise copy diagnostic failure")
                del previous
        del pairs, base_x, base_dy
    tensors.update(X=x, dY=dy)


def gpu_phase(args):
    """Verify pinned installed consumers before importing/executing GPU dependencies."""
    path = args.output / f"{args.phase}.json"
    record = dict(status="running", phase=args.phase, checks=[], script_sha256=digest(__file__),
                  native_provenance_contract=args.native_provenance, binary_attestation=False,
                  original_driver_sha256=ORIGINAL_DRIVER_SHA256, numerical_acceptance=False,
                  capture={"status": "not-reached"}, copy_add={})
    started = time.monotonic()
    write(path, record)
    try:
        record["sources"] = verify_sources([PACKAGE], PY_SOURCES)
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
        validate_numerics(torch, gemm_module.general_gemm, tensors, record["checks"], args.output, record)
        affinity["final"] = sorted(os.sched_getaffinity(0))
        affinity["cpu_counts"]["final"] = len(affinity["final"])
        affinity["thread_environment"]["after_work"] = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MAX_JOBS")}
        require_affinity(initial_affinity, "projection phase completion")
        record.update(status="diagnostic-complete", outcome="nonreproduced")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        record["outcome"] = ("original-numerical-failure" if str(error).startswith("numerical failure:")
                             else "diagnostic-or-preflight-failure")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        # Evidence failures cannot replace the original numerical exception.
        active_error = sys.exc_info()[0] is not None
        try:
            write(path, record)
        except BaseException as error:
            if not active_error:
                raise
            print(f"KDA_CAPTURE_RESULT_WRITE_ERROR {type(error).__name__}: {error}", file=sys.stderr, flush=True)


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
