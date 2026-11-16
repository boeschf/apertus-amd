"""Inventory fixed installed TE source/metadata/ELF files without vendor imports or GPU use.

Missing files and version/hash mismatches are observations, not job failures.
This does not alter the tuning gate or attest source-to-binary correspondence.
"""

import argparse
import csv
import hashlib
from importlib.metadata import PathDistribution
import io
import json
import os
from pathlib import Path
import stat
import time


IMAGE = "cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0"
COMMIT = "130099ce48dc806398611de9960f8e0c486ad5f1"
VERSION = "2.17.1+130099ce4"
PACKAGE = Path("/opt/venv/lib/python3.12/site-packages")
WORKSPACE = Path("/workspace/TransformerEngine")
# Small frozen duplication: the selected file must not import an unfrozen sibling.
EXPECTED = {
    "transformer_engine/pytorch/cpp_extensions/gemm.py": "957974b26d32ab74f1521780d96e2d9e56a30e3a8420d7bb066c69cb4c6a5e9d",
    "transformer_engine/pytorch/module/base.py": "2b3ed9049450cd7002185584a9e4a0ee6176b40d3f2e33f1f30636622ab3691f",
    "transformer_engine/pytorch/module/layernorm_linear.py": "d5f177055fa489f7b922dccbb7f21c16f36b806116d6cab7bd0d5ca3b7400c73",
    "transformer_engine/common/gemm/rocm_gemm.cu": "66451919925d0206853597fac8e15e806254c08a1bdb77710f21ba45269f0016",
    "transformer_engine/common/gemm/cublaslt_gemm.cu": "d21c3828938fac30fd171ecd1b384514debec894beebc02aa82c5f36f6dae668",
    "transformer_engine/pytorch/csrc/extensions/gemm.cpp": "f68a35ccf3238f25cee096a7149d24121a2ae764f62800f26c6cddafffb51dd1",
}
NATIVE = tuple(name for name in EXPECTED if name.endswith((".cu", ".cpp")))


def deadline(state):
    """Bound cooperative reads; the existing commands runtime enforces the hard timeout."""
    if time.monotonic() >= state["deadline"]:
        raise TimeoutError("metadata inventory deadline")


def fingerprint(info):
    """Detect inode/content-metadata changes across a read or inventory."""
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def selected_path(path, root):
    """Observe absence/symlinks without reading targets outside the selected root."""
    try:
        path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "observation": "absent"}, None
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root.resolve()):
        return {"path": str(path), "observation": "external-symlink-not-followed", "target": str(resolved)}, None
    try:
        info = resolved.stat()
    except FileNotFoundError:
        return {"path": str(path), "observation": "dangling-symlink-not-followed"}, None
    if not stat.S_ISREG(info.st_mode):
        return {"path": str(path), "observation": "not-regular-file"}, None
    return {"path": str(path), "resolved": str(resolved), "bytes": info.st_size, "observation": "present"}, resolved


def inspect_file(path, root, state, *, expected=None, text_limit=None):
    """Hash bounded regular files and optionally capture bounded UTF-8 metadata text."""
    deadline(state)
    result, resolved = selected_path(path, root)
    if resolved is None:
        return result
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > state["remaining"]:
            result["hash"] = "skipped-total-byte-budget"
            return result
        if text_limit is not None and before.st_size > text_limit:
            raise ValueError(f"metadata text bound exceeded: {path}")
        h, blocks, count, magic = hashlib.sha256(), [], 0, b""
        while True:
            deadline(state)
            block = stream.read(min(1024 * 1024, state["remaining"] + 1))
            if not block:
                break
            if len(block) > state["remaining"]:
                raise ValueError(f"file grew beyond inventory byte budget: {path}")
            state["remaining"] -= len(block)
            count += len(block)
            if not magic:
                magic = block[:4]
            h.update(block)
            if text_limit is not None:
                if count > text_limit:
                    raise ValueError(f"metadata text grew beyond bound: {path}")
                blocks.append(block)
        if fingerprint(before) != fingerprint(os.fstat(stream.fileno())) or fingerprint(before) != fingerprint(resolved.stat()):
            raise ValueError(f"inventory file changed: {path}")
    result.update(sha256=h.hexdigest(), elf=magic == b"\x7fELF", stat_fingerprint=list(fingerprint(before)))
    if expected is not None:
        result["expected_sha256"] = expected
        result["matches_expected"] = result["sha256"] == expected
    if text_limit is not None:
        result["text"] = b"".join(blocks).decode("utf-8")
    return result


def record_summary(text, state, entry):
    """Count bounded RECORD text incrementally and retain limited match/schema evidence."""
    reader = csv.reader(io.StringIO(text))
    try:
        for number, row in enumerate(reader, 1):
            deadline(state)
            entry["record_entry_count"] = number
            name = row[0] if row else ""
            if len(row) != 3:
                entry["record_schema_valid"] = False
                entry["record_bad_row_width_count"] += 1
                if len(entry["record_bad_row_width_diagnostics"]) < 8:
                    entry["record_bad_row_width_diagnostics"].append(
                        {"row_number": number, "width": len(row), "path_prefix": name[:512]})
                else:
                    entry["record_bad_row_width_diagnostics_truncated_count"] += 1
            if name in EXPECTED:
                entry["expected_sources_listed_in_RECORD"][name] = True
            if name in EXPECTED or Path(name).name == "build_info.txt" or ".so" in Path(name).suffixes:
                entry["relevant_record_entry_match_count"] += 1
                if len(entry["relevant_record_entries"]) < 64:
                    entry["relevant_record_entries"].append(row)
                else:
                    entry["relevant_record_entries_truncated_count"] += 1
    except csv.Error as error:
        entry.update(record_schema_valid=False, record_parse_error=str(error)[:1024],
                     record_parse_error_physical_line=reader.line_num)
        return
    entry["record_count_complete"] = True


def inventory(package, workspace, *, max_hash_bytes, seconds, result=None):
    """Collect into the caller's record so failures retain all preceding observations."""
    state = {"remaining": max_hash_bytes, "deadline": time.monotonic() + seconds}
    if result is None:
        result = {}
    result.update({"scope": "installed TE metadata/files only; GPUs unused", "sources": [], "distributions": [],
              "build_info": [], "elf_candidates": [], "runtime_loaded_selection": "not observed; no imports or dlopen",
              "binary_attestation": False, "tuning_readiness": "no automatic change to existing gate"})
    for root, names in ((package, EXPECTED), (workspace, NATIVE)):
        for name in names:
            result["sources"].append(inspect_file(root / name, root, state, expected=EXPECTED[name]))
    metadata_snapshots = []
    directories = sorted(set(package.glob("transformer_engine*.dist-info")) |
                         set(package.glob("transformer-engine*.dist-info")))
    if len(directories) > 16:
        raise ValueError("unexpected number of TE distributions")
    for directory in directories:
        deadline(state)
        # Guard the metadata directory itself before PathDistribution reads it.
        if not directory.resolve().is_relative_to(package.resolve()) or not directory.is_dir():
            result["distributions"].append({"path": str(directory), "observation": "external-or-nondirectory-not-followed"})
            continue
        entry = {"path": str(directory), "record_entry_count": 0, "record_count_complete": False,
                 "record_schema_valid": None, "record_bad_row_width_count": 0,
                 "record_bad_row_width_diagnostics": [], "record_bad_row_width_diagnostics_truncated_count": 0,
                 "relevant_record_entries": [], "relevant_record_entry_match_count": 0,
                 "relevant_record_entries_truncated_count": 0,
                 "expected_sources_listed_in_RECORD": {name: False for name in EXPECTED}}
        result["distributions"].append(entry)
        files = {}
        for name, limit in (("METADATA", 1024 * 1024), ("RECORD", 8 * 1024 * 1024)):
            item = files[name] = inspect_file(directory / name, package, state, text_limit=limit)
            entry[name.lower() + "_observation"] = item["observation"]
            entry[name.lower() + "_sha256"] = item.get("sha256")
            if "stat_fingerprint" in item:
                metadata_snapshots.append((directory / name, tuple(item["stat_fingerprint"]),
                                           item["sha256"], 1024 * 1024 if name == "METADATA" else 8 * 1024 * 1024))
        if "text" in files["RECORD"]:
            entry["record_schema_valid"] = True
            record_summary(files["RECORD"]["text"], state, entry)
        if "text" not in files["METADATA"]:
            entry["observation"] = "metadata-unavailable"
            continue
        # Only stdlib metadata parsing; never import the distribution's code.
        dist = PathDistribution(directory)
        metadata = dist.metadata
        entry.update(name=metadata.get("Name"), version=metadata.get("Version"),
                     matches_expected_version=metadata.get("Version") == VERSION)
    build_info = inspect_file(package / "transformer_engine/build_info.txt", package, state, text_limit=16384)
    if "text" in build_info:
        fields = dict(line.split(":", 1) for line in build_info["text"].splitlines() if ":" in line)
        build_info["fields"] = {key.strip(): value.strip() for key, value in fields.items()}
        build_info["matches_expected_COMMIT_ID"] = build_info["fields"].get("COMMIT_ID") == COMMIT
    result["build_info"].append(build_info)
    candidates = set()
    for directory in (package, package / "transformer_engine", package / "transformer_engine/wheel_lib"):
        if not directory.resolve().is_relative_to(package.resolve()) or not directory.is_dir():
            continue
        for pattern in ("libtransformer_engine*.so*", "transformer_engine_torch*.so*", "transformer_engine_rocm_torch*.so*"):
            candidates.update(directory.glob(pattern))
    if len(candidates) > 32:
        raise ValueError("unexpected number of TE ELF candidates")
    for path in sorted(candidates):
        result["elf_candidates"].append(inspect_file(path, package, state))
    for path, before, before_sha256, text_limit in metadata_snapshots:
        after = inspect_file(path, package, state, text_limit=text_limit)
        if after.get("hash") == "skipped-total-byte-budget":
            raise ValueError(f"metadata recheck exceeds inventory hash budget: {path}")
        if tuple(after.get("stat_fingerprint", ())) != before or after.get("sha256") != before_sha256:
            raise ValueError(f"distribution metadata changed during inventory: {path}")
    deadline(state)
    result["hashed_bytes"] = max_hash_bytes - state["remaining"]
    if any(entry.get("record_schema_valid") is False for entry in result["distributions"]):
        raise ValueError("invalid TE RECORD schema; see retained per-distribution diagnostics")
    return result


def validate_launch(settings, environ):
    """Require the parent's explicit metadata-only resource envelope without device calls."""
    resource = settings["resources"]
    if (settings["image"]["sha256"] != IMAGE or resource["nodes"] != 1 or resource["gpus_per_node"] != 4
            or resource["memory_gb"] != 2 or resource["time"] != "00:05:00" or resource.get("reservation") != "cscs"
            or int(environ["SLURM_NTASKS"]) != 4 or not 0 <= int(environ["SLURM_PROCID"]) == int(environ["SLURM_LOCALID"]) < 4):
        raise ValueError("requires immutable image, one node/four CPU ranks, 2GiB, five minutes and cscs")


def main(argv=None):
    """Write one exclusive per-rank observation; absence/mismatch is not execution failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-hash-bytes", type=int, required=True)
    parser.add_argument("--deadline-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    if args.max_hash_bytes != 1073741824 or args.deadline_seconds != 110:
        raise ValueError("requires reviewed one-GiB/110-second inventory bounds")
    settings = json.loads(Path("/apertus/resolved.json").read_text())
    validate_launch(settings, os.environ)
    rank = os.environ["SLURM_PROCID"]
    path = args.output / f"te-source-inventory-{rank}.json"
    result = {"status": "running", "rank": int(rank), "image": IMAGE,
              "gpu_use": False, "vendor_imports": False, "binary_attestation": False}
    with path.open("x") as stream:
        stream.write(json.dumps(result) + "\n")
    try:
        inventory(PACKAGE, WORKSPACE, max_hash_bytes=args.max_hash_bytes, seconds=args.deadline_seconds, result=result)
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)


if __name__ == "__main__":
    main()
