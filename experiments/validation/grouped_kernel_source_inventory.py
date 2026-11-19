"""Bounded source-only TE inventory; no vendor imports, GPU use or binary attestation."""

import ast
import base64
import csv
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
PACKAGE = Path("/opt/venv/lib/python3.12/site-packages")
DIST = "transformer_engine-2.17.1+130099ce4.dist-info"
TE = "transformer_engine/pytorch/"
KERNELS = TE + "triton_kernels"
TARGET = KERNELS + "/grouped_gemm.py"
WRAPPER = KERNELS + "/gmm/gmm_wrapper.py"
PINS = {TE + "module/grouped_linear.py": (92320, "bc52bcbeb1e5c6e696c3869814d9e9bebe37d507a06beb5e16ccef43de988dea"),
        TE + "cpp_extensions/gemm.py": (29761, "957974b26d32ab74f1521780d96e2d9e56a30e3a8420d7bb066c69cb4c6a5e9d")}
MAX_FILES, FILE_BYTES, TOTAL_BYTES, ENTRIES = 64, 2 * 1024**2, 8 * 1024**2, 128
RECORD_BYTES, RECORD_ROWS, LINE_BYTES = 8 * 1024**2, 50000, 65536
COMMAND = ["python3", "-u", "/apertus/validation.py", "."]


def directory(fd, parts):
    """Open every path component without following symlinks; caller owns returned FD."""
    current = os.dup(fd)
    try:
        for part in parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def fingerprint(info):
    """Ignore atime, retaining file identity and content-change indicators."""
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def record_rows(root, wanted, result, deadline):
    """Stream bounded raw RECORD bytes; retain selected rows only after correspondence checks."""
    parent = directory(root, [DIST])
    try:
        fd = os.open("RECORD", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            summary = result["record"] = dict(name=DIST + "/RECORD", bytes=before.st_size,
                initial_stat=fingerprint(before), bytes_read=0, rows=0, complete=False)
            if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= RECORD_BYTES:
                raise ValueError("RECORD not regular or exceeds 8MiB read cap")
            digest, owners, logical_bytes = hashlib.sha256(), set(), 0
            def lines():
                nonlocal logical_bytes
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("RECORD deadline")
                    data = stream.readline(min(LINE_BYTES + 1, before.st_size - summary["bytes_read"] + 1))
                    if not data:
                        return
                    summary["bytes_read"] += len(data)
                    digest.update(data)
                    summary["sha256"] = digest.hexdigest()
                    logical_bytes += len(data)
                    if len(data) > LINE_BYTES or summary["bytes_read"] > before.st_size or logical_bytes > 262144:
                        raise ValueError("RECORD line/logical-row/read-byte cap")
                    if b"\0" in data:
                        raise ValueError("RECORD NUL")
                    yield data.decode("utf-8")
            for fields in csv.reader(lines(), strict=True):
                logical_bytes = 0
                summary["rows"] += 1
                if summary["rows"] > RECORD_ROWS or len(fields) != 3:
                    raise ValueError("RECORD row cap/schema")
                if fields[0] not in wanted:
                    continue
                if fields[0] in owners:
                    raise ValueError("duplicate selected RECORD row")
                row = wanted[fields[0]]
                expected = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(row["sha256"])).decode().rstrip("=")
                row["RECORD"] = dict(row=fields, number=summary["rows"], matches=fields[1:] == [expected, str(row["bytes"])])
                if not row["RECORD"]["matches"]:
                    raise ValueError("selected RECORD correspondence mismatch")
                owners.add(fields[0])
            summary.update(sha256=digest.hexdigest(), final_stat=fingerprint(os.fstat(stream.fileno())))
            if (summary["bytes_read"] != before.st_size or summary["final_stat"] != fingerprint(before)
                    or fingerprint(os.stat("RECORD", dir_fd=parent, follow_symlinks=False)) != fingerprint(before)):
                raise ValueError("RECORD changed during read")
            if owners != set(wanted):
                raise ValueError("selected source missing RECORD row")
            summary["complete"] = True
    finally:
        os.close(parent)


def inventory(package, result, *, seconds=45):
    """Export only selected regular UTF8 files under bounded FD-owned TE scopes."""
    deadline, remaining = time.monotonic() + seconds, TOTAL_BYTES
    anchor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        root = directory(anchor, Path(package).absolute().parts[1:])
    finally:
        os.close(anchor)
    def read(name, limit, source=False):
        nonlocal remaining
        if time.monotonic() >= deadline:
            raise TimeoutError("inventory deadline")
        parent = directory(root, name.split("/")[:-1])
        try:
            fd = os.open(name.split("/")[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size > min(limit, remaining if source else limit):
                    raise ValueError("not regular or byte cap exceeded before read")
                data = os.read(fd, before.st_size + 1)
                signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                if (len(data) != before.st_size or signature(before) != signature(os.fstat(fd))
                        or signature(before) != signature(os.stat(name.split("/")[-1], dir_fd=parent, follow_symlinks=False))):
                    raise ValueError("file changed during read")
                if data.startswith(b"\x7fELF"):
                    raise ValueError("binary payload forbidden")
                if source:
                    remaining -= len(data)
                return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "text": data.decode("utf-8")}
            finally:
                os.close(fd)
        finally:
            os.close(parent)
    try:
        result.update(files=[], metadata={})
        selected, entry_count = [], 0
        def scan(name, depth):
            nonlocal entry_count
            fd = directory(root, name.split("/"))
            try:
                with os.scandir(fd) as entries:
                    for entry in entries:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("source snapshot deadline")
                        entry_count += 1
                        if entry_count > ENTRIES or entry.is_symlink():
                            raise ValueError("directory entry cap or symlink")
                        path = name + "/" + entry.name
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name != "__pycache__" and (depth or entry.name == "gmm"):
                                if depth >= 4:
                                    raise ValueError("source directory depth cap")
                                scan(path, depth + 1)
                        elif entry.name.endswith(".py"):
                            selected.append(path)
                            if len(selected) > MAX_FILES:
                                raise ValueError("source file-count cap")
            finally:
                os.close(fd)
        scan(KERNELS, 0)
        if TARGET not in selected or WRAPPER not in selected:
            raise ValueError("missing target or source file-count cap exceeded")
        for name in sorted(list(PINS) + selected):
            row = read(name, FILE_BYTES, True)
            result["files"].append(row)
            if name in PINS and (row["bytes"], row["sha256"]) != PINS[name]:
                raise ValueError("known caller pin mismatch: " + name)
            tree = ast.parse(row["text"], filename=name)
            row["imports"] = sorted({("." * n.level + (n.module or "") + ":" + a.name) if isinstance(n, ast.ImportFrom)
                                      else a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                                      for a in n.names})
        for name, cap in ((DIST + "/METADATA", 262144), ("transformer_engine/build_info.txt", 4096)):
            result["metadata"][name] = read(name, cap)
        metadata = Parser().parsestr(result["metadata"][DIST + "/METADATA"]["text"])
        if (len(metadata.get_all("Name", [])) != 1 or metadata["Name"].lower().replace("_", "-") != "transformer-engine"
                or metadata.get_all("Version") != ["2.17.1+130099ce4"]):
            raise ValueError("TE distribution identity mismatch")
        wanted = {r["name"]: r for r in result["files"]}
        record_rows(root, wanted, result, deadline)
        result["source_identity"] = hashlib.sha256(json.dumps(result["files"], sort_keys=True).encode()).hexdigest()
    finally:
        os.close(root)


def validate_launch(settings, env):
    """Require the reviewed image, site-default four-rank CPU placement and 60s command."""
    r, p = settings["resources"], settings["procedure"]
    expected = dict(nodes=1, gpus_per_node=4, cpus_per_node=192, time="00:05:00", reservation="cscs",
                    mpi="pmix", cpu_bind="map_ldom:0,1,2,3", distribution="block:block")
    if (settings["image"]["sha256"] != IMAGE or any(r.get(k) != v for k, v in expected.items())
            or r.get("cpus_per_task") is not None or p["kind"] != "commands" or p["timeout"] != 60
            or p["commands"] != [COMMAND] or int(env["SLURM_NTASKS"]) != 4 or int(env["SLURM_NODEID"]) != 0
            or not 0 <= int(env["SLURM_PROCID"]) == int(env["SLURM_LOCALID"]) < 4 or len(os.sched_getaffinity(0)) != 48):
        raise ValueError("image/resource/rank/affinity contract mismatch")


def main():
    """Exclusively close per-rank evidence, retaining explicit failures and partial sources."""
    settings = json.loads(Path("/apertus/resolved.json").read_text())
    validate_launch(settings, os.environ)
    result = dict(status="failed", rank=int(os.environ["SLURM_PROCID"]), image=IMAGE, gpu_use=False, binary_attestation=False)
    with (Path(sys.argv[1]) / f"grouped-kernel-source-{result['rank']}.json").open("x") as stream:
        try:
            result["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            if result["probe_sha256"] != settings["validation"]["sha256"]:
                raise ValueError("executed probe hash mismatch")
            inventory(PACKAGE, result)
            result["status"] = "observed"
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            result["closed_at"] = datetime.now(timezone.utc).isoformat()
            json.dump(result, stream, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
