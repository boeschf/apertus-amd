"""Portable compiler objects and immutable generations shared by compatible runs.

Existing objects win; new keys are added without mixing whole-object versions.
Triton group pointers are relocated when hydrating each private rank cache.
Call capture only with quiescent private caches, and publish only after checks.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import time

from .artifacts import sha256, write_json


SCHEMA = 2
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
TRITON_KEY = re.compile(r"[A-Za-z0-9_-]{16,128}\Z")
TRITON_SUFFIXES = {".json", ".source", ".hsaco", ".cubin", ".ptx", ".llir", ".ttir", ".ttgir", ".amdgcn"}


def canonical(value: object) -> bytes:
    """Serialize content identities independently of formatting and insertion order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(descriptor: dict) -> str:
    """Address compatible inputs, excluding experiment names and runtime paths."""
    return hashlib.sha256(canonical(descriptor)).hexdigest()


def object_key(name: str) -> str:
    """Validate a portable artifact name and return its indivisible merge unit."""
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts:
        raise ValueError(f"unsafe cache path: {name}")
    parts = path.parts
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise ValueError(f"unsafe cache path: {name}")
    if len(parts) == 3 and parts[0] == "triton" and TRITON_KEY.fullmatch(parts[1]):
        if path.suffix in TRITON_SUFFIXES and not any(word in path.name.lower() for word in ("autotun", "pickle")):
            return "/".join(parts[:2])
    # Only installed AITER modules and source-matched dataset helpers are
    # portable here. Arbitrary Ninja/Inductor/MIOpen build trees are not.
    if len(parts) == 2 and parts[0] == "aiter" and path.suffix == ".so":
        return name
    if len(parts) > 2 and parts[:2] == ("helpers", "megatron") and path.suffix == ".so":
        return name
    raise ValueError(f"unsupported cache artifact: {name}")


def group_bytes(data: bytes, names: set[str], *, source: Path | None = None,
                destination: Path | None = None) -> bytes:
    """Canonicalize group pointers or relocate them into the final private root."""
    value = json.loads(data)
    if not isinstance(value, dict) or set(value) != {"child_paths"} or not isinstance(value["child_paths"], dict):
        raise ValueError("unsupported Triton group schema")
    children = {}
    for name, pointer in value["child_paths"].items():
        if name not in names or not isinstance(pointer, str):
            raise ValueError("Triton group references an absent artifact")
        expected = str(source / name) if source is not None else name
        if pointer != expected:
            raise ValueError("Triton group points outside its compilation object")
        children[name] = str(destination / name) if destination is not None else name
    if not children:
        raise ValueError("empty Triton compilation group")
    return canonical({"child_paths": children})


def capture(payload: Path, destination: Path, *, max_bytes: int) -> dict:
    """Capture quiescent portable objects in a new bundle; omit mutable cache state."""
    destination.mkdir()
    files, skipped, total = {}, [], 0
    try:
        candidates = {}
        for path in sorted(payload.rglob("*")):
            name = str(path.relative_to(payload))
            if path.is_symlink():
                raise ValueError(f"nonportable cache link: {name}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"nonportable cache entry: {name}")
            try:
                key = object_key(name)
            except ValueError:
                skipped.append(name)
                continue
            candidates.setdefault(key, {})[name] = path
        with tarfile.open(destination / "payload.tar", "x", format=tarfile.PAX_FORMAT) as archive:
            for key, members in candidates.items():
                groups = {}
                if key.startswith("triton/"):
                    names = {path.name for path in members.values()}
                    for name, path in members.items():
                        if path.name.startswith("__grp__"):
                            if path.stat().st_size > 2**20:
                                raise ValueError("oversized Triton group")
                            groups[name] = group_bytes(path.read_bytes(), names, source=path.parent)
                    if not any(path.name.startswith("__grp__") for path in members.values()) or not any(
                        path.suffix in {".hsaco", ".cubin"} for path in members.values()
                    ):
                        skipped.append(key + ":incomplete-object")
                        continue
                for name, path in members.items():
                    data = groups.get(name)
                    stat = path.stat()
                    size = len(data) if data is not None else stat.st_size
                    total += size
                    if total > max_bytes or len(files) >= 100_000:
                        raise ValueError("cache exceeds capture budget")
                    member = tarfile.TarInfo(name)
                    member.size, member.mode = size, 0o600
                    # Integer nanoseconds are retained separately; tar float mtime
                    # roundtrips are insufficient for compiled-helper reuse checks.
                    with (io.BytesIO(data) if data is not None else path.open("rb")) as stream:
                        archive.addfile(member, stream)
                    files[name] = {"sha256": hashlib.sha256(data).hexdigest() if data is not None else sha256(path),
                                   "size": size, "mtime_ns": stat.st_mtime_ns}
        record = {"schema": SCHEMA, "files": files, "skipped": skipped,
                  "sha256": sha256(destination / "payload.tar")}
        write_json(destination / "manifest.json", record)
        return record
    except BaseException:
        shutil.rmtree(destination)
        raise


def objects(files: dict) -> dict:
    """Group byte identities, excluding timestamps, by indivisible compiled object."""
    result = {}
    for name, metadata in files.items():
        result.setdefault(object_key(name), {})[name] = {k: metadata[k] for k in ("size", "sha256")}
    return result


def merge_objects(current: dict, incoming: dict) -> tuple[dict, list[str], int]:
    """Keep occupied keys, add complete missing objects, and count differing contributions."""
    merged = dict(current)
    accepted = []
    differing = 0
    for key, value in incoming.items():
        if key in current:
            differing += value != current[key]
            continue
        merged[key] = value
        accepted.append(key)
    return merged, accepted, differing


def verify(bundle: Path, record: dict, *, max_bytes: int) -> None:
    """Verify archive bytes, regular entries, budgets, manifests, and group closure."""
    archive = bundle / "payload.tar"
    if record["schema"] != SCHEMA or sha256(archive) != record["sha256"]:
        raise ValueError("compiler cache checksum/schema mismatch")
    actual, groups, children, total = {}, {}, {}, 0
    with tarfile.open(archive, "r:") as stream:
        for entry in stream:
            object_key(entry.name)
            if not entry.isfile() or entry.name in actual:
                raise ValueError("nonregular or duplicate cache member")
            total += entry.size
            if total > max_bytes or len(actual) >= 100_000:
                raise ValueError("cache exceeds restoration budget")
            with stream.extractfile(entry) as source:
                if Path(entry.name).name.startswith("__grp__"):
                    if entry.size > 2**20:
                        raise ValueError("oversized Triton group")
                    data = source.read()
                    groups[entry.name] = data
                    checksum = hashlib.sha256(data).hexdigest()
                else:
                    checksum = hashlib.file_digest(source, "sha256").hexdigest()
            actual[entry.name] = {"size": entry.size, "sha256": checksum}
            path = PurePosixPath(entry.name)
            children.setdefault(path.parent, set()).add(path.name)
    if objects(actual) != objects(record["files"]):
        raise ValueError("cache file manifest mismatch")
    for name, data in groups.items():
        parent = PurePosixPath(name).parent
        # Index once: scanning every member for every group is quadratic across
        # thousands of kernels, repeated for every rank during publication.
        group_bytes(data, children[parent])


def _generation(namespace: Path, generation: str, descriptor: dict, max_bytes: int) -> tuple[Path, dict]:
    if not DIGEST.fullmatch(generation):
        raise ValueError("invalid cache generation")
    directory = namespace / "generations" / generation
    if not (directory / "COMPLETE").is_file():
        raise ValueError("incomplete cache generation")
    record = json.loads((directory / "manifest.json").read_text())
    if record["descriptor"] != descriptor or record["generation"] != generation or \
            identity(objects(record["files"])) != generation:
        raise ValueError("incompatible compiler cache generation")
    verify(directory, record, max_bytes=max_bytes)
    return directory, record


def stage(root: Path, descriptor: dict, destination: Path, *, max_bytes: int,
          generation: str | None = None) -> dict:
    """Stage one selected immutable snapshot per node; explicit warm misses fail."""
    namespace = root / identity(descriptor)
    if generation is None:
        generation = (namespace / "CURRENT").read_text().strip()
    source, record = _generation(namespace, generation, descriptor, max_bytes)
    destination.mkdir()
    try:
        shutil.copyfile(source / "payload.tar", destination / "payload.tar")
        write_json(destination / "manifest.json", record)
        verify(destination, record, max_bytes=max_bytes)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return record


def restore(staged: Path, destination: Path, *, max_bytes: int) -> dict:
    """Hydrate a fresh private cache from a node snapshot, relocating final paths."""
    record = json.loads((staged / "manifest.json").read_text())
    verify(staged, record, max_bytes=max_bytes)
    destination.mkdir()
    try:
        with tarfile.open(staged / "payload.tar", "r:") as stream:
            for entry in stream:
                target = destination / entry.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output, stream.extractfile(entry) as source:
                    shutil.copyfileobj(source, output)
                modified = record["files"][entry.name]["mtime_ns"]
                os.utime(target, ns=(modified, modified))
        for group in (destination / "triton").glob("*/__grp__*.json"):
            group.write_bytes(group_bytes(group.read_bytes(), {p.name for p in group.parent.iterdir()},
                                          destination=group.parent.resolve()))
    except BaseException:
        shutil.rmtree(destination)
        raise
    return record


@contextmanager
def publication_lock(namespace: Path, timeout: float):
    """Serialize writers with a deadline; process death releases the advisory lock."""
    namespace.mkdir(parents=True, exist_ok=True)
    with (namespace / "publish.lock").open("a") as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("cache publication lock deadline exceeded")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(root: Path, descriptor: dict, bundles: list[Path], *,
            max_bytes: int, timeout: float = 30) -> dict:
    """Add missing whole objects after checks; return generation and per-call difference count.

    The caller owns contributor completeness (all ranks for distributed runs).
    Existing keys win, including against cold rebuilds. For new keys, bundle
    order selects the producer; the lock serializes competing publications.
    Difference counts belong to the run, never the immutable generation manifest.
    """
    namespace = root / identity(descriptor)
    with publication_lock(namespace, timeout):
        current = (namespace / "CURRENT").read_text().strip() if (namespace / "CURRENT").exists() else None
        previous = _generation(namespace, current, descriptor, max_bytes) if current else None
        merged = objects(previous[1]["files"]) if previous else {}
        # Pick one complete source per object; never combine versions file-by-file.
        sources = {key: previous for key in merged}
        differing = 0
        for bundle in bundles:
            incoming = json.loads((bundle / "manifest.json").read_text())
            verify(bundle, incoming, max_bytes=max_bytes)
            merged, accepted, count = merge_objects(merged, objects(incoming["files"]))
            differing += count
            sources.update({key: (bundle, incoming) for key in accepted})
        if not merged:
            raise ValueError("no portable compiled artifacts to publish")
        files = {name: sources[key][1]["files"][name] for key, members in merged.items() for name in members}
        if sum(item["size"] for item in files.values()) > max_bytes:
            raise ValueError("merged cache exceeds restoration budget")
        generation = identity(merged)
        generations = namespace / "generations"
        generations.mkdir(exist_ok=True)
        target = generations / generation
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix=".generation-", dir=generations) as temporary:
                staging = Path(temporary) / "ready"
                staging.mkdir()
                with tarfile.open(staging / "payload.tar", "x", format=tarfile.PAX_FORMAT) as output:
                    unique = {str(bundle): (bundle, record) for bundle, record in sources.values()}
                    for bundle, record in unique.values():
                        with tarfile.open(bundle / "payload.tar", "r:") as archive:
                            for entry in archive:
                                if sources.get(object_key(entry.name), (None,))[0] == bundle:
                                    with archive.extractfile(entry) as stream:
                                        output.addfile(entry, stream)
                record = {"schema": SCHEMA, "status": "ready", "descriptor": descriptor,
                          "generation": generation, "files": files,
                          "sha256": sha256(staging / "payload.tar")}
                write_json(staging / "manifest.json", record)
                verify(staging, record, max_bytes=max_bytes)
                _sync(staging / "payload.tar")
                _sync(staging / "manifest.json")
                (staging / "COMPLETE").write_text("complete\n")
                _sync(staging / "COMPLETE")
                _sync(staging)
                staging.rename(target)
                _sync(generations)
        else:
            _, record = _generation(namespace, generation, descriptor, max_bytes)
        # Reusing identical content must not rewrite an immutable generation.
        fd, temporary = tempfile.mkstemp(prefix=".CURRENT-", dir=namespace)
        pointer = Path(temporary)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(generation + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            pointer.replace(namespace / "CURRENT")
            _sync(namespace)
        finally:
            pointer.unlink(missing_ok=True)
        return {**record, "differing_contributions": differing}
