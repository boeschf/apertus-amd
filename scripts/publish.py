#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Publish an existing SquashFS, source archive, and provenance with Lustre striping."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


DEFAULT_STORE = Path("/capstor/store/cscs/cscs/public/team-ml/apertus-amd-images")
LUSTRE_STRIPES = (
    "--component-end", "4M", "--stripe-count", "1", "--stripe-size", "4M",
    "--component-end", "64M", "--stripe-count", "4", "--stripe-size", "4M",
    "--component-end", "-1", "--stripe-count", "32", "--stripe-size", "4M",
)


def configure_striping(directory: Path) -> None:
    """Set progressive striping on an empty publication directory when on Lustre."""
    filesystem = subprocess.check_output(
        ["stat", "--file-system", "--format=%T", "--", str(directory)], text=True).strip()
    if filesystem == "lustre":
        # Files inherit at creation; the final rename preserves their layout.
        # Fail before copying if Lustre tooling is missing or setstripe fails.
        subprocess.run(["lfs", "setstripe", *LUSTRE_STRIPES, str(directory)],
                       check=True, stdout=sys.stderr)


def digest(path):
    """Return the SHA-256 and byte size of a regular artifact."""
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": checksum, "bytes": path.stat().st_size}


def sync_directory(path):
    """Persist directory entries before reporting successful publication."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(image, source, provenance, store) -> Path:
    """Atomically publish an immutable object and return its absolute image path."""
    record = json.loads(provenance.read_text())
    if not isinstance(record, dict) or not {"source", "recipe", "build_args"} <= record.keys():
        raise ValueError("provenance requires source, recipe, and build_args")
    if not isinstance(record["source"], dict) or not {"repository", "head"} <= record["source"].keys():
        raise ValueError("source provenance requires repository and head (null if unknown)")
    if not isinstance(record["recipe"], str) or not isinstance(record["build_args"], dict):
        raise ValueError("recipe must be a string and build_args an object")
    store = store.resolve()
    staging = store / ".staging"
    locks = store / ".locks"
    objects = store / "objects" / "sha256"
    for directory in (staging, locks, objects):
        directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="publication-", dir=staging))
    lock = None
    try:
        configure_striping(temporary)
        artifacts = {}
        for name, original in (("image.sqsh", image), ("source.tar.gz", source)):
            target = temporary / name
            with original.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            target.chmod(0o644)
            artifacts[name] = digest(target)
        with (temporary / "image.sqsh").open("rb") as stream:
            if stream.read(4) != b"hsqs":
                raise ValueError("image is not a little-endian SquashFS file")
        # Validate the supplied archive without extracting or executing its contents.
        with tarfile.open(temporary / "source.tar.gz", "r:gz") as archive:
            for _ in archive:
                pass
        manifest = {"schema": 1, "artifacts": artifacts, "provenance": record}
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        manifest_path.chmod(0o644)
        checksum = artifacts["image.sqsh"]["sha256"]
        destination = objects / checksum
        candidate_lock = locks / checksum
        # Only the invocation that successfully creates the lock may remove it.
        candidate_lock.mkdir()
        lock = candidate_lock
        (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "host": os.uname().nodename}))
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise ValueError("published object must not be a symlink")
            existing = json.loads((destination / "manifest.json").read_text())
            if existing != manifest:
                raise ValueError("same image already published with different provenance or source")
            for name, expected in artifacts.items():
                if (destination / name).is_symlink() or digest(destination / name) != expected:
                    raise ValueError(f"published artifact is corrupt: {name}")
        else:
            temporary.chmod(0o755)
            sync_directory(temporary)
            # Same-filesystem rename publishes image, source, and manifest together.
            temporary.rename(destination)
            sync_directory(objects)
        return destination / "image.sqsh"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if lock is not None:
            shutil.rmtree(lock)


def main():
    """Parse standalone publication inputs and report an actionable failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE,
                        help=f"shared image store (default: {DEFAULT_STORE})")
    args = parser.parse_args()
    try:
        print(publish(args.image, args.source, args.provenance, args.store))
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Publication failed: {error}\n")


if __name__ == "__main__":
    main()
