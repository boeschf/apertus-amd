"""Content verification and immutable-file helpers used during run preparation."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile


def sha256(path: Path) -> str:
    """Hash complete file bytes with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically replace a controller-owned record on its own filesystem."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def image_manifest(image: Path, checksum: str) -> dict:
    """Check a published image reference, trusting the immutable publication store."""
    directory = image.parent
    if directory.is_symlink():
        raise ValueError("image object must not be a symlink")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != 1:
        raise ValueError("unsupported image manifest schema")
    artifacts = manifest["artifacts"]
    if artifacts["image.sqsh"]["sha256"] != checksum:
        raise ValueError("image selection disagrees with manifest")
    if image.is_symlink() or not image.is_file() or image.stat().st_size != artifacts["image.sqsh"]["bytes"]:
        raise ValueError(f"missing or mismatched image: {image}")
    with image.open("rb"):
        pass
    return manifest


def cache_image(registry: Path, target: Path, checksum: str) -> None:
    """Verify or stage one image without replacing an existing cache object."""
    def verify(path: Path) -> None:
        if not stat.S_ISREG(path.lstat().st_mode) or sha256(path) != checksum:
            raise ValueError(f"invalid cached image: {path}; refusing to replace it")

    # The managed cache path has four directory components under the checkout.
    # Do not follow redirections back to the registry or to another cache.
    for directory in reversed(target.parents[:4]):
        directory.mkdir(exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"invalid image cache directory: {directory}")
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    else:
        verify(target)
        return

    manifest = image_manifest(registry, checksum)
    fd, name = tempfile.mkstemp(prefix=".image.sqsh-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as destination, registry.open("rb") as source:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            destination.flush()
            if os.fstat(destination.fileno()).st_size != manifest["artifacts"]["image.sqsh"]["bytes"]:
                raise ValueError(f"cached image size differs from registry manifest: {registry}")
            os.fchmod(destination.fileno(), 0o444)
            os.fsync(destination.fileno())
        verify(temporary)
        # Publish atomically on the target filesystem, with no overwrite even if
        # another launch won the race. The link is only made after verification.
        try:
            os.link(temporary, target)
        except FileExistsError:
            verify(target)
    finally:
        temporary.unlink(missing_ok=True)
