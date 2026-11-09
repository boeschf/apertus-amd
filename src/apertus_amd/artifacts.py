"""Content verification and immutable-file helpers used during run preparation."""

import hashlib
import json
import os
from pathlib import Path
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
