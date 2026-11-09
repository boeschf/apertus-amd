"""Prepare retained archives from exact Git commits, never live working files."""

from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

from .artifacts import sha256, write_json


@contextmanager
def exclusive(directory: Path):
    """Fail on an existing lock and remove only a lock acquired by this caller."""
    directory.mkdir()
    try:
        yield
    finally:
        directory.rmdir()


def extract(archive: Path, destination: Path) -> None:
    """Extract a source archive with Python's containment-aware data filter."""
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter="data")


def prepare(selection: dict, root: Path, *, local: Path) -> dict:
    """Retain committed source once; no remote access, mirror, or history bundle."""
    commit = selection["commit"]
    destination = root / commit
    root.mkdir(parents=True, exist_ok=True)

    def git(*args, **kwargs):
        return subprocess.run(["git", "-C", str(local), *args], check=True, capture_output=True, **kwargs)

    if git("rev-parse", f"{commit}^{{commit}}", text=True).stdout.strip() != commit:
        raise ValueError("resolved Git commit differs from requested identity")
    with exclusive(root / f".{commit}.lock"):
        if destination.exists():
            record = json.loads((destination / "manifest.json").read_text())
            if record["selection"]["excluded_submodules"] != selection.get("excluded_submodules", []):
                raise ValueError("retained source selection mismatch")
            if sha256(destination / "source.tar") != record["sha256"]:
                raise ValueError("retained source checksum mismatch")
            return {**record, "directory": str(destination)}
        tree = git("rev-parse", f"{commit}^{{tree}}", text=True).stdout.strip()
        excluded = {}
        for entry in git("ls-tree", "-rz", commit).stdout.split(b"\0"):
            if not entry:
                continue
            metadata, raw_name = entry.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            name = raw_name.decode()
            if kind == "commit":
                if name not in selection.get("excluded_submodules", []):
                    raise ValueError(f"required submodule is not materialized: {name}")
                excluded[name] = oid
            elif mode == "120000":
                link = git("cat-file", "blob", oid, text=True).stdout
                if PurePosixPath(link).is_absolute():
                    raise ValueError(f"absolute source symlink: {name}")
        temporary = Path(tempfile.mkdtemp(prefix=".source-", dir=destination.parent))
        try:
            archive = temporary / "source.tar"
            git("archive", "--format=tar", f"--output={archive}", commit)
            # Validate extraction now, not for the first time inside an allocation.
            check = temporary / "check"
            check.mkdir()
            extract(archive, check)
            for path in check.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    with path.open("rb") as stream:
                        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                            raise ValueError(f"unmaterialized Git LFS input: {path.relative_to(check)}")
            files = {str(path.relative_to(check)): sha256(path) for path in check.rglob("*")
                     if path.is_file() and not path.is_symlink()}
            shutil.rmtree(check)
            record = {"schema": 2, "selection": {**selection, "excluded_submodules": selection.get("excluded_submodules", [])}, "tree": tree,
                        "excluded_submodules": excluded, "files": files,
                       "sha256": sha256(archive)}
            write_json(temporary / "manifest.json", record)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {**record, "directory": str(destination)}
