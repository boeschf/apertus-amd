"""Committed-source preparation, integrity checking, and safe extraction."""

import io
from pathlib import Path
import subprocess
import tarfile

import pytest

from apertus_amd import sources


def test_source_uses_committed_bytes_with_no_origin_or_history_copy(tmp_path, monkeypatch):
    repository = tmp_path / "checkout"
    repository.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True, text=True).stdout.strip()
    git("init")
    (repository / "model.py").write_text("committed\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    commit = git("rev-parse", "HEAD")
    (repository / "model.py").write_text("dirty\n")
    original = subprocess.run
    def checked(command, **kwargs):
        assert not {"fetch", "bundle", "remote", "clone"}.intersection(command)
        return original(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", checked)
    selection = {"repository": "irrelevant-provenance-url", "commit": commit, "excluded_submodules": []}
    record = sources.prepare(selection, tmp_path / "sources", local=repository)
    directory = Path(record["directory"])
    assert directory.name == commit
    assert {p.name for p in directory.iterdir()} == {"source.tar", "manifest.json"}
    with tarfile.open(directory / "source.tar") as archive:
        assert archive.extractfile("model.py").read() == b"committed\n"
    assert sources.prepare(selection, tmp_path / "sources", local=repository) == record
    (directory / "source.tar").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        sources.prepare(selection, tmp_path / "sources", local=repository)


def test_source_extraction_rejects_escaping_links(tmp_path):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as stream:
        entry = tarfile.TarInfo("../../escape")
        entry.size = 1
        stream.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        sources.extract(archive, tmp_path / "output")
