"""Committed-source preparation, integrity checking, and safe extraction."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from apertus_amd import sources
from apertus_amd.artifacts import sha256, write_json


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
    manifest = directory / "manifest.json"
    altered = json.loads(manifest.read_text())
    altered["tree"] = "changed-tree"
    manifest.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="commit/tree mismatch"):
        sources.prepare(selection, tmp_path / "sources", local=repository)


def test_source_extraction_rejects_escaping_links(tmp_path):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as stream:
        entry = tarfile.TarInfo("../../escape")
        entry.size = 1
        stream.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        sources.extract(archive, tmp_path / "output")


def test_auxiliary_archive_freezes_only_the_verified_member(tmp_path):
    commit = "1" * 40
    parent = tmp_path / commit
    parent.mkdir()
    name = "fla/ops/kda/chunk_bwd.py"
    data = b"def backward():\n    pass\n"
    path = parent / "source.tar"
    manifest = parent / "manifest.json"
    dest = tmp_path / "auxiliary.py"
    digest = hashlib.sha256(data).hexdigest()

    def prepare(*, link=False, duplicate=False, wrong=False):
        with tarfile.open(path, "w") as archive:
            for _ in range(2 if duplicate else 1):
                m = tarfile.TarInfo(name)
                if link:
                    m.type, m.linkname = tarfile.SYMTYPE, "other.py"
                    archive.addfile(m)
                else:
                    m.size = len(data)
                    archive.addfile(m, io.BytesIO(data))
        write_json(manifest, {"selection": {"commit": commit}, "files": {name: "0" * 64 if wrong else digest},
                              "sha256": sha256(path)})
        return dict(commit=commit, path=name, sha256=digest, manifest_sha256=sha256(manifest))

    for options in ({"link": True}, {"duplicate": True}, {"wrong": True}):
        with pytest.raises(ValueError, match="auxiliary source"):
            sources.freeze_auxiliary_file(prepare(**options), tmp_path, dest)
        assert not dest.exists()
    selected = prepare()
    sources.freeze_auxiliary_file(selected, tmp_path, dest)
    assert dest.read_bytes() == data
    with pytest.raises(FileExistsError):
        sources.freeze_auxiliary_file(selected, tmp_path, dest)
