"""Exercise standalone publication, idempotence, corruption, and ownership failures."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/publish.py"


@pytest.fixture
def inputs(tmp_path):
    image = tmp_path / "image.sqsh"
    # A signature fixture tests publication mechanics, not SquashFS execution.
    image.write_bytes(b"hsqs" + b"publication fixture" * 10)
    source = tmp_path / "source.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        entry = tarfile.TarInfo("Containerfile")
        data = b"FROM scratch\n"
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"source": {"repository": "https://example.invalid/images", "head": "a" * 40},
                                      "recipe": "Containerfile", "build_args": {}}))
    return image, source, provenance, tmp_path / "store"


def invoke(script, inputs, cwd, *, umask=-1):
    image, source, provenance, store = inputs
    return subprocess.run([sys.executable, str(script), str(image), str(source), str(provenance), "--store", str(store)],
                          cwd=cwd, capture_output=True, text=True, umask=umask)


@pytest.mark.parametrize("existing", [False, True])
def test_team_publication_under_private_umask(inputs, tmp_path, existing):
    store = inputs[3]
    public = [store, store / "objects", store / "objects/sha256"]
    if existing:
        for directory in public:
            directory.mkdir()
            directory.chmod(0o2770)
    result = invoke(SCRIPT, inputs, tmp_path, umask=0o077)
    assert result.returncode == 0, result.stderr
    image = Path(result.stdout.strip())
    for directory in [*public, image.parent]:
        assert directory.stat().st_mode & 0o055 == 0o055
    for directory in public:
        assert directory.stat().st_mode & 0o3775 == 0o3775
    for artifact in image.parent.iterdir():
        assert artifact.stat().st_mode & 0o044 == 0o044
        assert artifact.stat().st_mode & 0o022 == 0
        assert artifact.stat().st_gid == store.stat().st_gid
    for shared in (store / ".staging", store / ".locks"):
        assert shared.stat().st_mode & 0o7777 == 0o3770


def test_copied_script_is_self_contained_and_idempotent(inputs, tmp_path):
    script = tmp_path / "standalone.py"
    shutil.copyfile(SCRIPT, script)
    first = invoke(script, inputs, tmp_path)
    assert first.returncode == 0, first.stderr
    second = invoke(script, inputs, tmp_path)
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    checksum = hashlib.sha256(inputs[0].read_bytes()).hexdigest()
    image = inputs[3].resolve() / "objects/sha256" / checksum / "image.sqsh"
    assert first.stdout == f"{image}\n"
    obj = image.parent
    assert {p.name for p in obj.iterdir()} == {"manifest.json", "image.sqsh", "source.tar.gz"}
    assert not list((inputs[3] / ".staging").iterdir())
    assert not list((inputs[3] / ".locks").iterdir())
    before = (obj / "manifest.json").read_bytes()
    inputs[2].write_text(inputs[2].read_text().replace("Containerfile", "Differentfile"))
    conflict = invoke(script, inputs, tmp_path)
    assert conflict.returncode != 0 and not conflict.stdout
    assert (obj / "manifest.json").read_bytes() == before


def test_corrupt_existing_image_is_not_overwritten(inputs, tmp_path):
    result = invoke(SCRIPT, inputs, tmp_path)
    assert result.returncode == 0, result.stderr
    image = Path(result.stdout.strip())
    image.write_bytes(b"corrupt")
    assert invoke(SCRIPT, inputs, tmp_path).returncode != 0
    assert image.read_bytes() == b"corrupt"


def test_foreign_lock_is_preserved(inputs, tmp_path):
    checksum = hashlib.sha256(inputs[0].read_bytes()).hexdigest()
    lock = inputs[3] / ".locks" / checksum
    lock.mkdir(parents=True)
    (lock / "owner.json").write_text("foreign publisher")
    assert invoke(SCRIPT, inputs, tmp_path).returncode != 0
    assert (lock / "owner.json").read_text() == "foreign publisher"
    assert not list((inputs[3] / "objects/sha256").iterdir())


def test_failed_copy_never_publishes_partial_object(inputs, monkeypatch):
    spec = importlib.util.spec_from_file_location("standalone_publisher", SCRIPT)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    def fail(src, dst, size):
        dst.write(b"partial")
        raise OSError("injected copy failure")
    monkeypatch.setattr(publisher.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="injected"):
        publisher.publish(*inputs)
    assert not list((inputs[3] / ".staging").iterdir())
    assert not list((inputs[3] / "objects/sha256").iterdir())


@pytest.mark.parametrize("filesystem,status", [("lustre", 0), ("lustre", 23), ("tmpfs", 23)])
def test_striping_precedes_copy_and_failure_never_publishes(inputs, tmp_path, monkeypatch,
                                                          filesystem, status):
    tools = tmp_path / "tools"
    tools.mkdir()
    stat = tools / "stat"
    stat.write_text(f"#!{sys.executable}\nprint({filesystem!r})\n")
    stat.chmod(0o755)
    log = tmp_path / "stripe-call.json"
    lfs = tools / "lfs"
    lfs.write_text(f'''#!{sys.executable}
import json, pathlib, sys
directory = pathlib.Path(sys.argv[-1])
assert directory.parent.name == '.staging'
assert directory.is_dir() and not list(directory.iterdir())
pathlib.Path({str(log)!r}).write_text(json.dumps(sys.argv[1:]))
print('layout diagnostic')
sys.exit({status})
''')
    lfs.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    result = invoke(SCRIPT, inputs, tmp_path)
    if filesystem == "lustre":
        arguments = json.loads(log.read_text())
        assert arguments[:-1] == [
            "setstripe", "--component-end", "4M", "--stripe-count", "1", "--stripe-size", "4M",
            "--component-end", "64M", "--stripe-count", "4", "--stripe-size", "4M",
            "--component-end", "-1", "--stripe-count", "32", "--stripe-size", "4M",
        ]
        assert "layout diagnostic" in result.stderr
    else:
        assert not log.exists()
    if filesystem == "lustre" and status:
        assert result.returncode != 0 and not result.stdout
        assert not list((inputs[3] / "objects/sha256").iterdir())
    else:
        assert result.returncode == 0, result.stderr
        image = Path(result.stdout.strip())
        assert result.stdout == f"{image}\n" and image.read_bytes() == inputs[0].read_bytes()
    assert not list((inputs[3] / ".staging").iterdir())
    assert not list((inputs[3] / ".locks").iterdir())
