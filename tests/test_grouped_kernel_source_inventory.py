"""Owned temporary TE trees only; no installed vendor packages or scheduler calls."""

import base64
import builtins
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

PROBE = Path(__file__).resolve().parents[1] / "experiments/validation/grouped_kernel_source_inventory.py"


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "triton", "transformer_engine", "aiter"}:
            pytest.fail("vendor import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("grouped_source_inventory", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(probe, tmp_path, monkeypatch):
    texts = {k: "# pinned caller\n" for k in probe.PINS}
    texts[probe.TARGET] = "from .helpers import kernel\nimport math\n"
    texts[probe.KERNELS + "/helpers.py"] = "# helper\n"
    texts[probe.WRAPPER] = "from .kernels.core import call\n"
    texts[probe.KERNELS + "/gmm/kernels/core.py"] = "# nested kernel\n"
    rows = []
    for name, text in texts.items():
        data = text.encode(); path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        digest = hashlib.sha256(data).digest()
        rows.append([name, "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("="), str(len(data))])
    monkeypatch.setattr(probe, "PINS", {k: (len(texts[k].encode()), hashlib.sha256(texts[k].encode()).hexdigest()) for k in probe.PINS})
    dist = tmp_path / probe.DIST; dist.mkdir()
    (dist / "METADATA").write_text("Name: transformer_engine\nVersion: 2.17.1+130099ce4\n")
    (tmp_path / "transformer_engine/build_info.txt").write_text("fixture build\n")
    with (dist / "RECORD").open("w") as stream:
        csv.writer(stream).writerows(rows)
    return tmp_path


def test_export_exact_deterministic_and_imports(probe, tree):
    a, b = {}, {}
    probe.inventory(tree, a); probe.inventory(tree, b)
    assert a == b and json.loads(json.dumps(a, sort_keys=True)) == a
    target = next(r for r in a["files"] if r["name"] == probe.TARGET)
    assert target["imports"] == [".helpers:kernel", "math"] and target["RECORD"]["matches"]
    assert target["text"].encode() == (tree / probe.TARGET).read_bytes()


@pytest.mark.parametrize("damage", ["missing", "symlink", "parent_symlink", "oversize", "total", "files", "entries", "pin", "metadata", "record", "binary", "syntax"])
def test_fail_closed(probe, tree, monkeypatch, damage):
    target = tree / probe.TARGET
    if damage == "missing": target.unlink()
    elif damage == "symlink":
        target.unlink(); target.symlink_to("helpers.py")
    elif damage == "parent_symlink":
        directory = tree / probe.KERNELS; directory.rename(directory.with_name("elsewhere")); directory.symlink_to("elsewhere")
    elif damage == "oversize": monkeypatch.setattr(probe, "FILE_BYTES", 1)
    elif damage == "total": monkeypatch.setattr(probe, "TOTAL_BYTES", 1)
    elif damage == "files": monkeypatch.setattr(probe, "MAX_FILES", 1)
    elif damage == "entries": monkeypatch.setattr(probe, "ENTRIES", 1)
    elif damage == "pin": (tree / next(iter(probe.PINS))).write_text("# drift\n")
    elif damage == "metadata": (tree / probe.DIST / "METADATA").write_text("Name: wrong\nVersion: 0\n")
    elif damage == "record": (tree / probe.DIST / "RECORD").write_text("")
    elif damage == "binary": target.write_bytes(b"\x7fELF")
    else: target.write_text("def ?")
    with pytest.raises((ValueError, OSError, SyntaxError)):
        probe.inventory(tree, {})


def settings(probe):
    return dict(image=dict(sha256=probe.IMAGE), resources=dict(nodes=1, gpus_per_node=4,
        cpus_per_node=192, time="00:05:00", reservation="cscs", mpi="pmix",
        cpu_bind="map_ldom:0,1,2,3", distribution="block:block"),
        procedure=dict(kind="commands", timeout=60, commands=[probe.COMMAND]))


@pytest.mark.parametrize("damage", [None, "image", "cpu_override", "affinity", "mpi", "rank"])
def test_launch_contract(probe, monkeypatch, damage):
    value = settings(probe)
    env = dict(SLURM_NTASKS="4", SLURM_NODEID="0", SLURM_PROCID="2", SLURM_LOCALID="2")
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda _: set(range(48 if damage != "affinity" else 4)))
    if damage == "image": value["image"]["sha256"] = "0" * 64
    elif damage == "cpu_override": value["resources"]["cpus_per_task"] = 48
    elif damage == "mpi": value["resources"]["mpi"] = "none"
    elif damage == "rank": env["SLURM_LOCALID"] = "1"
    if damage is None: probe.validate_launch(value, env)
    else:
        with pytest.raises(ValueError): probe.validate_launch(value, env)


@pytest.mark.parametrize("failure", [False, "missing", "record", "nested"])
def test_main_exclusive_closed_failure_evidence(probe, tree, monkeypatch, tmp_path, failure):
    value = settings(probe); value["validation"] = dict(sha256=hashlib.sha256(PROBE.read_bytes()).hexdigest())
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k: json.dumps(value) if str(p) == "/apertus/resolved.json" else original(p, *a, **k))
    monkeypatch.setattr(probe, "PACKAGE", tree)
    monkeypatch.setattr(probe, "validate_launch", lambda *a: None)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setattr(sys, "argv", [str(PROBE), str(tmp_path)])
    if failure == "missing": (tree / probe.TARGET).unlink()
    elif failure == "nested": (tree / probe.WRAPPER).unlink()
    elif failure == "record": (tree / probe.DIST / "RECORD").write_text('"unterminated,,\n')
    if failure:
        with pytest.raises((ValueError, csv.Error)): probe.main()
    else: probe.main()
    path = tmp_path / "grouped-kernel-source-0.json"
    row = json.loads(original(path))
    assert row["status"] == ("failed" if failure else "observed") and row["closed_at"]
    before = path.read_bytes()
    with pytest.raises(FileExistsError): probe.main()
    assert path.read_bytes() == before


def test_deadline(probe, tree):
    with pytest.raises(TimeoutError): probe.inventory(tree, {}, seconds=-1)


def test_large_record_stream_selected_only(probe, tree):
    path = tree / probe.DIST / "RECORD"
    selected = path.read_bytes()
    unrelated = ("unused/" + "a" * 300 + ",,\n").encode() * 11000
    path.write_bytes(unrelated + selected)
    result = {}; probe.inventory(tree, result)
    assert 3_000_000 < result["record"]["bytes_read"] < 4_000_000
    assert result["record"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["record"]["rows"] == 11000 + len(result["files"])
    assert result["record"]["complete"] and "text" not in result["record"]
    assert "unused/" not in json.dumps(result)
    assert any(r["name"].endswith("gmm/kernels/core.py") for r in result["files"])


@pytest.mark.parametrize("damage", ["oversize", "rows", "line", "malformed", "duplicate", "missing", "symlink", "fifo", "changed", "deadline", "nul"])
def test_stream_record_failures(probe, tree, monkeypatch, damage):
    path = tree / probe.DIST / "RECORD"
    if damage == "oversize":
        with path.open("ab") as stream: stream.truncate(probe.RECORD_BYTES + 1)
    elif damage == "rows": path.write_bytes(b"unrelated,,\n" * 50001)
    elif damage == "line": path.write_bytes(b"a" * (probe.LINE_BYTES + 1) + b",,\n")
    elif damage == "malformed": path.write_bytes(b'"unterminated,,\n')
    elif damage == "duplicate": path.write_bytes(path.read_bytes() * 2)
    elif damage == "missing": path.write_bytes(b"")
    elif damage == "symlink":
        path.unlink(); path.symlink_to("METADATA")
    elif damage == "fifo":
        path.unlink(); probe.os.mkfifo(path)
    elif damage == "changed":
        original = probe.os.stat
        def changed(name, *args, **kwargs):
            if name == "RECORD":
                with path.open("ab") as stream: stream.write(b"drift,,\n")
            return original(name, *args, **kwargs)
        monkeypatch.setattr(probe.os, "stat", changed)
    elif damage == "deadline":
        original = probe.record_rows
        monkeypatch.setattr(probe, "record_rows", lambda root, wanted, result, deadline: original(root, wanted, result, -1))
    else: path.write_bytes(b"nul\0,,\n")
    result = {}
    with pytest.raises((ValueError, OSError, csv.Error, TimeoutError)):
        probe.inventory(tree, result)
    assert "source_identity" not in result
    assert not result.get("record", {}).get("complete")
    if damage == "oversize": assert result["record"]["bytes_read"] == 0


@pytest.mark.parametrize("damage", ["wrapper", "nested_symlink", "depth", "entries", "bytes", "skip_cache", "outside_scope"])
def test_nested_scope_bounds(probe, tree, monkeypatch, damage):
    base = tree / probe.KERNELS
    if damage == "wrapper": (tree / probe.WRAPPER).unlink()
    elif damage == "nested_symlink": (base / "gmm/alias").symlink_to("kernels", target_is_directory=True)
    elif damage == "depth": (base / "gmm/a/b/c/d").mkdir(parents=True)
    elif damage == "entries":
        for i in range(128): (base / f"gmm/entry{i}").touch()
    elif damage == "bytes":
        with (base / "gmm/kernels/core.py").open("wb") as stream: stream.truncate(probe.FILE_BYTES + 1)
    elif damage == "skip_cache":
        cache = base / "gmm/__pycache__"; cache.mkdir()
        for i in range(140): (cache / f"ignore{i}.py").touch()
    else:
        unrelated = base / "unrelated"; unrelated.mkdir(); (unrelated / "no.py").write_text("invalid syntax ???")
    result = {}
    if damage in ("skip_cache", "outside_scope"):
        probe.inventory(tree, result)
        assert len(result["files"]) == 6
    else:
        with pytest.raises((ValueError, OSError)): probe.inventory(tree, result)


def test_record_multiline_csv_hashes_raw_bytes(probe, tree):
    path = tree / probe.DIST / "RECORD"
    path.write_bytes(b'"unrelated\nname",,\r\n' + path.read_bytes())
    result = {}; probe.inventory(tree, result)
    assert result["record"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["record"]["rows"] == len(result["files"]) + 1
