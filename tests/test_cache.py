"""Generation reconciliation, private rank hydration, relocation, and failure gates."""

import fcntl
import io
import json
from pathlib import Path
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from apertus_amd import cache
from apertus_amd.artifacts import sha256


LIMIT = 2**20


def bundle(tmp_path, name, kernels):
    live = tmp_path / name
    for key, data in kernels.items():
        obj = live / "triton" / (key * 32)
        obj.mkdir(parents=True)
        (obj / "kernel.hsaco").write_bytes(data)
        (obj / "kernel.json").write_text('{"name":"kernel"}')
        (obj / "__grp__kernel.json").write_text(json.dumps({"child_paths": {
            filename: str(obj / filename) for filename in ("kernel.hsaco", "kernel.json")}}))
    result = tmp_path / (name + "-bundle")
    cache.capture(live, result, max_bytes=LIMIT)
    return result


def publish(root, bundles):
    return cache.publish(root, {}, bundles, max_bytes=LIMIT)


def test_rank_contributions_relocate_to_private_copies_and_pinned_generation(tmp_path):
    root = tmp_path / "store"
    first = publish(root, [bundle(tmp_path, "rank0", {"A": b"one"}), bundle(tmp_path, "rank1", {"B": b"two"})])
    # Two simulated nodes stage the same explicit selection. A later publisher
    # advances CURRENT between them; no rank is allowed to follow it afresh.
    selection = first["generation"]
    node0, node1 = tmp_path / "node0", tmp_path / "node1"
    cache.stage(root, {}, node0, generation=selection, max_bytes=LIMIT)
    second = publish(root, [bundle(tmp_path, "other-experiment", {"C": b"three"})])
    cache.stage(root, {}, node1, generation=selection, max_bytes=LIMIT)
    assert selection != second["generation"]
    for rank, staged in enumerate((node0, node0, node1, node1)):
        live = tmp_path / f"private-{rank}"
        assert cache.restore(staged, live, max_bytes=LIMIT)["generation"] == selection
        assert {p.name for p in (live / "triton").iterdir()} == {"A" * 32, "B" * 32}
        for group in live.glob("triton/*/__grp__*.json"):
            for name, pointer in json.loads(group.read_text())["child_paths"].items():
                assert Path(pointer) == group.parent / name
                assert Path(pointer).is_file()
        # Re-capture under a different private path must have the same object IDs.
        captured = cache.capture(live, tmp_path / f"recapture-{rank}", max_bytes=LIMIT)
        assert cache.objects(captured["files"]) == cache.objects(first["files"])
        (live / "triton" / ("A" * 32) / "kernel.hsaco").write_bytes(b"private mutation")


def test_existing_objects_win_for_cold_and_stale_producers_with_new_additions(tmp_path):
    root = tmp_path / "store"
    first = publish(root, [bundle(tmp_path, "base", {"A": b"base"})])
    current = publish(root, [bundle(tmp_path, "cold-rebuild", {"A": b"rebuilt", "B": b"new"})])
    assert current["differing_contributions"] == 1
    assert cache.objects(current["files"])["triton/" + "A" * 32] == cache.objects(first["files"])["triton/" + "A" * 32]
    merged = publish(root, [bundle(tmp_path, "stale-reader", {"A": b"base", "C": b"other"})])
    assert len(cache.objects(merged["files"])) == 3
    assert merged["differing_contributions"] == 0
    manifest = root / cache.identity({}) / "generations" / merged["generation"] / "manifest.json"
    before = manifest.read_bytes()
    repeated = publish(root, [bundle(tmp_path, "divergent", {"A": b"another rebuild"})])
    assert repeated["generation"] == merged["generation"]
    assert repeated["differing_contributions"] == 1
    assert manifest.read_bytes() == before
    assert "differing_contributions" not in json.loads(before)


def test_concurrent_writers_keep_one_overlapping_object_and_all_disjoint_additions(tmp_path):
    root = tmp_path / "store"
    inputs = [bundle(tmp_path, f"rank-{key}", {"Z": key.encode(), key: key.encode()}) for key in "ABCD"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda item: publish(root, [item]), inputs))
    selected = cache.stage(root, {}, tmp_path / "staged", max_bytes=LIMIT)
    assert len(cache.objects(selected["files"])) == 5
    assert sum(r["differing_contributions"] for r in records) == 3
    key = "triton/" + "Z" * 32
    assert all(cache.objects(r["files"])[key] == cache.objects(selected["files"])[key] for r in records)


def test_publication_interruption_retains_previous_generation(tmp_path, monkeypatch):
    root = tmp_path / "store"
    first = publish(root, [bundle(tmp_path, "first", {"A": b"one"})])
    incoming = bundle(tmp_path, "second", {"B": b"two"})
    original = Path.replace
    def interrupted(path, destination):
        if Path(destination).name == "CURRENT":
            raise OSError("simulated interruption before CURRENT")
        return original(path, destination)
    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="interruption"):
        publish(root, [incoming])
    assert cache.stage(root, {}, tmp_path / "staged", max_bytes=LIMIT)["generation"] == first["generation"]


def test_lock_wait_is_bounded(tmp_path):
    namespace = tmp_path / cache.identity({})
    namespace.mkdir()
    with (namespace / "publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError):
            cache.publish(tmp_path, {}, [], max_bytes=LIMIT, timeout=0)


def test_incomplete_groups_and_mutable_state_are_not_valid_objects(tmp_path):
    source = bundle(tmp_path, "fixture", {"A": b"binary"})
    live = tmp_path / "fixture"
    (live / "triton" / ("A" * 32) / "autotune.json").write_text("{}")
    captured = cache.capture(live, tmp_path / "filtered", max_bytes=LIMIT)
    assert any("autotune" in name for name in captured["skipped"])
    assert cache.objects(captured["files"]) == cache.objects(json.loads((source / "manifest.json").read_text())["files"])
    (live / "triton" / ("A" * 32) / "kernel.hsaco").unlink()
    with pytest.raises(ValueError, match="absent artifact"):
        cache.capture(live, tmp_path / "incomplete", max_bytes=LIMIT)
    assert not (tmp_path / "incomplete").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "triton/" + "A" * 32 + "/kernel.hsaco"])
def test_archive_validation_precedes_extraction(tmp_path, name):
    staged = tmp_path / "staged"
    staged.mkdir()
    archive = staged / "payload.tar"
    with tarfile.open(archive, "w") as output:
        entry = tarfile.TarInfo(name)
        if name.startswith("triton"):
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/escape"
        else:
            entry.size = 1
        output.addfile(entry, io.BytesIO(b"x"))
    (staged / "manifest.json").write_text(json.dumps({"schema": cache.SCHEMA, "files": {}, "sha256": sha256(archive)}))
    with pytest.raises(ValueError):
        cache.restore(staged, tmp_path / "private", max_bytes=LIMIT)
    assert not (tmp_path / "private").exists()


def test_failed_capture_and_oversized_merge_never_advance_current(tmp_path):
    root = tmp_path / "store"
    first = publish(root, [bundle(tmp_path, "one", {"A": b"one"})])
    incoming = bundle(tmp_path, "two", {"B": b"two"})
    with pytest.raises(ValueError, match="budget"):
        cache.publish(root, {}, [incoming], max_bytes=1)
    assert cache.stage(root, {}, tmp_path / "staged", max_bytes=LIMIT)["generation"] == first["generation"]


def test_corrupt_snapshot_is_rejected_before_private_hydration(tmp_path):
    root = tmp_path / "store"
    publish(root, [bundle(tmp_path, "one", {"A": b"binary"})])
    staged = tmp_path / "staged"
    cache.stage(root, {}, staged, max_bytes=LIMIT)
    (staged / "payload.tar").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cache.restore(staged, tmp_path / "private", max_bytes=LIMIT)
    assert not (tmp_path / "private").exists()


def test_capture_rejects_links_without_publishing_partial_bundle(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    (live / "link").symlink_to("/previous/run")
    with pytest.raises(ValueError, match="nonportable"):
        cache.capture(live, tmp_path / "bundle", max_bytes=LIMIT)
    assert not (tmp_path / "bundle").exists()


def test_first_contributor_wins_as_a_complete_group_without_later_file_splicing(tmp_path):
    first = bundle(tmp_path, "first", {"A": b"first binary"})
    bundle(tmp_path, "later", {"A": b"different binary"})
    live = tmp_path / "later"
    obj = live / "triton" / ("A" * 32)
    # A later version has extra diagnostics; they must not leak into the first group.
    (obj / "kernel.ttir").write_text(f'#loc = loc("{live}/inductor.py":1:1)\n')
    later = tmp_path / "later-complete"
    cache.capture(live, later, max_bytes=LIMIT)
    result = publish(tmp_path / "store", [first, later])
    assert result["differing_contributions"] == 1
    assert result["files"] == json.loads((first / "manifest.json").read_text())["files"]
    with tarfile.open(later / "payload.tar") as archive:
        assert archive.extractfile("triton/" + "A" * 32 + "/kernel.ttir").read() == (obj / "kernel.ttir").read_bytes()


def test_corrupt_differing_contribution_is_rejected_even_when_key_already_exists(tmp_path):
    root = tmp_path / "store"
    first = publish(root, [bundle(tmp_path, "first", {"A": b"good"})])
    incoming = bundle(tmp_path, "incoming", {"A": b"different"})
    (incoming / "payload.tar").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        publish(root, [incoming])
    assert cache.stage(root, {}, tmp_path / "staged", max_bytes=LIMIT)["generation"] == first["generation"]
