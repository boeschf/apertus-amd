"""Image cache reuse and no-overwrite contracts without a registry or Slurm."""

import hashlib

import pytest

from apertus_amd import artifacts


@pytest.fixture
def image_paths(tmp_path):
    payload = b"fixture"
    digest = hashlib.sha256(payload).hexdigest()
    registry = tmp_path / "offline-registry" / "image.sqsh"
    target = tmp_path / "cache/images/sha256" / digest / "image.sqsh"
    return registry, target, payload, digest


@pytest.mark.parametrize("valid", [True, False])
def test_cache_hit_checks_digest_without_accessing_registry(image_paths, monkeypatch, valid):
    registry, target, payload, digest = image_paths
    target.parent.mkdir(parents=True)
    contents = payload if valid else b"corrupt"
    target.write_bytes(contents)
    before = target.stat()
    monkeypatch.setattr(artifacts, "image_manifest", lambda *args: pytest.fail("touched registry on cache hit"))
    if valid:
        artifacts.cache_image(registry, target, digest)
    else:
        with pytest.raises(ValueError, match="refusing to replace"):
            artifacts.cache_image(registry, target, digest)
    assert target.stat().st_ino == before.st_ino and target.read_bytes() == contents


def test_cache_refuses_symlinked_directory(image_paths, tmp_path):
    registry, target, _, digest = image_paths
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (tmp_path / "cache").symlink_to(unrelated, target_is_directory=True)
    with pytest.raises(ValueError, match="invalid image cache directory"):
        artifacts.cache_image(registry, target, digest)
    assert not list(unrelated.iterdir())


def test_cache_refuses_symlinked_image(image_paths, tmp_path):
    registry, target, payload, digest = image_paths
    unrelated = tmp_path / "unrelated.sqsh"
    unrelated.write_bytes(payload)
    target.parent.mkdir(parents=True)
    target.symlink_to(unrelated)
    with pytest.raises(ValueError, match="invalid cached image"):
        artifacts.cache_image(registry, target, digest)
    assert target.is_symlink()


def test_simultaneous_publication_keeps_verified_winner(image_paths, monkeypatch):
    registry, target, payload, digest = image_paths
    registry.parent.mkdir()
    registry.write_bytes(payload)
    monkeypatch.setattr(artifacts, "image_manifest", lambda *a: {"artifacts": {"image.sqsh": {"bytes": len(payload)}}})

    def other_writer(temporary, destination):
        assert temporary.read_bytes() == payload
        assert temporary.stat().st_mode & 0o777 == 0o444
        assert destination == target and not destination.exists()
        destination.write_bytes(payload)
        raise FileExistsError

    monkeypatch.setattr(artifacts.os, "link", other_writer)
    artifacts.cache_image(registry, target, digest)
    assert target.read_bytes() == payload
    assert not list(target.parent.glob(".image.sqsh-*"))
