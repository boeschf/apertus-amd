"""CPU checks for the guarded signed schedule probe and its evidence gate."""

import json
from pathlib import Path
import runpy

import pytest

from apertus_amd.artifacts import sha256
from apertus_amd.cli import plan
from apertus_amd.runs import run


ROOT = Path(__file__).resolve().parents[1]
V = runpy.run_path(ROOT / "experiments/validation/post_combine_schedule.py")


def planned():
    return plan(ROOT / "experiments/chonk-post-combine-schedule-i910bfcf936d3",
                site=ROOT / "config.toml", submit=True, gpus_per_node=1, cpus_per_task=48,
                cpu_bind="cores", time="00:35:00", cache_mode="cold", exclude="nid002450",
                cache_image=True, reservation="cscs")


def test_schedule_recipe_is_guarded():
    p = planned()
    assert p["source"]["commit"] == V["COMMIT"] and p["image"]["sha256"] == V["IMAGE"]
    assert p["args"] == [] and p["entrypoint"] == V["ENTRYPOINT"]
    assert p["resources"]["nodes"] == p["resources"]["gpus_per_node"] == 1
    assert p["procedure"]["kind"] == "probe" and "auxiliary_source" not in p["procedure"]
    with pytest.raises(ValueError, match="launch blocked"):
        run(p)


def test_schedule_rejects_other_rank_or_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(tree=V["TREE"], sha256=V["ARCHIVE"],
                                        selection=dict(commit=V["COMMIT"]))))
    verify = V["verify_inputs"]
    monkeypatch.setitem(verify.__globals__, "MANIFEST", sha256(manifest))
    p = planned()
    verify(p, manifest, V["ENTRYPOINT"], [], dict(WORLD_SIZE="1", RANK="0"))
    with pytest.raises(ValueError, match="inputs"):
        verify(p, manifest, V["ENTRYPOINT"], [], dict(WORLD_SIZE="1", RANK="1"))
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        verify(p, manifest, V["ENTRYPOINT"], [], dict(WORLD_SIZE="1", RANK="0"))


@pytest.mark.parametrize("marks,verified,status", [(1, 100, "passed"), (0, 100, "failed"),
                                                  (2, 100, "failed"), (1, 0, "failed")])
def test_schedule_check_needs_verified_imports_and_unique_pass(tmp_path, marks, verified, status):
    output = tmp_path / "output"
    output.mkdir()
    (output / "imports-0.json").write_text(json.dumps(dict(verified_modules=verified)))
    (output / "rank-0.log").write_text((V["PASS_LINE"] + "\n") * marks)
    if status == "passed":
        V["check"](tmp_path)
    else:
        with pytest.raises(ValueError, match="missing"):
            V["check"](tmp_path)
    assert json.loads((output / "checks.json").read_text())["status"] == status
