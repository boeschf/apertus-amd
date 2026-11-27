"""CPU checks for the signed-module admission and isolated KDA A/B gate."""

from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from apertus_amd.cli import plan
from apertus_amd.runs import run


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = runpy.run_path(ROOT / "experiments/validation/kda_backward_release.py")


def reports(drop=64 << 20):
    entry = 512 << 20
    fingerprints = dict.fromkeys(VALIDATOR["OUTPUTS"], {"sha256": "1" * 64})
    return [
        dict(arm=arm, module_sha256=VALIDATOR["DIGEST"][arm], target="gfx942",
             shape=list(VALIDATOR["SHAPE"]), backward_calls_verified=1,
             trials=[dict(seconds=0.1, allocated=entry, reserved=entry, peak_reserved=2 * entry,
                          peak_allocated=2 * entry - (drop if arm == "release" else 0))
                     for _ in range(VALIDATOR["SAMPLES"])],
             outputs=[deepcopy(fingerprints) for _ in range(VALIDATOR["SAMPLES"])])
        for arm in VALIDATOR["ARMS"]
    ]


def test_parity_is_separate_from_repeatable_memory_improvement():
    verdict = VALIDATOR["verdict"]
    assert verdict(reports())["repeatable_saving"]
    no_gain = verdict(reports(0))
    assert no_gain["status"] == "passed" and not no_gain["repeatable_saving"]
    single = reports()
    single[2]["trials"] = single[3]["trials"]
    assert not verdict(single)["repeatable_saving"]
    numerics = reports()
    numerics[2]["outputs"][1]["q"] = {"sha256": "2" * 64}
    assert verdict(numerics)["status"] == "failed"
    unmatched = reports()
    unmatched[1]["trials"][0]["allocated"] -= 1
    assert not verdict(unmatched)["matched_backward_entry"]


def test_incomplete_or_wrong_module_evidence_rejected():
    data = reports()
    data[0]["trials"].pop()
    with pytest.raises(ValueError, match="incomplete arm"):
        VALIDATOR["verdict"](data)
    data = reports()
    data[1]["module_sha256"] = data[0]["module_sha256"]
    with pytest.raises(ValueError, match="incomplete arm"):
        VALIDATOR["verdict"](data)


def test_finder_only_replaces_the_exact_kda_backward_module(tmp_path):
    path = tmp_path / "chunk_bwd.py"
    finder = VALIDATOR["CandidateFinder"](path)
    target = finder.find_spec("fla.ops.kda.chunk_bwd")
    assert Path(target.origin) == path
    for module in ("fla", "fla.ops.kda", "fla.ops.kda.chunk", "fla.ops.kda.chunk_fwd"):
        assert finder.find_spec(module) is None


def test_single_gpu_signed_source_plan_is_guarded():
    p = plan(ROOT / "experiments/chonk-kda-backward-release-i910bfcf936d3", site=ROOT / "config.toml",
             submit=True, gpus_per_node=1, cpus_per_task=48, cpu_bind="cores", time="00:35:00",
             reservation="cscs", cache_mode="cold", cache_image=True, exclude="nid002450")
    assert p["resources"]["nodes"] == p["resources"]["gpus_per_node"] == 1
    assert p["source"]["commit"] == VALIDATOR["COMMIT"]
    assert p["entrypoint"] == str(VALIDATOR["FILE"])
    assert p["procedure"]["timeout"] == 1500 and p["cache"]["mode"] == "cold"
    with pytest.raises(ValueError, match="launch blocked"):
        run(p)
