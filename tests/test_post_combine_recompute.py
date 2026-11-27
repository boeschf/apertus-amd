"""CPU contract checks for the guarded post-combine replay probe."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from apertus_amd.cli import plan
from apertus_amd.runs import run


ROOT = Path(__file__).resolve().parents[1]
V = runpy.run_path(ROOT / "experiments/validation/post_combine_recompute.py")


@pytest.mark.parametrize("arm", ["A", "B"])
def test_replay_contains_only_projection_and_add_and_precedes_norm_backward(arm):
    events = []
    latent, shared, residual, scaled = (object() for _ in range(4))
    summed = SimpleNamespace(bytes=V["SUM_BYTES"])
    summed.untyped_storage = lambda: SimpleNamespace(nbytes=lambda: summed.bytes)

    def region(x, s):
        assert (x, s) == (latent, shared)
        events.append("projection_add")
        return summed

    class Checkpoint:
        def __init__(self, *, fp8):
            assert fp8 is None

        def checkpoint(self, fn, *args):
            assert fn is region
            events.append("checkpoint")
            return fn(*args)

        def discard_output_and_register_recompute(self, trigger):
            assert trigger is scaled
            assert summed.bytes == V["SUM_BYTES"]
            events.append("discard_and_register")
            summed.bytes = 0

    def post_norm(value, norm):
        assert value == (summed, None) and norm is model.norm
        assert summed.bytes == V["SUM_BYTES"]
        events.append("norm")
        return scaled, None

    def bda(value, r, p):
        assert value == (scaled, None) and r is residual and p == 0.0
        events.append("residual")
        return scaled

    model = SimpleNamespace(region=region, checkpoint=Checkpoint, norm=object(), post_norm=post_norm, bda=bda)
    assert V["post_combine"](model, arm, latent, shared, residual) is scaled
    assert events == (["projection_add", "norm", "residual"] if arm == "A" else
                      ["checkpoint", "projection_add", "norm", "discard_and_register", "residual"])


def evidence():
    reports, samples = [], []
    count = V["IN_FLIGHT"]
    for arm in V["ARMS"]:
        inputs = {f"{k}:{i}": k for k in V["FIXTURES"] for i in range(count)}
        values = {f"{k}:{i}": k for k in V["DERIVATIVES"] for i in range(count)}
        reports.append(dict(arm=arm, calls=V["expected_calls"](arm), fixtures={**inputs, "projection": "p", "norm": "n"},
                            derivatives=values, wgrad=[[1, 0]] * count))
        samples.append([dict(
            entry=[100, 400], retained=[250 if arm == "A" else 200, 400],
            forward_peak=[300 if arm == "A" else 275, 410],
            backward_peak=[360 if arm == "A" else 320, 430],
            forward_seconds=1.0, backward_seconds=2.0, wgrad=[[1, 0]] * count)
            for _ in range(V["SAMPLES"])])
    return reports, samples


def test_component_gate_uses_absolute_allocated_peaks_and_retains_reservation():
    result = V["verdict"](*evidence())
    assert result["status"] == "passed"
    for pair in result["paired_B_minus_A"]:
        assert pair["retained_allocated"] == -50
        assert pair["backward_peak_allocated"] == -40
        assert pair["absolute_peak_reserved"] == 0
    assert result["expected_sum_bytes"] == 224 * 2**20


@pytest.mark.parametrize("fault", ["gradient", "entry", "backward_peak", "forward_peak"])
def test_component_gate_rejects_mismatch_or_shifted_peak(fault):
    reports, samples = evidence()
    if fault == "gradient":
        reports[2]["derivatives"]["projection_wgrad:0"] += "!"
    elif fault == "entry":
        samples[2][0]["entry"][0] += 1
    elif fault == "backward_peak":
        for sample in samples[2]:
            sample["backward_peak"][0] = 360
    else:
        for sample in samples[2]:
            sample["forward_peak"][0] = 361
    assert V["verdict"](reports, samples)["status"] == "failed"


def test_component_rejects_duplicate_wgrad_and_is_guarded():
    reports, samples = evidence()
    samples[1][0]["wgrad"] = [[2, 0]] * V["IN_FLIGHT"]
    with pytest.raises(ValueError, match="delayed-wgrad"):
        V["verdict"](reports, samples)
    recipe = ROOT / "experiments/chonk-post-combine-recompute-i910bfcf936d3"
    p = plan(recipe, site=ROOT / "config.toml", submit=True, gpus_per_node=1,
             cpus_per_task=48, cpu_bind="cores", time="00:35:00", cache_mode="cold",
             exclude="nid002450", cache_image=True, reservation="cscs")
    assert p["source"]["commit"] == V["COMMIT"] and p["image"]["sha256"] == V["IMAGE"]
    assert p["args"] == [] and p["entrypoint"] == V["ENTRYPOINT"]
    assert p["resources"]["nodes"] == p["resources"]["gpus_per_node"] == 1
    assert p["procedure"]["kind"] == "probe" and "auxiliary_source" not in p["procedure"]
    with pytest.raises(ValueError, match="launch blocked"):
        run(p)
