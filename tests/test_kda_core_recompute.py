"""CPU checks for the guarded Megatron KDA core-checkpoint comparison."""

from pathlib import Path
import runpy
import tomllib
from types import SimpleNamespace

import pytest

from apertus_amd.cli import plan
from apertus_amd.runs import edf, run


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/chonk-kda-core-recompute-i910bfcf936d3"
V = runpy.run_path(ROOT / "experiments/validation/kda_core_recompute.py")


def evidence():
    count = V["IN_FLIGHT"]
    reports, trials = [], []
    for arm in V["ARMS"]:
        reports.append(dict(
            arm=arm, modules=V["modules"](arm), calls=V["expected_calls"](arm),
            shape=list(V["SHAPE"]), in_flight=count, target="gfx942",
            module_sha256=V["RELEASE"]["sha256"], delay_wgrad_compute=False, conv_backend="triton",
            fixtures={**{f"{k}:{i}": {"sha256": "a"} for k in ("input", "cotangent") for i in range(count)},
                      "param:weight": {"sha256": "b"}},
            numerics={**{f"{k}:{i}": {"sha256": "c"} for k in ("output", "input_grad") for i in range(count)},
                      "param_grad:weight": {"sha256": "d"}},
        ))
        retained, backward = ([130, 208], [190, 252]) if arm == "A" else ([150, 220], [195, 246])
        trials.append([dict(entry=[100, 200], retained=retained[:], forward_peak=[170, 225],
                            backward_peak=backward[:], forward_seconds=1.0, backward_seconds=2.0)
                       for _ in range(V["SAMPLES"])])
    return reports, trials


def test_abba_keeps_tradeoffs_separate():
    data = V["verdict"](*evidence())
    assert data["status"] == "passed"
    for delta in data["paired_B_minus_A"]:
        assert delta["forward_retained_allocated"] == 20
        assert delta["backward_peak_allocated"] == 5
        assert delta["backward_peak_above_entry_allocated"] == -15
        assert delta["combined_peak_reserved"] == -6
    assert V["expected_calls"]("A")["backward"]["chunk_kda"] == V["IN_FLIGHT"]
    assert V["expected_calls"]("B")["backward"]["chunk_kda"] == 0


def test_parity_and_forward_entry_are_independent_guards():
    reports, trials = evidence()
    reports[2]["numerics"]["input_grad:0"] = {"sha256": "unequal"}
    assert V["verdict"](reports, trials)["status"] == "failed"
    assert V["verdict"](reports, trials)["differences_from_first_arm"][2]["numerics"] == ["input_grad:0"]
    reports, trials = evidence()
    trials[1][0]["entry"][0] -= 1
    assert V["verdict"](reports, trials)["status"] == "failed"
    assert not V["verdict"](reports, trials)["matched_entry_allocated"]


@pytest.mark.parametrize("fault", ["calls", "signed_module", "missing_gradient", "missing_sample", "backend"])
def test_incomplete_or_unmatched_evidence_is_rejected(fault):
    reports, trials = evidence()
    if fault == "calls":
        reports[1]["calls"]["backward"]["chunk_kda"] += 1
    elif fault == "signed_module":
        reports[1]["module_sha256"] = "wrong"
    elif fault == "missing_gradient":
        reports[1]["numerics"].pop("param_grad:weight")
    elif fault == "missing_sample":
        trials[1].pop()
    else:
        reports[1]["conv_backend"] = "cuda"
    with pytest.raises(ValueError, match="incomplete or incomparable"):
        V["verdict"](reports, trials)


def test_phase_measurements_reject_wrong_boundaries():
    _, trials = evidence()
    trials[0][0]["backward_peak"][0] = 1
    with pytest.raises(ValueError, match="invalid phase measurement"):
        V["verdict"](evidence()[0], trials)


def test_standalone_config_preserves_full_run_linear_attention_frequency():
    config_a, config_b = (V["kda_config"](dict, "bf16", arm) for arm in ("A", "B"))
    full = tomllib.loads((ROOT / "experiments/chonk-native-fnuz-kda-release-i910bfcf936d3/experiment.toml").read_text())
    assert full["args"][full["args"].index("--linear-attention-freq") + 1] == "([1,1,1,0]*10+[1,0])"
    assert config_a["linear_attention_freq"] == [1, 1, 1, 0] * 10 + [1, 0]
    assert len(config_a["linear_attention_freq"]) == config_a["num_layers"] == 42
    assert config_a["linear_attention_freq"][0] == 1
    assert sum(config_a["linear_attention_freq"]) == 31
    assert {k for k in config_a if config_a[k] != config_b[k]} == {"recompute_modules"}


def test_schedule_keeps_all_graphs_then_drains_oldest_first():
    calls = []

    class Input:
        def __init__(self, index):
            self.index = index
            self.grad = None

        def detach(self):
            return Input(self.index)

        def requires_grad_(self):
            return self

    class Output:
        dtype, shape, grad_fn = "bf16", V["SHAPE"], object()

        def __init__(self, x):
            self.x = x

        def data_ptr(self):
            return self.x.index

        def backward(self, cotangent):
            assert cotangent == self.x.index
            calls.append(("backward", self.x.index))
            self.x.grad = cotangent

    class Model:
        def zero_grad(self, *, set_to_none):
            assert set_to_none

        def __call__(self, x, *, attention_mask):
            assert attention_mask is None
            calls.append(("forward", x.index))
            return Output(x), None

        def named_parameters(self):
            return iter(())

    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                           memory_allocated=lambda: 100, memory_reserved=lambda: 200,
                           max_memory_allocated=lambda: 150, max_memory_reserved=lambda: 225)
    indices = range(V["IN_FLIGHT"])
    receipts = {}
    V["schedule"](SimpleNamespace(cuda=cuda, bfloat16="bf16"), Model(), list(map(Input, indices)), indices,
                  advance=lambda: calls.append("boundary"), receipt=receipts.__setitem__)
    assert calls == ([("forward", i) for i in indices] + ["boundary"] + [("backward", i) for i in indices])
    assert all(receipts[f"input_grad:{i}"] == i for i in indices)


def test_single_gpu_plan_is_guarded_and_uses_signed_auxiliary(tmp_path):
    p = plan(RECIPE, site=ROOT / "config.toml", submit=True, gpus_per_node=1, cpus_per_task=48,
             cpu_bind="cores", time="00:35:00", reservation="cscs", cache_mode="cold",
             cache_image=True, exclude="nid002450")
    full = tomllib.loads((ROOT / "experiments/chonk-native-fnuz-kda-release-i910bfcf936d3/experiment.toml").read_text())
    start = full["args"].index("--recompute-modules") + 1
    assert V["modules"]("A") == full["args"][start: start + 4]
    assert V["modules"]("B") == [m for m in V["modules"]("A") if m != "linear_attn"]
    assert p["resources"]["nodes"] == p["resources"]["gpus_per_node"] == 1
    assert p["source"] == full["source"] and p["image"] == full["image"]
    assert p["procedure"]["auxiliary_source"] == V["RELEASE"] == full["procedure"]["auxiliary_source"]
    assert p["entrypoint"] == V["ENTRYPOINT"] and p["procedure"]["timeout"] == 1500
    assert p["cache"]["mode"] == "cold"
    assert str(tmp_path / "auxiliary.py") + ":/apertus/auxiliary.py:ro" in edf(p, tmp_path)
    finder = V["SignedBackwardFinder"]()
    assert Path(finder.find_spec(V["MODULE"]).origin) == V["AUXILIARY"]
    assert all(finder.find_spec(name) is None for name in ("fla", "fla.ops.kda", "fla.ops.kda.chunk"))
    with pytest.raises(ValueError, match="launch blocked"):
        run(p)
