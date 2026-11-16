"""CPU-only probe contract, evidence gates, and source-verification failure tests."""

from copy import deepcopy
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import pytest

from apertus_amd.artifacts import sha256
from apertus_amd.config import option, phases, resolve
from apertus_amd.runs import edf
from apertus_amd.runtime import training_command


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/moe-recompute-probe-m1072583a9632-icb5a384016a9"


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("moe_recompute_probe", ROOT / "experiments/validation/moe_recompute_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=[EXPERIMENT, ROOT / "experiments/moe-recompute-probe-me242f3c3c113-icb5a384016a9"])
def settings(request):
    return resolve(request.param, submit=True)


def test_explicit_probe_uses_source_staging_and_readonly_validator(probe, settings):
    probe.validate(settings)
    assert settings["source"]["commit"] in (probe.SOURCE, probe.ROUTING_FIX_SOURCE)
    assert settings["image"]["sha256"] == probe.IMAGE
    assert settings["procedure"]["kind"] == "train"
    assert settings["validation"]["sha256"] == sha256(Path(settings["validation"]["path"]))
    container = edf(settings, Path("/run"))
    assert "/apertus/validation.py:ro" in container
    assert 'com.hooks.cxi.enabled = "false"' in container
    assert 'com.hooks.aws_ofi_nccl.enabled = "false"' in container
    assert 'HWLOC_COMPONENTS = "-gl"' in container


def test_routing_fix_experiment_changes_only_source_pin():
    original = tomllib.loads((EXPERIMENT / "experiment.toml").read_text())
    candidate = tomllib.loads((ROOT / "experiments/moe-recompute-probe-me242f3c3c113-icb5a384016a9/experiment.toml").read_text())
    assert candidate["source"].pop("commit") == "e242f3c3c1136aeae498529bfde5e06bda9a7cca"
    assert original["source"].pop("commit") == "1072583a9632fd1902544f7eaea562e2578ef1de"
    assert candidate == original


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("method", ["histogram", "marin_histogram", "marin_exact", "other"])
@pytest.mark.parametrize("drop_priority", [False, True])
def test_branch_metadata_differs_only_for_training_histogram_no_grad(
        probe, training, grad_enabled, method, drop_priority):
    # Metadata coverage only: native GPU routing/statistics still require validation.
    kwargs = dict(training=training, grad_enabled=grad_enabled, method=method, drop_priority=drop_priority)
    original = probe.expected_topk_branch(probe.SOURCE, **kwargs)
    candidate = probe.expected_topk_branch(probe.ROUTING_FIX_SOURCE, **kwargs)
    original_plus_one = (training and grad_enabled and method in ("histogram", "marin_histogram", "marin_exact")) or drop_priority
    assert original == (9 if original_plus_one else 8)
    changed = training and not grad_enabled and method == "histogram" and not drop_priority
    assert candidate == (9 if changed else original)


def test_branch_metadata_rejects_unreviewed_source(probe):
    with pytest.raises(ValueError, match="unknown routing source"):
        probe.expected_topk_branch("0" * 40, training=True, grad_enabled=False,
                                   method="histogram", drop_priority=False)


def test_launcher_appended_save_has_explicit_native_save_interval(settings, tmp_path):
    phase = phases(settings, tmp_path / "output")[0]
    command = training_command(settings, Path(phase["output"]), tmp_path / "source", phase["extra_args"])
    assert command[1:3] == ["/apertus/validation.py", "train"]
    # Native validate_args at source 1072583a9632, arguments.py:1196-1203:
    # --save requires a save interval/iteration list, and any interval must be positive.
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--save-iters")
    args, _ = parser.parse_known_args(command[6:])
    assert args.save == str(tmp_path / "output/checkpoints")
    assert args.save_interval is not None or args.save_iters is not None
    assert args.save_interval == 1 and args.save_interval > 0
    assert command[-2:] == ["--save", args.save]


def test_explicit_args_satisfy_inspected_native_initialization_prerequisites(settings):
    # Source-inspected scalar prerequisites, not execution of GPU-dependent validate_args.
    args = settings["args"]
    assert int(option(args, "--max-position-embeddings")) >= int(option(args, "--seq-length"))
    assert int(option(args, "--hidden-size")) % int(option(args, "--num-attention-heads")) == 0
    world = settings["resources"]["nodes"] * settings["resources"]["gpus_per_node"]
    ep = int(option(args, "--expert-model-parallel-size"))
    assert world % ep == int(option(args, "--num-experts")) % ep == 0
    assert all(option(args, flag) == "1" for flag in ("--tensor-model-parallel-size",
        "--pipeline-model-parallel-size", "--context-parallel-size", "--expert-tensor-parallel-size"))
    assert int(option(args, "--global-batch-size")) == 2 * world * int(option(args, "--micro-batch-size"))
    assert option(args, "--recompute-granularity") == "selective"
    assert option(args, "--recompute-method") is None
    assert option(args, "--moe-latent-size") == "1792" and "--use-legacy-models" not in args
    assert "--mock-data" in args and not any(flag in args for flag in (
        "--data-path", "--train-data-path", "--valid-data-path", "--test-data-path"))
    assert option(args, "--tokenizer-type") == "NullTokenizer" and int(option(args, "--vocab-size")) > 0
    # The native PP1 inference restriction is inactive at its pinned default -1.
    assert int(option(args, "--inference-batch-times-seqlen-threshold", -1)) == -1


@pytest.mark.parametrize("mutation", [
    lambda s: s["source"].update(commit="0" * 40),
    lambda s: s["image"].update(sha256="0" * 64),
    lambda s: s["resources"].update(nodes=16),
    lambda s: s["resources"].update(gpus_per_node=1),
    lambda s: s.update(compare_run="previous"),
    lambda s: s["procedure"].update(kind="restart"),
    lambda s: s["args"].remove("--sssglu"),
    lambda s: s["args"].remove("--no-gradient-accumulation-fusion"),
    lambda s: s["args"].remove("--save-interval"),
    lambda s: s["args"].extend(["--save-interval", "0"]),
    lambda s: s["args"].extend(["--moe-latent-size", "3584"]),
    lambda s: s["args"].append("--moe-use-offloading-experts"),
    lambda s: s["args"].append("--cpu-offloading"),
    lambda s: s["environment"].update(TORCH_COMPILE_DISABLE="1"),
])
def test_probe_rejects_scope_changes(probe, settings, mutation):
    mutation(settings)
    with pytest.raises(ValueError):
        probe.validate(settings)


def test_numerical_gates_require_both_elementwise_and_normwise(probe):
    valid = {"finite": True, "max_tolerance_ratio": 0.5, "rms_error": 0.001, "reference_rms": 1.0}
    assert probe.accepted(valid)
    assert not probe.accepted({**valid, "max_tolerance_ratio": 1.01})
    assert not probe.accepted({**valid, "rms_error": 0.01})
    assert not probe.accepted({**valid, "finite": False})
    assert not probe.accepted({**valid, "rms_error": float("nan")})
    assert not probe.accepted({**valid, "reference_rms": float("inf")})
    assert not probe.accepted({**valid, "max_tolerance_ratio": None})
    assert probe.accepted({**valid, "reference_rms": 0.0, "rms_error": 0.0})
    assert not probe.accepted({**valid, "reference_rms": 0.0})


def test_effective_native_config_uses_only_pinned_config_fields(probe):
    activation = object()
    fields = {key: False for key in ("gradient_accumulation_fusion", "cpu_offloading",
                                    "fine_grained_activation_offloading", "moe_use_offloading_experts",
                                    "moe_shared_expert_overlap", "fp8", "fp4", "use_te_activation_func")}
    fields.update(activation_func=activation, bias_activation_fusion=True, cuda_graph_impl="none")
    # use_te_op_fuser is NOT a TransformerConfig field in the selected source.
    probe.validate_native_config(SimpleNamespace(**fields), activation, 2)
    for key in ("gradient_accumulation_fusion", "cpu_offloading", "moe_shared_expert_overlap", "fp8",
                "use_te_activation_func"):
        with pytest.raises(ValueError):
            probe.validate_native_config(SimpleNamespace(**{**fields, key: True}), activation, 2)
    with pytest.raises(ValueError):
        probe.validate_native_config(SimpleNamespace(**fields), activation, 1)
    with pytest.raises(ValueError):
        probe.validate_native_config(SimpleNamespace(**fields), object(), 2)


def event(enabled, rows):
    return {"grad_enabled": enabled, "histogram_rows_per_expert": [rows] * 256}


@pytest.mark.parametrize("microbatch", [0, 1])
def test_router_counts_only_one_grad_enabled_pass(probe, microbatch):
    previous, current = 4096 * microbatch, 4096 * (microbatch + 1)
    full = [event(False, previous), event(True, current)]
    activation = [event(True, current)]
    assert probe.check_router_trace(full, "moe", microbatch)
    assert probe.check_router_trace(activation, "moe_act", microbatch)
    assert not probe.check_router_trace([event(False, current), event(True, current + 4096)], "moe", microbatch)
    assert not probe.check_router_trace(full[:1], "moe", microbatch)
    assert not probe.check_router_trace(activation * 2, "moe_act", microbatch)
    assert not probe.check_router_trace([event(True, previous)], "moe_act", microbatch)
    damaged = deepcopy(full)
    damaged[-1]["histogram_rows_per_expert"][17] += 1
    assert not probe.check_router_trace(damaged, "moe", microbatch)


def test_local_load_oracle_uses_own_grad_enabled_map_without_waiving_replay(probe):
    original, replay, candidate = object(), object(), object()
    full_observations = [(None, original), (None, replay)]
    full_trace = [event(False, 0), event(True, 4096)]
    assert probe.state_update_routing(full_observations, full_trace) is replay
    assert probe.state_update_routing([(None, candidate)], [event(True, 4096)]) is candidate
    # The correction separates accounting from equivalence: every exact routing
    # gate remains required, even when the counter is correct for its replay map.
    required = probe.required_labels(["router.weight"])
    assert {"mb0/moe/replay-probabilities", "mb0/moe/replay-routing-map",
            "mb0/probabilities", "mb0/routing-map", "mb0/forward"} <= required


@pytest.mark.parametrize("observations,trace", [
    ([], []),
    ([(None, "map")], [event(False, 0)]),
    ([(None, "map")], []),
    ([(None, "map"), (None, "map")], [event(True, 4096), event(True, 8192)]),
])
def test_local_load_oracle_rejects_missing_or_double_counted_router_calls(probe, observations, trace):
    with pytest.raises(ValueError):
        probe.state_update_routing(observations, trace)


@pytest.fixture
def tied_boundary():
    # Synthetic captured values, NOT a claim about CPU/GPU Torch tie ordering.
    # Both tie orders are legal; the fixture represents observed same-k-stable
    # top-8 versus top-9-prefix behavior with different grad-enabled branches.
    scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.6] + [0.1] * 247
    logits = [math.log(value / (1 - value)) for value in scores]
    top8, top9 = list(range(8)), [0, 1, 2, 3, 4, 5, 6, 8, 7]
    calls = {name: {"grad_enabled": enabled, "topk_branch": 9 if enabled else 8,
                   "repeat8_exact": True, "repeat9_exact": True, "actual_matches_branch": True}
             for name, enabled in (("original", False), ("replay", True), ("candidate", True))}
    comparisons = {name: {"logits": {"exact_equal": True, "bitwise_equal": True, "finite": True, "max_abs": 0.0},
                         "same_observed_inputs": True, "actual_expert_sets": {"different_rows": rows}}
                   for name, rows in (("original_vs_replay", 1), ("original_vs_candidate", 1), ("replay_vs_candidate", 0))}
    return logits, scores, top8, top9, calls, comparisons


def test_tied_boundary_diagnostics_preserve_all_scores_and_actual_selected_experts(probe, tied_boundary):
    logits, scores, top8, top9, calls, comparisons = tied_boundary
    original = probe.routing_row(logits, scores, scores, top8, top8, top9)
    replay = probe.routing_row(logits, scores, scores, top9[:8], top8, top9)
    assert len(original["raw_logits"]) == len(original["sigmoid_biased_scores"]) == 256
    assert original["actual_selected8"] != replay["actual_selected8"]
    assert original["eighth_ninth_gap"] == 0
    assert original["tied_at_eighth"] == original["tied_at_ninth"] == [7, 8]
    assert probe.routing_diagnosis(calls, comparisons, all_changed_rows_tied=True) == \
        "observed_deterministic_topk_boundary_branch_difference"
    assert calls["original"]["grad_enabled"] is False and calls["replay"]["grad_enabled"] is True


@pytest.mark.parametrize("damage", ["logits", "bits", "input", "repeat", "branch_match", "branch_mode", "not_tied"])
def test_unresolved_diagnostics_require_explicit_same_mode_repeat(probe, tied_boundary, damage):
    _, _, _, _, calls, comparisons = tied_boundary
    if damage == "logits":
        comparisons["original_vs_replay"]["logits"]["exact_equal"] = False
    elif damage == "bits":
        comparisons["original_vs_replay"]["logits"]["bitwise_equal"] = False
    elif damage == "input":
        comparisons["original_vs_replay"]["same_observed_inputs"] = False
    elif damage == "repeat":
        calls["original"]["repeat8_exact"] = False
    elif damage == "branch_match":
        calls["replay"]["actual_matches_branch"] = False
    elif damage == "branch_mode":
        calls["original"]["topk_branch"] = 9
    assert probe.routing_diagnosis(calls, comparisons, all_changed_rows_tied=damage != "not_tied") == \
        "unresolved_same_mode_repeat_required"


def test_raw_row_nonfinite_values_remain_json_serializable(probe, tied_boundary):
    logits, scores, top8, top9, _, _ = tied_boundary
    logits[32] = float("nan")
    row = probe.routing_row(logits, scores, scores, top8, top8, top9)
    assert row["raw_logits"][32] is None
    json.dumps(row, allow_nan=False)


def test_equal_routes_do_not_claim_a_boundary_branch_failure(probe, tied_boundary):
    _, _, _, _, calls, comparisons = tied_boundary
    for pair in comparisons.values():
        pair["actual_expert_sets"]["different_rows"] = 0
    assert probe.routing_diagnosis(calls, comparisons, all_changed_rows_tied=True) == "no_expert_set_difference_observed"


def test_rng_fingerprints_are_stable_and_detect_changed_family(probe, monkeypatch):
    monkeypatch.setattr(probe, "tensor_hash", lambda value: f"hash:{value}")
    state = {"python": (1, 2), "numpy": ("state", 3), "cpu": "cpu-state", "cuda": "cuda-state",
             "tracker": {"expert-parallel-rng": "expert-state"}}
    before = probe.rng_fingerprints(state)
    assert before == probe.rng_fingerprints(deepcopy(state))
    state["tracker"]["expert-parallel-rng"] = "changed"
    after = probe.rng_fingerprints(state)
    assert before["tracker"] != after["tracker"]
    assert before["cpu"] == after["cpu"] and before["cuda"] == after["cuda"]


@pytest.mark.parametrize("source_name", ["SOURCE", "ROUTING_FIX_SOURCE"])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("failure", [None, "native", "capture", "registration"])
def test_routing_callthrough_is_transparent_owned_and_restored(probe, monkeypatch, source_name, grad_enabled, failure):
    class Tensor:
        def __init__(self, value):
            self.value = list(value)

        def detach(self):
            return self

        def clone(self):
            return Tensor(self.value)

    torch = SimpleNamespace(is_grad_enabled=lambda: grad_enabled)
    result, padding = object(), object()
    value = Tensor([1, 2, 3])
    calls = []

    class Router:
        def __init__(self):
            self.weight, self.qb_beta = Tensor([4]), Tensor([0])
            self.training = True
            self.config = SimpleNamespace(moe_expert_capacity_factor=None, moe_token_drop_policy="probs",
                                          moe_router_quantile_balancing_method="histogram")
            self.hooks = []
            self.native_calls = 0

        def register_forward_pre_hook(self, hook):
            if failure == "registration":
                raise RuntimeError("registration failed")
            self.hooks.append(hook)
            return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

        def routing(self, logits, *, padding_mask):
            self.native_calls += 1
            assert logits is value and padding_mask is padding
            assert torch.is_grad_enabled() is grad_enabled
            if failure == "native":
                raise RuntimeError("native failed")
            return result

        def __call__(self, tensor):
            for hook in self.hooks:
                hook(self, (tensor,))
            return self.routing(logits=tensor, padding_mask=padding)

    def snapshot(*args):
        if failure == "capture":
            raise RuntimeError("capture failed")
        return {"state": "unchanged"}

    monkeypatch.setattr(probe, "rng_snapshot", snapshot)
    router = Router()
    original = router.routing.__func__
    try:
        with probe.capture_router_calls(torch, router, calls, None, source=getattr(probe, source_name)):
            if failure:
                with pytest.raises(RuntimeError):
                    router(value)
                # Propagate a failure out of the observer scope, not just out of native routing.
                raise RuntimeError("scope failed")
            assert router(value) is result
            assert len(calls) == 1
            assert calls[0]["grad_enabled"] is calls[0]["routing_grad_enabled"] is grad_enabled
            assert calls[0]["topk_branch"] == (9 if grad_enabled or source_name == "ROUTING_FIX_SOURCE" else 8)
            value.value[0] = 99
            router.weight.value[0] = 99
            assert calls[0]["input"].value == calls[0]["logits"].value == [1, 2, 3]
            assert calls[0]["weight"].value == [4]
    except RuntimeError:
        assert failure is not None
    assert router.routing.__func__ is original and "routing" not in vars(router)
    assert router.hooks == []
    assert router.native_calls == (0 if failure in {"capture", "registration"} else 1)


def test_routing_observer_restores_existing_instance_override(probe, monkeypatch):
    existing = lambda *args, **kwargs: None
    router = SimpleNamespace(routing=existing, register_forward_pre_hook=lambda hook: SimpleNamespace(remove=lambda: None))
    monkeypatch.setattr(probe, "rng_snapshot", lambda *args: {})
    with pytest.raises(RuntimeError, match="scope failed"):
        with probe.capture_router_calls(SimpleNamespace(), router, [], None):
            raise RuntimeError("scope failed")
    assert router.routing is existing


def test_routing_method_restored_even_when_handle_removal_raises(probe):
    def remove():
        raise RuntimeError("remove failed")

    existing = lambda *args, **kwargs: None
    router = SimpleNamespace(routing=existing, register_forward_pre_hook=lambda hook: SimpleNamespace(remove=remove))
    with pytest.raises(RuntimeError, match="remove failed"):
        with probe.capture_router_calls(SimpleNamespace(), router, [], None):
            pass
    assert router.routing is existing


def test_new_output_gates_add_to_instead_of_replacing_existing_acceptance(probe):
    labels = probe.required_labels(["router.weight"])
    for mb in range(2):
        for suffix in ("forward", "forward-pre-backward", "input-gradient", "probabilities", "routing-map",
                       "moe/output-unmutated", "moe_act/output-unmutated", "moe/replay-probabilities",
                       "moe/replay-routing-map", "weight-gradient/router.weight", "accumulated-gradient/router.weight"):
            assert f"mb{mb}/{suffix}" in labels
    assert probe.MAX_MISMATCH_ROWS == 16


def test_router_snapshot_excludes_only_uninitialized_future_slots(probe):
    class Tensor:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, key):
            return Tensor(self.value[key])

        def detach(self):
            return self

        def clone(self):
            return deepcopy(self.value)

    buffers = {"qb_histogram": Tensor([[1, 2], [3, 4]]),
               "mbs_expert_load_samples": Tensor([[4096], ["uninitialized"]])}
    router = SimpleNamespace(expert_load_sample_count=SimpleNamespace(item=lambda: 1),
                             named_buffers=lambda: buffers.items())
    result = probe.router_snapshot(router)
    assert result == {"qb_histogram": [[1, 2], [3, 4]], "mbs_expert_load_samples": [[4096]]}


def make_evidence(probe, settings, run):
    (run / "output").mkdir()
    (run / "resolved.json").write_text(json.dumps(settings))
    for rank in range(16):
        value = {"status": "passed", "scope": probe.SCOPE, "rank": rank,
                  "source": settings["source"]["commit"], "image": probe.IMAGE,
                 "microbatches": [{"index": 0}, {"index": 1}], "parameters": {"router.weight": {}},
                 "checks": [{"label": label, "passed": True}
                            for label in sorted(probe.required_labels(["router.weight"]))]}
        (run / f"output/moe-recompute-{rank}.json").write_text(json.dumps(value))
        (run / f"output/imports-{rank}.json").write_text(json.dumps({"verified_modules": 100}))


def test_all_rank_acceptance_is_component_only(probe, settings, tmp_path):
    make_evidence(probe, settings, tmp_path)
    probe.check(tmp_path)
    value = json.loads((tmp_path / "output/checks.json").read_text())
    assert value["status"] == "passed"
    assert "not full Chonk" in value["scope"]
    assert value["optimizer_validation"] == value["restart_validation"] == "not performed"


@pytest.mark.parametrize("damage", ["missing_rank", "failed", "partial", "missing_gradient", "duplicate", "imports", "wrong_source"])
def test_all_rank_check_retains_failures(probe, settings, tmp_path, damage):
    make_evidence(probe, settings, tmp_path)
    path = tmp_path / "output/moe-recompute-15.json"
    value = json.loads(path.read_text())
    if damage == "missing_rank":
        path.unlink()
    elif damage == "imports":
        (tmp_path / "output/imports-15.json").write_text('{"verified_modules": 0}')
    else:
        if damage == "failed":
            value["checks"][0]["passed"] = False
        elif damage == "partial":
            value["microbatches"].pop()
        elif damage == "missing_gradient":
            value["checks"] = [r for r in value["checks"] if "weight-gradient" not in r["label"]]
        elif damage == "duplicate":
            value["checks"].append(value["checks"][0])
        elif damage == "wrong_source":
            value["source"] = probe.ROUTING_FIX_SOURCE if value["source"] == probe.SOURCE else probe.SOURCE
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        probe.check(tmp_path)
    assert json.loads((tmp_path / "output/checks.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("failure", ["initialize", "probe", "final_imports", None])
def test_train_verifies_before_compute_and_on_every_exit(probe, settings, tmp_path, monkeypatch, failure):
    source = tmp_path / "native"
    source.mkdir()
    (source / "pretrain_gpt.py").write_text("raise AssertionError('GPT entrypoint must not run')\n")
    run = tmp_path / "run"
    (run / "source").mkdir(parents=True)
    output = run / "output"
    output.mkdir()
    (run / "resolved.json").write_text(json.dumps(settings))
    (run / "source/manifest.json").write_text(json.dumps({"selection": {"commit": settings["source"]["commit"]},
        "files": {"pretrain_gpt.py": sha256(source / "pretrain_gpt.py")}}))
    calls = []

    def verify(*args):
        calls.append("verify")
        if failure == "final_imports" and calls.count("verify") == 3:
            raise ValueError("bad imports")
        return 4

    def initialize():
        calls.append("initialize")
        if failure == "initialize":
            raise ValueError("initialization failure")

    def compute(emit, record, **kwargs):
        calls.append("compute")
        if failure == "probe":
            raise ValueError("probe failure")
        emit({"label": "stub", "passed": True})

    monkeypatch.setattr(probe, "verify_imports", verify)
    monkeypatch.setattr(probe, "run_probe", compute)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", sys.argv.copy())
    monkeypatch.setitem(sys.modules, "megatron", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "megatron.training", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "megatron.training.initialize", SimpleNamespace(initialize_megatron=initialize))
    if failure:
        with pytest.raises(ValueError):
            probe.train(output, source, "pretrain_gpt.py", [], run=run)
    else:
        probe.train(output, source, "pretrain_gpt.py", [], run=run)
    assert calls[0] == calls[-1] == "verify"
    if "compute" in calls:
        assert calls[:4] == ["verify", "initialize", "verify", "compute"]
    record = json.loads((output / "moe-recompute-0.json").read_text())
    assert record["status"] == ("failed" if failure else "passed")
    assert record["source"] == settings["source"]["commit"]


def test_source_entrypoint_identity_rejected_before_native_imports(probe, settings, tmp_path, monkeypatch):
    (tmp_path / "resolved.json").write_text(json.dumps(settings))
    (tmp_path / "source").mkdir()
    (tmp_path / "source/manifest.json").write_text(json.dumps({"selection": {"commit": "0" * 40}}))
    monkeypatch.setattr(probe, "verify_imports", lambda *args: pytest.fail("must reject before imports"))
    with pytest.raises(ValueError, match="identity mismatch"):
        probe.train(tmp_path, tmp_path, "pretrain_gpt.py", [], run=tmp_path)
