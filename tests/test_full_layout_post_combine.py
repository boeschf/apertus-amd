"""CPU contracts for the signed full-layout replay diagnostic."""

from copy import deepcopy
import json
from pathlib import Path
from queue import Queue
import runpy
from types import SimpleNamespace

import pytest

from apertus_amd.cli import plan
from apertus_amd.runs import run


ROOT = Path(__file__).resolve().parents[1]
V = runpy.run_path(ROOT / "experiments/validation/chonk_fnuz_oom.py")


def planned(name="post-combine"):
    return plan(ROOT / f"experiments/chonk-native-fnuz-{name}-i910bfcf936d3",
                site=ROOT / "config.toml", submit=True, gpus_per_node=4, cpus_per_task=48,
                cpu_bind="map_ldom:0,1,2,3", time="01:00:00", cache_mode="cold",
                exclude="nid002450", cache_image=True, reservation="cscs")


def frozen():
    return {**planned(), "source_snapshot": dict(tree=V["POST_COMBINE_TREE"],
                                                sha256=V["POST_COMBINE_ARCHIVE"],
                                                manifest_sha256=V["POST_COMBINE_MANIFEST"])}


def test_guard_and_baseline_match():
    p, base = frozen(), planned("pp1-allocator")
    assert p["args"] == base["args"] + [V["POST_COMBINE_FLAG"]]
    assert p["resources"] == base["resources"] and p["prepare"] == base["prepare"]
    assert p["image"] == base["image"]
    assert all(base["environment"][key] == value for key, value in p["environment"].items())
    assert V["post_combine_profile"](p) is True
    assert V["post_combine_profile"](base) is False
    with pytest.raises(ValueError, match="launch blocked"):
        run(p)


@pytest.mark.parametrize("field", ["args", "source_snapshot", "environment", "procedure"])
def test_profile_rejects_unmatched_inputs(field):
    p = frozen()
    if field == "args":
        p[field].extend(["--train-iters", "2"])
    else:
        key, value = {"source_snapshot": ("tree", "mismatch"),
                      "environment": (V["PP1_ALLOCATOR_TRACE"], "1"),
                      "procedure": ("timeout", 2800)}[field]
        p[field][key] = value
    with pytest.raises(ValueError, match="unexpected full-layout"):
        V["post_combine_profile"](p)


def test_lifecycle_distinguishes_local_replay_and_drains_exactly_one():
    probe = V["PostCombineLifecycle"](16)
    queue = Queue()

    class MoE:
        layer_number = 4
        fc2_latent_proj = SimpleNamespace(wgrad_store=SimpleNamespace(context=queue))

        def local(self):
            pass

    moe = MoE()
    target = SimpleNamespace(run_function=moe.local, ctx=object(), outputs=object())
    other = SimpleNamespace(run_function=lambda: None, ctx=object(), outputs=object())

    def run_forward(m, output, shared, *, checkpoint=None):
        return output

    def run_replay(checkpoint, grad):
        checkpoint.ctx = checkpoint.outputs = None

    def run_drain(m, *, routed_experts, shared_experts):
        if routed_experts:
            queue.get_nowait()

    forward = probe.wrap_forward(run_forward)
    replay = probe.wrap_replay(run_replay, MoE.local)
    drain = probe.wrap_drain(run_drain)
    with pytest.raises(ValueError, match="did not arm"):
        forward(moe, None, None)
    assert forward(moe, "sum", None, checkpoint=target) == "sum"
    replay(other, None)
    queue.put(object())
    with pytest.raises(ValueError, match="queued exactly once"):
        drain(moe)
    replay(target, None)
    drain(moe, routed_experts=False, shared_experts=True)
    queue.put(object())
    with pytest.raises(ValueError, match="queued exactly once"):
        drain(moe)
    queue.get_nowait()
    drain(moe)
    assert probe.counts() == dict.fromkeys(("forward", "replay", "drain"), {"4": 1})


def test_scope_and_lifecycle_receipts_require_full_two_chunk_schedule():
    count = 2 * V["MICRO"]
    warmup = 9
    exits = [dict(event="schedule_exit", call=i + 1,
                  forward=i if i < count else None,
                  backward=i - warmup if i >= warmup else None)
             for i in range(count + warmup)]
    layers = V["POST_COMBINE_LAYERS"][1]
    exits[8]["post_combine_counts"] = dict(forward={str(k): 5 if k < 26 else 4 for k in layers},
                                           replay={}, drain={})
    peak = dict(allocated=100, reserved=120, first_call=10, last_call=count + warmup)
    exits[-1].update(backward_phase_peak_allocated=100, backward_phase_peak_reserved=120)
    rows = [*exits, dict(event="post_combine_backward_peak", **peak),
            dict(event="update_exit", calls=count + warmup, grad_norm=1., losses={}),
            dict(event="post_combine_lifecycle", counts=dict.fromkeys(
                ("forward", "replay", "drain"), {str(k): V["MICRO"] for k in layers}))]
    assert V["verify_post_combine_trace"](rows, 16) == peak
    nonfinite = deepcopy(rows)
    nonfinite[-2]["grad_norm"] = float("inf")
    with pytest.raises(ValueError, match="finite update"):
        V["verify_post_combine_trace"](nonfinite, 16)
    incomplete = deepcopy(rows)
    incomplete[-1]["counts"]["drain"] = {"4": V["MICRO"]}
    with pytest.raises(ValueError, match="lifecycle"):
        V["verify_post_combine_trace"](incomplete, 16)
    rows[8]["post_combine_counts"]["forward"]["4"] -= 1
    with pytest.raises(ValueError, match="54"):
        V["verify_post_combine_trace"](rows, 16)


def test_compare_retains_observed_per_rank_deltas_without_imputing_node_totals(tmp_path):
    base = tmp_path / V["PP1_REPLAY_BASELINE"]
    replay = tmp_path / "replay"
    for path, plan_data in ((base, planned("pp1-allocator")), (replay, frozen())):
        out = path / "output"
        out.mkdir(parents=True)
        (path / "resolved.json").write_text(json.dumps(plan_data))
        row = dict(event="schedule_exit", rank=16, call=9, forward=8, backward=None,
                   allocated=120 if path == base else 100, reserved=140, peak_allocated=125, peak_reserved=140)
        (out / "gpu-progress-rank-16.jsonl").write_text(json.dumps(row) + "\n{\"event\":")
    report = V["compare_post_combine_memory"](replay, base)
    fill = report["per_rank"][16]["schedule_exit:9"]
    assert fill["replay_minus_baseline"]["allocated"] == -20
    assert report["per_rank"][17]["schedule_exit:9"]["baseline"] is None
    assert report["fill_exit_node_deltas"][4] is None


def test_sync_save_plan_matches_accepted_training_and_is_guarded():
    p, base = planned("sync-save"), frozen()
    assert p["args"] == base["args"] + list(V["SYNC_SAVE_ARGS"])
    assert p["source"] == base["source"] and p["environment"] == base["environment"]
    assert p["phases"][0]["extra_args"] == ["--save", "/apertus/output/checkpoints"]
    p["source_snapshot"] = base["source_snapshot"]
    assert V["post_combine_profile"](p)
    with pytest.raises(ValueError, match="launch blocked"):
        run(p)
    p["args"][-1] = "pinned"
    with pytest.raises(ValueError, match="checkpoint options"):
        V["post_combine_profile"](p)


def test_sync_save_observer_records_full_state_and_native_return(tmp_path):
    target = tmp_path / "checkpoints"
    batch = V["BATCH"]
    args = SimpleNamespace(consumed_train_samples=batch, save=str(target), ckpt_format="torch_dist",
                           async_save=False, no_save_optim=False, no_save_rng=False,
                           ckpt_staging="streamed")
    checkpointing = SimpleNamespace(
        generate_state_dict=lambda: dict(iteration=1, args=args, model0={"a": 1}, model1={"b": 1},
                                         optimizer={0: "state"}, opt_param_scheduler={"num_steps": batch},
                                         rng_state=["state"]),
    )
    training = SimpleNamespace(get_args=lambda: args,
                               save_checkpoint_and_time=lambda iteration: checkpointing.generate_state_dict())
    trace = [dict(event="update_exit")]
    emit = lambda event, **fields: trace.append(dict(event=event, **fields))
    complete = V["observe_sync_save"](training, checkpointing, target, emit, lambda: 1)
    assert not complete()
    args.async_save = True
    with pytest.raises(ValueError, match="checkpoint boundary"):
        training.save_checkpoint_and_time(1)
    args.async_save = False
    assert training.save_checkpoint_and_time(1)["optimizer"]
    assert complete()
    with pytest.raises(ValueError, match="checkpoint boundary"):
        training.save_checkpoint_and_time(1)
    trace.append(dict(event="completed"))
    assert V["verify_sync_save_trace"](trace, 0) > 0
    del trace[2]
    with pytest.raises(ValueError, match="save evidence"):
        V["verify_sync_save_trace"](trace, 0)


def test_checkpoint_inventory_requires_valid_referenced_extents(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1\n")
    target = tmp_path / "iter_0000001"
    target.mkdir()
    (target / ".metadata").write_text("metadata")
    (target / "common.pt").write_text("common")
    (target / "metadata.json").write_text(json.dumps(dict(sharded_backend="torch_dist",
                                                           common_backend="torch")))
    shard = target / "__0_0.distcp"
    shard.write_bytes(b"abcde")
    entry = SimpleNamespace(relative_path=shard.name, offset=1, length=4)
    reader = lambda _: SimpleNamespace(read_metadata=lambda: SimpleNamespace(storage_data={0: entry}))
    result = V["checkpoint_inventory"](tmp_path, reader=reader)
    assert result["shards"] == result["metadata_extents"] == 1
    assert result["tensor_integrity"] == result["restoration"] == "not assessed"
    entry.length += 1
    with pytest.raises(ValueError, match="extent"):
        V["checkpoint_inventory"](tmp_path, reader=reader)
    entry.length -= 1
    (target / "__0_1.distcp").write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="unreferenced"):
        V["checkpoint_inventory"](tmp_path, reader=reader)
