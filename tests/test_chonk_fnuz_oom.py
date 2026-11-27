"""Exercise the launch guard and durable memory observations with CPU fixtures."""

from contextlib import contextmanager
import json
import runpy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from apertus_amd import config, memory_telemetry, runs, runtime
from apertus_amd.cli import plan


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "experiments/chonk-native-fnuz-oom-i910bfcf936d3"
FIT = ROOT / "experiments/chonk-native-fnuz-fit-i910bfcf936d3"
CHUNKED_FIT = ROOT / "experiments/chonk-native-fnuz-chunked-fit-i910bfcf936d3"
REPLAY_FIT = ROOT / "experiments/chonk-native-fnuz-replay-fit-i910bfcf936d3"
REDUCED_FIT = ROOT / "experiments/chonk-reduced24-native-fnuz-dense-fit-i910bfcf936d3"
REDUCED_CACHED_FIT = ROOT / "experiments/chonk-reduced24-native-fnuz-cached-fit-i910bfcf936d3"
REPLAY_FLAG = "--moe-resident-fp8-fnuz-recompute-fc1"
REPLAY_FIELD = "moe_resident_fp8_fnuz_recompute_fc1"
REDUCED_OPTIONS = {
    "--num-layers": "24",
    "--pipeline-model-parallel-layout": "Ettt|(ttt|)*6,tttL",
    "--linear-attention-freq": "([1,1,1,0]*6)",
    "--moe-layer-freq": "([0]*3+[1]*21)",
}
REDUCED_ARGV = [item for pair in REDUCED_OPTIONS.items() for item in pair]
OBSERVER = runpy.run_path(ROOT / "experiments/validation/chonk_fnuz_oom.py")
FULL_PROBE_OPTIONS = dict(site=ROOT / "config.toml", submit=True, gpus_per_node=4, cpus_per_task=48,
                          cpu_bind="map_ldom:0,1,2,3", time="01:00:00", cache_mode="cold", exclude="nid002450")


def test_one_full_size_update_without_checkpoint_or_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected scheduler invocation"))
    fit, diagnostic = plan(FIT, **FULL_PROBE_OPTIONS), plan(DIAGNOSTIC, **FULL_PROBE_OPTIONS)
    assert all(fit[key] == diagnostic[key]
               for key in ("source", "image", "entrypoint", "prepare", "resources"))
    assert diagnostic["environment"] == {**fit["environment"], "TORCHINDUCTOR_COMPILE_THREADS": "4"}
    args = list(fit["args"])
    args[args.index("--train-iters") + 1] = "1"
    save = args.index("--save-interval")
    del args[save:save + 2]
    assert diagnostic["args"] == args
    assert diagnostic["procedure"]["memory_telemetry"] is True
    assert diagnostic["procedure"]["memory_mapping_ranks"] == [48, 51, 52, 55, 56, 59, 60, 63]
    assert diagnostic["procedure"]["timeout"] == fit["procedure"]["timeout"] == 2700
    assert diagnostic["phases"][0]["extra_args"] == []
    assert "/apertus/cache" not in runs.edf(diagnostic, tmp_path)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(diagnostic)


def test_chunked_fit_keeps_the_dense_attribution_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected scheduler invocation"))
    dense, chunked = plan(DIAGNOSTIC, **FULL_PROBE_OPTIONS), plan(CHUNKED_FIT, **FULL_PROBE_OPTIONS)
    assert chunked["experiment"] != dense["experiment"]
    assert chunked["args"] == [*dense["args"], "--lm-head-token-chunk-size", "512"]
    assert chunked["source"] == {**dense["source"], "commit": "aee1e9abab07e644653f71da98e35835059de10d"}
    assert all(chunked[key] == dense[key] for key in (
        "resources", "environment", "image", "image_path", "entrypoint", "prepare", "cache", "validation", "phases"
    ))
    assert {k: v for k, v in chunked["procedure"].items() if k != "launch_blocked"} == {
        k: v for k, v in dense["procedure"].items() if k != "launch_blocked"
    }
    assert chunked["phases"][0]["extra_args"] == []
    assert "/apertus/cache" not in runs.edf(chunked, tmp_path)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(chunked)


def test_replay_fit_keeps_the_chunked_fit_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected scheduler invocation"))
    chunked, replay = plan(CHUNKED_FIT, **FULL_PROBE_OPTIONS), plan(REPLAY_FIT, **FULL_PROBE_OPTIONS)
    assert replay["experiment"] != chunked["experiment"]
    assert replay["args"] == [*chunked["args"], REPLAY_FLAG]
    assert replay["source"] == {**chunked["source"], "commit": "ceb8035c182c40711db4d256d99fc44e4b8db27e"}
    assert all(replay[key] == chunked[key] for key in (
        "resources", "environment", "image", "image_path", "entrypoint", "prepare", "cache", "validation", "phases"
    ))
    assert {k: v for k, v in replay["procedure"].items() if k != "launch_blocked"} == {
        k: v for k, v in chunked["procedure"].items() if k != "launch_blocked"
    }
    assert "/apertus/cache" not in runs.edf(replay, tmp_path)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(replay)


def test_replay_selection_requires_exact_flag_and_effective_boolean():
    requested, verify = OBSERVER["replay_requested"], OBSERVER["verify_effective_replay"]
    assert not requested([])
    assert requested([REPLAY_FLAG])
    for argv in ([REPLAY_FLAG, REPLAY_FLAG], [REPLAY_FLAG + "=true"]):
        with pytest.raises(ValueError, match="FC1 replay flag"):
            requested(argv)
    old = SimpleNamespace()
    enabled = SimpleNamespace(**{REPLAY_FIELD: True})
    invalid = SimpleNamespace(**{REPLAY_FIELD: 1})
    verify(old, old, False)
    verify(enabled, enabled, True)
    for args, cfg, replay in ((old, enabled, True), (enabled, old, True), (enabled, enabled, False),
                              (invalid, enabled, True)):
        with pytest.raises(ValueError, match="effective FC1 replay"):
            verify(args, cfg, replay)


def test_reduced_plan_changes_only_depth_related_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected scheduler invocation"))
    dense, reduced = plan(DIAGNOSTIC, **FULL_PROBE_OPTIONS), plan(REDUCED_FIT, **FULL_PROBE_OPTIONS)
    expected = list(dense["args"])
    for flag, value in REDUCED_OPTIONS.items():
        expected[expected.index(flag) + 1] = value
    assert reduced["args"] == expected
    assert all(reduced[k] == dense[k] for k in (
        "source", "image", "image_path", "environment", "resources", "prepare", "entrypoint",
        "validation", "cache", "phases",
    ))
    assert {k: v for k, v in reduced["procedure"].items() if k != "launch_blocked"} == {
        k: v for k, v in dense["procedure"].items() if k != "launch_blocked"
    }
    assert "/apertus/cache" not in runs.edf(reduced, tmp_path)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(reduced)


def test_reduced_recipe_rejects_other_depths_and_memory_changes():
    depth = OBSERVER["diagnostic_depth"]
    assert depth([], 512, True) == 42
    assert depth(REDUCED_ARGV, 0, False) == 24
    for argv, chunk, replay in ((REDUCED_ARGV, 512, False), (REDUCED_ARGV, 0, True),
                                (REDUCED_ARGV[0:2], 0, False),
                                (["--num-layers", "25"], 0, False)):
        with pytest.raises(ValueError, match="reduced diagnostic"):
            depth(argv, chunk, replay)


def test_serial_compile_plan_only_changes_build_concurrency(tmp_path):
    full = plan(ROOT / "experiments/chonk-native-fnuz-timeout600-i910bfcf936d3",
                **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
    serial = plan(ROOT / "experiments/chonk-native-fnuz-compile1-i910bfcf936d3",
                  **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
    assert {k for k in serial if serial[k] != full[k]} == {
        "experiment", "environment", "runtime_environment", "procedure"
    }
    changes = {"TORCHINDUCTOR_COMPILE_THREADS": "1", "MAX_JOBS": "1",
               "CMAKE_BUILD_PARALLEL_LEVEL": "1", "MAKEFLAGS": "-j1"}
    assert serial["environment"] == {**full["environment"], **changes}
    assert serial["runtime_environment"] == {**full["runtime_environment"], "MAX_JOBS": "1"}
    assert serial["procedure"] == {**full["procedure"], "launch_blocked": serial["procedure"]["launch_blocked"]}
    assert all(f'{key} = "{value}"' in runs.edf(serial, tmp_path) for key, value in changes.items())
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(serial)


def test_full_ownership_plan_preserves_serial_compile_inputs():
    kwargs = {**FULL_PROBE_OPTIONS, "cache_image": True, "reservation": "cscs"}
    serial = plan(ROOT / "experiments/chonk-native-fnuz-compile1-i910bfcf936d3", **kwargs)
    trace = plan(ROOT / "experiments/chonk-native-fnuz-ownership-i910bfcf936d3", **kwargs)
    assert {key for key in trace if trace[key] != serial[key]} == {"experiment", "environment", "procedure"}
    assert trace["environment"] == {**serial["environment"], "APERTUS_CHONK_OWNERSHIP_TRACE": "1"}
    assert trace["procedure"] == {**serial["procedure"], "launch_blocked": trace["procedure"]["launch_blocked"],
                                 "scratch_telemetry": True}
    select = OBSERVER["ownership_trace_requested"]
    assert select(trace["environment"], 42, 512, True, True)
    for inputs in ((24, 512, True, True), (42, 0, True, True), (42, 512, False, True), (42, 512, True, False)):
        with pytest.raises(ValueError, match="full-depth serial-build recipe"):
            select(trace["environment"], *inputs)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(trace)


def test_kda_release_plan_preserves_full_ownership_inputs(tmp_path):
    kwargs = {**FULL_PROBE_OPTIONS, "cache_image": True, "reservation": "cscs"}
    old = plan(ROOT / "experiments/chonk-native-fnuz-ownership-i910bfcf936d3", **kwargs)
    new = plan(ROOT / "experiments/chonk-native-fnuz-kda-release-i910bfcf936d3", **kwargs)
    aux = OBSERVER["KDA_RELEASE"]
    assert {key for key in new if new[key] != old[key]} == {"experiment", "procedure"}
    assert new["procedure"] == {**old["procedure"], "launch_blocked": new["procedure"]["launch_blocked"],
                                "auxiliary_source": aux}
    assert f"{tmp_path}/auxiliary.py:/apertus/auxiliary.py:ro" in runs.edf(new, tmp_path)
    assert "/apertus/auxiliary.py" not in runs.edf(old, tmp_path)
    admit = OBSERVER["kda_release_requested"]
    assert admit(new, True)
    assert not admit(old, True)
    with pytest.raises(ValueError, match="unexpected KDA release"):
        admit(new, False)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(new)


def test_kda_release_finder_and_first_backward_are_exact(monkeypatch):
    finder = OBSERVER["KDAReleaseFinder"]()
    assert finder.find_spec("fla.ops.kda.chunk") is None
    assert finder.find_spec("fla.ops.kda.chunk_bwd").origin == "/apertus/auxiliary.py"
    chunk = SimpleNamespace()
    calls = []

    def original(value):
        assert chunk.chunk_kda_bwd is original
        calls.append(value)
        return value

    chunk.chunk_kda_bwd = original
    monkeypatch.setitem(sys.modules, "fla.ops.kda.chunk", chunk)
    rows = []
    observe = OBSERVER["observe_first_kda_release"]
    observe(lambda event, **data: rows.append((event, data)), {"commit": "test"})
    assert chunk.chunk_kda_bwd("first") == "first"
    assert chunk.chunk_kda_bwd("second") == "second"
    assert calls == ["first", "second"]
    assert rows == [("kda_release_backward_enter", {"commit": "test"}),
                    ("kda_release_backward_return", {"commit": "test"})]


def test_ownership_records_deduplicate_views_without_retaining_tensors():
    import weakref

    class Tensor:
        device = "cuda:0"
        dtype = "torch.bfloat16"

        def __init__(self, storage, offset):
            self.storage, self.offset = storage, offset

        def untyped_storage(self):
            return self.storage

        def storage_offset(self):
            return self.offset

    cuda = SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=19),
                           memory_stats=lambda: {"num_alloc_retries": 2, "num_ooms": 0})
    rows = []
    emit = lambda event, **values: rows.append({"event": event, **values})
    scope = {"call": 258, "forward": None, "backward": 248}
    drain = OBSERVER["BackwardDrainTrace"](emit, lambda: scope)
    observer = OBSERVER["DrainOwnership"](SimpleNamespace(Tensor=Tensor, cuda=cuda), emit, drain, 10)
    base = SimpleNamespace(_cdata=7, data_ptr=lambda: 100, nbytes=lambda: 64)
    x, view = Tensor(base, 0), Tensor(base, 2)
    refs = weakref.ref(x), weakref.ref(view)
    hook = observer.wrap_p2p(lambda **kwargs: {"recv_next": "the original work"})
    assert hook(tensor_send_prev=x, tensor_recv_next=view) == {"recv_next": "the original work"}
    assert not rows
    scope.update(call=259, backward=249, chunk=1)
    observer.record("call_begin")
    assert hook(tensor_send_prev=x, tensor_recv_next=view) == {"recv_next": "the original work"}
    observer.record("call_return", context=scope)
    del x, view
    assert all(ref() is None for ref in refs)
    s = rows[1]["storage"]
    assert s["sampled_storage_bytes"] == 64
    assert s["roles"] == {"tensor_send_prev": {"storages": 1, "bytes": 64},
                          "tensor_recv_next": {"storages": 1, "bytes": 64}}
    assert s["examples"]["tensor_recv_next"][0]["offset"] == 2
    assert rows[0]["allocator"]["num_alloc_retries"] == 2
    assert rows[0]["allocator"]["active_bytes.all.current"] is None
    json.dumps(rows)
    rows.append({"event": "schedule_exit", **scope})
    OBSERVER["verify_ownership"](rows, 10)
    with pytest.raises(ValueError, match="call_return"):
        OBSERVER["verify_ownership"]([r for r in rows if r.get("boundary") != "call_return"], 10)


def test_scratch_inventory_accounts_deleted_and_deduplicates_hard_links(tmp_path):
    root = tmp_path / "scratch"
    root.mkdir()
    cache = root / "cache-10"
    cache.mkdir()
    p = cache / "object"
    p.write_bytes(b"x" * 4096)
    (cache / "alias").hardlink_to(p)
    (root / "no-follow").symlink_to(tmp_path)
    dump = root / "core_123.gpu"
    dump.write_bytes(b"x" * 4096)
    live = dump.stat().st_blocks * 512
    with dump.open("rb") as descriptor:
        dump.unlink()
        # Use the real Linux descriptor scan to preserve its kernel deletion marker.
        row = memory_telemetry.scratch_inventory(root, [os.getpid()], include_tree=True)
        assert row["bytes_by_tag"] == {"rank_cache": p.stat().st_blocks * 512, "dump_like:deleted": live}
        assert any(x["path"] == dump.name and x["deleted"] for x in row["largest"])
        assert not descriptor.closed
    bounded = memory_telemetry.scratch_inventory(root, [], include_tree=True, limit=1)
    assert bounded["truncated"] and bounded["status"] == "partial"


def test_ownership_budget_keeps_call_endpoints_and_original_exception():
    rows = []
    emit = lambda event, **values: rows.append({"event": event, **values})
    scope = {"call": 261, "forward": None, "backward": 255, "chunk": 0}
    drain = OBSERVER["BackwardDrainTrace"](emit, lambda: scope)
    cuda = SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=0), memory_stats=dict)
    observer = OBSERVER["DrainOwnership"](SimpleNamespace(Tensor=object, cuda=cuda), emit, drain, 16)
    observer.MAX_EVENTS = 2
    observer.record("call_begin")
    observer.record("detail_first")
    observer.record("detail_limited")
    observer.record("call_return", context=scope)
    assert [r.get("boundary") for r in rows] == ["call_begin", "detail_first", None, "call_return"]
    assert rows[2]["event"] == "ownership_limit"

    def failed(*args, **kwargs):
        raise RuntimeError("original")

    observe = observer.wrap_layer(failed)
    layer = SimpleNamespace(layer=SimpleNamespace(layer_number=12))
    with pytest.raises(RuntimeError, match="original"):
        observe(None, layer)
    assert observer.layer is None


def test_reduced_drain_control_retains_full_recipe_except_depth_and_trace():
    full = plan(ROOT / "experiments/chonk-native-fnuz-timeout600-i910bfcf936d3",
                **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
    control = plan(ROOT / "experiments/chonk-reduced24-native-fnuz-drain-i910bfcf936d3",
                   **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
    assert {k for k in control if control[k] != full[k]} == {"experiment", "args", "environment", "procedure"}
    expected = list(full["args"])
    for flag, value in REDUCED_OPTIONS.items():
        expected[expected.index(flag) + 1] = value
    assert control["args"] == expected
    assert control["environment"] == {**full["environment"], "APERTUS_CHONK_REDUCED_DRAIN_TRACE": "1"}
    assert control["procedure"] == {**full["procedure"], "launch_blocked": control["procedure"]["launch_blocked"]}
    env = control["environment"]
    assert OBSERVER["reduced_drain_requested"](env)
    assert OBSERVER["diagnostic_depth"](expected, 512, True, True) == 24
    assert OBSERVER["notify_trace_requested"](env, 24, 512, True, True)
    with pytest.raises(ValueError, match="unexpected reduced diagnostic"):
        OBSERVER["diagnostic_depth"](full["args"], 512, True, True)
    with pytest.raises(ValueError, match="unexpected reduced diagnostic"):
        OBSERVER["diagnostic_depth"](expected, 0, False, True)
    with pytest.raises(ValueError, match="requires the 600-second"):
        OBSERVER["reduced_drain_requested"]({**env, "UCCL_EP_CPU_TIMEOUT_SECS": "100"})
    assert not OBSERVER["ck_training_mapping_required"](0, True)
    assert OBSERVER["ck_training_mapping_required"](16, True)
    assert OBSERVER["ck_training_mapping_required"](0, False)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(control)


@pytest.mark.parametrize("fails", [False, True])
def test_backward_drain_observer_scopes_tail_and_preserves_original(fails):
    from contextlib import nullcontext

    rows = []
    active = dict(call=258, forward=None, backward=248)
    cls = OBSERVER["BackwardDrainTrace"]
    observer = cls(lambda e, **kw: rows.append(dict(event=e, **kw)), lambda: active)
    node = SimpleNamespace(name="mlp")
    backward = observer.wrap_context(lambda obj, _: nullcontext(obj))
    dw = observer.wrap_wgrad(lambda obj: obj)
    with backward(node, "mlp backward") as value:
        assert value is node
    assert dw(node) is node and not rows
    active.update(call=259, backward=249)
    with backward(node, "mlp backward") as value:
        assert value is node
        assert dw(node) is node
    rows.append(dict(event="schedule_exit", **active))
    for phase in cls.PHASES:
        def original():
            if fails and phase == "optimizer_step":
                raise RuntimeError("native optimizer fault")
            return node
        wrapped = observer.wrap_phase(original, phase)
        if fails and phase == "optimizer_step":
            with pytest.raises(RuntimeError, match="native optimizer fault"):
                wrapped()
        else:
            assert wrapped() is node
    if fails:
        assert rows[-1]["returned"] is False
        with pytest.raises(ValueError, match="invalid drain trace span"):
            OBSERVER["verify_backward_drain"](rows, 16)
    else:
        OBSERVER["verify_backward_drain"](rows, 16)


def test_cached_reduced_fit_preserves_recipe_and_deadlines(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected scheduler invocation"))
    old = plan(REDUCED_FIT, **FULL_PROBE_OPTIONS)
    cached = plan(REDUCED_CACHED_FIT, **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
    assert old["experiment"] != cached["experiment"]
    assert all(cached[k] == old[k] for k in (
        "args", "entrypoint", "prepare", "source", "image", "environment", "validation", "cache", "phases",
    ))
    assert cached["source"]["commit"] == "4b21bb5781ee6db66952a9b6ab128b688c7dc1d1"
    assert cached["resources"] == {**old["resources"], "reservation": "cscs"}
    assert cached["procedure"] == {**old["procedure"], "startup_trace": True,
                                   "launch_blocked": cached["procedure"]["launch_blocked"]}
    assert "gpu_identity_timeout" not in cached["procedure"]
    assert cached["registry_image_path"] == old["image_path"]
    assert Path(cached["image_path"]).name == "image.sqsh"
    assert cached["step"][-1] == "--startup-trace"
    frozen = {**cached, "id": "fixture", "source_snapshot": {"directory": "/signed"}}
    assert "#SBATCH --exclusive\n" in runs.batch_script(frozen, tmp_path)
    assert ":/apertus/cache" not in runs.edf(frozen, tmp_path)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(cached)


def test_effective_reduced_model_masks_layout_and_loss():
    verify = OBSERVER["verify_reduced_model"]
    good = SimpleNamespace(
        linear_attention_freq=[1, 1, 1, 0] * 6,
        moe_layer_freq=[0] * 3 + [1] * 21,
        pipeline_model_parallel_layout=REDUCED_OPTIONS["--pipeline-model-parallel-layout"],
        cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="native",
    )
    verify(good, good)
    for key, value in (("moe_layer_freq", [1] * 24), ("linear_attention_freq", 4),
                       ("pipeline_model_parallel_layout", "Et"), ("cross_entropy_fusion_impl", "te")):
        with pytest.raises(ValueError, match="effective reduced-model"):
            verify(good, SimpleNamespace(**{**vars(good), key: value}))


@pytest.mark.parametrize("argv,expected", [
    ([], 0), (["--lm-head-token-chunk-size", "512"], 512), (["--lm-head-token-chunk-size=512"], 512),
    (["--lm-head-token-chunk-size", "256"], None), (["--lm-head-token-chunk-size"], None),
    (["--lm-head-token-chunk-size=512", "--lm-head-token-chunk-size=0"], None),
])
def test_head_chunk_selection(argv, expected):
    if expected is None:
        with pytest.raises(ValueError, match="chunk"):
            OBSERVER["head_chunk_size"](argv)
    else:
        assert OBSERVER["head_chunk_size"](argv) == expected


def test_effective_head_must_match_invocation_and_native_ce():
    verify = OBSERVER["verify_effective_head"]
    old = SimpleNamespace()
    native = SimpleNamespace(lm_head_token_chunk_size=512, cross_entropy_loss_fusion=True,
                             cross_entropy_fusion_impl="native")
    verify(old, old, 0)
    verify(native, native, 512)
    for args, cfg, cap in ((old, native, 512), (native, old, 512), (native, native, 0)):
        with pytest.raises(ValueError, match="expected LM-head chunk size"):
            verify(args, cfg, cap)
    wrong = SimpleNamespace(**{**vars(native), "cross_entropy_fusion_impl": "te"})
    with pytest.raises(ValueError, match="requires the native fused CE"):
        verify(native, wrong, 512)


def head_rows(chunked=True):
    def row(event, **metadata):
        return dict(event=event, call=5, forward=4, backward=None, **metadata)

    rows = [row("output_head_enter")]
    tokens = 512 if chunked else 8192
    for _ in range(8192 // tokens):
        rows += [
            row("ce_enter", shape=[tokens // 2, 2, 200064], dtype="torch.bfloat16", bytes=tokens * 200064 * 2),
            row("ce_fp32_logits_exit", shape=[tokens // 2, 2, 200064], dtype="torch.float32", bytes=tokens * 200064 * 4),
            row("ce_exit"),
        ]
    return [*rows, row("output_head_exit")]


@pytest.mark.parametrize("defect", [
    None, "dense", "missing", "extra", "wrong-order", "wrong-call", "wrong-backward",
    "wrong-vocab", "wrong-dtype", "wrong-bytes", "missing-head-exit", "duplicate-head-enter",
])
def test_first_head_trace_requires_exact_chunks(defect):
    rows = head_rows()
    if defect == "dense":
        del rows[4:-1]
        rows[1].update(shape=[4096, 2, 200064], bytes=3277848576)
        rows[2].update(shape=[4096, 2, 200064], bytes=6555697152)
    elif defect == "missing":
        del rows[1:4]
    elif defect == "extra":
        rows[1:1] = rows[1:4]
    elif defect == "wrong-order":
        rows[1], rows[2] = rows[2], rows[1]
    elif defect == "wrong-call":
        rows[2]["call"] = 6
    elif defect == "wrong-backward":
        rows[2]["backward"] = 0
    elif defect == "wrong-vocab":
        rows[2]["shape"][2] -= 1
    elif defect == "wrong-dtype":
        rows[2]["dtype"] = "torch.bfloat16"
    elif defect == "wrong-bytes":
        rows[2]["bytes"] -= 1
    elif defect == "missing-head-exit":
        rows.pop()
    elif defect == "duplicate-head-enter":
        rows.insert(0, rows[0])
    verify = OBSERVER["verify_head_trace"]
    if defect:
        with pytest.raises(ValueError, match="LM-head"):
            verify(rows, 512)
    else:
        rows += [{**row, "call": 6} for row in rows]
        verify(rows, 512)


@pytest.mark.parametrize("mode", [
    "dense", "chunked", "missing-chunk-rank", "missing-effective-size",
    "replay", "missing-replay-rank", "missing-replay-flag", "missing-pressure-exit", "missing-replay-backward",
    "reduced", "missing-reduced-depth", "missing-dense-head",
])
def test_frozen_argv_controls_all_rank_fit_checks(tmp_path, mode):
    output = tmp_path / "output"
    output.mkdir()
    reduced = mode in {"reduced", "missing-reduced-depth", "missing-dense-head"}
    argv = list(REDUCED_ARGV) if reduced else ([] if mode == "dense" else ["--lm-head-token-chunk-size", "512"])
    replay = mode in {"replay", "missing-replay-rank", "missing-replay-flag",
                      "missing-pressure-exit", "missing-replay-backward"}
    if replay:
        argv.append(REPLAY_FLAG)
    (tmp_path / "resolved.json").write_text(json.dumps({
        "args": argv, "procedure": {"memory_mapping_ranks": [48]}
    }))
    for rank in range(64):
        enter = {} if mode == "dense" or (mode == "missing-effective-size" and rank == 1) else {
            "lm_head_token_chunk_size": 512
        }
        if replay and not (mode == "missing-replay-flag" and rank == 5):
            enter[REPLAY_FIELD] = True
        if reduced:
            enter = dict(num_layers=42 if mode == "missing-reduced-depth" and rank == 6 else 24,
                         lm_head_token_chunk_size=0, **{REPLAY_FIELD: False})
        evidence = [
            {"event": "entry"}, {"event": "compiler_config", "inductor_compile_threads": 4},
            {"event": "update_enter", "inductor_compile_threads": 4, **enter},
        ]
        if rank >= 48:
            if reduced:
                evidence += head_rows(chunked=(mode == "missing-dense-head" and rank == 63))
            else:
                evidence += head_rows() if argv and not (mode == "missing-chunk-rank" and rank == 63) else [
                    {"event": "ce_fp32_logits_exit"}
                ]
        if replay and 16 <= rank < 32 and not (mode == "missing-pressure-exit" and rank == 20):
            evidence.append(dict(event="schedule_exit", call=9, forward=8, backward=None))
        if replay and not (mode == "missing-replay-rank" and rank == 63):
            backward = None if mode == "missing-replay-backward" and rank == 19 else 0
            evidence.extend(dict(event=event, call=12, forward=11, backward=backward)
                            for event in ("fc1_replay_enter", "fc1_replay_exit"))
        evidence += [{"event": "update_exit"}, {"event": "completed"}]
        (output / f"gpu-progress-rank-{rank}.jsonl").write_text(
            "".join(json.dumps(dict(rank=rank, **row)) + "\n" for row in evidence)
        )
        host = {"event": "exit:0", "node_memory": {"MemAvailable": 100}}
        if rank == 48:
            host["vm_mapping"] = {"status": "sampled", "mappings": [{"name": "/dev/zero (deleted)"}]}
        (output / f"host-memory-rank-{rank}.jsonl").write_text(json.dumps(host) + "\n")
        (output / f"imports-{rank}.json").write_text('{"verified_modules": 1}')
        (output / f"rank-{rank}.log").write_text("UCCL_EP_TRANSPORT=cxi initialized\n")
    if mode.startswith("missing"):
        match = {
            "missing-chunk-rank": "rank 63: unexpected first LM-head",
            "missing-effective-size": "rank 1 missing effective",
            "missing-replay-rank": "rank 63 missing FC1 replay",
            "missing-replay-flag": "rank 5 missing effective FC1 replay",
            "missing-pressure-exit": "rank 20 did not clear forward warmup call 9",
            "missing-replay-backward": "rank 19 missing FC1 replay in backward",
            "missing-reduced-depth": "rank 6 missing effective reduced dense-head recipe",
            "missing-dense-head": "rank 63: unexpected first LM-head",
        }[mode]
        with pytest.raises(ValueError, match=match):
            OBSERVER["check"](tmp_path)
    else:
        OBSERVER["check"](tmp_path)
    result = json.loads((output / "checks.json").read_text())
    assert result["status"] == ("failed" if mode.startswith("missing") else "passed")
    if mode in ("chunked", "replay"):
        assert result["lm_head_verified_ranks"] == list(range(48, 64))
        assert result["lm_head_chunks_per_rank"] == 16
        assert result["observed_mapping_ranks"] == [48]
    else:
        assert "lm_head_verified_ranks" not in result
    if mode == "replay":
        assert result["fc1_replay_verified_ranks"] == list(range(64))
        assert result["pp_rank_1_verified_ranks"] == list(range(16, 32))
    else:
        assert "fc1_replay_verified_ranks" not in result
    if mode == "reduced":
        assert result["num_layers"] == 24 and result["lm_head"] == "dense"
        assert result["fc1_replay"] is False and "not full Chonk" in result["scope"]


@pytest.mark.parametrize("requested,effective", [("4", 4), ("4", 32), ("32", 4), (None, 4)])
def test_effective_compile_limit_is_required(monkeypatch, requested, effective):
    if requested is None:
        monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    else:
        monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", requested)
    if (requested, effective) == ("4", 4):
        OBSERVER["verify_compile_threads"](effective)
    else:
        with pytest.raises(ValueError, match="expected 4 Inductor compile threads"):
            OBSERVER["verify_compile_threads"](effective)


@pytest.mark.parametrize("missing", [None, "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "MAKEFLAGS"])
def test_serial_compile_profile_checks_all_controls(monkeypatch, missing):
    serial = OBSERVER["SERIAL_BUILD_ENV"]
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    for key, value in serial.items():
        monkeypatch.setenv(key, value)
    verify = OBSERVER["verify_compile_threads"]
    if missing is not None:
        monkeypatch.delenv(missing)
        with pytest.raises(ValueError, match="full serial build environment"):
            verify(1)
    else:
        assert verify(1) == {"serial_build_environment": serial}
        with pytest.raises(ValueError, match="expected 1 Inductor compile threads"):
            verify(4)


@pytest.mark.parametrize("rank_list", ["[48, 48]", "[-1]", "[64]", "[true]", "[]"])
def test_mapping_rank_contract(tmp_path, rank_list):
    trial = tmp_path / "oom-fixture"
    trial.mkdir()
    text = (DIAGNOSTIC / "experiment.toml").read_text()
    text = text.replace("[48, 51, 52, 55, 56, 59, 60, 63]", rank_list)
    text = text.replace("../validation/chonk_fnuz_oom.py", str(ROOT / "experiments/validation/chonk_fnuz_oom.py"))
    (trial / "experiment.toml").write_text(text)
    with pytest.raises(ValueError, match="memory_mapping_ranks"):
        config.resolve(trial, site=ROOT / "config.toml", submit=True)


def test_boundary_records_entry_even_when_native_call_fails():
    events = []

    def observe(event, **info):
        events.append((event, info))

    def native(x):
        if not x:
            raise RuntimeError("native failure")
        return x

    active = None
    wrapped = OBSERVER["trace_boundary"](native, "ce", observe, lambda: active,
                                         lambda x: {"input": x}, lambda x: {"result": x})
    assert wrapped(5) == 5 and not events
    active = {"call": 5}
    assert wrapped(7) == 7
    with pytest.raises(RuntimeError, match="native failure"):
        wrapped(0)
    assert events == [("ce_enter", {"call": 5, "input": 7}), ("ce_exit", {"call": 5, "result": 7}),
                      ("ce_enter", {"call": 5, "input": 0})]


@pytest.mark.parametrize("outcome", ["return", "timeout", "before-target"])
@pytest.mark.parametrize("prefix", [False, True])
def test_dependency_trace_preserves_waits_and_bounds_capture(outcome, prefix):
    calls, rows = [], []
    comp, comm, uccl = (SimpleNamespace(cuda_stream=i) for i in (0, 1, 2))
    context = dict(call=8, forward=7, backward=2)
    error = RuntimeError("native timeout")

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is False

        def record(self, stream):
            self.i = len(calls)
            self.stream = stream
            calls.append(("record", self.i))

        def query(self):
            calls.append(("query", self.i))
            return self.stream is comp

    cuda = SimpleNamespace(Event=Event, current_stream=lambda: comp)
    trace = OBSERVER["DependencyTrace"](cuda, lambda e, **r: rows.append(dict(event=e, **r)), lambda: comm)

    @contextmanager
    def actual_wait(node, name):
        calls.append(("original wait", name))
        try:
            yield node.stream
        finally:
            calls.append(("original release", name))

    acquire = trace.wrap_context(actual_wait)
    exchange = trace.wrap_buffer_dispatch(lambda _buffer: "native result")
    buffer = SimpleNamespace(get_comm_stream=lambda: uccl)

    def target():
        if outcome == "timeout":
            raise error
        return "native result"

    dispatch = trace.wrap_target(target)
    trace.begin(dict(call=7, forward=6, backward=1))
    trace.finish("before_target")
    assert not calls and not rows
    trace.begin(context)
    if prefix:
        autograd = SimpleNamespace(backward=lambda _: None)
        recv_prev_wait_handles = [object()]

        def receive():
            recv_prev_wait_handles.pop(0)

        def combined(forward, *, pre_forward):
            autograd.backward(forward())
            pre_forward()

        trace.wrap_prefix_step(combined, autograd)(lambda: None, pre_forward=receive)
    for name in OBSERVER["DEPENDENCY_NODES"]:
        stream = comm if "combine" in name or name == "moe_dispatch forward" else comp
        if name == "moe_dispatch forward" and outcome == "before-target":
            trace.finish("before_target")
            break
        with acquire(SimpleNamespace(stream=stream), name) as selected:
            assert selected is stream
            if name == "moe_combine backward":
                assert exchange(buffer) == "native result"
            if name == "moe_dispatch forward":
                if outcome == "timeout":
                    with pytest.raises(RuntimeError, match="native timeout") as caught:
                        dispatch()
                    assert caught.value is error
                else:
                    assert dispatch() == "native result"
    [summary] = [row for row in rows if row["event"] == "dependency_observation"]
    assert summary["boundary"] == {
        "return": "target_return", "timeout": "target_exception", "before-target": "before_target",
    }[outcome]
    assert summary["query_errors"] == summary["record_errors"] == {}
    assert [i for op, i in calls if op == "query"] == list(reversed([i for op, i in calls if op == "record"]))
    expected = OBSERVER["DEPENDENCY_NODES"][:len(OBSERVER["DEPENDENCY_NODES"]) - (outcome == "before-target")]
    assert tuple(name for op, name in calls if op == "original wait") == expected
    assert tuple(name for op, name in calls if op == "original release") == expected
    if outcome == "return":
        OBSERVER["verify_dependency_trace"](rows, 48, prefix)
    else:
        with pytest.raises(ValueError, match="unusable dependency"):
            OBSERVER["verify_dependency_trace"](rows, 48, prefix)
    trace.begin(context)
    trace.finish("before_target")
    assert len(rows) == (7 if prefix else 1) and not trace.markers


def test_dependency_trace_requires_notify_trace():
    requested = OBSERVER["dependency_trace_requested"]
    flag = {"APERTUS_CHONK_DEPENDENCY_TRACE": "1"}
    assert not requested({}, True)
    assert requested(flag, True)
    with pytest.raises(ValueError, match="requires the full-size"):
        requested(flag, False)


@pytest.mark.parametrize("outcome", ["consumed", "empty", "error", "no-callback"])
def test_prefix_trace_keeps_native_callbacks_and_restores_autograd(outcome):
    stream = SimpleNamespace(cuda_stream=0)
    rows, original_calls = [], []
    error = RuntimeError("receive error")
    recv_prev_wait_handles = [object()] if outcome in ("consumed", "error") else []

    class Event:
        def __init__(self, *, enable_timing):
            assert not enable_timing

        def record(self, selected):
            assert selected is stream

        def query(self):
            return True

    def native_backward(value):
        original_calls.append(("backward", value))

    autograd = SimpleNamespace(backward=native_backward)
    cuda = SimpleNamespace(Event=Event, current_stream=lambda: stream)
    trace = OBSERVER["DependencyTrace"](cuda, lambda e, **r: rows.append(dict(event=e, **r)), lambda: stream)

    def receive(vp_stage):
        original_calls.append(("receive", vp_stage))
        if recv_prev_wait_handles:
            recv_prev_wait_handles.pop(0)
        if outcome == "error":
            raise error

    def combined(forward, *, pre_forward=None):
        autograd.backward(forward())
        assert autograd.backward is native_backward
        if pre_forward:
            pre_forward(1)
        autograd.backward("later")
        return "result"

    run = trace.wrap_prefix_step(combined, autograd)
    trace.begin(dict(call=8, forward=7, backward=2))
    if outcome == "error":
        with pytest.raises(RuntimeError, match="receive error") as caught:
            run(lambda: "plan", pre_forward=receive)
        assert caught.value is error
    else:
        assert run(lambda: "plan", pre_forward=None if outcome == "no-callback" else receive) == "result"
    assert autograd.backward is native_backward and not recv_prev_wait_handles
    trace.finish("before_target")
    host = rows[:-1]
    pairs = [(name, phase) for name in OBSERVER["PREFIX_BOUNDARIES"] for phase in ("before", "after")]
    count = {"no-callback": 4, "error": 5}.get(outcome, 6)
    assert [(r["boundary"], r["phase"]) for r in host] == pairs[:count]
    expected = {"consumed": [1, 0], "empty": [0, 0], "error": [1], "no-callback": []}[outcome]
    assert [r["queued"] for r in host if "queued" in r] == expected
    assert original_calls.count(("backward", "plan")) == 1
    assert original_calls.count(("receive", 1)) == (outcome != "no-callback")


def test_prefix_trace_requires_dependency_trace():
    flag = {"APERTUS_CHONK_PREFIX_TRACE": "1"}
    requested = OBSERVER["prefix_trace_requested"]
    assert requested(flag, True) and not requested({}, True)
    with pytest.raises(ValueError, match="requires the call-8"):
        requested(flag, False)


@pytest.mark.parametrize("receiver", [False, True])
def test_pair_trace_preserves_native_work_and_matches_context(receiver, monkeypatch):
    rank = 48 if receiver else 32
    rows, calls = [], []
    context = dict(call=7, forward=6, backward=1) if receiver else dict(call=8, forward=7, backward=0)
    stream = SimpleNamespace(cuda_stream=1)
    work = object()
    tensor = SimpleNamespace(shape=(4096, 2, 3584), dtype="torch.bfloat16")

    class Event:
        def __init__(self, *, enable_timing):
            assert not enable_timing

        def record(self, selected):
            assert selected is stream

        def query(self):
            return False

    cuda = SimpleNamespace(Event=Event, current_stream=lambda: stream, current_device=lambda: 0)
    emit = lambda event, **fields: rows.append(dict(event=event, **fields))
    deps = OBSERVER["DependencyTrace"](cuda, emit, lambda: stream)
    cls = OBSERVER["P2PPairTrace"]
    pair = cls(cuda, emit, lambda: context, rank, lambda: stream, deps)
    # The sender samples synchronously at isend return as well as periodically.
    monkeypatch.setitem(cls.begin.__globals__, "Thread", lambda **kw: SimpleNamespace(start=lambda: None))

    def native_post(**kwargs):
        calls.append(kwargs)
        return work

    native = pair.wrap_native_post(native_post)
    # Unrelated posts must not appear in the trace.
    assert native(tensor=tensor, **({"src": 32} if receiver else {"dst": 48})) is work
    assert rows == []
    pair.begin()
    if not receiver:
        pair.wrap_backward_step(lambda **kw: kw["pre_backward"]())(pre_backward=lambda: None)
        pair.wrap_chunk_wait(lambda plan: "same")(object())
    post = pair.wrap_forward_post(lambda: native(tensor=tensor, **({"src": 32} if receiver else {"dst": 48})))
    assert post() is work
    if receiver:
        context = dict(call=8, forward=7, backward=2)
        deps.begin(context)
        recv_prev_wait_handles = [work]
        cleaned = []
        deallocate = pair.wrap_deallocation(lambda: cleaned.append(True))

        def receive():
            assert recv_prev_wait_handles.pop(0) is work
            deallocate()

        pair.wrap_receiver_wait(receive)()
        assert cleaned == [True]
        assert "receiver_wait_return" in deps.seen
    assert len(calls) == 2
    OBSERVER["verify_pair_trace"](rows, rank)
    if receiver:
        next(r for r in rows if r["event"] == "p2p_pair_wait_enter")["work_id"] = id(tensor)
    else:
        next(r for r in rows if r["event"] == "p2p_pair_sender_sample")["markers"].pop("sender_isend_ready")
    with pytest.raises(ValueError, match="missing"):
        OBSERVER["verify_pair_trace"](rows, rank)


def test_pair_trace_requires_prefix_trace():
    flag = {"APERTUS_CHONK_P2P_PAIR_TRACE": "1"}
    requested = OBSERVER["pair_trace_requested"]
    assert requested(flag, True) and not requested({}, True)
    with pytest.raises(ValueError, match="requires the call-8 prefix"):
        requested(flag, False)


def test_later_trace_requires_node_verified_prewarm_and_guard():
    p = plan(ROOT / "experiments/chonk-native-fnuz-later-p2p-trace-i910bfcf936d3",
             **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
    check = OBSERVER["later_trace_requested"]
    assert check(p["environment"], True) and not check({}, True)
    with pytest.raises(ValueError, match="requires node-verified CK prewarm"):
        check(p["environment"], False)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(p)


def test_later_phase_trace_requires_later_p2p_and_guard():
    p = plan(ROOT / "experiments/chonk-native-fnuz-later-phase-trace-i910bfcf936d3",
             **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
    check = OBSERVER["later_phase_requested"]
    assert check(p["environment"], True) and not check({}, True)
    with pytest.raises(ValueError, match="requires the matched later P2P"):
        check(p["environment"], False)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(p)


def test_reverse_ready_plans_require_the_signed_handoff():
    full = plan(ROOT / "experiments/chonk-native-fnuz-reverse-ready-i910bfcf936d3",
                **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
    with pytest.raises(ValueError, match="requires the later-phase"):
        OBSERVER["reverse_ready_requested"](full["environment"], False)
    assert OBSERVER["reverse_ready_requested"](full["environment"], True)
    ep = plan(ROOT / "experiments/chonk-native-fnuz-reverse-ready-ep16-i910bfcf936d3",
              **{**FULL_PROBE_OPTIONS, "time": "00:25:00"}, cache_image=True, reservation="cscs")
    assert ep["args"] == ["ep16-ready-order"] and ep["resources"]["nodes"] == 4
    assert full["source"]["commit"] == ep["source"]["commit"]
    for p in (ep, full):
        assert ":/apertus/cache" not in runs.edf(p, Path("/run"))
        with pytest.raises(ValueError, match="launch blocked"):
            runs.run(p)


def test_stall_trace_plan_changes_only_diagnostic_flag_and_metadata():
    previous, current = [plan(ROOT / f"experiments/chonk-native-fnuz-{name}-i910bfcf936d3",
                              **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
                         for name in ("reverse-ready", "stall-trace")]
    assert {k for k in current if current[k] != previous[k]} == {"experiment", "environment", "procedure"}
    assert current["environment"] == {**previous["environment"], "APERTUS_CHONK_STALL_TRACE": "1"}
    assert current["procedure"] == {**previous["procedure"], "launch_blocked": current["procedure"]["launch_blocked"]}
    assert current["source"]["commit"] == OBSERVER["STALL_SOURCE"]
    requested = OBSERVER["stall_trace_requested"]
    assert not requested(previous["environment"], True)
    assert requested(current["environment"], True)
    with pytest.raises(ValueError, match="requires the reverse-readiness"):
        requested(current["environment"], False)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(current)


@pytest.fixture
def stall_fixture():
    rows, streams = [], []

    class Event:
        def record(self, stream):
            streams.append(stream.cuda_stream)

        def query(self):
            return False

    compute = SimpleNamespace(cuda_stream=7)
    comm = SimpleNamespace(cuda_stream=8)
    cuda = SimpleNamespace(current_stream=lambda: compute, Event=lambda **kw: Event())
    return cuda, compute, comm, rows, streams


def test_stall_trace_sender_scopes_autograd_and_pre_launch_inputs(stall_fixture):
    cuda, compute, _, rows, streams = stall_fixture
    context = dict(call=10, forward=9, backward=0)
    trace = OBSERVER["StallTrace"](cuda, lambda e, **fields: rows.append(dict(event=e, **fields)),
                                   lambda: context, (10, 9, 0))
    node = SimpleNamespace(stream=compute)
    expert = SimpleNamespace(_backend=SimpleNamespace(row=object()))

    @OBSERVER["contextmanager"]
    def context_manager(node, name):
        yield node

    def phase(label, sentinel):
        return trace.wrap_host(lambda: sentinel, label, lambda: trace.row_id is not None)

    calls = [phase(label, object()) for label in ("weights", "padding", "destinations", "row_quantize")]
    launch = trace.wrap_sender_gemm(lambda *args: "gemm")

    def native_backward(expert):
        for fn in calls:
            fn()
        assert launch(expert._backend, "fc2_dgrad", [], [], [object()]) == "gemm"
        assert trace.row_id is None
        calls[-1]()  # Post-FC2 quantization is not part of the observation.
        return expert

    resident = trace.wrap_resident(native_backward)
    backward = trace.wrap_host(lambda: resident(expert), "mlp_autograd", lambda: trace.in_mlp)
    acquire = trace.wrap_mlp_context(context_manager)
    with acquire(node, "attn backward"):
        assert not rows
    with acquire(node, "mlp backward") as seen:
        assert seen is node and backward() is expert
    count = len(rows)
    with acquire(node, "mlp backward"):
        assert backward() is expert
    assert len(rows) == count and streams == [7, 7]
    assert sum(r.get("boundary") == "row_quantize" for r in rows) == 2
    OBSERVER["verify_stall_trace"](rows, 16)


@pytest.mark.parametrize("fail", [False, True])
def test_stall_trace_reverse_preserves_native_objects_and_samples_after_forward(fail, stall_fixture):
    cuda, _, comm, rows, streams = stall_fixture
    context = dict(call=9, forward=8, backward=3)
    trace = OBSERVER["StallTrace"](cuda, lambda e, **fields: rows.append(dict(event=e, **fields)),
                                   lambda: context, (9, 8, 3))
    buffer = SimpleNamespace(get_comm_stream=lambda: comm)
    tensor, handle, ready, output = [object() for _ in range(4)]
    capture = trace.wrap_capture(lambda: ready)
    stream_ptr = trace.wrap_native_args(lambda b, device: 7)
    waited = []
    wait = trace.wrap_existing_wait(waited.append)

    def native(b, x, h, *, previous_event):
        assert b is buffer and x is tensor and h is handle and previous_event is ready
        assert stream_ptr(b, "device") == 7
        return output

    dispatch = trace.wrap_internode(native)

    def reverse(ctx):
        event = dispatch(buffer, tensor, handle, previous_event=capture())
        wait(event)
        return event

    run = trace.wrap_reverse(reverse, SimpleNamespace(node="moe_combine backward"))
    assert run(None) is output and waited == [output]
    assert not trace.in_reverse and not trace.internode

    def forward():
        if fail:
            raise RuntimeError("original failure")
        return output

    observed = trace.wrap_final_dispatch(forward)
    if fail:
        with pytest.raises(RuntimeError, match="original failure"):
            observed()
    else:
        assert observed() is output
        assert observed() is output
    assert streams == [7, 8, 8, 8, 7, 7]
    OBSERVER["verify_stall_trace"](rows, 48)
    with pytest.raises(ValueError, match="missing stall host"):
        OBSERVER["verify_stall_trace"]([r for r in rows if r["boundary"] != "capture_ready_event"], 48)


@pytest.mark.parametrize("fail", [False, True])
def test_autograd_handoff_scopes_real_node_hooks_and_removes_them(fail, stall_fixture):
    cuda, _, _, rows, streams = stall_fixture
    context = dict(call=10, forward=9, backward=0)
    stall = OBSERVER["StallTrace"](cuda, lambda *_: None, lambda: context, (10, 9, 0))
    stall.in_mlp = True
    trace = OBSERVER["GraphHandoffTrace"](cuda, lambda e, **kw: rows.append(dict(event=e, **kw)), stall)

    class Node:
        def __init__(self, name, *children):
            self.name = lambda: name
            self.next_functions = [(child, 0) for child in children]
            self.pre, self.post = [], []

        @staticmethod
        def attach(hooks, callback):
            hooks.append(callback)
            return SimpleNamespace(remove=lambda: hooks.remove(callback))

        def register_prehook(self, fn):
            return self.attach(self.pre, fn)

        def register_hook(self, fn):
            return self.attach(self.post, fn)

    resident = Node(trace.NODES["resident"])
    unpermute = Node(trace.NODES["unpermute"], resident)
    outputs = (SimpleNamespace(grad_fn=Node("MakeViewlessTensorBackward", unpermute)),)
    grad = object()
    triton = trace.wrap_permute(lambda x: x)

    def original(node, out, dy):
        assert out is outputs and dy is grad
        assert unpermute.pre[0]((grad,)) is None
        assert triton(grad) is grad
        if fail:
            raise RuntimeError("original backward failed")
        assert unpermute.post[0]((grad,), (grad,)) is None
        assert resident.pre[0]((grad,)) is None
        return grad

    observed = trace.wrap_autograd(original)
    if fail:
        with pytest.raises(RuntimeError, match="original backward failed"):
            observed(SimpleNamespace(name="mlp"), outputs, grad)
        with pytest.raises(ValueError, match="missing native autograd handoff"):
            OBSERVER["verify_handoff_trace"](rows, 16)
    else:
        assert observed(SimpleNamespace(name="mlp"), outputs, grad) is grad
        assert streams == [7] * 5
        OBSERVER["verify_handoff_trace"](rows, 16)
    assert not any((trace.in_unpermute, unpermute.pre, unpermute.post, resident.pre))


def test_call8_phase_trace_retains_node_and_existing_chunk_wait_boundaries():
    context = dict(call=8, forward=7, backward=0)
    rows = []
    handler = SimpleNamespace(**{f"{op}_{phase}_callback": lambda _: None
                                 for op in ("register", "remove") for phase in ("start", "end")})
    trace = OBSERVER["SenderPhaseTrace"](lambda e, **kw: rows.append(dict(event=e, **kw)), handler,
                                         (8, 7, 0), live_nodes=True)
    marker = object()
    kwargs = {name: (lambda: marker) for name in OBSERVER["SENDER_CALLBACKS"]}

    @OBSERVER["contextmanager"]
    def original(node, name):
        yield node

    acquire = trace.wrap_context(original)
    wait = trace.wrap_chunk_wait(lambda value: value, lambda: 8, lambda: 8)
    with trace.capture(context, kwargs):
        assert kwargs["pre_forward"]() is marker
        assert kwargs["pre_backward"]() is marker
        with acquire(marker, "mlp backward") as value:
            assert value is marker
        assert wait(marker) is marker
        assert kwargs["post_forward"]() is marker
        assert kwargs["post_backward"]() is marker
    OBSERVER["verify_sender_phases"](rows, 32, (8, 7, 0), live_nodes=True)
    with pytest.raises(ValueError, match="missing call-8 sender node evidence"):
        OBSERVER["verify_sender_phases"](
            [r for r in rows if r["event"] != "later_sender_node"], 32, (8, 7, 0), live_nodes=True)


def test_handoff_launch_has_only_trace_delta_and_is_guarded():
    old, new = [plan(ROOT / f"experiments/chonk-native-fnuz-{n}-trace-i910bfcf936d3",
                     **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs") for n in ("stall", "handoff")]
    assert {k for k in new if new[k] != old[k]} == {"experiment", "environment", "procedure"}
    assert new["environment"] == {**old["environment"], "APERTUS_CHONK_HANDOFF_TRACE": "1"}
    assert new["procedure"] == {**old["procedure"], "launch_blocked": new["procedure"]["launch_blocked"]}
    requested = OBSERVER["handoff_trace_requested"]
    assert requested(new["environment"], True)
    with pytest.raises(ValueError, match="requires the full-layout stall"):
        requested(new["environment"], False)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(new)


def test_extended_timeout_launch_preserves_outer_deadlines_and_is_guarded():
    old, new = [plan(ROOT / f"experiments/chonk-native-fnuz-{n}-i910bfcf936d3",
                     **FULL_PROBE_OPTIONS, cache_image=True, reservation="cscs")
                for n in ("handoff-trace", "timeout600")]
    assert {k for k in new if new[k] != old[k]} == {"experiment", "environment", "procedure"}
    assert new["environment"] == {**old["environment"], "UCCL_EP_CPU_TIMEOUT_SECS": "600"}
    assert new["procedure"] == {**old["procedure"], "launch_blocked": new["procedure"]["launch_blocked"]}
    requested = OBSERVER["extended_timeout_requested"]
    assert not requested(old["environment"], True)
    assert requested(new["environment"], True)
    for env, handoff in ((new["environment"], False), ({"UCCL_EP_CPU_TIMEOUT_SECS": "150"}, True)):
        with pytest.raises(ValueError, match="requires the full-layout handoff"):
            requested(env, handoff)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(new)


@pytest.mark.parametrize("fails", [False, True])
def test_extended_timeout_samples_original_markers_and_preserves_dispatch(fails, monkeypatch):
    from contextlib import nullcontext

    context = dict(call=10, forward=9, backward=2)
    labels = ("b2:before", "b2:after", "f9:before", "f9:after")
    original_markers = {k: ("7", SimpleNamespace(query=lambda: False)) for k in labels}
    cuda = SimpleNamespace(current_device=lambda: 0, device=lambda _: nullcontext())
    later = SimpleNamespace(target=(10, 9, 2), at=lambda _: True, active=lambda: context,
                            markers=original_markers)
    rows, jobs = [], []
    observer = OBSERVER["ExtendedTimeoutTrace"](cuda, lambda e, **kw: rows.append(dict(event=e, **kw)),
                                               later, None)
    monkeypatch.setitem(observer.observe.__globals__, "Thread", lambda **kw: SimpleNamespace(
        start=lambda: jobs.append(kw)))
    now = [0.0]
    monkeypatch.setitem(observer.observe.__globals__, "monotonic", lambda: now[0])

    def native(sentinel):
        job, = jobs
        stop, *rest = job["args"]

        def advance(delay):
            now[0] += delay
            return False

        original_markers.clear()  # The live sampler owns only its copy of the event references.
        monkeypatch.setattr(stop, "wait", advance)
        job["target"](stop, *rest)
        if fails:
            raise RuntimeError("original dispatch failure")
        return sentinel

    sentinel = object()
    dispatch = observer.wrap(native)
    if fails:
        with pytest.raises(RuntimeError, match="original dispatch failure"):
            dispatch(sentinel)
    else:
        assert dispatch(sentinel) is sentinel
        OBSERVER["verify_extended_timeout_trace"](rows, 32)
    assert rows[-1]["returned"] is not fails
    assert jobs[0]["args"][0].is_set()
    assert [r["offset_seconds"] for r in rows[1:-1]] == [101, 150, 300, 580]
    assert all(set(r["markers"]) == set(labels) for r in rows[1:-1])


@pytest.mark.parametrize("failed", [False, True])
def test_reverse_ready_sender_brackets_native_wait_and_only_first_dgrad(failed):
    rows, native, query = [], [], []
    context = dict(call=10, forward=9, backward=0)
    stream = SimpleNamespace(cuda_stream=9)

    class Event:
        def record(self, value):
            assert value is stream

        def query(self):
            query.append(True)
            return True

    cuda = SimpleNamespace(current_stream=lambda: stream, Event=lambda **kw: Event())
    observer = OBSERVER["ReverseReadyTrace"](cuda, lambda e, **kw: rows.append(dict(event=e, **kw)), 16,
                                             lambda: context)
    handle = object()
    queue = [handle]

    def callback(recv_next_wait_handles):
        def wait():
            native.append(recv_next_wait_handles.pop(0))
        return wait

    kwargs = dict(pre_backward=callback(queue))
    observer.callbacks(kwargs)
    kwargs["pre_backward"]()
    assert not queue and rows[0]["work_id"] == id(handle)

    def gemm(backend, operation, a, b, out, **kw):
        native.append(operation)
        if failed:
            raise RuntimeError("native failure")

    run = observer.wrap_gemm(gemm)
    if failed:
        with pytest.raises(RuntimeError, match="native failure"):
            run(None, "fc2_dgrad", [], [], [object()], layout="NN")
    else:
        run(None, "fc2_dgrad", [], [], [], layout="NN")
        for _ in range(2):
            run(None, "fc2_dgrad", [], [], [object()], layout="NN")
    observer.finish()
    assert native[0] is handle
    assert sum(r["event"] == "ready_fc2_dgrad_enter" for r in rows) == 1
    assert query
    if failed:
        with pytest.raises(ValueError, match="missing backward-0"):
            OBSERVER["verify_reverse_ready"](rows, 16)
    else:
        OBSERVER["verify_reverse_ready"](rows, 16)


@pytest.mark.parametrize("contiguous", [True, False])
def test_reverse_ready_layout_observer_forwards_original_tensor(contiguous):
    rows = []
    context = dict(call=9, forward=8, backward=3)
    cuda = SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=13))
    observer = OBSERVER["ReverseReadyTrace"](cuda, lambda e, **kw: rows.append(dict(event=e, **kw)), 48,
                                             lambda: context)
    native = SimpleNamespace(context=context, node="moe_combine backward")
    grad = SimpleNamespace(shape=(32, 1792), stride=lambda: (1792, 1) if contiguous else (1, 32),
                           dtype="torch.bfloat16", is_contiguous=lambda: contiguous)
    ctx = SimpleNamespace(async_finish=True)
    original = lambda cb_ctx, tensor: (cb_ctx, tensor)
    assert observer.wrap_reverse(original, native)(ctx, grad) == (ctx, grad)
    assert rows[0]["contiguous"] is contiguous
    OBSERVER["verify_reverse_ready"](rows, 48)


@pytest.mark.parametrize("outcome", ["pass", "no-compile", "capped", "error"])
def test_later_sender_capture_scoped_and_restores_compile_callbacks(outcome):
    rows, starts, ends, calls = [], [], [], []
    anchor = lambda _: None
    starts.append(anchor)
    handler = SimpleNamespace(register_start_callback=starts.append, remove_start_callback=starts.remove,
                              register_end_callback=ends.append, remove_end_callback=ends.remove)
    emit = lambda event, **fields: rows.append(dict(event=event, **fields))
    capture = OBSERVER["SenderPhaseTrace"](emit, handler)

    @contextmanager
    def acquire(node, name=None):
        yield node

    for target in ((9, 8, None), OBSERVER["LATER_SENDER_CALL"]):
        ctx = dict(zip(("call", "forward", "backward"), target))
        kwargs = {name: (lambda name=name: calls.append(name) or name) for name in OBSERVER["SENDER_CALLBACKS"]}

        def step():
            with capture.capture(ctx, kwargs):
                assert kwargs["pre_forward"]() == "pre_forward"
                with capture.wrap_context(acquire)(work := object(), "attn backward") as value:
                    assert value is work
                    if outcome != "no-compile":
                        n = 100 if outcome == "capped" else 1
                        for i in range(n):
                            event = SimpleNamespace(callback_trigger=SimpleNamespace(name="DYNAMO"), compile_id=str(i))
                            for callback in [*starts, *ends]:
                                callback(event)
                    if outcome == "error" and target == OBSERVER["LATER_SENDER_CALL"]:
                        raise RuntimeError("native failure")
                for name in OBSERVER["SENDER_CALLBACKS"][1:]:
                    assert kwargs[name]() == name

        if outcome == "error" and target == OBSERVER["LATER_SENDER_CALL"]:
            with pytest.raises(RuntimeError, match="native failure"):
                step()
        else:
            step()
        assert starts == [anchor] and not ends and capture.context is None
        if target == (9, 8, None):
            assert not rows
    end = rows[-1]
    assert end["returned"] is (outcome != "error")
    assert end["compile_events"] == (128 if outcome == "capped" else 0 if outcome == "no-compile" else 2)
    assert end["compile_limit_hit"] is (outcome == "capped")
    assert {row["node"] for row in rows if row["event"] == "later_sender_compile"} == (
        set() if outcome == "no-compile" else {"attn backward"})
    verify = OBSERVER["verify_sender_phases"]
    if outcome == "error":
        with pytest.raises(ValueError, match="missing later sender phase"):
            verify(rows, 16)
    else:
        verify(rows, 16)


def test_later_final_prefix_keeps_native_queue_and_reports_target_exception():
    rows = []
    stream = SimpleNamespace(cuda_stream=17)
    cuda = SimpleNamespace(current_stream=lambda: stream,
                           Event=lambda **kw: SimpleNamespace(record=lambda _: None, query=lambda: False))
    emit = lambda event, **fields: rows.append(dict(event=event, **fields))
    trace = OBSERVER["DependencyTrace"](cuda, emit, lambda: stream, target=OBSERVER["LATER_TARGETS"][3])
    old = OBSERVER["DependencyTrace"](cuda, emit, lambda: stream)
    context = dict(call=9, forward=8, backward=3)
    old.begin(context)
    trace.begin(context)
    assert old.context is None and trace.context == context
    autograd = SimpleNamespace(backward=lambda: "loss")
    native_backward = autograd.backward

    def callback(recv_prev_wait_handles):
        def wait():
            return recv_prev_wait_handles.pop(0)
        return wait

    handle = object()
    queue = [handle]
    later = OBSERVER["LaterP2PTrace"](cuda, emit, 48, lambda: context)
    kw = {"pre_forward": callback(queue)}
    later.callbacks(kw)

    def dispatch():
        raise RuntimeError("native timeout")

    def step(plan, *, pre_forward):
        assert plan() == "plan" and autograd.backward() == "loss" and pre_forward() is handle
        trace.wrap_target(dispatch)()

    with pytest.raises(RuntimeError, match="native timeout"):
        trace.wrap_prefix_step(step, autograd)(lambda: "plan", **kw)
    assert not queue and autograd.backward is native_backward
    assert [r["queued"] for r in rows if r["event"] == "prefix_boundary" and "queued" in r] == [1, 0]
    assert rows[-1]["event"] == "dependency_observation" and rows[-1]["boundary"] == "target_exception"
    assert rows[-1]["call"] == 9 and not rows[-1]["query_errors"]


@pytest.mark.parametrize("rank", [16, 32, 48])
@pytest.mark.parametrize("defect", [None, "wrong-work", "failed-dispatch", "query-error"])
def test_later_p2p_preserves_work_and_observes_only_named_edges(rank, defect):
    rows, active, waited = [], {}, []
    work, other_work = object(), object()
    tensor = SimpleNamespace(shape=(4096, 2, 3584), dtype="torch.bfloat16")
    stream = SimpleNamespace(cuda_stream=7)

    class Event:
        def record(self, where):
            assert where is stream

        def query(self):
            if defect == "query-error":
                raise RuntimeError("query failed")
            return False

    cuda = SimpleNamespace(current_stream=lambda: stream, Event=lambda **kwargs: Event())

    def emit(event, **fields):
        rows.append(dict(event=event, **fields))

    def activate(ctx):
        active.update(zip(("call", "forward", "backward"), ctx))

    def recv_fwd(recv_prev_wait_handles):
        def wait():
            waited.append(recv_prev_wait_handles.pop(0))
        return wait

    def recv_bwd(recv_next_wait_handles):
        def wait():
            waited.append(recv_next_wait_handles.pop(0))
        return wait

    trace = OBSERVER["LaterP2PTrace"](cuda, emit, rank, lambda: active)
    isend = trace.wrap_native(lambda **kw: work, "isend")
    irecv = trace.wrap_native(lambda **kw: work, "irecv")
    expect = []
    for edge, direction, operation, peer, post, wait in trace.edges:
        activate(post)
        native, field = (isend, "dst") if operation == "isend" else (irecv, "src")
        assert native(tensor=tensor, **{field: peer}) is work and not trace.post_edges
        callback = {f"post_{direction}": lambda: native(tensor=tensor, **{field: peer})}
        trace.callbacks(callback)
        assert callback[f"post_{direction}"]() is work and not trace.post_edges
        if wait:
            activate(wait)
            handle = other_work if defect == "wrong-work" and not expect else work
            queue = [handle, other_work]
            callbacks = {f"pre_{direction}": (recv_fwd if direction == "forward" else recv_bwd)(queue)}
            trace.callbacks(callbacks)
            callbacks[f"pre_{direction}"]()
            assert queue == [other_work]
            expect.append(handle)
    assert waited == expect
    assert [(x["edge"], x["operation"], x["peer"]) for x in rows if x["event"] == "later_p2p_post_return"] == [
        (edge, operation, peer) for edge, _, operation, peer, _, _ in OBSERVER["later_rank_edges"](rank)]

    def dispatch():
        if defect == "failed-dispatch":
            raise TimeoutError("native timeout")
        for event in ("notify_layout_enter", "notify_layout_enqueued", "notify_dispatch_observation"):
            emit(event, **active, returned=True, markers=dict.fromkeys(OBSERVER["NOTIFY_MARKERS"], True),
                 query_errors={})

    if expect:
        activate(trace.target)
        if defect == "failed-dispatch":
            with pytest.raises(TimeoutError, match="native timeout"):
                trace.wrap_dispatch(dispatch)()
            assert rows[-1]["boundary"] == "target_exception"
        else:
            trace.wrap_dispatch(dispatch)()
        trace.finish("before_target")
        assert sum(x["event"] == "later_p2p_gpu_observation" for x in rows) == 1
    verify = OBSERVER["verify_later_trace"]
    if defect and expect:
        with pytest.raises(ValueError, match="(missing matched later|unusable later P2P)"):
            verify(rows, rank)
    else:
        verify(rows, rank)


def test_ck_seed_copy_preserves_existing_object_and_rejects_changed_bytes(tmp_path):
    key = OBSERVER["CK_OBJECT"]
    source = tmp_path / key
    original = b"\x7fELF" + b"A" * (OBSERVER["CK_BYTES"] - 4)
    source.write_bytes(original)
    install = OBSERVER["install_ck_object"]
    expected = OBSERVER["ck_artifact"](OBSERVER["observe_ck_object"](source))
    target = tmp_path / "te_ck_jit" / key
    identity = install(source, target, expected)
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        install(source, target, expected)
    source.write_bytes(original[:-1] + b"B")
    with pytest.raises(ValueError, match="differs from the qualified"):
        install(source, target, expected)
    assert target.read_bytes() == original
    mapping = f"0-1 r-xp 0 {os.major(identity[0]):x}:{os.minor(identity[0]):x} {identity[1]} {target}"
    verify = OBSERVER["verify_ck_training"]
    seed = target, identity, expected, {"node": 0, **expected}
    verify(seed, maps=mapping)
    with pytest.raises(ValueError, match="did not reuse"):
        verify(seed, maps=mapping + " (deleted)")
    with pytest.raises(ValueError, match="did not reuse"):
        verify(seed, maps=mapping.replace(f" {identity[1]} ", f" {identity[1] + 1} "))


@pytest.mark.parametrize("defect", [None, "missing", "digest", "wrong-node", "rebuild"])
def test_ck_training_acceptance_requires_unchanged_mapped_seed(defect):
    verify = OBSERVER["verify_ck_rank"]
    handoff = dict(node=8, hostname="nid08", key=OBSERVER["CK_OBJECT"], bytes=OBSERVER["CK_BYTES"], sha256="a" * 64)
    rows = [dict(event=e, **OBSERVER["ck_rank_seed"](handoff, 32)) for e in
            ("ck_prewarm_seed_verified", "ck_prewarm_training_mapped")]
    log = ""
    if defect == "missing":
        rows.pop()
    elif defect == "digest":
        rows[-1]["sha256"] = "b" * 64
    elif defect == "wrong-node":
        rows[-1]["node"] = 9
    elif defect == "rebuild":
        log = f"[CK-JIT] JIT-compiling blob: {OBSERVER['CK_STEM']}.cpp"
    if defect:
        with pytest.raises(ValueError, match="CK prewarm evidence"):
            verify(rows, log, 32, handoff)
    else:
        verify(rows, log, 32, handoff)


def test_ck_prewarm_requires_matched_full_layout_trace():
    flag = {"APERTUS_CHONK_CK_PREWARM": "node-verified"}
    requested = OBSERVER["ck_prewarm_requested"]
    assert requested(flag, True) and not requested({}, True)
    with pytest.raises(ValueError, match="requires the matched full-layout"):
        requested(flag, False)
    with pytest.raises(ValueError, match="requires the matched full-layout"):
        requested({"APERTUS_CHONK_CK_PREWARM": "1"}, True)


def test_ck_handoffs_are_node_bound_even_when_all_sha256_differ():
    handoffs = [dict(node=n, hostname=f"nid{n:02}", key=OBSERVER["CK_OBJECT"], bytes=OBSERVER["CK_BYTES"],
                     sha256=f"{n:064x}") for n in range(16)]
    verify = OBSERVER["verify_ck_handoffs"]
    verify(handoffs)
    with pytest.raises(ValueError, match="wrong node"):
        OBSERVER["ck_rank_seed"](handoffs[2], 4)
    with pytest.raises(ValueError, match="invalid CK handoff"):
        verify([handoffs[1], handoffs[0], *handoffs[2:]])
    with pytest.raises(ValueError, match="host placement"):
        verify([dict(handoffs[0], hostname=handoffs[1]["hostname"]), *handoffs[1:]])
    with pytest.raises(ValueError, match="invalid native CK artifact"):
        verify([dict(handoffs[0], sha256="bad"), *handoffs[1:]])


@pytest.mark.parametrize("quiescent", [True, False])
def test_ck_worker_phase_retains_rejected_identity_only_when_quiescent(tmp_path, monkeypatch, quiescent):
    phase = OBSERVER["ck_worker_phase"]
    g = phase.__globals__
    evidence = {}
    cache = tmp_path / "cache"
    path = cache / "xdg/te_ck_jit" / OBSERVER["CK_OBJECT"]

    def fail(command, **kwargs):
        path.parent.mkdir()
        path.write_bytes(b"rejected")
        raise g["subprocess"].CalledProcessError(1, command)

    monkeypatch.setattr(g["subprocess"], "run", fail)
    monkeypatch.setitem(g, "ck_builder_quiescent", lambda: quiescent)
    with pytest.raises(g["subprocess"].CalledProcessError if quiescent else ValueError):
        phase(cache, tmp_path, 0, "cold", g["monotonic"]() + 10, evidence)
    assert evidence["quiescent"] is quiescent
    if quiescent:
        assert evidence["observed_object"] == OBSERVER["observe_ck_object"](path)
        assert evidence["observed_object"]["elf"] is False
    else:
        assert "observed_object" not in evidence


@pytest.mark.parametrize("defect", [None, "no-cold-build", "rebuilt", "not-mapped", "nonfinite", "foreign-object",
                                    "timeout", "orphan"])
def test_ck_node_handoff_requires_quiescent_cold_and_fresh_warm(tmp_path, monkeypatch, defect):
    qualify = OBSERVER["qualify_ck_node"]
    key = OBSERVER["CK_OBJECT"]
    build = tmp_path / "build"
    started = []
    original = b"\x7fELF" + b"X" * (OBSERVER["CK_BYTES"] - 4)

    def phase(cache, output, node, phase_name, deadline, evidence):
        path = cache / "xdg/te_ck_jit" / key
        started.append((phase_name, deadline))
        assert deadline > qualify.__globals__["monotonic"]()
        if phase_name == "cold":
            path.parent.mkdir(parents=True)
            path.write_bytes(original)
            log = "" if defect == "no-cold-build" else (
                f"[CK-JIT] JIT-compiling blob: {OBSERVER['CK_STEM']}.cpp\n[CK-JIT-BUILD] SUCCESS: {path}\n")
        else:
            log = "[CK-JIT] JIT-compiling" if defect == "rebuilt" else ""
            assert path.read_bytes() == original
            if defect == "foreign-object":
                path.chmod(0o644)
                path.write_bytes(original[:-1] + b"Z")
            if defect == "timeout":
                raise TimeoutError("child did not complete")
        observed = OBSERVER["observe_ck_object"](path)
        evidence.update(quiescent=defect != "orphan", observed_object=observed)
        torch, hip = OBSERVER["CK_VENDOR"]
        worker = dict(status="passed", pid=len(started), pgid=123, hostname=qualify.__globals__["socket"].gethostname(),
                      visible_gpus=4, target="gfx942:xnack-", torch=torch, hip=hip,
                      builder_sha256=OBSERVER["CK_BUILDER_SHA"], te_consumer=str(OBSERVER["CK_CONSUMER"]),
                      backends=OBSERVER["CK_BACKENDS"], mapped=defect != "not-mapped",
                      finite_bf16=dict.fromkeys(OBSERVER["CK_TENSORS"], defect != "nonfinite"),
                      norms=OBSERVER["CK_NORMS"], observed_object=observed)
        return worker, log, OBSERVER["ck_artifact"](observed)

    monkeypatch.setitem(qualify.__globals__, "ck_worker_phase", phase)
    if defect:
        with pytest.raises((ValueError, TimeoutError)):
            qualify(build, tmp_path, 0)
    else:
        assert qualify(build, tmp_path, 0)["key"] == key
    assert [name for name, _ in started] == (["cold"] if defect == "no-cold-build" else ["cold", "warm"])
    assert len({d for _, d in started}) == 1
    report = json.loads((tmp_path / "ck-prewarm-node-0.json").read_text())
    assert report["status"] == ("failed" if defect else "passed")
    assert report["cold"]["observed_object"]["bytes"] == OBSERVER["CK_BYTES"]


@pytest.mark.parametrize("failure", [None, "before-layout", "after-layout", "query"])
def test_first_dispatch_markers_are_bounded_and_preserve_failure(failure):
    calls, rows = [], []
    context = dict(call=7, forward=6, backward=1)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is False

        def record(self, stream):
            calls.append(stream)
            self.stream = stream

        def query(self):
            calls.append(("query", self.stream))
            if failure == "query" and self.stream == "UCCL":
                raise RuntimeError("unavailable")
            return self.stream == "Megatron"

    cuda = SimpleNamespace(Event=Event, current_stream=lambda: "Megatron")
    buffer = SimpleNamespace(get_comm_stream=lambda: "UCCL")
    observe = OBSERVER["NotifyTrace"](cuda, lambda event, **row: rows.append(dict(event=event, **row)),
                                     lambda: context)
    layout = observe.wrap_layout(lambda instance, x: x + 1)
    error = RuntimeError("native timeout")

    def native(x):
        if failure == "before-layout":
            raise error
        x = layout(buffer, x)
        if failure in ("after-layout", "query"):
            raise error
        return x

    dispatch = observe.wrap_dispatch(native)
    assert observe.wrap_dispatch(lambda: 3)() == 3 and not rows
    context.update(call=8, forward=7, backward=2)
    if failure:
        with pytest.raises(RuntimeError, match="native timeout") as caught:
            dispatch(4)
        assert caught.value is error
    else:
        assert dispatch(4) == 5
        OBSERVER["verify_notify_trace"](rows, 48)
    assert observe.capture is None
    final = rows[-1]
    assert final["event"] == "notify_dispatch_observation" and final["returned"] == (failure is None)
    if failure == "before-layout":
        assert not calls and final["markers"] == dict(producer=None, prior_uccl=None, layout=None)
    else:
        assert calls == ["Megatron", "UCCL", "UCCL", ("query", "UCCL"), ("query", "UCCL"), ("query", "Megatron")]
        assert final["markers"] == dict(producer=True, prior_uccl=None if failure == "query" else False,
                                       layout=None if failure == "query" else False)
    count = len(rows)
    assert observe.wrap_dispatch(lambda: 7)() == 7 and len(rows) == count


def test_dispatch_markers_require_full_mitigation_recipe():
    check = OBSERVER["notify_trace_requested"]
    assert not check({}, 24, 0, False)
    env = {"APERTUS_CHONK_NOTIFY_TRACE": "1"}
    assert check(env, 42, 512, True)
    for recipe in ((24, 0, False), (42, 512, False), (42, 0, True)):
        with pytest.raises(ValueError, match="notify|notification"):
            check(env, *recipe)


def test_mapping_snapshot_filters_sorts_and_preserves_identity(tmp_path):
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "smaps").write_text(
        "10000000-18000000 rw-s 00000000 00:01 702 /dev/zero (deleted)\n"
        "Rss: 32768 kB\nPss: 16384 kB\nShared_Dirty: 32768 kB\nVmFlags: rd wr sh\n"
        "18000000-18001000 rw-p 00000000 00:00 0\nRss: 65536 kB\n"
        "20000000-30000000 rw-s 00010000 00:01 703 /memfd:test (deleted)\n"
        "Rss: 65536 kB\nPss_Shmem: 32768 kB\n"
    )
    result = memory_telemetry.mapping_snapshot(123, tmp_path, limit=1)
    assert result["candidate_count"] == 2
    largest, = result["mappings"]
    assert largest["name"] == "/memfd:test (deleted)" and largest["device"] == "00:01"
    assert largest["inode"] == 703 and largest["Pss_Shmem"] == 32768 and largest["Rss"] == 65536
    assert memory_telemetry.mapping_snapshot(124, tmp_path)["status"] == "unavailable"


def test_process_tree_includes_workers_from_other_threads(tmp_path):
    for pid, children in ((10, "20 30"), (20, ""), (30, "")):
        path = tmp_path / str(pid)
        task = path / "task" / "123"
        task.mkdir(parents=True)
        (task / "children").write_text(children)
        (path / "status").write_text(f"Name:\ttest{pid}\nVmRSS:\t11 kB\nUid:\t22\n")
    assert memory_telemetry.process_tree(10, tmp_path) == {
        pid: {"Name": f"test{pid}", "VmRSS": "11 kB"} for pid in (10, 20, 30)
    }


def test_cgroup_paths_handle_mount_root(tmp_path):
    proc, mount = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    leaf = mount / "step_0" / "task_4"
    leaf.mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/slurm/job_7/step_0/task_4\n")
    (proc / "self/mountinfo").write_text(f"18 1 0:40 /slurm/job_7 {mount} ro - cgroup2 cgroup rw\n")
    assert list(memory_telemetry.cgroup_paths(proc, mount)) == [leaf, leaf.parent, mount]


def test_child_sigkill_does_not_discard_memory_evidence(tmp_path):
    path = tmp_path / "host-memory-rank-0.jsonl"
    cmd = [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"]
    with pytest.raises(ValueError, match=r"child failed \(-9\)"):
        runtime.child(cmd, tmp_path / "rank-0.log", cwd=tmp_path,
                      env={**os.environ, "LOCAL_RANK": "0"}, timeout=10, telemetry=path)
    live, exit_row = [json.loads(line) for line in path.read_text().splitlines()]
    assert live["event"] == "live" and exit_row["event"] == "exit:-9"
    assert live["pid"] == exit_row["pid"]
    assert {"MemAvailable", "MemTotal"} <= live["node_memory"].keys()
    assert "cgroups" in exit_row and "shm_used_bytes" in exit_row


def test_peer_failure_retains_pre_teardown_sample(tmp_path):
    path = tmp_path / "host-memory-rank-1.jsonl"
    (tmp_path / "failed-0.json").write_text("{}")
    with pytest.raises(RuntimeError, match="peer failed"):
        runtime.child([sys.executable, "-c", "import time; time.sleep(30)"],
                      tmp_path / "rank-1.log", cwd=tmp_path,
                      env={**os.environ, "LOCAL_RANK": "1"}, timeout=10, session=tmp_path, telemetry=path)
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == ["live", "interrupted:RuntimeError"]
