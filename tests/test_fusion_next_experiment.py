"""Keep the fusion attribution input matched and native profiling on update one."""

import builtins
import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from apertus_amd.cli import plan
from apertus_amd.config import option, resolve
from apertus_amd.runs import edf, step_command


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "experiments/moe-chonk-kda-pr73-sc-m1072583a9632-icb5a384016a9"
EXPERIMENT = ROOT / "experiments/moe-chonk-kda-pr73-sc-gaf-profile-m1072583a9632-icb5a384016a9"
PROFILE_ARGS = [
    "--profile", "--use-pytorch-profiler", "--profile-step-start", "1", "--profile-step-end", "2",
    "--profile-ranks", "0", "16", "32", "48", "--pytorch-profiler-collect-callstack",
    "--tensorboard-dir", "/apertus/output/tb",
]


def test_fusion_diagnostic_changes_only_fusion_and_explicit_profile_args():
    control = resolve(CONTROL, submit=True, cache_mode="cold", reservation="cscs", time="01:15:00")
    candidate = resolve(EXPERIMENT, submit=True, cache_mode="cold", reservation="cscs", time="01:15:00")
    expected = control["args"].copy()
    expected.remove("--no-gradient-accumulation-fusion")
    assert candidate["args"] == expected + PROFILE_ARGS
    for key in control.keys() - {"args", "experiment"}:
        assert candidate[key] == control[key], key
    assert candidate["environment"]["APERTUS_INDUCTOR_POLICY"] == "single_candidate_backend"
    assert candidate["environment"]["NCCL_P2P_DISABLE"] == "0"
    assert option(candidate["args"], "--train-iters") == option(candidate["args"], "--save-interval") == "26"


def test_fusion_plan_has_no_writes_subprocesses_or_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("planning must not write, execute or allocate")

    def guarded_open(original):
        def opened(file, mode="r", *args, **kwargs):
            assert not set(mode) & set("wax+"), mode
            return original(file, mode, *args, **kwargs)
        return opened

    monkeypatch.setattr(builtins, "open", guarded_open(builtins.open))
    monkeypatch.setattr(io, "open", guarded_open(io.open))
    for name in ("write_text", "write_bytes", "mkdir", "touch", "unlink", "rename", "replace", "symlink_to"):
        monkeypatch.setattr(Path, name, forbidden)
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    value = plan(EXPERIMENT, submit=True, cache_mode="cold", reservation="cscs", time="01:15:00")
    assert value["phases"] == [{"name": "train", "output": "/apertus/output", "extra_args": ["--save", "/apertus/output/checkpoints"]}]
    assert "--cpus-per-task=<allocated CPUs per node / ranks per node>" in value["step"]
    assert "source_snapshot" not in value and "id" not in value
    assert not any(flag in value["args"] for flag in ("--load", "--exit-interval"))


def test_fusion_direct_step_retains_control_placement_and_edf():
    value = resolve(EXPERIMENT, job_id="123", cache_mode="cold", time="01:15:00")
    cpus = value["resources"]["cpus_per_node"] // value["resources"]["gpus_per_node"]
    assert cpus == 48
    step = step_command(value, Path("<run>"), cpus=cpus)
    for flag in ("--nodes=16", "--ntasks=64", "--ntasks-per-node=4", "--cpus-per-task=48", "--distribution=block:block", "--gpu-bind=none", "--cpu-bind=map_ldom:0,1,2,3"):
        assert flag in step
    text = edf(value, Path("<run>"))
    for line in ('com.hooks.cxi.enabled = "false"', 'com.hooks.aws_ofi_nccl.enabled = "false"', 'HWLOC_COMPONENTS = "-gl"'):
        assert line in text


@pytest.mark.parametrize("rank,enabled", [(0, True), (16, True), (32, True), (48, True), (1, False), (63, False)])
def test_native_profiler_selection_and_schedule_model(rank, enabled):
    # Model the inspected pinned expressions, not a GPU/profiler implementation test.
    args = SimpleNamespace(profile=True, profile_ranks=[0, 16, 32, 48], use_pytorch_profiler=True,
                           profile_step_start=1, profile_step_end=2)
    assert (args.profile and rank in args.profile_ranks and args.use_pytorch_profiler) is enabled
    schedule = dict(wait=max(args.profile_step_start - 1, 0),
                    warmup=1 if args.profile_step_start > 0 else 0,
                    active=args.profile_step_end - args.profile_step_start, repeat=1)
    assert schedule == dict(wait=0, warmup=1, active=1, repeat=1)
    # start() uses slot 0; the loop steps to slot 1 before update 1, slot 2 before update 2.
    assert schedule["wait"] + schedule["warmup"] <= 1 < sum(schedule[k] for k in ("wait", "warmup", "active"))
    assert 2 >= sum(schedule[k] for k in ("wait", "warmup", "active"))


def test_native_profiler_default_excludes_shapes_and_chakra():
    value = resolve(EXPERIMENT, submit=True)
    assert "--pytorch-profiler-collect-callstack" in value["args"]
    assert not {"--pytorch-profiler-collect-shapes", "--pytorch-profiler-collect-chakra", "--record-shapes", "--record-memory-history"} & set(value["args"])
