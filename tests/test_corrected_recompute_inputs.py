"""CPU-only corrected-source preparation contracts; not run authorization or GPU validation."""

import builtins
from copy import deepcopy
import io
from pathlib import Path
import re
import subprocess
import tomllib

import pytest

from apertus_amd.cli import plan
from apertus_amd.config import option
from apertus_amd.runs import edf


ROOT = Path(__file__).resolve().parents[1]
OLD = "1072583a9632fd1902544f7eaea562e2578ef1de"
FIX = "e242f3c3c1136aeae498529bfde5e06bda9a7cca"
IMAGE = "cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0"
ARMS = ("", "moeact-")


def experiment(arm, commit):
    return ROOT / "experiments" / f"moe-chonk-kda-{arm}m{commit[:12]}-i{IMAGE[:12]}"


def inputs(arm, commit):
    return tomllib.loads((experiment(arm, commit) / "experiment.toml").read_text())


@pytest.mark.parametrize("arm", ARMS)
def test_each_corrected_arm_changes_only_its_committed_source_pin(arm):
    original, corrected = inputs(arm, OLD), inputs(arm, FIX)
    assert original["source"]["commit"] == OLD
    expected = deepcopy(original)
    expected["source"]["commit"] = FIX
    assert corrected == expected
    assert re.fullmatch(r"[0-9a-f]{40}", corrected["source"]["commit"])
    assert corrected["image"]["sha256"] == IMAGE
    assert len(experiment(arm, FIX).name) <= 64
    text = (experiment(arm, FIX) / "experiment.toml").read_text()
    assert "NOT READY FOR RUN" in text
    assert "compiler-control/repetition review" in text and "GPU resource authorization" in text
    assert "11-26/45-60" in text


def test_corrected_pair_differs_only_by_full_moe_versus_activation_recomputation():
    baseline, activation = inputs("", FIX), inputs("moeact-", FIX)
    expected = deepcopy(baseline)
    args = expected["args"]
    index = args.index("--recompute-modules")
    assert args[index + 1:index + 4] == ["layernorm", "linear_attn", "moe"]
    args[index + 3] = "moe_act"
    assert activation == expected
    differences = [(a, b) for a, b in zip(baseline["args"], activation["args"]) if a != b]
    assert differences == [("moe", "moe_act")]


@pytest.mark.parametrize("arm", ARMS)
def test_preparation_retains_duration_checkpoint_network_fusion_and_default_compilation(arm):
    value = inputs(arm, FIX)
    args, env = value["args"], value["environment"]
    assert value["nodes"] == 16
    assert value["entrypoint"] == "pretrain_gpt.py"
    assert value["procedure"] == {"kind": "train", "validation": "../validation/memory_trace.py"}
    assert option(args, "--train-iters") == option(args, "--save-interval") == "60"
    assert option(args, "--ckpt-format") == "torch"
    assert option(args, "--micro-batch-size") == "1" and option(args, "--global-batch-size") == "512"
    assert "--no-gradient-accumulation-fusion" in args and "--no-overlap-p2p-communication" in args
    assert env["NCCL_P2P_DISABLE"] == "1" and env["TORCH_COMPILE_DISABLE"] == "0"
    assert env["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:True"
    assert "APERTUS_INDUCTOR_POLICY" not in env
    assert not any("policy" in key.lower() for key in value)
    assert not any("policy" in arg.lower() for arg in args)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("launch", [{"submit": True}, {"job_id": "123456"}])
def test_standard_plan_is_pure_and_retains_64_ranks_and_network_safety(monkeypatch, arm, launch):
    def forbidden(*args, **kwargs):
        pytest.fail("planning attempted a subprocess or filesystem write")

    def readonly_open(original):
        def open_file(file, mode="r", *args, **kwargs):
            assert not any(flag in mode for flag in "wax+"), "planning attempted a write"
            return original(file, mode, *args, **kwargs)
        return open_file

    # Install guards only around planning/EDF rendering, not pytest's own I/O.
    with monkeypatch.context() as guard:
        for name in ("run", "Popen", "call", "check_call", "check_output"):
            guard.setattr(subprocess, name, forbidden)
        for name in ("write_text", "write_bytes", "mkdir", "touch", "rename", "replace", "unlink"):
            guard.setattr(Path, name, forbidden)
        guard.setattr(builtins, "open", readonly_open(builtins.open))
        guard.setattr(io, "open", readonly_open(io.open))
        value = plan(experiment(arm, FIX), site=ROOT / "config.toml", **launch)
        container = edf(value, Path("/unwritten-run"))
    resources, step = value["resources"], value["step"]
    assert resources["nodes"] == 16 and resources["gpus_per_node"] == 4
    assert resources["nodes"] * resources["gpus_per_node"] == 64
    assert resources["distribution"] == "block:block"
    assert resources["cpu_bind"] == "map_ldom:0,1,2,3"
    for argument in ("--nodes=16", "--ntasks=64", "--ntasks-per-node=4", "--gpus-per-task=1", "--gpu-bind=none"):
        assert argument in step
    assert "--distribution=block:block" in step
    assert option(value["args"], "--pipeline-model-parallel-size") == "4"
    assert option(value["args"], "--expert-model-parallel-size") == "16"
    assert option(value["args"], "--tensor-model-parallel-size") == "1"
    assert value["source"]["commit"] == FIX and value["image"]["sha256"] == IMAGE
    assert "APERTUS_INDUCTOR_POLICY" not in value["environment"]
    assert value["environment"]["NCCL_P2P_DISABLE"] == "1"
    assert 'com.hooks.cxi.enabled = "false"' in container
    assert 'com.hooks.aws_ofi_nccl.enabled = "false"' in container
    assert 'HWLOC_COMPONENTS = "-gl"' in container
    assert value["phases"] == [{"name": "train", "output": "/apertus/output",
                                "extra_args": ["--save", "/apertus/output/checkpoints"]}]
