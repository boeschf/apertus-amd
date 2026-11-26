"""CPU-only checks for Swiss-AI preparation, scheduler gating and evidence."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tomllib

import pytest

from apertus_amd import config, runs
from apertus_amd.cli import plan

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "config.toml"
EXPERIMENT = ROOT / "experiments/chonk-swissai-resident-preparation-i910bfcf936d3"
VALIDATOR = ROOT / "experiments/validation/chonk_swissai_preparation.py"
spec = importlib.util.spec_from_file_location("chonk_swissai_preparation", VALIDATOR)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def test_plan_contract(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected scheduler call"))
    planned = plan(EXPERIMENT, site=SITE, job_id="0", cache_mode="warm")
    resources = planned["resources"]
    for key, value in dict(account="csstaff", partition="mi300", nodes=16, gpus_per_node=4,
                           cpus_per_node=192, memory_gb=0, cpu_bind="map_ldom:0,1,2,3",
                           distribution="block:block", mpi="pmix", network="disable_rdzv_get",
                           uenv_passthrough="ignore").items():
        assert resources[key] == value
    assert "--ntasks=64" in planned["step"] and "--gpu-bind=none" in planned["step"]
    assert planned["source"]["commit"] == "e242f3c3c1136aeae498529bfde5e06bda9a7cca"
    assert planned["image"]["sha256"] == "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
    assert planned["prepare"] == [["make", "-C", "megatron/core/datasets"]]
    assert planned["phases"][0]["extra_args"] == ["--save", "/apertus/output/checkpoints"]
    assert planned["validation"]["path"] == str(VALIDATOR)
    argv = planned["args"]
    assert len({a for a in argv if a.startswith("--")}) == sum(a.startswith("--") for a in argv)
    for name, value in dict(ffn_hidden_size=8960, vocab_size=200063, make_vocab_size_divisible_by=128,
                            micro_batch_size=2, global_batch_size=4096, seq_length=4096,
                            lr_warmup_iters=100, lr_decay_iters=59604, train_iters=60, save_interval=60,
                            ckpt_format="torch_dist", pipeline_model_parallel_layout=validator.LAYOUT).items():
        assert config.option(argv, "--" + name.replace("_", "-")) == str(value)
    assert ((200063 + 1 + 127) // 128) * 128 == 200064
    assert 4096 // (16 * 2) == 128 and 4096 * 4096 == 16777216
    assert 100 * 4096 == 409600 and 59604 * 4096 == 244137984
    index = argv.index("--recompute-modules")
    assert argv[index + 1:index + 5] == ["layernorm", "linear_attn", "qkv", "shared_experts"]
    assert {"--kda-legacy-gate-out-proj-bias", "--overlap-grad-reduce", "--delay-wgrad-compute",
            "--overlap-moe-expert-parallel-comm", "--no-gradient-accumulation-fusion"} <= set(argv)
    assert not {"--moe-use-offloading-experts", "--moe-use-inplace-fp8-param", "--moe-use-fp8-dispatch",
                "--no-overlap-p2p-communication", "--lr-warmup-samples", "--lr-decay-samples", "--load",
                "--save", "--async-save", "--goldfish-loss", "--use-distributed-optimizer",
                "--overlap-param-gather", "--sequence-parallel"} & set(argv)
    env = planned["environment"]
    assert env["KDA_ALOG_PER_CHANNEL"] == env["NCCL_RUNTIME_CONNECT"] == "0"
    assert env["NCCL_P2P_DISABLE"] == "0" and env["UCCL_EP_TRANSPORT"] == "cxi"
    assert planned["runtime_environment"]["MAX_JOBS"] == "8"
    assert not {"UCCL_EP_DISPATCH_CONFIG", "UCCL_EP_COMBINE_CONFIG", "AITER_CONFIG_GEMM_BF16",
                "CUDA_DEVICE_MAX_CONNECTIONS", "APERTUS_INDUCTOR_POLICY"} & env.keys()
    edf = tomllib.loads(runs.edf(planned, Path("/no-write")))
    assert edf["entrypoint"] is True and edf["env"]["HWLOC_COMPONENTS"] == "-gl"
    assert edf["annotations"] == {"com": {"hooks": {"cxi": {"enabled": "false"},
        "aws_ofi_nccl": {"enabled": "false"}, "netstack": {"source": "host"}}}}


@pytest.mark.parametrize("allocation", [{"submit": True}, {"job_id": "123"}])
def test_launch_guard_precedes_scheduler_and_freeze(tmp_path, monkeypatch, allocation):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a))
    root = tmp_path / "must-not-exist"
    value = config.resolve(EXPERIMENT, site=SITE, data_root=root, **allocation)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(value)
    assert not root.exists()
    (tmp_path / "resolved.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="launch blocked"):
        runs.launch(tmp_path, "123")
    assert not calls and not (tmp_path / "launch.json").exists()


def test_closed_guard_is_not_a_boolean_override(tmp_path):
    text = (EXPERIMENT / "experiment.toml").read_text().split("launch_blocked =")[0]
    text = text.replace("../validation/chonk_swissai_preparation.py", str(VALIDATOR))
    path = tmp_path / "fixture" / "experiment.toml"
    path.parent.mkdir()
    path.write_text(text + "launch_blocked = false\n")
    with pytest.raises(ValueError, match="nonempty review reason"):
        config.resolve(path, site=SITE, submit=True)


@pytest.mark.parametrize("defect", ["resume", "checkpoint_exists", "cuda_override"])
def test_validator_blocks_unreviewed_native_inputs(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(validator, "train", lambda *a, **kw: pytest.fail("native training must not start"))
    checkpoint = tmp_path / "checkpoints"
    argv = ["--save", str(checkpoint)]
    if defect == "resume":
        argv[:0] = ["--load=/old/run"]
    elif defect == "checkpoint_exists":
        checkpoint.mkdir()
    else:
        monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    with pytest.raises(ValueError, match="fresh training|unreviewed execution overrides"):
        validator.run_training(tmp_path, tmp_path, "pretrain_gpt.py", argv)


def update_lines():
    return [f"[2026-11-20 12:00:00.000000] iteration {i}/60 | consumed samples: {i * 4096} | "
            "elapsed time per iteration (ms): 180000 | lm loss: 1.23 | grad norm: 2.4 | "
            "number of skipped iterations: 0 | number of nan iterations: 0 |" for i in range(1, 61)]


@pytest.mark.parametrize("old,new", [
    ("1.23", "nan"), ("2.4", "inf"), ("180000", "0"), ("iteration 3/", "iteration 2/"),
    ("samples: 12288", "samples: 12287"), ("grad norm: 2.4", "no grad norm"),
    ("skipped iterations: 0", "skipped iterations: 1"), ("nan iterations: 0", "nan iterations: 1"),
])
def test_invalid_update_fails(old, new):
    lines = update_lines()
    lines[2] = lines[2].replace(old, new)
    with pytest.raises(ValueError):
        validator.parse_updates(lines)


def test_missing_update_fails():
    with pytest.raises(ValueError, match="missing updates"):
        validator.parse_updates(update_lines()[:-1])


def checkpoint(tmp_path):
    output = tmp_path / "output"
    iteration = output / "checkpoints/iter_0000060"
    iteration.mkdir(parents=True)
    (iteration.parent / "latest_checkpointed_iteration.txt").write_text("60\n")
    (iteration / "metadata.json").write_text('{"sharded_backend":"torch_dist","common_backend":"torch"}')
    for child in ("common.pt", ".metadata", "__0_0.distcp"):
        (iteration / child).write_bytes(b"inventory deliberately does not load tensor data")
    return output, iteration


@pytest.mark.parametrize("defect", ["missing", "empty", "early", "symlink", "tracker"])
def test_checkpoint_failure_paths(tmp_path, defect):
    _, iteration = checkpoint(tmp_path)
    selected = iteration / "__0_0.distcp"
    if defect == "missing":
        selected.unlink()
    elif defect == "empty":
        selected.write_text("")
    elif defect == "early":
        (iteration.parent / "iter_0000001").mkdir()
    elif defect == "symlink":
        selected.unlink()
        selected.symlink_to(iteration / "common.pt")
    else:
        (iteration.parent / "latest_checkpointed_iteration.txt").write_text("59")
    with pytest.raises(ValueError):
        validator.inventory_checkpoint(iteration.parent)


def test_checker_lifecycle_keeps_outcomes_separate(tmp_path):
    output, _ = checkpoint(tmp_path)
    (output / "rank-63.log").write_text("\n".join(update_lines()))
    (output / "rank-0.log").write_text("successfully saved checkpoint from iteration      60 to /out\n")
    records = [dict(update=n, skipped=0, reruns=0, discarded_batches=0,
                    early_exit=False, forced_checkpoint=False, microbatches=128) for n in range(1, 61)]
    for rank in range(64):
        (output / f"imports-{rank}.json").write_text('{"verified_modules": 1}')
        (output / f"swissai-rank-{rank}.json").write_text(json.dumps({
            "rank": rank, "completed": True,
            "state": dict(args=validator.EFFECTIVE_ARGS, config=validator.EFFECTIVE_CONFIG,
                          geometry=validator.GEOMETRY, initial_samples=0, max_jobs="8"),
            "updates": records, "compiler": [{"update": n} for n in range(1, 61)],
            "memory": {"peak_allocated_bytes": 1, "peak_reserved_bytes": 2, "total_bytes": 3}}))
    validator.check(tmp_path)
    report = json.loads((output / "checks.json").read_text())
    assert report["status"] == "passed" and report["checkpoint"]["status"] == "inventoried"
    assert report["checkpoint"]["tensor_integrity"] == report["checkpoint"]["restoration"] == "not assessed"
    assert report["steady_state_throughput"] == "not qualified" and report["memory_feasibility"] == "not assessed"
    assert [row["count"] for row in report["update_windows"].values()] == [16, 16]
    bad = output / "swissai-rank-19.json"
    payload = json.loads(bad.read_text())
    payload["updates"][8]["reruns"] += 1
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="rerun"):
        validator.check(tmp_path)
    assert json.loads((output / "checks.json").read_text())["status"] == "failed"
