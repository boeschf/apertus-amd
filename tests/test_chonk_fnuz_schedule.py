"""CPU and scheduler-stub coverage of the combined FNUZ rehearsal gate."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from apertus_amd import config, runs
from apertus_amd.artifacts import write_json
from apertus_amd.cli import plan


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/chonk-native-fnuz-schedule-i910bfcf936d3"
FULL = ROOT / "experiments/chonk-swissai-resident-fnuz-i910bfcf936d3"
SCRIPT = ROOT / "experiments/validation/chonk_fnuz_schedule.py"
SPEC = importlib.util.spec_from_file_location("chonk_fnuz_schedule", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_plan_and_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected scheduler call"))
    site = ROOT / "config.toml"
    opts = dict(site=site, submit=True, reservation="cscs", gpus_per_node=4, cpus_per_task=48,
                cpu_bind="map_ldom:0,1,2,3", time="00:40:00", cache_mode="cold")
    planned = plan(EXP, **opts)
    full = config.resolve(FULL, site=site, submit=True)
    expected = {"--num-layers": "4", "--linear-attention-freq": "([1,1,1,0])",
                "--moe-layer-freq": "1", "--pipeline-model-parallel-size": "2",
                "--pipeline-model-parallel-layout": gate.LAYOUT, "--global-batch-size": "256",
                "--train-iters": "2", "--save-interval": None, "--log-device-memory-used": None}

    def by_flag(argv):
        result = {}
        for item in argv:
            if item.startswith("--"):
                key = item
                assert key not in result
                result[key] = []
            else:
                result[key].append(item)
        return result

    defaults = by_flag(full["args"])
    for key, value in expected.items():
        if value is None:
            del defaults[key]
        else:
            defaults[key] = [value]
    assert by_flag(planned["args"]) == defaults
    assert planned["source"] == full["source"]
    assert planned["source"]["commit"] == "4b21bb5781ee6db66952a9b6ab128b688c7dc1d1"
    assert planned["image"] == full["image"]
    assert planned["environment"] == full["environment"]
    assert planned["resources"]["nodes"] == 8
    assert planned["procedure"]["timeout"] == 1500
    assert planned["phases"] == [{"name": "probe", "output": "/apertus/output", "extra_args": []}]
    assert {"--ntasks=32", "--cpus-per-task=48", "--gpu-bind=none", "--mem=0G"} <= set(planned["step"])
    # The live declaration can be unblocked for an approved run; exercise an
    # explicitly blocked copy without relying on the current approval state.
    fixture = tmp_path / EXP.name
    fixture.mkdir()
    text = (EXP / "experiment.toml").read_text().replace("../validation/chonk_fnuz_schedule.py", str(SCRIPT))
    lines = [line for line in text.splitlines() if not line.startswith("launch_blocked")]
    (fixture / "experiment.toml").write_text("\n".join(lines) + '\nlaunch_blocked = "review required"\n')
    value = config.resolve(fixture, data_root=tmp_path / "never-created", **opts)
    with pytest.raises(ValueError, match="launch blocked"):
        runs.run(value)
    assert not (tmp_path / "never-created").exists()


def step(rank, n):
    return dict(update=n, skipped=0, reruns=0, discarded=0, checkpoint=False, early_exit=False,
                grad_norm=1.5, losses={"lm loss": 1.2} if rank in gate.LOSS_RANKS else {},
                versions_advanced=True, chunks=[dict(gate.EXPECTED_CHUNK) for _ in range(2)],
                optimizer_grad_views=[gate.EXPERT_PARAMS], finite_resident_weights=True)


@pytest.mark.parametrize("changes", [
    dict(skipped=1), dict(reruns=1), dict(discarded=1), dict(checkpoint=True), dict(early_exit=True),
    dict(grad_norm=float("nan")), dict(grad_norm=0.0), dict(versions_advanced=False),
    dict(chunks=[{"parameters": 32, "ready_events": 255}] * 2),
    dict(optimizer_grad_views=[63]), dict(finite_resident_weights=False),
    dict(losses={}), dict(losses={"lm loss": float("inf")}),
])
def test_bad_update(changes):
    with pytest.raises(ValueError, match="unqualified"):
        gate.verify_step({**step(16, 1), **changes}, 1, True)


@pytest.fixture
def evidence(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    for rank in range(gate.WORLD):
        write_json(output / f"schedule-rank-{rank}.json",
                   dict(rank=rank, completed=True, distributed_teardown=True, transport=gate.TRANSPORT,
                        effective=gate.SETTINGS, geometry=gate.GEOMETRY,
                        updates=[step(rank, 1), step(rank, 2)]))
        write_json(output / f"imports-{rank}.json", {"verified_modules": 20})
        (output / f"rank-{rank}.log").write_text("[Proxy] UCCL_EP_TRANSPORT=cxi initialized\n")
    return output


def test_all_ranks(evidence):
    gate.check(evidence.parent)
    report = json.loads((evidence / "checks.json").read_text())
    assert report["status"] == "passed"
    assert report["loss_ranks"] == list(range(16, 32))


@pytest.mark.parametrize("defect", ["missing", "teardown", "transport", "loss-stage", "no-cxi", "checkpoint"])
def test_reject_incomplete_evidence(evidence, defect):
    path = evidence / "schedule-rank-31.json"
    row = json.loads(path.read_text())
    if defect == "missing":
        path.unlink()
    elif defect == "no-cxi":
        (evidence / "rank-31.log").write_text("")
    elif defect == "checkpoint":
        (evidence / "checkpoints").mkdir()
    else:
        if defect == "teardown":
            row["distributed_teardown"] = False
        elif defect == "transport":
            row["transport"]["nvl_bytes"] -= 1
        else:
            row["updates"][0]["losses"] = {}
        write_json(path, row)
    with pytest.raises(ValueError):
        gate.check(evidence.parent)
    assert json.loads((evidence / "checks.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("defect", [None, "missing", "duplicate", "not-alias", "wrong-optimizer"])
def test_optimizer_views(monkeypatch, defect):
    class Wrapper:
        pass

    class MD:
        pass

    monkeypatch.setitem(sys.modules, "megatron.core.optimizer.optimizer",
                        NS(Float16OptimizerWithFloat16Params=Wrapper))
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer.md_decoupling", NS(MDDecoupling=MD))
    grad = NS(data_ptr=lambda: 17, size=lambda: (3840, 1792), stride=lambda: (1792, 1),
              dtype="fp32", device="cuda")
    param = NS(resident_fp8_delayed_wgrad=True, main_grad=grad, device="cuda")
    master = NS(grad=grad, dtype="fp32", device="cuda")
    chain = Wrapper()
    chain.optimizer = MD() if defect != "wrong-optimizer" else object()
    chain.float16_groups = [[param]]
    chain.fp32_from_float16_groups = [[master]]
    if defect == "missing":
        chain.float16_groups[0].clear()
        chain.fp32_from_float16_groups[0].clear()
    elif defect == "duplicate":
        chain.float16_groups[0].append(param)
        chain.fp32_from_float16_groups[0].append(master)
    elif defect == "not-alias":
        master.grad = NS(**{**vars(grad), "data_ptr": lambda: 18})
    args = NS(chained_optimizers=[chain]), [(None, None, [param])], NS(float32="fp32")
    if defect:
        with pytest.raises(ValueError, match="optimizer"):
            gate.verify_optimizer_grads(*args)
    else:
        assert gate.verify_optimizer_grads(*args) == 1


@pytest.mark.parametrize("defect", [None, "early-prepare", "extra-schedule", "undrained", "stale-version"])
def test_observer_orders_schedule_and_optimizer(monkeypatch, defect):
    class LayerWise:
        async_allgather = False

        def prepare_grad_norm(self):
            return False, 1.5

    class AllTrue:
        def all(self):
            return True

    args = NS(**gate.SETTINGS, consumed_train_samples=0, main_grads_dtype="fp32", params_dtype="bf16")
    rerun = NS(rerun_count=0, discarded_batches=0)
    mpu = NS(is_pipeline_last_stage=lambda **kw: True)
    names = dict(tp="tensor_model", pp="pipeline_model", vpp="virtual_pipeline_model",
                 ep="expert_model", etp="expert_tensor", cp="context", dp="data",
                 edp="expert_data")
    for key, name in names.items():
        setattr(mpu, f"get_{name}_parallel_world_size", lambda key=key: gate.GEOMETRY[key])
    chunks = []
    for _ in range(2):
        param = object()
        expert = NS(_pending=[], _outstanding=0, version=0, _version=None)
        expert._weight_version = lambda expert=expert, p=param: ((id(p), expert.version, id(p)),)
        chunks.append((NS(), expert, [param]))

    class Param:
        grad = None

    for (_, expert, params) in chunks:
        params[0] = Param()

    def scheduler():
        for chunk, expert, params in chunks:
            expert._version = expert._weight_version()
            expert._pending = [object()] if defect == "undrained" else []
            chunk._resident_fp8_grad_counts = {p: gate.MICRO for p in params}
            chunk._resident_fp8_grad_streams = {object()}
            chunk.bucket_groups = [NS(grad_reduce_handle=None)]
            chunk.expert_parallel_bucket_groups = []
        # Use real expected totals while keeping the test's tensors inert.
        return [{"lm loss": 1.2}]

    def train_step(forward, data, model, optimizer, schedule, cfg, run_schedule, iteration=None):
        if defect == "early-prepare":
            optimizer.prepare_grad_norm()
        run_schedule()
        if defect == "extra-schedule":
            run_schedule()
        _, norm = optimizer.prepare_grad_norm()
        if defect != "stale-version":
            for _, expert, _ in chunks:
                expert.version += 1
        return {"lm loss": 1.2}, 0, False, False, 0, norm, None, 0

    training = NS(train_step=train_step, get_args=lambda: args)
    for name, mod in {
        "torch": NS(float32="fp32", bfloat16="bf16", isfinite=lambda _: AllTrue(), stack=lambda _: AllTrue()),
        "megatron.core": NS(parallel_state=mpu),
        "megatron.training": NS(training=training),
        "megatron.core.num_microbatches_calculator": NS(get_num_microbatches=lambda: gate.MICRO),
        "megatron.core.optimizer.layer_wise_optimizer": NS(LayerWiseDistributedOptimizer=LayerWise),
        "megatron.core.pipeline_parallel.schedules": NS(forward_backward_pipelining_with_interleaving=scheduler),
        "megatron.core.rerun_state_machine": NS(get_rerun_state_machine=lambda: rerun),
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(gate, "verify_chunks", lambda model, torch: chunks)
    monkeypatch.setattr(gate, "EXPECTED_CHUNK", {"parameters": 1, "ready_events": gate.MICRO})
    monkeypatch.setattr(gate, "verify_optimizer_grads", lambda *a: gate.EXPERT_PARAMS)
    cfg = NS(microbatch_group_size_per_vp_stage=2, overlap_p2p_comm=True,
             overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=True,
             finalize_model_grads_func=object(), grad_sync_func=[object(), object()])
    optimizer = LayerWise()
    record = dict(rank=16, updates=[])
    gate.install_observer(record)
    call = lambda i: training.train_step(None, None, None, optimizer, None, cfg, scheduler, iteration=i)
    if defect:
        with pytest.raises(ValueError):
            call(0)
    else:
        call(0)
        args.consumed_train_samples += gate.BATCH
        call(1)
        assert [r["update"] for r in record["updates"]] == [1, 2]
    assert "prepare_grad_norm" not in vars(optimizer)


@pytest.mark.parametrize("defect", [None, "train", "teardown", "checkpoint"])
def test_failure_and_teardown(tmp_path, monkeypatch, defect):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "32")
    calls = []

    def destroy():
        calls.append("destroy")
        if defect == "teardown":
            raise RuntimeError("teardown failure")

    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(synchronize=lambda: calls.append("sync")),
        distributed=NS(is_initialized=lambda: True, destroy_process_group=destroy)))
    monkeypatch.setattr(gate, "transport_state", lambda: gate.TRANSPORT)
    monkeypatch.setattr(gate, "install_observer", lambda record: record.update(updates=[step(0, 1), step(0, 2)]))

    def train(*args, before_run):
        before_run()
        if defect == "train":
            raise RuntimeError("train failure")

    monkeypatch.setattr(gate, "train", train)
    if defect:
        with pytest.raises((ValueError, RuntimeError)):
            gate.run_training(tmp_path, tmp_path, "pretrain_gpt.py", ["--load=other"] if defect == "checkpoint" else [])
    else:
        gate.run_training(tmp_path, tmp_path, "pretrain_gpt.py", [])
    if defect == "checkpoint":
        assert not list(tmp_path.iterdir())
    else:
        record = json.loads((tmp_path / "schedule-rank-0.json").read_text())
        assert record["completed"] == record["distributed_teardown"] == (defect is None)
        assert calls == ([] if defect == "train" else ["sync", "destroy"])
