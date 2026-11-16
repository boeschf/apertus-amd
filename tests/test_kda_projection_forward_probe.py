"""CPU contract tests only; no installed Torch/native import or GPU oracle."""

import builtins
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from apertus_amd import config
from apertus_amd.runs import edf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/validation/kda_projection_forward_probe.py"
EXPERIMENT = ROOT / "experiments/chonk-kda-projection-forward-i910bfcf936d3"


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformer_engine", "triton", "aiter"}:
            pytest.fail("CPU import touched vendor")
        return original(name, *args, **kwargs)
    spec = importlib.util.spec_from_file_location("forward_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded)
        spec.loader.exec_module(module)
    return module


def settings():
    return config.resolve(EXPERIMENT, submit=True, time="00:05:00", reservation="cscs")


@pytest.fixture
def launch(probe, monkeypatch, tmp_path):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    env = {**value["environment"], "HWLOC_COMPONENTS": "-gl", "SLURM_PROCID": "0",
           "SLURM_LOCALID": "0", "SLURM_NODEID": "0", "SLURM_NTASKS": "4", "SLURM_JOB_ID": "123"}
    for key in list(os.environ):
        if key.startswith(("TE_HIPBLASLT_", "PYTORCH_TUNABLEOP_", "HIPBLASLT_TUNING")):
            monkeypatch.delenv(key)
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda pid: set(range(48)))
    monkeypatch.chdir(tmp_path)
    return value


def test_pure_plan_and_edf(probe, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("plan spawned"))
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: pytest.fail("plan wrote"))
    value = settings()
    assert value["procedure"]["commands"] == [probe.COMMAND]
    assert value["procedure"]["timeout"] == 180 and "source" not in value
    assert value["resources"]["cpus_per_node"] // value["resources"]["gpus_per_node"] == 48
    assert value["validation"]["sha256"] == probe.digest(SCRIPT)
    container = edf(value, Path("/run"))
    for text in ('com.hooks.cxi.enabled = "false"', 'com.hooks.aws_ofi_nccl.enabled = "false"',
                 'HWLOC_COMPONENTS = "-gl"', '/apertus/validation.py:ro'):
        assert text in container


@pytest.mark.parametrize("key,val", [("TE_HIPBLASLT_ALGO_LOAD", "old"), ("TE_HIPBLASLT_ALGO_SAVE", "x"),
    ("TE_HIPBLASLT_TUNING_RUN_COUNT", "1"), ("TE_HIPBLASLT_TUNING_ALGO_COUNT", "16"),
    ("PYTORCH_TUNABLEOP_ENABLED", "0"), ("HIPBLASLT_TUNING_OVERRIDE_FILE", "x")])
def test_forbidden_selection(probe, key, val):
    with pytest.raises(ValueError):
        probe.validate_environment(probe.DEFAULTS | {key: val})


def test_geometry_and_resource_refusal(probe, launch):
    assert probe.validate_launch(launch, os.environ) == (0, 0)
    for key, val in (("time", "00:06:00"), ("nodes", 2), ("cpus_per_task", 24)):
        changed = deepcopy(launch)
        changed["resources"][key] = val
        with pytest.raises(ValueError):
            probe.validate_launch(changed, os.environ)


class Tensor:
    dtype = "torch.bfloat16"
    def __init__(self, shape):
        self.shape = shape
    def stride(self):
        return (self.shape[1], 1)
    def storage_offset(self):
        return 0


def test_forward_only_native_contract_and_wrong_shape(probe):
    x, w, y = Tensor((4096, 3584)), Tensor((11036, 3584)), Tensor((4096, 11036))
    calls = []
    probe.invoke(lambda *a, **k: calls.append((a, k)), x, w, y)
    assert calls == [((w, x), dict(out_dtype=y.dtype, out=y, layout="TN", alpha=1.0, beta=0.0,
                                 accumulate=False, grad=False, use_split_accumulator=False, bias=None, gelu=False))]
    with pytest.raises(ValueError, match="shape/stride"):
        probe.invoke(lambda *a, **k: pytest.fail("wrong shape reached native"), x, w, Tensor((4096, 11040)))


@pytest.mark.parametrize("damage", [{"finite": False}, {"mismatched_elements": 1},
    {"rms_error": 0.1}, {"max_abs": math.nan}, {"reference_rms": math.inf}])
def test_unchanged_gates_fail_closed(probe, damage):
    metric = dict(finite=True, mismatched_elements=0, max_abs=0.0, rms_error=0.0, reference_rms=1.0)
    assert probe.metrics_pass(metric)
    assert not probe.metrics_pass(metric | damage)


def test_native_algorithm_metadata_not_table_ack(probe):
    log = ("m=11036 k=3584 n=4096 transa=T transb=N A_type=6 B_type=6 D_type=6 "
           "bias_type=4 grad=0 bias=0 gelu=0 use_fp8=0 scale_mode=tensor alpha=1 beta=0\n"
           "[INFO] Use hipBLASLt algo [0] 83530\n")
    assert probe.algorithm_metadata(log)["algo_id"] == 83530
    for bad in ("", log.replace("11036", "11040"), log.replace("D_type=6", "D_type=0"),
                log + "[INFO] Use hipBLASLt algo [0] 83530\n", log + "Loading autotune"):
        with pytest.raises(ValueError):
            probe.algorithm_metadata(bad)


def test_timing_counts_and_separation(probe):
    events, ticks = [], iter(range(36))
    class Event:
        def __init__(self, **kw):
            pass
        def record(self):
            events.append("event")
        def synchronize(self):
            events.append("complete")
        def elapsed_time(self, end):
            return 0.5
    cuda = SimpleNamespace(Event=Event, synchronize=lambda: events.append("sync"))
    samples = probe.measure(SimpleNamespace(cuda=cuda), lambda: events.append("native"), clock=lambda: next(ticks))
    assert len(samples) == 10 and events.count("native") == 12
    assert events == ["sync", "event", "native", "event", "complete"] * 12
    assert all(s == dict(host_return_seconds=1, completion_seconds=2, device_event_ms=0.5) for s in samples)


def test_provenance_failure_precedes_vendor_import(probe, tmp_path, monkeypatch):
    monkeypatch.setenv("NVTE_LOG_GEMM_CONFIG", "1")
    monkeypatch.setattr(probe, "validate_environment", lambda env, stage: None)
    monkeypatch.setattr(probe, "PACKAGE", tmp_path / "absent")
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: probe.VERSIONS[name])
    original = builtins.__import__
    def guarded(name, *a, **k):
        if name.startswith(("torch", "transformer_engine")):
            pytest.fail("provenance failure imported vendor")
        return original(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(FileNotFoundError):
        probe.gpu_worker(tmp_path, 0, 0)
    assert json.loads((tmp_path / "worker.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("outcome", ["pass", "failed", "timeout", "descendants", "late-output", "id-mismatch"])
def test_worker_same_pgid_private_caches_and_cleanup(probe, launch, tmp_path, monkeypatch, outcome):
    waits, stages = [], []
    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is False
        directory = Path(command[-1])
        assert command[3] == "worker" and command[5] == probe.CONTRACT
        stage = command[7]
        stages.append(stage)
        assert kwargs["env"]["NVTE_LOG_GEMM_CONFIG"] == ("1" if stage == "proof" else "0")
        for key in config.CACHE_PATHS:
            assert Path(kwargs["env"][key]).is_dir()
            assert str(tmp_path / "scratch") in kwargs["env"][key]
        def wait(timeout):
            assert 0 < timeout <= 160
            waits.append(timeout)
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(command, timeout)
            log = native_log(stage, 999 if outcome == "id-mismatch" and stage == "measurement" else 83530)
            kwargs["stdout"].write(log)
            kwargs["stdout"].flush()
            worker = {key: "same" for key in ("rank", "seed", "consumer_source", "driver_sha256", "provenance",
                      "tensors", "workspace", "mapped_libraries", "parameters", "local_device")}
            worker.update(status="passed" if outcome != "failed" else "failed", pgid=os.getpgrp(),
                          error="original numerical failure", algorithm=probe.algorithm_metadata(log, stage))
            probe.write(directory / "worker.json", worker)
            if outcome == "late-output" and stage == "measurement":
                kwargs["stdout"].write("unexpected buffered output\n")
            return int(outcome == "failed")
        return SimpleNamespace(wait=wait)
    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    monkeypatch.setattr(probe, "quiescent", lambda: outcome != "descendants")
    if outcome == "pass":
        probe.run(launch)
    else:
        with pytest.raises((ValueError, subprocess.TimeoutExpired)) as error:
            probe.run(launch)
        if outcome == "failed":
            assert "original numerical failure" in str(error.value)
    result = json.loads((tmp_path / "forward-rank-0/summary.json").read_text())
    assert result["status"] == ("passed" if outcome == "pass" else "failed")
    assert Path(result["scratch"]).exists() == (outcome != "pass")
    if outcome in {"pass", "descendants"}:
        assert stages == ["proof", "measurement"] and waits[1] < waits[0]
        for stage in stages:
            assert (tmp_path / f"forward-rank-0/{stage}/native.log").read_text() == native_log(stage)
    if outcome in {"failed", "timeout"}:
        assert stages == ["proof"]


def test_no_tuning_cli_or_unbounded_driver(probe, monkeypatch):
    assert len(SCRIPT.read_text().splitlines()) <= 400
    for mode in ("tune", "replay", "wgrad", "dgrad"):
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), mode, "--native-provenance", probe.CONTRACT])
        with pytest.raises(SystemExit):
            probe.main()


def test_secondary_evidence_error_never_replaces_numerical_failure(probe, monkeypatch):
    def fail(*args):
        raise OSError("evidence disk fixture")
    monkeypatch.setattr(probe, "write", fail)
    with pytest.raises(ValueError, match="numerical failure") as error:
        try:
            raise ValueError("original numerical failure")
        finally:
            probe.retain(Path("unused"), {})
    assert "secondary evidence error" in error.value.__notes__[0]
    with pytest.raises(OSError):
        probe.retain(Path("unused"), {})


def native_log(stage, algo=83530):
    descriptor = ("m=11036 k=3584 n=4096 transa=T transb=N A_type=6 B_type=6 D_type=6 "
                  "bias_type=4 grad=0 bias=0 gelu=0 use_fp8=0 scale_mode=tensor alpha=1 beta=0\n")
    return (descriptor if stage == "proof" else "") + f"[INFO] Use hipBLASLt algo [0] {algo}\n"


@pytest.mark.parametrize("extra", ["m=11036 k=3584 n=4096\n", "unexpected output\n",
    "[WARNING] selection failed\n", "[INFO] Use hipBLASLt algo [0] 83530\n", "Saving autotune\n"])
def test_measurement_log_rejects_all_descriptor_output_and_ambiguity(probe, extra):
    with pytest.raises(ValueError):
        probe.algorithm_metadata(native_log("measurement") + extra, "measurement")


def test_measurement_startup_must_be_quiet_before_provenance_or_import(probe, tmp_path, monkeypatch):
    monkeypatch.setenv("NVTE_LOG_GEMM_CONFIG", "1")
    for key, value in probe.DEFAULTS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(probe, "provenance", lambda: pytest.fail("wrong startup reached provenance/import"))
    with pytest.raises(ValueError, match="explicit default"):
        probe.gpu_worker(tmp_path, 0, 0, "measurement")


@pytest.fixture
def fake_gpu(probe, monkeypatch, tmp_path):
    """Fake control flow only: no CPU reimplementation of GEMM or CUDA oracle."""
    from contextlib import contextmanager
    from apertus_amd import runtime
    events, state = [], {}
    class FakeTensor(Tensor):
        def __mul__(self, scalar):
            return self
        def stride(self):
            return (1,) if len(self.shape) == 1 else super().stride()
        def is_contiguous(self):
            return True
        def numel(self):
            return math.prod(self.shape)
        def data_ptr(self):
            return 256
        def clone(self):
            return FakeTensor(self.shape)
    @contextmanager
    def affinity(record):
        yield frozenset(range(48))
    monkeypatch.setattr(runtime, "preserve_affinity", affinity)
    monkeypatch.setattr(runtime, "require_affinity", lambda *args: None)
    monkeypatch.setattr(probe, "provenance", lambda: {"fixed": True})
    monkeypatch.setattr(probe, "mapped_libraries", lambda: {"fixed": True})
    def gemm(*args, **kwargs):
        events.append("native")
        if not state.get("logged"):
            state["log"].write_text(native_log(state["stage"], state.get("algo", 83530)))
            state["logged"] = True
    module = SimpleNamespace(__file__=str(probe.PACKAGE / "transformer_engine/pytorch/cpp_extensions/gemm.py"),
                             general_gemm=gemm, get_cublas_workspace=lambda *args: FakeTensor((probe.WORKSPACE,)))
    cuda = SimpleNamespace(device_count=lambda: 4, set_device=lambda local: None, current_device=lambda: 0,
                           get_device_properties=lambda local: SimpleNamespace(gcnArchName="gfx942"), synchronize=lambda: None)
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip="fake"), bfloat16="fake",
        manual_seed=lambda seed: events.append(("seed", seed)), set_float32_matmul_precision=lambda value: None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
        randn=lambda *shape, **kw: FakeTensor(shape), empty=lambda *shape, **kw: FakeTensor(shape),
        empty_like=lambda t: FakeTensor(t.shape), equal=lambda a, b: True)
    root = SimpleNamespace(pytorch=SimpleNamespace(cpp_extensions=SimpleNamespace(gemm=module)))
    original = builtins.__import__
    def imported(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformer_engine"):
            assert os.environ["NVTE_LOG_GEMM_CONFIG"] == ("1" if state["stage"] == "proof" else "0")
            events.append("import")
            return torch if name == "torch" else root
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", imported)
    def numerical(*args):
        events.append("numeric")
        if state.get("descriptor_before_measurement"):
            with state["log"].open("a") as log:
                log.write("m=unexpected before timing\n")
        return {"passed": not state.get("numerical_failure")}
    def measured(torch, call):
        events.append("measure")
        for _ in range(12):
            call()
        if state.get("descriptor_during_measurement"):
            with state["log"].open("a") as log:
                log.write("m=unexpected timed descriptor\n")
        return [dict(host_return_seconds=1, completion_seconds=2, device_event_ms=0.5)] * 10
    monkeypatch.setattr(probe, "numerical_check", numerical)
    monkeypatch.setattr(probe, "measure", measured)
    def run(stage, **damage):
        state.clear()
        directory = tmp_path / stage
        directory.mkdir()
        state.update(stage=stage, log=directory / "native.log", **damage)
        for key, value in probe.DEFAULTS.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("NVTE_LOG_GEMM_CONFIG", "1" if stage == "proof" else "0")
        probe.gpu_worker(directory, 0, 0, stage)
        return json.loads((directory / "worker.json").read_text())
    return run, events


def test_proof_never_measures_and_fresh_default_validates_before_and_after(probe, fake_gpu):
    run, events = fake_gpu
    proof = run("proof")
    assert "samples" not in proof and events.count("native") == 1 and "measure" not in events
    events.clear()
    measurement = run("measurement")
    assert events.index("numeric") < events.index("measure") < len(events) - 1
    assert events.count("numeric") == 2 and events.count("native") == 13
    assert len(measurement["samples"]) == 10 and measurement["startup_logging"] == "0"
    probe.compare_workers(proof, measurement)


@pytest.mark.parametrize("damage", [{"algo": 999}, {"numerical_failure": True},
    {"descriptor_before_measurement": True}, {"descriptor_during_measurement": True}])
def test_failed_fresh_default_cannot_qualify(fake_gpu, damage):
    run, events = fake_gpu
    run("proof")
    events.clear()
    with pytest.raises(ValueError):
        run("measurement", **damage)
    if "descriptor_during_measurement" not in damage:
        assert "measure" not in events


def test_two_worker_deadline_never_resets(probe, launch, tmp_path, monkeypatch):
    ticks, waits = iter((100, 110, 270)), []
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    def spawn(command, **kwargs):
        assert command[7] == "proof"
        directory = Path(command[-1])
        kwargs["stdout"].write(native_log("proof"))
        def wait(timeout):
            waits.append(timeout)
            probe.write(directory / "worker.json", {"status": "passed", "pgid": os.getpgrp(),
                        "algorithm": probe.algorithm_metadata(native_log("proof"))})
            return 0
        return SimpleNamespace(wait=wait)
    monkeypatch.setattr(probe.subprocess, "Popen", spawn)
    with pytest.raises(TimeoutError, match="total two-worker"):
        probe.run(launch)
    assert waits == [150]
    record = json.loads((tmp_path / "forward-rank-0/summary.json").read_text())
    assert record["cleanup"] == "retained" and Path(record["scratch"]).exists()
