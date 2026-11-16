"""CPU-only input, isolation and pinned-policy contracts; no GPU correctness claim."""

import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from apertus_amd import config
from apertus_amd.cli import plan
from apertus_amd.runs import edf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/validation/combined_image_attention.py"
EXPERIMENT = ROOT / "experiments/combined-aiter-attention-i910bfcf936d3"
ARCHIVE = ROOT / "artifacts/image-builds/primus268-uccl52d4184-aiter-20261113-monitored/inputs/source-checkout/Alps-Images/apps/apertus-2/tests/primus"


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "aiter", "triton", "megatron"}:
            pytest.fail("CPU contract imported vendor code")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("combined_image_attention", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings():
    return config.resolve(EXPERIMENT, site=ROOT / "config.toml", submit=True, reservation="cscs", time="00:15:00")


def environment(rank=0):
    return {"SLURM_PROCID": str(rank), "SLURM_LOCALID": str(rank), "SLURM_NODEID": "0",
            "SLURM_NTASKS": "4", "OMP_NUM_THREADS": "1", "MAX_JOBS": "2", "NCCL_P2P_DISABLE": "1",
            "OMPI_MCA_memory": "^patcher", "HWLOC_COMPONENTS": "-gl"}


@pytest.mark.parametrize("rank", range(4))
def test_local_rank_contract(probe, rank):
    assert probe.validate_launch(settings(), environment(rank)) == (rank, rank)


@pytest.mark.parametrize("section,key,value", [
    ("image", "sha256", "0" * 64), ("resources", "nodes", 2),
    ("resources", "gpus_per_node", 1), ("resources", "time", "00:20:00"),
    ("resources", "reservation", "other"), ("resources", "distribution", "cyclic"),
    ("procedure", "timeout", 900), ("procedure", "commands", [["bash", "with-reference-env.sh"]]),
])
def test_rejects_unreviewed_envelopes(probe, section, key, value):
    changed = deepcopy(settings())
    changed[section][key] = value
    with pytest.raises(ValueError, match="image/layout/procedure"):
        probe.validate_launch(changed, environment())


@pytest.mark.parametrize("key,value", [("SLURM_LOCALID", "3"), ("SLURM_NODEID", "1"),
                                     ("SLURM_NTASKS", "8"), ("OMP_NUM_THREADS", "48"),
                                     ("MAX_JOBS", "128"), ("HWLOC_COMPONENTS", "gl")])
def test_rejects_rank_or_environment_drift(probe, key, value):
    with pytest.raises(ValueError):
        probe.validate_launch(settings(), environment() | {key: value})


@pytest.mark.parametrize("flag", ["BEVERIN_T6_FP8", "BEVERIN_TURBO_GG", "BEVERIN_AITER_ATTN"])
def test_rejects_broad_reference_selection(probe, flag):
    with pytest.raises(ValueError, match="broad reference"):
        probe.validate_launch(settings(), environment() | {flag: "1"})


@pytest.fixture
def inputs(probe, tmp_path, monkeypatch):
    tests, seed = tmp_path / "tests", tmp_path / "seed"
    tests.mkdir()
    seed.mkdir()
    for name in probe.PINS:
        (tests / name).write_text("def validate_gradient_error(*args): return {}\n")
        monkeypatch.setitem(probe.PINS, name, probe.digest(tests / name))
    (seed / "module_aiter_core.so").write_bytes(b"CPU fixture, not an ELF")
    content = {"gpu_archs": "gfx942", "cu_num": 228, "packages": probe.VERSIONS,
               "modules": {"module_aiter_core.so": probe.digest(seed / "module_aiter_core.so")}}
    (seed / "seed.json").write_text(json.dumps(content))
    monkeypatch.setattr(probe, "TESTS", tests)
    monkeypatch.setattr(probe, "SEED", seed)
    monkeypatch.setattr(probe, "SEED_HASH", probe.digest(seed / "seed.json"))
    return tests, seed


def test_pinned_inputs_match_without_vendor_import(probe, inputs):
    assert probe.verified_inputs()["seed"]["cu_num"] == 228


@pytest.mark.parametrize("target", ["script", "seed", "core"])
def test_input_drift_rejected_before_gpu(probe, inputs, target):
    tests, seed = inputs
    path = {"script": tests / "check-stack.py", "seed": seed / "seed.json",
            "core": seed / "module_aiter_core.so"}[target]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        probe.verified_inputs()


def test_failed_preimport_gate_retains_result_without_scratch(probe, inputs, tmp_path, monkeypatch):
    tests, _ = inputs
    (tests / "attention-accuracy.py").write_text("changed")
    monkeypatch.chdir(tmp_path)
    for key, value in environment().items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="mismatch"):
        probe.run(settings())
    record = json.loads((tmp_path / "attention-rank-0.json").read_text())
    assert record["status"] == "failed" and "scratch" not in record


def test_plan_is_pure_and_has_no_reference_bundle(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("plan attempted subprocess or write")
    with monkeypatch.context() as guard:
        guard.setattr(subprocess, "Popen", refuse)
        guard.setattr(Path, "write_text", refuse)
        guard.setattr(Path, "write_bytes", refuse)
        guard.setattr(Path, "mkdir", refuse)
        value = settings()
        plan(EXPERIMENT, submit=True, reservation="cscs", time="00:15:00")
    text = edf(value, "/future-run")
    assert 'HWLOC_COMPONENTS = "-gl"' in text
    assert 'com.hooks.cxi.enabled = "false"' in text and 'com.hooks.aws_ofi_nccl.enabled = "false"' in text
    assert "BEVERIN_T6_FP8" not in text and "with-reference-env" not in text


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("fail", [False, True])
def test_same_process_local_gpu_and_private_cleanup_with_fake_vendor(
        probe, inputs, tmp_path, monkeypatch, rank, fail):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(probe.os, "environ", environment(rank) | {"SLURM_JOB_ID": "123"})
    state = SimpleNamespace(device=None, checked=False)
    def select(device):
        state.device = device
    fake_torch = SimpleNamespace(version=SimpleNamespace(hip="CPU fixture"), set_num_threads=lambda n: None,
        cuda=SimpleNamespace(device_count=lambda: 4, set_device=select, current_device=lambda: state.device,
            get_device_properties=lambda device: SimpleNamespace(gcnArchName="gfx942", multi_processor_count=228)))
    fake_aiter = SimpleNamespace(flash_attn_func=object())
    guarded = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake_torch
        if name == "aiter":
            return fake_aiter
        return guarded(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: probe.VERSIONS[name])
    def numerical(torch, operation, policy):
        assert torch is fake_torch and operation is fake_aiter.flash_attn_func
        assert state.device == rank
        cache = Path(probe.os.environ["AITER_JIT_DIR"])
        assert cache.is_relative_to(Path(value["scratch_root"])) and cache != probe.SEED
        assert probe.digest(cache / "module_aiter_core.so") == probe.digest(probe.SEED / "module_aiter_core.so")
        state.checked = True
        if fail:
            raise ValueError("synthetic numerical failure")
        return {"fake_GPU_contract_only": True}
    monkeypatch.setattr(probe, "attention", numerical)
    if fail:
        with pytest.raises(ValueError, match="synthetic numerical failure"):
            probe.run(value)
    else:
        probe.run(value)
    record = json.loads((tmp_path / f"attention-rank-{rank}.json").read_text())
    assert state.checked and record["selected_device"] == rank
    assert record["status"] == ("failed" if fail else "passed")
    assert record["scratch_cleanup"] == "removed" and not Path(record["scratch"]).exists()
    assert record["affinity"]["initial"] == record["affinity"]["final"]


def test_accuracy_policy_is_the_existing_unmodified_image_policy(probe):
    # Only this pure-math helper is loaded; no source or native package is executed.
    for name, digest in probe.PINS.items():
        assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == digest
    path = ARCHIVE / "attention-accuracy.py"
    spec = importlib.util.spec_from_file_location("pinned_attention_policy_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.validate_gradient_error(0.005, 0.001, 0.001, 0.001)["max_abs_bound"] == 0.01
    for values in ((0.02, 0.001, 0.001, 0.001), (0.001, 0.001, 0.1, 0.001),
                   (0.001, 0.001, 0.001, 0.1), (float("nan"), 0.001, 0.001, 0.001)):
        with pytest.raises(AssertionError):
            module.validate_gradient_error(*values)
