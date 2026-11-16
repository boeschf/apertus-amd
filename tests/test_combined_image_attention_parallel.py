"""CPU contracts for the separately bounded parallel-JIT attention follow-up, not GPU correctness."""

import ast
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
SCRIPT = ROOT / "experiments/validation/combined_image_attention_parallel.py"
ORIGINAL = ROOT / "experiments/validation/combined_image_attention.py"
EXPERIMENT = ROOT / "experiments/combined-aiter-attention-parallel-i910bfcf936d3"
PREP = ROOT / "artifacts/investigations/performance-next/restart-20261113/attention-parallel-preparation"
AUDIT = PREP.parent / "attention-next-review"


def settings():
    return config.resolve(EXPERIMENT, site=ROOT / "config.toml", submit=True, reservation="cscs", time="00:15:00")


def environment(rank=0):
    return {"SLURM_PROCID": str(rank), "SLURM_LOCALID": str(rank), "SLURM_NODEID": "0",
            "SLURM_NTASKS": "4", "SLURM_JOB_ID": "123", "OMP_NUM_THREADS": "1", "MAX_JOBS": "8",
            "AITER_LOG_MORE": "1", "NCCL_P2P_DISABLE": "1", "OMPI_MCA_memory": "^patcher",
            "HWLOC_COMPONENTS": "-gl"}


@pytest.fixture
def probe(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "aiter", "triton", "megatron"}:
            pytest.fail("CPU contract imported vendor code")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("parallel_attention_cpu", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rank", range(4))
def test_exact_rank_envelope(probe, rank):
    assert probe.validate_launch(settings(), environment(rank)) == (rank, rank)


@pytest.mark.parametrize("section,key,value", [
    ("image", "sha256", "0" * 64), ("resources", "nodes", 2),
    ("resources", "gpus_per_node", 1), ("resources", "cpus_per_node", 96),
    ("resources", "cpu_bind", "none"), ("resources", "distribution", "block:cyclic"),
    ("resources", "time", "00:20:00"), ("resources", "reservation", "other"),
    ("procedure", "timeout", 900), ("procedure", "commands", [["bash", "with-reference-env.sh"]]),
])
def test_unreviewed_envelope_rejected(probe, section, key, value):
    changed = deepcopy(settings())
    changed[section][key] = value
    with pytest.raises(ValueError, match="image/layout/procedure"):
        probe.validate_launch(changed, environment())


@pytest.mark.parametrize("key,value", [
    ("MAX_JOBS", "2"), ("MAX_JOBS", "0"), ("MAX_JOBS", "16"), ("MAX_JOBS", "auto"),
    ("OMP_NUM_THREADS", "48"), ("AITER_LOG_MORE", "0"), ("AITER_LOG_MORE", "2"),
    ("PREBUILD_THREAD_NUM", "1"), ("PREBUILD_THREAD_NUM", ""), ("PREBUILD_THREAD_NUM", "2"),
    ("SLURM_LOCALID", "3"), ("SLURM_NODEID", "1"), ("SLURM_NTASKS", "8"),
    ("HWLOC_COMPONENTS", "gl"), ("BEVERIN_AITER_ATTN", "1"), ("BEVERIN_T6_FP8", "1"),
])
def test_environment_mismatch_precedes_vendor_import(probe, key, value, monkeypatch):
    monkeypatch.setattr(probe.os, "environ", environment() | {key: value})
    with pytest.raises(ValueError):
        probe.run(settings())


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
    cpp = tmp_path / "cpp_extension.py"
    cpp.write_text("# source fixture, not executed\n")
    monkeypatch.setattr(probe, "TESTS", tests)
    monkeypatch.setattr(probe, "SEED", seed)
    monkeypatch.setattr(probe, "SEED_HASH", probe.digest(seed / "seed.json"))
    monkeypatch.setattr(probe, "CPP_HASH", probe.digest(cpp))
    monkeypatch.setattr(probe.importlib.metadata, "distribution", lambda name: SimpleNamespace(locate_file=lambda path: cpp))
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: probe.VERSIONS[name])
    monkeypatch.setattr(probe.os, "environ", environment())
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda pid: set(range(48)))
    monkeypatch.chdir(tmp_path)
    return tests, seed, cpp


@pytest.mark.parametrize("target", ["script", "seed", "core", "cpp", "version", "affinity"])
def test_preimport_failure_retains_truthful_record_without_scratch(probe, inputs, target, monkeypatch):
    tests, seed, cpp = inputs
    if target == "version":
        monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: "wrong")
    elif target == "affinity":
        monkeypatch.setattr(probe.os, "sched_getaffinity", lambda pid: {0})
    else:
        {"script": tests / "attention-accuracy.py", "seed": seed / "seed.json",
         "core": seed / "module_aiter_core.so", "cpp": cpp}[target].write_bytes(b"changed")
    with pytest.raises(ValueError):
        probe.run(settings())
    record = json.loads(Path("attention-rank-0.json").read_text())
    assert record["status"] == "failed" and "scratch" not in record


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("failure", [None, "numerical", "device-count", "device-selection"])
def test_private_cache_local_gpu_and_cleanup(probe, inputs, tmp_path, monkeypatch, rank, failure):
    value = settings()
    value["scratch_root"] = str(tmp_path / "scratch")
    monkeypatch.setattr(probe.os, "environ", environment(rank))
    state = SimpleNamespace(device=None, checked=False)
    def select(device):
        state.device = device if failure != "device-selection" else (device + 1) % 4
    fake_torch = SimpleNamespace(version=SimpleNamespace(hip="CPU fixture"), set_num_threads=lambda n: None,
        cuda=SimpleNamespace(device_count=lambda: 1 if failure == "device-count" else 4,
            set_device=select, current_device=lambda: state.device,
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
    def numerical(torch, operation, policy):
        assert torch is fake_torch and operation is fake_aiter.flash_attn_func
        assert state.device == rank
        cache = Path(probe.os.environ["AITER_JIT_DIR"])
        assert cache.is_relative_to(Path(value["scratch_root"])) and cache != probe.SEED
        assert probe.digest(cache / "module_aiter_core.so") == probe.digest(probe.SEED / "module_aiter_core.so")
        assert len(list(cache.iterdir())) == 1
        state.checked = True
        if failure == "numerical":
            raise ValueError("synthetic numerical failure")
        return {"fake_GPU_contract_only": True}
    monkeypatch.setattr(probe, "attention", numerical)
    if failure:
        with pytest.raises(ValueError):
            probe.run(value)
    else:
        probe.run(value)
    record = json.loads(Path(f"attention-rank-{rank}.json").read_text())
    assert state.checked == (failure not in ("device-count", "device-selection"))
    assert record["status"] == ("failed" if failure else "passed")
    assert record["scratch_cleanup"] == "removed" and not Path(record["scratch"]).exists()
    assert record["jit_environment"] == {"MAX_JOBS": "8", "OMP_NUM_THREADS": "1", "AITER_LOG_MORE": "1"}
    if not failure:
        assert record["selected_device"] == rank and record["seed_unchanged"]
        assert record["affinity"]["initial"] == record["affinity"]["final"]


def function_text(path, name):
    text = path.read_text()
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(text, node)


def test_original_and_numerical_policy_are_unchanged():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == "5e504476629da255c08a5c540f6b3c230fd6879c741d903750e41d0649a38920"
    for name in ("attention", "verified_inputs", "digest"):
        assert function_text(SCRIPT, name) == function_text(ORIGINAL, name)
    assert (PREP / "original/combined_image_attention.py").read_bytes() == ORIGINAL.read_bytes()


def test_retained_ninja_consumer_pin(probe):
    path = AUDIT / "upstream/aiter/aiter/jit/utils/cpp_extension.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == probe.CPP_HASH
    text = path.read_text()
    assert 'max_jobs = os.environ.get("MAX_JOBS")' in text
    assert 'command.extend(["-j", str(num_workers)])' in text


def test_plan_is_pure_with_original_bounds_and_disabled_injection(monkeypatch):
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
    for setting in ('HWLOC_COMPONENTS = "-gl"', 'MAX_JOBS = "8"', 'AITER_LOG_MORE = "1"',
                    'com.hooks.cxi.enabled = "false"', 'com.hooks.aws_ofi_nccl.enabled = "false"'):
        assert setting in text
    assert "with-reference-env" not in text and "BEVERIN_T6_FP8" not in text
    assert value["procedure"]["timeout"] == 720 and value["resources"]["time"] == "00:15:00"


@pytest.fixture
def transport():
    spec = importlib.util.spec_from_file_location("attention_parallel_transport_cpu", PREP / "transport.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transport_is_preparation_only(transport, monkeypatch):
    monkeypatch.setattr(transport.sys, "executable", transport.PYTHON)
    monkeypatch.delenv("APERTUS_ATTENTION_PARALLEL_APPROVAL", raising=False)
    with pytest.raises(RuntimeError, match="PREPARATION ONLY"):
        transport.main("sbatch")


def test_transport_placement(transport):
    result = transport.forwarded("sbatch", ["--parsable", "/future-run/job.sh"])
    assert result[:3] == ["/usr/bin/sbatch", "--uenv-passthrough=ignore", "--nodelist=nid002458"]
    assert "nid002458" not in result[3]
    for node in ("nid002412", "nid002454", "nid002456", "nid002744", "nid002760", "nid002778"):
        assert node in result[3]
    assert {"741627", "741609", "738614", "737047"} <= transport.PROTECTED_JOBS
    assert transport.forwarded("srun", ["--jobid=999999"])[1] == "--uenv-passthrough=ignore"


@pytest.mark.parametrize("tool,arg", [("sbatch", "--exclude=nid1"), ("sbatch", "--nodelist=nid2"),
                                     ("sbatch", "-w"), ("srun", "--uenv-passthrough=all")])
def test_transport_rejects_second_owner(transport, tool, arg):
    with pytest.raises(ValueError):
        transport.forwarded(tool, [arg])


@pytest.fixture
def authorized_stub(transport, tmp_path, monkeypatch):
    # Synthetic CPU approval fixture only; cannot execute any scheduler command.
    monkeypatch.setattr(transport, "HERE", tmp_path)
    monkeypatch.setattr(transport.sys, "executable", transport.PYTHON)
    (tmp_path / "manifest.json").write_text("{}\n")
    evidence = tmp_path / "execution-fixture"
    evidence.mkdir()
    approval = {"execution_authorized": True, "maximum_submissions": 1, "experiment": transport.EXPERIMENT,
                "node": "nid002458", "execution_artifacts": str(evidence),
                "preparation_manifest_sha256": hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()}
    path = tmp_path / "synthetic-approval.json"
    path.write_text(json.dumps(approval))
    monkeypatch.setattr(transport.os, "environ", {"APERTUS_ATTENTION_PARALLEL_APPROVAL": str(path)})
    calls = []
    monkeypatch.setattr(transport.os, "execv", lambda executable, args: calls.append((executable, args)))
    return approval, path, evidence, calls


def test_transport_one_shot_with_scheduler_exec_stub(transport, authorized_stub, monkeypatch):
    _, _, evidence, calls = authorized_stub
    monkeypatch.setattr(transport.sys, "argv", ["sbatch", "--parsable", "/future-run/job.sh"])
    transport.main("sbatch")
    assert len(calls) == 1 and calls[0][0] == "/usr/bin/sbatch"
    assert (evidence / "sbatch-intent.json").exists()
    with pytest.raises(FileExistsError):
        transport.main("sbatch")
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("maximum_submissions", 2), ("node", "nid002456"),
                                         ("experiment", "old"), ("execution_authorized", False),
                                         ("preparation_manifest_sha256", "0" * 64)])
def test_transport_bad_approval_cannot_exec(transport, authorized_stub, field, value):
    approval, path, _, calls = authorized_stub
    path.write_text(json.dumps(approval | {field: value}))
    with pytest.raises(ValueError, match="approval"):
        transport.main("sbatch")
    assert not calls


@pytest.mark.parametrize("job,node", [("741627", "nid002458"), ("741609", "nid002458"),
                                    ("999999", "nid002456"), ("", "nid002458")])
def test_transport_protected_job_or_wrong_node_cannot_exec(transport, authorized_stub, monkeypatch, job, node):
    _, _, _, calls = authorized_stub
    transport.os.environ.update(SLURM_JOB_ID=job, SLURM_JOB_NODELIST=node)
    monkeypatch.setattr(transport.sys, "argv", ["srun", f"--jobid={job}"])
    with pytest.raises(ValueError, match="fresh owned"):
        transport.main("srun")
    assert not calls


def test_transport_owned_step_uses_direct_srun_stub(transport, authorized_stub, monkeypatch):
    _, _, _, calls = authorized_stub
    transport.os.environ.update(SLURM_JOB_ID="999999", SLURM_JOB_NODELIST="nid002458")
    monkeypatch.setattr(transport.sys, "argv", ["srun", "--jobid=999999", "--ntasks=4"])
    transport.main("srun")
    assert len(calls) == 1 and calls[0][0] == "/usr/bin/srun"


def test_launch_environment_cleans_inherited_scheduler_context():
    spec = importlib.util.spec_from_file_location("parallel_launch_cpu", PREP / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    value = launch.command_environment({"SLURM_JOB_ID": "741609", "SBATCH_NODELIST": "bad", "SRUN_CPUS_PER_TASK": "1",
                                        "PATH": "/usr/bin", "APERTUS_ATTENTION_PARALLEL_APPROVAL": "future"})
    assert not any(key.startswith(("SLURM_", "SBATCH_", "SRUN_")) for key in value)
    assert value["PATH"].startswith(str(PREP / "bin"))
    assert value["APERTUS_ATTENTION_PARALLEL_APPROVAL"] == "future"
