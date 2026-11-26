"""CPU gates for the CK backward and independent build diagnostics."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from apertus_amd.cli import plan
from apertus_amd.runs import edf, run


ROOT = Path(__file__).resolve().parents[1]
PROBE = runpy.run_path(ROOT / "experiments/validation/chonk_ck_attention_bwd_reuse.py")


def test_plan_and_guard():
    p = plan(ROOT / "experiments/chonk-ck-attention-bwd-reuse-i910bfcf936d3",
             site=ROOT / "config.toml", submit=True, gpus_per_node=1, cpus_per_task=48,
             cpu_bind="cores", time="00:25:00", exclude="nid002450", cache_image=True, reservation="cscs")
    PROBE["verify_launch"](p, {"SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NTASKS": "1"})
    assert p["resources"]["reservation"] == "cscs"
    assert p["resources"]["uenv_passthrough"] == "ignore"
    assert "--ntasks=1" in p["steps"][0]
    assert ":/apertus/cache" not in edf(p, Path("/run"))
    with pytest.raises(ValueError, match="launch blocked"):
        run(p)
    with pytest.raises(ValueError, match="unexpected one-rank"):
        PROBE["verify_launch"](p, {"SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NTASKS": "4"})


def test_object_record_requires_regular_elf(tmp_path):
    path = tmp_path / PROBE["OBJECT"]
    path.write_bytes(b"\x7fELF" + b"fixture" * 10)
    assert PROBE["object_record"](path)["bytes"] == 74
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="not a complete regular file"):
        PROBE["object_record"](alias)
    path.write_bytes(b"text" * 16)
    with pytest.raises(ValueError, match="not ELF"):
        PROBE["object_record"](path)


@pytest.mark.parametrize("defect", [None, "cold", "warm", "identity", "norms", "backend"])
def test_reuse_requires_exact_build_and_matching_results(defect):
    obj = {"bytes": 123, "sha256": "a" * 64}
    cold = {"object": obj, "norms": [1, 2, 3, 4], "backends": {"native": True}}
    warm = deepcopy(cold)
    cold_log = f"[CK-JIT] JIT-compiling blob: {PROBE['STEM']}.cpp\n[CK-JIT-BUILD] output : exact"
    warm_log = ""
    if defect == "cold":
        cold_log = cold_log.replace(PROBE["STEM"], "other")
    elif defect == "warm":
        warm_log = cold_log
    elif defect == "identity":
        warm["object"]["sha256"] = "b" * 64
    elif defect == "norms":
        warm["norms"][3] *= 2
    elif defect == "backend":
        warm["backends"]["native"] = False
    if defect:
        with pytest.raises(ValueError):
            PROBE["verify_reuse"](cold, warm, cold_log, warm_log, obj)
    else:
        PROBE["verify_reuse"](cold, warm, cold_log, warm_log, obj)


def test_comparison_requires_exact_cached_four_rank_plan():
    p = plan(ROOT / "experiments/chonk-ck-attention-bwd-compare-i910bfcf936d3",
             site=ROOT / "config.toml", submit=True, gpus_per_node=4, cpus_per_task=48,
             cpu_bind="map_ldom:0,1,2,3", time="00:25:00", exclude="nid002450",
             cache_image=True, reservation="cscs")
    verify = PROBE["verify_comparison_launch"]
    assert "--ntasks=4" in p["steps"][0] and "--gpu-bind=none" in p["steps"][0]
    assert ":/apertus/cache" not in edf(p, Path("/run"))
    e = {"SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NTASKS": "4", "SLURM_NODEID": "0"}
    for rank in range(4):
        assert verify(p, {**e, "SLURM_PROCID": str(rank), "SLURM_LOCALID": str(rank)}) == rank
    with pytest.raises(ValueError, match="launch blocked"):
        run(p)
    for bad in ({**p, "registry_image_path": None}, {**p, "resources": {**p["resources"], "cpu_bind": "cores"}}):
        with pytest.raises(ValueError, match="unexpected four-rank"):
            verify(bad, e)
    with pytest.raises(ValueError, match="unexpected four-rank"):
        verify(p, {**e, "SLURM_LOCALID": "3"})


def test_exact_mapping_requires_executable_original_inode(tmp_path):
    target = tmp_path / PROBE["OBJECT"]
    target.touch()
    info = target.stat()
    fields = f"{os.major(info.st_dev):x}:{os.minor(info.st_dev):x} {info.st_ino}"
    row = f"7f000000-7f001000 r-xp 00000000 {fields} {target}"
    mapped = PROBE["mapped_object"]
    assert mapped(target, row)
    assert not mapped(target, row + " (deleted)")
    assert not mapped(target, row.replace(f" {info.st_ino} ", f" {info.st_ino + 1} "))
    assert not mapped(target, row.replace("r-xp", "r--p"))


def test_seed_preserves_exact_object_and_refuses_existing_cache(tmp_path):
    source = tmp_path / PROBE["OBJECT"]
    original = b"\x7fELF" + b"native" * 15
    source.write_bytes(original)
    expected = PROBE["object_record"](source)
    target = tmp_path / "warm/xdg/te_ck_jit" / source.name
    identity = PROBE["seed_verified"](source, target, expected)
    assert PROBE["seed_identity"](target) == identity
    assert target.read_bytes() == original and target.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        PROBE["seed_verified"](source, target, expected)
    assert PROBE["seed_identity"](target) == identity
    source.write_bytes(original[:-1] + b"X")
    with pytest.raises(ValueError, match="source changed"):
        PROBE["seed_verified"](source, tmp_path / "warm-2/xdg/te_ck_jit" / source.name, expected)


def worker_report(expected):
    torch, hip = PROBE["VENDOR"]
    return dict(status="passed", key=PROBE["OBJECT"], image=PROBE["IMAGE"], visible_gpus=4, mapped=True,
                torch=torch, hip=hip, object=expected, norms=list(PROBE["REFERENCE_NORMS"]),
                finite_bf16=dict.fromkeys(PROBE["TENSOR_NAMES"], True),
                backends=dict(fused_attention_backend=PROBE["BACKEND"], use_fused_attention="1",
                              use_flash_attention="False", use_unfused_attention="False"))


@pytest.mark.parametrize("bad", ["backend", "fallback", "nonfinite", "norm", "version", "identity", "mapping"])
def test_comparison_worker_requires_qualified_numerics_and_route(bad):
    expected = {"bytes": 123, "sha256": "a" * 64}
    record = worker_report(expected)
    verify = PROBE["verify_comparison_worker"]
    verify(record, expected)
    if bad == "backend":
        record["backends"]["fused_attention_backend"] = "other"
    elif bad == "fallback":
        record["backends"]["use_unfused_attention"] = "True"
    elif bad == "nonfinite":
        record["finite_bf16"]["dv"] = False
    elif bad == "norm":
        record["norms"][3] *= 1.01
    elif bad == "version":
        record["hip"] = "other"
    elif bad == "mapping":
        record["mapped"] = False
    else:
        record["object"] = PROBE["REFERENCE"]
    with pytest.raises(ValueError, match="failed vendor, route, object or numerical qualification"):
        verify(record, expected)


@pytest.mark.parametrize("quiet", [True, False])
def test_failed_phase_records_object_only_after_quiescence(tmp_path, monkeypatch, quiet):
    phase = PROBE["comparison_phase"]
    g = phase.__globals__
    scratch = tmp_path / "cache"
    scratch.mkdir()
    evidence = {}
    target = scratch / "xdg/te_ck_jit" / PROBE["OBJECT"]

    def fail(*args, **kwargs):
        target.parent.mkdir()
        target.write_bytes(b"rejected object")
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(g["subprocess"], "run", fail)
    monkeypatch.setitem(g, "quiescent", lambda: quiet)
    with pytest.raises(subprocess.CalledProcessError if quiet else RuntimeError):
        phase(scratch, tmp_path, "build-a-cold", 30, evidence)
    assert evidence["quiescent"] is quiet
    if quiet:
        assert evidence["observed_object"] == PROBE["observed_object"](target)
        assert not evidence["observed_object"]["elf"]
        assert evidence["observed_object"]["bytes"] == 15
    else:
        assert "observed_object" not in evidence


@pytest.mark.parametrize("fail_first", [False, True])
def test_comparison_qualifies_distinct_builds_and_never_seeds_rejected_object(tmp_path, monkeypatch, fail_first):
    compare = PROBE["compare"]
    g = compare.__globals__
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(g["tempfile"], "mkdtemp", lambda **_: str(tmp_path / "scratch"))
    names = []

    def phase(directory, output, name, timeout, evidence):
        names.append(name)
        target = directory / "xdg/te_ck_jit" / PROBE["OBJECT"]
        log = ""
        if name.endswith("-cold"):
            target.parent.mkdir(parents=True)
            target.write_bytes(b"\x7fELF" + name.encode() * 10)
            log = (f"[CK-JIT] JIT-compiling blob: {PROBE['STEM']}.cpp\n"
                   f"[CK-JIT-BUILD] SUCCESS: {target}\n")
        evidence["observed_object"] = PROBE["observed_object"](target)
        if fail_first and name == "build-a-cold":
            raise ValueError("rejected cold object")
        expected = PROBE["object_record"](target)
        return {**worker_report(expected), "pid": len(names), "backward_seconds": 1.0}, log, expected

    monkeypatch.setitem(g, "comparison_phase", phase)
    if fail_first:
        with pytest.raises(ValueError, match="failed qualification"):
            compare({"scratch_root": "/dev/shm"})
    else:
        compare({"scratch_root": "/dev/shm"})
    summary = json.loads((tmp_path / "ck-attention-compare/summary.json").read_text())
    a, b = (summary["trials"][name] for name in PROBE["ATTEMPTS"])
    assert b["status"] == "passed"
    assert a["cold"]["observed_object"] != b["cold"]["observed_object"]
    assert not b["cold_matches_762106"]
    if fail_first:
        assert names == ["build-a-cold", "build-b-cold", "build-b-warm"]
        assert "warm" not in a and summary["status"] == "failed"
    else:
        assert names == ["build-a-cold", "build-a-warm", "build-b-cold", "build-b-warm"]
        assert a["status"] == summary["status"] == "passed"
        assert not summary["independent_builds_byte_identical"]
