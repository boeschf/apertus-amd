"""Actual SSSGLU supervisor under CPU subprocess stubs and strict closed records."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/validation/chonk_sssglu_qualification.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def h(tmp_path, monkeypatch):
    from apertus_amd.config import resolve
    p = load("sssglu_lifecycle_fixture", SCRIPT)
    f = load("sssglu_lifecycle_records", ROOT / "tests/test_chonk_sssglu_records.py")
    settings = resolve(ROOT / "experiments/chonk-sssglu-qualification-i910bfcf936d3", site=ROOT/"config.toml",
                       submit=True, reservation="cscs", time="00:10:00")
    settings["scratch_root"] = str(tmp_path/"scratch")
    monkeypatch.chdir(tmp_path)
    for key, value in dict(settings["environment"], SLURM_JOB_ID="999999", SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NODEID="0", SLURM_NTASKS="4").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(p.os, "sched_getaffinity", lambda _: set(range(48)))
    counter = [100_000_000]
    def clock(): counter[0] += 100_000_000; return counter[0]
    monkeypatch.setattr(p, "time", SimpleNamespace(monotonic=lambda: 1., monotonic_ns=clock))
    mode = {"failure": None}; calls = []; waits = []; stops = []; quiet = []
    class Child:
        def __init__(self, argv, *, env, stdout, stderr, start_new_session):
            assert start_new_session is False and stderr == p.subprocess.STDOUT
            family = argv[argv.index("--family")+1]; self.family = family; self.pid = 10000+len(calls); self.waits = 0
            directory = Path(argv[argv.index("--directory")+1]); scratch = Path(env["TMPDIR"]).parent.parent
            row = f.worker(family, 0, str(scratch)); row.update(pid=self.pid, pgid=p.os.getpgrp(), driver_sha256=p.sha(SCRIPT.read_bytes()))
            row.update(started_ns=clock()); row["finished_ns"] = row["started_ns"]+1000
            if family == "weighted" and mode["failure"] == "schema": row["cases"][0]["metrics"]["candidate/oracle/dp"]["checked"] += 1
            (directory/"worker.json").write_text(json.dumps(row))
            if family == "weighted" and mode["failure"] == "warning": stdout.write("recompile_limit\n"); stdout.flush()
            calls.append(dict(family=family, env=env, directory=directory, row=row))
        def wait(self, timeout):
            self.waits += 1; waits.append((self.family, timeout))
            if self.family == "weighted" and mode["failure"] == "timeout" and self.waits == 1:
                raise p.subprocess.TimeoutExpired("owned worker", timeout)
            return 1 if self.family == "weighted" and mode["failure"] == "exit" else 0
        def terminate(self): stops.append(self.family)
        def kill(self): pytest.fail("cooperative fake worker must not require kill")
    monkeypatch.setattr(p.subprocess, "Popen", Child)
    def quiescent(): quiet.append(True); return mode["failure"] != "quiescence"
    monkeypatch.setattr(p, "quiescent", quiescent)
    actual_validate = p.validate_rank
    def validate_cpu_rank(record, **kwargs):
        # Only alias pytest's private scratch prefix to the native namespace.
        # Every worker is first validated against its actual temporary source
        # path by run(); all other strict rank gates remain active here.
        assert Path(record["scratch"]).is_relative_to(tmp_path/"scratch")
        mapped = deepcopy(record); mapped["scratch"] = "/dev/shm/apertus-amd/sssglu-999999-0-cpu"
        for family, row in zip(p.FAMILIES, mapped["workers"]):
            owned = Path(mapped["scratch"])/family/"component-source"
            row["adaptation"] = p.expected_adaptation(owned)
            for info in row["baseline_wrappers"].values(): info["original_path"] = str(owned/"baseline.py")
            row["helper_observer"]["path"] = row["adaptation"]["helper_path"]
        kwargs["expected_scratch"] = mapped["scratch"]
        return actual_validate(mapped, **kwargs)
    monkeypatch.setattr(p, "validate_rank", validate_cpu_rank)
    return p, settings, mode, calls, waits, stops, quiet, tmp_path


def test_actual_two_family_supervisor_closes_records_and_retains_private_caches(h):
    p, settings, _, calls, waits, stops, quiet, tmp = h
    p.run(settings)
    result = json.loads((tmp/"sssglu-0/summary.json").read_text())
    assert result["status"] == "passed" and result["cleanup"] == "retained" and not stops
    assert [c["family"] for c in calls] == ["plain", "weighted"]
    assert waits == [("plain",470),("weighted",470)] and len(quiet) == 3
    assert result["workers"] == [c["row"] for c in calls]
    for variable in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR"):
        paths = [c["env"][variable] for c in calls]
        assert len(set(paths)) == 2 and all(Path(path).is_dir() for path in paths)
    assert Path(result["scratch"]).is_dir()
    with pytest.raises(FileExistsError): p.run(settings)
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["exit", "timeout", "warning", "schema", "quiescence"])
def test_supervisor_failure_never_becomes_qualified_or_removes_scratch(h, failure):
    p, settings, mode, calls, waits, stops, quiet, tmp = h; mode["failure"] = failure
    with pytest.raises((ValueError,p.subprocess.TimeoutExpired)): p.run(settings)
    result = json.loads((tmp/"sssglu-0/summary.json").read_text())
    assert result["status"] == "unqualified" and result["cleanup"] == "retained" and "error" in result
    assert Path(result["scratch"]).is_dir()
    assert len(calls) == (1 if failure == "quiescence" else 2)
    assert stops == (["weighted"] if failure == "timeout" else [])
    if failure == "timeout":
        assert waits[-1] == ("weighted",5) and len(result["workers"]) == 1
    if failure == "warning": assert result["logs"]["weighted"]["warnings"] == ["recompile_limit"]
    assert len(waits) == len(calls)+(failure == "timeout")
