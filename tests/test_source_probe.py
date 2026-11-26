"""CPU-only tests for bounded probes on a committed source snapshot."""

from pathlib import Path
import sys
import tomllib

import pytest

from apertus_amd import config, runs, runtime
from apertus_amd.artifacts import sha256


@pytest.fixture
def experiment(tmp_path):
    path = tmp_path / "fixture-probe" / "experiment.toml"
    path.parent.mkdir()
    path.write_text(
        'entrypoint = "tests/fixture.py"\nargs = ["--case", "codec"]\nnodes = 1\n'
        '[source]\nrepository = "unused"\ncommit = "' + "1" * 40 + '"\n'
        '[image]\nstore = "apertus-amd-images"\nsha256 = "' + "2" * 64 + '"\n'
        '[procedure]\nkind = "probe"\ntimeout = 60\n'
    )
    return path


def test_committed_python_probe_passes_exact_arguments(experiment):
    settings = config.resolve(experiment, job_id="0")
    assert settings["cache"]["mode"] == "cold"
    [phase] = config.phases(settings, Path("/output"))
    assert phase == {"name": "probe", "output": "/output", "extra_args": []}
    assert runtime.training_command(settings, Path("/output"), Path("/source"), []) == [
        sys.executable, "-m", "apertus_amd.entrypoint", "/output", "/source",
        "tests/fixture.py", "--case", "codec",
    ]


def test_probe_edf_mounts_readonly_source_without_shared_cache(experiment):
    settings = config.resolve(experiment, job_id="0")
    settings["source_snapshot"] = {"directory": "/frozen/source"}
    edf = tomllib.loads(runs.edf(settings, Path("/run")))
    assert "/frozen/source:/apertus/source:ro" in edf["mounts"]
    assert not any(":/apertus/cache" in mount for mount in edf["mounts"])


def test_rank_rejects_a_modified_frozen_manifest(tmp_path):
    manifest = tmp_path / "source" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text('{"commit":"first"}')
    settings = {
        "tooling": {}, "validation": None,
        "source_snapshot": {"manifest_sha256": sha256(manifest)},
    }
    runtime.verify_source_manifest(manifest.parent, settings)
    manifest.write_text('{"commit":"changed"}')
    with pytest.raises(ValueError, match="frozen source manifest changed"):
        runtime.verify_source_manifest(manifest.parent, settings)


@pytest.mark.parametrize("overrides", [{"cache_mode": "warm"}, {"compare_run": "/old"}])
def test_probe_rejects_shared_cache_and_comparison(experiment, overrides):
    with pytest.raises(ValueError, match="no comparison or shared cache"):
        config.resolve(experiment, job_id="0", **overrides)


def test_probe_needs_timeout(experiment):
    experiment.write_text(experiment.read_text().replace("timeout = 60\n", "timeout = 0\n"))
    with pytest.raises(ValueError, match="positive timeout"):
        config.resolve(experiment, job_id="0")


def test_startup_probe_renders_marked_launcher_without_workload_phases(experiment):
    experiment.write_text(experiment.read_text() + "startup_only = true\n")
    settings = config.resolve(experiment, job_id="1")
    assert config.phases(settings, Path("/output")) == []
    step = runs.step_command(settings, Path("/run"), cpus=48)
    assert step[-1] == "--startup-trace"
    assert "APERTUS_STARTUP_SHELL" in step[step.index("bash") + 2]
    assert settings["cache"]["timeout"] == 300


@pytest.mark.parametrize("extra", ["startup_only = false", "startup_only = true\nmemory_telemetry = true",
                                  "startup_only = true\ncheckpoint = true"])
def test_startup_probe_rejects_conflicting_modes(experiment, extra):
    experiment.write_text(experiment.read_text() + extra + "\n")
    with pytest.raises(ValueError, match="startup_only"):
        config.resolve(experiment, job_id="1")


def test_traced_training_probe_keeps_workload_phases(experiment):
    experiment.write_text(experiment.read_text() + "startup_trace = true\nmemory_telemetry = true\n")
    settings = config.resolve(experiment, job_id="1")
    assert config.phases(settings, Path("/output")) == [{"name": "probe", "output": "/output", "extra_args": []}]
    step = runs.step_command(settings, Path("/run"), cpus=48)
    assert step[-1] == "--startup-trace"
    assert "APERTUS_STARTUP_SHELL" in step[step.index("bash") + 2]


@pytest.mark.parametrize(("kind", "value"), [("probe", "false"), ("train", "true")])
def test_startup_trace_rejects_invalid_values_and_kinds(experiment, kind, value):
    experiment.write_text(experiment.read_text().replace('kind = "probe"', f'kind = "{kind}"')
                          + f"startup_trace = {value}\n")
    with pytest.raises(ValueError, match="startup_trace"):
        config.resolve(experiment, job_id="1")


def test_gpu_identity_timeout_is_a_separate_probe_setting(experiment):
    experiment.write_text(experiment.read_text() + "gpu_identity_timeout = 600\n")
    settings = config.resolve(experiment, job_id="1")
    assert settings["procedure"]["gpu_identity_timeout"] == 600
    assert settings["procedure"]["timeout"] == 60
    assert settings["cache"]["timeout"] == 300


@pytest.mark.parametrize(("kind", "value"), [("probe", "0"), ("probe", "true"), ("probe", "1.5"), ("train", "600")])
def test_gpu_identity_timeout_rejects_invalid_values_and_kinds(experiment, kind, value):
    experiment.write_text(experiment.read_text().replace('kind = "probe"', f'kind = "{kind}"')
                          + f"gpu_identity_timeout = {value}\n")
    with pytest.raises(ValueError, match="gpu_identity_timeout"):
        config.resolve(experiment, job_id="1")


@pytest.mark.parametrize("bundles", [[], ["/must-not-publish"]])
def test_probe_cannot_publish_even_with_contributions(tmp_path, monkeypatch, bundles):
    session = tmp_path / "session"
    session.mkdir()

    def child(*args, **kwargs):
        pytest.fail("a probe must never run a publication worker")

    monkeypatch.setattr(runtime, "child", child)
    result = runtime.complete(
        {
            "validation": None, "cache": {"timeout": 5, "max_gb": 1},
            "procedure": {"kind": "probe"}, "resources": {"time": "00:01:00"},
        },
        tmp_path, tmp_path, session, rank=0, source=tmp_path, env={},
        gathered=[{"bundles": bundles, "errors": []}], result={"cache": {}},
    )
    expected = "failed" if bundles else "not applicable (probe)"
    assert result["cache"]["publication"] == expected
