"""Exercise advisory sampling and bounded scheduler access without Slurm or GPUs."""

import json
import sys
from types import SimpleNamespace

import pytest

from apertus_amd import node_screen as screen


def put(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def kernel(tmp_path):
    proc, sysfs = tmp_path / "proc", tmp_path / "sys"
    put(proc, "buddyinfo", "Node 0, zone Normal 100 0 0 0 0 0 0 0 0 2 1\n"
        "Node 1, zone Normal 200 0 0 0 0 0 0 0 0 0 1\n"
        "Node 2, zone Normal 1 0 0 0 0 0 0 0 0 0 0\n")
    put(proc, "meminfo", "MemTotal: 10000 kB\nMemFree: 8000 kB\nMemAvailable: 9000 kB\n"
        "Slab: 30 kB\nUnevictable: 12 kB\n")
    put(proc, "uptime", "1234.5 10000\n")
    for node in range(3):
        put(sysfs, f"devices/system/node/node{node}/meminfo",
            f"Node {node} MemTotal: 4000 kB\nNode {node} MemFree: 3000 kB\n"
            f"Node {node} Slab: 10 kB\nNode {node} Unevictable: 4 kB\n")
    for index, node in enumerate((0, 1)):
        device = f"bus/pci/devices/0000:{index:02x}:00.0"
        put(sysfs, f"{device}/vendor", "0x1002\n")
        put(sysfs, f"{device}/class", "0x030200\n")
        put(sysfs, f"{device}/numa_node", f"{node}\n")
    # AMD host bridges are not GPUs.
    put(sysfs, "bus/pci/devices/0000:ff:00.0/vendor", "0x1002\n")
    put(sysfs, "bus/pci/devices/0000:ff:00.0/class", "0x060000\n")
    put(sysfs, "kernel/mm/transparent_hugepage/hpage_pmd_size", "2097152\n")
    put(sysfs, "kernel/mm/transparent_hugepage/enabled", "always [madvise] never\n")
    put(sysfs, "kernel/mm/transparent_hugepage/defrag", "always defer [madvise] never\n")
    return proc, sysfs


def test_raw_counts_page_size_and_target_extent_math():
    zones = screen.parse_buddyinfo("Node 7, zone Normal 9 8 7 6 5\n", 65536, 262144)
    assert zones == [{"numa_node": 7, "zone": "Normal", "order_counts": [9, 8, 7, 6, 5],
                      "free_bytes": 65536 * (9 + 16 + 28 + 48 + 80), "target_order": 2,
                      "target_sized_blocks": 7 + 12 + 20,
                      "high_order_free_bytes": 262144 * (7 + 12 + 20)}]


def test_memory_bytes_do_not_mislabel_unitless_hugepage_counts():
    assert screen.parse_meminfo("MemFree: 100 kB\nHugePages_Total: 4\n") == {"MemFree": 102400}


@pytest.mark.parametrize("target", [0, 1024, 4097, 3 * 4096])
def test_invalid_extent_is_not_silently_rounded(target):
    with pytest.raises(ValueError, match="power-of-two"):
        screen.parse_buddyinfo("", 4096, target)


def test_snapshot_is_read_only_and_ranks_worst_gpu_local_domain(kernel, monkeypatch):
    proc, sysfs = kernel
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    before = {path: path.read_bytes() for root in kernel for path in root.rglob("*") if path.is_file()}
    sample = screen.collect(proc=proc, sysfs=sysfs, page_bytes=4096)
    assert sample["base_page_bytes"] == 4096
    assert sample["target_bytes"] == 2097152
    assert sample["target_source"] == "kernel hpage_pmd_size"
    assert sample["gpu_numa_nodes"] == [0, 1]
    assert len(sample["gpus"]) == 2
    assert sample["score"]["worst_target_sized_blocks"] == 2
    assert sample["score"]["worst_numa_nodes"] == [1]
    assert sample["buddy_zones"][0]["order_counts"] == [100, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1]
    assert sample["meminfo_bytes"]["MemAvailable"] == 9000 * 1024
    assert sample["meminfo_bytes"]["Slab"] == 30 * 1024
    assert sample["numa_memory"]["0"]["available_bytes"] is None
    assert sample["numa_memory"]["0"]["free_bytes"] == 3000 * 1024
    assert sample["numa_memory"]["0"]["meminfo_bytes"]["Unevictable"] == 4 * 1024
    assert sample["uptime_seconds"] == 1234.5
    assert sample["slurm_job_id"] == "123"
    assert sample["started_at"] <= sample["finished_at"]
    assert sample["raw"][str(proc / "buddyinfo")] == (proc / "buddyinfo").read_text()
    assert sample["errors"] == {}
    assert "not an admission gate" in sample["advisory"]
    assert before == {path: path.read_bytes() for root in kernel for path in root.rglob("*") if path.is_file()}


def test_unknown_and_partial_topology_are_not_ranked(kernel):
    proc, sysfs = kernel
    put(sysfs, "bus/pci/devices/0000:01:00.0/numa_node", "-1\n")
    sample = screen.collect(proc=proc, sysfs=sysfs)
    assert sample["score"]["worst_target_sized_blocks"] is None
    assert "incomplete" in sample["score"]["reason"]
    sample = screen.collect(proc=proc, sysfs=sysfs, gpu_nodes=[0, 1, 2])
    assert sample["score"]["worst_target_sized_blocks"] == 0
    assert sample["gpu_topology_source"] == "explicit"
    assert screen.fragmentation_score(sample["buddy_zones"], [3])["worst_target_sized_blocks"] is None
    assert screen.fragmentation_score(sample["buddy_zones"], [])["worst_target_sized_blocks"] is None


def test_absent_kernel_counters_retain_errors_and_unknown_score(tmp_path):
    sample = screen.collect(proc=tmp_path / "proc", sysfs=tmp_path / "sys")
    assert sample["buddy_zones"] == []
    assert sample["score"]["worst_target_sized_blocks"] is None
    assert sample["uptime_seconds"] is None
    assert sample["target_source"] == "2 MiB fallback"
    assert sample["errors"]


def test_malformed_buddyinfo_retains_evidence(kernel):
    proc, sysfs = kernel
    put(proc, "buddyinfo", "unexpected kernel output\n")
    sample = screen.collect(proc=proc, sysfs=sysfs)
    assert sample["score"]["worst_target_sized_blocks"] is None
    assert "malformed" in sample["errors"][str(proc / "buddyinfo")]
    assert sample["raw"][str(proc / "buddyinfo")] == "unexpected kernel output\n"


def test_normal_zone_only_prevents_dma_counts_masking_fragmentation():
    zones = screen.parse_buddyinfo("Node 0, zone DMA32 0 0 999\nNode 0, zone Normal 1 0 0\n", 4096, 16384)
    assert screen.fragmentation_score(zones, [0])["worst_target_sized_blocks"] == 0


def sample_for(node, score):
    return {"hostname": node, "slurm_job_id": "123", "score": {"worst_target_sized_blocks": score}}


def fake_scheduler(monkeypatch, nodes, states=None, samples=None):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    calls = []

    def command(argv, timeout):
        calls.append((argv, timeout))
        result = {"command": argv, "returncode": 0, "stdout": "", "stderr": ""}
        if argv[:3] == ["scontrol", "show", "reservation"]:
            result["stdout"] = "ReservationName=cscs Nodes=test[01-10] State=ACTIVE"
        elif argv[:3] == ["scontrol", "show", "hostnames"]:
            result["stdout"] = "\n".join(nodes)
        elif argv[:3] == ["scontrol", "show", "node"]:
            state = (states or {}).get(argv[3], "IDLE")
            if isinstance(state, dict):
                result.update(state)
            else:
                result["stdout"] = f"NodeName={argv[3]} State={state}"
        elif argv[0] == "srun":
            node = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--nodelist="))
            sample = (samples or {}).get(node, sample_for(node, 0))
            if "returncode" in sample or "error" in sample:
                result.update(sample)
            else:
                result["stdout"] = json.dumps(sample)
        else:
            pytest.fail(f"unexpected command {argv}")
        return result

    monkeypatch.setattr(screen, "bounded_command", command)
    return calls


def test_survey_keeps_poor_nodes_and_records_busy_raced_and_unreachable(monkeypatch, tmp_path):
    nodes = ["good", "poor", "busy", "drain", "raced", "unknown", "unreachable"]
    calls = fake_scheduler(monkeypatch, nodes,
                           states={"busy": "MIXED", "drain": "IDLE+DRAIN",
                                   "unreachable": {"returncode": 1, "stderr": "unavailable"}},
                           samples={"good": sample_for("good", 100), "unknown": sample_for("unknown", None),
                                    "raced": {"returncode": 1, "stderr": "resources unavailable"}})
    output = tmp_path / "evidence"
    report = screen.survey(output, account="staff", partition="gpu", timeout=2, budget=30)
    assert json.loads((output / "survey.json").read_text()) == report
    assert [item["node"] for item in report["ranking"]] == ["good", "poor", "unknown"]
    status = {entry["node"]: entry["status"] for entry in report["nodes"]}
    assert status == {"good": "sampled", "poor": "sampled", "unknown": "sampled",
                      "busy": "skipped_busy_or_unavailable", "drain": "skipped_busy_or_unavailable",
                      "raced": "skipped_unreachable_or_raced", "unreachable": "skipped_unreachable"}
    runs = [argv for argv, timeout in calls if argv[0] == "srun"]
    assert len(runs) == 4
    for argv in runs:
        assert "--reservation=cscs" in argv
        assert "--immediate=1" in argv
        assert "--gres=none" in argv
        assert "--mem=128M" in argv
        assert "--time=00:01:00" in argv
        assert "--nice=10000" in argv
        assert "--account=staff" in argv
        assert "--partition=gpu" in argv
        assert "--exclusive" not in argv and "--overlap" not in argv
        assert argv[-5:] == [sys.executable, "-B", "-m", "apertus_amd.node_screen", "sample"]
    assert all(0 < timeout <= 2 for argv, timeout in calls)
    assert not {argv[0] for argv, timeout in calls} - {"scontrol", "srun"}


@pytest.mark.parametrize("result", [{"error": "timeout", "returncode": -15},
                                   {"stdout": "not json"},
                                   {"stdout": json.dumps(sample_for("wrong-node", 100))},
                                   {"stdout": json.dumps(sample_for("node", "invalid"))},
                                   {"stdout": "{}"},
                                   {"stdout": "[]"}])
def test_survey_retains_failed_or_invalid_sample(monkeypatch, tmp_path, result):
    fake_scheduler(monkeypatch, ["node"], samples={"node": {"returncode": 0, **result}})
    report = screen.survey(tmp_path / "evidence")
    assert report["nodes"][0]["status"].startswith("skipped_")
    assert report["ranking"] == []
    assert report["nodes"][0]["sample_command"]


def test_survey_budget_records_remaining_nodes_without_attempts(monkeypatch, tmp_path):
    calls = fake_scheduler(monkeypatch, ["one", "two"])
    # Start, reservation, hostnames, first node, first query, first srun, second node.
    ticks = iter([0, 0, 0, 0, 0, 0, 5])
    monkeypatch.setattr(screen.time, "monotonic", lambda: next(ticks))
    report = screen.survey(tmp_path / "evidence", budget=1)
    assert [entry["status"] for entry in report["nodes"]] == ["sampled", "skipped_budget"]
    assert len([argv for argv, _ in calls if argv[0] == "srun"]) == 1


def test_survey_discovery_failure_is_retained(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(screen, "bounded_command", lambda *args: {"returncode": 1, "stderr": "no reservation"})
    report = screen.survey(tmp_path / "evidence")
    assert "discovery failed" in report["error"]
    assert report["finished_at"]
    assert report["nodes"] == []
    assert json.loads((tmp_path / "evidence/survey.json").read_text()) == report


def test_never_borrows_existing_job_or_overwrites_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "unrelated")
    with pytest.raises(ValueError, match="outside an allocation"):
        screen.survey(tmp_path / "new")
    assert not (tmp_path / "new").exists()
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(FileExistsError):
        screen.survey(tmp_path)


@pytest.mark.parametrize("timeout,budget", [(0, 1), (1, -1), (float("inf"), 1), (1, float("nan"))])
def test_invalid_bounds_do_not_create_artifacts(monkeypatch, tmp_path, timeout, budget):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="finite and positive"):
        screen.survey(tmp_path / "new", timeout=timeout, budget=budget)
    assert not (tmp_path / "new").exists()


def test_command_timeout_terminates_only_owned_child():
    result = screen.bounded_command([sys.executable, "-c", "import time; time.sleep(30)"], 0.05)
    assert result["error"] == "timeout"
    assert result["returncode"] != 0


def test_missing_command_is_diagnostic(tmp_path):
    result = screen.bounded_command([str(tmp_path / "not-a-command")], 1)
    assert result["returncode"] is None
    assert result["error"]


def test_interruption_terminates_owned_process_before_propagating(monkeypatch):
    class Process:
        pid = 12345

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return "", ""

    process = Process()
    monkeypatch.setattr(screen.subprocess, "Popen", lambda *a, **kw: process)
    signals = []
    monkeypatch.setattr(screen.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(KeyboardInterrupt):
        screen.bounded_command(["srun", "owned-command"], 1)
    assert signals == [(12345, screen.signal.SIGTERM)]
    assert process.calls == 2


def test_unresponsive_owned_process_and_inherited_pipes_have_bounded_cleanup(monkeypatch):
    waits, closes, signals = [], [], []

    class Process:
        pid = 12345
        returncode = None
        stdout = SimpleNamespace(close=lambda: closes.append("stdout"))
        stderr = SimpleNamespace(close=lambda: closes.append("stderr"))

        def communicate(self, timeout):
            waits.append(timeout)
            raise screen.subprocess.TimeoutExpired("owned", timeout, output=b"partial", stderr=b"error")

        def wait(self, timeout):
            waits.append(timeout)
            raise screen.subprocess.TimeoutExpired("owned", timeout)

    monkeypatch.setattr(screen.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(screen.os, "killpg", lambda *args: signals.append(args))
    result = screen.bounded_command(["srun", "owned-command"], 0.5)
    assert result["error"] == "timeout"
    assert result["stdout"] == "partial"
    assert result["stderr"] == "error"
    assert waits == [0.5, 1, 1, 1]
    assert closes == ["stdout", "stderr"]
    assert signals == [(12345, screen.signal.SIGTERM), (12345, screen.signal.SIGKILL)]


def test_sample_cli_is_bounded_and_does_not_schedule(monkeypatch, capsys):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(sys, "argv", ["node-screen", "sample", "--samples", "2", "--interval", "0.1"])
    monkeypatch.setattr(screen, "collect", lambda **kwargs: {"sample": True})
    sleeps = []
    monkeypatch.setattr(screen.time, "sleep", sleeps.append)
    screen.main()
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [{"sample": True}] * 2
    assert sleeps == [0.1]


def test_sample_cli_requires_owned_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["node-screen", "sample"])
    with pytest.raises(SystemExit) as exit:
        screen.main()
    assert exit.value.code == 1


@pytest.mark.parametrize("options", [["--samples", "0"], ["--interval", "inf"],
                                   ["--interval", "0"], ["--target-bytes", "3"],
                                   ["--gpu-numa-nodes", "-1"]])
def test_sample_cli_rejects_unbounded_or_invalid_inputs(monkeypatch, options):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(sys, "argv", ["node-screen", "sample", *options])
    with pytest.raises(SystemExit) as exit:
        screen.main()
    assert exit.value.code == 1


def test_survey_cli_reads_only_site_scheduler_defaults(monkeypatch, tmp_path, capsys):
    site = put(tmp_path, "site.toml", '[slurm]\naccount="staff"\npartition="gpu"\n')
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", ["node-screen", "survey", "--site", str(site),
                                     "--output", str(output), "--gpu-numa-nodes", "1,0,1"])
    calls = []
    monkeypatch.setattr(screen, "survey", lambda *a, **kw: calls.append((a, kw)) or {})
    screen.main()
    assert calls[0][0] == (output,)
    assert calls[0][1]["account"] == "staff"
    assert calls[0][1]["partition"] == "gpu"
    assert calls[0][1]["gpu_nodes"] == [0, 1]
    assert capsys.readouterr().out.strip() == str(output / "survey.json")
