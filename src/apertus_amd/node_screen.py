"""Read-only, advisory buddy-allocator evidence; never a benchmark admission gate.

Run ``python -m apertus_amd.node_screen sample`` inside a CPU-only Slurm step.
The explicit ``survey`` command makes short, immediate allocations on idle nodes
in a reservation and retains skips rather than waiting for busy nodes.
No GPU runtime is loaded, no memory is filled, and no kernel state is changed.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import tomllib
import uuid

from .artifacts import write_json


EVIDENCE_ROOT = Path("artifacts/investigations/performance-next/node-screening")
ADVISORY = ("Advisory snapshot only, not an admission gate or a THP guarantee; "
            "buddy counts omit migratetypes, watermarks, reclaim and allocation races. "
            "Per-NUMA MemAvailable is null when the kernel does not expose it; "
            "MemFree is not an available-memory estimate.")


def timestamp() -> str:
    """Return an unambiguous UTC observation timestamp."""
    return datetime.now(timezone.utc).isoformat()


def parse_buddyinfo(text: str, page_bytes: int, target_bytes: int) -> list[dict]:
    """Retain raw order counts and derive target-sized extents for every zone."""
    if page_bytes <= 0 or target_bytes < page_bytes or target_bytes % page_bytes:
        raise ValueError("target bytes must be a power-of-two multiple of the base page size")
    pages = target_bytes // page_bytes
    if pages & (pages - 1):
        raise ValueError("target bytes must be a power-of-two multiple of the base page size")
    target_order = pages.bit_length() - 1
    zones = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*Node\s+(\d+),\s+zone\s+(\S+)\s+([\d\s]+)", line)
        if not match:
            raise ValueError(f"malformed buddyinfo line: {line!r}")
        counts = [int(item) for item in match[3].split()]
        extents = sum(count << (order - target_order) for order, count in enumerate(counts)
                      if order >= target_order)
        zones.append({"numa_node": int(match[1]), "zone": match[2], "order_counts": counts,
                      "free_bytes": sum(count * (page_bytes << order)
                                        for order, count in enumerate(counts)),
                      "target_order": target_order, "target_sized_blocks": extents,
                      "high_order_free_bytes": extents * target_bytes})
    return zones


def parse_meminfo(text: str) -> dict:
    """Normalize byte-valued meminfo fields; raw evidence retains unitless counters."""
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r"(?:Node\s+\d+\s+)?(\S+):\s+(\d+)\s+kB\s*", line)
        if match:
            values[match[1]] = int(match[2]) * 1024
    return values


def fragmentation_score(zones: list[dict], gpu_nodes: list[int]) -> dict:
    """Rank by the weakest GPU-local Normal zone, without filtering any nodes."""
    local = []
    for node in sorted(set(gpu_nodes)):
        normal = [zone for zone in zones if zone["numa_node"] == node and zone["zone"] == "Normal"]
        if not normal:
            return {"worst_target_sized_blocks": None, "reason": f"missing Normal zone for NUMA {node}"}
        local.append({"numa_node": node,
                      "target_sized_blocks": sum(zone["target_sized_blocks"] for zone in normal),
                      "free_bytes": sum(zone["free_bytes"] for zone in normal)})
    if not local:
        return {"worst_target_sized_blocks": None, "reason": "GPU-local NUMA topology unavailable"}
    worst = min(item["target_sized_blocks"] for item in local)
    return {"worst_target_sized_blocks": worst,
            "worst_numa_nodes": [item["numa_node"] for item in local if item["target_sized_blocks"] == worst],
            "gpu_local_normal": local}


def collect(*, proc: Path = Path("/proc"), sysfs: Path = Path("/sys"),
            page_bytes: int | None = None, target_bytes: int | None = None,
            gpu_nodes: list[int] | None = None) -> dict:
    """Read a local snapshot with per-file errors, without subprocesses or writes."""
    started = timestamp()
    errors, raw = {}, {}

    def read(path):
        try:
            text = path.read_text()
        except OSError as error:
            errors[str(path)] = str(error)
            return ""
        raw[str(path)] = text
        return text

    page_bytes = page_bytes if page_bytes is not None else os.sysconf("SC_PAGE_SIZE")
    thp_root = sysfs / "kernel/mm/transparent_hugepage"
    pmd_size = read(thp_root / "hpage_pmd_size").strip()
    if target_bytes is None:
        target_bytes = int(pmd_size) if pmd_size.isdigit() else 2 * 1024**2
        target_source = "kernel hpage_pmd_size" if pmd_size.isdigit() else "2 MiB fallback"
    else:
        target_source = "explicit"
    buddy = read(proc / "buddyinfo")
    try:
        zones = parse_buddyinfo(buddy, page_bytes, target_bytes)
    except ValueError as error:
        errors[str(proc / "buddyinfo")] = str(error)
        zones = []
    memory = parse_meminfo(read(proc / "meminfo"))
    numa_memory = {}
    for path in sorted((sysfs / "devices/system/node").glob("node[0-9]*/meminfo")):
        values = parse_meminfo(read(path))
        numa_memory[path.parent.name.removeprefix("node")] = {
            "meminfo_bytes": values, "free_bytes": values.get("MemFree"),
            "available_bytes": values.get("MemAvailable")}
    gpus = []
    for device in sorted((sysfs / "bus/pci/devices").glob("*")):
        try:
            vendor = int((device / "vendor").read_text().strip(), 16)
            device_class = int((device / "class").read_text().strip(), 16) >> 16
        except (OSError, ValueError):
            continue
        if vendor not in (0x1002, 0x10DE) or device_class not in (0x03, 0x12):
            continue
        numa = read(device / "numa_node").strip()
        gpus.append({"pci_address": device.name, "vendor": hex(vendor),
                     "numa_node": int(numa) if numa.lstrip("-").isdigit() else -1})
    explicit_topology = gpu_nodes is not None
    if gpu_nodes is None:
        gpu_nodes = sorted({gpu["numa_node"] for gpu in gpus if gpu["numa_node"] >= 0})
    score = fragmentation_score(zones, gpu_nodes)
    # A partial topology must not silently hide the worst GPU-local domain.
    if not explicit_topology and any(gpu["numa_node"] < 0 for gpu in gpus):
        score = {"worst_target_sized_blocks": None, "reason": "incomplete GPU-local NUMA topology"}
    uptime = read(proc / "uptime").split()
    try:
        uptime_seconds = float(uptime[0]) if uptime else None
    except ValueError as error:
        errors[str(proc / "uptime")] = str(error)
        uptime_seconds = None
    value = {"schema": 1, "started_at": started, "hostname": socket.gethostname(),
             "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "base_page_bytes": page_bytes,
             "target_bytes": target_bytes, "target_source": target_source,
             "buddy_zones": zones, "meminfo_bytes": memory, "numa_memory": numa_memory,
             "uptime_seconds": uptime_seconds,
             "gpus": gpus, "gpu_numa_nodes": gpu_nodes,
             "gpu_topology_source": "explicit" if explicit_topology else "PCI sysfs",
             "score": score, "advisory": ADVISORY}
    for name in ("enabled", "defrag"):
        read(thp_root / name)
    value.update(raw=raw, errors=errors, finished_at=timestamp())
    return value


def bounded_command(command: list[str], timeout: float) -> dict:
    """Bound a scheduler call, forwarding termination to its owned srun process group."""
    result = {"command": command, "started_at": timestamp()}
    try:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
    except OSError as error:
        return {**result, "returncode": None, "error": str(error), "finished_at": timestamp()}
    def terminate():
        # SIGTERM lets srun cancel its own allocation; never cancel by user/name.
        # The one-minute Slurm wall limit also bounds remote lifetime if srun fails.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                return process.communicate(timeout=1)
            except subprocess.TimeoutExpired as error:
                # Even an escaped descendant retaining a pipe must not hold up
                # benchmark scheduling. Slurm still owns the remote wall limit.
                process.stdout.close()
                process.stderr.close()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                return tuple(value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                             for value in (error.output, error.stderr))

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stdout, stderr = terminate()
        result["error"] = "timeout"
    except BaseException:
        # Ctrl-C must not leave a survey allocation running behind the caller.
        terminate()
        raise
    result.update(returncode=process.returncode, stdout=stdout, stderr=stderr, finished_at=timestamp())
    return result


def slurm_fields(text: str) -> dict:
    """Extract Slurm's one-line key/value records without interpreting hostlists."""
    return dict(re.findall(r"(?:^|\s)(\w+)=(\S+)", text))


def survey(output: Path, *, reservation: str = "cscs", account: str | None = None,
           partition: str | None = None, timeout: float = 10, budget: float = 60,
           target_bytes: int | None = None, gpu_nodes: list[int] | None = None) -> dict:
    """Record idle reservation nodes via bounded CPU-only srun; preserve all skips."""
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("survey must run outside an allocation; use sample inside an owned job")
    if not all(math.isfinite(value) and value > 0 for value in (timeout, budget)):
        raise ValueError("timeout and budget must be finite and positive")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "started_at": timestamp(), "reservation": reservation,
              "advisory": ADVISORY, "discovery": [], "nodes": [], "ranking": [],
              "budget_seconds": budget, "command_timeout_seconds": timeout}
    deadline = time.monotonic() + budget

    def save():
        write_json(output / "survey.json", report)

    def call(command):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"command": command, "returncode": None, "error": "survey budget exhausted"}
        return bounded_command(command, min(timeout, remaining))

    def discover(command):
        result = call(command)
        report["discovery"].append(result)
        save()
        if result.get("error") or result["returncode"] != 0:
            raise ValueError(f"reservation discovery failed; see {output / 'survey.json'}")
        return result["stdout"]

    try:
        fields = slurm_fields(discover(["scontrol", "show", "reservation", reservation, "--oneliner"]))
        hostlist = fields.get("Nodes")
        if not hostlist or hostlist in ("(null)", "ALL"):
            raise ValueError("reservation must provide an explicit node list")
        nodes = list(dict.fromkeys(discover(["scontrol", "show", "hostnames", hostlist]).split()))
        # Populate all entries first so an interrupted survey still retains its scope.
        report["nodes"] = [{"node": node, "status": "not_attempted"} for node in nodes]
        save()
        for entry in report["nodes"]:
            node = entry["node"]
            if time.monotonic() >= deadline:
                entry["status"] = "skipped_budget"
                save()
                continue
            state = call(["scontrol", "show", "node", node, "--oneliner"])
            entry["state_query"] = state
            if state.get("error") or state["returncode"] != 0:
                entry["status"] = "skipped_unreachable"
                save()
                continue
            fields = slurm_fields(state["stdout"])
            entry["slurm_state"] = fields.get("State")
            # Conservatively skip MIXED, completing, drained and non-responsive nodes.
            # --immediate handles the race between this query and allocation.
            if fields.get("State") not in ("IDLE", "IDLE+RESERVED"):
                entry["status"] = "skipped_busy_or_unavailable"
                save()
                continue
            command = ["srun", f"--reservation={reservation}", f"--nodelist={node}",
                       "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=128M",
                       "--gres=none", "--immediate=1", "--time=00:01:00", "--nice=10000",
                       "--job-name=node-screen", "--mpi=none", "--cpu-bind=none",
                       f"--chdir={Path.cwd()}"]
            if account:
                command.append(f"--account={account}")
            if partition:
                command.append(f"--partition={partition}")
            command.extend([sys.executable, "-B", "-m", "apertus_amd.node_screen", "sample"])
            if target_bytes is not None:
                command.extend(["--target-bytes", str(target_bytes)])
            if gpu_nodes is not None:
                command.extend(["--gpu-numa-nodes", ",".join(map(str, gpu_nodes))])
            result = call(command)
            entry["sample_command"] = result
            if result.get("error") or result["returncode"] != 0:
                entry["status"] = "skipped_unreachable_or_raced"
            else:
                try:
                    sample = json.loads(result["stdout"])
                    if sample["slurm_job_id"] is None or sample["hostname"].split(".")[0] != node.split(".")[0]:
                        raise ValueError("sample does not identify the requested scheduler-owned node")
                    score = sample["score"]["worst_target_sized_blocks"]
                    if score is not None and (type(score) is not int or score < 0):
                        raise ValueError("invalid fragmentation score")
                    entry.update(status="sampled", sample=sample)
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    entry.update(status="skipped_invalid_sample", error=str(error))
            save()
    except (OSError, ValueError) as error:
        report["error"] = str(error)
    finally:
        ranked = [entry for entry in report["nodes"] if entry["status"] == "sampled"]
        ranked.sort(key=lambda entry: (entry["sample"]["score"]["worst_target_sized_blocks"] is None,
                                      -(entry["sample"]["score"]["worst_target_sized_blocks"] or 0),
                                      entry["node"]))
        report["ranking"] = [{"node": entry["node"], **entry["sample"]["score"]} for entry in ranked]
        report["finished_at"] = timestamp()
        save()
    return report


def main() -> None:
    """Expose bounded local sampling and an explicitly invoked reservation survey."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="read-only JSONL collector inside an owned Slurm step")
    sample.add_argument("--samples", type=int, default=1)
    sample.add_argument("--interval", type=float, default=10)
    host = commands.add_parser("survey", help="immediately attempt idle reservation nodes; never queue")
    host.add_argument("--reservation", default="cscs")
    host.add_argument("--site", type=Path, default=Path("config.toml"))
    host.add_argument("--output", type=Path, help=f"new directory; default {EVIDENCE_ROOT}/<unique timestamp>")
    host.add_argument("--timeout", type=float, default=10, help="maximum seconds per scheduler command")
    host.add_argument("--budget", type=float, default=60, help="total survey seconds, plus termination grace")
    for command in (sample, host):
        command.add_argument("--target-bytes", type=int, help="power-of-two extent; default kernel PMD size or 2 MiB")
        command.add_argument("--gpu-numa-nodes", help="explicit topology override, e.g. 0,1,2,3")
    args = parser.parse_args()
    try:
        gpu_nodes = None
        if args.gpu_numa_nodes is not None:
            gpu_nodes = sorted(set(map(int, args.gpu_numa_nodes.split(","))))
            if not gpu_nodes or min(gpu_nodes) < 0:
                raise ValueError("GPU NUMA nodes must be nonnegative integers")
        if args.target_bytes is not None:
            parse_buddyinfo("", os.sysconf("SC_PAGE_SIZE"), args.target_bytes)
        if args.command == "sample":
            if not os.environ.get("SLURM_JOB_ID"):
                raise ValueError("sample requires an owned Slurm allocation (SLURM_JOB_ID)")
            if args.samples < 1 or not math.isfinite(args.interval) or args.interval <= 0:
                raise ValueError("samples and interval must be finite and positive")
            for index in range(args.samples):
                if index:
                    time.sleep(args.interval)
                print(json.dumps(collect(target_bytes=args.target_bytes, gpu_nodes=gpu_nodes)), flush=True)
        else:
            with args.site.open("rb") as stream:
                slurm = tomllib.load(stream)["slurm"]
            output = args.output or EVIDENCE_ROOT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
            report = survey(output, reservation=args.reservation, account=slurm.get("account"),
                            partition=slurm.get("partition"), timeout=args.timeout, budget=args.budget,
                            target_bytes=args.target_bytes, gpu_nodes=gpu_nodes)
            print(output / "survey.json")
            if "error" in report:
                parser.exit(1, report["error"] + "\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Failed: {error}\n")


if __name__ == "__main__":
    main()
