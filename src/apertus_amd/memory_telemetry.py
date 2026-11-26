"""Low-rate Linux process and node memory telemetry for bounded source probes."""

import json
import os
from pathlib import Path
import re
from time import time_ns


def text(path):
    """Return None for inaccessible or vanished procfs and cgroup files."""
    try:
        return path.read_text()
    except OSError:
        return None


def fields(path, names, sep=":"):
    """Read the selected kernel fields without collecting process command lines."""
    lines = text(path)
    if lines is None:
        return None
    result = {}
    for line in lines.splitlines():
        key, _, value = line.partition(sep)
        if key in names:
            result[key] = value.strip()
    return result


def process_tree(pid, proc=Path("/proc")):
    """Sample only the training leader and its currently visible descendants."""
    parents, rows = [pid], {}
    while parents:
        item = parents.pop()
        if item in rows:
            continue
        base = proc / str(item)
        row = fields(base / "status", {"Name", "VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM", "VmPin"})
        rows[item] = row
        if row is not None:
            for entry in (base / "task").glob("*/children"):
                parents.extend(int(s) for s in (text(entry) or "").split() if s.isdecimal())
    return rows


def cgroup_paths(proc=Path("/proc"), mount=Path("/sys/fs/cgroup")):
    """Resolve visible task/step/job cgroups without reading other jobs' cgroups."""
    for line in (text(proc / "self/cgroup") or "").splitlines():
        _, controllers, relative = line.split(":", 2)
        if controllers and "memory" not in controllers.split(","):
            continue
        root = mount / "memory" if controllers else mount
        relative = Path(relative)
        current = root / str(relative).lstrip("/")
        if ".." in current.parts:
            continue
        if not current.is_dir():
            for entry in (text(proc / "self/mountinfo") or "").splitlines():
                info, _, fs = entry.partition(" - ")
                parts = info.split()
                if (len(parts) < 6 or fs.split()[:1] != (["cgroup"] if controllers else ["cgroup2"])
                        or Path(parts[4]) != root or not relative.is_relative_to(parts[3])):
                    continue
                current = root / relative.relative_to(parts[3])
                break
        if not current.is_dir():
            # Preserve mount-level data as such if the exact task path is hidden.
            current = root
        for path in (current, *current.parents):
            if not path.is_relative_to(root):
                break
            yield path


_MAP = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+([rwxps-]{4})\s+([0-9a-f]+)\s+(\S+)\s+(\d+)\s*(.*)$")
_COUNTERS = {"Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Shared_Clean", "Shared_Dirty",
             "Private_Clean", "Private_Dirty", "Locked", "VmFlags"}


def mapping_snapshot(pid, proc=Path("/proc"), minimum_kib=128 * 1024, limit=32):
    """Retain largest process VMAs by resident size, without reading mapped contents."""
    try:
        with (proc / str(pid) / "smaps").open() as stream:
            rows, row = [], None
            for line in stream:
                if found := _MAP.match(line.rstrip("\n")):
                    start, end, perms, offset, device, inode, name = found.groups()
                    size = (int(end, 16) - int(start, 16)) // 1024
                    row = None
                    if size >= minimum_kib:
                        row = dict(start=start, end=end, perms=perms, offset=offset,
                                   device=device, inode=int(inode), name=name, size_kib=size)
                        rows.append(row)
                elif row is not None:
                    key, _, value = line.partition(":")
                    if key in _COUNTERS:
                        row[key] = value.strip() if key == "VmFlags" else int(value.split()[0])
    except OSError as exc:
        return dict(status="unavailable", error=str(exc))
    rows.sort(key=lambda row: (row.get("Rss", 0), row["size_kib"]), reverse=True)
    return dict(status="sampled", candidate_count=len(rows), minimum_kib=minimum_kib, mappings=rows[:limit])


def sample(stream, pid, *, node_lead, event, mappings=False):
    """Append one flushed observation from the supervisor, not the training child."""
    row = dict(unix_ns=time_ns(), event=event, pid=pid, process_tree=process_tree(pid))
    if node_lead:
        names = ("memory.current", "memory.max", "memory.peak", "memory.events", "memory.events.local",
                 "memory.usage_in_bytes", "memory.limit_in_bytes", "memory.max_usage_in_bytes",
                 "memory.failcnt", "memory.oom_control")
        row["cgroups"] = {str(group): {**{name: value.strip() for name in names
                                       if (value := text(group / name)) is not None},
                                      "stat": fields(group / "memory.stat", {
                                          "anon", "file", "shmem", "kernel", "slab", "pagetables",
                                          "total_rss", "total_cache", "total_shmem"
                                      }, sep=" ")}
                          for group in cgroup_paths()}
        row["node_memory"] = fields(Path("/proc/meminfo"), {
            "MemTotal", "MemAvailable", "MemFree", "AnonPages", "Shmem", "Unevictable", "Slab"
        })
        stat = os.statvfs("/dev/shm")
        row["shm_used_bytes"] = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
    if mappings:
        row["vm_mapping"] = mapping_snapshot(pid)
        row["vm_mapping_end_unix_ns"] = time_ns()
    print(json.dumps(row, separators=(",", ":")), file=stream, flush=True)
