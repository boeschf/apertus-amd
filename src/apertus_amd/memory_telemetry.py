"""Low-rate Linux process and node memory telemetry for bounded source probes."""

from collections import deque
from itertools import islice
import json
import os
from pathlib import Path
import re
import stat
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


def scratch_inventory(root, pids, *, include_tree, proc=Path("/proc"), limit=4096, keep=16):
    """Sample only job scratch and visible child descriptors; never open file contents."""
    inode_rows, covered, errors = {}, 0, 0
    truncated = False
    pending = deque([root]) if include_tree else deque()
    root_device = root.stat().st_dev

    def add(path, metadata, *, deleted=False):
        if metadata.st_dev != root_device or not stat.S_ISREG(metadata.st_mode):
            return
        key = (metadata.st_dev, metadata.st_ino)
        if key in inode_rows:
            return
        name = path.name
        tag = ("dump_like" if name.startswith(("core", "gpucore")) or name.endswith(".gpu")
               else "rank_cache" if any(part.startswith("cache-") for part in path.parts)
               else "other")
        inode_rows[key] = dict(path=str(path), device=key[0], inode=key[1], tag=tag,
                               bytes=metadata.st_blocks * 512, deleted=deleted)

    while pending and covered < limit:
        directory = pending.popleft()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if covered >= limit:
                        truncated = True
                        break
                    covered += 1
                    try:
                        meta = entry.stat(follow_symlinks=False)
                        if meta.st_dev == root_device and stat.S_ISDIR(meta.st_mode):
                            pending.append(Path(entry.path))
                        elif stat.S_ISREG(meta.st_mode):
                            add(Path(entry.path).relative_to(root), meta)
                    except OSError:
                        errors += 1
        except OSError:
            errors += 1
    truncated |= bool(pending)

    fd_count = 0
    truncated |= len(pids) > 32
    for pid in islice(pids, 32):
        try:
            with os.scandir(proc / str(pid) / "fd") as fds:
                for fd in fds:
                    if fd_count >= min(limit, 1024):
                        truncated = True
                        break
                    fd_count += 1
                    try:
                        target = os.readlink(fd.path)
                        deleted = target.endswith(" (deleted)")
                        path = Path(target.removesuffix(" (deleted)"))
                        if path.is_absolute() and ".." not in path.parts and path.is_relative_to(root):
                            add(path.relative_to(root), fd.stat(), deleted=deleted)
                    except OSError:
                        errors += 1
        except OSError:
            errors += 1
        if fd_count >= min(limit, 1024):
            break

    totals = {}
    for row in inode_rows.values():
        tag = f"{row['tag']}:deleted" if row["deleted"] else row["tag"]
        totals[tag] = totals.get(tag, 0) + row["bytes"]
    largest = sorted(inode_rows.values(), key=lambda row: row["bytes"], reverse=True)[:keep]
    return dict(status="partial" if truncated or errors else "sampled", tree=include_tree,
                entries=covered, descriptors=fd_count, errors=errors, truncated=truncated,
                bytes_by_tag=totals, largest=largest)


def sample(stream, pid, *, node_lead, event, mappings=False, scratch=None):
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
    if scratch is not None:
        try:
            row["scratch_inventory"] = scratch_inventory(scratch, row["process_tree"], include_tree=node_lead)
        except OSError as exc:
            row["scratch_inventory"] = dict(status="unavailable", error=type(exc).__name__)
        if node_lead:
            row["kernel_core_pattern"] = text(Path("/proc/sys/kernel/core_pattern"))
        row["scratch_inventory_end_unix_ns"] = time_ns()
    print(json.dumps(row, separators=(",", ":")), file=stream, flush=True)
