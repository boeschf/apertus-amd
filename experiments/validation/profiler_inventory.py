"""Bounded ROCm profiler file inventory: no vendor imports, subprocesses or GPU calls."""

import argparse
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import stat
import sys
import time


IMAGE = 'cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0'
MAX_FILES, TEXT_BYTES, HASH_BYTES = 32, 1048576, 16777216  # Four ranks: <=64MiB/job.
NAMES = ('bin/rocprofv3', 'lib/rocprofiler-sdk/rocprofv3.py', '.info/version',
         '.info/version-rocprofiler-sdk', 'include/rocprofiler-sdk/version.h',
         'share/rocprofiler-sdk/README.md', 'share/doc/rocprofiler-sdk/README.md',
         'share/doc/rocprofiler-sdk/rocprofv3.md')


def fingerprint(info):
    """Ignore read-induced atime updates while rejecting content/identity changes."""
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def atomic(path, value):
    """Replace this rank's reserved result atomically, without overwriting old invocations."""
    temporary = path.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def validate_launch(settings, env):
    """Require the reviewed metadata-only allocation; never select or initialize a GPU."""
    resource = settings['resources']
    if (settings['image']['sha256'] != IMAGE or resource['nodes'] != 1 or resource['gpus_per_node'] != 4
            or resource['time'] != '00:03:00' or resource.get('reservation') != 'cscs'
            or settings['procedure']['timeout'] != 60 or int(env['SLURM_NTASKS']) != 4
            or not 0 <= int(env['SLURM_PROCID']) == int(env['SLURM_LOCALID']) < 4):
        raise ValueError('requires baseline image, one node/four ranks, cscs, three minutes/command60s')


def inventory(opt, result, *, seconds=55):
    """Enumerate only four scoped ROCm roots/eight names, with bounded deduplicated reads."""
    deadline = time.monotonic() + seconds
    used, seen, roots = 0, {}, set()
    candidates = sorted(islice(opt.glob('rocm*'), 5))
    result.update(roots=[], files=[], enumeration_truncated=len(candidates) > 4, hashed_bytes=0)
    for alias in candidates[:4]:
        root = alias.resolve()
        entry = {'path': str(alias), 'resolved': str(root)}
        result['roots'].append(entry)
        if root.parent != opt.resolve() or not root.name.startswith('rocm') or not root.is_dir():
            entry['observation'] = 'external-or-nondirectory-not-followed'
            continue
        if root in roots:
            entry['observation'] = 'duplicate-root'
            continue
        roots.add(root)
        entry['observation'] = 'scoped-root'
        for name in NAMES:
            if time.monotonic() >= deadline:
                raise TimeoutError('profiler metadata inventory deadline')
            path = root / name
            row = {'path': str(path)}
            result['files'].append(row)
            target = path.resolve()
            if not target.is_relative_to(root):
                row.update(observation='external-symlink-not-followed', target=str(target))
                continue
            try:
                info = target.stat()
            except FileNotFoundError:
                row['observation'] = 'absent'
                continue
            row.update(resolved=str(target), bytes=info.st_size)
            if not stat.S_ISREG(info.st_mode):
                row['observation'] = 'not-regular-file'
                continue
            key = (info.st_dev, info.st_ino)
            if key in seen:
                row.update(observation='duplicate-file', duplicate_of=seen[key])
                continue
            seen[key] = str(target)
            if info.st_size > HASH_BYTES - used:
                row['observation'] = 'present-hash-byte-budget-skipped'
                continue
            h, text, count, magic = hashlib.sha256(), [], 0, b''
            with target.open('rb') as stream:
                while count < info.st_size:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('profiler metadata inventory deadline')
                    block = stream.read(min(262144, info.st_size - count))
                    if not block:
                        raise ValueError('selected file shortened during read')
                    magic = magic or block[:4]
                    h.update(block)
                    count, used = count + len(block), used + len(block)
                    result['hashed_bytes'] = used
                    if info.st_size <= TEXT_BYTES and magic != b'\x7fELF':
                        text.append(block)
                if fingerprint(os.fstat(stream.fileno())) != fingerprint(info) or fingerprint(target.stat()) != fingerprint(info):
                    raise ValueError('selected file changed during read')
            row.update(observation='present', sha256=h.hexdigest(), elf=magic == b'\x7fELF')
            if text:
                try:
                    row['text'] = b''.join(text).decode('utf-8')
                except UnicodeDecodeError:
                    row['text_observation'] = 'not-UTF8'
            elif info.st_size > TEXT_BYTES:
                row['text_observation'] = 'one-MiB-text-limit'
    assert len(result['files']) <= MAX_FILES and used <= HASH_BYTES
    result['scope'] = 'selected immutable-image files only; absence is not proof of tool unavailability'


def main(argv=None):
    """Retain atomic rank-owned metadata under the existing commands runtime deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    settings = json.loads(Path('/apertus/resolved.json').read_text())
    validate_launch(settings, os.environ)
    rank = int(os.environ['SLURM_PROCID'])
    path = args.output / f'profiler-inventory-{rank}.json'
    result = {'status': 'running', 'rank': rank, 'image': IMAGE, 'gpu_use': False,
              'vendor_imports': False, 'collector_executed': False, 'trace_acceptance': False,
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'resources': settings['resources'], 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'environment': {key: value for key, value in os.environ.items() if key.startswith(
                  ('SLURM_', 'ROCR_VISIBLE', 'HIP_VISIBLE', 'CUDA_VISIBLE', 'HWLOC_'))}}
    with path.open('x') as stream:
        json.dump(result, stream)
    try:
        inventory(Path('/opt'), result)
        result['status'] = 'observed'
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        active = sys.exc_info()[0] is not None
        try:
            atomic(path, result)
        except BaseException:
            if not active:
                raise


if __name__ == '__main__':
    main()
