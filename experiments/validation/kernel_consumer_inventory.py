"""Bounded installed Python-consumer text/RECORD inventory; no vendor imports or GPU use."""

import argparse
import csv
from email.parser import Parser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time

IMAGE = '910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490'
PACKAGE = Path('/opt/venv/lib/python3.12/site-packages')
MAX_FILES, FILE_BYTES, SOURCE_BYTES = 32, 1048576, 16777216
RECORD_BYTES, RECORD_ROWS, METADATA_BYTES = 8388608, 50000, 262144
INDEX_ENTRIES, INDEX_BYTES, MAX_DISTS = 4096, 524288, 16
EXACT = ('transformer_engine/pytorch/permutation.py', 'transformer_engine/pytorch/__init__.py',
         'transformer_engine/pytorch/attention.py', 'flash_attn/__init__.py',
         'aiter/ops/mha.py', 'aiter/__init__.py')


def fingerprint(info):
    """Exclude atime while detecting read-time content/identity drift."""
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def check(state):
    """Enforce the cooperative deadline beneath the unchanged outer timeout."""
    if time.monotonic() >= state['deadline']:
        raise TimeoutError('consumer inventory deadline')


def canonical(name):
    """Normalize distribution-name aliases without importing their packages."""
    return '-'.join(filter(None, name.lower().replace('_', '-').replace('.', '-').split('-')))


def allowed(name):
    """Select source only from fixed package scopes, never native files or filesystem walks."""
    path = PurePosixPath(name)
    return (name in EXACT or (name.startswith('transformer_engine/pytorch/attention/') and name.endswith('.py'))
            or (name.startswith('flash_attn/') and name.endswith('.py') and 'interface' in path.name))


def safe_parts(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or path.as_posix() != name or '..' in path.parts
            or '\\' in name or '\x00' in name):
        raise ValueError('unsafe RECORD/metadata path')
    return path.parts


def read_text(root, name, limit, state, bucket, row):
    """Read regular UTF8 through no-follow directory FDs, within independent byte budgets."""
    check(state)
    parts = safe_parts(name)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(source)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('selected source/metadata is not a regular file')
            row.update(path=str(root / name), resolved=str(root / name), bytes=before.st_size,
                       stat_fingerprint=fingerprint(before))
            if before.st_size > min(limit, state[bucket]):
                row['observation'] = 'byte-bound-not-read'
                raise ValueError('source/metadata byte budget exceeded; no truncation accepted')
            blocks, remaining = [], before.st_size
            while remaining:
                check(state)
                block = os.read(source, min(262144, remaining))
                if not block:
                    raise ValueError('selected file shortened during read')
                blocks.append(block)
                remaining -= len(block)
                state[bucket] -= len(block)
            if (fingerprint(before) != fingerprint(os.fstat(source))
                    or fingerprint(before) != fingerprint(os.stat(parts[-1], dir_fd=fd, follow_symlinks=False))):
                raise ValueError('selected file changed during read')
            data = b''.join(blocks)
            if data.startswith(b'\x7fELF'):
                raise ValueError('ELF disguised as selected Python source/metadata; not hashed')
            row['sha256'] = hashlib.sha256(data).hexdigest()
            text = data.decode('utf-8')
            row['observation'] = 'complete-UTF8'
            return text
        finally:
            os.close(source)
    finally:
        os.close(fd)


def inventory(package, result, *, seconds=110):
    """Bound one metadata index; derive all selected source filenames solely from complete RECORDs."""
    root = package.resolve(strict=True)
    if package.is_symlink() or root != package.absolute():
        raise ValueError('package root aliases/symlinks are not accepted')
    state = {'deadline': time.monotonic() + seconds, 'record': RECORD_BYTES,
             'metadata': METADATA_BYTES, 'source': SOURCE_BYTES}
    result.update(distributions=[], files=[], missing_RECORD_names=[], truncated=False,
                  index_entries=0, index_name_bytes=0, record_rows=0)
    index, owners, seen_dist = [], {}, set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                check(state)
                result['index_entries'] += 1
                result['index_name_bytes'] += len(os.fsencode(entry.name))
                if result['index_entries'] > INDEX_ENTRIES or result['index_name_bytes'] > INDEX_BYTES:
                    result['truncated'] = True
                    raise ValueError('metadata index bound exceeded')
                name = canonical(entry.name.removesuffix('.dist-info'))
                if entry.name.endswith('.dist-info') and name.startswith(('transformer-engine', 'flash-attn', 'amd-aiter', 'aiter-')):
                    index.append(entry.name)
                    if len(index) > MAX_DISTS:
                        raise ValueError('distribution count bound exceeded')
        for directory in sorted(index):
            dist = {'directory': directory, 'metadata': {}, 'record': {}, 'record_complete': False}
            result['distributions'].append(dist)
            text = read_text(root, directory + '/METADATA', 65536, state, 'metadata', dist['metadata'])
            metadata = Parser().parsestr(text)
            name, version = metadata.get('Name'), metadata.get('Version')
            if (len(metadata.get_all('Name', [])) != 1 or len(metadata.get_all('Version', [])) != 1
                    or not name or not version or canonical(name) in seen_dist
                    or not canonical(name).startswith(('transformer-engine', 'flash-attn', 'amd-aiter', 'aiter'))):
                raise ValueError('missing/duplicate ambiguous distribution identity')
            seen_dist.add(canonical(name))
            dist.update(name=name, version=version)
            text = read_text(root, directory + '/RECORD', RECORD_BYTES, state, 'record', dist['record'])
            for number, row in enumerate(csv.reader(io.StringIO(text), strict=True), 1):
                check(state)
                result['record_rows'] += 1
                if result['record_rows'] > RECORD_ROWS:
                    result['truncated'] = True
                    raise ValueError('RECORD row bound exceeded')
                if len(row) != 3:
                    raise ValueError('invalid RECORD row width')
                source = row[0]
                # Wheels may contain unrelated ../bin entries; reject them only if
                # they claim an allowed consumer scope, never follow/hash their targets.
                if not allowed(source):
                    continue
                safe_parts(source)
                if source in owners:
                    raise ValueError('duplicate ambiguous RECORD source owner')
                owners[source] = {'distribution': name, 'directory': directory, 'row': number,
                                  'record_hash': row[1], 'record_size': row[2]}
            dist['record_complete'] = True
        result['missing_RECORD_names'] = [name for name in EXACT if name not in owners]
        attention = sorted(name for name in owners if name.startswith('transformer_engine/pytorch/attention/'))
        selected = sorted(name for name in owners if name not in attention) + attention[:24]
        result['listed_source_count'] = len(owners)
        result['attention_subtree_listed'] = len(attention)
        result['flash_interfaces_listed'] = sum(name.startswith('flash_attn/') and 'interface' in name for name in owners)
        selected = selected[:MAX_FILES]
        inode_owners = {}
        for name in selected:
            row = {'name': name, 'provenance': owners[name]}
            result['files'].append(row)
            row['text'] = read_text(root, name, FILE_BYTES, state, 'source', row)
            identity = tuple(row['stat_fingerprint'][:2])
            if identity in inode_owners:
                raise ValueError('duplicate source inode/alias ambiguity')
            inode_owners[identity] = name
        if len(attention) > 24 or len(owners) > MAX_FILES:
            result['truncated'] = True
            raise ValueError('selected source count exceeded; partial text only')
        normalized = {'distributions': [{'name': d['name'], 'version': d['version'],
            'metadata_sha256': d['metadata']['sha256'], 'record_sha256': d['record']['sha256']} for d in result['distributions']],
            'files': [{'name': f['name'], 'sha256': f['sha256'], 'bytes': f['bytes']} for f in result['files']],
            'missing_RECORD_names': result['missing_RECORD_names']}
        result['normalized_identity'] = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    finally:
        result.update(source_hashed_bytes=SOURCE_BYTES - state['source'], record_hashed_bytes=RECORD_BYTES - state['record'],
                      metadata_hashed_bytes=METADATA_BYTES - state['metadata'])


def atomic(path, value, *, first=False):
    """Atomically reserve this invocation or replace its own result; never replace an old invocation."""
    temporary = path.with_suffix('.tmp')
    stream = temporary.open('x')
    try:
        with stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if first:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_launch(settings, env):
    """Reject unreviewed resources/ranks/masks before metadata reads, without GPU initialization."""
    resource, procedure = settings['resources'], settings['procedure']
    if (settings['image']['sha256'] != IMAGE or resource['nodes'] != 1 or resource['gpus_per_node'] != 4
            or resource['cpus_per_node'] != 192 or resource['time'] != '00:05:00' or resource.get('reservation') != 'cscs'
            or resource['distribution'] != 'block:block' or resource['cpu_bind'] != 'map_ldom:0,1,2,3'
            or procedure['kind'] != 'commands' or procedure['timeout'] != 120
            or procedure['commands'] != [['python3', '-u', '/apertus/validation.py', '.']]
            or int(env['SLURM_NTASKS']) != 4 or int(env['SLURM_NODEID']) != 0
            or not 0 <= int(env['SLURM_PROCID']) == int(env['SLURM_LOCALID']) < 4
            or len(os.sched_getaffinity(0)) != 48):
        raise ValueError('requires candidate image, one192CPU/four metadata ranks/48CPU masks, cscs5min/command120s')


def main(argv=None):
    """Retain observations atomically, including partial reads on failure; never initialize a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    settings = json.loads(Path('/apertus/resolved.json').read_text())
    validate_launch(settings, os.environ)
    rank = int(os.environ['SLURM_PROCID'])
    path = args.output / f'kernel-consumer-inventory-{rank}.json'
    result = {'status': 'running', 'rank': rank, 'image': IMAGE, 'gpu_use': False, 'vendor_imports': False,
              'binary_attestation': False, 'dynamic_consumer_proved': False, 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    atomic(path, result, first=True)
    try:
        inventory(PACKAGE, result)
        result['status'] = 'observed'
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        primary = sys.exception()
        try:
            atomic(path, result)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f'secondary evidence error: {type(error).__name__}: {error}')


if __name__ == '__main__':
    main()
