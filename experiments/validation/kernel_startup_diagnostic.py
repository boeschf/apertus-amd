"""Rank-tagged startup tracing and bounded installed-source inventory; no vendor imports."""

import argparse
import csv
from email.parser import Parser
import faulthandler
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import time

IMAGE = '910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490'
PACKAGE = Path('/opt/venv/lib/python3.12/site-packages')
MAX_FILES, FILE_BYTES, SOURCE_BYTES = 32, 1048576, 16777216
RECORD_BYTES, RECORD_ROWS, METADATA_BYTES = 8388608, 50000, 262144
INDEX_ENTRIES, INDEX_BYTES, MAX_DISTS = 4096, 524288, 16
MODULE_FORMS = (('aiter/jit/core.py', 'aiter/jit/core/__init__.py'),
                ('aiter/ops/flydsl/fmha_kernels.py', 'aiter/ops/flydsl/fmha_kernels/__init__.py'),
                ('aiter/ops/triton/mha.py', 'aiter/ops/triton/mha/__init__.py'),
                ('aiter/ops/triton/attention/mha.py', 'aiter/ops/triton/attention/mha/__init__.py'))
EXACT = ('transformer_engine/pytorch/permutation.py',
         'transformer_engine/pytorch/triton/permutation.py',
         'transformer_engine/pytorch/triton/__init__.py',
         'transformer_engine/pytorch/cpp_extensions/fused_attn.py',
         'transformer_engine/common/triton/permutation.py',
         'transformer_engine/common/triton/__init__.py',
         'torch/nn/functional.py', 'torch/nn/attention/__init__.py',
         'aiter/ops/mha.py', 'aiter/__init__.py') + tuple(name for forms in MODULE_FORMS for name in forms)


def marker(event, **fields):
    """Flush one bounded rank-tagged event to the retained launcher stderr."""
    print(json.dumps(dict(event=event, rank=os.environ.get('SLURM_PROCID'), pid=os.getpid(),
                          time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(), **fields)),
          file=sys.stderr, flush=True)


def traced_json(path, value):
    """Match runtime atomic JSON publication, exposing each potentially blocking boundary."""
    marker('json-temporary-begin', path=str(path))
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    temporary = Path(name)
    try:
        marker('json-temporary-end', path=str(path))
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            marker('json-flush-begin', path=str(path))
            stream.flush()
            marker('json-flush-end', path=str(path))
            marker('json-fsync-begin', path=str(path))
            os.fsync(stream.fileno())
            marker('json-fsync-end', path=str(path))
        marker('json-rename-begin', path=str(path))
        temporary.replace(path)
        marker('json-rename-end', path=str(path))
    finally:
        temporary.unlink(missing_ok=True)


def bootstrap(index):
    """Trace the unchanged frozen runtime, restoring diagnostic wrappers on every exit."""
    if index != '0':
        raise ValueError('startup diagnostic requires the sole command index 0')
    marker('python-bootstrap-entered')
    runtime, originals, argv = None, {}, sys.argv
    try:
        faulthandler.dump_traceback_later(20, repeat=False)
        marker('runtime-import-begin')
        from apertus_amd import runtime
        marker('runtime-import-end')
        originals = {name: getattr(runtime, name) for name in
                     ('verify_tooling', 'record_environment', 'sha256', 'write_json')}
        def traced_hash(path):
            marker('tooling-hash-begin', path=str(path))
            value = originals['sha256'](path)
            marker('tooling-hash-end', path=str(path))
            return value
        def verify(*args, **kwargs):
            marker('verify-tooling-begin')
            originals['verify_tooling'](*args, **kwargs)
            marker('verify-tooling-end')
        def record(*args, **kwargs):
            marker('record-environment-begin')
            originals['record_environment'](*args, **kwargs)
            marker('record-environment-end')
        runtime.verify_tooling, runtime.record_environment = verify, record
        runtime.sha256, runtime.write_json = traced_hash, traced_json
        sys.argv = [runtime.__file__, index]
        marker('runtime-main-begin')
        runtime.main()
        marker('runtime-main-end')
    finally:
        sys.argv = argv
        if runtime is not None:
            for name, original in originals.items():
                setattr(runtime, name, original)
        primary = sys.exception()
        try:
            faulthandler.cancel_dump_traceback_later()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f'secondary timer cleanup error: {type(error).__name__}: {error}')


def fingerprint(info):
    """Exclude atime while detecting read-time identity/content drift."""
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def check(state):
    """Enforce the cooperative 90-second deadline beneath the outer 120-second limit."""
    if time.monotonic() >= state['deadline']:
        raise TimeoutError('startup inventory deadline')


def canonical(name):
    """Normalize wheel distribution aliases without importing their packages."""
    return '-'.join(filter(None, name.lower().replace('_', '-').replace('.', '-').split('-')))


def selected_distribution(name):
    """Limit metadata selection to the exact consumer distribution families."""
    return name in {'torch', 'aiter', 'amd-aiter'} or name.startswith(('transformer-engine', 'aiter-'))


def safe_parts(name):
    """Reject noncanonical or escaping RECORD paths before filesystem access."""
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or path.as_posix() != name or '..' in path.parts
            or '\\' in name or '\x00' in name):
        raise ValueError('unsafe RECORD/metadata path')
    return path.parts


def read_text(root, name, limit, state, bucket, row):
    """Read bounded regular UTF8 using no-follow directory descriptors, never a native target."""
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
                raise ValueError('selected source/metadata is not regular')
            row.update(path=str(root / name), bytes=before.st_size, stat_fingerprint=fingerprint(before))
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
                raise ValueError('ELF disguised as selected text; not hashed')
            text = data.decode('utf-8')
            row.update(sha256=hashlib.sha256(data).hexdigest(), observation='complete-UTF8')
            return text
        finally:
            os.close(source)
    finally:
        os.close(fd)


def inventory(package, result, *, seconds=90):
    """Read only fixed source names listed by complete, bounded installed wheel RECORDs."""
    root = package.resolve(strict=True)
    if package.is_symlink() or root != package.absolute():
        raise ValueError('package root aliases/symlinks are not accepted')
    state = dict(deadline=time.monotonic() + seconds, record=RECORD_BYTES,
                 metadata=METADATA_BYTES, source=SOURCE_BYTES)
    result.update(distributions=[], files=[], missing_RECORD_names=[], module_resolution=[], truncated=False,
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
                if entry.name.endswith('.dist-info') and selected_distribution(canonical(entry.name[:-10].rsplit('-', 1)[0])):
                    index.append(entry.name)
                    if len(index) > MAX_DISTS:
                        raise ValueError('distribution count bound exceeded')
        for directory in sorted(index):
            dist = dict(directory=directory, metadata={}, record={}, record_complete=False)
            result['distributions'].append(dist)
            text = read_text(root, directory + '/METADATA', METADATA_BYTES, state, 'metadata', dist['metadata'])
            metadata = Parser().parsestr(text)
            name, version = metadata.get('Name'), metadata.get('Version')
            if (len(metadata.get_all('Name', [])) != 1 or len(metadata.get_all('Version', [])) != 1
                    or not name or not version or canonical(name) in seen_dist
                    or not selected_distribution(canonical(name))):
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
                if source not in EXACT:
                    continue
                safe_parts(source)
                if source in owners:
                    raise ValueError('duplicate ambiguous RECORD source owner')
                owners[source] = dict(distribution=name, directory=directory, row=number,
                                      record_hash=row[1], record_size=row[2])
            dist['record_complete'] = True
        result['missing_RECORD_names'] = [name for name in EXACT if name not in owners]
        for forms in MODULE_FORMS:
            present = [name for name in forms if name in owners]
            result['module_resolution'].append(dict(candidates=list(forms), RECORD_names=present,
                observation='ambiguous' if len(present) > 1 else 'RECORD-listed' if present else 'absent-from-RECORD'))
            if len(present) > 1:
                raise ValueError('ambiguous module/package RECORD forms')
        if len(owners) > MAX_FILES:
            result['truncated'] = True
            raise ValueError('selected source count exceeded')
        inodes = set()
        for name in sorted(owners):
            row = dict(name=name, provenance=owners[name])
            result['files'].append(row)
            try:
                row['text'] = read_text(root, name, FILE_BYTES, state, 'source', row)
            except FileNotFoundError:
                row['observation'] = 'RECORD-listed-file-missing'
                continue
            identity = tuple(row['stat_fingerprint'][:2])
            if identity in inodes:
                raise ValueError('duplicate source inode/alias ambiguity')
            inodes.add(identity)
        normalized = dict(distributions=[dict(name=d['name'], version=d['version'],
            metadata_sha256=d['metadata']['sha256'], record_sha256=d['record']['sha256']) for d in result['distributions']],
            files=[{k: f[k] for k in ('name', 'observation', 'sha256', 'bytes') if k in f} for f in result['files']],
            missing_RECORD_names=result['missing_RECORD_names'], module_resolution=result['module_resolution'])
        result['normalized_identity'] = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    finally:
        result.update(source_hashed_bytes=SOURCE_BYTES - state['source'], record_hashed_bytes=RECORD_BYTES - state['record'],
                      metadata_hashed_bytes=METADATA_BYTES - state['metadata'])


def atomic(path, value, *, first=False):
    """Reserve a new invocation without overwriting old results; replace only its own evidence."""
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
    """Require the approved candidate/resource shape before any installed metadata access."""
    resource, procedure = settings['resources'], settings['procedure']
    if (settings['image']['sha256'] != IMAGE or resource['nodes'] != 1 or resource['gpus_per_node'] != 4
            or resource['cpus_per_node'] != 192 or resource['time'] != '00:05:00' or resource.get('reservation') != 'cscs'
            or resource['distribution'] != 'block:block' or resource['cpu_bind'] != 'map_ldom:0,1,2,3'
            or procedure['kind'] != 'commands' or procedure['timeout'] != 120
            or procedure['commands'] != [['python3', '-u', '/apertus/validation.py', 'inventory']]
            or int(env['SLURM_NTASKS']) != 4 or int(env['SLURM_NODEID']) != 0
            or not 0 <= int(env['SLURM_PROCID']) == int(env['SLURM_LOCALID']) < 4
            or len(os.sched_getaffinity(0)) != 48):
        raise ValueError('requires candidate image, one192CPU/four metadata ranks/48CPU masks, cscs5min/command120s')


def main(argv=None):
    """Select startup bootstrap or retain bounded inventory, preserving primary failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('bootstrap', 'inventory'))
    parser.add_argument('index', nargs='?')
    args = parser.parse_args(argv)
    if args.mode == 'bootstrap':
        return bootstrap(args.index)
    if args.index is not None:
        parser.error('inventory takes no index')
    settings = json.loads(Path('/apertus/resolved.json').read_text())
    validate_launch(settings, os.environ)
    rank = int(os.environ['SLURM_PROCID'])
    path = Path(f'kernel-startup-inventory-{rank}.json')
    result = dict(status='running', rank=rank, image=IMAGE, gpu_use=False, vendor_imports=False,
                  binary_attestation=False, dynamic_consumer_proved=False, cpu_affinity=sorted(os.sched_getaffinity(0)))
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
