"""Bounded fake-wheel fixtures only; no vendor import, installed-package read or execution."""

import ast
import builtins
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess

import pytest

from apertus_amd.cli import plan
from apertus_amd.config import resolve
from apertus_amd.runs import edf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'experiments/validation/kernel_consumer_inventory.py'
EXPERIMENT = ROOT / 'experiments/chonk-kernel-consumer-inventory-i910bfcf936d3'


@pytest.fixture
def probe(monkeypatch):
    imported = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformer_engine', 'flash_attn', 'aiter', 'triton'}:
            pytest.fail('vendor import')
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    spec = importlib.util.spec_from_file_location('kernel_inventory_cpu', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wheel(root, distribution='transformer_engine_rocm10-2.17.dist-info', name='transformer-engine-rocm10', files=None, rows=None):
    metadata = root / distribution
    metadata.mkdir(parents=True)
    (metadata / 'METADATA').write_text(f'Name: {name}\nVersion: 2.17.1\n')
    files = files if files is not None else {'transformer_engine/pytorch/permutation.py': '# actual fixture\n'}
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    stream = io.StringIO()
    csv.writer(stream).writerows(rows if rows is not None else [(relative, '', '') for relative in files])
    (metadata / 'RECORD').write_text(stream.getvalue())
    return metadata


def settings():
    return resolve(EXPERIMENT, site=ROOT / 'config.toml', submit=True, reservation='cscs', time='00:05:00')


def environment(rank=0):
    return {'SLURM_JOB_ID': '123', 'SLURM_NTASKS': '4', 'SLURM_NODEID': '0',
            'SLURM_PROCID': str(rank), 'SLURM_LOCALID': str(rank)}


def test_actual_RECORD_selected_text_provenance_and_normalized_consistency(probe, tmp_path):
    wheel(tmp_path, files={'transformer_engine/pytorch/permutation.py': '# permutation\n',
        'transformer_engine/pytorch/attention/actual_binding.py': 'from flash_attn import flash_attn_func\n',
        'transformer_engine/pytorch/unselected.py': 'raise RuntimeError("never executed")\n'})
    wheel(tmp_path, 'flash_attn-2.dist-info', 'flash-attn', {'flash_attn/actual_interface.py': '# interface\n'})
    wheel(tmp_path, 'amd_aiter-1.dist-info', 'amd-aiter', {'aiter/ops/mha.py': '# mha\n', 'aiter/__init__.py': '# init\n'})
    first, second = {}, {}
    probe.inventory(tmp_path, first)
    probe.inventory(tmp_path, second)
    assert first == second and first['normalized_identity']
    assert len(first['files']) == 5 and first['flash_interfaces_listed'] == 1
    assert 'transformer_engine/pytorch/attention.py' in first['missing_RECORD_names']
    assert all(row['observation'] == 'complete-UTF8' and row['sha256'] and row['provenance']['row'] for row in first['files'])
    assert all(dist['record_complete'] and dist['record']['sha256'] and dist['metadata']['sha256'] for dist in first['distributions'])
    assert not any('unselected' in row['name'] for row in first['files'])


def test_no_vendor_import_subprocess_walk_or_unbounded_metadata_API(probe, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('subprocess'))
    tree = ast.parse(SCRIPT.read_text())
    roots = {n.module.split('.')[0] if isinstance(n, ast.ImportFrom) else n.names[0].name.split('.')[0]
             for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert roots <= {'argparse', 'csv', 'email', 'hashlib', 'io', 'json', 'os', 'pathlib', 'stat', 'sys', 'time'}
    assert not any(isinstance(n, ast.Attribute) and n.attr in {'walk', 'glob', 'rglob', 'iterdir', 'read_text'}
                   for n in ast.walk(tree) if getattr(n, 'lineno', 0) < 190)


@pytest.mark.parametrize('kind', ['duplicate-row', 'duplicate-dist', 'duplicate-header', 'hardlink-alias', 'symlink-inside', 'symlink-outside', 'unsafe-path', 'bad-width', 'bad-csv', 'ELF', 'invalid-UTF8', 'drift'])
def test_ambiguity_unsafe_or_changed_reads_fail_with_partial_observations(probe, tmp_path, monkeypatch, kind):
    metadata = wheel(tmp_path)
    source = tmp_path / 'transformer_engine/pytorch/permutation.py'
    if kind == 'duplicate-row':
        (metadata / 'RECORD').write_text('transformer_engine/pytorch/permutation.py,,\n' * 2)
    elif kind == 'duplicate-dist':
        wheel(tmp_path, 'transformer-engine-rocm10-duplicate.dist-info', files={})
    elif kind == 'duplicate-header':
        (metadata / 'METADATA').write_text('Name: transformer-engine\nName: other\nVersion: 2\n')
    elif kind in {'hardlink-alias', 'symlink-inside'}:
        alias = source.with_name('__init__.py')
        if kind == 'hardlink-alias':
            os.link(source, alias)
        else:
            alias.symlink_to(source)
        (metadata / 'RECORD').write_text('transformer_engine/pytorch/permutation.py,,\ntransformer_engine/pytorch/__init__.py,,\n')
    elif kind == 'symlink-outside':
        source.unlink()
        source.symlink_to('/outside/not-allowed.py')
    elif kind == 'unsafe-path':
        (metadata / 'RECORD').write_text('transformer_engine/pytorch/attention/../../../../escape.py,,\n')
    elif kind == 'bad-width':
        (metadata / 'RECORD').write_text('transformer_engine/pytorch/permutation.py,only-two\n')
    elif kind == 'bad-csv':
        (metadata / 'RECORD').write_text('"transformer_engine/pytorch/permutation.py,,\n')
    elif kind == 'ELF':
        source.write_bytes(b'\x7fELF disguised native bytes')
    elif kind == 'invalid-UTF8':
        source.write_bytes(b'\xff')
    else:
        original = probe.fingerprint
        count = [0]
        def fingerprint(value):
            count[0] += 1
            result = original(value)
            if count[0] == 3:
                result[2] += 1
            return result
        monkeypatch.setattr(probe, 'fingerprint', fingerprint)
    result = {}
    with pytest.raises((ValueError, OSError, UnicodeDecodeError, csv.Error)):
        probe.inventory(tmp_path, result)
    assert 'distributions' in result and 'source_hashed_bytes' in result and 'normalized_identity' not in result


@pytest.mark.parametrize('bound,value', [('RECORD_BYTES', 4), ('RECORD_ROWS', 0), ('METADATA_BYTES', 4),
    ('INDEX_ENTRIES', 0), ('INDEX_BYTES', 0), ('MAX_DISTS', 0), ('FILE_BYTES', 4), ('SOURCE_BYTES', 4)])
def test_all_independent_bounds_are_hard_failures(probe, tmp_path, monkeypatch, bound, value):
    wheel(tmp_path)
    monkeypatch.setattr(probe, bound, value)
    result = {}
    with pytest.raises(ValueError):
        probe.inventory(tmp_path, result)
    assert 'normalized_identity' not in result


def test_attention_cap_retains_partial_sources_and_rejects_completeness(probe, tmp_path):
    wheel(tmp_path, files={f'transformer_engine/pytorch/attention/module_{n:02d}.py': '# bounded\n' for n in range(25)})
    result = {}
    with pytest.raises(ValueError, match='partial text'):
        probe.inventory(tmp_path, result)
    assert len(result['files']) == 24 and result['truncated']
    assert result['attention_subtree_listed'] == 25


def test_total_file_cap_and_deadline(probe, tmp_path):
    wheel(tmp_path, files={f'flash_attn/interface_{n:02d}.py': '# bounded\n' for n in range(33)})
    result = {}
    with pytest.raises(ValueError):
        probe.inventory(tmp_path, result)
    assert len(result['files']) == 32 and result['truncated']
    with pytest.raises(TimeoutError):
        probe.inventory(tmp_path, {}, seconds=-1)


@pytest.mark.parametrize('rank', range(4))
@pytest.mark.parametrize('failed', [False, True])
def test_atomic_output_and_partial_failure_no_overwrite(probe, tmp_path, monkeypatch, rank, failed):
    launch = settings()
    original = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k: json.dumps(launch) if str(path) == '/apertus/resolved.json' else original(path, *a, **k))
    monkeypatch.setattr(probe.os, 'environ', environment(rank))
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    def inventory(package, result):
        result.update(files=[{'fixture': 'partial'}])
        if failed:
            raise ValueError('fixture budget failure')
    monkeypatch.setattr(probe, 'inventory', inventory)
    if failed:
        with pytest.raises(ValueError):
            probe.main([str(tmp_path)])
    else:
        probe.main([str(tmp_path)])
    path = tmp_path / f'kernel-consumer-inventory-{rank}.json'
    result = json.loads(path.read_text())
    assert result['status'] == ('failed' if failed else 'observed') and result['files']
    assert not result['gpu_use'] and not result['vendor_imports'] and not result['dynamic_consumer_proved']
    assert not path.with_suffix('.tmp').exists()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        probe.main([str(tmp_path)])
    assert path.read_bytes() == before


def test_pure_resolver_hooks_and_resource_gate(probe, monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail('plan wrote/called subprocess')
    with monkeypatch.context() as guard:
        guard.setattr(subprocess, 'Popen', refuse)
        guard.setattr(Path, 'write_text', refuse)
        guard.setattr(Path, 'write_bytes', refuse)
        guard.setattr(Path, 'mkdir', refuse)
        value = plan(EXPERIMENT, site=ROOT / 'config.toml', submit=True, reservation='cscs', time='00:05:00')
    assert value['procedure']['timeout'] == 120 and '--ntasks=4' in value['steps'][0]
    text = edf(value, '/future')
    for item in ('HWLOC_COMPONENTS = "-gl"', 'com.hooks.cxi.enabled = "false"', 'com.hooks.aws_ofi_nccl.enabled = "false"'):
        assert item in text
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    probe.validate_launch(settings(), environment())
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: {0})
    with pytest.raises(ValueError):
        probe.validate_launch(settings(), environment())


@pytest.mark.parametrize('existing', [False, True])
def test_atomic_serialization_failure_preserves_old_output_and_removes_owned_temporary(probe, tmp_path, existing):
    path = tmp_path / 'result.json'
    if existing:
        probe.atomic(path, {'old': True}, first=True)
    with pytest.raises(TypeError):
        probe.atomic(path, {'invalid': object()}, first=not existing)
    assert not path.with_suffix('.tmp').exists()
    assert path.exists() == existing
    if existing:
        assert json.loads(path.read_text()) == {'old': True}
    probe.atomic(path, {'new': True}, first=not existing)
    assert json.loads(path.read_text()) == {'new': True}


@pytest.mark.parametrize('workload_failed', [False, True])
def test_final_evidence_error_preserves_active_inventory_error(probe, tmp_path, monkeypatch, workload_failed):
    monkeypatch.setattr(probe, 'validate_launch', lambda *args: None)
    monkeypatch.setattr(Path, 'read_text', lambda *args, **kwargs: '{}')
    monkeypatch.setattr(probe.os, 'environ', environment())
    def inventory(package, result):
        if workload_failed:
            raise ValueError('original inventory failure')
    monkeypatch.setattr(probe, 'inventory', inventory)
    calls = []
    def atomic(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise OSError('final evidence fixture')
    monkeypatch.setattr(probe, 'atomic', atomic)
    with pytest.raises(ValueError if workload_failed else OSError) as error:
        probe.main([str(tmp_path)])
    if workload_failed:
        assert str(error.value) == 'original inventory failure'
        assert 'secondary evidence error: OSError: final evidence fixture' in error.value.__notes__
