"""Metadata inventory fixtures only: no actual ROCm reads, vendor imports or execution."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from apertus_amd import config
from apertus_amd.runs import batch_script, edf

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/chonk-profiler-sdk-inventory-icb5a384016a9'
PREPARATION = ROOT / 'artifacts/investigations/performance-next/restart-20261113/fusion-profiler-sdk-inventory-preparation'


def load(path):
    spec = importlib.util.spec_from_file_location('inventory_fixture_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe():
    return load(ROOT / 'experiments/validation/profiler_sdk_inventory.py')


def settings():
    return config.resolve(EXPERIMENT, submit=True, time='00:03:00', reservation='cscs')


def test_pure_plan_hooks_and_only_stdlib_imports(probe, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('subprocess'))
    monkeypatch.setattr(Path, 'write_text', lambda *a, **k: pytest.fail('plan write'))
    value = settings()
    assert value['procedure']['timeout'] == 60 and value['image']['sha256'] == probe.IMAGE
    container = edf(value, Path('/run'))
    for literal in ('com.hooks.cxi.enabled = "false"', 'com.hooks.aws_ofi_nccl.enabled = "false"', 'HWLOC_COMPONENTS = "-gl"'):
        assert literal in container
    batch = batch_script({**value, 'id': 'fixture'}, Path('/run'))
    assert '--time=00:03:00' in batch and '--ntasks-per-node=4' in batch and '--exclusive' in batch
    tree = ast.parse(Path(probe.__file__).read_text())
    imports = [n.module if isinstance(n, ast.ImportFrom) else n.names[0].name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert set(imports) <= {'argparse', 'hashlib', 'json', 'os', 'pathlib', 'stat', 'sys', 'time'}
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('glob', 'rglob', 'iterdir', 'walk') for n in ast.walk(tree))


def test_scoped_text_absence_alias_dedupe_and_external_symlink(probe, tmp_path):
    root = tmp_path / '_rocm_sdk_devel'
    (root / 'bin').mkdir(parents=True)
    (root / 'lib/rocprofiler-sdk').mkdir(parents=True)
    text = 'parser.add_argument("--runtime-period")\nupdate_env("ROCPROFILER_LIBRARY_CTOR", False)\n'
    script = root / 'lib/rocprofiler-sdk/rocprofv3.py'
    script.write_text(text)
    (root / 'bin/rocprofv3').symlink_to(script)
    (tmp_path / '_rocm_sdk_core').symlink_to(root)
    outside = tmp_path / 'outside.txt'
    outside.write_text('not read')
    (root / '.info').mkdir()
    (root / '.info/version').symlink_to(outside)
    result = {}
    probe.inventory(tmp_path, result)
    assert result['hashed_bytes'] == len(text) and len(result['files']) == 8
    assert sum(row.get('text') == text for row in result['files']) == 1
    assert any(row['observation'] == 'duplicate-file' for row in result['files'])
    assert any(row['observation'] == 'external-symlink-not-followed' for row in result['files'])
    assert any(row['observation'] == 'absent' for row in result['files'])
    assert any(row['observation'] == 'duplicate-root' for row in result['roots'])
    assert next(row for row in result['files'] if 'text' in row)['constructor_setting_lines'] == [
        {'line': 2, 'text': 'update_env("ROCPROFILER_LIBRARY_CTOR", False)'}]
    again = {}
    probe.inventory(tmp_path, again)
    assert again == result  # Same immutable files yield rank-independent observations.


def test_byte_text_root_deadline_bounds(probe, tmp_path, monkeypatch):
    root = tmp_path / '_rocm_sdk_devel'
    (root / 'bin').mkdir(parents=True)
    (root / 'bin/rocprofv3').write_bytes(b'12345')
    monkeypatch.setattr(probe, 'HASH_BYTES', 4)
    result = {}
    probe.inventory(tmp_path, result)
    assert result['hashed_bytes'] == 0
    assert result['files'][0]['observation'] == 'present-hash-byte-budget-skipped'
    monkeypatch.setattr(probe, 'HASH_BYTES', 64)
    monkeypatch.setattr(probe, 'TEXT_BYTES', 4)
    probe.inventory(tmp_path, result)
    assert result['files'][0]['text_observation'] == 'one-MiB-text-limit'
    with pytest.raises(TimeoutError):
        probe.inventory(tmp_path, {}, seconds=-1)
    (tmp_path / '_rocm_sdk_core').mkdir()
    (tmp_path / '_rocm_sdk_libraries').mkdir()
    (tmp_path / 'unselected-package').mkdir()
    probe.inventory(tmp_path, result)
    assert not result['enumeration_truncated'] and len(result['files']) == 24


@pytest.mark.parametrize('field,value', [('nodes', 2), ('gpus_per_node', 1), ('time', '00:05:00'), ('reservation', 'other')])
def test_invalid_resource_rejected_before_inventory(probe, field, value):
    launch = settings()
    launch['resources'][field] = value
    with pytest.raises(ValueError):
        probe.validate_launch(launch, dict(SLURM_NTASKS='4', SLURM_PROCID='0', SLURM_LOCALID='0'))


def test_atomic_rank_output_and_no_replacement_of_existing_work(probe, tmp_path, monkeypatch):
    original = Path.read_text
    launch = settings()
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **k: json.dumps(launch) if str(self) == '/apertus/resolved.json' else original(self, *a, **k))
    for key, value in dict(SLURM_NTASKS='4', SLURM_PROCID='0', SLURM_LOCALID='0').items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('SECRET_TOKEN', 'must-not-be-retained')
    monkeypatch.setenv('ROCM_HOME', '/fixture/declared-sdk')
    monkeypatch.setattr(probe, 'inventory', lambda opt, result: result.update(files=[], fixture=True))
    probe.main([str(tmp_path)])
    result = json.loads((tmp_path / 'profiler-sdk-inventory-0.json').read_text())
    assert result['status'] == 'observed' and result['gpu_use'] is False and result['collector_executed'] is False
    assert result['environment']['ROCM_HOME'] == '/fixture/declared-sdk' and 'SECRET_TOKEN' not in result['environment']
    assert not (tmp_path / 'profiler-sdk-inventory-0.tmp').exists()
    with pytest.raises(FileExistsError):
        probe.main([str(tmp_path)])


def test_transport_inactive_and_wrong_approval_never_executes(tmp_path, monkeypatch):
    transport = load(PREPARATION / 'transport.py')
    monkeypatch.setattr(transport.os, 'execv', lambda *a: pytest.fail('scheduler execution'))
    monkeypatch.delenv('APERTUS_PROFILER_SDK_INVENTORY_APPROVAL', raising=False)
    with pytest.raises(RuntimeError, match='PREPARATION ONLY'):
        transport.main('sbatch')
    monkeypatch.setattr(transport, 'HERE', tmp_path)
    (tmp_path / 'manifest.json').write_text('{}')
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps({'execution_authorized': True, 'maximum_submissions': 1, 'experiment': 'wrong'}))
    monkeypatch.setenv('APERTUS_PROFILER_SDK_INVENTORY_APPROVAL', str(approval))
    with pytest.raises(ValueError, match='exact inventory'):
        transport.main('sbatch')
    assert transport.NODE not in transport.EXCLUDED.split(',')
    assert {'nid002456', 'nid002744', 'nid002454'} <= set(transport.EXCLUDED.split(','))
    assert {'741662', '741653', '741609'} <= transport.PROTECTED


def test_exact_literal_roots_and_external_root_not_followed(probe, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    package = tmp_path / 'package'
    package.mkdir()
    (package / '_rocm_sdk_devel').symlink_to(outside)
    result = {}
    probe.inventory(package, result)
    assert [Path(row['path']).name for row in result['roots']] == list(probe.ROOTS)
    assert result['roots'][0]['observation'] == 'external-or-nondirectory-not-followed'
    assert result['files'] == []


def test_elf_is_hashed_not_decoded_or_executed(probe, tmp_path):
    root = tmp_path / '_rocm_sdk_devel'
    (root / 'bin').mkdir(parents=True)
    data = b'\x7fELF\x00fixture-not-executable'
    (root / 'bin/rocprofv3').write_bytes(data)
    result = {}
    probe.inventory(tmp_path, result)
    row = result['files'][0]
    assert row['elf'] and row['sha256'] == hashlib.sha256(data).hexdigest() and 'text' not in row
