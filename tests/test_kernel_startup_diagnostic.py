"""CPU fake-wheel and fake-runtime tests; never read installed vendors or launch a scheduler."""

import ast
import builtins
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from apertus_amd.config import resolve
from apertus_amd.runs import edf
from test_kernel_consumer_inventory import wheel, environment

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'experiments/validation/kernel_startup_diagnostic.py'
EXPERIMENT = ROOT / 'experiments/chonk-kernel-startup-diagnostic-i910bfcf936d3'


@pytest.fixture
def probe(monkeypatch):
    imported = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformer_engine', 'flash_attn', 'aiter', 'triton'}:
            pytest.fail('vendor import')
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    spec = importlib.util.spec_from_file_location('startup_diagnostic_cpu', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings():
    return resolve(EXPERIMENT, site=ROOT / 'config.toml', submit=True, reservation='cscs', time='00:05:00')


@pytest.mark.parametrize('failed', [False, True])
def test_bootstrap_orders_markers_one_shot_timer_and_restores_runtime(probe, monkeypatch, tmp_path, failed):
    import apertus_amd
    events = []
    monkeypatch.setattr(probe, 'marker', lambda event, **fields: events.append(event))
    monkeypatch.setattr(probe.faulthandler, 'dump_traceback_later', lambda seconds, **kw: events.append((seconds, kw)))
    monkeypatch.setattr(probe.faulthandler, 'cancel_dump_traceback_later', lambda: events.append('cancel'))
    runtime = SimpleNamespace(__file__='frozen/runtime.py', sha256=lambda path: 'digest', write_json=lambda *a: None)
    def verify(*args):
        assert runtime.sha256(tmp_path / 'tooling.py') == 'digest'
    def record(*args):
        runtime.write_json(tmp_path / 'environment-0.json', {'rank': 0})
    runtime.verify_tooling, runtime.record_environment = verify, record
    def main():
        assert sys.argv == ['frozen/runtime.py', '0']
        runtime.verify_tooling()
        runtime.record_environment()
        if failed:
            raise ValueError('primary runtime failure')
    runtime.main = main
    originals, argv = vars(runtime).copy(), sys.argv
    monkeypatch.setattr(apertus_amd, 'runtime', runtime, raising=False)
    if failed:
        with pytest.raises(ValueError, match='primary runtime failure'):
            probe.bootstrap('0')
    else:
        probe.bootstrap('0')
    assert vars(runtime) == originals and sys.argv is argv
    assert events[:4] == ['python-bootstrap-entered', (20, {'repeat': False}), 'runtime-import-begin', 'runtime-import-end']
    assert events[-1] == 'cancel'
    for begin, end in [('verify-tooling-begin', 'verify-tooling-end'), ('tooling-hash-begin', 'tooling-hash-end'),
                       ('record-environment-begin', 'record-environment-end'), ('json-fsync-begin', 'json-fsync-end'),
                       ('json-rename-begin', 'json-rename-end')]:
        assert events.index(begin) < events.index(end)
    assert json.loads((tmp_path / 'environment-0.json').read_text()) == {'rank': 0}


def test_bootstrap_import_failure_and_cleanup_preserve_primary(probe, monkeypatch):
    imported = builtins.__import__
    def fail(name, *args, **kwargs):
        if name == 'apertus_amd':
            raise ValueError('import failed')
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', fail)
    monkeypatch.setattr(probe.faulthandler, 'dump_traceback_later', lambda *a, **k: None)
    def cleanup():
        raise OSError('cancel failed')
    monkeypatch.setattr(probe.faulthandler, 'cancel_dump_traceback_later', cleanup)
    argv = sys.argv
    with pytest.raises(ValueError, match='import failed') as caught:
        probe.bootstrap('0')
    assert sys.argv is argv and 'cancel failed' in caught.value.__notes__[0]


def test_traced_publication_fsync_failure_preserves_old_json(probe, monkeypatch, tmp_path):
    path = tmp_path / 'environment-0.json'
    path.write_text('old')
    def fail(*args):
        raise OSError('fsync failed')
    monkeypatch.setattr(probe.os, 'fsync', fail)
    with pytest.raises(OSError, match='fsync failed'):
        probe.traced_json(path, {'rank': 0})
    assert path.read_text() == 'old' and list(tmp_path.iterdir()) == [path]


def test_exact_RECORD_provenance_and_missing_are_observations(probe, tmp_path):
    wheel(tmp_path, files={name: '# fixture\n' for name in probe.EXACT[:6]})
    wheel(tmp_path, 'torch-2.11.0+rocm.dist-info', 'torch', {'torch/nn/functional.py': '# torch text\n',
        'torch/unselected.py': 'raise RuntimeError("never execute")\n'})
    first, second = {}, {}
    probe.inventory(tmp_path, first)
    probe.inventory(tmp_path, second)
    assert first == second and len(first['files']) == 7
    assert all(row['provenance']['row'] and row['sha256'] for row in first['files'])
    assert 'torch/nn/attention/__init__.py' in first['missing_RECORD_names']
    assert any(row['name'] == 'transformer_engine/common/triton/permutation.py' for row in first['files'])
    assert not any('unselected' in row['name'] for row in first['files'])


@pytest.mark.parametrize('kind', ['symlink', 'directory-symlink', 'hardlink', 'duplicate', 'ELF', 'utf8', 'missing', 'bad-csv'])
def test_safe_source_failures_and_partial_evidence(probe, tmp_path, kind):
    metadata = wheel(tmp_path)
    source = tmp_path / probe.EXACT[0]
    if kind == 'symlink':
        source.unlink()
        source.symlink_to('/outside/never-read.py')
    elif kind == 'directory-symlink':
        source.parent.rename(source.parent.with_name('saved'))
        source.parent.symlink_to('saved')
    elif kind == 'hardlink':
        alias = tmp_path / probe.EXACT[1]
        alias.parent.mkdir(parents=True)
        os.link(source, alias)
        (metadata / 'RECORD').write_text(f'{probe.EXACT[0]},,\n{probe.EXACT[1]},,\n')
    elif kind == 'duplicate':
        (metadata / 'RECORD').write_text(f'{probe.EXACT[0]},,\n' * 2)
    elif kind == 'ELF':
        source.write_bytes(b'\x7fELF fake')
    elif kind == 'utf8':
        source.write_bytes(b'\xff')
    elif kind == 'missing':
        source.unlink()
    else:
        (metadata / 'RECORD').write_text('"unterminated')
    result = {}
    if kind == 'missing':
        probe.inventory(tmp_path, result)
        assert result['files'][0]['observation'] == 'RECORD-listed-file-missing'
    else:
        with pytest.raises((ValueError, OSError, UnicodeDecodeError, probe.csv.Error)):
            probe.inventory(tmp_path, result)
        assert 'normalized_identity' not in result and 'source_hashed_bytes' in result


@pytest.mark.parametrize('bound', ['RECORD_BYTES', 'RECORD_ROWS', 'METADATA_BYTES', 'INDEX_ENTRIES',
                                   'INDEX_BYTES', 'MAX_DISTS', 'FILE_BYTES', 'SOURCE_BYTES', 'MAX_FILES'])
def test_hard_bounds_and_deadline(probe, tmp_path, monkeypatch, bound):
    wheel(tmp_path)
    monkeypatch.setattr(probe, bound, 0)
    with pytest.raises(ValueError):
        probe.inventory(tmp_path, {})
    with pytest.raises(TimeoutError):
        probe.inventory(tmp_path, {}, seconds=-1)


@pytest.mark.parametrize('failed', [False, True])
def test_inventory_result_no_overwrite_and_primary_error(probe, monkeypatch, tmp_path, failed):
    launch, original = settings(), Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k: json.dumps(launch) if str(path) == '/apertus/resolved.json' else original(path, *a, **k))
    monkeypatch.setattr(probe.os, 'environ', environment())
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    monkeypatch.chdir(tmp_path)
    def inventory(*args):
        args[1]['files'] = [{'partial': True}]
        if failed:
            raise ValueError('inventory failed')
    monkeypatch.setattr(probe, 'inventory', inventory)
    if failed:
        with pytest.raises(ValueError, match='inventory failed'):
            probe.main(['inventory'])
    else:
        probe.main(['inventory'])
    path = tmp_path / 'kernel-startup-inventory-0.json'
    data = path.read_bytes()
    assert json.loads(data)['status'] == ('failed' if failed else 'observed')
    with pytest.raises(FileExistsError):
        probe.main(['inventory'])
    assert path.read_bytes() == data


def test_source_import_scope_and_launch_contract(probe, monkeypatch):
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split('.')[0] not in {'torch', 'aiter', 'triton', 'subprocess'} for n in node.names)
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: set(range(48)))
    probe.validate_launch(settings(), environment())
    monkeypatch.setattr(probe.os, 'sched_getaffinity', lambda pid: {0})
    with pytest.raises(ValueError):
        probe.validate_launch(settings(), environment())
    text = edf(settings(), Path('/future'))
    assert 'HWLOC_COMPONENTS = "-gl"' in text
    assert 'com.hooks.cxi.enabled = "false"' in text and 'com.hooks.aws_ofi_nccl.enabled = "false"' in text


@pytest.mark.parametrize('workload_failed', [False, True])
def test_final_evidence_failure_preserves_primary(probe, monkeypatch, workload_failed):
    monkeypatch.setattr(probe, 'validate_launch', lambda *args: None)
    monkeypatch.setattr(Path, 'read_text', lambda *args: '{}')
    monkeypatch.setattr(probe.os, 'environ', environment())
    def inventory(*args):
        if workload_failed:
            raise ValueError('primary inventory failure')
    monkeypatch.setattr(probe, 'inventory', inventory)
    calls = []
    def atomic(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise OSError('final evidence failed')
    monkeypatch.setattr(probe, 'atomic', atomic)
    with pytest.raises(ValueError if workload_failed else OSError) as caught:
        probe.main(['inventory'])
    if workload_failed:
        assert str(caught.value) == 'primary inventory failure'
        assert 'final evidence failed' in caught.value.__notes__[0]


def test_atomic_serialization_error_retains_old_result(probe, tmp_path):
    path = tmp_path / 'result.json'
    probe.atomic(path, {'old': True}, first=True)
    with pytest.raises(TypeError):
        probe.atomic(path, {'invalid': object()})
    assert json.loads(path.read_text()) == {'old': True}
    assert not path.with_suffix('.tmp').exists()


@pytest.mark.parametrize('forms', ['module', 'package', 'both', 'neither'])
def test_AITER_exact_module_forms_are_resolved_without_search(probe, tmp_path, forms):
    names = [name for pair in probe.MODULE_FORMS for name in
             (pair if forms == 'both' else pair[:1] if forms == 'module' else pair[1:] if forms == 'package' else ())]
    wheel(tmp_path, 'amd_aiter-1.0.dist-info', 'amd-aiter', {name: '# exact import target\n' for name in names})
    result = {}
    if forms == 'both':
        with pytest.raises(ValueError, match='ambiguous module/package'):
            probe.inventory(tmp_path, result)
        assert result['module_resolution'][0]['observation'] == 'ambiguous'
        assert 'normalized_identity' not in result
    else:
        probe.inventory(tmp_path, result)
        assert len(result['module_resolution']) == 4
        assert len(result['files']) == (0 if forms == 'neither' else 4)
        assert all(row['observation'] == ('absent-from-RECORD' if forms == 'neither' else 'RECORD-listed')
                   for row in result['module_resolution'])
