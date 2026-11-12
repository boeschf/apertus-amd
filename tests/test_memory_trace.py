"""Check diagnostic collection and lifecycle without requiring GPU libraries."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def trace():
    path = Path(__file__).parents[1] / 'experiments/validation/memory_trace.py'
    spec = importlib.util.spec_from_file_location('memory_trace', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_does_not_initialize_gpu(trace, monkeypatch, tmp_path):
    monkeypatch.setenv('TMPDIR', str(tmp_path / 'tmp'))
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    value = trace.snapshot(torch, 1, [], include_cache=True)
    assert 'gpu' not in value
    assert 'VmRSS:' in value['process_status']
    assert value['private_cache']['files'] == 0


def test_cache_scan_does_not_follow_directory_links(trace, tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'large').write_bytes(b'x' * 10000)
    (cache / 'link').symlink_to(outside, target_is_directory=True)
    (cache / 'kernel').write_bytes(b'abc')
    assert trace.cache_usage(cache)['logical_bytes'] == 3


@pytest.mark.parametrize('fails', [False, True])
def test_training_exit_stops_sampler_and_retains_outcome(trace, monkeypatch, tmp_path, fails):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='stub'))
    monkeypatch.setattr(trace, 'cgroup_paths', lambda: [])
    monkeypatch.setattr(trace, 'snapshot', lambda *a, **k: {'gpu': {'reserved_bytes.all.peak': 1}})

    def training(*args):
        if fails:
            raise ValueError('training failure')

    monkeypatch.setattr(trace, 'train', training)
    if fails:
        with pytest.raises(ValueError, match='training failure'):
            trace.trace_training(tmp_path, tmp_path, 'train.py', [])
    else:
        trace.trace_training(tmp_path, tmp_path, 'train.py', [])
    end = json.loads((tmp_path / 'memory-end-0.json').read_text())
    assert end['training_completed'] is not fails
    assert end['sampler_errors'] == []
    assert (tmp_path / 'memory-0.jsonl').read_text()


def test_sampler_error_is_not_reported_as_success(trace, monkeypatch, tmp_path):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='stub'))
    monkeypatch.setattr(trace, 'cgroup_paths', lambda: [])
    monkeypatch.setattr(trace, 'train', lambda *a: None)

    def snapshot(*args, include_cache):
        if include_cache:
            raise OSError('cannot sample')
        return {'gpu': {'reserved_bytes.all.peak': 1}}

    monkeypatch.setattr(trace, 'snapshot', snapshot)
    with pytest.raises(RuntimeError, match='cannot sample'):
        trace.trace_training(tmp_path, tmp_path, 'train.py', [])


def test_check_rejects_missing_rank_and_conflicting_override(trace, tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    env = {'PYTORCH_HIP_ALLOC_CONF': 'expandable_segments:True', 'TORCH_COMPILE_DISABLE': '1',
           'PYTORCH_CUDA_ALLOC_CONF': None, 'TORCHDYNAMO_DISABLE': None}
    (tmp_path / 'resolved.json').write_text(json.dumps({
        'resources': {'nodes': 1, 'gpus_per_node': 2}, 'environment': env}))
    for rank in range(2):
        (output / f'memory-start-{rank}.json').write_text(json.dumps({
            'environment': env, 'hostname': 'node'}))
        (output / f'memory-{rank}.jsonl').write_text('{}\n')
    end = {'training_completed': True, 'sampler_errors': [], 'final': {'gpu': {'peak': 1}}}
    (output / 'memory-end-0.json').write_text(json.dumps(end))
    with pytest.raises(FileNotFoundError):
        trace.check(tmp_path)
    assert not (output / 'checks.json').exists()
    (output / 'memory-end-1.json').write_text(json.dumps(end))
    trace.check(tmp_path)
    assert json.loads((output / 'checks.json').read_text())['numerical_validation'] == 'not requested'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    (output / 'memory-start-1.json').write_text(json.dumps({'environment': env, 'hostname': 'node'}))
    with pytest.raises(ValueError, match='confounding'):
        trace.check(tmp_path)


@pytest.mark.parametrize('fails', [False, True])
def test_compilation_observer_preserves_results_and_restores_on_failure(trace, monkeypatch, tmp_path, fails):
    counters = {'generated_kernel_count': 0, 'counters': {}}
    monkeypatch.setattr(trace, 'compilation_snapshot', lambda: counters)

    def original(value, *, fail=False):
        if fail:
            raise RuntimeError('step failed')
        counters['generated_kernel_count'] += 2
        return value

    module = SimpleNamespace(train_step=original)
    value = object()
    try:
        with trace.trace_compilation_steps(module, tmp_path, 0):
            assert module.train_step(value) is value
            if fails:
                module.train_step(value, fail=True)
    except RuntimeError as error:
        assert fails and str(error) == 'step failed'
    assert module.train_step is original
    rows = [json.loads(line) for line in (tmp_path / 'compilation-0.jsonl').read_text().splitlines()]
    assert [r['status'] for r in rows] == ['start', 'step', 'end']
    assert [r['generated_kernel_count'] for r in rows] == [0, 2, 2]
    assert [r['completed_train_steps'] for r in rows] == [0, 1, 1]
