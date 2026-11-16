"""CPU-only image-pair identity, full geometry and cache-isolation preparation contracts."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tomllib

from apertus_amd import cache
from apertus_amd.cli import plan
from apertus_amd.config import option, resolve
from apertus_amd.runs import edf


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'experiments/moe-chonk-kda-me242f3c3c113-icb5a384016a9'
NEW = ROOT / 'experiments/moe-chonk-kda-me242f3c3c113-i910bfcf936d3'
BASE = 'cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0'
CANDIDATE = '910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490'
ARCHIVE = 'e00b32a84e13e5de1b52d5f96e4882002212d0fe7756ab41062b85a275123498'
IDENTITY = ROOT / 'artifacts/investigations/performance-next/restart-20261113/native-dispatcher-combine8-preparation/source-identity.json'


def settings(directory):
    return resolve(directory, site=ROOT / 'config.toml', submit=True)


def test_exact_image_only_text_and_field_delta():
    old = (OLD / 'experiment.toml').read_text()
    new = (NEW / 'experiment.toml').read_text()
    assert old.count(BASE) == 1 and new == old.replace(BASE, CANDIDATE)
    a, b = tomllib.loads(old), tomllib.loads(new)
    a['image']['sha256'] = CANDIDATE
    assert a == b
    a, b = settings(OLD), settings(NEW)
    assert {key for key in a if a[key] != b[key]} == {'experiment', 'image', 'image_path'}
    assert a['image']['store'] == b['image']['store']


def test_only_full_geometry_and_ordinary_backend_selected():
    value = settings(NEW)
    args = value['args']
    assert value['nodes'] == value['resources']['nodes'] == 16
    assert value['resources']['gpus_per_node'] == 4
    for flag, expected in {'--tensor-model-parallel-size': '1', '--pipeline-model-parallel-size': '4',
        '--expert-model-parallel-size': '16', '--global-batch-size': '512', '--seq-length': '4096',
        '--train-iters': '60', '--save-interval': '60', '--moe-token-dispatcher-type': 'alltoall'}.items():
        assert option(args, flag) == expected
    layout = option(args, '--pipeline-model-parallel-layout')
    assert layout == option(settings(OLD)['args'], '--pipeline-model-parallel-layout')
    assert option(args, '--recompute-granularity') == 'selective'
    start, end = args.index('--recompute-modules') + 1, args.index('--no-gradient-accumulation-fusion')
    assert args[start:end] == ['layernorm', 'linear_attn', 'moe']
    assert '--mock-data' in args and '--no-gradient-accumulation-fusion' in args
    assert '--moe-flex-dispatcher-backend' not in args
    assert value['environment']['NCCL_P2P_DISABLE'] == '1' and value['environment']['TORCH_COMPILE_DISABLE'] == '0'
    assert value['procedure'] == {'kind': 'train', 'validation': '../validation/memory_trace.py'}
    assert not any(key in value['environment'] for key in ('MAX_JOBS', 'AITER_LOG_MORE', 'BEVERIN_AITER_ATTN', 'BEVERIN_TURBO_GG'))


def test_source_identity_and_auto_cache_image_separation():
    a, b = settings(OLD), settings(NEW)
    identity = json.loads(IDENTITY.read_text())
    assert identity['selection'] == b['source'] and identity['sha256'] == ARCHIVE
    assert b['source']['commit'] == 'e242f3c3c1136aeae498529bfde5e06bda9a7cca'
    assert a['cache'] == b['cache'] and b['cache']['mode'] == 'auto'
    # GPU is deliberately unresolved, not a claim to have probed a device or chosen a generation.
    descriptor = {'format': cache.SCHEMA, 'adapter': 3, 'image': BASE, 'source': ARCHIVE,
                  'gpu': '<future runtime-observed gcnArchName>', 'environment': a['environment'], 'prepare': a['prepare']}
    candidate = deepcopy(descriptor)
    candidate['image'] = CANDIDATE
    assert cache.identity(descriptor) != cache.identity(candidate)


def test_resolver_plan_and_cache_identity_have_no_subprocesses_or_writes(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('preparation resolver attempted subprocess/write')
    with monkeypatch.context() as guard:
        guard.setattr(subprocess, 'Popen', refuse)
        for name in ('write_text', 'write_bytes', 'mkdir'):
            guard.setattr(Path, name, refuse)
        original_open = Path.open
        def read_only(path, mode='r', *args, **kwargs):
            if any(flag in mode for flag in 'wax+'):
                refuse()
            return original_open(path, mode, *args, **kwargs)
        guard.setattr(Path, 'open', read_only)
        value = plan(NEW, site=ROOT / 'config.toml', submit=True)
        assert value['cache']['mode'] == 'auto' and value['resources']['nodes'] == 16
        assert '--ntasks=64' in value['step'] and '--ntasks-per-node=4' in value['step']
        assert cache.identity({'image': value['image']['sha256']})
    text = edf(value, '/future-run')
    assert 'HWLOC_COMPONENTS = "-gl"' in text
    assert 'com.hooks.cxi.enabled = "false"' in text and 'com.hooks.aws_ofi_nccl.enabled = "false"' in text
