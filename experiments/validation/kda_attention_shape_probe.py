"""Standalone nominal GQA accuracy qualification; no timing, tuning or training acceptance.

The stdlib parent owns one inherited-PGID worker; only a quiescent successful
worker permits private-cache cleanup. Failures leave scratch and primary errors.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

IMAGE = '910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490'
SOURCE = 'e242f3c3c1136aeae498529bfde5e06bda9a7cca'
PACKAGE = Path('/opt/venv/lib/python3.12/site-packages')
SEED = Path('/opt/aiter-jit')
SEED_HASH = '4876a08ad1c55d0892cd91ff361716f3ba44cf1499660509e2d2b8f3d954193b'
POLICY = Path('/opt/tests/apertus-2/primus/attention-accuracy.py')
POLICY_HASH = 'b5e72f177206d6b864971e4ad852ed8c58f835da0ff63c413e75b96a39f076e1'
ACCESSOR_HASH = 'cbc42f9e4a56b586a23bda2965122e8732115655be567462452c65bf01e768d7'
VERSIONS = {'torch': '2.14.0+rocm10.1.0', 'amd-aiter': '0.1.22.post1',
            'transformer-engine': '2.17.1+130099ce4', 'triton': '3.8.0+git669b31ac.rocm10.1.0'}
PINS = {
    'torch/nn/functional.py': '95ff403085bb179477a01df63acfddf59dfac0fb58829e99a9c8b5c2fe98899b',
    'torch/nn/attention/__init__.py': '0d92132ad4144612854f40dbadc74731055c7e72b1f16e93d4a000d3a72323eb',
    'aiter/__init__.py': '42cb63794d6567f864d03bcfd7f38519c5b1af0b2ad7f50f7a250a3c5fbf8882',
    'aiter/jit/core.py': 'ec2e1e51ddace53ec7999515bf8753fa76369915b29e834d81923916bec49e4f',
    'aiter/jit/utils/cpp_extension.py': '88bc7ddf34a6c1294b3a39b6a957b4ea17c6f837ed53fd89ef581785f49bd9bb',
    'aiter/ops/mha.py': 'd51713d062d42bcc58d9de316d5fbf5d817ca46edf574d1b7a3c06945da1ca2c',
    'aiter/ops/flydsl/fmha_kernels.py': '102e449b70e92ff3b15fe6f0ab7c74925c5067d78e972f7e369e83ecade12e6c',
    'aiter/ops/triton/attention/mha.py': '25808b75d6a13499254b8b0456359d3b8afe28dbfc0684fb71a69c4f6941dc07',
    'transformer_engine/pytorch/cpp_extensions/fused_attn.py': '7a8a3da25a7e81c460835bceb831d0dd4dbb131300b1bb3cd0b27d28bda003e6',
    'transformer_engine/pytorch/attention/dot_product_attention/backends.py': 'd891206b05d583af4522676fa876a1dd049066391fc4263250785554e97b906f',
    'transformer_engine/pytorch/attention/dot_product_attention/utils.py': 'a3fcd81a1d5342fd8ae533029a9af7af7cb822b10a80ba747692a6eece43d092',
    'transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py': 'd804c4ebc2dde5b09dcda5a9edd519ff50ab2458b0a8bec67035328f50c13200',
}
ELF = {'transformer_engine/libtransformer_engine.so': (401104592, '140dc626463ae6072121ca66a988c71954268f7b2c27b4d447894a97f2cba69a'),
       'transformer_engine/transformer_engine_torch.cpython-312-x86_64-linux-gnu.so': (3438144, '1bdeb20270594556bc705e2b9842439925cf70ab03611145d388a92d4cc9f620')}
COMMAND = ['python3', '-u', '/apertus/validation.py', 'run']
GPU_PLANNING_BYTES = 32 * 2**30
MIN_REPORTED_MEMORY_BYTES = 2 * GPU_PLANNING_BYTES
ATTENTION_FLAGS = {'NVTE_FLASH_ATTN': '1', 'NVTE_FUSED_ATTN': '1', 'NVTE_UNFUSED_ATTN': '1'}


def digest(path):
    """Hash only explicitly pinned consumers/ELFs, never an image payload."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    """Atomically retain state without masking an active primary failure."""
    primary = sys.exception()
    try:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, path)
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note(f'secondary evidence error: {error}')


def validate_launch(settings, env):
    """Reject unreviewed image/resources/environment before imports or scratch."""
    r, p = settings['resources'], settings['procedure']
    if (settings['image']['sha256'] != IMAGE or r['nodes'] != 1 or r['gpus_per_node'] != 4
            or r['cpus_per_node'] != 192 or r['time'] != '00:15:00' or r.get('reservation') != 'cscs'
            or r['distribution'] != 'block:block' or r['cpu_bind'] != 'map_ldom:0,1,2,3'
            or p['kind'] != 'commands' or p['timeout'] != 720 or p['commands'] != [COMMAND]):
        raise ValueError('unreviewed attention envelope')
    rank, local, node, world = (int(env[k]) for k in ('SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NODEID', 'SLURM_NTASKS'))
    if not (world == 4 and node == 0 and 0 <= rank == local < 4) or len(os.sched_getaffinity(0)) != 48:
        raise ValueError('requires four independent ranks with 48-CPU masks')
    if any(env.get(k) != v for k, v in {'OMP_NUM_THREADS': '1', 'MAX_JOBS': '16', 'HWLOC_COMPONENTS': '-gl', 'NCCL_P2P_DISABLE': '1'}.items()):
        raise ValueError('thread/network envelope mismatch')
    if any(env.get(k) != v for k, v in ATTENTION_FLAGS.items()):
        raise ValueError('baseline TE attention admission flags must all be 1')
    if 'PREBUILD_THREAD_NUM' in env or any(k.startswith('BEVERIN_') and v not in ('', '0') for k, v in env.items()):
        raise ValueError('broad optimization/build overrides forbidden')
    return rank, local


def provenance():
    """Check installed consumers and baked seed before vendor imports."""
    if {k: importlib.metadata.version(k) for k in VERSIONS} != VERSIONS:
        raise ValueError('package version mismatch')
    result = {}
    for relative, wanted in PINS.items():
        path = PACKAGE / relative
        if path.resolve(strict=True) != path or digest(path) != wanted:
            raise ValueError(f'consumer pin mismatch: {relative}')
        result[relative] = wanted
    for relative, (size, wanted) in ELF.items():
        path = PACKAGE / relative
        if path.resolve(strict=True) != path or path.stat().st_size != size or digest(path) != wanted:
            raise ValueError(f'ELF provenance mismatch: {relative}')
    if digest(POLICY) != POLICY_HASH or digest(SEED / 'seed.json') != SEED_HASH:
        raise ValueError('policy/seed identity mismatch')
    seed = json.loads((SEED / 'seed.json').read_text())
    if (seed['gpu_archs'] != 'gfx942' or seed['cu_num'] != 228
            or seed['packages'] != {k: VERSIONS[k] for k in ('torch', 'amd-aiter', 'triton')}
            or set(seed['modules']) != {'module_aiter_core.so'}):
        raise ValueError('seed metadata mismatch')
    for name, wanted in seed['modules'].items():
        if (SEED / name).is_symlink() or digest(SEED / name) != wanted:
            raise ValueError('seed core identity mismatch')
    return {'consumers': result, 'seed': seed, 'versions': VERSIONS, 'source_to_binary_attested': False}


def precision(torch):
    """Read the exact-revision verified public accessor, never set precision flags."""
    getter = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
    if hashlib.sha256(inspect.getsource(getter).encode()).hexdigest() != ACCESSOR_HASH:
        raise ValueError('installed precision getter differs from pinned source')
    allowed = getter()
    if allowed is not False:
        raise ValueError(f'MATH reduced-precision state must be False, observed {allowed!r}')
    return {'math_reduction_allowed': allowed, 'deterministic': torch.are_deterministic_algorithms_enabled(),
            'matmul_precision': torch.get_float32_matmul_precision()}


def reference(torch, inputs, upstream, dtype, tile=128, gradients=True):
    """Bound CPU score tiles; accumulate true seven-head KV gradients in reference dtype."""
    q, k, v = [x.detach().to(device='cpu', dtype=dtype).transpose(1, 2) for x in inputs]
    do = upstream.detach().to(device='cpu', dtype=dtype).transpose(1, 2)
    b, h, s, d = q.shape
    hk, sk = k.shape[1:3]
    if b != 1 or h % hk or sk != s or tile < 1:
        raise ValueError('reference requires dense square divisible GQA, B1')
    ratio, scale = h // hk, d ** -0.5
    out = torch.empty_like(q)
    grads = [torch.zeros_like(x) for x in (q, k, v)] if gradients else None
    for g in range(hk):
        head = slice(g * ratio, (g + 1) * ratio)
        kg, vg = k[:, g:g+1], v[:, g:g+1]
        for start in range(0, s, tile):
            stop = min(s, start + tile)
            qt, dt = q[:, head, start:stop], do[:, head, start:stop]
            scores = (qt * math.sqrt(scale)) @ (kg.transpose(-2, -1) * math.sqrt(scale))
            mask = torch.arange(sk)[None, :] > torch.arange(start, stop)[:, None]
            scores.masked_fill_(mask, -math.inf)
            p = scores.softmax(-1)
            out[:, head, start:stop] = p @ vg
            if gradients:
                dp = dt @ vg.transpose(-2, -1)
                ds = p * (dp - (p * dp).sum(-1, keepdim=True))
                grads[0][:, head, start:stop] = (ds @ kg) * scale
                grads[1][:, g:g+1] += ((ds.transpose(-2, -1) @ qt) * scale).sum(1, keepdim=True)
                grads[2][:, g:g+1] += (p.transpose(-2, -1) @ dt).sum(1, keepdim=True)
    return out.transpose(1, 2), None if grads is None else [x.transpose(1, 2) for x in grads]


def selfcheck(torch, sdpa_kernel, backend):
    """Worker-only tiny CPU check against monolithic FP32/FP64 SDPA autograd."""
    torch.manual_seed(123)
    for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-11)):
        q = torch.randn(1, 9, 4, 8, dtype=dtype, requires_grad=True)
        inputs = [q] + [torch.randn(1, 9, 2, 8, dtype=dtype, requires_grad=True) for _ in range(2)]
        do = torch.randn_like(q)
        with sdpa_kernel(backend.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(*[x.transpose(1, 2) for x in inputs], is_causal=True, enable_gqa=True).transpose(1, 2)
        expected = torch.autograd.grad(out, inputs, do)
        for tile in (1, 4, 9):
            actual, grads = reference(torch, inputs, do, dtype, tile)
            for a, e in zip([actual] + grads, [out] + list(expected)):
                torch.testing.assert_close(a, e, rtol=tolerance, atol=tolerance)
    return {'passed': True, 'FP32_tolerance': 1e-5, 'FP64_tolerance': 1e-11, 'tiles': [1, 4, 9]}


def spec(tensor):
    """Retain actual component storage, not a training-layout assertion."""
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'dtype': str(tensor.dtype), 'storage_offset': tensor.storage_offset()}


@contextmanager
def trace(module, names, record):
    """Observe verified Python boundaries without forcing routes or claiming native kernels."""
    originals = {}
    try:
        for name in names:
            original = getattr(module, name, None)
            if not callable(original):
                continue
            originals[name] = original
            def wrapper(*args, _fn=original, _name=name, **kwargs):
                record.append({'module': module.__name__, 'call': _name,
                               'tensors': [spec(x) for x in args if hasattr(x, 'storage_offset')][:4]})
                return _fn(*args, **kwargs)
            setattr(module, name, wrapper)
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def metrics(torch, actual, accurate, control, policy):
    """Apply unchanged whole-original-Q/K/V metrics, never per-group gates."""
    result = {}
    for name, a, r, c in zip(('q', 'k', 'v'), actual, accurate, control):
        if any(x is None or not torch.isfinite(x).all().item() for x in (a, r, c)):
            raise ValueError(f'missing/nonfinite {name} gradient')
        a, r, c = [x.detach().double().cpu() for x in (a, r, c)]
        norm = torch.linalg.vector_norm(r).item()
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('invalid FP64 reference norm')
        ae, ce = a - r, c - r
        result[name] = policy.validate_gradient_error(ae.abs().max().item(), ce.abs().max().item(),
            torch.linalg.vector_norm(ae).item() / norm, torch.linalg.vector_norm(ce).item() / norm)
    return result


def qualify(torch, mha, dpa_module, backends, sdpa_kernel, backend, policy, record, local):
    """Sequential full-shape arms and ONE full-head control; accuracy only, no timings."""
    torch.manual_seed(123)
    values = [torch.randn(1, 4096, h, 128, dtype=torch.bfloat16) for h in (28, 7, 7)]
    do = torch.randn_like(values[0])
    expected, _ = reference(torch, values, do, torch.float32, gradients=False)
    precise_out, accurate = reference(torch, values, do, torch.float64)
    del precise_out
    if not torch.isfinite(expected).all().item():
        raise ValueError('nonfinite FP32 reference output')
    inputs = [x.to(device=f'cuda:{local}').requires_grad_() for x in values]
    upstream = do.to(device=f'cuda:{local}')
    with sdpa_kernel(backend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(*[x.transpose(1, 2) for x in inputs],
            attn_mask=None, dropout_p=0.0, is_causal=True, scale=None, enable_gqa=True).transpose(1, 2)
    out.backward(upstream)
    controls = [x.grad.detach().cpu() for x in inputs]
    del out, inputs
    record['arms'] = {}
    for arm in ('te', 'aiter'):
        inputs = [x.to(device=f'cuda:{local}').requires_grad_() for x in values]
        item = record['arms'][arm] = {'inputs': [spec(x) for x in inputs], 'calls': []}
        with trace(mha, ('_flash_attn_forward', '_flash_attn_backward', 'fmha_v3_fwd', 'mha_fwd', 'fmha_v3_bwd', 'mha_bwd'), item['calls']), trace(backends, ('fused_attn_fwd', 'fused_attn_bwd'), item['calls']):
            if arm == 'aiter':
                item['public_binding'] = 'aiter.ops.mha.flash_attn_func'
                out = mha.flash_attn_func(*inputs, dropout_p=0.0, causal=True, deterministic=False, return_lse=True, how_v3_bf16_cvt=1)[0]
            else:
                model = dpa_module.DotProductAttention(28, 128, num_gqa_groups=7, attention_dropout=0.0, qkv_format='sbhd', attn_mask_type='causal', tp_size=1)
                model.train()
                entered = [x.transpose(0, 1) for x in inputs]
                item['entry'] = [spec(x) for x in entered]
                raw = model(*entered)
                item['exit'] = spec(raw)
                if tuple(raw.shape) != (4096, 1, 3584):
                    raise ValueError('unexpected TE SB(H*D) output')
                out = raw.reshape(4096, 1, 28, 128).transpose(0, 1)
                item['backend'] = {k: str(v) for k, v in dpa_module._attention_backends.items() if k not in ('attention_params',)}
            item['output'] = spec(out)
            if not torch.isfinite(out).all().item():
                raise ValueError('nonfinite component output')
            torch.testing.assert_close(out.float().cpu(), expected, rtol=0.03, atol=0.03)
            out.backward(upstream)
        item['gradients'] = metrics(torch, [x.grad for x in inputs], accurate, controls, policy)
        del out, inputs
        if arm == 'te':
            del raw, entered, model
    torch.cuda.synchronize()
    record['timing'] = 'deferred; Python route observations are not native-kernel proof'
    record['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(local)
    record['peak_host_rss_bytes'] = __import__('resource').getrusage(0).ru_maxrss * 1024


def validate_hardware(hardware):
    """Check identity and reported planning headroom, not physical capacity or peak fit."""
    if not hardware['arch'].startswith('gfx942') or hardware['cu'] != 228:
        raise ValueError('requires gfx942/228CU hardware identity')
    if hardware['reported_memory_bytes'] < MIN_REPORTED_MEMORY_BYTES:
        raise ValueError(f'reported memory {hardware["reported_memory_bytes"]} bytes below 64GiB planning floor; not a peak-fit guarantee')


def worker(directory, local):
    """Import vendor code only in the private inherited-PGID worker after source checks."""
    from apertus_amd.runtime import preserve_affinity, require_affinity
    record = {'status': 'failed', 'pgid': os.getpgrp(), 'source': SOURCE, 'scope': 'nominal accuracy only',
              'planning_bytes': {'host': 4 * 2**30, 'gpu': GPU_PLANNING_BYTES, 'proven_peak': False}}
    try:
        with preserve_affinity(record.setdefault('affinity', {})) as affinity:
            import torch
            record['precision'] = precision(torch)
            torch.set_num_threads(1)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            record['tiny_selfcheck'] = selfcheck(torch, sdpa_kernel, SDPBackend)
            if torch.cuda.device_count() != 4 or not torch.version.hip:
                raise ValueError('all four ROCm devices must remain visible')
            torch.cuda.set_device(local)
            record['selected_device'] = torch.cuda.current_device()
            if record['selected_device'] != local:
                raise ValueError('selected device differs from Slurm local rank')
            prop = torch.cuda.get_device_properties(local)
            record['hardware'] = {'arch': prop.gcnArchName, 'cu': prop.multi_processor_count,
                'reported_memory_bytes': prop.total_memory, 'nominal_physical_class_bytes': 128 * 2**30,
                'minimum_reported_memory_bytes': MIN_REPORTED_MEMORY_BYTES, 'peak_fit_proved': False}
            validate_hardware(record['hardware'])
            import aiter.ops.mha as mha
            from aiter.jit import core
            import importlib
            dpa = importlib.import_module('transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention')
            backends = importlib.import_module('transformer_engine.pytorch.attention.dot_product_attention.backends')
            record['routing_environment'] = {k: v for k, v in os.environ.items() if k.startswith(('NVTE_', 'AITER_', 'ENABLE_CK', 'FLASH_ATTENTION_'))}
            record['ENABLE_CK'] = core.ENABLE_CK
            for module, relative in ((mha, 'aiter/ops/mha.py'), (dpa, 'transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py')):
                if Path(module.__file__).resolve() != PACKAGE / relative:
                    raise ValueError('executed consumer imports differ from pins')
            maps = Path('/proc/self/maps').read_text()
            record['mapped_TE'] = {}
            for relative, (size, wanted) in ELF.items():
                path = PACKAGE / relative
                if str(path) not in maps or path.stat().st_size != size:
                    raise ValueError('expected TE library not mapped')
                record['mapped_TE'][str(path)] = {'bytes': size, 'provenance_sha256': wanted}
            spec_policy = importlib.util.spec_from_file_location('pinned_attention_policy', POLICY)
            policy = importlib.util.module_from_spec(spec_policy)
            spec_policy.loader.exec_module(policy)
        require_affinity(affinity, 'attention checks')
        torch.cuda.reset_peak_memory_stats(local)
        qualify(torch, mha, dpa, backends, sdpa_kernel, SDPBackend, policy, record, local)
        require_affinity(affinity, 'attention completion')
        record['status'] = 'passed'
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if 'selected_device' in record:
            try:
                record['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(local)
            except BaseException as error:
                record['peak_read_error'] = str(error)
        save(directory / 'worker.json', record)


def quiescent():
    """Do not clean caches until all other live members of the outer PGID exit."""
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == os.getpgrp() and fields[0] not in ('Z', 'X'):
            return False
    return True


def run(settings):
    """One private worker under the unchanged commands-runtime process-group deadline."""
    from apertus_amd.config import CACHE_PATHS
    rank, local = validate_launch(settings, os.environ)
    directory = Path(f'attention-shape-rank-{rank}').resolve()
    directory.mkdir()
    record = {'status': 'failed', 'cleanup': 'retained', 'image': IMAGE, 'execution_scope': 'accuracy qualification only'}
    try:
        record['provenance'] = provenance()
        root = Path(settings['scratch_root'])
        root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f'attention-shape-{os.environ["SLURM_JOB_ID"]}-{rank}-', dir=root))
        record['scratch'] = str(scratch)
        env = dict(os.environ)
        for key, relative in CACHE_PATHS.items():
            cache = scratch / relative
            cache.mkdir()
            env[key] = str(cache)
        for name in record['provenance']['seed']['modules']:
            shutil.copyfile(SEED / name, Path(env['AITER_JIT_DIR']) / name)
        with (directory / 'native.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--directory', str(directory)],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
            status = process.wait(timeout=690)
        completed = json.loads((directory / 'worker.json').read_text())
        if status or completed['status'] != 'passed' or completed['pgid'] != os.getpgrp():
            raise ValueError(f'attention worker failed: {completed.get("error", status)}')
        if not quiescent():
            raise ValueError('worker descendants remain; scratch retained')
        if digest(SEED / 'seed.json') != SEED_HASH or any(digest(SEED / name) != wanted for name, wanted in record['provenance']['seed']['modules'].items()):
            raise ValueError('immutable seed changed')
        shutil.rmtree(scratch)
        record.update(status='passed', cleanup='removed')
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save(directory / 'summary.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker'))
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if (args.mode == 'worker') != (args.directory is not None):
        parser.error('only private worker requires directory')
    settings = json.loads(Path('/apertus/resolved.json').read_text())
    _, local = validate_launch(settings, os.environ)
    worker(args.directory, local) if args.mode == 'worker' else run(settings)
