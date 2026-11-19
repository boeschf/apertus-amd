"""Preregistered numerical-gated output-gate timing component; no production acceptance."""
import time
from collections.abc import Mapping

import argparse

import ast

import hashlib

import importlib.metadata

import importlib.util

import json

import math

import os

from pathlib import Path

import tempfile

IMAGE = '910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490'

GIT = '11ef10939935619f2c723ba326e3808a750a924e'

HIP = '7.16.26385'

VERSIONS = {'torch': '2.14.0+rocm10.1.0', 'transformer-engine': '2.17.1+130099ce4', 'triton': '3.8.0+git669b31ac.rocm10.1.0'}

ELEMENT_RTOL, ELEMENT_ATOL, RMS_RTOL, RMS_ATOL = 4 * 2**-8, 2**-8, 4 * 2**-8, 1e-5

METHOD_SHA = 'a58c58e24108f37f6e5582189481550ae8edb4b3c9e57f754d11e881d71da4f7'

HELPER_SHA = '1ea728c04b1ae0dc3bfd208612821da6ff2aba2dc4c8803939b583399eb02268'

CASES = ('actual-contiguous', 'actual-strided-offset', 'tail-4097', 'short-17')

METHOD = '''@jit_fuser
def _apply_output_gate(self, x, gate):
    x_dtype = x.dtype
    gate = gate.contiguous()
    gate = gate.view(*x.shape)
    x = x * torch.sigmoid(gate.float())
    x = x.to(x_dtype)
    return x
'''

HELPER = '''# Adapted from Imanol Schlag's b3-0005-fused-out-gate.patch.
# Default-off BF16 output gate; caller validates ordinary equal-sized CUDA tensors.
# FP32 sigmoid/multiply and original BF16 stores; no compiled-baseline gain claimed.
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_fwd(x_ptr, g_ptr, y_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + i, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + i, (x * tl.sigmoid(g)).to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _gate_bwd(x_ptr, g_ptr, dy_ptr, dx_ptr, dg_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + i, mask=m, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + i, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    tl.store(dx_ptr + i, (dy * s).to(dx_ptr.dtype.element_ty), mask=m)
    tl.store(dg_ptr + i, (dy * x * s * (1.0 - s)).to(dg_ptr.dtype.element_ty), mask=m)


_BLOCK = 4096


class _SigmoidGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        x = x.contiguous()
        g = gate.contiguous().view(x.shape)
        y = torch.empty_like(x)
        n = x.numel()
        _gate_fwd[(triton.cdiv(n, _BLOCK),)](x, g, y, n, BLOCK=_BLOCK, num_warps=8)
        ctx.save_for_backward(x, g)
        ctx.gshape = gate.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        x, g = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        dg = torch.empty_like(g)
        n = x.numel()
        _gate_bwd[(triton.cdiv(n, _BLOCK),)](x, g, dy, dx, dg, n, BLOCK=_BLOCK, num_warps=8)
        return dx, dg.view(ctx.gshape)


def sigmoid_gate(x, gate):
    """x * sigmoid(gate) (computed in fp32, returned in x.dtype); gate has x's numel."""
    return _SigmoidGate.apply(x, gate)
'''

def sha(data):
    """Hash component source bytes, never an image payload."""
    return hashlib.sha256(data).hexdigest()

def source_tree():
    """Verify frozen source; adapt only decorator and unused self, not the body."""
    if sha(METHOD.encode()) != METHOD_SHA or sha(HELPER.encode()) != HELPER_SHA:
        raise ValueError('embedded component source drift')
    tree = ast.parse(METHOD)
    node = tree.body[0]
    if ast.unparse(node.decorator_list[0]) != 'jit_fuser' or node.args.args[0].arg != 'self':
        raise ValueError('unexpected method adaptation')
    node.decorator_list = []
    node.args.args = node.args.args[1:]
    return ast.fix_missing_locations(tree)

def validate(settings, env):
    """Require proposed four-rank command resources and unchanged compilation defaults."""
    r, p = settings['resources'], settings['procedure']
    if (settings['image']['sha256'] != IMAGE or r['nodes'] != 1 or r['gpus_per_node'] != 4
            or r['cpus_per_node'] != 192 or r.get('reservation') != 'cscs' or r['time'] != '00:15:00'
            or r['mpi'] != 'pmix' or r['cpu_bind'] != 'map_ldom:0,1,2,3' or r['distribution'] != 'block:block'
            or p['kind'] != 'commands' or p['timeout'] != 720
            or p['commands'] != [['python3', '-u', '/apertus/validation.py', 'run']]):
        raise ValueError('unreviewed component envelope')
    rank, local, node, world = (int(env[k]) for k in ('SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NODEID', 'SLURM_NTASKS'))
    if not (world == 4 and node == 0 and 0 <= rank == local < 4) or len(os.sched_getaffinity(0)) != 48:
        raise ValueError('four local ranks/48-CPU masks required')
    if env.get('TORCH_COMPILE_DISABLE') != '0' or env.get('TORCHDYNAMO_DISABLE', '0') != '0' or env.get('OMP_NUM_THREADS') != '1' or env.get('MAX_JOBS') != '2':
        raise ValueError('compiled baseline, OMP1/MAX2 required; no settings changed')
    return rank, local

def metric_pass(m):
    """Apply proposed pointwise gates to every element and whole-tensor RMS."""
    try:
        counts = ('numel', 'mismatched_elements', 'bf16_mismatches', 'raw_bf16_mismatches', 'max_bf16_ulp')
        if m['finite'] is not True or any(type(m[k]) is not int or m[k] < 0 for k in counts) or m['numel'] == 0:
            return False
        if any(m[k] > m['numel'] for k in counts[1:4]) or m['max_bf16_ulp'] > 65535 or m['bf16_mismatches'] > m['raw_bf16_mismatches']:
            return False
        if (m['max_bf16_ulp'] == 0) != (m['bf16_mismatches'] == 0):
            return False
        return (not m['mismatched_elements'] and all(type(m[k]) in (int, float) and math.isfinite(m[k]) and m[k] >= 0 for k in ('max_abs', 'rms_error', 'reference_rms'))
                and m['rms_error'] <= RMS_ATOL + RMS_RTOL * m['reference_rms'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False

def compare(torch, actual, expected):
    """Reduce full tensors; ordered ULP mismatches collapse signed zero, raw bits do not."""
    a = actual.detach().reshape(-1)
    m = dict(finite=True, numel=a.numel(), mismatched_elements=0, bf16_mismatches=0, raw_bf16_mismatches=0, max_bf16_ulp=0, max_abs=0.0)
    se = sr = 0.0
    for start in range(0, a.numel(), 131072):
        v = a[start:start+131072].double()
        r = expected(start, start+v.numel()).double()
        if v.shape != r.shape:
            raise ValueError('reference shape mismatch')
        if not bool(torch.isfinite(v).all() and torch.isfinite(r).all()):
            m['finite'] = False
            continue
        e = (v - r).abs()
        m['mismatched_elements'] += int((e > ELEMENT_ATOL + ELEMENT_RTOL * r.abs()).sum().item())
        m['max_abs'] = max(m['max_abs'], e.max().item())
        se += e.square().sum().item(); sr += r.square().sum().item()
        words = [x.to(torch.bfloat16).contiguous().view(torch.int16).int() for x in (v, r)]
        m['raw_bf16_mismatches'] += int((words[0] != words[1]).sum().item())
        ordered = [torch.where(w < 0, -32768 - w, w) for w in words]
        ulp = (ordered[0] - ordered[1]).abs()
        m['max_bf16_ulp'] = max(m['max_bf16_ulp'], ulp.max().item())
        m['bf16_mismatches'] += int((ulp != 0).sum().item())
    m.update(rms_error=math.sqrt(se / m['numel']), reference_rms=math.sqrt(sr / m['numel']))
    m['passed'] = metric_pass(m)
    return m

def analytic(torch, values, start, end, key):
    """FP64 continuous derivatives of BF16 values/upstream, with final BF16 stores."""
    x, g, dy = [t.reshape(-1)[start:end].double() for t in values]
    s = g.sigmoid()
    result = {'forward': x * s, 'dx': dy * s, 'dgate': dy * x * s * (1 - s)}[key]
    return result.to(torch.bfloat16).double()

def tensors(torch, case, device):
    """Build actual/short cases with stress values and optional noncontiguous offset views."""
    xs, gs = ((4096, 1, 3584), (4096, 1, 28, 128)) if case.startswith('actual') else ((1, 1, 4097 if case == 'tail-4097' else 17),) * 2
    def make(shape):
        if case == 'actual-strided-offset':
            return torch.randn(*shape[:-1], 2 * shape[-1] + 1, device=device, dtype=torch.bfloat16)[..., 1::2]
        return torch.randn(shape, device=device, dtype=torch.bfloat16)
    x, g, dy = make(xs), make(gs), make(xs)
    for offset, value in enumerate((-80, 0, 80, 1e-3, -1e-3, -20, 20)):
        g[..., offset::8] = value
    x[..., 0] = 0; x[..., 1] = 64; x[..., 2] = -64
    return x, g, dy

def spec(t):
    """Retain actual shape/stride/offset, not a production layout claim."""
    return dict(shape=list(t.shape), stride=list(t.stride()), offset=t.storage_offset(), dtype=str(t.dtype))

def save(path, record):
    """Atomic evidence; preserve an active original failure if writing also fails."""
    import sys
    primary = sys.exception()
    try:
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note(f'evidence failure: {error}')

def expected_inputs(name):
    """Describe exact shapes/strides/offsets produced by the fixed case constructors."""
    xs, gs = ((4096, 1, 3584), (4096, 1, 28, 128)) if name.startswith('actual') else ((1, 1, 4097 if name == 'tail-4097' else 17),) * 2
    result = []
    for shape in (xs, gs, xs):
        backing = (*shape[:-1], 2 * shape[-1] + 1) if name == 'actual-strided-offset' else shape
        stride, size = [], 1
        for dim in reversed(backing):
            stride.insert(0, size); size *= dim
        if name == 'actual-strided-offset': stride[-1] = 2
        result.append(dict(shape=list(shape), stride=stride, offset=int(name == 'actual-strided-offset'), dtype='torch.bfloat16'))
    return result

def validate_record(r, rank, expected_sha):
    """Fail closed on missing cases/arms/tensors or sampled rather than full metrics."""
    if 'error' in r or any(type(r[k]) is not int for k in ('rank', 'local_rank', 'selected_device')) or r['timings'] is not False or r['sampling'] is not False:
        raise ValueError('invalid rank/timing/sampling evidence')
    if r['probe_sha256'] != expected_sha or not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(c not in '0123456789abcdef' for c in expected_sha):
        raise ValueError('run validation identity mismatch')
    if r['versions'] != VERSIONS or r['torch_git'] != GIT or r['torch_hip'] != HIP or r['policy'] != 'PROPOSED-parent-review-required':
        raise ValueError('version/policy provenance mismatch')
    if r['tolerances'] != dict(element_rtol=ELEMENT_RTOL, element_atol=ELEMENT_ATOL, rms_rtol=RMS_RTOL, rms_atol=RMS_ATOL) or any(type(v) not in (int, float) for v in r['tolerances'].values()):
        raise ValueError('proposed tolerance tuple mismatch')
    scratch = Path(r['scratch'])
    if not scratch.is_absolute() or scratch == Path('/') or '..' in scratch.parts or Path(r['helper_path']) != scratch / 'component-source/megatron/core/extensions/beverin_gate.py':
        raise ValueError('helper origin outside recorded private source')
    affinity = r['affinity']; initial = affinity['initial']
    if not isinstance(initial, list) or len(initial) != 48 or any(type(c) is not int or c < 0 for k in ('initial', 'post_gpu_init', 'final') for c in affinity[k]) or initial != sorted(set(initial)) or any(affinity[k] != initial for k in ('post_gpu_init', 'final')) or affinity['errors'] != [] or 'initialization_error' in affinity:
        raise ValueError('incomplete/changed 48-CPU affinity evidence')
    if r['status'] != 'passed' or r['image'] != IMAGE or r['rank'] != rank or r['local_rank'] != rank or r['selected_device'] != rank or r['helper_sha256'] != HELPER_SHA or r['method_sha256'] != METHOD_SHA or set(r['cases']) != set(CASES):
        raise ValueError('incomplete GPU numerical evidence')
    for name, case in r['cases'].items():
        if case['inputs'] != expected_inputs(name) or any(type(s['offset']) is not int or any(type(v) is not int for k in ('shape', 'stride') for v in s[k]) for s in case['inputs']):
            raise ValueError('input shape/stride/offset/dtype evidence mismatch')
        if set(case['arms']) != {'compiled-baseline', 'candidate'}:
            raise ValueError('missing numerical arm')
        count = 4096 * 3584 if name.startswith('actual') else 4097 if name == 'tail-4097' else 17
        for checks in [*case['arms'].values(), case['candidate_vs_compiled']]:
            if set(checks) != {'forward', 'dx', 'dgate'} or any(m['numel'] != count or m.get('passed') is not True or not metric_pass(m) for m in checks.values()):
                raise ValueError('missing/full-count numerical metrics failed')

def numerical(torch, baseline, candidate, record, device, provided):
    """Sequential baseline/candidate graphs; full forward/dx/dgate metrics, no timings."""
    record['cases'] = {}
    for case in CASES:
        values = provided[case]
        snapshots = [x.clone() for x in values]
        item = record['cases'][case] = {'inputs': [spec(x) for x in values], 'arms': {}}
        results = {}
        for name, operation in (('compiled-baseline', baseline), ('candidate', candidate)):
            x, g = [v.detach().requires_grad_() for v in values[:2]]
            if any(type(v) is not torch.Tensor or not v.is_cuda or v.dtype != torch.bfloat16 for v in (x, g)) or x.device != g.device or x.numel() != g.numel():
                raise ValueError('candidate admission mismatch')
            out = operation(x, g)
            out.backward(values[2])
            result = {'forward': out.detach().cpu(), 'dx': x.grad.detach().cpu(), 'dgate': g.grad.detach().cpu()}
            if any(not torch.equal(a, b) for a, b in zip(values, snapshots)):
                raise ValueError('input/upstream mutated')
            if result['forward'].shape != x.shape or result['dx'].shape != x.shape or result['dgate'].shape != g.shape:
                raise ValueError('output/gradient shape mismatch')
            results[name] = result
            del out, x, g
        cpu_values = [v.detach().cpu().contiguous().view(-1) for v in values]
        for name, outputs in results.items():
            checks = item['arms'][name] = {}
            for key, actual in outputs.items():
                checks[key] = compare(torch, actual, lambda s, e, k=key: analytic(torch, cpu_values, s, e, k))
        item['candidate_vs_compiled'] = {k: compare(torch, results['candidate'][k], lambda s, e, k=k: results['compiled-baseline'][k].reshape(-1)[s:e]) for k in results['candidate']}
        if not all(m['passed'] for checks in [*item['arms'].values(), item['candidate_vs_compiled']] for m in checks.values()):
            raise ValueError(f'numerical gate failed: {case}')
        del results, values, snapshots, cpu_values
    torch.cuda.synchronize()

PERF_CASES = CASES[:2]
WARM, WARM_BLOCKS, BLOCKS, BATCH = 20, 2, 12, 16


def order(rank, block):
    """Counterbalance within device without changing canonical input allocation order."""
    return 'ABBA' if (rank + block) % 2 == 0 else 'BAAB'


def case_order(rank):
    return PERF_CASES if rank % 2 == 0 else PERF_CASES[::-1]


def counter_sources():
    """Observe canonical package Python sources, not guessed native kernel symbols."""
    du = importlib.import_module('torch._dynamo.utils')
    im = importlib.import_module('torch._inductor.metrics')
    tj = importlib.import_module('triton.runtime.jit')
    result = {}
    for module, relative, token in ((du, 'torch/_dynamo/utils.py', 'counters'), (im, 'torch/_inductor/metrics.py', 'generated_kernel_count'), (tj, 'triton/runtime/jit.py', 'device_caches')):
        path = Path('/opt/venv/lib/python3.12/site-packages') / relative
        if Path(module.__file__).resolve() != path or path.resolve() != path or path.stat().st_size > 1048576:
            raise ValueError('UNQUALIFIED counter-source origin/size')
        text = path.read_text()
        if token not in text: raise ValueError('UNQUALIFIED counter-source field')
        result[relative] = dict(path=str(path), sha256=sha(text.encode()))
    return du, im, result


def counters(du, im, helper):
    """Read existing mappings/fields without resetting counters or populating caches."""
    result = {'dynamo': {}, 'generated_kernel_count': im.generated_kernel_count, 'triton_keys': {}}
    if not isinstance(du.counters, Mapping) or type(im.generated_kernel_count) is not int or im.generated_kernel_count < 0:
        raise ValueError('UNQUALIFIED compiler-counter types')
    for group in ('frames', 'stats', 'inductor', 'graph_break'):
        values = du.counters.get(group, {})
        if not isinstance(values, Mapping) or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in values.items()):
            raise ValueError('UNQUALIFIED Dynamo-counter types')
        result['dynamo'][group] = dict(sorted(values.items()))
    for name in ('_gate_fwd', '_gate_bwd'):
        caches = vars(getattr(helper, name)).get('device_caches')
        if not isinstance(caches, Mapping): raise ValueError('UNQUALIFIED Triton cache ABI')
        result['triton_keys'][name] = {}
        for device, value in caches.items():
            if not isinstance(value, tuple) or not value or not isinstance(value[0], Mapping):
                raise ValueError('UNQUALIFIED Triton cache-entry ABI')
            result['triton_keys'][name][str(device)] = sorted(sha(repr(k).encode()) for k in value[0])
    return result


def iterate(operation, leaves, dy, count):
    """Identical allocation/backward/release/reset work inside both arms' batches."""
    x, g = leaves
    for _ in range(count):
        out = operation(x, g)
        out.backward(dy)
        del out
        x.grad = None; g.grad = None


def sample(torch, operation, leaves, dy, events, device):
    """Three clocks for the same batch; endpoints/reset metrics are not compiler changes."""
    start, end = events
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter_ns(); start.record()
    iterate(operation, leaves, dy, BATCH)
    end.record(); returned = time.perf_counter_ns()
    end.synchronize(); completed = time.perf_counter_ns()
    elapsed = start.elapsed_time(end)
    return dict(status='complete', t0_ns=t0, return_ns=returned, complete_ns=completed, event_ms=elapsed,
        batch=BATCH, completion_us=(completed-t0)/1000/BATCH, host_return_us=(returned-t0)/1000/BATCH,
        device_span_us=elapsed*1000/BATCH, peak_allocated=torch.cuda.max_memory_allocated(device), peak_reserved=torch.cuda.max_memory_reserved(device))


def median(values):
    values = sorted(values); n = len(values)
    return (values[(n-1)//2] + values[n//2]) / 2


def decision(samples, rank):
    """This component's preregistered rule; never a universal optimization threshold."""
    result = {}
    for case in PERF_CASES:
        gains, na, nb, arms = [], [], [], {'A': [], 'B': []}
        for block in range(BLOCKS):
            rows = [s for s in samples if s['case'] == case and s['block'] == block]
            pair = {a: [s['completion_us'] for s in rows if s['arm'] == a] for a in 'AB'}
            a, b = sum(pair['A'])/2, sum(pair['B'])/2
            gains.append((a-b)/a); na.append(abs(pair['A'][0]-pair['A'][1])/a); nb.append(abs(pair['B'][0]-pair['B'][1])/b)
            for arm in 'AB': arms[arm] += pair[arm]
        noise = max(median(na), median(nb)); gain = median(gains)
        p95 = {a: sorted(v)[math.ceil(.95*len(v))-1] for a, v in arms.items()}
        result[case] = dict(rank=rank, block_gains=gains, median_gain=gain, noise=noise,
            positive_blocks=sum(g > 0 for g in gains), p95=p95,
            positive_timing=gain > max(.05, 2*noise) and sum(g > 0 for g in gains) >= 10 and p95['B'] <= 1.05*p95['A'],
            memory_review='required; cumulative allocator baseline/context and per-sample peaks retained')
    return result


def proof_ok(proof, expected_device=None):
    """Require complete typed snapshots, both helper caches and stable warm state."""
    try:
        before, pre, warm, measured = [proof[k] for k in ('before_pre', 'after_pre', 'after_warm', 'after_measure')]
        for index, snapshot in enumerate((before, pre, warm, measured)):
            if set(snapshot) != {'dynamo', 'generated_kernel_count', 'triton_keys'}:
                return False
            if type(snapshot['generated_kernel_count']) is not int or snapshot['generated_kernel_count'] < 0:
                return False
            groups = snapshot['dynamo']
            if (not isinstance(groups, Mapping) or not {'stats', 'graph_break'} <= set(groups)
                    or any(not isinstance(g, Mapping) or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in g.items()) for g in groups.values())):
                return False
            helpers = snapshot['triton_keys']
            if not isinstance(helpers, Mapping) or set(helpers) != {'_gate_fwd', '_gate_bwd'}:
                return False
            for devices in helpers.values():
                if not isinstance(devices, Mapping) or (index and not devices):
                    return False
                if index and expected_device is not None and set(devices) != {str(expected_device)}:
                    return False
                for device, keys in devices.items():
                    if (not isinstance(device, str) or not device or not isinstance(keys, list) or not keys
                            or any(not isinstance(k, str) or len(k) != 64 or k == '0'*64 or any(c not in '0123456789abcdef' for c in k) for k in keys)
                            or keys != sorted(set(keys))):
                        return False
        stats = pre['dynamo']['stats']; old = before['dynamo']['stats']
        return (stats.get('unique_graphs', 0) > old.get('unique_graphs', 0) and stats.get('calls_captured', 0) > old.get('calls_captured', 0)
                and pre['dynamo']['graph_break'] == before['dynamo']['graph_break'] and warm == measured)
    except (KeyError, TypeError, ValueError):
        return False


def validate_performance(r, rank, expected_sha):
    """Validate complete paired raw samples, units, order and evidence before native success."""
    if 'error' in r or r['status'] != 'passed' or type(r['rank']) is not int or r['rank'] != rank or r['probe_sha256'] != expected_sha or r['image'] != IMAGE or r['protocol'] != [WARM, WARM_BLOCKS, BLOCKS, BATCH] or not proof_ok(r['compiler'], rank):
        raise ValueError('UNQUALIFIED performance provenance/compiler proof')
    if set(r['phases']) != {'pre', 'post'} or any(v['status'] != 'passed' or len(v['sha256']) != 64 for v in r['phases'].values()):
        raise ValueError('missing strict numerical phases')
    for key, value in (('helper_sha256', HELPER_SHA), ('method_sha256', METHOD_SHA), ('versions', VERSIONS), ('torch_git', GIT), ('torch_hip', HIP)):
        if r['component'][key] != value: raise ValueError('component source/version drift')
    if r['component']['tolerances'] != dict(element_rtol=ELEMENT_RTOL, element_atol=ELEMENT_ATOL, rms_rtol=RMS_RTOL, rms_atol=RMS_ATOL) or Path(r['component']['helper_path']) != Path(r['scratch']) / 'component-source/megatron/core/extensions/beverin_gate.py':
        raise ValueError('component policy/import path mismatch')
    if set(r['counter_sources']) != {'torch/_dynamo/utils.py', 'torch/_inductor/metrics.py', 'triton/runtime/jit.py'} or any(len(v['sha256']) != 64 or Path(v['path']) != Path('/opt/venv/lib/python3.12/site-packages') / k for k, v in r['counter_sources'].items()):
        raise ValueError('missing canonical counter-source fingerprints')
    if set(r['warm_stability']) != set(PERF_CASES):
        raise ValueError('missing fixed warm-block snapshots')
    for snapshots in r['warm_stability'].values():
        if not isinstance(snapshots, list) or len(snapshots) != 2 or not proof_ok(
                dict(r['compiler'], after_warm=snapshots[0], after_measure=snapshots[1]), rank):
            raise ValueError('UNQUALIFIED warm compiler/cache instability')
    expected = [(case, block, slot, arm) for case in case_order(rank) for block in range(BLOCKS) for slot, arm in enumerate(order(rank, block))]
    if len(r['samples']) != len(expected) or r['operands_unchanged'] is not True:
        raise ValueError('incomplete samples/operand evidence')
    previous_complete = None
    for row, key in zip(r['samples'], expected):
        if any(type(row[k]) is not int for k in ('block', 'slot', 'batch')) or (row['case'], row['block'], row['slot'], row['arm']) != key or row['status'] != 'complete' or row['batch'] != BATCH:
            raise ValueError('sample pairing/order mismatch')
        if any(type(row[k]) is not int or row[k] < 0 for k in ('t0_ns', 'return_ns', 'complete_ns', 'peak_allocated', 'peak_reserved')) or not row['t0_ns'] <= row['return_ns'] <= row['complete_ns']:
            raise ValueError('invalid raw clock/memory types')
        if previous_complete is not None and row['t0_ns'] < previous_complete:
            raise ValueError('overlapping/reused sample clocks')
        previous_complete = row['complete_ns']
        for name, value in (('completion_us', (row['complete_ns']-row['t0_ns'])/1000/BATCH), ('host_return_us', (row['return_ns']-row['t0_ns'])/1000/BATCH), ('device_span_us', row['event_ms']*1000/BATCH)):
            if type(row[name]) not in (int, float) or not math.isfinite(row[name]) or row[name] < 0 or not math.isclose(row[name], value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('invalid named clock units')
        if row['completion_us'] <= 0 or type(row['event_ms']) not in (int, float) or not math.isfinite(row['event_ms']) or row['event_ms'] <= 0:
            raise ValueError('invalid completion/event interval')
    if r['decision'] != decision(r['samples'], rank): raise ValueError('decision differs from all raw samples')


def run(settings):
    """One outer commands PGID; all outcomes retain caches/samples and never auto-confirm."""
    from apertus_amd.config import CACHE_PATHS
    from apertus_amd.runtime import preserve_affinity, require_affinity
    rank, local = validate(settings, os.environ); expected = settings['validation']['sha256']
    paths = {phase: Path(f'output-gate-{phase}-{rank}.json') for phase in ('pre', 'post', 'performance')}
    if any(p.exists() for p in paths.values()): raise FileExistsError('existing component evidence')
    r = dict(status='unqualified', rank=rank, image=IMAGE, probe_sha256=sha(Path(__file__).read_bytes()), protocol=[WARM, WARM_BLOCKS, BLOCKS, BATCH], samples=[], phases={})
    provided = baseline = helper = None
    def phase(name):
        numeric = dict(base, status='failed', timings=False, sampling=False)
        try:
            numerical(torch, baseline, helper.sigmoid_gate, numeric, f'cuda:{local}', provided)
            numeric['status'] = 'passed'; validate_record(numeric, rank, expected)
        except BaseException as error:
            numeric.update(status='failed', error=str(error)); raise
        finally:
            primary = __import__('sys').exception(); save(paths[name], numeric)
            try: r['phases'][name] = dict(path=str(paths[name]), sha256=sha(paths[name].read_bytes()), status=numeric['status'])
            except BaseException as secondary:
                if primary is None: raise
                primary.add_note(f'numerical evidence hash failure: {secondary}')
    try:
        if r['probe_sha256'] != expected or {k: importlib.metadata.version(k) for k in VERSIONS} != VERSIONS: raise ValueError('frozen source/version mismatch')
        tree = source_tree(); root = Path(settings['scratch_root']); root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f'output-gate-perf-{os.environ["SLURM_JOB_ID"]}-{rank}-', dir=root)); r['scratch'] = str(scratch)
        for key, relative in CACHE_PATHS.items():
            cache = scratch / relative; cache.mkdir(); os.environ[key] = str(cache)
        path = scratch / 'component-source/megatron/core/extensions/beverin_gate.py'; path.parent.mkdir(parents=True); path.write_text(HELPER)
        base = dict(status='failed', image=IMAGE, rank=rank, local_rank=local, selected_device=local, probe_sha256=expected, versions=VERSIONS,
            torch_git=GIT, torch_hip=HIP, helper_sha256=HELPER_SHA, method_sha256=METHOD_SHA, helper_path=str(path), scratch=str(scratch),
            policy='PROPOSED-parent-review-required', tolerances=dict(element_rtol=ELEMENT_RTOL, element_atol=ELEMENT_ATOL, rms_rtol=RMS_RTOL, rms_atol=RMS_ATOL))
        with preserve_affinity(base.setdefault('affinity', {})) as affinity:
            import torch
            if torch.__version__ != VERSIONS['torch'] or torch.version.git_version != GIT or torch.version.hip != HIP or torch._dynamo.config.suppress_errors: raise ValueError('Torch/default compiler identity mismatch')
            torch.set_num_threads(1)
            if torch.cuda.device_count() != 4: raise ValueError('all four devices required')
            torch.cuda.set_device(local); prop = torch.cuda.get_device_properties(local)
            if torch.cuda.current_device() != local or not prop.gcnArchName.startswith('gfx942') or prop.multi_processor_count != 228: raise ValueError('local hardware mismatch')
            namespace = {'torch': torch}; exec(compile(tree, '<frozen-e242-output-gate>', 'exec'), namespace); baseline = torch.compile(namespace['_apply_output_gate'])
            spec_helper = importlib.util.spec_from_file_location('frozen_output_gate_perf', path); helper = importlib.util.module_from_spec(spec_helper); spec_helper.loader.exec_module(helper)
            if Path(helper.__file__).resolve() != path or sha(path.read_bytes()) != HELPER_SHA: raise ValueError('imported helper source drift')
            r['component'] = dict(base, helper_path=str(Path(helper.__file__).resolve()))
            du, im, sources = counter_sources(); r['counter_sources'] = sources
        require_affinity(affinity, 'performance component')
        torch.manual_seed(123); provided = {case: tensors(torch, case, f'cuda:{local}') for case in CASES}
        snapshots = {case: [v.clone() for v in values] for case, values in provided.items()}
        r['compiler'] = {'before_pre': counters(du, im, helper)}; phase('pre'); r['compiler']['after_pre'] = counters(du, im, helper)
        leaves = {case: [v.detach().requires_grad_() for v in provided[case][:2]] for case in PERF_CASES}; operations = {'A': baseline, 'B': helper.sigmoid_gate}
        events = {case: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for case in PERF_CASES}
        r['warm_stability'] = {}
        for case in PERF_CASES:
            warm_snapshots = r['warm_stability'][case] = []
            for arm in 'AB': iterate(operations[arm], leaves[case], provided[case][2], WARM)
            for block in range(WARM_BLOCKS):
                for arm in order(rank, block): iterate(operations[arm], leaves[case], provided[case][2], BATCH)
                warm_snapshots.append(counters(du, im, helper))
            if not proof_ok(dict(r['compiler'], after_warm=warm_snapshots[0], after_measure=warm_snapshots[1]), local):
                raise ValueError('UNQUALIFIED warm compiler/cache instability')
        torch.cuda.synchronize(); r['compiler']['after_warm'] = counters(du, im, helper)
        for case in case_order(rank):
            for block in range(BLOCKS):
                for slot, arm in enumerate(order(rank, block)):
                    row = dict(case=case, block=block, slot=slot, arm=arm, status='started'); r['samples'].append(row)
                    row.update(sample(torch, operations[arm], leaves[case], provided[case][2], events[case], local))
        r['compiler']['after_measure'] = counters(du, im, helper)
        r['operands_unchanged'] = all(torch.equal(v, old) for case in CASES for v, old in zip(provided[case], snapshots[case]))
        if not r['operands_unchanged']: raise ValueError('timed operands mutated')
        phase('post'); r['compiler']['after_post'] = counters(du, im, helper)
        if not proof_ok(r['compiler'], local): raise ValueError('UNQUALIFIED compilation/cache changes or missing proof')
        r['decision'] = decision(r['samples'], rank); r['status'] = 'passed'; validate_performance(r, rank, expected)
        require_affinity(affinity, 'performance completion')
    except BaseException as error:
        r.update(status='unqualified', error=f'{type(error).__name__}: {error}')
        if provided is not None and r['phases'].get('pre', {}).get('status') == 'passed' and 'post' not in r['phases']:
            try: phase('post')
            except BaseException as secondary: error.add_note(f'post-numerical failure: {secondary}')
        raise
    finally:
        save(paths['performance'], r)


def check(directory):
    """Offline frozen-run audit of four rank-paired timing/numerical records, never pooling."""
    expected = json.loads((directory / 'resolved.json').read_text())['validation']['sha256']
    if sha((directory / 'validation.py').read_bytes()) != expected: raise ValueError('frozen validation drift')
    records = []
    for rank in range(4):
        r = json.loads((directory / f'output/0/output-gate-performance-{rank}.json').read_text())
        for name in ('pre', 'post'):
            path = directory / f'output/0/output-gate-{name}-{rank}.json'
            if sha(path.read_bytes()) != r['phases'][name]['sha256']: raise ValueError('numeric-phase hash drift')
            validate_record(json.loads(path.read_text()), rank, expected)
        validate_performance(r, rank, expected); records.append(r)
    save(directory / 'output/performance-checks.json', dict(status='passed', per_rank=[r['decision'] for r in records],
        positive_timing_signal=all(d['positive_timing'] for r in records for d in r['decision'].values()),
        eligibility='requires memory review and separate independent confirmation; no production/image approval'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('run', 'check')); parser.add_argument('directory', nargs='?', type=Path); args = parser.parse_args()
    check(args.directory) if args.mode == 'check' else run(json.loads(Path('/apertus/resolved.json').read_text()))
