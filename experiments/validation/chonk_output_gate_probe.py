"""Numerical-only component: default-compiled e242 gate versus frozen patched helper.

Pointwise budgets are a proposal requiring parent approval before any GPU run.
One process uses the existing commands PGID; private compiler caches are retained.
"""
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


def numerical(torch, baseline, candidate, record, device):
    """Sequential baseline/candidate graphs; full forward/dx/dgate metrics, no timings."""
    torch.manual_seed(123)
    record['cases'] = {}
    for case in CASES:
        values = tensors(torch, case, device)
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


def run(settings):
    """One commands-runtime process, retained rank-private compiler caches, no publication."""
    from apertus_amd.config import CACHE_PATHS
    from apertus_amd.runtime import preserve_affinity, require_affinity
    rank, local = validate(settings, os.environ)
    path = Path(f'output-gate-{rank}.json')
    if path.exists():
        raise FileExistsError(path)
    record = dict(status='failed', image=IMAGE, rank=rank, local_rank=local, policy='PROPOSED-parent-review-required', timings=False, sampling=False,
                  tolerances=dict(element_rtol=ELEMENT_RTOL, element_atol=ELEMENT_ATOL, rms_rtol=RMS_RTOL, rms_atol=RMS_ATOL))
    try:
        tree = source_tree()
        record['probe_sha256'] = sha(Path(__file__).read_bytes())
        if record['probe_sha256'] != settings['validation']['sha256']:
            raise ValueError('frozen validation source mismatch')
        if {k: importlib.metadata.version(k) for k in VERSIONS} != VERSIONS:
            raise ValueError('vendor version mismatch')
        root = Path(settings['scratch_root']); root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f'output-gate-{os.environ["SLURM_JOB_ID"]}-{rank}-', dir=root))
        record.update(scratch=str(scratch), cache_cleanup='retained until outer command process group is quiescent')
        for key, relative in CACHE_PATHS.items():
            cache = scratch / relative; cache.mkdir(); os.environ[key] = str(cache)
        helper = scratch / 'component-source/megatron/core/extensions/beverin_gate.py'; helper.parent.mkdir(parents=True); helper.write_text(HELPER)
        with preserve_affinity(record.setdefault('affinity', {})) as affinity:
            import torch
            if torch.__version__ != VERSIONS['torch'] or torch.version.git_version != GIT or not torch.version.hip:
                raise ValueError('Torch version/git/HIP mismatch')
            record.update(versions=VERSIONS, torch_git=torch.version.git_version, torch_hip=torch.version.hip)
            if torch._dynamo.config.suppress_errors:
                raise ValueError('silent compiler-error eager fallback is forbidden')
            torch.set_num_threads(1)
            if torch.cuda.device_count() != 4:
                raise ValueError('all four devices must remain visible')
            torch.cuda.set_device(local)
            prop = torch.cuda.get_device_properties(local)
            if not prop.gcnArchName.startswith('gfx942') or prop.multi_processor_count != 228 or torch.cuda.current_device() != local:
                raise ValueError('hardware/local-rank mismatch')
            namespace = {'torch': torch}; exec(compile(tree, '<frozen-e242-output-gate>', 'exec'), namespace)
            baseline = torch.compile(namespace['_apply_output_gate'])
            module_spec = importlib.util.spec_from_file_location('frozen_chonk_output_gate', helper)
            module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
            record.update(helper_path=str(Path(module.__file__).resolve()), helper_sha256=sha(helper.read_bytes()), method_sha256=METHOD_SHA, selected_device=local, pgid=os.getpgrp())
        require_affinity(affinity, 'gate numerical checks')
        numerical(torch, baseline, module.sigmoid_gate, record, f'cuda:{local}')
        require_affinity(affinity, 'gate completion')
        record['status'] = 'passed'
        validate_record(record, rank, settings['validation']['sha256'])
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save(path, record)


def check(run_path):
    """Require four independent GPU numerical records; CPU preparation is not success."""
    expected_sha = json.loads((run_path / 'resolved.json').read_text())['validation']['sha256']
    if sha((run_path / 'validation.py').read_bytes()) != expected_sha:
        raise ValueError('run frozen validation differs from resolved identity')
    records = [json.loads((run_path / f'output/0/output-gate-{r}.json').read_text()) for r in range(4)]
    for rank, r in enumerate(records):
        validate_record(r, rank, expected_sha)
    save(run_path / 'output/checks.json', {'status': 'passed', 'scope': 'component numerical only; checkpoint/optimizer/cost not checked'})


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('run', 'check')); parser.add_argument('directory', type=Path, nargs='?'); args = parser.parse_args()
    if args.mode == 'check':
        check(args.directory)
    else:
        run(json.loads(Path('/apertus/resolved.json').read_text()))
