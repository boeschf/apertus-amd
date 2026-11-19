"""Isolated git/AST/stub and stdlib mathematics; never import Torch or Triton."""
import ast
import builtins
import hashlib
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / 'experiments/patches/chonk-fused-output-gate-e242.patch'
COMMIT = 'e242f3c3c1136aeae498529bfde5e06bda9a7cca'
TARGET = 'megatron/core/transformer/attention.py'
HELPER = 'megatron/core/extensions/beverin_gate.py'
PREP = ROOT / 'artifacts/investigations/performance-next/restart-20261116/imanol-patch-integration/output-gate'


@pytest.fixture(autouse=True)
def no_vendor_import(monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'triton', 'aiter', 'transformer_engine', 'megatron'):
            pytest.fail('host vendor import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)


def method(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.fixture
def applied(tmp_path):
    base = (PREP / 'base-attention.py.txt').read_bytes()
    pins = __import__('json').loads((PREP / 'PINS.json').read_text())
    assert hashlib.sha256(base).hexdigest() == pins['base_file_sha256']
    destination = tmp_path / TARGET
    destination.parent.mkdir(parents=True)
    destination.write_bytes(base)
    # Apply only in the private fixture, outside the external git checkout.
    env = dict(os.environ)
    for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'):
        env.pop(name, None)
    subprocess.run(['git', 'apply', '--check', str(PATCH)], cwd=tmp_path, env=env, check=True, capture_output=True)
    subprocess.run(['git', 'apply', str(PATCH)], cwd=tmp_path, env=env, check=True, capture_output=True)
    files = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*.py')}
    assert files == {TARGET, HELPER}
    return base.decode(), destination.read_text(), (tmp_path / HELPER).read_text()


def test_exact_apply_original_method_and_only_callsite_change(applied):
    base, changed, helper = applied
    old, new = ast.parse(base), ast.parse(changed)
    assert ast.dump(method(old, '_apply_output_gate')) == ast.dump(method(new, '_apply_output_gate'))
    assert ast.get_source_segment(base, method(old, '_apply_output_gate')) == ast.get_source_segment(changed, method(new, '_apply_output_gate'))
    core = method(new, '_core_attn_and_output_proj')
    outer = next(n for n in core.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'gate is not None')
    assert isinstance(outer.body[1], ast.Import) and outer.body[1].names[0].name == 'os'
    assert outer.body[2].body[0].module == 'megatron.core.extensions.beverin_gate'
    original = method(old, '_core_attn_and_output_proj')
    old_outer = next(n for n in original.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'gate is not None')
    outer.body = old_outer.body
    assert ast.dump(old) == ast.dump(new)  # No checkpoint/model/other method edits.
    ast.parse(helper)
    assert len(PATCH.read_text().splitlines()) <= 150


def test_original_kernel_autograd_AST_and_copy_contract(applied):
    _, _, helper = applied
    original = (ROOT / __import__('json').loads((PREP / 'PINS.json').read_text())['original_patch']).read_text()
    original_helper = '\n'.join(line[1:] for line in original.split('@@ -0,0 +1,62 @@\n')[1].splitlines() if line.startswith('+'))
    a, b = ast.parse(original_helper), ast.parse(helper)
    for name in ('_gate_fwd', '_gate_bwd', 'sigmoid_gate'):
        assert ast.dump(method(a, name)) == ast.dump(method(b, name))
    ca = next(n for n in a.body if isinstance(n, ast.ClassDef))
    cb = next(n for n in b.body if isinstance(n, ast.ClassDef))
    assert ast.dump(ca) == ast.dump(cb)
    assert 'x = x.contiguous()' in helper and 'g = gate.contiguous().view(x.shape)' in helper
    assert 'dy = dy.contiguous()' in helper and 'dg.view(ctx.gshape)' in helper
    assert 'ctx.save_for_backward(x, g)' in helper and '_BLOCK = 4096' in helper
    assert '0.5 ms' not in helper and 'TORCH_COMPILE_DISABLE' not in helper


class TensorFixture:
    def __init__(self, **overrides):
        self.is_cuda, self.dtype, self.device, self.layout, self.is_nested, self.count = True, 'bf16', 'cuda:0', 'strided', False, 16
        self.__dict__.update(overrides)

    def numel(self):
        return self.count


def route(applied, monkeypatch, x, gate, flag):
    _, changed, _ = applied
    outer = next(n for n in method(ast.parse(changed), '_core_attn_and_output_proj').body
                 if isinstance(n, ast.If) and ast.unparse(n.test) == 'gate is not None')
    calls = []
    def fused(x, gate):
        calls.append('fused')
        return 'candidate'
    original = builtins.__import__
    def imported(name, *args, **kwargs):
        if name == 'megatron.core.extensions.beverin_gate':
            calls.append('helper-import-stub')
            return SimpleNamespace(sigmoid_gate=fused)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', imported)
    if flag is None:
        monkeypatch.delenv('BEVERIN_FUSED_OUT_GATE', raising=False)
    else:
        monkeypatch.setenv('BEVERIN_FUSED_OUT_GATE', flag)
    def fallback(x, gate):
        calls.append('baseline')
        return 'original'
    namespace = {'torch': SimpleNamespace(Tensor=TensorFixture, bfloat16='bf16', strided='strided'),
                 'self': SimpleNamespace(_apply_output_gate=fallback), 'core_attn_out': x, 'gate': gate}
    fragment = ast.Module(body=outer.body[1:3], type_ignores=[])
    exec(compile(ast.fix_missing_locations(fragment), '<isolated-gate-route>', 'exec'), namespace)
    return namespace['core_attn_out'], calls


@pytest.mark.parametrize('flag', [None, '0', '', 'true'])
def test_default_off_never_imports_helper(applied, monkeypatch, flag):
    assert route(applied, monkeypatch, TensorFixture(), TensorFixture(), flag) == ('original', ['baseline'])


def test_supported_flag_on_invokes_candidate(applied, monkeypatch):
    assert route(applied, monkeypatch, TensorFixture(), TensorFixture(), '1') == ('candidate', ['helper-import-stub', 'fused'])


@pytest.mark.parametrize('side', ['x', 'gate'])
@pytest.mark.parametrize('override', [{'dtype': 'fp64'}, {'dtype': 'fp32'}, {'dtype': 'fp8'}, {'is_cuda': False},
    {'device': 'cuda:1'}, {'layout': 'sparse'}, {'is_nested': True}, {'count': 0}, {'count': 15}])
def test_unsupported_flag_on_uses_baseline(applied, monkeypatch, side, override):
    x, g = TensorFixture(), TensorFixture()
    (x if side == 'x' else g).__dict__.update(override)
    assert route(applied, monkeypatch, x, g, '1') == ('original', ['baseline'])


def test_tensor_subclasses_fall_back(applied, monkeypatch):
    class Subclass(TensorFixture):
        pass
    assert route(applied, monkeypatch, Subclass(), TensorFixture(), '1') == ('original', ['baseline'])


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    e = math.exp(value)
    return e / (1 + e)


@pytest.mark.parametrize('gate', [-1000, -80, -20, -1, 0, 1, 20, 80, 1000])
@pytest.mark.parametrize('x', [-7.5, 0.0, 2.0])
def test_stdlib_FP64_formula_gradients(gate, x):
    # Mathematical identities only: not Triton execution or BF16 cast equivalence.
    dy, epsilon = -0.75, 1e-4
    s = sigmoid(gate)
    dx, dg = dy * s, dy * x * s * (1 - s)
    loss = lambda xx, gg: dy * xx * sigmoid(gg)
    dxf = (loss(x + epsilon, gate) - loss(x - epsilon, gate)) / (2 * epsilon)
    dgf = (loss(x, gate + epsilon) - loss(x, gate - epsilon)) / (2 * epsilon)
    assert all(math.isfinite(v) for v in (s, dx, dg, dxf, dgf))
    assert math.isclose(dx, dxf, rel_tol=1e-8, abs_tol=1e-10)
    assert math.isclose(dg, dgf, rel_tol=1e-7, abs_tol=1e-10)
    if gate == 0:
        assert s == 0.5 and dg == dy * x / 4
