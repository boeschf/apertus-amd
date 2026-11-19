"""Owned e242 apply/AST/metadata/math checks; never import Torch or Triton."""
import ast
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'artifacts/investigations/performance-next/restart-20261116/sssglu-isolated-candidate-review'
PATCH=ROOT/'experiments/patches/chonk-fused-sssglu-e242.patch'
BASE='megatron/core/fusions/fused_bias_sssglu.py'
HELPER='megatron/core/fusions/beverin_sssglu.py'


class Tensor:
    def __init__(self,shape=(7,3840),dtype='bf16',**changes):
        self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device='cuda:0';self.is_cuda=True;self.layout='strided';self.is_nested=False;self.contiguous=True;self.offset=0
        self.__dict__.update(changes)
    def is_contiguous(self):return self.contiguous
    def storage_offset(self):return self.offset


def eligible(enabled=True):
    tree=ast.parse((REVIEW/'after'/BASE).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_beverin_sssglu_eligible')
    namespace=dict(_BEVERIN_FUSED_SSSGLU=enabled,torch=SimpleNamespace(Tensor=Tensor,bfloat16='bf16',float32='fp32',strided='strided'))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<owned eligibility>','exec'),namespace)
    return namespace['_beverin_sssglu_eligible']


def test_exact_owned_apply_and_paths(tmp_path):
    target=tmp_path/BASE;target.parent.mkdir(parents=True);target.write_bytes((REVIEW/'before'/BASE).read_bytes())
    subprocess.run(['git','apply','--check',str(PATCH)],cwd=tmp_path,check=True,capture_output=True)
    subprocess.run(['git','apply',str(PATCH)],cwd=tmp_path,check=True,capture_output=True)
    for name in (BASE,HELPER):
        assert (tmp_path/name).read_bytes()==(REVIEW/'after'/name).read_bytes()
        ast.parse((tmp_path/name).read_text())
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*.py'))==sorted([BASE,HELPER])


def test_default_off_and_supported_inputs():
    f=eligible();p=Tensor((7,1),'fp32')
    assert f(Tensor()) and f(Tensor(),p)
    assert f(Tensor((0,3840)),Tensor((0,1),'fp32'))
    assert not eligible(False)(Tensor(),p)
    text=(REVIEW/'after'/BASE).read_text()
    assert '_os.environ.get("BEVERIN_FUSED_SSSGLU", "0") == "1"' in text


@pytest.mark.parametrize('damage',['width','cpu','dtype','stride','offset','nested','prob_dtype','prob_shape','prob_device','prob_stride'])
def test_unsupported_metadata_falls_back(damage):
    x=Tensor();p=Tensor((7,1),'fp32')
    if damage=='width':x.shape=(7,4096)
    elif damage=='cpu':x.is_cuda=False
    elif damage=='dtype':x.dtype='fp32'
    elif damage=='stride':x.contiguous=False
    elif damage=='offset':x.offset=1
    elif damage=='nested':x.is_nested=True
    elif damage=='prob_dtype':p.dtype='bf16'
    elif damage=='prob_shape':p.shape=(7,)
    elif damage=='prob_device':p.device='cuda:1'
    else:p.contiguous=False
    assert not eligible()(x,p)


def test_original_baseline_functions_unchanged_and_narrow_callsite_guards():
    before=ast.parse((REVIEW/'before'/BASE).read_text());after=ast.parse((REVIEW/'after'/BASE).read_text())
    names={'bias_sssglu_impl','weighted_bias_sssglu_impl'}
    for node in before.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name not in names:
            assert ast.dump(node)==ast.dump(next(n for n in after.body if type(n) is type(node) and n.name==node.name))
    source=(REVIEW/'after'/BASE).read_text()
    assert 'bias is None and not fp8_input_store and not cpu_offload_input and _beverin_sssglu_eligible(input)' in source
    assert 'elif not fp8_input_store and weights is not None and _beverin_sssglu_eligible(input, weights)' in source
    helper=(REVIEW/'after'/HELPER).read_text()
    assert 'TORCH_COMPILE_DISABLE' not in helper and 'autotune' not in helper and 'matmul' not in helper
    assert 'gp.to(ctx.w_dtype).view(ctx.w_shape)' in helper and 'dtype=torch.float32' in helper


def test_fused_kernel_math_structure():
    helper=(REVIEW/'after'/HELPER).read_text()
    assert 'y = (0.5 + u / (1.0 + tl.abs(u))) * b' in helper
    assert 'acc += tl.sum(tl.where(mask, gate * b * g, 0.0), axis=1)' in helper
    assert 'ga = g * b / (den * den)' in helper and 'gb = g * gate' in helper
    assert 'g = g * p[:, None]' in helper and 'atomic' not in helper


def test_analytic_full_input_and_row_probability_gradients():
    def value(a,b,p):return (.5+(a-1)/(1+abs(a-1)))*b*p
    epsilon=1e-6
    for a in (-3.,-1.,0.,1.,2.,4.):
        for b in (-2.,0.,3.):
            for p in (-0.,0.,.25,1.,1.5):
                gate=.5+(a-1)/(1+abs(a-1));analytic=(b*p/(1+abs(a-1))**2,gate*p,gate*b)
                point=[a,b,p]
                for index,gradient in enumerate(analytic):
                    plus=point.copy();minus=point.copy();plus[index]+=epsilon;minus[index]-=epsilon
                    observed=(value(*plus)-value(*minus))/(2*epsilon)
                    assert math.isclose(observed,gradient,rel_tol=3e-6,abs_tol=5e-6)
    # Row probability gradient is the complete feature reduction, independent of p.
    row=[(-1.,2.,.25),(1.,-3.,2.),(2.,1.,-.5)]
    contributions=[(.5+(a-1)/(1+abs(a-1)))*b*g for a,b,g in row]
    def total(probability):return sum(value(a,b,probability)*g for a,b,g in row)
    observed=(total(.3+epsilon)-total(.3-epsilon))/(2*epsilon)
    assert math.isclose(observed,sum(contributions),rel_tol=1e-8,abs_tol=1e-8)
