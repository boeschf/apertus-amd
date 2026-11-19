"""Vendor-free CPU contracts for the deliberately incomplete SSSGLU component."""
import ast
import builtins
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'experiments/validation/chonk_sssglu_qualification.py'
REVIEW=ROOT/'artifacts/investigations/performance-next/restart-20261116/sssglu-isolated-candidate-review'


@pytest.fixture
def p(monkeypatch):
    original=builtins.__import__
    def guard(name,*a,**kw):
        if name.split('.')[0] in {'torch','triton','transformer_engine','megatron','aiter','uccl'}:pytest.fail('vendor import')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guard)
    spec=importlib.util.spec_from_file_location('sssglu_cpu',SCRIPT);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_exact_assets_and_only_declared_source_adapters(p,tmp_path):
    assert p.IMPLEMENTATION_READY is True and p.TILE==64 and p.D==1920
    assert p.FAMILIES=={'plain':(4096,129,1,0),'weighted':(32768,129,1,0)}
    files={'before':REVIEW/'before/megatron/core/fusions/fused_bias_sssglu.py','applied':REVIEW/'after/megatron/core/fusions/fused_bias_sssglu.py','helper':REVIEW/'after/megatron/core/fusions/beverin_sssglu.py'}
    for name,path in files.items():assert getattr(p,name.upper()).encode()==path.read_bytes() and p.sha(path.read_bytes())==p.ASSET_SHA[name]
    identity=p.export_sources(tmp_path/'owned')
    assert 'torch.compile' in identity['adapter'] and 'backend=' not in identity['adapter']
    for name,info in identity['exports'].items():
        source=p.BEFORE if name=='baseline' else p.APPLIED
        old,new=ast.parse(source),ast.parse(Path(info['path']).read_text())
        for node in old.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                other=next(n for n in new.body if type(n) is type(node) and n.name==node.name)
                assert ast.dump(node)==ast.dump(other)
    assert Path(identity['helper_path']).read_text()==p.HELPER
    assert len({info['path'] for info in identity['exports'].values()})==3


def test_preexisting_module_ownership_rejected_before_writes(p,tmp_path,monkeypatch):
    monkeypatch.setitem(p.sys.modules,'megatron.unrelated',object())
    with pytest.raises(ValueError,match='preexisting'):p.export_sources(tmp_path/'refuse')
    assert not (tmp_path/'refuse').exists()


def test_import_time_flag_and_apply_proxy_restore_once(p,tmp_path,monkeypatch):
    path=tmp_path/'export.py';path.write_text('import os\nflag=os.environ.get("BEVERIN_FUSED_SSSGLU")\n')
    monkeypatch.setenv('BEVERIN_FUSED_SSSGLU','prior')
    try:
        off=p.load_export('sssglu_test_off',path,False);on=p.load_export('sssglu_test_on',path,True)
        assert off.flag=='0' and on.flag=='1'
        assert p.sys.modules['sssglu_test_off'] is off and p.sys.modules['sssglu_test_on'] is on
        with pytest.raises(ValueError,match='preexisting component'):p.load_export('sssglu_test_on',path,True)
    finally:
        p.sys.modules.pop('sssglu_test_off',None);p.sys.modules.pop('sssglu_test_on',None)
    assert p.os.environ['BEVERIN_FUSED_SSSGLU']=='prior'
    value=(object(),None);calls=[]
    class Original:
        @staticmethod
        def apply(*args):calls.append(args);return value
    module=SimpleNamespace(Function=Original);seen={}
    with p.observe_class(module,'Function',seen):assert module.Function.apply(value) is value
    assert module.Function is Original and calls==[(value,)] and seen=={'Function':1}
    with pytest.raises(RuntimeError):
        with p.observe_class(module,'Function',seen):raise RuntimeError('original')
    assert module.Function is Original


def metric(count):
    return dict(finite=True,numel=count,checked=count,mismatches=0,max_abs=0.,rms_error=0.,reference_rms=0.,raw_mismatches=0,max_ulp=0,passed=True)


@pytest.mark.parametrize('damage',['count-bool','finite','mismatch','rms','empty','policy-type','policy-value'])
def test_closed_metric_and_policy_failures(p,damage):
    m=metric(0 if damage=='empty' else 1);assert p.metric_pass(m)
    if damage.startswith('policy'):
        policy=dict(p.POLICY);policy['rms_atol']=True if damage=='policy-type' else .1
        with pytest.raises(ValueError,match='policy'):p.validate_policy(policy)
        return
    if damage=='count-bool':m['checked']=True
    elif damage=='finite':m['finite']=False
    elif damage=='mismatch':m['mismatches']=1
    elif damage=='rms':m['rms_error']=1.
    else:m['max_abs']=1.
    assert not p.metric_pass(m)


@pytest.mark.parametrize('family',['plain','weighted'])
@pytest.mark.parametrize('failure',[None,'hash','numerical'])
def test_actual_numerical_orchestration_fixed_operands_and_partial_failure(p,monkeypatch,family,failure):
    class Value:
        shape=(1,1920)
        dtype,device='fixture-dtype','fixture-device'
        def stride(self):return (3841,2)
        def storage_offset(self):return 1
    cases={rows:dict(host=(Value(),Value(),Value()),device=(Value(),Value(),Value()),strided_upstream=family=='weighted' and rows==129) for rows in p.FAMILIES[family]}
    calls=[];hashes=[]
    def hashed(torch,value):
        hashes.append(value)
        return 'b'*64 if failure=='hash' and len(hashes)==4 else 'a'*64
    monkeypatch.setattr(p,'tensor_hash',hashed)
    def evaluate(torch,module,selected,values):
        calls.append((module,selected,values));return dict.fromkeys(('y','dx_first','dx_second','dp') if selected=='weighted' else ('y','dx_first','dx_second'),Value())
    monkeypatch.setattr(p,'evaluate',evaluate)
    def compare(torch,actual,reference,rows):
        result=metric(rows*1920)
        if failure=='numerical':result.update(passed=False,mismatches=1)
        return result
    monkeypatch.setattr(p,'compare',compare)
    record={}
    if failure:
        with pytest.raises(ValueError):p.numerical_cases(None,family,cases,'baseline','candidate',record)
        assert len(record['cases'])==1
    else:
        p.numerical_cases(None,family,cases,'baseline','candidate',record)
        assert [r['rows'] for r in record['cases']]==list(p.FAMILIES[family])
        assert len(calls)==8 and all(calls[i][2] is calls[i+1][2] for i in range(0,8,2))
        assert sum(len(r['metrics']) for r in record['cases'])==(48 if family=='weighted' else 36)


def test_timeout_terminate_kill_reap_preserves_primary(p,monkeypatch):
    order=[];primary=p.subprocess.TimeoutExpired('native',470)
    monkeypatch.setattr(p.time,'monotonic',lambda:0.)
    class Child:
        n=0
        def wait(self,timeout):
            order.append(('wait',timeout));self.n+=1
            if self.n==1:raise primary
            if self.n==2:raise p.subprocess.TimeoutExpired('terminate',timeout)
            return -9
        def terminate(self):order.append('terminate')
        def kill(self):order.append('kill')
    with pytest.raises(p.subprocess.TimeoutExpired) as error:p.wait_worker(Child(),480)
    assert error.value is primary and order==[('wait',470.),'terminate',('wait',5),'kill',('wait',5)]


def test_actual_resolver_and_invalid_native_cli_admission(p,monkeypatch):
    from apertus_amd.config import resolve
    settings=resolve(ROOT/'experiments/chonk-sssglu-qualification-i910bfcf936d3',site=ROOT/'config.toml',submit=True,reservation='cscs',time='00:10:00')
    monkeypatch.setattr(p.os,'sched_getaffinity',lambda _:set(range(48)))
    for rank in range(4):
        env=dict(settings['environment'],SLURM_PROCID=str(rank),SLURM_LOCALID=str(rank),SLURM_NODEID='0',SLURM_NTASKS='4')
        assert p.validate_launch(settings,env)==rank
    with pytest.raises(FileNotFoundError):p.check(Path('/never'))
    monkeypatch.setattr(p.sys,'argv',['validation.py','worker'])
    with pytest.raises(SystemExit):p.main()


def test_staged_real_number_equations_finite_differences(p):
    # This tests equations only, not the Torch/Triton implementation or rounding.
    def f(a,b,prob):return (.5+(a-1)/(1+abs(a-1)))*b*prob
    step=1e-6
    for a in (-80.,0.,1.,1.0078125,80.):
        for prob in (-0.,0.,.25,2.5):
            b=-.25;gate=.5+(a-1)/(1+abs(a-1));expected=(b*prob/(1+abs(a-1))**2,gate*prob,gate*b)
            for axis,derivative in enumerate(expected):
                plus=[a,b,prob];minus=plus.copy();plus[axis]+=step;minus[axis]-=step
                assert math.isclose((f(*plus)-f(*minus))/(2*step),derivative,rel_tol=2e-5,abs_tol=1e-5)
