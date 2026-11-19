"""Actual proof orchestration with fake native math, never native vendor imports."""
import ast
import builtins
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'experiments/validation/chonk_sssglu_qualification.py'


@pytest.fixture
def p(monkeypatch):
    original=builtins.__import__
    def guard(name,*a,**kw):
        if name.split('.')[0] in {'torch','triton','megatron','transformer_engine','aiter','uccl'}:pytest.fail('real vendor import')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guard)
    spec=importlib.util.spec_from_file_location('sssglu_native_proof_fixture',SCRIPT);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def metric():
    return dict(finite=True,numel=7,checked=7,mismatches=0,max_abs=0.,rms_error=0.,reference_rms=1.,raw_mismatches=0,max_ulp=0,passed=True)


def harness(p,monkeypatch,family):
    class Tensor:
        is_cuda=True; layout='strided'; is_nested=False; device='cuda:0'; ndim=2
        def __init__(self,shape=(7,3840),dtype='bf16',offset=0,contiguous=True):self.shape,self.dtype,self.offset,self.contiguous=shape,dtype,offset,contiguous
        def is_contiguous(self):return self.contiguous
        def storage_offset(self):return self.offset
        def detach(self):return self
        def cpu(self):return self
    graph=[0]
    class Kernel:
        def __init__(self):self.cache={}
        def __getitem__(self,grid):
            def launch():self.cache['0']=['a'*64]
            return launch
    helper=SimpleNamespace(_fwd=Kernel(),_bwd=Kernel())
    original_name='WeightedSSSGLUFunction' if family=='weighted' else 'SSSGLUFunction'
    selected='TritonWeightedSSSGLU' if family=='weighted' else 'TritonSSSGLU'
    class Original:
        @staticmethod
        def apply(*args):graph[0]+=1;return Tensor((7,1920))
    class Selected:
        @staticmethod
        def apply(x,*args):
            if x.shape[0]:helper._fwd[None]();helper._bwd[None]()
            return Tensor((x.shape[0],1920))
    setattr(helper,selected,Selected)
    torch=SimpleNamespace(Tensor=Tensor,bfloat16='bf16',float32='fp32',strided='strided')
    guard_node=next(n for n in ast.parse(p.APPLIED).body if isinstance(n,ast.FunctionDef) and n.name=='_beverin_sssglu_eligible')
    def module(enabled=None):
        m=SimpleNamespace();setattr(m,original_name,Original)
        ns=dict(torch=torch,_BEVERIN_FUSED_SSSGLU=enabled)
        exec(compile(ast.Module(body=[guard_node],type_ignores=[]),'<actual source guard>','exec'),ns)
        m._beverin_sssglu_eligible=ns['_beverin_sssglu_eligible'];m._BEVERIN_FUSED_SSSGLU=enabled
        def public(x,bias,prob=None):
            admitted=enabled and m._beverin_sssglu_eligible(x,prob if family=='weighted' else None)
            function=getattr(helper,selected) if admitted else getattr(m,original_name)
            return function.apply(x,prob) if family=='weighted' else function.apply(x)
        m.bias_sssglu_impl=public;m.weighted_bias_sssglu_impl=public
        return m
    baseline,off,candidate=module(),module(False),module(True)
    def tiny(torch,rank,name='ordinary'):
        width=17280 if name=='dense' else 1920
        return Tensor((7,2*width),offset=name=='input_offset',contiguous=name!='input_stride'),Tensor((7,1),'bf16' if name=='prob_bf16' else 'fp32',offset=name=='prob_offset'),width
    monkeypatch.setattr(p,'tiny_inputs',tiny)
    monkeypatch.setattr(p,'compare',lambda *a:metric())
    monkeypatch.setattr(p,'evaluate',lambda torch,m,f,values:{'y':p.public_forward(m,f,values[0],values[1])})
    def snapshot():return dict(dynamo={'frames':{},'stats':dict(unique_graphs=graph[0],calls_captured=graph[0]),'inductor':{},'graph_break':{}},generated_kernel_count=graph[0],triton_keys={n:deepcopy(getattr(helper,n).cache) for n in ('_fwd','_bwd')})
    # Snapshots intentionally bind original JIT owners, like bind_helper does.
    kernels={n:getattr(helper,n) for n in ('_fwd','_bwd')}
    def snapshot():return dict(dynamo={'frames':{},'stats':dict(unique_graphs=graph[0],calls_captured=graph[0]),'inductor':{},'graph_break':{}},generated_kernel_count=graph[0],triton_keys={n:deepcopy(k.cache) for n,k in kernels.items()})
    return torch,Tensor,helper,baseline,off,candidate,snapshot


@pytest.mark.parametrize('family',['plain','weighted'])
def test_actual_default_off_admitted_empty_and_fallback_orchestrators(p,monkeypatch,family):
    torch,Tensor,helper,base,off,candidate,snapshot=harness(p,monkeypatch,family);record={}
    p.prove_default_off(torch,family,base,off,snapshot,record,0)
    proof=record['default_off'];assert proof['baseline_original_calls']==proof['default_off_original_calls']==1
    assert proof['helper_loaded_before'] is proof['helper_loaded_after'] is False
    for rows in (7,0):
        entry={'rows':rows};values=(Tensor((rows,3840)),Tensor((rows,1),'fp32'),None)
        p.evaluate_proven(torch,base,family,values,entry,'baseline',snapshot,helper,0)
        p.evaluate_proven(torch,candidate,family,values,entry,'candidate',snapshot,helper,0)
        assert entry['dispatch']==dict(baseline_original=1,candidate_original=0,candidate_helper=1)
        assert entry['launches']==dict(_fwd=int(rows>0),_bwd=int(rows>0))
        assert set(entry['compiler'])=={'before_baseline','after_baseline','before_candidate','after_candidate'}
    p.prove_fallbacks(torch,family,base,candidate,helper,0,record)
    assert len(record['fallbacks'])==(5 if family=='weighted' else 3)
    assert all(r['guard'] is False and r['original_calls']==1 and r['helper_calls']==0 for r in record['fallbacks'])


@pytest.mark.parametrize('damage',['off-enabled','helper-loaded','no-compile'])
def test_default_off_failures(p,monkeypatch,damage):
    torch,Tensor,helper,base,off,candidate,snapshot=harness(p,monkeypatch,'plain')
    if damage=='off-enabled':off._BEVERIN_FUSED_SSSGLU=True
    elif damage=='helper-loaded':monkeypatch.setitem(p.sys.modules,p.HELPER_MODULE,object())
    else:
        constant=snapshot();snapshot=lambda:deepcopy(constant)
    with pytest.raises(ValueError):p.prove_default_off(torch,'plain',base,off,snapshot,{},0)


@pytest.mark.parametrize('damage',['guard-miss','empty-launch','wrong-device','native-error'])
def test_candidate_failures_preserve_partial_dispatch(p,monkeypatch,damage):
    torch,Tensor,helper,base,off,candidate,snapshot=harness(p,monkeypatch,'plain');rows=0 if damage=='empty-launch' else 7
    if damage=='guard-miss':candidate._beverin_sssglu_eligible=lambda *a:False
    if damage=='empty-launch':
        class Selected:
            @staticmethod
            def apply(x):helper._fwd[None]();return x
        helper.TritonSSSGLU=Selected
    if damage=='wrong-device':
        original=snapshot
        def snapshot():
            result=original()
            for devices in result['triton_keys'].values():
                if '0' in devices:devices['1']=devices.pop('0')
            return result
    if damage=='native-error':
        def evaluate(*a):raise RuntimeError('native primary')
        monkeypatch.setattr(p,'evaluate',evaluate)
    entry={'rows':rows}
    with pytest.raises((ValueError,RuntimeError)):p.evaluate_proven(torch,candidate,'plain',(Tensor((rows,3840)),Tensor((rows,1),'fp32'),None),entry,'candidate',snapshot,helper,0)
    assert 'before_candidate' in entry['compiler']
    if damage=='native-error':assert 'native primary' in entry['proof_error']


def test_load_cleanup_and_proxy_foreign_ownership(p,tmp_path,monkeypatch):
    path=tmp_path/'broken.py';path.write_text('raise RuntimeError("load failure")\n')
    with pytest.raises(RuntimeError):p.load_export('failed_owned_sssglu',path,False)
    assert 'failed_owned_sssglu' not in p.sys.modules
    sentinel=object();monkeypatch.setitem(p.sys.modules,'already_owned_sssglu',sentinel)
    with pytest.raises(ValueError,match='preexisting'):p.load_export('already_owned_sssglu',path,False)
    assert p.sys.modules['already_owned_sssglu'] is sentinel
    original=object();foreign=object();module=SimpleNamespace(Function=original)
    with p.observe_class(module,'Function',{}):module.Function=foreign
    assert module.Function is foreign


@pytest.mark.parametrize('warning',[None,'recompile_limit','CACHE_SIZE_LIMIT',"WON\x27T CONVERT",'Falling back to eager'])
def test_closed_log_fingerprint_and_warning_detection(p,tmp_path,warning):
    path=tmp_path/'native.log';path.write_text(warning or 'ordinary native output\n')
    result=p.audit_log(path)
    assert result['sha256']==p.sha(path.read_bytes()) and result['bytes']==path.stat().st_size
    assert bool(result['warnings']) is (warning is not None)


def test_log_cap_precedes_read(p):
    path=SimpleNamespace(stat=lambda:SimpleNamespace(st_size=8*2**20+1),read_bytes=lambda:pytest.fail('oversize log read'))
    with pytest.raises(ValueError,match='8MiB'):p.audit_log(path)


def test_exact_counter_sources_and_readonly_helper_cache_abi(p,tmp_path,monkeypatch):
    assets=ROOT/'artifacts/investigations/performance-next/restart-20261116/output-gate-counter-source-review'
    modules={}
    for relative,(size,checksum) in p.COUNTER_PINS.items():
        source=assets/relative;assert source.stat().st_size==size and p.sha(source.read_bytes())==checksum
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(source.read_bytes())
        name={'torch/_dynamo/utils.py':'torch._dynamo.utils','torch/_inductor/metrics.py':'torch._inductor.metrics','triton/runtime/jit.py':'triton.runtime.jit'}[relative]
        modules[name]=SimpleNamespace(__file__=str(path))
    modules['torch._dynamo.utils'].counters={}
    modules['torch._inductor.metrics'].generated_kernel_count=0
    monkeypatch.setattr(p,'PACKAGE',tmp_path);monkeypatch.setattr(p.importlib,'import_module',modules.__getitem__)
    record={};snapshot,jit=p.compiler_reader(record)
    assert len(record['counter_sources'])==3 and snapshot()['triton_keys']=={'_fwd':{},'_bwd':{}}
    helper_path=tmp_path/'helper.py';helper_path.write_text(p.HELPER)
    class JIT:
        def __init__(self):self.fn=lambda:None;self.device_caches={}
    jit.JITFunction=JIT;helper=SimpleNamespace(__file__=str(helper_path),_fwd=JIT(),_bwd=JIT())
    monkeypatch.setattr(p.inspect,'getfile',lambda obj:tmp_path/'triton/runtime/jit.py' if obj is JIT else helper_path)
    read=p.bind_helper(snapshot,jit,helper,helper_path)
    assert read()['triton_keys']=={'_fwd':{},'_bwd':{}} and not helper._fwd.device_caches
    helper._fwd.device_caches={0:({'native-key':object()}, {}, 'target')}
    assert len(read()['triton_keys']['_fwd']['0'][0])==64
    helper._fwd.device_caches={0:[]}
    with pytest.raises(ValueError,match='tuple'):read()
    modules['torch._inductor.metrics'].generated_kernel_count=True
    with pytest.raises(ValueError,match='ABI'):snapshot()
