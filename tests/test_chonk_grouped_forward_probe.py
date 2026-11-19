"""CPU fakes only; neither importing vendor packages nor running a scheduler/GPU."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"experiments/validation/chonk_grouped_forward_probe.py"


@pytest.fixture
def p(monkeypatch):
    original=builtins.__import__
    def guard(name,*a,**k):
        if name.split('.')[0] in {'torch','transformer_engine','triton','aiter'}: pytest.fail('host vendor import')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guard)
    spec=importlib.util.spec_from_file_location('grouped_forward_cpu',PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def metric(n=1):
    return dict(finite=True,numel=n,checked=n,mismatches=0,max_abs=0.,rms_error=0.,reference_rms=0.)


@pytest.mark.parametrize('key,value',[('finite',False),('finite',1),('checked',0),('mismatches',1),('mismatches',False),('max_abs',float('nan')),('rms_error',.01),('reference_rms',float('inf')),('numel',-1)])
def test_independent_strict_gates(p,key,value):
    m=metric();assert p.metrics_pass(m);m[key]=value;assert not p.metrics_pass(m)


def test_empty_and_extra_schema(p):
    assert p.metrics_pass(metric(0));assert not p.metrics_pass(metric()|{'waiver':True})
    assert not p.metrics_pass(metric(0)|{'max_abs':1.})


def test_pins_and_selected_natural_config(p):
    base=ROOT/'artifacts/investigations/performance-next/restart-20261116'
    assert {k:list(v) for k,v in p.PINS.items()}==json.loads((base/'grouped-forward-selector-implementation/installed-pins.json').read_text())
    config=json.loads((base/'grouped-kernel-config-review/gfx942-GMM.json').read_text())
    for splits in p.CASES.values():
        assert sum(splits) in (32768,22447)
        for name,k,n in p.PROJECTIONS:
            key='small_shapes' if sum(splits)<50000 and n<=2816 else 'n_heavy' if n/k>2 else 'invalid'
            assert key==('n_heavy' if name=='fc1' else 'small_shapes')
            assert all(type(v) is int and v>0 for v in config['gmm'][key].values())
    data=(base/'grouped-kernel-config-review/gfx942-GMM.json').read_bytes()
    assert (len(data),hashlib.sha256(data).hexdigest())==p.PINS[p.CONFIG]


@pytest.mark.parametrize('failure',[False,True])
def test_observer_forward_identity_error_and_restore(p,failure):
    token=object();calls=[];error=RuntimeError('native')
    def original(*a,**k):
        calls.append((a,k))
        if failure: raise error
        return token
    module=SimpleNamespace(**dict.fromkeys(p.ALIASES,original));expected=vars(module).copy()
    if failure:
        with pytest.raises(RuntimeError) as caught:
            with p.observe(module,expected) as counts: module.general_grouped_gemm(token,key=token)
        assert caught.value is error and sum(counts.values())==0
    else:
        with p.observe(module,expected) as counts: assert module.general_grouped_gemm(token,key=token) is token
        assert counts[p.ALIASES[0]]==1
    assert calls==[((token,),{'key':token})] and vars(module)==expected


def test_observer_ownership_and_wrong_alias(p):
    original=lambda:None;replacement=lambda:1;module=SimpleNamespace(**dict.fromkeys(p.ALIASES,original))
    with p.observe(module,vars(module).copy()): module.general_grouped_gemm=replacement
    assert module.general_grouped_gemm is replacement
    with pytest.raises(ValueError):
        with p.observe(module,dict.fromkeys(p.ALIASES,original)): pytest.fail('wrong identity')


def workers(p):
    checks=[dict(key=f'{name}/{case}',splits=splits,experts=[metric(rows*n) for rows in splits]) for name,_,n in p.PROJECTIONS for case,splits in p.CASES.items()]
    shared=dict(rank=0,seed=42,pgid=99,provenance={},origins={},precision={},mapped_elfs=[],layouts={},baseline_environment_initial=dict(p.BASELINE),baseline_environment_final=dict(p.BASELINE),driver_sha256=p.digest(PATH),checks=checks,
                operands={r['key']:dict(input='a'*64,weights=['b'*64]*16) for r in checks})
    return [dict(shared,status='passed',pid=100+i,selector=str(i),counts=dict(zip(p.ALIASES,(4,0) if i==0 else (0,4)))) for i in range(2)]


@pytest.mark.parametrize('damage',[None,'pid','pgid','branch','digest','coverage','numerical','empty'])
def test_pair_strict(p,damage):
    a,b=json.loads(json.dumps(workers(p)))
    if damage=='pid':b['pid']=a['pid']
    elif damage=='pgid':b['pgid']+=1
    elif damage=='branch':b['counts']=a['counts']
    elif damage=='digest':b['operands']['fc1/balanced']['input']='c'*64
    elif damage=='coverage':b['checks'][1]=b['checks'][0]
    elif damage=='numerical':b['checks'][0]['experts'][0]['mismatches']=1
    elif damage=='empty':b['checks'][1]['experts'][0]['numel']=1
    if damage is None:p.pair(a,b)
    else:
        with pytest.raises(ValueError):p.pair(a,b)


def test_environment_startup_no_override(p):
    p.environment({});p.environment({p.SELECTOR:'1'},'1')
    for env in ({p.SELECTOR:'1'},{'TE_HIPBLASLT_ALGO_SAVE':'table'},{'NVTE_USE_CK_GROUPED_GEMM':'1'}):
        with pytest.raises(ValueError):p.environment(env)


@pytest.mark.parametrize('failure',[None,'exit','timeout','quiescence'])
def test_supervisor_reap_private_cache_and_failure_retention(p,tmp_path,monkeypatch,failure):
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('SLURM_JOB_ID','999999')
    monkeypatch.setattr(p,'validate_launch',lambda *a:(0,0));monkeypatch.setattr(p,'quiescent',lambda:failure!='quiescence')
    monkeypatch.setattr(p,'environment',lambda *a:None)
    values=workers(p);pgid=p.os.getpgrp()
    for row in values:row['pgid']=pgid
    seen=[]
    class Child:
        def __init__(self,argv,**kwargs):
            i=len(seen);self.pid=100+i;self.row=values[i];self.directory=Path(argv[argv.index('--directory')+1]);seen.append(kwargs)
            assert kwargs['start_new_session'] is False
        def wait(self,timeout):
            if failure=='timeout':raise p.subprocess.TimeoutExpired('worker',timeout)
            (self.directory/'worker.json').write_text(json.dumps(self.row))
            return 1 if failure=='exit' else 0
    monkeypatch.setattr(p.subprocess,'Popen',Child)
    settings={'scratch_root':str(tmp_path/'scratch')}
    if failure:
        with pytest.raises((ValueError,p.subprocess.TimeoutExpired)):p.run(settings)
    else:p.run(settings)
    summary=json.loads((tmp_path/'grouped-forward-0/summary.json').read_text())
    assert summary['status']==('failed' if failure else 'passed')
    assert Path(summary['scratch']).exists()==bool(failure)
    if not failure:assert seen[0]['env']['TRITON_CACHE_DIR']!=seen[1]['env']['TRITON_CACHE_DIR']


def test_actual_resolver_admission(p,monkeypatch):
    from apertus_amd.config import resolve
    value=resolve(ROOT/'experiments/chonk-grouped-forward-selector-i910bfcf936d3',site=ROOT/'config.toml',submit=True,time='00:05:00',reservation='cscs')
    monkeypatch.setattr(p.os,'sched_getaffinity',lambda _:set(range(48)))
    for rank in range(4):assert p.validate_launch(value,dict(SLURM_PROCID=str(rank),SLURM_LOCALID=str(rank),SLURM_NODEID='0',SLURM_NTASKS='4'))==(rank,rank)


def test_preimport_pin_refusal(p,tmp_path,monkeypatch):
    path=tmp_path/'consumer.py';path.write_text('# exact\n')
    monkeypatch.setattr(p,'PACKAGE',tmp_path)
    monkeypatch.setattr(p,'PINS',{'consumer.py':(path.stat().st_size,p.digest(path))})
    monkeypatch.setattr(p.importlib.metadata,'version',lambda name:p.VERSIONS[name])
    assert p.provenance()['binary_attestation'] is False
    path.write_text('# drift\n')
    with pytest.raises(ValueError,match='installed pin'):p.provenance()


def test_no_backward_timing_or_precision_mutation(p):
    import ast
    tree=ast.parse(PATH.read_text())
    forbidden={'backward','set_float32_matmul_precision','manual_seed_all','Event','setsid'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='allow_tf32' for t in n.targets) for n in ast.walk(tree))


def test_actual_745389_baseline_environment_regression(p):
    path=ROOT/'artifacts/experiment-data/runs/chonk-kda-attention-shape-i910bfcf936d3/20261116T132351Z-748396a3/output/0/attention-shape-rank-0/worker.json'
    actual=json.loads(path.read_text())['routing_environment']
    flags={k:v for k,v in actual.items() if k.startswith('NVTE_USE_')}
    assert flags==p.BASELINE
    assert p.environment(flags)==flags
    for selector in ('0','1'):assert p.environment(flags|{p.SELECTOR:selector},selector)==flags
    for key in flags:
        with pytest.raises(ValueError):p.environment(flags|{key:'0'})
    with pytest.raises(ValueError):p.environment(flags|{'NVTE_USE_UNKNOWN':'0'})
    assert p.environment({})==dict.fromkeys(flags)


@pytest.mark.parametrize('failure',[False,True])
def test_device_selection_stub_and_source_order(p,failure):
    import ast
    calls=[];state={'device':0}
    def select(local):calls.append(local);state['device']=local if not failure else 0
    cuda=SimpleNamespace(device_count=lambda:4,set_device=select,current_device=lambda:state['device'],get_device_properties=lambda _:SimpleNamespace(gcnArchName='gfx942'))
    fake=SimpleNamespace(cuda=cuda,version=SimpleNamespace(hip='10.1'))
    if failure:
        with pytest.raises(ValueError):p.select_device(fake,3)
    else:p.select_device(fake,3)
    assert calls==[3]
    tree=ast.parse(PATH.read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    selected=next(n.lineno for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='select_device')
    te_import=next(n.lineno for n in ast.walk(worker) if isinstance(n,ast.Import) and any(a.name=='transformer_engine.pytorch' for a in n.names))
    assert selected<te_import
