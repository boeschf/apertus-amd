"""Vendor-free checks for upstream source identity, invocation and failure ownership."""
import ast
import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'experiments/validation/uccl_library_ep_tests.py'
ASSETS = ROOT/'artifacts/investigations/uccl-upstream-tests-20261119/frozen'


@pytest.fixture
def p(monkeypatch):
    original = builtins.__import__
    def guard(name, *a, **k):
        if name.split('.')[0] in {'torch','uccl','deep_ep','triton','transformer_engine','megatron'}:
            pytest.fail('host vendor import')
        return original(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', guard)
    spec = importlib.util.spec_from_file_location('uccl_library_fixture', SCRIPT)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings():
    from apertus_amd.config import resolve
    return resolve(ROOT/'experiments/uccl-library-ep-i910bfcf936d3',site=ROOT/'config.toml',
                   submit=True,reservation='cscs',time='00:30:00')


def test_exact_sources_and_all_upstream_test_assertions_preserved(p,settings):
    bench,metadata=p.sources(ASSETS,settings)
    assert len(metadata['files'])==10 and bench==ASSETS/'upstream/ep/bench'
    baseline=ASSETS.parent/'base-source'
    for name in metadata['files']:
        if name.endswith('.py') and name!='ep/bench/buffer.py':
            assert (baseline/name).read_bytes()==(ASSETS/'upstream'/name).read_bytes()
            assert ast.dump(ast.parse((baseline/name).read_text()))==ast.dump(ast.parse((ASSETS/'upstream'/name).read_text()))
    assert metadata['files']['ep/bench/buffer.py']=='d30ad5231f476c726c246240eec9b6a2f9124e16752900ca562653623e77a768'


def test_all_rank_launch_admissions_and_override_rejection(p,settings,monkeypatch):
    monkeypatch.setattr(p.os,'sched_getaffinity',lambda _:set(range(48)))
    for rank in range(8):
        env=dict(settings['environment'],HWLOC_COMPONENTS='-gl',SLURM_PROCID=str(rank),SLURM_LOCALID=str(rank%4),SLURM_NODEID=str(rank//4),SLURM_NTASKS='8')
        for phase in p.PHASES:assert p.validate_launch(settings,env,phase)==(rank,rank%4,rank//4)
        with pytest.raises(ValueError,match='Config'):p.validate_launch(settings,env|{'UCCL_EP_COMBINE_CONFIG':'20,4,256,8,128'},'internode')
    settings['procedure']['timeout']=361
    with pytest.raises(ValueError,match='launch contract'):p.validate_launch(settings,env,'internode')


def test_atomic_evidence_never_overwrites(p,tmp_path):
    path=tmp_path/'record.json';p.write_new(path,{'status':'original'})
    with pytest.raises(FileExistsError):p.write_new(path,{'status':'replacement'})
    assert json.loads(path.read_text())=={'status':'original'}
    assert list(tmp_path.iterdir())==[path]


@pytest.mark.parametrize('phase',['intranode','internode','low_latency','dual_mode'])
def test_actual_upstream_function_selection_and_teardown(p,monkeypatch,phase):
    calls=[];group=object()
    class Buffer:
        def __init__(self,*a,**k):calls.append(('buffer',a,k))
        @staticmethod
        def get_low_latency_rdma_size_hint(*a):calls.append(('hint',a));return 4096
        def destroy(self):calls.append(('destroy',))
    def test_main(*a,**k):calls.append(('test_main',a,k));return 17
    module=SimpleNamespace(test_main=test_main,test_loop=lambda *a:calls.append(('test_loop',a)),
        main=lambda *a:calls.append(('main',a)),compute_buffer_sizes=lambda *a:(1024,2048))
    utils=SimpleNamespace(init_dist_under_torchrun=lambda *a:(3,8,group))
    torch=SimpleNamespace(manual_seed=lambda r:calls.append(('seed',r)),
        distributed=SimpleNamespace(barrier=lambda:calls.append(('barrier',)),destroy_process_group=lambda:calls.append(('destroy_group',))))
    original=builtins.__import__
    monkeypatch.setattr(builtins,'__import__',lambda name,*a,**k:torch if name=='torch' else original(name,*a,**k))
    result=p.invoke(phase,module,utils,Buffer,3,3)
    if phase=='intranode':
        assert calls[0][0]=='test_loop' and calls[0][1][:2]==(3,4)
        assert calls[0][1][2].hidden==7168
    elif phase=='dual_mode':
        assert calls[0][0]=='main' and calls[0][1][:2]==(3,4)
        assert calls[0][1][2].num_experts==16
    else:
        test=next(c for c in calls if c[0]=='test_main')
        assert test[2]['skip_benchmark'] is True
        assert calls[-3:]==[('barrier',),('destroy',),('destroy_group',)] and not p.KEEPALIVE
        if phase=='internode':assert test[1][1:7]==(64,3,4,8,2,3)
        else:
            assert test[1][:6]==(128,7168,288,8,3,8)
            assert test[2]['use_logfmt'] is False and test[2]['dispatch_use_fp8'] is True


def test_native_assertion_is_not_suppressed_or_followed_by_collective_teardown(p,monkeypatch):
    primary=AssertionError('upstream original assertion');calls=[]
    class Buffer:
        def __init__(self,*a,**k):pass
        def destroy(self):pytest.fail('unsafe failure-path buffer collective')
    def reject(*a,**k):raise primary
    module=SimpleNamespace(test_main=reject,compute_buffer_sizes=lambda *a:(1024,2048))
    utils=SimpleNamespace(init_dist_under_torchrun=lambda *a:(0,8,object()))
    original=builtins.__import__
    torch=SimpleNamespace(manual_seed=lambda r:None,distributed=SimpleNamespace(barrier=lambda:pytest.fail('failure collective')))
    monkeypatch.setattr(builtins,'__import__',lambda name,*a,**k:torch if name=='torch' else original(name,*a,**k))
    with pytest.raises(AssertionError) as caught:p.invoke('internode',module,utils,Buffer,0,0)
    assert caught.value is primary and len(p.KEEPALIVE)==1


def test_source_drift_rejected_before_vendor_work(p,settings,tmp_path):
    import shutil
    shutil.copytree(ASSETS,tmp_path/'assets')
    (tmp_path/'assets/upstream/ep/bench/test_internode.py').write_text('# wrong source\n')
    with pytest.raises(ValueError,match='file identity'):p.sources(tmp_path/'assets',settings)


def test_rendezvous_maps_local_and_global_ranks_without_spawn(p,tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','999999')
    p.write_new(tmp_path/'rdzv-node-1.json',dict(job_id='999999',address='nid-test',port=12345))
    env=p.rendezvous(tmp_path,'intranode',7,3,1)
    assert env['RANK']=='3' and env['WORLD_SIZE']=='4' and env['LOCAL_WORLD_SIZE']=='4'
    p.write_new(tmp_path/'rdzv-world.json',dict(job_id='999999',address='nid-zero',port=12346))
    env=p.rendezvous(tmp_path,'internode',7,3,1)
    assert env['RANK']=='7' and env['WORLD_SIZE']=='8'
