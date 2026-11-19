"""Vendor-free strict worker/rank/offline SSSGLU closure fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'experiments/validation/chonk_sssglu_qualification.py'
spec=importlib.util.spec_from_file_location('sssglu_records_cpu',SCRIPT)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
SHA=p.sha(SCRIPT.read_bytes())


def metric(n):
    return dict(finite=True,numel=n,checked=n,mismatches=0,max_abs=0.,rms_error=0.,reference_rms=0.,raw_mismatches=0,max_ulp=0,passed=True)


def snapshot(rank, populated=False):
    return dict(dynamo={k:{} for k in ('frames','stats','inductor','graph_break')},generated_kernel_count=0,
        triton_keys={k:{str(rank):['a'*64]} if populated else {} for k in ('_fwd','_bwd')})


def case(family,rows,rank):
    nominal=family=='plain' and rows==4096;strided=family=='weighted' and rows==129
    layout=lambda shape,stride,offset,dtype:dict(shape=shape,stride=stride,offset=offset,dtype=dtype,device=f'cuda:{rank}')
    x=layout([rows,1,3840] if nominal else [rows,3840],[3840,3840,1] if nominal else [3840,1],0,'torch.bfloat16')
    prob=layout([rows,1],[1,1],0,'torch.float32')
    g=layout([rows,1,1920] if nominal else [rows,1920],[1920,1920,1] if nominal else [3841,2] if strided else [1920,1],int(strided),'torch.bfloat16')
    names=['y','dx_first','dx_second']+(['dp'] if family=='weighted' else [])
    compiler={k:snapshot(rank) for k in p.SNAPSHOTS};compiler['after_candidate']=snapshot(rank,bool(rows))
    return dict(rows=rows,strided_upstream=strided,metrics={f'{comparison}/{name}':metric(rows if name=='dp' else rows*1920) for comparison in ('baseline/oracle','candidate/oracle','candidate/baseline') for name in names},
        host_hashes=['b'*64]*3,device_pre=['b'*64]*3,device_post=['b'*64]*3,upstream={k:g[k] for k in ('shape','stride','offset')},layouts=[x,prob,g],
        dispatch=dict(baseline_original=1,candidate_original=0,candidate_helper=1),compiler=compiler,launches=dict.fromkeys(('_fwd','_bwd'),int(rows>0)))


def worker(family,rank,scratch):
    owned=Path(scratch)/family/'component-source';before=snapshot(rank);after=snapshot(rank)
    after['dynamo']['stats']=dict(unique_graphs=1,calls_captured=1);after['generated_kernel_count']=1
    names=['sssglu','sssglu_back']+(['weighted_sssglu','weighted_sssglu_back'] if family=='weighted' else [])
    fallback_names=['input_stride','input_offset','dense']+(['prob_bf16','prob_offset'] if family=='weighted' else [])
    start=100 if family=='plain' else 300
    return dict(status='passed',proof_status='passed',rank=rank,family=family,pid=1000+rank*2+int(family=='weighted'),pgid=2000+rank,driver_sha256=SHA,policy=dict(p.POLICY),image=p.IMAGE,source=p.SOURCE,
        asset_sha256=dict(p.ASSET_SHA),started_ns=start,finished_ns=start+100,versions=dict(p.VERSIONS),local_device=rank,torch_git=p.GIT,torch_hip=p.HIP,
        affinity=dict(initial=list(range(48)),post_gpu_init=list(range(48)),final=list(range(48)),restoration_attempted=False,restoration_applied=False,errors=[]),
        adaptation=p.expected_adaptation(owned),compiler_suppression=False,baseline_wrappers={n:dict(wrapper_type='builtins.function',original_path=str(owned/'baseline.py')) for n in names},
        counter_sources={n:dict(path=str(p.PACKAGE/n),bytes=size,sha256=h) for n,(size,h) in p.COUNTER_PINS.items()},
        default_off=dict(flag=False,helper_loaded_before=False,helper_loaded_after=False,baseline_original_calls=1,default_off_original_calls=1,compiler_baseline_before=before,compiler_baseline_after=after,metric=metric(7*1920)),
        helper_observer=dict(installation='explicit owned import after default-off proof to install observers; not production lazy-import attestation',path=str(owned/'megatron/core/fusions/beverin_sssglu.py')),
        cases=[case(family,rows,rank) for rows in p.FAMILIES[family]],fallbacks=[dict(name=n,original_calls=1,helper_calls=0,guard=False,rows=7,width=17280 if n=='dense' else 1920,metric=metric(7*(17280 if n=='dense' else 1920))) for n in fallback_names])


def rank_record(rank):
    scratch=f'/dev/shm/apertus-amd/sssglu-123-{rank}-private'
    return dict(status='passed',rank=rank,pgid=2000+rank,driver_sha256=SHA,image=p.IMAGE,source=p.SOURCE,policy=dict(p.POLICY),started_ns=1,finished_ns=500,scratch=scratch,cleanup='retained',
        workers=[worker(f,rank,scratch) for f in p.FAMILIES],logs={f:dict(sha256=p.sha(b''),bytes=0,warnings=[]) for f in p.FAMILIES})


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def run_fixture(tmp_path):
    directory=tmp_path/'run';directory.mkdir();(directory/'validation.py').write_bytes(SCRIPT.read_bytes())
    from apertus_amd.config import resolve
    root=SCRIPT.parents[2]
    settings=resolve(root/'experiments/chonk-sssglu-qualification-i910bfcf936d3',site=root/'config.toml',submit=True,reservation='cscs',time='00:10:00')
    settings['validation']['sha256']=SHA;write(directory/'resolved.json',settings)
    commands=[]
    for rank in range(4):
        r=rank_record(rank);base=directory/f'output/0/sssglu-{rank}';write(base/'summary.json',r)
        for family,w in zip(p.FAMILIES,r['workers']):
            write(base/family/'worker.json',w);(base/family/'native.log').write_bytes(b'')
        command=dict(status='passed',rank=rank,argv=['python3','-u','/apertus/validation.py','run'])
        write(directory/f'output/0/rank-{rank}.json',command);commands.append(command)
    write(directory/'result.json',dict(status='passed',execution='passed',commands=[dict(argv=commands[0]['argv'],returncode=0,ranks=commands)]))
    return directory


def test_all_ranks_workers_cases():
    metrics=0
    for rank in range(4):
        r=rank_record(rank);assert p.validate_rank(r,rank=rank,expected_sha=SHA) is r
        metrics+=sum(len(c['metrics']) for w in r['workers'] for c in w['cases'])
    assert metrics==336 and p.IMPLEMENTATION_READY is True


@pytest.mark.parametrize('damage',['bool','suppression','dp','compile','source','pid','warning','launch','extra','affinity','layout','counter'])
def test_closed_refusals(damage):
    r=rank_record(0);w=r['workers'][1]
    if damage=='bool':w['local_device']=False
    elif damage=='suppression':w['compiler_suppression']=True
    elif damage=='dp':w['cases'][0]['metrics']['candidate/oracle/dp']['numel']*=1920
    elif damage=='compile':w['default_off']['compiler_baseline_after']['generated_kernel_count']=0
    elif damage=='source':w['adaptation']['exports']['candidate']['sha256']='c'*64
    elif damage=='pid':w['pid']=r['workers'][0]['pid']
    elif damage=='warning':r['logs']['weighted']['warnings']=['falling back to eager']
    elif damage=='launch':w['cases'][-1]['launches']['_fwd']=True
    elif damage=='extra':w['cases'][0]['extra']=1
    elif damage=='affinity':w['affinity']['final'][0]=True
    elif damage=='layout':w['cases'][0]['layouts'][0]['device']='cuda:1'
    else:w['counter_sources']['torch/_dynamo/utils.py']['bytes']=True
    with pytest.raises((ValueError,TypeError)):p.validate_rank(r,rank=0,expected_sha=SHA)


def test_offline_happy_exclusive_and_tamper(tmp_path):
    directory=run_fixture(tmp_path);before={f:f.read_bytes() for f in directory.rglob('*') if f.is_file()}
    assert p.check(directory)['metrics']==336
    assert before=={f:f.read_bytes() for f in before}
    with pytest.raises(ValueError):p.check(directory,directory/'derived.json')
    target=tmp_path/'derived.json';p.check(directory,target)
    with pytest.raises(FileExistsError):p.check(directory,target)
    log=directory/'output/0/sssglu-0/plain/native.log';log.write_text('changed')
    with pytest.raises(ValueError):p.check(directory)


def test_offline_worker_and_result_tampering(tmp_path):
    directory=run_fixture(tmp_path);path=directory/'output/0/sssglu-0/plain/worker.json'
    r=json.loads(path.read_text());r['pid']+=1;write(path,r)
    with pytest.raises(ValueError):p.check(directory)


@pytest.mark.parametrize('damage',['result_bool','global_pid','cross_rank_hash','source','warning','ordered_cases'])
def test_offline_global_bindings(tmp_path,damage):
    directory=run_fixture(tmp_path)
    if damage=='result_bool':
        path=directory/'result.json';r=json.loads(path.read_text());r['commands'][0]['returncode']=False;write(path,r)
    elif damage=='source':(directory/'validation.py').write_text('changed')
    elif damage=='warning':(directory/'output/0/sssglu-1/plain/native.log').write_text('falling back to eager')
    else:
        base=directory/'output/0/sssglu-1';r=json.loads((base/'summary.json').read_text());w=r['workers'][0]
        if damage=='global_pid':w['pid']=1000
        elif damage=='cross_rank_hash':
            for name in ('host_hashes','device_pre','device_post'):w['cases'][0][name]=['e'*64]*3
        else:w['cases'].reverse()
        write(base/'summary.json',r);write(base/'plain/worker.json',w)
    with pytest.raises(ValueError):p.check(directory)


def test_quiescence_bounded_same_pgid(tmp_path,monkeypatch):
    real_path=Path
    proc=tmp_path/'proc';proc.mkdir();child=proc/'987654';child.mkdir()
    (child/'stat').write_text('987654 (a child name) S 1 123 0')
    monkeypatch.setattr(p,'Path',lambda value:proc if value=='/proc' else real_path(value))
    monkeypatch.setattr(p.os,'getpgrp',lambda:123)
    assert not p.quiescent()
    (child/'stat').write_text('987654 (a child name) Z 1 123 0')
    assert p.quiescent()
