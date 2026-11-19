"""Filesystem-only quota agreement with real temporary files and CPU threads."""
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'experiments/validation/native_dispatcher_contributor_capture_probe.py'
spec=importlib.util.spec_from_file_location('quota_probe',SCRIPT)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def common():
    return dict(job_id='987654',validation_sha256='a'*64,source_manifest_sha256='b'*64,tooling_sha256='c'*64,
                frame=dict(image=p.IMAGE,source=p.SOURCE,tooling='c'*64,layer=4,invocation=0,microbatch=0,phase='replay',kind='forward'),
                group=list(range(8)),owners=[list(range(32*r,32*r+32)) for r in range(8)])


def prepared(na,nr):
    metadata={};lazy=[]
    for d in p.preflight_plan_arrays(na,nr):
        name=d['name']; arm=next((arm for arm in p.MODES if name.startswith(arm+'_')),None)
        suffix=name[len(arm)+1:] if arm else None
        if arm and suffix in (*p.BOUNDARIES[arm],'expert_probs'):
            shape=(4096,1,64) if suffix in ('source','postcombine') else tuple(d['shape'])
            lazy.append((name,arm,suffix,shape,'fp32' if suffix=='expert_probs' else 'bf16',d['dtype']))
        else: metadata[name]=SimpleNamespace(shape=tuple(d['shape']),dtype=d['dtype'])
    return metadata,lazy,lambda *a:pytest.fail('planning consumed owned roles')


def proposals(counts=None):
    counts=counts or [(32768,4096)]*8
    return [p.prepare_rank_proposal(r,common(),prepared(*counts[r])) for r in range(8)]


def test_uniform_and_hotspot_quotas():
    for counts in (None,[(196608,24576),(65536,8192)]+[(0,0)]*6):
        ps=proposals(counts);d=p.strict_allocate_quotas(ps,common())
        assert d['status']=='QUOTA_APPROVED' and sum(d['job_quotas'])<=p.JOB_CAP and max(d['job_quotas'])<=p.RANK_CAP
        for r,proposal in enumerate(ps):
            assert p.approved_prepared_quotas(r,proposal,prepared(*(counts[r] if counts else (32768,4096))),d)==d['stage_quotas']
        if counts: assert d['job_quotas'][0]>64*2**20


def test_all_rank_maxima_rejected_before_payload(tmp_path):
    ps=proposals([(262144,32768)]*8)
    assert all(pr['data_bytes']+2**20+2*65536+2**20<=p.RANK_CAP for pr in ps)
    with pytest.raises(ValueError,match='byte cap'):p.strict_allocate_quotas(ps,common())
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('damage',['rank-bool','duplicate','frame-bool','group-bool','owner-bool','source','validation','tooling','shape-bool','bytes','hash','mixed','tuple'])
def test_bad_proposals(damage):
    ps=proposals();target=ps[0]
    if damage=='rank-bool':target['rank']=False
    if damage=='duplicate':ps[1]['rank']=0
    if damage=='frame-bool':target['common']['frame']['invocation']=False
    if damage=='group-bool':target['common']['group'][0]=False
    if damage=='owner-bool':target['common']['owners'][0][0]=False
    if damage=='source':target['common']['frame']['source']='d'*40
    if damage=='validation':target['common']['validation_sha256']='not-a-hash'
    if damage=='tooling':target['common']['tooling_sha256']='d'*64
    if damage=='shape-bool':target['arrays'][0]['shape'][0]=True
    if damage=='bytes':target['data_bytes']+=1
    if damage=='hash':target['role_plan_sha256']='0'*64
    if damage=='mixed':target['common']['job_id']='different'
    if damage=='tuple':target['common']['group']=tuple(target['common']['group'])
    with pytest.raises(ValueError):p.strict_allocate_quotas(ps,common())


def test_eight_threads_approve_and_stale_rejected(tmp_path):
    ps=proposals()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(p.coordinate_preflight,tmp_path,r,ps[r],2) for r in range(8)]
        decisions=[f.result(timeout=4) for f in futures]
    assert all(d==decisions[0] for d in decisions)
    directory=next(tmp_path.iterdir());assert len(list(directory.iterdir()))==9
    assert not any('payload' in f.name or 'GLOBAL' in f.name for f in directory.iterdir())
    with pytest.raises(FileExistsError):p.coordinate_preflight(tmp_path,0,ps[0],.1)


def test_timeout_is_bounded_and_error_retained(tmp_path):
    with pytest.raises(TimeoutError):p.coordinate_preflight(tmp_path,3,proposals()[3],.1)
    directory=next(tmp_path.iterdir());assert (directory/'error-3.json').exists()


@pytest.mark.parametrize('damage',['decision','proposal','peer-error','overcap'])
def test_filesystem_refusals(tmp_path,damage):
    ps=proposals([(262144,32768)]*8 if damage=='overcap' else None)
    directory=tmp_path/('quota-'+hashlib.sha256(p.preflight_json(common())).hexdigest());directory.mkdir()
    for rank in range(8):
        if rank!=1:p.preflight_publish(directory/f'proposal-{rank}.json',ps[rank])
    if damage=='proposal':
        bad=deepcopy(ps[0]);bad['role_plan_sha256']='0'*64
        (directory/'proposal-0.json').write_bytes(p.preflight_json(bad))
    if damage=='decision':
        decision=p.strict_allocate_quotas(ps,common());decision['proposal_sha256'][0]='0'*64
        p.preflight_publish(directory/'decision.json',decision)
    if damage=='peer-error':p.preflight_publish(directory/'error-0.json',dict(status='PREFLIGHT_REJECTED'))
    with pytest.raises(ValueError):p.coordinate_preflight(tmp_path,1,ps[1],.2)
    assert not any(f.is_dir() for f in directory.iterdir())


def test_bounded_strict_json(tmp_path):
    path=tmp_path/'bad.json';path.write_bytes(b'{"rank":0,"rank":1}')
    with pytest.raises(ValueError):p.preflight_read(path)
    path.write_bytes(b' '*65537)
    with pytest.raises(ValueError):p.preflight_read(path)
    with pytest.raises(ValueError):p.preflight_publish(tmp_path/'huge.json',{'x':'a'*65536})


def test_prepared_mutation_refused_and_native_unchanged():
    ps=proposals();decision=p.strict_allocate_quotas(ps,common());pr=prepared(32768,4096)
    pr[0]['a_experts'].shape=(4096,7)
    with pytest.raises(ValueError):p.approved_prepared_quotas(0,ps[0],pr,decision)
    before=ROOT/'artifacts/investigations/performance-next/restart-20261116/uccl-contributor-global-preflight/typed-serializer-before.py.txt'
    def bodies(text):
        result={}
        for n in ast.parse(text).body:
            if not isinstance(n,(ast.FunctionDef,ast.ClassDef)) or n.name in {'write_rank_stage','preflight_publish'}: continue
            if n.name=='ContributorCapture': n.body=[m for m in n.body if not isinstance(m,ast.FunctionDef) or m.name not in {'serialize_quiescent','__exit__'}]
            result[n.name]=ast.dump(n,include_attributes=False)
        return result
    old,new=bodies(before.read_text()),bodies(SCRIPT.read_text())
    assert all(new[name]==body for name,body in old.items())
    assert p.IMPLEMENTATION_READY is False
