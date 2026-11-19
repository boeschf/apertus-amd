"""Actual eight-stage synthetic disk fixture; no vendor/native/GPU imports."""
import ast
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


p=load(ROOT/'experiments/validation/native_contributor_identity.py','global_closure_identity')
s=load(ROOT/'tests/test_native_contributor_serialization.py','closure_serialization_fixtures')
q=load(ROOT/'tests/test_native_contributor_quota.py','closure_quota_fixtures')
OBSERVER=ROOT/'experiments/validation/native_dispatcher_contributor_capture_probe.py'


@pytest.fixture(scope='module')
def staged(tmp_path_factory):
    base=tmp_path_factory.mktemp('actual-eight-stages');root=base/'capture';root.mkdir()
    frozen=base/'frozen_observer.py';frozen.write_bytes(OBSERVER.read_bytes());sha=hashlib.sha256(frozen.read_bytes()).hexdigest()
    binding=s.binding();binding['frozen']['validation_sha256']=sha
    tooling=hashlib.sha256(json.dumps(binding['frozen']['tooling'],sort_keys=True).encode()).hexdigest()
    common=dict(job_id='112233',validation_sha256=sha,source_manifest_sha256=binding['frozen']['source_manifest_sha256'],tooling_sha256=tooling,
                frame=dict(image=p.IMAGE,source=p.SOURCE,tooling=tooling,layer=4,invocation=0,microbatch=0,phase='replay',kind='forward'),group=list(range(8)),owners=binding['owners'])
    observer=s.p
    take,_,_=s.roles();prepared=observer.typed_roles(take,0)
    proposals=[observer.prepare_rank_proposal(r,common,prepared if r==0 else q.prepared(32768,4096)) for r in range(8)]
    decision=observer.strict_allocate_quotas(proposals,common)
    template=base/'template'
    index=observer.write_rank_stage(template,0,binding,take,prepared_roles=prepared,job_quotas=decision['stage_quotas'],quota_approval=dict(proposal=proposals[0],decision=decision))
    quota=root/('quota-'+hashlib.sha256(observer.preflight_json(common)).hexdigest());quota.mkdir()
    for r,proposal in enumerate(proposals):observer.preflight_publish(quota/f'proposal-{r}.json',proposal)
    observer.preflight_publish(quota/'decision.json',decision)
    for r in range(8):
        directory=root/f'contributors-rank-{r}';directory.mkdir();row=json.loads(json.dumps(index))
        row['rank']=r;row['binding']['local_experts']=list(range(r*32,r*32+32))
        row['binding_sha256']=hashlib.sha256(json.dumps(row['binding'],sort_keys=True).encode()).hexdigest()
        row['quota_approval']['proposal']=proposals[r]
        for d in row['descriptors']:
            original=template/d['file'];role=d['name'][3:]
            d.update(rank=r,name=f'r{r}.{role}',file=f'r{r}.{role}.bin');os.link(original,directory/d['file'])
        row['fields']={role:f'r{r}.{role}' for role in row['fields']}
        for entry in row['boundaries']:entry.update(rank=r,array=entry['array'].replace('r0.',f'r{r}.',1))
        (directory/'rank-index.json').write_text(json.dumps(row,sort_keys=True))
    return root,frozen,sha,common


@pytest.fixture(autouse=True)
def forbid_vendor_imports(monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','triton','transformer_engine','uccl','deep_ep','megatron','aiter'}:pytest.fail('vendor import')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)


def check(staged,root=None,target=None):
    source,frozen,sha,common=staged
    return p.check_staged_capture(source if root is None else root,frozen_observer=frozen,expected_observer_sha256=sha,expected_common=common,target=target)


def clone(staged,tmp_path):
    source=staged[0];root=tmp_path/'capture';root.mkdir()
    for folder in source.iterdir():
        dest=root/folder.name;dest.mkdir()
        for path in folder.iterdir():
            if path.suffix=='.bin':os.link(path,dest/path.name)
            else:shutil.copyfile(path,dest/path.name)
    return root


def index_change(root,change):
    path=root/'contributors-rank-0/rank-index.json';value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value,sort_keys=True))


def test_actual_global_positive(staged,tmp_path):
    target=tmp_path/'diagnostic.json';summary=check(staged,target=target)
    assert summary['status']=='CAPTURE_CLOSED_DIAGNOSTIC' and summary['contributors']==262144 and summary['boundaries']==88
    assert summary['tokens']==32768 and not any(summary[k] for k in ('native_numerics_accepted','full1792_equivalence','performance_claim'))
    assert summary['logical_accounted_bytes']>8*25_000_000 and summary['logical_accounted_bytes']<=p.JOB_CAP
    assert json.loads(target.read_text())==summary
    assert sum(len(list(d.glob('*.bin'))) for d in staged[0].glob('contributors-rank-*'))==264


@pytest.mark.parametrize('damage',['rank','frame-bool','quota','missing','pending','alias','hash','peer-error'])
def test_early_tamper_refusals(staged,tmp_path,damage):
    root=clone(staged,tmp_path)
    if damage=='rank':index_change(root,lambda i:i.update(rank=False))
    if damage=='frame-bool':index_change(root,lambda i:i['frame'].update(invocation=False))
    if damage=='quota':index_change(root,lambda i:i['stage_quotas'].__setitem__(0,i['full_job_quotas'][0]))
    if damage=='missing':(root/'contributors-rank-0/r0.alltoall_source.bin').unlink()
    if damage=='pending':index_change(root,lambda i:i.update(status='CAPTURE_CLOSED_DIAGNOSTIC'))
    if damage=='alias':index_change(root,lambda i:i['fields'].update(a_probs=i['fields']['f_probs']))
    if damage=='hash':index_change(root,lambda i:i['descriptors'][0].update(sha256='0'*64))
    if damage=='peer-error':(next(root.glob('quota-*'))/'error-7.json').write_text('{}')
    with pytest.raises((ValueError,FileNotFoundError)):check(staged,root)
    assert not any(root.glob('*GLOBAL*'))


def test_joined_probability_tamper_even_with_rehashed_file(staged,tmp_path):
    root=clone(staged,tmp_path);path=root/'contributors-rank-0/r0.alltoall_expert_probs.bin'
    data=bytearray(path.read_bytes());struct.pack_into('<I',data,0,0x3f000000);path.unlink();path.write_bytes(data)
    def change(index):
        next(d for d in index['descriptors'] if d['file']==path.name)['sha256']=hashlib.sha256(data).hexdigest()
    index_change(root,change)
    with pytest.raises(ValueError,match='expert probability bits'):check(staged,root)


@pytest.mark.parametrize('failure',['hash','nonfinite'])
def test_all_mappings_closed_on_failure(staged,tmp_path,monkeypatch,failure):
    root=clone(staged,tmp_path);closed=[];original=p.TypedArray.close
    def close(value):closed.append(value);return original(value)
    monkeypatch.setattr(p.TypedArray,'close',close)
    if failure=='hash':index_change(root,lambda i:i['descriptors'][2].update(sha256='0'*64))
    else:
        path=root/'contributors-rank-0/r0.a_probs.bin';data=bytearray(path.read_bytes());struct.pack_into('<I',data,0,0x7f800000)
        path.unlink();path.write_bytes(data)
        index_change(root,lambda i:next(d for d in i['descriptors'] if d['file']==path.name).update(sha256=hashlib.sha256(data).hexdigest()))
    with pytest.raises(ValueError,match='hash' if failure=='hash' else 'nonfinite'):check(staged,root)
    assert closed and len({id(v) for v in closed})==len(closed)
    if failure=='nonfinite':assert len(closed)==264
    for value in closed:
        with pytest.raises(ValueError):value.data.nbytes


def test_new_target_failure_and_io_priority(staged,tmp_path,monkeypatch):
    target=tmp_path/'prior.json';target.write_text('preserve')
    with pytest.raises(ValueError):check(staged,target=target)
    assert target.read_text()=='preserve'
    with pytest.raises(ValueError):check(staged,target=staged[0]/'forbidden.json')
    root=clone(staged,tmp_path);(root/'contributors-rank-0/r0.alltoall_source.bin').unlink()
    failure=tmp_path/'failure.json'
    with pytest.raises(ValueError):check(staged,root,target=failure)
    assert json.loads(failure.read_text())['status']=='CAPTURE_REJECTED'
    monkeypatch.setattr(p,'publish_capture_diagnostic',lambda *a:(_ for _ in ()).throw(OSError('injected evidence I/O')))
    with pytest.raises(ValueError) as caught:check(staged,root,target=tmp_path/'io.json')
    assert any('secondary failure evidence' in note for note in caught.value.__notes__)


def test_frozen_hash_and_existing_functions_unchanged(staged):
    with pytest.raises(ValueError,match='observer hash'):p.check_staged_capture(staged[0],frozen_observer=staged[1],expected_observer_sha256='0'*64,expected_common=staged[3])
    before=ROOT/'artifacts/investigations/performance-next/restart-20261116/uccl-contributor-global-closure/native_contributor_identity.py.before.txt'
    def bodies(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old,new=bodies(before.read_text()),bodies((ROOT/'experiments/validation/native_contributor_identity.py').read_text())
    assert all(new[name]==node for name,node in old.items()) and p.NATIVE_OBSERVER_READY is False
