"""CPU-only approved prepared-role staging; one physical real-geometry rank writer."""
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'experiments/validation/native_dispatcher_contributor_capture_probe.py'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


p=load(SCRIPT,'staging_probe')
s=load(ROOT/'tests/test_native_contributor_serialization.py','staging_role_fixtures')
q=load(ROOT/'tests/test_native_contributor_quota.py','staging_quota_fixtures')


def capture_fixture(tmp_path):
    binding=s.binding();binding['group']=tuple(binding['group']);binding['local_experts']=tuple(binding['local_experts'])
    digest=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    capture=p.ContributorCapture(None,tmp_path,0,frozen=binding['frozen'])
    capture.bindings={arm:dict(metadata=deepcopy(binding),sha256=digest) for arm in p.MODES};capture.replays=dict.fromkeys(p.MODES,1)
    take,released,taken=s.roles(); keys=[]
    for arm in p.MODES:
        keys.extend((arm,n) for n in (*p.BOUNDARIES[arm],'router_mask','router_probs','expert_probs','expert_counts'))
    keys.extend(('alltoall',n) for n in ('source_te_map','input_splits','output_splits','forward_chunk_sizes','forward_chunk_order','restore_chunk_sizes','restore_chunk_order'))
    keys.extend(('flex',n) for n in ('token_indices','token_probs','dispatched_indices','dispatched_probs','flex_te_map'))
    assert len(keys)==31
    for arm,name in keys:
        capture.entries.append(dict(arm=arm,boundary=name,rank=0,layer=4,invocation=0,microbatch=0,phase='replay',kind='forward',image=p.IMAGE,source=p.SOURCE,binding_sha256=digest))
        capture.owned.append(object())
    def owned_take(arm,name):
        field=take(arm,name);release=field.release
        def finish():release();capture.owned[keys.index((arm,name))]=None
        field.release=finish;return field
    tooling=hashlib.sha256(json.dumps(binding['frozen']['tooling'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    common=dict(job_id='555777',validation_sha256=binding['frozen']['validation_sha256'],source_manifest_sha256=binding['frozen']['source_manifest_sha256'],tooling_sha256=tooling,
                frame=dict(image=p.IMAGE,source=p.SOURCE,tooling=tooling,layer=4,invocation=0,microbatch=0,phase='replay',kind='forward'),group=list(range(8)),owners=binding['owners'])
    return capture,owned_take,released,taken,common


def setup_capture(tmp_path,monkeypatch):
    capture,take,released,taken,common=capture_fixture(tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID','555777');monkeypatch.setattr(p,'owned_role_reader',lambda c:take)
    return capture,released,taken,common


def seed_peers(tmp_path,common,maximum=False):
    directory=tmp_path/('quota-'+hashlib.sha256(p.preflight_json(common)).hexdigest());directory.mkdir()
    for r in range(1,8):p.preflight_publish(directory/f'proposal-{r}.json',p.prepare_rank_proposal(r,common,q.prepared(262144,32768) if maximum else q.prepared(32768,4096)))


def test_once_prepared_eight_file_proposals_and_real_writer(tmp_path,monkeypatch):
    capture,released,taken,common=setup_capture(tmp_path,monkeypatch)
    original=p.typed_roles;prepared=[]
    def once(*args):
        assert not prepared;value=original(*args);prepared.append(value);return value
    monkeypatch.setattr(p,'typed_roles',once)
    with ThreadPoolExecutor(max_workers=8) as pool:
        peers=[pool.submit(p.coordinate_preflight,tmp_path,r,p.prepare_rank_proposal(r,common,q.prepared(32768,4096)),5) for r in range(1,8)]
        index=p.stage_capture_quiescent(capture,5)
        decisions=[f.result(timeout=6) for f in peers]
    assert index['status']=='INCOMPLETE_JOB' and len(prepared)==1 and len(taken)==len(released)==31 and all(v is None for v in capture.owned)
    assert index['stage_quotas']==index['job_quotas']==decisions[0]['stage_quotas']
    assert index['full_job_quotas']==decisions[0]['job_quotas'] and index['job_id']=='555777'
    out=tmp_path/'contributors-rank-0';arrays=s.identity.read_arrays(out,index['descriptors'],manifest_bytes=(out/'rank-index.json').stat().st_size)
    combined={};fields=[]
    for r in range(8):
        fields.append({})
        for role,name in index['fields'].items():
            old=arrays[name];value=s.identity.TypedArray(old.data,old.dtype,old.shape);value.rank=r
            alias=f'r{r}.{role}';combined[alias]=value;fields[-1][role]=alias
    joined=s.identity.parse_metadata(combined,fields,list(range(8)),common['owners'],list(range(256)))
    assert len(joined['pairs'])==262144
    for a in combined.values():a.close()
    for a in arrays.values():a.close()


@pytest.mark.parametrize('failure',['timeout','cap','mutation'])
def test_prewrite_rejection_retains_lazy_owned_fields(tmp_path,monkeypatch,failure):
    capture,released,taken,common=setup_capture(tmp_path,monkeypatch)
    if failure!='timeout':seed_peers(tmp_path,common,maximum=failure=='cap')
    if failure=='mutation':
        original=p.typed_roles;state=[]
        def prepare(*a):value=original(*a);state.append(value);return value
        monkeypatch.setattr(p,'typed_roles',prepare);coordinate=p.coordinate_preflight
        def mutate(*a):
            decision=coordinate(*a);state[0][0]['a_experts'].shape=(4096,7);return decision
        monkeypatch.setattr(p,'coordinate_preflight',mutate)
    with pytest.raises((ValueError,TimeoutError)):p.stage_capture_quiescent(capture,.05 if failure=='timeout' else 2)
    assert not (tmp_path/'contributors-rank-0').exists() and len(taken)==len(released)==18
    assert sum(v is not None for v in capture.owned)==13
    capture.__exit__(None,None,None);assert capture.owned==[]


def test_io_fault_releases_payload_without_false_index(tmp_path,monkeypatch):
    capture,released,taken,common=setup_capture(tmp_path,monkeypatch);seed_peers(tmp_path,common)
    original=p.stream_wire
    def fail(field,sink,dtype,**kwargs):
        if dtype=='u16':sink.write(b'partial');raise OSError('payload I/O primary')
        return original(field,sink,dtype,**kwargs)
    monkeypatch.setattr(p,'stream_wire',fail)
    with pytest.raises(OSError,match='payload I/O primary'):p.stage_capture_quiescent(capture,2)
    assert (tmp_path/'contributors-rank-0').exists() and not (tmp_path/'contributors-rank-0/rank-index.json').exists()
    assert len(taken)==len(released) and len(taken)<31


def test_snapshot_error_and_native_exception_remain_primary(tmp_path,monkeypatch):
    capture,*_=setup_capture(tmp_path,monkeypatch);capture.error(ValueError('snapshot error'))
    monkeypatch.setattr(p,'stage_capture_quiescent',lambda c:pytest.fail('stage after observer error'))
    primary=RuntimeError('native primary')
    with pytest.raises(RuntimeError) as caught:
        with capture:
            capture.serialize_quiescent();raise primary
    assert caught.value is primary and capture.summary['native_exception']=='RuntimeError: native primary'
    assert not capture.summary['rank_serialized'] and list(tmp_path.iterdir())==[]


def test_exit_preserves_successful_stage_diagnostics(tmp_path,monkeypatch):
    capture,*_=setup_capture(tmp_path,monkeypatch)
    marker=dict(status='INCOMPLETE_JOB',rank_serialized=True,descriptors=[{'sha256':'a'*64}],quota_approval={'status':'QUOTA_APPROVED'})
    monkeypatch.setattr(p,'stage_capture_quiescent',lambda c:marker)
    with capture:capture.serialize_quiescent()
    assert capture.summary['descriptors']==marker['descriptors'] and capture.summary['status']=='INCOMPLETE_JOB'
    assert not capture.summary['implementation_ready']


def test_publish_cleanup_cannot_mask_primary(tmp_path,monkeypatch):
    primary=OSError('link primary')
    def fail_link(*a):raise primary
    monkeypatch.setattr(p.os,'link',fail_link);original=Path.unlink
    def fail_unlink(path,*a,**k):
        if path.name.startswith('.quota-'):raise OSError('unlink secondary')
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'unlink',fail_unlink)
    with pytest.raises(OSError) as caught:p.preflight_publish(tmp_path/'decision.json',{'status':'QUOTA_APPROVED'})
    assert caught.value is primary and any('secondary' in note for note in primary.__notes__)


def test_exact_authorized_ast_changes_and_disabled_cli():
    old=ROOT/'artifacts/investigations/performance-next/restart-20261116/uccl-contributor-approved-staging/native_dispatcher_contributor_capture_probe.py.before.txt'
    def nodes(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=nodes(old.read_text()),nodes(SCRIPT.read_text());changed=set()
    for name,node in a.items():
        if name=='ContributorCapture':
            before={n.name:ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.FunctionDef)}
            after={n.name:ast.dump(n,include_attributes=False) for n in b[name].body if isinstance(n,ast.FunctionDef)}
            assert {k for k in before if before[k]!=after[k]}=={'serialize_quiescent','__exit__'}
        elif ast.dump(node,include_attributes=False)!=ast.dump(b[name],include_attributes=False):changed.add(name)
    assert changed=={'write_rank_stage','preflight_publish'}
    assert p.IMPLEMENTATION_READY is False
    with pytest.raises(ValueError):p.verify_contributor_artifacts(None,0,{})
