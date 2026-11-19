"""Quiescent owned-role CPU fakes; no Torch/vendor imports or native execution."""
import ast
from collections.abc import Sequence
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'experiments/validation/native_dispatcher_contributor_capture_probe.py'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


p=load(SCRIPT,'typed_observer')
identity=load(ROOT/'experiments/validation/native_contributor_identity.py','typed_offline_identity')


class Words(Sequence):
    def __init__(self,n,get): self.n,self.get=n,get
    def __len__(self): return self.n
    def __getitem__(self,i):
        if not 0<=i<self.n: raise IndexError(i)
        return self.get(i)


def roles(damage=None):
    """Full4096-token/rank source and uniform32768-assignment rank, virtual tensors."""
    fields={}; released=[]; taken=[]
    def add(arm,name,shape,dtype,words): fields[arm,name]=(shape,dtype,words)
    for arm in p.MODES:
        add(arm,'router_mask',(4096,256),'bool',Words(4096*256,lambda i:int(32*((i//256)%8)<=i%256<32*((i//256)%8)+8)))
        add(arm,'router_probs',(4096,256),'fp32',Words(4096*256,lambda i:0x3e000000 if 32*((i//256)%8)<=i%256<32*((i//256)%8)+8 else 0))
        add(arm,'expert_counts',(32,),'i64',[4096]*8+[0]*24)
        add(arm,'expert_probs',(32768,),'fp32',Words(32768,lambda i:0x3e000000))
        for boundary in p.BOUNDARIES[arm]:
            n=4096 if boundary in ('source','postcombine','received','local') else 32768
            shape=(4096,1,64) if boundary in ('source','postcombine') else (n,64)
            add(arm,boundary,shape,'bf16',Words(n*64,lambda i:0x3f80))
    add('flex','token_indices',(4096,8),'i64',Words(32768,lambda i:32*((i//8)%8)+7-i%8))
    add('flex','token_probs',(4096,8),'fp32',Words(32768,lambda i:0x3e000000))
    add('flex','dispatched_indices',(4096,8),'i64',Words(32768,lambda i:7-i%8))
    add('flex','dispatched_probs',(4096,8),'fp32',Words(32768,lambda i:0x3e000000))
    def source_map(i):
        t,j=divmod(i,513)
        if j==512: return 8
        if j<8: return (t%8)*4096+(7-j)*512+t//8
        if 256<=j<264: return 32*(t%8)+7-(j-256)
        raise AssertionError('undefined source map tail accessed')
    def flex_map(i):
        r,j=divmod(i,65)
        if j==64: return 8
        if j<8: return (7-j)*4096+r
        if 32<=j<40: return 7-(j-32)
        raise AssertionError('undefined flex map tail accessed')
    add('alltoall','source_te_map',(4096,513),'i32',Words(4096*513,source_map))
    add('flex','flex_te_map',(4096,65),'i32',Words(4096*65,flex_map))
    sizes=[512 if e<8 else 0 for s in range(8) for e in range(32)]
    order=[s*32+e for e in range(32) for s in range(8)]
    for name,shape,values in (('forward_chunk_sizes',(8,32),sizes),('forward_chunk_order',(256,),order),
                              ('restore_chunk_sizes',(256,),[sizes[i] for i in order]),('restore_chunk_order',(256,),[order.index(i) for i in range(256)]),
                              ('input_splits',(8,),[4096]*8),('output_splits',(8,),[4096]*8)):
        add('alltoall',name,shape,'i64',values)
    if damage=='missing': fields.pop(('alltoall','router_mask'))
    if damage=='counts': fields['flex','expert_counts']=((32,),'i64',[0]*32)
    if damage=='map': fields['alltoall','source_te_map']=((4096,513),'i32',Words(4096*513,lambda i:0))
    def take(arm,name):
        assert (arm,name) not in taken
        taken.append((arm,name)); shape,dtype,words=fields[arm,name]
        return p.WireField(shape,dtype,words,lambda:released.append((arm,name)))
    return take,released,taken


def binding():
    return dict(group=list(range(8)),owners=[list(range(32*r,32*r+32)) for r in range(8)],
                source_columns=list(range(256)),local_experts=list(range(32)),
                frozen=dict(validation_sha256='a'*64,source_manifest_sha256='b'*64,tooling={'runtime':'c'*64},source_files={'selected.py':'d'*64}))


def test_real_roles_writer_and_offline_parser(tmp_path):
    take,released,taken=roles(); out=tmp_path/'rank0'
    index=p.write_rank_stage(out,0,binding(),take,job_quotas=[64*2**20]*8)
    assert index['status']=='INCOMPLETE_JOB' and not index['implementation_ready']
    assert len(index['fields'])==18 and len(index['boundaries'])==11 and set(released)==set(taken)
    arrays=identity.read_arrays(out,index['descriptors'],manifest_bytes=(out/'rank-index.json').stat().st_size)
    assert sum(d['bytes'] for d in index['descriptors'])==index['data_bytes']
    # Uniform synthetic ranks share metadata VALUES, but independent descriptors/rank labels.
    all_arrays={}; fields=[]
    for rank in range(8):
        fields.append({})
        for role,name in index['fields'].items():
            original=arrays[name]; value=identity.TypedArray(original.data,original.dtype,original.shape); value.rank=rank
            new=f'r{rank}.{role}'; all_arrays[new]=value; fields[-1][role]=new
    joined=identity.parse_metadata(all_arrays,fields,list(range(8)),binding()['owners'],list(range(256)))
    assert len(joined['pairs'])==262144
    for value in all_arrays.values(): value.close()
    for value in arrays.values(): value.close()
    with pytest.raises(FileExistsError): p.write_rank_stage(out,0,binding(),roles()[0],job_quotas=[64*2**20]*8)


@pytest.mark.parametrize('damage',['missing','counts','map'])
def test_bad_metadata_before_rank_files(tmp_path,damage):
    with pytest.raises((ValueError,KeyError)): p.write_rank_stage(tmp_path/'no-index',0,binding(),roles(damage)[0],job_quotas=[64*2**20]*8)
    assert not (tmp_path/'no-index').exists()


def test_caps_before_writes_and_stream_integrity(tmp_path):
    with pytest.raises(ValueError,match='global quota'): p.write_rank_stage(tmp_path/'bad',0,binding(),lambda *a:pytest.fail('take before cap'),job_quotas=[p.RANK_CAP]*8)
    with pytest.raises(ValueError,match='preflight'): p.write_rank_stage(tmp_path/'small',0,binding(),roles()[0],job_quotas=[1]*8)
    assert not (tmp_path/'small').exists()
    for dtype,finite,values in (('u32','fp32',[0x7f800000]),('u16','bf16',[0x7f80])):
        with pytest.raises(ValueError): p.stream_wire(p.WireField((1,),dtype,values),io.BytesIO(),dtype,lower=0,upper=2**32-1,finite=finite)
    sink=io.BytesIO(); result=p.stream_wire(p.WireField((3,),'u32',[0x80000000,0,0x3f800000]),sink,'u32',lower=0,upper=2**32-1,finite='fp32')
    assert sink.getvalue()==struct.pack('<3I',0x80000000,0,0x3f800000)
    assert result['sha256']==hashlib.sha256(sink.getvalue()).hexdigest()


def test_partial_io_never_publishes_index(tmp_path,monkeypatch):
    def fail(field,sink,*a,**k): sink.write(b'partial'); raise OSError('injected I/O failure')
    monkeypatch.setattr(p,'stream_wire',fail)
    with pytest.raises(OSError): p.write_rank_stage(tmp_path/'partial',0,binding(),roles()[0],job_quotas=[64*2**20]*8)
    assert (tmp_path/'partial').exists() and not (tmp_path/'partial/rank-index.json').exists()


def test_torch_bridge_fake_and_ownership():
    class Tensor:
        dtype='float32'; shape=(2,)
        def detach(self): return self
        def cpu(self): return self
        def contiguous(self): return self
        def view(self,dtype): return SimpleNamespace(numpy=lambda:bytearray(struct.pack('<2I',0x80000000,0x3f800000)))
    torch=SimpleNamespace(Tensor=Tensor,bool='bool',float32='float32',bfloat16='bfloat16',int64='int64',int32='int32',uint8='uint8')
    capture=SimpleNamespace(torch=torch,entries=[dict(arm='flex',boundary='token_probs')],owned=[Tensor()])
    take=p.owned_role_reader(capture); value=take('flex','token_probs')
    assert list(value.words)==[0x80000000,0x3f800000]
    with pytest.raises(ValueError): take('flex','token_probs')
    value.release(); assert capture.owned==[None]


def test_native_body_callbacks_unchanged_and_disabled():
    original=ROOT/'artifacts/investigations/performance-next/restart-20261116/uccl-contributor-typed-serialization/metadata-revision-original.py.txt'
    before=ast.parse(original.read_text()); after=ast.parse(SCRIPT.read_text())
    def body(tree,name):
        node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        if name=='ContributorCapture': node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name not in {'serialize_quiescent','__exit__'}]
        return ast.dump(node,include_attributes=False)
    for name in ('run_probe','gpu','observe','accepted','compare','validate_record','ContributorCapture'):
        assert body(before,name)==body(after,name)
    assert p.IMPLEMENTATION_READY is False and not p.wire_bound()['native_preflight_proved']
    assert p.wire_bound()['global_closed_bytes']<p.JOB_CAP<p.wire_bound()['uncoordinated_eight_rank_bytes']
    with pytest.raises(ValueError): p.verify_contributor_artifacts(None,0,{})
