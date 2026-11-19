"""CPU-only typed evidence and real-cardinality virtual-array contributor tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'artifacts/investigations/performance-next/restart-20261116'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


p = load(ROOT/'experiments/validation/native_contributor_identity.py', 'real_identity')
old = load(PREP/'uccl-offline-identity-preparation/identity.py', 'identity')
sys.modules['identity'] = old
toy = load(PREP/'uccl-offline-identity-preparation/test_identity.py', 'old_identity_fixtures')
sys.modules.pop('identity', None)


def test_five_old_fixtures():
    import unittest
    result = unittest.TestResult(); unittest.defaultTestLoader.loadTestsFromTestCase(toy.Tests).run(result)
    assert result.wasSuccessful(), (result.errors, result.failures)
    for args in (toy.fixture(), toy.fixture([[2*d+1, 2*d] for d in range(8)])):
        expected = old.build(*args); actual = p.build(*args)
        assert {p.coordinates(k): v for k,v in actual['pairs'].items()} == expected['pairs']
        assert [[p.coordinates(k) for k in rows] for rows in actual['a2a']] == expected['a2a']


def test_dense_and_flex_slot_order_not_equal():
    args = toy.fixture()
    for tokens in args[3]:
        for slots in tokens: slots.reverse()
    assert p.build(*args)['pairs']
    args[6][0][0][0].reverse()
    with pytest.raises(ValueError): p.build(*args)


@pytest.mark.parametrize('kind', range(10))
def test_toy_corruption(kind):
    args = toy.fixture()
    if kind == 0: args[6][0][0][0][0] = 1
    if kind == 1: args[6][0][0][1][0] ^= 1
    if kind == 2: args[8][0][1] = args[8][0][0]
    if kind == 3: args[5][0][0][0] = -1
    if kind == 4: args[7][0][0][-1] += 1
    if kind == 5: args[10][0] = list(range(len(args[10][0])))
    if kind == 6: args[4][0][0][0] = (0, 0x3e800001)
    if kind == 7: args[6][0].pop()
    if kind == 8: args[3][0][0][1] = args[3][0][0][0]
    if kind == 9: args[1][0][0] = 255
    with pytest.raises(ValueError): p.build(*args)


def real_fixture():
    group = [91,17,62,8,35,104,6,77]
    owners = [list(range(d*32,(d+1)*32)) for d in range(8)]
    def routes(reverse):
        return [p.Row(4096, lambda t: p.Row(8, lambda j: (32*(t%8)+(7-j if reverse else j), 0x3e000000))) for _ in group]
    def source_row(t):
        def get(i):
            if i == 512: return 8
            if 0 <= i < 8: return (t%8)*4096+(7-i)*512+t//8
            if 256 <= i < 264: return 32*(t%8)+7-(i-256)
            raise ValueError('inactive tail')
        return p.Row(513,get)
    source = [p.Row(4096,source_row) for _ in group]
    received = [p.Row(4096,lambda r: (p.Row(8,lambda j:7-j),p.Row(8,lambda j:0x3e000000))) for _ in group]
    def flex_row(r):
        return p.Row(65,lambda i: 8 if i==64 else (7-i)*4096+r if i<8 else 7-(i-32) if 32<=i<40 else -99)
    fm = [p.Row(4096,flex_row) for _ in group]
    sizes = [512 if e<8 else 0 for s in range(8) for e in range(32)]
    order = [s*32+e for e in range(32) for s in range(8)]
    inverse = [order.index(i) for i in range(256)]
    return (group, owners, list(range(256)), routes(False), routes(True), source, received, fm,
            [[4096]*8 for _ in group], [[4096]*8 for _ in group],
            [(sizes,order) for _ in group], [([sizes[i] for i in order],inverse) for _ in group])


def test_real_32768_tokens_262144_assignments():
    args=real_fixture(); result=p.associate(*args)
    assert sum(result['tokens'])==32768 and len(result['pairs'])==262144
    assert result['a2a']==result['flex']
    assert p.coordinates(result['a2a'][7][-1])==(7,4095,231)
    args[9][0][0]+=1
    with pytest.raises(ValueError,match='transpose'): p.associate(*args)


def test_real_typed_metadata_parser():
    args=real_fixture(); arrays={}; fields=[]
    for r in range(8):
        fields.append({})
        def put(name, shape, values, dtype='i32'):
            values=list(values)
            raw=struct.pack('<'+str(len(values))+p.DTYPES[dtype][0],*values)
            name_id=f'r{r}.{name}'; value=p.TypedArray(raw,dtype,shape); value.rank=r
            arrays[name_id]=value; fields[r][name]=name_id
        for prefix, routes in (('a',args[3]),('f',args[4])):
            put(prefix+'_experts',(4096,8),(e for slots in routes[r] for e,b in slots),'i64')
            put(prefix+'_probs',(4096,8),(b for slots in routes[r] for e,b in slots),'u32')
        for prefix, maps, experts in (('source',args[5],256),('flex',args[7],32)):
            put(prefix+'_counts',(4096,),(8 for _ in range(4096)))
            put(prefix+'_rows',(32768,),(row[j] for row in maps[r] for j in range(8)))
            put(prefix+'_cols',(32768,),(row[experts+j] for row in maps[r] for j in range(8)))
        put('received_experts',(4096,8),(e for indices,bits in args[6][r] for e in indices),'i64')
        put('received_probs',(4096,8),(b for indices,bits in args[6][r] for b in bits),'u32')
        for name, values in (('send',args[8][r]),('recv',args[9][r]),('forward_sizes',args[10][r][0]),
                             ('forward_order',args[10][r][1]),('restore_sizes',args[11][r][0]),('restore_order',args[11][r][1])):
            put(name,(len(values),),values)
    result=p.parse_metadata(arrays,fields,*args[:3])
    assert len(result['pairs'])==262144
    arrays[fields[0]['a_experts']].rank=1
    with pytest.raises(ValueError,match='rank/dtype'): p.parse_metadata(arrays,fields,*args[:3])


def test_raw_bits_and_positive_zero():
    result=p.build(*toy.fixture()); labels=[p.key(0,0,0),p.key(0,0,1),p.key(0,1,3)]
    sources=[[[0x8000]*64,[0x4000]*64]]
    p.check_input_bits(labels,result['probabilities'],sources,[[0x8000]*64,[0]*64,[0]*64])
    with pytest.raises(ValueError): p.check_input_bits(labels,result['probabilities'],sources,[[0x8000]*64]*3)


def descriptor(name, rank, data, shape, dtype='i32'):
    return dict(name=name,file=name+'.bin',rank=rank,dtype=dtype,shape=shape,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def test_typed_reader_and_compact_513_columns(tmp_path):
    raw=struct.pack('<4i',8,0,256,123)
    d=descriptor('counts',0,raw,[4]); (tmp_path/d['file']).write_bytes(raw)
    arrays=p.read_arrays(tmp_path,[d]); assert list(arrays['counts'])==[8,0,256,123]
    arrays['counts'].close()
    dense=p.TypedArray(bytes(2*513*4),'i32',(2,513)); assert len(dense[0])==513
    compact=p.ActiveMap([1],[0],[255],256)
    assert compact[0][0]==0 and compact[0][256]==255 and compact[0][-1]==1
    with pytest.raises(ValueError): compact[0][1]
    with pytest.raises(ValueError): p.ActiveMap([2],[0],[255],256)
    dense.close()


@pytest.mark.parametrize('damage',['hash','bytes','rank','escape','name','dtype','cap','job-cap','shape'])
def test_reader_refusals(tmp_path,monkeypatch,damage):
    data=struct.pack('<i',1); d=descriptor('value',0,data,[1]); (tmp_path/d['file']).write_bytes(data)
    if damage=='hash': d['sha256']='0'*64
    if damage=='bytes': d['bytes']=8
    if damage=='rank': d['rank']=True
    if damage=='escape': d['file']='../value.bin'
    if damage=='name': d['name']='../x'
    if damage=='dtype': d['dtype']='f32'
    if damage=='cap': monkeypatch.setattr(p,'RANK_CAP',3)
    if damage=='job-cap': monkeypatch.setattr(p,'JOB_CAP',3)
    if damage=='shape': d['shape']=[True]
    with pytest.raises(ValueError): p.read_arrays(tmp_path,[d])


def closure_fixture(group=None):
    """Build only joined-shape evidence to isolate the88-entry binding contract."""
    owners=[list(range(32*r,32*r+32)) for r in range(8)]; group=list(range(8)) if group is None else group
    joined=dict(group=group,owners=owners,tokens=[4096]*8,source=[range(32768)]*8,received=[[]]*8,
                a2a=[[]]*8,flex=[[]]*8,probabilities=range(p.MAX_ASSIGNMENTS))
    frame=dict(image=p.IMAGE,source=p.SOURCE,tooling='a'*64,layer=4,invocation=0,microbatch=0,phase='replay',kind='forward')
    entries=[]; arrays={}; buffers={count:bytes(count*64*2) for count in (0,4096,32768)}
    for r in range(8):
        for arm,boundaries in p.BOUNDARIES.items():
            for b in boundaries:
                count=4096 if b in ('source','postcombine') else 32768 if b=='returned' else 0
                name=f'{r}-{arm}-{b}'; arrays[name]=p.TypedArray(buffers[count],'u16',(count,64)); arrays[name].rank=r
                entries.append(dict(rank=r,arm=arm,boundary=b,array=name,frame=frame,group=group,owners=owners,channels=list(p.CHANNELS)))
    return frame,group,owners,entries,joined,arrays


def test_fixed_88_closure():
    frame,group,owners,entries,joined,arrays=closure_fixture()
    assert len(entries)==88 and p.validate_closure(frame,group,owners,entries,joined,arrays)
    with pytest.raises(ValueError): p.validate_closure(frame,group,owners,entries[:-1],joined,arrays)
    with pytest.raises(ValueError): p.validate_closure(frame,group,owners,entries+[entries[0]],joined,arrays)
    damaged=dict(frame,tooling='bad')
    with pytest.raises(ValueError): p.validate_closure(damaged,group,owners,entries,joined,arrays)


@pytest.mark.parametrize('field,value',[
    ('invocation',False),('microbatch',False),('invocation',0.0),('microbatch',0.0),('layer',4.0),
])
def test_entry_frame_strict_ordinals(field,value):
    frame,group,owners,entries,joined,arrays=closure_fixture()
    entries=copy.deepcopy(entries); entries[0]['frame'][field]=value
    with pytest.raises(ValueError,match='boundary binding'): p.validate_closure(frame,group,owners,entries,joined,arrays)


@pytest.mark.parametrize('binding,path,value',[
    ('group',(0,),False),('group',(1,),True),('group',(0,),0.0),
    ('owners',(0,0),False),('owners',(0,1),True),('owners',(0,0),0.0),
    ('channels',(0,),False),('channels',(0,),0.0),
])
def test_entry_order_strict_types(binding,path,value):
    frame,group,owners,entries,joined,arrays=closure_fixture()
    entries=copy.deepcopy(entries); target=entries[0][binding]
    for index in path[:-1]: target=target[index]
    target[path[-1]]=value
    with pytest.raises(ValueError,match='boundary binding'): p.validate_closure(frame,group,owners,entries,joined,arrays)


@pytest.mark.parametrize('binding',['group','owners','channels'])
def test_entry_json_lists_not_tuples(binding):
    args=closure_fixture(); entries=copy.deepcopy(args[3]); entries[0][binding]=tuple(entries[0][binding])
    with pytest.raises(ValueError,match='boundary binding'): p.validate_closure(*args[:3],entries,*args[4:])


def test_closure_nonmonotonic_group_order():
    assert p.validate_closure(*closure_fixture([91,17,62,8,35,104,6,77]))


@pytest.mark.parametrize('binding,value', [('group',False),('group',0.0),('owners',False),('owners',0.0)])
def test_canonical_order_cannot_alias_joined_integer_bindings(binding,value):
    frame,group,owners,entries,joined,arrays=closure_fixture()
    # Even consistently relabelled entries cannot weaken the already joined
    # canonical integer group/ownership evidence through Python scalar equality.
    group,owners,entries=copy.deepcopy((group,owners,entries))
    if binding=='group':
        group[0]=value
        for entry in entries: entry['group'][0]=value
    else:
        owners[0][0]=value
        for entry in entries: entry['owners'][0][0]=value
    with pytest.raises(ValueError,match='native ownership/group'):
        p.validate_closure(frame,group,owners,entries,joined,arrays)


def test_offline_only():
    import ast
    tree=ast.parse((ROOT/'experiments/validation/native_contributor_identity.py').read_text())
    names={n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import)}
    assert not names & {'torch','transformer_engine','uccl','triton'}
    assert p.NATIVE_OBSERVER_READY is False
