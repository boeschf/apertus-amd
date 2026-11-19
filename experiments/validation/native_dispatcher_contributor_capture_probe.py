"""PARTIAL contributor observer around the unchanged EP8 gate; native CLI disabled.

Two full-MoE-checkpointed microbatches use native routing and TE grouped experts.
Unfused BF16 leaf gradients are summed explicitly in FP32, not DDP/main_grad.
Only GPU children import vendor modules; CPU planning and checking are read-only.
"""

from contextlib import contextmanager
from copy import deepcopy
import faulthandler
import hashlib
import inspect
from importlib import metadata
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys

from apertus_amd.artifacts import sha256, write_json
from apertus_amd.config import option
from apertus_amd.entrypoint import verify_imports

CHANNELS = tuple(range(0, 1792, 28))
RANK_CAP, JOB_CAP = 192 * 2**20, 512 * 2**20
REAL_TOKENS, REAL_ASSIGNMENTS = 32768, 262144
IMPLEMENTATION_READY = False
CURRENT_CAPTURE = None
BOUNDARIES = {'alltoall': ('source', 'expert_input', 'expert_output', 'returned', 'postcombine'),
              'flex': ('source', 'received', 'expert_input', 'expert_output', 'local', 'postcombine')}


def chunk_rows(sizes, order):
    """Expand actual source-defined chunk operands, never an opaque native handle."""
    if any(type(n) is not int or n < 0 for n in sizes) or sum(sizes) > REAL_ASSIGNMENTS:
        raise ValueError('chunk size bound')
    if any(type(i) is not int for i in order) or sorted(order) != list(range(len(sizes))):
        raise ValueError('chunk order')
    chunks, start = [], 0
    for size in sizes:
        chunks.append(range(start, start + size)); start += size
    return [row for i in order for row in chunks[i]]


def stream_i32(words, sink, count, *, lower, upper, byte_cap):
    """Validate immutable quiescent words; recheck replay and hash before final flush.

    A changed replay may leave partial output, never a successful typed record.
    """
    import struct
    if type(count) is not int or count < 0 or count > REAL_TOKENS * 513 or count * 4 > byte_cap:
        raise ValueError('typed metadata byte/count bound')
    seen = 0; validated = hashlib.sha256()
    for word in words():
        if type(word) is not int or not lower <= word <= upper or not -2**31 <= word < 2**31:
            raise ValueError('typed metadata word')
        seen += 1
        if seen > count: raise ValueError('metadata count overflow')
        validated.update(struct.pack('<i', word))
    if seen != count: raise ValueError('metadata count incomplete')
    digest = hashlib.sha256(); block = bytearray(); seen = 0
    for word in words():
        if type(word) is not int or not lower <= word <= upper or not -2**31 <= word < 2**31 or seen >= count:
            raise ValueError('changed metadata replay')
        seen += 1
        block.extend(struct.pack('<i', word))
        if len(block) == 65536:
            sink.write(block); digest.update(block); block.clear()
    digest.update(block)
    if seen != count or digest.digest() != validated.digest(): raise ValueError('changed metadata replay')
    if block: sink.write(block)
    return dict(bytes=count * 4, sha256=digest.hexdigest(), dtype='int32', byte_order='little')


def active_map_words(rows, experts, output_rows):
    """Emit only count/active TE prefixes, zeroing undefined tails after quiescence."""
    if experts not in (32, 256) or not 0 <= output_rows <= REAL_ASSIGNMENTS:
        raise ValueError('TE map dimensions')
    for row in rows:
        if len(row) != 2 * experts + 1: raise ValueError('TE map width')
        count = row[-1]
        if type(count) is not int or not 0 <= count <= 8: raise ValueError('TE active count')
        for index in range(experts):
            value = row[index] if index < count else 0
            if type(value) is not int or not 0 <= value < max(1, output_rows): raise ValueError('TE output row')
            yield value
        for index in range(experts):
            value = row[experts + index] if index < count else 0
            if type(value) is not int or not 0 <= value < experts: raise ValueError('TE expert column')
            yield value
        yield count


class WireField:
    """Quiescent flat raw-word sequence with explicit logical shape and ownership."""
    def __init__(self, shape, dtype, words, release=lambda: None):
        self.shape, self.dtype, self.words, self.release = tuple(shape), dtype, words, release
        if (not self.shape or any(type(n) is not int or n < 0 for n in self.shape)
                or dtype not in ('bool','fp32','bf16','i64','i32','u32','u16')
                or math.prod(self.shape) != len(words)): raise ValueError('wire field shape/type/size')


def wire_bound():
    """Return byte bounds; global closure bounds are conditional, NOT local preflight."""
    # Across a valid global join: sum expert rows/arm=A; sum received rows<=A.
    a, t = REAL_ASSIGNMENTS, REAL_TOKENS
    payload = (4*t + 7*a)*64*2
    metadata = 4*a*4 + (t+2*a)*4 + (8*a)*8 + (a+2*a)*4 + 2*a*4 + 8*(16+4*256+64)*4
    # Each rank can individually receive every assignment; no cross-rank proof yet.
    local_payload = (4*4096 + 4*a + 32768 + 2*t)*64*2
    local_metadata = 4*32768*4 + (4096+2*32768)*4 + t*8*8 + (t+2*a)*4 + 2*a*4 + (16+4*256+64)*4
    return dict(global_closed_bytes=payload+metadata+8*2**20,
                local_worst_bytes=local_payload+local_metadata+2**20,
                uncoordinated_eight_rank_bytes=8*(local_payload+local_metadata+2**20),
                index_reserve_per_rank=2**20, native_preflight_proved=False)


def stream_wire(field, sink, dtype, *, lower, upper, finite=None):
    """Validate/write explicit LE raw words in <=1MiB chunks; errors leave partial files."""
    import struct
    formats = {'i32': 'i', 'u32': 'I', 'u16': 'H'}
    if dtype not in formats or math.prod(field.shape) > REAL_ASSIGNMENTS*64:
        raise ValueError('wire type/count bound')
    word = struct.Struct('<'+formats[dtype]); block = bytearray(); digest = hashlib.sha256(); count = 0; expected=math.prod(field.shape)
    for value in field.words:
        if type(value) is not int or not lower <= value <= upper: raise ValueError('wire word bounds')
        if finite == 'fp32' and value & 0x7f800000 == 0x7f800000: raise ValueError('nonfinite probability')
        if finite == 'bf16' and value & 0x7f80 == 0x7f80: raise ValueError('nonfinite BF16')
        count += 1
        if count > expected: raise ValueError('wire count overflow')
        block.extend(word.pack(value))
        if len(block) >= 2**20:
            sink.write(block); digest.update(block); block.clear()
    if count != expected: raise ValueError('wire count incomplete')
    if block: sink.write(block); digest.update(block)
    return dict(bytes=count*word.size, sha256=digest.hexdigest())


def typed_roles(take, rank):
    """Convert actual owned metadata one field at a time; payloads remain lazy.

    take(arm,name) returns a WireField over immutable quiescent raw words.
    No tensor API, host conversion or payload lookup is performed by callbacks.
    """
    from array import array
    roles, used = {}, set()
    def consume(arm, name, shape, dtype, convert):
        identity = arm, name
        if identity in used: raise ValueError('duplicate consumed role')
        used.add(identity); field = take(arm, name)
        try:
            if field.shape != tuple(shape) or field.dtype != dtype: raise ValueError(f'role shape/dtype: {identity}')
            return convert(field.words)
        finally: field.release()
    def ints(words, lower, upper):
        result = array('i')
        for value in words:
            if type(value) is not int or not lower <= value <= upper: raise ValueError('metadata integer bounds')
            result.append(value)
        return result
    def probs(words):
        result = array('I')
        for value in words:
            if type(value) is not int or not 0 <= value < 2**32 or value & 0x7f800000 == 0x7f800000:
                raise ValueError('nonfinite/raw probability word')
            result.append(value)
        return result
    def add(name, shape, dtype, words): roles[name] = WireField(shape, dtype, words)
    canonical = None
    for arm in MODES:
        def selected(words):
            values = array('i')
            for t in range(4096):
                start = len(values)
                for e in range(256):
                    value = words[t*256+e]
                    if type(value) is not int or value not in (0,1): raise ValueError('router mask word')
                    if value: values.append(e)
                if len(values)-start != 8: raise ValueError('router requires exactly8 selected experts')
            return values
        experts = consume(arm, 'router_mask', (4096,256), 'bool', selected)
        def selected_probs(words):
            # Check the complete dense router tensor, but retain selected raw words only.
            for value in words:
                if type(value) is not int or not 0 <= value < 2**32 or value & 0x7f800000 == 0x7f800000:
                    raise ValueError('nonfinite dense router probability')
            return array('I', (words[t*256+experts[t*8+j]] for t in range(4096) for j in range(8)))
        bits = consume(arm, 'router_probs', (4096,256), 'fp32', selected_probs)
        if canonical is None:
            canonical = experts, bits
            add('a_experts',(4096,8),'i32',experts); add('a_probs',(4096,8),'u32',bits)
        elif canonical != (experts,bits): raise ValueError('cross-arm dense routing bits')
    ae, ap = canonical
    fe = consume('flex','token_indices',(4096,8),'i64',lambda w:ints(w,0,255))
    fp = consume('flex','token_probs',(4096,8),'fp32',probs)
    for t in range(4096):
        selected = dict(zip(ae[t*8:t*8+8],ap[t*8:t*8+8]))
        if len(set(fe[t*8:t*8+8])) != 8 or any(selected.get(fe[t*8+j]) != fp[t*8+j] for j in range(8)):
            raise ValueError('FLEX actual topk slot/bits versus dense membership')
    add('f_experts',(4096,8),'i32',fe); add('f_probs',(4096,8),'u32',fp)
    def compact(arm, name, inputs, experts, selected):
        totals = [0]*experts
        for r in range(inputs):
            for e in selected(r): totals[e] += 1
        starts, n = [], 0
        for count in totals: starts.append(n); n += count
        if n > REAL_ASSIGNMENTS: raise ValueError('map assignment bound')
        def convert(words):
            counts, rows, cols = array('i'), array('i'), array('i'); seen = [0]*experts
            for r in range(inputs):
                active = sorted(selected(r),reverse=True); offset = r*(2*experts+1)
                count = words[offset+2*experts]
                if type(count) is not int or count != len(active): raise ValueError('map active count')
                counts.append(count)
                for j,e in enumerate(active):
                    position = starts[e]+seen[e]; seen[e] += 1
                    if type(words[offset+j]) is not int or type(words[offset+experts+j]) is not int or words[offset+j] != position or words[offset+experts+j] != e:
                        raise ValueError('map active prefix/formula')
                    rows.append(position); cols.append(e)
            return counts, rows, cols
        result = consume(arm,name,(inputs,2*experts+1),'i32',convert)
        prefix = 'source' if arm == 'alltoall' else 'flex'
        for suffix, values in zip(('counts','rows','cols'),result): add(prefix+'_'+suffix,(len(values),),'i32',values)
        return totals
    source_counts = compact('alltoall','source_te_map',4096,256,lambda t:ae[t*8:t*8+8])
    field = take('flex','dispatched_indices')
    try:
        nr = field.shape[0]
        if not 0 <= nr <= REAL_TOKENS or field.shape != (nr,8) or field.dtype != 'i64': raise ValueError('received shape/dtype')
        ri = ints(field.words,-1,31)
    finally: field.release()
    rp = consume('flex','dispatched_probs',(nr,8),'fp32',probs)
    for r in range(nr):
        active = [e for e in ri[r*8:r*8+8] if e >= 0]
        if not active or len(set(active)) != len(active): raise ValueError('received expert multiplicity')
        if any(ri[r*8+j] == -1 and rp[r*8+j] != 0 for j in range(8)): raise ValueError('nonlocal probability not positive zero')
    add('received_experts',(nr,8),'i32',ri); add('received_probs',(nr,8),'u32',rp)
    flex_counts = compact('flex','flex_te_map',nr,32,lambda r:[e for e in ri[r*8:r*8+8] if e >= 0])
    sizes = consume('alltoall','forward_chunk_sizes',(8,32),'i64',lambda w:ints(w,0,REAL_ASSIGNMENTS))
    forward = consume('alltoall','forward_chunk_order',(256,),'i64',lambda w:ints(w,0,255))
    restore_sizes = consume('alltoall','restore_chunk_sizes',(256,),'i64',lambda w:ints(w,0,REAL_ASSIGNMENTS))
    restore = consume('alltoall','restore_chunk_order',(256,),'i64',lambda w:ints(w,0,255))
    expected = [s*32+e for e in range(32) for s in range(8)]
    if list(forward) != expected or list(restore) != [expected.index(i) for i in range(256)] or list(restore_sizes) != [sizes[i] for i in expected]:
        raise ValueError('source-defined chunk transpose/inverse')
    counts = [sum(sizes[s*32+e] for s in range(8)) for e in range(32)]
    na = sum(counts); nf = sum(flex_counts)
    if na > REAL_ASSIGNMENTS or counts != flex_counts: raise ValueError('cross-arm expert counts')
    send = consume('alltoall','input_splits',(8,),'i64',lambda w:ints(w,0,REAL_ASSIGNMENTS))
    recv = consume('alltoall','output_splits',(8,),'i64',lambda w:ints(w,0,REAL_ASSIGNMENTS))
    if list(send) != [sum(source_counts[d*32:d*32+32]) for d in range(8)] or list(recv) != [sum(sizes[s*32:s*32+32]) for s in range(8)]:
        raise ValueError('local splits/counts')
    for name, values in (('send',send),('recv',recv),('forward_sizes',sizes),('forward_order',forward),('restore_sizes',restore_sizes),('restore_order',restore)):
        add(name,(len(values),),'i32',values)
    for arm, expected_counts in (('alltoall',counts),('flex',flex_counts)):
        value = consume(arm,'expert_counts',(32,),'i64',lambda w:ints(w,0,REAL_ASSIGNMENTS))
        if list(value) != expected_counts: raise ValueError('expert count values')
        add(arm+'_expert_counts',(32,),'i32',value)
    # Scalar expert probabilities are finite now; contributor association requires the global join.
    lazy = []
    for arm, rows in (('alltoall',na),('flex',nf)):
        lazy.append((arm+'_expert_probs',arm,'expert_probs',(rows,),'fp32','u32'))
        for boundary in BOUNDARIES[arm]:
            n = 4096 if boundary in ('source','postcombine') else nr if boundary in ('received','local') else 32768 if boundary == 'returned' else rows
            shape = (4096,1,64) if boundary in ('source','postcombine') else (n,64)
            lazy.append((arm+'_'+boundary,arm,boundary,shape,'bf16','u16'))
    return roles, lazy, take


def write_rank_stage(directory, rank, binding, take, *, job_quotas, prepared_roles=None, quota_approval=None):
    """Write a local incomplete stage, optionally reusing an approved prepared tuple.

    Quotas are eight integer byte allowances fixed by an offline caller before ANY
    rank writes; local shape checks alone cannot establish their global legitimacy.
    """
    if type(rank) is not int or not 0 <= rank < 8 or len(job_quotas) != 8 or any(type(n) is not int or not 0 < n <= RANK_CAP for n in job_quotas) or sum(job_quotas) > JOB_CAP:
        raise ValueError('pre-write global quota bound')
    if prepared_roles is None:
        if quota_approval is not None: raise ValueError('approval requires prepared roles')
        prepared_roles=typed_roles(take,rank)
    if type(prepared_roles) is not tuple or len(prepared_roles)!=3 or prepared_roles[2] is not take:
        raise ValueError('same prepared tuple/take required')
    metadata, lazy, take = prepared_roles
    if quota_approval is not None:
        if set(quota_approval)!={'proposal','decision'}: raise ValueError('quota approval fields')
        approved=approved_prepared_quotas(rank,quota_approval['proposal'],prepared_roles,quota_approval['decision'])
        if preflight_json(job_quotas)!=preflight_json(approved): raise ValueError('payload stage allowances required')
    if set(binding) != {'group','owners','source_columns','local_experts','frozen'}:
        raise ValueError('binding fields')
    frozen = binding['frozen']
    if set(frozen) != {'validation_sha256','tooling','source_manifest_sha256','source_files'} or not frozen['tooling'] or not frozen['source_files']:
        raise ValueError('frozen binding fields')
    for name in ('validation_sha256','source_manifest_sha256'):
        value=frozen[name]
        if not isinstance(value,str) or len(value)!=64 or set(value)-set('0123456789abcdef'): raise ValueError('frozen binding digest')
    tooling_sha=hashlib.sha256(json.dumps(frozen['tooling'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    frame = dict(image=IMAGE,source=SOURCE,tooling=tooling_sha,layer=4,invocation=0,microbatch=0,phase='replay',kind='forward')
    if binding['group'] != list(range(8)) and tuple(binding['group']) != tuple(range(8)): raise ValueError('group order')
    if binding['owners'] != [list(range(r*32,(r+1)*32)) for r in range(8)]: raise ValueError('owner order')
    if binding['source_columns'] != list(range(256)) or tuple(binding['local_experts']) != tuple(range(rank*32,(rank+1)*32)):
        raise ValueError('binding column/local expert order')
    if any(not isinstance(frame[k],str) or len(frame[k]) != 64 or set(frame[k])-set('0123456789abcdef') for k in ('tooling',)): raise ValueError('tooling hash')
    if quota_approval is not None:
        common=quota_approval['proposal']['common']
        if (preflight_json(frame)!=preflight_json(common['frame']) or preflight_json(list(binding['group']))!=preflight_json(common['group'])
                or preflight_json(binding['owners'])!=preflight_json(common['owners']) or frozen['validation_sha256']!=common['validation_sha256']
                or frozen['source_manifest_sha256']!=common['source_manifest_sha256']): raise ValueError('approved writer binding')
    entries, boundaries, fields = [], [], {}; total = 0
    pending = [(name,field.shape,field.dtype,lambda field=field:field,None) for name,field in metadata.items()]
    for name,arm,boundary,shape,source_dtype,dtype in lazy:
        pending.append((name,shape,dtype,lambda arm=arm,boundary=boundary:take(arm,boundary),source_dtype))
    widths = {'i32':4,'u32':4,'u16':2}
    data_bytes = sum(math.prod(shape)*widths[dtype] for _,shape,dtype,_,_ in pending)
    if data_bytes+2**20 > min(job_quotas[rank],RANK_CAP): raise ValueError('rank quota/data preflight')
    directory = Path(directory); directory.mkdir()  # Exclusive directory; partial failures are retained.
    for name,shape,dtype,factory,source_dtype in pending:
        field = factory()
        try:
            if field.shape != tuple(shape) or field.dtype != (source_dtype or dtype): raise ValueError('staged role shape/dtype')
            physical_shape = (4096,64) if shape == (4096,1,64) else tuple(shape)
            filename = f'r{rank}.{name}.bin'
            with (directory/filename).open('xb') as sink:
                result = stream_wire(WireField(physical_shape,field.dtype,field.words),sink,dtype,
                                     lower=-1 if name=='received_experts' else 0,upper=65535 if dtype=='u16' else 2**32-1 if dtype=='u32' else REAL_ASSIGNMENTS,
                                     finite='bf16' if dtype=='u16' else 'fp32' if dtype=='u32' else None)
            total += result['bytes']; array_name = f'r{rank}.{name}'
            entries.append(dict(name=array_name,file=filename,rank=rank,dtype=dtype,shape=list(physical_shape),**result))
            if name in metadata and not name.endswith(('expert_counts',)):
                fields[name] = array_name
            for arm in MODES:
                if name.startswith(arm+'_') and name[len(arm)+1:] in BOUNDARIES[arm]:
                    boundaries.append(dict(rank=rank,arm=arm,boundary=name[len(arm)+1:],array=array_name,frame=frame,
                                           group=list(binding['group']),owners=binding['owners'],channels=list(CHANNELS)))
        finally:
            primary=sys.exception()
            try: field.release()
            except Exception as error:
                if primary is None: raise
                primary.add_note(f'secondary field release: {error}')
    index = dict(status='INCOMPLETE_JOB',rank_serialized=True,implementation_ready=False,rank=rank,
                 frame=frame,binding=binding,binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest(),descriptors=entries,fields=fields,boundaries=boundaries,
                 job_quotas=list(job_quotas),stage_quotas=list(job_quotas),data_bytes=total,
                 pending='all8-rank file/cap validation,88-boundary closure,identity join and expert probability association')
    if quota_approval is not None:
        index['quota_approval']=deepcopy(quota_approval)
        index['full_job_quotas']=list(quota_approval['decision']['job_quotas'])
        index['job_id']=quota_approval['proposal']['common']['job_id']
    encoded = json.dumps(index,sort_keys=True,allow_nan=False).encode()
    # Charge the offline reader's descriptor JSON as well as the actual index bytes.
    charge = len(encoded)+len(json.dumps(entries,sort_keys=True).encode())
    if charge > 2**20 or total+charge > job_quotas[rank]: raise ValueError('rank index byte cap')
    with (directory/'rank-index.json').open('xb') as stream: stream.write(encoded)
    return index


def preflight_json(value):
    """Encode canonical strict JSON for immutable file identities (no numeric coercion)."""
    def validate(item):
        if type(item) is dict:
            if any(type(k) is not str for k in item): raise ValueError('preflight JSON keys')
            for v in item.values(): validate(v)
        elif type(item) is list:
            for v in item: validate(v)
        elif type(item) not in (str,int,float,bool,type(None)):
            raise ValueError('preflight JSON type')
    validate(value)
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def preflight_plan_arrays(assignments, received):
    """Return the exact33-role shape/byte schema; no payload reads or identity claim."""
    if any(type(n) is not int for n in (assignments,received)) or not 0 <= received <= REAL_TOKENS or not received <= assignments <= min(8*received,REAL_ASSIGNMENTS):
        raise ValueError('preflight received/assignment geometry')
    shapes = {name:('i32',[4096,8]) for name in ('a_experts','f_experts')}
    shapes.update({name:('u32',[4096,8]) for name in ('a_probs','f_probs')})
    shapes.update(source_counts=('i32',[4096]),source_rows=('i32',[32768]),source_cols=('i32',[32768]),
                  received_experts=('i32',[received,8]),received_probs=('u32',[received,8]),
                  flex_counts=('i32',[received]),flex_rows=('i32',[assignments]),flex_cols=('i32',[assignments]))
    shapes.update({name:('i32',[8]) for name in ('send','recv')})
    shapes.update({name:('i32',[256]) for name in ('forward_sizes','forward_order','restore_sizes','restore_order')})
    for arm in MODES:
        shapes[arm+'_expert_counts']=('i32',[32]); shapes[arm+'_expert_probs']=('u32',[assignments])
        for boundary in BOUNDARIES[arm]:
            rows = 4096 if boundary in ('source','postcombine') else received if boundary in ('received','local') else 32768 if boundary=='returned' else assignments
            shapes[arm+'_'+boundary]=('u16',[rows,64])
    return [dict(name=name,dtype=dtype,shape=shape,bytes=math.prod(shape)*(2 if dtype=='u16' else 4)) for name,(dtype,shape) in sorted(shapes.items())]


def prepare_rank_proposal(rank, common, prepared_roles):
    """Plan already-owned typed_roles output once; never call take or consume snapshots.

    Keep this SAME prepared object alive for future separately reviewed writer wiring.
    The current native writer does not yet accept/reuse prepared_roles.
    """
    metadata,lazy,_ = prepared_roles
    if len(metadata)!=20 or len(lazy)!=13: raise ValueError('prepared role partition')
    for name,arm,boundary,shape,source_dtype,dtype in lazy:
        if arm not in MODES or boundary not in (*BOUNDARIES[arm],'expert_probs') or name!=arm+'_'+boundary or (source_dtype,dtype)!=(('fp32','u32') if boundary=='expert_probs' else ('bf16','u16')):
            raise ValueError('prepared lazy role binding')
    arrays=[dict(name=name,dtype=f.dtype,shape=list(f.shape),bytes=math.prod(f.shape)*(2 if f.dtype=='u16' else 4)) for name,f in metadata.items()]
    arrays.extend(dict(name=name,dtype=dtype,shape=[4096,64] if tuple(shape)==(4096,1,64) else list(shape),
                       bytes=math.prod(shape)*(2 if dtype=='u16' else 4)) for name,arm,boundary,shape,source_dtype,dtype in lazy)
    arrays.sort(key=lambda d:d['name'])
    proposal=dict(status='PREPARED_ROLES',rank=rank,common=deepcopy(common),arrays=arrays,
                  data_bytes=sum(d['bytes'] for d in arrays),role_plan_sha256=hashlib.sha256(preflight_json(arrays)).hexdigest())
    validate_preflight_proposal(proposal,common)
    return proposal


def validate_preflight_proposal(proposal, common):
    """Validate exact native JSON identity and prepared role descriptors, not payload hashes."""
    sha=lambda s:type(s) is str and len(s)==64 and not set(s)-set('0123456789abcdef')
    if type(common) is not dict or set(common)!={'job_id','validation_sha256','source_manifest_sha256','tooling_sha256','frame','group','owners'}:
        raise ValueError('preflight common fields')
    if type(common['job_id']) is not str or not common['job_id'].isascii() or not common['job_id'].isdigit() or len(common['job_id'])>32:
        raise ValueError('preflight job identity')
    if not all(sha(common[k]) for k in ('validation_sha256','source_manifest_sha256','tooling_sha256')):
        raise ValueError('preflight source/tooling hashes')
    frame=dict(image=IMAGE,source=SOURCE,tooling=common['tooling_sha256'],layer=4,invocation=0,microbatch=0,phase='replay',kind='forward')
    if type(common['frame']) is not dict or type(common['group']) is not list or type(common['owners']) is not list or any(type(row) is not list for row in common['owners']) or preflight_json(common['frame'])!=preflight_json(frame) or preflight_json(common['group'])!=preflight_json(list(range(8))) or preflight_json(common['owners'])!=preflight_json([list(range(32*r,32*r+32)) for r in range(8)]):
        raise ValueError('preflight native frame/group/owners')
    if type(proposal) is not dict or set(proposal)!={'status','rank','common','arrays','data_bytes','role_plan_sha256'} or proposal['status']!='PREPARED_ROLES' or type(proposal['rank']) is not int or not 0<=proposal['rank']<8:
        raise ValueError('preflight proposal identity')
    if preflight_json(proposal['common'])!=preflight_json(common): raise ValueError('preflight mixed identity')
    arrays=proposal['arrays']
    if type(arrays) is not list or len(arrays)!=33 or len(preflight_json(proposal))>65536-1024:
        raise ValueError('preflight proposal size/roles')
    for d in arrays:
        if type(d) is not dict or set(d)!={'name','dtype','shape','bytes'} or type(d['shape']) is not list or any(type(n) is not int or n<0 for n in d['shape']) or type(d['bytes']) is not int:
            raise ValueError('preflight descriptor types')
    by_name={d['name']:d for d in arrays}
    if len(by_name)!=33: raise ValueError('preflight duplicate roles')
    try: na=by_name['alltoall_expert_probs']['shape'][0]; nr=by_name['received_experts']['shape'][0]
    except (KeyError,IndexError): raise ValueError('preflight role dimensions') from None
    if preflight_json(arrays)!=preflight_json(preflight_plan_arrays(na,nr)) or not sha(proposal['role_plan_sha256']) or proposal['role_plan_sha256']!=hashlib.sha256(preflight_json(arrays)).hexdigest():
        raise ValueError('preflight shape/byte/plan hash')
    if type(proposal['data_bytes']) is not int or proposal['data_bytes']!=sum(d['bytes'] for d in arrays):
        raise ValueError('preflight data bytes')
    return na,nr


def strict_allocate_quotas(proposals, framecommon):
    """Allocate all actual byte plans plus fixed reserves BEFORE payload directories exist."""
    if type(proposals) is not list or len(proposals)!=8: raise ValueError('eight preflight proposals required')
    geometry=[validate_preflight_proposal(p,framecommon) for p in proposals]
    if sorted(p['rank'] for p in proposals)!=list(range(8)): raise ValueError('duplicate/missing preflight ranks')
    ordered=sorted(proposals,key=lambda p:p['rank']); data=[p['data_bytes'] for p in ordered]
    quotas=[n+2**20+65536+(65536+2**20 if r==0 else 0) for r,n in enumerate(data)]
    if any(n>RANK_CAP for n in quotas) or sum(quotas)>JOB_CAP: raise ValueError('preflight rank/job byte cap')
    if sum(na for na,nr in geometry)!=REAL_ASSIGNMENTS or sum(nr for na,nr in geometry)>REAL_ASSIGNMENTS:
        raise ValueError('preflight aggregate declared geometry')
    return dict(status='QUOTA_APPROVED',common_sha256=hashlib.sha256(preflight_json(framecommon)).hexdigest(),
                proposal_sha256=[hashlib.sha256(preflight_json(p)).hexdigest() for p in ordered],data_bytes=data,
                job_quotas=quotas,stage_quotas=[n+2**20 for n in data])


def approved_prepared_quotas(rank, proposal, prepared_roles, decision):
    """Rebind the SAME immutable prepared roles; return payload-only writer allowances.

    Only use a decision returned by coordinate_preflight (all8 hashes verified).
    Native writer reuse of prepared_roles is still unwired, never call take twice.
    """
    actual=prepare_rank_proposal(rank,proposal['common'],prepared_roles)
    if preflight_json(actual)!=preflight_json(proposal) or decision['status']!='QUOTA_APPROVED' or decision['common_sha256']!=hashlib.sha256(preflight_json(proposal['common'])).hexdigest() or decision['proposal_sha256'][rank]!=hashlib.sha256(preflight_json(proposal)).hexdigest():
        raise ValueError('approved prepared-role binding')
    if preflight_json(decision['stage_quotas'])!=preflight_json([n+2**20 for n in decision['data_bytes']]) or decision['data_bytes'][rank]!=proposal['data_bytes']:
        raise ValueError('approved stage byte binding')
    return decision['stage_quotas']


def preflight_publish(path, value):
    """Publish <=64KiB JSON atomically/exclusively with file and directory fsync."""
    import tempfile
    encoded=preflight_json(value)
    if len(encoded)>65536: raise ValueError('preflight JSON byte cap')
    fd,temporary=tempfile.mkstemp(prefix='.quota-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream: stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary,path)  # Atomic no-replace publication; stale objects are never overwritten.
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        primary=sys.exception()
        try: Path(temporary).unlink()
        except OSError as error:
            if primary is None: raise
            primary.add_note(f'secondary preflight temporary cleanup: {error}')


def preflight_read(path):
    """Read bounded strict JSON; duplicate keys/nonfinite values are invalid evidence."""
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ValueError('duplicate preflight JSON key')
            result[k]=v
        return result
    if path.is_symlink(): raise ValueError('preflight symlink')
    with path.open('rb') as stream: data=stream.read(65537)
    if len(data)>65536: raise ValueError('preflight file byte cap')
    return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite preflight JSON')))


def coordinate_preflight(output, rank, proposal, timeout_seconds=20):
    """Filesystem-only bounded quota agreement; NOT native approval or GLOBAL completion.

    Output is an existing job-owned shared directory. Namespace binds job+all common
    identities; repeated/stale rank publication fails rather than reusing a decision.
    """
    import time
    from itertools import islice
    if type(rank) is not int or not 0<=rank<8 or type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds) or not 0<timeout_seconds<=20:
        raise ValueError('preflight rank/deadline')
    common=proposal['common']; validate_preflight_proposal(proposal,common)
    if proposal['rank']!=rank: raise ValueError('preflight caller rank')
    directory=Path(output)/('quota-'+hashlib.sha256(preflight_json(common)).hexdigest())
    directory.mkdir(exist_ok=True)
    if directory.is_symlink() or len(list(islice(directory.iterdir(),33)))>32: raise ValueError('preflight namespace/entry bound')
    deadline=time.monotonic()+timeout_seconds
    try:
        preflight_publish(directory/f'proposal-{rank}.json',proposal)
        while True:
            if time.monotonic()>=deadline: raise TimeoutError('bounded preflight timeout')
            if any((directory/f'error-{r}.json').exists() for r in range(8)): raise ValueError('peer preflight error')
            files=[directory/f'proposal-{r}.json' for r in range(8)]; decision=directory/'decision.json'
            if all(p.exists() for p in files):
                proposals=[preflight_read(p) for p in files]
                if any(p['rank']!=r for r,p in enumerate(proposals)) or preflight_json(proposals[rank])!=preflight_json(proposal): raise ValueError('preflight file/rank/hash binding')
                expected=strict_allocate_quotas(proposals,common)
                if rank==0: preflight_publish(decision,expected)
                if decision.exists():
                    actual=preflight_read(decision)
                    if preflight_json(actual)!=preflight_json(expected): raise ValueError('preflight decision/proposal hash mismatch')
                    if time.monotonic()>=deadline: raise TimeoutError('bounded preflight timeout')
                    return actual
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError('bounded preflight timeout')
            time.sleep(min(.01,remaining))
    except BaseException as error:
        try: preflight_publish(directory/f'error-{rank}.json',dict(status='PREFLIGHT_REJECTED',rank=rank,error=ascii(error)[:256]))
        except (OSError,ValueError): pass
        raise


def stage_capture_quiescent(capture, timeout_seconds=20):
    """Prepare once, agree quotas via files, then write ONE incomplete local stage.

    Caller must already be beyond the native synchronize/barrier; no new collective
    or synchronization is added. Global closure/numerics remain independently blocked.
    """
    if capture.errors or set(capture.replays)!=set(MODES) or any(type(n) is not int or n!=1 for n in capture.replays.values()) or set(capture.bindings)!=set(MODES):
        raise ValueError('incomplete/error capture before staging')
    bindings=[]
    for arm in MODES:
        entry=capture.bindings[arm]; binding=deepcopy(entry['metadata'])
        if entry['sha256']!=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest(): raise ValueError('capture binding hash')
        for name in ('group','local_experts'):
            if type(binding[name]) not in (list,tuple) or any(type(v) is not int for v in binding[name]): raise ValueError('capture integer group/owner order')
            binding[name]=list(binding[name])  # Only documented native tuple containers are normalized.
        bindings.append(binding)
    if preflight_json(bindings[0])!=preflight_json(bindings[1]): raise ValueError('capture mode binding mismatch')
    binding=bindings[0]; frozen=binding['frozen']
    if preflight_json(frozen)!=preflight_json(capture.frozen): raise ValueError('capture frozen source/tooling mismatch')
    tooling=hashlib.sha256(json.dumps(frozen['tooling'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    common=dict(job_id=os.environ['SLURM_JOB_ID'],validation_sha256=frozen['validation_sha256'],
                source_manifest_sha256=frozen['source_manifest_sha256'],tooling_sha256=tooling,
                frame=dict(image=IMAGE,source=SOURCE,tooling=tooling,layer=4,invocation=0,microbatch=0,phase='replay',kind='forward'),
                group=binding['group'],owners=binding['owners'])
    if len(capture.entries)!=31: raise ValueError('complete owned role count required')
    for entry in capture.entries:
        expected=dict(rank=capture.rank,layer=4,invocation=0,microbatch=0,phase='replay',kind='forward',image=IMAGE,source=SOURCE)
        if preflight_json({k:entry[k] for k in expected})!=preflight_json(expected) or entry['binding_sha256']!=capture.bindings[entry['arm']]['sha256']:
            raise ValueError('owned entry frozen/frame binding')
    take=owned_role_reader(capture)
    prepared=typed_roles(take,capture.rank)
    proposal=prepare_rank_proposal(capture.rank,common,prepared)
    decision=coordinate_preflight(capture.output,capture.rank,proposal,timeout_seconds)
    quotas=approved_prepared_quotas(capture.rank,proposal,prepared,decision)
    return write_rank_stage(Path(capture.output)/f'contributors-rank-{capture.rank}',capture.rank,binding,take,
                            job_quotas=quotas,prepared_roles=prepared,quota_approval=dict(proposal=proposal,decision=decision))


def owned_role_reader(capture):
    """Build a quiescent-only Torch bridge; never call from a native callback.

    One field is copied to CPU/retained at a time; release drops its owned snapshot.
    No float casts are used for probability or BF16 payload words.
    """
    import struct
    from collections.abc import Sequence
    indices = {(e['arm'],e['boundary']):i for i,e in enumerate(capture.entries)}
    if len(indices) != len(capture.entries) or len(capture.entries) != len(capture.owned): raise ValueError('owned role duplicate/length')
    class Words(Sequence):
        def __init__(self, data, code): self.data,self.word=data,struct.Struct('<'+code)
        def __len__(self): return self.data.nbytes//self.word.size
        def __getitem__(self,i):
            if not 0 <= i < len(self): raise IndexError(i)
            return self.word.unpack_from(self.data,i*self.word.size)[0]
    active = set()
    def take(arm,name):
        if active: raise ValueError('previous staged CPU role not released')
        i=indices[arm,name]; value=capture.owned[i]
        if value is None: raise ValueError('owned role already consumed')
        if isinstance(value,tuple): return WireField((8,),'i64',value,lambda:capture.owned.__setitem__(i,None))
        t=capture.torch
        if not isinstance(value,t.Tensor):
            if type(value).__module__ != 'numpy' or str(value.dtype) != 'int64' or tuple(value.shape)!=(8,): raise ValueError('owned host splits')
            return WireField((8,),'i64',value.tolist(),lambda:capture.owned.__setitem__(i,None))
        kinds={t.bool:('bool','B'),t.float32:('fp32','I'),t.bfloat16:('bf16','H'),t.int64:('i64','q'),t.int32:('i32','i')}
        if value.dtype not in kinds or sys.byteorder!='little': raise ValueError('owned tensor dtype/endian')
        dtype,code=kinds[value.dtype]; host=value.detach().cpu().contiguous()
        data=memoryview(host.view(t.uint8).numpy()).cast('B'); active.add(i)
        words=Words(data,code)
        def release():
            data.release(); active.remove(i); capture.owned[i]=None
        return WireField(tuple(host.shape),dtype,words,release)
    return take


class ContributorCapture:
    """Own producer-stream snapshots; incomplete implementation cannot claim closure."""
    def __init__(self, torch, output, rank, *, frozen=None):
        self.torch, self.output, self.rank = torch, output, rank
        self.frame, self.index = None, None
        self.owned, self.entries, self.handles, self.errors = [], [], [], []
        self.seen, self.replays, self.bytes = set(), {}, 0
        self.summary = {'status': 'partial', 'implementation_ready': False}
        self.frozen, self.bindings, self.shapes = deepcopy(frozen), {}, {}

    def error(self, error):
        if len(self.errors) < 32: self.errors.append(f'{type(error).__name__}: {error}'[:512])

    def selected(self):
        return self.frame is not None and self.frame[1] == 0 and self.torch.is_grad_enabled()

    def snapshot(self, name, tensor, *, projected=False, rows=None, width=None, dtype=None,
                 shape=None, host=False, maxrows=REAL_ASSIGNMENTS):
        """Use owned detached copies on the current producer stream; no host transfers."""
        try:
            if not self.selected(): return
            arm = self.frame[0]; key = (arm, name)
            if arm not in self.bindings: raise ValueError('missing frozen/group frame binding')
            if self.replays.get(arm) != 1 or key in self.seen: raise ValueError('missing/duplicate replay event')
            if not isinstance(tensor, self.torch.Tensor) or (not host and not tensor.is_cuda): raise ValueError('CUDA tensor required')
            actual = tuple(tensor.shape)
            if shape is not None and actual != shape: raise ValueError('exact tensor shape')
            shape = actual
            if not shape or (rows is not None and shape[0] != rows) or shape[0] > maxrows:
                raise ValueError('physical row bound')
            if width is not None and shape[-1] != width: raise ValueError('external latent/metadata width')
            if tensor.dtype != dtype: raise ValueError('typed snapshot dtype')
            elements = math.prod(shape[:-1]) * 64 if projected else math.prod(shape)
            size = elements * tensor.element_size()
            if self.bytes + size + 4*2**20 > RANK_CAP: raise ValueError('rank snapshot cap')
            owned = tensor.detach().index_select(len(shape)-1, self.index) if projected else tensor.detach().clone()
            self.entries.append(dict(arm=arm, boundary=name, rank=self.rank, layer=4, invocation=0,
                microbatch=0, phase='replay', kind='forward', image=IMAGE, source=SOURCE,
                shape=list(shape), stride=list(tensor.stride()), dtype=str(dtype), bytes=size,
                storage_offset=tensor.storage_offset(), channels=list(CHANNELS) if projected else None,
                binding_sha256=self.bindings[arm]['sha256'], storage='cuda' if tensor.is_cuda else 'cpu-native-metadata',
                row_identity='physical row; offline metadata association required'))
            self.owned.append(owned); self.seen.add(key); self.bytes += size
        except Exception as error: self.error(error)

    def latent(self, name, value, rows=None):
        expected = (4096, 1, 1792) if name in ('source', 'postcombine') else (value.shape[0], 1792)
        self.snapshot(name, value, projected=True, rows=rows, width=1792, dtype=self.torch.bfloat16,
                      shape=expected, maxrows=REAL_TOKENS if name in ('received', 'local') else REAL_ASSIGNMENTS)

    def router(self, result):
        if not self.selected(): return
        try:
            arm = self.frame[0]; self.replays[arm] = self.replays.get(arm, 0) + 1
            if self.replays[arm] != 1: raise ValueError('duplicate grad-enabled replay router')
            if not isinstance(result, tuple) or len(result) != 2: raise ValueError('router tuple')
            for name, value, dtype in zip(('router_probs', 'router_mask'), result, (self.torch.float32, self.torch.bool)):
                self.snapshot(name, value, shape=(4096, 256), dtype=dtype)
        except Exception as error: self.error(error)

    def host_splits(self, name, value):
        """Own native already-host split metadata; never initiate a device transfer."""
        key = (self.frame[0], name)
        if key in self.seen or self.frame[0] not in self.bindings: raise ValueError('split frame/duplicate')
        if isinstance(value, (tuple, list)):
            if len(value) != 8 or any(type(x) is not int or not 0 <= x <= REAL_ASSIGNMENTS for x in value):
                raise ValueError('split list words')
            owned = tuple(value)
        elif type(value).__module__ == 'numpy' and type(value).__name__ == 'ndarray':
            if tuple(value.shape) != (8,) or str(value.dtype) != 'int64': raise ValueError('native NumPy split shape/dtype')
            owned = value.copy()
        elif isinstance(value, self.torch.Tensor):
            self.snapshot(name, value, shape=(8,), dtype=self.torch.int64, host=True)
            return
        else: raise ValueError('unsupported native split storage')
        if self.bytes + 64 + 4*2**20 > RANK_CAP: raise ValueError('split rank cap')
        self.entries.append(dict(arm=self.frame[0], boundary=name, shape=[8], dtype='int64', bytes=64,
            storage='cpu-native-metadata', binding_sha256=self.bindings[self.frame[0]]['sha256'],
            rank=self.rank, layer=4, invocation=0, microbatch=0, phase='replay', kind='forward', image=IMAGE, source=SOURCE))
        self.owned.append(owned); self.seen.add(key); self.bytes += 64

    def method(self, dispatcher, mode, name, args, result, before=False):
        """Capture only source-defined public method boundaries, never decode handles."""
        if not self.selected(): return
        try:
            t = self.torch
            if mode != self.frame[0] or self.replays.get(mode) != 1: raise ValueError('method replay frame')
            if name == 'dispatch_preprocess' and not before and mode == 'alltoall':
                if not isinstance(result, tuple) or len(result) != 2: raise ValueError('preprocess tuple')
                value, probs = result
                if not isinstance(value, t.Tensor) or tuple(value.shape) != (32768, 1792) or value.dtype != t.bfloat16 or not value.is_cuda:
                    raise ValueError('alltoall preprocess latent')
                if not isinstance(probs, t.Tensor) or tuple(probs.shape) != (32768,) or probs.dtype != t.float32 or not probs.is_cuda:
                    raise ValueError('alltoall preprocess probabilities')
                self.snapshot('source_te_map', dispatcher.reversed_local_input_permutation_mapping, shape=(4096, 513), dtype=t.int32)
            elif name == 'token_dispatch' and not before and mode == 'alltoall':
                if not isinstance(result, tuple) or len(result) != 3 or result[1] is not None:
                    raise ValueError('alltoall dispatch tuple')
                if not isinstance(result[0], t.Tensor) or tuple(result[0].shape) != (result[0].shape[0], 1792) or result[0].dtype != t.bfloat16 or not result[0].is_cuda or result[0].shape[0] > REAL_ASSIGNMENTS:
                    raise ValueError('alltoall dispatch latent')
                if not isinstance(result[2], t.Tensor) or tuple(result[2].shape) != (result[0].shape[0],) or result[2].dtype != t.float32:
                    raise ValueError('alltoall dispatch probabilities')
                # The native dispatch has already completed its scheduled metadata DtoH.
                for field in ('input_splits', 'output_splits'):
                    self.host_splits(field, getattr(dispatcher, field))
            elif name == 'token_dispatch' and not before and mode == 'flex':
                if not isinstance(result, tuple) or len(result) != 3 or result[1] is not None: raise ValueError('dispatch tuple')
                manager = dispatcher._comm_manager
                self.latent('received', result[0])
                received = result[0].shape[0]
                if received > REAL_TOKENS or tuple(result[0].shape) != (received, 1792): raise ValueError('flex received row bound')
                if result[2] is not manager.dispatched_probs or not isinstance(result[2], t.Tensor) or tuple(result[2].shape) != (received, 8) or result[2].dtype != t.float32:
                    raise ValueError('flex dispatch probability tuple')
                self.shapes[(mode, 'received')] = received
                for field, dtype in (('token_indices', t.int64), ('token_probs', t.float32), ('dispatched_indices', t.int64), ('dispatched_probs', t.float32)):
                    rows = 4096 if field.startswith('token_') else received
                    self.snapshot(field, getattr(manager, field), shape=(rows, 8), dtype=dtype)
            elif name == 'dispatch_postprocess' and before and mode == 'alltoall':
                self.snapshot('forward_chunk_sizes', dispatcher.num_global_tokens_per_local_expert, shape=(8, 32), dtype=t.int64, host=True)
                self.snapshot('forward_chunk_order', dispatcher.sort_input_by_local_experts, shape=(256,), dtype=t.int64, host=True)
            elif name == 'dispatch_postprocess' and not before:
                self.expert_tuple(result, mode, save=False)
                if mode == 'flex':
                    self.snapshot('flex_te_map', dispatcher._comm_manager.reversed_mapping_for_combine,
                                  shape=(self.shapes[(mode, 'received')], 65), dtype=t.int32)
            elif name == 'combine_preprocess' and before and mode == 'alltoall':
                sizes = dispatcher.num_global_tokens_per_local_expert
                self.snapshot('restore_chunk_sizes', sizes.T.ravel(), shape=(256,), dtype=t.int64, host=True)
                self.snapshot('restore_chunk_order', dispatcher.restore_output_by_local_experts, shape=(256,), dtype=t.int64, host=True)
            elif name == 'combine_preprocess' and not before and mode == 'flex': self.latent('local', result, self.shapes[(mode, 'received')])
            elif name == 'token_combine' and not before and mode == 'alltoall': self.latent('returned', result, 4096*8)
        except Exception as error: self.error(error)

    def expert_tuple(self, inputs, mode, *, save):
        """Validate descriptors now; defer count sums/value checks until quiescence."""
        t = self.torch
        if mode != self.frame[0] or self.replays.get(mode) != 1 or not isinstance(inputs, tuple) or len(inputs) != 3:
            raise ValueError('expert input/replay tuple')
        value, counts, probs = inputs
        if not all(isinstance(x, t.Tensor) for x in inputs): raise ValueError('expert tensor types')
        rows = value.shape[0]
        if tuple(value.shape) != (rows, 1792) or not 0 <= rows <= REAL_ASSIGNMENTS or value.dtype != t.bfloat16 or not value.is_cuda:
            raise ValueError('expert latent shape/dtype')
        if tuple(probs.shape) != (rows,) or probs.dtype != t.float32 or not probs.is_cuda:
            raise ValueError('expert probability shape/dtype')
        if tuple(counts.shape) != (32,) or counts.dtype != t.int64: raise ValueError('expert counts shape/dtype')
        previous = self.shapes.get((mode, 'expert'))
        if previous is not None and previous != rows: raise ValueError('expert row mismatch')
        self.shapes[(mode, 'expert')] = rows
        if save:
            self.latent('expert_input', value)
            self.snapshot('expert_probs', probs, shape=(rows,), dtype=t.float32)
            self.snapshot('expert_counts', counts, shape=(32,), dtype=t.int64, host=True)

    def bind_models(self, models):
        """Bind local group getters and already frozen inputs before any callback work."""
        required = {'validation_sha256', 'tooling', 'source_manifest_sha256', 'source_files'}
        if not isinstance(self.frozen, dict) or set(self.frozen) != required: raise ValueError('frozen input labels')
        if any(not isinstance(self.frozen[k], str) or len(self.frozen[k]) != 64 for k in ('validation_sha256', 'source_manifest_sha256')):
            raise ValueError('frozen digest labels')
        if not self.frozen['tooling'] or not self.frozen['source_files']: raise ValueError('frozen module/source labels')
        for mode, model in models.items():
            dispatcher = model.token_dispatcher
            group = tuple(self.torch.distributed.get_process_group_ranks(dispatcher.ep_group))
            combined = tuple(self.torch.distributed.get_process_group_ranks(dispatcher.tp_ep_group))
            owners = tuple(model.local_expert_indices)
            if group != tuple(range(8)) or combined != group or model.num_local_experts != 32 or owners != tuple(range(self.rank*32, (self.rank+1)*32)):
                raise ValueError('native group order/expert ownership')
            binding = dict(group=group, owners=[list(range(r*32, (r+1)*32)) for r in range(8)],
                           source_columns=list(range(256)), local_experts=owners, frozen=self.frozen)
            self.bindings[mode] = dict(metadata=binding, sha256=hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest())

    def attach(self, models):
        """Capture setup failures remain secondary to the unchanged native workload."""
        try: self.attach_checked(models)
        except Exception as error: self.error(error)

    def attach_checked(self, models):
        """Prepare private channel indices before invocation; use only public hooks."""
        self.bind_models(models)
        self.index = self.torch.tensor(CHANNELS, dtype=self.torch.int64, device='cuda')
        for mode, model in models.items():
            def source(module, inputs, result, mode=mode):
                try:
                    if self.selected() and self.frame[0] == mode:
                        if not isinstance(result, tuple) or len(result) != 2 or result[1] is not None: raise ValueError('fc1 tuple')
                        self.latent('source', result[0], 4096)
                except Exception as error: self.error(error)
            def postcombine(module, inputs, mode=mode):
                try:
                    if self.selected() and self.frame[0] == mode:
                        if len(inputs) != 1: raise ValueError('fc2 input tuple')
                        self.latent('postcombine', inputs[0], 4096)
                except Exception as error: self.error(error)
            def expert_before(module, inputs, mode=mode):
                try:
                    if self.selected() and self.frame[0] == mode:
                        self.expert_tuple(inputs, mode, save=True)
                except Exception as error: self.error(error)
            def expert_after(module, inputs, result, mode=mode):
                try:
                    if self.selected() and self.frame[0] == mode:
                        if not isinstance(result, tuple) or len(result) != 2 or result[1] is not None: raise ValueError('expert result tuple')
                        rows = self.shapes[(mode, 'expert')]
                        self.latent('expert_output', result[0], rows)
                except Exception as error: self.error(error)
            self.handles.append(model.fc1_latent_proj.register_forward_hook(source))
            self.handles.append(model.fc2_latent_proj.register_forward_pre_hook(postcombine))
            self.handles.append(model.experts.register_forward_pre_hook(expert_before))
            self.handles.append(model.experts.register_forward_hook(expert_after))

    @contextmanager
    def arm(self, mode, microbatch):
        global CURRENT_CAPTURE
        previous, owner = self.frame, CURRENT_CAPTURE
        self.frame, CURRENT_CAPTURE = (mode, microbatch), self
        try: yield
        finally: self.frame, CURRENT_CAPTURE = previous, owner

    def serialize_quiescent(self):
        """Stage only after native quiescence; observer errors never replace native checks."""
        try:
            if self.errors: raise ValueError('capture errors prevent rank staging')
            self.summary.update(stage_capture_quiescent(self))
        except Exception as error:
            self.error(error)
            self.summary.update(status='partial',rank_serialized=False,implementation_ready=False)

    def __enter__(self): return self

    def __exit__(self, kind, error, traceback):
        for handle in reversed(self.handles):
            try: handle.remove()
            except Exception as problem: self.error(problem)
        self.summary.update(status='INCOMPLETE_JOB' if self.summary.get('rank_serialized') else 'partial', implementation_ready=False, bytes=self.bytes,
            entries=len(self.entries), errors=self.errors,
            bindings=self.bindings, deferred_checks='all8-rank actual file caps,88-boundary closure,identity join,expert probability association and native numerical gates',
            native_exception=None if error is None else f'{kind.__name__}: {error}'[:512])
        self.owned.clear()
        return False


def verify_contributor_artifacts(output, rank, summary):
    """Fail closed until typed serialization and independent real-data join are complete."""
    raise ValueError('contributor observer implementation is partial; no execution approval')


SOURCE = "e242f3c3c1136aeae498529bfde5e06bda9a7cca"
IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
UCCL = "52d4184c89deb5631be6dac6a9b1c40896c9e26e"
SCOPE = "EP8 combine-send8 native dispatcher component; not EP16/full KDA/DDP/performance acceptance"
MODES = ("alltoall", "flex")
RTOL, ATOL, RMS_RTOL, RMS_ATOL = 0.02, 1e-5, 2**-7, 1e-7
PACKAGES = {"torch": "2.14.0+rocm10.1.0", "triton": "3.8.0+git669b31ac.rocm10.1.0",
            "transformer-engine": "2.17.1+130099ce4", "uccl": "0.1.1", "deep_ep": "0.1.0"}
REQUIRED = {"--num-layers": "42", "--hidden-size": "3584", "--ffn-hidden-size": "17280",
            "--num-attention-heads": "28", "--seq-length": "4096", "--moe-latent-size": "1792",
            "--moe-ffn-hidden-size": "1920", "--moe-shared-expert-intermediate-size": "1920",
            "--num-experts": "256", "--moe-router-topk": "8", "--expert-model-parallel-size": "8",
            "--tensor-model-parallel-size": "1", "--expert-tensor-parallel-size": "1",
            "--context-parallel-size": "1", "--pipeline-model-parallel-size": "1",
            "--micro-batch-size": "1", "--global-batch-size": "16", "--main-grads-dtype": "fp32",
            "--moe-token-dispatcher-type": "alltoall", "--moe-flex-dispatcher-backend": "deepep",
            "--moe-router-dtype": "fp32", "--moe-router-score-function": "sigmoid",
            "--moe-router-topk-scaling-factor": "2.5", "--moe-router-load-balancing-type": "quantile_balancing",
            "--moe-router-quantile-balancing-method": "histogram", "--moe-router-quantile-balancing-ema": "0",
            "--moe-router-quantile-balancing-num-bins": "1000", "--moe-aux-loss-coeff": "0.0",
            "--recompute-granularity": "selective", "--recompute-modules": "moe",
            "--transformer-impl": "transformer_engine", "--distributed-timeout-minutes": "3"}
ENV = {"OMP_NUM_THREADS": "1", "MAX_JOBS": "2", "NCCL_P2P_DISABLE": "1",
       "OMPI_MCA_memory": "^patcher", "TORCH_COMPILE_DISABLE": "0", "UCCL_EP_TRANSPORT": "cxi",
       "UCCL_SOCKET_IFNAME": "hsn", "UCCL_CXI_THREADING": "safe",
       "UCCL_EP_COMBINE_CONFIG": "20,4,256,8,128"}
CONFIG_FIELDS = ('num_sms', 'num_max_nvl_chunked_send_tokens', 'num_max_nvl_chunked_recv_tokens',
                 'num_max_rdma_chunked_send_tokens', 'num_max_rdma_chunked_recv_tokens')
COMBINE_CONFIG = '20,4,256,8,128'
WRAPPER_SHA256 = 'd30ad5231f476c726c246240eec9b6a2f9124e16752900ca562653623e77a768'


def validate_config_environment(env):
    """Allow only the explicit combine-send8 override, never a dispatch override."""
    if 'UCCL_EP_DISPATCH_CONFIG' in env or env.get('UCCL_EP_COMBINE_CONFIG') != COMBINE_CONFIG:
        raise ValueError('requires combine20,4,256,8,128 and absent dispatch override')


def getter_observation(buffer, env):
    """Read public getter fields and source-consumed environment; not a per-call trace."""
    relevant = {k: v for k, v in env.items() if k.startswith('UCCL_')}
    overrides = {k: env[k] for k in ('UCCL_EP_DISPATCH_CONFIG', 'UCCL_EP_COMBINE_CONFIG', 'EFA_HOME') if k in env}
    configs = (buffer.get_dispatch_config(8), buffer.get_combine_config(8))
    return dict(scope='public getter observations, not actual per-call Config traces',
                world=8, num_sms=buffer.num_sms, override_environment=overrides, uccl_environment=relevant,
                constructor_inputs=dict(dispatch=[20,6,256,6,128], combine=[20,4,256,8,128]),
                effective={name: {field: int(getattr(config, field)) for field in CONFIG_FIELDS}
                           for name, config in zip(('dispatch', 'combine'), configs)})


def validate_getter_observation(observation, previous=None):
    """Fail closed on getter/override drift, including constructor receive rounding."""
    validate_config_environment(observation['override_environment'])
    wanted = dict(dispatch=dict(zip(CONFIG_FIELDS, (20,6,256,6,132))),
                  combine=dict(zip(CONFIG_FIELDS, (20,4,256,8,128))))
    if observation['world'] != 8 or observation['num_sms'] != 20 or observation['effective'] != wanted:
        raise ValueError(f'effective EP8 getter Config differs: {observation}')
    if previous is not None and observation != previous:
        raise ValueError('getter fields or observed UCCL/override environment changed during work')


def timeout_identity():
    """Observe the installed GNU timeout before exec; fail closed on another utility."""
    path = Path('/usr/bin/timeout')
    version = subprocess.check_output([str(path), '--version'], text=True, timeout=5)
    help_text = subprocess.check_output([str(path), '--help'], text=True, timeout=5)
    if 'GNU coreutils' not in version or any(flag not in help_text for flag in ('--foreground', '--signal', '--kill-after')):
        raise ValueError('requires GNU timeout foreground/signal/kill-after contract')
    return dict(path=str(path), realpath=str(path.resolve()), sha256=sha256(path), version=version, help=help_text)


def train(output, source, entrypoint, argv, *, run=Path('/apertus')):
    """Exec GNU foreground timeout in the outer runtime-owned group; never fork a session."""
    settings = json.loads((run / 'resolved.json').read_text())
    validate(settings)
    rank = int(os.environ['RANK'])
    observation = timeout_identity()
    observation.update(wrapper_pid=os.getpid(), wrapper_pgid=os.getpgrp(), seconds=660, kill_after_seconds=10)
    write_json(output / f'timeout-{rank}.json', observation)
    env = dict(os.environ, APERTUS_NATIVE_DISPATCHER_PGID=str(os.getpgrp()))
    command = ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '660s',
               sys.executable, '-u', str(Path(__file__).resolve()), 'gpu', str(output), str(source), entrypoint, *argv]
    os.execve(command[0], command, env)


def effective_flags(method, args, kwargs):
    """Bind actual native positional/keyword arguments and declared defaults."""
    bound = inspect.signature(method).bind(*args, **kwargs)
    bound.apply_defaults()
    return {k: bound.arguments[k] for k in ('async_finish', 'allocate_on_comm_stream')}


def validate(settings):
    """Reject changes to the bounded component's source, geometry and numerical path."""
    validate_config_environment(settings['environment'])
    r = settings["resources"]
    if settings["source"]["commit"] != SOURCE or settings["image"]["sha256"] != IMAGE:
        raise ValueError("native dispatcher requires exact source/image")
    if (r["nodes"], r["gpus_per_node"], r["time"], r.get("reservation")) != (2, 4, "00:30:00", "cscs"):
        raise ValueError("requires two nodes/eight GPUs, cscs, 30 minutes")
    if r["distribution"].split(":")[0] != "block" or settings["procedure"]["kind"] != "train" or settings.get("compare_run"):
        raise ValueError("requires block train/check component")
    args = settings["args"]
    if any(option(args, k) != v for k, v in REQUIRED.items()) or any(
        flag not in args for flag in ("--bf16", "--sssglu", "--disable-bias-linear", "--moe-grouped-gemm",
                                     "--moe-permute-fusion", "--enable-experimental", "--no-gradient-accumulation-fusion")):
        raise ValueError("native component arguments differ")
    forbidden = {"--fp8", "--fp4", "--cpu-offloading", "--fine-grained-activation-offloading",
                 "--moe-use-offloading-experts", "--moe-offload-main-grad", "--moe-offload-activations",
                 "--moe-shared-expert-overlap", "--overlap-moe-expert-parallel-comm", "--use-te-op-fuser",
                 "--use-distributed-optimizer", "--use-layer-wise-distributed-optimizer"}
    if any(a.split("=", 1)[0] in forbidden for a in args) or any(settings["environment"].get(k) != v for k, v in ENV.items()):
        raise ValueError("unsupported optimization/environment")


def accepted(metrics):
    """Keep both existing finite elementwise and RMS numerical gates, including zeros."""
    keys = ("max_tolerance_ratio", "rms_error", "reference_rms")
    return metrics["finite"] and all(metrics.get(k) is not None and math.isfinite(metrics[k]) for k in keys) and \
        metrics["max_tolerance_ratio"] <= 1 and metrics["rms_error"] <= RMS_ATOL + RMS_RTOL * metrics["reference_rms"]


def compare(torch, actual, expected, label, emit, *, contributions=1, scale=None):
    """Check all elements, propagating per-contribution accumulation budgets only."""
    if actual is None or expected is None or actual.shape != expected.shape:
        emit(label, False, error="missing tensor/shape mismatch")
        return
    a, b = actual.detach().float(), expected.detach().float()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    m = dict(finite=finite, max_tolerance_ratio=None, rms_error=None, reference_rms=None)
    if finite:
        error = (a - b).abs()
        allowance = ATOL * contributions + RTOL * (b.abs() if scale is None else scale)
        m.update(max_tolerance_ratio=(error / allowance).max().item(), max_abs=error.max().item(),
                 rms_error=error.square().mean().sqrt().item(), reference_rms=b.square().mean().sqrt().item())
    emit(label, accepted(m), **m)


def exact(torch, a, b, label, emit):
    """Integer routing, RNG and state have no tolerance."""
    emit(label, a is not None and b is not None and a.shape == b.shape and bool(torch.equal(a, b)))


def rng_state(torch, tracker):
    """Own every RNG family so paired native arms start identically."""
    import numpy as np
    return (random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state().clone(), {k: v.clone() for k, v in tracker.get_states().items()})


def restore_rng(torch, tracker, state):
    """Restore the saved native RNG families without changing the native graph."""
    import numpy as np
    random.setstate(state[0])
    np.random.set_state(deepcopy(state[1]))
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])
    tracker.set_states({k: v.clone() for k, v in state[4].items()})


def rng_digest(state):
    """Fingerprint host/device/tracker state after completed backward."""
    host = pickle.dumps(state[:2], protocol=5)
    data = host + state[2].cpu().numpy().tobytes() + state[3].cpu().numpy().tobytes()
    for k, v in sorted(state[4].items()):
        data += k.encode() + v.cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def coverage(counts, rank):
    """Reject assumed coverage: counts are actual source-to-destination assignments."""
    return len(counts) == 8 and len(set(counts)) > 1 and all(c >= 0 for c in counts) and \
        sum(c for i, c in enumerate(counts) if i != rank and i // 4 == rank // 4) > 0 and \
        sum(c for i, c in enumerate(counts) if i // 4 != rank // 4) > 0


def identity():
    """Verify installed compatibility modules, build pin and unchanged vendor versions."""
    import deep_ep
    import deep_ep.buffer
    import deep_ep.utils
    import uccl.ep
    from megatron.core.transformer.moe import fused_a2a
    versions = {k: metadata.version(k) for k in PACKAGES}
    if versions != PACKAGES:
        raise ValueError(f"package identity differs: {versions}")
    pin = f"uccl https://github.com/uccl-project/uccl.git {UCCL}"
    if pin not in Path("/opt/alps/apertus2-rocm/source-pins.txt").read_text().splitlines():
        raise ValueError("UCCL build pin differs")
    paths = {m.__name__: str(Path(m.__file__).resolve()) for m in (deep_ep, deep_ep.buffer, deep_ep.utils, uccl.ep)}
    root = Path("/opt/venv/lib/python3.12/site-packages")
    expected_paths = {"deep_ep": str(root / "deep_ep/__init__.py"),
                      "deep_ep.buffer": str(root / "deep_ep/buffer.py"),
                      "deep_ep.utils": str(root / "deep_ep/utils.py"),
                      "uccl.ep": str(root / "uccl/ep.abi3.so")}
    if paths != expected_paths or \
            deep_ep.Buffer is not fused_a2a.Buffer or deep_ep.Buffer.__module__ != "deep_ep.buffer":
        raise ValueError("native consumer did not load installed UCCL compatibility Buffer")
    if deep_ep.buffer.__dict__.get("Buffer") is not deep_ep.Buffer:
        raise ValueError("unexpected compatibility class")
    # The installed wrapper must consume the real extension, not an unrelated DeepEP.
    wrapper = Path(deep_ep.buffer.__file__).read_text()
    if "uccl" not in wrapper or "comm_stream_ptr=int(self._comm_stream.cuda_stream)" not in wrapper or "_record_stream_safe" not in wrapper:
        raise ValueError("compatibility wrapper does not identify UCCL")
    patches = {"0001-alps-cxi-rocm-ep.patch": "d590f34fdf41a3c8582bb5237cde125485a4d6208c41d405bdf86e1d01a42d00",
               "0002-preserve-torch-resource-ownership.patch": "36e04b242db03a0fd24ce123483ebc3dd8d708de7ecce49b618a3f8f2072971b"}
    if any(sha256(Path("/opt/alps/apertus2-rocm/patches/uccl") / name) != value for name, value in patches.items()):
        raise ValueError("retained UCCL corrections differ")
    return {"versions": versions, "paths": paths, "uccl_pin": UCCL,
            "patches": patches, "files": {k: sha256(Path(v)) for k, v in paths.items()}}


@contextmanager
def observe(torch, model, mode, observations, handles, events):
    """Call through native dispatcher unchanged; own evidence, never substitute routing."""
    dispatcher = model.token_dispatcher
    originals = {}
    hook = None
    def router_hook(module, inputs, result):
        probs, routing = result
        row = {"probs": probs.detach().clone(), "map": routing.detach().clone(),
               "enabled": torch.is_grad_enabled(), "gradient": None}
        observations.append(row)
        if CURRENT_CAPTURE is not None: CURRENT_CAPTURE.router(result)
        if probs.requires_grad:
            probs.register_hook(lambda grad: row.update(gradient=grad.detach().clone()))

    def wrap(name):
        original = getattr(dispatcher, name)
        originals[name] = (name in vars(dispatcher), vars(dispatcher).get(name))
        def call(*args, **kwargs):
            flags = effective_flags(original, args, kwargs) if mode == 'flex' and name in {'token_dispatch', 'token_combine'} else {}
            if CURRENT_CAPTURE is not None: CURRENT_CAPTURE.method(dispatcher, mode, name, args, None, before=True)
            result = original(*args, **kwargs)
            if CURRENT_CAPTURE is not None: CURRENT_CAPTURE.method(dispatcher, mode, name, args, result)
            if name == "token_dispatch":
                if mode == "flex":
                    manager = dispatcher._comm_manager
                    handles.append(manager.handle)  # Own until ALL backward work is complete.
                    events.append({"kind": name, "enabled": torch.is_grad_enabled(),
                                   "handle": id(manager.handle), "buffer": id(sys.modules[
                                       "megatron.core.transformer.moe.fused_a2a"]._buffer),
                                    **flags,
                                   "grad_fn": type(result[0].grad_fn).__name__,
                                   "indices": manager.token_indices.detach().clone()})
                else:
                    events.append({"kind": name, "input_splits": list(map(int, dispatcher.input_splits)),
                                   "output_splits": list(map(int, dispatcher.output_splits))})
            elif name == "dispatch_postprocess":
                events.append({"kind": name, "counts": result[1].detach().clone()})
            elif mode == "flex" and name == "token_combine":
                events.append({"kind": name, "enabled": torch.is_grad_enabled(),
                               **flags,
                               "grad_fn": type(result.grad_fn).__name__})
            return result
        setattr(dispatcher, name, call)
    try:
        hook = model.router.register_forward_hook(router_hook)
        for name in ("dispatch_preprocess", "token_dispatch", "dispatch_postprocess", "combine_preprocess", "token_combine"):
            wrap(name)
        yield
    finally:
        if hook is not None:
            hook.remove()
        for name in originals:
            owned, value = originals[name]
            if owned: setattr(dispatcher, name, value)
            else: delattr(dispatcher, name)


def required_labels(parameters):
    """Enumerate complete coverage independently of claimed status."""
    result = {"identity", "groups", "initial-weights", "post-destroy-storage", "post-destroy-rccl", "buffer-lifetime"}
    for mode in MODES:
        result.add(f"{mode}/weights-unmodified")
    for mb in range(2):
        p = f"mb{mb}"
        result.update(f"{p}/{x}" for x in ("forward", "input-gradient", "probability-gradient", "router-state", "rng"))
        for mode in MODES:
            result.update(f"{p}/{mode}/{x}" for x in ("replay", "coverage", "metadata", "output-unmutated"))
        for name in parameters:
            result.update(f"{p}/{kind}/{name}" for kind in ("gradient", "accumulated-gradient"))
    return result


def run_probe(emit, record, source, expected, *, capture):
    """Run paired native checkpointed MoELayers and destroy transport only after backward."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.transformer.moe import fused_a2a
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher, MoEFlexTokenDispatcher
    from megatron.core.transformer.spec_utils import build_module
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args
    verify_imports(source, expected)
    record["identity"] = identity()
    if record['identity']['files']['deep_ep.buffer'] != WRAPPER_SHA256:
        raise ValueError('combine8 requires the audited installed compatibility wrapper')
    emit("identity", True)
    groups = {"ep": parallel_state.get_expert_model_parallel_group(),
              "expt_tp": parallel_state.get_expert_tensor_parallel_group(),
              "tp_ep": parallel_state.get_expert_tensor_and_model_parallel_group()}
    record["groups"] = {k: dist.get_process_group_ranks(g) for k, g in groups.items()}
    topology = [None] * 8
    import socket
    dist.all_gather_object(topology, (socket.gethostname(), torch.cuda.current_device()))
    record["topology"] = topology
    emit("groups", record["groups"]["ep"] == record["groups"]["tp_ep"] == list(range(8)) and
         len(record["groups"]["expt_tp"]) == 1 and groups["ep"] is not dist.group.WORLD and
         len({t[0] for t in topology[:4]}) == len({t[0] for t in topology[4:]}) == 1 and
         topology[0][0] != topology[4][0] and [t[1] for t in topology] == list(range(4)) * 2)
    config = core_transformer_config_from_args(get_args())
    if get_num_microbatches() != 2 or config.gradient_accumulation_fusion or config.fp8 or config.fp4:
        raise ValueError("effective native precision/accumulation differs")
    from megatron.core.activations import sssglu_act
    if config.activation_func is not sssglu_act or not config.bias_activation_fusion or config.use_te_activation_func or \
            config.cuda_graph_impl != "none" or any(getattr(config, k) for k in (
                "cpu_offloading", "fine_grained_activation_offloading", "moe_use_offloading_experts", "moe_shared_expert_overlap")):
        raise ValueError("effective activation/offload/overlap path differs")
    spec = get_moe_module_spec(use_te=True, num_experts=256, moe_grouped_gemm=True)
    models = {}
    for mode, cls in zip(MODES, (MoEAlltoAllTokenDispatcher, MoEFlexTokenDispatcher)):
        selected = deepcopy(config)
        selected.moe_token_dispatcher_type = mode
        model = build_module(spec, config=selected, layer_number=4).cuda().train()
        model.set_layer_number(4)
        if type(model) is not MoELayer or type(model.experts) is not TEGroupedMLP or \
                type(model.token_dispatcher) is not cls or model.num_local_experts != 32 or not model.moe_layer_recompute:
            raise ValueError("native classes/recompute differ")
        if model.token_dispatcher.ep_group is not groups["ep"] or model.token_dispatcher.tp_ep_group is not groups["tp_ep"]:
            raise ValueError("native group collection differs")
        models[mode] = model
    # Own the native singleton explicitly, as the pinned Flex tests do; never
    # replace it while saved autograd handles exist. Preserve native configs.
    record['getters_before_hints'] = getter_observation(fused_a2a.Buffer, os.environ)
    validate_getter_observation(record['getters_before_hints'])
    emit('getter-configs-before-hints', True, observation=record['getters_before_hints'])
    hints = [fused_a2a.Buffer.get_dispatch_config(8), fused_a2a.Buffer.get_combine_config(8)]
    if fused_a2a._buffer is not None:
        raise ValueError("unexpected pre-existing native buffer")
    fused_a2a._buffer = fused_a2a.Buffer(
        group=groups["tp_ep"],
        num_nvl_bytes=max(c.get_nvl_buffer_size_hint(1792 * 2, 8) for c in hints),
        num_rdma_bytes=max(c.get_rdma_buffer_size_hint(1792 * 2, 8) for c in hints),
        num_qps_per_rank=64, explicitly_destroy=True, is_intranode=False)
    if not type(fused_a2a._buffer.runtime).__module__.startswith("uccl.ep") or fused_a2a._buffer.runtime.get_num_rdma_ranks() != 2:
        raise ValueError("native Buffer does not consume the two-node UCCL extension")
    record["buffer_runtime"] = f"{type(fused_a2a._buffer.runtime).__module__}.{type(fused_a2a._buffer.runtime).__qualname__}"
    params = {m: dict(model.named_parameters()) for m, model in models.items()}
    if params["alltoall"].keys() != params["flex"].keys():
        raise ValueError("parameter inventory differs")
    if any(p.dtype != torch.bfloat16 for values in params.values() for p in values.values()):
        raise ValueError("expected native BF16 leaf parameters")
    with torch.no_grad():
        for name, p in params["alltoall"].items():
            params["flex"][name].copy_(p)
        buffers = dict(models["alltoall"].named_buffers())
        for name, b in models["flex"].named_buffers():
            b.copy_(buffers[name])
    record["parameters"] = {k: list(p.shape) for k, p in params["alltoall"].items()}
    initial = {m: {k: p.detach().clone() for k, p in ps.items()} for m, ps in params.items()}
    emit("initial-weights", all(torch.equal(initial["alltoall"][k], initial["flex"][k]) for k in initial["flex"]))
    totals = {m: {k: torch.zeros_like(p, dtype=torch.float32) for k, p in ps.items()} for m, ps in params.items()}
    budgets = {k: torch.zeros_like(p, dtype=torch.float32) for k, p in params["alltoall"].items()}
    tracker = tensor_parallel.get_cuda_rng_tracker()
    handles, buffer_ids, held = [], [], []
    record["microbatches"] = []
    capture.attach(models)
    for mb in range(2):
        generator = torch.Generator(device="cuda").manual_seed(1000 + 8 * mb + dist.get_rank())
        x = torch.randn(4096, 1, 3584, device="cuda", dtype=torch.bfloat16, generator=generator)
        dy = torch.randn(x.shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
        start = rng_state(torch, tracker)
        reference = None
        states, rngs, row = {}, {}, {"index": mb, "metadata": {}}
        for mode, model in models.items():
            restore_rng(torch, tracker, start)
            observations, events = [], []
            value = x.detach().clone().requires_grad_(True)
            with capture.arm(mode, mb), observe(torch, model, mode, observations, handles, events):
                output, bias = model(value)
                before = output.detach().clone()
                if bias is not None:
                    raise ValueError("unexpected output bias")
                output.backward(dy)
            pfx = f"mb{mb}/{mode}"
            exact(torch, output, before, f"{pfx}/output-unmutated", emit)
            if len(observations) != 2 or [o["enabled"] for o in observations] != [False, True]:
                raise ValueError("expected native full-MoE original/replay routing")
            first, replay = observations
            emit(f"{pfx}/replay", torch.equal(first["map"], replay["map"]) and
                 torch.equal(first["probs"], replay["probs"]) and bool((replay["map"].sum(-1) == 8).all()))
            if replay["gradient"] is None or not bool((replay["gradient"] != 0).any()):
                raise ValueError("missing/zero native probability gradient")
            route = replay["map"]
            expert_counts = route.sum(0).long()
            all_counts = [torch.empty_like(expert_counts) for _ in range(8)]
            dist.all_gather(all_counts, expert_counts, group=groups["ep"])
            matrix = torch.stack(all_counts).reshape(8, 8, 32)
            send = matrix[dist.get_rank()].sum(-1).tolist()
            recv = matrix[:, dist.get_rank()].sum(-1).tolist()
            expected_counts = matrix[:, dist.get_rank()].sum(0).cpu()
            emit(f"{pfx}/coverage", coverage(send, dist.get_rank()) and sum(send) == 4096 * 8)
            metadata_ok = True
            for event in events:
                if event["kind"] == "dispatch_postprocess":
                    metadata_ok &= bool(torch.equal(event.pop("counts").cpu().long(), expected_counts))
                elif mode == "alltoall" and event["kind"] == "token_dispatch":
                    metadata_ok &= event["input_splits"] == send and event["output_splits"] == recv
                elif mode == "flex" and event["kind"] == "token_dispatch":
                    idx = event.pop("indices")
                    reconstructed = torch.zeros_like(route).scatter(1, idx.long(), True)
                    metadata_ok &= bool(torch.equal(reconstructed, route)) and event["async_finish"] and event["allocate_on_comm_stream"]
                    buffer_ids.append(event["buffer"])
                    if event["enabled"]:
                        metadata_ok &= event["grad_fn"] == "FusedDispatchBackward"
                elif mode == "flex" and event["kind"] == "token_combine" and event["enabled"]:
                    metadata_ok &= event["grad_fn"] == "FusedCombineBackward" and event['async_finish'] and event['allocate_on_comm_stream']
            emit(f"{pfx}/metadata", metadata_ok and len(events) == (6 if mode == "flex" else 4))
            row["metadata"][mode] = {"send_assignments": send, "receive_assignments": recv,
                                      "expert_counts": expected_counts.tolist(), "events": events}
            states[mode] = {k: b.detach().clone() for k, b in model.router.named_buffers()}
            rngs[mode] = rng_digest(rng_state(torch, tracker))
            if mode == "alltoall":
                reference = (before, value.grad.detach().clone(), replay)
            else:
                compare(torch, before, reference[0], f"mb{mb}/forward", emit)
                compare(torch, value.grad, reference[1], f"mb{mb}/input-gradient", emit)
                compare(torch, replay["gradient"], reference[2]["gradient"], f"mb{mb}/probability-gradient", emit)
                if not torch.equal(replay["map"], reference[2]["map"]) or not torch.equal(replay["probs"], reference[2]["probs"]):
                    raise ValueError("paired native routing differs")
            for name, parameter in params[mode].items():
                if parameter.grad is None or parameter.grad.dtype != torch.bfloat16:
                    raise ValueError(f"missing/non-BF16 native gradient: {mode}/{name}")
                totals[mode][name].add_(parameter.grad.float())
                if mode == "alltoall":
                    budgets[name].add_(parameter.grad.float().abs())
                else:
                    compare(torch, parameter.grad, params["alltoall"][name].grad, f"mb{mb}/gradient/{name}", emit)
                    compare(torch, totals[mode][name], totals["alltoall"][name], f"mb{mb}/accumulated-gradient/{name}",
                            emit, contributions=mb + 1, scale=budgets[name])
            held.append((output.detach(), before))
            del output, value
        # Metric sample buffers contain unused/uninitialized slots: only active slots are state.
        def active(k, b, state):
            return b[:int(state["expert_load_sample_count"].item())] if k in {"mbs_expert_load_samples", "seq_expert_load_samples"} else b
        emit(f"mb{mb}/router-state", states["alltoall"].keys() == states["flex"].keys() and all(
            torch.equal(active(k, b, states["alltoall"]), active(k, states["flex"][k], states["flex"]))
            for k, b in states["alltoall"].items()))
        emit(f"mb{mb}/rng", rngs["alltoall"] == rngs["flex"])
        record["microbatches"].append(row)
        for model in models.values():
            model.zero_grad(set_to_none=True)
    for mode, ps in params.items():
        emit(f"{mode}/weights-unmodified", all(torch.equal(p, initial[mode][k]) for k, p in ps.items()))
    record['getters_after_work'] = getter_observation(fused_a2a.Buffer, os.environ)
    validate_getter_observation(record['getters_after_work'], record['getters_before_hints'])
    emit('getter-configs-after-work', True, observation=record['getters_after_work'])
    # The only global synchronization is AFTER every native backward, before destruction.
    torch.cuda.synchronize()
    dist.barrier()
    capture.serialize_quiescent()
    buffer = fused_a2a._buffer
    emit("buffer-lifetime", buffer is not None and len(set(buffer_ids)) == 1 and buffer_ids[0] == id(buffer) and len(handles) == 4)
    stream = buffer.get_comm_stream()
    for values in totals.values():
        held.extend((v, v.detach().clone()) for v in values.values())
    storage = torch.ones(4096, device="cuda")
    storage.record_stream(stream)
    buffer.destroy()
    fused_a2a._buffer = None
    handles.clear()
    emit("post-destroy-storage", bool((storage == 1).all()) and all(torch.equal(a, b) for a, b in held))
    with torch.cuda.stream(stream):
        after = torch.tensor([dist.get_rank() + 1], device="cuda", dtype=torch.int32)
    torch.cuda.current_stream().wait_stream(stream)
    dist.all_reduce(after, group=groups["ep"])
    emit("post-destroy-rccl", after.item() == 36)
    torch.cuda.synchronize()
    dist.barrier()


def gpu(output, source, entrypoint, argv, *, run=Path("/apertus")):
    """Use ordinary frozen-source initialization, import verification and bounded exit."""
    if os.getpgrp() != int(os.environ.get('APERTUS_NATIVE_DISPATCHER_PGID', '-1')):
        raise ValueError('GPU worker escaped outer runtime process group')
    settings = json.loads((run / "resolved.json").read_text())
    validate(settings)
    validate_config_environment(os.environ)
    manifest = json.loads((run / "source/manifest.json").read_text())
    if manifest["selection"]["commit"] != SOURCE or entrypoint != settings["entrypoint"] or sha256(source / entrypoint) != manifest["files"][entrypoint]:
        raise ValueError("source/entrypoint identity differs")
    rank, local = int(os.environ["RANK"]), int(os.environ["SLURM_LOCALID"])
    if int(os.environ["WORLD_SIZE"]) != 8 or rank != int(os.environ["SLURM_NODEID"]) * 4 + local:
        raise ValueError("requires eight direct block Slurm ranks")
    if any(os.environ.get(k) != v for k, v in ENV.items()) or os.environ.get("HWLOC_COMPONENTS") != "-gl":
        raise ValueError("effective component environment differs")
    source = source.resolve()
    sys.path.insert(0, str(source))
    sys.argv = [str(source / entrypoint), *argv]
    record = dict(status="failed", rank=rank, source=SOURCE, image=IMAGE, scope=SCOPE, checks=[],
                  worker_pid=os.getpid(), worker_pgid=os.getpgrp(),
                  accumulation="explicit FP32 sum of unfused BF16 leaf gradients, not DDP/main_grad",
                  policy=dict(rtol=RTOL, atol=ATOL, rms_rtol=RMS_RTOL, rms_atol=RMS_ATOL))
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, exit=False)
    with (output / f"native-dispatcher-{rank}.json").open("x") as summary, \
            (output / f"native-dispatcher-{rank}.jsonl").open("x", buffering=1) as stream:
        def emit(label, passed, **details):
            row = dict(label=label, passed=bool(passed), **details)
            record["checks"].append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        try:
            import torch
            if torch.cuda.device_count() != 4 or not torch.version.hip:
                raise ValueError("all four ROCm GPUs must remain visible")
            torch.cuda.set_device(local)
            torch.set_num_threads(1)
            record["local_device"] = torch.cuda.current_device()
            record["visible_devices"] = torch.cuda.device_count()
            from megatron.training.initialize import initialize_megatron
            verify_imports(source, manifest["files"])
            initialize_megatron()
            verify_imports(source, manifest["files"])
            capture = ContributorCapture(torch, output, rank, frozen=dict(
                validation_sha256=settings['validation']['sha256'], tooling=deepcopy(settings['tooling']),
                source_manifest_sha256=sha256(run / 'source/manifest.json'), source_files=deepcopy(manifest['files'])))
            try:
                with capture:
                    run_probe(emit, record, source, manifest["files"], capture=capture)
            finally:
                record['contributor_capture'] = capture.summary
            validate_record(record, rank)
            verify_contributor_artifacts(output, rank, record['contributor_capture'])
            record["status"] = "passed"
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            try:
                # On a native failure do not free a live buffer; the bounded child exits.
                import torch.distributed as dist
                if dist.is_initialized() and record["status"] == "passed":
                    from megatron.core import parallel_state
                    parallel_state.destroy_model_parallel()
                    dist.destroy_process_group()
                write_json(output / f"imports-{rank}.json", {"verified_modules": verify_imports(source, manifest["files"])})
            except BaseException as error:
                record.update(status="failed", cleanup_error=str(error))
                raise
            finally:
                summary.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
                if record["status"] == "passed":
                    faulthandler.cancel_dump_traceback_later()


def validate_record(record, rank):
    """Reject missing, failed or duplicated coverage, regardless of claimed status."""
    validate_getter_observation(record['getters_before_hints'])
    validate_getter_observation(record['getters_after_work'], record['getters_before_hints'])
    checks = record.get("checks", [])
    labels = [r["label"] for r in checks]
    if record.get("source") != SOURCE or record.get("image") != IMAGE or record.get("scope") != SCOPE or \
            record.get("rank") != rank or not record.get("parameters") or not checks or \
            not all(r["passed"] for r in checks) or len(labels) != len(set(labels)) or \
            not (required_labels(record["parameters"]) | {'getter-configs-before-hints', 'getter-configs-after-work'}).issubset(labels) or \
            [m["index"] for m in record.get("microbatches", [])] != [0, 1]:
        raise ValueError("incomplete/failed native dispatcher evidence")


def check(run):
    """Require complete evidence from all eight ranks; CPU checks are not GPU acceptance."""
    try:
        settings = json.loads((run / "resolved.json").read_text())
        validate(settings)
        for rank in range(8):
            record = json.loads((run / f"output/native-dispatcher-{rank}.json").read_text())
            validate_record(record, rank)
            verify_contributor_artifacts(run / 'output', rank, record['contributor_capture'])
            if record["status"] != "passed" or json.loads((run / f"output/imports-{rank}.json").read_text())["verified_modules"] <= 0:
                raise ValueError("rank failed/import evidence missing")
        result = dict(status="passed", scope=SCOPE, ranks=8, source=SOURCE, image=IMAGE)
    except Exception as error:
        write_json(run / "output/checks.json", dict(status="failed", scope=SCOPE, error=str(error)))
        raise
    write_json(run / "output/checks.json", result)


if __name__ == "__main__":
    if not IMPLEMENTATION_READY:
        raise SystemExit("PARTIAL CPU IMPLEMENTATION: typed real-data closure/identity join required; native execution disabled")
    if len(sys.argv) >= 5 and sys.argv[1] == "train":
        train(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif len(sys.argv) >= 5 and sys.argv[1] == 'gpu':
        gpu(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        check(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: native_dispatcher_probe.py train OUTPUT SOURCE ENTRYPOINT [ARGS...] | check RUN")
