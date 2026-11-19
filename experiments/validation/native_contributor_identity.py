"""CPU-only bounded contributor evidence reader/join; no native observer or CLI.

Descriptors are an explicit proposed offline format, not an attested observer wire ABI.
Contributor keys encode (EP-group index, source token, global expert), never payloads.
"""
from array import array
from collections.abc import Sequence
import hashlib
import json
import mmap
from pathlib import Path
import struct

IMAGE = '910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490'
SOURCE = 'e242f3c3c1136aeae498529bfde5e06bda9a7cca'
MAX_TOKENS, MAX_ASSIGNMENTS = 32768, 262144
RANK_CAP, JOB_CAP = 192*2**20, 512*2**20
CHANNELS = tuple(range(0, 1792, 28))
BOUNDARIES = {'alltoall': ('source', 'expert_input', 'expert_output', 'returned', 'postcombine'),
              'flex': ('source', 'received', 'expert_input', 'expert_output', 'local', 'postcombine')}
DTYPES = {'i32': ('i', 4), 'u32': ('I', 4), 'i64': ('q', 8), 'u16': ('H', 2)}
NATIVE_OBSERVER_READY = False


def need(condition, message):
    """Reject incomplete or inconsistent evidence without guessing identities."""
    if not condition:
        raise ValueError(message)


def integer(value, lower, upper):
    """Check exact Python integer type and inclusive bounds (bool is not an integer)."""
    return type(value) is int and lower <= value <= upper


def key(source, token, expert):
    """Pack bounded contributor coordinates independently of payload values."""
    need(integer(source, 0, 7) and integer(token, 0, MAX_TOKENS-1) and integer(expert, 0, 255), 'contributor key')
    return ((source*MAX_TOKENS+token)*256)+expert


def coordinates(value):
    """Decode a packed contributor label into EP index, token and global expert."""
    need(integer(value, 0, 8*MAX_TOKENS*256-1), 'packed key')
    return value//(MAX_TOKENS*256), (value//256) % MAX_TOKENS, value % 256


class Row(Sequence):
    """Noncopying row of a typed array or virtual active-prefix TE map."""
    def __init__(self, size, get):
        self.size, self.get = size, get

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.size))]
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        return self.get(index)


class TypedArray(Sequence):
    """Read explicit LE words from a buffer, without native-endian casts or row lists."""
    def __init__(self, buffer, dtype, shape):
        need(dtype in DTYPES and len(shape) in (1, 2) and all(integer(n, 0, MAX_ASSIGNMENTS*513) for n in shape), 'array shape/dtype')
        self.shape, self.dtype = tuple(shape), dtype
        self.word = struct.Struct('<'+DTYPES[dtype][0])
        self.data = memoryview(buffer)
        need(self.data.nbytes == self.word.size*self.count, 'array byte count')
        self.backing = buffer
        self.rank = None

    @property
    def count(self):
        """Return the product of declared dimensions."""
        return self.shape[0]*(self.shape[1] if len(self.shape) == 2 else 1)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if not integer(index, 0, len(self)-1):
            raise IndexError(index)
        if len(self.shape) == 2:
            return Row(self.shape[1], lambda j: self.word.unpack_from(self.data, (index*self.shape[1]+j)*self.word.size)[0])
        return self.word.unpack_from(self.data, index*self.word.size)[0]

    def close(self):
        """Release the view and owned read-only file mapping after consumers finish."""
        self.data.release()
        if isinstance(self.backing, mmap.mmap):
            self.backing.close()


class ActiveMap(Sequence):
    """Expose compact count/destination/column arrays as TE's logical (2E+1) rows.

    Inactive tails are deliberately inaccessible; no uninitialized bytes are decoded.
    """
    def __init__(self, counts, destinations, columns, experts):
        need(integer(experts, 1, 256) and len(counts) <= MAX_ASSIGNMENTS, 'map geometry')
        self.counts, self.destinations, self.columns, self.experts = counts, destinations, columns, experts
        self.offsets = array('I', [0])
        for count in counts:
            need(integer(count, 0, min(8, experts)), 'map count')
            self.offsets.append(self.offsets[-1]+count)
        need(self.offsets[-1] <= MAX_ASSIGNMENTS and len(destinations) == len(columns) == self.offsets[-1], 'active map sizes')

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, index):
        if not integer(index, 0, len(self)-1):
            raise IndexError(index)
        n, e, start = self.counts[index], self.experts, self.offsets[index]
        def get(j):
            if j == 2*e:
                return n
            if j < n:
                return self.destinations[start+j]
            if e <= j < e+n:
                return self.columns[start+j-e]
            raise ValueError('inactive TE map tail')
        return Row(2*e+1, get)


def te_rows(inputs, columns, actual):
    """Validate expert-major/token-stable rows and descending-column active prefixes."""
    need(0 < len(columns) <= 256 and len(set(columns)) == len(columns) and all(integer(e, 0, 255) for e in columns), 'TE columns')
    need(len(actual) == len(inputs), 'map rows')
    lookup = {e: c for c, e in enumerate(columns)}
    counts = [0]*len(columns)
    for s, t, selected in inputs:
        need(len(selected) <= 8 and len(set(selected)) == len(selected), 'TE selected experts')
        for e in selected:
            need(e in lookup, 'TE expert ownership')
            counts[lookup[e]] += 1
    need(sum(counts) <= MAX_ASSIGNMENTS, 'TE assignment bound')
    starts, total = [], 0
    for count in counts:
        starts.append(total); total += count
    used = [0]*len(columns); labels = array('Q', [0])*total
    for i, (s, t, selected) in enumerate(inputs):
        row = actual[i]; e = len(columns)
        need(len(row) == 2*e+1 and integer(row[-1], 0, 8) and row[-1] == len(selected), 'TE map multiplicity')
        for j, c in enumerate(sorted((lookup[x] for x in selected), reverse=True)):
            position = starts[c]+used[c]; used[c] += 1
            need(type(row[j]) is int and type(row[e+j]) is int and row[j] == position and row[e+j] == c, 'TE active prefix/order')
            labels[position] = key(s, t, columns[c])
    return labels


def chunk_rows(sizes, order):
    """Expand supplied CPU chunk operands to output-row -> input-row permutation."""
    need(len(sizes) == len(order) <= 256 and all(integer(n, 0, MAX_ASSIGNMENTS) for n in sizes), 'chunk sizes')
    need(all(integer(i, 0, len(sizes)-1) for i in order) and sorted(order) == list(range(len(sizes))), 'chunk order')
    need(sum(sizes) <= MAX_ASSIGNMENTS, 'chunk bound')
    starts, total = [], 0
    for n in sizes:
        starts.append(total); total += n
    return array('I', (row for i in order for row in range(starts[i], starts[i]+sizes[i])))


def reorder(rows, permutation):
    """Apply a bounded complete output-to-input permutation without suppressing duplicates."""
    need(len(rows) == len(permutation) <= MAX_ASSIGNMENTS, 'permutation size')
    seen = bytearray(len(rows)); result = array('Q')
    for index in permutation:
        need(integer(index, 0, len(rows)-1) and not seen[index], 'row permutation')
        seen[index] = 1; result.append(rows[index])
    return result


def build(group, owners, columns, routes_a, routes_f, source_maps, received,
          flex_maps, transfer, forward, restore):
    """Join explicit routing/maps; alltoall slot order is NOT treated as FLEX slot order.

    routes_a contains canonical selected (global expert, raw FP32 word) pairs from
    dense routing membership; routes_f preserves actual source top-k slot order.
    received contains actual (local expert slot words, FP32 probability words).
    The transfer argument is explicit source TE row references, not opaque handles.
    """
    n = len(group)
    need(n == 8 and all(integer(g, 0, 2**31-1) for g in group) and len(set(group)) == n, 'group order')
    need(all(len(x) == n for x in (owners, routes_a, routes_f, source_maps, received, flex_maps, transfer, forward, restore)), 'rank counts')
    flat = [e for row in owners for e in row]
    need(len(flat) == len(set(flat)) and set(flat) == set(columns) and all(integer(e, 0, 255) for e in flat), 'ownership')
    need(all(0 < len(row) <= 32 for row in owners), 'local expert bound')
    owner = {e: d for d, row in enumerate(owners) for e in row}
    local = [{e: i for i, e in enumerate(row)} for row in owners]
    need(sum(map(len, routes_a)) <= MAX_TOKENS, 'token bound')
    need(sum(map(len, transfer)) <= MAX_ASSIGNMENTS, 'transfer bound')
    probabilities, source_rows = {}, []
    for s in range(n):
        need(len(routes_a[s]) == len(routes_f[s]), 'source token counts')
        for t, slots in enumerate(routes_a[s]):
            for values in (slots, routes_f[s][t]):
                need(0 < len(values) <= 8, 'topk count')
                experts = set()
                for e, p in values:
                    need(integer(e, 0, 255) and e in owner and e not in experts, 'duplicate/unknown expert')
                    need(integer(p, 0, 2**32-1) and p & 0x7f800000 != 0x7f800000, 'probability bits')
                    experts.add(e)
            need(dict(slots) == dict(routes_f[s][t]), 'cross-arm keyed probability bits')
            for e, p in slots:
                probabilities[key(s, t, e)] = p
        inputs = Row(len(routes_a[s]), lambda t, s=s: (s, t, [e for e, _ in routes_a[s][t]]))
        source_rows.append(te_rows(inputs, columns, source_maps[s]))
    need(len(probabilities) <= MAX_ASSIGNMENTS, 'assignment bound')
    a2a, flex, receive_labels = [], [], []
    for d in range(n):
        before = array('Q')
        for s, row in transfer[d]:
            need(integer(s, 0, n-1) and integer(row, 0, len(source_rows[s])-1), 'transfer reference')
            label = source_rows[s][row]
            need(owner[label % 256] == d, 'transfer ownership')
            before.append(label)
        wanted = {k for k in probabilities if owner[k % 256] == d}
        need(len(before) == len(wanted) and set(before) == wanted, 'transfer multiplicity')
        sorted_rows = reorder(before, forward[d])
        need(reorder(sorted_rows, restore[d]) == before, 'chunk inverse')
        a2a.append(sorted_rows)
        labels = array('Q')
        for s, tokens in enumerate(routes_f):
            for t, slots in enumerate(tokens):
                if not any(e in local[d] for e, _ in slots):
                    continue
                r = len(labels)
                need(r < len(received[d]), 'received missing row')
                indices, bits = received[d][r]
                need(len(indices) == len(bits) == len(slots), 'received slot shape')
                for j, (e, p) in enumerate(slots):
                    need(type(indices[j]) is int and type(bits[j]) is int and indices[j] == local[d].get(e, -1)
                         and bits[j] == (p if e in local[d] else 0), 'received slot order/bits')
                labels.append(key(s, t, 0))
        need(len(labels) == len(received[d]), 'received multiplicity')
        receive_labels.append(labels)
        def received_input(r):
            s, t, _ = coordinates(labels[r])
            return s, t, [e for e, _ in routes_f[s][t] if e in local[d]]
        rows = te_rows(Row(len(labels), received_input), owners[d], flex_maps[d])
        need(len(rows) == len(wanted) and set(rows) == wanted, 'flex multiplicity')
        flex.append(rows)
    positions = {label: (d, r) for d, rows in enumerate(flex) for r, label in enumerate(rows)}
    pairs = {label: ((d, r), positions[label]) for d, rows in enumerate(a2a) for r, label in enumerate(rows)}
    need(len(pairs) == len(probabilities) and pairs.keys() == probabilities.keys(), 'contributor closure')
    return dict(group=list(group), owners=[list(x) for x in owners], tokens=[len(x) for x in routes_a], a2a=a2a, flex=flex, received=receive_labels, source=source_rows,
                pairs=pairs, probabilities=probabilities)


def associate(group, owners, columns, routes_a, routes_f, source_maps, received,
              flex_maps, send, recv, forward_chunks, restore_chunks):
    """Derive TP1 source-EP ordered alltoall transfer ranges from checked split operands."""
    need(len(send) == len(recv) == 8 and all(len(row) == 8 for row in (*send, *recv)), 'split shape')
    need(all(integer(x, 0, MAX_ASSIGNMENTS) for row in (*send, *recv) for x in row), 'split words')
    need(all(send[s][d] == recv[d][s] for s in range(8) for d in range(8)), 'split transpose')
    need(sum(sum(row) for row in send) <= MAX_ASSIGNMENTS, 'split assignment bound')
    # Source TE global column order must group consecutive destination blocks.
    need([e for own in owners for e in sorted(own, key=columns.index)] == list(columns), 'source destination column order')
    transfers = [[] for _ in range(8)]
    for s in range(8):
        start = 0
        for d in range(8):
            transfers[d].extend((s, row) for row in range(start, start+send[s][d])); start += send[s][d]
        need(start == sum(len(slots) for slots in routes_a[s]), 'source split total')
    need(len(forward_chunks) == len(restore_chunks) == 8, 'chunk ranks')
    return build(group, owners, columns, routes_a, routes_f, source_maps, received, flex_maps, transfers,
                 [chunk_rows(*x) for x in forward_chunks], [chunk_rows(*x) for x in restore_chunks])


def check_input_bits(labels, probabilities, source_bits, actual_bits):
    """Check all64 raw BF16 gather-or-positive-zero channels, never infer identity by bits."""
    need(len(labels) == len(actual_bits) <= MAX_ASSIGNMENTS, 'payload rows')
    for label, actual in zip(labels, actual_bits):
        s, t, _ = coordinates(label); source = source_bits[s][t]
        need(len(source) == len(actual) == 64, 'fixed channel count')
        zero = probabilities[label] & 0x7fffffff == 0
        for a, b in zip(actual, source):
            need(integer(a, 0, 65535) and integer(b, 0, 65535), 'BF16 words')
            need(a == (0 if zero else b), 'gather/positive-zero bits')


def read_arrays(directory, descriptors, manifest_bytes=0):
    """Preflight complete whole-file descriptors/caps, then hash and map read-only arrays.

    Caller owns returned TypedArrays and must close them after offline analysis.
    Caps cover declared file bytes plus manifest bytes, not Python working memory.
    """
    need(integer(manifest_bytes, 0, JOB_CAP) and len(descriptors) <= 2048, 'descriptor bound')
    manifest_bytes += len(json.dumps(descriptors, sort_keys=True, allow_nan=False).encode())
    root = Path(directory).resolve(strict=True); sizes = [manifest_bytes]*8
    prepared, names, files, total = [], set(), set(), manifest_bytes
    for d in descriptors:
        need(set(d) == {'name', 'file', 'rank', 'dtype', 'shape', 'bytes', 'sha256'}, 'descriptor fields')
        name, filename, rank = d['name'], d['file'], d['rank']
        need(isinstance(name, str) and name and all(c.isascii() and (c.isalnum() or c in '_.-') for c in name) and name not in names and isinstance(filename, str)
             and filename not in ('', '.', '..') and Path(filename).name == filename and filename not in files, 'descriptor name/path')
        need(integer(rank, 0, 7) and d['dtype'] in DTYPES and isinstance(d['shape'], list) and len(d['shape']) in (1, 2), 'descriptor rank/type')
        need(all(integer(x, 0, MAX_ASSIGNMENTS*513) for x in d['shape']), 'descriptor shape')
        count = d['shape'][0]*(d['shape'][1] if len(d['shape']) == 2 else 1)
        need(type(d['bytes']) is int and d['bytes'] == count*DTYPES[d['dtype']][1], 'descriptor bytes')
        sha = d['sha256']; need(isinstance(sha, str) and len(sha) == 64 and not set(sha)-set('0123456789abcdef'), 'descriptor hash')
        path = root/filename
        need(not path.is_symlink() and path.resolve(strict=True).parent == root and path.is_file() and path.stat().st_size == d['bytes'], 'file bytes/path')
        sizes[rank] += d['bytes']; total += d['bytes']
        need(sizes[rank] <= RANK_CAP and total <= JOB_CAP, 'capture byte cap')
        names.add(name); files.add(filename); prepared.append((name, path, d))
    arrays = {}
    try:
        for name, path, d in prepared:
            with path.open('rb') as stream:
                h = hashlib.file_digest(stream, 'sha256').hexdigest()
                need(h == d['sha256'], 'file hash')
                backing = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) if d['bytes'] else b''
            arrays[name] = TypedArray(backing, d['dtype'], d['shape'])
            arrays[name].rank = d['rank']
        return arrays
    except BaseException:
        for value in arrays.values():
            value.close()
        raise


def strict_json_equal(actual, expected):
    """Compare JSON bindings without bool/int/float or tuple/list coercion."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (all(type(k) is str for k in actual) and actual.keys() == expected.keys()
                and all(strict_json_equal(actual[k], value) for k, value in expected.items()))
    if type(expected) is list:
        return len(actual) == len(expected) and all(strict_json_equal(a, b) for a, b in zip(actual, expected))
    return type(expected) in (str, int, float, bool, type(None)) and actual == expected


def validate_closure(frame, group, owners, entries, joined, arrays):
    """Validate fixed native frame and exactly88 boundary descriptors bound to joined rows."""
    need(set(frame) == {'image', 'source', 'tooling', 'layer', 'invocation', 'microbatch', 'phase', 'kind'}, 'frame fields')
    need(frame['image'] == IMAGE and frame['source'] == SOURCE and frame['phase'] == 'replay' and frame['kind'] == 'forward', 'frame identity')
    need(all(type(frame[k]) is int for k in ('layer', 'invocation', 'microbatch')) and (frame['layer'], frame['invocation'], frame['microbatch']) == (4, 0, 0), 'frame ordinal')
    sha = frame['tooling']; need(isinstance(sha, str) and len(sha) == 64 and not set(sha)-set('0123456789abcdef'), 'tooling hash')
    need(strict_json_equal(list(group), joined['group']) and strict_json_equal(owners, joined['owners']) and len(owners) == 8 and all(len(x) == 32 for x in owners)
         and sorted(e for row in owners for e in row) == list(range(256)), 'native ownership/group')
    need(joined['tokens'] == [4096]*8, 'native source token geometry')
    need(all(len(rows) == 32768 for rows in joined['source'])
         and len(joined['probabilities']) == MAX_ASSIGNMENTS, 'native top8 assignment closure')
    wanted = {(r, a, b) for r in range(8) for a, boundaries in BOUNDARIES.items() for b in boundaries}
    seen, names = set(), set()
    for entry in entries:
        need(set(entry) == {'rank', 'arm', 'boundary', 'array', 'frame', 'group', 'owners', 'channels'}, 'boundary fields')
        r, a, b = entry['rank'], entry['arm'], entry['boundary']; identity = r, a, b
        need(integer(r, 0, 7) and identity in wanted and identity not in seen, 'boundary rank/duplicate')
        need(strict_json_equal(entry['frame'], frame) and strict_json_equal(entry['group'], list(group))
             and strict_json_equal(entry['owners'], owners) and strict_json_equal(entry['channels'], list(CHANNELS)), 'boundary binding')
        name = entry['array']; need(name in arrays and name not in names, 'boundary array')
        value = arrays[name]
        if b in ('source', 'postcombine'):
            rows = joined['tokens'][r]
        elif b in ('received', 'local'):
            rows = len(joined['received'][r])
        elif b == 'returned':
            rows = len(joined['source'][r])
        else:
            rows = len(joined['a2a' if a == 'alltoall' else 'flex'][r])
        need(value.rank == r and value.dtype == 'u16' and value.shape == (rows, 64), 'boundary rank/shape/dtype')
        seen.add(identity); names.add(name)
    need(seen == wanted, 'partial boundary closure')
    return True


def parse_metadata(arrays, fields, group, owners, columns):
    """Join proposed explicit typed metadata names; producer hook wiring remains unverified.

    Each rank has exactly the named fields below; map arrays contain active prefixes
    only. Source alltoall experts/probabilities must be extracted from selected dense
    membership, not reconstructed by sorting FLEX slots or matching activation bits.
    """
    required = {'a_experts', 'a_probs', 'f_experts', 'f_probs', 'source_counts', 'source_rows', 'source_cols',
                'received_experts', 'received_probs', 'flex_counts', 'flex_rows', 'flex_cols',
                'send', 'recv', 'forward_sizes', 'forward_order', 'restore_sizes', 'restore_order'}
    need(len(fields) == len(owners) == 8 and list(columns) == list(range(256))
         and owners == [list(range(32*r, 32*(r+1))) for r in range(8)], 'native geometry')
    need(all(set(f) == required for f in fields), 'metadata fields')
    used = [name for f in fields for name in f.values()]
    need(len(used) == len(set(used)) and all(name in arrays for name in used), 'metadata arrays')
    typed = []
    for r, f in enumerate(fields):
        rank = {name: arrays[value] for name, value in f.items()}
        for name, value in rank.items():
            expected = ('u32',) if name.endswith('probs') else ('i32', 'i64') if name.endswith('experts') else ('i32',)
            need(value.rank == r and value.dtype in expected, 'metadata rank/dtype')
        for prefix in ('a', 'f'):
            need(rank[prefix+'_experts'].shape == rank[prefix+'_probs'].shape == (4096, 8), 'native source top8')
        nr = len(rank['received_experts'])
        need(nr <= MAX_TOKENS and rank['received_experts'].shape == rank['received_probs'].shape == (nr, 8), 'received typed shape')
        for prefix, count in (('source', 4096), ('flex', nr)):
            need(rank[prefix+'_counts'].shape == (count,) and len(rank[prefix+'_rows'].shape) == len(rank[prefix+'_cols'].shape) == 1, 'compact map dimensions')
        need(rank['send'].shape == rank['recv'].shape == (8,), 'typed splits')
        for prefix in ('forward', 'restore'):
            need(rank[prefix+'_sizes'].shape == rank[prefix+'_order'].shape == (256,), 'typed chunks')
        typed.append(rank)
    def routes(rank, prefix):
        experts, probs = rank[prefix+'_experts'], rank[prefix+'_probs']
        return Row(len(experts), lambda t: Row(8, lambda j: (experts[t][j], probs[t][j])))
    def compact(rank, prefix, experts):
        return ActiveMap(rank[prefix+'_counts'], rank[prefix+'_rows'], rank[prefix+'_cols'], experts)
    return associate(group, owners, columns, [routes(r, 'a') for r in typed], [routes(r, 'f') for r in typed],
                     [compact(r, 'source', 256) for r in typed],
                     [Row(len(r['received_experts']), lambda t, r=r: (r['received_experts'][t], r['received_probs'][t])) for r in typed],
                     [compact(r, 'flex', 32) for r in typed], [r['send'] for r in typed], [r['recv'] for r in typed],
                     [(r['forward_sizes'], r['forward_order']) for r in typed],
                     [(r['restore_sizes'], r['restore_order']) for r in typed])


def load_frozen_observer(path, expected_sha256):
    """Load only explicitly SHA-pinned offline helper code, never a native entry point."""
    import importlib.util
    path = Path(path)
    need(not path.is_symlink() and path.is_file() and 0 < path.stat().st_size <= 2**20, 'frozen observer size/origin')
    path = path.resolve(strict=True); data = path.read_bytes()
    need(len(data) == path.stat().st_size and type(expected_sha256) is str
         and hashlib.sha256(data).hexdigest() == expected_sha256, 'frozen observer hash')
    spec = importlib.util.spec_from_file_location('_pinned_contributor_offline_helpers', path)
    need(spec is not None and Path(spec.origin) == path, 'frozen observer import origin')
    module = importlib.util.module_from_spec(spec)
    # Execute verified bytes without creating a pycache in the authoritative capture.
    exec(compile(data, str(path), 'exec'), module.__dict__)
    need(Path(module.__file__) == path and module.IMPLEMENTATION_READY is False, 'frozen helper identity/readiness')
    return module


def read_capture_json(path, limit):
    """Read an immutable regular JSON object with bounded bytes and strict keys/constants."""
    path = Path(path)
    need(not path.is_symlink() and path.is_file() and path.stat().st_size <= limit, 'capture JSON path/size')
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'duplicate capture JSON key'); result[k] = v
        return result
    with path.open('rb') as stream:
        data = stream.read(limit+1)
    need(len(data) <= limit, 'capture JSON byte cap')
    value = json.loads(data, object_pairs_hook=pairs,
                       parse_constant=lambda s: need(False, 'nonfinite capture JSON'))
    need(type(value) is dict, 'capture JSON object')
    return value, data


def publish_capture_diagnostic(target, summary):
    """Publish a new <=1MiB external diagnostic atomically; never overwrite evidence."""
    import os
    import sys
    import tempfile
    data = json.dumps(summary, sort_keys=True, allow_nan=False).encode()
    need(len(data) <= 2**20, 'global diagnostic byte reserve')
    fd, temporary = tempfile.mkstemp(prefix='.closure-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        primary = sys.exception()
        try: Path(temporary).unlink()
        except OSError as error:
            if primary is None: raise
            primary.add_note(f'secondary diagnostic cleanup: {error}')


def check_staged_capture(root, *, frozen_observer, expected_observer_sha256, expected_common, target=None):
    """Verify actual eight-rank staged capture; return DIAGNOSTIC closure, not numerics.

    All inputs must stay immutable throughout this offline host check. The frozen
    observer's helper imports require the pinned host tooling; no vendor is imported.
    Optional target is a NEW file outside root, including on rejection.
    """
    root = Path(root)
    need(not root.is_symlink() and root.is_dir(), 'capture root')
    root = root.resolve(strict=True)
    if target is not None:
        target = Path(target).absolute()
        need(not target.resolve().is_relative_to(root) and not target.exists() and not target.is_symlink(), 'new external diagnostic target required')
    arrays = {}
    try:
        observer = load_frozen_observer(frozen_observer, expected_observer_sha256)
        need(expected_common['validation_sha256'] == expected_observer_sha256, 'expected frozen validation binding')
        namespace = 'quota-'+hashlib.sha256(observer.preflight_json(expected_common)).hexdigest()
        quota = root/namespace
        expected_quota_files = {f'proposal-{r}.json' for r in range(8)} | {'decision.json'}
        need(not quota.is_symlink() and quota.is_dir() and {p.name for p in quota.iterdir()} == expected_quota_files, 'incomplete/stale/error quota namespace')
        need(all(not (quota/name).is_symlink() and (quota/name).is_file() for name in expected_quota_files), 'quota file origin')
        need({p.name for p in root.iterdir() if p.name.startswith('quota-')} == {namespace}, 'stale/mixed quota namespace')
        proposals = [observer.preflight_read(quota/f'proposal-{r}.json') for r in range(8)]
        need(all(type(p['rank']) is int and p['rank'] == r for r, p in enumerate(proposals)), 'proposal file rank')
        decision = observer.strict_allocate_quotas(proposals, expected_common)
        need(strict_json_equal(observer.preflight_read(quota/'decision.json'), decision), 'quota decision/hash mismatch')
        indices, ranks, fields, boundaries, totals, index_hashes = [], [], [], [], [], []
        need({p.name for p in root.iterdir() if p.name.startswith('contributors-rank-')} == {f'contributors-rank-{r}' for r in range(8)}, 'rank-stage set')
        frozen = None; inode_roles = {}
        metadata_names = {'a_experts','a_probs','f_experts','f_probs','source_counts','source_rows','source_cols',
                          'received_experts','received_probs','flex_counts','flex_rows','flex_cols','send','recv',
                          'forward_sizes','forward_order','restore_sizes','restore_order'}
        required_index = {'status','rank_serialized','implementation_ready','rank','frame','binding','binding_sha256',
                          'descriptors','fields','boundaries','job_quotas','stage_quotas','data_bytes','pending',
                          'quota_approval','full_job_quotas','job_id'}
        for r in range(8):
            directory = root/f'contributors-rank-{r}'
            need(not directory.is_symlink() and directory.is_dir(), 'rank directory origin')
            index, raw = read_capture_json(directory/'rank-index.json', 2**20)
            need(set(index) == required_index and index['status'] == 'INCOMPLETE_JOB' and index['rank_serialized'] is True
                 and index['implementation_ready'] is False and type(index['rank']) is int and index['rank'] == r, 'rank pending/status schema')
            need(index['pending']=='all8-rank file/cap validation,88-boundary closure,identity join and expert probability association', 'rank pending marker')
            need(strict_json_equal(index['frame'], expected_common['frame']) and index['job_id'] == expected_common['job_id'], 'rank frame/job binding')
            for name, expected in (('job_quotas',decision['stage_quotas']),('stage_quotas',decision['stage_quotas']),('full_job_quotas',decision['job_quotas']),
                                   ('quota_approval',dict(proposal=proposals[r],decision=decision))):
                need(strict_json_equal(index[name], expected), 'rank quota approval binding')
            binding = index['binding']; need(type(binding) is dict and set(binding) == {'group','owners','source_columns','local_experts','frozen'}, 'rank binding schema')
            current = binding['frozen']
            need(type(current) is dict and set(current) == {'validation_sha256','source_manifest_sha256','tooling','source_files'}, 'frozen binding fields')
            need(current['validation_sha256'] == expected_observer_sha256 and current['source_manifest_sha256'] == expected_common['source_manifest_sha256']
                 and hashlib.sha256(json.dumps(current['tooling'],sort_keys=True,allow_nan=False).encode()).hexdigest() == expected_common['tooling_sha256'], 'frozen source/tooling binding')
            need(type(current['source_files']) is dict and current['source_files'] and all(type(k) is str and type(v) is str and len(v)==64 and not set(v)-set('0123456789abcdef') for k,v in current['source_files'].items()), 'frozen source files')
            if frozen is None: frozen = current
            need(strict_json_equal(current,frozen), 'cross-rank frozen files/tooling')
            for name, expected in (('group',expected_common['group']),('owners',expected_common['owners']),('source_columns',list(range(256))),('local_experts',list(range(r*32,r*32+32)))):
                need(strict_json_equal(binding[name],expected), 'rank group/owner/column binding')
            need(index['binding_sha256'] == hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest(), 'rank binding hash')
            descriptors = index['descriptors']; need(type(descriptors) is list and len(descriptors)==33, 'rank descriptor count')
            plans = {d['name']:d for d in proposals[r]['arrays']}; seen = set(); data_bytes = 0
            for d in descriptors:
                need(type(d) is dict and set(d)=={'name','file','rank','dtype','shape','bytes','sha256'} and type(d['name']) is str, 'array descriptor schema')
                role = d['name'].removeprefix(f'r{r}.')
                need(role in plans and role not in seen and d['name']==f'r{r}.{role}' and d['file']==f'r{r}.{role}.bin', 'array role/name/alias')
                plan = plans[role]
                need(strict_json_equal({k:d[k] for k in ('dtype','shape','bytes')},{k:plan[k] for k in ('dtype','shape','bytes')}) and type(d['rank']) is int and d['rank']==r, 'descriptor planned bytes/rank')
                need(type(d['sha256']) is str and len(d['sha256'])==64 and not set(d['sha256'])-set('0123456789abcdef'), 'descriptor SHA256')
                path = directory/d['file']
                need(not path.is_symlink() and path.is_file() and path.stat().st_size==d['bytes'], 'payload file/size')
                stat = path.stat(); inode = stat.st_dev,stat.st_ino
                need(inode not in inode_roles or inode_roles[inode]==role, 'aliased physical roles')
                inode_roles[inode]=role; seen.add(role); data_bytes += d['bytes']
            need({p.name for p in directory.iterdir()} == {d['file'] for d in descriptors}|{'rank-index.json'}, 'unexpected/missing rank objects')
            need(type(index['data_bytes']) is int and data_bytes==index['data_bytes']==proposals[r]['data_bytes'], 'actual rank byte plan')
            need(strict_json_equal(index['fields'],{name:f'r{r}.{name}' for name in metadata_names}), 'metadata role association')
            need(type(index['boundaries']) is list and len(index['boundaries'])==11, 'local boundary count')
            expected_boundaries = {(arm,b) for arm,bs in BOUNDARIES.items() for b in bs}; actual_boundaries = set()
            for entry in index['boundaries']:
                identity = entry['arm'],entry['boundary']
                need(identity in expected_boundaries and identity not in actual_boundaries, 'local boundary identity')
                arm,b = identity
                need(strict_json_equal(entry,dict(rank=r,arm=arm,boundary=b,array=f'r{r}.{arm}_{b}',frame=expected_common['frame'],
                                                 group=expected_common['group'],owners=expected_common['owners'],channels=list(CHANNELS))), 'boundary frozen binding')
                actual_boundaries.add(identity)
            charge = len(raw)+len(json.dumps(descriptors,sort_keys=True).encode())
            need(charge<=2**20 and data_bytes+charge<=decision['stage_quotas'][r], 'actual rank index/stage cap')
            total = data_bytes+charge+(quota/f'proposal-{r}.json').stat().st_size
            if r==0: total += (quota/'decision.json').stat().st_size+2**20
            need(total<=decision['job_quotas'][r] and total<=RANK_CAP, 'actual rank total cap')
            totals.append(total); indices.append(index); ranks.append(directory); fields.append(index['fields']); boundaries.extend(index['boundaries'])
            index_hashes.append(hashlib.sha256(raw).hexdigest())
        need(sum(totals)<=JOB_CAP, 'aggregate actual capture cap')
        for directory,index in zip(ranks,indices):
            loaded = read_arrays(directory,index['descriptors'],manifest_bytes=(directory/'rank-index.json').stat().st_size)
            arrays.update(loaded)
        # Recheck finite words independently, even if someone recomputed a descriptor hash.
        for value in arrays.values():
            if value.dtype not in ('u16','u32'): continue
            mask = 0x7f80 if value.dtype=='u16' else 0x7f800000
            words = struct.iter_unpack('<H' if value.dtype=='u16' else '<I',value.data)
            try:
                for (word,) in words: need(word & mask != mask, 'nonfinite captured raw words')
            finally: del words  # Release exported buffers before any mmap teardown, including errors.
        joined = parse_metadata(arrays,fields,expected_common['group'],expected_common['owners'],list(range(256)))
        validate_closure(expected_common['frame'],expected_common['group'],expected_common['owners'],boundaries,joined,arrays)
        for arm,label_name in (('alltoall','a2a'),('flex','flex')):
            source_bits = [arrays[f'r{r}.{arm}_source'] for r in range(8)]
            for r,labels in enumerate(joined[label_name]):
                check_input_bits(labels,joined['probabilities'],source_bits,arrays[f'r{r}.{arm}_expert_input'])
                probabilities=arrays[f'r{r}.{arm}_expert_probs']; counts=arrays[f'r{r}.{arm}_expert_counts']; actual=[0]*32
                need(len(probabilities)==len(labels) and counts.shape==(32,), 'auxiliary probability/count shape')
                for row,label in enumerate(labels):
                    need(probabilities[row]==joined['probabilities'][label], 'expert probability bits by joined label')
                    actual[expected_common['owners'][r].index(label % 256)]+=1
                need(list(counts)==actual, 'expert counts by joined labels')
        summary=dict(status='CAPTURE_CLOSED_DIAGNOSTIC',ranks=8,boundaries=88,tokens=MAX_TOKENS,contributors=len(joined['pairs']),
                     logical_accounted_bytes=sum(totals),rank_accounted_bytes=totals,index_sha256=index_hashes,
                     observer_sha256=expected_observer_sha256,common=expected_common,native_numerics_accepted=False,
                     full1792_equivalence=False,performance_claim=False)
        if target is not None: publish_capture_diagnostic(target,summary)
        return summary
    except BaseException as error:
        if target is not None and not target.exists():
            try: publish_capture_diagnostic(target,dict(status='CAPTURE_REJECTED',error=ascii(error)[:512],native_numerics_accepted=False))
            except Exception as secondary: error.add_note(f'secondary failure evidence: {secondary}')
        raise
    finally:
        for value in arrays.values(): value.close()
