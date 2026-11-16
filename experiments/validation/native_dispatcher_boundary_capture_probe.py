"""Capture-only canonical projection boundaries around the unchanged EP8 gate.

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

BOUNDARY_TOKENS = tuple(range(0, 4096, 64))
BOUNDARY_BYTES = 64 * 1024 * 1024
BOUNDARY_METADATA_BYTES = 1024 * 1024
BOUNDARY_COUNT = 48
BOUNDARY_HANDLES = 24
PROJECTIONS = {'fc1_latent_proj': (3584, 1792), 'fc2_latent_proj': (1792, 3584)}


def boundary_keys():
    """Enumerate expected source-oriented snapshots, including replay-only dgrad."""
    return {f'{mode}/mb{mb}/{phase}/{name}/{side}/{kind}'
            for mode in ('alltoall', 'flex') for mb in range(2)
            for phase in ('original', 'replay') for name in PROJECTIONS
            for side in ('input', 'output') for kind in ('forward', 'dgrad')
            if kind == 'forward' or phase == 'replay'}


def verify_boundary_artifacts(output, rank, summary):
    """Verify diagnostic bits/index independently; never rewrite native acceptance."""
    index = output / f'boundary-{rank}.json'
    if index.stat().st_size > BOUNDARY_METADATA_BYTES:
        raise ValueError('boundary metadata budget exceeded')
    value = json.loads(index.read_text())
    if summary.get('status') != 'complete' or sha256(index) != summary.get('index_sha256') or value['status'] != 'complete':
        raise ValueError('incomplete boundary capture')
    entries = value['entries']
    keys = [entry['key'] for entry in entries]
    if len(keys) != BOUNDARY_COUNT or set(keys) != boundary_keys() or len(set(keys)) != len(keys) or \
            value['rank'] != rank or value['tokens'] != list(BOUNDARY_TOKENS) or value['errors']:
        raise ValueError('boundary coverage/identity differs')
    payload = output / f'boundary-{rank}.bin'
    if payload.stat().st_size + index.stat().st_size > BOUNDARY_BYTES or index.stat().st_size > BOUNDARY_METADATA_BYTES:
        raise ValueError('boundary byte budget exceeded')
    offset = 0
    with payload.open('rb') as stream:
        for entry in entries:
            width = PROJECTIONS[entry['projection']][entry['side'] == 'output']
            element_size = {'torch.bfloat16': 2, 'torch.float32': 4}.get(entry['dtype'])
            key = f"{entry['mode']}/mb{entry['microbatch']}/{entry['phase']}/{entry['projection']}/{entry['side']}/{entry['kind']}"
            if entry['shape'] != [4096, 1, width] or entry['captured_shape'] != [64, 1, width] or \
                    entry['rank'] != rank or element_size is None or entry['bytes'] != 64 * width * element_size or \
                    key != entry['key'] or entry['enabled'] != (entry['phase'] == 'replay'):
                raise ValueError('boundary canonical row/shape/dtype identity differs')
            if entry['offset'] != offset:
                raise ValueError('boundary offsets differ')
            raw = stream.read(entry['bytes'])
            if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('boundary payload integrity failure')
            offset += len(raw)
        if stream.read(1):
            raise ValueError('unexpected boundary payload tail')


class BoundaryCapture:
    """Own bounded snapshots; callbacks return None and never perform CPU copies/I/O.

    Native exceptions and numerical rejection take priority over observer failures.
    A failed native API path never invokes quiescent serialization.
    """

    def __init__(self, torch, output, rank):
        self.torch, self.output, self.rank = torch, output, rank
        self.entries, self.owned, self.handles, self.errors = [], [], [], []
        self.frame, self.seen, self.bytes = None, set(), 0
        self.serialized, self.native_exception, self.summary = False, None, {'status': 'partial'}

    def error(self, error):
        """Bound error metadata without raising into native computation."""
        if len(self.errors) < 32:
            self.errors.append(f'{type(error).__name__}: {error}'[:512])

    def snapshot(self, tensor, name, side, phase, kind, frame):
        """Copy actual canonical rows on their current producer stream only."""
        try:
            key = f'{frame[0]}/mb{frame[1]}/{phase}/{name}/{side}/{kind}'
            width = PROJECTIONS[name][side == 'output']
            if key not in boundary_keys() or key in self.seen or len(self.entries) >= BOUNDARY_COUNT:
                raise ValueError('unexpected/duplicate boundary event')
            if not isinstance(tensor, self.torch.Tensor) or tuple(tensor.shape) != (4096, 1, width) or not tensor.is_cuda:
                raise ValueError('noncanonical projection shape/device')
            if tensor.dtype not in (self.torch.bfloat16, self.torch.float32):
                raise ValueError('unsupported boundary bit dtype')
            size = 64 * width * tensor.element_size()
            if self.bytes + size + BOUNDARY_METADATA_BYTES > BOUNDARY_BYTES:
                raise ValueError('boundary byte budget exceeded')
            entry = dict(key=key, rank=self.rank, mode=frame[0], microbatch=frame[1], phase=phase,
                         enabled=phase == 'replay', projection=name, side=side, kind=kind,
                         requires_grad=bool(tensor.requires_grad),
                         dtype=str(tensor.dtype), shape=list(tensor.shape), stride=list(tensor.stride()),
                         storage_offset=int(tensor.storage_offset()), captured_shape=[64, 1, width], bytes=size)
            # Clone the canonical strided token slice on the current producer
            # stream; no shared device-index tensor or native view is retained.
            owned = tensor.detach()[::64].clone()
            self.entries.append(entry)
            self.owned.append(owned)
            self.seen.add(key)
            self.bytes += size
            return True
        except Exception as error:
            self.error(error)
            return False

    def observe_tensor(self, tensor, name, side):
        """Observe forward and replay dgrad, never substitute a gradient/tensor."""
        frame = self.frame
        if frame is None:
            raise ValueError('projection hook outside labelled native arm')
        enabled = bool(self.torch.is_grad_enabled())
        phase = 'replay' if enabled else 'original'
        captured = self.snapshot(tensor, name, side, phase, 'forward', frame)
        if captured and enabled and tensor.requires_grad:
            if len(self.handles) >= BOUNDARY_HANDLES:
                self.error(ValueError('boundary hook count exceeded'))
                return
            def gradient(grad):
                self.snapshot(grad, name, side, phase, 'dgrad', frame)
                return None
            self.handles.append(tensor.register_hook(gradient))

    def attach(self, models):
        """Install only public Module/tensor hooks; preserve native tuple outputs."""
        for mode, model in models.items():
            for name in PROJECTIONS:
                try:
                    self.attach_projection(getattr(model, name), mode, name)
                except Exception as error:
                    self.error(error)

    def attach_projection(self, module, mode, name):
        """Register one projection's hooks; caller retains partial handles on error."""
        def before(module, inputs):
            try:
                if self.frame is None or self.frame[0] != mode or len(inputs) != 1:
                    raise ValueError('projection input/call label differs')
                self.observe_tensor(inputs[0], name, 'input')
            except Exception as error:
                self.error(error)
            return None
        def after(module, inputs, result):
            try:
                if not isinstance(result, tuple) or len(result) != 2 or result[1] is not None:
                    raise ValueError('projection must return native (Tensor, None)')
                self.observe_tensor(result[0], name, 'output')
            except Exception as error:
                self.error(error)
            return None
        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after))

    @contextmanager
    def arm(self, mode, microbatch):
        """Supply immutable rank/token-independent call labels around native work."""
        previous = self.frame
        self.frame = (mode, microbatch)
        try:
            yield
        finally:
            self.frame = previous

    def serialize_quiescent(self):
        """Serialize bits only at the existing all-backwards-complete GPU barrier."""
        try:
            offset = 0
            with (self.output / f'boundary-{self.rank}.bin').open('xb') as stream:
                for entry, owned in zip(self.entries, self.owned):
                    raw = owned.view(self.torch.uint8).cpu().numpy().tobytes()
                    if len(raw) != entry['bytes']:
                        raise ValueError('boundary byte count differs')
                    entry.update(offset=offset, sha256=hashlib.sha256(raw).hexdigest())
                    stream.write(raw)
                    offset += len(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self.serialized = True
        except Exception as error:
            self.error(error)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.native_exception = f'{kind.__name__}: {error}'[:512]
        for handle in reversed(self.handles):
            try:
                handle.remove()
            except Exception as problem:
                self.error(problem)
        complete = self.serialized and not self.errors and self.seen == boundary_keys() and error is None
        status = 'complete' if complete else 'partial'
        self.summary = dict(status=status, entries=len(self.entries), bytes=self.bytes, errors=self.errors,
                            native_exception=self.native_exception, scope='canonical boundary diagnostic only')
        try:
            value = dict(self.summary, rank=self.rank, tokens=list(BOUNDARY_TOKENS), entries=self.entries,
                         byte_order=sys.byteorder)
            if len(json.dumps(value).encode()) > BOUNDARY_METADATA_BYTES:
                raise ValueError('boundary metadata budget exceeded')
            path = self.output / f'boundary-{self.rank}.json'
            if path.exists():
                raise FileExistsError('refusing to replace existing boundary evidence')
            write_json(path, value)
            self.summary['index_sha256'] = sha256(path)
        except Exception as problem:
            self.error(problem)
            self.summary['status'] = 'partial'
        self.owned.clear()
        return False

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
        if probs.requires_grad:
            probs.register_hook(lambda grad: row.update(gradient=grad.detach().clone()))

    def wrap(name):
        original = getattr(dispatcher, name)
        originals[name] = original
        def call(*args, **kwargs):
            flags = effective_flags(original, args, kwargs) if mode == 'flex' and name in {'token_dispatch', 'token_combine'} else {}
            result = original(*args, **kwargs)
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
            elif mode == "flex":
                events.append({"kind": name, "enabled": torch.is_grad_enabled(),
                               **flags,
                               "grad_fn": type(result.grad_fn).__name__})
            return result
        setattr(dispatcher, name, call)
    try:
        hook = model.router.register_forward_hook(router_hook)
        for name in ("token_dispatch", "dispatch_postprocess", "token_combine"):
            wrap(name)
        yield
    finally:
        if hook is not None:
            hook.remove()
        for name in originals:
            delattr(dispatcher, name)  # Methods were inherited, not instance overrides.


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
            capture = BoundaryCapture(torch, output, rank)
            try:
                with capture:
                    run_probe(emit, record, source, manifest["files"], capture=capture)
            finally:
                record['boundary_capture'] = capture.summary
            validate_record(record, rank)
            verify_boundary_artifacts(output, rank, record['boundary_capture'])
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
            verify_boundary_artifacts(run / 'output', rank, record['boundary_capture'])
            if record["status"] != "passed" or json.loads((run / f"output/imports-{rank}.json").read_text())["verified_modules"] <= 0:
                raise ValueError("rank failed/import evidence missing")
        result = dict(status="passed", scope=SCOPE, ranks=8, source=SOURCE, image=IMAGE)
    except Exception as error:
        write_json(run / "output/checks.json", dict(status="failed", scope=SCOPE, error=str(error)))
        raise
    write_json(run / "output/checks.json", result)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "train":
        train(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif len(sys.argv) >= 5 and sys.argv[1] == 'gpu':
        gpu(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        check(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: native_dispatcher_probe.py train OUTPUT SOURCE ENTRYPOINT [ARGS...] | check RUN")
