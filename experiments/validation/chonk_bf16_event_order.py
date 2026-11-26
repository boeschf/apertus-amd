"""Compare native current-stream event capture before and after a gated BF16 copy."""

import ctypes as C
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import sys
from time import monotonic, sleep

from apertus_amd.artifacts import sha256, write_json


IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")
UTILS = PACKAGES / "deep_ep/utils.py"
UTILS_SHA = "30f49700526437387571671ca75c99c3d2811f0f45e1bc6fd1cb164ca3f90bf9"
BUFFER_SHA = "d30ad5231f476c726c246240eec9b6a2f9124e16752900ca562653623e77a768"
EXTENSION_SHA = "fb347417de0ebbf3b0e87a514fa7896f9dd2eaa23b72fb7ba7f86cd7fed18d89"
HIP_SONAME = "libamdhip64.so.7"
VENDOR = ("2.14.0+rocm10.1.0", "7.16.26385")
SHAPE = (8192, 3584)
ARMS = ("before-copy", "after-copy", "contiguous-control")
READY_SECONDS, OBSERVE_SECONDS, DRAIN_SECONDS = 5, 1, 10
WAIT_VALUE_ATTRIBUTE, SIGNAL_MEMORY, WAIT_EQ = 10013, 2, 1
EP, LOCAL_EXPERTS, TOPK, TOKENS, WIDTH = 16, 16, 8, 128, 1792
NATIVE_TRANSPORT = dict(UCCL_EP_TRANSPORT="cxi", UCCL_SOCKET_IFNAME="hsn", UCCL_CXI_THREADING="safe",
                        NCCL_P2P_DISABLE="0")


def verify_launch(settings, env):
    """Require the reviewed cached-image, one-rank Slurm command."""
    resources = dict(nodes=1, gpus_per_node=1, cpus_per_task=48, cpu_bind="cores", account="csstaff",
                     partition="mi300", reservation="cscs", memory_gb=0, time="00:10:00",
                     exclude="nid002450", mpi="pmix", distribution="block:block",
                     network="disable_rdzv_get", uenv_passthrough="ignore")
    if (settings["image"]["sha256"] != IMAGE or not settings.get("registry_image_path")
            or settings["procedure"]["kind"] != "commands" or settings["procedure"]["timeout"] != 300
            or settings["procedure"]["commands"] != [["python3", "-u", "/apertus/validation.py", "run"]]
            or not settings["launch"]["submit"] or any(settings["resources"].get(k) != v for k, v in resources.items())
            or any(env.get(k) != v for k, v in dict(SLURM_PROCID="0", SLURM_LOCALID="0",
                                                    SLURM_NTASKS="1", SLURM_NODEID="0").items())):
        raise ValueError("unexpected BF16 event-order launch")


def verify_ep16_launch(settings, env):
    """Require the reviewed native CXI allocation and block rank placement."""
    resources = dict(nodes=4, gpus_per_node=4, cpus_per_task=48, cpu_bind="map_ldom:0,1,2,3", account="csstaff",
                     partition="mi300", reservation="cscs", memory_gb=0, time="00:20:00",
                     exclude="nid002450", mpi="pmix", distribution="block:block",
                     network="disable_rdzv_get", uenv_passthrough="ignore")
    rank, local, node = (int(env[k]) for k in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID"))
    if (settings["image"]["sha256"] != IMAGE or not settings.get("registry_image_path")
            or settings["procedure"]["kind"] != "commands" or settings["procedure"]["timeout"] != 600
            or settings["procedure"]["commands"] != [["python3", "-u", "/apertus/validation.py", "ep16"]]
            or not settings["launch"]["submit"] or any(settings["resources"].get(k) != v for k, v in resources.items())
            or not 0 <= rank < EP or (node, local) != divmod(rank, 4) or env["SLURM_NTASKS"] != str(EP)
            or any(env.get(k) != v for k, v in NATIVE_TRANSPORT.items())
            or any(k in env for k in ("UCCL_EP_DISPATCH_CONFIG", "UCCL_EP_COMBINE_CONFIG", "EFA_HOME"))):
        raise ValueError("unexpected native EP16 BF16 event-order launch")
    return rank, local


class HipSignal:
    """Own an eight-byte HIP signal used only by three reviewed streams."""

    def __init__(self, device=0):
        # Bind to the HIP instance already loaded by Torch; do not open a second copy.
        self.lib = C.CDLL(HIP_SONAME, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        ptr, i32, u32, u64 = C.c_void_p, C.c_int, C.c_uint, C.c_uint64
        for name, args in {
            "hipDeviceGetAttribute": (C.POINTER(i32), i32, i32),
            "hipExtMallocWithFlags": (C.POINTER(ptr), C.c_size_t, u32),
            "hipStreamWriteValue64": (ptr, ptr, u64, u32),
            "hipStreamWaitValue64": (ptr, ptr, u64, u32, u64),
            "hipFree": (ptr,),
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, i32
        supported = i32()
        self.call("hipDeviceGetAttribute", C.byref(supported), WAIT_VALUE_ATTRIBUTE, device)
        if supported.value != 1:
            raise RuntimeError("HIP stream wait-value is not supported; no substitute gate")
        self.ptr = ptr()
        self.call("hipExtMallocWithFlags", C.byref(self.ptr), 8, SIGNAL_MEMORY)
        if not self.ptr.value:
            raise RuntimeError("HIP signal allocation returned a null pointer")

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} returned HIP status {status}")

    def hold(self, stream):
        self.call("hipStreamWaitValue64", stream.cuda_stream, self.ptr, 1, WAIT_EQ, 2**64 - 1)

    def write(self, stream, value):
        self.call("hipStreamWriteValue64", stream.cuda_stream, self.ptr, value, 0)

    def close(self):
        self.call("hipFree", self.ptr)


def await_events(events, seconds, boundary):
    """Poll for completion with a bound; never synchronize a stream or device."""
    events = tuple(events)
    deadline = monotonic() + seconds
    while not all(event.query() for event in events):
        if monotonic() >= deadline:
            raise TimeoutError(f"GPU markers did not complete: {boundary}")
        sleep(0.005)


def verify_held(arm, samples):
    """Accept only an independently progressing consumer and a demonstrably held producer."""
    if not samples or any(s["producer_done"] or not s["consumer_prefix"] for s in samples):
        raise ValueError(f"{arm}: gate did not hold the producer or independent consumer progress was absent")
    dependent = [s["consumer_done"] for s in samples]
    if (arm == "after-copy" and any(dependent)) or (arm != "after-copy" and not dependent[-1]):
        raise ValueError(f"{arm}: consumer readiness is inconsistent with its capture point")


def run_arm(torch, utils, gate, streams, arm, source, golden, record):
    """Run one capture arm and release its gate even if observation fails."""
    producer, consumer, release = streams
    control = arm == "contiguous-control"
    if source.is_contiguous() != control or source.dtype != torch.bfloat16 or tuple(source.shape) != SHAPE:
        raise ValueError("incorrect input layout for the BF16 copy probe")
    events = {k: torch.cuda.Event(enable_timing=False) for k in
              ("producer_done", "consumer_prefix", "consumer_done", "release_done")}
    # Prime lazy event handles and the producer-stream copy allocation before holding that stream.
    for event, stream in zip(events.values(), (producer, consumer, consumer, release)):
        event.record(stream)
    with torch.cuda.stream(producer):
        warm = source.contiguous()
    events["producer_done"].record(producer)
    await_events(events.values(), READY_SECONDS, "setup")
    del warm
    gate.write(release, 0)
    events["release_done"].record(release)
    await_events((events["release_done"],), READY_SECONDS, "signal reset")
    held = False
    record["samples"] = []
    try:
        with torch.cuda.stream(producer):
            if arm == "before-copy":
                ready = utils.EventOverlap(utils.EventHandle())
            if not control:
                held = True
                gate.hold(producer)
            payload = source.contiguous()
            if arm != "before-copy":
                ready = utils.EventOverlap(utils.EventHandle())
            if control:
                # Hold independent successor work, not the already-ready no-copy payload.
                held = True
                gate.hold(producer)
            events["producer_done"].record(producer)
        record["copy"] = payload.data_ptr() != source.data_ptr()
        if record["copy"] == control or not payload.is_contiguous():
            raise ValueError("unexpected BF16 materialization")
        events["consumer_prefix"].record(consumer)
        with torch.cuda.stream(consumer):
            ready.current_stream_wait()
            events["consumer_done"].record(consumer)
        await_events((events["consumer_prefix"],), READY_SECONDS, "independent consumer prefix")
        start = monotonic()
        while True:
            record["samples"].append({k: events[k].query() for k in
                                      ("consumer_done", "producer_done", "consumer_prefix")})
            if monotonic() - start >= OBSERVE_SECONDS:
                break
            sleep(0.05)
        verify_held(arm, record["samples"])
    finally:
        # No freeing the signal or inspecting the payload until the gate and all submitted work drain.
        if held:
            gate.write(release, 1)
            for name, stream in zip(("release_done", "producer_done", "consumer_done"),
                                    (release, producer, consumer)):
                events[name].record(stream)
            await_events((events[k] for k in ("release_done", "producer_done", "consumer_done")),
                         DRAIN_SECONDS, "gate release and drain")
            record["drained"] = True
    record["exact_bf16"] = bool(torch.equal(payload.cpu(), golden))
    if not record["exact_bf16"]:
        raise ValueError("BF16 tensor did not match the exact reference after the gate drained")


def run():
    """Persist per-arm observations and fail closed without training or publishing artifacts."""
    from apertus_amd.runtime import preserve_affinity, require_affinity

    evidence = dict(status="failed", image=IMAGE, hostname=socket.gethostname(), arms={})
    report = Path("bf16-event-order.json")
    try:
        with preserve_affinity(evidence.setdefault("affinity", {})) as affinity:
            import torch

            if (torch.__version__, torch.version.hip) != VENDOR or torch.cuda.device_count() != 1:
                raise ValueError("expected the pinned Torch/HIP and one visible GPU")
            torch.cuda.set_device(0)
            evidence["target"] = torch.cuda.get_device_properties(0).gcnArchName
            if evidence["target"].split(":")[0] != "gfx942":
                raise ValueError("expected gfx942")
            import deep_ep.utils as utils

            if Path(utils.__file__).resolve() != UTILS or sha256(UTILS) != UTILS_SHA:
                raise ValueError("unexpected deep_ep.utils source")
        require_affinity(affinity, "BF16 event-order component")
        if len(affinity) != 48:
            raise ValueError("expected 48 bound CPUs")
        evidence.update(torch=torch.__version__, hip=torch.version.hip, utils_sha256=UTILS_SHA, shape=SHAPE)
        streams = tuple(torch.cuda.Stream() for _ in range(3))
        evidence["streams"] = [s.cuda_stream for s in streams]
        if len(set(evidence["streams"])) != 3 or 0 in evidence["streams"]:
            raise ValueError("expected distinct nondefault producer, consumer and release streams")
        word = torch.arange(SHAPE[0] * SHAPE[1], dtype=torch.int32).remainder_(257).sub_(128)
        transposed = word.reshape(SHAPE[::-1]).to(torch.bfloat16)
        golden = transposed.T.contiguous()
        strided = transposed.to("cuda").T
        dense = golden.to("cuda")
        gate = HipSignal()
        for arm in ARMS:
            record = evidence["arms"][arm] = {}
            write_json(report, evidence)
            run_arm(torch, utils, gate, streams, arm, dense if arm == "contiguous-control" else strided, golden, record)
        gate.close()
        require_affinity(affinity, "BF16 event-order component completion")
        evidence["status"] = "passed"
    except BaseException as exc:
        # If release/drain failed, do not call hipFree on a possibly live gate; the runner owns teardown.
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(report, evidence)


def route_index(key, slot, epoch):
    """Map scalar or broadcast tensor coordinates to the exact expert index."""
    rank = (key + slot * 3 + epoch * 5) % EP
    expert = (key // EP + slot * 3 + epoch) % LOCAL_EXPERTS
    return rank * LOCAL_EXPERTS + expert


def routes(torch, key, epoch):
    """Give every source row eight distinct destinations and epoch-specific experts."""
    slot = torch.arange(TOPK, device=key.device)[None, :]
    return route_index(key[:, None], slot, epoch).to(torch.int64)


def row_values(torch, keys, epoch):
    """Generate exact BF16 inputs for the deterministic native FNUZ conversion."""
    col = torch.arange(WIDTH, device=keys.device, dtype=torch.int32)[None, :]
    return ((keys[:, None] * 5 + col * 3 + 11 * epoch) % 127 - 63).to(torch.bfloat16)


def wire_values(torch, keys, epoch):
    payload = row_values(torch, keys, epoch).to(torch.float8_e4m3fnuz)
    cols = torch.arange(WIDTH // 128, device=keys.device, dtype=torch.int32)[None, :]
    scales = ((keys[:, None] + cols + epoch) % 7 + 1).to(torch.float32) / 64
    return payload, scales


def tags(torch, keys):
    slots = torch.arange(TOPK, device=keys.device, dtype=torch.int32)[None, :]
    # Encode row and slot as exactly representable FP32 fractions strictly above zero.
    return ((keys[:, None] + 1) * TOPK + slots + 1).to(torch.float32) / 32768


def exact(torch, actual, expected, label, wire=False):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"{label}: dtype or shape differs")
    if wire:
        actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
    if not torch.equal(actual, expected):
        raise ValueError(f"{label}: values differ")


def exchange_handle(torch, buffer, keys, rank, epoch, record):
    """Create a native FNUZ routing handle and verify payload, metadata and counts."""
    outgoing = routes(torch, keys, epoch)
    payload, scales = wire_values(torch, keys, epoch)
    n, nr, ne, which, layout = buffer.get_dispatch_layout(outgoing, EP * LOCAL_EXPERTS,
                                                         previous_event=buffer.capture(), async_finish=True,
                                                         allocate_on_comm_stream=True)
    (rx, sf), indices, probs, counts, handle, done = buffer.dispatch(
        (payload, scales), num_tokens_per_rank=n, num_tokens_per_rdma_rank=nr, num_tokens_per_expert=ne,
        is_token_in_rank=which, topk_idx=outgoing, topk_weights=tags(torch, keys), previous_event=layout,
        async_finish=True, allocate_on_comm_stream=True)
    done.current_stream_wait()
    all_keys = torch.arange(EP * TOKENS, device=keys.device, dtype=torch.int32)
    all_routes = routes(torch, all_keys, epoch)
    is_local = all_routes // LOCAL_EXPERTS == rank
    expected_keys = all_keys[is_local.any(dim=1)]
    recv_keys = ((probs.amax(dim=1) * 32768).to(torch.int32) - 1) // TOPK - 1
    exact(torch, recv_keys.sort().values, expected_keys, "received routing keys")
    if len(handle) != 12 or len(recv_keys) != TOKENS * TOPK or sf.stride() != (WIDTH // 128, 1):
        raise ValueError("unexpected internode handle or unaligned scale geometry")
    ix = all_routes[recv_keys.long()]
    local = ix // LOCAL_EXPERTS == rank
    wanted, wanted_sf = wire_values(torch, recv_keys, epoch)
    exact(torch, rx, wanted, "FNUZ wire bytes", wire=True)
    exact(torch, sf, wanted_sf, "FP32 scales")
    exact(torch, probs, tags(torch, recv_keys).masked_fill(~local, 0), "FP32 route probabilities")
    exact(torch, indices, torch.where(local, ix - rank * LOCAL_EXPERTS, -1), "local expert indices")
    exact(torch, torch.tensor(counts, dtype=torch.int64, device=keys.device),
          torch.bincount((all_routes[is_local] % LOCAL_EXPERTS), minlength=LOCAL_EXPERTS), "local expert counts")
    record.update(tokens=len(recv_keys), scale_row_bytes=WIDTH // 128 * 4, exact_wire=True,
                  exact_scales=True, exact_metadata=True)
    return handle, recv_keys, local


def cached_dispatch(buffer, bf16, handle, ready):
    """Enqueue the native cached reverse path; it must not return routing metadata."""
    result, indices, probs, counts, returned_handle, done = buffer.dispatch(
        bf16, handle=handle, previous_event=ready, async_finish=True, allocate_on_comm_stream=True)
    if any(x is not None for x in (indices, probs, counts, returned_handle)) or done.event is None:
        raise ValueError("cached BF16 dispatch returned unexpected metadata")
    return result, done


def cached_after_copy(torch, utils, buffer, gate, streams, source, handle, record, report):
    """Call native cached dispatch behind the after-copy event, then verify the held boundary."""
    producer, release = streams
    if source.is_contiguous() or source.dtype != torch.bfloat16 or tuple(source.shape) != (TOKENS, WIDTH):
        raise ValueError("expected a noncontiguous BF16 gradient fixture")
    comm = buffer.get_comm_stream()
    producer_done, prior_comm, after_comm, consumer_done, release_done = (
        torch.cuda.Event(enable_timing=False) for _ in range(5))
    for event, stream in zip((producer_done, prior_comm, after_comm, consumer_done, release_done),
                             (producer, comm, comm, producer, release)):
        event.record(stream)
    with torch.cuda.stream(producer):
        warm = source.contiguous()
    producer_done.record(producer)
    await_events((producer_done, prior_comm, after_comm, consumer_done, release_done), READY_SECONDS, "cached setup")
    del warm
    gate.write(release, 0)
    release_done.record(release)
    await_events((release_done,), READY_SECONDS, "cached reset")
    prior_comm.record(comm)
    record.update(entered=True, held_samples=[])
    report()
    try:
        with torch.cuda.stream(producer):
            gate.hold(producer)
            bf16 = source.contiguous()
            copy = bf16.data_ptr() != source.data_ptr() and bf16.is_contiguous()
            if not copy:
                raise ValueError("gated arm did not materialize new contiguous storage")
            ready = utils.EventOverlap(utils.EventHandle())
            producer_done.record(producer)
            start = monotonic()
            result, done = cached_dispatch(buffer, bf16, handle, ready)
            record.update(copy=copy, native_host_seconds=monotonic() - start)
            after_comm.record(comm)
            done.current_stream_wait()
            consumer_done.record(producer)
        await_events((prior_comm,), READY_SECONDS, "cached prior UCCL stream")
        stop = monotonic() + OBSERVE_SECONDS
        while True:
            record["held_samples"].append(dict(prior_comm=prior_comm.query(), producer_done=producer_done.query(),
                                               after_comm=after_comm.query(), consumer_done=consumer_done.query()))
            if monotonic() >= stop:
                break
            sleep(0.05)
        verify_cached_held(record["held_samples"])
    finally:
        gate.write(release, 1)
        for event, stream in zip((release_done, consumer_done, after_comm), (release, producer, comm)):
            event.record(stream)
        await_events((release_done, consumer_done, after_comm), DRAIN_SECONDS, "cached release and drain")
        record["drained"] = True
        report()
    return result


def verify_cached_held(samples):
    """Require independently ready UCCL prior work with both dependent paths held."""
    expected = dict(prior_comm=True, producer_done=False, after_comm=False, consumer_done=False)
    if not samples or any(s != expected for s in samples):
        raise ValueError("cached BF16 dispatch did not remain gated behind the pending copy")


def reverse_combine(torch, buffer, rank, keys, recv_keys, local, handle, epoch):
    """Check BF16 token sums and FP32 per-route gradients independently."""
    dests = routes(torch, keys, epoch) // LOCAL_EXPERTS
    contribution = rank + 1 + epoch * 16
    # Integer BF16 values while the reductions stay in the exactly representable range.
    tokens = torch.full((len(recv_keys), WIDTH), contribution, dtype=torch.bfloat16, device=keys.device)
    probs = (tags(torch, recv_keys) + contribution).masked_fill(~local, 0)
    combined, gradients, done = buffer.combine(tokens, handle, topk_weights=probs,
                                               previous_event=buffer.capture(), async_finish=True,
                                               allocate_on_comm_stream=True)
    done.current_stream_wait()
    marker = torch.cuda.Event(enable_timing=False)
    marker.record()
    await_events((marker,), DRAIN_SECONDS, "BF16/FP32 reverse combine")
    expected = (dests + (1 + epoch * 16)).sum(1).to(torch.bfloat16)[:, None].expand(-1, WIDTH)
    exact(torch, combined, expected, "BF16 reverse token sum")
    exact(torch, gradients, tags(torch, keys) + dests + (1 + epoch * 16), "FP32 reverse probability gradients")


def run_ep16(rank, local):
    """Run two outstanding native internode handles with no training or source modification."""
    from apertus_amd.runtime import preserve_affinity, require_affinity

    dist_up = False
    report_path = Path(f"bf16-ep16-rank-{rank}.json")
    record = dict(status="failed", rank=rank, local=local, image=IMAGE, hostname=socket.gethostname(), handles={})

    def report():
        write_json(report_path, record)

    try:
        with preserve_affinity(record.setdefault("affinity", {})) as affinity:
            import torch

            if (torch.__version__, torch.version.hip) != VENDOR or torch.cuda.device_count() != 4:
                raise ValueError("expected the pinned Torch/HIP and four visible GPUs")
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
            record["target"] = torch.cuda.get_device_properties(local).gcnArchName
            if record["target"].split(":")[0] != "gfx942":
                raise ValueError("expected gfx942")
            import deep_ep.buffer as native
            import deep_ep.utils as utils
            import uccl.ep as uccl

            for module, path, digest in ((native, PACKAGES / "deep_ep/buffer.py", BUFFER_SHA),
                                         (utils, UTILS, UTILS_SHA),
                                         (uccl, PACKAGES / "uccl/ep.abi3.so", EXTENSION_SHA)):
                if Path(module.__file__).resolve() != path or sha256(path) != digest:
                    raise ValueError(f"unexpected native module identity: {path}")
        require_affinity(affinity, "BF16 EP16 event-order component")
        if len(affinity) != 48 or uccl.num_max_nvl_peers() != 4:
            raise ValueError("unexpected CPU binding or local UCCL topology")
        record["stack"] = dict(torch=VENDOR[0], hip=VENDOR[1], buffer=BUFFER_SHA, utils=UTILS_SHA,
                               extension=EXTENSION_SHA)
        report()
        dist = torch.distributed
        deadline = timedelta(seconds=180)
        store = dist.FileStore(str(Path("bf16-ep16.store").resolve()), EP)
        store.set_timeout(deadline)
        dist.init_process_group("nccl", store=store, rank=rank, world_size=EP, timeout=deadline, device_id=device)
        dist_up = True
        hosts = [None] * EP
        dist.all_gather_object(hosts, record["hostname"])
        if len(set(hosts)) != 4 or any(hosts[i:i + 4] != [hosts[i]] * 4 for i in range(0, EP, 4)):
            raise ValueError("expected one four-rank block per host")
        configs = (native.Buffer.get_dispatch_config(EP), native.Buffer.get_combine_config(EP))
        size = [max(getattr(c, f"get_{kind}_buffer_size_hint")(WIDTH * 2, EP) for c in configs)
                for kind in ("nvl", "rdma")]
        if size != [64974592, 63350400]:
            raise ValueError(f"unexpected UCCL buffer size: {size}")
        os.environ.update(LOCAL_RANK=str(local), LOCAL_WORLD_SIZE="4")
        buffer = native.Buffer(dist.group.WORLD, *size, explicitly_destroy=True)
        if (buffer.runtime.get_num_rdma_ranks() != 4 or not buffer.scratch.is_cuda
                or "UCCL_EP_TRANSPORT=cxi initialized"
                not in Path(f"rank-{rank}.log").read_text(errors="replace")):
            raise ValueError("expected cached internode dispatch over native CXI with GPU scratch")
        producer, release = (torch.cuda.Stream(device=device) for _ in range(2))
        if len({0, producer.cuda_stream, release.cuda_stream, buffer.get_comm_stream().cuda_stream}) != 4:
            raise ValueError("expected distinct nondefault producer, release and UCCL streams")
        keys = torch.arange(rank * TOKENS, (rank + 1) * TOKENS, dtype=torch.int32, device=device)
        record.update(hosts=hosts, buffer_bytes=size, native_cxi=True)
        arms = [record["handles"].setdefault(str(i), {}) for i in range(2)]
        handles = [exchange_handle(torch, buffer, keys, rank, i, arms[i]) for i in range(2)]
        if handles[0][0] is handles[1][0] or torch.equal(handles[0][1].sort().values, handles[1][1].sort().values):
            raise ValueError("expected distinct outstanding handles and different routing keys")
        gate = HipSignal(local)
        # Consume the older handle after the newer is already live.
        for epoch, (handle, recv_keys, local_mask) in enumerate(handles):
            values = row_values(torch, keys, epoch)
            expected = row_values(torch, recv_keys, epoch)
            control, done = cached_dispatch(buffer, values, handle, buffer.capture())
            done.current_stream_wait()
            control_ready = torch.cuda.Event(enable_timing=False)
            control_ready.record()
            await_events((control_ready,), DRAIN_SECONDS, "contiguous cached dispatch control")
            exact(torch, control, expected, "contiguous cached BF16 control")
            arms[epoch]["exact_contiguous_control"] = True
            source = values.T.contiguous().T
            initialized = torch.cuda.Event(enable_timing=False)
            initialized.record()
            await_events((initialized,), READY_SECONDS, "strided fixture upload")
            report()
            result = cached_after_copy(torch, utils, buffer, gate, (producer, release), source, handle, arms[epoch], report)
            exact(torch, result, expected, "cached BF16 reverse dispatch")
            arms[epoch]["exact_cached_bf16"] = True
            reverse_combine(torch, buffer, rank, keys, recv_keys, local_mask, handle, epoch)
            arms[epoch].update(exact_combined_bf16=True, exact_probability_gradients=True)
            report()
        gate.close()
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        dist_up = False
        require_affinity(affinity, "BF16 EP16 event-order completion")
        record["status"] = "passed"
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["group_initialized_at_failure"] = dist_up
        # Never enter a collective from a single failing rank; the runner aborts the step.
        raise
    finally:
        report()


if __name__ == "__main__":
    settings = json.loads(Path("/apertus/resolved.json").read_text())
    if sys.argv[1:] == ["run"]:
        verify_launch(settings, os.environ)
        run()
    elif sys.argv[1:] == ["ep16"]:
        run_ep16(*verify_ep16_launch(settings, os.environ))
    else:
        raise ValueError("expected the reviewed component command")
