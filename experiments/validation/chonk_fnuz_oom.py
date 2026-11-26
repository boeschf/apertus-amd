"""Stream full-layout schedule and device memory evidence before a possible kill."""

from contextlib import contextmanager
from functools import wraps
import importlib
from inspect import unwrap
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
from threading import Event as ThreadEvent, Lock, Thread
from time import monotonic, monotonic_ns, time_ns

from apertus_amd.artifacts import sha256, write_json
from apertus_amd.config import CACHE_PATHS, option
from apertus_amd.entrypoint import train


WORLD, MICRO, BATCH = 64, 128, 4096
SEQ, MBS = 4096, 2
GEOMETRY = dict(tensor_model=1, pipeline_model=4, virtual_pipeline_model=2,
                expert_model=16, expert_tensor=1, context=1, data=16, expert_data=1)
COMPILE_THREADS = 4
TRACE_RANKS = range(48, WORLD)
TRACE_CALL_LIMIT = 8
HEAD_CHUNK = 512
HEAD_VOCAB = 200064
FIRST_HEAD_CALL = 5
HEAD_FLAG = "--lm-head-token-chunk-size"
REPLAY_FLAG = "--moe-resident-fp8-fnuz-recompute-fc1"
REPLAY_FIELD = "moe_resident_fp8_fnuz_recompute_fc1"
PRESSURE_RANKS = range(16, 32)
PRESSURE_CALL = 9
REDUCED_LAYERS = 24
REDUCED_LAYOUT = "Ettt|(ttt|)*6,tttL"
REDUCED_KDA = "([1,1,1,0]*6)"
REDUCED_MOE = "([0]*3+[1]*21)"
REDUCED_DRAIN_TRACE = "APERTUS_CHONK_REDUCED_DRAIN_TRACE"
DRAIN_FIRST_CALL = 259
DRAIN_LAST_CALL = 267
NOTIFY_TRACE = "APERTUS_CHONK_NOTIFY_TRACE"
DEPENDENCY_TRACE = "APERTUS_CHONK_DEPENDENCY_TRACE"
PREFIX_TRACE = "APERTUS_CHONK_PREFIX_TRACE"
P2P_PAIR_TRACE = "APERTUS_CHONK_P2P_PAIR_TRACE"
LATER_P2P_TRACE = "APERTUS_CHONK_LATER_P2P_TRACE"
LATER_PHASE_TRACE = "APERTUS_CHONK_LATER_PHASE_TRACE"
REVERSE_READY_TRACE = "APERTUS_CHONK_REVERSE_READY_TRACE"
STALL_TRACE = "APERTUS_CHONK_STALL_TRACE"
HANDOFF_TRACE = "APERTUS_CHONK_HANDOFF_TRACE"
UCCL_CPU_TIMEOUT = "UCCL_EP_CPU_TIMEOUT_SECS"
STALL_SOURCE = "041c1d753028e4190efb342accb98a6f362ee7dd"
CK_PREWARM = "APERTUS_CHONK_CK_PREWARM"
CK_NODE_VERIFIED = "node-verified"
CK_IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
CK_STEM = ("fmha_bwd_d128_bf16_batch_b16x128x128x16x128x16x32x128x128_r1x4x1_r4x1x1_"
           "r1x4x1_w16x16x32_w16x16x16_o1_maxq0_npad_nbias_ndbias_mask_ndropout_ndeterministic_ntrload_gfx9")
CK_OBJECT = CK_STEM + ".so.40ee1ee0"
CK_BYTES = 114880
CK_PACKAGE = Path("/opt/venv/lib/python3.12/site-packages/transformer_engine")
CK_CONSUMER = CK_PACKAGE / "pytorch/attention/dot_product_attention/dot_product_attention.py"
CK_BUILDER_SHA = "7e9906204c933e1abbc3ac0564f003c1c77b65a80d8cf3ecade1759facdbec8f"
CK_VENDOR = ("2.14.0+rocm10.1.0", "7.16.26385")
CK_BACKENDS = dict(fused_attention_backend="NVTE_Fused_Attn_Backend.NVTE_CK",
                   use_fused_attention="1", use_flash_attention="False", use_unfused_attention="False")
CK_TENSORS = ("output", "dq", "dk", "dv")
CK_NORMS = (369.9413757324219, 331.6982727050781, 333.5680847167969, 369.4879150390625)
CK_NODE_TIMEOUT = 240
PAIR_SENDERS = range(32, 48)
PAIR_SAMPLE_SECONDS, PAIR_MAX_SAMPLES = 5, 96
PAIR_SENDER_MARKERS = (
    "sender_call_entry", "sender_backward_receive_before", "sender_backward_receive_after",
    "sender_forward_chunk_wait_before", "sender_forward_chunk_wait_after", "sender_isend_ready",
)
NOTIFY_CALL = 8
NOTIFY_MARKERS = ("producer", "prior_uccl", "layout")
# Each tuple supplies direction, source/target stage and native post/wait contexts.
LATER_EDGES = {
    "f9": ("forward", 1, 2, (10, 9, 0), (9, 8, 1), (10, 9, 2)),
    "f8": ("forward", 2, 3, (9, 8, 1), (8, 7, 2), (9, 8, 3)),
    "b2": ("backward", 3, 2, (8, 7, 2), (9, 8, 1), (10, 9, 2)),
}
LATER_TARGETS = {2: (10, 9, 2), 3: (9, 8, 3)}
LATER_SENDER_CALL = LATER_EDGES["f9"][3]
CALL8_SENDER = (8, 7, 0)
SENDER_CALLBACKS = ("pre_forward", "pre_backward", "post_forward", "post_backward")
SENDER_COMPILE_LIMIT = 64
SENDER_SLOW_NODE_LIMIT = 32
SENDER_SLOW_NODE_NS = 250_000_000
READY_WAIT_MARKERS = ("backward0:before", "backward0:after")
READY_DGRAD_MARKERS = ("fc2_dgrad:before", "fc2_dgrad:after")
PREFIX_BOUNDARIES = ("forward_batch_plan", "loss_backward", "forward_receive_callback")
DEPENDENCY_NODES = (
    "pre_process forward", "post_process backward", "moe_post_combine backward",
    "moe_combine backward", "attn forward", "moe_pre_dispatch forward",
    "mlp backward", "moe_post_dispatch backward", "moe_dispatch forward",
)


def head_chunk_size(argv):
    """Accept only the traced dense reference or the separately qualified token cap."""
    if sum(arg.split("=", 1)[0] == HEAD_FLAG for arg in argv) > 1:
        raise ValueError("duplicate LM head chunk size in diagnostic inputs")
    value = option(argv, HEAD_FLAG, "0")
    if value not in ("0", str(HEAD_CHUNK)):
        raise ValueError("unexpected LM head chunk size in diagnostic inputs")
    return int(value)


def verify_head_trace(rows, chunk):
    """Prove the initial loss actually used the declared full-vocabulary tiling."""
    chunk = chunk or SEQ * MBS
    logits = {"ce_enter": ("torch.bfloat16", 2), "ce_fp32_logits_exit": ("torch.float32", 4)}
    tile = ["ce_enter", "ce_fp32_logits_exit", "ce_exit"]
    expected = ["output_head_enter", *(tile * (SEQ * MBS // chunk)), "output_head_exit"]
    traced = set(expected)
    first = [row for row in rows if row.get("call") == FIRST_HEAD_CALL and row["event"] in traced]
    if [row["event"] for row in first] != expected:
        raise ValueError("unexpected first LM-head tile count or boundary order")
    if any(row.get("forward") != FIRST_HEAD_CALL - 1 or "backward" not in row
           or row["backward"] is not None for row in first):
        raise ValueError("first LM-head chunks did not run at the forward-only warmup boundary")
    for row in first:
        event = row["event"]
        if event in logits:
            dtype, item_bytes = logits[event]
            if (row.get("shape") != [chunk // MBS, MBS, HEAD_VOCAB] or row.get("dtype") != dtype
                    or row.get("bytes") != chunk * HEAD_VOCAB * item_bytes):
                raise ValueError(f"incorrectly sized or typed LM-head evidence: {event}")


def verify_effective_head(args, config, chunk):
    """Match the native CLI and model config to the invocation; older dense source has no field."""
    effective = tuple(getattr(obj, "lm_head_token_chunk_size", 0) for obj in (args, config))
    if any(type(size) is not int or size != chunk for size in effective):
        raise ValueError(f"expected LM-head chunk size {chunk}; got args/config {effective}")
    if chunk and any(not obj.cross_entropy_loss_fusion or obj.cross_entropy_fusion_impl != "native"
                     for obj in (args, config)):
        raise ValueError("chunked LM head requires the native fused CE")


def replay_requested(argv):
    """Read the opt-in replay flag, preserving the older stored-operand recipe."""
    flags = [arg for arg in argv if arg.split("=", 1)[0] == REPLAY_FLAG]
    if flags not in ([], [REPLAY_FLAG]):
        raise ValueError("unexpected FC1 replay flag in diagnostic inputs")
    return bool(flags)


def verify_effective_replay(args, config, replay):
    """Match the native CLI and transformer config, including old pins with no field."""
    if any(getattr(obj, REPLAY_FIELD, False) is not replay for obj in (args, config)):
        raise ValueError(f"unexpected effective FC1 replay setting (requested {replay})")


def diagnostic_depth(argv, chunk, replay, reduced_drain=False):
    """Admit only the full model or one of the two exact reduced-depth recipes."""
    value = option(argv, "--num-layers", "42")
    if value == "42" and not reduced_drain:
        return 42
    expected_memory = (HEAD_CHUNK, True) if reduced_drain else (0, False)
    if (value != str(REDUCED_LAYERS) or (chunk, replay) != expected_memory
            or any(option(argv, flag) != expected for flag, expected in (
                ("--pipeline-model-parallel-layout", REDUCED_LAYOUT),
                ("--linear-attention-freq", REDUCED_KDA), ("--moe-layer-freq", REDUCED_MOE)))):
        raise ValueError("unexpected reduced diagnostic depth, topology, or memory options")
    return REDUCED_LAYERS


def verify_reduced_model(args, cfg):
    """Check the parsed layer masks, layout, and native loss on every rank."""
    for obj in (args, cfg):
        if (obj.linear_attention_freq != [1, 1, 1, 0] * 6
                or obj.moe_layer_freq != [0] * 3 + [1] * 21
                or str(obj.pipeline_model_parallel_layout) != REDUCED_LAYOUT
                or not obj.cross_entropy_loss_fusion or obj.cross_entropy_fusion_impl != "native"):
            raise ValueError("unexpected effective reduced-model layout, masks, or loss")


def verify_replay_trace(rows, rank):
    """Require one observed replay on every rank and forward warmup clearance on PP rank 1."""
    observed = [row for row in rows if row["event"] in ("fc1_replay_enter", "fc1_replay_exit")]
    if ([row["event"] for row in observed] != ["fc1_replay_enter", "fc1_replay_exit"]
            or any(observed[0].get(k) != observed[1].get(k) for k in ("call", "forward", "backward"))
            or type(observed[0].get("call")) is not int or observed[0]["call"] < 1
            or type(observed[0].get("backward")) is not int or observed[0]["backward"] < 0):
        raise ValueError(f"rank {rank} missing FC1 replay in backward")
    if rank in PRESSURE_RANKS:
        if not any(row["event"] == "schedule_exit" and row.get("call") == PRESSURE_CALL
                   and row.get("forward") == PRESSURE_CALL - 1
                   and "backward" in row and row["backward"] is None for row in rows):
            raise ValueError(f"rank {rank} did not clear forward warmup call {PRESSURE_CALL}")
        if observed[0]["call"] <= PRESSURE_CALL:
            raise ValueError(f"rank {rank} replay occurred before warmup finished")


def notify_trace_requested(environment, layers, chunk, replay, reduced_drain=False):
    """Allow dispatch markers only for the reviewed mitigated recipes."""
    flag = environment.get(NOTIFY_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and (
            layers != (REDUCED_LAYERS if reduced_drain else 42) or chunk != HEAD_CHUNK or not replay)):
        raise ValueError("unexpected dispatch notification trace recipe")
    return flag == "1"


def dependency_trace_requested(environment, notify_trace):
    """Require the existing three-marker dispatch trace as the control."""
    flag = environment.get(DEPENDENCY_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and not notify_trace):
        raise ValueError("dependency trace requires the full-size notification trace")
    return flag == "1"


def prefix_trace_requested(environment, dependency_trace):
    """Extend the dependency trace only for its already-qualified input shape."""
    flag = environment.get(PREFIX_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and not dependency_trace):
        raise ValueError("prefix trace requires the call-8 dependency trace")
    return flag == "1"


def pair_trace_requested(environment, prefix_trace):
    """Use the matched-pair observer only on the full prefix diagnostic."""
    flag = environment.get(P2P_PAIR_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and not prefix_trace):
        raise ValueError("P2P pair trace requires the call-8 prefix trace")
    return flag == "1"


def ck_prewarm_requested(environment, pair_trace):
    """Keep this prewarm specific to the reviewed full-layout diagnostic."""
    flag = environment.get(CK_PREWARM, "0")
    if flag not in ("0", CK_NODE_VERIFIED) or (flag == CK_NODE_VERIFIED and not pair_trace):
        raise ValueError("node-verified CK prewarm requires the matched full-layout P2P trace")
    return flag == CK_NODE_VERIFIED


def later_trace_requested(environment, ck_prewarm):
    """Keep later-call capture specific to the node-verified full-layout retest."""
    flag = environment.get(LATER_P2P_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and not ck_prewarm):
        raise ValueError("later P2P trace requires node-verified CK prewarm")
    return flag == "1"


def later_phase_requested(environment, later_trace):
    """Only extend the reviewed later P2P capture, never the workload itself."""
    flag = environment.get(LATER_PHASE_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and not later_trace):
        raise ValueError("later phase trace requires the matched later P2P trace")
    return flag == "1"


def reverse_ready_requested(environment, later_phase):
    """Use this trace only atop the complete reviewed later-phase observer."""
    flag = environment.get(REVERSE_READY_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and not later_phase):
        raise ValueError("reverse-ready trace requires the later-phase diagnostic")
    return flag == "1"


def stall_trace_requested(environment, reverse_ready):
    """Keep the stall observer on the reviewed full-layout trace and source path."""
    flag = environment.get(STALL_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and not reverse_ready):
        raise ValueError("stall trace requires the reverse-readiness diagnostic")
    return flag == "1"


def handoff_trace_requested(environment, stall_trace):
    flag = environment.get(HANDOFF_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and not stall_trace):
        raise ValueError("handoff trace requires the full-layout stall diagnostic")
    return flag == "1"


def extended_timeout_requested(environment, handoff_trace):
    """Enable timed snapshots only for the one reviewed 600-second diagnostic."""
    timeout = environment.get(UCCL_CPU_TIMEOUT)
    if timeout is not None and (timeout != "600" or not handoff_trace):
        raise ValueError("600-second UCCL host timeout requires the full-layout handoff diagnostic")
    return timeout is not None


def reduced_drain_requested(environment):
    """Require the original extended-timeout trace chain for the depth control."""
    flag = environment.get(REDUCED_DRAIN_TRACE, "0")
    if flag not in ("0", "1") or (flag == "1" and environment.get(UCCL_CPU_TIMEOUT) != "600"):
        raise ValueError("reduced drain control requires the 600-second UCCL host timeout")
    return flag == "1"


def ck_training_mapping_required(rank, reduced_drain):
    """PP0 owns layers 1–3 and 13–15 in the depth control, all of them KDA."""
    return not reduced_drain or rank >= 16


def observe_ck_object(path):
    """Report rejected size/ELF/digest evidence without following nonregular files."""
    info = path.lstat()
    record = dict(key=path.name, bytes=info.st_size, regular=stat.S_ISREG(info.st_mode))
    if record["regular"]:
        with path.open("rb") as stream:
            record["elf"] = stream.read(4) == b"\x7fELF"
        record["sha256"] = sha256(path)
    return record


def ck_artifact(record):
    """Admit the exact key and qualified size, not a universal digest."""
    if (record.get("key") != CK_OBJECT or record.get("regular") is not True
            or record.get("elf") is not True or record.get("bytes") != CK_BYTES
            or not re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", ""))):
        raise ValueError(f"invalid native CK artifact: {record}")
    return {k: record[k] for k in ("key", "bytes", "sha256")}


def ck_file_identity(path):
    """Retain inode and mutation identity independently of the digest."""
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns


def verify_ck_object(path, expected):
    """Verify this particular node's bytes and return the unchanged file identity."""
    before = ck_file_identity(path)
    if ck_artifact(observe_ck_object(path)) != expected or ck_file_identity(path) != before:
        raise ValueError(f"CK object differs from the qualified node artifact: {path}")
    return before


def ck_executable_mapping(path, identity, maps):
    """Reject a same-name replacement or deleted mapping."""
    dev, inode, *_ = identity
    device = os.major(dev), os.minor(dev)
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == str(path) and "x" in fields[1]:
            if int(fields[4]) == inode and tuple(int(n, 16) for n in fields[3].split(":")) == device:
                return True
    return False


def install_ck_object(source, target, expected):
    """Verify the node-local source and copy without replacing any cache object."""
    source_identity = verify_ck_object(source, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, target.open("xb") as copy:
        shutil.copyfileobj(original, copy)
    target.chmod(0o444)
    if verify_ck_object(source, expected) != source_identity:
        raise ValueError("qualified CK source changed during copy")
    return verify_ck_object(target, expected)


def ck_prewarm_worker(report):
    """Run the component-qualified native TE B2/S4096 fixture in a fresh process."""
    from apertus_amd.runtime import preserve_affinity, require_affinity

    record = dict(status="failed", pid=os.getpid(), pgid=os.getpgrp(), hostname=socket.gethostname())
    try:
        record["builder_sha256"] = sha256(CK_PACKAGE / "lib/ck_jit/ck_jit_compile.sh")
        if record["builder_sha256"] != CK_BUILDER_SHA:
            raise ValueError("CK build script differs from the component fixture")
        with preserve_affinity(record.setdefault("affinity", {})) as affinity:
            import torch

            record.update(torch=torch.__version__, hip=torch.version.hip, visible_gpus=torch.cuda.device_count())
            if ((record["torch"], record["hip"]) != CK_VENDOR or record["visible_gpus"] != 4
                    or os.environ["LOCAL_RANK"] != "0"):
                raise ValueError("CK worker requires the qualified Torch/HIP and four visible GPUs on local rank zero")
            torch.cuda.set_device(0)
            record["target"] = torch.cuda.get_device_properties(0).gcnArchName
            if record["target"].split(":")[0] != "gfx942":
                raise ValueError("CK worker requires gfx942")
            module = importlib.import_module(
                "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention")
            record["te_consumer"] = str(Path(module.__file__).resolve())
        require_affinity(affinity, "CK worker")
        if len(affinity) != 48 or record["te_consumer"] != str(CK_CONSUMER):
            raise ValueError("CK worker CPU binding or TE consumer differs from qualification")
        rng = torch.Generator(device="cuda").manual_seed(123)
        values = [torch.randn((SEQ, MBS, n, 128), device="cuda", dtype=torch.bfloat16, generator=rng)
                  .requires_grad_() for n in (28, 7, 7)]
        grad = torch.randn((SEQ, MBS, 3584), device="cuda", dtype=torch.bfloat16, generator=rng)
        model = module.DotProductAttention(28, 128, num_gqa_groups=7, attention_dropout=0.0,
                                          qkv_format="sbhd", attn_mask_type="causal", tp_size=1).train()
        if model.deterministic:
            raise ValueError("unexpected deterministic CK worker route")
        torch.cuda.synchronize()
        out = model(*values)
        torch.cuda.synchronize()
        record["backends"] = {k: str(module._attention_backends[k]) for k in CK_BACKENDS}
        if tuple(out.shape) != tuple(grad.shape) or out.dtype != torch.bfloat16 or record["backends"] != CK_BACKENDS:
            raise ValueError("unexpected CK output or backend")
        start = monotonic()
        out.backward(grad)
        torch.cuda.synchronize()
        record["backward_seconds"] = monotonic() - start
        tensors = [out, *(x.grad for x in values)]
        shapes = [grad.shape, *(x.shape for x in values)]
        valid = [t is not None and t.shape == shape and t.dtype == torch.bfloat16 and t.isfinite().all().item()
                 for t, shape in zip(tensors, shapes)]
        norms = [float(t.float().norm().item()) if ok else None for t, ok in zip(tensors, valid)]
        record.update(finite_bf16=dict(zip(CK_TENSORS, valid)),
                      norms=[n if n is not None and math.isfinite(n) else None for n in norms])
        target = Path(os.environ["XDG_CACHE_HOME"]) / "te_ck_jit" / CK_OBJECT
        identity = ck_file_identity(target)
        record["observed_object"] = observe_ck_object(target)
        record["mapped"] = ck_executable_mapping(target, identity, Path("/proc/self/maps").read_text())
        if ck_file_identity(target) != identity:
            raise ValueError("CK object changed while worker checked its mapping")
        require_affinity(affinity, "CK worker completion")
        verify_ck_worker(record, ck_artifact(record["observed_object"]))
        record["status"] = "passed"
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(report, record)


def verify_ck_worker(record, expected):
    """Require the component-qualified route, mapping, and output/gradient norms."""
    if (record["visible_gpus"] != 4 or record["target"].split(":")[0] != "gfx942"
            or (record["torch"], record["hip"]) != CK_VENDOR or record["builder_sha256"] != CK_BUILDER_SHA
            or record["te_consumer"] != str(CK_CONSUMER) or record["backends"] != CK_BACKENDS
            or record["mapped"] is not True or ck_artifact(record["observed_object"]) != expected
            or record["finite_bf16"] != dict.fromkeys(CK_TENSORS, True)
            or len(record["norms"]) != len(CK_NORMS)
            or any(type(n) not in (int, float) or not math.isclose(n, ref, rel_tol=0.005)
                   for n, ref in zip(record["norms"], CK_NORMS))):
        raise ValueError("CK worker differs from the native component qualification")


def ck_builder_quiescent():
    """Confirm no descendant compiler survives in this rank's training PGID."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == os.getpgrp() and fields[0] not in ("Z", "X"):
            return False
    return True


def ck_worker_phase(cache_root, output, node, phase, deadline, phase_evidence):
    """Run one fresh native worker and inspect its object only after quiescence."""
    env = dict(os.environ)
    for name, relative in CACHE_PATHS.items():
        path = cache_root / relative
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    target = cache_root / "xdg/te_ck_jit" / CK_OBJECT
    log = output / f"ck-prewarm-node-{node}-{phase}.log"
    report = output / f"ck-prewarm-node-{node}-{phase}.json"
    phase_evidence.update(log=log.name, report=report.name, object_path=str(target))
    try:
        with log.open("x") as stream:
            timeout = deadline - monotonic()
            if timeout <= 0:
                raise TimeoutError("node CK prewarm deadline exhausted")
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "ck-prewarm-worker", str(report)],
                           stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=True)
    finally:
        phase_evidence["quiescent"] = ck_builder_quiescent()
        if not phase_evidence["quiescent"]:
            raise ValueError(f"CK {phase} descendants remain")
        try:
            phase_evidence["observed_object"] = observe_ck_object(target)
        except OSError as error:
            phase_evidence["observation_error"] = str(error)
    worker = json.loads(report.read_text())
    expected = ck_artifact(phase_evidence["observed_object"])
    if (worker["status"] != "passed" or worker["pgid"] != os.getpgrp()
            or worker["hostname"] != socket.gethostname()):
        raise ValueError(f"CK {phase} did not pass in the owned local process group")
    verify_ck_worker(worker, expected)
    return worker, log.read_text(errors="replace"), expected


def require_ck_cold_compile(log, source):
    """Reject unproven cold objects before making even the verification copy."""
    lines = log.splitlines()
    if (lines.count(f"[CK-JIT] JIT-compiling blob: {CK_STEM}.cpp") != 1
            or f"[CK-JIT-BUILD] SUCCESS: {source}" not in lines):
        raise ValueError("missing exact CK cold-build success evidence")


def verify_ck_pair(cold, warm, cold_log, warm_log, source, expected):
    """Gate handoff on exact cold compilation and fresh-process reuse of the verified copy."""
    verify_ck_worker(cold, expected)
    verify_ck_worker(warm, expected)
    require_ck_cold_compile(cold_log, source)
    if (cold["status"] != "passed" or warm["status"] != "passed"
            or cold["pid"] == warm["pid"] or cold["pgid"] != warm["pgid"] or cold["hostname"] != warm["hostname"]
            or "[CK-JIT] JIT-compiling" in warm_log or "[CK-JIT-BUILD]" in warm_log
            or any(not math.isclose(a, b, rel_tol=0.005) for a, b in zip(cold["norms"], warm["norms"]))):
        raise ValueError("CK node build did not pass cold-build and fresh-process reuse gates")


def qualify_ck_node(build, output, node):
    """Build once, verify one warm copy, and retain node evidence before any handoff."""
    start = monotonic()
    deadline = start + CK_NODE_TIMEOUT
    report_path = output / f"ck-prewarm-node-{node}.json"
    report = dict(status="failed", node=node, hostname=socket.gethostname(), cold={})
    source = build / "cold/xdg/te_ck_jit" / CK_OBJECT
    target = build / "warm/xdg/te_ck_jit" / CK_OBJECT
    try:
        build.mkdir()
        cold, cold_log, expected = ck_worker_phase(build / "cold", output, node, "cold", deadline, report["cold"])
        require_ck_cold_compile(cold_log, source)
        source_identity = verify_ck_object(source, expected)
        before = install_ck_object(source, target, expected)
        report["warm"] = {}
        write_json(report_path, report)
        warm, warm_log, actual = ck_worker_phase(build / "warm", output, node, "warm", deadline, report["warm"])
        if (actual != expected or verify_ck_object(target, expected) != before
                or verify_ck_object(source, expected) != source_identity):
            raise ValueError("CK node object changed across warm verification")
        verify_ck_pair(cold, warm, cold_log, warm_log, source, expected)
        if monotonic() >= deadline:
            raise TimeoutError("node CK prewarm deadline exhausted")
        report.update(status="passed", artifact=expected, seconds=monotonic() - start)
        return ck_handoff(report, node)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(report_path, report)


def ck_handoff(report, node):
    """Bind the two qualified identities to their originating node."""
    if (report.get("status") != "passed" or report.get("node") != node
            or not report.get("hostname") or not 0 < report["seconds"] <= CK_NODE_TIMEOUT
            or any(report[phase]["quiescent"] is not True
                   or ck_artifact(report[phase]["observed_object"]) != report["artifact"] for phase in ("cold", "warm"))):
        raise ValueError(f"node {node} missing a qualified CK handoff")
    return dict(node=node, hostname=report["hostname"], **report["artifact"])


def ck_rank_seed(handoff, rank):
    """Select only the artifact from the rank's own block-placed node."""
    if handoff["node"] != rank // 4:
        raise ValueError(f"rank {rank} received a CK handoff from the wrong node")
    return dict(rank=rank, node=handoff["node"], key=handoff["key"], bytes=handoff["bytes"], sha256=handoff["sha256"])


def verify_ck_handoffs(handoffs):
    """Validate all node identities without requiring identical bytes across nodes."""
    if len(handoffs) != WORLD // 4 or len({row["hostname"] for row in handoffs}) != WORLD // 4:
        raise ValueError("unexpected CK node handoff count or host placement")
    for node, row in enumerate(handoffs):
        artifact = ck_artifact({**row, "elf": True, "regular": True})
        if not row["hostname"] or row != dict(node=node, hostname=row["hostname"], **artifact):
            raise ValueError(f"invalid CK handoff for node {node}")


def prewarm_ck(output, rank, emit):
    """Build once per node, seed one private object per rank, then gate training."""
    from apertus_amd.runtime import exchange, wait

    settings = json.loads((output.parent / "resolved.json").read_text())
    local = int(os.environ["LOCAL_RANK"])
    xdg = Path(os.environ["XDG_CACHE_HOME"])
    if (settings["image"]["sha256"] != CK_IMAGE or not settings.get("registry_image_path")
            or settings["cache"]["mode"] != "cold"
            or settings["procedure"]["timeout"] != 2700 or settings["cache"]["timeout"] != 300
            or settings["resources"]["nodes"] != WORLD // 4 or settings["resources"]["gpus_per_node"] != 4
            or rank % 4 != local or rank // 4 != int(os.environ["SLURM_NODEID"])
            or xdg.name != "xdg" or xdg.parent.name != f"cache-{rank}"
            or not xdg.resolve().is_relative_to("/dev/shm/apertus-amd")):
        raise ValueError("unexpected full-layout CK prewarm inputs")
    node = rank // 4
    build = xdg.parent.parent / "ck-prewarm"
    source = build / "cold/xdg/te_ck_jit" / CK_OBJECT
    session = output / ".runtime"
    ready = [session / f"ck-prewarm-ready-{n}.json" for n in range(WORLD // 4)]
    timeout = settings["cache"]["timeout"]
    if local == 0:
        emit("ck_prewarm_build_enter", node=node)
        record = qualify_ck_node(build, output, node)
        write_json(ready[node], record)
        emit("ck_prewarm_build_exit", **record)
    handoffs = wait(ready, session, timeout)
    verify_ck_handoffs(handoffs)
    own = handoffs[node]
    if own["hostname"] != socket.gethostname():
        raise ValueError("CK source is not on this rank's node")
    rank_seed = ck_rank_seed(own, rank)
    artifact = {k: own[k] for k in ("key", "bytes", "sha256")}
    target = xdg / "te_ck_jit" / CK_OBJECT
    identity = install_ck_object(source, target, artifact)
    records = exchange(session, "ck-prewarm-rank", rank, WORLD, rank_seed, timeout)
    if records != [ck_rank_seed(handoffs[r // 4], r) for r in range(WORLD)]:
        raise ValueError("invalid CK rank handoff")
    fields = {k: v for k, v in rank_seed.items() if k != "rank"}
    emit("ck_prewarm_seed_verified", **fields)
    return target, identity, artifact, fields


def verify_ck_training(seed, emit=None, maps=None):
    """Require the actual training process to load the unchanged private seed."""
    target, identity, artifact, fields = seed
    if maps is None:
        maps = Path("/proc/self/maps").read_text()
    if verify_ck_object(target, artifact) != identity or not ck_executable_mapping(target, identity, maps):
        raise ValueError("training did not reuse the unchanged qualified CK object")
    if emit is not None:
        emit("ck_prewarm_training_mapped", **fields)


def verify_ck_rank(rows, log_text, rank, handoff, require_mapping=True):
    """Require a private seed, plus actual reuse on ranks that run CK attention."""
    names = ("ck_prewarm_seed_verified", "ck_prewarm_training_mapped")
    seed = ck_rank_seed(handoff, rank)
    expected = [dict(event=name, **seed) for name in (names if require_mapping else names[:1])]
    actual = [{k: r.get(k) for k in ("event", *seed)} for r in rows if r["event"] in names]
    if actual != expected or f"JIT-compiling blob: {CK_STEM}.cpp" in log_text:
        raise ValueError(f"rank {rank} missing unchanged CK prewarm evidence or rebuilt the seed")


class P2PPairTrace:
    """Observe one native P2P pair without replacing its operations or Work handles."""

    def __init__(self, cuda, emit, active, rank, get_comm_stream, dependencies):
        self.cuda, self.emit, self.active = cuda, emit, active
        self.sender = rank in PAIR_SENDERS
        self.peer = rank + 16 if self.sender else rank - 16
        self.get_comm_stream = get_comm_stream
        self.dependencies = dependencies
        self.active_post, self.active_wait = False, False
        self.posted = False
        self.stop = ThreadEvent()
        self.lock = Lock()
        self.events = {}

    def at(self, call, forward, backward):
        context = self.active()
        return context is not None and (context["call"], context["forward"], context["backward"]) == (
            call, forward, backward)

    def log(self, event, **detail):
        self.emit(event, **self.active(), peer=self.peer, **detail)

    def mark_sender(self, name):
        stream = self.cuda.current_stream()
        try:
            event = self.cuda.Event(enable_timing=False)
            event.record(stream)
        except (RuntimeError, OSError) as error:
            self.log("p2p_pair_record_error", marker=name, error=str(error))
            return
        with self.lock:
            self.events[name] = (str(stream.cuda_stream), event)

    def sample(self, count, context):
        with self.lock:
            events, posted = list(self.events.items()), self.posted
        markers, errors = {}, {}
        for name, (stream, event) in reversed(events):
            try:
                state = event.query()
            except (RuntimeError, OSError) as error:
                state = None
                errors[name] = str(error)
            markers[name] = dict(stream=stream, complete=state)
        if not self.stop.is_set():
            self.emit("p2p_pair_sender_sample", **context, peer=self.peer, sample=count, posted=posted,
                      markers=markers, query_errors=errors)
        return posted and bool(markers) and all(m["complete"] is True for m in markers.values())

    def begin(self):
        if not self.sender or not self.at(8, 7, 0):
            return
        self.mark_sender("sender_call_entry")
        context = dict(self.active())
        device = self.cuda.current_device()

        def observe():
            try:
                with self.cuda.device(device):
                    for count in range(1, PAIR_MAX_SAMPLES + 1):
                        if self.stop.wait(PAIR_SAMPLE_SECONDS) or self.sample(count, context):
                            return
            except (RuntimeError, OSError) as error:
                self.emit("p2p_pair_sampler_error", **context, peer=self.peer, error=str(error))

        Thread(target=observe, name="chonk-p2p-observer", daemon=True).start()

    def wrap_backward_step(self, original):
        @wraps(original)
        def step(*args, **kwargs):
            callback = kwargs.get("pre_backward")
            if self.at(8, 7, 0) and callback is not None:
                @wraps(callback)
                def pre_backward(*a, **kw):
                    self.mark_sender("sender_backward_receive_before")
                    result = callback(*a, **kw)
                    self.mark_sender("sender_backward_receive_after")
                    return result
                kwargs["pre_backward"] = pre_backward
            return original(*args, **kwargs)
        return step

    def wrap_chunk_wait(self, original):
        @wraps(original)
        def wait(plan):
            if not self.at(8, 7, 0) or self.cuda.current_stream() != self.get_comm_stream():
                return original(plan)
            self.mark_sender("sender_forward_chunk_wait_before")
            result = original(plan)
            self.mark_sender("sender_forward_chunk_wait_after")
            return result
        return wait

    def wrap_forward_post(self, original):
        @wraps(original)
        def post(*args, **kwargs):
            if not (self.at(8, 7, 0) if self.sender else self.at(7, 6, 1)):
                return original(*args, **kwargs)
            self.active_post = True
            try:
                return original(*args, **kwargs)
            finally:
                self.active_post = False
        return post

    def wrap_native_post(self, original):
        @wraps(original)
        def post(*args, **kwargs):
            peer = kwargs.get("dst" if self.sender else "src")
            if not self.active_post or peer != self.peer:
                return original(*args, **kwargs)
            operation = "isend" if self.sender else "irecv"
            tensor = kwargs["tensor"]
            if self.sender:
                self.mark_sender("sender_isend_ready")
            self.log("p2p_pair_post_enter", operation=operation, shape=list(tensor.shape), dtype=str(tensor.dtype))
            work = original(*args, **kwargs)
            self.log("p2p_pair_post_return", operation=operation, work_id=id(work) if work is not None else None)
            if self.sender:
                with self.lock:
                    self.posted = True
                self.sample(0, self.active())
            return work
        return post

    def wrap_receiver_wait(self, callback):
        @wraps(callback)
        def wait(*args, **kwargs):
            queue = self.dependencies.receive_queue(callback)
            self.log("p2p_pair_wait_enter", work_id=id(queue[0]) if queue else None)
            self.active_wait = True
            try:
                return callback(*args, **kwargs)
            finally:
                self.active_wait = False
        return wait

    def wrap_deallocation(self, original):
        @wraps(original)
        def deallocate(*args, **kwargs):
            if not self.active_wait:
                return original(*args, **kwargs)
            # This is the very next call after the existing Work.wait(). Scope
            # the marker to the outer call, not its recursive tensor teardown.
            self.active_wait = False
            self.dependencies.mark("receiver_wait_return", self.cuda.current_stream())
            self.log("p2p_pair_wait_return")
            try:
                return original(*args, **kwargs)
            finally:
                self.active_wait = True
        return deallocate


def verify_pair_trace(rows, rank):
    """Require matched local receive identity and usable sender readiness evidence on success."""
    sender = rank in PAIR_SENDERS
    operation = "isend" if sender else "irecv"
    peer = rank + 16 if sender else rank - 16
    names = ["p2p_pair_post_enter", "p2p_pair_post_return"]
    if not sender:
        names += ["p2p_pair_wait_enter", "p2p_pair_wait_return"]
    events = [d for d in rows if d["event"] in names]
    contexts = [(8, 7, 0)] * 2 if sender else [(7, 6, 1)] * 2 + [(8, 7, 2)] * 2
    if ([d["event"] for d in events] != names
            or [(d.get("call"), d.get("forward"), d.get("backward")) for d in events] != contexts
            or any(d.get("peer") != peer for d in events)
            or any(d.get("operation") != operation for d in events[:2])
            or events[0].get("dtype") != "torch.bfloat16" or events[0].get("shape") != [SEQ, MBS, 3584]
            or type(events[1].get("work_id")) is not int
            or (not sender and events[1]["work_id"] != events[2].get("work_id"))):
        raise ValueError(f"rank {rank} missing matched native P2P trace")
    if sender:
        samples = [d for d in rows if d["event"] == "p2p_pair_sender_sample" and d.get("sample") == 0]
        if (len(samples) != 1 or samples[0].get("query_errors") != {}
                or samples[0].get("posted") is not True or samples[0].get("peer") != peer
                or set(samples[0].get("markers", {})) != set(PAIR_SENDER_MARKERS)
                or any(not m.get("stream") or type(m.get("complete")) is not bool
                       for m in samples[0]["markers"].values())):
            raise ValueError(f"rank {rank} missing usable sender GPU markers")
    if any(d["event"] in ("p2p_pair_record_error", "p2p_pair_sampler_error") for d in rows):
        raise ValueError(f"rank {rank} P2P trace instrumentation error")


class DependencyTrace:
    """Record a one-call prefix on the actual streams surrounding existing waits."""

    def __init__(self, cuda, emit, get_comm_stream, target=(NOTIFY_CALL, 7, 2)):
        self.cuda, self.emit, self.get_comm_stream = cuda, emit, get_comm_stream
        self.target = target
        self.context, self.node = None, None
        self.markers = []
        self.seen = set()
        self.record_errors = {}
        self.used = False

    def mark(self, label, stream):
        if label in self.seen:
            raise ValueError(f"repeated dependency trace boundary: {label}")
        self.seen.add(label)
        try:
            if stream is None:
                raise ValueError("stream unavailable")
            event = self.cuda.Event(enable_timing=False)
            event.record(stream)
        except (RuntimeError, OSError, ValueError) as error:
            self.record_errors[label] = str(error)
        else:
            self.markers.append((label, str(stream.cuda_stream), event))

    def begin(self, context):
        if self.used or context["call"] != self.target[0]:
            return
        if (context["forward"], context["backward"]) != self.target[1:]:
            raise ValueError("unexpected schedule at dependency trace boundary")
        self.used = True
        self.context = context
        self.mark("call_entry:compute", self.cuda.current_stream())
        self.mark("call_entry:schedule_comm", self.get_comm_stream())

    def wrap_context(self, original):
        @contextmanager
        @wraps(original)
        def acquire(node, name=None):
            if self.context is None or name not in DEPENDENCY_NODES:
                with original(node, name) as value:
                    yield value
                return
            previous, self.node = self.node, name
            try:
                self.mark(f"{name}:before_wait", node.stream)
                with original(node, name) as value:
                    self.mark(f"{name}:after_wait", node.stream)
                    yield value
                    if self.context is not None:
                        self.mark(f"{name}:after_body", node.stream)
            finally:
                self.node = previous
        return acquire

    def wrap_buffer_dispatch(self, original):
        @wraps(original)
        def dispatch(buffer, *args, **kwargs):
            if self.context is None or self.node != "moe_combine backward":
                return original(buffer, *args, **kwargs)
            stream = buffer.get_comm_stream()
            self.mark("reverse_bf16:uccl_before", stream)
            result = original(buffer, *args, **kwargs)
            self.mark("reverse_bf16:uccl_after", stream)
            return result
        return dispatch

    @staticmethod
    def receive_queue(callback):
        """Read the reviewed callback's receive queue without changing it."""
        callback = unwrap(callback)
        cells = dict(zip(callback.__code__.co_freevars, callback.__closure__ or ()))
        cell = cells.get("recv_prev_wait_handles")
        if cell is None:
            return None
        handles = cell.cell_contents
        return handles if isinstance(handles, list) else None

    def wrap_prefix_step(self, original, autograd, pair=None):
        """Bracket the pinned combined step without replacing its native receive wait."""
        @wraps(original)
        def step(forward_step_func, *args, **kwargs):
            if self.context is None:
                return original(forward_step_func, *args, **kwargs)
            context = self.context
            native_backward = autograd.backward
            receive = kwargs.get("pre_forward")
            plan_returned = False
            loss_started = False
            receive_started = False

            def boundary(name, phase, **detail):
                self.mark(f"{name}:{phase}", self.cuda.current_stream())
                self.emit("prefix_boundary", **context, boundary=name, phase=phase, **detail)

            @wraps(forward_step_func)
            def forward(*a, **kw):
                nonlocal plan_returned
                boundary("forward_batch_plan", "before")
                value = forward_step_func(*a, **kw)
                boundary("forward_batch_plan", "after")
                plan_returned = True
                return value

            @wraps(native_backward)
            def backward(*a, **kw):
                nonlocal loss_started
                if not plan_returned or loss_started or receive_started:
                    return native_backward(*a, **kw)
                loss_started = True
                autograd.backward = native_backward
                boundary("loss_backward", "before")
                value = native_backward(*a, **kw)
                boundary("loss_backward", "after")
                return value

            @wraps(receive)
            def pre_forward(*a, **kw):
                nonlocal receive_started
                receive_started = True
                queue = self.receive_queue(receive)
                boundary("forward_receive_callback", "before", queued=len(queue) if queue is not None else None)
                value = (pair.wrap_receiver_wait(receive) if pair is not None else receive)(*a, **kw)
                boundary("forward_receive_callback", "after", queued=len(queue) if queue is not None else None)
                return value

            autograd.backward = backward
            try:
                if receive is not None:
                    kwargs["pre_forward"] = pre_forward
                return original(forward, *args, **kwargs)
            finally:
                autograd.backward = native_backward
        return step

    def wrap_target(self, original):
        @wraps(original)
        def dispatch(*args, **kwargs):
            if self.context is None:
                return original(*args, **kwargs)
            returned = False
            try:
                result = original(*args, **kwargs)
                returned = True
                return result
            finally:
                self.finish("target_return" if returned else "target_exception")
        return dispatch

    def finish(self, boundary):
        """Query latest-to-earliest; never wait for an event or retain tensors."""
        if self.context is None:
            return
        context, self.context = self.context, None
        results, query_errors = [], {}
        for label, stream, event in reversed(self.markers):
            complete = None
            try:
                complete = event.query()
            except (RuntimeError, OSError) as error:
                query_errors[label] = str(error)
            results.append(dict(label=label, stream=stream, complete=complete))
        self.markers.clear()
        self.emit("dependency_observation", **context, boundary=boundary, markers=results[::-1],
                  record_errors=self.record_errors, query_errors=query_errors)


def verify_dependency_trace(rows, rank, prefix=False, pair=False, target=(NOTIFY_CALL, 7, 2)):
    """Require usable markers at the known first dispatch on a passing run."""
    observations = [r for r in rows if r["event"] == "dependency_observation" and r.get("call") == target[0]]
    if len(observations) != 1:
        raise ValueError(f"rank {rank} missing dependency observation")
    row = observations[0]
    markers = row["markers"]
    expected = ["call_entry:compute", "call_entry:schedule_comm"]
    if prefix:
        pairs = [(label, phase) for label in PREFIX_BOUNDARIES for phase in ("before", "after")]
        expected += [f"{label}:{phase}" for label, phase in pairs]
        if pair:
            expected.insert(expected.index("forward_receive_callback:after"), "receiver_wait_return")
        host = [r for r in rows if r["event"] == "prefix_boundary" and r.get("call") == target[0]]
        counts = [r.get("queued") for r in host if r.get("boundary") == "forward_receive_callback"]
        if ([(r.get("boundary"), r.get("phase")) for r in host] != pairs
                or any((r.get("call"), r.get("forward"), r.get("backward")) != target for r in host)
                or len(counts) != 2 or any(type(n) is not int or n < 0 for n in counts)
                or counts[0] - counts[1] not in (0, 1)):
            raise ValueError(f"rank {rank} unusable prefix observation")
    for node in DEPENDENCY_NODES:
        expected += [f"{node}:before_wait", f"{node}:after_wait"]
        if node == "moe_combine backward":
            expected += ["reverse_bf16:uccl_before", "reverse_bf16:uccl_after"]
        if node != "moe_dispatch forward":
            expected.append(f"{node}:after_body")
    if ((row.get("call"), row.get("forward"), row.get("backward")) != target
            or row.get("boundary") != "target_return" or row.get("record_errors") != {} or row.get("query_errors") != {}
            or [m.get("label") for m in markers] != expected
            or any(type(m.get("complete")) is not bool or not m.get("stream") for m in markers)):
        raise ValueError(f"rank {rank} unusable dependency observation")


class NotifyTrace:
    """Observe the first target dispatch without changing its kernels or waiting on GPU work."""

    def __init__(self, cuda, emit, active, target=(NOTIFY_CALL, 7, 2)):
        self.cuda, self.emit, self.active = cuda, emit, active
        self.target = target
        self.used = False
        self.capture = None

    def wrap_layout(self, original):
        @wraps(original)
        def layout(buffer, *args, **kwargs):
            capture = self.capture
            if capture is None:
                return original(buffer, *args, **kwargs)
            context, markers = capture
            if markers:
                raise ValueError("multiple layout calls in the traced FNUZ dispatch")
            producer = self.cuda.current_stream()
            comm = buffer.get_comm_stream()
            self.emit("notify_layout_enter", **context, uccl_stream_is_caller=comm == producer)
            self.mark(markers, "producer", producer)
            self.mark(markers, "prior_uccl", comm)
            result = original(buffer, *args, **kwargs)
            self.mark(markers, "layout", comm)
            self.emit("notify_layout_enqueued", **context)
            return result

        return layout

    def mark(self, markers, name, stream):
        event = self.cuda.Event(enable_timing=False)
        event.record(stream)
        markers[name] = event

    def wrap_dispatch(self, original):
        @wraps(original)
        def dispatch(*args, **kwargs):
            context = self.active()
            if self.used or context is None or context["call"] != self.target[0]:
                return original(*args, **kwargs)
            if (context["forward"], context["backward"]) != self.target[1:]:
                raise ValueError("unexpected microbatches at notify trace boundary")
            self.used = True
            markers = {}
            self.capture = (context, markers)
            returned = False
            try:
                result = original(*args, **kwargs)
                returned = True
                return result
            finally:
                self.capture = None
                states, query_errors = {}, {}
                # Querying an event never waits for it. Do not mask the training
                # exception if a marker is missing or the device reports an error.
                # Query latest work first: querying an earlier marker false and a
                # later one true could otherwise just reflect progress between queries.
                for name in reversed(NOTIFY_MARKERS):
                    if name not in markers:
                        states[name] = None
                        continue
                    try:
                        states[name] = markers[name].query()
                    except (RuntimeError, OSError) as exc:
                        states[name] = None
                        query_errors[name] = str(exc)
                self.emit("notify_dispatch_observation", **context, returned=returned,
                          markers=states, query_errors=query_errors)

        return dispatch


def verify_notify_trace(rows, rank, target=(NOTIFY_CALL, 7, 2)):
    """Require one correctly scoped set of marker observations on a passing run."""
    keys = ("notify_layout_enter", "notify_layout_enqueued", "notify_dispatch_observation")
    entries = [r for r in rows if r["event"] in keys and r.get("call") == target[0]]
    if ([r["event"] for r in entries] != list(keys)
            or any((r.get("call"), r.get("forward"), r.get("backward")) != target for r in entries)
            or entries[-1].get("returned") is not True or entries[-1].get("query_errors") != {}
            or set(entries[-1].get("markers", {})) != set(NOTIFY_MARKERS)
            or any(type(v) is not bool for v in entries[-1]["markers"].values())):
        raise ValueError(f"rank {rank} missing usable dispatch notification markers")


def later_rank_edges(rank):
    """Resolve the named schedule edges to global peers with the same pipeline lane."""
    stage, lane = divmod(rank, 16)
    for edge, (direction, sender, receiver, send, post, wait) in LATER_EDGES.items():
        if stage == sender:
            yield edge, direction, "isend", receiver * 16 + lane, send, None
        elif stage == receiver:
            yield edge, direction, "irecv", sender * 16 + lane, post, wait


class LaterP2PTrace:
    """Link native posts, local Work identity and GPU wait markers at the two later dispatches."""

    def __init__(self, cuda, emit, rank, active):
        self.cuda, self.emit, self.active = cuda, emit, active
        self.edges = tuple(later_rank_edges(rank))
        self.target = LATER_TARGETS.get(rank // 16)
        self.post_edges = ()
        self.markers, self.record_errors = {}, {}
        self.done = False

    def at(self, context):
        current = self.active()
        return context is not None and current is not None and tuple(
            current[k] for k in ("call", "forward", "backward")) == context

    def wrap_post(self, callback, edges):
        @wraps(callback)
        def post(*args, **kwargs):
            previous, self.post_edges = self.post_edges, edges
            try:
                return callback(*args, **kwargs)
            finally:
                self.post_edges = previous
        return post

    def wrap_native(self, original, operation):
        @wraps(original)
        def native(*args, **kwargs):
            peer = kwargs.get("dst" if operation == "isend" else "src")
            match = next((e for e in self.post_edges if (e[2], e[3]) == (operation, peer)), None)
            if match is None:
                return original(*args, **kwargs)
            fields = dict(edge=match[0], operation=operation, peer=peer, **self.active())
            tensor = kwargs["tensor"]
            self.emit("later_p2p_post_enter", **fields, shape=list(tensor.shape), dtype=str(tensor.dtype))
            work = original(*args, **kwargs)
            self.emit("later_p2p_post_return", **fields, work_id=id(work) if work is not None else None)
            return work
        return native

    def mark(self, name):
        try:
            stream = self.cuda.current_stream()
            event = self.cuda.Event(enable_timing=False)
            event.record(stream)
            self.markers[name] = (str(stream.cuda_stream), event)
        except (RuntimeError, OSError) as exc:
            self.record_errors[name] = str(exc)

    def wrap_wait(self, callback, edge, direction, peer):
        @wraps(callback)
        def wait(*args, **kwargs):
            cell_name = "recv_prev_wait_handles" if direction == "forward" else "recv_next_wait_handles"
            cells = dict(zip(callback.__code__.co_freevars, callback.__closure__ or ()))
            cell = cells.get(cell_name)
            queue = cell.cell_contents if cell is not None else None
            is_queue = isinstance(queue, list)
            fields = dict(edge=edge, peer=peer, **self.active())
            work_id = id(queue[0]) if is_queue and queue else None
            self.mark(f"{edge}:before")
            self.emit("later_p2p_wait_enter", **fields, work_id=work_id, queued=len(queue) if is_queue else None)
            value = callback(*args, **kwargs)
            self.mark(f"{edge}:after")
            self.emit("later_p2p_wait_return", **fields, queued=len(queue) if is_queue else None)
            return value
        return wait

    def callbacks(self, kwargs):
        """Scope wrappers to specific calls/directions; keep the original callbacks and queues."""
        for direction in ("forward", "backward"):
            post = f"post_{direction}"
            edges = tuple(e for e in self.edges if e[1] == direction and self.at(e[4]))
            if edges and kwargs.get(post) is not None:
                kwargs[post] = self.wrap_post(kwargs[post], edges)
            pre = f"pre_{direction}"
            for edge, d, operation, peer, _, wait in self.edges:
                if d == direction and operation == "irecv" and self.at(wait) and kwargs.get(pre) is not None:
                    kwargs[pre] = self.wrap_wait(kwargs[pre], edge, d, peer)

    def finish(self, boundary):
        if self.done or not self.at(self.target):
            return
        self.done = True
        markers, errors = {}, {}
        for label, (stream, event) in reversed(self.markers.items()):
            try:
                complete = event.query()
            except (RuntimeError, OSError) as exc:
                complete = None
                errors[label] = str(exc)
            markers[label] = dict(stream=stream, complete=complete)
        self.markers.clear()
        self.emit("later_p2p_gpu_observation", **self.active(), boundary=boundary, markers=markers,
                  record_errors=self.record_errors, query_errors=errors)

    def wrap_dispatch(self, original):
        @wraps(original)
        def dispatch(*args, **kwargs):
            returned = False
            try:
                result = original(*args, **kwargs)
                returned = True
                return result
            finally:
                self.finish("target_return" if returned else "target_exception")
        return dispatch


class ExtendedTimeoutTrace:
    """Query existing GPU markers near the old cutoff and during the extended wait."""

    OFFSETS = (101, 150, 300, 580)

    def __init__(self, cuda, emit, later, stall):
        self.cuda, self.emit, self.later, self.stall = cuda, emit, later, stall
        self.used = False

    def observe(self, stop, lock, context, device, markers, start):
        try:
            with self.cuda.device(device):
                for offset in self.OFFSETS:
                    if stop.wait(max(0, start + offset - monotonic())):
                        return
                    states, errors = {}, {}
                    for name, (stream, event) in reversed(tuple(markers.items())):
                        try:
                            complete = event.query()
                        except (RuntimeError, OSError) as exc:
                            complete = None
                            errors[name] = str(exc)
                        states[name] = dict(stream=stream, complete=complete)
                    with lock:
                        if not stop.is_set():
                            self.emit("extended_timeout_sample", **context, offset_seconds=offset,
                                      elapsed_seconds=monotonic() - start, markers=states, query_errors=errors)
        except (RuntimeError, OSError) as exc:
            with lock:
                if not stop.is_set():
                    self.emit("extended_timeout_sampler_error", **context, error=str(exc))

    def wrap(self, original):
        @wraps(original)
        def dispatch(*args, **kwargs):
            if self.used or not self.later.at(self.later.target):
                return original(*args, **kwargs)
            self.used = True
            context = dict(self.later.active())
            markers = dict(self.later.markers)
            if self.stall is not None:
                markers.update(self.stall.markers)
            stop, lock, start = ThreadEvent(), Lock(), monotonic()
            device = self.cuda.current_device()
            self.emit("extended_timeout_enter", **context, host_timeout_seconds=600,
                      marker_labels=sorted(markers))
            Thread(target=self.observe, args=(stop, lock, context, device, markers, start),
                   name="chonk-dispatch-observer", daemon=True).start()
            returned = False
            try:
                value = original(*args, **kwargs)
                returned = True
                return value
            finally:
                stop.set()
                with lock:
                    self.emit("extended_timeout_exit", **context, returned=returned,
                              elapsed_seconds=monotonic() - start)
        return dispatch


def verify_extended_timeout_trace(rows, rank):
    """Check that the original dispatch ran with the requested observer in scope."""
    records = [r for r in rows if r["event"].startswith("extended_timeout_")]
    stage = rank // 16
    target = LATER_TARGETS[stage]
    labels = ({"b2:before", "b2:after", "f9:before", "f9:after"} if stage == 2 else
              {"f8:before", "f8:after", "producer_prefix", "uccl_prefix", "uccl_native_entry",
               "uccl_cached_return", "consumer:before", "consumer:after"})
    if (len(records) < 2 or records[0]["event"] != "extended_timeout_enter"
            or records[0]["host_timeout_seconds"] != 600 or records[-1]["event"] != "extended_timeout_exit"
            or set(records[0]["marker_labels"]) != labels
            or records[-1]["returned"] is not True
            or any(tuple(r[k] for k in ("call", "forward", "backward")) != target for r in records)):
        raise ValueError(f"rank {rank} missing extended timeout dispatch records")
    for offset, row in zip(ExtendedTimeoutTrace.OFFSETS, records[1:-1]):
        if (row["event"] != "extended_timeout_sample" or row["offset_seconds"] != offset
                or row["elapsed_seconds"] < offset or row["query_errors"]
                or set(row["markers"]) != set(records[0]["marker_labels"])
                or any(type(m["complete"]) is not bool or not m["stream"] for m in row["markers"].values())):
            raise ValueError(f"rank {rank} unusable extended timeout sample")
    if len(records) - 2 > len(ExtendedTimeoutTrace.OFFSETS):
        raise ValueError(f"rank {rank} excess extended timeout samples")


def verify_later_trace(rows, rank):
    """Require every matched edge and usable nonblocking GPU marker samples on a passing update."""
    waits = set()
    for edge, _, operation, peer, post, wait in later_rank_edges(rank):
        events = [r for r in rows if r.get("edge") == edge and r["event"].startswith("later_p2p_")]
        names = ["later_p2p_post_enter", "later_p2p_post_return"]
        contexts = [post, post]
        if wait is not None:
            names += ["later_p2p_wait_enter", "later_p2p_wait_return"]
            contexts += [wait, wait]
            waits.update((f"{edge}:before", f"{edge}:after"))
        if ([r["event"] for r in events] != names
                or [tuple(r.get(k) for k in ("call", "forward", "backward")) for r in events] != contexts
                or any(r.get("peer") != peer for r in events)
                or any(r.get("operation") != operation for r in events[:2])
                or events[0].get("shape") != [SEQ, MBS, 3584] or events[0].get("dtype") != "torch.bfloat16"
                or type(events[1].get("work_id")) is not int
                or (wait is not None and (events[1]["work_id"] != events[2].get("work_id")
                    or type(events[2].get("queued")) is not int or type(events[3].get("queued")) is not int
                    or events[2]["queued"] != events[3]["queued"] + 1 or events[3]["queued"] < 0))):
            raise ValueError(f"rank {rank} missing matched later P2P edge {edge}")
    if waits:
        target = LATER_TARGETS[rank // 16]
        observations = [r for r in rows if r["event"] == "later_p2p_gpu_observation"]
        if len(observations) != 1:
            raise ValueError(f"rank {rank} missing later P2P GPU observation")
        row = observations[0]
        if (tuple(row.get(k) for k in ("call", "forward", "backward")) != target
                or row.get("boundary") != "target_return" or row.get("record_errors") != {}
                or row.get("query_errors") != {} or set(row["markers"]) != waits
                or any(type(m.get("complete")) is not bool or not m.get("stream") for m in row["markers"].values())):
            raise ValueError(f"rank {rank} unusable later P2P GPU observation")
        verify_notify_trace(rows, rank, target)


class SenderPhaseTrace:
    """Observe CPU work and vendor Dynamo callback spans without touching compiled functions."""

    def __init__(self, emit, handler, target=LATER_SENDER_CALL, live_nodes=False):
        self.emit, self.handler = emit, handler
        self.target, self.live_nodes = target, live_nodes
        self.context, self.node = None, None
        self.compile_events = self.slow_nodes = 0
        self.node_count = 0

    def log(self, event, **detail):
        self.emit(event, **self.context, monotonic_ns=monotonic_ns(), **detail)

    def compile(self, phase, args):
        if self.context is not None and self.compile_events < 2 * SENDER_COMPILE_LIMIT:
            self.compile_events += 1
            self.log("later_sender_compile", phase=phase, trigger=args.callback_trigger.name,
                     compile_id=str(args.compile_id), node=self.node)

    @contextmanager
    def capture(self, context, kwargs):
        current = tuple(context[k] for k in ("call", "forward", "backward"))
        if current != self.target:
            yield
            return
        self.context = context
        def start(args):
            self.compile("start", args)

        def end(args):
            self.compile("end", args)

        self.handler.register_start_callback(start)
        self.handler.register_end_callback(end)
        returned = False
        try:
            self.log("later_sender_enter")
            for name in SENDER_CALLBACKS:
                if kwargs.get(name) is not None:
                    kwargs[name] = self.wrap_callback(kwargs[name], name)
            yield
            returned = True
        finally:
            self.handler.remove_end_callback(end)
            self.handler.remove_start_callback(start)
            try:
                self.log("later_sender_exit", returned=returned, compile_events=self.compile_events,
                         compile_limit_hit=self.compile_events == 2 * SENDER_COMPILE_LIMIT,
                         slow_nodes=self.slow_nodes)
            finally:
                self.context = None

    def wrap_callback(self, callback, name):
        @wraps(callback)
        def run(*args, **kwargs):
            self.log("later_sender_callback", boundary=name, phase="enter")
            returned = False
            try:
                value = callback(*args, **kwargs)
                returned = True
                return value
            finally:
                self.log("later_sender_callback", boundary=name, phase="exit", returned=returned)
        return run

    def wrap_chunk_wait(self, original, current_stream, comm_stream):
        @wraps(original)
        def wait(*args, **kwargs):
            if self.context is None or current_stream() != comm_stream():
                return original(*args, **kwargs)
            self.log("later_sender_chunk_wait", phase="enter")
            returned = False
            try:
                value = original(*args, **kwargs)
                returned = True
                return value
            finally:
                self.log("later_sender_chunk_wait", phase="exit", returned=returned)
        return wait

    def wrap_context(self, original):
        @contextmanager
        @wraps(original)
        def acquire(node, name=None):
            if self.context is None:
                with original(node, name) as value:
                    yield value
                return
            previous, self.node = self.node, name
            start = monotonic_ns()
            index = self.node_count
            self.node_count += 1
            record = self.live_nodes and index < 192
            if record:
                self.log("later_sender_node", index=index, node=name, phase="enter")
            try:
                with original(node, name) as value:
                    yield value
            finally:
                self.node = previous
                elapsed = monotonic_ns() - start
                if record:
                    self.log("later_sender_node", index=index, node=name, phase="exit", elapsed_ns=elapsed)
                if elapsed >= SENDER_SLOW_NODE_NS and self.slow_nodes < SENDER_SLOW_NODE_LIMIT:
                    self.slow_nodes += 1
                    self.log("later_sender_slow_node", node=name, elapsed_ns=elapsed)
        return acquire


class ReverseReadyTrace:
    """Observe final-stage gradient layout and stage-1's first pre-replay dgrad."""

    def __init__(self, cuda, emit, rank, active):
        self.cuda, self.emit, self.active = cuda, emit, active
        self.sender = rank in PRESSURE_RANKS
        self.target = LATER_SENDER_CALL if self.sender else LATER_TARGETS[3]
        self.markers, self.errors = {}, {}
        self.seen_dgrad = False

    def at_target(self):
        row = self.active()
        return row is not None and tuple(row[k] for k in ("call", "forward", "backward")) == self.target

    def mark(self, label):
        try:
            stream = self.cuda.current_stream()
            event = self.cuda.Event(enable_timing=False)
            event.record(stream)
            self.markers[label] = str(stream.cuda_stream), event
        except (RuntimeError, OSError) as exc:
            self.errors[label] = str(exc)

    def sample(self, boundary):
        statuses, errors = {}, {}
        for label, (stream, event) in reversed(self.markers.items()):
            try:
                complete = event.query()
            except (RuntimeError, OSError) as exc:
                complete = None
                errors[label] = str(exc)
            statuses[label] = dict(stream=stream, complete=complete)
        self.emit("ready_sender_gpu", **self.active(), boundary=boundary, markers=statuses,
                  record_errors=dict(self.errors), query_errors=errors)

    def callbacks(self, kwargs):
        if not self.sender or not self.at_target() or kwargs.get("pre_backward") is None:
            return
        callback = kwargs["pre_backward"]
        fn = unwrap(callback)

        @wraps(callback)
        def wait(*args, **kw):
            cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
            cell = cells.get("recv_next_wait_handles")
            queue = cell.cell_contents if cell is not None else None
            known_queue = isinstance(queue, list)
            self.mark(READY_WAIT_MARKERS[0])
            self.emit("ready_backward0_wait_enter", **self.active(),
                      queued=len(queue) if known_queue else None,
                      work_id=id(queue[0]) if known_queue and queue else None)
            value = callback(*args, **kw)
            self.mark(READY_WAIT_MARKERS[1])
            self.emit("ready_backward0_wait_return", **self.active(), queued=len(queue) if known_queue else None)
            return value

        kwargs["pre_backward"] = wait

    def wrap_gemm(self, original):
        @wraps(original)
        def gemm(backend, operation, a, b, out, *args, **kwargs):
            if not self.at_target() or self.seen_dgrad or operation != "fc2_dgrad" or not out:
                return original(backend, operation, a, b, out, *args, **kwargs)
            self.seen_dgrad = True
            self.mark(READY_DGRAD_MARKERS[0])
            self.sample("dgrad_enter")
            start = monotonic_ns()
            self.emit("ready_fc2_dgrad_enter", **self.active(), experts=len(out), layout=kwargs.get("layout"))
            value = original(backend, operation, a, b, out, *args, **kwargs)
            self.emit("ready_fc2_dgrad_return", **self.active(), host_ns=monotonic_ns() - start)
            self.mark(READY_DGRAD_MARKERS[1])
            self.sample("dgrad_return")
            return value
        return gemm

    def finish(self):
        if self.sender and self.at_target():
            self.sample("call_exit")
            self.markers.clear()

    def wrap_reverse(self, original, dispatch_trace):
        @wraps(original)
        def backward(ctx, grad_output, *args, **kwargs):
            if not self.at_target() or dispatch_trace.context is None or dispatch_trace.node != "moe_combine backward":
                return original(ctx, grad_output, *args, **kwargs)
            self.emit("ready_reverse_grad_enter", **self.active(), shape=list(grad_output.shape),
                      stride=list(grad_output.stride()), dtype=str(grad_output.dtype),
                      contiguous=grad_output.is_contiguous(), async_finish=ctx.async_finish,
                      stream=str(self.cuda.current_stream().cuda_stream))
            value = original(ctx, grad_output, *args, **kwargs)
            self.emit("ready_reverse_grad_return", **self.active())
            return value
        return backward


def verify_reverse_ready(rows, rank):
    """Require usable metadata at either the sender or final-stage boundary."""
    sender = rank in PRESSURE_RANKS
    target = LATER_SENDER_CALL if sender else LATER_TARGETS[3]
    evidence = [row for row in rows if row["event"].startswith("ready_")]
    if any(tuple(row.get(k) for k in ("call", "forward", "backward")) != target for row in evidence):
        raise ValueError(f"rank {rank} reverse-ready trace was out of scope")
    events = [row["event"] for row in evidence]
    if sender:
        expected = ("ready_backward0_wait_enter", "ready_backward0_wait_return", "ready_sender_gpu",
                    "ready_fc2_dgrad_enter", "ready_fc2_dgrad_return", "ready_sender_gpu", "ready_sender_gpu")
        if tuple(events) != expected:
            raise ValueError(f"rank {rank} missing backward-0 or FC2 dgrad observations")
        start, end, before, enter, returned, after, exit_row = evidence
        if (type(start["work_id"]) is not int or type(start["queued"]) is not int
                or type(end["queued"]) is not int or start["queued"] != end["queued"] + 1
                or end["queued"] < 0 or enter["layout"] != "NN" or enter["experts"] < 1 or returned["host_ns"] < 0):
            raise ValueError(f"rank {rank} unusable backward-0 or FC2 dgrad observations")
        for row, boundary, labels in ((before, "dgrad_enter", READY_WAIT_MARKERS + READY_DGRAD_MARKERS[:1]),
                                      (after, "dgrad_return", READY_WAIT_MARKERS + READY_DGRAD_MARKERS),
                                      (exit_row, "call_exit", READY_WAIT_MARKERS + READY_DGRAD_MARKERS)):
            if (row["boundary"] != boundary or row["record_errors"] or row["query_errors"]
                    or set(row["markers"]) != set(labels)
                    or any(type(m.get("complete")) is not bool or not m.get("stream") for m in row["markers"].values())):
                raise ValueError(f"rank {rank} unusable backward-0 or FC2 dgrad GPU markers")
    else:
        if events != ["ready_reverse_grad_enter", "ready_reverse_grad_return"]:
            raise ValueError(f"rank {rank} missing reverse BF16 gradient layout")
        grad = evidence[0]
        if (grad["dtype"] != "torch.bfloat16" or len(grad["shape"]) != 2 or len(grad["stride"]) != 2
                or type(grad["contiguous"]) is not bool or grad["async_finish"] is not True
                or grad["stream"] in ("", "0")):
            raise ValueError(f"rank {rank} unusable reverse BF16 gradient layout")


class StallTrace:
    """Bracket the first MLP backward and cached reverse dispatch on the pinned schedule."""

    def __init__(self, cuda, emit, active, target):
        self.cuda, self.emit, self.active, self.target = cuda, emit, active, target
        self.used = self.in_mlp = self.in_reverse = self.internode = False
        self.resident_seen = False
        self.final_sampled = False
        self.row_id = None
        self.counts = {}
        self.markers, self.record_errors = {}, {}

    def at_target(self):
        row = self.active()
        return row is not None and tuple(row[k] for k in ("call", "forward", "backward")) == self.target

    def log(self, boundary, **detail):
        self.emit("stall_span", **self.active(), boundary=boundary, monotonic_ns=monotonic_ns(), **detail)

    @contextmanager
    def span(self, boundary):
        index = self.counts.get(boundary, 0)
        self.counts[boundary] = index + 1
        # Local EP has sixteen experts; leave room for ancillary padding/recompute.
        if index >= 32:
            if index == 32:
                self.log(boundary, phase="limit", index=index)
            yield
            return
        self.log(boundary, phase="enter", index=index)
        try:
            yield
        except BaseException:
            self.log(boundary, phase="error", index=index)
            raise
        else:
            self.log(boundary, phase="return", index=index)

    def mark(self, label, stream):
        try:
            event = self.cuda.Event(enable_timing=False)
            event.record(stream)
            self.markers[label] = (str(stream.cuda_stream), event)
        except (RuntimeError, OSError) as exc:
            self.record_errors[label] = str(exc)

    def sample(self, boundary):
        statuses, errors = {}, {}
        for label, (stream, event) in reversed(self.markers.items()):
            try:
                complete = event.query()
            except (RuntimeError, OSError) as exc:
                complete = None
                errors[label] = str(exc)
            statuses[label] = dict(stream=stream, complete=complete)
        self.emit("stall_gpu", **self.active(), boundary=boundary, markers=statuses,
                  record_errors=dict(self.record_errors), query_errors=errors)

    def wrap_mlp_context(self, original):
        @contextmanager
        @wraps(original)
        def acquire(node, name=None):
            if not self.at_target() or self.used or name != "mlp backward":
                with original(node, name) as value:
                    yield value
                return
            self.used = True
            self.mark("mlp_wait:before", node.stream)
            with self.span("mlp_context"):
                with original(node, name) as value:
                    self.mark("mlp_wait:after", node.stream)
                    self.sample("mlp_body_enter")
                    self.in_mlp = True
                    try:
                        yield value
                    finally:
                        self.in_mlp = False
        return acquire

    def wrap_host(self, original, boundary, enabled):
        @wraps(original)
        def run(*args, **kwargs):
            if not enabled(*args):
                return original(*args, **kwargs)
            with self.span(boundary):
                return original(*args, **kwargs)
        return run

    def wrap_resident(self, original):
        @wraps(original)
        def backward(expert, *args, **kwargs):
            if not self.in_mlp or self.resident_seen:
                return original(expert, *args, **kwargs)
            self.resident_seen = True
            self.row_id = id(expert._backend.row)
            try:
                self.sample("resident_enter")
                with self.span("resident_backward"):
                    return original(expert, *args, **kwargs)
            finally:
                self.row_id = None
        return backward

    def wrap_sender_gemm(self, original):
        @wraps(original)
        def gemm(backend, operation, a, b, out, *args, **kwargs):
            if self.row_id == id(backend.row) and operation == "fc2_dgrad" and out:
                self.sample("fc2_entry")
                self.row_id = None
            return original(backend, operation, a, b, out, *args, **kwargs)
        return gemm

    def wrap_reverse(self, original, dependencies):
        @wraps(original)
        def backward(ctx, *args, **kwargs):
            if not self.at_target() or self.used or dependencies.node != "moe_combine backward":
                return original(ctx, *args, **kwargs)
            self.used = self.in_reverse = True
            try:
                with self.span("reverse_backward"):
                    return original(ctx, *args, **kwargs)
            finally:
                self.in_reverse = False
        return backward

    def wrap_capture(self, original):
        @wraps(original)
        def capture(*args, **kwargs):
            if not self.in_reverse:
                return original(*args, **kwargs)
            self.mark("producer_prefix", self.cuda.current_stream())
            with self.span("capture_ready_event"):
                return original(*args, **kwargs)
        return capture

    def wrap_internode(self, original):
        @wraps(original)
        def dispatch(buffer, x, handle=None, *args, **kwargs):
            if not self.in_reverse or handle is None:
                return original(buffer, x, handle, *args, **kwargs)
            stream = buffer.get_comm_stream()
            self.mark("uccl_prefix", stream)
            self.internode = True
            try:
                with self.span("cached_internode"):
                    value = original(buffer, x, handle, *args, **kwargs)
                self.mark("uccl_cached_return", stream)
                return value
            finally:
                self.internode = False
        return dispatch

    def wrap_existing_wait(self, original):
        @wraps(original)
        def wait(event, *args, **kwargs):
            if not self.in_reverse:
                return original(event, *args, **kwargs)
            stream = self.cuda.current_stream()
            self.mark("consumer:before", stream)
            with self.span("output_wait"):
                value = original(event, *args, **kwargs)
            self.mark("consumer:after", stream)
            return value
        return wait

    def wrap_native_args(self, original):
        @wraps(original)
        def stream_ptr(buffer, *args, **kwargs):
            if not self.internode:
                return original(buffer, *args, **kwargs)
            # Last argument evaluated by the pinned Python wrapper before its
            # native dispatch call. This does not observe inside the C++ call.
            with self.span("native_dispatch_args_ready"):
                value = original(buffer, *args, **kwargs)
                self.mark("uccl_native_entry", buffer.get_comm_stream())
                return value
        return stream_ptr

    def wrap_final_dispatch(self, original):
        @wraps(original)
        def dispatch(*args, **kwargs):
            if not self.at_target() or not self.used or self.final_sampled:
                return original(*args, **kwargs)
            self.final_sampled = True
            try:
                return original(*args, **kwargs)
            finally:
                self.sample("next_forward_dispatch")
                self.markers.clear()
        return dispatch


class GraphHandoffTrace:
    """Observe the actual autograd nodes without replacing the engine or custom op."""

    NODES = {"unpermute": "GeneratedBackwardFor_te_moe_unpermute_mask_map_fwd_defaultBackward",
             "resident": "_ResidentExpertFunctionBackward"}

    def __init__(self, cuda, emit, stall):
        self.cuda, self.emit, self.stall = cuda, emit, stall
        self.in_unpermute = False
        self.used = False
        self.gpu = StallTrace(cuda, lambda _, **kw: emit("handoff_gpu", **kw), stall.active, stall.target)

    @classmethod
    def find_nodes(cls, outputs):
        # Allow the viewless wrapper or a short view chain, but never walk the
        # whole model graph or keep references beyond the scoped backward.
        pending = [(getattr(t, "grad_fn", None), 0) for t in outputs]
        seen, matches = set(), {}
        while pending and len(seen) < 24:
            node, depth = pending.pop(0)
            if node is None or node in seen or depth > 6:
                continue
            seen.add(node)
            for name, expected in cls.NODES.items():
                if node.name() == expected:
                    if name in matches:
                        raise ValueError(f"ambiguous {name} autograd node")
                    matches[name] = node
            if node is not matches.get("resident"):
                pending.extend((edge[0], depth + 1) for edge in node.next_functions)
        if matches.keys() != cls.NODES.keys():
            raise ValueError("missing native unpermute or resident autograd node")
        return matches

    def log(self, boundary, phase, **detail):
        self.emit("handoff_host", **self.stall.active(), boundary=boundary, phase=phase,
                  monotonic_ns=monotonic_ns(), **detail)

    def node_hook(self, boundary):
        def observe(*_):
            self.log(boundary, "enter")
            self.in_unpermute = boundary == "unpermute_pre"
            self.gpu.mark(boundary, self.cuda.current_stream())
            if boundary == "resident_pre":
                self.gpu.sample("resident_pre")
        return observe

    def wrap_autograd(self, original):
        @wraps(original)
        def backward(node, outputs, output_grad):
            if not self.stall.in_mlp or self.used or node.name != "mlp":
                return original(node, outputs, output_grad)
            self.used = True
            handles = []
            try:
                try:
                    children = self.find_nodes(outputs)
                    handles.append(children["unpermute"].register_prehook(self.node_hook("unpermute_pre")))
                    handles.append(children["unpermute"].register_hook(self.node_hook("unpermute_post")))
                    handles.append(children["resident"].register_prehook(self.node_hook("resident_pre")))
                    self.log("bind", "return", nodes={k: v.name() for k, v in children.items()})
                    del children
                except (ValueError, RuntimeError) as exc:
                    for handle in handles:
                        handle.remove()
                    handles.clear()
                    self.log("bind", "error", error=str(exc))
                return original(node, outputs, output_grad)
            finally:
                for handle in reversed(handles):
                    handle.remove()
                self.in_unpermute = False
        return backward

    def wrap_permute(self, original):
        @wraps(original)
        def permute(*args, **kwargs):
            if not self.in_unpermute:
                return original(*args, **kwargs)
            self.log("te_permute", "enter")
            self.gpu.mark("te_permute_before", self.cuda.current_stream())
            result = original(*args, **kwargs)
            self.log("te_permute", "return")
            self.gpu.mark("te_permute_after", self.cuda.current_stream())
            return result
        return permute


def verify_handoff_trace(rows, rank):
    """Require the no-probability unpermute path and usable stream observations."""
    host = [r for r in rows if r["event"] == "handoff_host"]
    gpu = [r for r in rows if r["event"] == "handoff_gpu"]
    expected = [("bind", "return"), ("unpermute_pre", "enter"), ("te_permute", "enter"),
                ("te_permute", "return"), ("unpermute_post", "enter"), ("resident_pre", "enter")]
    if ([(r["boundary"], r["phase"]) for r in host] != expected
            or host[0].get("nodes") != GraphHandoffTrace.NODES
            or [r["monotonic_ns"] for r in host] != sorted(r["monotonic_ns"] for r in host)
            or any(tuple(r[k] for k in ("call", "forward", "backward")) != LATER_SENDER_CALL for r in host + gpu)):
        raise ValueError(f"rank {rank} missing native autograd handoff boundaries")
    labels = {"unpermute_pre", "te_permute_before", "te_permute_after", "unpermute_post", "resident_pre"}
    if (len(gpu) != 1 or gpu[0]["boundary"] != "resident_pre" or gpu[0]["record_errors"]
            or gpu[0]["query_errors"] or set(gpu[0]["markers"]) != labels
            or any(type(m["complete"]) is not bool or not m["stream"] for m in gpu[0]["markers"].values())):
        raise ValueError(f"rank {rank} unusable autograd handoff GPU evidence")


def verify_stall_trace(rows, rank):
    """Require bounded, usable subphase records on the sender or final-stage rank."""
    sender = rank in PRESSURE_RANKS
    target = LATER_SENDER_CALL if sender else LATER_TARGETS[3]
    spans = [r for r in rows if r["event"] == "stall_span"]
    samples = [r for r in rows if r["event"] == "stall_gpu"]
    if any(tuple(r.get(k) for k in ("call", "forward", "backward")) != target for r in spans + samples):
        raise ValueError(f"rank {rank} stall observation outside target call")
    times = [r.get("monotonic_ns") for r in spans]
    if any(type(t) is not int for t in times) or times != sorted(times):
        raise ValueError(f"rank {rank} invalid stall host clocks")
    required = ({"mlp_context", "mlp_autograd", "resident_backward", "weights", "padding", "destinations", "row_quantize"}
                if sender else {"reverse_backward", "capture_ready_event", "cached_internode",
                                "native_dispatch_args_ready", "output_wait"})
    counts, stack = {}, []
    for row in spans:
        name, phase, index = (row.get(k) for k in ("boundary", "phase", "index"))
        if phase == "enter" and index == counts.get(name, 0) and index < 32:
            counts[name] = index + 1
            stack.append((name, index))
        elif phase == "return" and stack and stack[-1] == (name, index):
            stack.pop()
        else:
            raise ValueError(f"rank {rank} invalid or incomplete stall host span")
    if stack or not required <= counts.keys():
        raise ValueError(f"rank {rank} missing stall host boundaries")
    labels = ({"mlp_wait:before", "mlp_wait:after"} if sender else
              {"producer_prefix", "uccl_prefix", "uccl_native_entry", "uccl_cached_return",
               "consumer:before", "consumer:after"})
    expected = ["mlp_body_enter", "resident_enter", "fc2_entry"] if sender else ["next_forward_dispatch"]
    if ([r["boundary"] for r in samples] != expected or any(
            r["record_errors"] or r["query_errors"] or set(r["markers"]) != labels
            or any(type(m["complete"]) is not bool or not m["stream"] for m in r["markers"].values())
            for r in samples)):
        raise ValueError(f"rank {rank} unusable stall GPU markers")


def verify_sender_phases(rows, rank, target=LATER_SENDER_CALL, live_nodes=False):
    """Require valid sender phase coverage, including an explicit zero-compile observation."""
    seen = [r for r in rows if r["event"].startswith("later_sender_")]
    if (not seen or seen[0]["event"] != "later_sender_enter" or seen[-1]["event"] != "later_sender_exit"
            or seen[-1].get("returned") is not True
            or any(tuple(r.get(k) for k in ("call", "forward", "backward")) != target for r in seen)):
        raise ValueError(f"rank {rank} missing later sender phase coverage")
    callbacks = [r for r in seen if r["event"] == "later_sender_callback"]
    times = [r.get("monotonic_ns") for r in seen]
    compiles = sum(r["event"] == "later_sender_compile" for r in seen)
    if ([(r.get("boundary"), r.get("phase")) for r in callbacks]
            != [(name, phase) for name in SENDER_CALLBACKS for phase in ("enter", "exit")]
            or any(r.get("returned") is not True for r in callbacks[1::2])
            or any(type(t) is not int for t in times) or times != sorted(times)
            or not 0 <= compiles <= 2 * SENDER_COMPILE_LIMIT or seen[-1].get("compile_events") != compiles):
        raise ValueError(f"rank {rank} unusable later sender phase observations")
    if live_nodes:
        waits = [r for r in seen if r["event"] == "later_sender_chunk_wait"]
        if ([r["phase"] for r in waits] != ["enter", "exit"] or waits[1].get("returned") is not True
                or waits[1]["monotonic_ns"] > next(r["monotonic_ns"] for r in callbacks
                                                    if r["boundary"] == "post_forward")):
            raise ValueError(f"rank {rank} missing call-8 sender chunk wait evidence")
        nodes = [r for r in seen if r["event"] == "later_sender_node"]
        opened, closed = {}, set()
        for r in nodes:
            index, name = r["index"], r["node"]
            if r["phase"] == "enter" and index == len(opened):
                opened[index] = name
            elif r["phase"] == "exit" and index in opened and index not in closed and opened[index] == name:
                closed.add(index)
            else:
                raise ValueError(f"rank {rank} unusable call-8 sender node order")
        if not opened or closed != set(opened):
            raise ValueError(f"rank {rank} missing call-8 sender node evidence")


class BackwardDrainTrace:
    """Bound host tracing to the last backward calls and optimizer boundaries."""

    PHASES = ("grad_finalize", "grad_norm", "optimizer_step")
    MAX_SPANS_PER_CALL = 64

    def __init__(self, emit, active):
        self.emit, self.active = emit, active
        self.counts = {}

    def context(self):
        row = self.active()
        if (row is not None and DRAIN_FIRST_CALL <= row["call"] <= DRAIN_LAST_CALL
                and row["forward"] is None and row["backward"] is not None):
            return row

    @contextmanager
    def span(self, boundary, context):
        key = context.get("call")
        index = self.counts.get(key, 0)
        self.counts[key] = index + 1
        if index >= self.MAX_SPANS_PER_CALL:
            if index == self.MAX_SPANS_PER_CALL:
                self.emit("drain_span_limit", **context)
            yield
            return
        fields = dict(context, boundary=boundary, index=index)
        self.emit("drain_enter", **fields, monotonic_ns=monotonic_ns())
        returned = False
        try:
            yield
            returned = True
        finally:
            self.emit("drain_exit", **fields, returned=returned, monotonic_ns=monotonic_ns())

    def wrap_context(self, original):
        @contextmanager
        @wraps(original)
        def acquire(node, name=None):
            context = self.context()
            if context is not None and name and name.endswith(" backward"):
                with self.span(name, context), original(node, name) as value:
                    yield value
            else:
                with original(node, name) as value:
                    yield value
        return acquire

    def wrap_wgrad(self, original):
        @wraps(original)
        def run(node, *args, **kwargs):
            context = self.context()
            if context is None:
                return original(node, *args, **kwargs)
            with self.span(f"{node.name} wgrad", context):
                return original(node, *args, **kwargs)
        return run

    def wrap_phase(self, original, name):
        @wraps(original)
        def run(*args, **kwargs):
            with self.span(name, {}):
                return original(*args, **kwargs)
        return run


def verify_backward_drain(rows, rank):
    """A successful update must pass the tail schedule and real optimizer phases."""
    pending, seen, phases = {}, set(), []
    calls = {r["call"]: r["backward"] for r in rows if r["event"] == "schedule_exit"
             and DRAIN_FIRST_CALL <= r["call"] <= DRAIN_LAST_CALL and r["forward"] is None}
    for row in rows:
        event = row["event"]
        if event == "drain_span_limit":
            raise ValueError(f"rank {rank} exhausted drain trace budget")
        if event not in ("drain_enter", "drain_exit"):
            continue
        call, name = row.get("call"), row["boundary"]
        key = call, row["index"]
        if call is None:
            if event == "drain_enter":
                phases.append(name)
        elif call not in calls or row.get("forward") is not None or row["backward"] != calls[call]:
            raise ValueError(f"rank {rank} drain trace outside final backward")
        if event == "drain_enter" and key not in seen:
            seen.add(key)
            pending[key] = row
        elif (event == "drain_exit" and key in pending and pending[key]["boundary"] == name
              and row["monotonic_ns"] >= pending[key]["monotonic_ns"] and row["returned"] is True):
            del pending[key]
        else:
            raise ValueError(f"rank {rank} invalid drain trace span")
    if (not calls or pending or phases != list(BackwardDrainTrace.PHASES)
            or {c for c, _ in seen if c is not None} != calls.keys()):
        raise ValueError(f"rank {rank} missing backward drain or optimizer evidence")


def verify_compile_threads(effective):
    """Reject a missing or ignored Inductor limit before the diagnostic runs."""
    requested = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if requested != str(COMPILE_THREADS) or effective != COMPILE_THREADS:
        raise ValueError(f"expected {COMPILE_THREADS} Inductor compile threads, "
                         f"requested {requested!r}, effective {effective}")


def trace_boundary(original, name, observe, active, describe=None, describe_result=None):
    """Log entry and exit of a native call without synchronizing or retaining its tensors."""
    @wraps(original)
    def wrapped(*args, **kwargs):
        context = active()
        if context is None:
            return original(*args, **kwargs)
        detail = describe(*args, **kwargs) if describe is not None else {}
        observe(f"{name}_enter", **context, **detail)
        result = original(*args, **kwargs)
        detail = describe_result(result) if describe_result is not None else {}
        observe(f"{name}_exit", **context, **detail)
        return result

    return wrapped


def install(emit, head_chunk, replay, layers, notify_trace=False, dependency_trace=False,
            prefix_trace=False, pair_trace=False, ck_seed=None, later_trace=False, later_phase=False,
            reverse_ready=False, stall_trace=False, handoff_trace=False, extended_timeout=False,
            reduced_drain=False):
    """Observe native overlap schedule boundaries without adding GPU synchronization."""
    import torch
    from torch._inductor import config as inductor
    from megatron.core import parallel_state as mpu
    from megatron.core.fusions import fused_cross_entropy
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.pipeline_parallel.utils import ScheduleNode
    from megatron.core.rerun_state_machine import get_rerun_state_machine
    from megatron.core.transformer.moe import token_dispatcher
    from megatron.training import training

    verify_compile_threads(inductor.compile_threads)
    emit("compiler_config", inductor_compile_threads=inductor.compile_threads)
    original_step = training.train_step
    original_combined = schedules.combined_1f1b_schedule_for_interleaved_pipelining
    last_hip = float("-inf")
    calls, updates = 0, 0
    rank = int(os.environ["RANK"])
    active_call = None
    replay_seen = False
    dependencies = None
    if dependency_trace and rank in TRACE_RANKS:
        from megatron.core.pipeline_parallel.utils import ScheduleNode, get_comm_stream

        dependencies = DependencyTrace(torch.cuda, emit, get_comm_stream)
    pair = None
    if pair_trace and (rank in PAIR_SENDERS or rank in TRACE_RANKS):
        from megatron.core.pipeline_parallel.utils import get_comm_stream

        pair = P2PPairTrace(torch.cuda, emit, lambda: active_call, rank, get_comm_stream, dependencies)
    later = LaterP2PTrace(torch.cuda, emit, rank, lambda: active_call) if later_trace and rank >= 16 else None
    sender_phases, final_phases = None, None
    if (later_phase and rank in PRESSURE_RANKS) or (handoff_trace and rank in PAIR_SENDERS):
        from torch._dynamo.callback import callback_handler
        sender_phases = (SenderPhaseTrace(emit, callback_handler) if rank in PRESSURE_RANKS else
                         SenderPhaseTrace(emit, callback_handler, CALL8_SENDER, live_nodes=True))
    if later_phase and rank in TRACE_RANKS:
        final_phases = DependencyTrace(torch.cuda, emit, get_comm_stream, target=LATER_TARGETS[3])
    readiness = (ReverseReadyTrace(torch.cuda, emit, rank, lambda: active_call)
                 if reverse_ready and (rank in PRESSURE_RANKS or rank in TRACE_RANKS) else None)
    stall = (StallTrace(torch.cuda, emit, lambda: active_call, readiness.target)
             if stall_trace and readiness is not None else None)
    handoff = GraphHandoffTrace(torch.cuda, emit, stall) if handoff_trace and rank in PRESSURE_RANKS else None
    drain = BackwardDrainTrace(emit, lambda: active_call) if reduced_drain else None

    def snapshot(event, **extra):
        cuda = torch.cuda
        emit(event, allocated=cuda.memory_allocated(), reserved=cuda.memory_reserved(),
             peak_allocated=cuda.max_memory_allocated(), peak_reserved=cuda.max_memory_reserved(), **extra)

    def hip():
        nonlocal last_hip
        last_hip = monotonic()
        try:
            free, total = torch.cuda.mem_get_info()
        except (RuntimeError, OSError) as exc:
            emit("hip_snapshot_unavailable", error=str(exc))
        else:
            emit("hip_snapshot", free_bytes=free, total_bytes=total)

    @wraps(original_combined)
    def combined(*args, **kwargs):
        nonlocal calls, active_call
        calls += 1
        progress = dict(call=calls, forward=kwargs["f_virtual_microbatch_id"],
                        backward=kwargs["b_virtual_microbatch_id"])
        snapshot("schedule_enter", **progress)
        active_call = progress
        try:
            if pair is not None:
                pair.begin()
                if kwargs.get("post_forward") is not None and (
                    pair.at(8, 7, 0) if pair.sender else pair.at(7, 6, 1)
                ):
                    kwargs["post_forward"] = pair.wrap_forward_post(kwargs["post_forward"])
            if dependencies is not None:
                dependencies.begin(progress)
            if later is not None:
                later.callbacks(kwargs)
            if final_phases is not None:
                final_phases.begin(progress)
            if readiness is not None:
                readiness.callbacks(kwargs)
            if sender_phases is None:
                result = original_combined(*args, **kwargs)
            else:
                with sender_phases.capture(progress, kwargs):
                    result = original_combined(*args, **kwargs)
        finally:
            if dependencies is not None:
                dependencies.finish("before_target")
            if later is not None:
                later.finish("before_target")
            if final_phases is not None:
                final_phases.finish("before_target")
            if readiness is not None:
                readiness.finish()
            active_call = None
        snapshot("schedule_exit", **progress)
        if monotonic() - last_hip >= 5:
            hip()
        return result

    def logits_input(tensor):
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), bytes=tensor.numel() * tensor.element_size())

    def head_context():
        return active_call if rank in TRACE_RANKS and calls <= TRACE_CALL_LIMIT else None

    def dispatch_context():
        if replay and rank in PRESSURE_RANKS and calls == PRESSURE_CALL:
            return active_call
        return head_context()

    def observe_replay(event, **extra):
        nonlocal replay_seen
        snapshot(event, **extra)
        if event == "fc1_replay_exit":
            replay_seen = True

    targets = (
        (GPTModel, "_postprocess", "output_head", None, None, head_context),
        (LanguageModule, "compute_language_model_loss", "ce",
          lambda self, labels, logits: logits_input(logits), None, head_context),
        (fused_cross_entropy, "calculate_logits_max", "ce_fp32_logits",
          lambda vocab_parallel_logits: logits_input(vocab_parallel_logits),
          lambda result: logits_input(result[0]), head_context),
        (token_dispatcher, "fused_fp8_dispatch_fwd", "fnuz_dispatch",
          lambda x_fp8, *args, **kwargs: logits_input(x_fp8), None, dispatch_context),
    )

    @wraps(original_step)
    def step(forward, data, model, optimizer, scheduler, config, schedule, iteration=None):
        nonlocal updates
        verify_compile_threads(inductor.compile_threads)
        args = training.get_args()
        verify_effective_head(args, config, head_chunk)
        verify_effective_replay(args, config, replay)
        if layers == REDUCED_LAYERS:
            verify_reduced_model(args, config)
        geometry = {k: getattr(mpu, f"get_{k}_parallel_world_size")() for k in GEOMETRY}
        if (geometry != GEOMETRY or get_num_microbatches() != MICRO or iteration != 0 or updates
                or (later_trace and mpu.get_pipeline_model_parallel_rank() != rank // 16)
                or args.train_iters != 1 or args.global_batch_size != BATCH or args.micro_batch_size != MBS
                or args.seq_length != SEQ or args.num_layers != layers or config.num_layers != layers
                or args.consumed_train_samples != 0
                or args.load is not None or args.save is not None or args.fp8 is not None
                or not args.overlap_grad_reduce or not args.align_grad_reduce
                or args.main_grads_dtype != torch.float32 or args.params_dtype != torch.bfloat16
                or not isinstance(optimizer, LayerWiseDistributedOptimizer)
                or schedule is not schedules.forward_backward_pipelining_with_interleaving
                or not (config.moe_fp8_dispatch_fnuz and config.moe_resident_fp8_fnuz
                        and config.delay_wgrad_compute and config.overlap_moe_expert_parallel_comm
                        and config.overlap_p2p_comm and not config.gradient_accumulation_fusion
                        and config.cuda_graph_impl == "none"
                        and config.microbatch_group_size_per_vp_stage == 4)):
            raise ValueError("unexpected one-update full-layout diagnostic recipe")
        snapshot("update_enter", pp_rank=mpu.get_pipeline_model_parallel_rank(),
                 inductor_compile_threads=inductor.compile_threads, geometry=geometry,
                 num_layers=layers, lm_head_token_chunk_size=head_chunk, **{REPLAY_FIELD: replay})
        hip()
        rerun = get_rerun_state_machine()
        before = rerun.rerun_count
        schedules.combined_1f1b_schedule_for_interleaved_pipelining = combined
        originals = []
        try:
            def hook(module, attr, wrap):
                if isinstance(module, type) and isinstance(module.__dict__.get(attr), staticmethod):
                    original = module.__dict__[attr]
                    replacement = staticmethod(wrap(original.__func__))
                else:
                    original = getattr(module, attr)
                    replacement = wrap(original)
                originals.append((module, attr, original))
                setattr(module, attr, replacement)

            if ck_seed is not None and ck_training_mapping_required(rank, reduced_drain):
                backends = importlib.import_module(
                    "transformer_engine.pytorch.attention.dot_product_attention.backends")
                original_bwd = backends.fused_attn_bwd
                ck_seen = False

                @wraps(original_bwd)
                def ck_backward(*a, **kw):
                    nonlocal ck_seen
                    value = original_bwd(*a, **kw)
                    if not ck_seen:
                        maps = Path("/proc/self/maps").read_text()
                        if str(ck_seed[0]) in maps:
                            verify_ck_training(ck_seed, emit, maps)
                            ck_seen = True
                    return value

                originals.append((backends, "fused_attn_bwd", original_bwd))
                backends.fused_attn_bwd = ck_backward
            if pair is not None:
                from megatron.core.pipeline_parallel import combined_1f1b

                hook(torch.distributed, "isend" if pair.sender else "irecv", pair.wrap_native_post)
                if pair.sender:
                    from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan

                    hook(TransformerModelChunkSchedulePlan, "wait_current_stream", pair.wrap_chunk_wait)
                    hook(combined_1f1b, "combined_forward_backward_step", pair.wrap_backward_step)
                else:
                    hook(schedules, "deallocate_output_tensor", pair.wrap_deallocation)
            if rank in TRACE_RANKS or (replay and rank in PRESSURE_RANKS):
                for module, attr, name, describe, describe_result, context in targets:
                    if rank not in TRACE_RANKS and name != "fnuz_dispatch":
                        continue
                    fn = getattr(module, attr)
                    originals.append((module, attr, fn))
                    setattr(module, attr, trace_boundary(fn, name, snapshot, context,
                                                          describe, describe_result))
            if replay:
                from megatron.core.transformer.moe.resident_fp8 import ResidentFP8Experts

                fn = ResidentFP8Experts._replay_fc1
                originals.append((ResidentFP8Experts, "_replay_fc1", fn))
                ResidentFP8Experts._replay_fc1 = trace_boundary(
                    fn, "fc1_replay", observe_replay, lambda: None if replay_seen else active_call
                )
            if notify_trace and rank in TRACE_RANKS:
                from deep_ep import Buffer

                observer = NotifyTrace(torch.cuda, emit, lambda: active_call)
                for module, attr, wrap in (
                    (Buffer, "get_dispatch_layout", observer.wrap_layout),
                    (token_dispatcher, "fused_fp8_dispatch_fwd", observer.wrap_dispatch),
                ):
                    fn = getattr(module, attr)
                    originals.append((module, attr, fn))
                    setattr(module, attr, wrap(fn))
                if dependencies is not None:
                    for module, attr, wrap in (
                        (ScheduleNode, "stream_acquire_context", dependencies.wrap_context),
                        (Buffer, "dispatch", dependencies.wrap_buffer_dispatch),
                        (token_dispatcher, "fused_fp8_dispatch_fwd", dependencies.wrap_target),
                    ):
                        fn = getattr(module, attr)
                        originals.append((module, attr, fn))
                        setattr(module, attr, wrap(fn))
                    if prefix_trace:
                        from megatron.core.pipeline_parallel import combined_1f1b

                        hook(combined_1f1b, "combined_forward_backward_step",
                             lambda fn: dependencies.wrap_prefix_step(fn, torch.autograd, pair))
            if later is not None:
                hook(torch.distributed, "isend", lambda fn: later.wrap_native(fn, "isend"))
                hook(torch.distributed, "irecv", lambda fn: later.wrap_native(fn, "irecv"))
                if later.target is not None:
                    from deep_ep import Buffer

                    observer = NotifyTrace(torch.cuda, emit, lambda: active_call, later.target)
                    hook(Buffer, "get_dispatch_layout", observer.wrap_layout)
                    hook(token_dispatcher, "fused_fp8_dispatch_fwd", observer.wrap_dispatch)
                    hook(token_dispatcher, "fused_fp8_dispatch_fwd", later.wrap_dispatch)
            if sender_phases is not None:
                hook(ScheduleNode, "stream_acquire_context", sender_phases.wrap_context)
                if handoff_trace and rank in PAIR_SENDERS:
                    hook(TransformerModelChunkSchedulePlan, "wait_current_stream",
                         lambda fn: sender_phases.wrap_chunk_wait(fn, torch.cuda.current_stream, get_comm_stream))
            if final_phases is not None:
                hook(ScheduleNode, "stream_acquire_context", final_phases.wrap_context)
                hook(Buffer, "dispatch", final_phases.wrap_buffer_dispatch)
                hook(token_dispatcher, "fused_fp8_dispatch_fwd", final_phases.wrap_target)
                hook(combined_1f1b, "combined_forward_backward_step",
                     lambda fn: final_phases.wrap_prefix_step(fn, torch.autograd))
            if readiness is not None:
                if readiness.sender:
                    from megatron.core.transformer.moe.resident_fp8 import _NativeBackend

                    hook(_NativeBackend, "gemm", readiness.wrap_gemm)
                else:
                    from megatron.core.transformer.moe.fused_a2a import FusedCombine

                    hook(FusedCombine, "backward", lambda fn: readiness.wrap_reverse(fn, final_phases))
            if stall is not None:
                if readiness.sender:
                    from megatron.core.tensor_parallel import random
                    from megatron.core.transformer.moe.resident_fp8 import _Rows
                    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

                    hook(ScheduleNode, "stream_acquire_context", stall.wrap_mlp_context)
                    hook(ScheduleNode, "default_backward_func",
                         lambda fn: stall.wrap_host(fn, "mlp_autograd", lambda node, *_: stall.in_mlp and node.name == "mlp"))
                    hook(random.CheckpointWithoutOutput, "_recompute",
                         lambda fn: stall.wrap_host(fn, "norm_recompute", lambda *_: stall.in_mlp))
                    hook(random, "_get_share_storage",
                         lambda fn: stall.wrap_host(fn, "share_storage", lambda *_: stall.in_mlp))
                    hook(ResidentFP8Experts, "_backward", stall.wrap_resident)
                    hook(ResidentFP8Experts, "_get_weights",
                         lambda fn: stall.wrap_host(fn, "weights", lambda *_: stall.row_id is not None))
                    hook(_Rows, "pad", lambda fn: stall.wrap_host(fn, "padding", lambda *_: stall.row_id is not None))
                    hook(ResidentFP8Experts, "_destinations",
                         lambda fn: stall.wrap_host(fn, "destinations", lambda *_: stall.row_id is not None))
                    hook(Float8BlockQuantizer, "quantize_impl",
                         lambda fn: stall.wrap_host(fn, "row_quantize", lambda q, *_: id(q) == stall.row_id))
                    hook(_NativeBackend, "gemm", stall.wrap_sender_gemm)
                    if handoff is not None:
                        from transformer_engine.pytorch.triton import permutation

                        hook(ScheduleNode, "default_backward_func", handoff.wrap_autograd)
                        hook(permutation, "permute_with_mask_map", handoff.wrap_permute)
                else:
                    from deep_ep.utils import EventOverlap
                    from megatron.core.transformer.moe import fused_a2a

                    hook(FusedCombine, "backward", lambda fn: stall.wrap_reverse(fn, final_phases))
                    hook(fused_a2a, "EventHandle", stall.wrap_capture)
                    hook(Buffer, "internode_dispatch", stall.wrap_internode)
                    hook(Buffer, "_ll_compute_stream_ptr", stall.wrap_native_args)
                    hook(EventOverlap, "current_stream_wait", stall.wrap_existing_wait)
                    hook(token_dispatcher, "fused_fp8_dispatch_fwd", stall.wrap_final_dispatch)
            if extended_timeout and later is not None and later.target is not None:
                timeout_trace = ExtendedTimeoutTrace(torch.cuda, emit, later, stall)
                hook(token_dispatcher, "fused_fp8_dispatch_fwd", timeout_trace.wrap)
            if drain is not None:
                from megatron.core.models.gpt.fine_grained_callables import TransformerLayerNode

                hook(ScheduleNode, "stream_acquire_context", drain.wrap_context)
                hook(TransformerLayerNode, "backward_dw", drain.wrap_wgrad)
                for owner, attr, boundary in (
                    (config, "finalize_model_grads_func", "grad_finalize"),
                    (LayerWiseDistributedOptimizer, "prepare_grad_norm", "grad_norm"),
                    (LayerWiseDistributedOptimizer, "step_after_grad_norm", "optimizer_step"),
                ):
                    if not callable(getattr(owner, attr)):
                        raise ValueError(f"missing reduced drain phase: {boundary}")
                    hook(owner, attr, lambda fn, name=boundary: drain.wrap_phase(fn, name))
            result = original_step(forward, data, model, optimizer, scheduler, config, schedule, iteration)
        finally:
            if pair is not None:
                pair.stop.set()
            for module, attr, fn in reversed(originals):
                setattr(module, attr, fn)
            schedules.combined_1f1b_schedule_for_interleaved_pipelining = original_combined
        loss, skip, save, early_exit, _, norm, _, _ = result
        losses = {k: float(v) for k, v in loss.items()}
        if (skip or save or early_exit or rerun.rerun_count != before or rerun.discarded_batches
                or norm is None or not math.isfinite(norm) or norm <= 0 or not calls
                or set(losses) != ({"lm loss"} if mpu.is_pipeline_last_stage(ignore_virtual=True) else set())
                or not all(math.isfinite(v) for v in losses.values())):
            raise ValueError("one-update diagnostic did not reach a qualified update")
        snapshot("update_exit", calls=calls, grad_norm=float(norm), losses=losses)
        hip()
        updates += 1
        return result

    training.train_step = step
    return lambda: updates


def run_training(output, source, entrypoint, argv):
    """Persist progress incrementally, including the last entered schedule call."""
    rejected = {"--load", "--save", "--pretrained-checkpoint", "--finetune", "--async-save", "--profile",
                "--log-device-memory-used"}
    if int(os.environ["WORLD_SIZE"]) != WORLD or any(a.split("=", 1)[0] in rejected for a in argv):
        raise ValueError("diagnostic requires 64 ranks without checkpoint I/O or profiling")
    head_chunk = head_chunk_size(argv)
    replay = replay_requested(argv)
    reduced_drain = reduced_drain_requested(os.environ)
    layers = diagnostic_depth(argv, head_chunk, replay, reduced_drain)
    notify_trace = notify_trace_requested(os.environ, layers, head_chunk, replay, reduced_drain)
    dependency_trace = dependency_trace_requested(os.environ, notify_trace)
    prefix_trace = prefix_trace_requested(os.environ, dependency_trace)
    pair_trace = pair_trace_requested(os.environ, prefix_trace)
    ck_prewarm = ck_prewarm_requested(os.environ, pair_trace)
    later_trace = later_trace_requested(os.environ, ck_prewarm)
    later_phase = later_phase_requested(os.environ, later_trace)
    reverse_ready = reverse_ready_requested(os.environ, later_phase)
    stall_trace = stall_trace_requested(os.environ, reverse_ready)
    handoff_trace = handoff_trace_requested(os.environ, stall_trace)
    extended_timeout = extended_timeout_requested(os.environ, handoff_trace)
    if stall_trace and json.loads((output.parent / "resolved.json").read_text())["source"]["commit"] != STALL_SOURCE:
        raise ValueError("stall trace requires the pinned signed readiness source")
    rank = int(os.environ["RANK"])
    with (output / f"gpu-progress-rank-{rank}.jsonl").open("x") as stream:
        emit_lock = Lock()

        def emit(event, **fields):
            with emit_lock:
                if not stream.closed:
                    print(json.dumps(dict(unix_ns=time_ns(), rank=rank, event=event, **fields),
                                     separators=(",", ":")), file=stream, flush=True)

        observer = {}
        emit("entry")
        ck_seed = None
        try:
            if ck_prewarm:
                ck_seed = prewarm_ck(output, rank, emit)
            train(output, source, entrypoint, argv,
                  before_run=lambda: observer.update(
                      count=install(emit, head_chunk, replay, layers, notify_trace, dependency_trace,
                                    prefix_trace, pair_trace, ck_seed, later_trace, later_phase, reverse_ready,
                                    stall_trace, handoff_trace, extended_timeout, reduced_drain)))
            import torch

            if observer["count"]() != 1:
                raise ValueError("expected exactly one observed update")
            torch.cuda.synchronize()
            if ck_seed is not None and ck_training_mapping_required(rank, reduced_drain):
                verify_ck_training(ck_seed)
            torch.distributed.destroy_process_group()
        except BaseException as exc:
            emit("error", error=f"{type(exc).__name__}: {exc}")
            raise
        emit("completed")


def check(run):
    """On success, require all ranks' durable evidence; failure traces remain separate."""
    output = run / "output"
    result = dict(status="failed", scope="one cold full-layout update; no checkpoint, cache or throughput verdict")
    try:
        frozen = json.loads((run / "resolved.json").read_text())
        mapped = set(frozen["procedure"]["memory_mapping_ranks"])
        chunk = head_chunk_size(frozen["args"])
        replay = replay_requested(frozen["args"])
        reduced_drain = reduced_drain_requested(frozen.get("environment", {}))
        layers = diagnostic_depth(frozen["args"], chunk, replay, reduced_drain)
        reduced = layers == REDUCED_LAYERS
        notify_trace = notify_trace_requested(frozen.get("environment", {}), layers, chunk, replay, reduced_drain)
        dependency_trace = dependency_trace_requested(frozen.get("environment", {}), notify_trace)
        prefix_trace = prefix_trace_requested(frozen.get("environment", {}), dependency_trace)
        pair_trace = pair_trace_requested(frozen.get("environment", {}), prefix_trace)
        ck_prewarm = ck_prewarm_requested(frozen.get("environment", {}), pair_trace)
        later_trace = later_trace_requested(frozen.get("environment", {}), ck_prewarm)
        later_phase = later_phase_requested(frozen.get("environment", {}), later_trace)
        reverse_ready = reverse_ready_requested(frozen.get("environment", {}), later_phase)
        stall_trace = stall_trace_requested(frozen.get("environment", {}), reverse_ready)
        handoff_trace = handoff_trace_requested(frozen.get("environment", {}), stall_trace)
        extended_timeout = extended_timeout_requested(frozen.get("environment", {}), handoff_trace)
        if stall_trace and frozen["source"]["commit"] != STALL_SOURCE:
            raise ValueError("stall trace requires the pinned signed readiness source")
        if ck_prewarm:
            result["scope"] = "one full-layout update with node-verified CK prewarm; no checkpoint or throughput verdict"
            handoffs = []
            for node in range(WORLD // 4):
                stem = f"ck-prewarm-node-{node}"
                report = json.loads((output / f"{stem}.json").read_text())
                handoffs.append(ck_handoff(report, node))
                cold, warm = (json.loads((output / f"{stem}-{phase}.json").read_text()) for phase in ("cold", "warm"))
                cold_log, warm_log = ((output / f"{stem}-{phase}.log").read_text(errors="replace")
                                      for phase in ("cold", "warm"))
                if cold["hostname"] != report["hostname"]:
                    raise ValueError(f"node {node} CK report and builder host disagree")
                verify_ck_pair(cold, warm, cold_log, warm_log, report["cold"]["object_path"], report["artifact"])
            verify_ck_handoffs(handoffs)
        if reduced:
            scope = ("one 24-layer drain control with node-verified CK prewarm; not full Chonk"
                     if reduced_drain else "one cold 24-layer reduced-model update; not full Chonk")
            result.update(scope=scope, num_layers=layers, lm_head="chunked" if chunk else "dense", fc1_replay=replay)
        observed_mappings = []
        for r in range(WORLD):
            gpu = [json.loads(line) for line in (output / f"gpu-progress-rank-{r}.jsonl").read_text().splitlines()]
            host = [json.loads(line) for line in (output / f"host-memory-rank-{r}.jsonl").read_text().splitlines()]
            imports = json.loads((output / f"imports-{r}.json").read_text())
            if (not gpu or gpu[0]["event"] != "entry" or gpu[-1]["event"] != "completed"
                    or any(row["rank"] != r for row in gpu)
                    or sum(row["event"] == "update_exit" for row in gpu) != 1
                    or [row["inductor_compile_threads"] for row in gpu
                        if row["event"] in ("compiler_config", "update_enter")] != [COMPILE_THREADS] * 2
                    or not host or host[-1]["event"] != "exit:0" or not imports["verified_modules"]):
                raise ValueError(f"rank {r} missing completed diagnostic evidence")
            if chunk and [row.get("lm_head_token_chunk_size") for row in gpu
                          if row["event"] == "update_enter"] != [chunk]:
                raise ValueError(f"rank {r} missing effective LM-head chunk size {chunk}")
            if reduced:
                recipe = [(row.get("num_layers"), row.get("lm_head_token_chunk_size"), row.get(REPLAY_FIELD))
                          for row in gpu if row["event"] == "update_enter"]
                if recipe != [(layers, chunk, replay)] or recipe[0][2] is not replay:
                    description = "drain-control" if reduced_drain else "dense-head"
                    raise ValueError(f"rank {r} missing effective reduced {description} recipe")
            if replay:
                effective = [row.get(REPLAY_FIELD) for row in gpu if row["event"] == "update_enter"]
                if len(effective) != 1 or effective[0] is not True:
                    raise ValueError(f"rank {r} missing effective FC1 replay flag")
                verify_replay_trace(gpu, r)
            if r % 4 == 0 and not any(row.get("node_memory") for row in host):
                raise ValueError(f"rank {r} missing node memory samples")
            if r in mapped:
                mappings = [row["vm_mapping"] for row in host if "vm_mapping" in row]
                if not mappings:
                    raise ValueError(f"rank {r} missing mapping sample attempts")
                if any(row["status"] == "sampled" and row["mappings"] for row in mappings):
                    observed_mappings.append(r)
            if pair_trace and (r in PAIR_SENDERS or r in TRACE_RANKS):
                verify_pair_trace(gpu, r)
            if later_trace and r >= 16:
                verify_later_trace(gpu, r)
            if later_phase and r in PRESSURE_RANKS:
                verify_sender_phases(gpu, r)
            if reverse_ready and (r in PRESSURE_RANKS or r in TRACE_RANKS):
                verify_reverse_ready(gpu, r)
            if stall_trace and (r in PRESSURE_RANKS or r in TRACE_RANKS):
                verify_stall_trace(gpu, r)
            if handoff_trace and r in PRESSURE_RANKS:
                verify_handoff_trace(gpu, r)
            if handoff_trace and r in PAIR_SENDERS:
                verify_sender_phases(gpu, r, CALL8_SENDER, live_nodes=True)
            if extended_timeout and r >= 32:
                verify_extended_timeout_trace(gpu, r)
            if reduced_drain:
                verify_backward_drain(gpu, r)
            if r in TRACE_RANKS:
                if notify_trace:
                    verify_notify_trace(gpu, r)
                if dependency_trace:
                    verify_dependency_trace(gpu, r, prefix_trace, pair_trace)
                if later_phase:
                    verify_dependency_trace(gpu, r, True, False, target=LATER_TARGETS[3])
                if chunk or reduced:
                    try:
                        verify_head_trace(gpu, chunk)
                    except ValueError as exc:
                        raise ValueError(f"rank {r}: {exc}") from exc
                elif not any(row["event"] == "ce_fp32_logits_exit" for row in gpu):
                    raise ValueError(f"rank {r} missing native loss boundary")
            log = (output / f"rank-{r}.log").read_text(errors="replace")
            if "UCCL_EP_TRANSPORT=cxi initialized" not in log:
                raise ValueError(f"rank {r} missing native CXI initialization")
            if ck_prewarm:
                verify_ck_rank(gpu, log, r, handoffs[r // 4], ck_training_mapping_required(r, reduced_drain))
        if (output / "checkpoints").exists():
            raise ValueError("unexpected checkpoint directory")
        result.update(status="passed", ranks=WORLD, updates=1, observed_mapping_ranks=observed_mappings)
        if chunk:
            result.update(lm_head_token_chunk_size=chunk, lm_head_first_call=FIRST_HEAD_CALL,
                          lm_head_chunks_per_rank=SEQ * MBS // chunk, lm_head_verified_ranks=list(TRACE_RANKS))
        if replay:
            result.update(fc1_replay_verified_ranks=list(range(WORLD)),
                          pp_rank_1_warmup_call=PRESSURE_CALL, pp_rank_1_verified_ranks=list(PRESSURE_RANKS))
        if notify_trace:
            result.update(notify_trace_ranks=list(TRACE_RANKS), notify_trace_call=NOTIFY_CALL)
        if dependency_trace:
            result.update(dependency_trace_ranks=list(TRACE_RANKS), dependency_trace_call=NOTIFY_CALL)
        if prefix_trace:
            result.update(prefix_trace_ranks=list(TRACE_RANKS), prefix_trace_call=NOTIFY_CALL)
        if pair_trace:
            result.update(p2p_pair_senders=list(PAIR_SENDERS), p2p_pair_receivers=list(TRACE_RANKS))
        if ck_prewarm:
            result.update(ck_prewarm_nodes=handoffs, ck_prewarm_verified_ranks=list(range(WORLD)))
            if reduced_drain:
                result["ck_training_mapped_ranks"] = [r for r in range(WORLD) if ck_training_mapping_required(r, True)]
        if later_trace:
            result.update(later_p2p_edges=list(LATER_EDGES), later_p2p_verified_ranks=list(range(16, WORLD)),
                          later_dispatch_targets=LATER_TARGETS)
        if later_phase:
            result.update(later_sender_phase_ranks=list(PRESSURE_RANKS), later_final_phase_ranks=list(TRACE_RANKS))
        if reverse_ready:
            result.update(reverse_ready_sender_ranks=list(PRESSURE_RANKS), reverse_ready_final_ranks=list(TRACE_RANKS))
        if stall_trace:
            result.update(stall_sender_ranks=list(PRESSURE_RANKS), stall_final_ranks=list(TRACE_RANKS))
        if handoff_trace:
            result.update(autograd_handoff_ranks=list(PRESSURE_RANKS), call8_sender_phase_ranks=list(PAIR_SENDERS))
        if extended_timeout:
            result.update(uccl_cpu_timeout_seconds=600, dispatch_sample_ranks=list(range(32, WORLD)))
        if reduced_drain:
            result.update(backward_drain_verified_ranks=list(range(WORLD)))
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        write_json(output / "checks.json", result)


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"] and len(sys.argv) >= 6:
        run_training(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif sys.argv[1:2] == ["check"] and len(sys.argv) == 3:
        check(Path(sys.argv[2]))
    elif (sys.argv[1:2] == ["ck-prewarm-worker"] and len(sys.argv) == 3
          and os.environ.get(CK_PREWARM) == CK_NODE_VERIFIED):
        ck_prewarm_worker(Path(sys.argv[2]))
    else:
        raise SystemExit("expected train, check, or an enabled CK prewarm worker")
