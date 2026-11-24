"""Profile local rank 0 of every node with the image's rocprofv3, gated by
iteration markers instead of a wall-clock window.

Derived from rocprof_local_rank0.py (same structure, same switches, same
per-node profiled-rank contract). The one functional difference is the
collection mode:

- COLLECTION_MODE="period" reproduces the previous behavior: rocprofv3 -P
  start_delay:duration:repeat. That window is anchored to profiler init and
  must be retuned per regime; it missed steady state twice because the first
  full-iteration warmup stretched from T0+1164 (run d3968c8d, morning) to
  T0+1639 (run f054d220, afternoon) between two identical 10-iteration runs.

- COLLECTION_MODE="marker" (default) replaces the wall-clock window with
  rocprofv3's selected-regions mode: --selected-regions makes the profiler
  record only between roctxProfilerResume(0) and roctxProfilerPause(0). The
  profiled rank's wrapper installs an iteration trigger that counts
  forward-backward schedule calls (megatron.core.pipeline_parallel.schedules.
  get_forward_backward_func, invoked once per train_step) and fires
  Resume before schedule call ITER_RESUME and Pause after schedule call
  ITER_PAUSE returns, so the trace holds ITER_PAUSE-ITER_RESUME complete
  training iterations regardless of how long warmup takes.

Why this is safe for the host-memory headroom the old deferral protected:
both OOM kills happened while collection was ACTIVE during torch.compile
warmup. In selected-regions mode nothing is recorded before the first
Resume, so the InterceptQueue stays idle through warmup by construction -
there is no deferred-start delay left to tune.

Marker call path: rocprofv3 preloads librocprofiler-sdk-roctx.so when
--marker-trace is set, and a ctypes dlopen of its soname returns that
already-loaded instance, so the trigger binds the live profiling session
without any path setup. The image's roctx python bindings are tried second
because they may bind the legacy roctracer library rather than the SDK one.

Failure modes are deliberately inert: if the schedule hook cannot be
installed or the marker calls cannot be bound, the watchdog records the
failure, and if it cannot even fall back the profiler simply records nothing
(empty DBs, training unaffected). All trigger state is written to
trigger-<rank>.json, which check() requires in marker mode, so a silent
no-op cannot pass validation.

The watchdog covers a broken trigger: if Resume has not fired
WATCHDOG_RESUME_AFTER seconds into training it fires Resume itself and
Pauses COLLECTION_DURATION seconds later (the old calibrated window length).
If the marker Resume fired but the marker Pause never arrives within
COLLECTION_DURATION + WATCHDOG_PAUSE_GRACE seconds, the watchdog Pauses.
Watchdog Resume/Pause run on the watchdog thread, not the training thread;
if this rocprofv3 build turns out to treat the pause state as per-thread,
the evidence in trigger-<rank>.json shows exactly which thread fired what.

Per-rank semantics: each of the 16 profiled ranks counts its own schedule
calls and gates its own session. Pipeline stages are staggered anyway; what
the trace guarantees is that every profiled rank captured the same
iteration span of its own training. Non-profiled ranks never run the
before_run hook, so nothing about their execution changes.
"""

import ctypes
import functools
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import types

from apertus_amd.artifacts import write_json
from apertus_amd.entrypoint import train

# Testable escape hatch; the container environment never sets this variable.
ROCPROFV3 = os.environ.get("APERTUS_ROCPROFV3", "/opt/venv/bin/rocprofv3")
PROFILED_LOCAL_RANK = 0
CHILD_GUARD = "APERTUS_ROCPROF_CHILD"

# Optional instrumentation switches; flip to True to re-enable a feature.
MARKER_TRACE_ENABLED = True   # append --marker-trace to the rocprofv3 command
NVTX_RANGES_ENABLED = True    # enable Megatron's named NVTX/ROCTx ranges
TOLERANT_NVTX_POP = True      # recover instead of raise on NVTX pop mismatch
CALL_SAMPLING_ENABLED = False  # per-rank Python call-stack sampler thread
CALL_SAMPLE_INTERVAL = 0.1

# Collection mode: "marker" gates the profiler on iteration markers via
# --selected-regions; "period" is the legacy wall-clock -P window.
COLLECTION_MODE = "marker"

# Marker-mode window: schedule calls are 0-indexed train_step iterations.
# ITER_RESUME=3, ITER_PAUSE=7 captures complete iterations 4-7 (backward
# recompiles finished by iteration 3-4 in every observed regime; checkpoint
# I/O starts at iteration 10 with --save-interval 10). Four iterations at
# the observed ~20-40 s cadence is the same 100-160 s window the -P
# calibration targeted, so the 2 GiB perfetto buffer math is unchanged
# (~7.8 MiB/s fill, ~1.2 GiB peak).
ITER_RESUME = 3
ITER_PAUSE = 7

# Watchdog fallback, anchored at before_run time (python start + a few
# seconds). Slowest observed warmup ended T0+1639 with checkpoint I/O at
# T0+1909; a T0+1800 resume lands in steady iterations 5-7 of the slow
# regime. Regimes that warm up faster simply get a later-than-ideal window -
# the watchdog exists so a broken trigger still yields a usable trace.
WATCHDOG_RESUME_AFTER = 1800.0
WATCHDOG_PAUSE_GRACE = 60.0

# Legacy -P window, used by COLLECTION_MODE="period" and as the watchdog's
# fallback window length. Calibration history lives in rocprof_local_rank0.py;
# summary: warmup ends ranged T0+574 (5-iter fast regime) to T0+1639 (10-iter
# afternoon run), steady iterations ~24-40 s apart, checkpoint I/O from
# T0+1453 (morning) / T0+1909 (afternoon).
COLLECTION_START_DELAY = 1200
COLLECTION_DURATION = 150
COLLECTION_REPEAT = 1

# Kernel dispatches, HIP runtime API and RCCL collectives; memory-copy tracing
# was removed because its system-memory InterceptQueue records contributed to a
# host-memory OOM kill on a profiled node. The marker flag must stay on in
# marker mode: rocprofv3 preloads librocprofiler-sdk-roctx.so only under
# --marker-trace, and that preload is what carries roctxProfilerResume/Pause.
TRACE_FLAGS = ["--kernel-trace", "--hip-runtime-trace", "--rccl-trace"]
if MARKER_TRACE_ENABLED:
    TRACE_FLAGS.append("--marker-trace")
# Requested perfetto budget, per profiled rank: 2 GiB buffer with discard fill
# policy and a 16 MiB shared-memory hint, so a full buffer drops later records
# instead of growing or blocking.
PERFETTO_FLAGS = ["--perfetto-buffer-size", "2097152",
                  "--perfetto-shmem-size-hint", "16384",
                  "--perfetto-buffer-fill-policy", "discard"]


def profiler_window_flags(mode: str = COLLECTION_MODE) -> list[str]:
    """Return the rocprofv3 flags selecting when collection is active.

    The two modes are mutually exclusive in rocprofv3 (enforced by its CLI
    validation), so exactly one is emitted.
    """
    if mode == "marker":
        # --selected-regions-ref-count makes nested resume/pause pairs
        # reference-counted instead of fatal; the trigger fires one pair, but
        # this keeps an accidental double-fire from ending the window early.
        return ["--selected-regions", "--selected-regions-ref-count"]
    if mode == "period":
        return ["-P", f"{COLLECTION_START_DELAY}:{COLLECTION_DURATION}:{COLLECTION_REPEAT}"]
    raise ValueError(f"unknown collection mode: {mode}")


def profiled_ranks(world: int, gpus_per_node: int) -> list[int]:
    """Ranks whose Slurm local rank equals PROFILED_LOCAL_RANK."""
    return [rank for rank in range(world) if rank % gpus_per_node == PROFILED_LOCAL_RANK]


def bind_roctx_profiler_control(evidence: dict) -> tuple | None:
    """Bind roctxProfilerResume/Pause from the SDK library rocprofv3 preloaded.

    The soname dlopen returns the already-preloaded instance when
    --marker-trace set LD_PRELOAD (rocprofv3:1926), which is the only
    guarantee needed that the calls reach the live session. The image's roctx
    python package is the fallback because its libpyroctx may link the legacy
    roctracer library, whose pause/resume would not reach the SDK profiler.
    """
    candidates = ["librocprofiler-sdk-roctx.so.1",
                  "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/"
                  "librocprofiler-sdk-roctx.so.1"]
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            resume = library.roctxProfilerResume
            pause = library.roctxProfilerPause
            for function in (resume, pause):
                function.argtypes = [ctypes.c_uint32]
                function.restype = ctypes.c_int
            evidence["trigger_path"] = f"ctypes:{candidate}"
            return resume, pause
        except (OSError, AttributeError) as error:
            evidence.setdefault("trigger_errors", []).append(f"ctypes {candidate}: {error!r}")
    try:
        import roctx  # image python bindings; may wrap legacy roctx
        evidence["trigger_path"] = "python:roctx"
        return roctx.profilerResume, roctx.profilerPause
    except (ImportError, AttributeError) as error:
        evidence.setdefault("trigger_errors", []).append(f"roctx module: {error!r}")
    evidence["trigger_path"] = "none"
    return None


def install_iteration_trigger(evidence: dict) -> None:
    """Gate the profiler session on training-iteration markers.

    Wraps megatron.core.pipeline_parallel.schedules.get_forward_backward_func,
    which Megatron's train_step calls exactly once per iteration, and fires
    Resume before call ITER_RESUME and Pause after call ITER_PAUSE returns.
    Runs in the before_run hook, before pretrain_gpt.py imports the training
    modules, so their from-imports bind the counting wrapper; the sys.modules
    walk mirrors patch_nvtx_range_pop for anything already imported.
    """
    import megatron.core.pipeline_parallel.schedules as schedules

    control = bind_roctx_profiler_control(evidence)
    state = {"n": 0, "resumed": False, "paused": False, "times": []}
    resumed, paused = threading.Event(), threading.Event()
    evidence["schedule_hook"] = "megatron.core.pipeline_parallel.schedules.get_forward_backward_func"
    evidence["resume_iter"] = ITER_RESUME
    evidence["pause_iter"] = ITER_PAUSE
    evidence["trigger_state"] = state

    def write_trigger() -> None:
        write_json(Path(os.environ["APERTUS_OUTPUT_DIR"]) / f"trigger-{os.environ['RANK']}.json",
                   {"rank": int(os.environ["RANK"]), "hostname": socket.gethostname(),
                    "pid": os.getpid(), "resumed": state["resumed"],
                    "paused": state["paused"], **evidence,
                    "iterations": state["times"]})

    def fire(bound, label: str, done: threading.Event) -> None:
        bound(ctypes.c_uint32(0))
        state[label] = True
        state["times"].append({label: time.strftime("%H:%M:%S"), "call": state["n"]})
        done.set()
        write_trigger()

    if control is not None and ITER_RESUME < ITER_PAUSE:
        resume, pause = control
        original = schedules.get_forward_backward_func

        @functools.wraps(original)
        def counting_get_forward_backward_func():
            schedule = original()

            def counted_schedule(*args, **kwargs):
                # Count schedule executions, not get_forward_backward_func
                # calls: warmup or other callers may fetch the schedule
                # without running it, and only executions are iterations.
                index = state["n"]
                state["n"] += 1
                state["times"].append({"call": index, "start": time.strftime("%H:%M:%S")})
                if index == ITER_RESUME and not state["resumed"]:
                    fire(resume, "resumed", resumed)
                try:
                    return schedule(*args, **kwargs)
                finally:
                    if index + 1 == ITER_PAUSE and not state["paused"]:
                        fire(pause, "paused", paused)
            return counted_schedule

        schedules.get_forward_backward_func = counting_get_forward_backward_func
        for module in list(sys.modules.values()):
            if module is not None and getattr(module, "get_forward_backward_func", None) is original:
                module.get_forward_backward_func = counting_get_forward_backward_func
        evidence["trigger_installed"] = True
    else:
        evidence["trigger_installed"] = False

    def watchdog() -> None:
        """Fallback resume/pause if the marker path never completes."""
        if not resumed.wait(timeout=WATCHDOG_RESUME_AFTER):
            if control is None:
                evidence["watchdog"] = "no marker control bound; gave up"
                write_trigger()
                return
            fire(resume, "resumed", resumed)
            evidence["watchdog"] = f"marker resume missing; fell back at +{WATCHDOG_RESUME_AFTER}s"
            time.sleep(COLLECTION_DURATION)
        elif not paused.wait(timeout=COLLECTION_DURATION + WATCHDOG_PAUSE_GRACE):
            evidence["watchdog"] = "marker pause missing; fell back after grace"
            time.sleep(0)
        else:
            return
        if control is not None and not state["paused"]:
            fire(pause, "paused", paused)

    threading.Thread(target=watchdog, name="profiler-marker-watchdog", daemon=True).start()


def enable_nvtx_ranges() -> None:
    """Register the nvtx shim and enable Megatron's named ranges before imports.

    Runs as the entrypoint's before_run hook, strictly before pretrain_gpt.py and
    therefore before megatron.core.utils is imported: decorated call sites decide
    at decoration time, so late enablement would leave them inert.
    """
    import torch

    if "nvtx" not in sys.modules:
        shim = types.ModuleType("nvtx")

        def annotate(message=None, color=None):  # noqa: ARG001 - color has no ROCTx effect
            def decorator(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    torch.cuda.nvtx.range_push(message or func.__qualname__)
                    try:
                        return func(*args, **kwargs)
                    finally:
                        torch.cuda.nvtx.range_pop()
                return wrapper
            return decorator

        shim.annotate = annotate
        sys.modules["nvtx"] = shim
    from megatron.core.utils import configure_nvtx_profiling
    configure_nvtx_profiling(True)

    if TOLERANT_NVTX_POP:
        patch_nvtx_range_pop()


def patch_nvtx_range_pop() -> None:
    """Make megatron.core.utils.nvtx_range_pop recover instead of raise.

    Megatron keeps its own message stack beside the torch NVTX stack and the
    pop helper raises on mismatch (megatron/core/utils.py:2365-2372). The KDA
    push/pop pairs are not exception-safe, so one caught transient exception
    inside a pushed region desynchronizes the message stack and the next
    schedule-level pop kills training. The replacement keeps Megatron's stack
    bookkeeping (so the trace range sequence stays balanced: push and pop still
    alternate) but treats a mismatch as a warning and an empty stack as a
    no-op, matching the lenient behavior of torch.cuda.nvtx.range_pop itself.
    """
    import logging

    import torch
    import megatron.core.utils as mc_utils

    def tolerant_nvtx_range_pop(msg=None, suffix=None) -> None:
        if not mc_utils._nvtx_enabled:
            return
        if msg is None:
            msg = mc_utils._nvtx_range_get_func_path()
        if suffix is not None:
            msg = f"{msg}.{suffix}"
        if not mc_utils._nvtx_range_messages:
            logging.getLogger(__name__).warning(
                "nvtx_range_pop(%s): stack empty; skipping torch pop", msg)
            return
        last_msg = mc_utils._nvtx_range_messages.pop()
        if msg is not None and msg != last_msg:
            logging.getLogger(__name__).warning(
                "nvtx_range_pop(%s): stack mismatch (expected %s); "
                "popping %s from torch stack", msg, last_msg, last_msg)
        torch.cuda.nvtx.range_pop()

    original = mc_utils.nvtx_range_pop
    mc_utils.nvtx_range_pop = tolerant_nvtx_range_pop

    # Rebind the name in every module that imported it before this patch.
    # `from megatron.core.utils import nvtx_range_pop` (schedules.py,
    # kimi_delta_attention.py, gated_delta_net.py, mlp.py, attention.py,
    # transformer_layer.py, muon_logging.py) binds the original function
    # object directly, so replacing only the module attribute leaves those
    # call sites strict. Modules imported after this patch pick up the
    # tolerant version automatically; this walk covers the ones already
    # loaded (megatron.core's package __init__ transitively imports several).
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "nvtx_range_pop", None) is original:
            module.nvtx_range_pop = tolerant_nvtx_range_pop


def sampler_loop(path: Path, interval: float, stop: threading.Event, failures: list) -> None:
    """Sample every interpreter thread's Python call stack into JSON lines."""
    try:
        with path.open("w", buffering=1) as stream:
            while not stop.is_set():
                names = {thread.ident: thread.name for thread in threading.enumerate()}
                threads = {}
                for identifier, frame in sys._current_frames().items():
                    stack = []
                    while frame is not None and len(stack) < 200:
                        code = frame.f_code
                        stack.append([code.co_filename, frame.f_lineno, code.co_name])
                        frame = frame.f_back
                    threads[str(identifier)] = {"name": names.get(identifier), "stack": stack}
                stream.write(json.dumps({"time": time.time(), "threads": threads},
                                        separators=(",", ":")) + "\n")
                stop.wait(interval)
    except Exception as error:  # Retained as run evidence; never kills training.
        failures.append(repr(error))


def sample_python_calls(output: Path, source: Path, entrypoint: str, argv: list[str],
                        before_run=None) -> None:
    """Run verified training on a profiled rank with one call-sampling thread."""
    rank = int(os.environ["RANK"])
    interval = float(os.environ.get("APERTUS_CALL_SAMPLE_INTERVAL", str(CALL_SAMPLE_INTERVAL)))
    if not interval > 0:
        raise ValueError("APERTUS_CALL_SAMPLE_INTERVAL must be positive")
    stop, failures = threading.Event(), []
    worker = threading.Thread(target=sampler_loop,
                              args=(output / f"call-sample-{rank}.jsonl", interval, stop, failures),
                              name="python-call-sampler", daemon=True)
    worker.start()
    completed = False
    try:
        train(output, source, entrypoint, argv, before_run=before_run)
        completed = True
    finally:
        stop.set()
        worker.join(timeout=30)
        write_json(output / f"call-sample-{rank}-summary.json", {
            "rank": rank, "hostname": socket.gethostname(), "pid": os.getpid(),
            "interval_seconds": interval, "training_completed": completed,
            "sampler_errors": failures})
    if failures:
        raise RuntimeError(f"python call sampling failed: {failures}")


def run_train(output: Path, source: Path, entrypoint: str, argv: list[str]) -> None:
    """Select the ordinary or profiler-relaunched execution path for this rank."""
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if local != PROFILED_LOCAL_RANK:
        train(output, source, entrypoint, argv)
        return
    if os.environ.get(CHILD_GUARD) == "1":
        os.environ["APERTUS_OUTPUT_DIR"] = str(output)
        evidence = {
            "collection_mode": COLLECTION_MODE,
            "resume_iter": ITER_RESUME, "pause_iter": ITER_PAUSE,
            "call_sampling": "enabled" if CALL_SAMPLING_ENABLED else "disabled"}

        def before_run():
            if NVTX_RANGES_ENABLED:
                enable_nvtx_ranges()
            if COLLECTION_MODE == "marker":
                # After enable_nvtx_ranges: the schedules import then binds the
                # tolerant nvtx pop, and the sys.modules walk in the trigger
                # covers modules imported before either patch. A trigger failure
                # must not kill a 16-node job: record it, write the trigger file
                # so check() reports the gap, and keep training unprofiled.
                try:
                    install_iteration_trigger(evidence)
                except Exception as error:  # noqa: BLE001 - evidence, not crash
                    evidence["trigger_installed"] = False
                    evidence["trigger_errors"] = [f"install: {error!r}"]
                finally:
                    write_json(output / f"trigger-{rank}.json",
                               {"rank": rank, "hostname": socket.gethostname(),
                                "pid": os.getpid(), "resumed": False,
                                "paused": False, **evidence})
            write_json(output / f"ranges-{rank}.json", {
                "rank": rank, "hostname": socket.gethostname(), "pid": os.getpid(),
                "nvtx_ranges": "enabled" if NVTX_RANGES_ENABLED else "disabled",
                "configure": "megatron.core.utils.configure_nvtx_profiling(True)"
                             if NVTX_RANGES_ENABLED else "not requested",
                "nvtx_shim": "sys.modules['nvtx'] backed by torch.cuda.nvtx"
                             if NVTX_RANGES_ENABLED else "not registered",
                "marker_trace": "enabled" if MARKER_TRACE_ENABLED else "disabled",
                **evidence})
        if CALL_SAMPLING_ENABLED:
            sample_python_calls(output, source, entrypoint, argv, before_run=before_run)
        else:
            train(output, source, entrypoint, argv, before_run=before_run)
        return
    directory = output / "rocprof"
    directory.mkdir(exist_ok=True)
    inner = [sys.executable, str(Path(__file__).resolve()), "train",
             str(output), str(source), entrypoint, *argv]
    command = [ROCPROFV3, *TRACE_FLAGS, *PERFETTO_FLAGS, "-d", str(directory),
               *profiler_window_flags(), "--", *inner]
    # Written before exec so the launch intent exists even if execv fails.
    write_json(directory / f"profiled-{rank}.json", {
        "rank": rank, "local_rank": local, "hostname": socket.gethostname(),
        "pid": os.getpid(), "rocprofv3": ROCPROFV3, "argv": command,
        "collection_mode": COLLECTION_MODE})
    os.environ[CHILD_GUARD] = "1"
    os.execv(ROCPROFV3, command)


def check(run: Path) -> None:
    """Require profiler launch intent, hook evidence and trace output per profiled rank."""
    settings = json.loads((run / "resolved.json").read_text())
    gpus = settings["resources"]["gpus_per_node"]
    world = settings["resources"]["nodes"] * gpus
    profiled = profiled_ranks(world, gpus)
    output = run / "output"
    missing = []
    for rank in profiled:
        if not (output / "rocprof" / f"profiled-{rank}.json").is_file():
            missing.append(f"profiled-{rank}.json")
        if not (output / f"ranges-{rank}.json").is_file():
            missing.append(f"ranges-{rank}.json")
        if COLLECTION_MODE == "marker":
            trigger = output / f"trigger-{rank}.json"
            if not trigger.is_file():
                missing.append(f"trigger-{rank}.json")
            else:
                state = json.loads(trigger.read_text())
                if not (state.get("resumed") and state.get("paused")):
                    missing.append(f"trigger-{rank}.json incomplete")
        if CALL_SAMPLING_ENABLED:
            samples = output / f"call-sample-{rank}.jsonl"
            if not samples.is_file() or samples.stat().st_size == 0:
                missing.append(f"call-sample-{rank}.jsonl")
    traces = sorted(path for path in (output / "rocprof").rglob("*")
                    if path.is_file() and not path.name.startswith("profiled-")) \
        if (output / "rocprof").is_dir() else []
    if len(traces) < len(profiled):
        missing.append(f"rocprof trace files {len(traces)} < {len(profiled)} profiled ranks")
    write_json(output / "checks.json", {
        "status": "failed" if missing else "passed",
        "scope": "rocprofv3 per-node launch evidence, instrumentation-switch states and "
                 "marker-trigger evidence; not numerical, checkpoint or memory validation",
        "profiled_ranks": profiled, "trace_file_count": len(traces),
        "trace_files": [str(path.relative_to(output)) for path in traces[:256]],
        "collection_mode": COLLECTION_MODE,
        "marker_trace": "enabled" if MARKER_TRACE_ENABLED else "disabled",
        "nvtx_ranges": "enabled" if NVTX_RANGES_ENABLED else "disabled",
        "call_sampling": "enabled" if CALL_SAMPLING_ENABLED else "disabled",
        "missing": missing})
    if missing:
        raise ValueError(f"rocprof evidence incomplete: {missing[:8]}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        run_train(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif sys.argv[1] == "check":
        check(Path(sys.argv[2]))
    else:
        raise ValueError(f"unknown operation: {sys.argv[1]}")
