"""Matched synthetic public-TE grouped forward timing; no training-speedup claim.

Fresh workers inherit the runtime-owned process group and retain private caches.
Only worker entry imports vendor modules; CPU checks must remain vendor-free.
"""

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
from statistics import median
import subprocess
import sys
import tempfile
import time

IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
PACKAGE = Path("/opt/venv/lib/python3.12/site-packages")
PREFIX = "transformer_engine/pytorch/"
SELECTOR = "NVTE_USE_GROUPED_GEMM_TRITON"
BASELINE = {"NVTE_USE_ROCM": "1", "NVTE_USE_CAST_TRANSPOSE_TRITON": "1", "NVTE_USE_HIPBLASLT": "1"}
ALIASES = ("general_grouped_gemm", "general_grouped_gemm_triton")
PROJECTIONS = (("fc1", 1792, 3840), ("fc2", 1920, 1792))
SPLITS = {"balanced": [2048] * 16, "ragged": [0, 1, 127, 128, 129, 511, 1023, 2047, 2048, 2049, 3071, 3072, 4095, 4096, 17, 33]}
CASES = tuple(f"{p}/{case}" for p, _, _ in PROJECTIONS for case in SPLITS)
VERSIONS = {"torch": "2.14.0+rocm10.1.0", "transformer-engine": "2.17.1+130099ce4", "triton": "3.8.0+git669b31ac.rocm10.1.0"}
OPERANDS_SHA = "11b6959f4647d13c636fc49d5b2bcfa634e9f5a414867a16e7ccd84fe41fc587"
PINS = {
    "transformer_engine/build_info.txt": (98, "891ae1c35e89a9d833eb624f09ac017dcf1256870df5c692795c5bfb21de4395"),
    "transformer_engine/libtransformer_engine.so": (401104592, "140dc626463ae6072121ca66a988c71954268f7b2c27b4d447894a97f2cba69a"),
    PREFIX + "cpp_extensions/gemm.py": (29761, "957974b26d32ab74f1521780d96e2d9e56a30e3a8420d7bb066c69cb4c6a5e9d"),
    PREFIX + "module/grouped_linear.py": (92320, "bc52bcbeb1e5c6e696c3869814d9e9bebe37d507a06beb5e16ccef43de988dea"),
    PREFIX + "triton_kernels/__init__.py": (177, "900f1106a90ff13f796e70567eb440431e1049b8700b1ef67a1bebda3d57b9b3"),
    PREFIX + "triton_kernels/cast.py": (6215, "35821855471c4e045c5409e711a04e6b848388e015d48d35f4ea4d4f784f7f5e"),
    PREFIX + "triton_kernels/cast_transpose.py": (46403, "a2fc9be19c263a93e257403cff0d5bd44464bc72a6285c53564f995f6175282a"),
    PREFIX + "triton_kernels/common.py": (3502, "6402b9fc0ed5c02172faf9e57a177c7a1bcfdc8bddb27ddb9f29cae233421c11"),
    PREFIX + "triton_kernels/gmm/__init__.py": (248, "e8d7e3343fa1056a020bb525eac28e06ad09d4b9d149ccb7b46c1e1d7f0ff51a"),
    PREFIX + "triton_kernels/gmm/configs/gfx942-GMM.json": (5493, "6e1d4efb9dd8c6b6c12a1148b558120ec6f96afbea4eba125c75cb3e7c8f009a"),
    PREFIX + "triton_kernels/gmm/gmm_common.py": (24152, "dea1fb56056958030a4e434d80a73897e796883269c1ad7a7998812ed4cd92de"),
    PREFIX + "triton_kernels/gmm/gmm_kernels.py": (23953, "55853bedd639b6a3bd58948e4146aada80e0e6685f22a9fc45f240be3d8f790e"),
    PREFIX + "triton_kernels/gmm/gmm_wrapper.py": (21002, "b6e8af2255e951ab22c2445da1fd071fc78e1090cb1ce5e11345f34b5f6350b6"),
    PREFIX + "triton_kernels/gmm/pid_preprocessing.py": (2774, "8c08db51222eb51cd77e3a5654993b4ad498b123a419bae003f13d06d1858cdf"),
    PREFIX + "triton_kernels/grouped_gemm.py": (7361, "96d765fa50f97f1e86784076e46b3dcf040c4ca3a5e3f5e9384dff418bbc6553"),
    PREFIX + "triton_kernels/layernorm.py": (15829, "c67b394ebd197fb542657749c4b463c6fae7153e3d27d273d5102eeb9ea14509"),
    PREFIX + "triton_kernels/norms_common.py": (21528, "11adf8980005e5937179803bbc3c95d9c0722561687d7b098f45ff723454eada"),
    PREFIX + "triton_kernels/rmsnorm.py": (18802, "cb7b9c75149694c65f53315a3d77453392f3e89e081e3f754bf3866e1ca15773"),
    PREFIX + "triton_kernels/utils.py": (3825, "4b95eb88f26018e0ec5312d38b9fee7c7c23737dfda9f25d2afabd5789b9839b"),
    "transformer_engine/transformer_engine_torch.cpython-312-x86_64-linux-gnu.so": (3438144, "1bdeb20270594556bc705e2b9842439925cf70ab03611145d388a92d4cc9f620"),
    "torch/_dynamo/utils.py": (214231, "e550c68c924e1c0babaf31fc629e900c1d3b6ecbcf10aa88442d39b405b7102a"),
    "torch/_inductor/metrics.py": (15211, "d3cb2560945d4e284c24c2728621a0a4c95eae70a36a31801c9292a2180c3e2e"),
    "triton/runtime/jit.py": (44747, "38b8151fcceec1a0210a696775b7c64d7fac5e9b69eee2dda7e93c1e2a377509"),
    "triton/runtime/autotuner.py": (21876, "119055c98dfef66a1ae8a72e4bdf44185993ee24902ced5086caa7f9de9cef16"),
}
COMMAND = ["python3", "-u", "/apertus/validation.py", "run"]
# CPU/source qualification only; scheduler execution requires a separate freeze approval.
IMPLEMENTATION_READY = True
DIMS = {name: (k, n) for name, k, n in PROJECTIONS}
CACHE = ('before_pre','after_pre','warm1','warm2','before_measure','after_measure','after_post')


def digest(path):
    """Hash bounded selected sources/ELFs, never the image or a compiler cache."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): h.update(block)
    return h.hexdigest()


def object_digest(value):
    """Hash canonical JSON for exact cross-worker operand identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def retain(path, record):
    """Publish this invocation's new record without masking a primary native error."""
    primary = sys.exception()
    try:
        temporary = path.with_suffix(".tmp")
        with temporary.open("x") as stream: json.dump(record, stream, indent=2, allow_nan=False)
        if path.exists(): raise FileExistsError(path)
        os.replace(temporary, path)
    except BaseException as error:
        if primary is None: raise
        primary.add_note(f"secondary evidence error: {error}")


def environment(env, selector=None):
    """Preserve exact image baseline flags and reject unreviewed backend overrides."""
    if selector not in (None, "0", "1"): raise ValueError("startup selector value")
    for k, v in env.items():
        if k.startswith(("TE_HIPBLASLT_", "PYTORCH_TUNABLEOP_", "HIPBLASLT_TUNING", "NVTE_USE_", "NVTE_GROUPED_")):
            if k in BASELINE and v == BASELINE[k]: continue
            if k != SELECTOR or v != selector: raise ValueError(f"unreviewed backend environment: {k}")
    if env.get(SELECTOR) != selector: raise ValueError("startup selector mismatch")
    return {k: env.get(k) for k in BASELINE}


def validate_launch(settings, env):
    """Admit only the separately reviewed ten-minute four-rank public-API protocol."""
    r, p = settings["resources"], settings["procedure"]
    wanted = dict(nodes=1, gpus_per_node=4, cpus_per_node=192, mpi="pmix", time="00:10:00",
                  reservation="cscs", cpu_bind="map_ldom:0,1,2,3", distribution="block:block")
    rank, local = int(env["SLURM_PROCID"]), int(env["SLURM_LOCALID"])
    if (settings["image"]["sha256"] != IMAGE or any(r.get(k) != v for k, v in wanted.items())
            or r.get("cpus_per_task") is not None or p["kind"] != "commands" or p["timeout"] != 510
            or p["commands"] != [COMMAND] or int(env["SLURM_NTASKS"]) != 4 or int(env["SLURM_NODEID"]) != 0
            or not 0 <= rank == local < 4 or len(os.sched_getaffinity(0)) != 48): raise ValueError("launch contract")
    return rank, local


def expected_provenance():
    """Return the JSON-native pinned component identity, without importing vendor code."""
    return dict(image=IMAGE, versions=VERSIONS, pins={k: list(v) for k, v in PINS.items()}, binary_attestation=False)


def provenance():
    """Verify canonical installed bytes before imports; not compiled-source attestation."""
    if {k: importlib.metadata.version(k) for k in VERSIONS} != VERSIONS: raise ValueError("package versions")
    for name, (size, checksum) in PINS.items():
        path = PACKAGE / name
        if path.resolve(strict=True) != path or path.stat().st_size != size or digest(path) != checksum:
            raise ValueError(f"installed pin: {name}")
    return expected_provenance()


def metrics_pass(m):
    """Apply the unchanged independent finite/full-output/element/RMS expert gates."""
    if set(m) != {"finite", "numel", "checked", "mismatches", "max_abs", "rms_error", "reference_rms"}: return False
    if m["finite"] is not True or any(type(m[k]) is not int for k in ("numel", "checked", "mismatches")): return False
    if m["numel"] < 0 or m["checked"] != m["numel"] or m["mismatches"] != 0: return False
    if any(type(m[k]) not in (int, float) or not math.isfinite(m[k]) or m[k] < 0 for k in ("max_abs", "rms_error", "reference_rms")): return False
    return m["rms_error"] <= 1e-6 + 2**-7 * m["reference_rms"] and (m["numel"] > 0 or m["max_abs"] == m["rms_error"] == m["reference_rms"] == 0)


@contextmanager
def observe(module, expected):
    """Count pre/post public consumers only; restore owned aliases even on failure."""
    if set(expected) != set(ALIASES) or any(getattr(module, k) is not v for k, v in expected.items()): raise ValueError("alias identity")
    counts, wrappers = dict.fromkeys(ALIASES, 0), {}
    try:
        for name, original in expected.items():
            def wrapped(*args, _name=name, _original=original, **kwargs):
                value = _original(*args, **kwargs); counts[_name] += 1; return value
            wrappers[name] = wrapped; setattr(module, name, wrapped)
        yield counts
    finally:
        for name, wrapper in wrappers.items():
            if getattr(module, name) is wrapper: setattr(module, name, expected[name])


def tensor_digest(tensor):
    """Hash exact BF16 words only outside warm/timed batches."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(dtype=__import__("torch").uint8).numpy().tobytes()).hexdigest()


def layout(tensor):
    """Record synthetic storage metadata, not a production layout capture."""
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype), device=str(tensor.device), offset=tensor.storage_offset())


def rotate(rank, block):
    """Use the same fixed case rotation in every chronological worker slot."""
    offset = (rank + block) % 4
    return CASES[offset:] + CASES[:offset]


def memory(torch, device):
    """Observe allocator-context peaks outside the timed forward loop."""
    return dict(allocated=torch.cuda.max_memory_allocated(device), reserved=torch.cuda.max_memory_reserved(device))


def calls(operation, count):
    """Issue ordinary public calls and release each output inside the common loop."""
    for _ in range(count):
        output = operation()
        del output


def sample(torch, operation, events, device, log_fd, clock=time.monotonic_ns):
    """Measure one fixed batch with common documented host and event endpoints."""
    start, end = events
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    log_before = os.fstat(log_fd).st_size
    start.record(); t0 = clock()
    calls(operation, 16)
    returned = clock(); end.record(); end.synchronize(); completed = clock()
    elapsed = start.elapsed_time(end)
    return dict(status="complete", batch=16, t0_ns=t0, return_ns=returned, complete_ns=completed,
                event_ms=elapsed, completion_us=(completed-t0)/16000, host_return_us=(returned-t0)/16000,
                device_span_us=elapsed*1000/16, peak_allocated=torch.cuda.max_memory_allocated(device),
                peak_reserved=torch.cuda.max_memory_reserved(device), log_before=log_before, log_after=os.fstat(log_fd).st_size)


def stable_cache(snapshot, selector, rank):
    """Reject ambiguous eager cache proof without manufacturing compilation."""
    keys = snapshot["gmm_keys"]
    if selector == "1" and (set(keys) != {str(rank)} or not keys[str(rank)]): raise ValueError("missing current-device grouped cache")
    if selector == "0" and keys: raise ValueError("unexpected baseline grouped cache")


def measure_protocol(torch, operations, phase, snapshot, record, log_fd):
    """Run the real fixed phase/timing sequence; injected functions enable CPU testing."""
    device, rank, selector = record["local_device"], record["rank"], record["selector"]
    events = {case: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for case in CASES}
    # Initialize event resources before any measured sample, without extra forwards.
    for start, end in events.values(): start.record(); end.record(); end.synchronize()
    record["cache"] = {"before_pre": snapshot()}; record["phase_memory"] = {}
    torch.cuda.reset_peak_memory_stats(device); phase("pre"); torch.cuda.synchronize(device)
    record["phase_memory"]["pre"] = memory(torch, device); record["cache"]["after_pre"] = snapshot()
    stable_cache(record["cache"]["after_pre"], selector, rank)
    torch.cuda.reset_peak_memory_stats(device)
    for chunk in range(2):
        for case in rotate(rank, chunk): calls(operations[case], 10)
        torch.cuda.synchronize(device); record["cache"][f"warm{chunk+1}"] = snapshot()
    record["warm_calls_per_case"] = [10, 10]; record["phase_memory"]["warm"] = memory(torch, device)
    record["cache"]["before_measure"] = snapshot()
    if any(record["cache"][key] != record["cache"]["after_pre"] for key in ("warm1", "warm2", "before_measure")):
        raise ValueError("fixed warm compiler/cache instability")
    measured_allocated = measured_reserved = 0
    for block in range(12):
        for case in rotate(rank, block):
            row = dict(case=case, block=block, status="started"); record["samples"].append(row)
            row.update(sample(torch, operations[case], events[case], device, log_fd))
            measured_allocated = max(measured_allocated, row["peak_allocated"])
            measured_reserved = max(measured_reserved, row["peak_reserved"])
    record["phase_memory"]["measure"] = dict(allocated=measured_allocated, reserved=measured_reserved)
    record["cache"]["after_measure"] = snapshot()
    torch.cuda.reset_peak_memory_stats(device); phase("post"); torch.cuda.synchronize(device)
    record["phase_memory"]["post"] = memory(torch, device); record["cache"]["after_post"] = snapshot()
    if any(v != record["cache"]["after_pre"] for k, v in record["cache"].items() if k != "before_pre"):
        raise ValueError("compiler/cache changed during measurement/postchecks")


def cache_reader(kernel_module, rank):
    """Bind one exact source-reviewed Heuristics-to-JITFunction chain, read-only."""
    jit = importlib.import_module("triton.runtime.jit")
    tuner = importlib.import_module("triton.runtime.autotuner")
    du = importlib.import_module("torch._dynamo.utils")
    im = importlib.import_module("torch._inductor.metrics")
    for cls, relative in ((jit.JITFunction, "triton/runtime/jit.py"), (tuner.Heuristics, "triton/runtime/autotuner.py")):
        if Path(inspect.getfile(cls)).resolve() != PACKAGE / relative: raise ValueError("cache class origin")
    wrapper = kernel_module.gmm_kernel
    if type(wrapper) is not tuner.Heuristics or type(vars(wrapper).get("fn")) is not jit.JITFunction: raise ValueError("grouped wrapper ABI")
    function = vars(wrapper)["fn"]
    if Path(inspect.getfile(function.fn)).resolve() != PACKAGE / (PREFIX + "triton_kernels/gmm/gmm_kernels.py"):
        raise ValueError("grouped kernel function origin")
    def snapshot():
        result = dict(dynamo={}, generated_kernel_count=im.generated_kernel_count, gmm_keys={})
        if type(im.generated_kernel_count) is not int or im.generated_kernel_count < 0 or not isinstance(du.counters, Mapping): raise ValueError("compiler counter ABI")
        for group in ("frames", "stats", "inductor", "graph_break"):
            values = du.counters.get(group, {})
            if not isinstance(values, Mapping) or any(type(k) is not str or type(v) is not int or v < 0 for k, v in values.items()): raise ValueError("compiler counter words")
            result["dynamo"][group] = dict(values)
        caches = vars(function).get("device_caches")
        if not isinstance(caches, Mapping): raise ValueError("grouped device cache ABI")
        for device, entry in caches.items():
            if not isinstance(entry, tuple) or not entry or not isinstance(entry[0], Mapping): raise ValueError("grouped cache entry ABI")
            result["gmm_keys"][str(device)] = sorted(hashlib.sha256(repr(k).encode()).hexdigest() for k in entry[0])
        return result
    return snapshot


def prepare_cases(torch, te, record):
    """Replay745622's exact CPU generator draw order, retaining fixed device buffers."""
    generator = torch.Generator(device="cpu").manual_seed(42)
    cases, operands = {}, {}
    for name, k, n in PROJECTIONS:
        layer = te.GroupedLinear(16, k, n, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=False,
                                device="cuda", single_grouped_weight=False, single_grouped_bias=False)
        weights = [torch.randn(n, k, dtype=torch.bfloat16, generator=generator)*0.02 for _ in range(16)]
        hashes = [tensor_digest(w) for w in weights]
        for i, w in enumerate(weights): getattr(layer, f"weight{i}").requires_grad_(False).copy_(w)
        for case, splits in SPLITS.items():
            key = f"{name}/{case}"; host = torch.randn(sum(splits), k, dtype=torch.bfloat16, generator=generator)*0.25
            operands[key] = dict(input=tensor_digest(host), weights=hashes)
            x = host.to(device="cuda"); sizes = torch.tensor(splits, dtype=torch.int32, device=x.device)
            if sizes.shape != (16,) or sizes.stride() != (1,): raise ValueError("row-count contract")
            cases[key] = dict(layer=layer, input=x, sizes=sizes, splits=list(splits), n=n)
    if object_digest(operands) != OPERANDS_SHA: raise ValueError("qualified operand digest mismatch")
    record["operands"] = operands
    return cases


def operand_hashes(cases):
    """Rehash actual resident device operands outside timing, including every weight."""
    return {key: dict(input=tensor_digest(c["input"]), weights=[tensor_digest(getattr(c["layer"], f"weight{i}")) for i in range(16)]) for key, c in cases.items()}


def numerical_phase(torch, cases, module, aliases, record, phase):
    """Check every expert against the unchanged full FP32 reference before/after timing."""
    record["checks"][phase] = []; record["output_hashes"][phase] = {}
    with observe(module, aliases) as counts:
        for key in CASES:
            c = cases[key]; x, layer, splits, n = c["input"], c["layer"], c["splits"], c["n"]
            y = layer(x, list(splits), m_splits_tensor=c["sizes"]); torch.cuda.synchronize()
            if tuple(y.shape) != (len(x), n) or y.dtype != torch.bfloat16 or layer.fp8 or layer.is_debug_iter() or layer.fuse_wgrad_accumulation:
                raise ValueError("public forward contract")
            if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32: raise ValueError("reference precision changed")
            record["layouts"][key] = dict(input=layout(x), weights=[layout(getattr(layer, f"weight{i}")) for i in range(16)],
                output=layout(y), row_counts=layout(c["sizes"]), source="canonical CPU generator seed42; hashes before CUDA transfer")
            experts, start = [], 0
            for i, rows in enumerate(splits):
                actual = y[start:start+rows].float(); reference = x[start:start+rows].float() @ getattr(layer, f"weight{i}").float().T
                finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all() and torch.isfinite(x[start:start+rows]).all() and torch.isfinite(getattr(layer, f"weight{i}")).all())
                diff = actual-reference
                m = dict(finite=finite, numel=rows*n, checked=rows*n, mismatches=int((diff.abs()>0.02+0.02*reference.abs()).sum().item()),
                    max_abs=diff.abs().max().item() if rows else 0., rms_error=diff.square().mean().sqrt().item() if rows else 0.,
                    reference_rms=reference.square().mean().sqrt().item() if rows else 0.)
                m = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in m.items()}
                experts.append(m); start += rows
            record["checks"][phase].append(dict(key=key, splits=splits, experts=experts))
            if not all(metrics_pass(m) for m in experts): raise ValueError("independent FP32 numerical gate")
            record["output_hashes"][phase][key] = tensor_digest(y)
            del x, y, actual, reference, diff
    record["branch"][phase] = dict(counts)
    selected = (4, 0) if record["selector"] == "0" else (0, 4)
    if counts != dict(zip(ALIASES, selected)): raise ValueError("selected consumer counts")
    if any(getattr(module, k) is not v for k, v in aliases.items()): raise ValueError("aliases not restored")
    record["aliases_restored"] = True


def worker(directory, rank, slot, selector):
    """Execute one private eager worker; all error paths retain raw evidence/caches."""
    record = dict(status="unqualified", rank=rank, local_device=rank, slot=slot, selector=selector, pid=os.getpid(), pgid=os.getpgrp(),
                  driver_sha256=digest(__file__), started_ns=time.monotonic_ns(), samples=[], checks={}, branch={}, output_hashes={}, layouts={})
    phase = None
    try:
        record["baseline_environment_initial"] = environment(os.environ, selector); record["provenance"] = provenance()
        from apertus_amd.runtime import preserve_affinity, require_affinity
        with preserve_affinity(record.setdefault("affinity", {})) as initial:
            import torch
            if torch.cuda.device_count() != 4 or not torch.version.hip: raise ValueError("all four ROCm devices required")
            torch.cuda.set_device(rank)
            if torch.cuda.current_device() != rank or not torch.cuda.get_device_properties(rank).gcnArchName.startswith("gfx942"): raise ValueError("local gfx942 device")
            import transformer_engine.pytorch as te
            module = importlib.import_module("transformer_engine.pytorch.module.grouped_linear")
            native = importlib.import_module("transformer_engine.pytorch.cpp_extensions.gemm")
            candidate = importlib.import_module("transformer_engine.pytorch.triton_kernels.grouped_gemm")
            kernels = importlib.import_module("transformer_engine.pytorch.triton_kernels.gmm.gmm_kernels")
            if torch.cuda.current_device() != rank: raise ValueError("device changed during TE imports")
        record["precision"] = dict(matmul=torch.get_float32_matmul_precision(), allow_tf32=torch.backends.cuda.matmul.allow_tf32)
        if record["precision"] != dict(matmul="highest", allow_tf32=False): raise ValueError("reference precision")
        aliases = dict(zip(ALIASES, (native.general_grouped_gemm, candidate.general_grouped_gemm_triton)))
        record["origins"] = {"GroupedLinear": inspect.getfile(te.GroupedLinear), **{k: inspect.getfile(v) for k, v in aliases.items()}}
        wanted = ("module/grouped_linear.py", "cpp_extensions/gemm.py", "triton_kernels/grouped_gemm.py")
        for key, relative in zip(("GroupedLinear", *ALIASES), wanted):
            if Path(record["origins"][key]).resolve() != PACKAGE / (PREFIX+relative): raise ValueError("callable origin")
        mapped = {line.split(maxsplit=5)[5] for line in Path("/proc/self/maps").read_text().splitlines() if len(line.split(maxsplit=5)) == 6}
        record["mapped_elfs"] = sorted(str(PACKAGE/name) for name in PINS if name.endswith(".so"))
        if not set(record["mapped_elfs"]) <= mapped: raise ValueError("pinned ELF mappings absent")
        snapshot = cache_reader(kernels, rank)
        with torch.no_grad():
            cases = prepare_cases(torch, te, record)
            record["operands_device_pre"] = operand_hashes(cases)
            if record["operands_device_pre"] != record["operands"]: raise ValueError("resident pre-timing operands differ")
            operations = {key: (lambda c=c: c["layer"](c["input"], list(c["splits"]), m_splits_tensor=c["sizes"])) for key, c in cases.items()}
            def phase(name):
                require_affinity(initial, f"before {name}")
                numerical_phase(torch, cases, module, aliases, record, name)
            measure_protocol(torch, operations, phase, snapshot, record, sys.stdout.fileno())
            record["operands_post"] = operand_hashes(cases)
        if record["operands_post"] != record["operands"]: raise ValueError("resident operands changed")
        record["baseline_environment_final"] = environment(os.environ, selector)
        if record["baseline_environment_initial"] != record["baseline_environment_final"]: raise ValueError("baseline flags changed")
        if provenance() != record["provenance"]: raise ValueError("pins changed")
        require_affinity(initial, "grouped timing completion")
        if object_digest(record["operands"]) != OPERANDS_SHA: raise ValueError("qualified operand digest mismatch")
        record["status"] = "passed"
        record["finished_ns"] = time.monotonic_ns()
        validate_worker(record, rank=rank, slot=slot, expected_sha=digest(__file__), expected_operands=record["operands"], expected_provenance=expected_provenance())
    except BaseException as error:
        record["status"] = "unqualified"
        record["error"] = f"{type(error).__name__}: {error}"
        if phase is not None and "pre" in record["branch"] and "post" not in record["branch"]:
            try:
                with torch.no_grad(): phase("post")
            except BaseException as secondary: error.add_note(f"postcheck after primary failure: {secondary}")
        raise
    finally:
        record["finished_ns"] = time.monotonic_ns(); retain(directory/"worker.json", record)


def quiescent():
    """Bounded own-runtime-PGID check; no child state or cache is deleted."""
    for index, entry in enumerate(Path("/proc").iterdir()):
        if index >= 8192: return False
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: fields = (entry/"stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError): continue
        if int(fields[2]) == os.getpgrp() and fields[0] not in {"Z", "X"}: return False
    return True


def wait_worker(child, deadline):
    """Reserve ten seconds for direct-child stop/reap inside the shared deadline.

    The outer runtime still owns descendant process-group teardown; never kill
    this shared PGID or create a nested session from the supervisor.
    """
    try:
        remaining = deadline-time.monotonic()-10
        if remaining <= 0: raise TimeoutError("shared worker deadline leaves no shutdown reserve")
        return child.wait(timeout=remaining)
    except BaseException as primary:
        try:
            try: child.terminate()
            except ProcessLookupError: pass
            try:
                child.wait(timeout=max(0, min(5, deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                try: child.kill()
                except ProcessLookupError: pass
                child.wait(timeout=max(0, min(5, deadline-time.monotonic())))
        except BaseException as secondary:
            primary.add_note(f"direct-child teardown: {type(secondary).__name__}: {secondary}")
        raise


def run(settings):
    """Reap four sequential workers under one deadline; retain all private caches."""
    from apertus_amd.config import CACHE_PATHS
    if settings["validation"]["sha256"] != digest(__file__): raise ValueError("executed validation SHA mismatch")
    rank, _ = validate_launch(settings, os.environ); environment(os.environ)
    output = Path(f"grouped-performance-{rank}").resolve(); output.mkdir()
    Path(settings["scratch_root"]).mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f'grouped-perf-{os.environ["SLURM_JOB_ID"]}-{rank}-', dir=settings["scratch_root"]))
    record = dict(status="unqualified", rank=rank, pgid=os.getpgrp(), driver_sha256=digest(__file__),
                  started_ns=time.monotonic_ns(), cleanup="retained", scratch=str(scratch), workers=[])
    deadline = time.monotonic()+480
    try:
        for slot, arm in enumerate("ABBA" if rank % 2 == 0 else "BAAB"):
            selector = "0" if arm == "A" else "1"
            directory = output/str(slot); directory.mkdir(); env = dict(os.environ, **{SELECTOR: selector})
            for key, relative in CACHE_PATHS.items():
                cache = scratch/str(slot)/relative; cache.mkdir(parents=True, exist_ok=True); env[key] = str(cache)
            remaining = deadline-time.monotonic()
            if remaining <= 10: raise TimeoutError("shared worker deadline leaves no shutdown reserve")
            with (directory/"native.log").open("x") as log:
                child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--directory", str(directory),
                                          "--slot", str(slot), "--selector", selector], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
                status = wait_worker(child, deadline)
            value = json.loads((directory/"worker.json").read_text())
            if status or value.get("status") != "passed" or value.get("pid") != child.pid or value.get("pgid") != os.getpgrp():
                raise ValueError("closed worker failed")
            if object_digest(value.get("operands")) != OPERANDS_SHA: raise ValueError("qualified operand digest mismatch")
            validate_worker(value, rank=rank, slot=slot, expected_sha=digest(__file__), expected_operands=record["workers"][0]["operands"] if record["workers"] else value["operands"], expected_provenance=expected_provenance())
            record["workers"].append(value)
        if time.monotonic() > deadline or not quiescent(): raise ValueError("deadline/quiescence")
        record["status"] = "passed"
        record["finished_ns"] = time.monotonic_ns()
        validate_rank(record, rank=rank, expected_sha=digest(__file__), expected_operands=record["workers"][0]["operands"], expected_provenance=expected_provenance(), expected_scratch=str(scratch))
    except BaseException as error:
        record["status"] = "unqualified"
        record["error"] = f"{type(error).__name__}: {error}"; raise
    finally:
        record["finished_ns"] = time.monotonic_ns(); retain(output/"summary.json", record)


def need(ok, message):
    """Reject a closed-record contract violation without coercion."""
    if not ok: raise ValueError(message)


def keys(value, expected):
    """Require an exact plain-JSON object schema."""
    need(type(value) is dict and set(value)==set(expected),'record keys')


def integer(value, minimum=0):
    """Require an integer, excluding bool, within the stated lower bound."""
    need(type(value) is int and value>=minimum,'integer range/type')


def finite(value, minimum=0):
    """Require a finite numeric scalar, excluding bool."""
    need(type(value) in (int,float) and math.isfinite(value) and value>=minimum,'finite number')


def sha(value):
    """Require a lowercase SHA256 hex string."""
    need(type(value) is str and len(value)==64 and not set(value)-set('0123456789abcdef'),'SHA256')


def same(a, b):
    """Compare JSON recursively with exact types and finite floats."""
    need(type(a) is type(b),'JSON type mismatch')
    if type(a) is dict:
        keys(a,b)
        for k in b: same(a[k],b[k])
    elif type(a) is list:
        need(len(a)==len(b),'JSON length mismatch')
        for x,y in zip(a,b): same(x,y)
    else:
        need(type(a) in (str,int,float,bool,type(None)) and a==b,'JSON value mismatch')
        if type(a) is float: need(math.isfinite(a),'nonfinite JSON')


def validate_worker(r, *, rank, slot, expected_sha, expected_operands, expected_provenance):
    """Return unchanged r only after complete fixed worker qualification."""
    integer(rank); integer(slot); need(rank<4 and slot<4,'rank/slot'); sha(expected_sha)
    required='status rank local_device slot selector pid pgid driver_sha256 started_ns finished_ns baseline_environment_initial baseline_environment_final provenance operands operands_device_pre operands_post layouts output_hashes checks branch aliases_restored warm_calls_per_case cache samples phase_memory precision affinity origins mapped_elfs'.split()
    keys(r,required)
    for k in ('rank','local_device','slot','pid','pgid','started_ns','finished_ns'): integer(r[k],1 if k in ('pid','pgid','started_ns','finished_ns') else 0)
    need(r['status']=='passed' and r['rank']==r['local_device']==rank and r['slot']==slot,'worker identity/status')
    need(r['selector']==('0' if ('ABBA' if rank%2==0 else 'BAAB')[slot]=='A' else '1'),'selector order')
    need(r['driver_sha256']==expected_sha and r['started_ns']<r['finished_ns'],'source/interval'); same(r['provenance'],expected_provenance)
    for k in ('baseline_environment_initial','baseline_environment_final'): same(r[k],BASELINE)
    keys(expected_operands,CASES)
    for value in expected_operands.values():
        keys(value,('input','weights')); sha(value['input']); need(type(value['weights']) is list and len(value['weights'])==16,'weights')
        for h in value['weights']: sha(h)
    same(r['operands'],expected_operands); same(r['operands_device_pre'],expected_operands); same(r['operands_post'],expected_operands)
    same(r['precision'],dict(matmul='highest',allow_tf32=False)); same(r['warm_calls_per_case'],[10,10]); need(r['aliases_restored'] is True,'aliases not restored')
    for name in ('output_hashes','checks','branch'): keys(r[name],('pre','post'))
    keys(r['layouts'],CASES)
    for phase in ('pre','post'):
        keys(r['output_hashes'][phase],CASES)
        for h in r['output_hashes'][phase].values(): sha(h)
        counts=dict(zip(ALIASES,(4,0) if r['selector']=='0' else (0,4))); same(r['branch'][phase],counts)
        checks=r['checks'][phase]; need(type(checks) is list and len(checks)==4,'four checks'); seen=set()
        for check in checks:
            keys(check,('key','splits','experts')); case=check['key']; need(type(case) is str and case in CASES and case not in seen,'check coverage'); seen.add(case)
            projection,kind=case.split('/'); k,n=DIMS[projection]; splits=SPLITS[kind]; same(check['splits'],splits)
            need(type(check['experts']) is list and len(check['experts'])==16,'expert coverage')
            for rows,m in zip(splits,check['experts']):
                keys(m,('finite','numel','checked','mismatches','max_abs','rms_error','reference_rms'))
                need(m['finite'] is True,'nonfinite tensors')
                for field in ('numel','checked','mismatches'): integer(m[field])
                need(m['numel']==m['checked']==rows*n and m['mismatches']==0,'element gate')
                for field in ('max_abs','rms_error','reference_rms'): finite(m[field])
                need(m['rms_error']<=1e-6+2**-7*m['reference_rms'],'RMS gate')
                if rows==0: need(m['max_abs']==m['rms_error']==m['reference_rms']==0,'empty expert')
    for case,record in r['layouts'].items():
        keys(record,('input','weights','output','row_counts','source')); need(type(record['source']) is str and bool(record['source']),'operand source')
        projection,kind=case.split('/'); k,n=DIMS[projection]; rows=sum(SPLITS[kind]); device=f'cuda:{rank}'
        spec=lambda shape,stride,dtype:dict(shape=shape,stride=stride,dtype=dtype,device=device,offset=0)
        same(record['input'],spec([rows,k],[k,1],'torch.bfloat16')); same(record['output'],spec([rows,n],[n,1],'torch.bfloat16'))
        same(record['row_counts'],spec([16],[1],'torch.int32')); same(record['weights'],[spec([n,k],[k,1],'torch.bfloat16') for _ in range(16)])
    keys(r['cache'],CACHE)
    for phase,c in r['cache'].items():
        keys(c,('dynamo','generated_kernel_count','gmm_keys')); keys(c['dynamo'],('frames','stats','inductor','graph_break')); integer(c['generated_kernel_count'])
        for group in c['dynamo'].values():
            need(type(group) is dict and all(type(k) is str for k in group),'Dynamo schema')
            for count in group.values(): integer(count)
        need(type(c['gmm_keys']) is dict,'cache mapping')
        for device,hashes in c['gmm_keys'].items():
            need(type(device) is str and device==str(rank) and type(hashes) is list,'cache device')
            for h in hashes: sha(h)
            need(hashes==sorted(set(hashes)),'cache keys not sorted/unique')
        if r['selector']=='0': need(c['gmm_keys']=={},'unexpected baseline grouped cache')
        if phase!='before_pre' and r['selector']=='1': need(set(c['gmm_keys'])=={str(rank)} and bool(c['gmm_keys'][str(rank)]),'candidate cache empty')
        if phase!='before_pre': same(c,r['cache']['after_pre'])
    need(type(r['samples']) is list and len(r['samples'])==48,'48 samples')
    expected=[(j,case) for j in range(12) for case in CASES[(rank+j)%4:]+CASES[:(rank+j)%4]]; previous=r['started_ns']
    for row,(block,case) in zip(r['samples'],expected):
        keys(row,'case block batch status t0_ns return_ns complete_ns event_ms completion_us host_return_us device_span_us peak_allocated peak_reserved log_before log_after'.split())
        integer(row['block']); integer(row['batch']); need((row['block'],row['case'],row['batch'],row['status'])==(block,case,16,'complete'),'sample order')
        for k in ('t0_ns','return_ns','complete_ns'): integer(row[k],1)
        t0,t1,t2=row['t0_ns'],row['return_ns'],row['complete_ns']; need(previous<=t0<=t1<=t2<=r['finished_ns'] and t0<t2,'sample chronology'); previous=t2
        for k in ('event_ms','completion_us','host_return_us','device_span_us'): finite(row[k])
        need(row['event_ms']>0 and row['completion_us']>0 and row['device_span_us']>0,'positive clocks')
        need(row['host_return_us']==(t1-t0)/16000 and row['completion_us']==(t2-t0)/16000 and row['device_span_us']==row['event_ms']*1000/16,'clock units')
        for k in ('peak_allocated','peak_reserved'): integer(row[k],1)
        for k in ('log_before','log_after'): integer(row[k])
        need(row['peak_allocated']<=row['peak_reserved'] and row['log_before']==row['log_after'],'memory/native log growth')
    keys(r['phase_memory'],('pre','warm','measure','post'))
    for m in r['phase_memory'].values():
        keys(m,('allocated','reserved')); integer(m['allocated']); integer(m['reserved']); need(m['allocated']<=m['reserved'],'phase memory')
    prefix=str(PACKAGE)+'/'
    same(r['origins'],dict(GroupedLinear=prefix+'transformer_engine/pytorch/module/grouped_linear.py',general_grouped_gemm=prefix+'transformer_engine/pytorch/cpp_extensions/gemm.py',general_grouped_gemm_triton=prefix+'transformer_engine/pytorch/triton_kernels/grouped_gemm.py'))
    same(r['mapped_elfs'],sorted([prefix+'transformer_engine/libtransformer_engine.so',prefix+'transformer_engine/transformer_engine_torch.cpython-312-x86_64-linux-gnu.so']))
    a=r['affinity']; keys(a,('initial','post_gpu_init','final','restoration_attempted','restoration_applied','errors'))
    for field in ('initial','post_gpu_init','final'):
        need(type(a[field]) is list and len(a[field])==48,'affinity mask')
        for cpu in a[field]: integer(cpu)
        need(len(set(a[field]))==48,'affinity duplicates')
    same(a['initial'],a['final']); same(a['errors'],[])
    need(type(a['restoration_attempted']) is bool and type(a['restoration_applied']) is bool,'affinity flags')
    need((a['post_gpu_init']==a['initial'] and not a['restoration_attempted'] and not a['restoration_applied']) or (a['restoration_attempted'] and a['restoration_applied']),'affinity restoration')
    return r


def validate_rank(summary, *, rank, expected_sha, expected_operands, expected_provenance, expected_scratch=None):
    """Check ordered closed workers; producer independently proves reap and quiescence."""
    keys(summary,'status rank pgid driver_sha256 started_ns finished_ns cleanup scratch workers'.split())
    for k in ('rank','pgid','started_ns','finished_ns'): integer(summary[k],0 if k=='rank' else 1)
    need(summary['status']=='passed' and summary['rank']==rank and summary['driver_sha256']==expected_sha and summary['cleanup']=='retained','rank identity/lifecycle')
    need(type(summary['scratch']) is str,'scratch type'); path=PurePosixPath(summary['scratch'])
    need(str(path)==summary['scratch'] and '..' not in path.parts and str(path).startswith('/dev/shm/apertus-amd/'),'dedicated scratch path')
    if expected_scratch is not None: need(summary['scratch']==expected_scratch,'scratch ownership binding')
    need(type(summary['workers']) is list and len(summary['workers'])==4,'four workers'); previous=summary['started_ns']; pids=set()
    for slot,r in enumerate(summary['workers']):
        validate_worker(r,rank=rank,slot=slot,expected_sha=expected_sha,expected_operands=expected_operands,expected_provenance=expected_provenance)
        need(r['pgid']==summary['pgid'] and r['pid'] not in pids and previous<=r['started_ns'],'rank process/order'); previous=r['finished_ns']; pids.add(r['pid'])
    need(previous<=summary['finished_ns'] and summary['started_ns']<summary['finished_ns'],'parent interval')
    return summary


def analyze(samples, eligible):
    """Require complete positive durations and external full qualification per worker."""
    expected = {(r, s, c, j) for r in range(4) for s in range(4) for c in CASES for j in range(12)}
    if set(samples) != expected or set(eligible) != {(r, s) for r in range(4) for s in range(4)}:
        raise ValueError('incomplete fixed schedule')
    if any(value is not True for value in eligible.values()):
        raise ValueError('failed numerical/provenance/warm/lifecycle qualification')
    for value in samples.values():
        if set(value) != {'completion', 'event'} or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in value.values()):
            raise ValueError('invalid clock durations')
    result = {}
    for case in CASES:
        clocks = {}
        for clock in ('completion', 'event'):
            ranks = []
            for rank in range(4):
                order = 'ABBA' if rank % 2 == 0 else 'BAAB'
                a = [s for s in range(4) if order[s] == 'A']
                b = [s for s in range(4) if order[s] == 'B']
                early, late, gaps_a, gaps_b = [], [], [], []
                for j in range(12):
                    ae, al, be, bl = [math.log(samples[rank, s, case, j][clock]) for s in (*a, *b)]
                    early.append(ae-be); late.append(al-bl)
                    gaps_a.append(abs(ae-al)); gaps_b.append(abs(be-bl))
                delta = [(x+y)/2 for x, y in zip(early, late)]
                estimate = median(delta)
                eta = max(median(gaps_a), median(gaps_b), 1.4826*median(abs(x-estimate) for x in delta))
                ranks.append(dict(estimate=estimate, eta=eta, deltas=delta,
                                  early=median(early), late=median(late)))
            if clock == 'completion':
                passed = all(min(r['deltas']) > r['eta'] and r['early'] > r['eta'] and r['late'] > r['eta'] for r in ranks)
            else:
                passed = all(min(r['deltas']) >= -r['eta'] and r['early'] >= -r['eta'] and r['late'] >= -r['eta'] for r in ranks)
            strata = {name: median(ranks[r]['estimate'] for r in indices) for name, indices in (('ABBA', (0, 2)), ('BAAB', (1, 3)))}
            clocks[clock] = dict(ranks=ranks, strata=strata, ratio=math.exp(median(r['estimate'] for r in ranks)), passed=passed)
        result[case] = dict(clocks=clocks, passed=all(c['passed'] for c in clocks.values()))
    return dict(cases=result, passed=all(c['passed'] for c in result.values()))


def check(directory, derived_destination=None):
    """Audit immutable run records before classifying; write only an exclusive new target."""
    directory=Path(directory).resolve(); frozen=directory/'validation.py'; actual_sha=digest(frozen)
    need(actual_sha==digest(__file__),'checker/frozen source mismatch')
    settings=json.loads((directory/'resolved.json').read_text()); need(settings['validation']['sha256']==actual_sha,'resolved frozen SHA mismatch')
    need(settings['image']['sha256']==IMAGE and settings['procedure']['kind']=='commands' and settings['procedure']['commands']==[COMMAND] and settings['procedure']['timeout']==510,'offline execution contract')
    resources=settings['resources']; wanted=dict(nodes=1,gpus_per_node=4,cpus_per_node=192,mpi='pmix',time='00:10:00',reservation='cscs',cpu_bind='map_ldom:0,1,2,3',distribution='block:block')
    need(all(type(resources.get(k)) is type(v) and resources[k]==v for k,v in wanted.items()) and resources.get('cpus_per_task') is None,'offline resource contract')
    samples={}; eligible={}; pids=set(); expected_operands=None
    for rank in range(4):
        base=directory/f'output/0/grouped-performance-{rank}'; summary=json.loads((base/'summary.json').read_text())
        for slot in range(4):
            worker_record=json.loads((base/str(slot)/'worker.json').read_text())
            need(object_digest(worker_record['operands'])==OPERANDS_SHA,'qualified operand digest mismatch')
            same(worker_record,summary['workers'][slot])
            if expected_operands is None: expected_operands=worker_record['operands']
        validate_rank(summary,rank=rank,expected_sha=actual_sha,expected_operands=expected_operands,expected_provenance=expected_provenance())
        command=json.loads((directory/f'output/0/rank-{rank}.json').read_text())
        need(command['status']=='passed' and command['rank']==rank and command['argv']==COMMAND,'runtime rank failure')
        for w in summary['workers']:
            need(w['pid'] not in pids,'duplicate global PID'); pids.add(w['pid']); eligible[rank,w['slot']]=True
            for row in w['samples']: samples[rank,w['slot'],row['case'],row['block']]=dict(completion=row['completion_us'],event=row['device_span_us'])
    need(len(pids)==16 and len(samples)==768,'global schedule count')
    result=json.loads((directory/'result.json').read_text()); need(result['status']==result['execution']=='passed','run outcome')
    output=dict(status='qualified',driver_sha256=actual_sha,workers=16,samples=768,expert_checks=2048,analysis=analyze(samples,eligible),scope='fixed synthetic replica consistency only')
    if derived_destination is not None:
        target=Path(derived_destination).resolve()
        need(not target.is_relative_to(directory),'derived destination must be outside authoritative run')
        need(not target.exists() and not target.with_suffix('.tmp').exists(),'derived target already exists'); retain(target,output)
    return output


def main():
    """Expose the reviewed native protocol and vendor-free immutable-record checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "check")); parser.add_argument("--directory", type=Path)
    parser.add_argument("--derived-destination", type=Path)
    parser.add_argument("--slot", type=int, choices=range(4)); parser.add_argument("--selector", choices=("0", "1")); args = parser.parse_args()
    if args.mode == 'check':
        if args.directory is None or args.slot is not None or args.selector is not None: parser.error('check requires run directory only')
        print(json.dumps(check(args.directory,args.derived_destination),indent=2)); return
    if not IMPLEMENTATION_READY: raise RuntimeError("grouped timing implementation awaits strict validation and CPU qualification")
    if args.derived_destination is not None: parser.error('derived destination is check-only')
    if (args.mode == "worker") != (args.directory is not None and args.slot is not None and args.selector is not None): parser.error("worker needs directory/slot/selector")
    settings = json.loads(Path("/apertus/resolved.json").read_text())
    if settings['validation']['sha256'] != digest(__file__): raise ValueError('executed validation SHA mismatch')
    rank, _ = validate_launch(settings, os.environ)
    run(settings) if args.mode == "run" else worker(args.directory, rank, args.slot, args.selector)


if __name__ == "__main__": main()
