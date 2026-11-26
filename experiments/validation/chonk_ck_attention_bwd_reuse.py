"""Time the exact native attention backward and verify private CK object reuse."""

import importlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from time import monotonic

from apertus_amd.artifacts import sha256, write_json
from apertus_amd.config import CACHE_PATHS


IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
STEM = ("fmha_bwd_d128_bf16_batch_b16x128x128x16x128x16x32x128x128_r1x4x1_r4x1x1_"
        "r1x4x1_w16x16x32_w16x16x16_o1_maxq0_npad_nbias_ndbias_mask_ndropout_ndeterministic_ntrload_gfx9")
OBJECT = STEM + ".so.40ee1ee0"
PACKAGE = Path("/opt/venv/lib/python3.12/site-packages/transformer_engine")
SHAPE = (4096, 2)
COLD_TIMEOUT, WARM_TIMEOUT = 900, 180
COMPARE_COLD_TIMEOUT = 240
ATTEMPTS = ("build-a", "build-b")
VENDOR = ("2.14.0+rocm10.1.0", "7.16.26385")
BACKEND = "NVTE_Fused_Attn_Backend.NVTE_CK"
REFERENCE = {"bytes": 114880, "sha256": "c87378fb7245bac1ca43b6065fc4f7454d2bf7b840ec1ce45e68a72c6eb517b7"}
REFERENCE_NORMS = (369.9413757324219, 331.6982727050781, 333.5680847167969, 369.4879150390625)
TENSOR_NAMES = ("output", "dq", "dk", "dv")


def verify_launch(settings, environ):
    """Fail closed on an unexpected image, workload or rank placement."""
    resources = settings["resources"]
    if (settings["image"]["sha256"] != IMAGE or not settings.get("registry_image_path")
            or settings["procedure"]["kind"] != "commands" or settings["procedure"]["timeout"] != 1200
            or resources["nodes"] != 1 or resources["gpus_per_node"] != 1
            or resources["cpus_per_task"] != 48 or resources["time"] != "00:25:00"
            or any(environ.get(k) != v for k, v in (("SLURM_PROCID", "0"), ("SLURM_LOCALID", "0"),
                                                    ("SLURM_NTASKS", "1")))):
        raise ValueError("unexpected one-rank CK attention diagnostic launch")


def verify_comparison_launch(settings, environ):
    """Require the reviewed cached-image, four-rank component step."""
    resources = settings["resources"]
    rank = environ.get("SLURM_PROCID")
    expected_resources = dict(nodes=1, gpus_per_node=4, cpus_per_task=48, time="00:25:00",
                              cpu_bind="map_ldom:0,1,2,3", account="csstaff", partition="mi300",
                              reservation="cscs", exclude="nid002450", memory_gb=0,
                              distribution="block:block", mpi="pmix", network="disable_rdzv_get",
                              uenv_passthrough="ignore")
    if (settings["image"]["sha256"] != IMAGE or not settings.get("registry_image_path")
            or settings["procedure"]["kind"] != "commands" or settings["procedure"]["timeout"] != 1200
            or settings["procedure"]["commands"] != [["python3", "-u", "/apertus/validation.py", "compare"]]
            or settings["scratch_root"] != "/dev/shm/apertus-amd" or not settings["launch"]["submit"]
            or any(resources.get(k) != v for k, v in expected_resources.items())
            or rank not in ("0", "1", "2", "3") or environ.get("SLURM_LOCALID") != rank
            or environ.get("SLURM_NODEID") != "0" or environ.get("SLURM_NTASKS") != "4"):
        raise ValueError("unexpected four-rank CK build comparison launch")
    return int(rank)


def observed_object(path):
    """Preserve identity even for a rejected regular artifact; never follow a symlink."""
    info = path.lstat()
    record = {"key": path.name, "bytes": info.st_size, "regular": stat.S_ISREG(info.st_mode)}
    if record["regular"]:
        with path.open("rb") as stream:
            record["elf"] = stream.read(4) == b"\x7fELF"
        record["sha256"] = sha256(path)
    return record


def object_record(path):
    """Validate the expected completed ELF, then capture its content identity."""
    record = observed_object(path)
    if not record["regular"] or record["bytes"] < 64:
        raise ValueError(f"CK object is not a complete regular file: {path}")
    if not record["elf"]:
        raise ValueError(f"CK object is not ELF: {path}")
    return {name: record[name] for name in ("bytes", "sha256")}


def mapped_object(path, maps):
    """Distinguish the exact file from a deleted mapping or a same-name replacement."""
    info = path.lstat()
    device = os.major(info.st_dev), os.minor(info.st_dev)
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == str(path) and "x" in fields[1]:
            if int(fields[4]) == info.st_ino and tuple(int(n, 16) for n in fields[3].split(":")) == device:
                return True
    return False


def quiescent():
    """Check that the bounded command process group has no surviving compiler."""
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


def worker(output, visible=1):
    """Run the same public TE consumer in a separate cold or warm process."""
    from apertus_amd.runtime import preserve_affinity, require_affinity

    record = {"status": "failed", "pgid": os.getpgrp()}
    comparison = visible == 4
    try:
        if comparison:
            record.update(key=OBJECT, pid=os.getpid(), hostname=socket.gethostname(), image=IMAGE,
                          build_environment={k: os.environ.get(k) for k in (
                              "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                              "ROCM_HOME", "ROCM_PATH", "OMP_NUM_THREADS", "MAX_JOBS", "TMPDIR", "XDG_CACHE_HOME")},
                          hipcc_on_path=shutil.which("hipcc"),
                          ck_builder_sha256=sha256(PACKAGE / "lib/ck_jit/ck_jit_compile.sh"))
        with preserve_affinity(record.setdefault("affinity", {})) as affinity:
            import torch

            record.update(torch=torch.__version__, hip=torch.version.hip)
            if comparison and (record["torch"], record["hip"]) != VENDOR:
                raise ValueError("unexpected Torch/ROCm version")
            record["visible_gpus"] = torch.cuda.device_count()
            if not torch.version.hip or record["visible_gpus"] != visible:
                raise ValueError(f"expected {visible} visible ROCm GPU(s)")
            device = int(os.environ["SLURM_LOCALID"]) if comparison else 0
            torch.cuda.set_device(device)
            gpu = torch.cuda.get_device_properties(device)
            record["target"] = gpu.gcnArchName
            if gpu.gcnArchName.split(":")[0] != "gfx942":
                raise ValueError("expected gfx942")
            module = importlib.import_module(
                "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention")
            path = PACKAGE / "pytorch/attention/dot_product_attention/dot_product_attention.py"
            if Path(module.__file__).resolve() != path:
                raise ValueError("TE consumer import differs from pinned image")
        require_affinity(affinity, "TE attention backward")
        if len(affinity) != 48:
            raise ValueError("expected 48 bound CPUs")
        record["te_consumer"] = str(path)
        target = Path(os.environ["XDG_CACHE_HOME"]) / "te_ck_jit" / OBJECT
        rng = torch.Generator(device="cuda").manual_seed(123)
        values = [torch.randn((*SHAPE, heads, 128), dtype=torch.bfloat16, device="cuda", generator=rng)
                  .requires_grad_() for heads in (28, 7, 7)]
        dout = torch.randn((*SHAPE, 28 * 128), dtype=torch.bfloat16, device="cuda", generator=rng)
        model = module.DotProductAttention(28, 128, num_gqa_groups=7, attention_dropout=0.0,
                                           qkv_format="sbhd", attn_mask_type="causal", tp_size=1).train()
        if model.deterministic:
            raise ValueError("unexpected deterministic attention route")
        torch.cuda.synchronize()
        start = monotonic()
        out = model(*values)
        torch.cuda.synchronize()
        record["forward_seconds"] = monotonic() - start
        record["backends"] = {k: str(v) for k, v in module._attention_backends.items() if k != "attention_params"}
        if tuple(out.shape) != (*SHAPE, 3584) or out.dtype != torch.bfloat16:
            raise ValueError("unexpected TE attention output")
        if comparison and not native_route(record["backends"]):
            raise ValueError("expected native NVTE_CK, without attention fallback")
        start = monotonic()
        out.backward(dout)
        torch.cuda.synchronize()
        record["backward_seconds"] = monotonic() - start
        tensors = [out, *(x.grad for x in values)]
        expected_shapes = [out.shape, *(x.shape for x in values)]
        valid = [t is not None and t.shape == shape and t.dtype == torch.bfloat16 and t.isfinite().all().item()
                 for t, shape in zip(tensors, expected_shapes)]
        norms = [float(t.float().norm().item()) if ok else None for t, ok in zip(tensors, valid)]
        record.update(finite_bf16=dict(zip(TENSOR_NAMES, valid)),
                      norms=[n if n is not None and math.isfinite(n) else None for n in norms])
        if not all(valid):
            raise ValueError("invalid output or nonfinite native gradients")
        if any(n is None or n <= 0 for n in record["norms"]):
            raise ValueError("invalid output or gradient norm")
        maps = Path("/proc/self/maps").read_text()
        mapped = mapped_object(target, maps) if comparison else str(target) in maps
        record["mapped"] = mapped
        if not mapped:
            raise ValueError("expected private CK backward object was not mapped")
        require_affinity(affinity, "TE attention completion")
        record.update(status="passed", object=object_record(target))
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output, record)


def phase(scratch, directory, name, timeout):
    """Run a fresh child in the controller-owned process group."""
    env = dict(os.environ)
    for key, relative in CACHE_PATHS.items():
        path = scratch / relative
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    report = directory / f"{name}.json"
    log = directory / f"{name}.log"
    with log.open("x") as stream:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "worker", str(report)],
                       stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=True)
    record = json.loads(report.read_text())
    if record["status"] != "passed" or record["pgid"] != os.getpgrp() or not quiescent():
        raise ValueError(f"{name} worker did not finish quiescently")
    return record, log.read_text(errors="replace")


def verify_reuse(cold, warm, cold_log, warm_log, expected):
    """Require a cold build, no warm build, and matching native results."""
    if f"[CK-JIT] JIT-compiling blob: {STEM}.cpp" not in cold_log or "[CK-JIT-BUILD]" not in cold_log:
        raise ValueError("no cold build evidence for the exact CK blob")
    if "[CK-JIT-BUILD]" in warm_log or "[CK-JIT] JIT-compiling" in warm_log:
        raise ValueError("warm process recompiled a CK blob")
    if cold["object"] != expected or warm["object"] != expected:
        raise ValueError("executed CK object identity differs from the verified seed")
    if (cold["backends"] != warm["backends"] or len(cold["norms"]) != 4 or len(warm["norms"]) != 4
            or any(not math.isclose(a, b, rel_tol=0.005) for a, b in zip(cold["norms"], warm["norms"]))):
        raise ValueError("cold and warm attention results disagree")


def native_route(backends):
    """Require the exact TE fused CK selection seen in the historical fixture."""
    return all(backends.get(k) == v for k, v in {
        "fused_attention_backend": BACKEND, "use_fused_attention": "1",
        "use_flash_attention": "False", "use_unfused_attention": "False",
    }.items())


def verify_comparison_worker(record, expected):
    """Require native CK, the qualified stack, exact mapping, and reference norms."""
    norms = record["norms"]
    if (record["key"] != OBJECT or record["image"] != IMAGE or record["mapped"] is not True
            or record["visible_gpus"] != 4 or record["object"] != expected
            or (record["torch"], record["hip"]) != VENDOR
            or not native_route(record["backends"])
            or record["finite_bf16"] != dict.fromkeys(TENSOR_NAMES, True)
            or len(norms) != 4 or any(n is None or not math.isclose(n, r, rel_tol=0.005)
                                     for n, r in zip(norms, REFERENCE_NORMS))):
        raise ValueError("CK worker failed vendor, route, object or numerical qualification")


def seed_verified(source, target, expected):
    """Copy only the qualified whole object into a new private cache."""
    if object_record(source) != expected:
        raise ValueError("qualified CK source changed before copy")
    target.parent.mkdir(parents=True)
    with source.open("rb") as original, target.open("xb") as copy:
        shutil.copyfileobj(original, copy)
    target.chmod(0o444)
    if object_record(target) != expected or object_record(source) != expected:
        raise ValueError("CK seed copy failed byte verification")
    return seed_identity(target)


def seed_identity(path):
    """Detect replacement and in-place mutation in addition to content hashing."""
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns


def comparison_phase(scratch, directory, name, timeout, evidence):
    """Record bounded worker evidence even if it fails; never inspect a live cache."""
    env = dict(os.environ)
    for key, relative in CACHE_PATHS.items():
        path = scratch / relative
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    report = directory / f"{name}.json"
    log = directory / f"{name}.log"
    target = scratch / "xdg/te_ck_jit" / OBJECT
    evidence.update(key=OBJECT, path=str(target), report=report.name, log=log.name)
    try:
        with log.open("x") as stream:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "compare-worker", str(report)],
                           stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=True)
    finally:
        evidence["quiescent"] = quiescent()
        if not evidence["quiescent"]:
            # The enclosing runner owns this process group and will stop it.
            raise RuntimeError(f"{name}: CK process group still has live descendants")
        try:
            evidence["observed_object"] = observed_object(target)
        except OSError as error:
            evidence["observation_error"] = str(error)
    record = json.loads(report.read_text())
    if record["status"] != "passed" or record["pgid"] != os.getpgrp() or record["hostname"] != socket.gethostname():
        raise ValueError(f"{name} worker did not pass in the owned process group")
    expected = object_record(target)
    verify_comparison_worker(record, expected)
    return record, log.read_text(errors="replace"), expected


def compare(settings):
    """Compare independent four-visible-GPU builds and their private-copy reuse."""
    directory = Path("ck-attention-compare").resolve()
    directory.mkdir()
    result = {"status": "failed", "image": IMAGE, "reference_job": "762106", "reference_object": REFERENCE,
              "scope": "one node, two independent rank-zero builds; no training or publication",
              "hostname": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"], "trials": {}}
    summary = directory / "summary.json"
    try:
        root = Path(settings["scratch_root"])
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.resolve().is_relative_to("/dev/shm"):
            raise ValueError("expected job-owned node-local scratch")
        scratch = Path(tempfile.mkdtemp(prefix=f"ck-compare-{result['job_id']}-", dir=root))
        result["scratch"] = str(scratch)
        write_json(summary, result)
        for name in ATTEMPTS:
            trial = result["trials"][name] = {"status": "failed", "cold": {}}
            cold_dir, warm_dir = scratch / name / "cold", scratch / name / "warm"
            cold_dir.mkdir(parents=True)
            source = cold_dir / "xdg/te_ck_jit" / OBJECT
            target = warm_dir / "xdg/te_ck_jit" / OBJECT
            write_json(summary, result)
            try:
                cold, cold_log, expected = comparison_phase(cold_dir, directory, f"{name}-cold",
                                                             COMPARE_COLD_TIMEOUT, trial["cold"])
                if (f"[CK-JIT] JIT-compiling blob: {STEM}.cpp" not in cold_log.splitlines()
                        or f"[CK-JIT-BUILD] SUCCESS: {source}" not in cold_log.splitlines()):
                    raise ValueError("missing exact cold-build success evidence")
                trial.update(cold_qualified=True, cold_backward_seconds=cold["backward_seconds"])
                before = seed_verified(source, target, expected)
                trial["warm"] = {}
                write_json(summary, result)
                warm, warm_log, actual = comparison_phase(warm_dir, directory, f"{name}-warm",
                                                           WARM_TIMEOUT, trial["warm"])
                if (before != seed_identity(target) or actual != expected or object_record(source) != expected
                        or cold["pid"] == warm["pid"]):
                    raise ValueError("worker not fresh or CK cache object changed across reuse")
                verify_reuse(cold, warm, cold_log, warm_log, expected)
                trial.update(status="passed", warm_backward_seconds=warm["backward_seconds"])
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                trial["error"] = f"{type(error).__name__}: {error}"
            finally:
                observed = trial["cold"].get("observed_object", {})
                if "sha256" in observed:
                    trial["cold_matches_762106"] = all(observed[k] == v for k, v in REFERENCE.items())
                write_json(summary, result)
        a, b = (result["trials"][name]["cold"].get("observed_object", {}) for name in ATTEMPTS)
        if "sha256" in a and "sha256" in b:
            result["independent_builds_byte_identical"] = (a["bytes"], a["sha256"]) == (b["bytes"], b["sha256"])
        if any(trial["status"] != "passed" for trial in result["trials"].values()):
            raise ValueError("at least one independent CK build or reuse failed qualification")
        result["status"] = "passed"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(summary, result)


def run(settings):
    """Build cold and copy only the verified object into a fresh private cache."""
    directory = Path("ck-attention-reuse").resolve()
    directory.mkdir()
    result = {"status": "failed", "image": IMAGE, "scope": "one GPU native CK attention backward; no publication"}
    try:
        root = Path(settings["scratch_root"])
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.resolve().is_relative_to("/dev/shm"):
            raise ValueError("expected job-owned node-local scratch")
        scratch = Path(tempfile.mkdtemp(prefix=f"ck-attn-{os.environ['SLURM_JOB_ID']}-", dir=root))
        result["scratch"] = str(scratch)
        cold_dir, warm_dir = scratch / "cold", scratch / "warm"
        cold_dir.mkdir()
        cold, cold_log = phase(cold_dir, directory, "cold", COLD_TIMEOUT)
        source = cold_dir / "xdg/te_ck_jit" / OBJECT
        expected = object_record(source)
        target = warm_dir / "xdg/te_ck_jit" / OBJECT
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        target.chmod(0o444)
        before = target.stat()
        if object_record(target) != expected:
            raise ValueError("CK seed copy failed verification")
        warm, warm_log = phase(warm_dir, directory, "warm", WARM_TIMEOUT)
        after = target.stat()
        if (before.st_ino, before.st_mtime_ns) != (after.st_ino, after.st_mtime_ns):
            raise ValueError("warm process replaced or modified the seed object")
        if object_record(source) != expected or object_record(target) != expected:
            raise ValueError("CK cache object changed after reuse")
        verify_reuse(cold, warm, cold_log, warm_log, expected)
        shutil.rmtree(scratch)
        result.update(status="passed", object=expected, cold_backward_seconds=cold["backward_seconds"],
                      warm_backward_seconds=warm["backward_seconds"], cleanup="removed owned quiescent scratch")
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(directory / "summary.json", result)


if __name__ == "__main__":
    settings = json.loads(Path("/apertus/resolved.json").read_text())
    if sys.argv[1:] == ["compare"]:
        if verify_comparison_launch(settings, os.environ) == 0:
            compare(settings)
    elif len(sys.argv) == 3 and sys.argv[1] == "compare-worker":
        if verify_comparison_launch(settings, os.environ) != 0:
            raise ValueError("only local rank zero may build or consume the CK object")
        worker(Path(sys.argv[2]), visible=4)
    elif sys.argv[1:] == ["run"]:
        verify_launch(settings, os.environ)
        run(settings)
    elif len(sys.argv) == 3 and sys.argv[1] == "worker":
        verify_launch(settings, os.environ)
        worker(Path(sys.argv[2]))
    else:
        raise ValueError("unexpected CK attention diagnostic command")
