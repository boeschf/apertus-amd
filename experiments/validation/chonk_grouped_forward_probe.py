"""Synthetic TE grouped forward accuracy only; runtime owns outer PGID and termination."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
PACKAGE = Path("/opt/venv/lib/python3.12/site-packages")
PREFIX = "transformer_engine/pytorch/"
CONFIG = PREFIX + "triton_kernels/gmm/configs/gfx942-GMM.json"
SELECTOR = "NVTE_USE_GROUPED_GEMM_TRITON"
BASELINE = {"NVTE_USE_ROCM":"1", "NVTE_USE_CAST_TRANSPOSE_TRITON":"1", "NVTE_USE_HIPBLASLT":"1"}
ALIASES = ("general_grouped_gemm", "general_grouped_gemm_triton")
PROJECTIONS = (("fc1", 1792, 3840), ("fc2", 1920, 1792))
CASES = {"balanced": [2048] * 16, "ragged": [0,1,127,128,129,511,1023,2047,2048,2049,3071,3072,4095,4096,17,33]}
VERSIONS = {"torch": "2.14.0+rocm10.1.0", "transformer-engine": "2.17.1+130099ce4", "triton": "3.8.0+git669b31ac.rocm10.1.0"}
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
    CONFIG: (5493, "6e1d4efb9dd8c6b6c12a1148b558120ec6f96afbea4eba125c75cb3e7c8f009a"),
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
}
COMMAND = ["python3", "-u", "/apertus/validation.py", "run"]


def digest(path):
    """Hash selected source/ELF/operand evidence, never an image or cache payload."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): h.update(block)
    return h.hexdigest()


def retain(path, record):
    """Replace only this invocation's owned record, preserving an active primary error."""
    primary = sys.exception()
    try:
        temporary = path.with_suffix(".tmp")
        with temporary.open("x") as stream: json.dump(record, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    except BaseException as error:
        if primary is None: raise
        primary.add_note(f"secondary evidence error: {error}")


def environment(env, selector=None):
    """Reject inherited backend/search overrides; selector is immutable worker startup state."""
    if selector not in (None,"0","1"): raise ValueError("startup selector value")
    for k, v in env.items():
        if k.startswith(("TE_HIPBLASLT_", "PYTORCH_TUNABLEOP_", "HIPBLASLT_TUNING", "NVTE_USE_", "NVTE_GROUPED_")):
            if k in BASELINE and v == BASELINE[k]: continue
            if k != SELECTOR or v != selector: raise ValueError(f"unreviewed backend environment: {k}")
    if env.get(SELECTOR) != selector: raise ValueError("startup selector mismatch")
    return {k:env.get(k) for k in BASELINE}


def validate_launch(settings, env):
    """Require image910 and the existing default-bound four-rank 180s command contract."""
    r, p = settings["resources"], settings["procedure"]
    wanted = dict(nodes=1, gpus_per_node=4, cpus_per_node=192, mpi="pmix", time="00:05:00",
                  reservation="cscs", cpu_bind="map_ldom:0,1,2,3", distribution="block:block")
    rank, local = int(env["SLURM_PROCID"]), int(env["SLURM_LOCALID"])
    if (settings["image"]["sha256"] != IMAGE or any(r.get(k) != v for k,v in wanted.items())
            or r.get("cpus_per_task") is not None or p["kind"] != "commands" or p["timeout"] != 180
            or p["commands"] != [COMMAND] or int(env["SLURM_NTASKS"]) != 4 or int(env["SLURM_NODEID"]) != 0
            or not 0 <= rank == local < 4 or len(os.sched_getaffinity(0)) != 48): raise ValueError("launch contract")
    return rank, local


def provenance():
    """Check installed bytes before vendor import; no compiled-source attestation."""
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    if versions != VERSIONS: raise ValueError("package versions")
    for name, (size, sha) in PINS.items():
        path = PACKAGE / name
        if path.resolve(strict=True) != path or path.stat().st_size != size or digest(path) != sha: raise ValueError(f"installed pin: {name}")
    return dict(image=IMAGE, versions=versions, pins=PINS, binary_attestation=False)


def mapped_elfs():
    """Require the pinned TE ELF paths actually mapped in this worker, without binary attestation."""
    paths={line.split(maxsplit=5)[5] for line in Path("/proc/self/maps").read_text().splitlines()
           if len(line.split(maxsplit=5))==6}
    expected={str(PACKAGE/name) for name in PINS if name.endswith(".so")}
    if not expected <= paths: raise ValueError("pinned TE ELF mappings absent")
    return sorted(expected)


def metrics_pass(m):
    """Apply finite/full-element/elementwise/RMS gates independently, including empty experts."""
    if set(m) != {"finite","numel","checked","mismatches","max_abs","rms_error","reference_rms"}: return False
    if m["finite"] is not True or any(type(m[k]) is not int for k in ("numel","checked","mismatches")): return False
    if m["numel"] < 0 or m["checked"] != m["numel"] or m["mismatches"] != 0: return False
    if any(type(m[k]) not in (int,float) or not math.isfinite(m[k]) or m[k] < 0 for k in ("max_abs","rms_error","reference_rms")): return False
    return m["rms_error"] <= 1e-6 + 2**-7 * m["reference_rms"] and (m["numel"] > 0 or m["max_abs"] == m["rms_error"] == m["reference_rms"] == 0)


@contextmanager
def observe(module, expected):
    """Delegate once, count successful consumers and restore only owned aliases on all paths."""
    if set(expected) != set(ALIASES) or any(getattr(module,k) is not v for k,v in expected.items()): raise ValueError("alias identity")
    counts, wrappers = dict.fromkeys(ALIASES,0), {}
    try:
        for name, original in expected.items():
            def wrapped(*args, _name=name, _original=original, **kwargs):
                value = _original(*args, **kwargs); counts[_name] += 1; return value
            wrappers[name] = wrapped; setattr(module,name,wrapped)
        yield counts
    finally:
        for name, wrapper in wrappers.items():
            if getattr(module,name) is wrapper: setattr(module,name,expected[name])


def branch_pass(selector, counts):
    """Exactly four successful selected calls; no calls through the opposite consumer."""
    return selector in ("0","1") and counts == dict(zip(ALIASES, (4,0) if selector == "0" else (0,4))) and all(type(v) is int for v in counts.values())


def pair(a, b):
    """Reject incomplete/stale/aliased worker evidence; require exact shared operands and pins."""
    keys = {f"{p}/{case}" for p,_,_ in PROJECTIONS for case in CASES}
    for row, selector in ((a,"0"),(b,"1")):
        if (row.get("status") != "passed" or row.get("selector") != selector or not branch_pass(selector,row.get("counts",{}))
                or set(row.get("operands",{})) != keys or len(row.get("checks",[])) != 4): raise ValueError("incomplete worker")
        if {check["key"] for check in row["checks"]} != keys: raise ValueError("check coverage")
        for check in row["checks"]:
            projection,case=check["key"].split("/"); n=dict((p,n) for p,_,n in PROJECTIONS)[projection]
            if (set(check) != {"key","experts","splits"} or check["splits"] != CASES[case] or len(check["experts"]) != 16
                    or any(m["numel"] != rows*n or not metrics_pass(m) for rows,m in zip(CASES[case],check["experts"]))): raise ValueError("numerical worker")
        for operands in row["operands"].values():
            if set(operands) != {"input","weights"} or len(operands["weights"]) != 16: raise ValueError("operand schema")
            if any(type(h) is not str or len(h)!=64 or set(h)-set("0123456789abcdef") for h in [operands["input"],*operands["weights"]]): raise ValueError("operand hashes")
        if row["seed"] != 42 or type(row["pid"]) is not int or row["pid"]<=0: raise ValueError("worker identity")
        initial,final=row["baseline_environment_initial"],row["baseline_environment_final"]
        if any(set(flags)!=set(BASELINE) or any(v not in (None,"1") for v in flags.values()) for flags in (initial,final)): raise ValueError("baseline flags")
        if any(v is not None and final[k]!=v for k,v in initial.items()): raise ValueError("baseline flag removed")
    if a["pid"] == b["pid"] or a["pgid"] != b["pgid"]: raise ValueError("fresh inherited-PGID workers required")
    for key in ("rank","seed","provenance","origins","operands","driver_sha256","precision","mapped_elfs","layouts","baseline_environment_initial","baseline_environment_final"):
        if a[key] != b[key]: raise ValueError(f"worker mismatch: {key}")


def tensor_digest(tensor):
    """Hash exact contiguous CPU BF16 bytes; postflight device copies are explicitly untimed."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(dtype=__import__("torch").uint8).numpy().tobytes()).hexdigest()


def layout(tensor):
    """Record actual synthetic storage metadata, never a production layout capture."""
    return dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),device=str(tensor.device),offset=tensor.storage_offset())


def select_device(torch, local):
    """Select the Slurm local device before TE can resolve Triton's active target."""
    if torch.cuda.device_count()!=4 or not torch.version.hip: raise ValueError("all four ROCm devices required")
    torch.cuda.set_device(local)
    if torch.cuda.current_device()!=local or not torch.cuda.get_device_properties(local).gcnArchName.startswith("gfx942"): raise ValueError("gfx942/local device required")


def worker(directory, rank, local, selector):
    """Execute four public forwards, never backward, benchmarking or forced config selection."""
    record = dict(status="failed",rank=rank,selector=selector,seed=42,pid=os.getpid(),pgid=os.getpgrp(),driver_sha256=digest(__file__),checks=[],operands={})
    try:
        record["baseline_environment_initial"] = environment(os.environ,selector); record["provenance"] = provenance()
        from apertus_amd.runtime import preserve_affinity, require_affinity
        with preserve_affinity(record.setdefault("affinity",{})) as initial:
            import torch
            select_device(torch,local)
            import transformer_engine.pytorch as te
            module = importlib.import_module("transformer_engine.pytorch.module.grouped_linear")
            native = importlib.import_module("transformer_engine.pytorch.cpp_extensions.gemm")
            candidate = importlib.import_module("transformer_engine.pytorch.triton_kernels.grouped_gemm")
            if torch.cuda.current_device()!=local: raise ValueError("device changed during TE imports")
            record["local_device"] = torch.cuda.current_device()
        record["precision"] = dict(matmul=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32)
        if record["precision"] != dict(matmul="highest",allow_tf32=False): raise ValueError("reference precision not inherited highest/noTF32")
        expected = dict(zip(ALIASES,(native.general_grouped_gemm,candidate.general_grouped_gemm_triton)))
        origins = {"GroupedLinear":inspect.getfile(te.GroupedLinear),**{k:inspect.getfile(v) for k,v in expected.items()}}
        for key, relative in zip(("GroupedLinear",*ALIASES),("module/grouped_linear.py","cpp_extensions/gemm.py","triton_kernels/grouped_gemm.py")):
            if Path(origins[key]).resolve() != PACKAGE / (PREFIX + relative): raise ValueError("loaded callable origin")
        record["origins"] = origins
        record["mapped_elfs"] = mapped_elfs()
        generator = torch.Generator(device="cpu").manual_seed(42)
        with torch.no_grad(), observe(module,expected) as counts:
            for name,k,n in PROJECTIONS:
                if torch.cuda.current_device()!=local: raise ValueError("device changed before constructor")
                layer = te.GroupedLinear(16,k,n,bias=False,params_dtype=torch.bfloat16,fuse_wgrad_accumulation=False,device="cuda",single_grouped_weight=False,single_grouped_bias=False)
                weights = [torch.randn(n,k,dtype=torch.bfloat16,generator=generator)*0.02 for _ in range(16)]
                hashes = [tensor_digest(w) for w in weights]
                for i,w in enumerate(weights): getattr(layer,f"weight{i}").requires_grad_(False).copy_(w)
                for case,splits in CASES.items():
                    if torch.cuda.current_device()!=local: raise ValueError("device changed before data")
                    key = f"{name}/{case}"; host = torch.randn(sum(splits),k,dtype=torch.bfloat16,generator=generator)*0.25
                    before = dict(input=tensor_digest(host),weights=hashes); record["operands"][key] = before
                    x = host.to(device="cuda"); sizes = torch.tensor(splits,dtype=torch.int32,device=x.device)
                    if sizes.shape != (16,) or sizes.stride() != (1,) or sum(splits) != len(x) or min(splits)<0: raise ValueError("row-count contract")
                    require_affinity(initial,"grouped forward")
                    y = layer(x,list(splits),m_splits_tensor=sizes); torch.cuda.synchronize()
                    if tuple(y.shape) != (len(x),n) or y.dtype != torch.bfloat16 or layer.fp8 or layer.is_debug_iter() or layer.fuse_wgrad_accumulation: raise ValueError("public forward contract")
                    record.setdefault("layouts",{})[key]=dict(input=layout(x),weights=[layout(getattr(layer,f"weight{i}")) for i in range(16)],output=layout(y),row_counts=layout(sizes),source="canonical CPU generator seed42; hashes before CUDA transfer")
                    if torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32: raise ValueError("reference precision changed")
                    experts=[]; start=0
                    for i,rows in enumerate(splits):
                        actual = y[start:start+rows].float(); reference = x[start:start+rows].float() @ getattr(layer,f"weight{i}").float().T
                        finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all() and torch.isfinite(weights[i]).all() and torch.isfinite(host[start:start+rows]).all())
                        diff=actual-reference
                        m=dict(finite=finite,numel=rows*n,checked=rows*n,mismatches=int((diff.abs()>0.02+0.02*reference.abs()).sum().item()),max_abs=diff.abs().max().item() if rows else 0.,rms_error=diff.square().mean().sqrt().item() if rows else 0.,reference_rms=reference.square().mean().sqrt().item() if rows else 0.)
                        m={key:None if isinstance(value,float) and not math.isfinite(value) else value for key,value in m.items()}
                        experts.append(m); start+=rows
                    record["checks"].append(dict(key=key,experts=experts,splits=splits))
                    if not all(metrics_pass(m) for m in experts): raise ValueError("independent FP32 numerical gate")
                    if before != dict(input=tensor_digest(x),weights=[tensor_digest(getattr(layer,f"weight{i}")) for i in range(16)]): raise ValueError("operand mutation")
                    del x,y,host,actual,reference,diff
                del layer,weights
            record["counts"] = dict(counts)
        if not branch_pass(selector,record["counts"]): raise ValueError("selected consumer counts")
        require_affinity(initial,"grouped completion"); record["baseline_environment_final"] = environment(os.environ,selector)
        if any(v is not None and record["baseline_environment_final"][k]!=v for k,v in record["baseline_environment_initial"].items()): raise ValueError("baseline flag removed")
        if provenance() != record["provenance"]: raise ValueError("pins changed")
        record["status"]="passed"
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"; raise
    finally: retain(directory/"worker.json",record)


def quiescent():
    """Bounded own-outer-PGID quiescence check before deleting private scratch."""
    for index, entry in enumerate(Path("/proc").iterdir()):
        if index >= 8192: return False
        if not entry.name.isdigit() or int(entry.name)==os.getpid(): continue
        try: fields=(entry/"stat").read_text().rsplit(")",1)[1].split()
        except (FileNotFoundError,ProcessLookupError): continue
        if int(fields[2])==os.getpgrp() and fields[0] not in {"Z","X"}: return False
    return True


def run(settings):
    """Supervise fresh sequential children under one deadline; failure retains scratch."""
    from apertus_amd.config import CACHE_PATHS
    rank,local=validate_launch(settings,os.environ); environment(os.environ)
    output=Path(f"grouped-forward-{rank}").resolve(); output.mkdir()
    Path(settings["scratch_root"]).mkdir(parents=True,exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix=f"grouped-{os.environ['SLURM_JOB_ID']}-{rank}-",dir=settings["scratch_root"]))
    record=dict(status="failed",cleanup="retained",scratch=str(scratch),pgid=os.getpgrp(),workers=[]); deadline=time.monotonic()+160
    try:
        for selector in ("0","1"):
            directory=output/selector; directory.mkdir(); env=dict(os.environ,**{SELECTOR:selector})
            for key,relative in CACHE_PATHS.items():
                cache=scratch/selector/relative; cache.mkdir(parents=True,exist_ok=True); env[key]=str(cache)
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError("shared worker deadline")
            with (directory/"native.log").open("x") as log:
                child=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"worker","--directory",str(directory),"--selector",selector],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
                status=child.wait(timeout=remaining)
            value=json.loads((directory/"worker.json").read_text())
            if (status or value.get("status")!="passed" or value.get("pgid")!=os.getpgrp() or value.get("pid")!=child.pid
                    or value.get("rank")!=rank or value.get("selector")!=selector or value.get("driver_sha256")!=digest(__file__)): raise ValueError("closed worker failed")
            record["workers"].append(value)
        pair(*record["workers"])
        if time.monotonic()>deadline or not quiescent(): raise ValueError("deadline/quiescence; scratch retained")
        shutil.rmtree(scratch); record.update(status="passed",cleanup="removed")
    except BaseException as error:
        record["error"]=f"{type(error).__name__}: {error}"; raise
    finally: retain(output/"summary.json",record)


def main():
    """Expose the fixed accuracy-only command and its private startup-selector child."""
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("mode",choices=("run","worker"))
    parser.add_argument("--directory",type=Path); parser.add_argument("--selector",choices=("0","1")); args=parser.parse_args()
    if (args.mode=="worker") != (args.directory is not None and args.selector is not None): parser.error("worker needs private directory and selector")
    settings=json.loads(Path("/apertus/resolved.json").read_text()); rank,local=validate_launch(settings,os.environ)
    if args.mode=="run": run(settings)
    else: worker(args.directory,rank,local,args.selector)


if __name__ == "__main__": main()
