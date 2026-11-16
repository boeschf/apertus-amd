"""Check unchanged candidate attention numerics with bounded eight-worker private JIT builds."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile


IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
TESTS = Path("/opt/tests/apertus-2/primus")
PINS = {
    "check-stack.py": "723b5d42bdd410ea172764f13148260e6e57243f0ae977506fde5faba6aca83b",
    "attention-accuracy.py": "b5e72f177206d6b864971e4ad852ed8c58f835da0ff63c413e75b96a39f076e1",
}
SEED = Path("/opt/aiter-jit")
SEED_HASH = "4876a08ad1c55d0892cd91ff361716f3ba44cf1499660509e2d2b8f3d954193b"
VERSIONS = {"torch": "2.14.0+rocm10.1.0", "amd-aiter": "0.1.22.post1",
            "triton": "3.8.0+git669b31ac.rocm10.1.0"}
CPP_HASH = "88bc7ddf34a6c1294b3a39b6a957b4ea17c6f837ed53fd89ef581785f49bd9bb"
FORBIDDEN_FLAGS = ("BEVERIN_TURBO_GG", "BEVERIN_T6_EP", "BEVERIN_T6_FP8", "BEVERIN_T6_FUSED",
                   "BEVERIN_AITER_ATTN", "BEVERIN_K7_BATCH", "BEVERIN_NS_WS")


def digest(path):
    """Hash a selected script, manifest or core module, never the image payload."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_launch(settings, env):
    """Require only the reviewed eight-worker, four-rank, original-budget envelope."""
    resources, procedure = settings["resources"], settings["procedure"]
    if (settings["image"]["sha256"] != IMAGE or resources["nodes"] != 1
            or resources["gpus_per_node"] != 4 or resources["time"] != "00:15:00"
            or resources["cpus_per_node"] != 192 or resources["cpu_bind"] != "map_ldom:0,1,2,3"
            or resources.get("reservation") != "cscs" or resources["distribution"] != "block:block"
            or procedure["kind"] != "commands" or procedure["timeout"] != 720
            or procedure["commands"] != [["python3", "-u", "/apertus/validation.py"]]):
        raise ValueError("unexpected attention image/layout/procedure")
    rank, local, node, world = (int(env[key]) for key in
                              ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID", "SLURM_NTASKS"))
    if not 0 <= rank == local < 4 or node != 0 or world != 4:
        raise ValueError("requires four independent local GPU ranks on one node")
    if any(env.get(key) != value for key, value in
           {"OMP_NUM_THREADS": "1", "MAX_JOBS": "8", "AITER_LOG_MORE": "1", "NCCL_P2P_DISABLE": "1",
            "OMPI_MCA_memory": "^patcher", "HWLOC_COMPONENTS": "-gl"}.items()):
        raise ValueError("unexpected attention thread/transport environment")
    if "PREBUILD_THREAD_NUM" in env:
        raise ValueError("PREBUILD_THREAD_NUM must be absent")
    if any(env.get(key) not in (None, "0") for key in FORBIDDEN_FLAGS):
        raise ValueError("broad reference optimization settings must not be enabled")
    return rank, local


def verified_inputs():
    """Verify the existing attention oracle and seed before any vendor import or GPU work."""
    scripts = {name: digest(TESTS / name) for name in PINS}
    if scripts != PINS or digest(SEED / "seed.json") != SEED_HASH:
        raise ValueError("pinned image-local attention script/seed mismatch")
    seed = json.loads((SEED / "seed.json").read_text())
    if seed["gpu_archs"] != "gfx942" or seed["cu_num"] != 228 or seed["packages"] != VERSIONS:
        raise ValueError("unexpected seed architecture/packages")
    modules = seed["modules"]
    if set(modules) != {"module_aiter_core.so"}:
        raise ValueError("unexpected core seed modules")
    if any((SEED / name).is_symlink() or digest(SEED / name) != checksum for name, checksum in modules.items()):
        raise ValueError("baked core identity mismatch")
    return {"scripts": scripts, "seed": seed}


def verified_jit_source():
    """Verify the selected Ninja consumer as text without importing AITER."""
    path = Path(importlib.metadata.distribution("amd-aiter").locate_file("aiter/jit/utils/cpp_extension.py"))
    if path.is_symlink() or digest(path) != CPP_HASH:
        raise ValueError("pinned AITER Ninja consumer source mismatch")
    return {"path": str(path), "sha256": CPP_HASH}


def attention(torch, flash_attn_func, policy):
    """Use the image's attention shapes and numerical policy, without its unrelated Turbo checks."""
    torch.manual_seed(123)
    q = torch.randn(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    actual = flash_attn_func(q, k, v, dropout_p=0.0, causal=True, deterministic=False,
                            return_lse=True, how_v3_bf16_cvt=1)
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    references = [tensor.detach().float().cpu().requires_grad_() for tensor in (q, k, v)]
    expected = torch.nn.functional.scaled_dot_product_attention(
        *[tensor.transpose(1, 2) for tensor in references], is_causal=True).transpose(1, 2)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.03, atol=0.03)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient.float().cpu())
    precise = [tensor.detach().double().cpu().requires_grad_() for tensor in (q, k, v)]
    precise_out = torch.nn.functional.scaled_dot_product_attention(
        *[tensor.transpose(1, 2) for tensor in precise], is_causal=True).transpose(1, 2)
    precise_out.backward(gradient.double().cpu())
    controls = [tensor.detach().clone().requires_grad_() for tensor in (q, k, v)]
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        control_out = torch.nn.functional.scaled_dot_product_attention(
            *[tensor.transpose(1, 2) for tensor in controls], is_causal=True).transpose(1, 2)
    control_out.backward(gradient)
    metrics = {}
    for name, tensor, reference, accurate, control in zip(("q", "k", "v"), (q, k, v), references, precise, controls):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all().item():
            raise ValueError(f"missing/nonfinite attention gradient: {name}")
        actual_grad = tensor.grad.detach().double().cpu()
        accurate_grad = accurate.grad.detach()
        control_grad = control.grad.detach().double().cpu()
        if not (torch.isfinite(accurate_grad).all().item() and torch.isfinite(control_grad).all().item()):
            raise ValueError(f"nonfinite attention reference/control: {name}")
        norm = torch.linalg.vector_norm(accurate_grad).item()
        if norm <= 0:
            raise ValueError(f"zero attention reference norm: {name}")
        actual_error, control_error = actual_grad - accurate_grad, control_grad - accurate_grad
        item = policy.validate_gradient_error(
            actual_error.abs().max().item(), control_error.abs().max().item(),
            torch.linalg.vector_norm(actual_error).item() / norm,
            torch.linalg.vector_norm(control_error).item() / norm)
        item["original_elementwise_mismatches"] = (~torch.isclose(
            tensor.grad.float().cpu(), reference.grad, rtol=0.05, atol=0.005)).sum().item()
        metrics[name] = item
    torch.cuda.synchronize()
    return {"shape": [1, 128, 2, 128], "dtype": "bfloat16", "causal": True,
            "dropout": 0.0, "how_v3_bf16_cvt": 1, "forward_rtol": 0.03, "forward_atol": 0.03,
            "gradients": metrics}


def run(settings):
    """Own private JIT caches and retain per-rank results under the ordinary bounded commands runtime."""
    from apertus_amd.config import CACHE_PATHS
    from apertus_amd.runtime import preserve_affinity, require_affinity

    rank, local = validate_launch(settings, os.environ)
    path = Path(f"attention-rank-{rank}.json")
    record = {"status": "running", "rank": rank, "local_rank": local, "image_sha256": IMAGE,
              "scope": "isolated AITER attention forward/backward; no reference Turbo/dispatcher/training acceptance"}
    def save():
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    save()
    try:
        if len(os.sched_getaffinity(0)) != 48:
            raise ValueError("requires inherited 48-CPU rank affinity")
        record["inputs"] = verified_inputs()
        record["versions"] = {name: importlib.metadata.version(name) for name in VERSIONS}
        if record["versions"] != VERSIONS:
            raise ValueError("pinned attention compute versions differ")
        record["jit_consumer"] = verified_jit_source()
        record["jit_environment"] = {key: os.environ[key] for key in ("MAX_JOBS", "OMP_NUM_THREADS", "AITER_LOG_MORE")}
        root = Path(settings["scratch_root"])
        root.mkdir(parents=True, exist_ok=True)
        # Synchronous image-local JIT children inherit the outer commands process group.
        with tempfile.TemporaryDirectory(prefix=f"attention-{os.environ['SLURM_JOB_ID']}-{rank}-", dir=root) as private:
            record["scratch"] = private
            for key, relative in CACHE_PATHS.items():
                directory = Path(private) / relative
                directory.mkdir()
                os.environ[key] = str(directory)
            cache = Path(os.environ["AITER_JIT_DIR"])
            modules = record["inputs"]["seed"]["modules"]
            for name in modules:
                shutil.copyfile(SEED / name, cache / name)
            affinity = record["affinity"] = {}
            with preserve_affinity(affinity) as initial:
                import torch
                if torch.cuda.device_count() != 4 or not torch.version.hip:
                    raise ValueError("all four ROCm devices must remain visible")
                torch.cuda.set_device(local)
                torch.set_num_threads(1)
                properties = torch.cuda.get_device_properties(local)
                if not properties.gcnArchName.startswith("gfx942") or properties.multi_processor_count != 228:
                    raise ValueError("requires gfx942 MI300A")
                from aiter import flash_attn_func
            record["selected_device"] = torch.cuda.current_device()
            if record["selected_device"] != local:
                raise ValueError("attention GPU differs from Slurm local rank")
            spec = importlib.util.spec_from_file_location("pinned_attention_policy", TESTS / "attention-accuracy.py")
            policy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(policy)
            require_affinity(initial, "attention numerical checks")
            save()
            record["numerics"] = attention(torch, flash_attn_func, policy)
            require_affinity(initial, "attention completion")
            record["seed_unchanged"] = digest(SEED / "seed.json") == SEED_HASH and all(
                digest(SEED / name) == checksum for name, checksum in modules.items())
            if not record["seed_unchanged"]:
                raise ValueError("immutable attention seed changed")
        record.update(status="passed", scratch_cleanup="removed")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        if record.get("scratch"):
            record["scratch_cleanup"] = "removed" if not Path(record["scratch"]).exists() else "retained"
        raise
    finally:
        save()


if __name__ == "__main__":
    run(json.loads(Path("/apertus/resolved.json").read_text()))
