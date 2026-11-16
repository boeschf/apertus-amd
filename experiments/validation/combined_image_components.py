"""Select each local GPU before same-process embedded component tests; no training acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys


IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
SCRIPTS = {
    "aiter": Path("/opt/tests/apertus-2/primus/aiter-seed-smoke.py"),
    "uccl": Path("/opt/tests/apertus-2/primus/uccl-lifetime-smoke.py"),
}
SCRIPT_SHA256 = {
    "aiter": "af5b15651397c7b3758072b1c7c4f78e48c9f1608c9e6bcc5c72996a47cd588c",
    "uccl": "d782cf20994aa6a2464a144944c67f6524b5e14016a0ab32c20329b3ac517698",
}
SEED = Path("/opt/aiter-jit/seed.json")
SEED_SHA256 = "4876a08ad1c55d0892cd91ff361716f3ba44cf1499660509e2d2b8f3d954193b"


def validate_launch(settings, env):
    """Require the explicit candidate, two-node/eight-rank block layout and bounded component policy."""
    resources, procedure = settings["resources"], settings["procedure"]
    if (settings["image"]["sha256"] != IMAGE or resources["nodes"] != 2
            or resources["gpus_per_node"] != 4 or resources["time"] != "00:30:00"
            or resources.get("reservation") != "cscs" or resources["distribution"].split(":")[0] != "block"
            or procedure["kind"] != "commands" or procedure["timeout"] != 660
            or procedure["commands"] != [["python3", "-u", "/apertus/validation.py", phase] for phase in SCRIPTS]):
        raise ValueError("unexpected component image/layout/procedure")
    rank, local, node, world = (int(env[name]) for name in
                              ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID", "SLURM_NTASKS"))
    if world != 8 or not 0 <= rank < 8 or not 0 <= local < 4 or rank != node * 4 + local:
        raise ValueError("requires eight block-placed Slurm ranks, four per node")
    required = {"OMP_NUM_THREADS": "1", "MAX_JOBS": "2", "NCCL_P2P_DISABLE": "1",
                "OMPI_MCA_memory": "^patcher", "HWLOC_COMPONENTS": "-gl"}
    if any(env.get(name) != value for name, value in required.items()):
        raise ValueError("component thread/transport environment differs")
    return rank, local, node


def environment():
    """Record relevant effective settings without exposing unrelated submitting-shell secrets."""
    return {name: value for name, value in os.environ.items()
            if name.startswith(("SLURM_", "UCCL_", "FI_", "NCCL_", "RCCL_", "OMPI_", "PMIX_", "HWLOC_"))
            or name in {"OMP_NUM_THREADS", "MAX_JOBS", "BASH_ENV", "ROCR_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "AITER_JIT_DIR", "TMPDIR",
                        "GPU_ARCHS", "CU_NUM", "AITER_USE_SYSTEM_TRITON"}}


def small_bytes(path):
    """Read only the pinned small embedded scripts/manifest, never an image or cache tree."""
    with path.open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError(f"embedded metadata/script exceeds bound: {path}")
    return raw


def run(phase, settings):
    """Run the unchanged embedded main after local-device selection, retaining per-rank evidence."""
    rank, local, node = validate_launch(settings, os.environ)
    record = {"status": "running", "phase": phase, "image_sha256": IMAGE, "rank": rank,
              "local_rank": local, "node": node, "pid": os.getpid(), "hostname": socket.gethostname(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "environment_before": environment(),
              "embedded_scripts": {}, "scope": "component correctness only; no gradients/full-model/performance acceptance"}
    output = Path(f"component-rank-{rank}.json")  # Existing commands runtime supplies a phase-specific cwd.
    def save():
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    save()
    previous = sys.argv
    try:
        for name, path in SCRIPTS.items():
            checksum = hashlib.sha256(small_bytes(path)).hexdigest()
            record["embedded_scripts"][name] = {"path": str(path), "sha256": checksum}
            if checksum != SCRIPT_SHA256[name]:
                raise ValueError(f"embedded script hash mismatch: {name}")
        raw = small_bytes(SEED)
        record["seed_manifest"] = {"path": str(SEED), "sha256": hashlib.sha256(raw).hexdigest(),
                                   "content": json.loads(raw)}
        if record["seed_manifest"]["sha256"] != SEED_SHA256:
            raise ValueError("baked seed manifest hash mismatch")
        import torch  # Lazy: CPU planning/import never loads vendor code or initializes a GPU.

        count = torch.cuda.device_count()
        record["visible_device_count"] = count
        if count != 4 or not torch.version.hip:
            raise ValueError("all four ROCm GPUs must remain visible to each rank")
        torch.cuda.set_device(local)
        torch.set_num_threads(1)
        record["selected_device"] = torch.cuda.current_device()
        record["devices"] = [{"index": index, "name": torch.cuda.get_device_properties(index).name,
                               "arch": torch.cuda.get_device_properties(index).gcnArchName,
                               "compute_units": torch.cuda.get_device_properties(index).multi_processor_count}
                              for index in range(count)]
        if record["selected_device"] != local:
            raise ValueError("selected GPU differs from Slurm local rank")
        save()  # Preserve script/seed/device identity before embedded GPU work or a native abort.
        sys.argv = [str(SCRIPTS[phase])]
        runpy.run_path(str(SCRIPTS[phase]), run_name="__main__")
        record["seed_manifest_after_sha256"] = hashlib.sha256(small_bytes(SEED)).hexdigest()
        if record["seed_manifest_after_sha256"] != SEED_SHA256:
            raise ValueError("baked seed manifest changed")
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        sys.argv = previous
        record["environment_after"] = environment()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=SCRIPTS)
    args = parser.parse_args()
    run(args.phase, json.loads(Path("/apertus/resolved.json").read_text()))
