"""Admit one frozen Q/K preflight and retain canonical child-process ownership."""

import hashlib
import json
import os
from pathlib import Path


IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"


def command(root, env, affinity, assets=Path("/qk-input")):
    """Verify the effective resource settings and the frozen input manifest."""
    settings = json.loads((root / "resolved.json").read_text())
    resources = settings["resources"]
    expected = dict(nodes=1, gpus_per_node=1, cpus_per_task=8, reservation="cscs",
                    memory_gb=16, mpi="none", time="00:08:00", cpu_bind="cores")
    if (settings["image"]["sha256"] != IMAGE or settings["procedure"]["timeout"] != 450
            or any(resources.get(k) != v for k, v in expected.items())):
        raise ValueError("unexpected preflight image or resource settings")
    if (any(env.get(k) != "0" for k in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID"))
            or env.get("SLURM_NTASKS") != "1" or len(affinity) != 8):
        raise ValueError("requires one rank with eight bound CPUs")
    actual = hashlib.sha256((assets / "bundle.json").read_bytes()).hexdigest()
    if settings["environment"].get("APERTUS_QK_BUNDLE_SHA256") != actual:
        raise ValueError("Q/K input manifest differs")
    return ["/bin/bash", "--noprofile", "--norc", "/qk-input/native-task.sh",
            actual, "--allow-cold-fla-autotune"]


if __name__ == "__main__":
    argv = command(Path("/apertus"), os.environ, os.sched_getaffinity(0))
    os.chdir("/qk-output")
    os.execv(argv[0], argv)
