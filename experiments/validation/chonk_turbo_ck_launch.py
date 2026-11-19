"""Admit the pinned one-device component before executing the native driver."""

import hashlib
import json
import os
from pathlib import Path


IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
PINS = {
    "chonk_turbo_forward.py": "8317557873ae664440e3b5ffebfbfce6389efed0d60318a92aec6ee5f215c837",
    "pins.json": "2a1a2b0ed98e800d3d09ec2391a3028c7e2cf896784987e15ed7153d248219d8",
}
PYTHON = "/opt/venv/bin/python3"


def admit(settings, environ, affinity, assets):
    """Verify placement, resolved inputs and both read-only workload files."""
    resources = settings["resources"]
    expected = {"nodes": 1, "gpus_per_node": 1, "cpus_per_task": 48, "reservation": "cscs",
                "time": "00:06:00", "mpi": "pmix", "cpu_bind": "cores"}
    if (settings["image"]["sha256"] != IMAGE
            or any(resources.get(key) != value for key, value in expected.items())
            or settings["procedure"]["timeout"] != 300):
        raise ValueError("unexpected image, resources or command timeout")
    layout = {"SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1", "SLURM_PROCID": "0",
              "SLURM_LOCALID": "0", "SLURM_NODEID": "0"}
    if any(environ.get(key) != value for key, value in layout.items()) or len(affinity) != 48:
        raise ValueError("requires one task with 48 allocated CPUs")
    visible = environ.get("ROCR_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0].strip():
        raise ValueError("requires exactly one visible ROCm GPU")
    for name, checksum in PINS.items():
        path = assets / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"pinned input differs: {name}")


def main():
    """Keep the native driver in the existing runtime-owned child process group."""
    root, output = Path("/apertus"), Path.cwd()
    assets = root / "turbo-input"
    if output != root / "output/0":
        raise ValueError("requires the run-owned command output directory")
    settings = json.loads((root / "resolved.json").read_text())
    admit(settings, os.environ, os.sched_getaffinity(0), assets)
    cache = output / "triton"
    cache.mkdir(mode=0o700)
    env = dict(os.environ, TRITON_CACHE_DIR=str(cache))
    os.execve(PYTHON, [PYTHON, "-I", "-B", "-u", str(assets / "chonk_turbo_forward.py"),
                       "--device", "0", "--private-cache", str(cache),
                       "--result", str(output / "turbo-forward.jsonl")], env)


if __name__ == "__main__":
    main()
