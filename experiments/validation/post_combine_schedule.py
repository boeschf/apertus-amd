"""Run and check the reviewed signed local post-combine schedule A/B."""

import json
import os
from pathlib import Path
import sys

from apertus_amd.artifacts import sha256, write_json
from apertus_amd.entrypoint import train as run_entrypoint


COMMIT = "2988ed82702cb97eef6e23c1cfa263c69f594d9a"
TREE = "d10bc1e1e764cd5da916011a32bb895071aea5b3"
ARCHIVE = "d6ddcfaa036b6be5d0ca3dd691e20d83cc7061b252f31614f3605ffc86563348"
MANIFEST = "6b8165598bf892c22ea847d7e8b08e3ada3cc53b23bfb71b06b4d3e238bb5090"
IMAGE = "910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490"
ENTRYPOINT = "tests/test_utils/python_scripts/qualify_post_combine_replay_schedule.py"
PASS_LINE = "Five local schedule microbatches passed bitwise parity and exact delayed drain."


def verify_inputs(frozen, manifest, entrypoint, args, env):
    """Reject other topologies, source selections, and extra fixture arguments."""
    resources = frozen["resources"]
    if (
        entrypoint != ENTRYPOINT or args or frozen["entrypoint"] != ENTRYPOINT or frozen["args"]
        or env.get("WORLD_SIZE") != "1" or env.get("RANK") != "0"
        or resources["nodes"] != 1 or resources["gpus_per_node"] != 1
        or frozen["image"]["sha256"] != IMAGE or frozen["source"]["commit"] != COMMIT
        or frozen["procedure"]["kind"] != "probe" or frozen["procedure"].get("auxiliary_source")
        or not __debug__
    ):
        raise ValueError("unexpected schedule A/B inputs (assertions must be enabled)")
    if sha256(manifest) != MANIFEST:
        raise ValueError("unexpected schedule source manifest SHA-256")
    identity = json.loads(manifest.read_text())
    if (
        identity["tree"] != TREE or identity["sha256"] != ARCHIVE
        or identity["selection"]["commit"] != COMMIT
    ):
        raise ValueError("unexpected signed schedule source identity")


def train(output, source, entrypoint, args):
    """Execute the signed entry point without code injection; verify its source imports."""
    frozen = json.loads((output.parent / "resolved.json").read_text())
    manifest = Path("/apertus/source/manifest.json")
    verify_inputs(frozen, manifest, entrypoint, args, os.environ)
    run_entrypoint(output, source, entrypoint, args, manifest=manifest)


def check(run):
    """Require verified imports and the signed fixture's terminal parity/lifecycle marker."""
    output = run / "output"
    result = dict(status="failed", scope="single-GPU local schedule; no distributed, fit or throughput verdict")
    try:
        imports = json.loads((output / "imports-0.json").read_text())
        lines = (output / "rank-0.log").read_text().splitlines()
        if imports["verified_modules"] <= 0 or lines.count(PASS_LINE) != 1:
            raise ValueError("missing source import receipt or unique schedule-pass marker")
        result.update(status="passed", verified_modules=imports["verified_modules"], microbatches=5,
                      initial_live_forwards=4, bitwise_equal=True, delayed_projection_wgrad="one per backward")
    finally:
        write_json(output / "checks.json", result)


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["train", output, source, entrypoint, *args]:
            train(Path(output), Path(source), entrypoint, args)
        case ["check", run]:
            check(Path(run))
        case _:
            raise SystemExit("unexpected schedule A/B invocation")
