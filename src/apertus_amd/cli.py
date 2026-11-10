"""Plan or execute explicit experiments with the same launch options."""

import argparse
import json
from pathlib import Path
import subprocess

from .config import CACHE_PATHS, phases, resolve
from .runs import run, step_command


def plan(experiment: Path, **options) -> dict:
    """Show resolved inputs and commands without fetching, probing, or writing."""
    value = resolve(experiment, **options)
    value["phases"] = phases(value, Path("/apertus/output"))
    value["step"] = step_command(value, Path("<run>"))
    if value["procedure"]["kind"] == "commands":
        step = value.pop("step")
        value["steps"] = [[*step, str(index)] for index in range(len(value["phases"]))]
        return value
    value["runtime_environment"] = {name: f"<private-cache>/{relative}" for name, relative in CACHE_PATHS.items()}
    value["runtime_environment"].update(RANK="<Slurm rank>", WORLD_SIZE="<Slurm tasks>",
                                         LOCAL_RANK="<Slurm local rank / visible GPU index>", MASTER_ADDR="<rank-zero node>",
                                         MASTER_PORT="<rank-zero port>", PYTHONPATH="<source>:/apertus/tooling",
                                         PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                                         OMP_NUM_THREADS=value["environment"].get("OMP_NUM_THREADS", "<bound CPU count>"),
                                         MAX_JOBS=value["environment"].get("MAX_JOBS", "<bound CPU count>"))
    return value


def main() -> None:
    """Expose one input surface for planning and either launch mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--site", type=Path, default=Path("config.toml"))
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--data-root", type=Path)
    launch = parser.add_mutually_exclusive_group(required=True)
    launch.add_argument("--job-id")
    launch.add_argument("--submit", action="store_true")
    for flag in ("nodes", "gpus-per-node", "cpus-per-task", "memory-gb"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--time")
    parser.add_argument("--reservation", help="optional Slurm reservation for a new allocation")
    parser.add_argument("--dependency", help="batch dependency: afterany:JOB_ID or afterok:JOB_ID")
    parser.add_argument("--cpu-bind")
    parser.add_argument("--distribution", help="Slurm block:<within-node policy>; defaults to the site setting")
    parser.add_argument("--cache-mode", choices=("auto", "cold", "warm"), default="auto")
    parser.add_argument("--compare-run", type=Path, help="compare validation results against a retained run")
    args = vars(parser.parse_args())
    command, experiment, checkout = (args.pop(name) for name in ("command", "experiment", "checkout"))
    try:
        if command == "plan":
            print(json.dumps(plan(experiment, **args), indent=2, sort_keys=True))
        else:
            value = resolve(experiment, **args)
            if checkout is None and value["procedure"]["kind"] != "commands":
                parser.error("run requires --checkout containing the selected commit")
            print(run(value, checkout))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        parser.exit(1, f"Failed: {detail}\n")


if __name__ == "__main__":
    main()
