"""One resolver for explicit experiment inputs and full-node launch defaults."""

from pathlib import Path
import re
import tomllib

from .artifacts import sha256


CACHE_PATHS = {"TMPDIR": "tmp", "XDG_CACHE_HOME": "xdg", "TRITON_CACHE_DIR": "triton",
               "TORCH_EXTENSIONS_DIR": "torch-extensions", "TORCHINDUCTOR_CACHE_DIR": "inductor",
               "AITER_JIT_DIR": "aiter", "MIOPEN_USER_DB_PATH": "miopen-db", "MIOPEN_CUSTOM_CACHE_DIR": "miopen"}
RESERVED_ENV = {*CACHE_PATHS, "PYTHONPATH", "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HWLOC_COMPONENTS"}


def option(args: list[str], name: str, default=None):
    """Read a native CLI option in either --name value or --name=value form."""
    for index in range(len(args) - 1, -1, -1):
        item = args[index]
        if item == name:
            if index + 1 == len(args):
                raise ValueError(f"missing value for {name}")
            return args[index + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default


def phases(experiment: dict, output: Path) -> list[dict]:
    """Expand the selected procedure into explicit commands."""
    if experiment["procedure"]["kind"] == "commands":
        return [{"name": str(index), "output": str(output / str(index)), "argv": command}
                for index, command in enumerate(experiment["procedure"]["commands"])]
    restart = experiment["procedure"]["kind"] == "restart"
    result = []
    for name in (("continuous", "prefix", "resume") if restart else ("train",)):
        directory = output / name if restart else output
        args = ["--save", str(directory / "checkpoints")]
        if name == "prefix":
            args += ["--exit-interval", str(experiment["procedure"]["split"])]
        if name == "resume":
            args += ["--load", str(output / "prefix/checkpoints")]
        result.append({"name": name, "output": str(directory), "extra_args": args})
    return result


def resolve(experiment: Path, *, site: Path = Path("config.toml"), **overrides) -> dict:
    """Read settings without subprocesses or writes; retain unresolved allocation values."""
    path = experiment / "experiment.toml" if experiment.is_dir() else experiment
    value = tomllib.loads(path.read_text())
    unknown = set(value) - {"nodes", "entrypoint", "args", "prepare", "source", "image", "environment", "procedure"}
    if unknown:
        raise ValueError(f"unknown experiment settings: {sorted(unknown)}")
    if "nodes" in value and (type(value["nodes"]) is not int or value["nodes"] < 1):
        raise ValueError("experiment nodes must be a positive integer")
    name = path.parent.name
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise ValueError("experiment directory must have a simple experiment ID")
    value.setdefault("procedure", {"kind": "train"})
    procedure = value["procedure"]
    if procedure["kind"] not in {"train", "restart", "commands"}:
        raise ValueError("procedure must be train, restart, or commands")
    validation = None
    if "validation" in procedure:
        script = (path.parent / procedure["validation"]).resolve()
        if procedure["kind"] == "commands" or script.suffix != ".py" or not script.is_file():
            raise ValueError("training validation requires an existing Python script")
        validation = {"path": str(script), "sha256": sha256(script)}
    if procedure["kind"] == "restart" and validation is None:
        raise ValueError("restart procedure requires an explicit validation script")
    if procedure["kind"] == "commands":
        if any(key in value for key in ("source", "entrypoint", "args", "prepare")):
            raise ValueError("commands procedure uses image executables, not source/training settings")
        commands = procedure.get("commands")
        if not isinstance(commands, list) or not commands or any(
            not isinstance(command, list) or not command or any(not isinstance(a, str) or not a or "\0" in a for a in command)
            for command in commands
        ):
            raise ValueError("commands must contain nonempty command argument arrays")
        if type(procedure.get("timeout")) is not int or procedure["timeout"] <= 0:
            raise ValueError("commands require a positive timeout in seconds")
        if overrides.get("compare_run") or overrides.get("cache_mode", "auto") not in (None, "auto"):
            raise ValueError("commands do not use training comparison or compiler caches")
    else:
        if overrides.get("compare_run") and validation is None:
            raise ValueError("compare-run requires an explicit validation script")
        entrypoint = Path(value["entrypoint"])
        if entrypoint.is_absolute() or ".." in entrypoint.parts:
            raise ValueError("entrypoint must be inside the selected source")
        if not re.fullmatch(r"[0-9a-f]{40}", value["source"]["commit"]):
            raise ValueError("source requires a full commit")
    if not re.fullmatch(r"[0-9a-f]{64}", value["image"]["sha256"]):
        raise ValueError("image requires a published SHA-256")
    args = value.get("args", [])
    if not isinstance(args, list) or any(not isinstance(a, str) or "\0" in a for a in args):
        raise ValueError("args must be native command-line strings")
    value.setdefault("prepare", [])
    if any(not command or not isinstance(command, list) or any(not isinstance(a, str) for a in command)
           for command in value["prepare"]):
        raise ValueError("prepare must contain command argument arrays")
    reserved = {"--save"}
    if procedure["kind"] == "restart":
        reserved |= {"--load", "--exit-interval"}
        steps = int(option(args, "--train-iters", 0))
        split = procedure["split"]
        if type(split) is not int or not 0 < split < steps or int(option(args, "--save-interval", 0)) != split:
            raise ValueError("restart split must precede train-iters and match save-interval")
    if any(a.split("=", 1)[0] in reserved for a in args):
        raise ValueError(f"procedure owns these arguments: {sorted(reserved)}")
    settings = tomllib.loads(site.read_text())
    if settings["cache"]["max_gb"] <= 0 or settings["cache"]["timeout"] <= 0:
        raise ValueError("cache capacity and coordination timeout must be positive")
    resources = dict(settings["slurm"])
    if "nodes" in value:
        resources["nodes"] = value["nodes"]
    for key in ("nodes", "gpus_per_node", "cpus_per_task", "memory_gb", "time", "cpu_bind", "distribution", "reservation"):
        if overrides.get(key) is not None:
            resources[key] = overrides[key]
    for key in ("nodes", "gpus_per_node", "cpus_per_node"):
        if type(resources[key]) is not int or resources[key] < 1:
            raise ValueError(f"{key} must be positive")
    if resources.get("cpus_per_task") is not None and resources["cpus_per_task"] < 1:
        raise ValueError("cpus-per-task must be positive")
    if resources["distribution"].split(":", 1)[0] != "block":
        raise ValueError("node-level Slurm rank distribution must be block")
    if resources["memory_gb"] < 0 or not re.fullmatch(r"\d{2,3}:[0-5]\d:[0-5]\d", resources["time"]):
        raise ValueError("invalid memory or HH:MM:SS time")
    if overrides.get("cpu_bind") is None and (overrides.get("cpus_per_task") or
            resources["gpus_per_node"] != settings["slurm"]["gpus_per_node"]):
        resources["cpu_bind"] = "cores"
    job_id = overrides.get("job_id")
    if bool(job_id) == bool(overrides.get("submit")):
        raise ValueError("select exactly one of job-id or submit")
    if job_id is not None and not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("job-id must be numeric")
    mode = overrides.get("cache_mode") or "auto"
    if mode not in {"auto", "cold", "warm"}:
        raise ValueError("cache mode must be auto, cold, or warm")
    env = {**settings.get("environment", {}), **value.get("environment", {})}
    if any(k in RESERVED_ENV or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or not isinstance(v, str)
           for k, v in env.items()):
        raise ValueError("invalid or runtime-owned environment setting")
    scratch = Path(settings["scratch_root"])
    if ".." in scratch.parts or not scratch.is_relative_to("/dev/shm") or scratch == Path("/dev/shm"):
        raise ValueError("scratch_root must be a dedicated /dev/shm subdirectory")
    root = Path(overrides["data_root"]) if overrides.get("data_root") else site.parent / settings["data_root"]
    image = value["image"]
    image_path = Path(settings["image_stores"][image["store"]]) / "objects/sha256" / image["sha256"] / "image.sqsh"
    return {"experiment": name, **value, "validation": validation, "environment": env, "resources": resources,
            "image_path": str(image_path.resolve()), "data_root": str(root.resolve()),
            "scratch_root": str(scratch), "cache": {**settings["cache"], "mode": mode},
            "launch": {"job_id": job_id, "submit": bool(overrides.get("submit"))},
            "compare_run": str(Path(overrides["compare_run"]).resolve()) if overrides.get("compare_run") else None}
