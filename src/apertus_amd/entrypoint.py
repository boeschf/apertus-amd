"""Execute the selected Megatron entry point and verify its source imports."""

import json
import os
from pathlib import Path
import runpy
import sys

from apertus_amd.artifacts import sha256, write_json


def verify_imports(source: Path, expected: dict) -> int:
    """Assert every loaded Megatron file comes from the selected materialization."""
    modules = {}
    for name, module in list(sys.modules.items()):
        if not (name == "megatron" or name.startswith("megatron.") or name in {"gpt_builders", "model_provider"}):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        relative = str(path.relative_to(source))  # Reject image-bundled modules.
        checksum = sha256(path)
        if relative in expected:
            if checksum != expected[relative]:
                raise ValueError(f"source file changed: {relative}")
        elif not path.name.endswith(".so"):
            raise ValueError(f"untracked runtime module: {relative}")
        modules[name] = {"path": str(path), "sha256": checksum}
    return len(modules)


def train(output: Path, source: Path, entrypoint: str, argv: list[str], *,
          manifest: Path = Path("/apertus/source/manifest.json"), before_run=None) -> None:
    """Run source with optional experiment setup; verify imports even on failure."""
    source = source.resolve()
    expected = json.loads(manifest.read_text())["files"]
    if sha256(source / entrypoint) != expected[entrypoint]:
        raise ValueError("entry point checksum mismatch")
    sys.path.insert(0, str(source))
    sys.argv = [str(source / entrypoint), *argv]
    try:
        if before_run is not None:
            before_run()
        verify_imports(source, expected)
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        count = verify_imports(source, expected)
        rank = int(os.environ["RANK"])
        write_json(output / f"imports-{rank}.json", {"verified_modules": count, "source": str(source)})


if __name__ == "__main__":
    train(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
