#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Build publishable image artifacts through alps-extended-images' manual helper."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


MANUAL_BUILD = Path("manual-build/manual-build.sh")


def resolve_source(source: Path) -> Path:
    """Resolve a checkout or source snapshot containing the manual-build helper."""
    source = source.resolve()
    if not (source / MANUAL_BUILD).is_file():
        raise ValueError(f"source does not contain {MANUAL_BUILD}")
    return source


def source_environment(source: Path) -> dict[str, str]:
    """Prevent a Git-less source context from inheriting a parent checkout."""
    return dict(os.environ, GIT_CEILING_DIRECTORIES=str(source.parent))


def list_candidates(source: Path) -> None:
    """Delegate local selector discovery to the image repository without snapshotting."""
    source = resolve_source(source)
    # Discovery's YAML dependency belongs to the image helper; keep it in a
    # uv-managed subprocess rather than adding dependencies to the build tool.
    subprocess.run(["uv", "run", "--no-project", "--python", sys.executable,
                    "--with", "pyyaml", "bash", str(MANUAL_BUILD), "list"],
                   cwd=source, env=source_environment(source), check=True)


def write_json(path: Path, value: object) -> None:
    """Atomically replace a build-owned record, including the completion marker."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot(source: Path, archive: Path) -> dict:
    """Archive working files, including ignored files, but never Git metadata."""
    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    metadata = {"repository": git("remote", "get-url", "origin"),
                "head": git("rev-parse", "HEAD"),
                "short_head": git("rev-parse", "--short", "HEAD")}

    def keep(entry):
        if ".git" in Path(entry.name).parts:
            return None
        if not (entry.isfile() or entry.isdir() or entry.issym() or entry.islnk()):
            raise ValueError(f"unsupported source file: {entry.name}")
        return entry

    with tarfile.open(archive, "x:gz") as stream:
        for path in sorted(source.iterdir()):
            stream.add(path, arcname=path.name, filter=keep)
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {**metadata, "archive_sha256": checksum}


def build(source: Path, output: Path, selector: list[str], *, processors: int = 4,
          isolation: str = "oci") -> dict:
    """Snapshot, build, and export into a new directory; retain evidence on failure.

    Podman storage and the build allocation belong to the caller.
    Only successful exports with a full filesystem read receive provenance.json,
    the final input required by the standalone publisher.
    """
    if not ((len(selector) == 4 and selector[0] == "base") or
            (len(selector) == 3 and selector[0] == "app")):
        raise ValueError("selector must be: base FAMILY NAME VARIANT, or app NAME VARIANT")
    if processors < 1 or isolation not in {"oci", "chroot"}:
        raise ValueError("processors must be positive; isolation must be oci or chroot")
    source, output = resolve_source(source), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("build output must be outside the source checkout")
    # manual-build emits a double-quoted cd command without shell-escaping its root.
    if any(char in str(output) for char in '\\"$`\n\r'):
        raise ValueError("output path contains characters unsupported by manual-build")
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    context = output / "context"
    context.mkdir()
    archive = output / "source.tar.gz"
    metadata = snapshot(source, archive)
    write_json(evidence / "source.json", metadata)
    with tarfile.open(archive) as stream:
        stream.extractall(context, filter="data")

    env = source_environment(context)
    env.update(CI_COMMIT_SHA=metadata["head"], CI_COMMIT_SHORT_SHA=metadata["short_head"],
               CSCS_CI_ORIG_CLONE_URL=env.get("CSCS_CI_ORIG_CLONE_URL", metadata["repository"]),
               ENROOT_MAX_PROCESSORS=str(processors))
    env.setdefault("ENROOT_SQUASH_OPTIONS", "-comp zstd -Xcompression-level 3 -b 1048576 -mem 1024M")
    # Keep Enroot extraction workspace beside the build's retained evidence.
    temporary = output / "export-tmp"
    temporary.mkdir()
    env["ENROOT_TEMP_PATH"] = str(temporary)
    settings = {key: env[key] for key in (
        "CI_COMMIT_SHA", "CI_COMMIT_SHORT_SHA", "CSCS_CI_ORIG_CLONE_URL", "IMAGE_PREFIX",
        "ALPS_REV", "ENROOT_MAX_PROCESSORS", "ENROOT_SQUASH_OPTIONS", "ENROOT_LIBRARY_PATH",
    ) if key in env}
    write_json(evidence / "settings.json", settings)
    commands = []

    def run(name, command, *, stdout=None):
        print(f"{name}: {evidence / (name + '.log')}", file=sys.stderr, flush=True)
        step = {"name": name, "argv": command, "cwd": str(context)}
        commands.append(step)
        write_json(evidence / "commands.json", commands)
        with (evidence / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=context, env=env,
                                    stdout=log if stdout is None else stdout, stderr=log)
        step["returncode"] = result.returncode
        write_json(evidence / "commands.json", commands)
        if result.returncode:
            raise ValueError(f"{name} failed ({result.returncode}); see {evidence / (name + '.log')}")

    script = evidence / "build.sh"
    with script.open("w") as stream:
        run("generate", ["bash", str(MANUAL_BUILD), *selector], stdout=stream)

    # Let Bash expand manual-build's script (including OCI_CREATED) into exact argv.
    # Recording a single Podman invocation avoids a second shell parser or a copy
    # of the image repository's recipe/tag/build-argument logic.
    captured = evidence / "build-argv"
    env["APERTUS_BUILD_ARGV"] = str(captured)
    run("render", ["bash", "-c", '''
podman() {
    if [ -e "$APERTUS_BUILD_ARGV" ]; then
        echo 'Expected exactly one podman build invocation' >&2
        return 1
    fi
    printf '%s\\0' "$@" > "$APERTUS_BUILD_ARGV"
}
export -f podman
bash "$1"
''', "render-build", str(script)])
    argv = captured.read_bytes().decode().removesuffix("\0").split("\0")
    if argv[0] != "build" or argv[-1] != ".":
        raise ValueError("manual-build must emit a podman build of the extracted context")
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("-f", required=True)
    parser.add_argument("-t", action="append", required=True)
    parser.add_argument("--build-arg", action="append", default=[])
    flags, _ = parser.parse_known_args(argv[1:])
    build_args = dict(value.split("=", 1) for value in flags.build_arg)
    canonical = flags.t[0]
    # An iidfile pins the image actually built even if another process retags it.
    iid = evidence / "image-id"
    run("build", ["podman", *argv[:-1], f"--isolation={isolation}", f"--iidfile={iid}", "."])
    image_id = iid.read_text().strip()
    inspect_path = evidence / "inspect.json"
    with inspect_path.open("w") as stream:
        run("inspect", ["podman", "image", "inspect", image_id], stdout=stream)
    inspected = json.loads(inspect_path.read_text())[0]
    if inspected["Id"].removeprefix("sha256:") != image_id.removeprefix("sha256:"):
        raise ValueError("inspected image differs from the build's iidfile")
    image = output / "image.sqsh"
    run("export", ["enroot", "import", "-x", "tar", "-o", str(image), f"podman://{image_id}"])
    # -pf emits the entire filesystem data stream; discard data but retain errors.
    run("full-read", ["unsquashfs", "-no-progress", "-processors", str(processors),
                      "-data-queue", "32", "-frag-queue", "32", "-pf", "-", str(image)],
        stdout=subprocess.DEVNULL)
    record = {
        "source": metadata, "recipe": flags.f, "build_args": build_args,
        "canonical_ref": canonical, "id": canonical.rsplit("/", 1)[-1],
        "selector": selector, "image_config_id": inspected["Id"],
        "image_manifest_digest": inspected.get("Digest"),
        "image_labels": inspected.get("Labels") or inspected.get("Config", {}).get("Labels"),
        "build_options": {"isolation": isolation},
        "export_options": {"extract_mode": "tar"}, "settings": settings,
        "validation": {"squashfs_full_read": "passed", "runtime": "not tested"},
    }
    # This record is the completion marker. Failures above leave diagnostic inputs
    # and logs, but never a bundle that publish.py can accept.
    write_json(output / "provenance.json", record)
    return {"id": record["id"], "output": str(output),
            "image": str(image), "source": str(archive),
            "provenance": str(output / "provenance.json")}


def argument_parser(description: str = __doc__, *, include_output: bool = True) -> argparse.ArgumentParser:
    """Define build arguments shared with the build-and-publish convenience tool."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", type=Path, required=True)
    if include_output:
        parser.add_argument("--output", type=Path, help="build output directory (required unless --list)")
        parser.add_argument("--list", action="store_true", help="list repository build selectors without building")
    parser.add_argument("--processors", type=int, default=4, help="SquashFS export/read workers (default: 4)")
    parser.add_argument("--isolation", choices=("oci", "chroot"), default="oci")
    parser.add_argument("selector", nargs="*" if include_output else "+",
                        help="base FAMILY NAME VARIANT, or app NAME VARIANT")
    return parser


def main() -> None:
    """List build candidates or build and print the resulting artifact paths."""
    parser = argument_parser()
    args = parser.parse_args()
    if args.list:
        if args.output is not None or args.selector:
            parser.error("--list cannot be combined with --output or a build selector")
    elif args.output is None or not args.selector:
        parser.error("building requires --output and a build selector")
    try:
        if args.list:
            list_candidates(args.source)
            return
        result = build(args.source, args.output, args.selector,
                       processors=args.processors, isolation=args.isolation)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError,
            tarfile.TarError, argparse.ArgumentError) as error:
        parser.exit(1, f"Build failed: {error}\n")


if __name__ == "__main__":
    main()
