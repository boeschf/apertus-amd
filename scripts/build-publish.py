#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Build and publish an image using the adjacent build.py and publish.py tools."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from build import argument_parser, build
from publish import DEFAULT_STORE, publish


def main() -> None:
    """Build in temporary storage, publish, then clean up; retain failed work."""
    parser = argument_parser(__doc__, include_output=False)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE,
                        help=f"shared image store (default: {DEFAULT_STORE})")
    args = parser.parse_args()
    workspace = Path(tempfile.mkdtemp(prefix="image-build-publish-"))
    output = workspace / "build"
    print(f"Build workspace: {workspace}", file=sys.stderr, flush=True)
    try:
        artifacts = build(args.source, output, args.selector,
                          processors=args.processors, isolation=args.isolation)
        result = publish(Path(artifacts["image"]), Path(artifacts["source"]),
                         Path(artifacts["provenance"]), args.store)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError,
            tarfile.TarError, argparse.ArgumentError) as error:
        parser.exit(1, f"Build/publication failed: {error}\nRetained workspace: {workspace}\n")
    # Remove only the workspace created by this invocation, after publication.
    try:
        shutil.rmtree(workspace)
    except OSError as error:
        parser.exit(1, f"Published successfully, but workspace cleanup failed: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
