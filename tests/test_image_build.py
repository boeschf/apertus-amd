"""Exercise build-to-publication composition and retained failures using tool stubs."""

import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("standalone_build", ROOT / "scripts/build.py")
builder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder_module)
build = builder_module.build


@pytest.mark.parametrize("export_status", [0, 23])
@pytest.mark.parametrize("cleanup_status", [0, 17])
def test_enroot_workaround_preserves_export_status(tmp_path, export_status, cleanup_status):
    library = tmp_path / "custom installation"
    library.mkdir()
    original = '''trap 'common::rmall "${tmpdir}" 2> /dev/null; docker rm -f -v "${tmpdir##*/}" > /dev/null 2>&1' EXIT\n'''
    (library / "docker.sh").write_text(original)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    env = {"ENROOT_LIBRARY_PATH": str(library)}
    record = builder_module.enroot_cleanup_workaround(env, evidence)
    assert record["applied"] and record["original_sha256"] != record["patched_sha256"]
    assert (library / "docker.sh").read_text() == original
    # Exercise the trap in Bash: both cleanup operations may fail, and a real
    # export failure must still propagate even when cleanup succeeds.
    script = '''set -e
tmpdir=/unused/workspace
engine=podman
common::rmall() { return "$CLEANUP_STATUS"; }
podman() { printf '%s\\n' "$*" > "$CALLS"; return "$CLEANUP_STATUS"; }
source "$HELPER"
exit "$EXPORT_STATUS"
'''
    result = subprocess.run(["bash", "-c", script], env={**os.environ,
        "HELPER": str(Path(env["ENROOT_LIBRARY_PATH"]) / "docker.sh"), "CALLS": str(tmp_path / "calls"),
        "CLEANUP_STATUS": str(cleanup_status), "EXPORT_STATUS": str(export_status)}, capture_output=True)
    assert result.returncode == export_status
    assert (tmp_path / "calls").read_text().strip() == "rm -f -v workspace"


def test_enroot_workaround_discovers_prefix_and_leaves_unknown_helpers(tmp_path):
    library = tmp_path / "custom-prefix/lib/enroot"
    library.mkdir(parents=True)
    helper = library / "docker.sh"
    helper.write_text("# Already fixed or a different Enroot implementation.\n")
    executable = tmp_path / "enroot"
    executable.write_text(f'#!/bin/bash\nconfig::export ENROOT_LIBRARY_PATH "{library}"\n')
    executable.chmod(0o755)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    env = {"PATH": str(tmp_path)}
    record = builder_module.enroot_cleanup_workaround(env, evidence)
    assert record["applied"] is False and "original_sha256" in record
    assert "ENROOT_LIBRARY_PATH" not in env
    assert not (evidence / "enroot-library").exists()
    assert helper.read_text() == "# Already fixed or a different Enroot implementation.\n"


@pytest.fixture
def builder(tmp_path, monkeypatch):
    temporary = tmp_path / "node-local"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    library = tmp_path / "installed-enroot"
    library.mkdir()
    (library / "docker.sh").write_text('''trap 'common::rmall "${tmpdir}" 2> /dev/null; docker rm -f -v "${tmpdir##*/}" > /dev/null 2>&1' EXIT\n''')
    monkeypatch.setenv("ENROOT_LIBRARY_PATH", str(library))
    source = tmp_path / "checkout"
    source.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init")
    git("remote", "add", "origin", "https://example.invalid/images.git")
    helper = source / "manual-build/manual-build.sh"
    helper.parent.mkdir()
    # Match the real helper's emitted shell, including runtime timestamp expansion,
    # quoted arguments, canonical/stable tags, and Git-less root discovery.
    helper.write_text('''#!/bin/bash
set -eu
root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cat <<EOF
#!/bin/bash
set -eu
cd "$root"
podman build --format docker \\
  -f Containerfile \\
  -t localhost/images/fixture:alps7-dev-abc \\
  -t localhost/images/fixture:alps7-dev \\
  --build-arg BASE_IMAGE="localhost/base:canonical" \\
  --build-arg OCI_REVISION="$CI_COMMIT_SHA" \\
  --build-arg OCI_CREATED="\\$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
  --build-arg OCI_DESCRIPTION="a value with spaces" .
printf 'Built canonical image\\n'
EOF
''')
    (source / "Containerfile").write_text("FROM scratch\n")
    (source / ".gitignore").write_text("ignored-input\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    head = git("rev-parse", "HEAD")
    (source / "Containerfile").write_text("FROM scratch\n# dirty input\n")
    (source / "ignored-input").write_text("required ignored bytes")
    (source / "untracked-input").write_text("untracked bytes")
    tools = tmp_path / "tools"
    tools.mkdir()
    program = '''import json, os, pathlib, sys
tool = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
stage = 'inspect' if tool == 'podman' and args[0] == 'image' else {'podman': 'build', 'enroot': 'export', 'unsquashfs': 'full-read'}[tool]
if os.environ.get('FAIL_STAGE') == stage:
    if stage == 'export':
        pathlib.Path(args[args.index('-o') + 1]).write_bytes(b'partial')
    sys.exit(23)
if stage == 'build':
    assert pathlib.Path('Containerfile').read_text().endswith('# dirty input\\n')
    assert pathlib.Path('ignored-input').read_text() == 'required ignored bytes'
    assert not pathlib.Path('.git').exists()
    for arg in args:
        if arg.startswith('--iidfile='):
            pathlib.Path(arg.split('=', 1)[1]).write_text('sha256:' + 'a' * 64)
elif stage == 'inspect':
    assert args[-1] == 'sha256:' + 'a' * 64
    print(json.dumps([{'Id': 'a' * 64, 'Digest': 'sha256:' + 'b' * 64}]))
elif stage == 'export':
    assert args[-1] == 'podman://sha256:' + 'a' * 64
    workspace = pathlib.Path(os.environ['ENROOT_TEMP_PATH'])
    assert workspace.parent == pathlib.Path(os.environ['TMPDIR'])
    assert workspace.is_dir()
    assert pathlib.Path(os.environ['ENROOT_LIBRARY_PATH']).name == 'enroot-library'
    assert 'exit "${status}"' in pathlib.Path(os.environ['ENROOT_LIBRARY_PATH'], 'docker.sh').read_text()
    pathlib.Path(args[args.index('-o') + 1]).write_bytes(b'hsqs fixture')
else:
    assert pathlib.Path(args[-1]).read_bytes() == b'hsqs fixture'
'''
    for tool in ("podman", "enroot", "unsquashfs"):
        path = tools / tool
        path.write_text(f"#!{sys.executable}\n{program}")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    # A nested Git checkout around the output exercises the Git ceiling: the
    # helper must build from the extracted context, not discover this parent.
    parent = tmp_path / "parent-repo"
    parent.mkdir()
    subprocess.run(["git", "init", str(parent)], check=True, capture_output=True)
    return source, parent / "build output", head


def test_build_composes_with_unmodified_copied_publisher(builder, tmp_path):
    source, output, head = builder
    script = tmp_path / "build.py"
    shutil.copyfile(ROOT / "scripts/build.py", script)
    # -S disables site-packages, including the project's editable installation.
    built = subprocess.run([sys.executable, "-S", str(script),
                            "--source", str(source), "--output", str(output),
                            "app", "apertus-2", "megatron-primus268-rocm"],
                           check=True, capture_output=True, text=True, cwd=tmp_path)
    result = json.loads(built.stdout)
    record = json.loads(Path(result["provenance"]).read_text())
    assert record["id"] == "fixture:alps7-dev-abc"
    assert record["source"]["head"] == head
    assert record["build_args"]["OCI_REVISION"] == head
    assert record["build_args"]["OCI_DESCRIPTION"] == "a value with spaces"
    assert "$(date" not in record["build_args"]["OCI_CREATED"]
    assert "build_script" not in record and "commands" not in record
    assert str(output) not in json.dumps(record)
    assert record["build_options"] == {"isolation": "oci"}
    assert record["export_options"]["extract_mode"] == "tar"
    assert record["export_options"]["enroot_cleanup"]["applied"] is True
    commands = json.loads((output / "evidence/commands.json").read_text())
    assert commands[2]["cwd"] == str(output / "context")
    assert (output / "evidence/build.sh").is_file()
    workspace = Path(json.loads((output / "evidence/export-workspace.json").read_text())["path"])
    assert not workspace.exists()
    with tarfile.open(result["source"]) as archive:
        assert not any(".git" in Path(entry.name).parts for entry in archive)
        assert archive.extractfile("Containerfile").read().endswith(b"# dirty input\n")
        assert archive.extractfile("ignored-input").read() == b"required ignored bytes"
        assert archive.extractfile("untracked-input").read() == b"untracked bytes"
    assert (source / "Containerfile").read_text().endswith("# dirty input\n")
    publisher = tmp_path / "publish.py"
    shutil.copyfile(ROOT / "scripts/publish.py", publisher)
    published = subprocess.run([sys.executable, "-S", str(publisher), result["image"], result["source"],
                                result["provenance"], "--store", str(tmp_path / "store")],
                               check=True, capture_output=True, text=True, cwd=tmp_path)
    manifest = Path(published.stdout.strip()).parent / "manifest.json"
    assert json.loads(manifest.read_text())["provenance"] == record
    with pytest.raises(FileExistsError):
        build(source, output, ["base", "rocm", "primus", "26.8"])
    assert json.loads(Path(result["provenance"]).read_text()) == record


@pytest.mark.parametrize("stage", ["generate", "build", "inspect", "export", "full-read"])
def test_failed_build_retains_evidence_without_publishable_provenance(builder, monkeypatch, stage):
    source, output, _ = builder
    if stage == "generate":
        (source / "manual-build/manual-build.sh").write_text("exit 23\n")
    monkeypatch.setenv("FAIL_STAGE", stage)
    with pytest.raises(ValueError, match=f"{stage} failed"):
        build(source, output, ["base", "rocm", "primus", "26.8"])
    assert (output / "source.tar.gz").is_file()
    assert (output / "evidence/build.sh").is_file()
    assert not (output / "provenance.json").exists()
    steps = json.loads((output / "evidence/commands.json").read_text())
    assert steps[-1]["name"] == stage and steps[-1]["returncode"] == 23
    assert (output / f"evidence/{stage}.log").is_file()
    if stage in {"export", "full-read"}:
        workspace = Path(json.loads((output / "evidence/export-workspace.json").read_text())["path"])
        assert workspace.is_dir() and workspace.parent == Path(os.environ["TMPDIR"])


@pytest.fixture
def combined_tools(tmp_path, monkeypatch):
    scripts = tmp_path / "copied tools"
    scripts.mkdir()
    for name in ("build.py", "publish.py", "build-publish.py"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    return scripts, temporary


def combined_command(scripts, source, store):
    return [sys.executable, "-S", str(scripts / "build-publish.py"),
            "--source", str(source), "--store", str(store),
            "base", "rocm", "primus", "26.8"]


def test_combined_copied_tools_publish_and_clean_workspace(builder, combined_tools, tmp_path):
    source, _, _ = builder
    scripts, temporary = combined_tools
    result = subprocess.run(combined_command(scripts, source, tmp_path / "store"),
                            check=True, capture_output=True, text=True, cwd=tmp_path)
    image = Path(result.stdout.strip())
    assert result.stdout == f"{image}\n" and image.is_absolute()
    assert image.read_bytes() == b"hsqs fixture"
    assert (image.parent / "source.tar.gz").is_file()
    assert json.loads((image.parent / "manifest.json").read_text())["provenance"]["id"] == "fixture:alps7-dev-abc"
    assert not list(temporary.iterdir())


def test_combined_build_failure_never_publishes(builder, combined_tools, tmp_path, monkeypatch):
    source, _, _ = builder
    scripts, temporary = combined_tools
    monkeypatch.setenv("FAIL_STAGE", "full-read")
    store = tmp_path / "store"
    result = subprocess.run(combined_command(scripts, source, store),
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0 and not result.stdout
    assert not store.exists()
    workspace, = temporary.glob("image-build-publish-*")
    export_workspace, = temporary.glob("apertus-image-export-*")
    assert str(export_workspace) in result.stderr
    assert str(workspace) in result.stderr
    assert (workspace / "build/evidence/full-read.log").is_file()
    assert not (workspace / "build/provenance.json").exists()


def test_combined_publish_failure_retains_artifacts_for_direct_retry(builder, combined_tools, tmp_path):
    source, _, _ = builder
    scripts, temporary = combined_tools
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("preserve")
    result = subprocess.run(combined_command(scripts, source, blocked),
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0 and not result.stdout
    workspace, = temporary.iterdir()
    assert str(workspace) in result.stderr
    artifacts = [workspace / "build" / name for name in ("image.sqsh", "source.tar.gz", "provenance.json")]
    before = [path.read_bytes() for path in artifacts]
    retry = subprocess.run([sys.executable, "-S", str(scripts / "publish.py"),
                            *map(str, artifacts), "--store", str(tmp_path / "store")],
                           check=True, capture_output=True, text=True, cwd=tmp_path)
    assert Path(retry.stdout.strip()).read_bytes() == before[0]
    assert [path.read_bytes() for path in artifacts] == before
    assert blocked.read_text() == "preserve"


def test_source_containment_and_escaping_symlinks(builder, tmp_path):
    source, output, _ = builder
    with pytest.raises(ValueError, match="outside"):
        build(source, source / "output", ["base", "rocm", "primus", "26.8"])
    assert not (source / "output").exists()
    (source / "external-input").symlink_to(tmp_path / "outside")
    with pytest.raises(tarfile.FilterError):
        build(source, output, ["base", "rocm", "primus", "26.8"])
    assert not (output / "provenance.json").exists()


@pytest.mark.parametrize("status", [0, 23])
def test_copied_build_lists_gitless_source_without_build_side_effects(tmp_path, monkeypatch, status):
    parent = tmp_path / "parent"
    subprocess.run(["git", "init", str(parent)], check=True, capture_output=True)
    source = parent / "source snapshot"
    helper = source / "manual-build/manual-build.sh"
    helper.parent.mkdir(parents=True)
    helper.write_text('''#!/bin/bash
set -eu
[[ "$#" == 1 && "$1" == list ]]
[[ "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" == "$PWD" ]]
if [[ "$LIST_STATUS" != 0 ]]; then
    echo 'invalid repository metadata' >&2
    exit "$LIST_STATUS"
fi
cat candidates.txt
''')
    candidates = "base rocm example 1.0\napp example rocm\n"
    (source / "candidates.txt").write_text(candidates)
    tools = tmp_path / "tools"
    tools.mkdir()
    # Isolate uv's dependency setup while exercising the real CLI-to-helper handoff.
    uv = tools / "uv"
    uv.write_text('''#!/bin/sh
while [ "$#" -gt 0 ] && [ "$1" != bash ]; do shift; done
test "$#" -gt 0 || exit 99
exec "$@"
''')
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("LIST_STATUS", str(status))
    script = tmp_path / "build.py"
    shutil.copyfile(ROOT / "scripts/build.py", script)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = subprocess.run([sys.executable, "-S", str(script), "--source", str(source), "--list"],
                            capture_output=True, text=True, cwd=tmp_path)
    assert (result.returncode == 0) == (status == 0)
    assert result.stdout == (candidates if status == 0 else "")
    if status:
        assert "invalid repository metadata" in result.stderr
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("arguments", [[], ["--list", "--output", "unused"],
                                     ["--list", "app", "example", "rocm"]])
def test_build_rejects_incomplete_or_conflicting_modes(tmp_path, arguments):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/build.py"),
                            "--source", str(tmp_path), *arguments],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 2
    assert not list(tmp_path.iterdir())
