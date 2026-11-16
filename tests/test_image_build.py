"""Exercise build-to-publication composition and retained failures using tool stubs."""

import json
import hashlib
import importlib.util
import os
from pathlib import Path
import re
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
@pytest.mark.parametrize("engine_status", [0, 19])
def test_enroot_workaround_preserves_primary_failure_and_rejects_cleanup_failure(
        tmp_path, export_status, cleanup_status, engine_status):
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
podman() { printf '%s\\n' "$*" > "$CALLS"; return "$ENGINE_STATUS"; }
source "$HELPER"
exit "$EXPORT_STATUS"
'''
    result = subprocess.run(["bash", "-c", script], env={**os.environ,
        "HELPER": str(Path(env["ENROOT_LIBRARY_PATH"]) / "docker.sh"), "CALLS": str(tmp_path / "calls"),
        "CLEANUP_STATUS": str(cleanup_status), "ENGINE_STATUS": str(engine_status),
        "EXPORT_STATUS": str(export_status)}, capture_output=True)
    assert result.returncode == (export_status or cleanup_status or engine_status)
    if export_status:
        assert not (tmp_path / "calls").exists()
    else:
        assert (tmp_path / "calls").read_text().strip() == "rm -f -v workspace"


@pytest.mark.parametrize("export_status", [0, 23])
def test_deferred_enroot_trap_records_objects_without_cleanup(tmp_path, export_status):
    library, evidence = tmp_path / "library", tmp_path / "evidence"
    library.mkdir()
    evidence.mkdir()
    original = '''trap 'common::rmall "${tmpdir}" 2> /dev/null; docker rm -f -v "${tmpdir##*/}" > /dev/null 2>&1' EXIT\n'''
    (library / "docker.sh").write_text(original)
    env = {"ENROOT_LIBRARY_PATH": str(library)}
    builder_module.enroot_cleanup_workaround(env, evidence, defer_cleanup=True)
    script = '''set -e
tmpdir="/unused/path with spaces/workspace"
engine=podman
common::rmall() { printf 'unexpected cleanup' > "$CALLS"; return 17; }
podman() { printf 'unexpected engine' > "$CALLS"; return 19; }
source "$HELPER"
exit "$EXPORT_STATUS"
'''
    result = subprocess.run(["bash", "-c", script], env={**os.environ, **env,
        "HELPER": str(Path(env["ENROOT_LIBRARY_PATH"]) / "docker.sh"),
        "CALLS": str(tmp_path / "calls"), "EXPORT_STATUS": str(export_status)}, capture_output=True)
    assert result.returncode == export_status
    assert not (tmp_path / "calls").exists()
    fields = (evidence / "enroot-cleanup-request.nul").read_bytes().split(b"\0")
    assert fields == [str(export_status).encode(), b"/unused/path with spaces/workspace", b"podman", b"workspace", b""]
    assert (library / "docker.sh").read_text() == original


@pytest.mark.parametrize("export_status", [0, 23])
def test_deferred_trap_record_failure_never_becomes_success(tmp_path, export_status):
    library, evidence = tmp_path / "library", tmp_path / "evidence"
    library.mkdir()
    evidence.mkdir()
    (library / "docker.sh").write_text('''trap 'common::rmall "${tmpdir}" 2> /dev/null; docker rm -f -v "${tmpdir##*/}" > /dev/null 2>&1' EXIT\n''')
    env = {"ENROOT_LIBRARY_PATH": str(library)}
    builder_module.enroot_cleanup_workaround(env, evidence, defer_cleanup=True)
    # An existing directory makes the trap's record write fail deterministically.
    Path(env["APERTUS_ENROOT_CLEANUP_RECORD"]).mkdir()
    result = subprocess.run(["bash", "-c", 'tmpdir=/unused/workspace; engine=podman; source "$HELPER"; exit "$EXPORT_STATUS"'],
        env={**os.environ, **env, "HELPER": str(Path(env["ENROOT_LIBRARY_PATH"]) / "docker.sh"),
             "EXPORT_STATUS": str(export_status)}, capture_output=True)
    if export_status:
        assert result.returncode == export_status
    else:
        assert result.returncode != 0


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("export_status", [0, 23])
@pytest.mark.parametrize("cleanup_status", [0, 17])
@pytest.mark.parametrize("engine_status", [0, 19])
def test_audited_enroot_daemon_body_has_one_cleanup_owner(
        tmp_path, deferred, export_status, cleanup_status, engine_status):
    library, evidence, target = tmp_path / "library", tmp_path / "evidence", tmp_path / "enroot-fixture"
    for directory in (library, evidence, target):
        directory.mkdir()
    original = (ROOT / "tests/fixtures/enroot-daemon-import.sh").read_text()
    body = re.search(r'^docker::daemon::import\(\) \(\n.*?^\)$', original, re.MULTILINE | re.DOTALL).group()
    assert hashlib.sha256(body.encode()).hexdigest() == builder_module.ENROOT_DAEMON_IMPORT_SHA256
    (library / "docker.sh").write_text(original)
    env = {"ENROOT_LIBRARY_PATH": str(library)}
    record = builder_module.enroot_cleanup_workaround(env, evidence, defer_cleanup=deferred)
    script = '''set -euo pipefail
common::checkcmd() { :; }
common::realpath() { printf '%s\\n' "$1"; }
common::mktmpdir() { printf '%s\\n' "$TARGET"; }
common::chdir() { cd "$1"; }
common::log() { :; }
common::fixperms() { :; }
common::rmall() { printf 'workspace\\n' >> "$CLEANUPS"; return "$CLEANUP_STATUS"; }
docker::daemon::_config_squash() { :; }
tar() { cat > /dev/null; }
podman() {
    printf '%s\\n' "$*" >> "$CALLS"
    case "$1" in
      create) : > "$CONTAINER_STATE" ;;
      export) return "$EXPORT_STATUS" ;;
      rm) [ -f "$CONTAINER_STATE" ] || return 29
          [ "$ENGINE_STATUS" = 0 ] || return "$ENGINE_STATUS"
          command rm "$CONTAINER_STATE" ;;
      *) return 31 ;;
    esac
}
source "$HELPER"
docker::daemon::import podman://abc "$IMAGE_OUTPUT" '' tar
'''
    result = subprocess.run(["bash", "-c", script], env={**os.environ, **env,
        "HELPER": str(Path(env["ENROOT_LIBRARY_PATH"]) / "docker.sh"), "TARGET": str(target),
        "IMAGE_OUTPUT": str(tmp_path / "image.sqsh"), "CALLS": str(tmp_path / "calls"),
        "CLEANUPS": str(tmp_path / "cleanups"), "CONTAINER_STATE": str(tmp_path / "container-state"),
        "EXPORT_STATUS": str(export_status), "CLEANUP_STATUS": str(cleanup_status),
        "ENGINE_STATUS": str(engine_status)}, capture_output=True)
    assert result.returncode == (export_status if deferred else export_status or cleanup_status or engine_status), result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    if deferred or export_status:
        assert not any(line.startswith('rm ') for line in calls)
        assert not (tmp_path / "cleanups").exists()
        assert (tmp_path / "container-state").exists()
    else:
        assert calls.count('rm -f -v enroot-fixture') == 1
        assert (tmp_path / "cleanups").read_text() == 'workspace\n'
    if deferred:
        assert (evidence / "enroot-cleanup-request.nul").read_bytes().split(b'\0') == [
            str(export_status).encode(), os.fsencode(target), b'podman', b'enroot-fixture', b'']
    assert record['inline_container_cleanup'] == ('deferred' if deferred else 'moved-to-exit-trap')
    assert (library / "docker.sh").read_text() == original


@pytest.mark.parametrize("deferred", [False, True])
def test_audited_enroot_cleanup_keeps_external_engine_cwd_until_removal(tmp_path, deferred):
    library, evidence, target = tmp_path / "library", tmp_path / "evidence", tmp_path / "enroot-fixture"
    for directory in (library, evidence, target):
        directory.mkdir()
    original = (ROOT / "tests/fixtures/enroot-daemon-import.sh").read_text()
    (library / "docker.sh").write_text(original)
    env = {"ENROOT_LIBRARY_PATH": str(library)}
    builder_module.enroot_cleanup_workaround(env, evidence, defer_cleanup=deferred)
    engine = tmp_path / "podman"
    engine.write_text(f"#!{sys.executable}\n" + '''import os, sys
with open(os.environ["CALLS"], "a") as stream:
    try:
        cwd = os.getcwd()
    except FileNotFoundError:
        stream.write(sys.argv[1] + " CWD_MISSING\\n")
        sys.exit(41)
    stream.write(sys.argv[1] + " " + cwd + "\\n")
''')
    engine.chmod(0o755)
    # Execute the real generated daemon-import function and trap, not a hand-written
    # replacement. Only our disposable test directory is removed by common::rmall.
    script = '''set -euo pipefail
common::checkcmd() { :; }
common::realpath() { printf '%s\\n' "$1"; }
common::mktmpdir() { printf '%s\\n' "$TARGET"; }
common::chdir() { cd "$1"; }
common::log() { :; }
common::fixperms() { :; }
common::rmall() { command rm -rf -- "$1"; }
docker::daemon::_config_squash() { :; }
tar() { cat > /dev/null; }
source "$HELPER"
docker::daemon::import podman://abc "$IMAGE_OUTPUT" '' tar
'''
    result = subprocess.run(["bash", "-c", script], env={**os.environ, **env,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "HELPER": str(Path(env["ENROOT_LIBRARY_PATH"]) / "docker.sh"), "TARGET": str(target),
        "IMAGE_OUTPUT": str(tmp_path / "image.sqsh"), "CALLS": str(tmp_path / "calls")},
        capture_output=True, text=True)
    calls = (tmp_path / "calls").read_text().splitlines()
    assert result.returncode == 0, (result.stderr, calls)
    assert calls == [f"create {target}", f"export {target}"] + ([] if deferred else [f"rm {target}"])
    assert target.exists() is deferred
    if deferred:
        assert (evidence / "enroot-cleanup-request.nul").read_bytes().split(b"\0") == [
            b"0", os.fsencode(target), b"podman", b"enroot-fixture", b""]
    assert (library / "docker.sh").read_text() == original


@pytest.mark.parametrize("damage", ['create', 'remove', 'duplicate', 'extra-cleanup'])
@pytest.mark.parametrize("deferred", [False, True])
def test_unaudited_enroot_daemon_protocol_is_not_partially_patched(tmp_path, damage, deferred):
    library, evidence = tmp_path / "library", tmp_path / "evidence"
    library.mkdir()
    evidence.mkdir()
    text = (ROOT / "tests/fixtures/enroot-daemon-import.sh").read_text()
    if damage == 'create':
        text = text.replace('create --name', 'create --other')
    elif damage == 'remove':
        text = text.replace('"${engine}" rm', '"${engine}"  rm')
    elif damage == 'duplicate':
        text += text
    else:
        text = text.replace('    # Cleanup the container', '    rm -rf unreviewed\n    # Cleanup the container')
    (library / "docker.sh").write_text(text)
    with pytest.raises(ValueError, match='unaudited Enroot'):
        builder_module.enroot_cleanup_workaround({'ENROOT_LIBRARY_PATH': str(library)}, evidence, defer_cleanup=deferred)
    assert not (evidence / 'enroot-library').exists()


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
    if os.environ.get('APERTUS_ENROOT_CLEANUP_RECORD'):
        directory = workspace / 'enroot-fixture'
        directory.mkdir()
        pathlib.Path(os.environ['APERTUS_ENROOT_CLEANUP_RECORD']).write_bytes(
            b'0\\0' + os.fsencode(directory) + b'\\0podman\\0enroot-fixture\\0')
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


def test_explicit_deferred_export_keeps_workspace_and_records_caller_handoff(builder, monkeypatch):
    source, output, _ = builder
    original = builder_module.shutil.rmtree

    def refuse_export_cleanup(path, *args, **kwargs):
        assert not Path(path).name.startswith("apertus-image-export-"), "deferred workspace was traversed"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(builder_module.shutil, "rmtree", refuse_export_cleanup)
    built = build(source, output, ["app", "apertus-2", "megatron-primus268-rocm"], defer_export_cleanup=True)
    handoff = json.loads(Path(built["deferred_cleanup"]).read_text())
    assert Path(handoff["workspace"]).is_dir()
    assert Path(handoff["enroot_directory"]).is_dir()
    assert handoff["container_name"] == "enroot-fixture"
    assert handoff["engine"] == "podman"
    provenance = json.loads(Path(built["provenance"]).read_text())
    assert provenance["validation"]["squashfs_full_read"] == "passed"
    assert provenance["export_options"]["cleanup"] == "deferred"
    assert provenance["export_options"]["enroot_cleanup"]["cleanup"] == "deferred"


def test_copied_standalone_deferred_cli_needs_no_project_imports(builder, tmp_path):
    source, output, _ = builder
    script = tmp_path / "build.py"
    shutil.copyfile(ROOT / "scripts/build.py", script)
    result = subprocess.run([sys.executable, "-S", str(script), "--source", str(source),
        "--output", str(output), "--defer-export-cleanup", "app", "apertus-2", "megatron-primus268-rocm"],
        capture_output=True, text=True, check=True)
    handoff = json.loads(Path(json.loads(result.stdout)["deferred_cleanup"]).read_text())
    assert Path(handoff["workspace"]).is_dir()


@pytest.mark.parametrize("blocked", [None, "build", "export", "full-read", "cleanup-export"])
def test_external_phase_gate_prevents_unsafe_nested_operation_and_completion(builder, blocked):
    source, output, _ = builder
    gates = []

    def gate(name):
        gates.append(name)
        if name == blocked:
            raise ValueError("external health/capacity gate refused")

    if blocked:
        with pytest.raises(ValueError, match="external health/capacity"):
            build(source, output, ["app", "apertus-2", "megatron-primus268-rocm"], phase_gate=gate)
        assert not (output / "provenance.json").exists()
        commands = json.loads((output / "evidence/commands.json").read_text())
        assert blocked not in [row["name"] for row in commands]
        if blocked in {"export", "full-read", "cleanup-export"}:
            workspace = Path(json.loads((output / "evidence/export-workspace.json").read_text())["path"])
            assert workspace.is_dir()
    else:
        result = build(source, output, ["app", "apertus-2", "megatron-primus268-rocm"], phase_gate=gate)
        assert gates == ["generate", "render", "build", "inspect", "export", "full-read", "cleanup-export"]
        assert json.loads(Path(result["provenance"]).read_text())["build_options"]["external_phase_gate"] is True


@pytest.mark.parametrize("damage", ["missing", "duplicate", "failure", "escape", "dotdot", "engine", "container", "oversized"])
def test_deferred_request_rejects_ambiguous_or_unowned_objects(tmp_path, damage):
    evidence, workspace = tmp_path / "evidence", tmp_path / "workspace"
    evidence.mkdir()
    directory = str(workspace / "container")
    status, engine, container = "0", "podman", "container"
    if damage == "failure":
        status = "23"
    elif damage == "escape":
        directory = str(tmp_path / "elsewhere/container")
    elif damage == "dotdot":
        directory = str(workspace / "../container")
    elif damage == "engine":
        engine = "/unowned/podman"
    elif damage == "container":
        container = "other"
    raw = b"\0".join(value.encode() for value in (status, directory, engine, container, ""))
    if damage == "duplicate":
        raw *= 2
    elif damage == "oversized":
        raw = b"x" * 65537
    if damage != "missing":
        (evidence / "enroot-cleanup-request.nul").write_bytes(raw)
    with pytest.raises((ValueError, FileNotFoundError)):
        builder_module.deferred_cleanup_request(evidence, workspace, {"PATH": "/usr/bin:/bin"})
    assert not (evidence / "deferred-cleanup.json").exists()


def test_deferred_cleanup_requires_the_exact_known_enroot_trap(tmp_path):
    library, evidence = tmp_path / "library", tmp_path / "evidence"
    library.mkdir()
    evidence.mkdir()
    (library / "docker.sh").write_text("# different helper\n")
    with pytest.raises(ValueError, match="unknown Enroot trap"):
        builder_module.enroot_cleanup_workaround({"ENROOT_LIBRARY_PATH": str(library)}, evidence, defer_cleanup=True)
    assert not (evidence / "enroot-library").exists()


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


def test_combined_wrapper_does_not_bypass_deferred_cleanup_ownership(builder, combined_tools, tmp_path):
    source, _, _ = builder
    scripts, temporary = combined_tools
    command = combined_command(scripts, source, tmp_path / "store")
    command.insert(3, "--defer-export-cleanup")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert not list(temporary.iterdir())
    assert not (tmp_path / "store").exists()


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
