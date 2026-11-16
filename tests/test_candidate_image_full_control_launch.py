"""CPU scheduler stubs exercise the real CLI/freeze boundary, never image/archive payloads or Slurm."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apertus_amd import cli, runs

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'artifacts/investigations/performance-next/restart-20261113/candidate-image-full-control-launch-preparation'
EXPERIMENT = ROOT / 'experiments/moe-chonk-kda-me242f3c3c113-i910bfcf936d3'


def load(name):
    spec = importlib.util.spec_from_file_location('candidate_full_cpu_' + name, PREP / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def transport(monkeypatch):
    module = load('transport')
    monkeypatch.setattr(module.sys, 'executable', module.PYTHON)
    return module


def test_inert_and_placement_owner(transport, monkeypatch):
    monkeypatch.delenv('APERTUS_CANDIDATE_FULL_CONTROL_APPROVAL', raising=False)
    with pytest.raises(RuntimeError, match='PREPARATION ONLY'):
        transport.main('sbatch')
    value = transport.forwarded('sbatch', ['--parsable', '/future/job.sh'])
    assert value[:2] == ['/usr/bin/sbatch', '--uenv-passthrough=ignore']
    assert not any(arg.startswith('--nodelist') for arg in value)
    assert all(node in value[2] for node in ('nid002456', 'nid002458', 'nid002744'))
    assert 'nid002412' not in value[2] and 'nid002454' not in value[2]
    assert {'741678', '741671', '741670', '741662', '741653', '741609', '738614', '737047'} == transport.PROTECTED_JOBS
    for arg in ('--nodelist=nid1', '--dependency=afterok:741678', '--exclude=nid1', '--uenv-passthrough=all'):
        with pytest.raises(ValueError):
            transport.forwarded('sbatch', [arg])


@pytest.fixture
def approved_fixture(transport, tmp_path, monkeypatch):
    # Synthetic CPU proof only. No real approval is created or used.
    monkeypatch.setattr(transport, 'HERE', tmp_path)
    (tmp_path / 'manifest.json').write_text('{}\n')
    proof = tmp_path / 'synthetic-A2-acceptance.json'
    proof.write_text('{"job_id":"741678","fixture_only":true}\n')
    evidence = tmp_path / 'execution-fixture'
    evidence.mkdir()
    approval = {'execution_authorized': True, 'maximum_submissions': 1, 'experiment': transport.EXPERIMENT,
        'preparation_manifest_sha256': transport.digest(tmp_path / 'manifest.json'), 'execution_artifacts': str(evidence),
        'fresh_disjoint_placement_reviewed': True, 'compatible_component_dependencies_reviewed': True,
        'a2_predecessor': {'job_id': '741678', 'terminal_passed_reviewed': True, 'ABBA_complete_reviewed': True,
                           'acceptance_path': str(proof), 'acceptance_sha256': transport.digest(proof)}}
    path = tmp_path / 'synthetic-approval.json'
    path.write_text(json.dumps(approval))
    monkeypatch.setattr(transport.os, 'environ', {'APERTUS_CANDIDATE_FULL_CONTROL_APPROVAL': str(path)})
    calls = []
    monkeypatch.setattr(transport.os, 'execv', lambda program, args: calls.append((program, args)))
    return approval, path, evidence, calls


@pytest.mark.parametrize('field', ['terminal_passed_reviewed', 'ABBA_complete_reviewed', 'acceptance_sha256', 'job_id'])
def test_actual_A2_proof_gates(transport, approved_fixture, field):
    value, path, _, calls = approved_fixture
    value['a2_predecessor'][field] = False if field.endswith('reviewed') else 'wrong'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        transport.approval_inputs()
    assert not calls


@pytest.mark.parametrize('field', ['fresh_disjoint_placement_reviewed', 'compatible_component_dependencies_reviewed'])
def test_other_parent_reviews_required(transport, approved_fixture, field):
    value, path, _, _ = approved_fixture
    del value[field]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        transport.approval_inputs()


def test_real_cli_freeze_and_receipt_race_boundary(transport, approved_fixture, tmp_path, monkeypatch):
    _, _, evidence, calls = approved_fixture
    expected = json.loads((PREP / 'resolved-input.json').read_text())
    expected['data_root'] = str(tmp_path / 'data')
    (tmp_path / 'resolved-input.json').write_text(json.dumps(expected))
    (tmp_path / 'tooling-hashes.json').write_bytes((PREP / 'tooling-hashes.json').read_bytes())
    monkeypatch.setattr(runs, 'image_manifest', lambda *args: {})
    monkeypatch.setattr(runs.sources, 'prepare', lambda *args, **kwargs: {'directory': str(tmp_path / 'source-fixture'),
        'sha256': 'e00b32a84e13e5de1b52d5f96e4882002212d0fe7756ab41062b85a275123498', 'tree': 'CPU fixture',
        'files': {'pretrain_gpt.py': 'CPU fixture'}})
    submitted = []
    def submit(command, **kwargs):
        assert command[:2] == ['sbatch', '--parsable']
        directory = Path(command[2]).parent
        assert not (directory / 'submission.json').exists()
        with monkeypatch.context() as scoped:
            scoped.setattr(transport.sys, 'argv', command)
            transport.main('sbatch')
        submitted.append(directory)
        return SimpleNamespace(stdout='999999\n')
    monkeypatch.setattr(runs.subprocess, 'run', submit)
    monkeypatch.setattr(transport.sys, 'argv', ['python3', 'run', str(EXPERIMENT), '--site', str(ROOT / 'config.toml'),
        '--data-root', str(tmp_path / 'data'), '--checkout', str(tmp_path / 'source-fixture'),
        '--submit', '--reservation', 'cscs', '--time', '01:15:00', '--cache-mode', 'auto'])
    cli.main()
    directory = submitted[0]
    assert len(calls) == 1 and (directory / 'submission.json').exists()
    assert transport.validate_run(directory)['resources']['nodes'] == 16
    assert f'export PYTHONPATH={directory}/tooling' in (directory / 'job.sh').read_text()
    assert f'exec {transport.PYTHON} -m apertus_amd.runs' in (directory / 'job.sh').read_text()
    # The owned batch may begin before the host receipt exists; intent was durable first.
    (directory / 'submission.json').unlink()
    transport.os.environ.update(SLURM_JOB_ID='999999')
    step = runs.step_command(json.loads((directory / 'resolved.json').read_text()), directory, cpus=48, job_id='999999')
    monkeypatch.setattr(transport.sys, 'argv', step)
    transport.main('srun')
    assert calls[-1][0] == '/usr/bin/srun'
    transport.os.environ['SLURM_JOB_ID'] = '741678'
    monkeypatch.setattr(transport.sys, 'argv', [arg.replace('999999', '741678') for arg in step])
    with pytest.raises(ValueError, match='fresh owned'):
        transport.main('srun')
    transport.os.environ.pop('SLURM_JOB_ID')
    monkeypatch.setattr(transport.sys, 'argv', ['sbatch', '--parsable', str(directory / 'job.sh')])
    with pytest.raises(FileExistsError):
        transport.main('sbatch')
    # Moving image into [annotations] retains the string but violates actual TOML ordering/semantics.
    edf = directory / 'container.toml'
    text = edf.read_text(); first, rest = text.split('\n', 1)
    edf.write_text(rest + first + '\n')
    with pytest.raises(ValueError, match='EDF semantics'):
        transport.validate_run(directory)


def test_canonical_entrypoint_and_clean_environment():
    launch = load('launch')
    assert launch.COMMAND[:7] == ['uv', 'run', '--locked', 'python3', '-m', 'apertus_amd.cli', 'run']
    assert launch.COMMAND[-2:] == ['--cache-mode', 'auto']
    value = launch.command_environment({'SLURM_JOB_ID': '741678', 'SBATCH_NODELIST': 'bad', 'SRUN_CPUS_PER_TASK': '1', 'PATH': '/usr/bin'})
    assert not any(key.startswith(('SLURM_', 'SBATCH_', 'SRUN_')) for key in value)
