"""Exercise approved component transport with scheduler stubs, never allocations or image payloads."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apertus_amd import config, runs

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'artifacts/investigations/performance-next/restart-20261116/bounded-component-launch'


@pytest.fixture
def transport(monkeypatch):
    spec = importlib.util.spec_from_file_location('bounded_transport', HERE / 'transport.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, 'executable', module.PYTHON)
    return module


@pytest.fixture(params=['inventory', 'forward'])
def approved(transport, tmp_path, monkeypatch, request):
    component = request.param
    monkeypatch.setattr(transport, 'HERE', tmp_path / 'prep')
    monkeypatch.setattr(transport, 'ROOT', tmp_path)
    transport.HERE.mkdir()
    evidence = transport.HERE / 'execution' / component
    evidence.mkdir(parents=True)
    (transport.HERE / 'manifest.json').write_text('{}')
    value = config.resolve(ROOT / 'experiments' / transport.COMPONENTS[component], site=ROOT / 'config.toml',
                           submit=True, time='00:05:00', reservation='cscs')
    value['data_root'] = str(tmp_path / 'artifacts/experiment-data')
    (transport.HERE / f'{component}-resolved.json').write_text(json.dumps(value))
    source = ROOT / 'src/apertus_amd'
    tooling = {str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in source.glob('*.py')}
    (transport.HERE / 'tooling-hashes.json').write_text(json.dumps(tooling))
    approval = dict(component=component, experiment=transport.COMPONENTS[component], node=transport.NODES[component],
        execution_authorized=True, maximum_submissions=1, manifest_sha256=transport.digest(transport.HERE / 'manifest.json'),
        execution_artifacts=str(evidence))
    path = transport.HERE / 'approval.json'
    path.write_text(json.dumps(approval))
    monkeypatch.setattr(transport.os, 'environ', {'APERTUS_BOUNDED_COMPONENT_APPROVAL': str(path)})
    calls = []
    monkeypatch.setattr(transport.os, 'execv', lambda program, args: calls.append(args))
    monkeypatch.setattr(runs, 'image_manifest', lambda *args: {})
    submitted = []
    def submit(command, **kwargs):
        assert command[:2] == ['sbatch', '--parsable']
        directory = Path(command[2]).parent
        assert not (directory / 'submission.json').exists()
        transport.scheduler('sbatch', command[1:])
        submitted.append(directory)
        return SimpleNamespace(stdout='999999\n')
    monkeypatch.setattr(runs.subprocess, 'run', submit)
    return component, value, approval, path, evidence, calls, submitted


def test_no_approval_is_inert(transport, monkeypatch):
    monkeypatch.delenv('APERTUS_BOUNDED_COMPONENT_APPROVAL', raising=False)
    with pytest.raises(RuntimeError):
        transport.scheduler('sbatch', ['--parsable', '/future/job.sh'])


def test_real_freeze_and_exact_disjoint_step(transport, approved):
    component, value, _, _, evidence, calls, submitted = approved
    directory = runs.run(value)
    assert submitted == [directory] and len(calls) == 1
    assert calls[0][:3] == ['/usr/bin/sbatch', '--uenv-passthrough=ignore', f'--nodelist={transport.NODES[component]}']
    other = set(transport.NODES.values()) - {transport.NODES[component]}
    assert all(node in calls[0][3] for node in transport.EXCLUDED | other)
    # Test the race where the batch starts before the host writes its receipt.
    (directory / 'submission.json').unlink()
    transport.os.environ.update(SLURM_JOB_ID='999999', SLURM_JOB_NODELIST=transport.NODES[component], SLURM_JOB_USER='boeschf')
    actual = json.loads((directory / 'resolved.json').read_text())
    args = [*runs.step_command(actual, directory, cpus=48, job_id='999999')[1:], '0']
    transport.scheduler('srun', args)
    assert calls[-1][:2] == ['/usr/bin/srun', '--uenv-passthrough=ignore']
    with pytest.raises(FileExistsError):
        transport.scheduler('srun', args)
    assert (evidence / 'sbatch-intent.json').exists()


@pytest.mark.parametrize('damage', ['input', 'tooling', 'EDF', 'batch'])
def test_frozen_run_tampering_rejected(transport, approved, damage):
    component, value, _, _, _, _, _ = approved
    directory = runs.run(value)
    path = directory / {'input': 'resolved.json', 'tooling': 'tooling/apertus_amd/runtime.py',
                        'EDF': 'container.toml', 'batch': 'job.sh'}[damage]
    text = path.read_text()
    if damage == 'EDF':
        text = text.replace('com.hooks.cxi.enabled = "false"', 'com.hooks.cxi.enabled = "true"')
    else:
        text += '\n# tampered\n'
    path.write_text(text)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        transport.validate_run(directory, component)


@pytest.mark.parametrize('field,wrong', [('execution_authorized', False), ('maximum_submissions', 2),
    ('manifest_sha256', 'bad'), ('node', 'nid002744'), ('component', 'other')])
def test_approval_scope_refused(transport, approved, field, wrong):
    _, _, approval, path, _, _, _ = approved
    approval[field] = wrong
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError):
        transport.approval()


def test_dirty_scheduler_environment_refused(transport, approved):
    _, value, _, _, _, calls, _ = approved
    transport.os.environ['SBATCH_WAIT'] = '1'
    with pytest.raises(ValueError, match='cleaned'):
        runs.run(value)
    assert not calls


@pytest.mark.parametrize('damage', ['foreign-job', 'other-node', 'other-user', 'extra-arg'])
def test_foreign_or_modified_step_refused(transport, approved, damage):
    component, value, _, _, _, calls, _ = approved
    directory = runs.run(value)
    actual = json.loads((directory / 'resolved.json').read_text())
    job = '744590' if damage == 'foreign-job' else '999999'
    transport.os.environ.update(SLURM_JOB_ID=job,
        SLURM_JOB_NODELIST='nid002744' if damage == 'other-node' else transport.NODES[component],
        SLURM_JOB_USER='other' if damage == 'other-user' else 'boeschf')
    args = [*runs.step_command(actual, directory, cpus=48, job_id=job)[1:], '0']
    if damage == 'extra-arg':
        args.append('--uenv-passthrough=all')
    with pytest.raises(ValueError):
        transport.scheduler('srun', args)
    assert len(calls) == 1
