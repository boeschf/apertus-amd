"""CPU-only diagnostic-tail and scheduler guards; no actual scheduler or image invocation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apertus_amd import config, runs

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'artifacts/investigations/performance-next/restart-20261116/startup-diagnostic-launch'


@pytest.fixture
def transport(monkeypatch):
    spec = importlib.util.spec_from_file_location('startup_transport', HERE / 'transport.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, 'executable', module.PYTHON)
    return module


def fixture_value():
    # Real resolver fixture from the already-existing metadata experiment; no native code.
    return config.resolve(ROOT / 'experiments/chonk-kernel-consumer-inventory-i910bfcf936d3',
                          site=ROOT / 'config.toml', submit=True, reservation='cscs', time='00:05:00')


def test_inert_without_separate_approval(transport, monkeypatch):
    monkeypatch.delenv('APERTUS_STARTUP_DIAGNOSTIC_APPROVAL', raising=False)
    with pytest.raises(RuntimeError):
        transport.scheduler('sbatch', ['--parsable', '/future/job.sh'])


def test_diagnostic_tail_preserves_runtime_resources_entrypoint_and_stdin(transport):
    value, directory = fixture_value(), Path('/future')
    args = [*runs.step_command(value, directory, cpus=48, job_id='999999')[1:], '0']
    actual = transport.diagnostic_step(args, value, directory, '999999')
    split = args.index('env')
    assert actual[:split] == args[:split]
    assert actual[split:split+2] == ['/bin/sh', '-c']
    assert 'stage=container-command-entered' in actual[split+2]
    assert 'stage=bash-startup-complete' in actual[actual.index('bash')+2]
    assert actual[-7:] == ['python3', '-X', 'importtime', '-u', '/apertus/validation.py', 'bootstrap', '0']
    assert '--gpu-bind=none' in actual and '--ntasks=4' in actual
    assert not any('stdin' in argument or 'BASH_ENV' in argument or 'entrypoint' in argument for argument in actual)


@pytest.mark.parametrize('damage', ['job', 'ranks', 'cpu', 'command', 'extra'])
def test_modified_step_refused(transport, damage):
    value, directory = fixture_value(), Path('/future')
    args = [*runs.step_command(value, directory, cpus=48, job_id='999999')[1:], '0']
    if damage == 'extra':
        args.append('--uenv-passthrough=all')
    else:
        before, after = {'job': ('--jobid=999999', '--jobid=744590'), 'ranks': ('--ntasks=4', '--ntasks=1'),
            'cpu': ('--cpus-per-task=48', '--cpus-per-task=24'), 'command': ('apertus_amd.runtime', 'other')}[damage]
        args[args.index(before)] = after
    with pytest.raises(ValueError):
        transport.diagnostic_step(args, value, directory, '999999')


@pytest.fixture
def approved(transport, tmp_path, monkeypatch):
    monkeypatch.setattr(transport, 'HERE', tmp_path)
    evidence = tmp_path / 'execution'
    evidence.mkdir()
    (tmp_path / 'manifest.json').write_text('{}')
    value = dict(execution_authorized=True, maximum_submissions=1, experiment=transport.EXPERIMENT,
                 node=transport.NODE, manifest_sha256=transport.digest(tmp_path / 'manifest.json'), execution_artifacts=str(evidence))
    path = tmp_path / 'approval.json'
    path.write_text(json.dumps(value))
    monkeypatch.setattr(transport.os, 'environ', {'APERTUS_STARTUP_DIAGNOSTIC_APPROVAL': str(path)})
    return value, path, evidence


@pytest.mark.parametrize('key,bad', [('execution_authorized', False), ('maximum_submissions', 2),
    ('experiment', 'other'), ('node', 'nid002744'), ('manifest_sha256', 'bad')])
def test_approval_refuses_expanded_scope(transport, approved, key, bad):
    value, path, _ = approved
    value[key] = bad
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        transport.approval()


def test_exact_one_shot_scheduler_boundary_and_receipt_race(transport, approved, monkeypatch):
    _, _, evidence = approved
    value, directory = fixture_value(), Path('/future')
    monkeypatch.setattr(transport, 'validate_run', lambda path: value)
    calls = []
    monkeypatch.setattr(transport.os, 'execv', lambda program, args: calls.append(args))
    transport.scheduler('sbatch', ['--parsable', str(directory / 'job.sh')])
    assert calls[0][:3] == ['/usr/bin/sbatch', '--uenv-passthrough=ignore', f'--nodelist={transport.NODE}']
    # The durable intent, not a submission receipt, admits the owned step.
    transport.os.environ.update(SLURM_JOB_ID='999999', SLURM_JOB_NODELIST=transport.NODE, SLURM_JOB_USER='boeschf')
    args = [*runs.step_command(value, directory, cpus=48, job_id='999999')[1:], '0']
    transport.scheduler('srun', args)
    assert calls[-1][:2] == ['/usr/bin/srun', '--uenv-passthrough=ignore']
    assert calls[-1][2] == '--label'
    assert 'bootstrap' in calls[-1]
    with pytest.raises(FileExistsError):
        transport.scheduler('srun', args)
    assert (evidence / 'sbatch-intent.json').exists()


def test_real_diagnostic_runner_freeze_and_marker_transport(transport, approved, tmp_path, monkeypatch):
    _, _, evidence = approved
    monkeypatch.setattr(transport, 'ROOT', tmp_path)
    value = config.resolve(ROOT / 'experiments' / transport.EXPERIMENT, site=ROOT / 'config.toml',
                           submit=True, reservation='cscs', time='00:05:00')
    value['data_root'] = str(tmp_path / 'artifacts/experiment-data')
    (tmp_path / 'resolved-input.json').write_text(json.dumps(value))
    tooling = {str(path.relative_to(ROOT / 'src')): transport.digest(path)
               for path in (ROOT / 'src/apertus_amd').glob('*.py')}
    (tmp_path / 'tooling-hashes.json').write_text(json.dumps(tooling))
    monkeypatch.setattr(runs, 'image_manifest', lambda *args: {})
    calls = []
    monkeypatch.setattr(transport.os, 'execv', lambda program, args: calls.append(args))
    def submit(command, **kwargs):
        transport.scheduler('sbatch', command[1:])
        return SimpleNamespace(stdout='999999\n')
    monkeypatch.setattr(runs.subprocess, 'run', submit)
    directory = runs.run(value)
    assert transport.validate_run(directory)['validation']['sha256'] == value['validation']['sha256']
    (directory / 'submission.json').unlink()
    transport.os.environ.update(SLURM_JOB_ID='999999', SLURM_JOB_NODELIST=transport.NODE, SLURM_JOB_USER='boeschf')
    actual = json.loads((directory / 'resolved.json').read_text())
    args = [*runs.step_command(actual, directory, cpus=48, job_id='999999')[1:], '0']
    transport.scheduler('srun', args)
    assert calls[-1][-3:] == ['/apertus/validation.py', 'bootstrap', '0']
    assert '--label' in calls[-1] and 'importtime' in calls[-1]
    path = directory / 'container.toml'
    path.write_text(path.read_text().replace('com.hooks.cxi.enabled = "false"', 'com.hooks.cxi.enabled = "true"'))
    with pytest.raises(ValueError, match='EDF differs'):
        transport.validate_run(directory)
    assert (evidence / 'sbatch-intent.json').exists()
