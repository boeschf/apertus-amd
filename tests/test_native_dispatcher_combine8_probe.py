"""CPU-only contracts/failure gates; no vendor imports or GPU correctness claims."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from apertus_amd.config import resolve
from apertus_amd.runs import edf
from apertus_amd.runtime import training_command

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/native-dispatcher-combine8-me242f3c3c113-i910bfcf936d3"


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("native_dispatcher_combine8_probe", ROOT / "experiments/validation/native_dispatcher_combine8_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings():
    return resolve(EXPERIMENT, submit=True, reservation="cscs", time="00:30:00")


def test_explicit_source_staged_validator(probe, settings):
    probe.validate(settings)
    text = edf(settings, Path("/run"))
    assert "/apertus/validation.py:ro" in text
    assert 'com.hooks.cxi.enabled = "false"' in text
    assert 'com.hooks.aws_ofi_nccl.enabled = "false"' in text
    assert 'HWLOC_COMPONENTS = "-gl"' in text
    command = training_command(settings, Path("/output"), Path("/source"), [])
    assert command[1:3] == ["/apertus/validation.py", "train"]
    assert settings["source"]["commit"] == probe.SOURCE


@pytest.mark.parametrize("mutate", [
    lambda s: s["source"].update(commit="0" * 40),
    lambda s: s["image"].update(sha256="0" * 64),
    lambda s: s["resources"].update(nodes=16),
    lambda s: s["resources"].update(time="01:00:00"),
    lambda s: s["resources"].update(reservation=None),
    lambda s: s["procedure"].update(kind="commands"),
    lambda s: s["args"].extend(["--expert-model-parallel-size", "16"]),
    lambda s: s["args"].append("--fp8"),
    lambda s: s["args"].append("--moe-shared-expert-overlap"),
    lambda s: s["args"].remove("--no-gradient-accumulation-fusion"),
    lambda s: s["environment"].update(UCCL_EP_TRANSPORT="rdma"),
    lambda s: s["environment"].update(NCCL_P2P_DISABLE="0"),
])
def test_reject_scope_drift(probe, settings, mutate):
    mutate(settings)
    with pytest.raises(ValueError):
        probe.validate(settings)


@pytest.mark.parametrize("counts,rank,ok", [
    ([1, 2, 3, 4, 5, 6, 7, 8], 0, True),
    ([1] * 8, 0, False),
    ([2, 0, 0, 0, 1, 0, 0, 0], 0, False),
    ([2, 1, 0, 0, 0, 0, 0, 0], 0, False),
    ([1, 2, -1, 4, 5, 6, 7, 8], 0, False),
])
def test_independent_coverage(probe, counts, rank, ok):
    assert probe.coverage(counts, rank) is ok


def test_numerical_policy_and_zero_reference(probe):
    assert (probe.RTOL, probe.ATOL, probe.RMS_RTOL, probe.RMS_ATOL) == (0.02, 1e-5, 2**-7, 1e-7)
    m = dict(finite=True, max_tolerance_ratio=0.5, rms_error=0.001, reference_rms=1)
    assert probe.accepted(m)
    for changes in (dict(finite=False), dict(max_tolerance_ratio=1.01), dict(rms_error=0.01),
                    dict(reference_rms=0), dict(rms_error=float("nan")), dict(max_tolerance_ratio=None)):
        assert not probe.accepted({**m, **changes})
    assert probe.accepted({**m, "reference_rms": 0, "rms_error": 0})


def valid_record(probe):
    snapshot = probe.getter_observation(FakeBuffer, {'UCCL_EP_COMBINE_CONFIG': probe.COMBINE_CONFIG})
    return dict(source=probe.SOURCE, image=probe.IMAGE, scope=probe.SCOPE, rank=0,
                getters_before_hints=snapshot, getters_after_work=deepcopy(snapshot),
                parameters={"router.weight": [256, 3584]}, microbatches=[dict(index=0), dict(index=1)],
                checks=[dict(label=k, passed=True) for k in sorted(probe.required_labels(["router.weight"]) | {'getter-configs-before-hints','getter-configs-after-work'})])


def test_record_missing_gradient_partial_duplicate_failed(probe):
    record = valid_record(probe)
    probe.validate_record(record, 0)
    for label in probe.required_labels(record["parameters"]):
        damaged = deepcopy(record)
        damaged["checks"] = [r for r in damaged["checks"] if r["label"] != label]
        with pytest.raises(ValueError):
            probe.validate_record(damaged, 0)
    for damaged in [dict(record, checks=record["checks"] * 2), dict(record, microbatches=[dict(index=0)]),
                    dict(record, parameters={}), dict(record, checks=[dict(label="identity", passed=False)])]:
        with pytest.raises(ValueError):
            probe.validate_record(damaged, 0)


def test_no_gpu_import_for_cpu_planning(probe):
    # Importing the file and evaluating contracts above must never import vendors.
    import ast
    tree = ast.parse(Path(probe.__file__).read_text())
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("torch" in ast.unparse(n) or "megatron" in ast.unparse(n) for n in top_imports)


def test_source_staging_keeps_normal_checked_cache_lifecycle_with_exec_deadline(probe, settings, monkeypatch, tmp_path):
    assert settings["procedure"]["kind"] == "train"
    assert "timeout" not in settings["procedure"]
    import json, os
    (tmp_path / 'resolved.json').write_text(json.dumps(settings))
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setattr(probe, 'timeout_identity', lambda: {'version': 'test'})
    captured = {}
    def execute(path, command, env):
        captured.update(path=path, command=command, env=env)
    monkeypatch.setattr(probe.os, 'execve', execute)
    before = dict(os.environ)
    probe.train(tmp_path, Path('/source'), 'pretrain_gpt.py', ['native'], run=tmp_path)
    assert captured['command'][:5] == ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '660s']
    assert captured['env'] == {**before, 'APERTUS_NATIVE_DISPATCHER_PGID': str(os.getpgrp())}
    assert dict(os.environ) == before


def test_effective_async_flags_bind_actual_positional_arguments(probe):
    def native(x, sf=None, probs=None, async_finish=True, allocate_on_comm_stream=True):
        pass
    assert probe.effective_flags(native, (1,), {}) == dict(async_finish=True, allocate_on_comm_stream=True)
    assert probe.effective_flags(native, (1, None, None, False, False), {}) == dict(async_finish=False, allocate_on_comm_stream=False)
    assert probe.effective_flags(native, (1,), {'async_finish': False})['async_finish'] is False


@pytest.mark.parametrize('mode', ['term', 'kill', 'normal-helper', 'outer-timeout'])
def test_real_foreground_timeout_group_ownership_and_cleanup(probe, tmp_path, monkeypatch, mode):
    import json, os, signal, subprocess, sys
    from apertus_amd import runtime
    monkeypatch.setattr(runtime, 'CHILD_GRACE', 0.2)
    probe.timeout_identity()
    helper = "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)"
    script = tmp_path / 'worker.py'
    script.write_text(
        'import os,signal,subprocess,sys,time,json\nfrom pathlib import Path\n'
        f'helper=subprocess.Popen([sys.executable,"-c",{helper!r}])\n'
        f'Path({str(tmp_path / "pids.json")!r}).write_text(json.dumps([os.getpid(),os.getpgrp(),helper.pid,os.getpgid(helper.pid)]))\n'
        + ('signal.signal(signal.SIGTERM,signal.SIG_IGN)\n' if mode in {'kill','outer-timeout'} else '')
        + ('sys.exit(0)\n' if mode == 'normal-helper' else 'time.sleep(20)\n'))
    command = ['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=0.2s',
               '10s' if mode in {'outer-timeout','normal-helper'} else '0.5s',sys.executable,str(script)]
    env = dict(os.environ)
    if mode == 'normal-helper':
        runtime.child(command,tmp_path/'log',cwd=tmp_path,env=env,timeout=3)
    else:
        with pytest.raises((ValueError, subprocess.TimeoutExpired)) as error:
            runtime.child(command,tmp_path/'log',cwd=tmp_path,env=env,timeout=0.5 if mode=='outer-timeout' else 3)
        if mode in {'term','kill'}:
            # This host GNU timeout reports 124 even after foreground KILL;
            # retain observed status rather than inventing a mandatory 137.
            assert '(124)' in str(error.value) or '(137)' in str(error.value)
    pid, pgid, helper_pid, helper_pgid = json.loads((tmp_path/'pids.json').read_text())
    assert pgid == helper_pgid and pgid != os.getpgrp()
    assert not runtime.group_running(pgid)
    assert dict(os.environ) == env
    print(dict(mode=mode, worker_pid=pid, worker_pgid=pgid, helper_pid=helper_pid, helper_pgid=helper_pgid,
               quiescent=True, outcome='normal' if mode=='normal-helper' else str(error.value)))


def test_outer_unreaped_type_reaches_retention_guard(tmp_path, monkeypatch):
    import json, os, sys
    from apertus_amd import runtime
    original = runtime.stop_child
    def fail(process):
        original(process)  # Keep the injected-failure CPU test itself quiescent.
        raise runtime.UnreapedChildError('injected outer stop failure')
    monkeypatch.setattr(runtime, 'stop_child', fail)
    with pytest.raises(runtime.UnreapedChildError):
        runtime.child(['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=0.1s','1s',sys.executable,'-c','pass'],
                      tmp_path/'child.log',cwd=tmp_path,env=dict(os.environ),timeout=2)
    monkeypatch.setattr(runtime, 'stop_child', original)
    scratch, session = tmp_path/'scratch', tmp_path/'session'
    scratch.mkdir(); session.mkdir()
    (scratch/'owner.json').write_text('{}')
    status = dict(status='failed', children_reaped=False)
    runtime.finalize(tmp_path,tmp_path,session,scratch,{'node':0},rank=0,world=1,local=0,ranks_per_node=1,
                     timeout=0.01,status=status,result=dict(status='failed'))
    assert scratch.exists() and not (session/'exited-0.json').exists()


def test_bounded_pinned_source_conformance_without_vendor_imports(probe):
    import ast, hashlib, json, tarfile
    base = ROOT/'artifacts/investigations/performance-next/restart-20261113/native-dispatcher-preparation/source'/probe.SOURCE
    manifest = json.loads((base/'manifest.json').read_text())
    assert manifest['selection']['commit'] == probe.SOURCE
    with tarfile.open(base/'source.tar') as archive:
        paths = ['megatron/core/transformer/moe/token_dispatcher.py','megatron/core/transformer/moe/fused_a2a.py',
                 'megatron/core/transformer/moe/moe_layer.py','megatron/core/parallel_state.py']
        trees = {}
        for path in paths:
            raw = archive.extractfile(path).read()
            assert hashlib.sha256(raw).hexdigest() == manifest['files'][path]
            trees[path] = ast.parse(raw)
    classes = {n.name:n for n in trees[paths[0]].body if isinstance(n,ast.ClassDef)}
    assert {'MoEFlexTokenDispatcher','MoEAlltoAllTokenDispatcher','_DeepepManager'} <= classes.keys()
    for name in ('token_dispatch','token_combine'):
        method = next(n for n in classes['MoEFlexTokenDispatcher'].body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert [ast.literal_eval(n) for n in method.args.defaults[-2:]] == [True,True]
    attrs = {n.attr for n in ast.walk(trees[paths[0]]) if isinstance(n,ast.Attribute)}
    assert {'ep_group','tp_ep_group','_comm_manager','handle','token_indices','input_splits','output_splits'} <= attrs
    assert {'FusedDispatch','FusedCombine'} <= {n.name for n in trees[paths[1]].body if isinstance(n,ast.ClassDef)}
    groups = {n.name for n in trees[paths[3]].body if isinstance(n,ast.FunctionDef)}
    assert {'get_expert_model_parallel_group','get_expert_tensor_parallel_group','get_expert_tensor_and_model_parallel_group'} <= groups


class FakeConfig:
    """CPU model of the audited Config fields/receive rounding, not a native import."""
    def __init__(self, sms, nvl_send, nvl_recv, rdma_send, rdma_recv):
        self.num_sms, self.num_max_nvl_chunked_send_tokens, self.num_max_nvl_chunked_recv_tokens = sms, nvl_send, nvl_recv
        self.num_max_rdma_chunked_send_tokens = rdma_send
        self.num_max_rdma_chunked_recv_tokens = ((rdma_recv + rdma_send - 1) // rdma_send) * rdma_send


class FakeBuffer:
    num_sms = 20
    @staticmethod
    def get_dispatch_config(world):
        assert world == 8
        return FakeConfig(20,6,256,6,128)
    @staticmethod
    def get_combine_config(world):
        assert world == 8
        return FakeConfig(20,4,256,8,128)


@pytest.mark.parametrize('extra', [
    {}, {'UCCL_EP_COMBINE_CONFIG': ''}, {'UCCL_EP_COMBINE_CONFIG': '20,4,256,6,128'},
    {'UCCL_EP_COMBINE_CONFIG': '64,8,512,16,512'},
    {'UCCL_EP_COMBINE_CONFIG': '20,4,256,8,128', 'UCCL_EP_DISPATCH_CONFIG': ''},
    {'UCCL_EP_COMBINE_CONFIG': '20,4,256,8,128', 'UCCL_EP_DISPATCH_CONFIG': '20,6,256,6,128'},
])
def test_strict_override_environment(probe, settings, extra):
    settings['environment'].pop('UCCL_EP_COMBINE_CONFIG')
    settings['environment'].update(extra)
    with pytest.raises(ValueError): probe.validate(settings)
    with pytest.raises(ValueError): probe.validate_config_environment(extra)


def test_getter_observations_are_effective_and_fail_closed(probe):
    env={'UCCL_EP_COMBINE_CONFIG':probe.COMBINE_CONFIG, 'UCCL_EP_TRANSPORT':'cxi'}
    snapshot=probe.getter_observation(FakeBuffer,env)
    probe.validate_getter_observation(snapshot)
    assert snapshot['constructor_inputs']['dispatch'][-1]==128
    assert snapshot['effective']['dispatch']['num_max_rdma_chunked_recv_tokens']==132
    assert snapshot['effective']['combine']['num_max_rdma_chunked_recv_tokens']==128
    for field,value in [('num_sms',64),('num_max_rdma_chunked_send_tokens',6)]:
        damaged=deepcopy(snapshot); damaged['effective']['combine'][field]=value
        with pytest.raises(ValueError): probe.validate_getter_observation(damaged)
    changed=deepcopy(snapshot);changed['uccl_environment']['UCCL_EP_TRANSPORT']='other'
    with pytest.raises(ValueError): probe.validate_getter_observation(changed,snapshot)
    for key in ('getters_before_hints','getters_after_work'):
        record=valid_record(probe);record[key]['effective']['combine']['num_sms']=64
        with pytest.raises(ValueError): probe.validate_record(record,0)


def test_new_experiment_changes_only_explicit_combine_override_and_validator():
    import tomllib
    old=tomllib.loads((ROOT/'experiments/native-dispatcher-me242f3c3c113-i910bfcf936d3/experiment.toml').read_text())
    new=tomllib.loads((EXPERIMENT/'experiment.toml').read_text())
    assert new['environment'].pop('UCCL_EP_COMBINE_CONFIG')=='20,4,256,8,128'
    assert new['procedure'].pop('validation')=='../validation/native_dispatcher_combine8_probe.py'
    old['procedure'].pop('validation')
    assert new==old


def test_numerical_and_lifecycle_body_byte_identical(probe):
    import ast
    old=(ROOT/'experiments/validation/native_dispatcher_probe.py').read_text()
    new=Path(probe.__file__).read_text()
    def functions(text):
        return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(old),functions(new)
    for name in ('accepted','compare','exact','rng_state','restore_rng','rng_digest','coverage','observe',
                 'required_labels','effective_flags','train','timeout_identity'):
        assert a[name]==b[name],name
    start='    for mb in range(2):\n'
    assert a['run_probe'][a['run_probe'].index(start):a['run_probe'].index('    # The only global synchronization')] == \
        b['run_probe'][b['run_probe'].index(start):b['run_probe'].index("    record['getters_after_work']")]
    end='\n\ndef gpu('
    assert old[old.index('    # The only global synchronization'):old.index(end)] == \
        new[new.index('    # The only global synchronization'):new.index(end)]


def test_audited_wrapper_knob_fields_and_rounding_are_real(probe):
    import tarfile
    archive=ROOT/'artifacts/image-builds/primus268-uccl52d4184-aiter-20261112/uccl-history-audit/uccl-52d4184-source.tar.gz'
    with tarfile.open(archive) as t:
        wrapper=t.extractfile('ep/bench/buffer.py').read().decode()
        binding=t.extractfile('ep/src/uccl_ep.cc').read().decode()
        config=t.extractfile('ep/include/ep_config.hpp').read().decode()
    assert 'Buffer._config_from_env("UCCL_EP_COMBINE_CONFIG")' in wrapper
    assert '_, env_config = Buffer._validated_env_configs()' in wrapper
    assert 'config = self.get_combine_config(self.group_size) if config is None else config' in wrapper
    assert 'num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);' in config
    assert all('"'+field+'"' in binding for field in probe.CONFIG_FIELDS)
