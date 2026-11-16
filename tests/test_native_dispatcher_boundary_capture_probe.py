"""CPU fake-hook/bit-integrity tests; never import Torch or run a GPU oracle."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from apertus_amd.config import resolve

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'experiments/validation/native_dispatcher_boundary_capture_probe.py'


@pytest.fixture
def probe():
    spec=importlib.util.spec_from_file_location('boundary_capture_probe',PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Handle:
    def __init__(self, hooks, hook):
        self.hooks,self.hook=hooks,hook
    def remove(self):
        if self.hook in self.hooks:self.hooks.remove(self.hook)


class Tensor:
    is_cuda=True
    requires_grad=True
    def __init__(self,width,dtype='torch.bfloat16',shape=None):
        self.shape=shape or (4096,1,width);self.dtype=dtype;self.hooks=[];self.copies=0
    def stride(self):return (self.shape[-1],self.shape[-1],1)
    def storage_offset(self):return 7
    def element_size(self):return 2 if self.dtype=='torch.bfloat16' else 4
    def detach(self):return self
    def index_select(self,dim,indices):
        assert dim==0 and list(indices)==list(range(0,4096,64))
        return Owned(self.shape[-1],self.dtype)
    def __getitem__(self,index):
        assert index==slice(None,None,64)
        return Owned(self.shape[-1],self.dtype)
    def register_hook(self,hook):
        self.hooks.append(hook);return Handle(self.hooks,hook)
    def backward_sample(self):
        for hook in list(self.hooks):assert hook(self) is None


class Owned(Tensor):
    def __init__(self,width,dtype):
        super().__init__(width,dtype,(64,1,width))
        self.raw=(struct.pack('<H',0x3f81) if dtype=='torch.bfloat16' else struct.pack('<I',0x3f800001))*(64*width)
    def clone(self):return self
    def view(self,dtype):assert dtype=='uint8';return self
    def cpu(self):self.copies+=1;return self
    def numpy(self):return self
    def tobytes(self):return self.raw


class Torch:
    Tensor=Tensor;long='long';bfloat16='torch.bfloat16';float32='torch.float32';uint8='uint8'
    enabled=False
    def tensor(self,values,**kwargs):return tuple(values)
    def is_grad_enabled(self):return self.enabled


class Module:
    def __init__(self):self.before=[];self.after=[]
    def register_forward_pre_hook(self,hook):
        self.before.append(hook);return Handle(self.before,hook)
    def register_forward_hook(self,hook):
        self.after.append(hook);return Handle(self.after,hook)
    def call(self,x,y):
        for hook in self.before:assert hook(self,(x,)) is None
        result=(y,None)
        for hook in self.after:assert hook(self,(x,),result) is None
        return result


def exercise(probe,capture,torch,dtype='torch.bfloat16'):
    models={mode:SimpleNamespace(**{name:Module() for name in probe.PROJECTIONS}) for mode in probe.MODES}
    capture.attach(models)
    for mode in probe.MODES:
        for mb in range(2):
            with capture.arm(mode,mb):
                for enabled in (False,True):
                    torch.enabled=enabled
                    for name,(a,b) in probe.PROJECTIONS.items():
                        x,y=Tensor(a,dtype),Tensor(b,dtype)
                        assert getattr(models[mode],name).call(x,y)[0] is y
                        if enabled:x.backward_sample();y.backward_sample()
    return models


@pytest.mark.parametrize('dtype,word',[
    ('torch.bfloat16',struct.pack('<H',0x3f81)),('torch.float32',struct.pack('<I',0x3f800001))])
def test_owned_hooks_quiescent_bit_serialization_and_integrity(probe,tmp_path,dtype,word):
    torch=Torch()
    with probe.BoundaryCapture(torch,tmp_path,0) as capture:
        models=exercise(probe,capture,torch,dtype)
        assert len(capture.entries)==48
        assert all(o.copies==0 for o in capture.owned)
        assert not list(tmp_path.iterdir())
        capture.serialize_quiescent()
    assert all(not m.before and not m.after for model in models.values() for m in vars(model).values())
    assert capture.summary['status']=='complete'
    probe.verify_boundary_artifacts(tmp_path,0,capture.summary)
    assert (tmp_path/'boundary-0.bin').read_bytes()[:len(word)]==word
    with (tmp_path/'boundary-0.bin').open('r+b') as stream:stream.write(b'BAD')
    with pytest.raises(ValueError,match='integrity'):probe.verify_boundary_artifacts(tmp_path,0,capture.summary)


def test_api_failure_does_not_serialize_and_removes_hooks(probe,tmp_path):
    torch=Torch();capture=probe.BoundaryCapture(torch,tmp_path,0)
    with pytest.raises(RuntimeError,match='native first'):
        with capture:
            models={mode:SimpleNamespace(**{name:Module() for name in probe.PROJECTIONS}) for mode in probe.MODES}
            capture.attach(models)
            with capture.arm('alltoall',0):
                models['alltoall'].fc1_latent_proj.call(Tensor(3584),Tensor(1792))
            assert all(o.copies==0 for o in capture.owned)
            raise RuntimeError('native first')
    assert capture.summary['status']=='partial'
    assert not (tmp_path/'boundary-0.bin').exists()
    assert all(not m.before and not m.after for model in models.values() for m in vars(model).values())


def test_bounds_shape_duplicates_and_missing_gradient_fail_closed(probe,tmp_path,monkeypatch):
    torch=Torch()
    with probe.BoundaryCapture(torch,tmp_path,0) as c:
        with c.arm('alltoall',0):
            c.observe_tensor(Tensor(17), 'fc1_latent_proj','input')
            c.observe_tensor(Tensor(3584), 'fc1_latent_proj','input')
            c.observe_tensor(Tensor(3584), 'fc1_latent_proj','input')
        monkeypatch.setattr(probe,'BOUNDARY_BYTES',probe.BOUNDARY_METADATA_BYTES+1)
        with c.arm('flex',0):c.observe_tensor(Tensor(3584),'fc1_latent_proj','input')
    assert c.summary['status']=='partial' and len(c.errors)==3
    assert not (tmp_path/'boundary-0.bin').exists()
    with pytest.raises(ValueError):probe.verify_boundary_artifacts(tmp_path,0,c.summary)


def test_observer_failure_does_not_replace_original_failure(probe,tmp_path,monkeypatch):
    torch=Torch();capture=probe.BoundaryCapture(torch,tmp_path,0)
    monkeypatch.setattr(probe,'write_json',lambda *a:(_ for _ in ()).throw(OSError('index failure')))
    with pytest.raises(RuntimeError,match='original API'):
        with capture:raise RuntimeError('original API')
    assert capture.summary['status']=='partial' and any('index failure' in e for e in capture.errors)


def test_missing_replay_dgrad_cannot_complete(probe,tmp_path,monkeypatch):
    torch=Torch()
    monkeypatch.setattr(Tensor,'backward_sample',lambda self:None)
    with probe.BoundaryCapture(torch,tmp_path,0) as capture:
        exercise(probe,capture,torch)
        capture.serialize_quiescent()
    assert capture.summary['entries']==32 and capture.summary['status']=='partial'
    with pytest.raises(ValueError,match='incomplete'):probe.verify_boundary_artifacts(tmp_path,0,capture.summary)


def test_partial_hook_registration_is_additive_and_removed(probe,tmp_path):
    class BrokenModule(Module):
        def register_forward_hook(self,hook):raise OSError('hook registration failed')
    models={mode:SimpleNamespace(fc1_latent_proj=BrokenModule(),fc2_latent_proj=Module()) for mode in probe.MODES}
    capture=probe.BoundaryCapture(Torch(),tmp_path,0)
    with pytest.raises(RuntimeError,match='original native'):
        with capture:
            capture.attach(models)  # Must not preempt native execution.
            raise RuntimeError('original native')
    assert any('hook registration failed' in e for e in capture.errors)
    assert all(not m.before and not m.after for model in models.values() for m in vars(model).values())


def test_serialization_error_is_additive_to_original_numerical_rejection(probe,tmp_path,monkeypatch):
    torch=Torch();original=Path.open
    def fail_payload(path,*args,**kwargs):
        if path.name.endswith('.bin'):raise OSError('payload write failed')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_payload)
    with probe.BoundaryCapture(torch,tmp_path,0) as capture:
        exercise(probe,capture,torch)
        capture.serialize_quiescent()  # Must not raise ahead of original gates.
    assert capture.summary['status']=='partial' and any('payload write failed' in e for e in capture.errors)
    fields=probe.CONFIG_FIELDS
    getter=dict(world=8,num_sms=20,override_environment={'UCCL_EP_COMBINE_CONFIG':probe.COMBINE_CONFIG},
                effective=dict(dispatch=dict(zip(fields,(20,6,256,6,132))),combine=dict(zip(fields,(20,4,256,8,128)))))
    record=dict(source=probe.SOURCE,image=probe.IMAGE,scope=probe.SCOPE,rank=0,parameters={'weight':[1]},
                getters_before_hints=getter,getters_after_work=deepcopy(getter),microbatches=[{'index':0},{'index':1}],
                checks=[dict(label=k,passed=k!='mb0/forward') for k in probe.required_labels(['weight'])],boundary_capture=capture.summary)
    with pytest.raises(ValueError,match='native dispatcher evidence'):
        probe.validate_record(record,0)


def test_scientific_sections_and_original_validation_are_unchanged(probe):
    old=(ROOT/'experiments/validation/native_dispatcher_combine8_probe.py').read_text();new=PATH.read_text()
    def funcs(text):return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    a,b=funcs(old),funcs(new)
    for name in ('validate','accepted','compare','exact','rng_state','restore_rng','rng_digest','coverage','observe','required_labels',
                 'validate_record','train','timeout_identity','identity','getter_observation','validate_getter_observation'):
        assert a[name]==b[name],name
    amended=b['run_probe'].replace(', *, capture)',')').replace('    capture.attach(models)\n','').replace('with capture.arm(mode, mb), observe(', 'with observe(').replace('    capture.serialize_quiescent()\n','')
    assert amended==a['run_probe']
    gpu=b['gpu'];assert gpu.index('validate_record(record, rank)')<gpu.index('verify_boundary_artifacts(output, rank')
    assert 'run_probe(emit, record' in gpu and 'record[\'boundary_capture\'] = capture.summary' in gpu
    block="""            capture = BoundaryCapture(torch, output, rank)
            try:
                with capture:
                    run_probe(emit, record, source, manifest["files"], capture=capture)
            finally:
                record['boundary_capture'] = capture.summary"""
    assert gpu.replace(block,'            run_probe(emit, record, source, manifest["files"])').replace(
        "            verify_boundary_artifacts(output, rank, record['boundary_capture'])\n",'')==a['gpu']


def test_input_identity_budget_and_no_vendor_import(probe):
    assert probe.BOUNDARY_TOKENS==tuple(64*j for j in range(64))
    assert probe.BOUNDARY_BYTES*8==512*1024*1024 and len(probe.boundary_keys())==48
    exp=ROOT/'experiments/native-dispatcher-boundary-me242f3c3c113-i910bfcf936d3'
    settings=resolve(exp,submit=True,reservation='cscs',time='00:30:00',cache_mode='cold')
    probe.validate(settings)
    import tomllib
    original=tomllib.loads((ROOT/'experiments/native-dispatcher-combine8-me242f3c3c113-i910bfcf936d3/experiment.toml').read_text())
    captured=tomllib.loads((exp/'experiment.toml').read_text())
    assert captured['procedure'].pop('validation')=='../validation/native_dispatcher_boundary_capture_probe.py'
    original['procedure'].pop('validation')
    assert captured==original
    tree=ast.parse(PATH.read_text())
    assert all('import torch' not in ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
