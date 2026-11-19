"""Root-wired record validation and offline classification; no vendor imports."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'experiments/validation/chonk_grouped_forward_performance.py'
ARTIFACT=ROOT/'artifacts/investigations/performance-next/restart-20261116/grouped-forward-performance-record-validation'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


p=load('root_performance_records',DRIVER)
fixtures=load('root_performance_fixture_source',ARTIFACT/'test_validator.py')
namespace=dict(vars(p)); namespace['PACKAGE']=str(p.PACKAGE)+'/'
fixtures.v=SimpleNamespace(**namespace)
original_worker=fixtures.worker


def worker(rank=0,slot=0):
    r=original_worker(rank,slot)
    r['operands_device_pre']=r['operands']
    r['affinity']=dict(initial=list(range(48)),post_gpu_init=list(range(48)),final=list(range(48)),restoration_attempted=False,restoration_applied=False,errors=[])
    return r


fixtures.worker=worker
test_complete_all_ranks_nonmutating_and_counts=fixtures.test_complete_all_ranks_nonmutating_and_counts
test_worker_failures=fixtures.test_worker_failures
test_rank_failures=fixtures.test_rank_failures


def test_required_native_fields_and_empty_baseline_cache():
    for field in ('affinity','origins','mapped_elfs','operands_device_pre'):
        r=worker();del r[field]
        with pytest.raises(ValueError):p.validate_worker(r,rank=0,slot=0,**fixtures.kwargs())
    r=worker();r['cache']['before_pre']['gmm_keys']={'0':['e'*64]}
    with pytest.raises(ValueError):p.validate_worker(r,rank=0,slot=0,**fixtures.kwargs())
    r=worker();r['operands_device_pre']={}
    with pytest.raises(ValueError):p.validate_worker(r,rank=0,slot=0,**fixtures.kwargs())


def test_analysis_is_exact_approved_function():
    reference=ROOT/'artifacts/investigations/performance-next/restart-20261116/grouped-forward-performance-protocol-audit/analysis.py'
    def function(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='analyze'),include_attributes=False)
    assert function(reference)==function(DRIVER)


def offline_tree(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();(run/'validation.py').write_bytes(DRIVER.read_bytes());checksum=p.digest(DRIVER)
    settings=dict(validation=dict(sha256=checksum),image=dict(sha256=p.IMAGE),procedure=dict(kind='commands',commands=[p.COMMAND],timeout=510),
                  resources=dict(nodes=1,gpus_per_node=4,cpus_per_node=192,mpi='pmix',time='00:10:00',reservation='cscs',cpu_bind='map_ldom:0,1,2,3',distribution='block:block'))
    (run/'resolved.json').write_text(json.dumps(settings));(run/'result.json').write_text(json.dumps(dict(status='passed',execution='passed')))
    monkeypatch.setattr(p,'OPERANDS_SHA',p.object_digest(fixtures.OPERANDS))
    for rank in range(4):
        base=run/f'output/0/grouped-performance-{rank}';base.mkdir(parents=True)
        summary=fixtures.summary(rank);summary['driver_sha256']=checksum
        for slot,r in enumerate(summary['workers']):
            r['driver_sha256']=checksum;r['provenance']=p.expected_provenance()
            directory=base/str(slot);directory.mkdir();(directory/'worker.json').write_text(json.dumps(r))
        (base/'summary.json').write_text(json.dumps(summary))
        (run/f'output/0/rank-{rank}.json').write_text(json.dumps(dict(status='passed',rank=rank,argv=p.COMMAND)))
    return run


def test_offline_exclusive_derived_without_native_execution(tmp_path,monkeypatch):
    run=offline_tree(tmp_path,monkeypatch);target=tmp_path/'derived.json'
    assert p.IMPLEMENTATION_READY is True
    result=p.check(run,target);assert result['status']=='qualified' and result['analysis']['passed'] is False
    before=target.read_bytes()
    with pytest.raises(ValueError):p.check(run,target)
    assert target.read_bytes()==before
    with pytest.raises(ValueError):p.check(run,run/'overwrite.json')


def test_offline_worker_file_mismatch_rejected(tmp_path,monkeypatch):
    run=offline_tree(tmp_path,monkeypatch);file=run/'output/0/grouped-performance-0/0/worker.json'
    r=json.loads(file.read_text());r['pid']+=1;file.write_text(json.dumps(r))
    with pytest.raises(ValueError):p.check(run)
