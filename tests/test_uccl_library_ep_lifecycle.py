"""CPU evidence for the actual EP launcher ownership and terminal-record paths."""
import builtins
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'artifacts/investigations/uccl-upstream-tests-20261119/frozen'


@pytest.fixture
def p():
    spec=importlib.util.spec_from_file_location('upstream_ep_lifecycle',ROOT/'experiments/validation/uccl_library_ep_tests.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('preexisting',[False,True])
def test_actual_eight_rank_initial_barrier_agreement(p,tmp_path,monkeypatch,preexisting):
    output=tmp_path/'output/0';output.mkdir(parents=True)
    monkeypatch.setenv('SLURM_JOB_ID','999999')
    barrier=tmp_path/'uccl_barrier_old_ht_0';barrier.write_text('preserve')
    original=Path.glob
    monkeypatch.setattr(Path,'glob',lambda path,pattern:iter([barrier] if preexisting else []) if path==Path('/dev/shm') else original(path,pattern))
    def check(rank):
        try:return p.barrier_ownership(tmp_path,output,'intranode',rank)
        except ValueError as error:return str(error)
    with ThreadPoolExecutor(max_workers=8) as pool:result=list(pool.map(check,range(8)))
    assert len(list(output.glob('initial-ep-shm-*.json')))==8
    if preexisting:assert all('preexisting' in r for r in result)
    else:assert result==[[]]*8
    assert barrier.read_text()=='preserve'


@pytest.mark.parametrize('outcome',['success','assertion','evidence_failure'])
def test_actual_launcher_terminal_evidence_and_primary_error(p,tmp_path,monkeypatch,outcome):
    from apertus_amd.config import resolve
    settings=resolve(ROOT/'experiments/uccl-library-ep-i910bfcf936d3',site=ROOT/'config.toml',
                     submit=True,reservation='cscs',time='00:30:00')
    settings['scratch_root']=str(tmp_path/'scratch')
    monkeypatch.chdir(tmp_path)
    for key,value in dict(settings['environment'],HWLOC_COMPONENTS='-gl',SLURM_PROCID='0',SLURM_LOCALID='0',SLURM_NODEID='0',SLURM_NTASKS='8',SLURM_JOB_ID='999999').items():monkeypatch.setenv(key,value)
    for key in ('UCCL_EP_DISPATCH_CONFIG','UCCL_EP_COMBINE_CONFIG'):monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(p.os,'sched_getaffinity',lambda _:set(range(48)))
    monkeypatch.setattr(p,'barrier_ownership',lambda *a:[])
    monkeypatch.setattr(p,'rendezvous',lambda *a:dict(WORLD_SIZE='4',RANK='0',LOCAL_RANK='0'))
    bench=ASSETS/'upstream/ep/bench'
    modules={name:SimpleNamespace(__file__=str(bench/f'{name}.py')) for name in ('test_intranode','utils','buffer')}
    modules['buffer'].Buffer=object()
    modules['deep_ep.buffer']=SimpleNamespace(__file__=str(bench/'buffer.py'))
    ep=SimpleNamespace(__file__=str(bench/'buffer.py'),num_max_nvl_peers=lambda:4)
    torch=SimpleNamespace(version=SimpleNamespace(hip='fixture'),set_num_threads=lambda n:None,
                          cuda=SimpleNamespace(device_count=lambda:4,set_device=lambda n:None,current_device=lambda:0))
    original_import=builtins.__import__
    def importing(name,*a,**k):
        if name=='torch':return torch
        if name=='uccl.ep':return SimpleNamespace(ep=ep)
        if name.split('.')[0] in {'uccl','deep_ep','triton','transformer_engine','megatron'}:pytest.fail('unexpected real vendor import')
        return original_import(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',importing)
    original_module=p.importlib.import_module
    monkeypatch.setattr(p.importlib,'import_module',lambda name:modules[name] if name in modules else original_module(name))
    for name,module in modules.items():monkeypatch.setitem(sys.modules,name,module)
    versions={'torch':'2.14.0+rocm10.1.0','uccl':'0.1.1','deep_ep':'0.1.0'}
    monkeypatch.setattr(p.importlib.metadata,'version',lambda name:versions[name])
    primary=AssertionError('original upstream test failure')
    def invoke(*a):
        if outcome!='success':raise primary
        return {'entrypoint':'synthetic fixture'}
    monkeypatch.setattr(p,'invoke',invoke)
    original_write=p.write_new
    def writing(path,value):
        if outcome=='evidence_failure' and path.name=='uccl-library-rank-0.json':raise OSError('secondary evidence I/O')
        return original_write(path,value)
    monkeypatch.setattr(p,'write_new',writing)
    if outcome=='success':
        p.run('intranode',settings,root=ASSETS)
        record=json.loads((tmp_path/'uccl-library-rank-0.json').read_text())
        assert record['status']=='passed' and record['selected_device']==0
        assert set(record['origins'])=={'test_intranode','utils','buffer'}
    else:
        with pytest.raises(AssertionError) as caught:p.run('intranode',settings,root=ASSETS)
        assert caught.value is primary
        if outcome=='evidence_failure':assert any('secondary evidence I/O' in note for note in primary.__notes__)
        else:assert json.loads((tmp_path/'uccl-library-rank-0.json').read_text())['status']=='failed'
    assert (tmp_path/'uccl-library-start-0.json').is_file()
    ready=json.loads((tmp_path/'uccl-library-ready-0.json').read_text())
    assert Path(ready['scratch']).is_dir()
    assert ready['installed_buffer']['sha256']=='d30ad5231f476c726c246240eec9b6a2f9124e16752900ca562653623e77a768'
