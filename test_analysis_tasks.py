"""Software tests of failure isolation and result status, not MD experiments."""
import json
from types import SimpleNamespace
import h5py
import pytest
import numpy as np
import analysis_tasks
from analyze_quartz import spectrum_blocks
from summarize_experiment import compare_runs


def test_failed_energy_does_not_block_other_tasks_and_resume(tmp_path,monkeypatch):
    for name in ['candidate','reference']:
        with h5py.File(tmp_path/(name+'.h5'),'w') as f:
            f.attrs.update(completed_steps=1,thermalization_steps=0,requested_steps=1,stable=True,failure_reason='')
    monkeypatch.setattr(analysis_tasks,'Context',lambda c,r,out:SimpleNamespace(completed=1,production=True,common=2))
    def failure(x):raise OSError('test missing MPI')
    calls=[]
    def thermo(x):calls.append(1);return {'temperature_K':{'test_value':300}}
    monkeypatch.setattr(analysis_tasks,'TASKS',{'energy':failure,'thermo':thermo})
    argv=['analysis','--candidate',str(tmp_path/'candidate.h5'),'--reference',str(tmp_path/'reference.h5'),'--output',str(tmp_path/'out')]
    monkeypatch.setattr('sys.argv',argv)
    with pytest.raises(SystemExit):analysis_tasks.main()
    result=json.loads((tmp_path/'out/metrics.json').read_text())
    assert result['trajectory_completed'] and not result['physical_analysis_completed']
    assert result['temperature_K']['test_value']==300
    monkeypatch.setattr(analysis_tasks,'TASKS',{'energy':lambda x:{'energy_change_eV':0},'thermo':thermo})
    analysis_tasks.main()
    result=json.loads((tmp_path/'out/metrics.json').read_text())
    assert result['physical_analysis_completed'] and len(calls)==1


def test_missing_analysis_is_not_missing_trajectory(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';out=tmp_path/'report'
    for p in [old,new,out]:p.mkdir()
    (new/'nvt.summary.json').write_text(json.dumps(dict(production_ps=15000,requested_production_ps=15000)))
    compare_runs(old,new,report_output=out)
    state=json.loads((out/'run_status.json').read_text())['nvt']
    assert state['trajectory_completed'] is True
    assert state['physical_analysis_completed'] is False
    (new/'nvt_analysis').mkdir()
    (new/'nvt_analysis/metrics.json').write_text(json.dumps({'production_available':False}))
    compare_runs(old,new,report_output=out)
    state=json.loads((out/'run_status.json').read_text())['nvt']
    assert state['trajectory_completed'] is True
    assert state['physical_analysis_completed'] is False


def test_spectral_cache_preserves_uncached_definition(tmp_path):
    # Software equality test only; not a new dynamics experiment.
    values=np.random.default_rng(4).normal(size=(2048,2,3))
    numbers=np.array([8,14])
    expected=spectrum_blocks(values,numbers,0,len(values),1,16.)
    observed=spectrum_blocks(values,numbers,0,len(values),1,16.,cache=tmp_path/'spectra')
    class Unreadable:
        def __getitem__(self,key):raise AssertionError('Cached blocks should not reread input')
    resumed=spectrum_blocks(Unreadable(),numbers,0,len(values),1,16.,cache=tmp_path/'spectra')
    np.testing.assert_array_equal(expected[0],observed[0])
    for z in numbers:
        np.testing.assert_array_equal(expected[1][z],observed[1][z])
        np.testing.assert_array_equal(expected[1][z],resumed[1][z])


def test_parallel_spectral_blocks_match_serial(tmp_path):
    values=np.random.default_rng(5).normal(size=(2048,2,3));numbers=np.array([8,14])
    with h5py.File(tmp_path/'source.h5','w') as f:f.create_dataset('v',data=values)
    expected=spectrum_blocks(values,numbers,0,len(values),1,16.)
    with h5py.File(tmp_path/'source.h5','r') as f:
        actual=spectrum_blocks(f['v'],numbers,0,len(values),1,16.,cache=tmp_path/'parallel',workers=2)
    np.testing.assert_array_equal(expected[0],actual[0])
    for z in numbers:np.testing.assert_array_equal(expected[1][z],actual[1][z])
