"""Convert official Forecast output and invoke the preserved quartz evaluator.

No potential is used during neural forecasting. BKS is offline in the existing
validated MPICH environment, not installed into the new TrajCast environment.
"""
import argparse
import json
import os
import subprocess

import ase.io
import h5py
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from trajcast_based.acquire import ROOT, RUN


def main(tag):
    out=RUN/tag
    meta=json.loads((out/'run.json').read_text())
    assert meta['trajectory_completed']
    target=out/'trajectory.h5'
    if not target.exists():
        expected=meta['production_steps']+meta['warmup_steps']+1
        with h5py.File(target,'x') as f:
            f.attrs['conversion_completed']=False
            f.attrs['timestep_fs']=30.
            for i, atoms in enumerate(ase.io.iread(out/'trajectory.extxyz',index=':')):
                if i==0:
                    f['numbers']=atoms.numbers; f['cell']=atoms.cell.array; f['masses']=atoms.get_masses()
                    q=f.create_dataset('positions',(expected,len(atoms),3),dtype='f8',chunks=(64,len(atoms),3),compression='lzf')
                    v=f.create_dataset('velocities',q.shape,dtype='f8',chunks=q.chunks,compression='lzf')
                np.testing.assert_array_equal(atoms.numbers, f['numbers'][:])
                np.testing.assert_allclose(atoms.cell.array, f['cell'][:],atol=1e-12)
                q[i]=atoms.positions;v[i]=atoms.arrays['velocities']
            assert i+1==expected
            f.attrs['conversion_completed']=True
    with h5py.File(target) as f: assert f.attrs['conversion_completed']
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'src')
    env['LD_LIBRARY_PATH']='/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib'
    env['OMP_NUM_THREADS']='4'
    env['OPENBLAS_NUM_THREADS']='1'
    subprocess.run(['/data1/wyd/envs/mlip/bin/python',str(ROOT/'src/evaluate_quartz_trajectory.py'),
        str(target),str(out/'physics'), '--dt-fs','30','--discard-frames',str(meta['warmup_steps']+1),
        '--evaluate-bks','--paper-vdos',str(ROOT/'data/trajcast_quartz/paper_source/fig3_vdos.csv'),
        '--paper-energy',str(ROOT/'data/trajcast_quartz/paper_source/fig3_energy.csv')],check=True,env=env,cwd=ROOT)
    # Do not hide a potential-definition mismatch by only reporting centered distributions.
    energy=pd.read_csv(out/'physics.energy.csv')
    paper=pd.read_csv(ROOT/'data/trajcast_quartz/paper_source/fig3_energy.csv')
    comparison={'warning':'Absolute differences are recorded, not assumed to be a constant zero-point shift.'}
    for field in energy:
        values=energy[field].dropna().to_numpy()
        comparison[field]={'mean_eV':float(values.mean()),'std_eV':float(values.std())}
        for ref in paper:
            refvalues=paper[ref].dropna().to_numpy()
            comparison[field][ref]={'mean_eV':float(refvalues.mean()),'std_eV':float(refvalues.std()),
                'mean_difference_eV':float(values.mean()-refvalues.mean()),
                'uncentered_wasserstein_eV':float(wasserstein_distance(values,refvalues))}
    (out/'energy_conventions.json').write_text(json.dumps(comparison,indent=2)+'\n')
    meta['physical_analysis_completed']=True
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',default='official30_100ps')
    main(p.parse_args().tag)
