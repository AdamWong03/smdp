"""Common-cadence 1 fs spectra from saved CMIS interior velocities."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physics.statistics import spectrum_blocks,stats
from cmis.train_formal import save_json


def main():
    p=argparse.ArgumentParser()
    for name in ['candidate','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/'dense_vdos.json'
    if target.exists() and json.loads(target.read_text()).get('completed'):return
    with h5py.File(a.candidate,'r') as c,h5py.File(a.reference,'r') as r:
        h=float(c.attrs['timestep_fs']);start=int(c.attrs['thermalization_steps'])*int(h)
        completed=int(c.attrs['completed_steps'])*int(h)
        if completed<=start:
            save_json(target,dict(completed=False,reason='No complete production segment'));return
        z=c['atomic_numbers'][:];rs=round(1/float(r.attrs['output_interval_fs']))
        frames=min(completed-start+1,(len(r['time_fs'])-1)//rs+1)
        cand=spectrum_blocks(c['dense/velocities'],z,start,frames,1,1.,cache=a.output/'dense_spectral/model')
        ref=spectrum_blocks(r['velocities_A_per_fs'],z,0,frames,rs,1.,cache=a.output/'dense_spectral/reference')
    if cand is None or ref is None:
        save_json(target,dict(completed=False,reason='Trajectory too short for the reference spectral convention'));return
    frequency,cp=cand;_,rp=ref;metrics=dict(cadence_fs=1.,nyquist_THz=float(frequency[-1]),elements={})
    fig,axes=plt.subplots(len(cp),1,figsize=(8,3*len(cp)));rows=[]
    for ax,z in zip(np.atleast_1d(axes),sorted(cp)):
        cm,rm=cp[z].mean(0),rp[z].mean(0)
        metrics['elements'][str(z)]=dict(overlap=float(np.minimum(cm,rm).sum()),
            block_overlap=stats(np.minimum(cp[z],rp[z]).sum(1)))
        ax.plot(frequency,cm,label='CMIS queried 1 fs');ax.plot(frequency,rm,label='Reference 1 fs')
        ax.set(xlabel='THz',xlim=(0,160 if z in [1,8] else 60),title=f'Z={z}');ax.legend()
        rows.extend(dict(element=int(z),frequency_THz=float(f),model=float(u),reference=float(v))
                    for f,u,v in zip(frequency,cm,rm))
    fig.tight_layout();fig.savefig(a.output/'dense_vdos.png',dpi=160);plt.close(fig)
    pd.DataFrame(rows).to_csv(a.output/'dense_vdos.csv',index=False)
    save_json(target,dict(completed=True,metrics=metrics,
        convention='Neural interior queries with committed post-CSVR endpoint velocities at segment boundaries'))


if __name__=='__main__':main()
