"""Fixed-final validation prediction and 20 ps NVE; never selects a checkpoint."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from conditional_transition.common import DATA,originals
from conditional_transition.oracle import create_oracle
from physics.units import KINETIC_TO_EV,KB_EV_PER_K
from physics.water_motion import motion_partition
from relational_inference import load_relational
from smdpr.train import write_json
from train_relational import evaluate
from dynamics_diagnosis.common import ARTIFACTS
from dynamics_diagnosis.experiment import errors


def analyze(path,rawpath,first_frame,system,output):
    if output.exists():return
    calc,cell,z,m,verification=create_oracle(system)
    rows=[]
    try:
        with h5py.File(path) as candidate,h5py.File(rawpath) as raw:
            stop=int(candidate.attrs['completed_steps'])
            dt=float(candidate.attrs['timestep_fs']);stride=round(dt/float(raw.attrs['output_interval_fs']))
            selected=np.unique(np.r_[np.linspace(0,stop,min(1000,stop+1),dtype=int),
                [min(stop,round(t*1000/dt)) for t in [1,5,10,20]]])
            mol=raw['molecule_ids'][:] if system=='water' else None
            for index in selected:
                q=candidate['positions'][index].astype(float);v=candidate['velocities'][index].astype(float)
                ri=first_frame+int(index)*stride
                rq=raw['positions_unwrapped_A'][ri].astype(float);rv=raw['velocities_A_per_fs'][ri].astype(float)
                ke=lambda u:float(.5*KINETIC_TO_EV*np.sum(m[:,None]*u*u))
                ck,rk=ke(v),ke(rv);cp,rp=calc.evaluate(q)['pe'],calc.evaluate(rq)['pe']
                row=dict(time_ps=index*dt/1000,temperature_K=2*ck/((3*len(m)-3)*KB_EV_PER_K),
                    reference_temperature_K=2*rk/((3*len(m)-3)*KB_EV_PER_K),
                    KE_eV=ck,reference_KE_eV=rk,PE_eV=cp,reference_PE_eV=rp,
                    total_E_eV=ck+cp,reference_total_E_eV=rk+rp,
                    position_RMSE_A=float(np.sqrt(np.mean((q-rq)**2))),
                    velocity_RMSE_A_fs=float(np.sqrt(np.mean((v-rv)**2))))
                if mol is not None:
                    motion=motion_partition(q,v,cell,z,m,mol);ref=motion_partition(rq,rv,cell,z,m,mol)
                    for key in ['translation_eV','rotation_eV','internal_eV']:
                        row[key]=float(motion[key]);row['reference_'+key]=float(ref[key])
                rows.append(row)
    finally:calc.close()
    df=pd.DataFrame(rows);df.to_csv(output.with_suffix('.csv'),index=False)
    record=dict(completed=True,trajectory=str(path),initial_source=str(rawpath),initial_frame=first_frame,
        final_time_ps=float(df.time_ps.iloc[-1]),mean_temperature_K=float(df.temperature_K.mean()),
        final_temperature_K=float(df.temperature_K.iloc[-1]),
        reference_mean_temperature_K=float(df.reference_temperature_K.mean()),
        energy_change_eV=float(df.total_E_eV.iloc[-1]-df.total_E_eV.iloc[0]),
        reference_energy_change_eV=float(df.reference_total_E_eV.iloc[-1]-df.reference_total_E_eV.iloc[0]),
        checkpoint_selection=False)
    if system=='water':
        record['motion']={key:dict(mean=float(df[key].mean()),reference_mean=float(df['reference_'+key].mean()))
                          for key in ['translation_eV','rotation_eV','internal_eV']}
    write_json(output,record)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run;cfg=json.loads((run/'config.json').read_text());system=cfg['system']
    output=run/'evaluation';output.mkdir(exist_ok=True)
    cp=run/'step_25000.pt'
    model,saved=load_relational(cp)
    torch.backends.cuda.matmul.allow_tf32=False
    dest=output/'validation_endpoint.json'
    if not dest.exists():
        graphs=originals(saved['config'],'val')
        result=evaluate(model,DataLoader(graphs,batch_size=8),torch.device('cuda'))
        write_json(dest,dict(checkpoint=str(cp),fixed_optimizer_steps=25000,**result))
    # Reuse frozen evaluation-only banks, never their states as training labels.
    calc,cell,z,m,verification=create_oracle(system)
    try:
        with h5py.File(DATA/system/'raw_nve'/('val_07' if system=='quartz' else 'val_04')/'trajectory.h5') as raw:
            mol=raw['molecule_ids'][:] if system=='water' else None
        for bank_name in ['reference','r3opt']:
            bank=ARTIFACTS/system/f'bank_{bank_name}.h5'
            with h5py.File(bank) as f:
                assert f.attrs['completed']
                assert 'evaluation' in f.attrs['purpose']
            errors(system,cfg['arm'],model,bank,calc,cell,z,m,mol,
                destination=output/f'local_{bank_name}.csv')
    finally:calc.close()
    del model;torch.cuda.empty_cache()
    multistep=output/'validation_multistep.json'
    horizon=30 if system=='quartz' else 5
    if not multistep.exists():
        with (output/'multistep.log').open('a') as log:
            subprocess.run([sys.executable,'-u','evaluate_relational_rollout.py','--checkpoint',str(cp),
                '--transitions',str(DATA/system/'transitions'/f'{horizon}fs/val.h5'),
                '--output',str(multistep)],stdout=log,stderr=subprocess.STDOUT,check=True)
    initials=[('val_07',0),('val_08',0)] if system=='quartz' else [('val_04',0),('val_04',10000)]
    for replica,(source,frame) in enumerate(initials):
        directory=output/f'nve20ps_{replica}';directory.mkdir(exist_ok=True)
        path=directory/'trajectory.h5';raw=DATA/system/'raw_nve'/source/'trajectory.h5'
        if not path.with_suffix('.summary.json').exists():
            if path.exists():raise RuntimeError('Interrupted trajectory retained; no silent rerun/overwrite')
            with (directory/'rollout.log').open('a') as log:
                subprocess.run([sys.executable,'-u','rollout_relational.py','--checkpoint',str(cp),
                    '--initial',str(raw),'--initial-frame',str(frame),'--output',str(path),
                    '--ensemble','nve','--steps',str(round(20000/horizon)),
                    '--execution-note','controlled learning experiment; shared GPU, no inference oracle'],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        analyze(path,raw,frame,system,directory/'physics.json')
    write_json(run/'evaluation_status.json',dict(validation_physics_completed=True,
        checkpoint=str(cp),fixed_final_step=25000,test_accessed=False,long_NVT_started=False))


if __name__=='__main__':main()
