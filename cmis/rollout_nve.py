"""Minimal force-free NVE autoregressive rollout with HDF5 streaming.

This code does NOT compute potential energy, RDF, VDOS or thermostat states;
run the project's validated reference-MD analyzers separately. For NVT,
reuse the already-audited CSVR thermostat in the smdp repository.
"""
import argparse
import json
import time

import numpy as np
import torch

try:
    from .cmis_model import CMIS, CMISConfig
    from .vesin_adapter import VesinGraph
except ImportError:
    from cmis_model import CMIS, CMISConfig
    from vesin_adapter import VesinGraph

K_B_EV_PER_K=8.617333262145e-5
AMU_AFS2_TO_EV=1.66053906660e-27*(1e5**2)/1.602176634e-19


def kinetic_temperature(v, mass):
    com=(mass[:,None]*v).sum(0)/mass.sum()
    v0=v-com
    energy=0.5*AMU_AFS2_TO_EV*(mass[:,None]*v0.square()).sum()
    dof=max(1,3*len(mass)-3)
    return float(2*energy/(dof*K_B_EV_PER_K))


@torch.no_grad()
def simulate(model,q,v,z,mass,cell,steps,output_h5,save_every=1,stop_tmin=20.,stop_tmax=2000.):
    import h5py
    model.eval()
    graph=VesinGraph(cell,model.cfg.cutoff)
    n=len(q);h=model.cfg.horizon_fs
    device=q.device
    q=q.clone();v=v.clone()
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    sync();begin=time.perf_counter()
    written=0;reason=None
    with h5py.File(output_h5,'x') as f:
        f.attrs.update({'horizon_fs':h,'length_unit':'Angstrom',
                        'velocity_unit':'Angstrom/fs','training_forces_required':False})
        qq=f.create_dataset('q_unwrapped',(0,n,3),maxshape=(None,n,3),chunks=(1,n,3),dtype='f4')
        vv=f.create_dataset('v',(0,n,3),maxshape=(None,n,3),chunks=(1,n,3),dtype='f4')
        times=f.create_dataset('time_fs',(0,),maxshape=(None,),chunks=True,dtype='f8')
        temp=f.create_dataset('kinetic_temperature_K',(0,),maxshape=(None,),chunks=True,dtype='f4')
        def save(step):
            nonlocal written
            for arr,data in ((qq,q),(vv,v)):
                arr.resize((written+1,n,3))
                arr[written]=data.detach().cpu().numpy()
            times.resize((written+1,));times[written]=step*h
            temp.resize((written+1,));temp[written]=kinetic_temperature(v,mass)
            written+=1
        save(0)
        for t in range(1,steps+1):
            output=model(q,v,z,mass,torch.tensor([h],device=device,dtype=q.dtype),graph)
            q_next=output['positions'][0];v_next=output['velocities'][0]
            if not torch.isfinite(q_next).all() or not torch.isfinite(v_next).all():
                reason=f'nonfinite_state_at_step_{t}';break
            temp_new=kinetic_temperature(v_next,mass)
            if temp_new < stop_tmin or temp_new > stop_tmax:
                q,v=q_next,v_next
                save(t)
                reason=f'temperature_safety_stop_at_step_{t}_T_{temp_new:.3f}'
                break
            q,v=q_next,v_next
            if t%save_every==0 or t==steps:save(t)
        if reason is not None:f.attrs['stop_reason']=reason
    sync();elapsed=time.perf_counter()-begin
    return {'completed_steps':t,'requested_steps':steps,'written_frames':written,
            'wallclock_s':elapsed,'nanoseconds_per_day':(t*h/1e6)/elapsed*86400,
            'stop_reason':reason}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--initial-state',required=True,help='NPZ keys q0,v0,z,mass,cell (single system)')
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--save-every',type=int,default=1)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=CMIS(CMISConfig(**ck['meta']['config']))
    model.load_state_dict(ck['state_dict']);model=model.to(a.device)
    data=np.load(a.initial_state,allow_pickle=False)
    q=torch.as_tensor(data['q0'],device=a.device,dtype=torch.float32)
    v=torch.as_tensor(data['v0'],device=a.device,dtype=torch.float32)
    z=torch.as_tensor(data['z'],device=a.device,dtype=torch.long)
    mass=torch.as_tensor(data['mass'],device=a.device,dtype=torch.float32)
    cell=torch.as_tensor(data['cell'],device=a.device,dtype=torch.float32)
    stats=simulate(model,q,v,z,mass,cell,a.steps,a.out,a.save_every)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
