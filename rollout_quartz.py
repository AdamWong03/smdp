"""Autonomous segment rollout, streaming output, no reference-force corrections."""
import argparse
import json
import math
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from periodic import PeriodicGraphBuilder, predict_batch
from evaluate_quartz import load_model
from train_quartz import diagnostics
from physics.units import KINETIC_TO_EV, KB_EV_PER_K


def temperature(v,m):
    return KINETIC_TO_EV*np.sum(m[:,None]*v*v)/((3*len(m)-3)*KB_EV_PER_K)


class CSVR:
    """Exact stochastic KE rescaling, preserving the initial COM velocity.

    Applied at macrostep endpoints, h=16 fs, tau from actual reference input.
    This is not identical to the reference's 1 fs thermostat discretization.
    """
    def __init__(self,m,dt,tau,seed):
        self.m=m; self.dof=3*len(m)-3; self.c=math.exp(-dt/tau)
        self.rng=np.random.default_rng(seed)

    def __call__(self,v):
        com=(v*self.m[:,None]).sum(0)/self.m.sum()
        internal=v-com
        ke=.5*KINETIC_TO_EV*np.sum(self.m[:,None]*internal**2)
        if not math.isfinite(ke) or ke<=0: raise FloatingPointError('Invalid pre-thermostat kinetic energy')
        r=self.rng.normal(); chi=self.rng.chisquare(self.dof-1)
        ratio=(1-self.c)*(.5*KB_EV_PER_K*300)/ke
        a2=self.c+ratio*(r*r+chi)+2*r*math.sqrt(self.c*ratio)
        alpha=math.copysign(math.sqrt(max(a2,0.)),math.sqrt(self.c)+math.sqrt(ratio)*r)
        return com+alpha*internal,alpha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ensemble',choices=['nve','nvt'],required=True)
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--thermalization-steps',type=int,default=0)
    p.add_argument('--thermostat-metadata',type=Path)
    p.add_argument('--seed',type=int,default=2026102201)
    p.add_argument('--device',default='cuda')
    args=p.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device(args.device)
    model,cfg=load_model(args.checkpoint,device)
    with h5py.File(args.initial) as f:
        q0=f['positions_unwrapped_A'][0].astype(np.float32);v0=f['velocities_A_per_fs'][0].astype(np.float32)
        cell=f['cell_A'][0]; z=f['atomic_numbers'][:];m=f['masses_u'][:]
    # Preserve initial COM; the model conserves it separately for every graph.
    q=torch.tensor(q0,device=device)[None];v=torch.tensor(v0,device=device)[None]
    cs=torch.tensor(cell,device=device,dtype=torch.float32)[None]
    zs=torch.tensor(z,device=device)[None];ms=torch.tensor(m,device=device,dtype=torch.float32)[None]
    builder=PeriodicGraphBuilder(model.config.cutoff_angstrom)
    # A short-distance linked-cell list is sufficient to detect collisions and
    # gives the exact minimum in quartz (reference nearest distance ~1.5 A).
    near=PeriodicGraphBuilder(3.)
    thermo=None;metadata={}
    if args.ensemble=='nvt':
        if args.thermostat_metadata is None: raise ValueError('Reference thermostat metadata required')
        metadata=json.loads(args.thermostat_metadata.read_text())
        assert metadata['ensemble']=='NVT-CSVR' and metadata['temperature_K']==300
        thermo=CSVR(m,16.,metadata['thermostat_tau_fs'],args.seed)
    f=h5py.File(args.output,'w')
    f.attrs.update(ensemble=args.ensemble,timestep_fs=16.,requested_steps=args.steps,thermalization_steps=args.thermalization_steps,
                   checkpoint=str(args.checkpoint),initial_trajectory=str(args.initial),dense_sample_dt_fs=1.,
                   dense_endpoint_velocity='post-CSVR at macro boundaries; pre-CSVR in NVE',thermostat_metadata=json.dumps(metadata))
    f.create_dataset('atomic_numbers',data=z);f.create_dataset('masses_u',data=m);f.create_dataset('cell',data=cell)
    def ds(name,shape,dtype='f4'):
        return f.create_dataset(name,shape=(0,*shape),maxshape=(None,*shape),chunks=(256,*shape),dtype=dtype,compression='lzf')
    stores={k:ds(k,(len(z),3)) for k in ['positions','velocities']}
    for k in ['time_fs','temperature_K','pre_thermostat_temperature_K','minimum_distance_A','thermostat_alpha','segment_join_derivative_RMS_A_fs']:
        stores[k]=ds(k,())
    dense=ds('dense/velocities',(len(z),3))
    buffers={k:[] for k in stores}; dense_buffer=[]
    def append(qn,vn,tm,pre,minimum,alpha,join):
        values=[qn,vn,tm,temperature(vn,m),pre,minimum,alpha,join]
        for key,value in zip(stores,values): buffers[key].append(value)
    def flush():
        for k,d in stores.items():
            if buffers[k]:
                a=np.asarray(buffers[k]); old=len(d);d.resize(old+len(a),axis=0);d[old:]=a;buffers[k].clear()
        if dense_buffer:
            a=np.concatenate(dense_buffer);old=len(dense);dense.resize(old+len(a),axis=0);dense[old:]=a;dense_buffer.clear()
        f.flush()
    def minimum(qn):
        e,s=near.one(qn,cell)
        if e.shape[1]==0: return 3.  # documented lower bound when no short pair exists
        return float(np.linalg.norm(qn[e[0]]+s-qn[e[1]],axis=1).min())
    append(q0,v0,0.,temperature(v0,m),minimum(q0),1.,0.)
    dense_buffer.append(v0[None]);flush()
    completed=0;failure=None;model_seconds=0.;graph_seconds=0.;stats={};last_left=None
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
    start=time.perf_counter()
    try:
        with torch.inference_mode():
            for step in range(1,args.thermalization_steps+args.steps+1):
                t=time.perf_counter(); edges,shifts,graphs=builder.batch(q,cs);graph_seconds+=time.perf_counter()-t
                if device.type=='cuda': torch.cuda.synchronize(device)
                t=time.perf_counter()
                out=model(q[0],v[0],zs[0],ms[0],edges,shifts,list(range(1,17)),graph_index=graphs)
                if device.type=='cuda': torch.cuda.synchronize(device)
                model_seconds+=time.perf_counter()-t
                segment_q=out['positions'].cpu().numpy();segment_v=out['velocities'].cpu().numpy()
                if not np.isfinite(segment_q).all() or not np.isfinite(segment_v).all():
                    failure='nonfinite segment prediction'
                    f.create_dataset('failure/positions',data=segment_q);f.create_dataset('failure/velocities',data=segment_v)
                    break
                qn=segment_q[-1];vn=segment_v[-1].copy();pre=temperature(vn,m);mind=minimum(qn)
                right=segment_q[0]-q[0].cpu().numpy()
                join=0. if last_left is None else float(np.sqrt(np.mean((right-last_left)**2)))
                last_left=segment_q[-1]-segment_q[-2]
                # Check before thermostat: no cooling of a divergent model step.
                if pre>2000 or pre<20 or mind<.8:
                    failure=f'pre-thermostat invalid state: T={pre:.6g} K, minimum={mind:.6g} A'
                alpha=1.
                if thermo is not None and failure is None: vn,alpha=thermo(vn)
                segment_v[-1]=vn
                append(qn,vn,step*16.,pre,mind,alpha,join);dense_buffer.append(segment_v)
                completed=step
                for k,x in diagnostics(out).items(): stats[k]=stats.get(k,0.)+np.array(x)
                q=torch.tensor(qn,device=device)[None];v=torch.tensor(vn,dtype=torch.float32,device=device)[None]
                if step%256==0 or failure:
                    flush();f.attrs['completed_steps']=completed
                    print(json.dumps(dict(step=step,elapsed_s=time.perf_counter()-start,temperature_K=float(pre),minimum_A=mind,failure=failure)),flush=True)
                if failure: break
    except Exception as error:
        failure=f'{type(error).__name__}: {error}'
        raise
    finally:
        flush();elapsed=time.perf_counter()-start
        f.attrs.update(completed_steps=completed,stable=failure is None,failure_reason=failure or '')
        f.close()
        result=dict(completed_steps=completed,production_steps=max(0,completed-args.thermalization_steps),
                    simulated_ps=completed*.016,production_ps=max(0,completed-args.thermalization_steps)*.016,
                    requested_production_ps=args.steps*.016,stable=failure is None,failure_reason=failure,
                    elapsed_s=elapsed,model_seconds=model_seconds,neighbor_seconds=graph_seconds,
                    ns_day_end_to_end=completed*16e-6/max(elapsed,1e-9)*86400,
                    ns_day_model_only_with_16_queries=completed*16e-6/max(model_seconds,1e-9)*86400,
                    ms_per_step=1000*elapsed/max(completed,1),peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                    hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                    parameters=sum(p.numel() for p in model.parameters()),
                    mode_statistics={k:(x/max(completed,1)).tolist() for k,x in stats.items()},
                    min_distance_convention='exact if <3 A; otherwise lower bound 3 A')
        args.output.with_suffix('.summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__': main()
