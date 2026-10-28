"""Unmodified official Forecast timestepping, streaming SMDP trajectory output."""
import argparse,json,time,shutil
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from trajcast.model.forecast import Forecast
from trajcast.data.atomic_graph import AtomicGraph
from trajcast_smdp.data import RUN,DATA
from trajcast_smdp.evaluate import load
from physics.units import KINETIC_TO_EV,KB_EV_PER_K


@torch.no_grad()
def main(run,ensemble,replica):
    run=Path(run);model,saved=load(run)
    cfg=yaml.safe_load((run/'config.yaml').read_text())['smdp']
    tag=f'{ensemble}_{replica}';out=run/tag;out.mkdir(exist_ok=True)
    path=out/'trajectory.h5'
    if path.exists(): raise FileExistsError('Existing autonomous trajectories are never overwritten')
    if ensemble=='nve':
        with h5py.File(DATA/'quartz/transitions/30fs/test.h5') as f:
            source=f['source_paths'][replica]
            if isinstance(source,bytes): source=source.decode()
        warm=0;steps=int(cfg['nve_ps']*1000//30)
    else:
        source=str(DATA/'quartz/physical_reference/nvt_15ns/production.h5')
        warm=round(cfg['nvt_thermalization_ps']*1000/30)
        steps=round(cfg['nvt_production_ps']*1000/30)+warm
    with h5py.File(source) as f:
        q=f['positions_unwrapped_A'][0].astype(float);v=f['velocities_A_per_fs'][0].astype(float)
        cell=f['cell_A'][0];z=f['atomic_numbers'][:];m=f['masses_u'][:]
    graph=AtomicGraph.from_atoms_dict(dict(positions=torch.tensor(q),velocities=torch.tensor(v),
        cell=torch.tensor(cell),pbc=torch.tensor([True]*3),atomic_numbers=torch.tensor(z[:,None],dtype=torch.long),
        atomic_masses=torch.tensor(m[:,None])),4.5,{8:0,14:1})
    seed=2026102800+replica
    engine=Forecast(dict(device=torch.device('cuda'),seed=seed,units='real',run=steps,
        timestep=30.,configuration=graph,model=model,extra_dof=3,temperature=300.,
        thermostat={'Tdamp':cfg['nvt_tau_fs']} if ensemble=='nvt' else False))
    frame=engine.start_graph;qunwrap=torch.tensor(q,device='cuda');nat=len(z);dof=3*nat-3
    masses=torch.tensor(m,device='cuda')
    def temp(vel): return float((masses[:,None]*vel.square()).sum()*KINETIC_TO_EV/(dof*KB_EV_PER_K))
    manifest=dict(ensemble=ensemble,reference_path=source,timestep_fs=30,seed=seed,
        thermostat_tau_fs=cfg['nvt_tau_fs'] if ensemble=='nvt' else None,
        thermalization_steps=warm,requested_steps=steps,completed_steps=0,trajectory_completed=False,
        units=dict(q='Angstrom',v='Angstrom/fs',mass='u'),checkpoint=str(run/'best.pt'),
        decoder_sampling='Neural segment velocities; boundary frames include the actual CSVR impulse in NVT',
        safety_limits=dict(min_pair_A=.65,max_pre_or_post_temperature_K=3000.),
        reference_forces_used_during_propagation=False)
    if ensemble=='nvt' and shutil.disk_usage(out).free<40*1024**3:
        manifest.update(failure_reason='Insufficient disk space: require 40 GiB before dense 15 ns output',
                        elapsed_seconds=0.,actual_duration_ps=0.,ns_day=0.)
        (out/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return
    failure=None;start=time.perf_counter();prop_s=0.;query_s=0.;io_s=0.
    with h5py.File(path,'x') as f:
        f['atomic_numbers']=z;f['masses_u']=m;f['cell']=cell
        f.attrs.update(completed_steps=0,thermalization_steps=warm,ensemble=ensemble,timestep_fs=30.,trajectory_completed=False)
        fields={key:f.create_dataset(key,(steps+1,nat,3),dtype='f4',chunks=(64,nat,3),compression='lzf')
                for key in ['positions','velocities']}
        dense=f.create_dataset('dense/velocities',(steps*30+1,nat,3),dtype='f4',chunks=(120,nat,3),compression='lzf')
        scalars={key:f.create_dataset(key,(steps+1,),dtype='f8') for key in
                 ['temperature_K','pre_thermostat_temperature_K','thermostat_alpha','minimum_distance_A']}
        fields['positions'][0]=q;fields['velocities'][0]=v;dense[0]=v
        scalars['temperature_K'][0]=temp(frame.velocities);scalars['thermostat_alpha'][0]=1.
        for step in range(1,steps+1):
            try:
                if step%1000==1 and shutil.disk_usage(out).free<5*1024**3:
                    raise OSError('Storage safety stop: less than 5 GiB free; no historical data deleted')
                torch.cuda.synchronize();tic=time.perf_counter()
                frame=engine._make_timestep(frame,step)
                torch.cuda.synchronize();prop_s+=time.perf_counter()-tic
                displacement=frame.target[:,:3]*model.rms_targets[0]
                qunwrap=qunwrap+displacement
                pre=temp(frame.target[:,3:]*model.rms_targets[1]);post=temp(frame.velocities)
                frame.compute_edge_vectors()
                mind=float(frame.edge_lengths.min()) if frame.num_edges else float('inf')
                if not torch.isfinite(frame.pos).all() or not torch.isfinite(frame.velocities).all():
                    raise FloatingPointError('nonfinite predicted phase space')
                tic=time.perf_counter();query=model.query(frame,list(range(1,31)))['velocities']
                query[-1]=frame.velocities
                torch.cuda.synchronize();query_s+=time.perf_counter()-tic
                tic=time.perf_counter()
                fields['positions'][step]=qunwrap.cpu().numpy();fields['velocities'][step]=frame.velocities.cpu().numpy()
                dense[(step-1)*30+1:step*30+1]=query.cpu().numpy()
                for key,value in [('temperature_K',post),('pre_thermostat_temperature_K',pre),
                    ('thermostat_alpha',np.sqrt(post/pre) if pre>0 else 1.),('minimum_distance_A',mind)]: scalars[key][step]=value
                f.attrs['completed_steps']=step;manifest['completed_steps']=step;io_s+=time.perf_counter()-tic
                if step%1000==0:
                    f.flush();print(json.dumps(dict(tag=tag,step=step,total=steps,T=post,min_A=mind,
                        elapsed_s=time.perf_counter()-start)),flush=True)
                if mind<.65 or max(pre,post)>3000 or min(pre,post)<1:
                    raise RuntimeError(f'physical safety stop: min={mind}, preT={pre}, postT={post}')
            except Exception as exc:
                failure=repr(exc);break
        manifest.update(trajectory_completed=failure is None,failure_reason=failure,
            elapsed_seconds=time.perf_counter()-start,propagation_seconds=prop_s,query_seconds=query_s,io_seconds=io_s,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),hardware=torch.cuda.get_device_name())
        manifest['actual_duration_ps']=manifest['completed_steps']*.03
        manifest['ns_day']=manifest['completed_steps']*.00003/manifest['elapsed_seconds']*86400
        f.attrs['trajectory_completed']=failure is None;f.attrs['failure_reason']=failure or ''
        f.flush()
    (out/'run.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--ensemble',choices=['nve','nvt'],required=True);p.add_argument('--replica',type=int,default=0)
    a=p.parse_args();main(a.run,a.ensemble,a.replica)
