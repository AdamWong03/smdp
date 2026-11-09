"""Frozen state-transfer diagnosis on common validation initial states.

No optimizer, teacher correction, checkpoint selection or production training.
True continuations are stored separately as evaluation-only labels.
"""
import argparse
import json
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from dynamics_diagnosis.common import ARTIFACTS, DATA, REPORT, SEED, checkpoints, save, source, initial_sources
from dynamics_diagnosis.reference import calculator,validate_integrator
from periodic import PeriodicGraphBuilder
from physics.units import KINETIC_TO_EV,KB_EV_PER_K
from physics.water_motion import motion_partition, signed_kinetic_error
from relational_inference import load_relational,vesin_graph

WINDOWS=[(0,1),(1,5),(5,20),(20,50)]


def ke(v,m):
    return .5*KINETIC_TO_EV*np.sum(m[...,None]*np.asarray(v,dtype=float)**2,axis=(-2,-1))


def temp(v,m):
    return 2*ke(v,m)/((3*len(m)-3)*KB_EV_PER_K)


@torch.inference_mode()
def predict(model,q,v,cell,z,m,builder):
    graphs=[vesin_graph(x,y,cell,z,m,model,builder,'cpu') for x,y in zip(q,v)]
    out=model(Batch.from_data_list(graphs).cuda())
    return (out['positions'].cpu().numpy().reshape(np.shape(q)),
            out['velocities'].cpu().numpy().reshape(np.shape(v)))


def autonomous(system,name,model,cell,z,m):
    directory=ARTIFACTS/system/name
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'validation_50ps.h5'
    if path.exists():
        with h5py.File(path) as f:
            if f.attrs.get('completed',False):return path
        raise RuntimeError(f'Unfinished diagnostic trajectory retained: {path}')
    originals=initial_sources(system)
    q,v=[],[]
    for src,frame in originals:
        with h5py.File(src) as f:
            np.testing.assert_allclose(f['cell_A'][frame],cell,atol=1e-7)
            q.append(f['positions_unwrapped_A'][frame]);v.append(f['velocities_A_per_fs'][frame])
    q,v=np.array(q,dtype=np.float32),np.array(v,dtype=np.float32)
    h=model.cfg.horizon_fs;n=int(np.ceil(50000/h))
    builder=PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    start=time.perf_counter()
    with h5py.File(path,'x') as f:
        f.attrs.update(completed=False,macro_dt_fs=h,purpose='frozen-model validation diagnosis',
            initial_sources=json.dumps([(str(p),i) for p,i in originals]),
            forces_during_inference=False,thermostat=False)
        f['cell']=cell;f['z']=z;f['mass']=m;f['time_fs']=np.arange(n+1)*h
        qs=f.create_dataset('q',(n+1,2,len(z),3),dtype='f4',chunks=(64,2,len(z),3),compression='lzf')
        vs=f.create_dataset('v',(n+1,2,len(z),3),dtype='f4',chunks=(64,2,len(z),3),compression='lzf')
        qs[0],vs[0]=q,v
        for i in range(1,n+1):
            q,v=predict(model,q,v,cell,z,m,builder)
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                f.attrs['failure']=f'nonfinite state at macro step {i}'
                raise FloatingPointError(f'{name}: nonfinite dynamics')
            qs[i],vs[i]=q,v
            if i%1000==0:
                f.flush()
                print(dict(event='autonomous',system=system,model=name,step=i,total=n),flush=True)
        f.attrs.update(completed=True,elapsed_seconds=time.perf_counter()-start)
    return path


def series(system,name,path,calc,z,m,cell,mol):
    dest=ARTIFACTS/system/name/'time_series.csv'
    if dest.exists():return
    rows=[]
    h=30 if system=='quartz' else 5
    with h5py.File(path) as f:
        # Same cadence for every candidate in a system, plus window boundaries.
        stride=1 if system=='quartz' else 5
        selected=np.unique(np.r_[np.arange(0,len(f['time_fs']),stride),len(f['time_fs'])-1,
                      [round(t*1000/h) for t in [1,5,10,20]]])
        for replica,(raw_path,initial) in enumerate(initial_sources(system)):
            with h5py.File(raw_path) as raw:
                lag=round(h/float(raw.attrs['output_interval_fs']))
                for index in selected:
                    q,v=f['q'][index,replica].astype(float),f['v'][index,replica].astype(float)
                    row=dict(system=system,model=name,replica=replica,time_ps=float(f['time_fs'][index]/1000),
                             temperature_K=float(temp(v,m)),KE_eV=float(ke(v,m)),PE_eV=calc.evaluate(q)['pe'])
                    row['total_E_eV']=row['KE_eV']+row['PE_eV']
                    for element in np.unique(z):
                        row[f'Z{element}_KE_eV']=float(ke(v[z==element],m[z==element]))
                    if mol is not None:
                        motion=motion_partition(q,v,cell,z,m,mol)
                        for key in ['translation_eV','rotation_eV','internal_eV']:
                            row[key]=float(motion[key])
                        row.update(bond_mean_A=float(motion['bond_lengths_A'].mean()),
                            bond_std_A=float(motion['bond_lengths_A'].std()),
                            angle_mean_deg=float(motion['angles_deg'].mean()),angle_std_deg=float(motion['angles_deg'].std()))
                    ri=initial+int(index)*lag
                    if ri<len(raw['time_fs']):
                        rq=raw['positions_unwrapped_A'][ri].astype(float)
                        rv=raw['velocities_A_per_fs'][ri].astype(float)
                        row.update(reference_temperature_K=float(temp(rv,m)),
                            reference_KE_eV=float(ke(rv,m)),reference_PE_eV=calc.evaluate(rq)['pe'],
                            position_RMSE_A=float(np.sqrt(np.mean((q-rq)**2))),
                            velocity_RMSE_A_fs=float(np.sqrt(np.mean((v-rv)**2))))
                        row['reference_total_E_eV']=row['reference_KE_eV']+row['reference_PE_eV']
                        if mol is not None:
                            ref=motion_partition(rq,rv,cell,z,m,mol)
                            for key in ['translation_eV','rotation_eV','internal_eV']:row['reference_'+key]=float(ref[key])
                    rows.append(row)
    pd.DataFrame(rows).to_csv(dest,index=False)


def state_bank(system,name,path,calc,cell):
    dest=ARTIFACTS/system/f'bank_{name}.h5'
    if dest.exists():
        with h5py.File(dest) as f:
            if f.attrs.get('completed'):return dest
        raise RuntimeError(f'Partial diagnostic bank: {dest}')
    rng=np.random.default_rng(SEED)
    states=[]
    originals=initial_sources(system)
    horizon=30 if system=='quartz' else 5
    if name=='reference':
        for replica,(rawpath,initial) in enumerate(originals):
            with h5py.File(rawpath) as f:
                dt=float(f.attrs['output_interval_fs']);lag=round(horizon/dt)
                for a,b in WINDOWS:
                    candidates=np.flatnonzero((f['time_fs'][:]/1000>=a)&(f['time_fs'][:]/1000<b))
                    candidates=candidates[candidates+lag<len(f['time_fs'])]
                    chosen=np.sort(rng.choice(candidates,32,replace=False))
                    for frame in chosen:
                        states.append((replica,f'{a}-{b}',int(frame),float(f['time_fs'][frame]/1000),
                            f['positions_unwrapped_A'][frame],f['velocities_A_per_fs'][frame]))
    else:
        with h5py.File(path) as f:
            times=f['time_fs'][:]/1000
            for replica in range(2):
                for a,b in WINDOWS:
                    candidates=np.flatnonzero((times>=a)&(times<b))
                    chosen=np.sort(rng.choice(candidates,32,replace=False))
                    for frame in chosen:
                        states.append((replica,f'{a}-{b}',int(frame),float(times[frame]),
                                      f['q'][frame,replica].astype(float),f['v'][frame,replica].astype(float)))
    dt=1. if system=='quartz' else .5
    with h5py.File(dest,'x') as f:
        f.attrs.update(completed=False,purpose='evaluation-only; prohibited as training data',
                       source_model=name,horizon_fs=horizon,dt_fs=dt,seed=SEED)
        f['replica']=[s[0] for s in states]
        f['window']=np.array([s[1].encode() for s in states])
        f['frame']=[s[2] for s in states];f['time_ps']=[s[3] for s in states]
        f['q']=np.array([s[4] for s in states]);f['v']=np.array([s[5] for s in states])
        qtrue=f.create_dataset('true_q',f['q'].shape,dtype='f8')
        vtrue=f.create_dataset('true_v',f['q'].shape,dtype='f8')
        pe0=f.create_dataset('initial_PE',(len(states),),dtype='f8')
        pet=f.create_dataset('true_PE',(len(states),),dtype='f8')
        for i,(_,_,_,_,q,v) in enumerate(states):
            pe0[i]=calc.evaluate(q)['pe']
            rq,rv=calc.short_trajectory(q,v,dt_fs=dt,steps=round(horizon/dt))
            qtrue[i],vtrue[i]=rq,rv
            pet[i]=calc.evaluate(rq)['pe']
        f.attrs['completed']=True
    return dest


def errors(system,name,model,bank_path,calc,cell,z,m,mol):
    bank_name=bank_path.stem.replace('bank_','')
    dest=ARTIFACTS/system/name/f'local_{bank_name}.csv'
    if dest.exists():return
    builder=PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    rows=[]
    with h5py.File(bank_path) as f:
        for begin in range(0,len(f['q']),8):
            q,v=f['q'][begin:begin+8],f['v'][begin:begin+8]
            pq,pv=predict(model,q,v,cell,z,m,builder)
            for j in range(len(q)):
                i=begin+j;tq,tv=f['true_q'][i],f['true_v'][i]
                initialke,trueke,modelke=map(float,[ke(v[j],m),ke(tv,m),ke(pv[j],m)])
                ep,ei,et=calc.evaluate(pq[j])['pe'],float(f['initial_PE'][i]),float(f['true_PE'][i])
                cross,quadratic=signed_kinetic_error(pv[j],tv,m)
                neighbor,shift=builder.one(q[j],cell)
                distance=np.linalg.norm(q[j][neighbor[0]]-q[j][neighbor[1]]+shift,axis=1)
                near=PeriodicGraphBuilder(3.);edges,shifts=near.one(q[j],cell)
                row=dict(system=system,model=name,state_source=bank_name,
                    origin='reference' if bank_name=='reference' else 'autonomous',
                    own_state=bank_name==name,replica=int(f['replica'][i]),window=f['window'][i].decode(),
                    frame=int(f['frame'][i]),time_ps=float(f['time_ps'][i]),
                    temperature_K=float(temp(v[j],m)),min_distance_A=float(np.linalg.norm(
                        q[j][edges[0]]-q[j][edges[1]]+shifts,axis=1).min()),
                    mean_neighbor_count=float(len(distance)/len(z)),
                    mean_edge_distance_A=float(distance.mean()),
                    q_RMSE_A=float(np.sqrt(np.mean((pq[j]-tq)**2))),
                    v_RMSE_A_fs=float(np.sqrt(np.mean((pv[j]-tv)**2))),
                    q_signed_mean_A=float((pq[j]-tq).mean()),v_signed_mean_A_fs=float((pv[j]-tv).mean()),
                    q_error_velocity_projection=float(np.sum((pq[j]-tq)*tv)/np.maximum(np.sum(tv*tv),1e-30)),
                    v_error_velocity_projection=float(np.sum((pv[j]-tv)*tv)/np.maximum(np.sum(tv*tv),1e-30)),
                    initial_KE_eV=initialke,initial_PE_eV=ei,
                    model_delta_KE_eV=modelke-initialke,MD_delta_KE_eV=trueke-initialke,
                    model_delta_PE_eV=ep-ei,MD_delta_PE_eV=et-ei,
                    model_delta_E_eV=ep+modelke-ei-initialke,
                    MD_delta_E_eV=et+trueke-ei-initialke,
                    signed_energy_discrepancy_eV=ep+modelke-et-trueke,
                    kinetic_error_cross_eV=float(cross),kinetic_error_quadratic_eV=float(quadratic))
                for axis,label in enumerate(['x','y','z']):
                    row['q_signed_'+label+'_A']=float((pq[j]-tq)[:,axis].mean())
                    row['v_signed_'+label+'_A_fs']=float((pv[j]-tv)[:,axis].mean())
                for element in np.unique(z):
                    select=z==element
                    row[f'Z{element}_model_delta_KE_eV']=float(ke(pv[j][select],m[select])-ke(v[j][select],m[select]))
                    row[f'Z{element}_MD_delta_KE_eV']=float(ke(tv[select],m[select])-ke(v[j][select],m[select]))
                if mol is not None:
                    motions=[motion_partition(np.asarray(x,dtype=float),np.asarray(y,dtype=float),cell,z,m,mol)
                             for x,y in [(q[j],v[j]),(tq,tv),(pq[j],pv[j])]]
                    for component in ['translation_eV','rotation_eV','internal_eV']:
                        row['initial_'+component]=float(motions[0][component])
                        row['model_delta_'+component]=float(motions[2][component]-motions[0][component])
                        row['MD_delta_'+component]=float(motions[1][component]-motions[0][component])
                        row['error_'+component]=float(motions[2][component]-motions[1][component])
                    row.update(bond_mean_A=float(motions[0]['bond_lengths_A'].mean()),
                        bond_std_A=float(motions[0]['bond_lengths_A'].std()),
                        angle_mean_deg=float(motions[0]['angles_deg'].mean()),
                        angle_std_deg=float(motions[0]['angles_deg'].std()))
                np.testing.assert_allclose(row['model_delta_E_eV']-row['MD_delta_E_eV'],
                                           row['signed_energy_discrepancy_eV'],atol=1e-9)
                rows.append(row)
    pd.DataFrame(rows).to_csv(dest,index=False)


def roundtrip(system,name,model,bank,cell,z,m):
    dest=ARTIFACTS/system/name/'roundtrip.json'
    if dest.exists():return
    builder=PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    with h5py.File(bank) as f:
        q,v=f['q'][:64],f['v'][:64]
    errors=[]
    for start in range(0,len(q),8):
        x,y=q[start:start+8],v[start:start+8]
        a,b=predict(model,x,y,cell,z,m,builder)
        aa,bb=predict(model,a,-b,cell,z,m,builder)
        errors.append([float(((aa-x)**2).mean()),float(((-bb-y)**2).mean())])
    save(dest,dict(q_RMSE_A=float(np.sqrt(np.mean(errors,axis=0)[0])),
        v_RMSE_A_fs=float(np.sqrt(np.mean(errors,axis=0)[1])),samples=len(q),
        interpretation='time-reversal round trip; not an energy conservation test'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--system',choices=['quartz','water'],required=True)
    parser.add_argument('--models',nargs='+',choices=list(checkpoints('water')))
    args=parser.parse_args();system=args.system
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ARTIFACTS.joinpath(system).mkdir(parents=True,exist_ok=True)
    with h5py.File(source(system)) as raw:
        cell,z,m=raw['cell_A'][0],raw['atomic_numbers'][:],raw['masses_u'][:]
        mol=raw['molecule_ids'][:] if system=='water' else None
        calc=calculator(system,raw)
        verification_path=REPORT/f'{system}_reference_verification.json'
        if args.models and verification_path.exists():
            verify=json.loads(verification_path.read_text())
        else:
            verify=validate_integrator(system,raw,calc)
            save(verification_path,verify)
        if not verify['passed']:raise RuntimeError('Original reference continuation mismatch')
    try:
        refbank=state_bank(system,'reference',None,calc,cell)
        models=checkpoints(system)
        # Shared k3_stage2 bank first: all predictors see exactly these same
        # autonomous inputs, in addition to their own trajectory's states.
        order=args.models or ['k3_stage2','r3opt','r3opt_two_step','k3_stage1','k1_stage1','k1_stage2']
        for name in order:
            model,_=load_relational(models[name])
            path=autonomous(system,name,model,cell,z,m)
            series(system,name,path,calc,z,m,cell,mol)
            bank=state_bank(system,name,path,calc,cell)
            errors(system,name,model,refbank,calc,cell,z,m,mol)
            errors(system,name,model,bank,calc,cell,z,m,mol)
            if name!='k3_stage2':
                errors(system,name,model,ARTIFACTS/system/'bank_k3_stage2.h5',calc,cell,z,m,mol)
            roundtrip(system,name,model,refbank,cell,z,m)
            save(ARTIFACTS/system/name/'completed.json',dict(completed=True,model=name,system=system,
                  checkpoint=str(models[name]),reference_trajectories='validation only',labels_used_for_training=False))
            print(dict(event='model_diagnosis_completed',system=system,model=name),flush=True)
            del model
        finished=[name for name in models if (ARTIFACTS/system/name/'completed.json').exists()]
        if len(finished)==len(models):
            save(ARTIFACTS/system/'status.json',dict(state='diagnosis_completed',models=finished))
    finally:calc.close()


if __name__=='__main__':main()
