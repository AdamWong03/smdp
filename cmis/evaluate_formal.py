"""Locked-checkpoint segment, multistep and modal evaluations on full splits."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .cmis_model import oscillatory_primitives
from .formal_data import HDF5Segments,selected_frames,strings
from .inference import load_cmis
from .train_formal import forward_batch,save_json


def rank_and_cosine(gram):
    values=np.linalg.eigvalsh(gram).clip(0);prob=values/max(values.sum(),1e-30)
    denominator=np.sqrt(np.outer(np.diag(gram),np.diag(gram))).clip(1e-30)
    return dict(effective_rank=float(np.exp(-np.sum(prob*np.log(prob.clip(1e-30))))),
                cosine=(gram/denominator).tolist())


@torch.inference_mode()
def segment_metrics(model,ds,device,batch_size,do_modes=True):
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    z=torch.as_tensor(ds.z,device=device,dtype=torch.long);mass=torch.as_tensor(ds.mass,device=device,dtype=torch.float32)
    times=torch.as_tensor(ds.times,device=device,dtype=torch.float32)
    # Dense evaluation uses real .5/1 fs reference queries in a separate pass;
    # formal supervised queries here use the exact cached training contract.
    sums=np.zeros((len(times),4));count=0;species={str(int(k)):np.zeros(4) for k in np.unique(ds.z)}
    gram=np.zeros((model.cfg.modes,model.cfg.modes));attention=np.zeros_like(gram)
    impulse_power=np.zeros(model.cfg.modes);frequency=[[] for _ in range(model.cfg.modes)]
    intervention=np.zeros(model.cfg.modes);intervention_v=np.zeros_like(intervention)
    for raw in loader:
        data={k:v.to(device) for k,v in raw.items()};b,n=data['q0'].shape[:2]
        out=forward_batch(model,data,z,mass,times,return_modes=do_modes)
        q=out['positions'].reshape(len(times),b,n,3).permute(1,0,2,3)
        v=out['velocities'].reshape(len(times),b,n,3).permute(1,0,2,3)
        dq=(q-data['q_target']).double();dv=(v-data['v_target']).double()
        for index in range(len(times)):
            sums[index]+=[float(dq[:,index].square().sum()),float(dv[:,index].square().sum()),
                          float(dq[:,index].abs().sum()),float(dv[:,index].abs().sum())]
        for number in np.unique(ds.z):
            mask=z==int(number);x=dq[:,-1,mask];y=dv[:,-1,mask]
            species[str(int(number))]+=[float(x.square().sum()),float(y.square().sum()),
                                        float(x.abs().sum()),float(y.abs().sum())]
        if not do_modes:
            count+=b
            continue
        k=model.cfg.modes;modal_v=torch.zeros((k,b*n,3),device=device)
        for suffix in ['initial','refined']:
            J=out['mode_impulses_'+suffix];omega=out['mode_omega_fs_'+suffix]
            edge=out['mode_edges_'+suffix];p,_=oscillatory_primitives(times.new_tensor([1.]),omega*model.cfg.horizon_fs)
            edge_dp=torch.einsum('ekb,ekbc->kec',p[0],J)
            modal_v.index_add_(1,edge[0],edge_dp);modal_v.index_add_(1,edge[1],-edge_dp)
            impulse_power+=J.double().square().sum((0,2,3)).cpu().numpy()
            for mode in range(k):frequency[mode].append(omega[:,mode].cpu().numpy())
        modal_v/=mass.repeat(b)[None,:,None]
        flattened=modal_v.double().flatten(1);gram+=(flattened@flattened.T).cpu().numpy()
        attention+=out['mode_attention_second'].double().sum(0).cpu().numpy()
        # Causal deletion recomputes both stages and the candidate midpoint.
        from .vesin_adapter import BatchGraph
        graph=BatchGraph(data['cell'],n,model.cfg.cutoff)
        gid=torch.arange(b,device=device).repeat_interleave(n)
        for mode in range(k):
            active=torch.ones(k,dtype=torch.bool,device=device);active[mode]=False
            removed=model(data['q0'].reshape(-1,3),data['v0'].reshape(-1,3),z.repeat(b),mass.repeat(b),
                times[-1:],graph,graph_index=gid,active_modes=active)
            intervention[mode]+=float((removed['positions'][0]-out['positions'][-1]).double().square().sum())
            intervention_v[mode]+=float((removed['velocities'][0]-out['velocities'][-1]).double().square().sum())
        count+=b
    denom=count*ds.n_atoms*3
    result=dict(samples=count,queries={})
    for t,row in zip(ds.times,sums):
        result['queries'][f'{t:g}']=dict(position_RMSE_A=float(np.sqrt(row[0]/denom)),
            velocity_RMSE_A_fs=float(np.sqrt(row[1]/denom)),position_MAE_A=float(row[2]/denom),
            velocity_MAE_A_fs=float(row[3]/denom))
    result['species_endpoint']={}
    for number,row in species.items():
        d=count*int(np.sum(ds.z==int(number)))*3
        result['species_endpoint'][number]=dict(position_RMSE_A=float(np.sqrt(row[0]/d)),
            velocity_RMSE_A_fs=float(np.sqrt(row[1]/d)),position_MAE_A=float(row[2]/d),velocity_MAE_A_fs=float(row[3]/d))
    if do_modes:result['modes']=dict(endpoint_velocity_contribution=rank_and_cosine(gram),
        impulse_amplitude_squared_share=(impulse_power/max(impulse_power.sum(),1e-30)).tolist(),
        mean_second_attention=(attention/(count*ds.n_atoms)).tolist(),
        deletion_position_RMS_change_A=np.sqrt(intervention/denom).tolist(),
        deletion_velocity_RMS_change_A_fs=np.sqrt(intervention_v/denom).tolist(),
        frequency_rad_fs=[dict(mean=float(np.concatenate(v).mean()),std=float(np.concatenate(v).std()),
                              quantiles=np.quantile(np.concatenate(v),[0,.1,.5,.9,1]).tolist()) for v in frequency],
        interpretation='Deletion effect and contribution covariance do not identify physical phonon modes')
    return result


class DenseEvaluation(HDF5Segments):
    """Evaluation labels at real 1 fs times, including unsupervised query times."""
    def __init__(self,path):
        super().__init__(path)
        self.times=np.arange(1,int(self.horizon)+1,dtype=float)
        with h5py.File(path,'r') as f:source=f.attrs['source_transition']
        self.qt=np.empty((len(self),len(self.times),self.n_atoms,3),dtype='f4');self.vt=np.empty_like(self.qt)
        with h5py.File(source,'r') as tr:
            paths=strings(tr['source_paths'][:]);tid=tr['source_trajectory'][:];frame=tr['source_frame'][:]
            dt=float(tr.attrs['source_dt_fs']);offset=np.rint(self.times/dt).astype(int)
            if not np.allclose(offset*dt,self.times):raise ValueError('Dense reference query not saved')
        for trajectory,path in enumerate(paths):
            ids=np.flatnonzero(tid==trajectory);indices=frame[ids,None]+offset[None,:]
            with h5py.File(path,'r') as raw:
                self.qt[ids]=selected_frames(raw['positions_unwrapped_A'],indices)
                self.vt[ids]=selected_frames(raw['velocities_A_per_fs'],indices)
    def __getitem__(self,index):
        data=super().__getitem__(index)
        data['q_target']=torch.from_numpy(self.qt[index].copy())
        data['v_target']=torch.from_numpy(self.vt[index].copy())
        return data


@torch.inference_mode()
def multistep(model,transition,device,batch_size):
    with h5py.File(transition,'r') as tr:
        q0=tr['positions_initial_A'][:];v0=tr['velocities_initial_A_per_fs'][:];cell=tr['cell_A'][:]
        numbers=tr['atomic_numbers'][:];mass=tr['masses_u'][:];lag=int(tr.attrs['lag_steps'])
        trajectory=tr['source_trajectory'][:];frame=tr['source_frame'][:];paths=strings(tr['source_paths'][:])
    valid=np.ones(len(q0),bool);references={k:(np.zeros_like(q0),np.zeros_like(v0)) for k in [1,2,4,8]}
    for tid,path in enumerate(paths):
        ids=np.flatnonzero(trajectory==tid)
        with h5py.File(path,'r') as raw:
            okay=frame[ids]+8*lag<len(raw['time_fs']);valid[ids[~okay]]=False;ids=ids[okay]
            for horizon in references:
                indices=frame[ids]+horizon*lag
                references[horizon][0][ids]=selected_frames(raw['positions_unwrapped_A'],indices)
                references[horizon][1][ids]=selected_frames(raw['velocities_A_per_fs'],indices)
    ids=np.flatnonzero(valid);total={k:np.zeros(2) for k in references};z=torch.as_tensor(numbers,device=device)
    m=torch.as_tensor(mass,device=device,dtype=torch.float32);times=torch.tensor([model.cfg.horizon_fs],device=device)
    for first in range(0,len(ids),batch_size):
        selected=ids[first:first+batch_size]
        data=dict(q0=torch.as_tensor(q0[selected],device=device),v0=torch.as_tensor(v0[selected],device=device),
                  cell=torch.as_tensor(cell[selected],device=device,dtype=torch.float32))
        for horizon in range(1,9):
            out=forward_batch(model,data,z,m,times)
            data['q0']=out['positions'][0].reshape(len(selected),len(numbers),3)
            data['v0']=out['velocities'][0].reshape_as(data['q0'])
            if horizon in total:
                q,v=references[horizon]
                total[horizon]+=[float(np.sum((data['q0'].cpu().numpy()-q[selected])**2)),
                                 float(np.sum((data['v0'].cpu().numpy()-v[selected])**2))]
    denominator=len(ids)*len(numbers)*3
    return dict(samples=len(ids),horizons={str(k):dict(time_fs=k*model.cfg.horizon_fs,
        position_RMSE_A=float(np.sqrt(v[0]/denominator)),velocity_RMSE_A_fs=float(np.sqrt(v[1]/denominator)))
        for k,v in total.items()})


def query_checks(model,ds,device):
    from .vesin_adapter import VesinGraph
    d={k:v.to(device) for k,v in ds[0].items()}
    z=torch.as_tensor(ds.z,device=device);m=torch.as_tensor(ds.mass,device=device,dtype=torch.float32)
    graph=VesinGraph(d['cell'],model.cfg.cutoff)
    h=model.cfg.horizon_fs;endpoint=d['q0'].new_tensor([h]);dense=torch.arange(1,int(h)+1,device=device,dtype=torch.float32)
    with torch.inference_mode():
        a=model(d['q0'],d['v0'],z,m,endpoint,graph)
        b=model(d['q0'],d['v0'],z,m,dense,graph)
        identity=dict(position_max_abs_A=float(abs(a['positions'][0]-b['positions'][-1]).max()),
                      velocity_max_abs_A_fs=float(abs(a['velocities'][0]-b['velocities'][-1]).max()))
        timings={}
        for name,t in [('endpoint',endpoint),('dense_1fs',dense)]:
            for _ in range(3):model(d['q0'],d['v0'],z,m,t,graph)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(50):model(d['q0'],d['v0'],z,m,t,graph)
            torch.cuda.synchronize();seconds=(time.perf_counter()-start)/50
            timings[name]=dict(mean_seconds=seconds,compute_graph_ns_day=h*1e-6*86400/seconds,
                queries=len(t),includes='two periodic graph rebuilds + neural forward; excludes thermostat/output')
    t=d['q0'].new_tensor([h*.37],requires_grad=True)
    output=model(d['q0'],d['v0'],z,m,t,graph)
    errors=[]
    for atom,axis in [(0,0),(1,1),(2,2)]:
        derivative=torch.autograd.grad(output['positions'][0,atom,axis],t,retain_graph=True)[0][0]
        errors.append(float(abs(derivative-output['velocities'][0,atom,axis]).detach()))
    return dict(endpoint_dense_identity=identity,qdot_v_max_abs_A_fs=max(errors),
                query_timing=timings,hardware=torch.cuda.get_device_name(),shared_gpu=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],required=True);a=p.parse_args()
    cfg=json.loads((a.run/'config.json').read_text());ds=HDF5Segments(Path(cfg['cache'])/(a.split+'.h5'))
    outdir=a.run/'evaluation';outdir.mkdir(exist_ok=True);target=outdir/(a.split+'.json')
    if target.exists():return
    model,saved=load_cmis(a.run/'best.pt');start=time.perf_counter()
    metrics=segment_metrics(model,ds,'cuda',cfg['batch_size'])
    metrics['dense_1fs']=segment_metrics(model,DenseEvaluation(ds.path),'cuda',cfg['batch_size'],do_modes=False)
    transition=Path(cfg['reference_root'])/cfg['system']/f"transitions/{model.cfg.horizon_fs:g}fs/{a.split}.h5"
    metrics['multistep']=multistep(model,transition,'cuda',cfg['batch_size'])
    metrics['query_checks']=query_checks(model,ds,'cuda')
    metrics.update(selected_epoch=saved['epoch']+1,checkpoint=str(a.run/'best.pt'),
                   elapsed_seconds=time.perf_counter()-start,split=a.split)
    save_json(target,metrics);print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
