"""Two prescribed rounds; all oracle labels start from the supplied state itself."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

from conditional_transition.common import DATA,ROOT,train_sources,transitions
from conditional_transition.oracle import create_oracle
from periodic import PeriodicGraphBuilder
from physics.units import KINETIC_TO_EV,KB_EV_PER_K
from relational_inference import load_relational,vesin_graph
from smdpr.train import write_json


def temperature(v,m):
    return float(KINETIC_TO_EV*np.sum(m[:,None]*np.asarray(v,dtype=float)**2)/((3*len(m)-3)*KB_EV_PER_K))


def provenance(system):
    sources=train_sources(system)
    for split in ['val','test']:
        with h5py.File(transitions(system,split)) as f:
            others=[p.decode() if isinstance(p,bytes) else str(p) for p in f['source_paths'][:]]
            if set(sources)&set(others):raise ValueError('Source split leakage')
    if not all(Path(p).parent.name.startswith('train_') for p in sources):
        raise ValueError('Non-training source')
    return sources


def autonomous_inputs(system,model,checkpoint,round_index,directory,sources):
    """Same 125 real starts and depth schedule in both collection rounds."""
    dest=directory/'visited_states.h5'
    if dest.exists():
        with h5py.File(dest) as f:
            if f.attrs.get('completed'):return dest
        raise RuntimeError('Partial visited-state pool retained; inspect explicitly before restart')
    rng=np.random.default_rng((20261102 if system=='quartz' else 20261103)+500)
    with h5py.File(transitions(system)) as tr:
        source_id=tr['source_trajectory'][:].astype(int);source_frame=tr['source_frame'][:]
        start_ids=[]
        for k in range(len(sources)):
            count=125//len(sources)+(k<125%len(sources))
            start_ids.extend(rng.choice(np.flatnonzero(source_id==k),count,replace=False).tolist())
        q0=tr['positions_initial_A'][:];v0=tr['velocities_initial_A_per_fs'][:];cells=tr['cell_A'][:]
        z,m=tr['atomic_numbers'][:],tr['masses_u'][:]
    depth_lists=[sorted(sum([rng.choice(np.arange(a,b+1),4,replace=False).tolist()
                 for a,b in [(1,4),(5,16),(17,64),(65,256)]],[])) for _ in start_ids]
    builder=PeriodicGraphBuilder(model.cfg.cutoff_angstrom);near=PeriodicGraphBuilder(3.)
    rows=[];failures=[];neural_steps=0
    start_time=time.perf_counter()
    for begin in range(0,len(start_ids),8):
        ids=start_ids[begin:begin+8]
        q=q0[ids].astype(np.float32);v=v0[ids].astype(np.float32)
        alive=np.ones(len(ids),bool)
        for depth in range(1,257):
            local=np.flatnonzero(alive)
            if not len(local):break
            graphs=[vesin_graph(q[j],v[j],cells[ids[j]],z,m,model,builder,'cpu') for j in local]
            with torch.inference_mode():out=model(Batch.from_data_list(graphs).cuda())
            q[local]=out['positions'].cpu().numpy().reshape(len(local),len(z),3)
            v[local]=out['velocities'].cpu().numpy().reshape(len(local),len(z),3)
            neural_steps+=len(local)
            for j in local:
                finite=bool(np.isfinite(q[j]).all() and np.isfinite(v[j]).all())
                temp=temperature(v[j],m) if finite else float('nan')
                minimum=float('nan')
                if finite:
                    e,s=near.one(q[j],cells[ids[j]])
                    minimum=float(np.linalg.norm(q[j][e[0]]-q[j][e[1]]+s,axis=1).min())
                if not finite or minimum<.65 or temp<1 or temp>3000:
                    failures.append(dict(start_id=int(ids[j]),depth=depth,temperature_K=temp,
                        min_distance_A=minimum,reason='existing numerical safety rule',q=q[j].tolist(),v=v[j].tolist()))
                    alive[j]=False;continue
                if depth in depth_lists[begin+j]:
                    rows.append((q[j].astype(float),v[j].astype(float),cells[ids[j]],
                        int(source_id[ids[j]]),int(source_frame[ids[j]]),int(ids[j]),depth,temp,minimum))
        write_json(directory/'collection_progress.json',dict(state='collecting',round=round_index,
            starts_completed=min(begin+8,len(start_ids)),total_starts=125,states=len(rows),
            neural_macro_steps=neural_steps,failed_rollouts=len(failures)))
        print(json.dumps(dict(starts=min(begin+8,125),states=len(rows),neural_macro_steps=neural_steps)),flush=True)
    with h5py.File(dest,'x') as f:
        f.attrs.update(completed=False,system=system,split='train',round=round_index,
            collector_checkpoint=str(checkpoint),purpose='training candidate states',
            sources=json.dumps(sources),neural_macro_steps=neural_steps,
            generation_seconds=time.perf_counter()-start_time)
        f['z']=z;f['mass']=m
        for column,name in enumerate(['q','v','cell','source_id','source_frame','start_id','depth','temperature_K','min_distance_A']):
            f[name]=np.asarray([r[column] for r in rows])
        failed=f.create_group('failed_rollouts')
        for i,record in enumerate(failures):
            g=failed.create_group(str(i));g['q']=record.pop('q');g['v']=record.pop('v')
            g.attrs['details']=json.dumps(record)
        f.attrs.update(completed=True,requested_states=2000,valid_states=len(rows),failed_rollouts=len(failures))
    return dest


def real_inputs(system,round_index,directory,sources):
    dest=directory/'visited_states.h5'
    if dest.exists():
        with h5py.File(dest) as f:
            if f.attrs.get('completed'):return dest
        raise RuntimeError('Partial reference-state pool retained')
    rng=np.random.default_rng((20261102 if system=='quartz' else 20261103)+800)
    with h5py.File(transitions(system)) as tr:
        original=[set(tr['source_frame'][:][tr['source_trajectory'][:]==i].tolist()) for i in range(len(sources))]
        z,m=tr['atomic_numbers'][:],tr['masses_u'][:]
        lag=int(tr.attrs['lag_steps'])
    rows=[]
    quotas=[4000//len(sources)+(k<4000%len(sources)) for k in range(len(sources))]
    first_quotas=[n//2 for n in quotas]
    remainder=2000-sum(first_quotas)
    for k,n in enumerate(quotas):
        if remainder and n%2:
            first_quotas[k]+=1;remainder-=1
    assert sum(first_quotas)==2000
    for k,path in enumerate(sources):
        count=quotas[k]
        with h5py.File(path) as raw:
            available=np.array([i for i in range(0,len(raw['time_fs'])-lag,lag) if i not in original[k]])
            chosen=rng.choice(available,count,replace=False)
            first=first_quotas[k]
            chosen=np.sort(chosen[:first] if round_index==1 else chosen[first:])
            for frame in chosen:
                rows.append((raw['positions_unwrapped_A'][frame].astype(float),
                    raw['velocities_A_per_fs'][frame].astype(float),raw['cell_A'][frame],k,int(frame)))
    assert len(rows)==2000
    with h5py.File(dest,'x') as f:
        f.attrs.update(completed=False,system=system,split='train',round=round_index,
                       sources=json.dumps(sources),collector_checkpoint='real training reference')
        f['z']=z;f['mass']=m
        for column,name in enumerate(['q','v','cell','source_id','source_frame']):
            f[name]=np.asarray([r[column] for r in rows])
        f.attrs.update(completed=True,requested_states=2000,valid_states=len(rows),neural_macro_steps=0)
    return dest


def label(system,inputs,directory):
    output=directory/'labels.h5'
    if output.exists():
        with h5py.File(output) as f:
            if f.attrs.get('completed'):return output
        raise RuntimeError('Partial oracle labels retained; inspect before recovery')
    calc,cell,z,m,verification=create_oracle(system)
    write_json(directory/'training_reference_verification.json',verification)
    h=30 if system=='quartz' else 5;dt=1 if system=='quartz' else .5
    start=time.perf_counter();accepted=[];failed=[]
    try:
        with h5py.File(inputs) as src,h5py.File(output,'x') as dst:
            np.testing.assert_array_equal(src['z'][:],z)
            np.testing.assert_allclose(src['mass'][:],m,rtol=0,atol=1e-8)
            dst.attrs.update(completed=False,split='train',system=system,
                candidate_pool=str(inputs),sources=src.attrs['sources'],horizon_fs=h,
                collector_checkpoint=src.attrs['collector_checkpoint'],round=src.attrs['round'],
                energy_convention='unchanged raw LAMMPS eV',purpose='true conditional successor training labels')
            dst['z']=z;dst['mass']=m
            for i in range(len(src['q'])):
                q,v=src['q'][i],src['v'][i]
                np.testing.assert_allclose(src['cell'][i],cell,atol=1e-7)
                try:
                    pe0=calc.evaluate(q)['pe'];rq,rv=calc.short_trajectory(q,v,dt_fs=dt,steps=round(h/dt))
                    pe1=calc.evaluate(rq)['pe']
                    if not np.isfinite(rq).all() or not np.isfinite(rv).all() or not np.isfinite(pe1):
                        raise FloatingPointError('Nonfinite reference successor')
                    accepted.append((i,rq,rv,pe0,pe1))
                except Exception as error:
                    failed.append(dict(candidate=i,reason=f'{type(error).__name__}: {error}'))
                    dst.attrs['failure_reason']=json.dumps(failed)
                    dst.attrs['attempted_queries']=i+1
                    dst.attrs['successful_queries']=len(accepted)
                    write_json(directory/'oracle_failure.json',dict(failures=failed,
                        candidate_pool=str(inputs),attempted_queries=i+1,successful_queries=len(accepted)))
                    # A broken LAMMPS instance must not be silently reused.
                    raise RuntimeError(f'Oracle failed; candidate {i} retained in {inputs}: {error}') from error
                if (i+1)%250==0:print(json.dumps(dict(labeled=i+1,total=len(src['q']))),flush=True)
            ids=np.array([r[0] for r in accepted])
            for name in ['q','v','cell','source_id','source_frame','start_id','depth','temperature_K','min_distance_A']:
                if name in src:dst[name]=src[name][ids]
            dst['next_q']=np.array([r[1] for r in accepted]);dst['next_v']=np.array([r[2] for r in accepted])
            dst['initial_PE_eV']=[r[3] for r in accepted];dst['next_PE_eV']=[r[4] for r in accepted]
            kinetic=lambda vel:.5*KINETIC_TO_EV*np.sum(m[None,:,None]*vel**2,axis=(1,2))
            dst['initial_KE_eV']=kinetic(dst['v'][:])
            dst['next_KE_eV']=kinetic(dst['next_v'][:])
            dst['initial_total_E_eV']=dst['initial_KE_eV'][:]+dst['initial_PE_eV'][:]
            dst['next_total_E_eV']=dst['next_KE_eV'][:]+dst['next_PE_eV'][:]
            dst['candidate_index']=ids
            dst.attrs.update(completed=True,attempted_queries=len(src['q']),successful_queries=len(accepted),
                reference_micro_steps=len(src['q'])*round(h/dt),oracle_seconds=time.perf_counter()-start,
                failures=json.dumps(failed),reference_dt_fs=dt,
                verification_continuations=3,verification_micro_steps=3*round(h/dt),
                kinetic_mass_convention='unchanged HDF5 evaluation masses; integrator masses in verification JSON')
    finally:calc.close()
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--round',type=int,choices=[1,2],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads((args.run/'config.json').read_text())
    directory=args.run/f'collection_round{args.round}';directory.mkdir(exist_ok=True)
    sources=provenance(cfg['system'])
    if cfg['arm']=='autonomous':
        model,_=load_relational(args.checkpoint);torch.backends.cuda.matmul.allow_tf32=False
        candidates=autonomous_inputs(cfg['system'],model,args.checkpoint,args.round,directory,sources)
        del model;torch.cuda.empty_cache()
    elif cfg['arm']=='extra_reference':
        candidates=real_inputs(cfg['system'],args.round,directory,sources)
    else:raise ValueError('Only data-aggregation branches collect labels')
    labels=label(cfg['system'],candidates,directory)
    with h5py.File(labels) as f:
        write_json(directory/'status.json',dict(state='collection_completed',labels=str(labels),
            round=args.round,valid_labels=len(f['q']),reference_micro_steps=int(f.attrs['reference_micro_steps'])))


if __name__=='__main__':main()
