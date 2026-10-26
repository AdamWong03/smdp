"""Relabel model-visited TRAIN states with their own BKS futures, never old futures."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from neural_segment.train import load_model
from neural_segment.labels import short_reference, atoms_state
from physics.bks import make_calculator, evaluate
from physics.units import KINETIC_TO_EV
from periodic import PeriodicGraphBuilder, predict_batch
from reference_data import QuartzSegments
from rollout_quartz import temperature


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--shared',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--round',type=int,choices=[1,2],required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite previously sampled states')
    torch.set_num_threads(4);model,_=load_model(args.checkpoint,args.device)
    data=QuartzSegments(args.data_root,'train',args.shared/'cache')
    rng=np.random.default_rng(20261026+args.round)
    starts=rng.choice(len(data),size=500,replace=False)
    ck=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    builder=PeriodicGraphBuilder(4.5);near=PeriodicGraphBuilder(3.)
    z=data.z.numpy();m=data.m.numpy();calc=make_calculator()
    kept=0;records=[]
    with h5py.File(args.output,'x') as f:
        f.attrs.update(completed=False,split='train',checkpoint_sha256=ck,checkpoint=str(args.checkpoint),
                       round=args.round,maximum_labels=2000,selection='500 fixed random training starts; states at 64,128,192,256 fs')
        shapes=dict(q=(17,len(m),3),v=(17,len(m),3),a=(5,len(m),3),cell=(3,3),potential_eV=(17,))
        arrays={k:f.create_dataset(k,shape=(0,*s),maxshape=(None,*s),chunks=(1,*s),dtype='f4' if k in ('q','v','a') else 'f8') for k,s in shapes.items()}
        for idx in starts:
            q=data.q[idx,0].to(args.device)[None];v=data.v[idx,0].to(args.device)[None]
            cell=data.cells[idx].numpy();cs=data.cells[idx].to(args.device)[None]
            zs=data.z.to(args.device)[None];ms=data.m.to(args.device)[None]
            for step in range(1,17):
                with torch.inference_mode():
                    o=predict_batch(model,q,v,zs,ms,cs,[16.],builder)
                    q=o['positions'][0][None];v=o['velocities'][0][None]
                qn=q[0].cpu().numpy().astype(float);vn=v[0].cpu().numpy().astype(float)
                finite=bool(np.isfinite(qn).all() and np.isfinite(vn).all())
                mind=None;temp=None
                if finite:
                    edges,shifts=near.one(qn,cell)
                    mind=float(np.linalg.norm(qn[edges[0]]+shifts-qn[edges[1]],axis=-1).min()) if edges.shape[1] else 3.
                    temp=float(temperature(vn,m))
                reason=None if finite and mind>=.8 and 20<=temp<=2000 else 'invalid model state'
                if step%4!=0 and reason is None:continue
                record=dict(training_index=int(idx),source_path=data.sources['paths'][data.sources['trajectories'][idx]],
                    source_frame=data.sources['frames'][idx],autonomous_time_fs=step*16,checkpoint_sha256=ck,
                    minimum_A=mind,temperature_K=temp,accepted=False,rejection_reason=reason)
                if reason is None:
                    try:
                        atoms=atoms_state(qn,vn,z,m,cell,calc);pe=evaluate(atoms,calc)[2]
                        record['initial_BKS_total_eV']=float(pe+.5*KINETIC_TO_EV*np.sum(m[:,None]*vn**2))
                        qs,vs,acc,potential=short_reference(qn,vn,z,m,cell,calc)
                        minima=[]
                        for qt in qs:
                            ee,ss=near.one(qt,cell)
                            minima.append(float(np.linalg.norm(qt[ee[0]]+ss-qt[ee[1]],axis=-1).min()))
                        temps=[temperature(vt,m) for vt in vs]
                        if not all(np.isfinite(x).all() for x in (qs,vs,acc,potential)) or min(minima)<.8 or min(temps)<20 or max(temps)>2000:
                            record['rejection_reason']='BKS short trajectory unstable'
                        else:
                            for key,value in zip(('q','v','a','cell','potential_eV'),(qs,vs,acc,cell,potential)):
                                arrays[key].resize(kept+1,axis=0);arrays[key][kept]=value
                            record.update(accepted=True,label_index=kept,reference_minimum_A=min(minima))
                            kept+=1
                    except Exception as error:
                        # Do not turn environment/software errors into silently rejected physics.
                        record['rejection_reason']=f'label_execution_error: {type(error).__name__}: {error}'
                        records.append(record)
                        f.create_dataset('failed_provenance_json',data=np.array([json.dumps(r) for r in records],dtype=h5py.string_dtype()))
                        raise
                if not record['accepted']:
                    group=f.require_group('rejected').create_group(str(len(records)))
                    group.create_dataset('positions',data=qn);group.create_dataset('velocities',data=vn)
                records.append(record)
                if reason is not None:break
            f.flush()
            if len(records)%100<4:print(json.dumps(dict(candidates=len(records),accepted=kept)),flush=True)
        f.create_dataset('provenance_json',data=np.array([json.dumps(r) for r in records],dtype=h5py.string_dtype()))
        f.attrs.update(completed=True,accepted=kept,rejected=sum(not r['accepted'] for r in records))
    calc.lmp.close()
    args.output.with_suffix('.json').write_text(json.dumps(dict(checkpoint_sha256=ck,accepted=kept,
        rejected=sum(not r['accepted'] for r in records),candidates=len(records),records=records),indent=2))


if __name__=='__main__':main()
