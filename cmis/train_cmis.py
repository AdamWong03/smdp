"""Minimal genuine training entrypoint for prepared *independent-NVE* segments.

Adapt existing raw HDF5 in smdp_reference_md_v1 to NPZ before using.
This script does NOT infer HDF5 field names, units, splits or trajectories.

Input NPZ keys:
 q0 [S,N,3], v0 [S,N,3], q_target [S,T,N,3],
 v_target [S,T,N,3], times_fs [T], z [N], mass [N],
 cell [3,3] (or cell [S,3,3]).
q must be unwrapped; v in A/fs; mass in amu. Targets from raw NVE.
Train and validation NPZ MUST be constructed from independent trajectories.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from cmis_model import CMIS, CMISConfig, small_cell_periodic_graph
from vesin_adapter import BatchGraph


class NPZSegments(Dataset):
    def __init__(self,path):
        f=np.load(path,mmap_mode='r',allow_pickle=False)
        must=['q0','v0','q_target','v_target','times_fs','z','mass','cell']
        missing=[k for k in must if k not in f]
        if missing:raise ValueError(f'{path}: missing keys: {missing}')
        self.a={k:np.asarray(f[k]) for k in must}
        a=self.a
        S,N,xyz=a['q0'].shape
        assert xyz==3 and a['v0'].shape==(S,N,3)
        assert a['q_target'].shape==(S,len(a['times_fs']),N,3)
        assert a['v_target'].shape==a['q_target'].shape
        assert a['z'].shape==(N,) and a['mass'].shape==(N,)
        assert a['cell'].shape in ((3,3),(S,3,3))
        assert (np.diff(a['times_fs'])>0).all()
        assert np.all(a['mass']>0)
        self.sharedcell=a['cell'].shape==(3,3)

    def __len__(self):return len(self.a['q0'])

    def __getitem__(self,i):
        a=self.a
        d={k:torch.from_numpy(np.array(a[k][i],copy=True)).float()
           for k in ('q0','v0','q_target','v_target')}
        d['cell']=torch.from_numpy(np.array(a['cell'] if self.sharedcell else a['cell'][i],copy=True)).float()
        return d


def mass_weighted_scales(ds):
    a=ds.a;m=a['mass'].astype(np.float64)
    # scales are computed on TRAIN only, including all available segment timepoints
    qt=np.asarray(a['q_target'],dtype=np.float64)
    vt=np.asarray(a['v_target'],dtype=np.float64)
    q0=np.asarray(a['q0'],dtype=np.float64)
    v0=np.asarray(a['v0'],dtype=np.float64)
    t=a['times_fs'].astype(np.float64)
    dq=qt-q0[:,None,:,:]-t[None,:,None,None]*v0[:,None,:,:]
    dv=vt-v0[:,None,:,:]
    w=m[None,None,:,None]
    sq=np.sqrt((w*dq*dq).sum()/(len(ds)*len(t)*3*m.sum()))
    sv=np.sqrt((w*dv*dv).sum()/(len(ds)*len(t)*3*m.sum()))
    return float(max(sq,1e-7)),float(max(sv,1e-9))


def supervised_loss(out,d, m,t,scale_q,scale_v,endpoint_weight=2.0):
    w=(m/m.sum())[None,:,None]
    lq=(w*((out['positions']-d['q_target'])/scale_q)**2).sum(dim=(1,2))/3
    lv=(w*((out['velocities']-d['v_target'])/scale_v)**2).sum(dim=(1,2))/3
    weights=torch.ones_like(lq); weights[-1]=endpoint_weight
    weights=weights/weights.sum()
    return (weights*(lq+lv)).sum(),(weights*lq).sum(),(weights*lv).sum()


def one_epoch(model,ds,optimizer,device,sq,sv,grad_accum,train,seed,
              debug_small_cell=False,batch_size=8):
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=train,
                      generator=generator,num_workers=0,pin_memory=False)
    if train:
        optimizer.zero_grad(set_to_none=True)
    z0=torch.as_tensor(ds.a['z'],device=device,dtype=torch.long)
    m0=torch.as_tensor(ds.a['mass'],device=device,dtype=torch.float32)
    times=torch.as_tensor(ds.a['times_fs'],device=device,dtype=torch.float32)
    accum={'total':0.,'pos':0.,'vel':0.}
    count=0
    for step,raw in enumerate(loader):
        d={k:v.to(device) for k,v in raw.items()}
        B,N=d['q0'].shape[:2]
        if train:
            reverse=torch.rand(B,device=device)<0.5
            q0,dq=d['q0'],d['q_target']
            v0,dv=d['v0'],d['v_target']
            q_rev=torch.cat([q0[:,None],dq[:,:-1]],dim=1).flip(dims=[1])
            v_rev=-torch.cat([v0[:,None],dv[:,:-1]],dim=1).flip(dims=[1])
            d['q0']=torch.where(reverse[:,None,None],dq[:,-1],q0)
            d['v0']=torch.where(reverse[:,None,None],-dv[:,-1],v0)
            d['q_target']=torch.where(reverse[:,None,None,None],q_rev,dq)
            d['v_target']=torch.where(reverse[:,None,None,None],v_rev,dv)
        graph=BatchGraph(d['cell'],N,model.cfg.cutoff,debug_small_cell)
        q=d['q0'].reshape(B*N,3)
        v=d['v0'].reshape(B*N,3)
        z=z0.repeat(B)
        m=m0.repeat(B)
        gid=torch.arange(B,device=device).repeat_interleave(N)
        targets={
            'q_target':d['q_target'].permute(1,0,2,3).reshape(len(times),B*N,3),
            'v_target':d['v_target'].permute(1,0,2,3).reshape(len(times),B*N,3),
        }
        if train:
            out=model(q,v,z,m,times,graph,graph_index=gid)
            loss,lq,lv=supervised_loss(out,targets,m,times,sq,sv)
            (loss/grad_accum).backward()
            if (step+1)%grad_accum==0 or (step+1)==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                out=model(q,v,z,m,times,graph,graph_index=gid)
                loss,lq,lv=supervised_loss(out,targets,m,times,sq,sv)
        for key,x in [('total',loss),('pos',lq),('vel',lv)]:
            accum[key]+=float(x.detach())*B
        count+=B
    return {k:v/count for k,v in accum.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True);p.add_argument('--val',required=True)
    p.add_argument('--out',required=True);p.add_argument('--horizon',type=float,required=True)
    p.add_argument('--epochs',type=int,default=150);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--grad-accum',type=int,default=1)
    p.add_argument('--debug-small-cell',action='store_true',help='ONLY for toy tests with cutoff below half the box height')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--width',type=int,default=96);p.add_argument('--modes',type=int,default=4)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    torch.manual_seed(20261106);np.random.seed(20261106)
    tr=NPZSegments(args.train);va=NPZSegments(args.val)
    for ds in (tr,va):
        ts=ds.a['times_fs']
        if abs(float(ts[-1])-args.horizon)>1e-5:raise ValueError('times_fs[-1] != --horizon')
    # Reversal implementation assumes t_i=(i+1)*h/T (equal spacings).
    ts=tr.a['times_fs']; step=np.diff(np.r_[0,ts])
    if not np.allclose(step,step[0],rtol=1e-5,atol=1e-5):
        raise ValueError('Time reversal augmentation requires uniformly spaced times_fs')
    if not np.array_equal(tr.a['z'],va.a['z']) or not np.allclose(tr.a['mass'],va.a['mass']):
        raise ValueError('Train/val species or mass mismatch')
    sq,sv=mass_weighted_scales(tr)
    cfg=CMISConfig(width=args.width,modes=args.modes,horizon_fs=args.horizon)
    model=CMIS(cfg).to(args.device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-6)
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=False)
    meta={'config':dataclasses.asdict(cfg),'scale_q':sq,'scale_v':sv,
          'train_npz':args.train,'val_npz':args.val,'seed':20261106}
    (outdir/'config.json').write_text(json.dumps(meta,indent=2))
    best=float('inf')
    with (outdir/'history.jsonl').open('w') as log:
        for ep in range(args.epochs):
            model.train();train_metrics=one_epoch(model,tr,optim,args.device,sq,sv,args.grad_accum,True,20261106+ep,args.debug_small_cell,args.batch_size)
            model.eval();val_metrics=one_epoch(model,va,optim,args.device,sq,sv,args.grad_accum,False,0,args.debug_small_cell,args.batch_size)
            row={'epoch':ep+1,'train':train_metrics,'val':val_metrics}
            log.write(json.dumps(row)+'\n');log.flush()
            if val_metrics['total']<best:
                best=val_metrics['total']
                torch.save({'state_dict':model.state_dict(),'meta':meta,'epoch':ep+1,
                            'optimizer':optim.state_dict()},outdir/'best.pt')
            if ep%5==0 or ep+1==args.epochs:print(row,flush=True)
    print('Finished: best normalized validation loss:',best)


if __name__=='__main__':main()