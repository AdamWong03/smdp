"""Full formal CMIS training on real HDF5 segments; no reference forces."""
import argparse
import dataclasses
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cmis_model import CMIS,CMISConfig
from .formal_data import HDF5Segments
from .vesin_adapter import BatchGraph


STOP=False


def stop(*_):
    global STOP
    STOP=True


def save_checkpoint(state,path):
    temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)


def save_json(path,record):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)


def reverse_segments(data,mask):
    q,v,qt,vt=data['q0'],data['v0'],data['q_target'],data['v_target']
    return dict(data,q0=torch.where(mask[:,None,None],qt[:,-1],q),
        v0=torch.where(mask[:,None,None],-vt[:,-1],v),
        q_target=torch.where(mask[:,None,None,None],torch.cat([q[:,None],qt[:,:-1]],1).flip(1),qt),
        v_target=torch.where(mask[:,None,None,None],-torch.cat([v[:,None],vt[:,:-1]],1).flip(1),vt))


def forward_batch(model,data,z0,m0,times,return_modes=False):
    b,n=data['q0'].shape[:2];device=data['q0'].device
    builder=BatchGraph(data['cell'],n,model.cfg.cutoff)
    gids=torch.arange(b,device=device).repeat_interleave(n)
    out=model(data['q0'].reshape(-1,3),data['v0'].reshape(-1,3),z0.repeat(b),m0.repeat(b),
              times,builder,graph_index=gids,return_modes=return_modes)
    return out


def losses(out,data,mass,scales,endpoint_weight):
    b,n=data['q0'].shape[:2];t=data['q_target'].shape[1]
    q=out['positions'].reshape(t,b,n,3).permute(1,0,2,3)-data['q_target']
    v=out['velocities'].reshape(t,b,n,3).permute(1,0,2,3)-data['v_target']
    weight=mass[None,None,:,None]/mass.sum()
    lq=(weight*(q/scales[0]).square()).sum((2,3))/3
    lv=(weight*(v/scales[1]).square()).sum((2,3))/3
    tw=torch.ones(t,device=q.device);tw[-1]=endpoint_weight;tw/=tw.sum()
    pq=(lq*tw).sum(1).mean();pv=(lv*tw).sum(1).mean()
    # Unweighted Cartesian endpoint metrics remain comparable to historical DSRT.
    metric=torch.stack([pq,pv,q[:,-1].square().mean(),v[:,-1].square().mean(),
                        q[:,-1].abs().mean(),v[:,-1].abs().mean(),q.square().mean(),v.square().mean()])
    return pq+pv,metric


def epoch(model,loader,optimizer,device,ds,scales,cfg,train,steps):
    model.train(train);total=np.zeros(8);seen=0;start=time.perf_counter()
    z=torch.as_tensor(ds.z,device=device,dtype=torch.long)
    mass=torch.as_tensor(ds.mass,device=device,dtype=torch.float32)
    times=torch.as_tensor(ds.times,device=device,dtype=torch.float32)
    for index,raw in enumerate(loader):
        data={key:value.to(device,non_blocking=True) for key,value in raw.items()}
        if train:data=reverse_segments(data,torch.rand(len(data['q0']),device=device)<.5)
        if train:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            out=forward_batch(model,data,z,mass,times)
            loss,metrics=losses(out,data,mass,scales,cfg['endpoint_weight'])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite formal loss')
            if train:
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();steps+=1
        b=len(data['q0']);seen+=b;total+=metrics.detach().cpu().double().numpy()*b
        if train and (index+1)%100==0:
            print(json.dumps(dict(event='train_progress',optimizer_steps=steps,batches=index+1,
                                  batches_total=len(loader),loss=float(loss.detach()))),flush=True)
    total/=seen
    return dict(samples=seen,total_loss=float(total[0]+total[1]),position_loss=float(total[0]),
        velocity_loss=float(total[1]),endpoint_position_RMSE_A=float(np.sqrt(total[2])),
        endpoint_velocity_RMSE_A_fs=float(np.sqrt(total[3])),endpoint_position_MAE_A=float(total[4]),
        endpoint_velocity_MAE_A_fs=float(total[5]),segment_position_RMSE_A=float(np.sqrt(total[6])),
        segment_velocity_RMSE_A_fs=float(np.sqrt(total[7])),seconds=time.perf_counter()-start),steps


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    cfg=json.loads((args.run/'config.json').read_text());seed=int(cfg['seed'])
    torch.set_default_dtype(torch.float32);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    device=torch.device('cuda');train=HDF5Segments(Path(cfg['cache'])/'train.h5')
    val=HDF5Segments(Path(cfg['cache'])/'val.h5')
    if (len(train),len(val))!=(5000,1250):raise ValueError('Incomplete formal splits')
    if set(train.source_paths)&set(val.source_paths):raise ValueError('Train/val leakage')
    np.testing.assert_array_equal(train.z,val.z);np.testing.assert_array_equal(train.mass,val.mass)
    np.testing.assert_array_equal(train.times,val.times)
    ts=train.times
    if not np.allclose(np.diff(np.r_[0,ts]),ts[0]) or ts[-1]!=cfg['model']['horizon_fs']:
        raise ValueError('Time reversal requires uniform real query labels through horizon')
    scales=train.scales();model=CMIS(CMISConfig(**cfg['model'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator().manual_seed(seed)
    common=dict(batch_size=cfg['batch_size'],num_workers=cfg['data_workers'],pin_memory=True)
    if cfg['data_workers']:common.update(persistent_workers=True,prefetch_factor=2)
    training=DataLoader(train,shuffle=True,generator=generator,**common)
    validation=DataLoader(val,shuffle=False,**common)
    best=float('inf');first=steps=0;elapsed_prior=0.
    if args.resume:
        saved=torch.load(args.run/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        first=saved['epoch'];best=saved['best'];steps=saved['optimizer_steps'];elapsed_prior=saved['elapsed_seconds']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);generator.set_state(saved['loader_rng'])
    elif (args.run/'last.pt').exists() or (args.run/'best.pt').exists():
        raise FileExistsError('Use --resume; never overwrite a completed run')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    start=time.perf_counter()
    meta=dict(config=dataclasses.asdict(model.cfg),scale_q=scales[0],scale_v=scales[1],
        seed=seed,parameters=sum(p.numel() for p in model.parameters()),
        train_samples=len(train),val_samples=len(val),times_fs=ts.tolist(),
        dataset=str(cfg['cache']),hardware=torch.cuda.get_device_name(device),
        initialized_from_pretrained=False,precision='FP32',upstream_commit=cfg['upstream_commit'])
    save_json(args.run/'metadata.json',meta)
    print(json.dumps(dict(event='formal_training_started',**meta)),flush=True)
    try:
        for ep in range(first,cfg['epochs']):
            tic=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
            tr,steps=epoch(model,training,optimizer,device,train,scales,cfg,True,steps)
            va,_=epoch(model,validation,optimizer,device,val,scales,cfg,False,steps)
            improved=va['total_loss']<best
            if improved:best=va['total_loss']
            record=dict(epoch=ep+1,optimizer_steps=steps,train=tr,validation=va,
                lr=optimizer.param_groups[0]['lr'],improved=improved,
                epoch_seconds=time.perf_counter()-tic,peak_cuda_bytes=torch.cuda.max_memory_allocated(device))
            payload=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),meta=meta,
                experiment_config=cfg,epoch=ep+1,best=best,optimizer_steps=steps,
                elapsed_seconds=elapsed_prior+time.perf_counter()-start,record=record,
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=generator.get_state())
            save_checkpoint(payload,args.run/'last.pt')
            if improved:save_checkpoint(payload,args.run/'best.pt')
            with (args.run/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            save_json(args.run/'status.json',dict(state='training',**record));print(json.dumps(record),flush=True)
            if STOP:
                save_json(args.run/'status.json',dict(state='paused',**record));return
        save_json(args.run/'status.json',dict(state='training_completed',epochs=cfg['epochs'],
            optimizer_steps=steps,best_validation_loss=best,elapsed_seconds=elapsed_prior+time.perf_counter()-start))
    except Exception as error:
        save_json(args.run/'failure.json',dict(reason=f'{type(error).__name__}: {error}',optimizer_steps=steps))
        raise


if __name__=='__main__':main()
