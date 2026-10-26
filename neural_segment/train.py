"""Formal segment/acceleration supervision and two equal-budget on-policy stages."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from neural_segment.model import NeuralSegment, SegmentConfig
from periodic import PeriodicGraphBuilder, predict_batch
from reference_data import QuartzSegments


def load_model(path,device):
    saved=torch.load(path,map_location=device,weights_only=False)
    model=NeuralSegment(SegmentConfig(**saved['config']['model'])).to(device).eval()
    model.load_state_dict(saved['model']);return model,saved['config']


class Supervision(Dataset):
    def __init__(self,base,acceleration):
        self.base=base
        with h5py.File(acceleration) as f:
            assert f.attrs['completed'] and f['done'][:].all()
            self.a=torch.tensor(f['a'][:])
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        b=self.base[i];b['q']=b['q'][:17];b['v']=b['v'][:17];b['a']=self.a[i]
        return b


class OnPolicy(Dataset):
    def __init__(self,path,base):
        with h5py.File(path) as f:
            assert f.attrs['completed']
            self.q=torch.tensor(f['q'][:]);self.v=torch.tensor(f['v'][:]);self.a=torch.tensor(f['a'][:]);self.cell=torch.tensor(f['cell'][:])
        self.z,self.m=base.z,base.m
    def __len__(self):return len(self.q)
    def __getitem__(self,i):
        return dict(q=self.q[i],v=self.v[i],a=self.a[i],cell=self.cell[i],z=self.z,mass=self.m,rollout_valid=torch.tensor(False))


def device_batch(b,device):return {k:v.to(device) for k,v in b.items()}


def errors(out,b,scales):
    tq=b['q'][:,:17].transpose(0,1).flatten(1,2)
    tv=b['v'][:,:17].transpose(0,1).flatten(1,2)
    eq=(out['positions'][1:]-tq[1:]).square().mean()
    ev=(out['velocities'][1:]-tv[1:]).square().mean()
    loss=eq/scales['displacement_residual_rms']**2+ev/scales['velocity_delta_rms']**2
    return loss,eq,ev


def validate(model,data,scales,builder,device):
    totals=np.zeros(3);count=0
    model.eval()
    with torch.inference_mode():
        for b in DataLoader(data,batch_size=4,shuffle=False):
            b=device_batch(b,device)
            out=predict_batch(model,b['q'][:,0],b['v'][:,0],b['z'],b['mass'],b['cell'],list(range(17)),builder)
            loss,eq,ev=errors(out,b,scales)
            n=len(b['q']);count+=n;totals+=np.array([float(loss),float(eq),float(ev)])*n
        # Fixed validation-only subset and fixed metric; no test selection.
        indices=torch.where(data.valid)[0][:128].tolist()
        long_loss=0.;long_count=0
        for b in DataLoader(torch.utils.data.Subset(data,indices),batch_size=4):
            b=device_batch(b,device);q=b['q'][:,0];v=b['v'][:,0]
            for _ in range(16):
                out=predict_batch(model,q,v,b['z'],b['mass'],b['cell'],[16.],builder)
                q=out['positions'][0].reshape_as(q);v=out['velocities'][0].reshape_as(v)
            target=int(np.flatnonzero(data.offsets==256)[0])
            loss=((q-b['q'][:,target])/scales['displacement_residual_rms']).square().mean()+((v-b['v'][:,target])/scales['velocity_delta_rms']).square().mean()
            long_loss+=float(loss)*len(q);long_count+=len(q)
    totals/=count
    return dict(segment_loss=totals[0],position_RMSE_A=np.sqrt(totals[1]),velocity_RMSE_A_fs=np.sqrt(totals[2]),
                rollout256_loss=long_loss/long_count,selection_metric=totals[0]+.25*long_loss/long_count,
                segment_samples=count,rollout_samples=long_count)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--shared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['direct','independent','full'],required=True)
    p.add_argument('--stage',type=int,choices=[0,1,2],required=True);p.add_argument('--device',default='cuda')
    args=p.parse_args();out=args.output/f'stage{args.stage}';out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists():raise FileExistsError('Existing stage; no implicit restart')
    report=json.loads((args.shared/'basis.json').read_text())
    assert report['selected_basis_count'] in (6,8)
    assert json.loads((args.shared/'bks_verification.json').read_text())['passed']
    assert json.loads((args.shared/'correctness.json').read_text())['passed']
    seed=20261026+args.stage;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    device=torch.device(args.device)
    train=QuartzSegments(args.data_root,'train',args.shared/'cache')
    val=QuartzSegments(args.data_root,'val',args.shared/'cache')
    scales=dict(report['train_scales']);scales.update(json.loads((args.shared/'acceleration_scale.json').read_text()))
    cfg=SegmentConfig(variant=args.variant,basis_count=report['selected_basis_count'])
    model=NeuralSegment(cfg).to(device)
    if args.stage:
        previous=torch.load(args.output/f'stage{args.stage-1}/best.pt',map_location=device,weights_only=False)
        model.load_state_dict(previous['model'])
    original=Supervision(train,args.shared/'original_acceleration.h5')
    additions=[OnPolicy(args.output/f'onpolicy_round{i}.h5',train) for i in range(1,args.stage+1)]
    dataset=ConcatDataset([original,*additions])
    if additions:
        added=sum(len(d) for d in additions)
        if not added:raise RuntimeError('No valid on-policy states: cannot silently skip data stage')
        weights=torch.cat([torch.full((len(original),),.7/len(original)),torch.full((added,),.3/added)])
        sampler=WeightedRandomSampler(weights,num_samples=5000,replacement=True)
    else:sampler=None
    loader=DataLoader(dataset,batch_size=4,shuffle=sampler is None,sampler=sampler,num_workers=0)
    epochs=80 if args.stage==0 else 20
    config=dict(model=asdict(cfg),model_family='neural_segment',stats=scales,seed=seed,stage=args.stage,
                epochs=epochs,batch_size=4,acceleration_weight=.1,original_fraction=.7 if additions else 1.,
                original_samples=5000,onpolicy_samples=sum(len(d) for d in additions),
                selection='all 1250 validation segments + 0.25 * normalized 256fs loss on first 128 valid validation starts',
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                hardware=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    builder=PeriodicGraphBuilder(4.5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4 if args.stage==0 else 1e-4,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=4,factor=.5,min_lr=1e-6)
    best=float('inf');stale=0;started=time.monotonic()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train();total=np.zeros(4);count=0
        for step,b in enumerate(loader):
            b=device_batch(b,device);optimizer.zero_grad(set_to_none=True)
            result=predict_batch(model,b['q'][:,0],b['v'][:,0],b['z'],b['mass'],b['cell'],list(range(17)),builder)
            loss,eq,ev=errors(result,b,scales)
            target=b['a'].transpose(0,1).flatten(1,2)
            la=((result['accelerations'][[0,4,8,12,16]]-target)/scales['acceleration_rms']).square().mean()
            objective=loss+.1*la
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite objective')
            objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            n=len(b['q']);count+=n;total+=np.array([float(objective.detach()),float(eq.detach()),float(ev.detach()),float(la.detach())])*n
            if step%250==0:print(json.dumps(dict(stage=args.stage,epoch=epoch,batch=step,loss=float(objective.detach()))),flush=True)
        validation=validate(model,val,scales,builder,device)
        metric=validation['selection_metric']
        # Unstable autonomous validation is reported as a failure, never hidden by thermostat.
        if not np.isfinite(metric):
            torch.save(dict(config=config,model=model.state_dict(),epoch=epoch,validation=validation),out/'failed_validation.pt')
            (out/'failure.json').write_text(json.dumps(dict(reason='Nonfinite validation propagation',epoch=epoch)))
            raise FloatingPointError('Nonfinite validation propagation; saved failed checkpoint, do not change architecture')
        scheduler.step(metric);improved=metric<best*(1-1e-4)
        if improved:best=metric;stale=0
        else:stale+=1
        record=dict(epoch=epoch,training=dict(zip(['loss','position_MSE','velocity_MSE','acceleration_loss'],(total/count).tolist())),
                    validation=validation,elapsed_s=time.monotonic()-started,lr=optimizer.param_groups[0]['lr'],
                    peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        saved=dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   epoch=epoch,validation=validation,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        torch.save(saved,out/'last.pt')
        if improved:torch.save(saved,out/'best.pt')
        if epoch>=19 and stale>=12:break
    (out/'complete.json').write_text(json.dumps(dict(completed=True,best_metric=best,epochs=epoch+1,elapsed_s=time.monotonic()-started)))


if __name__=='__main__':main()
