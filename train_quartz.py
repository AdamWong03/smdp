"""Formal 5000/1250 quartz segment training, then differentiable autonomous training."""
import argparse
from dataclasses import asdict
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from smdp_neural import NeuralSMDP, SMDPConfig
from training_objectives import SegmentStats, supervised_segment_loss
from periodic import PeriodicGraphBuilder, predict_batch
from reference_data import QuartzSegments, validate_split_manifest


def batch_to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def loss_for(model, batch, times, stats, builder, rollout=False):
    q, v = batch['q'], batch['v']
    out = predict_batch(model, q[:, 0], v[:, 0], batch['z'], batch['mass'], batch['cell'], times, builder)
    target_q = q[:, times.long()].transpose(0, 1).flatten(1, 2)
    target_v = v[:, times.long()].transpose(0, 1).flatten(1, 2)
    result = supervised_segment_loss(out, q[:, 0].flatten(0, 1), v[:, 0].flatten(0, 1),
                                     times, target_q, target_v, stats)
    result['segment_loss'] = result['loss']
    result['endpoint_q_mse'] = (out['positions'][-1]-target_q[-1]).square().mean()
    result['endpoint_v_mse'] = (out['velocities'][-1]-target_v[-1]).square().mean()
    if rollout:
        qnext, vnext = out['positions'][-1].reshape_as(q[:, 0]), out['velocities'][-1].reshape_as(v[:, 0])
        losses = []
        for frame in [17, 18]:  # actual t+32 and t+48, not interpolated
            step = predict_batch(model, qnext, vnext, batch['z'], batch['mass'], batch['cell'], [16.], builder)
            qnext, vnext = step['positions'][0].reshape_as(qnext), step['velocities'][0].reshape_as(vnext)
            losses.append(((qnext-q[:, frame])/stats.displacement_residual_rms).square().mean()
                          + ((vnext-v[:, frame])/stats.velocity_delta_rms).square().mean())
        result['rollout_loss'] = torch.stack(losses).mean()
        result['loss'] = 0.5 * (result['loss'] + result['rollout_loss'])
    return result, out


def diagnostics(out):
    return {
        'scalar_allocation': out['mode_scalar_allocation'].detach().mean((0, 2)).cpu().tolist(),
        'vector_allocation': out['mode_vector_allocation'].detach().mean((0, 2)).cpu().tolist(),
        'recurrent_route_by_mode': out['operator_routing'][..., 0].detach().mean((0, 1, 2)).cpu().tolist(),
        'cross_mode_attention': out['coupling_attention'].detach().mean((0, 1)).cpu().tolist(),
        'fusion_weights': out['fusion_weights'].detach().mean((0, 1)).flatten().cpu().tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--segment-epochs', type=int, default=100)
    p.add_argument('--rollout-epochs', type=int, default=24)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'history.jsonl').exists() and args.resume is None:
        raise FileExistsError('Existing run: supply --resume or choose a new output directory')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    sources = validate_split_manifest(args.data_root)
    train = QuartzSegments(args.data_root, 'train', args.output/'cache')
    val = QuartzSegments(args.data_root, 'val', args.output/'cache')
    stats = SegmentStats(**train.scales)
    cfg = SMDPConfig()
    model = NeuralSMDP(cfg).to(device)
    builder = PeriodicGraphBuilder(cfg.cutoff_angstrom)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=2e-6)
    config = dict(model=asdict(cfg), stats=asdict(stats), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  source_trajectories=sources, parameters=sum(p.numel() for p in model.parameters()),
                  training_queries='4,8,16 plus two random distinct internal integer times',
                  validation_queries=[1,4,8,12,16], rollout_steps=3,
                  hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                  torch_version=torch.__version__, units={'position':'Angstrom','velocity':'Angstrom/fs','cell':'row vectors','mass':'u'})
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    start_phase, start_epoch, best, stale, elapsed_prior = 'segment', 0, float('inf'), 0, 0.
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_phase, start_epoch = state['phase'], state['epoch']+1
        best, stale, elapsed_prior = state['best'], state['stale'], state['elapsed_s']
        torch.set_rng_state(state['torch_rng'].cpu()); random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
    started = time.perf_counter()
    for phase, epochs in [('segment', args.segment_epochs), ('rollout', args.rollout_epochs)]:
        if start_phase=='rollout' and phase=='segment': continue
        is_rollout = phase=='rollout'
        if is_rollout and start_phase!='rollout':
            model.load_state_dict(torch.load(args.output/'best_segment.pt',map_location=device,weights_only=False)['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=4,min_lr=1e-6)
            best, stale, start_epoch = float('inf'), 0, 0
        td = Subset(train, torch.where(train.valid)[0].tolist()) if is_rollout else train
        vd = Subset(val, torch.where(val.valid)[0].tolist()) if is_rollout else val
        train_loader = DataLoader(td,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True)
        val_loader = DataLoader(vd,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True)
        for epoch in range(start_epoch, epochs):
            model.train(); sums={}; count=0; gradient_norms={}
            for batch in train_loader:
                batch=batch_to(batch,device)
                extra=random.sample([i for i in range(1,16) if i not in [4,8]],2)
                times=torch.tensor(sorted([4,8,16]+extra),device=device,dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                losses,out=loss_for(model,batch,times,stats,builder,is_rollout)
                if not torch.isfinite(losses['loss']): raise FloatingPointError('Nonfinite training loss')
                losses['loss'].backward()
                if count==0:
                    for name,module in [('encoder',model.encoder),('modes',model.decompose),('recurrent',model.propagate.recurrent),('temporal',model.propagate.direct),('router',model.propagate.router),('interaction',model.couple),('decoder',model.decode)]:
                        norm=sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**0.5
                        gradient_norms[name]=norm
                        if not np.isfinite(norm) or norm==0: raise FloatingPointError(f'No finite nonzero gradient for {name}')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                n=len(batch['q']); count+=n
                for k,v in losses.items(): sums[k]=sums.get(k,0.)+float(v.detach())*n
            model.eval(); val_sums={}; nval=0; diag_sum={}
            with torch.no_grad():
                for batch in val_loader:
                    batch=batch_to(batch,device)
                    losses,out=loss_for(model,batch,torch.tensor([1.,4.,8.,12.,16.],device=device),stats,builder,is_rollout)
                    n=len(batch['q']); nval+=n
                    for k,v in losses.items(): val_sums[k]=val_sums.get(k,0.)+float(v)*n
                    for k,v in diagnostics(out).items(): diag_sum[k]=diag_sum.get(k,0.)+np.array(v)*n
            metric=val_sums['loss']/nval
            if not np.isfinite(metric): raise FloatingPointError('Nonfinite validation loss')
            scheduler.step(metric)
            improved=metric<best*(1-1e-4)
            best=min(best,metric); stale=0 if improved else stale+1
            row=dict(phase=phase,epoch=epoch,train={k:v/count for k,v in sums.items()},validation={k:v/nval for k,v in val_sums.items()},
                     mode_statistics={k:(v/nval).tolist() for k,v in diag_sum.items()},gradient_norms=gradient_norms,
                     lr=optimizer.param_groups[0]['lr'],elapsed_s=elapsed_prior+time.perf_counter()-started,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
            with (args.output/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            state=dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                       phase=phase,epoch=epoch,best=best,stale=stale,elapsed_s=row['elapsed_s'],
                       torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                       cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
            torch.save(state,args.output/'last.pt')
            if improved: torch.save(state,args.output/f'best_{phase}.pt')
            if stale>=args.patience and epoch>=19: break
        start_epoch=0
    (args.output/'training_complete.json').write_text(json.dumps(dict(best_checkpoint='best_rollout.pt',elapsed_s=elapsed_prior+time.perf_counter()-started)))


if __name__=='__main__': main()
