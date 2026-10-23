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
from training_objectives import SegmentStats, supervised_segment_loss, physical_segment_objectives
from periodic import PeriodicGraphBuilder, predict_batch
from r3_consistency import composition_consistency
from reference_data import QuartzSegments, validate_split_manifest


def batch_to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


OBJECTIVE_WEIGHTS = dict(kinematics=.2, species_velocity=.25, species_kinetic=.1, vacf=.1, join=.2)


def loss_for(model, batch, times, stats, builder, rollout_steps=0, composition_weight=0.0, composition_samples=2):
    q, v = batch['q'], batch['v']
    out = predict_batch(model, q[:, 0], v[:, 0], batch['z'], batch['mass'], batch['cell'], times, builder)
    target_q = q[:, times.long()].transpose(0, 1).flatten(1, 2)
    target_v = v[:, times.long()].transpose(0, 1).flatten(1, 2)
    result = supervised_segment_loss(out, q[:, 0].flatten(0, 1), v[:, 0].flatten(0, 1),
                                     times, target_q, target_v, stats)
    result['segment_loss'] = result['loss']
    result['endpoint_q_mse'] = (out['positions'][-1]-target_q[-1]).square().mean()
    result['endpoint_v_mse'] = (out['velocities'][-1]-target_v[-1]).square().mean()
    extra = physical_segment_objectives(out,target_q,target_v,times,batch['z'].flatten(),batch['mass'].flatten(),stats,graphs=len(q))
    result.update(extra)
    result['loss'] = result['loss'] + sum(OBJECTIVE_WEIGHTS[k]*x for k,x in extra.items())
    if rollout_steps:
        # Pushforward: all prefix states are autonomous; only the last two
        # segments retain gradients. No reference state is injected into rollout.
        qnext = out['positions'][-1].reshape_as(q[:,0]).detach()
        vnext = out['velocities'][-1].reshape_as(v[:,0]).detach()
        left = (out['positions'][-1]-out['positions'][torch.where(times==15)[0][0]]).detach()
        losses, joins, predicted_v, reference_v = [], [], [], []
        offset_index = {int(t):i for i,t in enumerate(QuartzSegments.offsets)}
        for index in range(1,rollout_steps):
            differentiable = torch.is_grad_enabled() and index >= rollout_steps-2
            with torch.set_grad_enabled(differentiable):
                step = predict_batch(model,qnext,vnext,batch['z'],batch['mass'],batch['cell'],[1.,15.,16.],builder)
                right = step['positions'][0]-qnext.flatten(0,1)
                # Compare the junction defect with the true trajectory, rather
                # than forcing a finite-difference derivative jump to zero.
                center = offset_index[index*16]
                ref_join = (q[:,offset_index[index*16+1]]-2*q[:,center]+q[:,offset_index[index*16-1]]).flatten(0,1)
                join = ((right-left-ref_join)/stats.velocity_delta_rms).square().mean()
                left = step['positions'][-1]-step['positions'][-2]
                qnext, vnext = step['positions'][-1].reshape_as(qnext), step['velocities'][-1].reshape_as(vnext)
                frame = offset_index[(index+1)*16]
                endpoint = ((qnext-q[:,frame])/stats.displacement_residual_rms).square().mean() + ((vnext-v[:,frame])/stats.velocity_delta_rms).square().mean()
                predicted_v.append(vnext); reference_v.append(v[:,frame])
                if index >= rollout_steps-2:
                    losses.append(endpoint); joins.append(join)
            if not differentiable:
                qnext,vnext,left = qnext.detach(),vnext.detach(),left.detach()
        result['rollout_loss'] = torch.stack(losses).mean()
        result['join'] = torch.stack(joins).mean()
        # Species amplitude and temporal correlations along the autonomous path.
        pred = torch.stack(predicted_v).flatten(1,2)
        ref = torch.stack(reference_v).flatten(1,2)
        temporal = physical_segment_objectives({'positions':pred*0,'velocities':pred},ref*0,ref,
                    torch.arange(len(pred),device=q.device)*16,batch['z'].flatten(),batch['mass'].flatten(),stats,graphs=len(q))
        result['rollout_species_kinetic'] = temporal['species_kinetic']
        result['rollout_vacf'] = temporal['vacf']
        result['loss'] = result['loss'] + result['rollout_loss'] + OBJECTIVE_WEIGHTS['join']*result['join'] + .1*(temporal['species_kinetic']+temporal['vacf'])
    if composition_weight:
        terms = composition_consistency(model, batch, out, times, stats, builder, samples=composition_samples)
        result.update(terms)
        result['loss'] = result['loss'] + composition_weight * terms['composition']
    return result, out


def diagnostics(out):
    result = {
        'scalar_allocation': out['mode_scalar_allocation'].detach().mean((0, 2)).cpu().tolist(),
        'vector_allocation': out['mode_vector_allocation'].detach().mean((0, 2)).cpu().tolist(),
        'recurrent_route_by_mode': out['operator_routing'][..., 0].detach().mean((0, 1, 2)).cpu().tolist(),
        'cross_mode_attention': out['coupling_attention'].detach().mean((0, 1)).cpu().tolist(),
    }
    if out['fusion_weights'] is not None:
        result['fusion_weights'] = out['fusion_weights'].detach().mean((0,1)).flatten().cpu().tolist()
    for key in ['mode_delta_q','mode_delta_v']:
        contributions=out[key][-1].detach()
        rms=contributions.square().mean((0,2)).sqrt()
        result[key+'_rms']=rms.cpu().tolist()
        result[key+'_rms_share']=(rms/rms.sum().clamp_min(1e-20)).cpu().tolist()
        total=contributions.sum(1)
        denom=total.square().sum().clamp_min(1e-20)
        result[key+'_signed_projection']=(torch.einsum('nkc,nc->k',contributions,total)/denom).cpu().tolist()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--segment-epochs', type=int, default=100)
    p.add_argument('--rollout-epochs', type=int, default=12, help='epochs for EACH of the 4/8/16-step phases')
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--resume', type=Path)
    p.add_argument('--r3-composition-weight', type=float, default=0.0)
    p.add_argument('--r3-composition-samples', type=int, default=2)
    p.add_argument('--spatial-mode-exchange', action='store_true')
    p.add_argument('--max-rollout-steps', type=int, choices=[8,16], default=16)
    args = p.parse_args()
    if args.r3_composition_weight < 0 or args.r3_composition_samples < 1:
        p.error('Invalid R3 composition weight or samples')
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
    cfg = SMDPConfig(additive_modes=True, exclude_self_exchange=True,
                     spatial_mode_exchange=args.spatial_mode_exchange)
    model = NeuralSMDP(cfg).to(device)
    # Preserve identical subsequent data/shuffle RNG for R3-A and R3-B.
    # The shared encoder/propagator/decoder are initialized before the optional
    # spatial module, so their initial weights match under the same seed.
    if args.r3_composition_weight:
        torch.manual_seed(args.seed + 41)
    builder = PeriodicGraphBuilder(cfg.cutoff_angstrom)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=2e-6)
    config = dict(model=asdict(cfg), stats=asdict(stats), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  source_trajectories=sources, parameters=sum(p.numel() for p in model.parameters()),
                  training_queries='0,1,4,8,15,16 plus one random adjacent triplet',
                  validation_queries=[0,1,4,8,12,15,16], rollout_steps=[4,8,16],
                  loss_weights=OBJECTIVE_WEIGHTS, pushforward_gradient_segments=2,
                  r3_composition_weight=args.r3_composition_weight,
                  r3_composition_samples=args.r3_composition_samples,
                  r3_spatial_mode_exchange=args.spatial_mode_exchange,
                  selection='complete validation split; 16-step autonomous metric in all rollout phases; no test selection',
                  hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                  torch_version=torch.__version__, units={'position':'Angstrom','velocity':'Angstrom/fs','cell':'row vectors','mass':'u'})
    import subprocess
    config['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
    config['rollout_valid_samples']={'train':int(train.valid.sum()),'val':int(val.valid.sum())}
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
    phases=[('segment',args.segment_epochs),('rollout4',args.rollout_epochs),('rollout8',args.rollout_epochs)]
    if args.max_rollout_steps == 16:
        phases.append(('rollout16',args.rollout_epochs))
    for phase_index,(phase, epochs) in enumerate(phases):
        if phase_index < [x[0] for x in phases].index(start_phase): continue
        is_rollout = phase!='segment'
        steps = int(phase.replace('rollout','')) if is_rollout else 0
        if is_rollout and phase!=start_phase:
            previous=phases[phase_index-1][0]
            model.load_state_dict(torch.load(args.output/f'best_{previous}.pt',map_location=device,weights_only=False)['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5 if steps==4 else 2e-5, weight_decay=1e-6)
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
                center=random.randint(2,14)
                times=torch.tensor(sorted({0,1,4,8,15,16,center-1,center,center+1}),device=device,dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                losses,out=loss_for(model,batch,times,stats,builder,steps,
                                    args.r3_composition_weight,args.r3_composition_samples)
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
                    losses,out=loss_for(model,batch,torch.tensor([0.,1.,4.,8.,12.,15.,16.],device=device),stats,builder,16 if is_rollout else 0,
                                        args.r3_composition_weight,args.r3_composition_samples)
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
    # All rollout phases use exactly the same 16-step validation objective;
    # R3 selects across stages instead of forcing the final stage to win.
    import shutil
    if args.r3_composition_weight:
        candidates=[]
        for phase,_ in phases:
            if not phase.startswith('rollout'): continue
            ckpt=torch.load(args.output/f'best_{phase}.pt',map_location='cpu',weights_only=False)
            candidates.append((float(ckpt['best']),phase))
        winner_metric,winner_phase=min(candidates)
    else:
        winner_phase=phases[-1][0]
        winner_metric=None
    shutil.copyfile(args.output/f'best_{winner_phase}.pt',args.output/'best_rollout.pt')
    (args.output/'training_complete.json').write_text(json.dumps(dict(
        best_checkpoint='best_rollout.pt',selection_phase=winner_phase,
        selection_validation_loss=winner_metric,
        elapsed_s=elapsed_prior+time.perf_counter()-started)))


if __name__=='__main__': main()
