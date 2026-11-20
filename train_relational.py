"""Formal endpoint training on existing Quartz 30 fs graphs. No MD generation."""
import argparse
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from trajcast.data.dataset import AtomicGraphDataset

from phase_space_transition import PhaseSpaceTransition, train_normalization
from relational_transition import (RelationalConfig, RelationalTransition,
                                   endpoint_loss_components, endpoint_objective)
from motion_loss import motion_balanced_objective, train_mode_rms
from selection_metrics import molecular_kinetic_partition, partition_relative_error
from validation_rollout import build_rollout_starts, rollout_partition_error

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def make_scheduler(optimizer, cfg):
    schedule = cfg.get('scheduler', {'type': 'plateau'})
    if schedule['type'] == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=.8, patience=20, min_lr=1e-6), 'plateau'
    if schedule['type'] != 'warmup_cosine':
        raise ValueError(f"Unknown scheduler: {schedule['type']}")
    warmup = int(schedule['warmup_steps'])
    total = int(cfg['total_optimizer_steps'])
    peak, initial, minimum = (float(cfg['learning_rate']),
                              float(schedule['initial_lr']),
                              float(schedule['minimum_lr']))
    if not 0 < initial <= peak or not 0 < minimum <= peak or not 0 < warmup < total:
        raise ValueError('Invalid warmup/cosine schedule')

    def multiplier(step):
        if step <= warmup:
            return (initial + (peak-initial) * step/warmup) / peak
        progress = min(1., (step-warmup)/(total-warmup))
        return (minimum + .5*(peak-minimum)*(1+math.cos(math.pi*progress))) / peak

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), 'warmup_cosine'


def module_gradient_norm(module):
    total = None
    for parameter in module.parameters():
        if parameter.grad is not None:
            value = parameter.grad.detach().float().square().sum()
            total = value if total is None else total + value
    return float(total.sqrt()) if total is not None else 0.


def gradient_groups(model):
    if isinstance(model, PhaseSpaceTransition):
        groups = {
            'input_encoder': torch.nn.ModuleList([
                model.embedding, model.relations, model.edge_initial]),
            'proposal_head': model.proposal_head,
            'correction_head': model.correction_head,
        }
    else:
        groups = {
            'input_encoder': torch.nn.ModuleList([
                model.element, model.kinematics, model.velocity_embedding, model.edge_embedding]),
            'proposal_head': model.proposal_head,
            'correction_head': model.correction_head,
        }
    groups.update({f'proposal_block_{i}': block for i, block in enumerate(model.proposal_blocks)})
    groups.update({f'refinement_block_{i}': block for i, block in enumerate(model.refinement_blocks)})
    return groups


def dataset_system(cfg):
    """Physical system label used to gate the molecular partition proxy."""
    name = str(cfg.get('dataset_name', '')).lower()
    if 'water' in name:
        return 'water'
    if 'quartz' in name:
        return 'quartz'
    return None


def is_phase_space(cfg):
    """True only for explicitly versioned phase-space configs."""
    architecture = cfg.get('architecture')
    if architecture in (None, 'dsrt', 'smdpr'):
        return False
    if architecture != 'phase_space_operator_v0':
        raise ValueError(f'Unknown architecture: {architecture}')
    return True


def build_model(cfg, rms_targets, aggregation_scale=None):
    if is_phase_space(cfg):
        if aggregation_scale is None:
            raise ValueError('Phase-space model requires the TRAIN aggregation scale')
        return PhaseSpaceTransition(cfg['model'], rms_targets=rms_targets,
                                    aggregation_scale=aggregation_scale)
    return RelationalTransition(cfg['model'], rms_targets)


def train_statistics(cfg, graphs):
    """Normalization scalars with one explicit source: the full TRAIN split."""
    motion = cfg.get('loss', {}).get('mode') == 'motion_balanced'
    if motion and dataset_system(cfg) != 'water':
        raise ValueError('motion-balanced loss is defined for molecular water only')
    mode_rms = train_mode_rms(graphs) if motion else None
    if is_phase_space(cfg):
        stats = train_normalization(graphs, float(cfg['model']['cutoff_angstrom']))
        if mode_rms is not None:
            stats['mode_rms'] = mode_rms
        return (stats['rms_targets'][0], stats['rms_targets'][1],
                stats['aggregation_scale'], stats)
    sq = torch.cat([g.displacements for g in graphs]).square().mean().sqrt().item()
    sv = torch.cat([g.update_velocities for g in graphs]).square().mean().sqrt().item()
    stats = dict(rms_targets=[sq, sv], aggregation_scale=None,
                 source='complete TRAIN split only')
    if mode_rms is not None:
        stats['mode_rms'] = mode_rms
    return sq, sv, None, stats


def training_diagnostics(model, batch, precision, proposal_weight):
    """One fixed training batch; gradients are observed but never optimized."""
    groups = gradient_groups(model)
    gradients = {}
    model.train()
    model.set_diagnostics(True)
    for label in ['final', 'weighted_proposal']:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch.pos.device.type, dtype=torch.bfloat16,
                            enabled=precision == 'bf16'):
            output = model(batch)
            pieces = endpoint_loss_components(
                model, output, batch.displacements, batch.update_velocities)
            objective = pieces['final'] if label == 'final' else proposal_weight * pieces['proposal']
        objective.backward()
        gradients[label] = {name: module_gradient_norm(module) for name, module in groups.items()}
    activations = model.diagnostics()
    model.zero_grad(set_to_none=True)
    model.set_diagnostics(False)
    return dict(gradient_norms=gradients, blocks=activations)


def dataset_settings(cfg):
    """Dataset naming/species metadata, with exact Quartz backward defaults."""
    return (cfg.get('dataset_name', 'quartz30'),
            {int(key): int(value) for key, value in
             cfg.get('atom_type_mapper', {8: 0, 14: 1}).items()},
            cfg.get('split_counts', {'train': 5000, 'val': 1250, 'test': 1000}))


def read_graphs(cache, split, cfg=None):
    cfg = cfg or {}
    name, mapper, counts = dataset_settings(cfg)
    path = cache / f'{name}_{split};P:64.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    # Upstream encodes dtype in filenames. Read existing f64 cache, then cast.
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dataset = AtomicGraphDataset(root=str(cache), name=f'{name}_{split}',
                                    cutoff_radius=float(cfg.get('model', {}).get(
                                        'cutoff_angstrom', 4.5)),
                                    files=[], atom_type_mapper=mapper)
        graphs = [dataset[i].clone().apply(lambda x: x.float() if x.is_floating_point() else x)
                  for i in range(len(dataset))]
    finally:
        torch.set_default_dtype(old)
    if len(graphs) != int(counts[split]):
        raise ValueError(f'Incomplete formal {split} dataset')
    # Attach molecule identities for the physical-proxy selection metric.
    # The cache itself does not carry them; the transition HDF5 does.  A
    # missing file or key leaves the graphs unchanged and disables the proxy.
    horizon = float(cfg.get('model', {}).get('horizon_fs', 0.0))
    transition = (Path(cache).parent.parent / 'transitions'
                  / f'{horizon:g}fs' / f'{split}.h5')
    if transition.is_file():
        import h5py
        with h5py.File(transition, 'r') as handle:
            if 'molecule_ids' in handle:
                ids = torch.as_tensor(handle['molecule_ids'][:], dtype=torch.long)
                if ids.numel() != len(graphs[0].pos):
                    raise ValueError('molecule_ids length does not match atoms')
                for graph in graphs:
                    graph.molecule_ids = ids.clone()
    return graphs


@torch.no_grad()
def evaluate(model, loader, device, precision='fp32', selection_cfg=None, system=None,
             loss_cfg=None, mode_rms=None, proposal_weight=0.02):
    model.eval()
    total = np.zeros(10, dtype=np.float64)
    count = 0
    motion_final = 0.0
    mode_sums = {k: 0.0 for k in ('translation', 'rotation', 'internal')}
    motion_loss = bool(loss_cfg and loss_cfg.get('mode') == 'motion_balanced')
    partition_pred = {k: 0.0 for k in ('translation', 'rotation', 'internal', 'total')}
    partition_ref = {k: 0.0 for k in ('translation', 'rotation', 'internal', 'total')}
    partition_count = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if hasattr(model, 'validate_graph'):
            model.validate_graph(batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == 'bf16'):
            out = model(batch)
            qe = out['displacements']-batch.displacements
            ve = out['velocities']-batch.update_velocities
            endpoint_loss, endpoint_pieces = endpoint_objective(
                model, out, batch.displacements, batch.update_velocities)
            if motion_loss:
                loss, pieces = motion_balanced_objective(
                    model, out, batch, mode_rms,
                    velocity_weights=(loss_cfg or {}).get('velocity_weights'),
                    position_weight=float((loss_cfg or {}).get('position_weight', 1.0)),
                    proposal_weight=proposal_weight)
            else:
                loss, pieces = endpoint_loss, endpoint_pieces
        n = batch.num_graphs
        # Endpoint metrics stay comparable across loss variants; the training
        # loss pieces are accumulated separately below.
        total += np.array([float(qe.square().mean()), float(ve.square().mean()),
                           float(qe.abs().mean()), float(ve.abs().mean()),
                           float(endpoint_pieces['final']), float(endpoint_pieces['proposal']),
                           float(endpoint_pieces['final_position']),
                           float(endpoint_pieces['final_velocity']),
                           float(endpoint_pieces['proposal_position']),
                           float(endpoint_pieces['proposal_velocity'])])*n
        if motion_loss:
            motion_final += float(pieces['final'])*n
            for key in mode_sums:
                mode_sums[key] += float(pieces[f'mode_{key}'])*n
        count += n
        # One-step physical proxy: only for molecular systems whose cache
        # carries molecule_ids.  Rollout modes compute the proxy separately in
        # the training loop; one-step partition is almost perfect on reference
        # states and is not used for rollout selection.
        selection_mode = (selection_cfg or {}).get(
            'mode', 'loss' if motion_loss else 'endpoint')
        if (selection_cfg and selection_mode in ('endpoint_partition', 'partition')
                and system == 'water'
                and getattr(batch, 'molecule_ids', None) is not None):
            for index in range(n):
                mask = batch.batch == index
                cell = batch.cell
                if cell.ndim == 3:
                    cell = cell[index]
                elif cell.shape == (3, 3):
                    cell = cell
                elif cell.shape[0] == n*3:
                    cell = cell.reshape(n, 3, 3)[index]
                else:
                    raise ValueError(f'unexpected batched cell shape {tuple(cell.shape)}')
                masses = batch.atomic_masses[mask].reshape(-1)
                molecules = batch.molecule_ids[mask]
                reference_positions = batch.pos[mask] + batch.displacements[mask]
                predicted = molecular_kinetic_partition(
                    out['positions'][mask], out['velocities'][mask],
                    masses, molecules, cell)
                reference = molecular_kinetic_partition(
                    reference_positions, batch.update_velocities[mask],
                    masses, molecules, cell)
                for name in partition_pred:
                    partition_pred[name] += float(predicted[name])
                    partition_ref[name] += float(reference[name])
                partition_count += 1
    if count == 0:
        raise ValueError('empty evaluation loader')
    total /= count
    selection_loss = (motion_final/count) if motion_loss else float(total[4])
    result = dict(samples=count, position_RMSE_A=float(np.sqrt(total[0])),
                  velocity_RMSE_A_fs=float(np.sqrt(total[1])), position_MAE_A=float(total[2]),
                  velocity_MAE_A_fs=float(total[3]), selection_loss=float(selection_loss),
                  endpoint_selection_loss=float(total[4]),
                  motion_selection_loss=(float(motion_final/count) if motion_loss else None),
                  mode_translation_loss=(float(mode_sums['translation']/count) if motion_loss else None),
                  mode_rotation_loss=(float(mode_sums['rotation']/count) if motion_loss else None),
                  mode_internal_loss=(float(mode_sums['internal']/count) if motion_loss else None),
                  proposal_loss=float(total[5]), normalized_position_MSE=float(total[6]),
                  normalized_velocity_MSE=float(total[7]),
                  proposal_normalized_position_MSE=float(total[8]),
                  proposal_normalized_velocity_MSE=float(total[9]))
    # E1 isolates the training loss: both arms select on the same endpoint
    # metric.  Without this, a motion-balanced training loss would silently
    # become the selection metric even when selection.mode == "endpoint".
    if selection_cfg and selection_cfg.get('mode') == 'endpoint':
        result['selection_loss'] = result['endpoint_selection_loss']
    if partition_count:
        pred = {k: v/partition_count for k, v in partition_pred.items()}
        ref = {k: v/partition_count for k, v in partition_ref.items()}
        components = tuple(selection_cfg.get('partition_components',
                                             ('translation', 'rotation', 'internal')))
        error = float(partition_relative_error(
            {k: torch.tensor(v) for k, v in pred.items()},
            {k: torch.tensor(v) for k, v in ref.items()}, components))
        weight = float(selection_cfg.get('partition_weight', 0.0))
        mode = selection_cfg.get('mode', 'loss' if motion_loss else 'endpoint')
        if mode in ('endpoint', 'loss'):
            selection = result['selection_loss']
        elif mode == 'partition':
            selection = error
        elif mode == 'endpoint_partition':
            selection = result['endpoint_selection_loss'] + weight*error
        elif mode in ('endpoint_rollout', 'rollout'):
            # The rollout proxy is computed by the training loop, which has
            # the model and the fixed validation starts.  Keep the selection
            # loss as the fallback for --evaluate-best or logging.
            selection = result['selection_loss']
        else:
            raise ValueError(f'Unknown selection mode: {mode}')
        result.update(selection_loss=float(selection),
                      partition_selection_loss=error,
                      partition_mode=mode,
                      partition_weight=weight,
                      partition_translation_pred_eV=pred['translation'],
                      partition_translation_ref_eV=ref['translation'],
                      partition_rotation_pred_eV=pred['rotation'],
                      partition_rotation_ref_eV=ref['rotation'],
                      partition_internal_pred_eV=pred['internal'],
                      partition_internal_ref_eV=ref['internal'],
                      partition_total_pred_eV=pred['total'],
                      partition_total_ref_eV=ref['total'])
    else:
        result.update(partition_selection_loss=None, partition_mode=None)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate-best', action='store_true')
    args = p.parse_args()
    cfg = json.loads((args.run/'config.json').read_text())
    if cfg['precision'] not in {'fp32', 'bf16'}:
        raise ValueError('precision must be fp32 or bf16')
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    # FP32 initially; no unverified mixed precision / TF32 change to the model.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    start = time.time()
    train = read_graphs(Path(cfg['cache']), 'train', cfg)
    val = read_graphs(Path(cfg['cache']), 'val', cfg)
    sq, sv, aggregation_scale, normalization = train_statistics(cfg, train)
    # Persist the frozen normalization inside every checkpoint config so the
    # inference loader never has to recompute it from a different data split.
    cfg = dict(cfg, normalization=dict(normalization, rms_targets=[sq, sv],
                                       aggregation_scale=aggregation_scale))
    model = build_model(cfg, (sq, sv), aggregation_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                  weight_decay=0., amsgrad=True)
    scheduler, scheduler_type = make_scheduler(optimizer, cfg)
    generator = torch.Generator().manual_seed(seed)
    def loader(graphs, shuffle):
        return DataLoader(graphs, batch_size=cfg['batch_size'], shuffle=shuffle,
                          num_workers=0, pin_memory=True, generator=generator if shuffle else None)
    train_loader, val_loader = loader(train, True), loader(val, False)
    selection_cfg = cfg.get('selection') or {}
    loss_cfg = cfg.get('loss') or {}
    motion_loss = loss_cfg.get('mode') == 'motion_balanced'
    system = dataset_system(cfg)
    if motion_loss and system != 'water':
        raise ValueError('motion-balanced loss is defined for molecular water only')
    rollout_starts = None
    if selection_cfg.get('mode') in ('endpoint_rollout', 'rollout'):
        if system != 'water':
            raise ValueError('Rollout partition selection is defined for water only')
        rollout_starts = build_rollout_starts(
            Path(cfg['cache']), 'val', cfg,
            steps=int(selection_cfg['rollout_steps']),
            samples=int(selection_cfg['rollout_samples']),
            seed=int(seed) + 17)
        print(json.dumps(dict(event='rollout_selection_starts',
                              samples=len(rollout_starts),
                              steps=int(selection_cfg['rollout_steps']))), flush=True)
    diagnostic_batch = next(iter(loader(train[:cfg['batch_size']], False))).to(
        device, non_blocking=True)
    if hasattr(model, 'validate_graph'):
        model.validate_graph(diagnostic_batch)
    if args.evaluate_best:
        saved = torch.load(args.run/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        test = read_graphs(Path(cfg['cache']), 'test', cfg)
        result = evaluate(model, loader(test, False), device, cfg['precision'],
                          cfg.get('selection'), dataset_system(cfg),
                          loss_cfg=loss_cfg,
                          mode_rms=cfg.get('normalization', {}).get('mode_rms'),
                          proposal_weight=cfg['proposal_weight'])
        result.update(checkpoint=str(args.run/'best.pt'), selected_epoch=saved['epoch']+1,
                      selected_optimizer_steps=saved.get('optimizer_steps'),
                      stopped_epoch=json.loads((args.run/'status.json').read_text()).get('last_epoch'),
                      actual_training_seconds=saved['elapsed_seconds'])
        (args.run/'test_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
        (args.run/'status.json').write_text(json.dumps(
            dict(state='stopped_and_test_completed', **result), indent=2)+'\n')
        print(json.dumps(result), flush=True)
        return
    first, best, stale, previous_seconds, optimizer_steps = 0, float('inf'), 0, 0., 0
    if args.resume:
        saved = torch.load(args.run/'last.pt', weights_only=False, map_location='cpu')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        first, best, stale = saved['epoch']+1, saved['best'], saved['stale']
        previous_seconds = saved['elapsed_seconds']
        optimizer_steps = saved.get('optimizer_steps', first * len(train_loader))
        torch.set_rng_state(saved['rng']); generator.set_state(saved['loader_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    elif (args.run/'last.pt').exists():
        raise FileExistsError('Use --resume; never overwrite an experiment')
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    metadata = dict(architecture=cfg.get('architecture', 'dsrt'), config=cfg,
                    normalization_train_only=[sq,sv], aggregation_scale=aggregation_scale,
                    parameters=sum(p.numel() for p in model.parameters()),
                    hardware=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    train_samples=len(train), val_samples=len(val), initialized_from_pretrained=False,
                    optimizer_steps_per_epoch=len(train_loader), scheduler_type=scheduler_type)
    (args.run/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(event='training_started', **metadata)), flush=True)
    for epoch in range(first,cfg['max_epochs']):
        tic = time.perf_counter(); model.train()
        running = torch.zeros(7,device=device); seen=0
        torch.cuda.reset_peak_memory_stats()
        for batch in train_loader:
            if optimizer_steps >= cfg.get('total_optimizer_steps', float('inf')):
                break
            batch=batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=cfg['precision'] == 'bf16'):
                out=model(batch)
                if motion_loss:
                    loss, pieces = motion_balanced_objective(
                        model, out, batch, cfg['normalization']['mode_rms'],
                        velocity_weights=loss_cfg.get('velocity_weights'),
                        position_weight=float(loss_cfg.get('position_weight', 1.0)),
                        proposal_weight=cfg['proposal_weight'])
                else:
                    loss, pieces=endpoint_objective(model,out,batch.displacements,batch.update_velocities,
                                                    proposal_weight=cfg['proposal_weight'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            if scheduler_type == 'warmup_cosine':
                scheduler.step()
            n=batch.num_graphs; seen+=n
            running+=torch.stack([loss.detach(),pieces['final'],pieces['proposal'],
                                  pieces['final_position'],pieces['final_velocity'],
                                  pieces['proposal_position'],pieces['proposal_velocity']])*n
        if seen == 0:
            break
        torch.cuda.synchronize(); train_seconds=time.perf_counter()-tic
        validation=evaluate(model,val_loader,device,cfg['precision'],
                            cfg.get('selection'), dataset_system(cfg),
                            loss_cfg=loss_cfg,
                            mode_rms=cfg.get('normalization', {}).get('mode_rms'),
                            proposal_weight=cfg['proposal_weight'])
        if rollout_starts is not None:
            rollout = rollout_partition_error(
                model, rollout_starts, device,
                steps=int(selection_cfg['rollout_steps']),
                components=tuple(selection_cfg.get(
                    'partition_components', ('translation', 'rotation', 'internal'))))
            validation.update(rollout)
            weight = float(selection_cfg.get('partition_weight', 0.0))
            if selection_cfg.get('mode') == 'endpoint_rollout':
                validation['selection_loss'] = (
                    validation['endpoint_selection_loss']
                    + weight*rollout['rollout_partition_error'])
            else:
                validation['selection_loss'] = rollout['rollout_partition_error']
        score=validation['selection_loss']; improved=score<best
        if improved: best=score; stale=0
        else: stale+=1
        if scheduler_type == 'plateau':
            scheduler.step(score)
        interval = int(cfg.get('diagnostics_interval_epochs', 0))
        diagnostics = (training_diagnostics(model, diagnostic_batch, cfg['precision'],
                                            cfg['proposal_weight'])
                       if interval and ((epoch+1) == 1 or (epoch+1) % interval == 0) else None)
        record=dict(epoch=epoch+1,train_total_loss=float(running[0]/seen),
                    train_final_loss=float(running[1]/seen), train_proposal_loss=float(running[2]/seen),
                    train_normalized_position_MSE=float(running[3]/seen),
                    train_normalized_velocity_MSE=float(running[4]/seen),
                    train_proposal_normalized_position_MSE=float(running[5]/seen),
                    train_proposal_normalized_velocity_MSE=float(running[6]/seen),
                    validation=validation,train_seconds=train_seconds,epoch_seconds=time.perf_counter()-tic,
                    samples_per_second=seen/train_seconds,lr=optimizer.param_groups[0]['lr'],
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(),improved=improved,
                    optimizer_steps=optimizer_steps,diagnostics=diagnostics)
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                     config=cfg,epoch=epoch,best=best,stale=stale,record=record,
                     optimizer_steps=optimizer_steps,
                     elapsed_seconds=previous_seconds+time.time()-start,rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state(),
                     numpy_rng=np.random.get_state(),python_rng=random.getstate())
        atomic_save(payload,args.run/'last.pt')
        if improved: atomic_save(payload,args.run/'best.pt')
        with (args.run/'history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        (args.run/'status.json').write_text(json.dumps(dict(state='training',**record),indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if STOP:
            (args.run/'status.json').write_text(json.dumps(dict(state='paused',last_epoch=epoch+1))+'\n')
            return
        patience = cfg.get('early_stopping_patience')
        if patience is not None and stale>=patience: break
        if optimizer_steps >= cfg.get('total_optimizer_steps', float('inf')): break
    saved=torch.load(args.run/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'])
    # First test access occurs AFTER validation-only model selection is finished.
    test=read_graphs(Path(cfg['cache']),'test',cfg)
    result=evaluate(model,loader(test,False),device,cfg['precision'],
                    cfg.get('selection'),dataset_system(cfg),
                    loss_cfg=loss_cfg,
                    mode_rms=cfg.get('normalization', {}).get('mode_rms'),
                    proposal_weight=cfg['proposal_weight'])
    result.update(checkpoint=str(args.run/'best.pt'),selected_epoch=saved['epoch']+1,
                  selected_optimizer_steps=saved.get('optimizer_steps'),
                  actual_training_seconds=previous_seconds+time.time()-start)
    (args.run/'test_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.run/'status.json').write_text(json.dumps(dict(state='training_and_test_completed',**result),indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
