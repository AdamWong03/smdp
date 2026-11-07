"""Formal 250k-step endpoint training plus a gradient-isolated query reader.

No test access during training. All 5,000 original starts are used every epoch.
The original architecture, loss, optimizer and step schedule remain available.
"""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from relational_transition import RelationalTransition, endpoint_objective
from smdpr.model import SMDPR
from smdpr.data import InteriorLabels, query_objective
from train_relational import read_graphs, evaluate, make_scheduler, atomic_save

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def write_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def mechanism_statistics(model, batch):
    """Fixed TRAIN batch; no optimization, no hidden state exposed to input."""
    if not isinstance(model, SMDPR):
        return None
    model.zero_grad(set_to_none=True)
    for block in model.modal_blocks:
        block.capture = True
    out = model(batch)
    loss, _ = endpoint_objective(model, out, batch.displacements,
                                 batch.update_velocities, proposal_weight=.02)
    loss.backward()
    blocks = []
    for block in model.modal_blocks:
        record = block.last_record
        affinity = record['affinity'].detach()
        attention = record['attention'].detach()
        blocks.append(dict(edge_affinity_mean=affinity.mean(0).tolist(),
            edge_affinity_std=affinity.std(0).tolist(),
            cross_mode_attention=attention.mean(0).tolist(),
            mode_RMS=record['modes'].detach().square().mean((0, 2)).sqrt().tolist(),
            scalar_modulation_RMS=float(record['scalar_modulation_RMS']),
            vector_coefficient_modulation_RMS=float(record['vector_coefficient_modulation_RMS']),
            gradient_norm=float(torch.stack([p.grad.square().sum() for p in block.parameters()
                                             if p.grad is not None]).sum().sqrt())))
        block.capture = False
        block.last_record = None
    result = dict(blocks=blocks, initial_mode_gradient_norms=
                  model.motion_state.embeddings.grad.norm(dim=-1).tolist())
    model.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((args.run/'config.json').read_text())
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    start = time.time()
    if not args.resume and ((args.run/'last.pt').exists() or (args.run/'history.jsonl').exists()):
        raise FileExistsError('Never overwrite an existing experiment; use --resume')
    train = read_graphs(Path(cfg['cache']), 'train', cfg)
    val = read_graphs(Path(cfg['cache']), 'val', cfg)
    scales = [float(torch.cat([getattr(g, field) for g in train]).square().mean().sqrt())
              for field in ['displacements', 'update_velocities']]
    model = (SMDPR(cfg['model'], scales, **cfg['motion'])
             if cfg['architecture']=='smdpr' else RelationalTransition(cfg['model'], scales)).to(device)
    has_query = isinstance(model, SMDPR) and model.query_head is not None
    query_parameters = list(model.query_head.parameters()) if has_query else []
    query_ids = {id(p) for p in query_parameters}
    main_parameters = [p for p in model.parameters() if id(p) not in query_ids]
    optimizer = torch.optim.AdamW(main_parameters, lr=cfg['learning_rate'],
                                  weight_decay=0., amsgrad=True)
    scheduler, kind = make_scheduler(optimizer, cfg)
    query_optimizer = query_scheduler = None
    train_labels = val_labels = None
    if has_query:
        train_labels = InteriorLabels(cfg['interior_labels'], 'train', train)
        val_labels = InteriorLabels(cfg['interior_labels'], 'val', val)
        if set(train_labels.provenance['source_paths']) & set(val_labels.provenance['source_paths']):
            raise ValueError('Source trajectory leakage')
        query_optimizer = torch.optim.AdamW(query_parameters, lr=cfg['learning_rate'],
                                            weight_decay=0., amsgrad=True)
        query_scheduler, _ = make_scheduler(query_optimizer, cfg)
    # Separate RNG for query sampling; it must not change original graph shuffling.
    generator = torch.Generator().manual_seed(seed)
    query_generator = torch.Generator().manual_seed(seed+100)
    def loader(graphs, shuffle):
        return DataLoader(graphs, batch_size=cfg['batch_size'], shuffle=shuffle,
                          num_workers=0, pin_memory=True, generator=generator if shuffle else None)
    train_loader, val_loader = loader(train, True), loader(val, False)
    diagnostic_batch = next(iter(loader(train[:cfg['batch_size']], False))).to(device)
    first, steps, best, previous_seconds = 0, 0, float('inf'), 0.
    if args.resume:
        saved = torch.load(args.run/'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        if has_query:
            query_optimizer.load_state_dict(saved['query_optimizer'])
            query_scheduler.load_state_dict(saved['query_scheduler'])
        first, steps, best = saved['epoch']+1, saved['optimizer_steps'], saved['best']
        previous_seconds = saved['elapsed_seconds']
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        generator.set_state(saved['loader_rng'])
        query_generator.set_state(saved['query_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    baseline = RelationalTransition(cfg['model'])
    metadata = dict(config=cfg, normalization_train_only=scales, parameters=sum(p.numel() for p in model.parameters()),
        baseline_parameters=sum(p.numel() for p in baseline.parameters()),
        query_parameters=sum(p.numel() for p in query_parameters), hardware=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__,
        train_samples=len(train), val_samples=len(val), initialized_from_pretrained=False,
        query_labels=train_labels.provenance if has_query else None,
        query_backbone_gradient='blocked by detach; separate optimizer',
        model_selection='validation endpoint only; test not loaded',
        steps_per_epoch=len(train_loader))
    del baseline
    write_json(args.run/'metadata.json', metadata)
    print(json.dumps(dict(event='formal_training_started', **metadata)), flush=True)
    for epoch in range(first, cfg['max_epochs']):
        if steps >= cfg['total_optimizer_steps']:
            break
        tic = time.perf_counter()
        model.train()
        running = torch.zeros(8, device=device)
        seen = 0
        torch.cuda.reset_peak_memory_stats()
        for batch in train_loader:
            if steps >= cfg['total_optimizer_steps']:
                break
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if has_query:
                query_optimizer.zero_grad(set_to_none=True)
            out = model(batch, return_context=has_query)
            loss, pieces = endpoint_objective(model, out, batch.displacements,
                batch.update_velocities, proposal_weight=cfg['proposal_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite endpoint loss at {steps}')
            loss.backward()
            # All query inputs were detached inside query(); this backward
            # cannot alter the already-computed endpoint gradients.
            query_loss = loss.new_zeros(())
            if has_query:
                count = len(train_labels.times)-1
                indices = torch.randperm(count, generator=query_generator)[:min(3, count)]
                query_loss, _ = query_objective(model, batch, out, train_labels, indices)
                if not torch.isfinite(query_loss):
                    raise FloatingPointError(f'Nonfinite query loss at {steps}')
                query_loss.backward()
                torch.nn.utils.clip_grad_norm_(query_parameters, cfg['gradient_clip'], error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(main_parameters, cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if has_query:
                query_optimizer.step()
                query_scheduler.step()
            steps += 1
            n = batch.num_graphs
            seen += n
            running += torch.stack([loss.detach(), pieces['final'], pieces['proposal'],
                pieces['final_position'], pieces['final_velocity'], pieces['proposal_position'],
                pieces['proposal_velocity'], query_loss.detach()])*n
            if steps % 100 == 0:
                write_json(args.run/'progress.json', dict(state='training', epoch=epoch+1,
                    optimizer_steps=steps, total_optimizer_steps=cfg['total_optimizer_steps'],
                    elapsed_seconds=previous_seconds+time.time()-start,
                    last_endpoint_loss=float(pieces['final']), lr=optimizer.param_groups[0]['lr']))
        torch.cuda.synchronize()
        train_seconds = time.perf_counter()-tic
        validation = evaluate(model, val_loader, device)
        if has_query:
            query_total = 0.
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch, True)
                    indices = torch.arange(len(val_labels.times)-1)
                    query_loss, _ = query_objective(model, batch, out, val_labels, indices)
                    query_total += float(query_loss)*batch.num_graphs
            validation['detached_query_normalized_loss'] = query_total/len(val)
        improved = validation['selection_loss'] < best
        if improved:
            best = validation['selection_loss']
        interval = cfg['diagnostics_interval_epochs']
        diagnostics = mechanism_statistics(model, diagnostic_batch) if epoch==0 or (epoch+1)%interval==0 else None
        values = (running/seen).tolist()
        record = dict(epoch=epoch+1, optimizer_steps=steps, train_total_loss=values[0],
            train_final_loss=values[1], train_proposal_loss=values[2],
            train_normalized_position_MSE=values[3], train_normalized_velocity_MSE=values[4],
            train_proposal_normalized_position_MSE=values[5],
            train_proposal_normalized_velocity_MSE=values[6], train_query_loss=values[7],
            validation=validation, lr=optimizer.param_groups[0]['lr'],
            train_seconds=train_seconds, epoch_seconds=time.perf_counter()-tic,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), improved=improved, diagnostics=diagnostics)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), query_optimizer=query_optimizer.state_dict() if has_query else None,
            query_scheduler=query_scheduler.state_dict() if has_query else None,
            config=cfg, epoch=epoch, optimizer_steps=steps, best=best, record=record,
            elapsed_seconds=previous_seconds+time.time()-start, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), loader_rng=generator.get_state(),
            query_rng=query_generator.get_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
        atomic_save(payload, args.run/'last.pt')
        if improved:
            atomic_save(payload, args.run/'best.pt')
        with (args.run/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        write_json(args.run/'status.json', dict(state='training', **record))
        print(json.dumps(record), flush=True)
        if STOP:
            write_json(args.run/'status.json', dict(state='paused', **record))
            return
    write_json(args.run/'status.json', dict(state='endpoint_training_completed',
        optimizer_steps=steps, selected_by='validation endpoint only',
        best_selection_loss=best, elapsed_seconds=previous_seconds+time.time()-start,
        test_completed=False, physical_analysis_completed=False))


if __name__ == '__main__':
    main()
