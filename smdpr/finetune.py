"""Predeclared two-step autonomous training, shared by SMDP-R and R3-Opt."""
import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from periodic import PeriodicGraphBuilder
from relational_inference import load_relational, vesin_graph
from relational_transition import endpoint_objective
from smdpr.data import InteriorLabels, TwoStepLabels, query_objective
from smdpr.model import SMDPR
from smdpr.train import write_json
from train_relational import read_graphs, atomic_save

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def two_step(model, graph, builder, proposal_weight):
    out = model(graph, return_context=isinstance(model, SMDPR))
    loss1, part1 = endpoint_objective(model, out, graph.displacements,
        graph.update_velocities, proposal_weight=proposal_weight)
    sizes = (graph.ptr[1:]-graph.ptr[:-1]).tolist()
    second = []
    start = 0
    for i, n in enumerate(sizes):
        # Topology uses detached CPU positions; features keep the prediction's
        # computation graph. No force model or truth replacement is used.
        second.append(vesin_graph(out['positions'][start:start+n], out['velocities'][start:start+n],
            graph.cell.reshape(-1, 3, 3)[i], graph.atomic_numbers[start:start+n],
            graph.atomic_masses[start:start+n], model, builder, graph.pos.device))
        start += n
    next_graph = Batch.from_data_list(second)
    out2 = model(next_graph)
    loss2, part2 = endpoint_objective(model, out2,
        graph.two_step_positions-out['positions'], graph.two_step_velocities,
        proposal_weight=proposal_weight)
    return .5*(loss1+loss2), out, part1, part2


@torch.no_grad()
def validate(model, loader, builder, device, proposal_weight):
    model.eval()
    totals = np.zeros(5)
    seen = 0
    for batch in loader:
        batch = batch.to(device)
        loss, _, a, b = two_step(model, batch, builder, proposal_weight)
        # Selection excludes auxiliary proposal loss, as in original training.
        totals += np.array([float(.5*(a['final']+b['final'])),
            float(a['final']), float(b['final']), float(a['proposal']), float(b['proposal'])])*batch.num_graphs
        seen += batch.num_graphs
    return dict(zip(['selection_loss', 'one_step_final_loss', 'two_step_final_loss',
                     'one_step_proposal_loss', 'two_step_proposal_loss'], (totals/seen).tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.run/'last.pt').exists():
        raise FileExistsError('Use --resume, never overwrite weights')
    device = torch.device('cuda')
    model, initial = load_relational(args.initial_checkpoint, device)
    cfg = initial['config'].copy()
    seed = cfg['seed']+200
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg['stage2'] = dict(total_updates=25000, initial_lr=3e-5, minimum_lr=1e-5,
        loss='mean of 2 autonomous normalized endpoint losses plus original proposal weight',
        selection='validation mean 1/2 step final endpoint loss',
        initial_checkpoint=str(args.initial_checkpoint), initial_selected_updates=initial.get('optimizer_steps'))
    write_json(args.run/'config.json', cfg)
    original_train = read_graphs(Path(cfg['cache']), 'train', cfg)
    original_val = read_graphs(Path(cfg['cache']), 'val', cfg)
    query_train = InteriorLabels(cfg['interior_labels'], 'train', original_train) if isinstance(model, SMDPR) else None
    data = Path('/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1')
    system = cfg.get('system', 'water' if cfg.get('dataset_name')=='water5' else 'quartz')
    cfg['system'] = system
    write_json(args.run/'config.json', cfg)
    transitions = data/system/'transitions'/f'{int(model.cfg.horizon_fs)}fs'
    train = TwoStepLabels(transitions/'train.h5', original_train)
    val = TwoStepLabels(transitions/'val.h5', original_val)
    if set(train.source_paths) & set(val.source_paths):
        raise ValueError('Train/validation source leakage')
    write_json(args.run/'source_selection.json', dict(train_sources=train.source_paths,
        val_sources=val.source_paths, train_starts=len(train.graphs),
        val_starts=len(val.graphs), excluded_train_start_indices=train.excluded,
        excluded_val_start_indices=val.excluded, reason='two true future endpoints must exist'))
    generator = torch.Generator().manual_seed(seed)
    query_rng = torch.Generator().manual_seed(seed+100)
    train_loader = DataLoader(train.graphs, batch_size=8, shuffle=True, generator=generator)
    val_loader = DataLoader(val.graphs, batch_size=8, shuffle=False)
    qparams = list(model.query_head.parameters()) if isinstance(model, SMDPR) else []
    ids = {id(p) for p in qparams}
    params = [p for p in model.parameters() if id(p) not in ids]
    optimizer = torch.optim.AdamW(params, lr=3e-5, weight_decay=0., amsgrad=True)
    qo = torch.optim.AdamW(qparams, lr=3e-5, weight_decay=0., amsgrad=True) if qparams else None
    builder = PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    first, steps, best, previous = 0, 0, float('inf'), 0.
    if args.resume:
        saved = torch.load(args.run/'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if qo:
            qo.load_state_dict(saved['query_optimizer'])
        first, steps, best = saved['epoch']+1, saved['optimizer_steps'], saved['best']
        generator.set_state(saved['loader_rng'])
        query_rng.set_state(saved['query_rng'])
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        previous = saved['elapsed_seconds']
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start_time = time.time()
    for epoch in range(first, 100000):
        if steps>=25000:
            break
        model.train()
        seen = 0
        total = 0.
        for graph in train_loader:
            if steps>=25000:
                break
            graph = graph.to(device)
            optimizer.zero_grad(set_to_none=True)
            if qo:
                qo.zero_grad(set_to_none=True)
            loss, out, _, _ = two_step(model, graph, builder, cfg['proposal_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite rollout loss at {steps}')
            loss.backward()
            if qo:
                times = torch.randperm(len(query_train.times)-1, generator=query_rng)[:3]
                qloss, _ = query_objective(model, graph, out, query_train, times)
                qloss.backward()
                torch.nn.utils.clip_grad_norm_(qparams, 10, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(params, 10, error_if_nonfinite=True)
            lr = 1e-5+.5*(3e-5-1e-5)*(1+math.cos(math.pi*steps/25000))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            if qo:
                for group in qo.param_groups:
                    group['lr'] = lr
                qo.step()
            steps += 1
            total += float(loss.detach())*graph.num_graphs
            seen += graph.num_graphs
        result = validate(model, val_loader, builder, device, cfg['proposal_weight'])
        improved = result['selection_loss']<best
        best = min(best, result['selection_loss'])
        record = dict(epoch=epoch+1, optimizer_steps=steps, validation=result,
            train_loss=total/seen, lr=optimizer.param_groups[0]['lr'], improved=improved)
        saved = dict(model=model.state_dict(), config=cfg, epoch=epoch, optimizer_steps=steps,
            best=best, optimizer=optimizer.state_dict(), query_optimizer=qo.state_dict() if qo else None,
            loader_rng=generator.get_state(), query_rng=query_rng.get_state(),
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            elapsed_seconds=previous+time.time()-start_time, record=record)
        atomic_save(saved, args.run/'last.pt')
        if improved:
            atomic_save(saved, args.run/'best.pt')
        with (args.run/'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        write_json(args.run/'status.json', dict(state='two_step_training', **record))
        print(json.dumps(record), flush=True)
        if STOP:
            write_json(args.run/'status.json', dict(state='paused', **record))
            return
    write_json(args.run/'status.json', dict(state='two_step_training_completed',
        optimizer_steps=steps, selected_by='validation mean 1/2 endpoint loss',
        best_selection_loss=best, test_completed=False))


if __name__ == '__main__':
    main()
