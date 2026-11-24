"""Train the position head with the FD velocity readout in the loop.

The velocity is fixed by the FDR (canonical by construction); only the
displacement head is trained.  This decouples structure (learned displacement)
from thermodynamics (FD velocity) and avoids feeding the position head a
velocity it was never trained on.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from fd_readout import FDReadout
from finetune_nvt_pairs import NVTPairs
from periodic import PeriodicGraphBuilder
from relational_inference import load_relational, vesin_graph

torch.set_default_dtype(torch.float32)


def make_batch(items, fd, model, builder, device):
    graphs = []
    for item in items:
        masses = torch.as_tensor(item['masses'], device=device)
        molecules = torch.as_tensor(item['molecules'], dtype=torch.long, device=device)
        q0 = torch.as_tensor(item['q0'], device=device)
        v0 = torch.as_tensor(item['v0'], device=device)
        cell = torch.as_tensor(item['cell'], device=device)
        v_fd = fd(q0, v0, masses, molecules, cell)
        graph = vesin_graph(q0, v_fd, item['cell'], item['z'], item['masses'],
                            model, builder, device)
        graph.displacements = torch.as_tensor(item['d'], device=device)
        graph.update_velocities = torch.as_tensor(item['v1'], device=device)
        graph.molecule_ids = molecules
        graphs.append(graph)
    return Batch.from_data_list(graphs)


def evaluate(model, fd, data, builder, device, batch_size=8):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for begin in range(0, len(data), batch_size):
            items = [data[i] for i in range(begin, min(begin+batch_size, len(data)))]
            batch = make_batch(items, fd, model, builder, device)
            output = model(batch)
            loss = ((output['displacements']-batch.displacements)
                    / model.rms_targets[0]).square().mean()
            total += float(loss)*batch.num_graphs
            count += batch.num_graphs
    model.train()
    return total/max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--train-limit', type=int, default=2000)
    parser.add_argument('--val-limit', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if (args.run/'last.pt').exists():
        raise FileExistsError(args.run/'last.pt')
    args.run.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model, saved = load_relational(args.base_checkpoint, device)
    model.train()
    fd = FDReadout().to(device)
    fd.eval()
    train = NVTPairs(args.pairs, 'train', args.train_limit)
    val = NVTPairs(args.pairs, 'val', args.val_limit)
    builder = PeriodicGraphBuilder(float(model.cfg.cutoff_angstrom))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    metadata = dict(method='position_head_with_fdr', base_checkpoint=str(args.base_checkpoint),
                    base_epoch=saved.get('epoch'), pairs=str(args.pairs),
                    train_pairs=len(train), val_pairs=len(val), steps=args.steps,
                    batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                    fd_gains=fd.gains.tolist(), temperature=fd.temperature,
                    hardware=torch.cuda.get_device_name())
    (args.run/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    order_rng = np.random.default_rng(args.seed)
    best = float('inf'); best_step = -1; start = time.time()
    for step in range(1, args.steps+1):
        indices = order_rng.integers(0, len(train), size=args.batch_size)
        items = [train[int(i)] for i in indices]
        batch = make_batch(items, fd, model, builder, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = ((output['displacements']-batch.displacements)
                / model.rms_targets[0]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0 or step == 1:
            print(json.dumps(dict(step=step, loss=float(loss.detach()),
                                  elapsed=time.time()-start)), flush=True)
        if step % 100 == 0 or step == args.steps:
            validation = evaluate(model, fd, val, builder, device)
            improved = validation < best
            if improved:
                best = validation; best_step = step
            payload = dict(model=model.state_dict(), config=saved['config'],
                           base_checkpoint=str(args.base_checkpoint), epoch=step,
                           best=best, best_step=best_step, validation=validation,
                           elapsed_seconds=time.time()-start)
            torch.save(payload, args.run/'last.pt.tmp')
            (args.run/'last.pt.tmp').replace(args.run/'last.pt')
            if improved:
                torch.save(payload, args.run/'best.pt.tmp')
                (args.run/'best.pt.tmp').replace(args.run/'best.pt')
            with (args.run/'history.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=step, validation=validation,
                                             improved=improved))+'\n')
            print(json.dumps(dict(step=step, validation=validation,
                                  improved=improved)), flush=True)
    print(json.dumps(dict(state='completed', best_step=best_step, best=best)), flush=True)


if __name__ == '__main__':
    main()
