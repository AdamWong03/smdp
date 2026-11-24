"""M0: fine-tune a trained propagator on 5 fs pairs from the reference NVT ensemble."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from periodic import PeriodicGraphBuilder
from relational_inference import load_relational, vesin_graph
from relational_transition import endpoint_objective

torch.set_default_dtype(torch.float32)


class NVTPairs(Dataset):
    def __init__(self, path, split, limit=None):
        with h5py.File(path, 'r') as handle:
            group = handle[split]
            self.q0 = group['positions_initial_A'][:limit].astype(np.float32)
            self.q1 = group['positions_final_A'][:limit].astype(np.float32)
            self.d = group['displacements_A'][:limit].astype(np.float32)
            self.v0 = group['velocities_initial_A_per_fs'][:limit].astype(np.float32)
            self.v1 = group['velocities_final_A_per_fs'][:limit].astype(np.float32)
            self.cell = np.asarray(handle['cell_A'][:], dtype=np.float32)
            self.z = np.asarray(handle['atomic_numbers'][:], dtype=np.int64)
            self.masses = np.asarray(handle['masses_u'][:], dtype=np.float32)
            self.molecules = np.asarray(handle['molecule_ids'][:], dtype=np.int64)

    def __len__(self):
        return len(self.q0)

    def __getitem__(self, index):
        return dict(q0=self.q0[index], q1=self.q1[index], d=self.d[index],
                    v0=self.v0[index], v1=self.v1[index],
                    cell=self.cell, z=self.z, masses=self.masses,
                    molecules=self.molecules)


def collate(items, model, builder, device):
    graphs = []
    for item in items:
        graph = vesin_graph(item['q0'], item['v0'], item['cell'], item['z'],
                            item['masses'], model, builder, device)
        graph.displacements = torch.as_tensor(item['d'], device=device)
        graph.update_velocities = torch.as_tensor(item['v1'], device=device)
        graph.molecule_ids = torch.as_tensor(item['molecules'], dtype=torch.long,
                                             device=device)
        graphs.append(graph)
    return Batch.from_data_list(graphs)


def evaluate(model, data, indices, batch_size, builder, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for begin in range(0, len(indices), batch_size):
            items = [data[i] for i in indices[begin:begin+batch_size]]
            batch = collate(items, model, builder, device)
            output = model(batch)
            _, pieces = endpoint_objective(model, output, batch.displacements,
                                           batch.update_velocities,
                                           proposal_weight=0.02)
            total += float(pieces['final'])*batch.num_graphs
            count += batch.num_graphs
    model.train()
    return total/max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--train-limit', type=int, default=5000)
    parser.add_argument('--val-limit', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if (args.run/'last.pt').exists():
        raise FileExistsError(args.run/'last.pt')
    args.run.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model, saved = load_relational(args.base_checkpoint, device)
    model.train()
    train = NVTPairs(args.pairs, 'train', args.train_limit)
    val = NVTPairs(args.pairs, 'val', args.val_limit)
    builder = PeriodicGraphBuilder(float(model.cfg.cutoff_angstrom))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train)/args.batch_size)
    total_epochs = math.ceil(args.steps/steps_per_epoch)

    def schedule(step):
        warm = min(500, max(1, args.steps//10))
        if step <= warm:
            return step/warm
        progress = min(1.0, (step-warm)/max(1, args.steps-warm))
        return 0.1 + 0.9*0.5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    metadata = dict(method='M0_ensemble_matched', base_checkpoint=str(args.base_checkpoint),
                    base_epoch=saved.get('epoch'), pairs=str(args.pairs),
                    train_pairs=len(train), val_pairs=len(val),
                    steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                    target='reference NVT 5 fs pairs (post-CSVR)', seed=args.seed,
                    hardware=torch.cuda.get_device_name())
    (args.run/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    order_rng = np.random.default_rng(args.seed)
    best = float('inf')
    best_epoch = -1
    step = 0
    start = time.time()
    for epoch in range(total_epochs):
        order = order_rng.permutation(len(train))
        running = 0.0
        seen = 0
        for begin in range(0, len(order), args.batch_size):
            if step >= args.steps:
                break
            items = [train[int(i)] for i in order[begin:begin+args.batch_size]]
            batch = collate(items, model, builder, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, pieces = endpoint_objective(model, output, batch.displacements,
                                              batch.update_velocities,
                                              proposal_weight=0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            running += float(loss.detach())*batch.num_graphs
            seen += batch.num_graphs
        validation = evaluate(model, val, np.arange(len(val)), args.batch_size,
                              builder, device)
        improved = validation < best
        if improved:
            best = validation
            best_epoch = epoch+1
        record = dict(epoch=epoch+1, step=step, train_loss=running/max(1, seen),
                      validation=validation, improved=improved)
        payload = dict(model=model.state_dict(), config=saved['config'],
                       base_checkpoint=str(args.base_checkpoint), epoch=epoch,
                       best=best, best_epoch=best_epoch, validation=validation,
                       elapsed_seconds=time.time()-start)
        torch.save(payload, args.run/'last.pt.tmp')
        (args.run/'last.pt.tmp').replace(args.run/'last.pt')
        if improved:
            torch.save(payload, args.run/'best.pt.tmp')
            (args.run/'best.pt.tmp').replace(args.run/'best.pt')
        with (args.run/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    print(json.dumps(dict(state='completed', best_epoch=best_epoch, best=best,
                          steps=step)), flush=True)


if __name__ == '__main__':
    main()
