"""Fine-tune a mode-aware correction with a differentiable 2-step rollout.

The endpoint-only fine-tune was null: the base is endpoint-optimal, so endpoint
loss gives the correction no gradient toward the NVT partition.  Here the
correction is trained on a short rollout and a mode partition loss at the final
reference state.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from mode_aware import ModeAwareTransition
from motion_loss import decompose_modes, mode_mse
from periodic import PeriodicGraphBuilder
from relational_inference import vesin_graph
from relational_transition import RelationalTransition, endpoint_objective
from rollout_data import build_sequences

torch.set_default_dtype(torch.float32)


def make_graph(q, v, cell, z, masses, molecules, model, builder, device):
    graph = vesin_graph(torch.as_tensor(q, device=device),
                        torch.as_tensor(v, device=device),
                        torch.as_tensor(cell, device=device),
                        torch.as_tensor(z, device=device),
                        torch.as_tensor(masses, device=device),
                        model, builder, device)
    graph.molecule_ids = torch.as_tensor(molecules, dtype=torch.long, device=device)
    return graph


def rollout_loss(model, sequences, builder, device, mode_rms, weights,
                 proposal_weight=0.0):
    graphs = [make_graph(s['q'][0], s['v'][0], s['cell'], s['z'], s['masses'],
                         s['molecules'], model, builder, device) for s in sequences]
    batch = Batch.from_data_list(graphs)
    out1 = model(batch)
    endpoints = []
    partitions = []
    for index, sequence in enumerate(sequences):
        mask = batch.batch == index
        q1 = torch.as_tensor(sequence['q'][1], device=device)
        v1 = torch.as_tensor(sequence['v'][1], device=device)
        q2 = torch.as_tensor(sequence['q'][2], device=device)
        v2 = torch.as_tensor(sequence['v'][2], device=device)
        q1p, v1p = out1['positions'][mask], out1['velocities'][mask]
        endpoint1 = 0.5*(((q1p-q1)/model.rms_targets[0]).square().mean()
                         + ((v1p-v1)/model.rms_targets[1]).square().mean())
        graph2 = make_graph(q1p, v1p, sequence['cell'], sequence['z'],
                            sequence['masses'], sequence['molecules'],
                            model, builder, device)
        out2 = model(graph2)
        q2p, v2p = out2['positions'], out2['velocities']
        endpoint2 = 0.5*(((q2p-q2)/model.rms_targets[0]).square().mean()
                         + ((v2p-v2)/model.rms_targets[1]).square().mean())
        pred = decompose_modes(q2p, v2p,
                               torch.as_tensor(sequence['masses'], device=device),
                               torch.as_tensor(sequence['molecules'], device=device),
                               torch.as_tensor(sequence['cell'], device=device))
        ref = decompose_modes(q2, v2,
                              torch.as_tensor(sequence['masses'], device=device),
                              torch.as_tensor(sequence['molecules'], device=device),
                              torch.as_tensor(sequence['cell'], device=device))
        partition, _ = mode_mse(pred, ref, mode_rms, weights)
        endpoints.append((endpoint1, endpoint2))
        partitions.append(partition)
    ep1 = torch.stack([x[0] for x in endpoints]).mean()
    ep2 = torch.stack([x[1] for x in endpoints]).mean()
    part = torch.stack(partitions).mean()
    total = 0.1*ep1 + ep2 + part
    return total, dict(endpoint1=ep1.detach(), endpoint2=ep2.detach(),
                       partition=part.detach())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--train-sequences', type=int, default=256)
    p.add_argument('--val-sequences', type=int, default=32)
    p.add_argument('--seed', type=int, default=20261121)
    args = p.parse_args()
    if (args.run/'last.pt').exists():
        raise FileExistsError(args.run/'last.pt')
    args.run.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    saved = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']
    normalization = cfg.get('normalization') or {}
    base = RelationalTransition(cfg['model'],
                                rms_targets=normalization.get('rms_targets', [0.0561, 0.01287]))
    base.load_state_dict(saved['model'], strict=True)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    model = ModeAwareTransition(base).to(device)
    data_root = Path(cfg['cache']).parent.parent
    train = build_sequences(data_root, 'train', args.train_sequences, args.seed)
    val = build_sequences(data_root, 'val', args.val_sequences, args.seed + 1)
    mode_rms = {'translation': 0.006254159458705605,
                'rotation': 0.013886540547892876,
                'internal': 0.01608279438434204}
    weights = {'translation': 0.5, 'rotation': 0.5, 'internal': 0.5}
    builder = PeriodicGraphBuilder(float(model.cfg.cutoff_angstrom))
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=args.lr,
                                  weight_decay=0.0)
    metadata = dict(architecture='mode_aware_finetune', base_checkpoint=str(args.base_checkpoint),
                    base_epoch=saved.get('epoch'), base_config=cfg,
                    mode_rms=mode_rms, mode_weights=weights,
                    train_sequences=len(train), val_sequences=len(val),
                    steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                    hardware=torch.cuda.get_device_name(),
                    loss='0.1*endpoint1 + endpoint2 + partition2')
    (args.run/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    best = float('inf'); best_epoch = -1; step = 0; start = time.time()
    for epoch in range(math.ceil(args.steps/(len(train)//args.batch_size))):
        model.train(); running = {k: 0.0 for k in ('total', 'endpoint1', 'endpoint2', 'partition')}
        seen = 0
        order = np.random.default_rng(args.seed+epoch).permutation(len(train))
        for offset in range(0, len(order), args.batch_size):
            if step >= args.steps:
                break
            sequences = [train[int(i)] for i in order[offset:offset+args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            total, pieces = rollout_loss(model, sequences, builder, device,
                                         mode_rms, weights)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 10.0,
                                           error_if_nonfinite=True)
            optimizer.step(); step += 1; seen += len(sequences)
            running['total'] += float(total.detach())*len(sequences)
            for key in ('endpoint1', 'endpoint2', 'partition'):
                running[key] += float(pieces[key])*len(sequences)
        # validation rollout
        model.eval()
        with torch.no_grad():
            vtotal = {k: 0.0 for k in ('total', 'endpoint1', 'endpoint2', 'partition')}
            for offset in range(0, len(val), args.batch_size):
                sequences = val[offset:offset+args.batch_size]
                total, pieces = rollout_loss(model, sequences, builder, device,
                                             mode_rms, weights)
                vtotal['total'] += float(total)*len(sequences)
                for key in ('endpoint1', 'endpoint2', 'partition'):
                    vtotal[key] += float(pieces[key])*len(sequences)
            vtotal = {k: v/len(val) for k, v in vtotal.items()}
        score = vtotal['partition'] + 0.1*vtotal['endpoint2']
        improved = score < best
        if improved:
            best = score; best_epoch = epoch+1
        record = dict(epoch=epoch+1, step=step, train={k: v/max(1, seen) for k, v in running.items()},
                      validation=vtotal, score=score, improved=improved)
        payload = dict(model=model.state_dict(),
                       config=dict(architecture='mode_aware_finetune', base_config=cfg),
                       base_checkpoint=str(args.base_checkpoint), epoch=epoch,
                       best=best, best_epoch=best_epoch, validation=vtotal,
                       elapsed_seconds=time.time()-start)
        torch.save(payload, args.run/'last.pt.tmp'); (args.run/'last.pt.tmp').replace(args.run/'last.pt')
        if improved:
            torch.save(payload, args.run/'best.pt.tmp'); (args.run/'best.pt.tmp').replace(args.run/'best.pt')
        (args.run/'history.jsonl').open('a').write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    print(json.dumps(dict(state='completed', best_epoch=best_epoch, best=best, steps=step)), flush=True)


if __name__ == '__main__':
    main()
