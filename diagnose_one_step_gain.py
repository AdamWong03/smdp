"""One-step mode-resolved gain/variance diagnostic on held-out NVE pairs.

Question: is the NVT partition bias already present in the one-step map on
data states, or does it only emerge during long autonomous rollouts?

For translation / rotation / internal modes we report

    alpha_x = <v_t, v_x>_m / <v_t, v_t>_m      (one-step gain)
    KE_x    = 0.5 <v_x, v_x>_m                 (mode kinetic energy)

with x = reference target or model prediction.  Mass-weighted inner products.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from motion_loss import decompose_batch
from relational_inference import load_relational
from train_relational import read_graphs

torch.set_default_dtype(torch.float32)


def mode_stats(parts, masses, molecules):
    masses = masses.double()
    stats = {}
    for key in ('translation', 'rotation', 'internal'):
        a = parts[key].double()
        norm = torch.einsum('n,nc,nc->', masses, a, a)
        stats[key] = {'norm': float(norm),
                      'ke_eV': float(0.5*103.64269*norm)}
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, saved = load_relational(args.checkpoint, device)
    model.eval()
    cfg = saved['config']
    base_cfg = cfg.get('base_config', cfg)
    graphs = read_graphs(Path(base_cfg['cache']), args.split, base_cfg)
    if args.limit:
        graphs = graphs[:args.limit]
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    sums = {k: {m: 0.0 for m in ('t_norm', 'ref_norm', 'pred_norm',
                                 'cross_ref', 'cross_pred', 'res_norm')}
            for k in ('translation', 'rotation', 'internal')}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            masses = batch.atomic_masses.reshape(-1)
            molecules = batch.molecule_ids.reshape(-1)
            v_t = batch.velocities
            v_ref = batch.update_velocities
            v_pred = output['velocities']
            p_t = decompose_batch(batch.pos, v_t, masses, molecules, batch.cell, batch.batch)
            p_ref = decompose_batch(batch.pos, v_ref, masses, molecules, batch.cell, batch.batch)
            p_pred = decompose_batch(batch.pos, v_pred, masses, molecules, batch.cell, batch.batch)
            m = masses.double()
            for key in sums:
                a_t, a_r, a_p = p_t[key].double(), p_ref[key].double(), p_pred[key].double()
                entry = {
                    't_norm': torch.einsum('n,nc,nc->', m, a_t, a_t),
                    'ref_norm': torch.einsum('n,nc,nc->', m, a_r, a_r),
                    'pred_norm': torch.einsum('n,nc,nc->', m, a_p, a_p),
                    'cross_ref': torch.einsum('n,nc,nc->', m, a_t, a_r),
                    'cross_pred': torch.einsum('n,nc,nc->', m, a_t, a_p),
                    'res_norm': torch.einsum('n,nc,nc->', m, a_r-a_p, a_r-a_p),
                }
                for name, value in entry.items():
                    sums[key][name] += float(value)
            count += batch.num_graphs
    result = {'checkpoint': str(args.checkpoint), 'split': args.split,
              'graphs': count, 'modes': {}}
    for key, value in sums.items():
        result['modes'][key] = {
            'alpha_ref': value['cross_ref']/value['t_norm'] if value['t_norm'] else None,
            'alpha_pred': value['cross_pred']/value['t_norm'] if value['t_norm'] else None,
            'ke_ref_eV': 0.5*103.64269*value['ref_norm'],
            'ke_pred_eV': 0.5*103.64269*value['pred_norm'],
            'variance_ratio': value['pred_norm']/value['ref_norm'] if value['ref_norm'] else None,
            'residual_norm_per_graph': value['res_norm']/count,
            'residual_norm_ratio': value['res_norm']/value['ref_norm'] if value['ref_norm'] else None,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
