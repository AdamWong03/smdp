"""Mode-resolved gain on on-policy (autonomously visited) states.

Reads the evaluation-only labels written by diagnose_water_bias.py
(diagnostic_labels.h5), recomputes the model prediction at each visited state,
and compares the one-step mode gain/variance with the SPC/Fw continuation.

This quantifies the covariate shift: the same model that is accurate on
held-out NVE pairs should be biased on its own NVT states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

from motion_loss import decompose_modes
from periodic import PeriodicGraphBuilder
from relational_inference import load_relational, vesin_graph

KINETIC_TO_EV = 103.64269


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True,
                        help='source trajectory for cell/masses/molecule_ids')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, saved = load_relational(args.checkpoint, device)
    model.eval()
    builder = PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    with h5py.File(args.trajectory, 'r') as handle:
        cell = np.asarray(handle['cell'][:], dtype=np.float64)
        masses = np.asarray(handle['masses_u'][:], dtype=np.float64)
        molecules = np.asarray(handle['molecule_ids'][:], dtype=np.int64)
        z = np.asarray(handle['atomic_numbers'][:], dtype=np.int64)
    groups = []
    with h5py.File(args.labels, 'r') as handle:
        names = sorted(k for k in handle.keys() if k.isdigit())
        for name in names:
            group = handle[name]
            groups.append(dict(
                window=group.attrs['window'],
                source_frame=int(group.attrs['source_frame']),
                q0=group['input_q_A'][:], v0=group['input_v_A_fs'][:],
                q1=group['true_q_A'][:], v1=group['true_v_A_fs'][:]))
    sums = {k: {m: 0.0 for m in ('t_norm', 'pred_norm', 'true_norm',
                                 'cross_pred', 'cross_true')}
            for k in ('translation', 'rotation', 'internal')}
    windows = {}
    for item in groups:
        graph = vesin_graph(item['q0'], item['v0'], cell, z, masses, model,
                            builder, device)
        graph.molecule_ids = torch.as_tensor(molecules, dtype=torch.long)
        with torch.inference_mode():
            out = model(graph)
        v_pred = out['velocities'].cpu().numpy()
        p0 = decompose_modes(torch.as_tensor(item['q0']), torch.as_tensor(item['v0']),
                             torch.as_tensor(masses), torch.as_tensor(molecules),
                             torch.as_tensor(cell))
        pp = decompose_modes(torch.as_tensor(item['q0']), torch.as_tensor(v_pred),
                             torch.as_tensor(masses), torch.as_tensor(molecules),
                             torch.as_tensor(cell))
        pt = decompose_modes(torch.as_tensor(item['q0']), torch.as_tensor(item['v1']),
                             torch.as_tensor(masses), torch.as_tensor(molecules),
                             torch.as_tensor(cell))
        m = torch.as_tensor(masses).double()
        for key in sums:
            a0 = p0[key].double().reshape(-1, 3)
            ap = pp[key].double().reshape(-1, 3)
            at = pt[key].double().reshape(-1, 3)
            entry = {
                't_norm': torch.einsum('n,nc,nc->', m, a0, a0),
                'pred_norm': torch.einsum('n,nc,nc->', m, ap, ap),
                'true_norm': torch.einsum('n,nc,nc->', m, at, at),
                'cross_pred': torch.einsum('n,nc,nc->', m, a0, ap),
                'cross_true': torch.einsum('n,nc,nc->', m, a0, at),
            }
            for name, value in entry.items():
                sums[key][name] += float(value)
            bucket = windows.setdefault(item['window'], {k: {m: 0.0 for m in entry}
                                                          for k in sums})
            for name, value in entry.items():
                bucket[key][name] += float(value)
    count = len(groups)
    result = {'labels': str(args.labels), 'samples': count, 'modes': {}}
    for key, value in sums.items():
        result['modes'][key] = {
            'alpha_true': value['cross_true']/value['t_norm'],
            'alpha_pred': value['cross_pred']/value['t_norm'],
            'variance_ratio': value['pred_norm']/value['true_norm'],
            'ke_input_eV': 0.5*KINETIC_TO_EV*value['t_norm']/count,
            'ke_true_eV': 0.5*KINETIC_TO_EV*value['true_norm']/count,
            'ke_pred_eV': 0.5*KINETIC_TO_EV*value['pred_norm']/count,
        }
    result['windows'] = {}
    for name, bucket in windows.items():
        n = sum(1 for item in groups if item['window'] == name)
        result['windows'][name] = {key: {
            'alpha_true': value['cross_true']/value['t_norm'],
            'alpha_pred': value['cross_pred']/value['t_norm'],
            'variance_ratio': value['pred_norm']/value['true_norm'],
        } for key, value in bucket.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
