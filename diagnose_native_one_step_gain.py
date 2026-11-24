"""One-step mode gain for native TrajCast checkpoints on held-out NVE pairs.

Same diagnostic as diagnose_one_step_gain.py, but for the upstream
EfficientTrajCastModel / AtomicGraph interface.  Compares the published water
model with our native epoch-145 checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.model.models import EfficientTrajCastModel
from trajcast.utils.atomic_computes import wrap_positions_back_to_box_torch

from motion_loss import decompose_modes

KINETIC_TO_EV = 103.64269


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--transitions', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--swap-elements', action='store_true',
                        help='map H->1, O->0 instead of H->0, O->1')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = EfficientTrajCastModel(
        config=state['resolved_native_model_config'],
        predicted_fields=['displacements', 'update_velocities'],
        rms_targets=state['rms_targets'].tolist(),
        mean_targets=[0.0, 0.0])
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    rms = state['rms_targets'].to(device)
    with h5py.File(args.transitions, 'r') as handle:
        q0 = handle['positions_initial_A'][:].astype(np.float64)
        v0 = handle['velocities_initial_A_per_fs'][:].astype(np.float64)
        q1 = handle['positions_final_A'][:].astype(np.float64)
        v1 = handle['velocities_final_A_per_fs'][:].astype(np.float64)
        cells = handle['cell_A'][:].astype(np.float64)
        z = handle['atomic_numbers'][:].astype(np.int64)
        masses = handle['masses_u'][:].astype(np.float64)
        molecules = handle['molecule_ids'][:].astype(np.int64)
    if args.limit:
        q0, v0, q1, v1, cells = q0[:args.limit], v0[:args.limit], q1[:args.limit], v1[:args.limit], cells[:args.limit]
    sums = {k: {m: 0.0 for m in ('t_norm', 'ref_norm', 'pred_norm',
                                 'cross_ref', 'cross_pred', 'res_norm')}
            for k in ('translation', 'rotation', 'internal')}
    abs_vel = 0.0
    sq_vel = 0.0
    abs_disp = 0.0
    sq_disp = 0.0
    count = 0
    with torch.no_grad():
        for index in range(len(q0)):
            cell_cpu = torch.as_tensor(cells[index])
            graph = AtomicGraph.from_atoms_dict(dict(
                positions=wrap_positions_back_to_box_torch(
                    torch.as_tensor(q0[index]), cell_cpu),
                velocities=torch.as_tensor(v0[index]),
                cell=cell_cpu,
                atomic_numbers=torch.as_tensor(z).reshape(-1, 1),
                atomic_masses=torch.as_tensor(masses).reshape(-1, 1),
                pbc=torch.tensor([True, True, True])),
                r_cut=6.0, atom_type_mapper=({1: 1, 8: 0} if args.swap_elements else {1: 0, 8: 1}))
            graph = graph.to(device)
            out = model(graph)
            target = out.target
            disp = target[:, :3]*rms[0]
            pred = target[:, 3:]*rms[1]
            q_pred = torch.as_tensor(q0[index], device=device) + disp
            masses_t = torch.as_tensor(masses)
            molecules_t = torch.as_tensor(molecules)
            cell_t = torch.as_tensor(cells[index])
            p0 = decompose_modes(torch.as_tensor(q0[index]), torch.as_tensor(v0[index]),
                                 masses_t, molecules_t, cell_t)
            pp = decompose_modes(torch.as_tensor(q0[index]), pred.cpu(),
                                 masses_t, molecules_t, cell_t)
            pt = decompose_modes(torch.as_tensor(q0[index]), torch.as_tensor(v1[index]),
                                 masses_t, molecules_t, cell_t)
            m = masses_t.double()
            for key in sums:
                a0 = p0[key].double()
                ap = pp[key].double()
                at = pt[key].double()
                entry = {
                    't_norm': torch.einsum('n,nc,nc->', m, a0, a0),
                    'ref_norm': torch.einsum('n,nc,nc->', m, at, at),
                    'pred_norm': torch.einsum('n,nc,nc->', m, ap, ap),
                    'cross_ref': torch.einsum('n,nc,nc->', m, a0, at),
                    'cross_pred': torch.einsum('n,nc,nc->', m, a0, ap),
                    'res_norm': torch.einsum('n,nc,nc->', m, at-ap, at-ap),
                }
                for name, value in entry.items():
                    sums[key][name] += float(value)
            abs_vel += float((pred.cpu()-torch.as_tensor(v1[index])).abs().sum())
            sq_vel += float((pred.cpu()-torch.as_tensor(v1[index])).square().sum())
            abs_disp += float((q_pred.cpu()-torch.as_tensor(q1[index])).abs().sum())
            sq_disp += float((q_pred.cpu()-torch.as_tensor(q1[index])).square().sum())
            count += 1
    result = {'checkpoint': str(args.checkpoint), 'samples': count, 'modes': {}}
    for key, value in sums.items():
        result['modes'][key] = {
            'alpha_true': value['cross_ref']/value['t_norm'],
            'alpha_pred': value['cross_pred']/value['t_norm'],
            'variance_ratio': value['pred_norm']/value['ref_norm'],
            'residual_over_reference': value['res_norm']/value['ref_norm'],
        }
    result['mae_velocity'] = abs_vel/(count*len(masses)*3)
    result['rmse_velocity'] = (sq_vel/(count*len(masses)*3))**0.5
    result['mae_displacement'] = abs_disp/(count*len(masses)*3)
    result['rmse_displacement'] = (sq_disp/(count*len(masses)*3))**0.5
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
