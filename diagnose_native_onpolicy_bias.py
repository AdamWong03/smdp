"""Signed one-step bias of a native TrajCast checkpoint on its own NVT states.

Native (AtomicGraph) analogue of diagnose_water_bias.py, without the relational
checkpoint loader.  Evaluation only.
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
from physics.spcfw import WaterReference
from physics.units import KB_EV_PER_K, KINETIC_TO_EV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--topology', type=Path, required=True)
    parser.add_argument('--states-per-window', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = EfficientTrajCastModel(
        config=state['resolved_native_model_config'],
        predicted_fields=['displacements', 'update_velocities'],
        rms_targets=state['rms_targets'].tolist(), mean_targets=[0.0, 0.0])
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    rms = state['rms_targets'].to(device)
    with h5py.File(args.trajectory, 'r') as handle:
        cell = np.asarray(handle['cell'][:], dtype=np.float64)
        z = np.asarray(handle['atomic_numbers'][:], dtype=np.int64)
        masses = np.asarray(handle['masses_u'][:], dtype=np.float64)
        molecules = np.asarray(handle['molecule_ids'][:], dtype=np.int64)
        dt = float(handle.attrs['timestep_fs'])
        stop = int(handle.attrs['completed_steps'])
        rng = np.random.default_rng(args.seed)
        selected = []
        for a, b in [(0, 1), (1, 5), (5, 20), (20, 50), (50, 100)]:
            candidates = np.flatnonzero((np.arange(stop)*dt/1000 >= a)
                                        & (np.arange(stop)*dt/1000 < b))
            selected.extend((int(i), f'{a}-{b}ps') for i in sorted(rng.choice(
                candidates, size=min(args.states_per_window, len(candidates)),
                replace=False)))
        calculator = WaterReference(args.topology, cell, z)
        rows = []
        try:
            for number, (index, window) in enumerate(selected):
                q = handle['positions'][index].astype(float)
                v = handle['velocities'][index].astype(float)
                tq, tv = calculator.short_trajectory(q, v)
                graph = AtomicGraph.from_atoms_dict(dict(
                    positions=wrap_positions_back_to_box_torch(
                        torch.as_tensor(q), torch.as_tensor(cell)),
                    velocities=torch.as_tensor(v), cell=torch.as_tensor(cell),
                    atomic_numbers=torch.as_tensor(z).reshape(-1, 1),
                    atomic_masses=torch.as_tensor(masses).reshape(-1, 1),
                    pbc=torch.tensor([True, True, True])),
                    r_cut=6.0, atom_type_mapper={1: 0, 8: 1}).to(device)
                with torch.no_grad():
                    target = model(graph).target
                pq = (torch.as_tensor(q, device=device) + target[:, :3]*rms[0]).cpu().numpy()
                pv = (target[:, 3:]*rms[1]).cpu().numpy()
                ke = lambda vel: .5*KINETIC_TO_EV*np.sum(masses[:, None]*np.asarray(vel)**2)
                p0 = decompose_modes(torch.as_tensor(q), torch.as_tensor(v),
                                     torch.as_tensor(masses), torch.as_tensor(molecules),
                                     torch.as_tensor(cell))
                pp = decompose_modes(torch.as_tensor(q), torch.as_tensor(pv),
                                     torch.as_tensor(masses), torch.as_tensor(molecules),
                                     torch.as_tensor(cell))
                pt = decompose_modes(torch.as_tensor(q), torch.as_tensor(tv),
                                     torch.as_tensor(masses), torch.as_tensor(molecules),
                                     torch.as_tensor(cell))
                row = dict(sample=number, window=window, frame=index,
                           input_temperature_K=2*ke(v)/((3*len(z)-3)*KB_EV_PER_K),
                           position_RMSE_A=float(np.sqrt(np.mean((pq-tq)**2))),
                           velocity_RMSE_A_fs=float(np.sqrt(np.mean((pv-tv)**2))),
                           kinetic_error_eV=float(ke(pv)-ke(tv)))
                for key in ('translation', 'rotation', 'internal'):
                    for tag, part in (('input', p0), ('pred', pp), ('true', pt)):
                        row[f'{key}_{tag}_eV'] = float(
                            0.5*KINETIC_TO_EV*torch.sum(
                                torch.as_tensor(masses).double()[:, None]
                                * part[key].double().square()))
                rows.append(row)
                if number % 32 == 0:
                    print(json.dumps(dict(sample=number, total=len(selected))), flush=True)
        finally:
            calculator.close()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'signed_errors.json').write_text(json.dumps(rows, indent=2)+'\n')
    summary = {}
    for key in ('kinetic_error_eV', 'position_RMSE_A', 'velocity_RMSE_A_fs',
                'input_temperature_K'):
        values = np.array([row[key] for row in rows])
        summary[key] = dict(mean=float(values.mean()), std=float(values.std()))
    for window in sorted(set(row['window'] for row in rows)):
        part = [row for row in rows if row['window'] == window]
        summary[window] = dict(
            n=len(part),
            kinetic_error_eV=float(np.mean([row['kinetic_error_eV'] for row in part])),
            internal_variance_ratio=float(np.mean([
                row['internal_pred_eV']/max(row['internal_true_eV'], 1e-9)
                for row in part])))
        for key in ('translation', 'rotation', 'internal'):
            summary[window][f'{key}_input_eV'] = float(np.mean(
                [row[f'{key}_input_eV'] for row in part]))
            summary[window][f'{key}_true_eV'] = float(np.mean(
                [row[f'{key}_true_eV'] for row in part]))
            summary[window][f'{key}_pred_eV'] = float(np.mean(
                [row[f'{key}_pred_eV'] for row in part]))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
