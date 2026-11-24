"""Build non-overlapping 5 fs pairs from the reference NVT trajectory.

The pairs are the evaluation ensemble: (q_t, v_t) -> (q_t+5fs, v_t+5fs), with
molecule-aware wrapping and minimum-image displacements.  No new labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def molecule_wrap(q, cell, molecules, masses):
    """Wrap whole molecules into the box; returns wrapped q and per-atom shift."""
    inv = np.linalg.inv(cell)
    mass = np.zeros(molecules.max()+1)
    com = np.zeros((molecules.max()+1, 3))
    np.add.at(mass, molecules, masses)
    for axis in range(3):
        np.add.at(com[:, axis], molecules, masses*q[:, axis])
    com /= mass[:, None]
    frac = com @ inv
    lattice = -np.floor(frac)
    shift = lattice @ cell
    return q + shift[molecules]


def minimum_image(d, cell):
    inv = np.linalg.inv(cell)
    frac = d @ inv
    frac -= np.round(frac)
    return frac @ cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-ps', type=float, default=250.0)
    parser.add_argument('--train-ps', type=float, default=1000.0)
    parser.add_argument('--val-ps', type=float, default=250.0)
    parser.add_argument('--test-ps', type=float, default=250.0)
    parser.add_argument('--stride-fs', type=float, default=50.0)
    args = parser.parse_args()
    with h5py.File(args.trajectory, 'r') as handle:
        cell = np.asarray(handle['cell_A'][0], dtype=np.float64)
        z = np.asarray(handle['atomic_numbers'][:], dtype=np.int16)
        masses = np.asarray(handle['masses_u'][:], dtype=np.float64)
        molecules = np.asarray(handle['molecule_ids'][:], dtype=np.int64)
        dt = float(handle.attrs['integration_dt_fs'])
        lag = int(round(5.0/dt))
        stride = int(round(args.stride_fs/dt))
        start = int(round(args.start_ps*1000/dt))
        blocks = {
            'train': (start, start + int(round(args.train_ps*1000/dt))),
            'val': (start + int(round(args.train_ps*1000/dt)),
                    start + int(round((args.train_ps+args.val_ps)*1000/dt))),
            'test': (start + int(round((args.train_ps+args.val_ps)*1000/dt)),
                     start + int(round((args.train_ps+args.val_ps+args.test_ps)*1000/dt))),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, 'w') as out:
            out.attrs.update(source=str(args.trajectory), horizon_fs=5.0,
                             stride_fs=args.stride_fs, start_ps=args.start_ps,
                             blocks=json.dumps({k: list(v) for k, v in blocks.items()}))
            out['cell_A'] = cell
            out['atomic_numbers'] = z
            out['masses_u'] = masses
            out['molecule_ids'] = molecules
            for split, (begin, end) in blocks.items():
                indices = np.arange(begin, end-lag, stride, dtype=np.int64)
                q0 = handle['positions_unwrapped_A'][indices].astype(np.float64)
                q1 = handle['positions_unwrapped_A'][indices+lag].astype(np.float64)
                v0 = handle['velocities_A_per_fs'][indices].astype(np.float64)
                v1 = handle['velocities_A_per_fs'][indices+lag].astype(np.float64)
                q0 = np.stack([molecule_wrap(q, cell, molecules, masses) for q in q0])
                q1 = np.stack([molecule_wrap(q, cell, molecules, masses) for q in q1])
                disp = np.stack([minimum_image(q1[i]-q0[i], cell) for i in range(len(q0))])
                group = out.create_group(split)
                group['positions_initial_A'] = q0.astype(np.float32)
                group['positions_final_A'] = q1.astype(np.float32)
                group['displacements_A'] = disp.astype(np.float32)
                group['velocities_initial_A_per_fs'] = v0.astype(np.float32)
                group['velocities_final_A_per_fs'] = v1.astype(np.float32)
                print(json.dumps(dict(split=split, pairs=len(indices),
                                      begin_ps=begin*dt/1000, end_ps=end*dt/1000)),
                      flush=True)
    print(json.dumps(dict(output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
