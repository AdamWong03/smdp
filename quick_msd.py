"""Quick molecular-COM MSD on a short trajectory (for 100 ps diagnostics).

The production diffusion task needs 100 ps blocks; this script uses one block and a
shorter fit window, so it is a diagnostic only, not a paper number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from physics.diffusion import time_averaged_msd
from physics.water_motion import molecular_geometry


def centres(values, cell, z, m, mol):
    ids, bonds, _, _ = molecular_geometry(values, cell, z, mol)
    local = m[ids[:, 1:], None] * bonds
    com = values[:, ids[:, 0]] + local.sum(axis=-2) / m[ids].sum(axis=-1)[:, None]
    return com - com.mean(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--positions-key', default='positions')
    parser.add_argument('--sample-ps', type=float, default=0.05)
    parser.add_argument('--start-ps', type=float, default=0.0)
    parser.add_argument('--duration-ps', type=float, default=100.0)
    parser.add_argument('--fit-start-ps', type=float, default=10.0)
    parser.add_argument('--fit-stop-ps', type=float, default=80.0)
    parser.add_argument('--label', default='candidate')
    args = parser.parse_args()

    with h5py.File(args.trajectory) as handle:
        dt = float(handle.attrs.get('timestep_fs',
                                    handle.attrs.get('output_interval_fs')))
        stride = int(round(args.sample_ps * 1000 / dt))
        start = int(round(args.start_ps * 1000 / dt))
        samples = int(round(args.duration_ps / args.sample_ps))
        stop = start + samples * stride
        q = handle[args.positions_key][start:stop:stride].astype(float)
        cell = handle['cell_A'][0] if 'cell_A' in handle else handle['cell'][:]
        z = handle['atomic_numbers'][:].astype(int)
        m = handle['masses_u'][:].astype(float)
        mol = handle['molecule_ids'][:].astype(int)
    com = centres(q, cell, z, m, mol)
    msd = time_averaged_msd(com)
    t = np.arange(len(msd)) * args.sample_ps
    mask = (t >= args.fit_start_ps) & (t <= args.fit_stop_ps)
    slope, intercept = np.polyfit(t[mask], msd[mask], 1)
    residual = msd[mask] - (slope * t[mask] + intercept)
    denominator = np.sum((msd[mask] - msd[mask].mean()) ** 2)
    print(json.dumps(dict(
        label=args.label,
        frames=len(q),
        duration_ps=float(t[-1]),
        fit_window_ps=[args.fit_start_ps, args.fit_stop_ps],
        slope_A2_ps=float(slope),
        diffusion_m2_s=float(slope / 6 * 1e-8),
        R2=float(1 - np.sum(residual ** 2) / denominator),
        msd_10ps_A2=float(np.interp(10, t, msd)),
        msd_50ps_A2=float(np.interp(50, t, msd)),
        msd_100ps_A2=float(np.interp(min(100, t[-1]), t, msd)),
    ), indent=2))


if __name__ == '__main__':
    main()
