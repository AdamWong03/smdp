"""Green-Kubo diffusion from the mass-weighted molecular translation ACF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from motion_loss import decompose_modes

KINETIC_TO_EV = 103.64269


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--positions-key', default='positions')
    parser.add_argument('--velocities-key', default='velocities')
    parser.add_argument('--start-ps', type=float, default=0.0)
    parser.add_argument('--duration-ps', type=float, default=100.0)
    parser.add_argument('--stride-fs', type=float, default=5.0)
    parser.add_argument('--max-lag-fs', type=float, default=5000.0)
    parser.add_argument('--label', default='candidate')
    args = parser.parse_args()
    with h5py.File(args.trajectory) as handle:
        dt = float(handle.attrs.get('timestep_fs', handle.attrs.get('output_interval_fs')))
        stride = int(round(args.stride_fs / dt))
        start = int(round(args.start_ps * 1000 / dt))
        frames = int(round(args.duration_ps * 1000 / args.stride_fs))
        indices = start + np.arange(frames) * stride
        q = handle[args.positions_key][indices].astype(np.float64)
        v = handle[args.velocities_key][indices].astype(np.float64)
        cell = handle['cell_A'][0] if 'cell_A' in handle else handle['cell'][:]
        z = handle['atomic_numbers'][:].astype(int)
        masses = handle['masses_u'][:].astype(float)
        molecules = handle['molecule_ids'][:].astype(int)
    translation = []
    for index in range(len(q)):
        part = decompose_modes(torch.as_tensor(q[index]), torch.as_tensor(v[index]),
                               torch.as_tensor(masses), torch.as_tensor(molecules),
                               torch.as_tensor(cell))
        translation.append(part['translation'].numpy())
    fields = np.stack(translation)
    weights = masses[None, :, None]
    maxlag = int(round(args.max_lag_fs / args.stride_fs))
    acf = []
    for lag in range(maxlag + 1):
        if lag == 0:
            value = np.sum(weights * fields * fields) / len(fields)
        else:
            value = np.sum(weights * fields[:-lag] * fields[lag:]) / (len(fields) - lag)
        acf.append(float(value))
    acf = np.asarray(acf)
    # mass-weighted per-atom translation ACF; normalize by the zero-lag value
    normalized = acf / acf[0]
    # Green-Kubo: D = (1/3) ∫ <v(0)·v(t)> dt.  <v(0)·v(t)> is the
    # mass-weighted average per atom, i.e. acf(t) / sum(m).
    vacuum = acf / masses.sum()
    lags_fs = np.arange(len(vacuum)) * args.stride_fs
    # 1 Å²/fs = 1e-5 m²/s
    diffusion = float(np.trapz(vacuum, lags_fs) / 3.0 * 1e-5)
    print(json.dumps(dict(
        label=args.label,
        frames=len(q),
        duration_ps=args.duration_ps,
        max_lag_fs=args.max_lag_fs,
        zero_lag_A2_fs2=float(acf[0]),
        normalized_5fs=float(normalized[1]) if len(normalized) > 1 else None,
        normalized_100fs=float(normalized[int(round(100 / args.stride_fs))])
        if len(normalized) > int(round(100 / args.stride_fs)) else None,
        normalized_500fs=float(normalized[int(round(500 / args.stride_fs))])
        if len(normalized) > int(round(500 / args.stride_fs)) else None,
        green_kubo_D_m2_s=diffusion,
    ), indent=2))


if __name__ == '__main__':
    main()
