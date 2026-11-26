"""Fit mode-resolved AR(p) coefficients from the reference NVT trajectory.

Yule-Walker on the mass-weighted mode velocity ACF.  The AR(p) prediction
error variance is the FD-consistent noise variance: it preserves the target
stationary mode KE.
"""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-ps', type=float, default=500.0)
    parser.add_argument('--duration-ps', type=float, default=100.0)
    parser.add_argument('--stride-fs', type=float, default=5.0)
    parser.add_argument('--order', type=int, default=10)
    parser.add_argument('--maxlag', type=int, default=40)
    args = parser.parse_args()
    with h5py.File(args.trajectory, 'r') as handle:
        # Stored cadence, not the integration step: some reference files use
        # 0.5 fs integration but 1.0 fs output.  Fall back to the integration
        # step for older files without output_interval_fs.
        dt = float(handle.attrs.get(
            'output_interval_fs',
            handle.attrs.get('integration_dt_fs', args.stride_fs)))
        stride = int(round(args.stride_fs/dt))
        start = int(round(args.start_ps*1000/dt))
        count = int(round(args.duration_ps*1000/dt))
        indices = np.arange(start, start+count, stride, dtype=np.int64)
        q = handle['positions_unwrapped_A'][indices].astype(np.float64)
        v = handle['velocities_A_per_fs'][indices].astype(np.float64)
        cell = np.asarray(handle['cell_A'][0], dtype=np.float64)
        z = np.asarray(handle['atomic_numbers'][:], dtype=np.int64)
        masses = np.asarray(handle['masses_u'][:], dtype=np.float64)
        molecules = np.asarray(handle['molecule_ids'][:], dtype=np.int64)
    keys = ('translation', 'rotation', 'internal')
    parts = {key: [] for key in keys}
    for index in range(len(indices)):
        part = decompose_modes(torch.as_tensor(q[index]), torch.as_tensor(v[index]),
                               torch.as_tensor(masses), torch.as_tensor(molecules),
                               torch.as_tensor(cell))
        for key in keys:
            parts[key].append(part[key].numpy().astype(np.float32))
    order = args.order
    maxlag = args.maxlag
    result = {'order': order, 'stride_fs': args.stride_fs, 'masses_u': masses.tolist(),
              'modes': {}}
    for key in keys:
        fields = np.stack(parts[key]).astype(np.float64)  # (T,N,3)
        weights = masses[None, :, None]
        acf = []
        for lag in range(maxlag+1):
            if lag == 0:
                value = np.sum(weights*fields*fields)
            else:
                value = np.sum(weights*fields[:-lag]*fields[lag:])
            acf.append(float(value))
        zero_lag = float(acf[0])
        acf = np.array(acf)/acf[0]
        R = np.empty((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = acf[abs(i-j)]
        r = acf[1:order+1]
        coefficients = np.linalg.solve(R, r)
        prediction_error = float(1.0 - np.dot(coefficients, r))
        ar_acf = [1.0] + [float(x) for x in r]
        for lag in range(order+1, maxlag+1):
            value = 0.0
            for k in range(1, order+1):
                value += coefficients[k-1]*ar_acf[abs(lag-k)]
            ar_acf.append(value)
        lags_fs = np.arange(maxlag + 1) * args.stride_fs
        per_atom_zero = zero_lag / len(fields) / masses.sum()
        empirical_d = float(np.trapezoid(acf * per_atom_zero, lags_fs) / 3.0 * 1e-5)
        ar_d = float(np.trapezoid(np.asarray(ar_acf) * per_atom_zero, lags_fs)
                     / 3.0 * 1e-5)
        result['modes'][key] = dict(
            acf=acf.tolist(), ar_acf=ar_acf,
            coefficients=coefficients.tolist(),
            prediction_error_ratio=prediction_error,
            zero_lag_A2_fs2=per_atom_zero,
            empirical_green_kubo_D_m2_s=empirical_d,
            ar_green_kubo_D_m2_s=ar_d)
        print(json.dumps(dict(mode=key, coefficients=[round(x, 5) for x in coefficients],
                              prediction_error_ratio=round(prediction_error, 5),
                              acf=[round(x, 4) for x in acf[:8]],
                              ar_acf=[round(x, 4) for x in ar_acf[:8]])), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
