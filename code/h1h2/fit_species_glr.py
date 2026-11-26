"""Fit species-resolved AR(p) coefficients from a quartz NVT reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-ps', type=float, default=5000.0)
    parser.add_argument('--duration-ps', type=float, default=100.0)
    parser.add_argument('--stride-fs', type=float, default=30.0)
    parser.add_argument('--order', type=int, default=10)
    args = parser.parse_args()
    with h5py.File(args.trajectory) as handle:
        dt = float(handle.attrs['output_interval_fs'])
        stride = int(round(args.stride_fs / dt))
        start = int(round(args.start_ps * 1000 / dt))
        count = int(round(args.duration_ps * 1000 / dt))
        indices = np.arange(start, start + count, stride, dtype=np.int64)
        v = handle['velocities_A_per_fs'][indices].astype(np.float64)
        z = handle['atomic_numbers'][:].astype(int)
        masses = handle['masses_u'][:].astype(np.float64)
    order = args.order
    maxlag = 40
    result = {'order': order, 'stride_fs': args.stride_fs,
              'species': sorted(int(x) for x in np.unique(z)), 'modes': {}}
    for element in result['species']:
        mask = z == element
        fields = v[:, mask]
        weights = masses[mask]
        acf = []
        for lag in range(maxlag + 1):
            if lag == 0:
                value = np.sum(weights[None, :, None] * fields * fields) / len(fields)
            else:
                value = np.sum(weights[None, :, None] * fields[:-lag] * fields[lag:]) / (len(fields) - lag)
            acf.append(float(value))
        acf = np.asarray(acf)
        acf = acf / acf[0]
        R = np.empty((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = acf[abs(i - j)]
        r = acf[1:order + 1]
        coefficients = np.linalg.solve(R, r)
        prediction_error = float(1.0 - np.dot(coefficients, r))
        ar_acf = [1.0] + [float(x) for x in r]
        for lag in range(order + 1, maxlag + 1):
            value = 0.0
            for k in range(1, order + 1):
                value += coefficients[k - 1] * ar_acf[abs(lag - k)]
            ar_acf.append(value)
        result['modes'][str(element)] = dict(
            coefficients=coefficients.tolist(),
            prediction_error_ratio=prediction_error,
            dof=3 * int(mask.sum()),
            acf=acf.tolist(), ar_acf=ar_acf)
        print(json.dumps(dict(element=int(element), dof=3 * int(mask.sum()),
                              prediction_error_ratio=round(prediction_error, 5),
                              coefficients=[round(x, 4) for x in coefficients])), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
