"""Mode kinetic-energy partition in time windows over an NVT trajectory.

Tests whether the partition bias is a transient from the initial condition or
a stationary property of the model's dynamics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_mode_gain import KINETIC_TO_EV, decompose, load_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--dt-fs', type=float, required=True)
    parser.add_argument('--window-ps', type=float, default=10.0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    window = load_window(args.trajectory, args.start, args.stop, args.stride)
    parts = decompose(window)
    masses = window['masses'].reshape(-1).astype(np.float64)
    frame_dt = args.dt_fs*args.stride
    frames_per_window = max(1, int(round(args.window_ps*1000/frame_dt)))
    total = parts['translation'].shape[0]
    rows = []
    for begin in range(0, total, frames_per_window):
        end = min(total, begin+frames_per_window)
        row = {'window_ps': [begin*frame_dt/1000, end*frame_dt/1000]}
        for key in ('translation', 'rotation', 'internal'):
            a = parts[key][begin:end].astype(np.float64)
            row[key+'_eV'] = float(0.5*KINETIC_TO_EV*np.einsum('n,tnc,tnc->', masses, a, a)/(end-begin))
        row['total_eV'] = sum(row[k+'_eV'] for k in ('translation', 'rotation', 'internal'))
        rows.append(row)
    result = {'label': args.label, 'trajectory': str(args.trajectory),
              'start': args.start, 'stop': args.stop, 'stride': args.stride,
              'dt_fs': args.dt_fs, 'frames': total, 'windows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'label': args.label, 'frames': total,
                      'window_ps': args.window_ps, 'rows': len(rows)}))
    for row in rows:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in row.items()}))


if __name__ == '__main__':
    main()
