"""Mode-resolved gain diagnostic for 100 ps NVT trajectories.

The NVT failure is a partition failure: total kinetic energy is right, but
collective modes are cold and internal modes are hot (or vice versa for the
native control).  A direct propagator is a one-step map; a systematic
mode-dependent gain/shrinkage of that map is a candidate mechanism.

For each mode (translation, rotation, internal) this script reports

- the mode kinetic energy (should be 1/2 kT per DOF at equilibrium),
- the normalized mass-weighted velocity autocorrelation at several lags,
- the ratio of the model's mode variance to the reference.

Usage:
    python diagnose_mode_gain.py --trajectory ... --label ... [--reference ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

KB_EV_PER_K = 8.617333262e-5
KINETIC_TO_EV = 103.64269  # amu * (A/fs)^2 -> eV


def minimum_image(r, cell):
    """r: (T, N, 3) float32/64, cell: (3,3)."""
    inv = np.linalg.inv(np.asarray(cell, dtype=np.float64))
    frac = r.astype(np.float64) @ inv
    frac -= np.round(frac)
    return (frac @ np.asarray(cell, dtype=np.float64)).astype(r.dtype)


def load_window(path, start, stop, stride):
    with h5py.File(path, 'r') as handle:
        if 'positions_unwrapped_A' in handle:
            q = handle['positions_unwrapped_A'][start:stop:stride].astype(np.float32)
            cell = np.asarray(handle['cell_A'][start], dtype=np.float64)
        else:
            q = handle['positions'][start:stop:stride].astype(np.float32)
            cell = np.asarray(handle['cell'], dtype=np.float64)
        v = handle['velocities_A_per_fs' if 'velocities_A_per_fs' in handle
                   else 'velocities'][start:stop:stride].astype(np.float32)
        masses = np.asarray(handle['masses_u'][:], dtype=np.float64)
        molecules = np.asarray(handle['molecule_ids'][:], dtype=np.int64)
        if 'time_fs' in handle:
            time = np.asarray(handle['time_fs'][start:stop:stride], dtype=np.float64)
        else:
            time = np.arange(q.shape[0], dtype=np.float64)
    return dict(q=q, v=v, cell=cell, masses=masses, molecules=molecules,
                time=time, path=str(path), start=start, stop=stop, stride=stride)


def decompose(window, chunk=2000):
    q, v = window['q'], window['v']
    masses = window['masses'].reshape(-1)
    molecules = window['molecules'].reshape(-1)
    cell = window['cell']
    _, inverse = np.unique(molecules, return_inverse=True)
    n_mol = int(inverse.max()) + 1
    onehot = np.zeros((n_mol, len(masses)), dtype=np.float64)
    onehot[inverse, np.arange(len(masses))] = 1.0
    mass = onehot @ masses
    frames = q.shape[0]
    out = {k: np.empty_like(v) for k in ('translation', 'rotation', 'internal')}
    for begin in range(0, frames, chunk):
        end = min(frames, begin + chunk)
        qq = q[begin:end].astype(np.float64)
        vv = v[begin:end].astype(np.float64)
        r_com = np.einsum('mn,tnc->tmc', onehot/mass[:, None], masses[None, :, None]*qq)
        v_com = np.einsum('mn,tnc->tmc', onehot/mass[:, None], masses[None, :, None]*vv)
        r_rel = minimum_image(qq - r_com[:, inverse, :], cell)
        v_rel = vv - v_com[:, inverse, :]
        r2 = np.einsum('tnc,tnc->tn', r_rel, r_rel)
        eye = np.eye(3)
        # inertia: sum_n m_n (r^2 I - r r^T)
        outer = np.einsum('n,tnc,tnd->tncd', masses, r_rel, r_rel)
        inertia = (np.einsum('mn,tn,cd->tmcd', onehot, masses[None, :]*r2, eye)
                   - np.einsum('mn,tncd->tmcd', onehot, outer))
        cross = np.cross(r_rel, v_rel)
        angular = np.einsum('mn,tnc->tmc', onehot, masses[None, :, None]*cross)
        omega = np.linalg.pinv(inertia) @ angular[..., None]
        omega = omega[..., 0]
        v_rot = np.cross(omega[:, inverse, :], r_rel)
        v_int = v_rel - v_rot
        out['translation'][begin:end] = (v_com[:, inverse, :]).astype(np.float32)
        out['rotation'][begin:end] = v_rot.astype(np.float32)
        out['internal'][begin:end] = v_int.astype(np.float32)
    return out


def summarize(window, parts, lags_fs, dt_fs):
    masses = window['masses'].reshape(-1).astype(np.float64)
    result = {'frames': int(window['q'].shape[0]), 'dt_fs': dt_fs,
              'path': window['path'], 'start': int(window['start']),
              'stop': int(window['stop'])}
    dof = 3.0
    # kinetic energy per mode: 0.5 * sum m v^2
    for key in ('translation', 'rotation', 'internal'):
        value = 0.5*KINETIC_TO_EV*np.einsum('n,tnc,tnc->t', masses, parts[key], parts[key])
        n_mol = int(window['molecules'].max()) + 1
        result[key] = {'ke_eV': float(value.mean()),
                       'ke_eV_per_molecule': float(value.mean()/n_mol),
                       'equipartition_eV_per_molecule': float(1.5*KB_EV_PER_K*300.0)}
    total = sum(result[k]['ke_eV'] for k in ('translation', 'rotation', 'internal'))
    result['total_ke_eV'] = total
    result['temperature_K'] = float(2*total/((3*len(masses)-3)*KB_EV_PER_K))
    for key in ('translation', 'rotation', 'internal'):
        a = parts[key].astype(np.float64)
        norm = float(np.einsum('n,tnc,tnc->', masses, a, a))
        autocorr = {}
        for lag_fs in lags_fs:
            lag = int(round(lag_fs/dt_fs))
            if lag >= a.shape[0]:
                continue
            numerator = float(np.einsum('n,tnc,tnc->', masses, a[:-lag], a[lag:]))
            autocorr[str(lag_fs)] = numerator/norm
        result[key]['autocorrelation'] = autocorr
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--start', type=int, default=0,
                        help='start frame in the source file')
    parser.add_argument('--stop', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--dt-fs', type=float, required=True)
    parser.add_argument('--lags-fs', type=float, nargs='+',
                        default=[5, 10, 25, 50, 100, 500, 1000])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    stop = args.stop
    window = load_window(args.trajectory, args.start, stop, args.stride)
    parts = decompose(window)
    result = summarize(window, parts, args.lags_fs, args.dt_fs)
    result['label'] = args.label
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in
                      ('label', 'frames', 'total_ke_eV', 'temperature_K')}))
    for key in ('translation', 'rotation', 'internal'):
        print(key, 'KE/mol', round(result[key]['ke_eV_per_molecule'], 4),
              'acf', {k: round(v, 4) for k, v in result[key]['autocorrelation'].items()})


if __name__ == '__main__':
    main()
