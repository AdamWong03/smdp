"""Read the first frame of the independent LAMMPS NVT dump into the screen HDF5 format."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import h5py
import numpy as np


def read_first_frame(path):
    with gzip.open(path, 'rt') as handle:
        while True:
            line = handle.readline()
            if not line:
                raise EOFError('no frame')
            if line.startswith('ITEM: TIMESTEP'):
                break
        step = int(handle.readline())
        assert handle.readline().startswith('ITEM: NUMBER OF ATOMS')
        n_atoms = int(handle.readline())
        assert handle.readline().startswith('ITEM: BOX BOUNDS')
        bounds = [list(map(float, handle.readline().split())) for _ in range(3)]
        header = handle.readline()
        assert header.startswith('ITEM: ATOMS')
        columns = header.split()[2:]
        rows = [handle.readline().split() for _ in range(n_atoms)]
    data = {name: np.array([row[i] for row in rows], dtype=np.float64)
            for i, name in enumerate(columns)}
    cell = np.diag([b[1]-b[0] for b in bounds])
    return step, data, cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True,
                        help='reference HDF5 for masses/molecule order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    step, data, cell = read_first_frame(args.dump)
    order = np.argsort(data['id'].astype(np.int64))
    z = np.where(data['type'][order].astype(np.int64) == 1, 8, 1).astype(np.int16)
    molecules = data['mol'][order].astype(np.int64)
    with h5py.File(args.reference, 'r') as ref:
        masses = np.asarray(ref['masses_u'][:], dtype=np.float64)
        ref_z = np.asarray(ref['atomic_numbers'][:], dtype=np.int16)
    if not np.array_equal(z, ref_z):
        raise ValueError(f'atomic number order mismatch: {z[:6]} vs {ref_z[:6]}')
    q = np.stack([data['xu'][order], data['yu'][order], data['zu'][order]], axis=1)
    v = np.stack([data['vx'][order], data['vy'][order], data['vz'][order]], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, 'w') as out:
        out.attrs.update(source=str(args.dump), source_step=step, integration_dt_fs=0.5,
                         ensemble='NVT-CSVR', temperature_K=300.0)
        out['positions_unwrapped_A'] = q[None].astype(np.float32)
        out['velocities_A_per_fs'] = v[None].astype(np.float32)
        out['cell_A'] = cell[None]
        out['atomic_numbers'] = z
        out['masses_u'] = masses
        out['molecule_ids'] = molecules
        out['time_fs'] = np.array([0.0])
    temperature = 2*(0.5*103.64269*np.sum(masses[:, None]*v**2))/((3*len(masses)-3)*8.617333262e-5)
    print(dict(output=str(args.output), step=step, temperature_K=float(temperature),
               atoms=len(z), molecules=int(molecules.max())))


if __name__ == '__main__':
    main()
