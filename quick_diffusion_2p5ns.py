"""Standalone 2.5 ns diffusion check (8 blocks, 50 fs sampling, 10-100 ps fit)."""
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


def load(path, key, masses, sample_ps=0.05, duration_ps=2500.0):
    with h5py.File(path) as handle:
        dt = float(handle.attrs.get('timestep_fs', handle.attrs.get('output_interval_fs')))
        stride = int(round(sample_ps * 1000 / dt))
        samples = int(round(duration_ps / sample_ps))
        q = handle[key][:samples * stride:stride].astype(float)
        cell = handle['cell_A'][0] if 'cell_A' in handle else handle['cell'][:]
        z = handle['atomic_numbers'][:].astype(int)
        mol = handle['molecule_ids'][:].astype(int)
    return centres(q, cell, z, masses, mol)


def analyse(values, sample_ps=0.05):
    blocks = np.array_split(values, 8)
    fits = []
    for block in blocks:
        msd = time_averaged_msd(block)
        t = np.arange(len(msd)) * sample_ps
        mask = (t >= 10) & (t <= 100)
        slope, intercept = np.polyfit(t[mask], msd[mask], 1)
        fits.append(slope / 6 * 1e-8)
    return dict(mean=float(np.mean(fits)), std=float(np.std(fits, ddof=1)), blocks=fits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-key', default='positions_unwrapped_A')
    parser.add_argument('--candidate-key', default='positions')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with h5py.File(args.candidate) as handle:
        masses = handle['masses_u'][:].astype(float)
    candidate = load(args.candidate, args.candidate_key, masses)
    reference = load(args.reference, args.reference_key, masses)
    result = dict(candidate=analyse(candidate), reference=analyse(reference))
    text = json.dumps(result, indent=2) + '\n'
    print(text)
    if args.output:
        args.output.write_text(text)


if __name__ == '__main__':
    main()
