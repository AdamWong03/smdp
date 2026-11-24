"""Quick quartz baseline vs NVT reference: species KE, RDF, VDOS, MSD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from physics.statistics import rdf_sample, spectrum_blocks

ROOT = Path('/data2/wyd/research/foundation_mlip_adaptation')
DATA = Path('/data1/wyd/research/foundation_mlip_adaptation')
BASE = ROOT / 'experiments' / 'quartz_baseline_100ps_20261122' / 'trajectory.h5'
REF = DATA / 'data/smdp_reference_md_v1/quartz/physical_reference/nvt_15ns/production.h5'
OUT = ROOT / 'experiments' / 'quartz_baseline_100ps_20261122' / 'analysis'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.trajectory) as b, h5py.File(REF) as r:
        z = b['atomic_numbers'][:].astype(int)
        cell = b['cell'][:]
        frames = len(b['positions'])
        qb = b['positions'][:].astype(float)
        dt = float(b.attrs['timestep_fs'])
        stride = int(round(dt))
        qr = r['positions_unwrapped_A'][
            5_000_000:5_000_000 + frames * stride:stride].astype(float)
        idx = np.unique(np.linspace(0, frames - 1, 512, dtype=int))
        result = {'frames': frames, 'dt_fs': dt, 'species_ke': {}}
        species_ke = b['post_thermostat_species_KE_eV'][:]
        for i, element in enumerate(b['kinetic_species'][:]):
            result['species_ke'][str(int(element))] = dict(
                mean_eV=float(species_ke[:, i].mean()),
                per_atom_eV=float(species_ke[:, i].mean() / np.sum(z == element)))
        result['temperature_K'] = dict(
            mean=float(b['temperature_K'][:].mean()),
            min=float(b['temperature_K'][:].min()),
            max=float(b['temperature_K'][:].max()))
        result['rdf'] = {}
        for first, second in [(14, 8), (8, 8), (14, 14)]:
            radius, cb = rdf_sample(qb[idx], cell, z, first, second, 6.0)
            _, cr = rdf_sample(qr[idx], cell, z, first, second, 6.0)
            result['rdf'][f'{first}-{second}'] = dict(
                rmse=float(np.sqrt(np.mean((cb.mean(0) - cr.mean(0)) ** 2))),
                peak_model=float(cb.mean(0).max()),
                peak_reference=float(cr.mean(0).max()))
        result['vdos'] = {}
        for element in [8, 14]:
            sb = spectrum_blocks(b['velocities'], z, 0, frames, 1, dt)
            sr = spectrum_blocks(r['velocities_A_per_fs'], z, 5_000_000,
                                 frames, stride, dt)
            if sb is None or sr is None:
                result['vdos'][str(element)] = dict(available=False)
                continue
            fb, pb = sb
            fr, pr = sr
            # spectrum_blocks returns per-block spectra per element; average blocks
            pb = np.asarray(pb[element]).mean(axis=0)
            pr = np.asarray(pr[element]).mean(axis=0)
            # normalize and interpolate to a common grid
            pb = pb / pb.max()
            pr = pr / pr.max()
            grid = np.linspace(0, min(fb.max(), fr.max()), 500)
            a = np.interp(grid, fb, pb)
            b_ = np.interp(grid, fr, pr)
            result['vdos'][str(element)] = dict(
                overlap=float(np.trapz(np.minimum(a, b_), grid) /
                              np.trapz(np.maximum(a, b_), grid)))
        # MSD plateau
        msd = []
        for element in [8, 14]:
            mask = z == element
            d = qb[:, mask] - qb[0, mask]
            msd.append(float(np.mean(np.sum(d[-1] ** 2, axis=-1))))
        result['msd_plateau_A2'] = {'8': msd[0], '14': msd[1]}
    (args.output / 'quartz_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
