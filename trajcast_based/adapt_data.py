"""Lossless official AtomicGraph adapter for the independent 16 fs splits.

The original upstream Trainer, normalization, batches and loss remain unchanged.
Preprocessing stores the exact cache format consumed by AtomicGraphDataset.
This avoids extxyz decimal rounding and preserves the actual MD atomic masses.
"""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from trajcast.data.atomic_graph import AtomicGraph
from trajcast_based.acquire import ROOT, RUN, sha


def build(source_root, out, horizon=16):
    torch.set_default_dtype(torch.float64)
    out.mkdir(parents=True, exist_ok=True)
    report, sets = {}, {}
    for split, count in [('train', 5000), ('val', 1250), ('test', 1000)]:
        source = source_root / f'quartz/transitions/{horizon}fs/{split}.h5'
        target = out / f'quartz{horizon}_{split};P:64.pt'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        with h5py.File(source) as f, ExitStack() as stack:
            assert f.attrs['horizon_fs'] == horizon and f.attrs['n_samples'] == count
            assert f.attrs['split'] == split and f.attrs['source_dt_fs'] == 1
            paths = [p.decode() if isinstance(p, bytes) else p for p in f['source_paths'][:]]
            sets[split] = set(paths)
            assert all(Path(p).parent.name.startswith(split + '_') for p in paths)
            raws = [stack.enter_context(h5py.File(p)) for p in paths]
            z, mass = f['atomic_numbers'][:], f['masses_u'][:]
            for raw in raws:
                np.testing.assert_array_equal(raw['atomic_numbers'][:], z)
                np.testing.assert_array_equal(raw['masses_u'][:], mass)
                np.testing.assert_array_equal(raw['atom_ids'][:], np.arange(1, len(z)+1))
                assert raw.attrs['ensemble'] == 'NVE'
            graphs, provenance, maxerr = [], [], {}
            def check(key, actual, expected, atol=1e-10):
                err = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
                maxerr[key] = max(err, maxerr.get(key, 0.))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=key)
            for i in range(count):
                tid, frame = int(f['source_trajectory'][i]), int(f['source_frame'][i])
                raw = raws[tid]
                q0, q1 = f['positions_initial_A'][i].astype('float64'), f['positions_final_A'][i].astype('float64')
                v0, v1 = f['velocities_initial_A_per_fs'][i].astype('float64'), f['velocities_final_A_per_fs'][i].astype('float64')
                cell = f['cell_A'][i]
                for t, q, v in [(frame, q0, v0), (frame+horizon, q1, v1)]:
                    check('raw_q', q, raw['positions_unwrapped_A'][t])
                    check('raw_v', v, raw['velocities_A_per_fs'][t])
                    check('cell', cell, raw['cell_A'][t])
                check('dt', raw['time_fs'][frame+horizon]-raw['time_fs'][frame], horizon)
                check('momentum', f['momenta_initial_u_A_per_fs'][i], mass[:, None]*v0, 5e-8)
                # Unwrap first, subtract second; never minimum-image a displacement target.
                disp = q1-q0
                check('stored_displacement', disp, f['delta_positions_A'][i], 2e-6)
                wrapped = (q0 @ np.linalg.inv(cell) % 1) @ cell
                d = dict(positions=torch.tensor(wrapped), cell=torch.tensor(cell),
                    pbc=torch.tensor([True]*3), atomic_numbers=torch.tensor(z.astype('int64')[:, None]),
                    atomic_masses=torch.tensor(mass[:, None]), velocities=torch.tensor(v0),
                    displacements=torch.tensor(disp), update_velocities=torch.tensor(v1), timestep=torch.tensor([float(horizon)]))
                g = AtomicGraph.from_atoms_dict(d, 4.5, {8: 0, 14: 1})
                check('graph_q1_unwrapped', g.pos.numpy() + g.displacements.numpy() + (q0-wrapped), q1)
                check('graph_v1', g.update_velocities.numpy(), v1)
                check('graph_mass', g.atomic_masses.numpy().ravel(), mass)
                graphs.append(g)
                provenance.append({'sample': i, 'trajectory': tid, 'frame': frame})
                if i % 500 == 0: print(split, i, flush=True)
            packed = InMemoryDataset.collate(graphs)
            torch.save(packed, target)
            report[split] = dict(samples=count, source=str(source), source_sha256=sha(source),
                source_paths=paths, cache=str(target), cache_sha256=sha(target), max_abs_error=maxerr,
                all_samples_verified=True, cell_first=cell.tolist(), masses=sorted(set(mass.tolist())))
            (out / f'{split}_provenance.json').write_text(json.dumps(provenance))
    assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    report['units'] = {'q': 'Angstrom', 'v': 'Angstrom/fs', 'mass': 'u', 'time': 'fs', 'cell': 'row vectors'}
    report['split_disjoint'] = True
    (out / 'adaptation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT/'data/smdp_reference_md_v1')
    p.add_argument('--out', type=Path, default=RUN/'independent16')
    a = p.parse_args()
    build(a.source, a.out)
