"""Rebuild only graph topology, retaining every historical physical label.

No MD generation or resplitting. Numeric verification covers every sample.
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import vesin
from torch_geometric.data import InMemoryDataset
from trajcast.data.dataset import AtomicGraphDataset


def dataset(root, split):
    return AtomicGraphDataset(root=str(root), name=f'water5_{split}',
                              cutoff_radius=6., files=[], atom_type_mapper={1: 0, 8: 1})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    old_root = args.data/'water/derived_graphs/dsrt_5fs'
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob('*.pt')):
        raise FileExistsError('New cache must be empty; inspect partial runs explicitly')
    neighbor_list = vesin.NeighborList(cutoff=6., full_list=True, sorted=True)
    report = dict(cutoff_A=6., label_changes=False, source_cache=str(old_root),
                  units=dict(position='angstrom', velocity='angstrom/fs', mass='u'), splits={})
    sources = []
    for split, count in [('train', 5000), ('val', 1250), ('test', 1000)]:
        old = dataset(old_root, split)
        assert len(old) == count
        graphs, edge_counts, max_err = [], [], {}
        with h5py.File(args.data/f'water/transitions/5fs/{split}.h5') as f:
            assert f.attrs['horizon_fs'] == 5. and f.attrs['n_samples'] == count
            source_paths = [v.decode() if isinstance(v, bytes) else str(v)
                            for v in f['source_paths'][:]]
            sources.append(set(source_paths))
            for index in range(count):
                g = old[index].clone()
                cell = g.cell.numpy().reshape(3, 3)
                q = g.pos.numpy()
                images = np.floor(q @ np.linalg.inv(cell))
                wrapped = q-images@cell
                i, j, S = neighbor_list.compute(wrapped, cell, True, quantities='ijS')
                # AtomicGraph uses neighbor-minus-central, unlike DSRT's sender API.
                shifts = (S-images[j]+images[i])@cell
                d = q[j]-q[i]+shifts
                assert np.max(np.linalg.norm(d, axis=-1)) < 6.+1e-10
                assert np.min(np.linalg.norm(d, axis=-1)) > 0.
                g.edge_index = torch.as_tensor(np.stack([i, j]), dtype=torch.long)
                g.shifts = torch.from_numpy(shifts)
                g.cutoff_radius = 6.
                # Explicitly verify all physical fields survived topology replacement.
                for field in old[index].keys():
                    if field in {'edge_index', 'shifts', 'cutoff_radius'}:
                        continue
                    a, b = old[index][field], g[field]
                    assert torch.equal(a, b) if torch.is_tensor(a) else a == b
                # Numeric sample-by-sample mapping to the formal HDF5.
                checks = {
                    'velocity': g.velocities.numpy()-f['velocities_initial_A_per_fs'][index],
                    'final_velocity': g.update_velocities.numpy()-f['velocities_final_A_per_fs'][index],
                    'displacement': g.displacements.numpy()-f['delta_positions_A'][index],
                    'mass': g.atomic_masses.numpy().ravel()-f['masses_u'][:],
                    'cell': cell-f['cell_A'][index],
                }
                qdiff = f['positions_initial_A'][index].astype(np.float64)-q
                checks['position_modulo_cell'] = qdiff-np.round(qdiff@np.linalg.inv(cell))@cell
                assert np.array_equal(g.atomic_numbers.numpy().ravel(), f['atomic_numbers'][:])
                # HDF5 stores LAMMPS type IDs; graph IDs are zero-based Z order.
                expected_types = np.array([{1: 0, 8: 1}[int(z)] for z in f['atomic_numbers'][:]])
                assert np.array_equal(g.atom_types.numpy().ravel(), expected_types)
                assert float(g.timestep) == 5.
                for key, value in checks.items():
                    error = float(np.max(np.abs(value)))
                    max_err[key] = max(max_err.get(key, 0.), error)
                    assert error < 3e-6, (split, index, key, error)
                graphs.append(g)
                edge_counts.append(g.num_edges)
                if (index+1) % 1000 == 0:
                    print(json.dumps(dict(split=split, verified=index+1)), flush=True)
        target = args.output/f'water5_{split};P:64.pt'
        torch.save(InMemoryDataset.collate(graphs), str(target)+'.tmp')
        os.replace(str(target)+'.tmp', target)
        report['splits'][split] = dict(samples=count, source_paths=source_paths,
            mean_directed_neighbors=float(np.mean(edge_counts)/192),
            maximum_mapping_error=max_err, bytes=target.stat().st_size)
        del old, graphs
    assert all(not sources[i]&sources[j] for i in range(3) for j in range(i))
    report['source_splits_disjoint'] = True
    (args.output/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
