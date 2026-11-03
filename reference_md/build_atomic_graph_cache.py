"""Build verified AtomicGraph caches from the formal transition HDF5 files.

Every selected endpoint is checked against its independent raw NVE source.
The output is the exact upstream TrajCast AtomicGraphDataset cache format used
by DSRT training; no coordinate rounding or new molecular dynamics is involved.
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


SPLIT_COUNTS = {'train': 5000, 'val': 1250, 'test': 1000}


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def build(source_root, system, horizon_fs, dataset_name, out, cutoff):
    torch.set_default_dtype(torch.float64)
    out.mkdir(parents=True, exist_ok=True)
    reports, source_sets = {}, {}
    for split, count in SPLIT_COUNTS.items():
        source = source_root / system / 'transitions' / f'{horizon_fs:g}fs' / f'{split}.h5'
        target = out / f'{dataset_name}_{split};P:64.pt'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        with h5py.File(source, 'r') as transitions, ExitStack() as stack:
            if (float(transitions.attrs['horizon_fs']) != horizon_fs or
                    int(transitions.attrs['n_samples']) != count or
                    text(transitions.attrs['split']) != split):
                raise ValueError(f'Unexpected transition metadata in {source}')
            lag = int(transitions.attrs['lag_steps'])
            source_dt = float(transitions.attrs['source_dt_fs'])
            if not np.isclose(lag * source_dt, horizon_fs):
                raise ValueError('lag_steps * source_dt_fs does not equal horizon')
            paths = [text(value) for value in transitions['source_paths'][:]]
            source_sets[split] = set(paths)
            if not all(Path(path).parent.name.startswith(split + '_') for path in paths):
                raise ValueError(f'{split} transition sources violate trajectory split')
            raw_files = [stack.enter_context(h5py.File(path, 'r')) for path in paths]
            atomic_numbers = transitions['atomic_numbers'][:]
            masses = transitions['masses_u'][:]
            mapper = {int(z): index for index, z in enumerate(sorted(set(atomic_numbers.tolist())))}
            for raw in raw_files:
                np.testing.assert_array_equal(raw['atomic_numbers'][:], atomic_numbers)
                np.testing.assert_array_equal(raw['masses_u'][:], masses)
                np.testing.assert_array_equal(raw['atom_ids'][:], np.arange(1, len(masses) + 1))
                if text(raw.attrs['ensemble']) != 'NVE':
                    raise ValueError('Transition source is not NVE')

            graphs, max_error = [], {}

            def check(key, actual, expected, atol=1e-10):
                error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
                max_error[key] = max(error, max_error.get(key, 0.0))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=key)

            for index in range(count):
                trajectory = int(transitions['source_trajectory'][index])
                frame = int(transitions['source_frame'][index])
                endpoint = frame + lag
                raw = raw_files[trajectory]
                q0 = transitions['positions_initial_A'][index].astype('float64')
                q1 = transitions['positions_final_A'][index].astype('float64')
                v0 = transitions['velocities_initial_A_per_fs'][index].astype('float64')
                v1 = transitions['velocities_final_A_per_fs'][index].astype('float64')
                cell = transitions['cell_A'][index]
                for raw_index, q, velocity in ((frame, q0, v0), (endpoint, q1, v1)):
                    check('raw_position', q, raw['positions_unwrapped_A'][raw_index])
                    check('raw_velocity', velocity, raw['velocities_A_per_fs'][raw_index])
                    check('raw_cell', cell, raw['cell_A'][raw_index])
                check('elapsed_time', raw['time_fs'][endpoint] - raw['time_fs'][frame], horizon_fs)
                check('initial_momentum', transitions['momenta_initial_u_A_per_fs'][index],
                      masses[:, None] * v0, atol=5e-8)
                displacement = q1 - q0
                check('stored_displacement', displacement,
                      transitions['delta_positions_A'][index], atol=2e-6)
                wrapped = (q0 @ np.linalg.inv(cell) % 1.0) @ cell
                graph = AtomicGraph.from_atoms_dict(dict(
                    positions=torch.tensor(wrapped), cell=torch.tensor(cell),
                    pbc=torch.tensor([True, True, True]),
                    atomic_numbers=torch.tensor(atomic_numbers.astype('int64')[:, None]),
                    atomic_masses=torch.tensor(masses[:, None]),
                    velocities=torch.tensor(v0), displacements=torch.tensor(displacement),
                    update_velocities=torch.tensor(v1),
                    timestep=torch.tensor([float(horizon_fs)])), cutoff, mapper)
                check('graph_endpoint_position',
                      graph.pos.numpy() + graph.displacements.numpy() + (q0 - wrapped), q1)
                check('graph_endpoint_velocity', graph.update_velocities.numpy(), v1)
                graphs.append(graph)
                if index % 500 == 0:
                    print(split, index, flush=True)
            torch.save(InMemoryDataset.collate(graphs), target)
            reports[split] = dict(samples=count, source=str(source), source_paths=paths,
                                  cache=str(target), lag_steps=lag, source_dt_fs=source_dt,
                                  atom_type_mapper=mapper, max_abs_error=max_error,
                                  all_samples_verified=True)

    if (source_sets['train'] & source_sets['val'] or source_sets['train'] & source_sets['test'] or
            source_sets['val'] & source_sets['test']):
        raise ValueError('Raw trajectory leakage between train/val/test')
    reports['metadata'] = dict(system=system, horizon_fs=horizon_fs, cutoff_angstrom=cutoff,
                               dataset_name=dataset_name, split_disjoint=True,
                               units={'position': 'Angstrom', 'velocity': 'Angstrom/fs',
                                      'mass': 'u', 'time': 'fs', 'cell': 'row vectors'})
    (out / 'adaptation.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--system', required=True)
    parser.add_argument('--horizon-fs', type=float, required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cutoff', type=float, default=4.5)
    args = parser.parse_args()
    build(args.source, args.system, args.horizon_fs, args.dataset_name, args.out, args.cutoff)
