"""Exact source-indexed segments from the existing independent quartz NVE splits."""
from pathlib import Path
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class QuartzSegments(Dataset):
    offsets = np.array(list(range(17)) + [32, 48], dtype=np.int64)

    def __init__(self, root, split, cache):
        root, cache = Path(root), Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        source = root / 'quartz/transitions/16fs' / (split + '.h5')
        expected = {'train': 5000, 'val': 1250, 'test': 1000}[split]
        cached = cache / (split + '_segments_v1.pt')
        with h5py.File(source) as f:
            assert f.attrs['horizon_fs'] == 16 and f.attrs['source_dt_fs'] == 1
            assert f.attrs['split'] == split and f.attrs['n_samples'] == expected
            paths = [Path(p.decode() if isinstance(p, bytes) else p) for p in f['source_paths'][:]]
            assert all(p.parent.name.startswith(split + '_') for p in paths)
            frames, traj = f['source_frame'][:], f['source_trajectory'][:]
            z, m, cells = f['atomic_numbers'][:], f['masses_u'][:], f['cell_A'][:]
            q0, q16 = f['positions_initial_A'][:], f['positions_final_A'][:]
            v0, v16 = f['velocities_initial_A_per_fs'][:], f['velocities_final_A_per_fs'][:]
        identity = {'source': str(source.resolve()), 'mtime_ns': source.stat().st_mtime_ns,
                    'offsets': self.offsets.tolist(), 'split': split}
        if cached.exists():
            state = torch.load(cached, weights_only=False, map_location='cpu')
            if state['identity'] != identity:
                raise ValueError('Source changed: use a new cache directory')
        else:
            q = np.empty((expected, len(self.offsets), len(z), 3), np.float32)
            v = np.empty_like(q)
            valid = np.zeros(expected, bool)
            for tid, path in enumerate(paths):
                selected = np.flatnonzero(traj == tid)
                with h5py.File(path) as raw:
                    assert raw.attrs['ensemble'] == 'NVE'
                    assert raw.attrs['output_interval_fs'] == 1
                    np.testing.assert_array_equal(raw['atomic_numbers'][:], z)
                    np.testing.assert_array_equal(raw['masses_u'][:], m)
                    np.testing.assert_array_equal(raw['atom_ids'][:], np.arange(1, len(z)+1))
                    indices = frames[selected, None] + self.offsets[None]
                    valid[selected] = indices[:, -1] < len(raw['time_fs'])
                    # Invalid continuation is never used for rollout; all endpoint samples remain.
                    indices = np.minimum(indices, len(raw['time_fs']) - 1)
                    for key, dst in [('positions_unwrapped_A', q), ('velocities_A_per_fs', v)]:
                        values = raw[key][:]
                        dst[selected] = values[indices]
                        del values
                    np.testing.assert_allclose(raw['cell_A'][:][frames[selected]], cells[selected], atol=1e-10)
                print(f'Loaded actual {split} segments: {path.parent.name}', flush=True)
            np.testing.assert_allclose(q[:, 0], q0, atol=2e-6)
            np.testing.assert_allclose(q[:, 16], q16, atol=2e-6)
            np.testing.assert_allclose(v[:, 0], v0, atol=1e-8)
            np.testing.assert_allclose(v[:, 16], v16, atol=1e-8)
            state = dict(identity=identity, q=torch.from_numpy(q), v=torch.from_numpy(v),
                         valid=torch.from_numpy(valid))
            torch.save(state, cached)
        self.q, self.v, self.valid = state['q'], state['v'], state['valid']
        self.cells = torch.tensor(cells, dtype=torch.float32)
        self.z, self.m = torch.tensor(z, dtype=torch.long), torch.tensor(m, dtype=torch.float32)
        self.sources = dict(paths=list(map(str, paths)), frames=frames.tolist(), trajectories=traj.tolist())
        self.scales = dict(displacement_residual_rms=float(np.sqrt(np.mean((q16-q0-16*v0)**2))),
                           velocity_delta_rms=float(np.sqrt(np.mean((v16-v0)**2))))

    def __len__(self): return len(self.q)

    def __getitem__(self, i):
        return dict(q=self.q[i], v=self.v[i], cell=self.cells[i], z=self.z, mass=self.m,
                    rollout_valid=self.valid[i])


def validate_split_manifest(root):
    """Metadata only; no test coordinates are used by training."""
    root = Path(root)
    metadata = {n: json.loads((root/n).read_text()) for n in
                ['dataset_metadata.json', 'physical_summary.json', 'long_reference_summary.json']}
    sources = {}
    for split in ['train', 'val', 'test']:
        with h5py.File(root / f'quartz/transitions/16fs/{split}.h5') as f:
            sources[split] = {p.decode() if isinstance(p, bytes) else str(p) for p in f['source_paths'][:]}
    assert not (sources['train'] & sources['val'] or sources['train'] & sources['test'] or sources['val'] & sources['test'])
    return {k: sorted(v) for k, v in sources.items()}
