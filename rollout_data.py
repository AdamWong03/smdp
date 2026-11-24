"""Short raw-trajectory sequences for differentiable rollout fine-tuning."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def discover(root: Path, split: str):
    paths = sorted((root/'raw_nve').glob(f'{split}_*/trajectory.h5'))
    if not paths:
        raise FileNotFoundError(f'no {split}_* raw trajectories under {root}')
    return paths


def build_sequences(root: Path, split: str, count: int, seed: int,
                    horizon_fs: float = 5.0, steps: int = 2):
    """Sample `count` sequences of `steps` consecutive horizon transitions."""
    paths = discover(root, split)
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        path = paths[int(rng.integers(0, len(paths)))]
        with h5py.File(path, 'r') as handle:
            dt = float(handle.attrs['integration_dt_fs'])
            lag = int(round(horizon_fs/dt))
            if not np.isclose(lag*dt, horizon_fs):
                raise ValueError('horizon is not an integer number of MD steps')
            n_frames = len(handle['positions_unwrapped_A'])
            if n_frames <= steps*lag:
                raise ValueError(f'{path} too short for {steps} steps')
            start = int(rng.integers(0, n_frames - steps*lag))
            q = np.stack([handle['positions_unwrapped_A'][start+k*lag]
                          for k in range(steps+1)]).astype(np.float32)
            v = np.stack([handle['velocities_A_per_fs'][start+k*lag]
                          for k in range(steps+1)]).astype(np.float32)
            cell = np.asarray(handle['cell_A'][start], dtype=np.float32)
            z = np.asarray(handle['atomic_numbers'][:], dtype=np.int64)
            masses = np.asarray(handle['masses_u'][:], dtype=np.float32)
            molecules = np.asarray(handle['molecule_ids'][:], dtype=np.int64)
        result.append(dict(q=q, v=v, cell=cell, z=z, masses=masses,
                           molecules=molecules, source=str(path), start=start))
    return result
