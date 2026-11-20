"""Short validation-only rollout proxy for model selection.

One-step partition on reference states is almost perfect for the frozen Water
model; the bias appears only after autoregressive rollout.  This module builds
a small, fixed set of validation starts and measures the partition error after
K deterministic NVE steps.  It never reads the test split and never selects a
checkpoint on a test trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from periodic import PeriodicGraphBuilder
from relational_inference import vesin_graph
from selection_metrics import molecular_kinetic_partition, partition_relative_error


@dataclass
class RolloutStart:
    positions: np.ndarray
    velocities: np.ndarray
    cell: np.ndarray
    atomic_numbers: np.ndarray
    masses: np.ndarray
    molecule_ids: np.ndarray
    reference_positions: np.ndarray
    reference_velocities: np.ndarray


def build_rollout_starts(cache: Path, split: str, cfg: dict, *, steps: int,
                         samples: int, seed: int) -> list[RolloutStart]:
    """Select fixed validation starts with a reference state `steps` ahead."""
    horizon = float(cfg['model']['horizon_fs'])
    transition = Path(cache).parent.parent / 'transitions' / f'{horizon:g}fs' / f'{split}.h5'
    if not transition.is_file():
        raise FileNotFoundError(transition)
    with h5py.File(transition, 'r') as handle:
        lag = int(handle.attrs['lag_steps'])
        dt = float(handle.attrs['source_dt_fs'])
        paths = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in handle['source_paths'][:]]
        trajectory = handle['source_trajectory'][:].astype(int)
        frame = handle['source_frame'][:].astype(int)
        cell = handle['cell_A'][:]
        numbers = handle['atomic_numbers'][:]
        masses = handle['masses_u'][:]
        molecules = handle['molecule_ids'][:]
        # Only starts whose full rollout has a reference state on the same
        # trajectory are eligible.  This is a coverage check, not test access.
        raw_lengths = []
        for path in paths:
            with h5py.File(path, 'r') as raw:
                raw_lengths.append(len(raw['positions_unwrapped_A']))
        eligible = [index for index in range(len(frame))
                    if frame[index] + steps*lag < raw_lengths[trajectory[index]]]
        if len(eligible) < samples:
            raise ValueError(f'only {len(eligible)} eligible validation starts')
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(eligible, size=samples, replace=False))
        result = []
        for index in chosen:
            path = paths[trajectory[index]]
            start = frame[index]
            end = start + steps*lag
            with h5py.File(path, 'r') as raw:
                q0 = raw['positions_unwrapped_A'][start].astype(np.float32)
                v0 = raw['velocities_A_per_fs'][start].astype(np.float32)
                q1 = raw['positions_unwrapped_A'][end].astype(np.float32)
                v1 = raw['velocities_A_per_fs'][end].astype(np.float32)
            result.append(RolloutStart(
                positions=q0, velocities=v0, cell=cell[index].astype(np.float64),
                atomic_numbers=numbers, masses=masses, molecule_ids=molecules,
                reference_positions=q1, reference_velocities=v1))
        return result


@torch.no_grad()
def rollout_partition_error(model, starts: list[RolloutStart], device,
                            steps: int, components: tuple[str, ...],
                            builder: PeriodicGraphBuilder | None = None) -> dict:
    """Roll out K NVE steps and compare final partition with the reference."""
    model.eval()
    builder = builder or PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    errors, pred_rows, ref_rows = [], [], []
    for start in starts:
        q = torch.as_tensor(start.positions, dtype=torch.float32, device=device)
        v = torch.as_tensor(start.velocities, dtype=torch.float32, device=device)
        cell = torch.as_tensor(start.cell, dtype=torch.float32, device=device)
        z = torch.as_tensor(start.atomic_numbers, dtype=torch.long, device=device)
        masses = torch.as_tensor(start.masses, dtype=torch.float32, device=device)
        molecules = torch.as_tensor(start.molecule_ids, dtype=torch.long, device=device)
        for _ in range(steps):
            graph = vesin_graph(q, v, cell, z, masses, model, builder, device)
            out = model(graph)
            q, v = out['positions'], out['velocities']
        ref_q = torch.as_tensor(start.reference_positions, dtype=torch.float32, device=device)
        ref_v = torch.as_tensor(start.reference_velocities, dtype=torch.float32, device=device)
        predicted = molecular_kinetic_partition(q, v, masses, molecules, cell)
        reference = molecular_kinetic_partition(ref_q, ref_v, masses, molecules, cell)
        errors.append(partition_relative_error(predicted, reference, components))
        pred_rows.append({k: float(v) for k, v in predicted.items()})
        ref_rows.append({k: float(v) for k, v in reference.items()})
    error = torch.stack(errors).mean()
    pred_mean = {k: float(np.mean([row[k] for row in pred_rows])) for k in pred_rows[0]}
    ref_mean = {k: float(np.mean([row[k] for row in ref_rows])) for k in ref_rows[0]}
    return dict(rollout_partition_error=float(error),
                rollout_steps=steps, rollout_samples=len(starts),
                rollout_translation_pred_eV=pred_mean['translation'],
                rollout_translation_ref_eV=ref_mean['translation'],
                rollout_rotation_pred_eV=pred_mean['rotation'],
                rollout_rotation_ref_eV=ref_mean['rotation'],
                rollout_internal_pred_eV=pred_mean['internal'],
                rollout_internal_ref_eV=ref_mean['internal'],
                rollout_total_pred_eV=pred_mean['total'],
                rollout_total_ref_eV=ref_mean['total'])
