"""Motion-balanced training objective for molecular direct propagators.

The global velocity MSE on Water is dominated by internal/rotational motion:
translation contributes only ~7.8% of the squared velocity.  This module
decomposes predicted and reference velocities into molecular translation,
rotation and internal components, normalizes each by its TRAIN-only RMS, and
returns a balanced objective.  It is a training change, not a new architecture.
"""
from __future__ import annotations

import torch

from selection_metrics import minimum_image


def _as_cell(cell: torch.Tensor, graphs: int) -> torch.Tensor:
    if cell.ndim == 3:
        return cell
    if cell.shape == (3, 3):
        return cell.unsqueeze(0).expand(graphs, -1, -1)
    if cell.shape[0] == graphs*3:
        return cell.reshape(graphs, 3, 3)
    raise ValueError(f'unexpected cell shape {tuple(cell.shape)}')


def decompose_modes(positions, velocities, masses, molecule_ids, cell):
    """Return per-atom translation/rotation/internal velocity components.

    All tensors are single-graph: positions/velocities [N,3], masses [N],
    molecule_ids [N], cell [3,3].  The decomposition is differentiable; the
    rotation inertia uses a pseudo-inverse so linear molecules remain valid.
    """
    dtype = velocities.dtype
    device = velocities.device
    masses = masses.reshape(-1)
    molecule_ids = molecule_ids.reshape(-1)
    m = masses.to(torch.float64)
    v = velocities.to(torch.float64)
    r = positions.to(torch.float64)
    _, inverse = torch.unique(molecule_ids, return_inverse=True)
    n_molecules = int(inverse.max()) + 1
    mass = torch.zeros(n_molecules, dtype=torch.float64, device=device).index_add(
        0, inverse, m)
    if bool((mass <= 0).any()):
        raise ValueError('molecule has non-positive total mass')
    v_com = torch.zeros(n_molecules, 3, dtype=torch.float64, device=device).index_add(
        0, inverse, m[:, None]*v) / mass[:, None]
    r_com = torch.zeros(n_molecules, 3, dtype=torch.float64, device=device).index_add(
        0, inverse, m[:, None]*r) / mass[:, None]
    r_rel = minimum_image(r - r_com[inverse], cell).to(torch.float64)
    v_rel = v - v_com[inverse]
    angular = torch.zeros(n_molecules, 3, dtype=torch.float64, device=device).index_add(
        0, inverse, m[:, None] * torch.cross(r_rel, v_rel, dim=-1))
    r2 = r_rel.square().sum(-1)
    eye = torch.eye(3, dtype=torch.float64, device=device)
    inertia = torch.zeros(n_molecules, 3, 3, dtype=torch.float64, device=device).index_add(
        0, inverse, m[:, None, None] * (r2[:, None, None]*eye - r_rel[:, :, None]*r_rel[:, None, :]))
    omega = (torch.linalg.pinv(inertia) @ angular.unsqueeze(-1)).squeeze(-1)
    v_rot = torch.cross(omega[inverse], r_rel, dim=-1)
    v_int = v_rel - v_rot
    v_trans = v_com[inverse]
    return dict(translation=v_trans.to(dtype),
                rotation=v_rot.to(dtype),
                internal=v_int.to(dtype),
                total=v.to(dtype))


def decompose_batch(positions, velocities, masses, molecule_ids, cell, batch):
    """Decompose a PyG batch graph by graph; returns concatenated components."""
    graphs = int(batch.max()) + 1
    cells = _as_cell(cell, graphs)
    parts = {k: [] for k in ('translation', 'rotation', 'internal', 'total')}
    for index in range(graphs):
        mask = batch == index
        part = decompose_modes(positions[mask], velocities[mask], masses[mask],
                               molecule_ids[mask], cells[index])
        for key in parts:
            parts[key].append(part[key])
    return {key: torch.cat(value, 0) for key, value in parts.items()}


def train_mode_rms(graphs) -> dict[str, float]:
    """TRAIN-only RMS of each mode, per atom, for normalization."""
    sums = {k: 0.0 for k in ('translation', 'rotation', 'internal')}
    count = 0
    for graph in graphs:
        part = decompose_modes(graph.pos, graph.update_velocities, graph.atomic_masses,
                               graph.molecule_ids, graph.cell)
        for key in sums:
            sums[key] += float(part[key].double().square().sum())
        count += len(graph.pos)
    if count == 0:
        raise ValueError('empty TRAIN split for mode normalization')
    # A degenerate mode (e.g. a synthetic pure-internal batch) still needs a
    # finite scale; real training splits have non-zero variance in all modes.
    return {key: max((value/count)**0.5, 1e-8) for key, value in sums.items()}


def mode_mse(predicted, reference, mode_rms, weights=None):
    weights = weights or {'translation': 1.0, 'rotation': 1.0, 'internal': 1.0}
    pieces = {}
    total = 0.0
    for key in ('translation', 'rotation', 'internal'):
        scale = float(mode_rms[key])
        if scale <= 0:
            raise ValueError(f'non-positive mode RMS for {key}')
        value = ((predicted[key]-reference[key])/scale).square().mean()
        pieces[key] = value
        total = total + float(weights.get(key, 1.0))*value
    normalizer = sum(float(weights.get(k, 1.0)) for k in pieces)
    return total/normalizer, pieces


def motion_balanced_objective(model, output, batch, mode_rms, *,
                              velocity_weights=None, position_weight=1.0,
                              proposal_weight=0.02):
    """Motion-balanced endpoint objective on a PyG batch.

    Position loss stays globally normalized; velocity loss is decomposed into
    molecular translation/rotation/internal and normalized by TRAIN-only RMS.
    """
    if getattr(batch, 'molecule_ids', None) is None:
        raise ValueError('motion-balanced loss requires molecule_ids')
    weights = velocity_weights or {'translation': 1.0, 'rotation': 1.0, 'internal': 1.0}
    position = ((output['displacements']-batch.displacements)
                / model.rms_targets[0]).square().mean()
    predicted_positions = output['positions']
    reference_positions = batch.pos + batch.displacements
    predicted = decompose_batch(predicted_positions, output['velocities'],
                                batch.atomic_masses, batch.molecule_ids,
                                batch.cell, batch.batch)
    reference = decompose_batch(reference_positions, batch.update_velocities,
                                batch.atomic_masses, batch.molecule_ids,
                                batch.cell, batch.batch)
    velocity, velocity_pieces = mode_mse(predicted, reference, mode_rms, weights)
    final = 0.5*(float(position_weight)*position + velocity)

    proposal_positions = batch.pos + output['proposal_displacements']
    proposal_pred = decompose_batch(proposal_positions, output['proposal_velocities'],
                                    batch.atomic_masses, batch.molecule_ids,
                                    batch.cell, batch.batch)
    proposal_velocity, proposal_pieces = mode_mse(
        proposal_pred, reference, mode_rms, weights)
    proposal = 0.5*(float(position_weight)*position + proposal_velocity)
    loss = final + float(proposal_weight)*proposal
    pieces = dict(final=final, proposal=proposal, final_position=position,
                  final_velocity=velocity, proposal_position=position,
                  proposal_velocity=proposal_velocity,
                  mode_translation=velocity_pieces['translation'],
                  mode_rotation=velocity_pieces['rotation'],
                  mode_internal=velocity_pieces['internal'])
    return loss, {key: value.detach() for key, value in pieces.items()}
