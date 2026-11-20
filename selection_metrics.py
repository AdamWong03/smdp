"""Physical-proxy selection metrics for finite-time propagators.

The endpoint objective is a conditional-mean metric.  These functions add a
cheap, validation-only kinetic-energy partition proxy so that model selection
can see the motion bias that endpoint RMSE misses.  They are measurement code:
they do not change the model, the training loss, or the physical reference.
"""
from __future__ import annotations

import torch

from physics.units import KINETIC_TO_EV


def _as_cell(cell: torch.Tensor) -> torch.Tensor:
    cell = torch.as_tensor(cell, dtype=torch.float64)
    if cell.shape == (1, 3, 3):
        cell = cell[0]
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be [3,3] or [1,3,3], got {tuple(cell.shape)}")
    return cell


def minimum_image(relative: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Minimum-image displacement for row-vector cells, in Angstrom."""
    cell = _as_cell(cell).to(relative.device)
    inverse = torch.linalg.inv(cell)
    fractional = relative.to(torch.float64) @ inverse
    fractional = fractional - torch.round(fractional)
    return (fractional @ cell).to(relative.dtype)


def molecular_kinetic_partition(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    molecule_ids: torch.Tensor,
    cell: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return translation/rotation/internal/total kinetic energy in eV.

    ``positions`` and ``velocities`` are [N,3] in Angstrom and Angstrom/fs;
    ``masses`` is [N] in amu.  Molecules are defined by ``molecule_ids``.
    Rotation is computed about each molecule's centre of mass with the
    minimum-image convention.  The function is deterministic and does not
    read reference energies.
    """
    if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions and velocities must both be [N,3]")
    n = positions.shape[0]
    if masses.numel() != n or molecule_ids.numel() != n:
        raise ValueError("masses and molecule_ids must have N entries")
    dtype = velocities.dtype
    device = velocities.device
    m = masses.to(torch.float64)
    v = velocities.to(torch.float64)
    r = positions.to(torch.float64)
    _, inverse = torch.unique(molecule_ids, return_inverse=True)
    n_molecules = int(inverse.max()) + 1
    eye = torch.eye(3, dtype=torch.float64, device=device)
    mass = torch.zeros(n_molecules, dtype=torch.float64, device=device).index_add(
        0, inverse, m)
    if bool((mass <= 0).any()):
        raise ValueError("molecule has non-positive total mass")
    v_com = torch.zeros(n_molecules, 3, dtype=torch.float64, device=device).index_add(
        0, inverse, m[:, None]*v) / mass[:, None]
    r_com = torch.zeros(n_molecules, 3, dtype=torch.float64, device=device).index_add(
        0, inverse, m[:, None]*r) / mass[:, None]
    r_rel = minimum_image(r - r_com[inverse], cell).to(torch.float64)
    v_rel = v - v_com[inverse]
    kinetic_atom = 0.5 * KINETIC_TO_EV * m * v.square().sum(-1)
    kinetic_molecule = torch.zeros(n_molecules, dtype=torch.float64, device=device).index_add(
        0, inverse, kinetic_atom)
    translation = 0.5 * KINETIC_TO_EV * mass * v_com.square().sum(-1)
    angular = torch.zeros(n_molecules, 3, dtype=torch.float64, device=device).index_add(
        0, inverse, m[:, None] * torch.cross(r_rel, v_rel, dim=-1))
    r2 = r_rel.square().sum(-1)
    outer = r_rel[:, :, None] * r_rel[:, None, :]
    inertia = torch.zeros(n_molecules, 3, 3, dtype=torch.float64, device=device).index_add(
        0, inverse, m[:, None, None] * (r2[:, None, None]*eye - outer))
    # A linear molecule has a zero eigenvalue; pinv reports rotation about the
    # available axes.  Water is non-linear, so this is exact in the target use.
    rotation = 0.5 * KINETIC_TO_EV * (
        angular * (torch.linalg.pinv(inertia) @ angular.unsqueeze(-1)).squeeze(-1)
    ).sum(-1)
    internal = kinetic_molecule - translation - rotation
    return dict(total=kinetic_molecule.sum().to(dtype),
                translation=translation.sum().to(dtype),
                rotation=rotation.sum().to(dtype),
                internal=internal.sum().to(dtype))


def partition_relative_error(
    predicted: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    components: tuple[str, ...] = ("translation", "rotation", "internal"),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean absolute relative error over the requested partition components."""
    errors = []
    for name in components:
        if name not in predicted or name not in reference:
            raise KeyError(name)
        errors.append((predicted[name] - reference[name]).abs()
                      / (reference[name].abs() + eps))
    return torch.stack(errors).mean()
