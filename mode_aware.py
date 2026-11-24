"""Mode-aware collective/internal correction for a frozen direct propagator.

The base model predicts a full per-atom velocity.  This wrapper adds a
zero-initialized correction with two explicit paths:

- collective: per-molecule COM translation + angular velocity;
- internal: per-atom motion in the molecular frame.

The correction is zero at initialization, so the fine-tuned model starts exactly
at the stable base checkpoint.  The base model can be frozen to test whether the
mode-aware path alone changes the NVT partition.
"""
from __future__ import annotations

import torch
from torch import nn

from selection_metrics import minimum_image


def _cell_for(cell, graphs):
    if cell.ndim == 3:
        return cell
    if cell.shape == (3, 3):
        return cell.unsqueeze(0).expand(graphs, -1, -1)
    if cell.shape[0] == graphs*3:
        return cell.reshape(graphs, 3, 3)
    raise ValueError(f'unexpected cell shape {tuple(cell.shape)}')


def _zero_init(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class ModeAwareCorrection(nn.Module):
    def __init__(self, scalar_channels, vector_channels, hidden=128):
        super().__init__()
        self.collective = nn.Sequential(
            nn.Linear(scalar_channels, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 6))
        self.internal = nn.Sequential(
            nn.Linear(scalar_channels + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3))
        _zero_init(self.collective[-1])
        _zero_init(self.internal[-1])
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, scalar, vector, positions, masses, molecule_ids, cell, batch):
        graphs = int(batch.max()) + 1
        cells = _cell_for(cell, graphs)
        vector_norm = vector.norm(dim=-1).mean(-1, keepdim=True)
        corrections = []
        for index in range(graphs):
            mask = batch == index
            s = scalar[mask]
            pos = positions[mask]
            mass = masses[mask].reshape(-1)
            mol = molecule_ids[mask].reshape(-1)
            _, inverse = torch.unique(mol, return_inverse=True)
            n_mol = int(inverse.max()) + 1
            count = torch.zeros(n_mol, dtype=s.dtype, device=s.device).index_add(
                0, inverse, torch.ones_like(mass))
            s_mol = torch.zeros(n_mol, s.shape[-1], dtype=s.dtype, device=s.device).index_add(
                0, inverse, s) / count[:, None]
            collective = self.collective(s_mol)
            mass_sum = torch.zeros(n_mol, dtype=mass.dtype, device=mass.device).index_add(
                0, inverse, mass)
            r_com = torch.zeros(n_mol, 3, dtype=pos.dtype, device=pos.device).index_add(
                0, inverse, mass[:, None]*pos) / mass_sum[:, None]
            r_rel = minimum_image(pos - r_com[inverse], cells[index]).to(pos.dtype)
            dv_collective = collective[:, :3][inverse] + torch.cross(
                collective[:, 3:][inverse], r_rel, dim=-1)
            dv_internal = self.internal(torch.cat([s, vector_norm[mask]], -1))
            corrections.append(dv_collective + dv_internal)
        return self.scale * torch.cat(corrections, 0)


class ModeAwareTransition(nn.Module):
    """Frozen-base + trainable mode-aware correction."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rms_targets = base.rms_targets
        ds = base.cfg.scalar_channels if hasattr(base.cfg, 'scalar_channels') else base.cfg['scalar_channels']
        dv = base.cfg.vector_channels if hasattr(base.cfg, 'vector_channels') else base.cfg['vector_channels']
        self.correction = ModeAwareCorrection(ds, dv)

    def validate_graph(self, data):
        return self.base.validate_graph(data) if hasattr(self.base, 'validate_graph') else None

    def forward(self, data, return_context=False):
        output = self.base(data, return_context=True)
        context = output.get('context', {})
        scalar = context.get('scalar')
        vector = context.get('vector')
        if scalar is None or vector is None:
            raise RuntimeError('mode-aware wrapper requires base context scalar/vector')
        batch = (data.batch if getattr(data, 'batch', None) is not None
                 else torch.zeros(data.num_nodes, dtype=torch.long, device=data.pos.device))
        correction = self.correction(
            scalar, vector, data.pos, data.atomic_masses,
            data.molecule_ids, data.cell, batch)
        horizon = float(self.base.cfg.horizon_fs)
        output['velocities'] = output['velocities'] + correction
        output['displacements'] = output['displacements'] + horizon*correction
        output['positions'] = data.pos + output['displacements']
        output['mode_correction'] = correction
        if return_context:
            output['context'] = context
        return output

    def set_diagnostics(self, enabled):
        if hasattr(self.base, 'set_diagnostics'):
            self.base.set_diagnostics(enabled)

    def diagnostics(self):
        return self.base.diagnostics() if hasattr(self.base, 'diagnostics') else []
