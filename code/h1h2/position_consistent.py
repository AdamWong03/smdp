"""Position-consistent GLE rollout.

The backbone supplies the drift (its predicted velocity), the GLR readout
supplies the memory/noise correction, and the position is integrated with the
same corrected velocity:

    v' = (1 - blend) v_backbone + blend v_GLR
    q' = q + h v'

This is the 1 fs alternative to the decoupled displacement update. It should
only be used with a horizon where the finite-difference drift is meaningful.
"""
from __future__ import annotations

import torch
from torch import nn


class PositionConsistentTransition(nn.Module):
    def __init__(self, base, readout, blend=0.1):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rms_targets = base.rms_targets
        self.readout = readout
        self.blend = float(blend)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def reset(self):
        if hasattr(self.readout, 'reset'):
            self.readout.reset()

    def validate_graph(self, data):
        return self.base.validate_graph(data) if hasattr(self.base, 'validate_graph') else None

    def forward(self, data, return_context=False):
        output = self.base(data, return_context=True)
        v_pred = output['velocities']
        v_fd = self.readout(data.pos, data.velocities, data.atomic_masses,
                            data.molecule_ids, data.cell)
        v_new = (1.0 - self.blend) * v_pred + self.blend * v_fd
        masses = data.atomic_masses.reshape(-1)
        com = (masses[:, None] * v_new).sum(0) / masses.sum()
        v_new = v_new - com

        horizon = float(self.cfg.horizon_fs)
        output['positions'] = data.pos + horizon * v_new
        output['velocities'] = v_new
        output['fd_velocity'] = v_fd
        self.readout.update(data.pos, v_new, data.atomic_masses,
                            data.molecule_ids, data.cell)
        if return_context:
            output['context'] = output.get('context', {})
        return output

    def set_diagnostics(self, enabled):
        if hasattr(self.base, 'set_diagnostics'):
            self.base.set_diagnostics(enabled)

    def diagnostics(self):
        return self.base.diagnostics() if hasattr(self.base, 'diagnostics') else []
