"""Driven position-consistent GLE rollout.

The backbone supplies the drift

    a_b = (v_backbone - v_t) / h

The GLR readout supplies the memory/noise prediction v_AR. The GLE update is

    v_{t+1} = v_AR + h a_b
    q_{t+1} = q_t + h v_{t+1}

This differs from blending: the AR process is the velocity model, and the
backbone enters as a force/drift, not as a 0.9 weight on the velocity.
"""
from __future__ import annotations

import torch
from torch import nn


class DrivenGLETransition(nn.Module):
    def __init__(self, base, readout):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rms_targets = base.rms_targets
        self.readout = readout
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
        v_ar = self.readout(data.pos, data.velocities, data.atomic_masses,
                            data.molecule_ids, data.cell)
        horizon = float(self.cfg.horizon_fs)
        drift = (v_pred - data.velocities) / horizon
        v_new = v_ar + horizon * drift
        masses = data.atomic_masses.reshape(-1)
        com = (masses[:, None] * v_new).sum(0) / masses.sum()
        v_new = v_new - com

        output['positions'] = data.pos + horizon * v_new
        output['velocities'] = v_new
        output['fd_velocity'] = v_ar
        output['drift'] = drift
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
