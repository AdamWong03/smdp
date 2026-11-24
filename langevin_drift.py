"""Stable Langevin integrator with a learned backbone drift.

    v' = v_backbone - gamma*h*v + sqrt(2*gamma*h*kB*T/m) * xi
    q' = q + h*v'

The backbone supplies the drift (its predicted velocity); the Langevin term
supplies friction and thermal noise. This is a stability/consistency test for
the position-consistent architecture, not the final memory kernel.
"""
from __future__ import annotations

import math

import torch
from torch import nn

KINETIC_TO_EV = 103.64269
KB_EV_PER_K = 8.617333262e-5


class LangevinDriftTransition(nn.Module):
    def __init__(self, base, temperature=300.0, gamma=0.1,
                 global_rescale=False):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rms_targets = base.rms_targets
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.global_rescale = bool(global_rescale)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def reset(self):
        pass

    def validate_graph(self, data):
        return self.base.validate_graph(data) if hasattr(self.base, 'validate_graph') else None

    def forward(self, data, return_context=False):
        output = self.base(data, return_context=True)
        v_pred = output['velocities']
        horizon = float(self.cfg.horizon_fs)
        masses = data.atomic_masses.reshape(-1)
        friction = self.gamma * horizon
        thermal_var = (2.0 * friction * KB_EV_PER_K * self.temperature
                       / (masses[:, None] * KINETIC_TO_EV))
        noise = torch.randn_like(v_pred) * torch.sqrt(thermal_var)
        v_new = v_pred - friction * data.velocities + noise
        if self.global_rescale:
            total = 0.5 * KINETIC_TO_EV * torch.sum(
                masses[:, None] * v_new.square())
            concentration = torch.as_tensor(
                1.5 * len(masses), dtype=total.dtype, device=total.device)
            rate = torch.as_tensor(
                1.0 / (KB_EV_PER_K * self.temperature),
                dtype=total.dtype, device=total.device)
            target = torch.distributions.Gamma(concentration, rate).sample()
            scale = torch.sqrt(torch.clamp(target / total.clamp_min(1e-12),
                                           max=1e6))
            v_new = v_new * scale
        com = (masses[:, None] * v_new).sum(0) / masses.sum()
        v_new = v_new - com

        output['positions'] = data.pos + horizon * v_new
        output['velocities'] = v_new
        output['langevin_noise'] = noise
        if return_context:
            output['context'] = output.get('context', {})
        return output

    def set_diagnostics(self, enabled):
        if hasattr(self.base, 'set_diagnostics'):
            self.base.set_diagnostics(enabled)

    def diagnostics(self):
        return self.base.diagnostics() if hasattr(self.base, 'diagnostics') else []
