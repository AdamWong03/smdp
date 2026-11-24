"""Mode-resolved Generalized Langevin Readout (GLR).

Each mode follows an AR(p) fitted by Yule-Walker from the reference ACF:

    v_mode(t) = sum_{k=1..p} a_k v_mode(t-k) + eps(t)

The prediction-error variance is the FD-consistent noise variance; the
per-mode variance rescale in the wrapper enforces the canonical stationary KE.
The readout keeps a p-step history of the mode velocity fields.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from motion_loss import decompose_modes

KINETIC_TO_EV = 103.64269
KB_EV_PER_K = 8.617333262e-5
KEYS = ('translation', 'rotation', 'internal')


class GLRReadout(nn.Module):
    def __init__(self, coefficients, prediction_error, temperature=300.0,
                 use_noise=True, slow_tau=0.0, slow_weight=0.0,
                 horizon_fs=5.0):
        super().__init__()
        self.register_buffer('coefficients',
                             torch.tensor(coefficients, dtype=torch.float32))
        self.register_buffer('prediction_error',
                             torch.tensor(prediction_error, dtype=torch.float32))
        self.temperature = float(temperature)
        self.order = int(self.coefficients.shape[1])
        self.use_noise = bool(use_noise)
        self.slow_tau = float(slow_tau)
        self.slow_weight = float(slow_weight)
        self.horizon_fs = float(horizon_fs)
        self.history = None
        self.slow_state = None

    def reset(self):
        self.history = None
        self.slow_state = None

    def forward(self, positions, velocities, masses, molecules, cell):
        masses = masses.reshape(-1)
        molecules = molecules.reshape(-1)
        current = decompose_modes(positions, velocities, masses, molecules, cell)
        if self.history is None:
            self.history = [{key: current[key] for key in KEYS}
                            for _ in range(self.order)]
        noise = torch.randn_like(velocities)
        noise_modes = decompose_modes(positions, noise, masses, molecules, cell)
        n_mol = int(molecules.max()) + 1
        n_atoms = len(masses)
        dofs = (3*n_mol - 3, 3*n_mol, 3*n_atoms - 6*n_mol)
        result = torch.zeros_like(velocities)
        for index, key in enumerate(KEYS):
            ar = torch.zeros_like(velocities)
            for lag in range(self.order):
                ar = ar + self.coefficients[index, lag]*self.history[-1-lag][key]
            if self.use_noise:
                target = (0.5*dofs[index]*KB_EV_PER_K*self.temperature
                          * self.prediction_error[index])
                norm = 0.5*KINETIC_TO_EV*torch.sum(
                    masses[:, None]*noise_modes[key].square())
                scale = torch.sqrt(torch.clamp(target/norm.clamp_min(1e-12), max=1e6))
                mode = ar + scale*noise_modes[key]
            else:
                mode = ar
            if key == 'translation' and self.slow_tau > 0.0 and self.slow_weight > 0.0:
                if self.slow_state is None:
                    self.slow_state = mode.detach().clone()
                else:
                    decay = math.exp(-self.horizon_fs/self.slow_tau)
                    sigma = mode.detach().std().clamp_min(1e-8)
                    slow_noise = noise_modes['translation']
                    slow_noise = (slow_noise
                                  / slow_noise.std().clamp_min(1e-8) * sigma)
                    self.slow_state = (decay*self.slow_state
                                       + math.sqrt(max(1.0 - decay*decay, 0.0))
                                       * slow_noise)
                mode = (math.sqrt(1.0 - self.slow_weight)*mode
                        + math.sqrt(self.slow_weight)*self.slow_state)
            result = result + mode
        com = (masses[:, None]*result).sum(0)/masses.sum()
        return result - com

    def update(self, positions, velocities, masses, molecules, cell):
        masses = masses.reshape(-1)
        molecules = molecules.reshape(-1)
        part = decompose_modes(positions, velocities, masses, molecules, cell)
        if self.history is None:
            self.history = [{key: part[key] for key in KEYS}
                            for _ in range(self.order)]
        else:
            self.history.append({key: part[key] for key in KEYS})
            self.history.pop(0)
