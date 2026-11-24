"""Species-resolved GLR readout for covalent solids (quartz).

The mode basis is the atomic species (Si, O), not molecular translation/rotation/
internal.  Each species follows an AR(p) velocity process fitted from the reference
NVT ACF; the per-species variance is rescaled to 3 N_s k_B T / 2.
"""
from __future__ import annotations

import torch
from torch import nn

KINETIC_TO_EV = 103.64269
KB_EV_PER_K = 8.617333262e-5


class SpeciesGLRReadout(nn.Module):
    def __init__(self, coefficients, prediction_error, species, dofs,
                 temperature=300.0, use_noise=True):
        super().__init__()
        self.register_buffer('coefficients',
                             torch.tensor(coefficients, dtype=torch.float32))
        self.register_buffer('prediction_error',
                             torch.tensor(prediction_error, dtype=torch.float32))
        self.species = [int(s) for s in species]
        self.dofs = [int(d) for d in dofs]
        self.temperature = float(temperature)
        self.order = int(self.coefficients.shape[1])
        self.use_noise = bool(use_noise)
        self.history = None

    def reset(self):
        self.history = None

    def _fields(self, velocities, atomic_numbers):
        z = atomic_numbers.reshape(-1)
        fields = []
        for element in self.species:
            field = torch.zeros_like(velocities)
            mask = z == element
            field[mask] = velocities[mask]
            fields.append(field)
        return fields

    def forward(self, positions, velocities, masses, atomic_numbers, cell):
        masses = masses.reshape(-1)
        z = atomic_numbers.reshape(-1)
        current = self._fields(velocities, z)
        if self.history is None:
            self.history = [[field.clone() for field in current]
                            for _ in range(self.order)]
        noise = torch.randn_like(velocities)
        noise_fields = self._fields(noise, z)
        result = torch.zeros_like(velocities)
        for index, element in enumerate(self.species):
            ar = torch.zeros_like(velocities)
            for lag in range(self.order):
                ar = ar + self.coefficients[index, lag]*self.history[-1-lag][index]
            if self.use_noise:
                target = (0.5*self.dofs[index]*KB_EV_PER_K*self.temperature
                          * self.prediction_error[index])
                mask = z == element
                norm = 0.5*KINETIC_TO_EV*torch.sum(
                    masses[mask, None]*noise_fields[index][mask].square())
                scale = torch.sqrt(torch.clamp(target/norm.clamp_min(1e-12), max=1e6))
                result = result + ar + scale*noise_fields[index]
            else:
                result = result + ar
        com = (masses[:, None]*result).sum(0)/masses.sum()
        return result - com

    def update(self, positions, velocities, masses, atomic_numbers, cell):
        current = self._fields(velocities, atomic_numbers.reshape(-1))
        if self.history is None:
            self.history = [[field.clone() for field in current]
                            for _ in range(self.order)]
        else:
            self.history.append([field.clone() for field in current])
            self.history.pop(0)


class SpeciesGLRTransition(nn.Module):
    def __init__(self, base, readout, blend=0.1, rescale=True):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rms_targets = base.rms_targets
        self.fd = readout
        self.blend = float(blend)
        self.rescale = bool(rescale)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def reset(self):
        if hasattr(self.fd, 'reset'):
            self.fd.reset()

    def validate_graph(self, data):
        return self.base.validate_graph(data) if hasattr(self.base, 'validate_graph') else None

    def forward(self, data, return_context=False):
        output = self.base(data, return_context=True)
        v_pred = output['velocities']
        v_fd = self.fd(data.pos, data.velocities, data.atomic_masses,
                       data.atomic_numbers, data.cell)
        v_new = (1.0 - self.blend)*v_pred + self.blend*v_fd
        masses = data.atomic_masses.reshape(-1)
        z = data.atomic_numbers.reshape(-1)
        for index, element in enumerate(self.fd.species):
            mask = z == element
            norm = 0.5*KINETIC_TO_EV*torch.sum(masses[mask, None]*v_new[mask].square())
            target = 0.5*self.fd.dofs[index]*KB_EV_PER_K*self.fd.temperature
            scale = torch.sqrt(torch.clamp(target/norm.clamp_min(1e-12), max=1e6))
            v_new = v_new.clone()
            v_new[mask] = v_new[mask]*scale
        com = (masses[:, None]*v_new).sum(0)/masses.sum()
        v_new = v_new - com
        dq = output['displacements']
        com_dq = (masses[:, None]*dq).sum(0)/masses.sum()
        output['displacements'] = dq - com_dq
        output['fd_velocity'] = v_fd
        output['velocities'] = v_new
        output['positions'] = data.pos + output['displacements']
        self.fd.update(data.pos, v_new, data.atomic_masses, data.atomic_numbers, data.cell)
        if return_context:
            output['context'] = output.get('context', {})
        return output

    def set_diagnostics(self, enabled):
        if hasattr(self.base, 'set_diagnostics'):
            self.base.set_diagnostics(enabled)

    def diagnostics(self):
        return self.base.diagnostics() if hasattr(self.base, 'diagnostics') else []
