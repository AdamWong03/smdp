"""Fluctuation-Dissipation Readout (FDR) for force-free propagators.

The backbone predicts the displacement and a mean velocity.  The FDR replaces
the velocity with a mode-resolved AR(1) update whose noise is tied to the gain
by the fluctuation-dissipation relation:

    v_mode' = a_mode * v_mode + sqrt(1 - a_mode^2) * noise_mode
    Var(noise_mode) = 2 * <KE_mode> / (dof_mode * m)  (target equipartition)

For |a_mode| < 1 the stationary mode kinetic energy is the canonical value,
independent of the backbone's mode bias.  The displacement keeps the backbone's
structural prediction plus a velocity-consistency correction.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from motion_loss import decompose_modes

KINETIC_TO_EV = 103.64269
KB_EV_PER_K = 8.617333262e-5


class FDReadout(nn.Module):
    def __init__(self, gains=(0.9816, 0.8385, -0.567), temperature=300.0):
        super().__init__()
        self.register_buffer('gains', torch.tensor(gains, dtype=torch.float32))
        self.temperature = float(temperature)

    def forward(self, positions, velocities, masses, molecules, cell):
        masses = masses.reshape(-1)
        molecules = molecules.reshape(-1)
        current = decompose_modes(positions, velocities, masses, molecules, cell)
        noise = torch.randn_like(velocities)
        noise_modes = decompose_modes(positions, noise, masses, molecules, cell)
        n_mol = int(molecules.max()) + 1
        n_atoms = len(masses)
        dofs = (3*n_mol - 3, 3*n_mol, 3*n_atoms - 6*n_mol)
        keys = ('translation', 'rotation', 'internal')
        result = torch.zeros_like(velocities)
        for index, key in enumerate(keys):
            gain = self.gains[index]
            mode_noise = noise_modes[key]
            norm = 0.5*KINETIC_TO_EV*torch.sum(masses[:, None]*mode_noise.square())
            target = 0.5*dofs[index]*KB_EV_PER_K*self.temperature
            scale = torch.sqrt(torch.clamp(target/norm.clamp_min(1e-12), max=1e6))
            result = result + gain*current[key] + torch.sqrt(
                torch.clamp(1.0 - gain*gain, min=0.0))*scale*mode_noise
        com = (masses[:, None]*result).sum(0)/masses.sum()
        return result - com


class FDTransition(nn.Module):
    """Frozen backbone + FD velocity readout; displacement stays from the backbone."""

    def __init__(self, base, gains=(0.9816, 0.8385, -0.567), temperature=300.0,
                 blend=1.0, correct_displacement=False, displacement_weight=None,
                 translation_displacement_weight=0.0, com_displacement_scale=1.0,
                 rescale=False, stochastic_rescale=False, rescale_tau=0.0,
                 global_rescale=False, readout=None):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rms_targets = base.rms_targets
        self.fd = (readout if readout is not None
                   else FDReadout(gains=gains, temperature=temperature))
        self.blend = blend
        self.correct_displacement = bool(correct_displacement)
        if displacement_weight is not None:
            self.displacement_weight = float(displacement_weight)
        else:
            self.displacement_weight = 1.0 if correct_displacement else 0.0
        self.translation_displacement_weight = float(translation_displacement_weight)
        self.com_displacement_scale = float(com_displacement_scale)
        self.rescale = bool(rescale)
        self.stochastic_rescale = bool(stochastic_rescale)
        self.rescale_tau = float(rescale_tau)
        self.global_rescale = bool(global_rescale)
        self.ke_target = None
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def reset(self):
        self.ke_target = None
        if hasattr(self.fd, 'reset'):
            self.fd.reset()

    def validate_graph(self, data):
        return self.base.validate_graph(data) if hasattr(self.base, 'validate_graph') else None

    def forward(self, data, return_context=False):
        output = self.base(data, return_context=True)
        v_pred = output['velocities']
        v_fd = self.fd(data.pos, data.velocities, data.atomic_masses,
                       data.molecule_ids, data.cell)
        horizon = float(self.cfg.horizon_fs)
        if isinstance(self.blend, (tuple, list)):
            masses = data.atomic_masses.reshape(-1)
            molecules = data.molecule_ids.reshape(-1)
            p_pred = decompose_modes(data.pos, v_pred, masses, molecules, data.cell)
            p_fd = decompose_modes(data.pos, v_fd, masses, molecules, data.cell)
            keys = ('translation', 'rotation', 'internal')
            v_new = torch.zeros_like(v_pred)
            for index, key in enumerate(keys):
                weight = float(self.blend[index])
                v_new = v_new + (1.0 - weight)*p_pred[key] + weight*p_fd[key]
        else:
            v_new = (1.0 - float(self.blend))*v_pred + float(self.blend)*v_fd
        if self.rescale:
            masses = data.atomic_masses.reshape(-1)
            molecules = data.molecule_ids.reshape(-1)
            part = decompose_modes(data.pos, v_new, masses, molecules, data.cell)
            n_mol = int(molecules.max()) + 1
            n_atoms = len(masses)
            dofs = (3*n_mol - 3, 3*n_mol, 3*n_atoms - 6*n_mol)
            keys = ('translation', 'rotation', 'internal')
            rescaled = torch.zeros_like(v_new)
            if self.ke_target is None:
                self.ke_target = [None, None, None]
            for index, key in enumerate(keys):
                norm = 0.5*KINETIC_TO_EV*torch.sum(masses[:, None]*part[key].square())
                if self.stochastic_rescale:
                    concentration = torch.as_tensor(
                        0.5*dofs[index], dtype=norm.dtype, device=norm.device)
                    rate = torch.as_tensor(
                        1.0/(KB_EV_PER_K*self.fd.temperature),
                        dtype=norm.dtype, device=norm.device)
                    if self.rescale_tau > 0.0:
                        alpha = math.exp(-float(horizon)/self.rescale_tau)
                        if self.ke_target[index] is None:
                            self.ke_target[index] = torch.distributions.Gamma(
                                concentration, rate).sample()
                        # Gamma AR(1): thinning + Gamma innovation preserves
                        # the Gamma(concentration, rate) stationary marginal.
                        innovation_shape = torch.clamp(
                            concentration*(1.0 - alpha), min=1e-6)
                        innovation = torch.distributions.Gamma(
                            innovation_shape, rate).sample()
                        target = alpha*self.ke_target[index] + innovation
                    else:
                        target = torch.distributions.Gamma(
                            concentration, rate).sample()
                    self.ke_target[index] = target
                else:
                    target = 0.5*dofs[index]*KB_EV_PER_K*self.fd.temperature
                scale = torch.sqrt(torch.clamp(target/norm.clamp_min(1e-12), max=1e6))
                rescaled = rescaled + scale*part[key]
            v_new = rescaled
        if self.global_rescale:
            masses = data.atomic_masses.reshape(-1)
            total = 0.5*KINETIC_TO_EV*torch.sum(masses[:, None]*v_new.square())
            concentration = torch.as_tensor(
                1.5*len(masses), dtype=total.dtype, device=total.device)
            rate = torch.as_tensor(
                1.0/(KB_EV_PER_K*self.fd.temperature),
                dtype=total.dtype, device=total.device)
            target = torch.distributions.Gamma(concentration, rate).sample()
            scale = torch.sqrt(torch.clamp(target/total.clamp_min(1e-12), max=1e6))
            v_new = v_new*scale
        masses = data.atomic_masses.reshape(-1)
        com_v = (masses[:, None]*v_new).sum(0)/masses.sum()
        v_new = v_new - com_v
        dq = output['displacements']
        com_dq = (masses[:, None]*dq).sum(0)/masses.sum()
        output['displacements'] = dq - com_dq
        output['fd_velocity'] = v_fd - (masses[:, None]*v_fd).sum(0)/masses.sum()
        output['velocities'] = v_new
        if hasattr(self.fd, 'update'):
            self.fd.update(data.pos, v_new, data.atomic_masses,
                           data.molecule_ids, data.cell)
        if self.displacement_weight:
            output['displacements'] = (output['displacements']
                                       + self.displacement_weight*horizon*(v_new - v_pred))
        if self.translation_displacement_weight:
            masses = data.atomic_masses.reshape(-1)
            molecules = data.molecule_ids.reshape(-1)
            delta = decompose_modes(data.pos, v_new - v_pred, masses, molecules, data.cell)
            output['displacements'] = (output['displacements']
                                       + self.translation_displacement_weight*horizon
                                       *delta['translation'])
        if self.com_displacement_scale != 1.0:
            masses = data.atomic_masses.reshape(-1)
            molecules = data.molecule_ids.reshape(-1)
            n_mol = int(molecules.max()) + 1
            com = torch.zeros(n_mol, 3, dtype=output['displacements'].dtype,
                              device=output['displacements'].device)
            mass = torch.zeros(n_mol, dtype=output['displacements'].dtype,
                               device=output['displacements'].device)
            com.index_add_(0, molecules, masses[:, None]*output['displacements'])
            mass.index_add_(0, molecules, masses)
            com = com / mass[:, None]
            output['displacements'] = (output['displacements']
                                       + (self.com_displacement_scale - 1.0)*com[molecules])
        output['positions'] = data.pos + output['displacements']
        if return_context:
            output['context'] = output.get('context', {})
        return output

    def set_diagnostics(self, enabled):
        if hasattr(self.base, 'set_diagnostics'):
            self.base.set_diagnostics(enabled)

    def diagnostics(self):
        return self.base.diagnostics() if hasattr(self.base, 'diagnostics') else []
