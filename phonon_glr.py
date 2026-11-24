"""Phonon-mode GLR readout for crystalline solids.

The mode basis is a set of mass-weighted phonon eigenvectors from a finite-
difference Hessian of the same potential that generated the reference MD.  Each
non-acoustic mode carries a scalar amplitude

    s_k = e_k^T (sqrt(m) * v)

with unit variance target k_B T / KINETIC_TO_EV (so that <KE_k> = k_B T / 2).
An AR(p) process is fitted to each mode's reference velocity ACF; the readout
predicts the next amplitude, adds the Yule-Walker innovation noise, and the
transition rescales each mode to a Gamma-distributed canonical kinetic energy.

The backbone still supplies the displacement.  Only the velocity is corrected,
exactly as in the molecular-mode GLR.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

KINETIC_TO_EV = 103.64269
KB_EV_PER_K = 8.617333262e-5


class PhononBasis(nn.Module):
    """Fixed phonon eigenvectors and per-mode AR coefficients."""

    def __init__(self, basis_path, coefficients_path=None, temperature=300.0):
        super().__init__()
        data = np.load(basis_path, allow_pickle=True)
        eigenvectors = data["eigenvectors"].astype(np.float32)
        acoustic = data["acoustic_mask"].astype(bool)
        keep = ~acoustic
        masses = data["masses"].astype(np.float32)
        frequencies = data["frequencies_thz"].astype(np.float32)[keep]

        self.register_buffer("eigenvectors",
                             torch.from_numpy(eigenvectors[:, keep].copy()))
        self.register_buffer("masses", torch.from_numpy(masses))
        self.register_buffer("m12", torch.sqrt(torch.from_numpy(
            np.repeat(masses, 3).astype(np.float32))))
        self.register_buffer("frequencies_thz",
                             torch.from_numpy(frequencies))
        edges = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 40.0])
        band_ids = np.digitize(np.abs(frequencies), edges[1:-1])
        self.register_buffer("band_ids",
                             torch.from_numpy(band_ids.astype(np.int64)))
        self.temperature = float(temperature)
        self.target_ke = 0.5 * KB_EV_PER_K * self.temperature
        self.target_variance = (KB_EV_PER_K * self.temperature
                                / KINETIC_TO_EV)
        self.n_modes = int(self.eigenvectors.shape[1])

        if coefficients_path is None:
            self.register_buffer("coefficients", torch.zeros(self.n_modes, 1))
            self.register_buffer("prediction_error", torch.ones(self.n_modes))
            self.has_coefficients = False
        else:
            coeff = np.load(coefficients_path, allow_pickle=True)
            c = coeff["coeff"].astype(np.float32)[keep]
            pe = coeff["pe"].astype(np.float32)[keep]
            self.register_buffer("coefficients", torch.from_numpy(c.copy()))
            self.register_buffer("prediction_error", torch.from_numpy(pe.copy()))
            self.has_coefficients = True
        self.order = int(self.coefficients.shape[1])

    def project(self, velocities):
        flat = velocities.reshape(-1) * self.m12
        return flat @ self.eigenvectors

    def reconstruct(self, amplitudes):
        flat = self.eigenvectors @ amplitudes
        return (flat / self.m12).reshape(-1, 3)


class PhononGLRReadout(nn.Module):
    """Per-mode AR(p) velocity readout in the phonon eigenbasis."""

    def __init__(self, coefficients, prediction_error, target_variance,
                 use_noise=True):
        super().__init__()
        self.register_buffer("coefficients", coefficients.float())
        self.register_buffer("prediction_error", prediction_error.float())
        self.target_variance = float(target_variance)
        self.order = int(self.coefficients.shape[1])
        self.use_noise = bool(use_noise)
        self.history = None

    def reset(self):
        self.history = None

    def forward(self, velocities, basis):
        current = basis.project(velocities)
        if self.history is None:
            self.history = [current.clone() for _ in range(self.order)]
        predicted = torch.zeros_like(current)
        for lag in range(self.order):
            predicted = predicted + self.coefficients[:, lag] * self.history[-1 - lag]
        if self.use_noise:
            scale = torch.sqrt(torch.clamp(self.prediction_error, min=0.0)
                               * self.target_variance)
            predicted = predicted + scale * torch.randn_like(predicted)
        return basis.reconstruct(predicted)

    def update(self, velocities, basis):
        current = basis.project(velocities)
        if self.history is None:
            self.history = [current.clone() for _ in range(self.order)]
        else:
            self.history.append(current.clone())
            self.history.pop(0)


class PhononGLRTransition(nn.Module):
    """Frozen backbone + phonon-mode GLR velocity correction."""

    def __init__(self, base, basis, readout=None, blend=0.1, rescale=True,
                 stochastic_rescale=True, rescale_tau=0.0,
                 rescale_partition='species', static_scale=None):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rms_targets = base.rms_targets
        self.basis = basis
        self.readout = readout
        if static_scale is None:
            self.static_scale = None
        else:
            self.register_buffer('static_scale',
                                 torch.as_tensor(static_scale, dtype=torch.float32))
        self.blend = float(blend)
        self.rescale = bool(rescale)
        self.stochastic_rescale = bool(stochastic_rescale)
        self.rescale_tau = float(rescale_tau)
        self.rescale_partition = str(rescale_partition)
        self.ke_target = None
        self.ke_targets = {}
        self.mode_scale = None
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def reset(self):
        self.ke_target = None
        self.ke_targets = {}
        self.mode_scale = None
        if self.readout is not None:
            self.readout.reset()

    def validate_graph(self, data):
        return self.base.validate_graph(data) if hasattr(self.base, "validate_graph") else None

    def _sample_target(self, shape, key=None):
        if not self.stochastic_rescale:
            return 0.5 * shape * KB_EV_PER_K * self.basis.temperature
        concentration = torch.as_tensor(0.5 * shape, dtype=torch.float32,
                                        device=self.basis.eigenvectors.device)
        rate = torch.as_tensor(1.0 / (KB_EV_PER_K * self.basis.temperature),
                               dtype=torch.float32,
                               device=self.basis.eigenvectors.device)
        sampled = torch.distributions.Gamma(concentration, rate).sample()
        if self.rescale_tau > 0.0 and key is not None:
            alpha = math.exp(-float(self.cfg.horizon_fs) / self.rescale_tau)
            previous = self.ke_targets.get(key, sampled)
            sampled = alpha * previous + (1.0 - alpha) * sampled
            self.ke_targets[key] = sampled
        return sampled

    def _rescale_species(self, velocities, atomic_numbers):
        masses = self.basis.masses
        z = atomic_numbers.reshape(-1)
        result = velocities.clone()
        for element in torch.unique(z):
            mask = z == element
            n_species = int(mask.sum())
            kinetic = 0.5 * KINETIC_TO_EV * torch.sum(
                masses[mask, None] * result[mask].square())
            target = self._sample_target(3.0 * n_species, key=int(element))
            scale = torch.sqrt(torch.clamp(target / kinetic.clamp_min(1e-12),
                                           max=1e6))
            result[mask] = result[mask] * scale
        return result

    def _rescale_bands(self, velocities):
        amplitudes = self.basis.project(velocities)
        kinetic = 0.5 * KINETIC_TO_EV * amplitudes.square()
        scale = torch.ones_like(amplitudes)
        for band in torch.unique(self.basis.band_ids):
            mask = self.basis.band_ids == band
            n_band = int(mask.sum())
            target = self._sample_target(float(n_band), key=int(band))
            band_ke = kinetic[mask].sum()
            scale[mask] = torch.sqrt(torch.clamp(target / band_ke.clamp_min(1e-12),
                                                 max=1e6))
        return self.basis.reconstruct(amplitudes * scale)

    def _rescale_global(self, velocities):
        amplitudes = self.basis.project(velocities)
        kinetic = 0.5 * KINETIC_TO_EV * amplitudes.square()
        target = self._sample_target(float(amplitudes.numel()), key=0)
        scale = torch.sqrt(torch.clamp(target / kinetic.sum().clamp_min(1e-12),
                                       max=1e6))
        return self.basis.reconstruct(amplitudes * scale)

    def _rescale(self, velocities, atomic_numbers):
        if self.rescale_partition == 'none':
            return velocities
        if self.rescale_partition == 'species':
            return self._rescale_species(velocities, atomic_numbers)
        if self.rescale_partition == 'bands':
            return self._rescale_bands(velocities)
        if self.rescale_partition == 'global':
            return self._rescale_global(velocities)
        amplitudes = self.basis.project(velocities)
        kinetic = 0.5 * KINETIC_TO_EV * amplitudes.square()
        n_modes = amplitudes.numel()
        if self.stochastic_rescale:
            concentration = torch.as_tensor(0.5, dtype=kinetic.dtype,
                                            device=kinetic.device)
            rate = torch.as_tensor(1.0 / (KB_EV_PER_K * self.basis.temperature),
                                   dtype=kinetic.dtype, device=kinetic.device)
            sampled = torch.distributions.Gamma(concentration, rate).sample((n_modes,))
            if self.rescale_tau > 0.0:
                alpha = math.exp(-float(self.cfg.horizon_fs) / self.rescale_tau)
                if self.ke_target is None:
                    self.ke_target = sampled.clone()
                target = alpha * self.ke_target + (1.0 - alpha) * sampled
                self.ke_target = target
            else:
                target = sampled
        else:
            target = torch.full_like(kinetic, 0.5 * KB_EV_PER_K
                                     * self.basis.temperature)
        if self.rescale_tau > 0.0:
            # Integral weak-coupling feedback: accumulate a slow multiplicative
            # scale so that the backbone's persistent mode bias is corrected,
            # while the mode phase and short-time ACF are left intact.
            if self.mode_scale is None:
                self.mode_scale = torch.ones_like(amplitudes)
            scaled = amplitudes * self.mode_scale
            kinetic_scaled = 0.5 * KINETIC_TO_EV * scaled.square()
            ratio = torch.clamp(target / kinetic_scaled.clamp_min(1e-12),
                                min=1e-6, max=1e6)
            exponent = float(self.cfg.horizon_fs) / self.rescale_tau
            self.mode_scale = torch.clamp(
                self.mode_scale * ratio ** exponent, min=0.1, max=10.0)
            return self.basis.reconstruct(amplitudes * self.mode_scale)
        scale = torch.sqrt(torch.clamp(target / kinetic.clamp_min(1e-12), max=1e6))
        return self.basis.reconstruct(amplitudes * scale)

    def forward(self, data, return_context=False):
        output = self.base(data, return_context=True)
        v_pred = output["velocities"]
        if self.readout is not None:
            v_fd = self.readout(data.velocities, self.basis)
            v_new = (1.0 - self.blend) * v_pred + self.blend * v_fd
        else:
            v_fd = torch.zeros_like(v_pred)
            v_new = v_pred
        if self.static_scale is not None:
            amplitudes = self.basis.project(v_new)
            v_new = self.basis.reconstruct(amplitudes * self.static_scale)
        if self.rescale:
            v_new = self._rescale(v_new, data.atomic_numbers)

        masses = data.atomic_masses.reshape(-1)
        com = (masses[:, None] * v_new).sum(0) / masses.sum()
        v_new = v_new - com

        output["fd_velocity"] = v_fd
        output["velocities"] = v_new
        output["positions"] = data.pos + output["displacements"]
        if self.readout is not None:
            self.readout.update(v_new, self.basis)
        if return_context:
            output["context"] = output.get("context", {})
        return output

    def set_diagnostics(self, enabled):
        if hasattr(self.base, "set_diagnostics"):
            self.base.set_diagnostics(enabled)

    def diagnostics(self):
        return self.base.diagnostics() if hasattr(self.base, "diagnostics") else []
