"""SACMP: Spectrally Anchored Canonical Mode Propagator.

Single-crystal, fixed-topology implementation. Position: Angstrom; velocity:
Angstrom/fs; mass: amu; time: fs. The canonical coordinates are
Q=sqrt(m)*(x-x_ref-COM_shift), P=sqrt(m)*(v-v_COM).
No trained BKS force or Hessian is evaluated during propagation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SACMPConfig:
    rank: int = 192
    hidden: int = 80
    radial_basis: int = 12
    cutoff_a: float = 4.5
    max_substep_fs: float = 4.0
    max_query_fs: float = 48.0
    coupling_scale: float = 0.004
    slow_edge_scale: float = 0.003
    fast_edge_scale: float = 0.001
    slow_onsite_freq: float = 0.025
    mode_trainable: bool = True


def _net(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs)
    )


class CanonicalPotential(nn.Module):
    """A single invariant scalar energy, with explicit slow and mixed terms.

    Differentiation of this scalar generates reciprocal momentum updates in
    BOTH canonical subspaces. The full topology and reference bonds stay fixed.
    """
    def __init__(self, cfg: SACMPConfig):
        super().__init__()
        self.cfg = cfg
        features = cfg.radial_basis + 2 + 8
        self.slow_net = _net(features, cfg.hidden, 2)
        self.fast_net = _net(features, cfg.hidden, 2)
        self.mixed_net = _net(features, cfg.hidden, 2)
        self.raw_slow_frequency = nn.Parameter(
            torch.tensor(math.log(math.expm1(cfg.slow_onsite_freq))))
        # Start at uncoupled stable modes, then let supervised dynamics learn mixing.
        for net in (self.fast_net, self.mixed_net):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, qf, qs, W, geometry):
        mass = geometry["mass"]
        n = mass.numel()
        bs = qs.shape[0]
        src, dst = geometry["edge_index"]
        rhat = geometry["direction"]
        rbf = geometry["rbf"]
        species = geometry["species"]
        mass_edge = geometry["edge_mass"]
        rootm = mass.sqrt().view(1, n, 1)
        fast = (qf @ W.T).reshape(bs, n, 3) / rootm
        slow = qs.reshape(bs, n, 3) / rootm
        df = fast[:, src] - fast[:, dst]
        ds = slow[:, src] - slow[:, dst]
        pf = (df * rhat).sum(-1)
        ps = (ds * rhat).sum(-1)
        ff = (df * df).sum(-1)
        ss = (ds * ds).sum(-1)
        fs = (df * ds).sum(-1)
        transverse_cross = fs - pf * ps
        transverse_slow = (ss - ps.square()).clamp_min(0.0)
        transverse_fast = (ff - pf.square()).clamp_min(0.0)
        features = torch.cat([
            rbf.unsqueeze(0).expand(bs, -1, -1),
            species.unsqueeze(0).expand(bs, -1, -1),
            (pf / .15).unsqueeze(-1),
            (ps / .15).unsqueeze(-1),
            (ff / .04).unsqueeze(-1),
            (ss / .04).unsqueeze(-1),
            (fs / .04).unsqueeze(-1),
            (pf * ps / .04).unsqueeze(-1),
            (transverse_cross / .04).unsqueeze(-1),
            (transverse_slow / .04).unsqueeze(-1)
        ], dim=-1)
        # Each edge is included once, and all energy terms vanish at q=0.
        zeros = torch.zeros_like(pf)
        # Strict energy decomposition: U_f depends only on q_f; U_s only
        # on q_s. The third scalar term U_fs generates cross-mode transfer.
        feat_s = torch.cat([
            rbf.unsqueeze(0).expand(bs, -1, -1),
            species.unsqueeze(0).expand(bs, -1, -1),
            zeros.unsqueeze(-1), (ps/.15).unsqueeze(-1),
            zeros.unsqueeze(-1), (ss/.04).unsqueeze(-1),
            zeros.unsqueeze(-1), zeros.unsqueeze(-1),
            zeros.unsqueeze(-1),
            (transverse_slow/.04).unsqueeze(-1)
        ], -1)
        feat_f = torch.cat([
            rbf.unsqueeze(0).expand(bs, -1, -1),
            species.unsqueeze(0).expand(bs, -1, -1),
            (pf/.15).unsqueeze(-1), zeros.unsqueeze(-1),
            (ff/.04).unsqueeze(-1), zeros.unsqueeze(-1),
            zeros.unsqueeze(-1), zeros.unsqueeze(-1),
            zeros.unsqueeze(-1),
            (transverse_fast/.04).unsqueeze(-1)
        ], -1)
        slow_coeff = F.softplus(self.slow_net(feat_s))
        fast_coeff = torch.tanh(self.fast_net(feat_f))
        fast_pair = .5 * self.cfg.fast_edge_scale * mass_edge * (
            fast_coeff[..., 0] * pf.square()
            + fast_coeff[..., 1] * transverse_fast)
        slow_pair = .5 * self.cfg.slow_edge_scale * mass_edge * (
            slow_coeff[..., 0] * ps.square()
            + slow_coeff[..., 1] * transverse_slow)
        mixed_coeff = torch.tanh(self.mixed_net(features))
        mixed_pair = self.cfg.coupling_scale * mass_edge * (
            mixed_coeff[..., 0] * pf * ps
            + mixed_coeff[..., 1] * transverse_cross)
        slow_omega = F.softplus(self.raw_slow_frequency)
        onsite = .5 * slow_omega.square() * qs.square().sum(-1)
        return onsite + slow_pair.sum(-1) + fast_pair.sum(-1) + mixed_pair.sum(-1)


class SACMP(nn.Module):
    """Orthogonal modal chart + exact harmonic fast phase + neural potential kick.

    For any fixed W, each propagation substep is a composition of exact
    symplectic flows (fast harmonic rotation, learned potential kick, slow drift).
    It is self-adjoint: Phi(-h) = inverse(Phi(h)), to solver precision.
    This statement does NOT imply conservation of the external BKS energy.
    """
    def __init__(self, cfg: SACMPConfig, reference, mass, atomic_numbers,
                 edge_index, edge_shift, initial_basis, initial_omega):
        super().__init__()
        reference = torch.as_tensor(reference, dtype=torch.float32)
        mass = torch.as_tensor(mass, dtype=torch.float32)
        atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_shift = torch.as_tensor(edge_shift, dtype=torch.float32)
        basis = torch.as_tensor(initial_basis, dtype=torch.float32)
        freq = torch.as_tensor(initial_omega, dtype=torch.float32)
        n = mass.numel()
        if reference.shape != (n, 3) or (mass <= 0).any():
            raise ValueError("Invalid reference geometry or masses")
        if basis.shape != (3*n, cfg.rank) or freq.shape != (cfg.rank,):
            raise ValueError("Mode basis and frequency shape mismatch")
        if not 0 < cfg.rank <= 3*n-3:
            raise ValueError("Rank must be between 1 and 3N-3")
        if cfg.max_substep_fs <= 0:
            raise ValueError("max_substep_fs must be positive")
        self.cfg = cfg
        self.register_buffer("reference", reference)
        self.register_buffer("mass", mass)
        self.register_buffer("atomic_numbers", atomic_numbers)
        self.register_buffer("basis_seed", basis)
        self.basis_raw = nn.Parameter(basis.clone(), requires_grad=cfg.mode_trainable)
        self.log_omega = nn.Parameter(freq.clamp(.002, .4).log())
        self.potential = CanonicalPotential(cfg)
        # Vesin returns directed full-list edges. Use ONE orientation to avoid
        # counting a physical pair twice; distinct periodic images are retained.
        src_all, dst_all = edge_index
        mask = src_all < dst_all
        src, dst = src_all[mask], dst_all[mask]
        if src.numel() == 0:
            raise ValueError("Empty periodic reference graph")
        dr = reference[src] + edge_shift[mask] - reference[dst]
        length = torch.linalg.vector_norm(dr, dim=-1).clamp_min(1e-9)
        if length.max() >= cfg.cutoff_a + 1e-3:
            raise ValueError("Reference graph exceeds configured cutoff")
        centers = torch.linspace(0.0, cfg.cutoff_a, cfg.radial_basis)
        rbf = torch.exp(-((length[:, None]-centers)/(cfg.cutoff_a/cfg.radial_basis))**2)
        # Distances are fixed reference values; no hard cutoff encountered at runtime.
        envelope = .5*(torch.cos(math.pi * length / cfg.cutoff_a)+1.)
        rbf = rbf*envelope[:, None]
        species = torch.stack([
            ((atomic_numbers[src]==8)&(atomic_numbers[dst]==8)).float(),
            ((atomic_numbers[src]==14)&(atomic_numbers[dst]==14)).float()
        ], dim=-1)
        self.register_buffer("edge_index", torch.stack([src, dst]))
        self.register_buffer("direction", dr / length[:, None])
        self.register_buffer("rbf", rbf)
        self.register_buffer("species", species)
        self.register_buffer("edge_mass", mass[src].mul(mass[dst]).sqrt())
        self.register_buffer("mode_mass", mass.sqrt().repeat_interleave(3))

    def geometry(self):
        return dict(mass=self.mass, edge_index=self.edge_index,
                    direction=self.direction, rbf=self.rbf,
                    species=self.species, edge_mass=self.edge_mass)

    def modes(self):
        n = self.mass.numel()
        root = self.mass.sqrt()
        # Remove the three mass-weighted translation directions before QR.
        raw = self.basis_raw.reshape(n, 3, self.cfg.rank)
        proj = (raw*root[:, None, None]).sum(dim=0)/root.square().sum().sqrt()
        centering = root[:, None, None]/root.square().sum().sqrt()
        constrained = (raw-centering*proj[None]).reshape(3*n, self.cfg.rank)
        W, R = torch.linalg.qr(constrained, mode="reduced")
        # Fix QR sign convention for reproducible channel identities.
        signs = torch.where(torch.diagonal(R) >= 0, 1.0, -1.0)
        return W*signs.unsqueeze(0)

    def _encode(self, x, v, W):
        if x.ndim == 2:
            x, v = x.unsqueeze(0), v.unsqueeze(0)
        if x.shape != v.shape or x.shape[-2:] != self.reference.shape:
            raise ValueError("Expected matching [batch, N, 3] x and v")
        root = self.mass.sqrt().view(1, -1, 1)
        msum = self.mass.sum()
        dx = x-self.reference
        cm_shift = (dx*self.mass.view(1, -1, 1)).sum(dim=1, keepdim=True)/msum
        cm_vel = (v*self.mass.view(1, -1, 1)).sum(dim=1, keepdim=True)/msum
        Q = ((dx-cm_shift)*root).flatten(start_dim=1)
        P = ((v-cm_vel)*root).flatten(start_dim=1)
        qf = Q@W
        pf = P@W
        qs = Q-qf@W.T
        ps = P-pf@W.T
        return qf, pf, qs, ps, cm_shift, cm_vel

    def _decode(self, qf, pf, qs, ps, cm_shift, cm_vel, W, h):
        bs = qf.shape[0]
        n = self.mass.numel()
        root = self.mass.sqrt().view(1, n, 1)
        Q = (qf@W.T+qs).reshape(bs, n, 3)
        P = (pf@W.T+ps).reshape(bs, n, 3)
        x = self.reference+cm_shift+h*cm_vel+Q/root
        v = cm_vel+P/root
        return x, v

    def _fast(self, qf, pf, h):
        omega = self.log_omega.exp().clamp(.002, .4)
        # Nonlinear action-dependent twists make d(x)/dt != decoded
        # physical velocity. Harmonic phase flow keeps q_dot=p exactly.
        theta = h*omega
        c, s = torch.cos(theta), torch.sin(theta)
        return c*qf+(s/omega)*pf, c*pf-(s*omega)*qf

    def _kick(self, qf, qs, pf, ps, step, W, geometry):
        # During evaluation callers may use torch.no_grad(), but obtaining
        # momentum kicks requires differentiation of a scalar neural energy.
        tracing = torch.is_grad_enabled()
        with torch.enable_grad():
            # IMPORTANT: the canonical qf and qs are mathematically
            # independent chart coordinates, but qs = Q-W@qf in the
            # encoding autograd graph. Differentiating U directly wrt
            # the original qf would incorrectly include a path through
            # qs, so the kick would depend on computation history and
            # cease to be reversible/symplectic. Clones create two
            # independent differentiation INPUT nodes while preserving
            # their paths to original coordinates for higher derivatives.
            fq = (qf.clone() if tracing else qf.detach().clone()).requires_grad_(True)
            sq = (qs.clone() if tracing else qs.detach().clone()).requires_grad_(True)
            scalar = self.potential(fq, sq, W, geometry).sum()
            gf, gs = torch.autograd.grad(
                scalar, (fq, sq), create_graph=tracing)
        # The slow subspace constraint is canonical: P_s = I-WW^T-C_COM.
        gs = gs-(gs@W)@W.T
        centroid = (gs.reshape(-1, self.mass.numel(), 3) *
                    self.mass.sqrt().view(1, -1, 1)).sum(dim=1)
        centroid = centroid / self.mass.sum()
        gs = gs-(centroid[:, None, :]*self.mass.sqrt()[None, :, None]).flatten(1)
        if not tracing:
            gf, gs = gf.detach(), gs.detach()
        return pf-step*gf, ps-step*gs

    def _propagate(self, initial, h, W, geometry):
        if abs(h) > self.cfg.max_query_fs+1e-8:
            raise ValueError("Requested time exceeds max_query_fs")
        qf, pf, qs, ps = initial[:4]
        if h == 0:
            return qf, pf, qs, ps
        pieces = max(1, math.ceil(abs(h)/self.cfg.max_substep_fs-1e-12))
        dt = h/pieces
        for _ in range(pieces):
            qf, pf = self._fast(qf, pf, dt*.5)
            pf, ps = self._kick(qf, qs, pf, ps, dt*.5, W, geometry)
            qs = qs+dt*ps
            pf, ps = self._kick(qf, qs, pf, ps, dt*.5, W, geometry)
            qf, pf = self._fast(qf, pf, dt*.5)
        return qf, pf, qs, ps

    def forward(self, x, v, times_fs):
        squeezed = x.ndim == 2
        if torch.is_tensor(times_fs):
            times = times_fs.detach().cpu().tolist()
        else:
            times = list(times_fs)
        if not times:
            raise ValueError("At least one query time is required")
        if any(not math.isfinite(float(t)) or
               abs(float(t)) > self.cfg.max_query_fs + 1e-8 for t in times):
            raise ValueError("Requested time is nonfinite or exceeds max_query_fs")
        W = self.modes()
        initial = self._encode(x, v, W)
        geometry = self.geometry()
        # Reuse the exact 4fs substep prefix for aligned queries. This
        # preserves the mathematical map while avoiding recomputation for
        # supervised 4/8/16fs targets and long dense trajectories.
        grid = self.cfg.max_substep_fs
        eligible = sorted({float(t) for t in times if t >= 0 and
                           abs(float(t)/grid-round(float(t)/grid)) < 1e-9})
        cached = {0.0: initial[:4]}
        prev = 0.0
        for t in eligible:
            if t > prev:
                cached[t] = self._propagate(cached[prev], t-prev, W, geometry)
                prev = t
        positions, velocities = [], []
        for t in times:
            t = float(t)
            state = cached.get(t)
            if state is None:
                state = self._propagate(initial, t, W, geometry)
            qf, pf, qs, ps = state
            xp, vp = self._decode(qf, pf, qs, ps, initial[4], initial[5], W, t)
            positions.append(xp.squeeze(0) if squeezed else xp)
            velocities.append(vp.squeeze(0) if squeezed else vp)
        return dict(positions=torch.stack(positions),
                    velocities=torch.stack(velocities))

    def learned_energy(self, x, v):
        """Surrogate conserved Hamiltonian (not BKS physical energy)."""
        W = self.modes()
        qf, pf, qs, ps, _, cm_vel = self._encode(x, v, W)
        omega = self.log_omega.exp().clamp(.002, .4)
        energy = .5*(pf.square()+omega.square()*qf.square()).sum(-1)
        energy = energy+.5*ps.square().sum(-1)
        energy = energy+self.potential(qf, qs, W, self.geometry())
        energy = energy+.5*self.mass.sum()*cm_vel.squeeze(1).square().sum(-1)
        return energy

    def export_config(self):
        return asdict(self.cfg)
