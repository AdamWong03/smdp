"""SMDP: learned motion modes, mode-conditioned propagation, coupled segment evolution.

This is an independent PyTorch model CORE, not the original action/Hessian/PET SMDP.

Contract:
    positions [N,3] Angstrom, velocities [N,3] Angstrom/fs
    atomic_numbers [N] long, masses [N] positive (amu)
    edge_index [2,E]: edge_index[0]=sender, edge_index[1]=receiver
    edge_shift [E,3] Cartesian Angstrom: displacement = pos[sender]+shift-pos[receiver]
    times_fs [T] in [0, horizon_fs]
    output {'positions': [T,N,3], 'velocities': [T,N,3], ...}

Graph construction should use a scalable periodic neighbor-list (e.g. vesin) in the
actual project. build_periodic_edges is a small-system reference fallback only.

The computational design is deliberately independent of generating functions,
Hessians, analytic oscillators and force predictions. It contains no hidden
state carried across independent MD steps; each step consumes full phase space.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SMDPConfig:
    scalar_dim: int = 128
    vector_dim: int = 32
    modes: int = 4
    encoder_layers: int = 4
    rbf_dim: int = 16
    cutoff_angstrom: float = 4.5
    horizon_fs: float = 16.0
    velocity_scale: float = 0.01  # Angstrom/fs, a fixed input/output unit scale
    recurrent_substeps: int = 2  # per half-stage, 4 per full propagation
    enforce_center_of_mass: bool = True


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


def period_edge_displacements(
    positions: torch.Tensor, edge_index: torch.Tensor, edge_shift: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src, dst = edge_index
    dr = positions[src] + edge_shift - positions[dst]
    distance = torch.linalg.vector_norm(dr, dim=-1).clamp_min(1e-8)
    return dr, distance, dr / distance[:, None]


def build_periodic_edges(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """O(N^2) minimum-image reference neighbor builder, suitable only for small cells.

    Cell rows are lattice vectors. This minimum-image-only implementation requires
    that the cutoff be below half the cell's shortest perpendicular dimension.
    For production and large supercells use a verified vesin neighbor builder.
    """
    n = positions.shape[0]
    diff = positions[None, :, :] - positions[:, None, :]  # [dst,src,3]
    frac = diff @ torch.linalg.inv(cell)
    offset = -torch.round(frac)
    mapped = diff + offset @ cell
    d = torch.linalg.vector_norm(mapped, dim=-1)
    mask = (d < cutoff) & (~torch.eye(n, device=positions.device, dtype=torch.bool))
    dst, src = torch.where(mask)
    edge_index = torch.stack([src, dst], 0)
    shift = offset[dst, src] @ cell
    return edge_index, shift


class RBF(nn.Module):
    def __init__(self, n: int, cutoff: float):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0.0, cutoff, n))
        self.gamma = (n / cutoff) ** 2
        self.cutoff = cutoff

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        x = torch.exp(-self.gamma * (distance[:, None] - self.centers) ** 2)
        envelope = (0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)).clamp(min=0.0)
        return x * torch.where(distance < self.cutoff, envelope, torch.zeros_like(envelope))[:, None]


class EquivariantMP(nn.Module):
    """Scalar/vector O(3)-equivariant local message passing (PaiNN-inspired).

    Scalar coefficients depend only on invariant inputs, while vector outputs
    are linear combinations of equivariant input vectors/edge directions.
    """
    def __init__(self, ds: int, dv: int, radial: int):
        super().__init__()
        self.ds, self.dv = ds, dv
        self.edge_mlp = mlp(2 * ds + radial + 1, ds, ds + 2 * dv)
        self.update_s = mlp(2 * ds + 2 * dv, ds, ds)
        self.vector_gate = nn.Linear(ds, dv)
        self.vector_mix = nn.Linear(dv, dv, bias=False)

    def forward(self, s, v, edge_index, edge_u, rbf, dv_parallel):
        src, dst = edge_index
        n = s.shape[0]
        feat = torch.cat([s[src], s[dst], rbf, dv_parallel[:, None]], dim=-1)
        msg = self.edge_mlp(feat)
        msg_s, gate = msg[:, :self.ds], msg[:, self.ds:].reshape(-1, 2, self.dv)
        msg_v = gate[:, 0, :, None] * edge_u[:, None, :] + gate[:, 1, :, None] * v[src]
        agg_s = torch.zeros_like(s).index_add_(0, dst, msg_s)
        agg_v = torch.zeros_like(v).index_add_(0, dst, msg_v)
        degree = torch.zeros((n, 1), device=s.device, dtype=s.dtype).index_add_(0, dst, torch.ones((dst.numel(), 1), device=s.device, dtype=s.dtype))
        inv_degree = degree.clamp_min(1.0).reciprocal()
        agg_s = agg_s * inv_degree
        agg_v = agg_v * inv_degree[:, :, None]
        inv = torch.cat([s, agg_s, torch.linalg.vector_norm(v, dim=-1), torch.linalg.vector_norm(agg_v, dim=-1)], dim=-1)
        s = s + self.update_s(inv)
        v_mixed = self.vector_mix(v.transpose(-1, -2)).transpose(-1, -2)
        v = v + torch.sigmoid(self.vector_gate(s))[:, :, None] * (agg_v + 0.1 * v_mixed)
        return s, v


class MotionEncoder(nn.Module):
    def __init__(self, cfg: SMDPConfig):
        super().__init__()
        ds, dv = cfg.scalar_dim, cfg.vector_dim
        self.cfg = cfg
        self.element = nn.Embedding(120, ds)
        self.mass_speed = mlp(2, ds, ds)
        self.velocity_embed = nn.Embedding(120, dv)
        self.rbf = RBF(cfg.rbf_dim, cfg.cutoff_angstrom)
        self.layers = nn.ModuleList([EquivariantMP(ds, dv, cfg.rbf_dim) for _ in range(cfg.encoder_layers)])

    def forward(self, q, v, z, masses, edge_index, edge_shift):
        if z.dtype != torch.long:
            z = z.long()
        vel_scaled = v / self.cfg.velocity_scale
        radial_vel = torch.linalg.vector_norm(vel_scaled, dim=-1)
        s = self.element(z) + self.mass_speed(torch.stack([masses.log(), radial_vel], dim=-1))
        vectors = self.velocity_embed(z)[:, :, None] * vel_scaled[:, None, :]
        dr, distance, unit = period_edge_displacements(q, edge_index, edge_shift)
        rbf = self.rbf(distance)
        src, dst = edge_index
        dv_parallel = ((vel_scaled[src] - vel_scaled[dst]) * unit).sum(-1)
        for layer in self.layers:
            s, vectors = layer(s, vectors, edge_index, unit, rbf, dv_parallel)
        return s, vectors


class LearnedMotionModes(nn.Module):
    def __init__(self, cfg: SMDPConfig):
        super().__init__()
        self.cfg = cfg
        self.assign_s = nn.Linear(cfg.scalar_dim, cfg.modes * cfg.scalar_dim)
        self.assign_v = nn.Linear(cfg.scalar_dim, cfg.modes * cfg.vector_dim)
        self.mode_embedding = nn.Parameter(0.01 * torch.randn(cfg.modes, cfg.scalar_dim))

    def forward(self, s, v):
        n, ds = s.shape
        k, dv = self.cfg.modes, self.cfg.vector_dim
        a_s = self.assign_s(s).reshape(n, k, ds).softmax(dim=1)
        a_v = self.assign_v(s).reshape(n, k, dv).softmax(dim=1)
        s_mode = a_s * s[:, None, :] + self.mode_embedding[None, :, :]
        v_mode = a_v[:, :, :, None] * v[:, None, :, :]
        return s_mode, v_mode, {'scalar_allocation': a_s, 'vector_allocation': a_v}


class RecurrentModeOperator(nn.Module):
    def __init__(self, cfg: SMDPConfig):
        super().__init__()
        ds, dv = cfg.scalar_dim, cfg.vector_dim
        self.cfg = cfg
        self.cell = nn.GRUCell(ds + 2, ds)
        self.vector_gate = nn.Linear(ds + 2, dv)
        self.vector_linear = nn.Linear(dv, dv, bias=False)

    def forward(self, s, v, dt):
        # s [T,N,K,Ds], v [T,N,K,Dv,3], dt [T] fs.
        t, n, k, ds = s.shape
        for _ in range(self.cfg.recurrent_substeps):
            a = dt / (self.cfg.horizon_fs * self.cfg.recurrent_substeps)
            time_embed = torch.stack([a, torch.sin(math.pi * a)], dim=-1)[:, None, None, :].expand(t, n, k, 2)
            x = torch.cat([s, time_embed], dim=-1)
            s_update = self.cell(x.reshape(-1, ds + 2), s.reshape(-1, ds)).reshape(t, n, k, ds)
            s = s + a[:, None, None, None] * (s_update - s)
            gate = torch.tanh(self.vector_gate(torch.cat([s, time_embed], dim=-1)))
            v_rot = self.vector_linear(v.transpose(-1, -2)).transpose(-1, -2)
            v = v + a[:, None, None, None, None] * gate[..., None] * v_rot
        return s, v


class DirectModeOperator(nn.Module):
    def __init__(self, cfg: SMDPConfig):
        super().__init__()
        self.cfg = cfg
        self.delta_s = mlp(cfg.scalar_dim + 2, cfg.scalar_dim, cfg.scalar_dim)
        self.vector_gate = mlp(cfg.scalar_dim + 2, cfg.scalar_dim, cfg.vector_dim)
        self.vector_linear = nn.Linear(cfg.vector_dim, cfg.vector_dim, bias=False)

    def forward(self, s, v, dt):
        t, n, k, _ = s.shape
        a = dt / self.cfg.horizon_fs
        emb = torch.stack([a, torch.sin(math.pi * a)], dim=-1)[:, None, None, :].expand(t, n, k, 2)
        feat = torch.cat([s, emb], dim=-1)
        v_delta = self.vector_linear(v.transpose(-1, -2)).transpose(-1, -2)
        v = v + a[:, None, None, None, None] * torch.tanh(self.vector_gate(feat))[..., None] * v_delta
        s = s + a[:, None, None, None] * self.delta_s(feat)
        return s, v


class ModeConditionedPropagation(nn.Module):
    def __init__(self, cfg: SMDPConfig):
        super().__init__()
        self.cfg = cfg
        self.recurrent = RecurrentModeOperator(cfg)
        self.direct = DirectModeOperator(cfg)
        self.router = mlp(cfg.scalar_dim + 2, cfg.scalar_dim, 2)

    def forward(self, s, v, dt):
        t, n, k, _ = s.shape
        a = dt / self.cfg.horizon_fs
        emb = torch.stack([a, torch.sin(math.pi * a)], dim=-1)[:, None, None, :].expand(t, n, k, 2)
        routing = self.router(torch.cat([s, emb], dim=-1)).softmax(dim=-1)
        s_rec, v_rec = self.recurrent(s, v, dt)
        s_dir, v_dir = self.direct(s, v, dt)
        s_new = routing[..., :1] * s_rec + routing[..., 1:] * s_dir
        v_new = routing[..., :1, None] * v_rec + routing[..., 1:, None] * v_dir
        return s_new, v_new, routing


class CrossModeExchange(nn.Module):
    """Atom-local cross-mode attention, including equivariant vector exchange."""
    def __init__(self, cfg: SMDPConfig):
        super().__init__()
        ds, dv = cfg.scalar_dim, cfg.vector_dim
        self.q = nn.Linear(ds, ds, bias=False)
        self.k = nn.Linear(ds, ds, bias=False)
        self.value = nn.Linear(ds, ds)
        self.s_proj = mlp(ds, ds, ds)
        self.vector_gate = nn.Linear(ds, dv)
        self.vector_linear = nn.Linear(dv, dv, bias=False)
        self.norm = nn.LayerNorm(ds)

    def forward(self, s, v):
        sn = self.norm(s)
        scores = torch.einsum('tnkd,tnjd->tnkj', self.q(sn), self.k(sn)) / math.sqrt(sn.shape[-1])
        attention = scores.softmax(dim=-1)
        s_message = torch.einsum('tnkj,tnjd->tnkd', attention, self.value(sn))
        v_message = torch.einsum('tnkj,tnjdc->tnkdc', attention, v)
        s_out = s + self.s_proj(s_message)
        v_message = self.vector_linear(v_message.transpose(-1, -2)).transpose(-1, -2)
        v_out = v + torch.tanh(self.vector_gate(s_out))[..., None] * v_message
        return s_out, v_out, attention


class EquivariantSegmentDecoder(nn.Module):
    def __init__(self, cfg: SMDPConfig):
        super().__init__()
        self.cfg = cfg
        self.mode_weights = nn.Linear(cfg.scalar_dim, 1)
        self.scalar_norm = nn.LayerNorm(cfg.scalar_dim)
        self.coeffs = mlp(cfg.scalar_dim + 2, cfg.scalar_dim, 2 * cfg.vector_dim)
        # Small nonzero initialization is essential: unlike an all-zero head,
        # it allows gradients to reach the mode/propagation network on step 1.
        nn.init.normal_(self.coeffs[-1].weight, std=1e-3)
        nn.init.zeros_(self.coeffs[-1].bias)

    def forward(self, s, v, q, vel, mass, times_fs, graph_index=None):
        t, n, k, ds = s.shape
        weights = self.mode_weights(s).softmax(dim=2)
        s_fused = (weights * s).sum(dim=2)
        v_fused = (weights[..., None] * v).sum(dim=2)
        a = times_fs / self.cfg.horizon_fs
        emb = torch.stack([a, torch.sin(math.pi * a)], dim=-1)[:, None, :].expand(t, n, 2)
        coeff = self.coeffs(torch.cat([self.scalar_norm(s_fused), emb], dim=-1))
        cq, cv = coeff.chunk(2, dim=-1)
        move_q = (cq[..., None] * v_fused).sum(dim=-2) / math.sqrt(v_fused.shape[-2])
        move_v = (cv[..., None] * v_fused).sum(dim=-2) / math.sqrt(v_fused.shape[-2])
        delta_q = times_fs[:, None, None] * self.cfg.velocity_scale * move_q
        delta_v = a[:, None, None] * self.cfg.velocity_scale * move_v
        if self.cfg.enforce_center_of_mass:
            if graph_index is None:
                graph_index = torch.zeros(n, dtype=torch.long, device=mass.device)
            # Each disconnected sample conserves its own initial COM motion.
            ng = int(graph_index.max()) + 1
            total_mass = mass.new_zeros(ng).index_add(0, graph_index, mass)
            def remove_com(delta):
                sums = delta.new_zeros(t, ng, 3).index_add(
                    1, graph_index, delta * mass[None, :, None])
                return delta - (sums / total_mass[None, :, None])[:, graph_index]
            delta_q, delta_v = remove_com(delta_q), remove_com(delta_v)
        pred_q = q[None, :, :] + times_fs[:, None, None] * vel[None, :, :] + delta_q
        pred_v = vel[None, :, :] + delta_v
        return pred_q, pred_v, weights


class NeuralSMDP(nn.Module):
    """Full neural segment predictor. Expensive atomic encoding runs once per call."""
    def __init__(self, config: SMDPConfig | None = None):
        super().__init__()
        cfg = config or SMDPConfig()
        self.config = cfg
        self.encoder = MotionEncoder(cfg)
        self.decompose = LearnedMotionModes(cfg)
        self.propagate = ModeConditionedPropagation(cfg)
        self.couple = CrossModeExchange(cfg)
        self.decode = EquivariantSegmentDecoder(cfg)

    def forward(self, positions, velocities, atomic_numbers, masses,
                edge_index, edge_shift, times_fs, graph_index=None):
        times_fs = torch.as_tensor(times_fs, dtype=positions.dtype, device=positions.device).reshape(-1)
        if torch.any(times_fs < 0) or torch.any(times_fs > self.config.horizon_fs + 1e-6):
            raise ValueError('times_fs must fall in [0, horizon_fs]')
        if positions.ndim != 2 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
            raise ValueError('positions and velocities must be [N,3]')
        if masses.ndim != 1 or masses.numel() != positions.shape[0] or not (masses > 0).all():
            raise ValueError('masses must be [N] positive')
        s, v = self.encoder(positions, velocities, atomic_numbers, masses, edge_index, edge_shift)
        sm, vm, allocation = self.decompose(s, v)
        n_times = times_fs.shape[0]
        sm = sm.unsqueeze(0).expand(n_times, -1, -1, -1)
        vm = vm.unsqueeze(0).expand(n_times, -1, -1, -1, -1)
        half = times_fs * 0.5
        s1, v1, router_1 = self.propagate(sm, vm, half)
        s2, v2, attention = self.couple(s1, v1)
        s3, v3, router_2 = self.propagate(s2, v2, half)
        next_pos, next_vel, fusion_weights = self.decode(
            s3, v3, positions, velocities, masses, times_fs, graph_index)
        return {
            'positions': next_pos,
            'velocities': next_vel,
            'mode_scalar_allocation': allocation['scalar_allocation'],
            'mode_vector_allocation': allocation['vector_allocation'],
            'operator_routing': torch.stack([router_1, router_2], dim=0),
            'coupling_attention': attention,
            'fusion_weights': fusion_weights,
        }
