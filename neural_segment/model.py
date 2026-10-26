"""Motion tokens -> recurrent evolution/coupling -> one temporal coefficient set.

Positions A, velocities A/fs, accelerations A/fs^2, masses u. There is no
potential, energy derivative, physical integrator or per-query graph pass.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from smdp_neural import SMDPConfig, MotionEncoder, RecurrentModeOperator, CrossModeExchange, mlp


@dataclass
class SegmentConfig(SMDPConfig):
    basis_count: int = 6
    variant: str = "full"  # full, independent, direct
    exclude_self_exchange: bool = True
    direct_scalar: int = 192
    direct_vector: int = 48


def integrated_legendre(u, count):
    """L_k(2u-1), integral_0^u L_k, and its second integral.

    Recurrences avoid ill-conditioned evaluation of expanded monomials.
    Both integrals vanish at u=0; derivatives are exact polynomial identities.
    """
    x = 2*u-1
    values = [torch.ones_like(x), x]
    for k in range(1, count+1):
        values.append(((2*k+1)*x*values[-1]-k*values[-2])/(k+1))
    first = [u]
    for k in range(1, count+1):
        first.append((values[k+1]-values[k-1])/(2*(2*k+1)))
    second = [u.square()/2]
    for k in range(1, count):
        second.append((first[k+1]-first[k-1])/(2*(2*k+1)))
    return torch.stack(values[:count], -1), torch.stack(first[:count], -1), torch.stack(second, -1)


def remove_com(coefficients, masses, graph_index):
    """Per-system and per-mode projection; never mixes disconnected graphs."""
    ng = int(graph_index.max())+1
    sums = coefficients.new_zeros((ng, *coefficients.shape[1:]))
    sums.index_add_(0, graph_index, coefficients*masses[:, None, None, None])
    total = masses.new_zeros(ng).index_add_(0, graph_index, masses)
    return coefficients-(sums/total[:, None, None, None])[graph_index]


class IndependentTokens(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Parameter(.02*torch.randn(cfg.modes, cfg.scalar_dim))
        self.scalar = nn.ModuleList([mlp(cfg.scalar_dim, cfg.scalar_dim, cfg.scalar_dim) for _ in range(cfg.modes)])
        self.vector = nn.ModuleList([nn.Linear(cfg.vector_dim, cfg.vector_dim, bias=False) for _ in range(cfg.modes)])
        self.gate = nn.ModuleList([nn.Linear(cfg.scalar_dim, cfg.vector_dim) for _ in range(cfg.modes)])

    def forward(self, s, v):
        states, vectors = [], []
        for k, (scalar, vector, gate) in enumerate(zip(self.scalar, self.vector, self.gate)):
            sk = s+scalar(s+self.embedding[k])+self.embedding[k]
            vk = vector(v.transpose(-1, -2)).transpose(-1, -2)
            states.append(sk)
            vectors.append(vk*torch.sigmoid(gate(sk))[..., None])
        return torch.stack(states, 1), torch.stack(vectors, 1)


class LocalRefine(nn.Module):
    """Capacity control for no-exchange variants: token-local residual update."""
    def __init__(self, ds, dv):
        super().__init__()
        self.scalar = mlp(ds, ds, ds)
        self.vector = nn.Linear(dv, dv, bias=False)
        self.gate = nn.Linear(ds, dv)

    def forward(self, s, v):
        s = s+.1*self.scalar(s)
        return s, v+.1*torch.tanh(self.gate(s))[..., None]*self.vector(v.transpose(-1, -2)).transpose(-1, -2)


class NeuralSegment(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.variant not in ("full", "independent", "direct"):
            raise ValueError(config.variant)
        self.config = config
        self.encoder = MotionEncoder(config)
        if config.variant == "direct":
            ds, dv = config.direct_scalar, config.direct_vector
            self.scalar_project = mlp(config.scalar_dim, ds, ds)
            self.vector_project = nn.Linear(config.vector_dim, dv, bias=False)
        else:
            ds, dv = config.scalar_dim, config.vector_dim
            self.tokens = IndependentTokens(config)
        inner = SMDPConfig(scalar_dim=ds, vector_dim=dv, horizon_fs=config.horizon_fs,
                           recurrent_substeps=config.recurrent_substeps, exclude_self_exchange=True)
        self.recurrent = RecurrentModeOperator(inner)
        self.exchange = CrossModeExchange(inner) if config.variant == "full" else LocalRefine(ds, dv)
        self.norm = nn.LayerNorm(ds)
        self.temporal = mlp(ds, ds, config.basis_count*dv)
        nn.init.normal_(self.temporal[-1].weight, std=1e-3)
        nn.init.zeros_(self.temporal[-1].bias)

    def coefficients(self, q, v, z, masses, edges, shifts, graph_index=None):
        if graph_index is None:
            graph_index = torch.zeros(len(q), dtype=torch.long, device=q.device)
        if q.shape != v.shape or q.shape[-1] != 3 or not bool((masses>0).all()):
            raise ValueError("Invalid phase space or masses")
        # Relative-to-COM input makes response independent of uniform boosts.
        ng = int(graph_index.max())+1
        total = masses.new_zeros(ng).index_add_(0, graph_index, masses)
        cm = v.new_zeros(ng, 3).index_add_(0, graph_index, masses[:, None]*v)/total[:, None]
        s, vectors = self.encoder(q, v-cm[graph_index], z, masses, edges, shifts)
        if self.config.variant == "direct":
            s = self.scalar_project(s)[:, None]
            vectors = self.vector_project(vectors.transpose(-1,-2)).transpose(-1,-2)[:, None]
        else:
            s, vectors = self.tokens(s, vectors)
        # Singleton dimension is a latent computation stage, NOT query time.
        s, vectors = s[None], vectors[None]
        half = q.new_tensor([self.config.horizon_fs/2])
        s, vectors = self.recurrent(s, vectors, half)
        attention = None
        if self.config.variant == "full":
            s, vectors, attention = self.exchange(s, vectors)
        else:
            s, vectors = self.exchange(s, vectors)
        s, vectors = self.recurrent(s, vectors, half)
        s, vectors = s[0], vectors[0]
        n, modes, dv, _ = vectors.shape
        weights = self.temporal(self.norm(s)).reshape(n, modes, self.config.basis_count, dv)
        coeff = torch.einsum('nmkd,nmdc->nmkc', weights, vectors)/math.sqrt(dv)
        coeff = coeff*(self.config.velocity_scale/self.config.horizon_fs)
        return remove_com(coeff, masses, graph_index), attention

    def query(self, q, v, coeff, times_fs):
        times = torch.as_tensor(times_fs, dtype=q.dtype, device=q.device).reshape(-1)
        if not bool(torch.isfinite(times).all()) or bool((times<0).any()) or bool((times>self.config.horizon_fs).any()):
            raise ValueError("Query must be finite and within the segment")
        h = self.config.horizon_fs
        a, iv, iq = integrated_legendre(times/h, self.config.basis_count)
        mode_q = h*h*torch.einsum('tk,nmkc->tnmc', iq, coeff)
        mode_v = h*torch.einsum('tk,nmkc->tnmc', iv, coeff)
        acceleration = torch.einsum('tk,nmkc->tnc', a, coeff)
        return dict(positions=q[None]+times[:, None, None]*v[None]+mode_q.sum(2),
                    velocities=v[None]+mode_v.sum(2), accelerations=acceleration,
                    mode_delta_q=mode_q, mode_delta_v=mode_v, coefficients=coeff)

    def forward(self, positions, velocities, atomic_numbers, masses, edge_index, edge_shift,
                times_fs, graph_index=None):
        coeff, attention = self.coefficients(positions, velocities, atomic_numbers, masses,
                                             edge_index, edge_shift, graph_index)
        out = self.query(positions, velocities, coeff, times_fs)
        out['coupling_attention'] = attention
        return out
