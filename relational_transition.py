"""Endpoint-supervised proposal/relational-refinement network.

Independent scalar/vector network: no TrajCast model inheritance or pretrained
weights. Reuses only this project's radial encoding and MLP primitives.
Input graph follows AtomicGraph: positions/shifts in A, velocities in A/fs,
masses in amu; official edge displacement = pos[edge[1]]-pos[edge[0]]+shifts.
Each forward predicts ONE fixed-horizon endpoint, not intermediate MD steps.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from smdp_neural import RBF, mlp


@dataclass
class RelationalConfig:
    scalar_channels: int = 128
    vector_channels: int = 32
    edge_channels: int = 64
    radial_channels: int = 16
    cutoff_angstrom: float = 4.5
    horizon_fs: float = 30.0
    proposal_layers: int = 2
    refinement_layers: int = 2
    velocity_scale: float = 0.01
    # Default preserves historical checkpoints/configs exactly.
    block_type: str = 'legacy'


def mass_mean(x, masses, batch):
    """Separate COM for every physical graph, including unequal atom counts."""
    ng = int(batch.max()) + 1
    # Autocast can make x BF16 while masses stay FP32. Accumulate the small
    # per-system reduction in the promoted precision and return x's dtype.
    acc_dtype = torch.promote_types(x.dtype, masses.dtype)
    weights = masses.to(acc_dtype)
    total = torch.zeros(ng, device=x.device, dtype=acc_dtype).index_add(0, batch, weights)
    sums = torch.zeros(ng, 3, device=x.device, dtype=acc_dtype).index_add(
        0, batch, weights[:, None] * x.to(acc_dtype))
    return (sums / total[:, None])[batch].to(x.dtype)


def mix_vectors(layer, v):
    # Channel mixing only: no xyz biases or axis-wise nonlinearities.
    return layer(v.transpose(-1, -2)).transpose(-1, -2)


def pair_mlp(network, nodes, src, dst, extras, factorized=True):
    """Exact affine factorization, NOT a low-rank/model approximation.

    W[s_i,s_j,e] = W_i s_i + W_j s_j + W_e e. Compute node
    projections N times rather than E times, and do not materialize [E,2*Ds].
    The unfactored path is retained solely for numerical/performance checks.
    """
    if not factorized:
        return network(torch.cat([nodes[src], nodes[dst], extras], -1))
    width = nodes.shape[-1]
    first = network[0]
    sender = F.linear(nodes, first.weight[:, :width])
    receiver = F.linear(nodes, first.weight[:, width:2*width])
    hidden = sender[src] + receiver[dst]
    hidden = hidden + F.linear(extras, first.weight[:, 2*width:], first.bias)
    return network[2](network[1](hidden))


class RelationEncoding(nn.Module):
    """Matched initial/proposed edge geometry with continuous image identity."""
    def __init__(self, cfg):
        super().__init__()
        self.rbf = RBF(cfg.radial_channels, cfg.cutoff_angstrom)
        self.scalar_size = 2 * cfg.radial_channels + 5
        self.cutoff = cfg.cutoff_angstrom

    def forward(self, d0, u0, d1, u1):
        r0 = d0.norm(dim=-1).clamp_min(1e-8)
        r1 = d1.norm(dim=-1).clamp_min(1e-8)
        n0, n1 = d0 / r0[:, None], d1 / r1[:, None]
        scalars = torch.cat([
            self.rbf(r0), self.rbf(r1),
            torch.stack([(u0 * n0).sum(-1), (u1 * n1).sum(-1),
                         u0.square().sum(-1), u1.square().sum(-1),
                         (n0 * n1).sum(-1)], -1),
        ], -1)
        vectors = torch.stack([n0, n1, u0, u1], 1)
        # Initial graph support is shared across stages. Candidate geometry is
        # a feature, not a second cutoff that can discard useful original edges.
        weight = torch.where(r0 < self.cutoff,
                             0.5 * (1 + torch.cos(torch.pi * r0 / self.cutoff)),
                             torch.zeros_like(r0))
        return scalars, vectors, weight


class RelationalBlock(nn.Module):
    """Persistently update edge state, then invariant/equivariant node state."""
    def __init__(self, cfg, relation_dim):
        super().__init__()
        ds, dv, de = cfg.scalar_channels, cfg.vector_channels, cfg.edge_channels
        self.ds, self.dv = ds, dv
        self.factorized_edges = True
        self.edge_update = mlp(2 * ds + de + relation_dim + 2 * dv, ds, de)
        self.edge_norm = nn.LayerNorm(de)
        self.message = mlp(de, ds, ds + 5 * dv)
        self.sender_mix = nn.Linear(dv, dv, bias=False)
        self.node_update = mlp(2 * ds + 3 * dv, ds, ds)
        self.node_norm = nn.LayerNorm(ds)
        self.vector_mix = nn.Linear(dv, dv, bias=False)
        self.vector_gate = nn.Linear(ds, 2 * dv)

    def forward(self, s, v, e, edges, relation):
        src, dst = edges
        invariants, directions, weight = relation
        # Projections allow oriented hidden states to affect edge computation.
        axis = directions[:, 0]
        projections = torch.cat([(v[src] * axis[:, None]).sum(-1),
                                 (v[dst] * axis[:, None]).sum(-1)], -1)
        extras = torch.cat([e, invariants, projections], -1)
        e = self.edge_norm(e + pair_mlp(self.edge_update, s, src, dst, extras,
                                       self.factorized_edges))
        message = self.message(e)
        ms = message[:, :self.ds]
        coefficients = message[:, self.ds:].reshape(-1, 5, self.dv)
        mv = torch.einsum('ekc,ekd->ecd', coefficients[:, :4], directions)
        mv = mv + coefficients[:, 4, :, None] * mix_vectors(self.sender_mix, v)[src]
        # Accumulate in state precision even when GEMMs use autocast.
        aggregate_s = torch.zeros_like(s).index_add(0, dst, (ms * weight[:, None]).to(s.dtype))
        aggregate_v = torch.zeros_like(v).index_add(0, dst, (mv * weight[:, None, None]).to(v.dtype))
        degree = s.new_zeros(len(s)).index_add(0, dst, weight).clamp_min(1)
        aggregate_s = aggregate_s / degree[:, None]
        aggregate_v = aggregate_v / degree[:, None, None]
        inv = torch.cat([s, aggregate_s, v.square().sum(-1),
                         aggregate_v.square().sum(-1), (v * aggregate_v).sum(-1)], -1)
        s = self.node_norm(s + self.node_update(inv))
        gates = torch.sigmoid(self.vector_gate(s)).reshape(len(s), 2, self.dv)
        v = v + gates[:, 0, :, None] * aggregate_v
        v = v + 0.1 * gates[:, 1, :, None] * mix_vectors(self.vector_mix, v)
        return s, v, e


class GeometryRelationalBlock(nn.Module):
    """Dual-state edge geometry, equivariant messages and node-local mixing.

    All learned vector projections run on nodes before edge indexing. Only
    invariant contractions are stored as edge descriptors; no triplets, rank-2
    features or attention tensors are constructed. Current and candidate axes
    have the same descriptor families but separate slots (not swap symmetry).
    """
    def __init__(self, cfg, relation_dim):
        super().__init__()
        ds, dv, de = cfg.scalar_channels, cfg.vector_channels, cfg.edge_channels
        self.ds, self.dv = ds, dv
        self.factorized_edges = True
        self.relation_vectors = nn.Linear(dv, 2 * dv, bias=False)
        self.edge_update = mlp(2 * ds + de + relation_dim + 7 * dv, ds, de)
        self.edge_norm = nn.LayerNorm(de)
        self.message = mlp(de, ds, ds + 5 * dv)
        self.sender_mix = nn.Linear(dv, dv, bias=False)
        self.scalar_update = mlp(2 * ds, ds, ds)
        self.scalar_norm = nn.LayerNorm(ds)
        self.mixing_vectors = nn.Linear(dv, 2 * dv, bias=False)
        self.mixing = mlp(ds + 2 * dv, ds, 2 * ds + dv)
        self.dot_to_scalar = nn.Linear(dv, ds, bias=False)
        self.output_norm = nn.LayerNorm(ds)

    def geometry_features(self, v, edges, directions):
        src, dst = edges
        left, right = mix_vectors(self.relation_vectors, v).split(self.dv, dim=1)
        left, right = left[src], right[dst]
        dot = (left * right).sum(-1)
        # r_perp.s_perp = r.s - (r.n)(s.n), without materializing
        # two [edges, channels, xyz] transverse tensors for each direction.
        features = []
        for axis in directions[:, :2].unbind(1):
            lp = (left * axis[:, None]).sum(-1)
            rp = (right * axis[:, None]).sum(-1)
            features.extend([lp, rp, dot - lp * rp])
        return torch.cat([*features, dot], -1)

    def forward(self, s, v, e, edges, relation):
        src, dst = edges
        invariants, directions, weight = relation
        geometry = self.geometry_features(v, edges, directions)
        extras = torch.cat([e, invariants, geometry], -1)
        e = self.edge_norm(e + pair_mlp(self.edge_update, s, src, dst, extras,
                                       self.factorized_edges))
        message = self.message(e)
        ms = message[:, :self.ds]
        coefficients = message[:, self.ds:].reshape(-1, 5, self.dv)
        mv = torch.einsum('ekc,ekd->ecd', coefficients[:, :4], directions)
        mv = mv + coefficients[:, 4, :, None] * mix_vectors(self.sender_mix, v)[src]
        aggregate_s = torch.zeros_like(s).index_add(0, dst, (ms * weight[:, None]).to(s.dtype))
        aggregate_v = torch.zeros_like(v).index_add(0, dst, (mv * weight[:, None, None]).to(v.dtype))
        degree = s.new_zeros(len(s)).index_add(0, dst, weight).clamp_min(1)
        aggregate_s = aggregate_s / degree[:, None]
        aggregate_v = aggregate_v / degree[:, None, None]
        s = self.scalar_norm(s + self.scalar_update(torch.cat([s, aggregate_s], -1)))
        v = v + aggregate_v
        u, w = mix_vectors(self.mixing_vectors, v).split(self.dv, dim=1)
        norm = (u.square().sum(-1) + 1e-8).sqrt()
        dot = (u * w).sum(-1)
        context = self.mixing(torch.cat([s, norm, dot], -1))
        scalar_delta, scalar_gate, vector_gate = context.split([self.ds, self.ds, self.dv], -1)
        s = self.output_norm(s + scalar_delta + scalar_gate * self.dot_to_scalar(dot))
        v = v + torch.tanh(vector_gate)[:, :, None] * u
        return s, v, e


class GatedGeometryRelationalBlock(nn.Module):
    """Geometry-complete block with bounded residual vector updates.

    This keeps the richer R2 relation descriptors but restores the stable R1
    update topology: aggregated vectors are scalar-gated before entering the
    residual stream and node-local vector mixing is a bounded correction.
    """
    def __init__(self, cfg, relation_dim):
        super().__init__()
        ds, dv, de = cfg.scalar_channels, cfg.vector_channels, cfg.edge_channels
        self.ds, self.dv = ds, dv
        self.factorized_edges = True
        self.relation_vectors = nn.Linear(dv, 2 * dv, bias=False)
        self.edge_update = mlp(2 * ds + de + relation_dim + 7 * dv, ds, de)
        self.edge_norm = nn.LayerNorm(de)
        self.message = mlp(de, ds, ds + 5 * dv)
        self.sender_mix = nn.Linear(dv, dv, bias=False)
        self.scalar_update = mlp(2 * ds, ds, ds)
        self.scalar_norm = nn.LayerNorm(ds)
        self.aggregate_gate = nn.Linear(ds, dv)
        self.mixing_vectors = nn.Linear(dv, 2 * dv, bias=False)
        self.mixing = mlp(ds + 2 * dv, ds, ds + dv)
        self.output_norm = nn.LayerNorm(ds)

    def geometry_features(self, v, edges, directions):
        src, dst = edges
        left, right = mix_vectors(self.relation_vectors, v).split(self.dv, dim=1)
        left, right = left[src], right[dst]
        dot = (left * right).sum(-1)
        features = []
        for axis in directions[:, :2].unbind(1):
            lp = (left * axis[:, None]).sum(-1)
            rp = (right * axis[:, None]).sum(-1)
            features.extend([lp, rp, dot - lp * rp])
        return torch.cat([*features, dot], -1)

    def forward(self, s, v, e, edges, relation):
        src, dst = edges
        invariants, directions, weight = relation
        geometry = self.geometry_features(v, edges, directions)
        extras = torch.cat([e, invariants, geometry], -1)
        e = self.edge_norm(e + pair_mlp(self.edge_update, s, src, dst, extras,
                                       self.factorized_edges))
        message = self.message(e)
        ms = message[:, :self.ds]
        coefficients = message[:, self.ds:].reshape(-1, 5, self.dv)
        mv = torch.einsum('ekc,ekd->ecd', coefficients[:, :4], directions)
        mv = mv + coefficients[:, 4, :, None] * mix_vectors(self.sender_mix, v)[src]
        aggregate_s = torch.zeros_like(s).index_add(0, dst, (ms * weight[:, None]).to(s.dtype))
        aggregate_v = torch.zeros_like(v).index_add(0, dst, (mv * weight[:, None, None]).to(v.dtype))
        degree = s.new_zeros(len(s)).index_add(0, dst, weight).clamp_min(1)
        aggregate_s = aggregate_s / degree[:, None]
        aggregate_v = aggregate_v / degree[:, None, None]
        s = self.scalar_norm(s + self.scalar_update(torch.cat([s, aggregate_s], -1)))
        v = v + torch.sigmoid(self.aggregate_gate(s))[:, :, None] * aggregate_v
        u, w = mix_vectors(self.mixing_vectors, v).split(self.dv, dim=1)
        norm = (u.square().sum(-1) + 1e-8).sqrt()
        dot = (u * w).sum(-1)
        scalar_delta, vector_gate = self.mixing(
            torch.cat([s, norm, dot], -1)).split([self.ds, self.dv], -1)
        s = self.output_norm(s + scalar_delta)
        v = v + 0.1 * torch.tanh(vector_gate)[:, :, None] * w
        return s, v, e


class EndpointHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gates = mlp(cfg.scalar_channels, cfg.scalar_channels, 2 * cfg.vector_channels)
        self.vectors = nn.Linear(cfg.vector_channels, cfg.vector_channels, bias=False)
        self.dv = cfg.vector_channels

    def forward(self, s, v):
        coefficients = self.gates(s).reshape(len(s), 2, self.dv)
        return torch.einsum('nkc,ncd->nkd', coefficients, mix_vectors(self.vectors, v)) / self.dv**0.5


class RelationalTransition(nn.Module):
    """Current relations -> candidate endpoint -> future-conditioned correction.

    All modules are trained from scratch. Physical q/v are never replaced by
    latent coordinates; the same original graph/image correspondence is used
    for both stages. Rebuild the graph from the new state on each autoregressive
    call using the existing AtomicGraph neighbor implementation.
    """
    def __init__(self, config=None, rms_targets=(0.07, 0.004)):
        super().__init__()
        self.cfg = config if isinstance(config, RelationalConfig) else RelationalConfig(**(config or {}))
        self.config = asdict(self.cfg)
        self.factorized_edges = True
        c = self.cfg
        block_classes = {'legacy': RelationalBlock, 'geometry': GeometryRelationalBlock,
                         'geometry_gated': GatedGeometryRelationalBlock}
        if c.block_type not in block_classes:
            raise ValueError(f'Unknown block_type: {c.block_type}')
        block_class = block_classes[c.block_type]
        if min(c.horizon_fs, c.velocity_scale, *rms_targets) <= 0:
            raise ValueError('Horizon and unit scales must be positive')
        self.register_buffer('rms_targets', torch.tensor(rms_targets))
        self.element = nn.Embedding(119, c.scalar_channels)
        self.kinematics = mlp(2, c.scalar_channels, c.scalar_channels)
        self.velocity_embedding = nn.Embedding(119, c.vector_channels)
        self.relations = RelationEncoding(c)
        self.edge_embedding = mlp(2 * c.scalar_channels + self.relations.scalar_size,
                                  c.scalar_channels, c.edge_channels)
        self.proposal_blocks = nn.ModuleList([
            block_class(c, self.relations.scalar_size) for _ in range(c.proposal_layers)])
        self.refinement_blocks = nn.ModuleList([
            block_class(c, self.relations.scalar_size) for _ in range(c.refinement_layers)])
        self.proposal_head = EndpointHead(c)
        self.correction_head = EndpointHead(c)

    def forward(self, data):
        q, vel = data.pos, data.velocities
        z, masses = data.atomic_numbers.reshape(-1).long(), data.atomic_masses.reshape(-1)
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(len(q), dtype=torch.long, device=q.device)
        if bool((masses <= 0).any()):
            raise ValueError('Masses must be positive')
        # AtomicGraph/torch_nl stores [central, neighbor], unlike the older
        # project's vesin adapter. Normalize once to [sender, receiver].
        edges = data.edge_index.flip(0)
        src, dst = edges
        if bool((batch[src] != batch[dst]).any()):
            raise ValueError('Cross-system edge in batched graph')
        if getattr(data, 'timestep', None) is not None:
            dt = torch.as_tensor(data.timestep, device=q.device, dtype=q.dtype)
            if not bool(torch.allclose(dt, torch.full_like(dt, self.cfg.horizon_fs))):
                raise ValueError('Data timestep differs from fixed model horizon')
        center = lambda value: value - mass_mean(value, masses, batch)
        vrel = center(vel) / self.cfg.velocity_scale
        s = self.element(z) + self.kinematics(torch.stack([masses.log(), vrel.square().sum(-1)], -1))
        v = self.velocity_embedding(z)[:, :, None] * vrel[:, None, :]
        d0 = q[src] - q[dst] + data.shifts
        u0 = vrel[src] - vrel[dst]
        initial = self.relations(d0, u0, d0, u0)
        e = pair_mlp(self.edge_embedding, s, src, dst, initial[0], self.factorized_edges)
        for block in self.proposal_blocks:
            s, v, e = block(s, v, e, edges, initial)
        proposal = self.proposal_head(s, v)
        dq0 = self.cfg.horizon_fs * vel + self.rms_targets[0] * center(proposal[:, 0])
        v0 = vel + self.rms_targets[1] * center(proposal[:, 1])
        # NO detach: final endpoint gradients pass through candidate relations.
        d1 = d0 + dq0[src] - dq0[dst]
        u1 = (v0[src] - v0[dst]) / self.cfg.velocity_scale
        future = self.relations(d0, u0, d1, u1)
        for block in self.refinement_blocks:
            s, v, e = block(s, v, e, edges, future)
        correction = self.correction_head(s, v)
        dq = dq0 + self.rms_targets[0] * center(correction[:, 0])
        vn = v0 + self.rms_targets[1] * center(correction[:, 1])
        return dict(displacements=dq, positions=q + dq, velocities=vn,
                    proposal_displacements=dq0, proposal_velocities=v0,
                    correction_displacements=dq-dq0, correction_velocities=vn-v0,
                    target=torch.cat([dq / self.rms_targets[0], vn / self.rms_targets[1]], -1))


def endpoint_objective(model, output, reference_displacements, reference_velocities,
                       proposal_weight=0.1):
    """Only real endpoint labels; no internal-time labels or auxiliary physics."""
    def error(dq, v):
        return 0.5 * (((dq-reference_displacements)/model.rms_targets[0]).square().mean()
                      + ((v-reference_velocities)/model.rms_targets[1]).square().mean())
    final = error(output['displacements'], output['velocities'])
    proposal = error(output['proposal_displacements'], output['proposal_velocities'])
    return final + proposal_weight * proposal, dict(final=final.detach(), proposal=proposal.detach())
