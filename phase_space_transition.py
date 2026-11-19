"""Design prototype: narrow edge transport, node-local phase-space coupling.

Independent reference implementation, not wired into the production trainer.
O(3) representations are 0e + 1o + 2e; e3nn supplies CG coefficients and
representation-aware linear maps. No upstream model/kernel code is vendored.
Physical inputs/outputs follow AtomicGraph: Angstrom, fs, amu; edges are
[receiver, sender] and d = q[sender] - q[receiver] + periodic shift.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from e3nn import o3


@dataclass
class PhaseSpaceConfig:
    node_channels: int = 128
    message_channels: int = 32
    edge_channels: int = 64
    geometry_channels: int = 4
    radial_channels: int = 16
    speed_channels: int = 8
    expansion: int = 4
    proposal_layers: int = 2
    refinement_layers: int = 2
    cutoff_angstrom: float = 4.5
    horizon_fs: float = 30.0
    velocity_scale: float = 0.01
    relation_conditioning: str = "predicted_future"


def irreps(channels):
    return o3.Irreps([(channels, (0, 1)), (channels, (1, -1)), (channels, (2, 1))])


def split_orders(x, channels):
    return [part.reshape(*x.shape[:-1], channels, 2*l+1)
            for l, part in enumerate(x.split([channels, 3*channels, 5*channels], -1))]


def join_orders(parts):
    return torch.cat([part.flatten(-2) for part in parts], -1)


def mlp(in_features, hidden, out_features):
    return nn.Sequential(nn.Linear(in_features, hidden), nn.SiLU(),
                         nn.Linear(hidden, out_features))


def cutoff_envelope(r, cutoff):
    return torch.where(r < cutoff, 0.5*(1+torch.cos(math.pi*r/cutoff)),
                       torch.zeros_like(r))


def motion_harmonics(u):
    """Smooth at u=0; magnitude is retained, not replaced with a unit axis."""
    bounded = u * torch.rsqrt(1+u.square().sum(-1, keepdim=True))
    return o3.spherical_harmonics([0, 1, 2], bounded, normalize=False,
                                 normalization="component")


def train_normalization(graphs, cutoff):
    """Frozen TRAIN-only scales required by PhaseSpaceTransition.

    ``graphs`` must yield AtomicGraph-like items with float position targets,
    ``edge_index`` in [central, neighbor] layout, periodic ``shifts`` and the
    formal endpoint targets. The function never reads validation/test data.
    """
    q2 = v2 = squared_cutoff = 0.
    atoms = components = edges = 0
    for g in graphs:
        q2 += g.displacements.double().square().sum().item()
        v2 += g.update_velocities.double().square().sum().item()
        src, dst = g.edge_index.flip(0)
        r = (g.pos[src]-g.pos[dst]+g.shifts).norm(dim=-1)
        if bool((r > cutoff+1e-6).any()):
            raise ValueError("Cache contains edges outside requested cutoff")
        squared_cutoff += cutoff_envelope(r, cutoff).double().square().sum().item()
        atoms += len(g.pos)
        components += g.displacements.numel()
        edges += int(g.num_edges)
    if not (atoms and components and edges):
        raise ValueError("TRAIN normalization requires non-empty graphs")
    return dict(rms_targets=[(q2/components)**.5, (v2/components)**.5],
                aggregation_scale=(squared_cutoff/atoms)**.5,
                samples=len(graphs) if hasattr(graphs, '__len__') else None,
                atoms=atoms, edges=edges,
                source="complete TRAIN split only",
                aggregation_definition=("sqrt(mean over atoms of sum over "
                                        "incoming edges of cosine_cutoff^2)"))


def center_by_mass(value, masses, batch, n_graphs):
    weights = masses.to(value.dtype)
    total = value.new_zeros(n_graphs).index_add(0, batch, weights)
    sums = value.new_zeros(n_graphs, 3).index_add(0, batch, value*weights[:, None])
    mean = (sums/total[:, None])[batch]
    return value-mean, mean


class JointRMS(nn.Module):
    """One invariant scale per atom, balanced over the three orders.

    No coordinate centering, graph statistics, xyz affine bias or LayerScale.
    The unnormalized state remains on the residual path.
    """
    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.channels, self.eps = channels, eps
        self.gain = nn.Parameter(torch.ones(3, channels))

    def forward(self, x):
        parts = split_orders(x, self.channels)
        variance = sum(p.square().mean((-2, -1), keepdim=True) for p in parts)/3
        scale = torch.rsqrt(variance+self.eps)
        return join_orders([p*scale*self.gain[l, :, None] for l, p in enumerate(parts)])


class PhaseEmbedding(nn.Module):
    """Element-conditioned scalar speed basis and regular motion harmonics."""
    def __init__(self, cfg):
        super().__init__()
        self.channels = cfg.node_channels
        self.element = nn.Embedding(119, self.channels)
        self.register_buffer("centers", torch.linspace(0, 1, cfg.speed_channels))
        self.gamma = float(cfg.speed_channels**2)
        self.speed = mlp(self.channels+cfg.speed_channels+1, self.channels,
                         3*self.channels)

    def forward(self, z, masses, relative_velocity):
        square = relative_velocity.square().sum(-1, keepdim=True)
        coordinate = square/(1+square)
        basis = torch.exp(-self.gamma*(coordinate-self.centers).square())
        coefficients = self.speed(torch.cat([self.element(z), basis, masses.log()[:, None]], -1))
        coefficients = coefficients.reshape(-1, 3, self.channels)
        harmonics = motion_harmonics(relative_velocity).split([1, 3, 5], -1)
        return join_orders([coefficients[:, l, :, None]*harmonics[l][:, None]
                            for l in range(3)])


class PhaseRelations(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cutoff = cfg.cutoff_angstrom
        self.register_buffer("centers", torch.linspace(0, self.cutoff, cfg.radial_channels))
        self.gamma = (cfg.radial_channels/self.cutoff)**2
        self.scalar_size = 2*cfg.radial_channels+5

    def forward(self, d0, u0, d1, u1):
        r0 = d0.norm(dim=-1).clamp_min(1e-8)
        r1 = d1.norm(dim=-1).clamp_min(1e-8)
        n0, n1 = d0/r0[:, None], d1/r1[:, None]
        rb0 = torch.exp(-self.gamma*(r0[:, None]-self.centers).square())
        rb1 = torch.exp(-self.gamma*(r1[:, None]-self.centers).square())
        features = torch.cat([rb0, rb1, torch.stack([
            (u0*n0).sum(-1), (u1*n1).sum(-1), u0.square().sum(-1),
            u1.square().sum(-1), (n0*n1).sum(-1)], -1)], -1)
        # Initial support and periodic image correspondence remain unchanged.
        basis = torch.stack([
            o3.spherical_harmonics([0, 1, 2], n0, False, normalization="component"),
            motion_harmonics(u0),
            o3.spherical_harmonics([0, 1, 2], n1, False, normalization="component"),
            motion_harmonics(u1)], 1)
        return features, basis, cutoff_envelope(r0, self.cutoff)


class NodePhaseCoupling(nn.Module):
    """Learned environment x (hidden state + receiver motion) on N atoms.

    Channelwise CG products follow all allowed paths within 0e/1o/2e;
    learned node-level linear maps mix channels before and after the product.
    No dense C^3 tensor product, angular grid or per-edge hidden tensor product.
    """
    def __init__(self, cfg):
        super().__init__()
        c = cfg.node_channels
        self.channels = c
        representation = irreps(c)
        # Unit floor preserves a smooth zero-message limit and amplitude
        # information. Dividing by near-zero environment RMS would cancel the
        # cutoff envelope almost everywhere and create a sharp boundary layer.
        self.environment_norm = JointRMS(c, eps=1.0)
        self.local = o3.Linear(representation, representation)
        self.motion = o3.Linear(representation, representation)
        self.environment = o3.Linear(representation, representation)
        instructions = [(i, j, k, "uuu", True)
                        for i, (_, a) in enumerate(representation)
                        for j, (_, b) in enumerate(representation)
                        for k, (_, out) in enumerate(representation) if out in a*b]
        self.product = o3.TensorProduct(representation, representation, representation,
            instructions, irrep_normalization="component", path_normalization="element")
        self.mix = o3.Linear(representation, representation)
        self.gates = mlp(3*c, cfg.expansion*c, 3*c)

    def forward(self, normalized_state, environment, motion):
        local = (self.local(normalized_state)+self.motion(motion))/math.sqrt(2)
        context = self.environment(self.environment_norm(environment))
        product = self.product(local, context)
        y = self.mix((context+product)/math.sqrt(2))
        scalars, vectors, tensors = split_orders(y, self.channels)
        invariants = torch.cat([scalars[..., 0],
            (vectors.square().mean(-1)+1e-8).sqrt(),
            (tensors.square().mean(-1)+1e-8).sqrt()], -1)
        scalar_delta, vg, tg = self.gates(invariants).split(self.channels, -1)
        # Only scalar coefficients see elementwise nonlinearities.
        return join_orders([scalar_delta[..., None],
                            vectors*torch.sigmoid(vg)[..., None],
                            tensors*torch.sigmoid(tg)[..., None]])


class PhaseSpaceBlock(nn.Module):
    def __init__(self, cfg, relation_size):
        super().__init__()
        c, m, e, g = (cfg.node_channels, cfg.message_channels,
                      cfg.edge_channels, cfg.geometry_channels)
        self.channels, self.message_channels, self.geometry_channels = c, m, g
        self.state_norm = JointRMS(c)
        self.edge_norm = nn.RMSNorm(e, eps=1e-8)
        self.geometry_projection = o3.Linear(irreps(c), o3.Irreps(f"{g}x1o+{g}x2e"))
        self.sender_scalar = nn.Linear(c, e, bias=False)
        self.receiver_scalar = nn.Linear(c, e, bias=False)
        # 2 endpoints x 4 relation bases x 2 orders, plus 2 pair contractions.
        self.edge_extra = nn.Linear(e+relation_size+18*g, e)
        self.edge_out = nn.Linear(e, e)
        self.message_gates = mlp(e, 2*e, 5*m)
        self.down = o3.Linear(irreps(c), irreps(m))
        self.up = o3.Linear(irreps(m), irreps(c))
        self.coupling = NodePhaseCoupling(cfg)

    def geometry_features(self, x, src, dst, bases):
        projected = self.geometry_projection(x)
        parts = [projected[:, :3*self.geometry_channels].reshape(-1, self.geometry_channels, 3),
                 projected[:, 3*self.geometry_channels:].reshape(-1, self.geometry_channels, 5)]
        features = []
        for values, basis in zip(parts, [bases[..., 1:4], bases[..., 4:9]]):
            left, right = values[src], values[dst]
            features.extend([torch.einsum("egm,ebm->ebg", side, basis).flatten(1)
                             for side in (left, right)])
            features.append((left*right).mean(-1))
        return torch.cat(features, -1)

    def forward(self, state, edge_state, edges, relation, motion, aggregation_scale):
        src, dst = edges
        invariants, bases, envelope = relation
        x = self.state_norm(state)
        scalar = x[:, :self.channels]
        geometry = self.geometry_features(x, src, dst, bases)
        extras = torch.cat([self.edge_norm(edge_state), invariants, geometry], -1)
        # Node projections precede indexing; no [E,2*C] concatenation.
        hidden = (self.sender_scalar(scalar)[src]+self.receiver_scalar(scalar)[dst]
                  +self.edge_extra(extras))/math.sqrt(3)
        edge_state = (edge_state+self.edge_out(torch.nn.functional.silu(hidden)))/math.sqrt(2)
        gates = self.message_gates(edge_state).reshape(-1, 5, self.message_channels)
        transport = split_orders(self.down(x), self.message_channels)
        aggregates = []
        # Pure PyTorch reference. This still materializes narrow edge messages;
        # no claim of fused-kernel speedup or measured GPU advantage.
        for l, values in enumerate(transport):
            basis = bases[..., l*l:(l+1)*(l+1)]
            message = gates[:, 0, :, None]*values[src]
            message = message+torch.einsum("ebc,ebm->ecm", gates[:, 1:], basis)
            message = message*envelope[:, None, None]/aggregation_scale
            aggregates.append(torch.zeros_like(values).index_add(0, dst, message))
        environment = self.up(join_orders(aggregates))
        delta = self.coupling(x, environment, motion)
        return (state+delta)/math.sqrt(2), edge_state


class PhaseEndpointHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.node_channels
        self.channels = c
        self.gates = mlp(c, 2*c, 2*c)
        self.vectors = o3.Linear(irreps(c), o3.Irreps(f"{c}x1o"))

    def forward(self, state):
        coefficients = self.gates(state[:, :self.channels]).reshape(-1, 2, self.channels)
        vectors = self.vectors(state).reshape(-1, self.channels, 3)
        return torch.einsum("nac,ncd->nad", coefficients, vectors)/math.sqrt(self.channels)


class PhaseSpaceTransition(nn.Module):
    """Prototype under the existing proposal/refinement and q/v contract.

    rms_targets and aggregation_scale must come only from TRAIN for any formal
    run. aggregation_scale = sqrt(mean_i sum_j cutoff(r_ij)^2). They are frozen
    buffers; do not recompute them on each input graph or simulation state.
    """
    def __init__(self, config, *, rms_targets, aggregation_scale):
        super().__init__()
        self.cfg = config if isinstance(config, PhaseSpaceConfig) else PhaseSpaceConfig(**config)
        self.config = asdict(self.cfg)
        c = self.cfg
        if min(c.node_channels, c.message_channels, c.edge_channels, c.geometry_channels,
               c.radial_channels, c.speed_channels, c.expansion, c.proposal_layers,
               c.refinement_layers, c.cutoff_angstrom, c.horizon_fs, c.velocity_scale,
               aggregation_scale, *rms_targets) <= 0:
            raise ValueError("All widths, horizons and normalization scales must be positive")
        if c.relation_conditioning not in {"predicted_future", "current_only"}:
            raise ValueError("Unknown relation_conditioning")
        self.register_buffer("rms_targets", torch.tensor(rms_targets))
        self.register_buffer("aggregation_scale", torch.tensor(float(aggregation_scale)))
        self.embedding = PhaseEmbedding(c)
        self.relations = PhaseRelations(c)
        self.edge_initial = mlp(self.relations.scalar_size, c.edge_channels, c.edge_channels)
        self.proposal_blocks = nn.ModuleList([PhaseSpaceBlock(c, self.relations.scalar_size)
                                             for _ in range(c.proposal_layers)])
        self.refinement_blocks = nn.ModuleList([PhaseSpaceBlock(c, self.relations.scalar_size)
                                               for _ in range(c.refinement_layers)])
        self.proposal_head = PhaseEndpointHead(c)
        self.correction_head = PhaseEndpointHead(c)

    def validate_graph(self, data):
        """Call at dataset/runner boundaries, not on every GPU forward."""
        if not torch.isfinite(data.pos).all() or not torch.isfinite(data.velocities).all():
            raise ValueError("Non-finite phase-space input")
        if not torch.isfinite(data.atomic_masses).all() or not (data.atomic_masses > 0).all():
            raise ValueError("Masses must be finite and positive")
        if getattr(data, "timestep", None) is not None:
            dt = torch.as_tensor(data.timestep)
            if not torch.allclose(dt, torch.full_like(dt, self.cfg.horizon_fs)):
                raise ValueError("Data timestep differs from model horizon")
        batch = getattr(data, "batch", None)
        if batch is not None and (batch[data.edge_index[0]] != batch[data.edge_index[1]]).any():
            raise ValueError("Cross-system edge")
        if batch is not None and getattr(data, "ptr", None) is None:
            raise ValueError("Batched inputs require ptr to avoid device-to-host size inference")

    def forward(self, data, return_context=False):
        q, velocity = data.pos, data.velocities
        masses, z = data.atomic_masses.reshape(-1), data.atomic_numbers.reshape(-1).long()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(len(q), dtype=torch.long, device=q.device)
            n_graphs = 1
        else:
            n_graphs = data.ptr.numel()-1
        def center(value):
            return center_by_mass(value, masses, batch, n_graphs)[0]
        relative = center(velocity)/self.cfg.velocity_scale
        state = self.embedding(z, masses, relative)
        motion = state
        src, dst = data.edge_index.flip(0)
        edges = (src, dst)
        d0 = q[src]-q[dst]+data.shifts
        u0 = relative[src]-relative[dst]
        initial = self.relations(d0, u0, d0, u0)
        edge_state = self.edge_initial(initial[0])
        for block in self.proposal_blocks:
            state, edge_state = block(state, edge_state, edges, initial, motion, self.aggregation_scale)
        proposal = self.proposal_head(state)
        dq0 = self.cfg.horizon_fs*velocity+self.rms_targets[0]*center(proposal[:, 0])
        v0 = velocity+self.rms_targets[1]*center(proposal[:, 1])
        if self.cfg.relation_conditioning == "predicted_future":
            d1 = d0+dq0[src]-dq0[dst]
            u1 = (v0[src]-v0[dst])/self.cfg.velocity_scale
            relation = self.relations(d0, u0, d1, u1)
            motion = self.embedding(z, masses, center(v0)/self.cfg.velocity_scale)
        else:
            relation = initial
        for block in self.refinement_blocks:
            state, edge_state = block(state, edge_state, edges, relation, motion, self.aggregation_scale)
        correction = self.correction_head(state)
        dq = dq0+self.rms_targets[0]*center(correction[:, 0])
        vn = v0+self.rms_targets[1]*center(correction[:, 1])
        output = dict(displacements=dq, positions=q+dq, velocities=vn,
            proposal_displacements=dq0, proposal_velocities=v0,
            correction_displacements=dq-dq0, correction_velocities=vn-v0,
            target=torch.cat([dq/self.rms_targets[0], vn/self.rms_targets[1]], -1))
        if return_context:
            output["context"] = dict(node=state, edge=edge_state, motion=motion)
        return output

    def set_diagnostics(self, enabled):
        """Observe block output RMS; hooks are removed outside diagnostic batches."""
        hooks = getattr(self, "_diagnostic_hooks", None)
        if enabled and hooks is None:
            self._diagnostic_records = []
            self._diagnostic_hooks = []
            for stage, blocks in [("proposal", self.proposal_blocks),
                                  ("refinement", self.refinement_blocks)]:
                for index, block in enumerate(blocks):
                    def capture(_module, _inputs, outputs, stage=stage, index=index):
                        state, edge_state = outputs
                        self._diagnostic_records.append(dict(
                            stage=stage, index=index,
                            node_rms=float(state.detach().square().mean().sqrt()),
                            edge_rms=float(edge_state.detach().square().mean().sqrt())))
                    self._diagnostic_hooks.append(block.register_forward_hook(capture))
        elif not enabled and hooks is not None:
            for hook in hooks:
                hook.remove()
            self._diagnostic_hooks = None

    def diagnostics(self):
        latest = {}
        for row in getattr(self, "_diagnostic_records", []):
            latest[(row["stage"], row["index"])] = row
        return list(latest.values())
