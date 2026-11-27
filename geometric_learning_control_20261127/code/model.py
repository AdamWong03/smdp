"""ParityFlow: momentum-parity-graded equivariant message-passing flow map.

The model receives only the central state (q, p) and a graph built at the
central configuration.  It predicts four spatial-vector fields:

    Q_even, Q_odd, P_even, P_odd

with mass-weighted position-delta and momentum units.  The even/odd labels are
momentum-reversal parities (tau=+1 / tau=-1), distinct from spatial O(3)
parity encoded by e3nn irreps.

No forces, energies, future coordinates, or numerical microsteps are used.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from e3nn import o3


def _slice_irrep(x: torch.Tensor, irreps: o3.Irreps, index: int) -> torch.Tensor:
    """Slice one irrep block (with multiplicity) from a flat feature tensor."""
    chunks = irreps.slices()
    return x[..., chunks[index]]


class ParityFlow(nn.Module):
    """Independent momentum-parity-graded equivariant flow map."""

    def __init__(
        self,
        num_species: int,
        hidden: int = 32,
        bottleneck: int = 16,
        blocks: int = 4,
        edge_rbf: int = 4,
        cutoff: float = 6.0,
        q_scale: float = 1.0,
        p_scale: float = 1.0,
        mass_mean: float = 1.0,
        use_edge_message: bool = False,
        use_global_q: bool = True,
        use_tp_readout: bool = False,
        use_receiver_gate: bool = False,
        use_two_endpoint: bool = False,
    ):
        super().__init__()
        self.num_species = int(num_species)
        self.hidden = int(hidden)
        self.cutoff = float(cutoff)
        self.edge_rbf = int(edge_rbf)
        self.q_scale = float(q_scale)
        self.p_scale = float(p_scale)
        self.mass_mean = float(mass_mean)
        self.use_edge_message = bool(use_edge_message)
        self.use_global_q = bool(use_global_q)
        self.use_tp_readout = bool(use_tp_readout)
        self.use_receiver_gate = bool(use_receiver_gate)
        self.use_two_endpoint = bool(use_two_endpoint)

        self.irreps_h = o3.Irreps(
            f"{hidden}x0e+{hidden}x1o+{hidden}x2e")
        self.irreps_b = o3.Irreps(
            f"{bottleneck}x0e+{bottleneck}x1o+{bottleneck}x2e")
        self.irreps_edge = o3.Irreps(
            f"{edge_rbf}x0e+{edge_rbf}x1o+{edge_rbf}x2e")
        self.sh = o3.SphericalHarmonics(
            "0e+1o+2e", normalize=True, normalization="component")

        self.register_buffer(
            "rbf_centers", torch.linspace(1.0, cutoff, edge_rbf))
        self.register_buffer(
            "rbf_width", torch.tensor(max(cutoff / max(edge_rbf - 1, 1), 1e-3)))

        # Initial even scalar features: species, log mass, speed^2, |q_rel|^2.
        self.even_scalar_mlp = nn.Sequential(
            nn.Linear(num_species + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Odd scalar is tanh(q_rel . p_tilde / scale) times an even MLP.
        self.odd_scalar_mlp = nn.Sequential(
            nn.Linear(num_species + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.even_q_linear = o3.Linear("1o", f"{hidden}x1o")
        self.odd_p_linear = o3.Linear("1o", f"{hidden}x1o")
        self.even_q2_tp = o3.FullyConnectedTensorProduct(
            "1o", "1o", "2e", internal_weights=True, shared_weights=True)
        self.even_p2_tp = o3.FullyConnectedTensorProduct(
            "1o", "1o", "2e", internal_weights=True, shared_weights=True)
        self.odd_qp_tp = o3.FullyConnectedTensorProduct(
            "1o", "1o", "2e", internal_weights=True, shared_weights=True)
        self.even_q2_linear = o3.Linear("2e", f"{hidden}x2e")
        self.even_p2_linear = o3.Linear("2e", f"{hidden}x2e")
        self.odd_qp_linear = o3.Linear("2e", f"{hidden}x2e")

        # Message tensor products (edge features are even under momentum reversal).
        self.msg_even = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_edge, self.irreps_h,
                internal_weights=True, shared_weights=True)
            for _ in range(blocks)
        ])
        self.msg_odd = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_edge, self.irreps_h,
                internal_weights=True, shared_weights=True)
            for _ in range(blocks)
        ])
        self.msg_even_linear = nn.ModuleList([
            o3.Linear(self.irreps_h, self.irreps_h)
            for _ in range(blocks)
        ])
        self.msg_odd_linear = nn.ModuleList([
            o3.Linear(self.irreps_h, self.irreps_h)
            for _ in range(blocks)
        ])

        # Node-local parity-graded tensor products.
        self.node_ee = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_h, self.irreps_b,
                internal_weights=True, shared_weights=True)
            for _ in range(blocks)
        ])
        self.node_oo = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_h, self.irreps_b,
                internal_weights=True, shared_weights=True)
            for _ in range(blocks)
        ])
        self.node_eo = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_h, self.irreps_b,
                internal_weights=True, shared_weights=True)
            for _ in range(blocks)
        ])
        self.node_oe = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_h, self.irreps_b,
                internal_weights=True, shared_weights=True)
            for _ in range(blocks)
        ])
        self.node_even_linear = nn.ModuleList([
            o3.Linear(self.irreps_b, self.irreps_h)
            for _ in range(blocks)
        ])
        self.node_odd_linear = nn.ModuleList([
            o3.Linear(self.irreps_b, self.irreps_h)
            for _ in range(blocks)
        ])
        # Even scalar gates: one gate per irrep multiplicity, per stream block.
        self.gate_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 3 * hidden),
            )
            for _ in range(blocks)
        ])
        self.post_linear_even = nn.ModuleList([
            o3.Linear(self.irreps_h, self.irreps_h)
            for _ in range(blocks)
        ])
        self.post_linear_odd = nn.ModuleList([
            o3.Linear(self.irreps_h, self.irreps_h)
            for _ in range(blocks)
        ])
        for gate in self.gate_mlp:
            gate[-1].bias.data.fill_(2.0)

        # Targeted direct even polar-vector message: a learnable radial scalar
        # times the relative edge vector, summed over neighbours.  This is an
        # even (0e x 1o -> 1o) geometric path that is added after the stream's
        # RMS normalization, so it cannot be suppressed by the global q l=1
        # feature.  It is shared across blocks with one learnable scale each.
        if self.use_edge_message:
            self.edge_message_mlp = nn.Sequential(
                nn.Linear(edge_rbf, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.edge_message_scale = nn.Parameter(
                torch.full((blocks,), 1.0))
        if self.use_receiver_gate:
            self.receiver_gate = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * hidden + edge_rbf, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(blocks)
            ])
        if self.use_two_endpoint:
            self.msg_even2 = nn.ModuleList([
                o3.FullyConnectedTensorProduct(
                    self.irreps_h, self.irreps_h, self.irreps_b)
                for _ in range(blocks)
            ])
            self.msg_even2_linear = nn.ModuleList([
                o3.Linear(self.irreps_b, self.irreps_h)
                for _ in range(blocks)
            ])
            self.msg_even2_scale = nn.Parameter(torch.ones(blocks))

        # Four spatial-vector heads.  The default reads the l=1 block of each
        # stream; the TP variant is a separate diagnostic readout.
        if self.use_tp_readout:
            self.q_even_head = o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_h, "1o")
            self.q_odd_head = o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_h, "1o")
            self.p_even_head = o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_h, "1o")
            self.p_odd_head = o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_h, "1o")
        else:
            self.q_even_head = o3.Linear(f"{hidden}x1o", "1o")
            self.q_odd_head = o3.Linear(f"{hidden}x1o", "1o")
            self.p_even_head = o3.Linear(f"{hidden}x1o", "1o")
            self.p_odd_head = o3.Linear(f"{hidden}x1o", "1o")

    @staticmethod
    def _rms_norm(x: torch.Tensor, irreps: o3.Irreps,
                  eps: float = 1e-6) -> torch.Tensor:
        """Even, O(3)-equivariant RMS normalization per irrep block."""
        chunks = irreps.slices()
        out = []
        for sl in chunks:
            xb = x[..., sl]
            scale = torch.sqrt(torch.mean(xb * xb, dim=-1, keepdim=True) + eps)
            out.append(xb / scale)
        return torch.cat(out, dim=-1)

    def _edge_features(self, edge_vec: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.vector_norm(edge_vec, dim=-1).clamp_min(1e-8)
        sh = self.sh(edge_vec / r[:, None])  # [E, 1x0e+1x1o+1x2e]
        diff = (r[:, None] - self.rbf_centers[None, :]) / self.rbf_width
        rbf = torch.exp(-diff * diff)  # [E, edge_rbf]
        envelope = torch.where(
            r < self.cutoff,
            0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0),
            torch.zeros_like(r),
        )
        rbf = rbf * envelope[:, None]
        # Reorder as edge_rbf x 0e, edge_rbf x 1o, edge_rbf x 2e.
        s0 = (rbf[:, :, None] * sh[:, None, 0:1]).reshape(len(r), -1)
        s1 = (rbf[:, :, None] * sh[:, None, 1:4]).reshape(len(r), -1)
        s2 = (rbf[:, :, None] * sh[:, None, 4:9]).reshape(len(r), -1)
        return torch.cat([s0, s1, s2], dim=-1)

    @staticmethod
    def _add_l1(x: torch.Tensor, delta: torch.Tensor,
                irreps: o3.Irreps) -> torch.Tensor:
        slices = irreps.slices()
        return torch.cat([x[..., slices[0]],
                          x[..., slices[1]] + delta,
                          x[..., slices[2]]], dim=-1)

    def _radial_basis(self, edge_vec: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.vector_norm(edge_vec, dim=-1).clamp_min(1e-8)
        diff = (r[:, None] - self.rbf_centers[None, :]) / self.rbf_width
        rbf = torch.exp(-diff * diff)
        envelope = torch.where(
            r < self.cutoff,
            0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0),
            torch.zeros_like(r),
        )
        return rbf * envelope[:, None]

    def _edge_message(self, edge_vec: torch.Tensor, receiver: torch.Tensor,
                      num_nodes: int) -> torch.Tensor:
        """Direct even polar-vector message w(r) * r_ij, summed over edges."""
        r = torch.linalg.vector_norm(edge_vec, dim=-1).clamp_min(1e-8)
        diff = (r[:, None] - self.rbf_centers[None, :]) / self.rbf_width
        rbf = torch.exp(-diff * diff)
        envelope = torch.where(
            r < self.cutoff,
            0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0),
            torch.zeros_like(r),
        )
        rbf = rbf * envelope[:, None]
        weight = self.edge_message_mlp(rbf)  # [E, hidden]
        unit = edge_vec / r[:, None]
        messages = weight[:, :, None] * unit[:, None, :]  # [E, hidden, 3]
        out = torch.zeros(num_nodes, self.hidden, 3,
                          dtype=messages.dtype, device=messages.device)
        out.index_add_(0, receiver, messages)
        return out.reshape(num_nodes, self.hidden * 3)

    @staticmethod
    def _aggregate(messages: torch.Tensor, receiver: torch.Tensor,
                   num_nodes: int) -> torch.Tensor:
        out = torch.zeros(num_nodes, messages.shape[-1],
                          dtype=messages.dtype, device=messages.device)
        out.index_add_(0, receiver, messages)
        count = torch.zeros(num_nodes, dtype=messages.dtype,
                            device=messages.device)
        count.index_add_(0, receiver, torch.ones_like(receiver, dtype=messages.dtype))
        return out / count.clamp_min(1.0)[:, None]

    def _initial(self, q_rel, p_tilde, masses, z_idx, edge_vec):
        n = q_rel.shape[0]
        onehot = torch.nn.functional.one_hot(
            z_idx, num_classes=self.num_species).to(q_rel.dtype)
        v = p_tilde / torch.sqrt(masses)[:, None]
        speed2 = (v * v).sum(-1, keepdim=True)
        q2 = (q_rel * q_rel).sum(-1, keepdim=True)
        scalars = torch.cat([
            onehot,
            torch.log(masses / self.mass_mean)[:, None],
            speed2 / (self.p_scale ** 2 / self.mass_mean),
            q2 / (self.q_scale ** 2),
        ], dim=-1)
        even0 = self.even_scalar_mlp(scalars)
        odd_scalar = (q_rel * p_tilde).sum(-1, keepdim=True)
        odd0 = torch.tanh(odd_scalar / (self.q_scale * self.p_scale)) * \
            self.odd_scalar_mlp(scalars)

        qn = q_rel / self.q_scale
        pn = p_tilde / self.p_scale
        if self.use_global_q:
            even1 = self.even_q_linear(qn)
        else:
            even1 = torch.zeros(q_rel.shape[0], self.hidden * 3,
                                dtype=qn.dtype, device=qn.device)
        odd1 = self.odd_p_linear(pn)
        even2 = self.even_q2_linear(self.even_q2_tp(qn, qn)) + \
            self.even_p2_linear(self.even_p2_tp(pn, pn))
        odd2 = self.odd_qp_linear(self.odd_qp_tp(qn, pn))

        even = torch.cat([even0, even1, even2], dim=-1)
        odd = torch.cat([odd0, odd1, odd2], dim=-1)
        return even, odd

    def forward(self, q_rel, p_tilde, masses, z_idx, edge_vec,
                edge_index, num_nodes):
        """Return Q_even, Q_odd, P_even, P_odd (mass-weighted units)."""
        edge_attr = self._edge_features(edge_vec)
        h_even, h_odd = self._initial(q_rel, p_tilde, masses, z_idx, edge_vec)
        h_even = self._rms_norm(h_even, self.irreps_h)
        h_odd = self._rms_norm(h_odd, self.irreps_h)
        sender, receiver = edge_index[0], edge_index[1]
        edge_msg = (self._edge_message(edge_vec, receiver, num_nodes)
                    if self.use_edge_message else None)
        rbf = (self._radial_basis(edge_vec)
               if self.use_receiver_gate else None)
        for block in range(len(self.msg_even)):
            # A. spatial messages with even geometric edges, same tau.
            me = self.msg_even[block](h_even[sender], edge_attr)
            if rbf is not None:
                gate_in = torch.cat([
                    _slice_irrep(h_even[sender], self.irreps_h, 0),
                    _slice_irrep(h_even[receiver], self.irreps_h, 0),
                    rbf,
                ], dim=-1)
                me = me * torch.sigmoid(
                    self.receiver_gate[block](gate_in))
            if self.use_two_endpoint:
                me2 = self.msg_even2[block](me, h_even[receiver])
                me = me + self.msg_even2_scale[block] * \
                    self.msg_even2_linear[block](me2)
            mo = self.msg_odd[block](h_odd[sender], edge_attr)
            me = self.msg_even_linear[block](self._aggregate(me, receiver, num_nodes))
            mo = self.msg_odd_linear[block](self._aggregate(mo, receiver, num_nodes))
            h_even = h_even + me
            h_odd = h_odd + mo

            # C. node-local parity-graded interaction.
            ee = self.node_ee[block](h_even, h_even)
            oo = self.node_oo[block](h_odd, h_odd)
            eo = self.node_eo[block](h_even, h_odd)
            oe = self.node_oe[block](h_odd, h_even)
            upd_e = self.node_even_linear[block](ee + oo)
            upd_o = self.node_odd_linear[block](eo + oe)
            h_even = h_even + upd_e
            h_odd = h_odd + upd_o

            # D. even scalar gate; preserves tau of both streams.
            scalars = _slice_irrep(h_even, self.irreps_h, 0)
            gates = torch.sigmoid(self.gate_mlp[block](scalars)).reshape(
                num_nodes, 3, self.hidden)
            e0 = _slice_irrep(h_even, self.irreps_h, 0) * gates[:, 0, :]
            e1 = _slice_irrep(h_even, self.irreps_h, 1) * \
                gates[:, 1, :].repeat_interleave(3, dim=1)
            e2 = _slice_irrep(h_even, self.irreps_h, 2) * \
                gates[:, 2, :].repeat_interleave(5, dim=1)
            o0 = _slice_irrep(h_odd, self.irreps_h, 0) * gates[:, 0, :]
            o1 = _slice_irrep(h_odd, self.irreps_h, 1) * \
                gates[:, 1, :].repeat_interleave(3, dim=1)
            o2 = _slice_irrep(h_odd, self.irreps_h, 2) * \
                gates[:, 2, :].repeat_interleave(5, dim=1)
            h_even = h_even + self.post_linear_even[block](
                torch.cat([e0, e1, e2], -1))
            h_odd = h_odd + self.post_linear_odd[block](
                torch.cat([o0, o1, o2], -1))
            h_even = self._rms_norm(h_even, self.irreps_h)
            h_odd = self._rms_norm(h_odd, self.irreps_h)
            if edge_msg is not None:
                h_even = self._add_l1(
                    h_even, self.edge_message_scale[block] * edge_msg,
                    self.irreps_h)

        if self.use_tp_readout:
            q_even = self.q_even_head(h_even, h_even)
            q_odd = self.q_odd_head(h_even, h_odd)
            p_even = self.p_even_head(h_even, h_even)
            p_odd = self.p_odd_head(h_even, h_odd)
        else:
            q_even = self.q_even_head(_slice_irrep(h_even, self.irreps_h, 1))
            q_odd = self.q_odd_head(_slice_irrep(h_odd, self.irreps_h, 1))
            p_even = self.p_even_head(_slice_irrep(h_even, self.irreps_h, 1))
            p_odd = self.p_odd_head(_slice_irrep(h_odd, self.irreps_h, 1))
        return dict(Q_even=q_even, Q_odd=q_odd,
                    P_even=p_even, P_odd=p_odd,
                    h_even=h_even, h_odd=h_odd)


class Ungraded(nn.Module):
    """Capacity-matched control without momentum-parity separation.

    Same spatial message TP, node-local TP and even-scalar gate primitives as
    ParityFlow, but one undifferentiated hidden stream and no parity selection.
    """

    def __init__(
        self,
        num_species: int,
        hidden: int = 56,
        bottleneck: int = 16,
        blocks: int = 4,
        edge_rbf: int = 4,
        cutoff: float = 6.0,
        q_scale: float = 1.0,
        p_scale: float = 1.0,
        mass_mean: float = 1.0,
    ):
        super().__init__()
        self.num_species = int(num_species)
        self.hidden = int(hidden)
        self.cutoff = float(cutoff)
        self.edge_rbf = int(edge_rbf)
        self.q_scale = float(q_scale)
        self.p_scale = float(p_scale)
        self.mass_mean = float(mass_mean)
        self.irreps_h = o3.Irreps(
            f"{hidden}x0e+{hidden}x1o+{hidden}x2e")
        self.irreps_b = o3.Irreps(
            f"{bottleneck}x0e+{bottleneck}x1o+{bottleneck}x2e")
        self.irreps_edge = o3.Irreps(
            f"{edge_rbf}x0e+{edge_rbf}x1o+{edge_rbf}x2e")
        self.sh = o3.SphericalHarmonics(
            "0e+1o+2e", normalize=True, normalization="component")
        self.register_buffer("rbf_centers", torch.linspace(1.0, cutoff, edge_rbf))
        self.register_buffer("rbf_width",
                             torch.tensor(max(cutoff / max(edge_rbf - 1, 1), 1e-3)))
        self.scalar_mlp = nn.Sequential(
            nn.Linear(num_species + 3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        self.pos_linear = o3.Linear("1o", f"{hidden}x1o")
        self.vel_linear = o3.Linear("1o", f"{hidden}x1o")
        self.q2_tp = o3.FullyConnectedTensorProduct(
            "1o", "1o", "2e", internal_weights=True, shared_weights=True)
        self.p2_tp = o3.FullyConnectedTensorProduct(
            "1o", "1o", "2e", internal_weights=True, shared_weights=True)
        self.qp_tp = o3.FullyConnectedTensorProduct(
            "1o", "1o", "2e", internal_weights=True, shared_weights=True)
        self.q2_linear = o3.Linear("2e", f"{hidden}x2e")
        self.p2_linear = o3.Linear("2e", f"{hidden}x2e")
        self.qp_linear = o3.Linear("2e", f"{hidden}x2e")
        self.msg = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_edge, self.irreps_h,
                internal_weights=True, shared_weights=True)
            for _ in range(blocks)
        ])
        self.msg_linear = nn.ModuleList([
            o3.Linear(self.irreps_h, self.irreps_h)
            for _ in range(blocks)
        ])
        self.node = nn.ModuleList([
            o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_h, self.irreps_b,
                internal_weights=True, shared_weights=True)
            for _ in range(blocks)
        ])
        self.node_linear = nn.ModuleList([
            o3.Linear(self.irreps_b, self.irreps_h)
            for _ in range(blocks)
        ])
        self.gate_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, 3 * hidden))
            for _ in range(blocks)
        ])
        self.post_linear = nn.ModuleList([
            o3.Linear(self.irreps_h, self.irreps_h)
            for _ in range(blocks)
        ])
        for gate in self.gate_mlp:
            gate[-1].bias.data.fill_(2.0)
        self.q_even_head = o3.Linear(f"{hidden}x1o", "1o")
        self.q_odd_head = o3.Linear(f"{hidden}x1o", "1o")
        self.p_even_head = o3.Linear(f"{hidden}x1o", "1o")
        self.p_odd_head = o3.Linear(f"{hidden}x1o", "1o")

    def _edge_features(self, edge_vec):
        # Same geometric edge construction as ParityFlow.
        return ParityFlow._edge_features(self, edge_vec)

    def _aggregate(self, messages, receiver, num_nodes):
        return ParityFlow._aggregate(messages, receiver, num_nodes)

    def _initial(self, q_rel, p_tilde, masses, z_idx):
        onehot = torch.nn.functional.one_hot(
            z_idx, num_classes=self.num_species).to(q_rel.dtype)
        v = p_tilde / torch.sqrt(masses)[:, None]
        scalars = torch.cat([
            onehot,
            torch.log(masses / self.mass_mean)[:, None],
            (v * v).sum(-1, keepdim=True) / (self.p_scale ** 2 / self.mass_mean),
            (q_rel * q_rel).sum(-1, keepdim=True) / (self.q_scale ** 2),
        ], dim=-1)
        s = self.scalar_mlp(scalars)
        qn = q_rel / self.q_scale
        pn = p_tilde / self.p_scale
        v1 = self.pos_linear(qn) + self.vel_linear(pn)
        v2 = self.q2_linear(self.q2_tp(qn, qn)) + \
            self.p2_linear(self.p2_tp(pn, pn)) + \
            self.qp_linear(self.qp_tp(qn, pn))
        return torch.cat([s, v1, v2], dim=-1)

    def forward(self, q_rel, p_tilde, masses, z_idx, edge_vec,
                edge_index, num_nodes):
        edge_attr = self._edge_features(edge_vec)
        h = self._initial(q_rel, p_tilde, masses, z_idx)
        h = self._rms_norm(h, self.irreps_h)
        sender, receiver = edge_index[0], edge_index[1]
        for block in range(len(self.msg)):
            m = self.msg[block](h[sender], edge_attr)
            m = self.msg_linear[block](
                self._aggregate(m, receiver, num_nodes))
            h = h + m
            upd = self.node[block](h, h)
            h = h + self.node_linear[block](upd)
            scalars = _slice_irrep(h, self.irreps_h, 0)
            gates = torch.sigmoid(self.gate_mlp[block](scalars)).reshape(
                num_nodes, 3, self.hidden)
            g0 = _slice_irrep(h, self.irreps_h, 0) * gates[:, 0, :]
            g1 = _slice_irrep(h, self.irreps_h, 1) * \
                gates[:, 1, :].repeat_interleave(3, dim=1)
            g2 = _slice_irrep(h, self.irreps_h, 2) * \
                gates[:, 2, :].repeat_interleave(5, dim=1)
            h = h + self.post_linear[block](torch.cat([g0, g1, g2], -1))
            h = self._rms_norm(h, self.irreps_h)
        h1 = _slice_irrep(h, self.irreps_h, 1)
        return dict(Q_even=self.q_even_head(h1), Q_odd=self.q_odd_head(h1),
                    P_even=self.p_even_head(h1), P_odd=self.p_odd_head(h1),
                    h=h)
