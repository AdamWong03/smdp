"""Reusable objectives and one-step/short-rollout APIs for SMDP training.

All targets must come from contiguous TRAIN NVE trajectories only. No file format,
MD engine, splitting scheme or physical units are silently inferred here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from smdp_neural import NeuralSMDP


@dataclass
class SegmentStats:
    displacement_residual_rms: float  # RMS of q(t+h)-q(t)-h*v(t), in Angstrom
    velocity_delta_rms: float         # RMS of v(t+h)-v(t), in Angstrom/fs

    def __post_init__(self):
        if self.displacement_residual_rms <= 0 or self.velocity_delta_rms <= 0:
            raise ValueError('target RMS values must be positive and calculated from training split only')


def supervised_segment_loss(
    output: dict[str, torch.Tensor],
    q0: torch.Tensor,
    v0: torch.Tensor,
    times_fs: torch.Tensor,
    reference_q: torch.Tensor,
    reference_v: torch.Tensor,
    stats: SegmentStats,
    endpoint_multiplier: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Joint endpoint and intermediate-frame supervision.

    Outputs [T,N,3], targets in the same continuous unwrapped coordinate chart.
    Displacements are compared relative to reference free drift, but the free
    drift cancels algebraically in the prediction-target residual.
    """
    pred_q, pred_v = output['positions'], output['velocities']
    if pred_q.shape != reference_q.shape or pred_v.shape != reference_v.shape:
        raise ValueError('model outputs and reference segment must have the same [T,N,3] shape')
    t = times_fs.numel()
    weight = torch.ones(t, dtype=pred_q.dtype, device=pred_q.device)
    weight[-1] = endpoint_multiplier
    weight = weight / weight.sum()
    q_squared = ((pred_q-reference_q)/stats.displacement_residual_rms).square().mean(dim=(-1,-2))
    v_squared = ((pred_v-reference_v)/stats.velocity_delta_rms).square().mean(dim=(-1,-2))
    lq = (q_squared * weight).sum()
    lv = (v_squared * weight).sum()
    return {'loss': lq+lv, 'position': lq, 'velocity': lv}


def predict_one_step(
    model: NeuralSMDP,
    q: torch.Tensor, v: torch.Tensor, z: torch.Tensor, masses: torch.Tensor,
    edge_builder: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    h_fs: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    edges, shifts = edge_builder(q)
    outputs = model(q, v, z, masses, edges, shifts, [h_fs])
    return outputs['positions'][0], outputs['velocities'][0]


def autoregressive_rollout(
    model: NeuralSMDP,
    q0: torch.Tensor, v0: torch.Tensor, z: torch.Tensor, masses: torch.Tensor,
    edge_builder: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    h_fs: float,
    steps: int,
    detach_between_steps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Short differentiable training rollouts OR evaluation rollouts.

    For long NVT runs, stream frames to disk; do not collect them in a tensor.
    The thermostat and production PBC neighbor list belong to the MD adapter.
    """
    q, v = q0, v0
    qs, vs = [], []
    for _ in range(steps):
        q, v = predict_one_step(model, q, v, z, masses, edge_builder, h_fs)
        qs.append(q); vs.append(v)
        if detach_between_steps:
            q, v = q.detach(), v.detach()
    return torch.stack(qs), torch.stack(vs)
