"""Supervised-anchor-preserving semigroup consistency for autonomous NVE SMDP.

One 16 fs transition must agree with two consecutive 8 fs transitions.
The 8+8 route rebuilds the periodic graph from the predicted intermediate
configuration. Both sides receive gradients; no BKS teacher or force calls.
"""
from __future__ import annotations
import torch
from periodic import predict_batch


def composition_consistency(model, batch, segment_out, times_fs, stats, builder, samples=2):
    """Return dimensionless loss and separately loggable position/velocity terms.

    segment_out is a batched segment prediction with [T, B*N, 3] outputs.
    It MUST include actual 8 fs and 16 fs queries. Sampling the first
    'samples' elements of a shuffled training batch bounds extra graph work.
    On validation the DataLoader order is fixed, so evaluation is repeatable.
    """
    if samples < 1:
        raise ValueError("composition samples must be positive")
    time_values = times_fs.detach().cpu().tolist()
    if 8.0 not in time_values or 16.0 not in time_values:
        raise ValueError("composition requires both 8 and 16 fs queries")
    i8, i16 = time_values.index(8.0), time_values.index(16.0)
    q0, v0 = batch["q"][:, 0], batch["v"][:, 0]
    count, n, _ = q0.shape
    take = min(samples, count)
    indices = torch.arange(take, device=q0.device)
    q8 = segment_out["positions"][i8].reshape(count, n, 3)[indices]
    v8 = segment_out["velocities"][i8].reshape(count, n, 3)[indices]
    q16 = segment_out["positions"][i16].reshape(count, n, 3)[indices]
    v16 = segment_out["velocities"][i16].reshape(count, n, 3)[indices]
    composed = predict_batch(model, q8, v8, batch["z"][indices],
                             batch["mass"][indices], batch["cell"][indices],
                             [8.0], builder)
    q88 = composed["positions"][0].reshape(take, n, 3)
    v88 = composed["velocities"][0].reshape(take, n, 3)
    q_loss = ((q88 - q16) / stats.displacement_residual_rms).square().mean()
    v_loss = ((v88 - v16) / stats.velocity_delta_rms).square().mean()
    return {"composition_q": q_loss, "composition_v": v_loss,
            "composition": q_loss + v_loss}
