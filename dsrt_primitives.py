"""Small invariant/equivariant neural primitives used by the DSRT core."""

import math

import torch
from torch import nn


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class RBF(nn.Module):
    """Gaussian radial basis with a cosine cutoff envelope."""

    def __init__(self, n: int, cutoff: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n))
        self.gamma = (n / cutoff) ** 2
        self.cutoff = cutoff

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        values = torch.exp(
            -self.gamma * (distance[:, None] - self.centers) ** 2
        )
        envelope = (
            0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        ).clamp(min=0.0)
        envelope = torch.where(
            distance < self.cutoff, envelope, torch.zeros_like(envelope)
        )
        return values * envelope[:, None]
