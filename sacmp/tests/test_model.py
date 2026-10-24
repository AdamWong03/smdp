"""Structural checks for the full canonical map (small CPU system)."""
import math
import pytest
import torch

from sacmp.model import SACMP, SACMPConfig


def make_model(dtype=torch.float64):
    torch.manual_seed(8)
    mass = torch.tensor([16., 28., 16., 28.])
    z = torch.tensor([8, 14, 8, 14])
    xref = torch.tensor([
        [0., 0., 0.], [1.6, 0., 0.], [.3, 1.7, 0.],
        [1.8, 1.8, .2]])
    src, dst = torch.where(~torch.eye(4, dtype=torch.bool))
    edges = torch.stack([src, dst])
    shifts = torch.zeros(len(src), 3)
    rank = 5
    # Remove translations before orthonormalizing the seed.
    M = torch.randn(4, 3, rank, dtype=torch.float64)
    root = mass.double().sqrt()
    cm = (M*root[:, None, None]).sum(0)/mass.sum().sqrt()
    M -= root[:, None, None]/mass.sum().sqrt()*cm[None]
    W = torch.linalg.qr(M.reshape(12, rank), mode="reduced")[0]
    omega = torch.linspace(.03, .15, rank)
    cfg = SACMPConfig(rank=rank, hidden=24, radial_basis=6,
                      cutoff_a=4.5, max_substep_fs=2.0)
    model = SACMP(cfg, xref, mass, z, edges, shifts, W, omega).to(dtype)
    with torch.no_grad():
        model.potential.mixed_net[-1].weight.normal_(std=.03)
        model.potential.mixed_net[-1].bias.normal_(std=.01)
    x = xref.to(dtype)+torch.randn(4, 3, dtype=dtype)*.03
    v = torch.randn(4, 3, dtype=dtype)*.007
    return model, x, v


def test_zero_step_and_translation():
    model, x, v = make_model()
    out = model(x, v, [0., 2.])
    torch.testing.assert_close(out["positions"][0], x, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(out["velocities"][0], v, atol=1e-12, rtol=1e-12)
    shift = torch.tensor([.6, -.2, .3], dtype=x.dtype)
    # The reference is fixed: a common translation is in the COM channel.
    other = model(x+shift, v, [2.])
    torch.testing.assert_close(other["positions"][0],
                               out["positions"][1]+shift,
                               atol=2e-11, rtol=2e-11)


def test_com_and_time_reverse():
    model, x, v = make_model()
    out = model(x, v, [8.])
    x1, v1 = out["positions"][0], out["velocities"][0]
    mass = model.mass[:, None]
    cm0 = (v*mass).sum(0)/mass.sum()
    cm1 = (v1*mass).sum(0)/mass.sum()
    torch.testing.assert_close(cm1, cm0, atol=5e-12, rtol=5e-12)
    reverse = model(x1, v1, [-8.])
    W = model.modes()
    initial = model._encode(x, v, W)
    internal_forward = model._propagate(initial, 8., W, model.geometry())
    internal_back = model._propagate(internal_forward, -8., W, model.geometry())
    reencoded = model._encode(x1, v1, W)
    print("SACMP DIAGNOSTIC",
          "internal_reverse_max", max(float((a-b).abs().max()) for a,b in zip(internal_back, initial[:4])),
          "reencode_max", max(float((a-b).abs().max()) for a,b in zip(reencoded[:4], internal_forward)),
          "slow_orthogonality", float((internal_forward[2]@W).abs().max()),
          "slow_momentum_orthogonality", float((internal_forward[3]@W).abs().max()))
    torch.testing.assert_close(reverse["positions"][0], x,
                               atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(reverse["velocities"][0], v,
                               atol=3e-10, rtol=3e-10)


def test_kinematic_tangent():
    model, x, v = make_model()
    eps = 1e-5
    o = model(x, v, [-eps, eps])
    derivative = (o["positions"][1]-o["positions"][0])/(2*eps)
    torch.testing.assert_close(derivative, v, atol=2e-8, rtol=2e-5)


def test_canonical_symplectic_jacobian():
    model, x, v = make_model()
    model.train()
    rootm = model.mass.sqrt().unsqueeze(-1)
    n = x.numel()
    state = torch.cat([(x*rootm).flatten(), (v*rootm).flatten()])
    def map_fn(y):
        q = (y[:n].reshape_as(x)/rootm)
        vel = (y[n:].reshape_as(v)/rootm)
        out = model(q, vel, [2.])
        return torch.cat([(out["positions"][0]*rootm).flatten(),
                          (out["velocities"][0]*rootm).flatten()])
    jac = torch.autograd.functional.jacobian(map_fn, state)
    ident = torch.eye(n, dtype=state.dtype)
    zero = torch.zeros_like(ident)
    J = torch.cat([torch.cat([zero, ident], dim=1),
                   torch.cat([-ident, zero], dim=1)], dim=0)
    error = jac.T@J@jac-J
    assert float(error.abs().max()) < 2e-8


def test_gradients_reach_all_operators():
    model, x, v = make_model()
    model.train()
    out = model(x, v, [4.])
    loss = (out["positions"]**2).sum()+10*(out["velocities"]**2).sum()
    loss.backward()
    for name, obj in [("basis", model.basis_raw),
                      ("phase", model.log_omega),
                      ("fast anharmonic potential", model.potential.fast_net[-1].weight),
                      ("slow potential", model.potential.slow_net[-1].weight),
                      ("coupling", model.potential.mixed_net[-1].weight)]:
        assert obj.grad is not None, name
        assert torch.isfinite(obj.grad).all(), name
        assert obj.grad.abs().max() > 0, name


def test_no_grad_rollout_requires_only_local_force_autograd():
    model, x, v = make_model()
    model.eval()
    with torch.inference_mode():
        # inference_mode is NOT compatible with autograd-derived forces.
        pass
    with torch.no_grad():
        out = model(x, v, [1., 16.])
    assert torch.isfinite(out["positions"]).all()
    assert torch.isfinite(out["velocities"]).all()
