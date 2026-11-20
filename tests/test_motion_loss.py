"""Tests for the motion-balanced objective decomposition."""
import math
from types import SimpleNamespace

import torch

from motion_loss import decompose_modes, mode_mse, train_mode_rms


def _inputs(positions, velocities, masses, molecule_ids, cell=None):
    return (torch.tensor(positions, dtype=torch.float64),
            torch.tensor(velocities, dtype=torch.float64),
            torch.tensor(masses, dtype=torch.float64),
            torch.tensor(molecule_ids, dtype=torch.long),
            torch.tensor(cell if cell is not None else [[10., 0, 0], [0, 10., 0], [0, 0, 10.]],
                         dtype=torch.float64))


def test_pure_translation_has_only_translation():
    q, v, m, ids, cell = _inputs(
        [[0., 0., 0.], [1., 0., 0.]],
        [[1., 0., 0.], [1., 0., 0.]],
        [16., 1.], [1, 1])
    part = decompose_modes(q, v, m, ids, cell)
    assert torch.allclose(part['translation'], v)
    assert part['rotation'].abs().max() < 1e-8
    assert part['internal'].abs().max() < 1e-8


def test_rigid_rotation_has_only_rotation():
    v = 0.1
    q, vv, m, ids, cell = _inputs(
        [[-0.5, 0., 0.], [0.5, 0., 0.]],
        [[0., -v, 0.], [0., v, 0.]],
        [1., 1.], [1, 1])
    part = decompose_modes(q, vv, m, ids, cell)
    assert part['translation'].abs().max() < 1e-8
    assert torch.allclose(part['rotation'], vv, atol=1e-6)
    assert part['internal'].abs().max() < 1e-6


def test_internal_stretch_has_only_internal():
    v = 0.1
    q, vv, m, ids, cell = _inputs(
        [[-0.5, 0., 0.], [0.5, 0., 0.]],
        [[v, 0., 0.], [-v, 0., 0.]],
        [1., 1.], [1, 1])
    part = decompose_modes(q, vv, m, ids, cell)
    assert part['translation'].abs().max() < 1e-8
    assert part['rotation'].abs().max() < 1e-8
    assert torch.allclose(part['internal'], vv, atol=1e-6)


def test_decomposition_is_differentiable():
    q, vv, m, ids, cell = _inputs(
        [[-0.5, 0., 0.], [0.5, 0., 0.]],
        [[0.1, 0., 0.], [-0.1, 0., 0.]],
        [1., 1.], [1, 1])
    vv.requires_grad_(True)
    part = decompose_modes(q, vv, m, ids, cell)
    loss = part['internal'].square().sum() + part['rotation'].square().sum()
    loss.backward()
    assert vv.grad is not None
    assert torch.isfinite(vv.grad).all()
    assert vv.grad.abs().sum() > 0


def test_train_mode_rms_and_mode_mse():
    q, vv, m, ids, cell = _inputs(
        [[-0.5, 0., 0.], [0.5, 0., 0.]],
        [[0.1, 0., 0.], [-0.1, 0., 0.]],
        [1., 1.], [1, 1])
    graph = SimpleNamespace(pos=q, update_velocities=vv, atomic_masses=m,
                            molecule_ids=ids, cell=cell)
    rms = train_mode_rms([graph])
    assert all(value > 0 for value in rms.values())
    part = decompose_modes(q, vv, m, ids, cell)
    zero, pieces = mode_mse(part, part, rms)
    assert float(zero) == 0.0
    shifted = {k: v + 0.01 for k, v in part.items() if k != 'total'}
    positive, _ = mode_mse(shifted, part, rms)
    assert float(positive) > 0
