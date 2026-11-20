"""Unit tests for the physical-proxy selection metric."""
import math

import torch

from physics.units import KINETIC_TO_EV
from selection_metrics import molecular_kinetic_partition, partition_relative_error


def _partition(positions, velocities, masses, molecule_ids, cell):
    return molecular_kinetic_partition(
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(velocities, dtype=torch.float32),
        torch.tensor(masses, dtype=torch.float32),
        torch.tensor(molecule_ids, dtype=torch.long),
        torch.tensor(cell, dtype=torch.float32),
    )


def test_pure_translation_has_no_rotation_or_internal():
    cell = torch.eye(3) * 10
    result = _partition(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [16.0, 1.0], [1, 1], cell)
    total = 0.5 * 17.0 * KINETIC_TO_EV
    assert math.isclose(float(result['total']), total, rel_tol=1e-6)
    assert math.isclose(float(result['translation']), total, rel_tol=1e-6)
    assert abs(float(result['rotation'])) < 1e-6
    assert abs(float(result['internal'])) < 1e-5


def test_rigid_rotation_is_rotation():
    cell = torch.eye(3) * 10
    v = 0.1
    result = _partition(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [[0.0, -v, 0.0], [0.0, v, 0.0]],
        [1.0, 1.0], [1, 1], cell)
    total = v * v * KINETIC_TO_EV
    assert math.isclose(float(result['total']), total, rel_tol=1e-6)
    assert abs(float(result['translation'])) < 1e-6
    assert math.isclose(float(result['rotation']), total, rel_tol=1e-5)
    assert abs(float(result['internal'])) < 1e-5


def test_internal_stretch_is_internal():
    cell = torch.eye(3) * 10
    v = 0.1
    result = _partition(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [[v, 0.0, 0.0], [-v, 0.0, 0.0]],
        [1.0, 1.0], [1, 1], cell)
    total = v * v * KINETIC_TO_EV
    assert math.isclose(float(result['total']), total, rel_tol=1e-6)
    assert abs(float(result['translation'])) < 1e-6
    assert abs(float(result['rotation'])) < 1e-6
    assert math.isclose(float(result['internal']), total, rel_tol=1e-5)


def test_periodic_wrap_preserves_partition():
    cell = torch.eye(3) * 5
    v = 0.1
    straight = _partition(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [[0.0, -v, 0.0], [0.0, v, 0.0]],
        [1.0, 1.0], [1, 1], cell)
    wrapped = _partition(
        [[4.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [[0.0, -v, 0.0], [0.0, v, 0.0]],
        [1.0, 1.0], [1, 1], cell)
    for key in ('total', 'translation', 'rotation', 'internal'):
        assert math.isclose(float(straight[key]), float(wrapped[key]), rel_tol=1e-5,
                            abs_tol=1e-6)


def test_partition_relative_error_is_zero_for_identical():
    value = {k: torch.tensor(1.0) for k in ('translation', 'rotation', 'internal')}
    assert float(partition_relative_error(value, value)) == 0.0
