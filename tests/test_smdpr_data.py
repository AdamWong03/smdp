"""Regression coverage for actual wrapped-cache/unwrapped-label conventions."""
import json

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

from smdpr.data import InteriorLabels
from smdpr.finetune import two_step
from smdpr.model import SMDPR
from periodic import PeriodicGraphBuilder
from test_relational_transition import graph


def make_graph():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return graph().apply(lambda t: t.float() if t.is_floating_point() else t)
    finally:
        torch.set_default_dtype(old)


def test_real_labels_carry_one_image_offset(tmp_path):
    g = make_graph()
    g.displacements = .03*torch.ones_like(g.pos)
    g.update_velocities = g.velocities+.001
    unwrapped = g.pos.numpy()-g.cell.numpy()[0]
    target = unwrapped+g.displacements.numpy()
    with h5py.File(tmp_path/'train.h5', 'w') as f:
        f.attrs.update(split='train', source_paths=json.dumps(['train/source.h5']),
            source_transition='train/transitions.h5', position_unit='Angstrom',
            velocity_unit='Angstrom/fs')
        for key, value in dict(times_fs=[30.], q0=unwrapped[None], v0=g.velocities[None],
            q_target=target[None, None], v_target=g.update_velocities[None, None],
            z=g.atomic_numbers.flatten(), mass=g.atomic_masses.flatten(), cell=g.cell[None]).items():
            f[key] = value
    labels = InteriorLabels(tmp_path, 'train', [g])
    np.testing.assert_allclose(labels.q[0, 0], g.pos.numpy()+g.displacements.numpy(), atol=3e-6)


def test_two_step_graph_rebuild_keeps_prediction_gradient():
    g = make_graph()
    g.displacements = g.velocities*30
    g.update_velocities = g.velocities*.9
    g.two_step_positions = g.pos+g.displacements*2
    g.two_step_velocities = g.velocities*.8
    model = SMDPR(dict(block_type='geometry_gated')).float()
    batch = Batch.from_data_list([g, g.clone()])
    loss, output, _, _ = two_step(model, batch, PeriodicGraphBuilder(4.5), .02)
    output['positions'].retain_grad()
    output['velocities'].retain_grad()
    loss.backward()
    assert output['positions'].grad is not None
    assert output['velocities'].grad is not None
    assert torch.isfinite(output['positions'].grad).all()
    assert output['positions'].grad.abs().sum()>0
    for name, p in model.named_parameters():
        if not name.startswith('query_head.'):
            assert p.grad is not None and p.grad.isfinite().all(), name
