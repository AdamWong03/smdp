"""Trainer/loader dispatch contracts for the phase-space operator."""
import json

import pytest
import torch
from torch_geometric.data import Batch, Data
from trajcast.data.atomic_graph import AtomicGraph

from phase_space_transition import PhaseSpaceTransition, train_normalization
from relational_transition import RelationalTransition, endpoint_objective
from relational_inference import load_relational
from train_relational import build_model, gradient_groups, is_phase_space, train_statistics


def make_graph(n=5, horizon=5., cutoff=4.5):
    dtype = torch.float64
    return AtomicGraph.from_atoms_dict(dict(
        positions=torch.rand(n, 3, dtype=dtype)*4,
        velocities=torch.randn(n, 3, dtype=dtype)*.01,
        atomic_numbers=torch.tensor([1, 8]*n)[:n, None],
        atomic_masses=torch.tensor([1., 16.]*n, dtype=dtype)[:n, None],
        cell=torch.tensor([[8., 0, 0], [1., 8., 0], [0, 1., 8.]], dtype=dtype),
        pbc=torch.ones(3, dtype=torch.bool), timestep=torch.tensor([horizon], dtype=dtype)),
        r_cut=cutoff, atom_type_mapper={1: 0, 8: 1})


def fake_targets():
    graphs = []
    for _ in range(2):
        graphs.append(Data(
            pos=torch.rand(4, 3, dtype=torch.float64)*3,
            edge_index=torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 2]]),
            shifts=torch.zeros(5, 3, dtype=torch.float64),
            displacements=torch.randn(4, 3, dtype=torch.float64)*.05,
            update_velocities=torch.randn(4, 3, dtype=torch.float64)*.01))
    return graphs


def tiny_phase_cfg():
    return dict(architecture='phase_space_operator_v0',
                model=dict(node_channels=12, message_channels=4, edge_channels=8,
                           geometry_channels=2, radial_channels=6, speed_channels=4,
                           expansion=2, cutoff_angstrom=4.5, horizon_fs=5.))


def test_train_normalization_is_train_only_and_frozen():
    graphs = fake_targets()
    stats = train_normalization(graphs, 4.5)
    q = torch.cat([g.displacements for g in graphs])
    v = torch.cat([g.update_velocities for g in graphs])
    assert stats['rms_targets'][0] == pytest.approx(q.square().mean().sqrt().item())
    assert stats['rms_targets'][1] == pytest.approx(v.square().mean().sqrt().item())
    torch.testing.assert_close(stats['samples'], 2)
    assert stats['atoms'] == 8 and stats['edges'] == 10
    assert stats['aggregation_scale'] > 0
    assert stats['source'].startswith('complete TRAIN')
    with pytest.raises(ValueError):
        train_normalization([], 4.5)


def test_train_statistics_dispatch_and_model_build():
    torch.set_default_dtype(torch.float64)
    try:
        cfg = tiny_phase_cfg()
        assert is_phase_space(cfg)
        graphs = fake_targets()
        sq, sv, alpha, stats = train_statistics(cfg, graphs)
        assert stats['aggregation_scale'] == alpha
        model = build_model(cfg, (sq, sv), alpha)
        assert isinstance(model, PhaseSpaceTransition)
        assert model.rms_targets[0].item() == pytest.approx(sq)
        assert model.aggregation_scale.item() == pytest.approx(alpha)
        with pytest.raises(ValueError, match='aggregation'):
            build_model(cfg, (sq, sv), None)
        assert not is_phase_space({})
        small = dict(model=dict(scalar_channels=8, vector_channels=2, edge_channels=8,
                                radial_channels=4, cutoff_angstrom=4.5, horizon_fs=5.,
                                proposal_layers=1, refinement_layers=1,
                                velocity_scale=.01, block_type='geometry_gated'))
        assert isinstance(build_model(small, (0.07, 0.004)), RelationalTransition)
    finally:
        torch.set_default_dtype(torch.float32)


def test_one_trainer_style_step_and_gradient_groups():
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(7)
        cfg = tiny_phase_cfg()
        graphs = [make_graph(5), make_graph(7)]
        batch = Batch.from_data_list(graphs)
        model = build_model(cfg, (0.06, 0.013), 2.0)
        groups = gradient_groups(model)
        assert {'input_encoder', 'proposal_head', 'correction_head',
                'proposal_block_0', 'proposal_block_1',
                'refinement_block_0', 'refinement_block_1'} == set(groups)
        output = model(batch)
        loss, _ = endpoint_objective(model, output, .1*batch.velocities,
                                     .9*batch.velocities, proposal_weight=.02)
        loss.backward()
        assert loss.isfinite()
        assert all(p.grad is not None and p.grad.isfinite().all()
                   for p in model.parameters())
    finally:
        torch.set_default_dtype(torch.float32)


def test_diagnostic_hooks_are_removed_and_report_four_blocks():
    torch.set_default_dtype(torch.float64)
    try:
        model = build_model(tiny_phase_cfg(), (0.06, 0.013), 2.0)
        batch = Batch.from_data_list([make_graph(5)])
        model.set_diagnostics(True)
        model(batch)
        model.set_diagnostics(False)
        records = model.diagnostics()
        assert len(records) == 4
        assert all(row['node_rms'] > 0 and row['edge_rms'] > 0 for row in records)
        assert getattr(model, '_diagnostic_hooks', None) is None
    finally:
        torch.set_default_dtype(torch.float32)


def test_checkpoint_loader_rebuilds_phase_space_model(tmp_path):
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(11)
        cfg = tiny_phase_cfg()
        model = build_model(cfg, (0.06, 0.013), 2.0)
        checkpoint = tmp_path/'best.pt'
        config = dict(cfg, normalization=dict(rms_targets=[0.06, 0.013],
                                               aggregation_scale=2.0))
        torch.save(dict(model=model.state_dict(), config=config), checkpoint)
        loaded, saved = load_relational(checkpoint, device='cpu')
        assert isinstance(loaded, PhaseSpaceTransition)
        assert saved['config']['architecture'] == 'phase_space_operator_v0'
        batch = Batch.from_data_list([make_graph(5)])
        with torch.no_grad():
            expected, actual = model(batch), loaded(batch)
        for key in ('displacements', 'velocities', 'proposal_displacements',
                    'proposal_velocities'):
            torch.testing.assert_close(expected[key], actual[key])
    finally:
        torch.set_default_dtype(torch.float32)
