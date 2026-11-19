"""Required algebra/gradient contracts for the design prototype; no fitting."""
import copy

import pytest
import torch
from e3nn import o3
from torch_geometric.data import Batch
from trajcast.data.atomic_graph import AtomicGraph

from phase_space_transition import (PhaseSpaceTransition, PhaseSpaceConfig,
    NodePhaseCoupling, PhaseSpaceBlock, PhaseRelations, irreps, motion_harmonics)
from relational_transition import endpoint_loss_components


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


@pytest.fixture
def setup():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(41)
    cfg = dict(node_channels=12, message_channels=4, edge_channels=8,
               geometry_channels=2, radial_channels=6, speed_channels=4,
               expansion=2, horizon_fs=5.)
    model = PhaseSpaceTransition(cfg, rms_targets=(.06, .013), aggregation_scale=2.)
    graphs = Batch.from_data_list([make_graph(5), make_graph(7)])
    model.validate_graph(graphs)
    yield model, graphs
    torch.set_default_dtype(previous)


KEYS = ["displacements", "velocities", "proposal_displacements", "proposal_velocities"]


def compare(a, b, atol=1e-9, rtol=1e-8):
    for key in KEYS:
        torch.testing.assert_close(a[key], b[key], atol=atol, rtol=rtol, msg=key)


@pytest.mark.parametrize("determinant", [1, -1])
def test_rotation_reflection_translation_boost(setup, determinant):
    model, graph = setup
    original = model(graph)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= determinant*torch.linalg.det(rotation)
    other = graph.clone()
    other.pos = graph.pos@rotation.T+3
    other.velocities = graph.velocities@rotation.T
    other.shifts = graph.shifts@rotation.T
    other.cell = graph.cell@rotation.T
    compare(model(other), {key: original[key]@rotation.T for key in KEYS})
    boost = torch.tensor([.004, -.003, .002])
    other = graph.clone()
    other.velocities += boost
    prediction = model(other)
    for key in KEYS:
        factor = 5 if "displacements" in key else 1
        torch.testing.assert_close(prediction[key], original[key]+factor*boost,
                                   atol=1e-10, rtol=1e-8)


def test_batch_permutation_com_and_state_loading(setup):
    model, graph = setup
    out = model(graph)
    singles = [model(item) for item in graph.to_data_list()]
    compare(out, {key: torch.cat([item[key] for item in singles]) for key in KEYS})
    perm = torch.randperm(len(graph.pos))
    inverse = torch.argsort(perm)
    other = graph.clone()
    for key in ["pos", "velocities", "atomic_numbers", "atomic_masses", "batch"]:
        other[key] = graph[key][perm]
    other.edge_index = inverse[graph.edge_index]
    compare(model(other), {key: out[key][perm] for key in KEYS})
    for k in range(2):
        mask = graph.batch == k
        masses = graph.atomic_masses[mask]
        for key in KEYS:
            factor = 5 if "displacements" in key else 1
            torch.testing.assert_close((out[key][mask]*masses).sum(0),
                factor*(graph.velocities[mask]*masses).sum(0), atol=1e-10, rtol=1e-9)
    restored = PhaseSpaceTransition(model.config, rms_targets=(.1, .1), aggregation_scale=1.)
    restored.load_state_dict(model.state_dict(), strict=True)
    compare(restored(graph), out)


def test_periodic_reimage_and_online_rebuild(setup):
    model, batch = setup
    graph = batch.to_data_list()[0]
    position = graph.pos.clone()
    position[::2] += graph.cell[0]
    position[1::2] -= 2*graph.cell[1]
    other = AtomicGraph.from_atoms_dict(dict(positions=position, velocities=graph.velocities,
        cell=graph.cell, pbc=graph.pbc, atomic_numbers=graph.atomic_numbers,
        atomic_masses=graph.atomic_masses, timestep=graph.timestep),
        r_cut=model.cfg.cutoff_angstrom, atom_type_mapper={1: 0, 8: 1})
    compare(model(other), model(graph))


def test_final_gradient_reaches_candidate_geometry_and_all_blocks(setup):
    model, graph = setup
    captured = []
    def capture(_module, inputs, _out):
        if inputs[2].requires_grad:
            for value in inputs[2:]:
                value.retain_grad()
                captured.append(value)
    hook = model.relations.register_forward_hook(capture)
    prediction = model(graph)
    hook.remove()
    pieces = endpoint_loss_components(model, prediction, .1*graph.velocities, .9*graph.velocities)
    pieces['final'].backward()  # No proposal loss can conceal a detached path.
    assert len(captured) == 2
    assert all(t.grad is not None and t.grad.isfinite().all() and t.grad.abs().sum()>0 for t in captured)
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(), name
        assert p.grad.abs().sum()>0, name


def test_second_order_and_receiver_motion_have_vector_gradient(setup):
    model, _ = setup
    block = model.proposal_blocks[0].coupling
    c = model.cfg.node_channels
    x, environment, motion = [torch.randn(6, 9*c, requires_grad=True) for _ in range(3)]
    output = block(x, environment, motion)[:, c:4*c]
    gradients = torch.autograd.grad(output.square().mean(), (x, environment, motion))
    for grad in gradients:
        assert grad[:, 4*c:].abs().sum()>0
        assert grad.isfinite().all()
    baseline = block(x, environment, motion)
    assert not torch.allclose(block(x, environment, torch.zeros_like(motion)), baseline)


def test_cutoff_edge_removal_has_smooth_limit(setup):
    model, _ = setup
    block, relations = model.proposal_blocks[0], model.relations
    c, e = model.cfg.node_channels, model.cfg.edge_channels
    state, motion = torch.randn(2, 9*c), torch.randn(2, 9*c)
    edge = torch.randn(1, e)
    edges = (torch.tensor([1]), torch.tensor([0]))
    empty_edges = (edges[0][:0], edges[1][:0])
    empty_vec = torch.empty(0, 3)
    empty_rel = relations(empty_vec, empty_vec, empty_vec, empty_vec)
    no_edge, _ = block(state, edge[:0], empty_edges, empty_rel, motion, model.aggregation_scale)
    distances = []
    for delta in [1e-2, 1e-3, 1e-4, 0.]:
        r = torch.tensor(model.cfg.cutoff_angstrom-delta, requires_grad=True)
        d = torch.stack([r, r*0, r*0])[None]
        u = torch.tensor([[.2, .1, 0.]])
        rel = relations(d, u, d, u)
        out, _ = block(state, edge, edges, rel, motion, model.aggregation_scale)
        derivative, = torch.autograd.grad(out.square().sum(), r)
        distances.append(float((out-no_edge).norm()))
        assert derivative.isfinite()
        if delta == 0:
            torch.testing.assert_close(out, no_edge, atol=1e-12, rtol=1e-12)
            assert derivative.abs()<1e-10
    assert distances[2] < distances[1] < distances[0]


def test_zero_speed_and_fp32_backward(setup):
    model, graph = setup
    u = torch.zeros(5, 3, requires_grad=True)
    torch.autograd.gradcheck(motion_harmonics, (u,), atol=1e-6, rtol=1e-5)
    model = model.float()
    graph = graph.apply(lambda x: x.float() if x.is_floating_point() else x)
    graph.velocities.zero_()
    graph.velocities.requires_grad_(True)
    out = model(graph)
    loss = out['displacements'].square().mean()+out['velocities'].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert graph.velocities.grad.isfinite().all()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())


def test_validation_and_current_only_graph_contract(setup):
    model, graph = setup
    other = graph.clone()
    other.timestep *= 2
    with pytest.raises(ValueError, match="timestep"):
        model.validate_graph(other)
    control = copy.deepcopy(model)
    control.cfg.relation_conditioning = 'current_only'
    full, current = model(graph), control(graph)
    torch.testing.assert_close(full['proposal_displacements'], current['proposal_displacements'])
    assert not torch.allclose(full['correction_displacements'], current['correction_displacements'])
    # Diagnostic switch removes BOTH future-edge and future-node-motion inputs.
    gradients = torch.autograd.grad(current['correction_displacements'].square().sum(),
        tuple(control.proposal_head.parameters()), allow_unused=True)
    assert all(g is None or g.abs().sum()==0 for g in gradients)
