"""Network identities on periodic graphs; no fit or dynamics experiment."""
import pytest
import torch
import copy
from torch_geometric.data import Batch
from trajcast.data.atomic_graph import AtomicGraph

from relational_transition import RelationalConfig, RelationalTransition, endpoint_objective


def graph(n=5):
    q = torch.rand(n, 3, dtype=torch.float64) * 5
    cell = torch.tensor([[7., 0., 0.], [2., 7., 0.], [1., 2., 7.]], dtype=q.dtype)
    return AtomicGraph.from_atoms_dict(dict(
        positions=q, cell=cell, pbc=torch.ones(3, dtype=torch.bool),
        velocities=torch.randn(n, 3, dtype=q.dtype) * .003,
        atomic_numbers=torch.tensor([8, 14] * n)[:n, None],
        atomic_masses=torch.tensor([16., 28.] * n, dtype=q.dtype)[:n, None],
        timestep=torch.tensor([30.], dtype=q.dtype)), r_cut=4.5, atom_type_mapper={8: 0, 14: 1})


@pytest.fixture(params=['legacy', 'geometry'])
def setup(request):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(17)
    model = RelationalTransition({'block_type': request.param}).double()
    batch = Batch.from_data_list([graph(5), graph(7)])
    yield model, batch
    torch.set_default_dtype(previous_dtype)


def compare(a, b):
    for key in ['displacements', 'velocities', 'proposal_displacements', 'proposal_velocities']:
        torch.testing.assert_close(a[key], b[key], atol=1e-10, rtol=1e-9)


def test_equivariance_translation_and_boost(setup):
    model, g = setup
    out = model(g)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=g.pos.dtype))
    rotation[:, 0] *= -torch.linalg.det(rotation)  # Include reflection, not just SO(3).
    transformed = g.clone()
    transformed.pos = g.pos @ rotation.T + 3
    transformed.velocities = g.velocities @ rotation.T
    transformed.shifts = g.shifts @ rotation.T
    transformed.cell = g.cell @ rotation.T
    predicted = model(transformed)
    compare(predicted, {key: value @ rotation.T for key, value in out.items()
                        if key != 'target'})
    boost = torch.tensor([.001, -.002, .003], dtype=g.pos.dtype)
    boosted = g.clone()
    boosted.velocities += boost
    predicted = model(boosted)
    for key in ['displacements', 'proposal_displacements']:
        torch.testing.assert_close(predicted[key], out[key] + 30*boost)
    for key in ['velocities', 'proposal_velocities']:
        torch.testing.assert_close(predicted[key], out[key] + boost)


def test_batch_momentum_and_permutation(setup):
    model, g = setup
    out = model(g)
    singles = [model(item) for item in g.to_data_list()]
    compare(out, {key: torch.cat([x[key] for x in singles]) for key in out})
    for k in range(2):
        mask = g.batch == k
        m, v = g.atomic_masses[mask], g.velocities[mask]
        torch.testing.assert_close((out['velocities'][mask]*m).sum(0), (v*m).sum(0))
        torch.testing.assert_close((out['displacements'][mask]*m).sum(0), 30*(v*m).sum(0))
    perm = torch.randperm(len(g.pos))
    inverse = torch.argsort(perm)
    pg = g.clone()
    for name in ['pos', 'velocities', 'atomic_numbers', 'atomic_masses', 'batch']:
        pg[name] = g[name][perm]
    pg.edge_index = inverse[g.edge_index]
    compare(model(pg), {key: value[perm] for key, value in out.items()})


def test_pbc_reimage_and_rebuild(setup):
    model, batch = setup
    g = batch.to_data_list()[0]
    new = g.clone()
    new.pos[::2] += g.cell[0]
    new.pos[1::2] -= 2*g.cell[1]
    # Recompute periodic neighbors using the actual upstream graph constructor.
    new = AtomicGraph.from_atoms_dict(dict(
        positions=new.pos, cell=g.cell, pbc=g.pbc, velocities=g.velocities,
        atomic_numbers=g.atomic_numbers, atomic_masses=g.atomic_masses,
        timestep=g.timestep), r_cut=4.5, atom_type_mapper={8:0,14:1})
    compare(model(g), model(new))


def test_final_gradient_reaches_proposal_through_relations(setup):
    model, g = setup
    future_tensors = []
    def capture(_module, inputs, _output):
        if inputs[2].requires_grad:
            inputs[2].retain_grad()
            inputs[3].retain_grad()
            future_tensors.extend([inputs[2], inputs[3]])
    hook = model.relations.register_forward_hook(capture)
    out = model(g)
    hook.remove()
    # Zero auxiliary weight: the main prediction must train the full network.
    loss, _ = endpoint_objective(model, out, .2*g.velocities, .9*g.velocities,
                                 proposal_weight=0)
    loss.backward()
    assert len(future_tensors) == 2
    for value in future_tensors:
        assert value.grad is not None and value.grad.isfinite().all()
        assert value.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_feedback_changes_prediction(setup):
    model, g = setup
    original = model(g)
    def no_feedback(module, inputs):
        d0, u0, d1, u1 = inputs
        return d0, u0, d0, u0
    handle = model.relations.register_forward_pre_hook(no_feedback)
    without = model(g)
    handle.remove()
    torch.testing.assert_close(original['proposal_displacements'], without['proposal_displacements'])
    assert (original['displacements'] - without['displacements']).abs().max() > 1e-9


def test_horizon_mismatch_rejected(setup):
    model, g = setup
    g.timestep.fill_(16)
    with pytest.raises(ValueError, match='timestep'):
        model(g)


def test_factoring_preserves_outputs_and_gradients(setup):
    model, g = setup
    plain = copy.deepcopy(model)
    for module in plain.modules():
        if hasattr(module, 'factorized_edges'):
            module.factorized_edges = False
    a, b = model(g), plain(g)
    compare(a, b)
    for network, output in [(model, a), (plain, b)]:
        loss, _ = endpoint_objective(network, output, .2*g.velocities, .9*g.velocities)
        loss.backward()
    for (name, pa), (_, pb) in zip(model.named_parameters(), plain.named_parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-9, rtol=1e-8, msg=name)


def test_fp32_forward_backward(setup):
    model, g = setup
    model = model.float()
    g = g.clone().apply(lambda x: x.float() if x.is_floating_point() else x)
    output = model(g)
    loss, _ = endpoint_objective(model, output, .2*g.velocities, .9*g.velocities)
    loss.backward()
    assert loss.isfinite()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_geometry_block_reads_current_and_candidate_axes():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(31)
        model = RelationalTransition({'block_type': 'geometry'}).double()
        g = Batch.from_data_list([graph(6)])
    finally:
        torch.set_default_dtype(previous_dtype)
    block = model.refinement_blocks[0]
    observed = []
    handle = block.register_forward_hook(lambda _m, inputs, _out: observed.append(inputs[-1][1]))
    model(g)
    handle.remove()
    directions = observed[0].detach().clone().requires_grad_(True)
    src, dst = g.edge_index.flip(0)
    v = torch.randn(len(g.pos), model.cfg.vector_channels, 3, dtype=torch.float64)
    value = block.geometry_features(v, (src, dst), directions).square().sum()
    grad, = torch.autograd.grad(value, directions)
    assert grad[:, 0].abs().sum() > 0
    assert grad[:, 1].abs().sum() > 0
