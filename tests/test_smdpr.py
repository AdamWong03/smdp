"""Identities and gradient paths of modal propagation; no fitting experiment."""
import pytest
import torch
from torch_geometric.data import Batch

from relational_transition import RelationalTransition, endpoint_objective
from smdpr.model import SMDPR
from test_relational_transition import graph, compare


@pytest.fixture
def setup():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(17)
    cfg = dict(block_type='geometry_gated')
    model = SMDPR(cfg).double()
    batch = Batch.from_data_list([graph(5), graph(7)])
    yield model, batch
    torch.set_default_dtype(old)


def test_disabled_exact_baseline_and_checkpoint(setup):
    model, batch = setup
    baseline = RelationalTransition(dict(block_type='geometry_gated')).double()
    state = {key: value for key, value in model.state_dict().items()
             if key in baseline.state_dict()}
    baseline.load_state_dict(state, strict=True)
    model.modes_enabled = False
    a, b = model(batch), baseline(batch)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_equivariance_and_query(setup):
    model, batch = setup
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= -torch.det(rotation)
    new = batch.clone()
    new.pos = batch.pos @ rotation.T + 2
    new.velocities = batch.velocities @ rotation.T
    new.shifts = batch.shifts @ rotation.T
    new.cell = batch.cell @ rotation.T
    out, transformed = model(batch, True), model(new, True)
    compare(transformed, {key: value @ rotation.T for key, value in out.items()
                          if key not in ['context', 'target']})
    query = model.query(batch, out, [0, 7, 30])
    rotated = model.query(new, transformed, [0, 7, 30])
    torch.testing.assert_close(rotated['positions'], query['positions'] @ rotation.T + 2)
    torch.testing.assert_close(rotated['velocities'], query['velocities'] @ rotation.T)
    assert torch.equal(query['positions'][0], batch.pos)
    assert torch.equal(query['velocities'][0], batch.velocities)
    assert torch.equal(query['positions'][-1], out['positions'])
    assert torch.equal(query['velocities'][-1], out['velocities'])


def test_batch_boost_momentum_permutation(setup):
    model, batch = setup
    out = model(batch)
    singles = [model(g) for g in batch.to_data_list()]
    compare(out, {key: torch.cat([item[key] for item in singles]) for key in out})
    for k in range(2):
        mask = batch.batch == k
        m = batch.atomic_masses[mask]
        torch.testing.assert_close((out['velocities'][mask]*m).sum(0),
                                   (batch.velocities[mask]*m).sum(0))
    boost = torch.tensor([.001, -.002, .003])
    changed = batch.clone()
    changed.velocities += boost
    boosted = model(changed)
    torch.testing.assert_close(boosted['velocities'], out['velocities'] + boost)
    torch.testing.assert_close(boosted['displacements'], out['displacements'] + 30*boost)
    base_query = model.query(batch, model(batch, True), [5., 17.])
    boost_query = model.query(changed, model(changed, True), [5., 17.])
    torch.testing.assert_close(boost_query['positions'], base_query['positions']+
                               torch.tensor([5., 17.])[:, None, None]*boost)
    torch.testing.assert_close(boost_query['velocities'], base_query['velocities']+boost)
    for k in range(2):
        mask = batch.batch==k
        m = batch.atomic_masses[mask]
        for j, time in enumerate([5., 17.]):
            torch.testing.assert_close((base_query['velocities'][j, mask]*m).sum(0),
                                       (batch.velocities[mask]*m).sum(0))
            torch.testing.assert_close(((base_query['positions'][j, mask]-batch.pos[mask])*m).sum(0),
                                       time*(batch.velocities[mask]*m).sum(0))
    perm = torch.randperm(len(batch.pos))
    inverse = torch.argsort(perm)
    pg = batch.clone()
    for name in ['pos', 'velocities', 'atomic_numbers', 'atomic_masses', 'batch']:
        pg[name] = batch[name][perm]
    pg.edge_index = inverse[batch.edge_index]
    compare(model(pg), {key: value[perm] for key, value in out.items()})


def test_pbc_reimage(setup):
    from trajcast.data.atomic_graph import AtomicGraph
    model, batch = setup
    g = batch.to_data_list()[0]
    q = g.pos.clone()
    q[::2] += g.cell[0]
    new = AtomicGraph.from_atoms_dict(dict(positions=q, cell=g.cell, pbc=g.pbc,
        velocities=g.velocities, atomic_numbers=g.atomic_numbers,
        atomic_masses=g.atomic_masses, timestep=g.timestep), r_cut=4.5,
        atom_type_mapper={8: 0, 14: 1})
    compare(model(g), model(new))


def test_modes_neighbor_persistence_and_gradients(setup):
    model, batch = setup
    for block in model.modal_blocks:
        block.capture = True
    out = model(batch, True)
    loss, _ = endpoint_objective(model, out, batch.velocities*.2,
                                 batch.velocities*.9, proposal_weight=0)
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith('query_head.'):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None, name
            assert parameter.grad.isfinite().all(), name
            assert parameter.grad.abs().sum() > 0, name
    assert (model.motion_state.embeddings.grad.abs().sum(-1) > 0).all()
    assert all(block.last_record['affinity'].shape[1] == 3 for block in model.modal_blocks)
    assert out['context']['modes'] is model.modal_blocks[-1].last_record['modes']
    model.set_active_modes([True, False, True])
    removed = model(batch)
    assert not torch.equal(removed['displacements'], out['displacements'])


def test_query_loss_isolated(setup):
    model, batch = setup
    out = model(batch, True)
    query = model.query(batch, out, [5, 16, 25])
    loss = query['positions'].square().mean() + query['velocities'].square().mean()
    loss.backward()
    for name, p in model.named_parameters():
        if name.startswith('query_head.'):
            assert p.grad is not None and p.grad.isfinite().all(), name
        else:
            assert p.grad is None, name


def test_fp32_full_width_and_capacity():
    cfg = dict(scalar_channels=192, vector_channels=48, edge_channels=96,
               block_type='geometry_gated')
    model = SMDPR(cfg).float()
    baseline = RelationalTransition(cfg).float()
    total = sum(p.numel() for p in model.parameters())
    original = sum(p.numel() for p in baseline.parameters())
    assert total/original <= 1.15, (original, total)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        batch = Batch.from_data_list([
            graph().apply(lambda x: x.float() if x.is_floating_point() else x)
            for _ in range(2)])
    finally:
        torch.set_default_dtype(old)
    out = model(batch)
    out['target'].square().mean().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
