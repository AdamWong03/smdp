"""R3 computational invariants; run with 'python -m pytest -q test_r3.py'."""
import math
import numpy as np
import pytest
import torch

from smdp_neural import NeuralSMDP, SMDPConfig, build_periodic_edges
from r3_consistency import composition_consistency
from periodic import PeriodicGraphBuilder, predict_batch
from training_objectives import SegmentStats


def fixture(spatial):
    torch.manual_seed(1707)
    cfg = SMDPConfig(scalar_dim=24, vector_dim=6, modes=3,
                     encoder_layers=2, rbf_dim=8, cutoff_angstrom=4.5,
                     additive_modes=True, exclude_self_exchange=True,
                     spatial_mode_exchange=spatial)
    model = NeuralSMDP(cfg).double()
    with torch.no_grad():
        model.decode.coeffs[-1].weight.normal_(0.0, .02)
    q = torch.tensor([[1.,1.,1.],[2.,1.,1.],[1.,2.,1.],
                      [2.,2.,2.],[3.,1.,2.],[1.,3.,2.]], dtype=torch.float64)
    v = torch.randn(6, 3, dtype=torch.float64)*.005
    z = torch.tensor([14,8,8,14,8,8])
    m = torch.where(z == 14, torch.tensor(28.085), torch.tensor(15.999)).double()
    cell = 12*torch.eye(3, dtype=torch.float64)
    return model,q,v,z,m,cell


def predict(model,q,v,z,m,c,times):
    edges, shifts = build_periodic_edges(q,c,4.5)
    return model(q,v,z,m,edges,shifts,times)


def test_r3_a_is_exact_r2_graph_when_spatial_disabled():
    old, q, v, z, m, cell = fixture(False)
    r3, *_ = fixture(True)
    shared = {k: value for k,value in r3.state_dict().items()
              if not k.startswith('spatial_exchange.')}
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, shared[key], atol=0, rtol=0)
    assert not any(k.startswith('spatial_exchange.') for k in old.state_dict())
    output = predict(old,q,v,z,m,cell,[0.,8.,16.])
    torch.testing.assert_close(output['positions'][0],q,atol=1e-12,rtol=0)


def test_r3_b_rotation_translation_pbc_com_and_gradients():
    model,q,v,z,m,cell = fixture(True)
    output = predict(model,q,v,z,m,cell,[0.,8.,16.])
    theta=.47
    R=torch.tensor([[math.cos(theta),-math.sin(theta),0],
                    [math.sin(theta), math.cos(theta),0],
                    [0,0,1]],dtype=q.dtype)
    transformed = predict(model,q@R.T + torch.tensor([.4,-.7,1.2]),
                          v@R.T,z,m,cell@R.T,[0.,8.,16.])
    torch.testing.assert_close(transformed['positions'],
                               output['positions']@R.T+torch.tensor([.4,-.7,1.2]),
                               atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(transformed['velocities'],output['velocities']@R.T,
                               atol=1e-9,rtol=1e-9)
    shifted=q.clone(); shifted[1]+=cell[0]
    image=predict(model,shifted,v,z,m,cell,[8.,16.])
    torch.testing.assert_close(image['positions'][-1,1]-cell[0],output['positions'][-1,1],atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(image['velocities'][-1],output['velocities'][-1],atol=1e-9,rtol=1e-9)
    delta_v=output['velocities'][-1]-v
    torch.testing.assert_close((m[:,None]*delta_v).sum(0),torch.zeros(3,dtype=m.dtype),atol=1e-9,rtol=0)
    loss=(output['positions'][-1]-q).square().sum()
    loss.backward()
    grads=[p.grad for p in model.spatial_exchange.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads)>0


@pytest.mark.parametrize('spatial',[False,True])
def test_composition_8_plus_8_uses_predicted_graph_and_backprop(spatial):
    model,q,v,z,m,cell = fixture(spatial)
    builder = PeriodicGraphBuilder(4.5)
    # Two distinct graph samples; the input supervision is synthetic and is
    # NOT claimed to be a physical MD trajectory.
    start=q[None].repeat(2,1,1).float()
    start[1] += torch.tensor([.13,.07,-.02])
    speed=v[None].repeat(2,1,1).float()
    elements=z[None].repeat(2,1)
    masses=m[None].repeat(2,1).float()
    cells=cell[None].repeat(2,1,1).float()
    model=model.float()
    outputs=predict_batch(model,start,speed,elements,masses,cells,[8.,16.],builder)
    batch={'q':start[:,None], 'v':speed[:,None],
           'z':elements,'mass':masses,'cell':cells}
    stats=SegmentStats(.01,.001)
    terms=composition_consistency(model,batch,outputs,torch.tensor([8.,16.]),stats,builder,samples=2)
    assert torch.isfinite(terms['composition'])
    assert terms['composition']>=0
    terms['composition'].backward()
    assert model.decode.coeffs[-1].weight.grad is not None
    assert torch.isfinite(model.decode.coeffs[-1].weight.grad).all()


def test_spatial_empty_graph_is_exact_noop():
    model,q,v,z,m,cell=fixture(True)
    s=torch.randn(2,6,3,24,dtype=torch.float64)
    vectors=torch.randn(2,6,3,6,3,dtype=torch.float64)
    edges=torch.empty((2,0),dtype=torch.long)
    shifts=q.new_zeros((0,3))
    so,vo=model.spatial_exchange(s,vectors,q,edges,shifts)
    torch.testing.assert_close(so,s,atol=0,rtol=0)
    torch.testing.assert_close(vo,vectors,atol=0,rtol=0)
