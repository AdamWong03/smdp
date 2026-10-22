"""Unit-level mathematical checks, not physical MD validation or training."""
import math
import torch
from smdp_neural import NeuralSMDP, SMDPConfig, build_periodic_edges


def make_fixture():
    torch.manual_seed(7)
    n = 9
    q = torch.rand(n, 3) * 5 + 2
    v = torch.randn(n, 3) * 0.004
    z = torch.tensor([14, 8, 8, 14, 8, 8, 14, 8, 8])
    mass = torch.where(z == 8, torch.tensor(15.999), torch.tensor(28.085))
    cell = torch.eye(3) * 12.0
    cfg = SMDPConfig(scalar_dim=32, vector_dim=8, encoder_layers=2, rbf_dim=8, modes=3,
                     additive_modes=True, exclude_self_exchange=True)
    model = NeuralSMDP(cfg).double().eval()
    # Avoid trivially passing equivariance checks with zero-initialized heads.
    with torch.no_grad():
        model.decode.coeffs[-1].weight.normal_(0.0, 0.03)
    return model, q.double(), v.double(), z, mass.double(), cell.double()


def predict(model,q,v,z,m,cell,times):
    edges, shifts=build_periodic_edges(q,cell,4.5)
    return model(q,v,z,m,edges,shifts,times)


def test_shape_zero_horizon_and_backward():
    model,q,v,z,m,cell=make_fixture()
    q=q.detach().requires_grad_(True)
    o=predict(model,q,v,z,m,cell,[0,4,8,16])
    assert o['positions'].shape==(4,9,3)
    assert o['velocities'].shape==(4,9,3)
    torch.testing.assert_close(o['positions'][0],q,atol=1e-12,rtol=0)
    torch.testing.assert_close(o['velocities'][0],v,atol=1e-12,rtol=0)
    loss = (o['positions'][-1]**2).mean()+10*(o['velocities'][-1]**2).mean()
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_translation_rotation_and_time_query():
    model,q,v,z,m,cell=make_fixture()
    o=predict(model,q,v,z,m,cell,[4,8,16])
    shift=torch.tensor([1.4,-2.3,0.7],dtype=q.dtype)
    ot=predict(model,q+shift,v,z,m,cell,[4,8,16])
    torch.testing.assert_close(ot['positions'],o['positions']+shift,atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(ot['velocities'],o['velocities'],atol=1e-9,rtol=1e-9)
    theta=.63
    rotation=torch.tensor([[math.cos(theta),-math.sin(theta),0.0],
                           [math.sin(theta),math.cos(theta),0.0],
                           [0.0,0.0,1.0]],dtype=q.dtype)
    orot=predict(model,q@rotation.T,v@rotation.T,z,m,cell@rotation.T,[4,8,16])
    torch.testing.assert_close(orot['positions'],o['positions']@rotation.T,atol=1e-9,rtol=1e-8)
    torch.testing.assert_close(orot['velocities'],o['velocities']@rotation.T,atol=1e-9,rtol=1e-8)
    final=predict(model,q,v,z,m,cell,[16])
    torch.testing.assert_close(final['positions'][0],o['positions'][-1],atol=1e-9,rtol=0)


def test_center_of_mass_and_periodic_shift():
    model,q,v,z,m,cell=make_fixture()
    o=predict(model,q,v,z,m,cell,[16])
    expected=q+(16*v)
    corr=o['positions'][0]-expected
    dv=o['velocities'][0]-v
    torch.testing.assert_close((corr*m[:,None]).sum(0),torch.zeros(3,dtype=q.dtype),atol=1e-10,rtol=0)
    torch.testing.assert_close((dv*m[:,None]).sum(0),torch.zeros(3,dtype=q.dtype),atol=1e-10,rtol=0)
    # Moving any atom by an integer cell vector must not change physical predictions.
    qshift=q.clone();qshift[0]+=cell[0]
    op=predict(model,qshift,v,z,m,cell,[16]);
    torch.testing.assert_close(op['positions'][0,0]-cell[0],o['positions'][0,0],atol=1e-8,rtol=1e-8)
    torch.testing.assert_close(op['positions'][0,1:],o['positions'][0,1:],atol=1e-8,rtol=1e-8)
    torch.testing.assert_close(op['velocities'],o['velocities'],atol=1e-8,rtol=1e-8)


if __name__=='__main__':
    test_shape_zero_horizon_and_backward()
    test_translation_rotation_and_time_query()
    test_center_of_mass_and_periodic_shift()
    print('PASS: shapes, gradients, time queries, translation, rotation, PBC images, COM')
