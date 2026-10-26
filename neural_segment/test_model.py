import numpy as np
import pytest
import torch
from neural_segment.model import SegmentConfig, NeuralSegment, integrated_legendre
from periodic import PeriodicGraphBuilder


def fixture(variant="full"):
    torch.manual_seed(26)
    q = torch.tensor([[.1,.2,.3], [1.7,.4,.2], [3.,1.9,.1], [.4,2.,1.]], dtype=torch.float64)
    v = torch.randn_like(q)*.005
    m = torch.tensor([16.,28.,16.,28.], dtype=q.dtype)
    z = torch.tensor([8,14,8,14])
    cell = np.array([[10.,0,0],[-3.,9.,0],[0,0,10.]])
    e, s = PeriodicGraphBuilder(4.5).one(q.numpy(), cell)
    model = NeuralSegment(SegmentConfig(variant=variant, scalar_dim=16, vector_dim=8,
        encoder_layers=2, direct_scalar=24, direct_vector=12)).double()
    return model, q, v, z, m, torch.tensor(e), torch.tensor(s,dtype=q.dtype), cell


@pytest.mark.parametrize("count", [6,8])
def test_integral_identities(count):
    for t in [0., .13, .5, 1.]:
        u = torch.tensor([t], dtype=torch.float64, requires_grad=True)
        a, v, q = integrated_legendre(u, count)
        for k in range(count):
            dv = torch.autograd.grad(v[0,k], u, retain_graph=True)[0]
            dq = torch.autograd.grad(q[0,k], u, retain_graph=True)[0]
            torch.testing.assert_close(dv, a[:,k], atol=2e-13, rtol=2e-13)
            torch.testing.assert_close(dq, v[:,k], atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("variant", ["full","independent","direct"])
def test_model_identities_and_gradients(variant):
    model,q,v,z,m,e,s,_ = fixture(variant)
    coeff,_ = model.coefficients(q,v,z,m,e,s)
    zero = model.query(q,v,coeff,[0.])
    torch.testing.assert_close(zero['positions'][0],q,atol=0,rtol=0)
    torch.testing.assert_close(zero['velocities'][0],v,atol=0,rtol=0)
    t = torch.tensor([6.3],dtype=q.dtype,requires_grad=True)
    out = model.query(q,v,coeff,t)
    for atom in range(len(q)):
        for axis in range(3):
            d = torch.autograd.grad(out['positions'][0,atom,axis],t,retain_graph=True)[0]
            torch.testing.assert_close(d,out['velocities'][:,atom,axis],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close((out['velocities'][0]*m[:,None]).sum(0),(v*m[:,None]).sum(0),atol=1e-12,rtol=1e-12)
    loss=out['positions'].square().sum()+out['velocities'].square().sum()
    loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    dense=model(q,v,z,m,e,s,list(range(17)))
    endpoint=model(q,v,z,m,e,s,[16.])
    torch.testing.assert_close(dense['positions'][-1],endpoint['positions'][0])


def test_e3_pbc_batch_and_single_encoding():
    model,q,v,z,m,e,s,cell=fixture()
    out=model(q,v,z,m,e,s,[16.])
    rot=torch.linalg.qr(torch.randn(3,3,dtype=q.dtype))[0]
    rot[:,0]*=-torch.linalg.det(rot)  # include improper rotations
    trans=q.new_tensor([3.,-2.,1.])
    other=model(q@rot+trans,v@rot,z,m,e,s@rot,[16.])
    torch.testing.assert_close(other['positions'],out['positions']@rot+trans,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(other['velocities'],out['velocities']@rot,atol=1e-11,rtol=1e-10)
    images=np.array([[0,0,0],[1,-1,0],[0,1,1],[-1,0,0]])@cell
    shifted=q+torch.tensor(images)
    ep,sp=PeriodicGraphBuilder(4.5).one(shifted.numpy(),cell)
    wrapped=model(shifted,v,z,m,torch.tensor(ep),torch.tensor(sp,dtype=q.dtype),[16.])
    torch.testing.assert_close(wrapped['positions']-torch.tensor(images)[None],out['positions'],atol=2e-8,rtol=1e-8)
    torch.testing.assert_close(wrapped['velocities'],out['velocities'],atol=2e-9,rtol=1e-8)
    batch=model(torch.cat([q,q+trans]),torch.cat([v,v+1.]),z.repeat(2),m.repeat(2),
                torch.cat([e,e+len(q)],1),s.repeat(2,1),[16.],
                graph_index=torch.arange(2).repeat_interleave(len(q)))
    torch.testing.assert_close(batch['positions'][:,:len(q)],out['positions'])
    torch.testing.assert_close(batch['positions'][:,len(q):],out['positions']+trans+16.)
    calls=[]
    hook=model.encoder.register_forward_hook(lambda *args:calls.append(1))
    model(q,v,z,m,e,s,list(range(17)))
    hook.remove()
    assert len(calls)==1
