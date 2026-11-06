"""Mathematical/software tests; no claim of physical accuracy."""
import math
import torch
from cmis_model import CMIS, CMISConfig, small_cell_periodic_graph

torch.set_default_dtype(torch.float64)
torch.manual_seed(23)
N=7
cell=torch.diag(torch.tensor([22.0,24.0,26.0]))
q=torch.tensor([[0.,0.,0.],[1.9,0.2,0.1],[3.1,1.2,-0.4],
                [1.,2.,1.],[3.,2.7,0.8],[4.5,-1.1,0.3],[-0.5,2.1,-1.]])
v=torch.randn(N,3)*.014
m=torch.tensor([16.,1.,1.,16.,1.,1.,16.])
z=torch.tensor([8,1,1,8,1,1,8])
cfg=CMISConfig(width=32,modes=4,encoder_layers=2,rbf_dim=10,cutoff=5.0,horizon_fs=5.0)
model=CMIS(cfg).double()
builder=lambda x:small_cell_periodic_graph(x,cell,cfg.cutoff)
times=torch.tensor([0.,1.0,2.5,5.])

def close(x,y,atol=1e-9):
    torch.testing.assert_close(x,y,atol=atol,rtol=1e-9)


def test_time_zero():
    out=model(q,v,z,m,times,builder)
    close(out['positions'][0],q)
    close(out['velocities'][0],v)


def test_linear_momentum_and_com():
    out=model(q,v,z,m,times,builder)
    p=(m[None,:,None]*out['velocities']).sum(1)
    close(p, (m[:,None]*v).sum(0).expand_as(p))
    com=(m[None,:,None]*out['positions']).sum(1)/m.sum()
    expected=(m[:,None]*q).sum(0)/m.sum() + times[:,None]*((m[:,None]*v).sum(0)/m.sum())
    close(com,expected)


def test_translation_boost():
    a=torch.tensor([2.1,-.8,.32]); boost=torch.tensor([.17,-.08,.04])
    baseline=model(q,v,z,m,times,builder)
    changed=model(q+a,v+boost,z,m,times,builder)
    close(changed['positions'],baseline['positions']+a+times[:,None,None]*boost,atol=2e-8)
    close(changed['velocities'],baseline['velocities']+boost,atol=2e-8)


def test_rotation_covariance():
    A=torch.randn(3,3)
    R,_=torch.linalg.qr(A)
    if torch.linalg.det(R)<0: R[:,0]=-R[:,0]
    # row-vector rotation (q @ R); rotate box rows too
    rotcell=cell@R
    rotbuilder=lambda x:small_cell_periodic_graph(x,rotcell,cfg.cutoff)
    base=model(q,v,z,m,times,builder)
    out=model(q@R,v@R,z,m,times,rotbuilder)
    close(out['positions'],base['positions']@R,atol=2e-8)
    close(out['velocities'],base['velocities']@R,atol=2e-8)


def test_periodic_images():
    qimg=q.clone()
    qimg[1]+=cell[0]
    qimg[3]-=cell[1]
    base=model(q,v,z,m,times,builder)
    changed=model(qimg,v,z,m,times,builder)
    expect=base['positions'].clone()
    expect[:,1]+=cell[0]
    expect[:,3]-=cell[1]
    close(changed['positions'],expect,atol=2e-8)
    close(changed['velocities'],base['velocities'],atol=2e-8)


def test_velocity_matches_position_time_derivative():
    t=torch.tensor([1.9],requires_grad=True)
    out=model(q,v,z,m,t,builder)
    # use forward-mode directional Jacobian derivative by autograd component-wise
    a,b=out['positions'].shape[1:]
    for i,j in [(0,0),(2,1),(6,2)]:
        deriv=torch.autograd.grad(out['positions'][0,i,j],t,retain_graph=True)[0]
        close(deriv.squeeze(),out['velocities'][0,i,j],atol=2e-9)


def test_mode_gradient_and_no_collapse_by_construction():
    out=model(q,v,z,m,torch.tensor([2.5,5.]),builder,return_modes=True)
    loss=(out['positions'][-1]-q).square().sum()+out['velocities'][-1].square().sum()
    model.zero_grad();loss.backward()
    grad=model.mode_embeddings.grad
    assert grad is not None and grad.abs().sum().item()>0
    # Different mode outputs are allowed and there is no softmax over their contributions.
    # This is NOT proof they learn independent modes after physical training.
    assert out['mode_impulses_initial'].shape[1]==cfg.modes


def test_no_edges_free_flight():
    def empty(x):
        return torch.empty((2,0),dtype=torch.long),x.new_zeros((0,3))
    out=model(q,v,z,m,times,empty)
    close(out['positions'],q[None]+times[:,None,None]*v[None])
    close(out['velocities'],v[None].expand(len(times),-1,-1))


def test_permutation_equivariance():
    perm=torch.tensor([3,5,2,0,6,4,1])
    base=model(q,v,z,m,times,builder)
    new=model(q[perm],v[perm],z[perm],m[perm],times,builder)
    close(new['positions'],base['positions'][:,perm],atol=2e-8)
    close(new['velocities'],base['velocities'][:,perm],atol=2e-8)



# Additional checks callable explicitly or through pytest discovery.
def test_query_independence():
    separate=model(q,v,z,m,torch.tensor([2.5]),builder)
    dense=model(q,v,z,m,torch.tensor([0.,1.,2.5,3.,5.]),builder)
    close(separate['positions'][0],dense['positions'][2])
    close(separate['velocities'][0],dense['velocities'][2])


def test_frozen_mode_intervention():
    mask=torch.tensor([True,False,True,True])
    baseline=model(q,v,z,m,torch.tensor([5.]),builder)
    altered=model(q,v,z,m,torch.tensor([5.]),builder,active_modes=mask)
    assert not torch.allclose(baseline['positions'],altered['positions'],atol=1e-14,rtol=1e-14)


def test_adaptive_frequency_gradient():
    model.zero_grad(set_to_none=True)
    out=model(q,v,z,m,torch.tensor([1.,2.,4.]),builder)
    out['positions'].square().sum().backward()
    g=model.step.mode_frequency_logit.grad
    assert g is not None and g.abs().max().item()>1e-15


def test_disconnected_batch_galilean_independence():
    B=2;N=len(q)
    def multi_graph(qflat):
        ea=[];sa=[]
        for b in range(B):
            ei,es=builder(qflat[b*N:(b+1)*N])
            ea.append(ei+b*N);sa.append(es)
        return torch.cat(ea,1),torch.cat(sa,0)
    graph_id=torch.arange(B).repeat_interleave(N)
    qbatch=torch.cat([q,q+torch.tensor([4.0,0.0,0.0])])
    vbatch=torch.cat([v,v+torch.tensor([0.018,0.0,0.0])])
    zbatch=z.repeat(B);mbatch=m.repeat(B)
    out=model(qbatch,vbatch,zbatch,mbatch,times,multi_graph,graph_index=graph_id)
    base=model(q,v,z,m,times,builder)
    close(out['positions'][:,:N],base['positions'],atol=2e-8)
    close(out['positions'][:,N:],base['positions']+torch.tensor([4.,0.,0.])+times[:,None,None]*torch.tensor([.018,0.,0.]),atol=2e-8)
    close(out['velocities'][:,N:],base['velocities']+torch.tensor([.018,0.,0.]),atol=2e-8)


if __name__=='__main__':
    import time
    ts=time.time()
    tests=[v for k,v in globals().items() if k.startswith('test_') and callable(v)]
    for t in tests:
        t(); print('PASS',t.__name__)
    print(f'{len(tests)} checks passed in {time.time()-ts:.1f}s')
