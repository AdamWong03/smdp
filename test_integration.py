import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from periodic import PeriodicGraphBuilder, predict_batch
from test_neural_smdp import make_fixture


def test_production_neighbors_triclinic_multiple_images():
    cell=np.array([[5.,0,0],[2.1,4.8,0],[.2,.7,5.3]])
    q=np.random.default_rng(4).random((7,3))@cell
    q[1]+=3*cell[0]-cell[1]
    b=PeriodicGraphBuilder(4.5)
    e,s=b.one(q,cell)
    atoms=Atoms('H7',positions=q,cell=cell,pbc=True)
    i,j,d=neighbor_list('ijD',atoms,4.5)
    expected=sorted((int(a),int(b),*np.round(v,5)) for a,b,v in zip(j,i,d))
    actual=sorted((int(a),int(b),*np.round(q[a]+v-q[b],5)) for a,b,v in zip(*e,s))
    assert expected==actual


def test_disjoint_batch_and_all_module_gradients():
    model,q,v,z,m,cell=make_fixture()
    builder=PeriodicGraphBuilder(4.5)
    qs=torch.stack([q,q+torch.tensor([2.,-1.,3.])])
    vs=torch.stack([v,v*1.3+0.01])
    zs=z.expand(2,-1); ms=m.expand(2,-1); cs=cell.expand(2,-1,-1)
    times=[0.,1.,8.,16.]
    out=predict_batch(model,qs,vs,zs,ms,cs,times,builder)
    for g in range(2):
        solo=predict_batch(model,qs[g:g+1],vs[g:g+1],zs[g:g+1],ms[g:g+1],cs[g:g+1],times,builder)
        for key in ['positions','velocities']:
            torch.testing.assert_close(out[key][:,g*len(q):(g+1)*len(q)],solo[key],atol=1e-10,rtol=1e-9)
    loss=out['positions'][-1].square().mean()+100*out['velocities'][-1].square().mean()
    loss.backward()
    for module in [model.encoder,model.decompose,model.propagate.recurrent,model.propagate.direct,model.propagate.router,model.couple,model.decode]:
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0
    for g in range(2):
        dv=out['velocities'][-1,g*len(q):(g+1)*len(q)]-vs[g]
        torch.testing.assert_close((dv*m[:,None]).sum(0),torch.zeros(3,dtype=q.dtype),atol=1e-10,rtol=0)


def test_messages_continuous_when_edge_crosses_cutoff():
    model,_,_,_,_,_=make_fixture()
    cell=torch.eye(3,dtype=torch.float64)[None]*20
    q=torch.tensor([[[2.,2.,2.],[6.5,2.,2.],[2.,3.5,2.]]],dtype=torch.float64)
    v=torch.tensor([[[.002,0.,0.],[0.,.004,0.],[0.,0.,.003]]],dtype=torch.float64)
    z=torch.tensor([[14,8,8]]);m=torch.tensor([[28.085,15.999,15.999]],dtype=torch.float64)
    builder=PeriodicGraphBuilder(4.5)
    a=q.clone();b=q.clone();a[0,1,0]-=1e-7;b[0,1,0]+=1e-7
    pa=predict_batch(model,a,v,z,m,cell,[16.],builder)
    pb=predict_batch(model,b,v,z,m,cell,[16.],builder)
    torch.testing.assert_close(pa['positions']-a,pb['positions']-b,atol=1e-7,rtol=0)
    torch.testing.assert_close(pa['velocities'],pb['velocities'],atol=1e-8,rtol=0)


def test_additive_attribution_and_no_self_exchange():
    model,q,v,z,m,cell=make_fixture()
    out=predict_batch(model,q[None],v[None],z[None],m[None],cell[None],[0.,1.,16.],PeriodicGraphBuilder(4.5))
    assert out['fusion_weights'] is None
    assert torch.count_nonzero(out['coupling_attention'].diagonal(dim1=-2,dim2=-1))==0
    torch.testing.assert_close(out['positions']-q[None]-torch.tensor([0.,1.,16.])[:,None,None]*v[None],
                               out['mode_delta_q'].sum(2),atol=1e-12,rtol=1e-8)
    torch.testing.assert_close(out['velocities']-v[None],out['mode_delta_v'].sum(2),atol=1e-12,rtol=1e-8)


def test_reference_kinematics_and_species_targets():
    from training_objectives import SegmentStats, physical_segment_objectives
    q=torch.randn(4,6,3,dtype=torch.float64);v=torch.randn_like(q)
    result=physical_segment_objectives({'positions':q,'velocities':v},q,v,torch.tensor([0.,1.,2.,16.]),
        torch.tensor([8,8,8,8,14,14]),torch.ones(6),SegmentStats(1.,1.))
    assert all(float(x)==0 for x in result.values())


def test_pushforward_long_rollout_backward():
    from train_quartz import loss_for
    from training_objectives import SegmentStats
    from reference_data import QuartzSegments
    model,q,v,z,m,cell=make_fixture()
    t=torch.tensor(QuartzSegments.offsets,dtype=q.dtype)
    batch=dict(q=(q[None]+t[:,None,None]*v[None])[None],v=v[None,None].expand(1,len(t),-1,-1),
        z=z[None],mass=m[None],cell=cell[None])
    result,_=loss_for(model,batch,torch.tensor([0.,1.,4.,8.,15.,16.],dtype=q.dtype),
                       SegmentStats(.1,.01),PeriodicGraphBuilder(4.5),16)
    assert all(torch.isfinite(x) for x in result.values())
    result['loss'].backward()
    assert model.couple.exchange_gate.weight.grad.abs().sum()>0
    assert model.propagate.recurrent.parameters().__next__().grad.abs().sum()>0
