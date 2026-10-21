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
