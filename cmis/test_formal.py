"""Production graph and formal segment algebra; no fitted toy models."""
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from cmis.cmis_model import CMIS,CMISConfig,oscillatory_primitives
from cmis.train_formal import reverse_segments
from cmis.vesin_adapter import VesinGraph


def test_production_triclinic_graph_matches_ase_and_reimages():
    rng=np.random.default_rng(19)
    cell=np.array([[8.,0.,0.],[3.,7.,0.],[1.,2.,8.]])
    q=rng.uniform(size=(18,3))@cell
    atoms=Atoms(numbers=[1,8,14]*6,positions=q,cell=cell,pbc=True)
    i,j,s=neighbor_list('ijS',atoms,4.5)
    expected=np.c_[j,i,s]
    builder=VesinGraph(torch.tensor(cell,dtype=torch.float64),4.5)
    edge,shift=builder(torch.tensor(q,dtype=torch.float64))
    actual_vectors=q[edge[0]]+shift.numpy()-q[edge[1]]
    expected_vectors=q[j]+s@cell-q[i]
    def canonical(e,vector):
        rows=np.c_[e.T,np.round(vector,9)]
        return rows[np.lexsort(tuple(rows[:,k] for k in range(4,-1,-1)))]
    np.testing.assert_allclose(canonical(edge.numpy(),actual_vectors),
                               canonical(expected[:,:2].T,expected_vectors),atol=1e-8)
    images=rng.integers(-3,4,size=(len(q),3));moved=q+images@cell
    ei,sh=builder(torch.tensor(moved,dtype=torch.float64))
    np.testing.assert_allclose(canonical(ei.numpy(),moved[ei[0]]+sh.numpy()-moved[ei[1]]),
                               canonical(edge.numpy(),actual_vectors),atol=1e-8)


def test_oscillatory_integrals_are_stable_for_small_query_time():
    t=torch.tensor([0.,1e-6,.01,.2,1.],dtype=torch.float64,requires_grad=True)
    omega=torch.tensor([[.025,3.5]],dtype=torch.float64)
    p,q=oscillatory_primitives(t,omega)
    for basis in range(3):
        derivative=torch.autograd.grad(q[:,0,0,basis].sum(),t,retain_graph=True)[0]
        torch.testing.assert_close(derivative,p[:,0,0,basis],atol=1e-12,rtol=1e-9)
    pf,qf=oscillatory_primitives(t.detach().float(),omega.float())
    torch.testing.assert_close(qf.double(),q.detach(),atol=1e-7,rtol=1e-5)


def test_time_reversal_uses_actual_earlier_states():
    initial=torch.zeros(1,2,3);targets=torch.stack([initial+1,initial+2,initial+3],1)
    data=dict(q0=initial,v0=initial+10,q_target=targets,v_target=targets+10,cell=torch.eye(3)[None])
    reversed=reverse_segments(data,torch.tensor([True]))
    torch.testing.assert_close(reversed['q0'],initial+3)
    torch.testing.assert_close(reversed['q_target'],torch.stack([initial+2,initial+1,initial],1))
    torch.testing.assert_close(reversed['v_target'],-(reversed['q_target']+10))
