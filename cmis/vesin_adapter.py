"""Production periodic graph adapter. Requires `vesin` and `numpy`.

Matches the existing project's convention:
  edge_index[0] = sender (j); edge_index[1] = receiver (i)
  shift in Angstrom, dr = q[sender] + shift - q[receiver].
No derivatives through discrete edge topology or image selection, while the
model differentiates continuously through q in its geometry calculations.
"""
import numpy as np
import torch


class VesinGraph:
    def __init__(self, cell: torch.Tensor, cutoff: float):
        import vesin
        self.cell=np.asarray(cell.detach().cpu().numpy(),dtype=np.float64)
        self.cutoff=float(cutoff)
        self.nl=vesin.NeighborList(cutoff=cutoff,full_list=True,sorted=True)
        self.cell_inv=np.linalg.inv(self.cell)

    def __call__(self, q: torch.Tensor):
        xyz=np.asarray(q.detach().cpu().numpy(),dtype=np.float64)
        images=np.floor(xyz@self.cell_inv)
        wrapped=xyz-images@self.cell
        i,j,shift=self.nl.compute(wrapped,self.cell,True,quantities='ijS')
        translation=(shift-images[j]+images[i])@self.cell
        ei=torch.as_tensor(np.stack((j,i)).astype(np.int64),device=q.device)
        es=torch.as_tensor(translation,device=q.device,dtype=q.dtype)
        return ei,es


class BatchGraph:
    """Handle B independent systems, each with N atoms and its own cell.
    Returns concatenated directed neighbor lists with global flat node indices.
    Use debug_small_cell=True only for small toy boxes, not production.
    """
    def __init__(self,cells:torch.Tensor,n_atoms:int,cutoff:float,debug_small_cell=False):
        self.n_atoms=n_atoms
        self.debug=debug_small_cell
        self.cutoff=cutoff
        self.cells=cells
        if not debug_small_cell:
            self.builders=[VesinGraph(cell,cutoff) for cell in cells]

    def __call__(self,flatq:torch.Tensor):
        edges=[];shifts=[]
        if self.debug:
            try:
                from .cmis_model import small_cell_periodic_graph
            except ImportError:
                from cmis_model import small_cell_periodic_graph
        for b in range(len(self.cells)):
            q=flatq[b*self.n_atoms:(b+1)*self.n_atoms]
            if self.debug:
                e,s=small_cell_periodic_graph(q,self.cells[b],self.cutoff)
            else:
                e,s=self.builders[b](q)
            edges.append(e+b*self.n_atoms)
            shifts.append(s)
        return torch.cat(edges,dim=1),torch.cat(shifts,dim=0)
