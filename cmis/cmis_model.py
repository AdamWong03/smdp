"""CMIS: Coupled Modal Impulse Segments.

Force-free finite-stride phase-space trajectory predictor.
Time unit fs, distance Angstrom, velocity Angstrom/fs, mass amu.

This is a standalone mathematical/model core, NOT a trained physical MD model.
Use a production periodic graph builder (vesin) for real runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CMISConfig:
    width: int = 96
    modes: int = 4
    encoder_layers: int = 3
    rbf_dim: int = 20
    cutoff: float = 4.5
    horizon_fs: float = 5.0
    velocity_scale: float = 0.01
    correction_scale: float = 0.5
    mode_update_scale: float = 0.2


def mlp(nin: int, hidden: int, nout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(nin, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, nout))


class RadialBasis(nn.Module):
    def __init__(self, n: int, cutoff: float):
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n))
        self.gamma = (n / cutoff) ** 2

    def forward(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.where(r < self.cutoff, 0.5 * (torch.cos(math.pi*r/self.cutoff)+1), 0.0)
        b = torch.exp(-self.gamma*(r[:, None]-self.centers[None, :])**2)
        return b * c[:, None], c


def edge_geometry(q: torch.Tensor, v: torch.Tensor,
                  edge_index: torch.Tensor, edge_shift: torch.Tensor,
                  vscale: float):
    src, dst = edge_index
    d = q[src] + edge_shift - q[dst]
    r = torch.linalg.vector_norm(d, dim=-1).clamp_min(1e-8)
    n = d / r[:, None]
    u = (v[src]-v[dst])/vscale
    radial_speed = (u*n).sum(-1)
    transverse = u - radial_speed[:, None]*n
    return r, n, radial_speed, transverse, torch.linalg.vector_norm(u, dim=-1)


def oscillatory_primitives(s: torch.Tensor, omega: torch.Tensor):
    """Analytic integrals of phi=[1, cos(omega*s), sin(omega*s)].

    omega is dimensionless angular frequency over one normalized macrostep.
    P(s)=int_0^s phi(u)du; Q(s)=int_0^s (s-u)phi(u)du.
    Inputs: s [T], omega [E,K]. Returns P,Q [T,E,K,3].
    """
    ss=s[:,None,None]
    w=omega[None,:,:]
    x=ss*w
    sinc_full=torch.sinc(x/math.pi)
    sinc_half=torch.sinc(x/(2*math.pi))
    P0=ss+torch.zeros_like(x)
    Pc=ss*sinc_full
    Ps=0.5*w*ss**2*sinc_half.square()
    Q0=0.5*ss.square()+torch.zeros_like(x)
    Qc=0.5*ss**2*sinc_half.square()
    # Avoid catastrophic cancellation of (x-sin(x))/omega**2 near zero.
    w_safe=torch.where(w.abs()<1e-8,torch.ones_like(w),w)
    Qs_exact=(x-torch.sin(x))/w_safe.square()
    Qs_taylor=(w*ss**3)/6-(w**3*ss**5)/120+(w**5*ss**7)/5040
    Qs=torch.where(x.abs()<0.1,Qs_taylor,Qs_exact)
    return torch.stack([P0,Pc,Ps],dim=-1),torch.stack([Q0,Qc,Qs],dim=-1)


class ScalarEncoder(nn.Module):
    """Local invariant encoder. Equivariant outputs arise from vector-valued edge bases."""
    def __init__(self, cfg: CMISConfig):
        super().__init__()
        d=cfg.width
        self.cfg=cfg
        self.elements=nn.Embedding(120, d)
        self.node=mlp(2,d,d)
        self.rbf=RadialBasis(cfg.rbf_dim,cfg.cutoff)
        self.edge_nets=nn.ModuleList([mlp(2*d+cfg.rbf_dim+2,d,d)
                                      for _ in range(cfg.encoder_layers)])
        self.updates=nn.ModuleList([mlp(2*d,d,d) for _ in range(cfg.encoder_layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg.encoder_layers)])

    def forward(self,q,v,z,m,edge_index,edge_shift):
        z=z.long()
        s=self.elements(z)+self.node(torch.stack((m.log(),
                     (v/self.cfg.velocity_scale).square().sum(-1)),dim=-1))
        src,dst=edge_index
        if src.numel()==0:
            return s
        r,_,vr,_,us=edge_geometry(q,v,edge_index,edge_shift,self.cfg.velocity_scale)
        basis,env=self.rbf(r)
        degree=torch.zeros(q.shape[0],device=q.device,dtype=q.dtype).index_add_(0,dst,env)
        norm=degree.clamp_min(1.0)[:,None]
        for edge_net, update, layer_norm in zip(self.edge_nets,self.updates,self.norms):
            edge_feat=torch.cat([s[src],s[dst],basis,vr[:,None],us[:,None]],dim=-1)
            msg=edge_net(edge_feat)*env[:,None]
            agg=torch.zeros_like(s).index_add_(0,dst,msg)/norm
            s=layer_norm(s+0.2*update(torch.cat([s,agg],dim=-1)))
        return s


class ModeCoupling(nn.Module):
    def __init__(self,width:int):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.q=nn.Linear(width,width,bias=False)
        self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False)
        self.output=nn.Linear(width,width)
        self.gate=nn.Linear(2*width,1)

    def forward(self,s,active_modes=None):
        h=self.norm(s)
        attn=torch.einsum('nkd,njd->nkj',self.q(h),self.k(h))/math.sqrt(h.shape[-1])
        if active_modes is not None:
            attn=attn.masked_fill(~active_modes[None,None,:],-1e9)
        attn=attn.softmax(dim=-1)
        ctx=torch.einsum('nkj,njd->nkd',attn,self.v(h))
        g=torch.sigmoid(self.gate(torch.cat([h,ctx],dim=-1)))
        result=s+0.2*g*self.output(ctx)
        if active_modes is not None:
            result=result*active_modes[None,:,None]
        return result, attn


class PairModeStep(nn.Module):
    """One geometry-dependent mode propagation and antisymmetric pair-impulse output."""
    def __init__(self,cfg: CMISConfig):
        super().__init__()
        self.cfg=cfg
        d=cfg.width
        self.rbf=RadialBasis(cfg.rbf_dim,cfg.cutoff)
        self.edge=mlp(2*d+cfg.rbf_dim+2,d,d)
        self.node_msg=nn.Linear(d,d)
        self.node_up=mlp(2*d,d,d)
        self.node_norm=nn.LayerNorm(d)
        self.initial_head=nn.Linear(d,6)   # 3 temporal basis x (radial, transverse)
        self.refine_head=nn.Linear(d,6)
        self.frequency_head=nn.Linear(d,1)
        self.omega_min=0.005   # rad/fs, not a fixed physical frequency
        self.omega_max=1.05
        init_freq=torch.logspace(math.log10(0.03),math.log10(0.7),cfg.modes)
        frac=((init_freq-self.omega_min)/(self.omega_max-self.omega_min)).clamp(.01,.99)
        self.mode_frequency_logit=nn.Parameter(torch.logit(frac))
        nn.init.zeros_(self.frequency_head.weight)
        nn.init.zeros_(self.frequency_head.bias)
        nn.init.normal_(self.initial_head.weight, std=2e-3)
        nn.init.zeros_(self.initial_head.bias)
        nn.init.normal_(self.refine_head.weight,std=2e-4)
        nn.init.zeros_(self.refine_head.bias)

    def forward(self, s, q, v, mass, edge_index, edge_shift,
                is_refinement: bool=False,active_modes=None):
        # Graphs are full directed neighbor lists. Retain each undirected
        # periodic-image edge once. For a < b, use only that orientation.
        src_all,dst_all=edge_index
        keep=src_all<dst_all
        src,dst=src_all[keep],dst_all[keep]
        shift=edge_shift[keep]
        n_atoms,n_modes,dim=s.shape
        if src.numel()==0:
            return s, (src,dst,torch.empty((0,n_modes,3,3),device=q.device,dtype=q.dtype),
                       torch.empty((0,n_modes),device=q.device,dtype=q.dtype))
        r,n,vr,trans,us=edge_geometry(q,v,torch.stack([src,dst]),shift,self.cfg.velocity_scale)
        basis,env=self.rbf(r)
        # Symmetric scalar inputs; directed vector bases change sign under i<->j.
        edge_in=torch.cat((s[src]+s[dst],(s[src]-s[dst]).abs(),
              basis[:,None,:].expand(-1,n_modes,-1),
              vr[:,None,None].expand(-1,n_modes,1),
              us[:,None,None].expand(-1,n_modes,1)),dim=-1)
        e=self.edge(edge_in)
        msg=self.node_msg(e)*env[:,None,None]
        agg=torch.zeros_like(s).index_add_(0,src,msg).index_add_(0,dst,msg)
        degree=torch.zeros(n_atoms,device=q.device,dtype=q.dtype).index_add_(0,src,env).index_add_(0,dst,env)
        agg=agg/degree.clamp_min(1.0)[:,None,None]
        s=self.node_norm(s+self.cfg.mode_update_scale*self.node_up(torch.cat((s,agg),dim=-1)))
        if active_modes is not None:
            s=s*active_modes[None,:,None]
        # One impulse amplitude per mode and [constant, cos, sin] temporal basis.
        head=self.refine_head if is_refinement else self.initial_head
        coeff=head(e).view(-1,n_modes,3,2)
        if is_refinement:
            coeff=coeff*self.cfg.correction_scale
        mu=(mass[src]*mass[dst])/(mass[src]+mass[dst])
        freq_logit=self.frequency_head(e).squeeze(-1)+self.mode_frequency_logit[None,:]
        omega_fs=self.omega_min+(self.omega_max-self.omega_min)*torch.sigmoid(freq_logit)
        J=(coeff[...,0,None]*n[:,None,None,:]+
           coeff[...,1,None]*trans[:,None,None,:]) * (mu*self.cfg.velocity_scale*env)[:,None,None,None]
        if active_modes is not None:
            J=J*active_modes[None,:,None,None]
        return s, (src,dst,J,omega_fs)


def _phase_from_impulses(q0,v0,mass,h,impulses,s,chunk_size=8):
    """Analytic phase-space trajectory for a list of adaptive oscillatory impulse fields.

    The graph operations are evaluated only at the initial and candidate midpoint
    geometries. Each time query is an analytic, differentiable evaluation.
    """
    n=q0.shape[0]
    outq=[]; outv=[]
    for first in range(0,s.numel(),chunk_size):
        sq=s[first:first+chunk_size]
        v_add=q0.new_zeros((sq.numel(),n,3))
        q_add=torch.zeros_like(v_add)
        for src,dst,J,omega_fs in impulses:
            if src.numel()==0:
                continue
            P,Q=oscillatory_primitives(sq,omega_fs*h)
            v_edges=torch.einsum('tekb,ekbc->tec',P,J)
            q_edges=torch.einsum('tekb,ekbc->tec',Q,J)
            v_add.index_add_(1,src,v_edges)
            v_add.index_add_(1,dst,-v_edges)
            q_add.index_add_(1,src,q_edges)
            q_add.index_add_(1,dst,-q_edges)
        outq.append(q0[None,:,:]+h*sq[:,None,None]*v0[None,:,:]+h*q_add/mass[None,:,None])
        outv.append(v0[None,:,:]+v_add/mass[None,:,None])
    return torch.cat(outq,0),torch.cat(outv,0)


class CMIS(nn.Module):
    def __init__(self,cfg: Optional[CMISConfig]=None):
        super().__init__()
        self.cfg=cfg or CMISConfig()
        c=self.cfg
        self.encoder=ScalarEncoder(c)
        self.mode_project=nn.Linear(c.width,c.width)
        self.mode_embeddings=nn.Parameter(torch.randn(c.modes,c.width)*0.03)
        self.first_coupling=ModeCoupling(c.width)
        self.second_coupling=ModeCoupling(c.width)
        self.step=PairModeStep(c)

    def forward(self, q, v, z, mass, times_fs,
                graph_builder: Callable[[torch.Tensor],tuple[torch.Tensor,torch.Tensor]],
                return_modes: bool=False, active_modes: Optional[torch.Tensor]=None,
                graph_index: Optional[torch.Tensor]=None):
        """q/v [N,3], z/mass [N], time queries [T].
        graph_builder(q) -> (edge_index[2,E], edge_shift[E,3]);
        must return a FULL directed PBC neighbor graph with shifts in Angstrom.
        Graph topology may be non-differentiable; positions inside edge geometry remain differentiable.
        """
        if q.ndim!=2 or q.shape[-1]!=3 or v.shape!=q.shape:
            raise ValueError('q and v must be [N,3]')
        if mass.shape!=(len(q),) or torch.any(mass<=0):
            raise ValueError('positive mass required for each atom')
        t=torch.as_tensor(times_fs,device=q.device,dtype=q.dtype).reshape(-1)
        if active_modes is not None:
            active_modes=torch.as_tensor(active_modes,dtype=torch.bool,device=q.device)
            if active_modes.shape!=(self.cfg.modes,) or not bool(active_modes.any()):
                raise ValueError('active_modes must be [K] bool with at least one active mode')
        h=float(self.cfg.horizon_fs)
        if torch.any(t<0) or torch.any(t>h+1e-6):
            raise ValueError('times must be within [0,horizon_fs]')
        # 1) Galilean-invariant features; COM translation is handled exactly by q+h*s*v0.
        if graph_index is None:
            vcom=(mass[:,None]*v).sum(0)/mass.sum()
            vcom_per_atom=vcom.expand_as(v)
        else:
            gids=graph_index.long()
            if gids.shape!=(len(q),):
                raise ValueError('graph_index must be [N]')
            B=int(gids.max().item())+1
            masses=torch.zeros(B,device=q.device,dtype=q.dtype).index_add_(0,gids,mass)
            momenta=torch.zeros((B,3),device=q.device,dtype=q.dtype).index_add_(0,gids,mass[:,None]*v)
            vcom_per_atom=(momenta/masses.clamp_min(1e-12)[:,None])[gids]
        vrel=v-vcom_per_atom
        e0,shift0=graph_builder(q)
        node=self.encoder(q,vrel,z,mass,e0,shift0)
        modes=self.mode_project(node)[:,None,:]+self.mode_embeddings[None,:,:]
        modes,attn0=self.first_coupling(modes,active_modes)
        modes,(src0,dst0,J0,omega0)=self.step(modes,q,vrel,mass,e0,shift0,False,active_modes)
        # 2) Geometry-dependent correction: midpoint is predicted, never teacher-forced.
        qmid,vmid=_phase_from_impulses(q,v,mass,h,[(src0,dst0,J0,omega0)],q.new_tensor([0.5]))
        qmid,vmid=qmid[0],vmid[0]
        em,shiftm=graph_builder(qmid)
        modes,attn1=self.second_coupling(modes,active_modes)
        modes,(src1,dst1,J1,omega1)=self.step(modes,qmid,vmid-vcom_per_atom,mass,em,shiftm,True,active_modes)
        # 3) Dense time query contains no additional message-passing calls.
        qout,vout=_phase_from_impulses(q,v,mass,h,
            [(src0,dst0,J0,omega0),(src1,dst1,J1,omega1)],t/h)
        result={'positions':qout,'velocities':vout}
        if return_modes:
            result['mode_impulses_initial']=J0
            result['mode_impulses_refined']=J1
            result['mode_omega_fs_initial']=omega0
            result['mode_omega_fs_refined']=omega1
            result['mode_attention_first']=attn0
            result['mode_attention_second']=attn1
            result['mode_states']=modes
            result['mode_edges_initial']=torch.stack([src0,dst0])
            result['mode_edges_refined']=torch.stack([src1,dst1])
        return result


def small_cell_periodic_graph(q: torch.Tensor,cell:torch.Tensor,cutoff:float):
    """ONLY a tiny-cell test reference, O(N^2), minimal image, not for production.
    Cell rows are lattice vectors; cutoff must be < half shortest box height.
    """
    n=len(q)
    dr=q[None,:,:]-q[:,None,:]  # [dst,src,xyz]
    inv=torch.linalg.inv(cell)
    frac=dr@inv
    offsets=-torch.round(frac)
    mapped=dr+offsets@cell
    r=torch.linalg.vector_norm(mapped,dim=-1)
    mask=(r<cutoff)&(~torch.eye(n,dtype=torch.bool,device=q.device))
    dst,src=torch.where(mask)
    edges=torch.stack([src,dst])
    shifts=offsets[dst,src]@cell
    return edges,shifts
