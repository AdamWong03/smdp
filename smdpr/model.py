"""SMDP-R: bounded modal conditioning inside existing geometric messages.

There is one unchanged R3-Opt backbone. Invariant persistent mode states use
the same edges and messages at each of its four blocks. No new geometry graph,
force calls or numeric integrator are added. Query losses cannot update it.
"""
import math

import torch
from torch import nn

from relational_transition import RelationalTransition, mass_mean, mix_vectors
from dsrt_primitives import mlp


class MotionModeState(nn.Module):
    def __init__(self, scalar_channels=192, modes=3, dimension=32):
        super().__init__(); self.modes=modes; self.dimension=dimension
        self.project=nn.Linear(scalar_channels, modes*dimension)
        self.embeddings=nn.Parameter(torch.randn(modes,dimension)*.03)

    def forward(self, scalar):
        return torch.nn.functional.silu(self.project(scalar).reshape(-1,self.modes,self.dimension)
                                       +self.embeddings)


class CrossModeCoupling(nn.Module):
    def __init__(self, dimension, rank=32):
        super().__init__()
        self.norm=nn.LayerNorm(dimension)
        self.q=nn.Linear(dimension,rank,bias=False)
        self.k=nn.Linear(dimension,rank,bias=False)
        self.v=nn.Linear(dimension,rank,bias=False)
        self.out=nn.Linear(rank,dimension)
        self.gate=nn.Linear(2*dimension,1)

    def forward(self, modes, active=None):
        x=self.norm(modes)
        scores=torch.einsum('nkr,njr->nkj',self.q(x),self.k(x))/math.sqrt(self.q.out_features)
        if active is not None:
            scores=scores.masked_fill(~active[None,None,:],-torch.inf)
        attention=scores.softmax(-1)
        context=self.out(torch.einsum('nkj,njr->nkr',attention,self.v(x)))
        gate=torch.sigmoid(self.gate(torch.cat([x,context],-1)))
        updated=modes+.2*gate*context
        if active is not None:
            updated=updated*active[None,:,None]
        return updated,attention


class ModalRelationalUpdate(nn.Module):
    def __init__(self, cfg, relation_dim, dimension=32, rank=32, modulation=.1):
        super().__init__(); self.modulation=modulation
        self.edge=nn.Linear(cfg.edge_channels+relation_dim,rank)
        self.message=nn.Linear(cfg.scalar_channels+5*cfg.vector_channels,rank)
        self.sender=nn.Linear(dimension,rank,bias=False)
        self.receiver=nn.Linear(dimension,rank,bias=False)
        self.affinity=nn.Linear(rank,1)
        self.update=mlp(dimension+rank,rank,dimension)
        self.norm=nn.LayerNorm(dimension)
        self.coupling=CrossModeCoupling(dimension,rank)
        self.read=nn.Linear(dimension,rank)
        self.condition=nn.Linear(rank,cfg.scalar_channels+5*cfg.vector_channels)
        self.capture=False; self.last_record=None

    def forward(self, modes, e, ms, coefficients, edges, relation, active):
        src,dst=edges;invariants,_,weight=relation
        context=self.edge(torch.cat([e,invariants],-1))
        sender=self.sender(modes)
        affinity=torch.sigmoid(self.affinity(torch.tanh(
            sender[src]+self.receiver(modes)[dst]+context[:,None])).squeeze(-1))
        if active is not None:
            affinity=affinity*active[None,:]
        transmitted=self.message(torch.cat([ms,coefficients.flatten(1)],-1))
        payload=transmitted[:,None,:]+sender[src]
        weights=weight[:,None]*affinity
        aggregate=modes.new_zeros(len(modes),modes.shape[1],context.shape[-1]).index_add(
            0,dst,weights[:,:,None]*payload)
        degree=modes.new_zeros(len(modes),modes.shape[1]).index_add(0,dst,weights).clamp_min(1.)
        aggregate=aggregate/degree[:,:,None]
        updated=self.norm(modes+.2*self.update(torch.cat([modes,aggregate],-1)))
        updated,attention=self.coupling(updated,active)
        # Only invariant node coefficients scale existing equivariant messages.
        # No K copies of the 192/48/96 graph processor are evaluated.
        read=self.read(updated)
        if active is None:
            read=read.mean(1)
        else:
            read=(read*active[None,:,None]).sum(1)/active.sum()
        node_condition=self.condition(read)
        condition=torch.tanh(.5*(node_condition[src]+node_condition[dst]))
        scale=1.+self.modulation*condition
        ns=ms.shape[-1]
        if self.capture:
            self.last_record=dict(affinity=affinity,attention=attention,edges=edges,
                modes=updated,scalar_modulation_RMS=(scale[:,:ns]-1).square().mean().sqrt(),
                vector_coefficient_modulation_RMS=(scale[:,ns:]-1).square().mean().sqrt())
        return updated,ms*scale[:,:ns],coefficients*scale[:,ns:].reshape_as(coefficients)


class SegmentQueryHead(nn.Module):
    """Detached, equivariant endpoint-anchored q/v query; not an integrator.

    Hermite interpolation is a boundary-respecting base. Independent learned
    interior q/v corrections need not satisfy qdot=v and are evaluated as such.
    """
    def __init__(self,cfg,total_mode_channels=96,hidden=48):
        super().__init__();self.cfg=cfg;self.dv=cfg.vector_channels
        self.context=nn.Linear(cfg.scalar_channels+total_mode_channels,hidden)
        self.time=mlp(hidden+3,hidden,2*cfg.vector_channels)
        self.vector=nn.Linear(cfg.vector_channels,cfg.vector_channels,bias=False)

    def forward(self,data,endpoint,times,scales):
        # Every source is detached, including endpoints and vector features.
        s=endpoint['context']['scalar'].detach();v=endpoint['context']['vector'].detach()
        modes=endpoint['context']['modes'].detach()
        q0=data.pos.detach();v0=data.velocities.detach()
        dq=endpoint['displacements'].detach();v1=endpoint['velocities'].detach()
        h=self.cfg.horizon_fs;t=torch.as_tensor(times,device=s.device,dtype=s.dtype).reshape(-1)
        if bool(((t<0)|(t>h)).any()):raise ValueError('Query outside model horizon')
        x=(t/h)[:,None,None];a=3*x*x-2*x*x*x;b=x*x*x-2*x*x+x;c=x*x*x-x*x
        base_q=q0[None]+a*dq[None]+h*b*v0[None]+h*c*v1[None]
        base_v=((6*x-6*x*x)/h)*dq[None]+(3*x*x-4*x+1)*v0[None]+(3*x*x-2*x)*v1[None]
        context=torch.nn.functional.silu(self.context(torch.cat([s,modes.flatten(1)],-1)))
        time_features=torch.stack([t/h,(t/h)**2,(t/h)**3],-1)
        hidden=context[None].expand(len(t),-1,-1)
        coeff=self.time(torch.cat([hidden,time_features[:,None].expand(-1,len(s),-1)],-1))
        coeff=coeff.reshape(len(t),len(s),2,self.dv)
        delta=torch.einsum('tnkc,ncd->tnkd',coeff,mix_vectors(self.vector,v))/self.dv**.5
        batch=getattr(data,'batch',None)
        if batch is None:batch=torch.zeros(len(s),device=s.device,dtype=torch.long)
        mass=data.atomic_masses.reshape(-1)
        groups=getattr(data,'num_graphs',1)
        total=mass.new_zeros(groups).index_add(0,batch,mass)
        sums=delta.new_zeros(len(t),groups,2,3).index_add(
            1,batch,delta*mass[None,:,None,None])
        delta=delta-(sums/total[None,:,None,None])[:,batch]
        dq_extra,dv_extra=delta[:,:,0],delta[:,:,1]
        envelope=x*(1-x)
        q=base_q+envelope*scales[0].detach()*dq_extra
        velocity=base_v+envelope*scales[1].detach()*dv_extra
        # Exact endpoint identity, including floating-point output convention.
        q=torch.where((t==0)[:,None,None],q0[None],q)
        q=torch.where((t==h)[:,None,None],endpoint['positions'].detach()[None],q)
        velocity=torch.where((t==0)[:,None,None],v0[None],velocity)
        velocity=torch.where((t==h)[:,None,None],v1[None],velocity)
        return dict(positions=q,velocities=velocity)


class SMDPR(RelationalTransition):
    def __init__(self,config=None,rms_targets=(.07,.004),modes=3,mode_dimension=32,
                 modulation=.1,query_head=True):
        # Construct backbone FIRST so a common RNG seed gives identical original
        # backbone initialization to the unmodified R3-Opt model.
        super().__init__(config,rms_targets)
        if self.cfg.block_type!='geometry_gated':raise ValueError('SMDP-R requires original geometry_gated')
        self.mode_count=modes;self.mode_dimension=mode_dimension;self.modes_enabled=True
        self.motion_state=MotionModeState(self.cfg.scalar_channels,modes,mode_dimension)
        self.modal_blocks=nn.ModuleList([ModalRelationalUpdate(self.cfg,self.relations.scalar_size,
            mode_dimension,modulation=modulation) for _ in range(len(self.proposal_blocks)+len(self.refinement_blocks))])
        self.query_head=SegmentQueryHead(self.cfg,modes*mode_dimension) if query_head else None
        self.register_buffer('active_modes',torch.ones(modes,dtype=torch.bool),persistent=False)
        self._use_ablation_mask=False

    def set_active_modes(self,mask):
        mask=torch.as_tensor(mask,device=self.active_modes.device,dtype=torch.bool)
        if mask.shape!=self.active_modes.shape or not bool(mask.any()):
            raise ValueError('At least one mode must remain active')
        self.active_modes.copy_(mask)
        self._use_ablation_mask=not bool(mask.all())

    def initialize_motion(self,s):
        if not self.modes_enabled:return None
        motion=self.motion_state(s)
        return motion*self.active_modes[None,:,None] if self._use_ablation_mask else motion

    def propagate_block(self,index,block,s,v,e,edges,relation,motion):
        if not self.modes_enabled:
            return super().propagate_block(index,block,s,v,e,edges,relation,motion)
        updated=motion
        def condition(new_edge,ms,coefficients):
            nonlocal updated
            updated,ms,coefficients=self.modal_blocks[index](motion,new_edge,ms,coefficients,
                edges,relation,self.active_modes if self._use_ablation_mask else None)
            return ms,coefficients
        s,v,e=block(s,v,e,edges,relation,message_transform=condition)
        return s,v,e,updated

    def query(self,data,output,times):
        if self.query_head is None:raise ValueError('No query head')
        return self.query_head(data,output,times,self.rms_targets)
