"""Independent motion tokens -> recurrent coupling -> continuous segment.

Uses IBM TrajCast's encoder/readout/conservation unchanged (Apache-2.0;
see ../trajcast_based/LICENSE.upstream). All parameters are trained from scratch.
No force, potential, frequency or integration step is evaluated here.
"""
import math
from collections import OrderedDict
import torch
from torch import nn
from e3nn import o3
from trajcast.model.models import EfficientTrajCastModel


def mlp(a,b,c):
    return nn.Sequential(nn.Linear(a,b),nn.SiLU(),nn.Linear(b,c))


def center_by_mass(x, data):
    m=data.atomic_masses.reshape(-1)
    batch=data.batch if getattr(data,'batch',None) is not None else torch.zeros(len(m),dtype=torch.long,device=m.device)
    ng=int(batch.max())+1
    shape=(len(m),)+(1,)*(x.ndim-1)
    total=m.new_zeros(ng).index_add(0,batch,m)
    sums=x.new_zeros((ng,)+x.shape[1:]).index_add(0,batch,x*m.reshape(shape))
    return x-(sums/total.reshape((ng,)+(1,)*(x.ndim-1)))[batch]


def bubble_basis(u, count):
    """Endpoint-null response and derivative on [0,1]."""
    x=2*u-1
    p=[torch.ones_like(u)]; d=[torch.zeros_like(u)]
    if count>1: p.append(x);d.append(torch.ones_like(u)*2)
    for k in range(1,count-1):
        p.append(((2*k+1)*x*p[k]-k*p[k-1])/(k+1))
        d.append(((2*k+1)*(2*p[k]+x*d[k])-k*d[k-1])/(k+1))
    env=u.square()*(1-u).square()
    denv=2*u*(1-u)*(1-2*u)
    return torch.stack(p,-1)*env[:,None],torch.stack(d,-1)*env[:,None]+torch.stack(p,-1)*denv[:,None]


def hermite(u, h, displacement, v0, v1):
    t=u[:,None,None]
    q=(3*t*t-2*t**3)*displacement[None]+h*(t**3-2*t*t+t)*v0[None]+h*(t**3-t*t)*v1[None]
    v=(6*t-6*t*t)/h*displacement[None]+(3*t*t-4*t+1)*v0[None]+(3*t*t-2*t)*v1[None]
    return q,v


class MotionAdapter(nn.Module):
    def __init__(self, irreps, count=4, basis=6, updates=4):
        super().__init__()
        self.modes=count;self.basis=basis;self.updates=updates
        self.irreps=o3.Irreps('16x0o+16x0e+16x1o+16x1e+8x2o+8x2e')
        self.inputs=nn.ModuleList([o3.Linear(irreps,self.irreps) for _ in range(count)])
        self.embed=nn.Parameter(.02*torch.randn(count,32))
        self.norm=o3.Norm(self.irreps,squared=True)
        n=self.irreps.num_irreps
        self.state_gate=mlp(n+32+1,64,n)
        self.state_update=o3.Linear(self.irreps,self.irreps)
        self.query=mlp(n+32,64,32);self.key=mlp(n+32,64,32)
        self.exchange=o3.Linear(self.irreps,self.irreps)
        self.exchange_gate=mlp(n+32,64,n)
        self.expand=o3.Linear(self.irreps,irreps)
        self.temporal=nn.ModuleList([o3.Linear(self.irreps,f'{basis}x1o') for _ in range(count)])
        # Identity at insertion, without loading ANY pretrained parameters.
        nn.init.zeros_(self.expand.weight)
        for head in self.temporal: nn.init.zeros_(head.weight)
        repeat=[]
        for mul,ir in self.irreps:
            for _ in range(mul): repeat.extend([len(set(repeat))]*ir.dim)
        # Map each irrep copy's invariant gate to its magnetic components.
        self.register_buffer('component_gate',torch.tensor(repeat,dtype=torch.long))
        self.register_buffer('self_mask',torch.eye(count,dtype=torch.bool))

    def invariants(self,z):
        return torch.cat([torch.log1p(self.norm(z)),self.embed[None].expand(len(z),-1,-1)],-1)

    def gate(self,x,g): return x*g[...,self.component_gate]

    def forward(self,data):
        original=data.node_features
        z=torch.stack([layer(original) for layer in self.inputs],1)
        attention=None
        for step in range(self.updates):
            inv=self.invariants(z)
            phase=z.new_full((*inv.shape[:-1],1),(step+1)/self.updates)
            g=torch.sigmoid(self.state_gate(torch.cat([inv,phase],-1)))
            z=z+self.gate(self.state_update(z),g)/self.updates
            inv=self.invariants(z)
            logits=torch.einsum('nmd,nkd->nmk',self.query(inv),self.key(inv))/math.sqrt(32)
            attention=logits.masked_fill(self.self_mask[None],-torch.inf).softmax(-1)
            received=torch.einsum('nmk,nkd->nmd',attention,self.exchange(z))
            z=z+self.gate(received,torch.sigmoid(self.exchange_gate(inv)))/self.updates
        # Additive, not competitive mode reconstruction. No softmax mode decoder.
        additions=self.expand(z)/math.sqrt(self.modes)
        data.node_features=original+additions.sum(1)
        data.mode_feature_contributions=additions
        data.mode_attention=attention
        coefficients=torch.stack([head(z[:,k]).reshape(len(z),self.basis,3)
                                  for k,head in enumerate(self.temporal)],1)/math.sqrt(self.modes)
        data.mode_coefficients=center_by_mass(coefficients,data)
        return data


class SMDP(EfficientTrajCastModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.horizon=float(self.config.get('horizon_fs',30.))
        self.basis=int(self.config.get('segment_basis',6))
        last=f'mp_layer_{int(self.num_mp_layers)-1}'
        irreps=self._int_out._modules[last].irreps_out['node_features']
        layers=OrderedDict()
        for name,layer in self._int_out.named_children():
            layers[name]=layer
            if name==last:
                layers['MotionModes']=MotionAdapter(irreps,4,self.basis,4)
        self._int_out=nn.Sequential(layers)

    def forward(self,data):
        data.segment_initial_velocities=data.velocities
        data=super().forward(data)
        # Original ConservationLayer preserves incoming P when net_lin_mom=[].
        # Also constrain COM displacement to h*v_COM for nonzero incoming P.
        disp=data.target[:,:3]*self.rms_targets[0]
        v0=data.segment_initial_velocities
        disp=center_by_mass(disp,data)+self.horizon*(v0-center_by_mass(v0,data))
        data.target=torch.cat([disp/self.rms_targets[0],data.target[:,3:]],-1)
        return data

    def query(self,data,times_fs):
        t=torch.as_tensor(times_fs,device=data.pos.device,dtype=data.pos.dtype).reshape(-1)
        if not bool(torch.isfinite(t).all()) or bool(((t<0)|(t>self.horizon)).any()):
            raise ValueError('Query outside the learned segment')
        q,v=hermite(t/self.horizon,self.horizon,data.target[:,:3]*self.rms_targets[0],
                    data.segment_initial_velocities,data.target[:,3:]*self.rms_targets[1])
        basis,derivative=bubble_basis(t/self.horizon,self.basis)
        coeff=data.mode_coefficients*self.rms_targets[0]
        mode_q=torch.einsum('tk,nmkc->tnmc',basis,coeff)
        mode_v=torch.einsum('tk,nmkc->tnmc',derivative/self.horizon,coeff)
        return dict(displacements=q+mode_q.sum(2),velocities=v+mode_v.sum(2),
                    mode_displacements=mode_q,mode_velocities=mode_v)
