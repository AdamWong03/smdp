"""CMIS checkpoint inference, sharing the established physical rollout harness."""
from pathlib import Path

import torch

from .cmis_model import CMIS,CMISConfig


def load_cmis(checkpoint,device='cuda'):
    saved=torch.load(Path(checkpoint),map_location='cpu',weights_only=False)
    model=CMIS(CMISConfig(**saved['meta']['config']))
    model.load_state_dict(saved['state_dict'],strict=True);model.to(device).eval()
    # Historical shared rollout metadata stores zero-based epoch indices.
    metadata=dict(saved,epoch=saved['epoch']-1)
    return model,metadata


@torch.inference_mode()
def segment(model,positions,velocities,cell,numbers,masses,device,builder,times):
    q=torch.as_tensor(positions,device=device,dtype=torch.float32)
    v=torch.as_tensor(velocities,device=device,dtype=torch.float32)
    z=torch.as_tensor(numbers,device=device,dtype=torch.long)
    m=torch.as_tensor(masses,device=device,dtype=torch.float32)
    def graph(value):
        edges,shifts=builder.one(value.detach().cpu().numpy(),cell)
        return (torch.as_tensor(edges,device=device,dtype=torch.long),
                torch.as_tensor(shifts,device=device,dtype=value.dtype))
    return model(q,v,z,m,torch.as_tensor(times,device=device,dtype=q.dtype),graph)


def step(model,positions,velocities,cell,numbers,masses,device='cuda',builder=None):
    result=segment(model,positions,velocities,cell,numbers,masses,device,builder,[model.cfg.horizon_fs])
    return result['positions'][0],result['velocities'][0]
