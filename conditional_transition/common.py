import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from relational_inference import vesin_graph
from periodic import PeriodicGraphBuilder
from train_relational import read_graphs

PROJECT=Path('/data1/wyd/research/foundation_mlip_adaptation')
ROOT=PROJECT/'experiments/conditional_transition_20261109'
DATA=PROJECT/'data/smdp_reference_md_v1'
ARMS=['original','two_step','extra_reference','autonomous']


def checkpoint(system):
    tag='quartz30' if system=='quartz' else 'water5'
    return PROJECT/'experiments/relational_transition'/f'{tag}_20261103_r3_opt/best.pt'


def normalize(g):
    keys=['pos','velocities','atomic_numbers','atomic_masses','edge_index','shifts',
          'displacements','update_velocities','timestep']
    data=Data(**{k:getattr(g,k).clone() for k in keys})
    data.cell=g.cell.reshape(1,3,3).clone()
    data.pbc=torch.ones(1,3,dtype=torch.bool)
    return data


def originals(cfg,split):
    return [normalize(g) for g in read_graphs(Path(cfg['cache']),split,cfg)]


def extra_graphs(files,model):
    import h5py
    graphs=[]
    builder=PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    for path in files:
        with h5py.File(path) as f:
            if not f.attrs.get('completed',False):raise RuntimeError(f'Incomplete label pool {path}')
            for i in range(len(f['q'])):
                q,v=f['q'][i],f['v'][i]
                graph=vesin_graph(q,v,f['cell'][i],f['z'][:],f['mass'][:],model,builder,'cpu')
                graph.displacements=torch.from_numpy((f['next_q'][i]-q).astype(np.float32))
                graph.update_velocities=torch.from_numpy(f['next_v'][i].astype(np.float32))
                graph.sample_id=torch.tensor([i])
                graphs.append(graph)
    if not graphs:raise RuntimeError('No valid extra labels; preserve failure instead of changing budget')
    return graphs


def transitions(system,split='train'):
    h=30 if system=='quartz' else 5
    return DATA/system/'transitions'/f'{h}fs/{split}.h5'


def train_sources(system):
    import h5py
    with h5py.File(transitions(system)) as f:
        return [p.decode() if isinstance(p,bytes) else str(p) for p in f['source_paths'][:]]


def seed_all(seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
