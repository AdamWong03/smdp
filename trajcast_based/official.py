"""Original TrajCast evaluation and Forecast, without model/thermostat changes."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import ase.io
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from trajcast_based.acquire import COMMIT, ROOT, RUN, UPSTREAM, sha


def load():
    torch.set_default_dtype(torch.float64)
    from trajcast.model.models import EfficientTrajCastModel
    assert subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip() == COMMIT
    cfg = RUN/'official/models/quartz/config_e3nn.yaml'
    weights = RUN/'official/models/quartz/state_dict_e3nn.pt'
    model = EfficientTrajCastModel.build_from_yaml(str(cfg))
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return model, cfg, weights


def evaluate(device):
    from trajcast.data.dataset import AtomicGraphDataset
    model, cfg, weights = load()
    from trajcast.utils.misc import GLOBAL_DEVICE
    GLOBAL_DEVICE.device = torch.device(device)
    model.to(device).eval()
    root = RUN/'official/datasets/quartz'
    data = AtomicGraphDataset(root=str(root), name='official_quartz_test', cutoff_radius=4.5,
                              files=['test.extxyz'], atom_type_mapper={8:0,14:1})
    assert len(data) == 1000
    sums = np.zeros((2, 3))
    started = time.perf_counter()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=2, shuffle=False):
            batch = batch.to(device)
            refs = [batch.displacements.clone(), batch.update_velocities.clone()]
            predicted = model(batch).target
            for k in range(2):
                pred = predicted[:,k*3:(k+1)*3] * model.rms_targets[k]
                err = pred-refs[k]
                sums[k] += [err.abs().sum().item(), err.square().sum().item(), err.numel()]
    result = dict(upstream_commit=COMMIT, checkpoint_sha256=sha(weights), test_samples=len(data),
        parameters=sum(p.numel() for p in model.parameters()), runtime_s=time.perf_counter()-started,
        architecture=str(cfg), rms_targets=model.rms_targets.cpu().tolist(),
        displacement_MAE_A=float(sums[0,0]/sums[0,2]), displacement_RMSE_A=float(np.sqrt(sums[0,1]/sums[0,2])),
        velocity_MAE_A_fs=float(sums[1,0]/sums[1,2]), velocity_RMSE_A_fs=float(np.sqrt(sums[1,1]/sums[1,2])),
        paper_table_S1={'displacement_MAE_A':0.0014124,'velocity_MAE_A_fs':0.0001137},
        future_adapter_irreps=str(model._int_out.mp_layer_3.irreps_out['node_features']))
    (RUN/'official_test_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def forecast(device, production_steps, warmup_steps, tag):
    from trajcast.model.forecast import Forecast
    model, cfg, weights = load()
    out = RUN/tag
    out.mkdir(parents=True, exist_ok=True)
    xyz=out/'trajectory.extxyz'
    if xyz.exists(): raise FileExistsError(xyz)
    atoms=ase.io.read(RUN/'official/datasets/quartz/test.extxyz',index=0)
    for k in ['displacements','update_velocities']: atoms.arrays.pop(k,None)
    atoms.info.pop('timestep',None)
    protocol=dict(units='real',run=production_steps+warmup_steps,temperature=300.,extra_dof=3,
        timestep=30.,configuration=atoms,model_type='EfficientTrajCast',model=model,
        type_mapper={8:0,14:1},thermostat={'Tdamp':75.},
        velocities={'temperature':300.,'linear':True,'angular':True,'distribution':'gaussian'},
        write={'filename':str(xyz),'every':1,'save_velocities':True,'format':'extxyz'},
        device=torch.device(device),seed=20261027,
        set_momenta={'linear':torch.zeros(3,device=device),'angular':torch.tensor([],device=device)})
    manifest=dict(upstream_commit=COMMIT,checkpoint_sha256=sha(weights),device=device,
        hardware=torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU',
        production_steps=production_steps,warmup_steps=warmup_steps,timestep_fs=30.,
        thermostat_tau_fs=75.,temperature_K=300.,dof=483,seed=20261027,
        initial_state='official test.extxyz frame 0; official Gaussian velocity initializer; zero linear momentum',
        angular_momentum_initialization=True,
        angular_note='Paper Gaussian initializer zeros both momenta; periodic rollout conserves linear momentum only, with 3N-3 DOF.',
        cell_A=atoms.cell.array.tolist(),masses_u=atoms.get_masses().tolist(),
        trajectory_completed=False,physical_analysis_completed=False,
        full_paper_duration=(production_steps==500000 and warmup_steps==50000))
    (out/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.perf_counter()
    engine=Forecast(protocol)
    engine.generate_trajectory()
    manifest.update(trajectory_completed=True,wall_seconds=time.perf_counter()-start)
    manifest['ns_per_day_including_initialization_and_extxyz_io']=(production_steps+warmup_steps)*30e-6/manifest['wall_seconds']*86400
    (out/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['evaluate','forecast'])
    p.add_argument('--device',default='cuda')
    p.add_argument('--production-steps',type=int,default=3334)
    p.add_argument('--warmup-steps',type=int,default=334)
    p.add_argument('--tag',default='official30_100ps')
    a=p.parse_args()
    if a.task=='evaluate': evaluate(a.device)
    else: forecast(a.device,a.production_steps,a.warmup_steps,a.tag)
