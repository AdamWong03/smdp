"""Frozen-checkpoint test errors, mode diagnostics and end-to-end timings."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.utils.misc import GLOBAL_DEVICE
from trajcast_smdp.data import RUN,dataset,fresh_graph
from trajcast_smdp.model import SMDP,center_by_mass
from trajcast_smdp.train import validate


def load(run,device='cuda'):
    torch.set_default_dtype(torch.float64);GLOBAL_DEVICE.device=torch.device(device)
    saved=torch.load(Path(run)/'best.pt',weights_only=False,map_location='cpu')
    weights=saved['model']
    model=SMDP(config=saved['config']['model'],rms_targets=weights['rms_targets'].tolist())
    model.load_state_dict(weights,strict=True);model.to(device).eval()
    return model,saved


@torch.no_grad()
def main(run):
    run=Path(run);model,saved=load(run);test=dataset(run,'test')
    result=validate(model,test,'cuda',rollout_samples=len(test))
    sums=np.zeros((30,4));count=0;att=[];features=[];coeff=[];signed=[];endpoint_effects=[]
    for batch in DataLoader(test,batch_size=2,shuffle=False):
        batch=batch.to('cuda');out=model(batch.clone());p=model.query(out,list(range(1,31)))
        q=batch.trajectory_q[:,1:31].transpose(0,1);v=batch.trajectory_v[:,1:31].transpose(0,1)
        for j,e in enumerate([p['displacements']-q,p['velocities']-v]):
            sums[:,2*j]+=e.abs().mean((1,2)).cpu().numpy()*batch.num_graphs
            sums[:,2*j+1]+=e.square().mean((1,2)).cpu().numpy()*batch.num_graphs
        count+=batch.num_graphs
        a=out.mode_feature_contributions
        att.append(out.mode_attention.mean(0).cpu().numpy())
        features.append(a.square().mean((0,2)).sqrt().cpu().numpy())
        coeff.append(out.mode_coefficients.square().mean((0,2,3)).sqrt().cpu().numpy())
        total=a.sum(1);signed.append((a*total[:,None]).sum((0,2)).cpu().numpy()/(float(total.square().sum())+1e-30))
        # Exact endpoint change on removing one final additive mode output,
        # holding preceding coupled evolution fixed (the readout is linear).
        pieces=[]
        for k in range(4):
            component=AtomicGraph(node_features=a[:,k])
            component=model._int_out.LinearCompression(component)
            component=model._int_out.LinearReadOut(component)
            physical=torch.cat([component.target[:,:3]*model.rms_targets[0],component.target[:,3:]*model.rms_targets[1]],-1)
            physical=center_by_mass(physical,out)
            pieces.append([float(physical[:,:3].square().mean().sqrt()),float(physical[:,3:].square().mean().sqrt())])
        endpoint_effects.append(pieces)
    pertime=[]
    for i in range(30):
        a=sums[i]/count
        pertime.append(dict(time_fs=i+1,position_MAE_A=float(a[0]),position_RMSE_A=float(a[1]**.5),
                            velocity_MAE_A_fs=float(a[2]),velocity_RMSE_A_fs=float(a[3]**.5)))
    # Timings use the identical model/GPU. Dense queries do not re-encode.
    g=test[0].to('cuda');g.batch=torch.zeros(g.num_nodes,dtype=torch.long,device='cuda')
    timing={}
    for label,times in [('endpoint',[30]),('dense_1fs',list(range(1,31)))]:
        for _ in range(5): model.query(model(g.clone()),times)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100): model.query(model(g.clone()),times)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        timing[label]=dict(model_query_seconds_per_step=elapsed/100,model_ns_day=.00003/(elapsed/100)*86400)
    out=model(g.clone());torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(1000): model.query(out,list(range(1,31)))
    torch.cuda.synchronize();timing['cached_dense_query_s']=(time.perf_counter()-start)/1000
    start=time.perf_counter()
    for _ in range(100): g.update_edge_index()
    torch.cuda.synchronize();timing['neighbor_update_seconds']=(time.perf_counter()-start)/100
    result.update(checkpoint=str(run/'best.pt'),selected_validation_record=saved['record'],
        errors_by_time=pertime,mode_diagnostics=dict(mean_off_diagonal_attention=np.mean(att,0).tolist(),
        mean_feature_RMS=np.mean(features,0).tolist(),mean_temporal_coefficient_RMS=np.mean(coeff,0).tolist(),
        signed_feature_projection=np.mean(signed,0).tolist(),
        endpoint_change_RMS_when_final_mode_output_removed_qA_vAfs=np.mean(endpoint_effects,0).tolist(),
        interpretation='Latent feature contributions, not identification of physical phonon modes'),
        timing=timing,hardware=torch.cuda.get_device_name(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        parameters=sum(p.numel() for p in model.parameters()))
    (run/'test_metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);a=p.parse_args();main(a.run)
