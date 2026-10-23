"""Frozen-checkpoint validation comparison; no optimizer and no MD trajectory generation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from evaluate_quartz import load_model
from reference_data import QuartzSegments
from periodic import PeriodicGraphBuilder,predict_batch
from train_quartz import batch_to,diagnostics
from physics.units import KINETIC_TO_EV
from analysis_tasks import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--r1',type=Path);p.add_argument('--r2',type=Path)
    p.add_argument('--checkpoints',type=Path,nargs='+');p.add_argument('--labels',nargs='+')
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if a.checkpoints:
        if not a.labels or len(a.labels)!=len(a.checkpoints) or len(set(a.labels))!=len(a.labels):
            p.error('Provide one unique --labels entry per checkpoint')
        models=dict(zip(a.labels,a.checkpoints))
    else:
        if not a.r1 or not a.r2:p.error('Provide --checkpoints/--labels or --r1/--r2')
        models={'R1_final':a.r1/'best_rollout.pt',**{k:a.r2/f'best_{k}.pt' for k in ['segment','rollout4','rollout8','rollout16']}}
    data=QuartzSegments(a.data_root,'val',a.output/'cache')
    lookup={int(t):i for i,t in enumerate(data.offsets)}
    for label,path in models.items():
        if (a.output/(label+'.json')).exists(): continue
        started=time.perf_counter();model,cfg=load_model(path,'cuda');builder=PeriodicGraphBuilder(model.config.cutoff_angstrom)
        records=[];deletions=[];kin=[];refkin=[];join=[];refjoin=[];diag={};single_count=0;mode_cos=[]
        with torch.inference_mode():
            for batch in DataLoader(data,batch_size=8,shuffle=False):
                b=batch_to(batch,'cuda');q,v=b['q'],b['v'];size,n=len(q),q.shape[2]
                out=predict_batch(model,q[:,0],v[:,0],b['z'],b['mass'],b['cell'],list(range(17)),builder)
                pq,pv=out['positions'],out['velocities']
                kin.extend((((pq[2:]-pq[:-2])/2-pv[1:-1]).reshape(15,size,n,3).square().mean((0,2,3))).cpu().tolist())
                refkin.extend((((q[:,2:17]-q[:,:15])/2-v[:,1:16]).square().mean((1,2,3))).cpu().tolist())
                for key,value in diagnostics(out).items(): diag[key]=diag.get(key,0.)+np.array(value)*size
                # Decoder-output deletion, leaving upstream dynamics/context fixed.
                # This measures reliance on each additive term, not physical uniqueness.
                for k in range(model.config.modes):
                    dq=out['mode_delta_q'][-1,:,k].reshape(size,n,3);dv=out['mode_delta_v'][-1,:,k].reshape(size,n,3)
                    erq=pq[-1].reshape(size,n,3)-dq-q[:,16];erv=pv[-1].reshape(size,n,3)-dv-v[:,16]
                    for i in range(size): deletions.append(dict(sample=single_count+i,mode=k+1,q_mse=float(erq[i].square().mean()),
                        v_mse=float(erv[i].square().mean()),removed_q_rms=float(dq[i].square().mean().sqrt()),removed_v_rms=float(dv[i].square().mean().sqrt())))
                components=out['mode_delta_v'][-1].permute(1,0,2).flatten(1)
                norm=components.norm(dim=1);cos=(components@components.T)/(norm[:,None]*norm[None,:]).clamp_min(1e-20)
                mode_cos.append(cos.cpu().numpy()*size)
                current_q,current_v=q[:,0],v[:,0];left=None
                for step in range(1,17):
                    if step>1: out=predict_batch(model,current_q,current_v,b['z'],b['mass'],b['cell'],[1.,15.,16.],builder)
                    qn=out['positions'][-1].reshape(size,n,3);vn=out['velocities'][-1].reshape(size,n,3)
                    target=lookup[step*16];prev=lookup[(step-1)*16]
                    qerr=(qn-q[:,target]).square().mean((1,2));verr=(vn-v[:,target]).square().mean((1,2))
                    route=out['operator_routing'][:,-1,...,0].reshape(2,size,n,4).mean((0,2))
                    weight=.5*KINETIC_TO_EV*b['mass'][:,:,None]
                    pred_ke=(weight*vn**2).sum((1,2));before=(weight*current_v**2).sum((1,2))
                    ref_ke=(weight*v[:,target]**2).sum((1,2));ref_before=(weight*v[:,prev]**2).sum((1,2))
                    for i in range(size):
                        if step>1 and not bool(b['rollout_valid'][i]):continue
                        row=dict(sample=single_count+i,step=step,q_mse=float(qerr[i]),v_mse=float(verr[i]),
                            predicted_KE_eV=float(pred_ke[i]),reference_KE_eV=float(ref_ke[i]),
                            initial_KE_eV=float((weight[i]*v[i,0]**2).sum()),
                            delta_KE_error_eV=float(pred_ke[i]-before[i]-ref_ke[i]+ref_before[i]))
                        for k in range(4):row[f'recurrent_{k+1}']=float(route[i,k])
                        for z in [8,14]:
                            mask=b['z'][i]==z
                            row[f'predicted_KE_Z{z}']=float((weight[i,mask]*vn[i,mask]**2).sum())
                            row[f'reference_KE_Z{z}']=float((weight[i,mask]*v[i,target,mask]**2).sum())
                            row[f'velocity_mse_Z{z}']=float((vn[i,mask]-v[i,target,mask]).square().mean())
                            initial=v[i,0,mask]
                            scale=initial.square().mean().clamp_min(1e-20)
                            row[f'predicted_vacf_Z{z}']=float((vn[i,mask]*initial).mean()/scale)
                            row[f'reference_vacf_Z{z}']=float((v[i,target,mask]*initial).mean()/scale)
                        records.append(row)
                    if step==2:
                        jump=out['positions'][0].reshape(size,n,3)-current_q-left
                        reference=q[:,lookup[17]]-2*q[:,16]+q[:,15]
                        valid=b['rollout_valid'];join.extend(jump[valid].square().mean((1,2)).cpu().tolist())
                        refjoin.extend(reference[valid].square().mean((1,2)).cpu().tolist())
                    left=(out['positions'][-1]-out['positions'][-2]).reshape(size,n,3)
                    current_q,current_v=qn,vn
                single_count+=size
                if single_count%200==0:print(label,single_count,flush=True)
        table=pd.DataFrame(records);table.to_csv(a.output/(label+'_windows.csv'),index=False)
        deletion=pd.DataFrame(deletions);deletion.to_csv(a.output/(label+'_mode_deletions.csv'),index=False)
        summary={}
        for step,group in table.groupby('step'):
            result=dict(samples=len(group),position_RMSE_A=float(np.sqrt(group.q_mse.mean())),velocity_RMSE_A_fs=float(np.sqrt(group.v_mse.mean())),
                predicted_KE_eV=float(group.predicted_KE_eV.mean()),reference_KE_eV=float(group.reference_KE_eV.mean()),
                KE_bias_eV=float((group.predicted_KE_eV-group.reference_KE_eV).mean()),delta_KE_error_eV=float(group.delta_KE_error_eV.mean()),
                species={str(z):dict(predicted_KE_eV=float(group[f'predicted_KE_Z{z}'].mean()),reference_KE_eV=float(group[f'reference_KE_Z{z}'].mean()),
                    velocity_RMSE_A_fs=float(np.sqrt(group[f'velocity_mse_Z{z}'].mean())),
                    predicted_vacf=float(group[f'predicted_vacf_Z{z}'].mean()),
                    reference_vacf=float(group[f'reference_vacf_Z{z}'].mean()),
                    vacf_window_RMSE=float(np.sqrt(((group[f'predicted_vacf_Z{z}']-group[f'reference_vacf_Z{z}'])**2).mean()))) for z in [8,14]},routing_correlations={})
            for k in range(4):
                route=group[f'recurrent_{k+1}'];corr=route.corr(group.delta_KE_error_eV)
                result['routing_correlations'][str(k+1)]=dict(mean=float(route.mean()),std=float(route.std()),
                    correlation_with_step_KE_error=float(corr) if np.isfinite(corr) else None)
            summary[str(int(step))]=result
        state=torch.load(path,map_location='cpu',weights_only=False)
        result=dict(checkpoint=str(path),checkpoint_epoch_zero_based=state['epoch'],checkpoint_phase=state['phase'],validation_samples=single_count,
            valid_long_windows=int(data.valid.sum()),steps=summary,
            kinematics_RMSE_A_fs=float(np.sqrt(np.mean(kin))),reference_kinematics_RMSE_A_fs=float(np.sqrt(np.mean(refkin))),
            join_derivative_jump_RMS_A_fs=float(np.sqrt(np.mean(join))),reference_join_RMS_A_fs=float(np.sqrt(np.mean(refjoin))),
            mode_statistics={k:(x/single_count).tolist() for k,x in diag.items()},velocity_mode_cosine_matrix=(sum(mode_cos)/single_count).tolist(),
            mode_deletion_scope='Only final decoder contribution is removed; upstream context held fixed. Not an identification of physically independent modes.',
            mode_deletions={str(k):dict(position_RMSE_A=float(np.sqrt(g.q_mse.mean())),velocity_RMSE_A_fs=float(np.sqrt(g.v_mse.mean())),
                removed_position_rms_A=float(g.removed_q_rms.mean()),removed_velocity_rms_A_fs=float(g.removed_v_rms.mean())) for k,g in deletion.groupby('mode')},
            elapsed_s=time.perf_counter()-started)
        save_json(a.output/(label+'.json'),result);print(label,'completed',result['elapsed_s'],flush=True)
        del model,state;torch.cuda.empty_cache()


if __name__=='__main__': main()
