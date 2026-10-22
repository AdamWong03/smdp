"""Once-only independent test evaluation after validation-selected model is fixed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from smdp_neural import NeuralSMDP, SMDPConfig
from periodic import PeriodicGraphBuilder, predict_batch
from reference_data import QuartzSegments
from train_quartz import batch_to, diagnostics


def load_model(path, device):
    state=torch.load(path,map_location=device,weights_only=False)
    model=NeuralSMDP(SMDPConfig(**state['config']['model'])).to(device).eval()
    model.load_state_dict(state['model'])
    return model,state['config']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    model,cfg=load_model(args.checkpoint,args.device)
    data=QuartzSegments(args.data_root,'test',args.output/'cache')
    builder=PeriodicGraphBuilder(model.config.cutoff_angstrom)
    sums={k:np.zeros(17) for k in ['q_mae','q_mse','v_mae','v_mse']}; n=0; diag={}
    derivative=[]; ref_derivative=[]; joins=[]; second=[]
    multi={k:dict(q=0.,v=0.,count=0) for k in [2,4,8,16]}
    ref_joins=[]
    offset_index={int(t):i for i,t in enumerate(QuartzSegments.offsets)}
    with torch.inference_mode():
        for b in DataLoader(data,batch_size=4,shuffle=False):
            b=batch_to(b,args.device); q,v=b['q'],b['v']
            out=predict_batch(model,q[:,0],v[:,0],b['z'],b['mass'],b['cell'],list(range(17)),builder)
            pq,pv=out['positions'],out['velocities']
            rq=q[:,:17].transpose(0,1).flatten(1,2); rv=v[:,:17].transpose(0,1).flatten(1,2)
            for name,error in [('q',pq-rq),('v',pv-rv)]:
                sums[name+'_mae']+=error.abs().mean((1,2)).cpu().numpy()*len(q)
                sums[name+'_mse']+=error.square().mean((1,2)).cpu().numpy()*len(q)
            derivative.append((((pq[2:]-pq[:-2])/2-pv[1:-1]).square().mean()).item())
            ref_derivative.append((((rq[2:]-rq[:-2])/2-rv[1:-1]).square().mean()).item())
            nextout=predict_batch(model,pq[-1].reshape_as(q[:,0]),pv[-1].reshape_as(v[:,0]),b['z'],b['mass'],b['cell'],[0.,1.,16.],builder)
            torch.testing.assert_close(nextout['positions'][0],pq[-1],rtol=0,atol=0)
            torch.testing.assert_close(nextout['velocities'][0],pv[-1],rtol=0,atol=0)
            joins.append(((nextout['positions'][1]-pq[-1]-(pq[-1]-pq[-2])).square().mean()).item())
            ref_joins.append(((q[:,offset_index[17]]-2*q[:,16]+q[:,15]).square().mean()).item())
            valid=b['rollout_valid']
            if valid.any(): second.append(((nextout['positions'][-1].reshape_as(q[:,0])[valid]-q[valid,17]).square().mean()).item())
            qn=nextout['positions'][-1].reshape_as(q[:,0]);vn=nextout['velocities'][-1].reshape_as(v[:,0])
            for step in range(2,17):
                if step>2:
                    mo=predict_batch(model,qn,vn,b['z'],b['mass'],b['cell'],[16.],builder)
                    qn=mo['positions'][0].reshape_as(qn);vn=mo['velocities'][0].reshape_as(vn)
                if step in multi and valid.any():
                    count=int(valid.sum());target=offset_index[16*step]
                    multi[step]['q']+=float((qn[valid]-q[valid,target]).square().mean())*count
                    multi[step]['v']+=float((vn[valid]-v[valid,target]).square().mean())*count
                    multi[step]['count']+=count
            n+=len(q)
            for k,x in diagnostics(out).items(): diag[k]=diag.get(k,0.)+np.array(x)*len(q)
    rows=[]
    for t in range(17):
        rows.append(dict(time_fs=t,position_MAE_A=sums['q_mae'][t]/n,position_RMSE_A=np.sqrt(sums['q_mse'][t]/n),
                         velocity_MAE_A_fs=sums['v_mae'][t]/n,velocity_RMSE_A_fs=np.sqrt(sums['v_mse'][t]/n)))
    with (args.output/'query_errors.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    result=dict(checkpoint=str(args.checkpoint),test_samples=n,endpoint=rows[-1],
                normalized_endpoint_loss=float(sums['q_mse'][-1]/n/cfg['stats']['displacement_residual_rms']**2+sums['v_mse'][-1]/n/cfg['stats']['velocity_delta_rms']**2),
                position_velocity_consistency_RMSE_A_fs=float(np.sqrt(np.mean(derivative))),
                reference_finite_difference_RMSE_A_fs=float(np.sqrt(np.mean(ref_derivative))),
                join_position_derivative_jump_RMS_A_fs=float(np.sqrt(np.mean(joins))),
                join_state_discontinuity=0.,two_step_position_RMSE_A=float(np.sqrt(np.mean(second))),
                reference_join_derivative_jump_RMS_A_fs=float(np.sqrt(np.mean(ref_joins))),
                multi_step={str(k):dict(time_fs=16*k,samples=x['count'],position_RMSE_A=float(np.sqrt(x['q']/max(1,x['count']))),
                    velocity_RMSE_A_fs=float(np.sqrt(x['v']/max(1,x['count'])))) for k,x in multi.items()},
                mode_statistics={k:(x/n).tolist() for k,x in diag.items()})
    (args.output/'test_metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
