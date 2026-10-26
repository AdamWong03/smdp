"""Train-only polynomial oracle and matched-GPU inference costs. No fitting network."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from neural_segment.model import SegmentConfig, NeuralSegment, integrated_legendre
from periodic import PeriodicGraphBuilder
from reference_data import QuartzSegments, validate_split_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'basis.json').exists(): raise FileExistsError('Frozen basis already exists')
    torch.set_num_threads(4);torch.manual_seed(20261026)
    manifest=validate_split_manifest(args.data_root)
    d=QuartzSegments(args.data_root,'train',args.output/'cache')
    times=np.arange(1,17,dtype=float)
    sq,sv=d.scales['displacement_residual_rms'],d.scales['velocity_delta_rms']
    report=dict(protocol='train-only normalized position+velocity least squares, exact initial state',
        samples=len(d),threshold_position_RMSE_A=.00025,threshold_velocity_RMSE_A_fs=.0000343,
        split_manifest=manifest,train_scales=d.scales,candidates={})
    selected=None
    for count in (6,8):
        _,iv,iq=integrated_legendre(torch.tensor(times/16),count)
        mat=np.concatenate([256*iq.numpy()/sq,16*iv.numpy()/sv])
        inverse=np.linalg.pinv(mat)
        eq=ev=0.;n=0
        for start in range(0,len(d),128):
            q=d.q[start:start+128,:17].double().numpy();v=d.v[start:start+128,:17].double().numpy()
            dq=q[:,1:]-q[:,:1]-times[None,:,None,None]*v[:,:1]
            dv=v[:,1:]-v[:,:1]
            target=np.concatenate([dq/sq,dv/sv],axis=1)
            c=np.einsum('kt,btnc->bknc',inverse,target)
            pred=np.einsum('tk,bknc->btnc',mat,c)
            eq+=np.square((pred[:,:16]-target[:,:16])*sq).sum()
            ev+=np.square((pred[:,16:]-target[:,16:])*sv).sum();n+=dq.size
        record=dict(position_RMSE_A=float(np.sqrt(eq/n)),velocity_RMSE_A_fs=float(np.sqrt(ev/n)))
        record['passed']=record['position_RMSE_A']<=.00025 and record['velocity_RMSE_A_fs']<=.0000343
        report['candidates'][str(count)]=record
        if selected is None and record['passed']:selected=count
    report['selected_basis_count']=selected
    (args.output/'basis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    if selected is None:raise RuntimeError('Neither proposed basis passes: stop, do not train')
    device=torch.device(args.device);builder=PeriodicGraphBuilder(4.5)
    q,v=d.q[0,0].to(device),d.v[0,0].to(device)
    z,m=d.z.to(device),d.m.to(device)
    e,s=builder.one(q.cpu().numpy(),d.cells[0].numpy())
    e=torch.tensor(e,device=device);s=torch.tensor(s,device=device,dtype=q.dtype)
    timing={}
    with torch.inference_mode():
        for variant in ('direct','independent','full'):
            cfg=SegmentConfig(basis_count=selected,variant=variant)
            model=NeuralSegment(cfg).to(device).eval()
            rec=dict(config=asdict(cfg),parameters=sum(x.numel() for x in model.parameters()),queries={})
            for queries in ([16.],list(range(1,17))):
                for _ in range(5):model(q,v,z,m,e,s,queries)
                if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                t=time.perf_counter()
                for _ in range(50):model(q,v,z,m,e,s,queries)
                if device.type=='cuda':torch.cuda.synchronize()
                rec['queries'][str(len(queries))]=dict(ms=1000*(time.perf_counter()-t)/50,
                    peak_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
            # Measured operator FLOPs, not a claim of exact hardware work count.
            from torch.profiler import profile,ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:
                model(q,v,z,m,e,s,[16.])
            rec['profiled_operator_flops']=sum(x.flops for x in prof.key_averages())
            timing[variant]=rec
    (args.output/'preflight_cost.json').write_text(json.dumps(dict(hardware=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
        includes='network only, fixed input graph, no BKS, no I/O; 50 calls per query count',models=timing),indent=2))


if __name__=='__main__':main()
