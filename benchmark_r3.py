"""Measure true endpoint-only versus dense time-query cost on identical hardware.

No reference trajectories are generated; this is a timed network+neighbor-list
inference loop from a frozen validation configuration. No speedup claim about BKS.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from evaluate_quartz import load_model
from reference_data import QuartzSegments
from periodic import PeriodicGraphBuilder, predict_batch


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def bench(model, builder, data, device, times, steps, warmup):
    sample=data[0]
    q=sample['q'][0].to(device).unsqueeze(0)
    v=sample['v'][0].to(device).unsqueeze(0)
    z=sample['z'].to(device).unsqueeze(0)
    m=sample['mass'].to(device).unsqueeze(0)
    cell=sample['cell'].to(device).unsqueeze(0)
    def step():
        return predict_batch(model,q,v,z,m,cell,times,builder)
    with torch.inference_mode():
        for _ in range(warmup):
            step()
        synchronize(device)
        t=time.perf_counter()
        for _ in range(steps):
            step()
        synchronize(device)
        elapsed=time.perf_counter()-t
    return dict(queries=len(times),steps=steps,elapsed_s=elapsed,
                ms_per_macrostep=1000*elapsed/steps,
                ns_per_day=steps*16*1e-6*86400/elapsed,
                includes=['periodic_graph_rebuild','network','CPU/GPU_transfer'],
                caveat='Fixed input phase-space state, throughput microbenchmark; no I/O or thermostat')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--warmup',type=int,default=20)
    args=p.parse_args()
    if args.steps <= 0 or args.warmup < 0: p.error('Invalid benchmark length')
    torch.set_num_threads(4)
    device=torch.device(args.device)
    model,conf=load_model(args.checkpoint,device)
    builder=PeriodicGraphBuilder(model.config.cutoff_angstrom)
    data=QuartzSegments(args.data_root,'val',args.output.parent/'benchmark_cache')
    values=[bench(model,builder,data,device,[16.],args.steps,args.warmup),
            bench(model,builder,data,device,list(range(1,17)),args.steps,args.warmup)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(checkpoint=str(args.checkpoint),hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                results=values,endpoint_to_dense_time_ratio=values[1]['elapsed_s']/values[0]['elapsed_s'])
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
