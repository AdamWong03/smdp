"""Training-only runtime/numerical equivalence check; no optimizer updates."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
import e3nn
from torch_geometric.data import Batch
from trajcast.utils.misc import GLOBAL_DEVICE
from trajcast_smdp.data import RUN,dataset
from trajcast_smdp.model import SMDP
from trajcast_smdp.train import segment_loss
from trajcast_smdp.runtime import compile_tensor_products


def main(run,iterations,eager_only=False,output='runtime_benchmark.json',compile_products=False):
    torch.set_default_dtype(torch.float64)
    GLOBAL_DEVICE.device=torch.device('cuda')
    saved=torch.load(run/'segment/last.pt',map_location='cpu',weights_only=False)
    model=SMDP(config=saved['config']['model'],rms_targets=saved['model']['rms_targets'].tolist()).cuda()
    model.load_state_dict(saved['model'])
    compiled=None
    if not eager_only:
        if compile_products:
            e3nn.set_optimization_defaults(jit_mode='eager')
            compiled=SMDP(config=saved['config']['model'],rms_targets=saved['model']['rms_targets'].tolist()).cuda()
            compiled.load_state_dict(saved['model'],strict=True)
            print('Compiling tensor products:',compile_tensor_products(compiled),flush=True)
            e3nn.set_optimization_defaults(jit_mode='script')
        else:
            compiled=copy.deepcopy(model)
            compiled._int_out.MotionModes.enable_compilation()
    ds=dataset(run,'train');batch=Batch.from_data_list([ds[0],ds[1]]).cuda()
    times=[5,10,15,20,25,30]
    def step(m):
        m.zero_grad(set_to_none=True)
        loss,p,_,_=segment_loss(m,batch.clone(),times)
        loss.backward()
        return loss.detach(),p['displacements'].detach(),p['velocities'].detach()
    if eager_only:
        for _ in range(3): step(model)
        measurements=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(iterations): step(model)
            torch.cuda.synchronize();measurements.append((time.perf_counter()-start)/iterations)
        result=dict(checkpoint_epoch=saved['record']['epoch'],training_samples=[0,1],iterations=iterations,
                    precision='float64',optimizer_steps=0,hardware=torch.cuda.get_device_name(),
                    shared_gpu=True,seconds_per_forward_backward=measurements,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated())
        (run/output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        return
    expected=step(model);start=time.perf_counter();actual=step(compiled)
    torch.cuda.synchronize();compile_s=time.perf_counter()-start
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)
    errors=[]
    for (name,p),(other,q) in zip(model.named_parameters(),compiled.named_parameters()):
        assert name==other
        if p.grad is None: assert q.grad is None;continue
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-10)
        errors.append(float((p.grad-q.grad).abs().max()))
    print(json.dumps(dict(compilation_seconds=compile_s,max_gradient_error=max(errors))),flush=True)
    label_compiled='compiled_products' if compile_products else 'compiled_modes'
    measurements={'eager':[],label_compiled:[]}
    for m in [model,compiled]:
        for _ in range(3): step(m)
    # Alternate order to reduce shared-GPU/load drift effects.
    for order in [[('eager',model),(label_compiled,compiled)],
                  [(label_compiled,compiled),('eager',model)]]:
        for label,m in order:
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(iterations): step(m)
            torch.cuda.synchronize()
            measurements[label].append((time.perf_counter()-start)/iterations)
    result=dict(checkpoint_epoch=saved['record']['epoch'],training_samples=[0,1],iterations=iterations,
                precision='float64',same_weights=True,optimizer_steps=0,hardware=torch.cuda.get_device_name(),
                shared_gpu=True,compilation_seconds=compile_s,max_gradient_error=max(errors),
                seconds_per_forward_backward=measurements,
                speedup=sum(measurements['eager'])/sum(measurements[label_compiled]),
                peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (run/output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--iterations',type=int,default=15);p.add_argument('--eager-only',action='store_true')
    p.add_argument('--output',default='runtime_benchmark.json')
    p.add_argument('--compile-products',action='store_true')
    args=p.parse_args();main(args.run,args.iterations,args.eager_only,args.output,args.compile_products)
