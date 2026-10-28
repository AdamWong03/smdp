"""Measure batch scaling on training data, without any optimizer updates."""
import argparse
import gc
import json
import os
import signal
import time
from pathlib import Path
import torch
from torch_geometric.data import Batch
from trajcast.utils.misc import GLOBAL_DEVICE
from trajcast_smdp.data import RUN,dataset
from trajcast_smdp.model import SMDP
from trajcast_smdp.train import segment_loss,rollout_loss


def main(run,pause_pid):
    torch.set_default_dtype(torch.float64);GLOBAL_DEVICE.device=torch.device('cuda')
    saved=torch.load(run/'segment/last.pt',map_location='cpu',weights_only=False)
    model=SMDP(config=saved['config']['model'],rms_targets=saved['model']['rms_targets'].tolist()).cuda()
    model.load_state_dict(saved['model'],strict=True);model.train()
    ds=dataset(run,'train')
    samples=[ds[i] for i in range(8)]
    times=[2,5,7,10,15,19,20,25,28,30]
    results=[];paused=False
    def step(size,rollout=False):
        model.zero_grad(set_to_none=True)
        batch=Batch.from_data_list(samples[:size]).cuda()
        loss,*_=segment_loss(model,batch.clone(),times)
        if rollout:
            rloss,_=rollout_loss(model,batch,8,2)
            loss=loss+.5*rloss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        return float(loss.detach())
    try:
        if pause_pid:
            command=Path(f'/proc/{pause_pid}/cmdline').read_bytes().replace(b'\0',b' ')
            if b' -m trajcast_smdp.train' not in command:
                raise RuntimeError('Refusing to pause an unrelated process')
            os.kill(pause_pid,signal.SIGSTOP);paused=True
        for size in [2,4,8]:
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                # COM and neighbor batching must not change individual predictions.
                with torch.no_grad():
                    together=model(Batch.from_data_list(samples[:size]).cuda()).target
                    apart=torch.cat([model(Batch.from_data_list(samples[i:i+2]).cuda()).target
                                     for i in range(0,size,2)])
                    torch.testing.assert_close(together,apart,rtol=1e-8,atol=1e-10)
                    batch_error=float((together-apart).abs().max())
                for _ in range(2): step(size)
                durations=[]
                for _ in range(2):
                    torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(5): step(size)
                    torch.cuda.synchronize();durations.append((time.perf_counter()-start)/5)
                segment_peak=torch.cuda.max_memory_allocated()
                # Retain the segment graph AND last two autoregressive gradients,
                # just as in the existing formal eight-step pushforward phase.
                torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                step(size,True);torch.cuda.synchronize()
                rollout_seconds=time.perf_counter()-start
                row=dict(batch_size=size,seconds_per_batch=durations,
                    samples_per_second=size/(sum(durations)/len(durations)),
                    segment_training_epoch_seconds=5000/size*sum(durations)/len(durations),
                    segment_peak_allocated_GiB=segment_peak/1024**3,
                    rollout8_peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
                    rollout8_batch_seconds=rollout_seconds,batch_prediction_max_error=batch_error,status='passed')
            except torch.cuda.OutOfMemoryError as error:
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                row=dict(batch_size=size,status='out_of_memory',reason=str(error))
            results.append(row);print(json.dumps(row),flush=True)
    finally:
        if paused: os.kill(pause_pid,signal.SIGCONT)
        report=dict(time_unix=time.time(),checkpoint_epoch=saved['record']['epoch'],
            hardware=torch.cuda.get_device_name(),precision='float64',optimizer_updates=0,
            data_split='train',samples=list(range(8)),queries_fs=times,
            training_temporarily_paused=paused,results=results,
            note='No checkpoint or optimizer changed. Timings include collation, transfer, forward, backward and clipping; exclude Adam and full validation. Existing trainer retains its GPU allocations.')
        (run/'batch_scaling.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--pause-training-pid',type=int,default=0);a=p.parse_args()
    main(a.run,a.pause_training_pid)
