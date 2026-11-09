"""25k updates with 16 supervised transitions/update; unmodified model/loss."""
import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from conditional_transition.common import originals,extra_graphs,seed_all,transitions
from periodic import PeriodicGraphBuilder
from relational_inference import load_relational
from relational_transition import endpoint_objective
from smdpr.data import TwoStepLabels
from smdpr.finetune import two_step
from smdpr.train import write_json
from train_relational import atomic_save,evaluate

STOP=False


def request_stop(*_):
    global STOP
    STOP=True


class Cycle:
    def __init__(self,size,generator):
        self.size=size;self.generator=generator
        self.order=torch.empty(0,dtype=torch.long);self.cursor=0

    def take(self,n):
        result=[]
        while len(result)<n:
            if self.cursor==len(self.order):
                self.order=torch.randperm(self.size,generator=self.generator);self.cursor=0
            count=min(n-len(result),len(self.order)-self.cursor)
            result.extend(self.order[self.cursor:self.cursor+count].tolist());self.cursor+=count
        return result

    def state(self):return dict(order=self.order,cursor=self.cursor,size=self.size)

    def restore(self,state):
        if state['size']!=self.size:raise ValueError('Changed pool on resume')
        self.order=state['order'];self.cursor=state['cursor']


def new_count(step):
    return (48*(step+1))//10-(48*step)//10


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--target-step',type=int,choices=[12500,25000],required=True)
    args=parser.parse_args();run=args.run
    cfg=json.loads((run/'config.json').read_text());arm=cfg['arm']
    seed_all(cfg['seed'])
    initial=Path(cfg['initial_checkpoint'])
    model,initial_saved=load_relational(initial)
    base=initial_saved['config']
    train=originals(base,'train');val=originals(base,'val')
    sources={}
    if arm=='two_step':
        train_labels=TwoStepLabels(transitions(cfg['system']),train)
        train=train_labels.graphs
        val_labels=TwoStepLabels(transitions(cfg['system'],'val'),val)
        sources=dict(train_sources=train_labels.source_paths,val_sources=val_labels.source_paths,
            excluded_train=train_labels.excluded,excluded_val=val_labels.excluded)
        if set(train_labels.source_paths)&set(val_labels.source_paths):raise ValueError('Split leakage')
    files=[]
    if arm in ['extra_reference','autonomous']:
        rounds=1 if args.target_step==12500 else 2
        files=[run/f'collection_round{i}/labels.h5' for i in range(1,rounds+1)]
    extra=extra_graphs(files,model) if files else []
    # Original and oracle graph schemas must be identical before mixed batches.
    if extra and set(extra[0].keys())!=set(train[0].keys()):
        # sample_id is provenance only, never a model input.
        for g in extra:
            if 'sample_id' in g:del g['sample_id']
        if set(extra[0].keys())!=set(train[0].keys()):
            raise ValueError((train[0].keys(),extra[0].keys()))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.,amsgrad=True)
    generator=torch.Generator().manual_seed(cfg['seed'])
    extra_generator=torch.Generator().manual_seed(cfg['seed']+1)
    orig_cycle=Cycle(len(train),generator)
    extra_cycle=Cycle(len(extra),extra_generator) if extra else None
    steps=0;previous=0.;old_draws=new_draws=0;previous_best=float('inf')
    if (run/'last.pt').exists():
        saved=torch.load(run/'last.pt',weights_only=False,map_location='cpu')
        if saved['experiment_config']!=cfg:raise ValueError('Experiment configuration changed')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        steps=saved['optimizer_steps'];previous=saved['elapsed_seconds']
        old_draws=saved['original_draws'];new_draws=saved['extra_draws']
        orig_cycle.restore(saved['original_cycle']);generator.set_state(saved['loader_rng'])
        extra_generator.set_state(saved['extra_rng'])
        if extra_cycle and saved.get('extra_cycle') and saved['extra_cycle']['size']==len(extra):
            extra_cycle.restore(saved['extra_cycle'])
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        previous_best=saved.get('best_validation_final_loss',float('inf'))
    if steps>=args.target_step:return
    if steps not in [0,12500] and not (run/'status.json').exists():raise RuntimeError('Untracked partial run')
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    builder=PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    validation=DataLoader(val,batch_size=8,shuffle=False)
    metadata=dict(experiment_config=cfg,initial_checkpoint=str(initial),
        normalization=model.rms_targets.detach().cpu().tolist(),architecture=base['model'],
        parameters=sum(p.numel() for p in model.parameters()),train_samples=len(train),val_samples=len(val),
        extra_labels=len(extra),extra_files=list(map(str,files)),device=torch.cuda.get_device_name(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),precision='FP32; TF32 disabled',
        source_selection=sources,criterion='fixed final update; no checkpoint selection by physics/test',
        transition_targets_per_update=16,oracle_used_in_inference=False)
    write_json(run/f'metadata_round{1 if args.target_step==12500 else 2}.json',metadata)
    print(json.dumps(dict(event='official_training_started',**metadata)),flush=True)
    start=time.time();interval_start=time.perf_counter();running=np.zeros(3);batches=0
    torch.cuda.reset_peak_memory_stats()
    while steps<args.target_step:
        model.train();optimizer.zero_grad(set_to_none=True)
        if arm=='two_step':
            ids=orig_cycle.take(8);batch=Batch.from_data_list([train[i] for i in ids]).cuda()
            loss,_,p1,p2=two_step(model,batch,builder,.02)
            loss.backward()
            final=float(.5*(p1['final']+p2['final']));prop=float(.5*(p1['proposal']+p2['proposal']))
            old_draws+=16
        else:
            n=new_count(steps) if extra_cycle else 0
            samples=[train[i] for i in orig_cycle.take(16-n)]
            if n:samples.extend(extra[i] for i in extra_cycle.take(n))
            order=torch.randperm(16,generator=generator).tolist()
            final=prop=total=0.
            for begin in [0,8]:
                batch=Batch.from_data_list([samples[i] for i in order[begin:begin+8]]).cuda()
                out=model(batch)
                micro,pieces=endpoint_objective(model,out,batch.displacements,batch.update_velocities,
                                               proposal_weight=.02)
                (.5*micro).backward()
                total+=.5*float(micro.detach());final+=.5*float(pieces['final']);prop+=.5*float(pieces['proposal'])
            loss=torch.tensor(total)
            old_draws+=16-n;new_draws+=n
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {steps}')
        torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
        lr=1e-5+.5*(3e-5-1e-5)*(1+math.cos(math.pi*steps/25000))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();steps+=1
        running+=[float(loss.detach()),final,prop];batches+=1
        if steps%100==0:
            write_json(run/'progress.json',dict(state='training',optimizer_steps=steps,
                target_steps=25000,supervised_transition_targets=steps*16,original_draws=old_draws,
                extra_draws=new_draws,lr=lr,elapsed_seconds=previous+time.time()-start))
        if steps%625==0 or STOP or steps==args.target_step:
            result=evaluate(model,validation,torch.device('cuda'))
            record=dict(optimizer_steps=steps,supervised_transition_targets=steps*16,
                train_total_loss=running[0]/batches,train_final_loss=running[1]/batches,
                train_proposal_loss=running[2]/batches,validation=result,lr=lr,
                original_draws=old_draws,extra_draws=new_draws,extra_pool_size=len(extra),
                interval_seconds=time.perf_counter()-interval_start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            previous_best=min(previous_best,result['selection_loss'])
            payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=base,
                experiment_config=cfg,optimizer_steps=steps,epoch=steps//625-1,record=record,
                best_validation_final_loss=previous_best,elapsed_seconds=previous+time.time()-start,
                original_draws=old_draws,extra_draws=new_draws,original_cycle=orig_cycle.state(),
                extra_cycle=extra_cycle.state() if extra_cycle else None,
                loader_rng=generator.get_state(),extra_rng=extra_generator.get_state(),
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state())
            atomic_save(payload,run/'last.pt')
            if steps in [12500,25000]:atomic_save(payload,run/f'step_{steps}.pt')
            with (run/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(run/'status.json',dict(state='training',**record))
            print(json.dumps(record),flush=True)
            running[:]=0;batches=0;interval_start=time.perf_counter()
            if STOP:
                write_json(run/'status.json',dict(state='paused',**record));return
    write_json(run/'status.json',dict(state='training_completed' if steps==25000 else 'round1_training_completed',
        optimizer_steps=steps,supervised_transition_targets=steps*16,original_draws=old_draws,
        extra_draws=new_draws,checkpoint_selection='fixed final step; no test/physics selection',
        physical_evaluation_completed=False))


if __name__=='__main__':main()
