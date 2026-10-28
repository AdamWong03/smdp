"""Original TrajCast Trainer with SMDP segment/rollout loss and validation hooks.

Retains its optimizer, normalization, batch loop orchestration and checkpoint
machinery. Only SMDP's necessary trajectory objective replaces endpoint MSE.
"""
import argparse
import copy
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
import torch
import yaml
import numpy as np
from torch_geometric.loader import DataLoader
import trajcast.model.training as upstream
from trajcast_smdp.model import SMDP
from trajcast_smdp.data import RUN,dataset,fresh_graph
from trajcast_based.train_original import epoch_scheduler_class

upstream.EfficientTrajCastModel=SMDP
upstream.CustomChainedScheduler=epoch_scheduler_class(upstream.CustomChainedScheduler)


def phase_error(model,q,v,refq,refv):
    return ((q-refq)/model.rms_targets[0]).square().mean()+((v-refv)/model.rms_targets[1]).square().mean()


def segment_loss(model,batch,times):
    out=model(batch)
    pred=model.query(out,times)
    ix=torch.as_tensor(times,dtype=torch.long,device=batch.pos.device)
    qref=batch.trajectory_q[:,ix].transpose(0,1);vref=batch.trajectory_v[:,ix].transpose(0,1)
    endpoint=phase_error(model,pred['displacements'][-1],pred['velocities'][-1],qref[-1],vref[-1])
    interior=phase_error(model,pred['displacements'][:-1],pred['velocities'][:-1],qref[:-1],vref[:-1])
    return .5*(endpoint+interior),pred,qref,vref


def rollout_loss(model,batch,steps,grad_steps=0):
    origin=batch.pos
    q=origin;v=batch.velocities
    total=origin.new_zeros(());rows=[];count=0
    for k in range(steps):
        active=grad_steps>0 and k>=steps-grad_steps
        with torch.set_grad_enabled(active):
            current=fresh_graph(batch,q,v)
            out=model(current)
            q=q+out.target[:,:3]*model.rms_targets[0]
            v=out.target[:,3:]*model.rms_targets[1]
            index=30 if k==0 else 30+k
            targetq=batch.trajectory_q[:,index];targetv=batch.trajectory_v[:,index]
            loss=phase_error(model,q-origin,v,targetq,targetv)
            if active or grad_steps==0: total=total+loss;count+=1
            rows.append(dict(time_fs=30*(k+1),loss=float(loss.detach()),
                position_RMSE_A=float((q.detach()-origin-targetq).square().mean().sqrt()),
                velocity_RMSE_A_fs=float((v.detach()-targetv).square().mean().sqrt()),
                velocity_squared=float(v.detach().square().mean()),
                reference_velocity_squared=float(targetv.square().mean())))
        if not active: q=q.detach();v=v.detach()
    return total/max(count,1),rows


class StageFinished(Exception): pass


def resume_handler(base,payload,stage_history):
    """Resume the original Trainer loop at an epoch boundary, not a new fit.

    Our last.pt is saved before upstream calls scheduler.step. Replaying the
    recorded validation scores restores exactly that schedule, including the
    final pending step. Never reset Adam moments or the random sampling stream.
    """
    class RestoringHandler(base):
        def load_latest(self,state):
            state.model.load_state_dict(payload['model'],strict=True)
            state.optimizer.load_state_dict(payload['optimizer'])
            sched=state.lr_scheduler
            if sched is not None:
                for group in state.optimizer.param_groups:
                    group['lr']=payload['config']['training']['optimizer_settings']['lr']
                for row in stage_history:
                    assert abs(state.optimizer.param_groups[0]['lr']-row['lr'])<1e-15
                    sched.step(row['selection_score'])
            torch.set_rng_state(payload['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in payload['cuda_rng']])
            if 'numpy_rng' in payload: np.random.set_state(payload['numpy_rng'])
            if 'python_rng' in payload: random.setstate(payload['python_rng'])
            score=min(row['selection_score'] for row in stage_history)
            print(json.dumps(dict(resumed_stage=payload['record']['stage'],
                next_epoch=payload['record']['epoch']+1,optimizer_restored=True,
                rng_restored=True,scheduler_replayed=True,lr=state.optimizer.param_groups[0]['lr'])),flush=True)
            return payload['record']['epoch'],score
    return RestoringHandler


class SegmentTrainer(upstream.Trainer):
    def _train_epoch(self,loader):
        self.model.train();sums=[0.,0.,0.];samples=0;rollsum=0.;rollcount=0
        for index,batch in enumerate(loader):
            batch=batch.to(self.device)
            # Dense labels are real frames. Always include endpoint; random times
            # supplement fixed phase anchors without additional GNN evaluations.
            times=sorted(set([5,10,15,20,25]+torch.randint(1,30,(4,),device='cpu').tolist()))+[30]
            self.optimizer.zero_grad(set_to_none=True)
            loss,pred,qref,vref=segment_loss(self.model,batch.clone(),times)
            if self.rollout_steps and index%self.settings['rollout_every_batches']==0:
                valid=[g for g in batch.to_data_list() if bool(g.rollout_valid)]
                if valid:
                    from torch_geometric.data import Batch
                    rloss,_=rollout_loss(self.model,Batch.from_data_list(valid),self.rollout_steps,
                                        self.settings['rollout_gradient_steps'])
                    loss=loss+self.settings['rollout_loss_weight']*rloss
                    rollsum+=float(rloss.detach());rollcount+=1
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite training loss batch {index}')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.max_grad_norm,error_if_nonfinite=True)
            self.optimizer.step()
            n=batch.num_graphs;samples+=n
            sums[0]+=float(loss.detach())*n
            sums[1]+=float((pred['displacements'][-1].detach()-qref[-1]).abs().mean())*n
            sums[2]+=float((pred['velocities'][-1].detach()-vref[-1]).abs().mean())*n
            if index%250==0:
                print(json.dumps(dict(stage=self.stage,batch=index,batches=len(loader),
                    mean_loss=sums[0]/samples,grad_norm=float(norm),wall_s=time.perf_counter()-self.started)),flush=True)
        self.last_rollout_training_loss=rollsum/max(rollcount,1)
        return torch.tensor(sums[0]/samples),dict(displacements=sums[1]/samples,update_velocities=sums[2]/samples)


@torch.no_grad()
def validate(model,validation,device,rollout_samples=128):
    model.eval();sums=dict(loss=0.,q2=0.,v2=0.,q_abs=0.,v_abs=0.);n=0
    for batch in DataLoader(validation,batch_size=2,shuffle=False):
        batch=batch.to(device)
        loss,p,q,v=segment_loss(model,batch,list(range(1,31)))
        count=batch.num_graphs;n+=count;sums['loss']+=float(loss)*count
        eq=p['displacements'][-1]-q[-1];ev=p['velocities'][-1]-v[-1]
        sums['q2']+=float(eq.square().mean())*count;sums['v2']+=float(ev.square().mean())*count
        sums['q_abs']+=float(eq.abs().mean())*count;sums['v_abs']+=float(ev.abs().mean())*count
    valid=[i for i in range(len(validation)) if bool(validation[i].rollout_valid)][:rollout_samples]
    rl=0.;traces=[]
    for batch in DataLoader(validation[valid],batch_size=2,shuffle=False):
        batch=batch.to(device);loss,trace=rollout_loss(model,batch,8)
        rl+=float(loss)*batch.num_graphs;traces.append(trace)
    import numpy as np
    meantrace=[{key:float(np.mean([t[k][key] for t in traces])) for key in traces[0][k]} for k in range(8)]
    return dict(samples=n,segment_loss=sums['loss']/n,
        endpoint_position_RMSE_A=(sums['q2']/n)**.5,endpoint_velocity_RMSE_A_fs=(sums['v2']/n)**.5,
        endpoint_position_MAE_A=sums['q_abs']/n,endpoint_velocity_MAE_A_fs=sums['v_abs']/n,
        rollout_samples=len(valid),rollout8_loss=rl/len(valid),rollout_trace=meantrace,
        selection_score=sums['loss']/n+.5*rl/len(valid))


def save_checkpoint(path,trainer,config,record):
    if shutil.disk_usage(path.parent).free<5*1024**3:
        raise OSError('Less than 5 GiB free; stop before checkpoint/trajectory corruption')
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model=trainer.model.state_dict(),optimizer=trainer.optimizer.state_dict(),config=config,
        record=record,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate()),tmp)
    tmp.replace(path)


def main(run,resume=None,compile_modes=False):
    # AtomicGraphDataset selects its cache filename from the global dtype,
    # before Trainer has an opportunity to apply model.precision.
    torch.set_default_dtype(torch.float64)
    run=Path(run)
    if (run/'training_started.json').exists() and resume is None:
        raise FileExistsError('Do not overwrite/restart a previous training implicitly; use explicit --resume')
    if (run/'training_complete.json').exists(): raise FileExistsError('Training is already complete')
    if not json.loads((run/'correctness.json').read_text())['passed']: raise RuntimeError('Correctness checks failed')
    config=yaml.safe_load((run/'config.yaml').read_text());settings=config['smdp']
    validation=dataset(run,'val');started=time.perf_counter();payload=None;history=[]
    if resume is not None:
        payload=torch.load(resume,map_location='cpu',weights_only=False)
        history=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines() if line.strip()]
        if not history or history[-1]!=payload['record']:
            raise RuntimeError('Resume must use the last complete recorded epoch, never roll back published progress')
        expected=copy.deepcopy(config);got=copy.deepcopy(payload['config'])
        # Stage-local paths/budgets differ; model and data must remain identical.
        if expected['model']!=got['model'] or expected['data']!=got['data']:
            raise RuntimeError('Resume model/data configuration mismatch')
        elapsed_since_start=time.time()-json.loads((run/'training_started.json').read_text())['started_unix']
        started-=elapsed_since_start
        receipt=dict(time_unix=time.time(),checkpoint=str(Path(resume).resolve()),
            record=payload['record'],compile_modes=compile_modes,hardware=torch.cuda.get_device_name(),
            previous_batch_size=payload['config']['training']['batch_size'],
            resumed_batch_size=config['training']['batch_size'],
            elapsed_since_last_completed_epoch_s=elapsed_since_start-payload['record']['wall_seconds'],
            source_version=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        with (run/'resumptions.jsonl').open('a') as stream: stream.write(json.dumps(receipt)+'\n')
    else:
        (run/'source_version.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
        (run/'training_started.json').write_text(json.dumps(dict(started_unix=time.time(),pretrained_weights=False,
            hardware=torch.cuda.get_device_name(),seed=config['training']['seed']),indent=2))
    global_best=min((r['selection_score'] for r in history),default=float('inf'));previous=None
    stages=[('segment',0,settings['segment_max_epochs'],settings['segment_patience'])]+[
        (f'rollout{k}',k,settings['rollout_stage_epochs'],settings['rollout_patience']) for k in settings['rollout_stages']]
    original_handler=upstream.CheckpointHandler
    reached_resume=payload is None
    for stage,steps,epochs,patience in stages:
        if not reached_resume:
            if stage!=payload['record']['stage']: continue
            reached_resume=True
        restoring=payload is not None and stage==payload['record']['stage']
        stage_dir=run/stage;stage_dir.mkdir(exist_ok=restoring)
        cfg=copy.deepcopy(config);cfg['training']['num_epochs']=epochs
        cfg['training']['checkpoint_settings']['root']=str(stage_dir/'checkpoints')
        cfg['training']['tensorboard_settings']['log_dir']=str(stage_dir/'tensorboard')
        if steps: cfg['training']['optimizer_settings']['lr']=.0003
        trainer=SegmentTrainer(copy.deepcopy(cfg));trainer.settings=settings;trainer.stage=stage
        trainer.rollout_steps=steps;trainer.started=started
        if compile_modes: trainer.model._int_out.MotionModes.enable_compilation()
        if previous is not None: trainer.model.load_state_dict(torch.load(previous,weights_only=False)['model'])
        if not restoring: trainer.dump_config_to_yaml(str(stage_dir/'resolved.yaml'))
        else: trainer.dump_config_to_yaml(str(stage_dir/f'resolved_resume_{int(time.time())}.yaml'))
        prior=[r for r in history if r['stage']==stage]
        stage_best=min((r['selection_score'] for r in prior),default=float('inf'))
        best_epoch=next((r['epoch'] for r in prior if r['selection_score']==stage_best),-1)
        upstream.CheckpointHandler=resume_handler(original_handler,payload,prior) if restoring else original_handler
        trainer.restart_latest=restoring
        def update(*,epoch,loss,model,lr,maes):
            nonlocal global_best,stage_best,best_epoch
            metrics=validate(model,validation,trainer.device,settings['validation_rollout_samples'])
            record=dict(stage=stage,epoch=epoch,train_loss=loss,train_rollout_loss=trainer.last_rollout_training_loss,
                batch_size=trainer.batch_size,optimizer_steps_per_epoch=(len(trainer.dataset)+trainer.batch_size-1)//trainer.batch_size,
                lr=lr,wall_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                parameters=sum(p.numel() for p in model.parameters()),**metrics)
            history.append(record)
            with (run/'training.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            for name,value in [('train',loss),('validation',metrics['segment_loss']),('selection',metrics['selection_score'])]:
                trainer.tensorboard.writer.add_scalar(name,value,epoch)
            trainer.tensorboard.writer.flush()
            print(json.dumps(record),flush=True)
            score=metrics['selection_score']
            if not torch.isfinite(torch.tensor(score)): raise FloatingPointError('Validation diverged')
            if score<stage_best:
                stage_best=score;best_epoch=epoch;save_checkpoint(stage_dir/'best.pt',trainer,cfg,record)
            if score<global_best:
                global_best=score;save_checkpoint(run/'best.pt',trainer,cfg,record)
            save_checkpoint(stage_dir/'last.pt',trainer,cfg,record)
            if epoch-best_epoch>=patience: raise StageFinished()
            return score
        trainer.tensorboard.update=update
        try:
            if not restoring or payload['record']['epoch']-best_epoch<patience: trainer.train()
        except StageFinished: print(f'{stage}: validation early stop',flush=True)
        finally: upstream.CheckpointHandler=original_handler
        trainer.tensorboard.writer.close()
        previous=stage_dir/'best.pt'
        del trainer;torch.cuda.empty_cache()
    (run/'training_complete.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started,
        best_selection_score=global_best,epochs=len(history),checkpoint=str(run/'best.pt'),
        initialization='from_scratch',model_selection='validation_only_all_stages'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--resume',type=Path);p.add_argument('--compile-modes',action='store_true')
    a=p.parse_args();main(a.run,a.resume,a.compile_modes)
