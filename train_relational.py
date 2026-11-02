"""Formal endpoint training on existing Quartz 30 fs graphs. No MD generation."""
import argparse
import json
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from trajcast.data.dataset import AtomicGraphDataset

from relational_transition import RelationalConfig, RelationalTransition, endpoint_objective

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def read_graphs(cache, split):
    path = cache / f'quartz30_{split};P:64.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    # Upstream encodes dtype in filenames. Read existing f64 cache, then cast.
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dataset = AtomicGraphDataset(root=str(cache), name=f'quartz30_{split}',
                                    cutoff_radius=4.5, files=[], atom_type_mapper={8:0,14:1})
        graphs = [dataset[i].clone().apply(lambda x: x.float() if x.is_floating_point() else x)
                  for i in range(len(dataset))]
    finally:
        torch.set_default_dtype(old)
    if len(graphs) != {'train':5000, 'val':1250, 'test':1000}[split]:
        raise ValueError(f'Incomplete formal {split} dataset')
    return graphs


@torch.no_grad()
def evaluate(model, loader, device, precision='fp32'):
    model.eval()
    total = np.zeros(6, dtype=np.float64)
    count = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == 'bf16'):
            out = model(batch)
            qe = out['displacements']-batch.displacements
            ve = out['velocities']-batch.update_velocities
            loss, pieces = endpoint_objective(model, out, batch.displacements, batch.update_velocities)
        n = batch.num_graphs
        total += np.array([float(qe.square().mean()), float(ve.square().mean()),
                           float(qe.abs().mean()), float(ve.abs().mean()),
                           float(pieces['final']), float(pieces['proposal'])])*n
        count += n
    total /= count
    return dict(samples=count, position_RMSE_A=float(np.sqrt(total[0])),
                velocity_RMSE_A_fs=float(np.sqrt(total[1])), position_MAE_A=float(total[2]),
                velocity_MAE_A_fs=float(total[3]), selection_loss=float(total[4]),
                proposal_loss=float(total[5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = json.loads((args.run/'config.json').read_text())
    if cfg['precision'] not in {'fp32', 'bf16'}:
        raise ValueError('precision must be fp32 or bf16')
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    # FP32 initially; no unverified mixed precision / TF32 change to the model.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    start = time.time()
    train = read_graphs(Path(cfg['cache']), 'train')
    val = read_graphs(Path(cfg['cache']), 'val')
    sq = torch.cat([g.displacements for g in train]).square().mean().sqrt().item()
    sv = torch.cat([g.update_velocities for g in train]).square().mean().sqrt().item()
    model = RelationalTransition(cfg['model'], (sq,sv)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                  weight_decay=0., amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.8, patience=20, min_lr=1e-6)
    generator = torch.Generator().manual_seed(seed)
    def loader(graphs, shuffle):
        return DataLoader(graphs, batch_size=cfg['batch_size'], shuffle=shuffle,
                          num_workers=0, pin_memory=True, generator=generator if shuffle else None)
    train_loader, val_loader = loader(train, True), loader(val, False)
    first, best, stale, previous_seconds, optimizer_steps = 0, float('inf'), 0, 0., 0
    if args.resume:
        saved = torch.load(args.run/'last.pt', weights_only=False, map_location='cpu')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        first, best, stale = saved['epoch']+1, saved['best'], saved['stale']
        previous_seconds = saved['elapsed_seconds']
        optimizer_steps = saved.get('optimizer_steps', first * len(train_loader))
        torch.set_rng_state(saved['rng']); generator.set_state(saved['loader_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    elif (args.run/'last.pt').exists():
        raise FileExistsError('Use --resume; never overwrite an experiment')
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    metadata = dict(config=cfg, normalization_train_only=[sq,sv], parameters=sum(p.numel() for p in model.parameters()),
                    hardware=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    train_samples=len(train), val_samples=len(val), initialized_from_pretrained=False,
                    optimizer_steps_per_epoch=len(train_loader))
    (args.run/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(event='training_started', **metadata)), flush=True)
    for epoch in range(first,cfg['max_epochs']):
        tic = time.perf_counter(); model.train()
        running = torch.zeros(3,device=device); seen=0
        torch.cuda.reset_peak_memory_stats()
        for batch in train_loader:
            batch=batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=cfg['precision'] == 'bf16'):
                out=model(batch)
                loss, pieces=endpoint_objective(model,out,batch.displacements,batch.update_velocities,
                                                proposal_weight=cfg['proposal_weight'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            n=batch.num_graphs; seen+=n
            running+=torch.stack([loss.detach(),pieces['final'],pieces['proposal']])*n
        torch.cuda.synchronize(); train_seconds=time.perf_counter()-tic
        validation=evaluate(model,val_loader,device,cfg['precision'])
        score=validation['selection_loss']; improved=score<best
        if improved: best=score; stale=0
        else: stale+=1
        scheduler.step(score)
        record=dict(epoch=epoch+1,train_total_loss=float(running[0]/seen),
                    train_final_loss=float(running[1]/seen), train_proposal_loss=float(running[2]/seen),
                    validation=validation,train_seconds=train_seconds,epoch_seconds=time.perf_counter()-tic,
                    samples_per_second=seen/train_seconds,lr=optimizer.param_groups[0]['lr'],
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(),improved=improved,
                    optimizer_steps=optimizer_steps)
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                     config=cfg,epoch=epoch,best=best,stale=stale,record=record,
                     optimizer_steps=optimizer_steps,
                     elapsed_seconds=previous_seconds+time.time()-start,rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state(),
                     numpy_rng=np.random.get_state(),python_rng=random.getstate())
        atomic_save(payload,args.run/'last.pt')
        if improved: atomic_save(payload,args.run/'best.pt')
        with (args.run/'history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        (args.run/'status.json').write_text(json.dumps(dict(state='training',**record),indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if STOP:
            (args.run/'status.json').write_text(json.dumps(dict(state='paused',last_epoch=epoch+1))+'\n')
            return
        if stale>=cfg['early_stopping_patience']: break
    saved=torch.load(args.run/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'])
    # First test access occurs AFTER validation-only model selection is finished.
    test=read_graphs(Path(cfg['cache']),'test')
    result=evaluate(model,loader(test,False),device,cfg['precision'])
    result.update(checkpoint=str(args.run/'best.pt'),selected_epoch=saved['epoch']+1,
                  selected_optimizer_steps=saved.get('optimizer_steps'),
                  actual_training_seconds=previous_seconds+time.time()-start)
    (args.run/'test_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.run/'status.json').write_text(json.dumps(dict(state='training_and_test_completed',**result),indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
