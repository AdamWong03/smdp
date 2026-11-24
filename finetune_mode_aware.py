"""Fine-tune a zero-initialized mode-aware correction on a frozen base model."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from mode_aware import ModeAwareTransition
from relational_transition import RelationalTransition, endpoint_objective
from train_relational import read_graphs

torch.set_default_dtype(torch.float32)


def evaluate(model, loader, device):
    model.eval()
    total = np.zeros(4, dtype=np.float64)
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            qe = out['displacements'] - batch.displacements
            ve = out['velocities'] - batch.update_velocities
            loss, pieces = endpoint_objective(
                model, out, batch.displacements, batch.update_velocities,
                proposal_weight=0.0)
            n = batch.num_graphs
            total += np.array([float(qe.square().mean()), float(ve.square().mean()),
                               float(qe.abs().mean()), float(ve.abs().mean())])*n
            count += n
    total /= count
    return dict(samples=count, position_RMSE_A=float(np.sqrt(total[0])),
                velocity_RMSE_A_fs=float(np.sqrt(total[1])),
                position_MAE_A=float(total[2]), velocity_MAE_A_fs=float(total[3]),
                selection_loss=float(pieces['final']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=20261121)
    args = p.parse_args()
    if (args.run/'last.pt').exists():
        raise FileExistsError(args.run/'last.pt')
    args.run.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    saved = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']
    normalization = cfg.get('normalization') or {}
    base = RelationalTransition(cfg['model'],
                                rms_targets=normalization.get('rms_targets', [0.07, 0.004]))
    base.load_state_dict(saved['model'], strict=True)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    model = ModeAwareTransition(base).to(device)
    train = read_graphs(Path(cfg['cache']), 'train', cfg)
    val = read_graphs(Path(cfg['cache']), 'val', cfg)
    loader = lambda graphs, shuffle: DataLoader(
        graphs, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=True, generator=torch.Generator().manual_seed(args.seed) if shuffle else None)
    train_loader, val_loader = loader(train, True), loader(val, False)
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=args.lr,
                                  weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train)/args.batch_size)
    total_epochs = math.ceil(args.steps/steps_per_epoch)

    def scale(step):
        warm = min(1000, max(1, args.steps//10))
        if step <= warm:
            return step/warm
        progress = min(1.0, (step-warm)/max(1, args.steps-warm))
        return 0.1 + 0.9*0.5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    metadata = dict(architecture='mode_aware_finetune', base_checkpoint=str(args.base_checkpoint),
                    base_epoch=saved.get('epoch'), base_config=cfg,
                    parameters_correction=sum(p.numel() for p in model.correction.parameters()),
                    parameters_total=sum(p.numel() for p in model.parameters()),
                    steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                    hardware=torch.cuda.get_device_name(),
                    initialized_zero_correction=True)
    (args.run/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    best = float('inf'); best_epoch = -1; step = 0
    start = time.time()
    history = []
    for epoch in range(total_epochs):
        model.train(); running = 0.0; seen = 0
        for batch in train_loader:
            if step >= args.steps:
                break
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss, pieces = endpoint_objective(
                model, out, batch.displacements, batch.update_velocities,
                proposal_weight=0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 10.0,
                                           error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            running += float(loss.detach())*batch.num_graphs; seen += batch.num_graphs
        validation = evaluate(model, val_loader, device)
        improved = validation['selection_loss'] < best
        if improved:
            best = validation['selection_loss']; best_epoch = epoch+1
        record = dict(epoch=epoch+1, step=step, train_loss=running/max(1, seen),
                      validation=validation, improved=improved)
        history.append(record)
        payload = dict(model=model.state_dict(),
                       config=dict(architecture='mode_aware_finetune', base_config=cfg),
                       base_checkpoint=str(args.base_checkpoint), epoch=epoch,
                       best=best, best_epoch=best_epoch, validation=validation,
                       elapsed_seconds=time.time()-start)
        torch.save(payload, args.run/'last.pt.tmp'); (args.run/'last.pt.tmp').replace(args.run/'last.pt')
        if improved:
            torch.save(payload, args.run/'best.pt.tmp'); (args.run/'best.pt.tmp').replace(args.run/'best.pt')
        (args.run/'history.jsonl').open('a').write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    print(json.dumps(dict(state='completed', best_epoch=best_epoch,
                          best=best, steps=step)), flush=True)


if __name__ == '__main__':
    main()
