"""Official TrajCast model, normalization, loss and per-epoch trainer.

Only run management is added: full validation, best/last checkpoints, RNG
resume, paper early stopping, and nonfinite-gradient rejection. Apache-2.0
upstream: IBM TrajCast c52f107; Thiemann et al., NMI 8, 764–776 (2026).
No pretrained parameters are read.
"""
import argparse
import copy
import json
import math
import os
import random
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from trajcast.model.training import Trainer

from water_reference.prepare import dataset

STOP = False


def stop(*_):
    global STOP
    STOP = True


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(temp, path)


def save(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(data, temp)
    os.replace(temp, path)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = np.zeros(4)
    count = 0
    rms = model.rms_targets.detach().cpu().numpy()
    for batch in loader:
        # Native normalization mutates only this newly collated batch.
        out = model(batch.to(device))
        err = out.target-torch.cat([out.displacements, out.update_velocities], -1)
        qe, ve = err[:, :3], err[:, 3:]
        n = out.num_graphs
        totals += np.array([qe.square().mean().item(), ve.square().mean().item(),
                            qe.abs().mean().item(), ve.abs().mean().item()])*n
        count += n
    totals /= count
    return dict(samples=count, selection_loss=float(.5*(totals[0]+totals[1])),
                normalized_position_MSE=float(totals[0]), normalized_velocity_MSE=float(totals[1]),
                position_RMSE_A=float(np.sqrt(totals[0])*rms[0]),
                velocity_RMSE_A_fs=float(np.sqrt(totals[1])*rms[1]),
                position_MAE_A=float(totals[2]*rms[0]),
                velocity_MAE_A_fs=float(totals[3]*rms[1]))


def correctness(trainer):
    """Real-data equations, gradients and effective batch memory; no updates."""
    model = trainer.model
    model.eval()
    batch = next(iter(DataLoader(trainer.dataset, batch_size=2))).to(trainer.device)
    torch.cuda.reset_peak_memory_stats()
    output = model(batch.clone())
    labels = torch.cat([output.displacements, output.update_velocities], -1)
    loss = trainer.loss_function(output.target, labels)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())
    target = output.target.detach().clone()
    rms = model.rms_targets.detach()
    momentum = torch.zeros(2, 3, device=trainer.device).index_add_(
        0, batch.batch, batch.atomic_masses*target[:, 3:]*rms[1])
    assert momentum.abs().max() < 1e-10
    model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated()
    del output, labels, loss
    with torch.no_grad():
        # Test an improper orthogonal transform, including reflection.
        R, _ = torch.linalg.qr(torch.randn(3, 3, device=trainer.device))
        if torch.linalg.det(R) > 0:
            R[:, 0] *= -1
        rotated = batch.clone()
        for field in ['pos', 'velocities', 'shifts', 'displacements', 'update_velocities', 'cell']:
            rotated[field] = rotated[field]@R
        expected = torch.cat([target[:, :3]@R, target[:, 3:]@R], -1)
        equivariance = (model(rotated).target-expected).abs().max().item()
        assert equivariance < 1e-8, equivariance
        remapped = batch.clone()
        offsets = torch.zeros_like(remapped.pos)
        offsets[0] = remapped.cell.reshape(-1, 3, 3)[0, 0]
        remapped.pos += offsets
        i, j = remapped.edge_index
        remapped.shifts += offsets[i]-offsets[j]
        periodic = (model(remapped).target-target).abs().max().item()
        assert periodic < 1e-8, periodic
    return dict(effective_batch=2, gradient_finite=True, nonzero_gradients=True,
                improper_equivariance_max_normalized=equivariance,
                periodic_reimage_max_normalized=periodic,
                net_momentum_max_u_A_fs=float(momentum.abs().max()),
                peak_forward_backward_cuda_bytes=peak, optimizer_updates=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    cfg = json.loads((args.run/'config.json').read_text())
    if (args.run/'last.pt').exists() and not args.resume and not args.check:
        raise FileExistsError('Existing experiment; use --resume')
    random.seed(cfg['training']['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    trainer = Trainer(copy.deepcopy(cfg))
    if args.check:
        result = correctness(trainer)
        write_json(args.run/'correctness.json', result)
        print(json.dumps(result), flush=True)
        return
    generator = torch.Generator().manual_seed(trainer.seed)
    train_loader = DataLoader(trainer.dataset, batch_size=trainer.batch_size,
                              shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    val_loader = DataLoader(dataset(Path(cfg['data']['root']), 'val'),
                            batch_size=trainer.batch_size, shuffle=False, num_workers=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, **cfg['experiment']['scheduler'])
    epoch0, best, stale, updates, prior_seconds = 0, math.inf, 0, 0, 0.
    if args.resume:
        state = torch.load(args.run/'last.pt', map_location='cpu', weights_only=False)
        trainer.model.load_state_dict(state['model'], strict=True)
        trainer.optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch0, best, stale, updates = state['epoch'], state['best'], state['stale'], state['optimizer_steps']
        prior_seconds = state['elapsed_seconds']
        torch.set_rng_state(state['rng']); generator.set_state(state['loader_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    metadata = dict(config=cfg, resolved_native_model_config=trainer.config['model'],
        parameters=sum(p.numel() for p in trainer.model.parameters()),
        normalization_train_only=trainer.model.rms_targets.tolist(),
        hardware=torch.cuda.get_device_name(), torch_version=torch.__version__,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        initialized_from_pretrained=False, training_samples=len(trainer.dataset),
        optimizer_updates_per_epoch=len(train_loader),
        maximum_optimizer_updates=trainer.num_epochs*len(train_loader),
        native_train_method='trajcast.model.training.Trainer._train_epoch')
    write_json(args.run/'metadata.json', metadata)
    print(json.dumps(dict(event='training_started', **metadata)), flush=True)

    def finite_gradients(optimizer, a, kw):
        finite = torch.stack([p.grad.isfinite().all() for group in optimizer.param_groups
                              for p in group['params'] if p.grad is not None]).all()
        if not finite:
            raise FloatingPointError('Nonfinite native gradient; optimizer update rejected')

    def progress(optimizer, a, kw):
        nonlocal updates
        updates += 1
        if updates == 1 or updates % 25 == 0:
            record = dict(state='training', epoch_in_progress=epoch+1,
                optimizer_steps=updates, elapsed_seconds=prior_seconds+time.monotonic()-started,
                lr=optimizer.param_groups[0]['lr'],
                peak_cuda_bytes=torch.cuda.max_memory_allocated())
            write_json(args.run/'progress.json', record)
            print(json.dumps(record), flush=True)

    trainer.optimizer.register_step_pre_hook(finite_gradients)
    trainer.optimizer.register_step_post_hook(progress)
    for epoch in range(epoch0, trainer.num_epochs):
        tick = time.monotonic()
        trainer.model.train()
        torch.cuda.reset_peak_memory_stats()
        train_loss, train_maes = trainer._train_epoch(train_loader)
        validation = evaluate(trainer.model, val_loader, trainer.device)
        score = validation['selection_loss']
        if not math.isfinite(float(train_loss)) or not math.isfinite(score):
            raise FloatingPointError('Nonfinite native train/val loss')
        improved = score < best
        best, stale = (score, 0) if improved else (best, stale+1)
        scheduler.step(score)
        record = dict(epoch=epoch+1, optimizer_steps=updates,
            train_final_loss=float(train_loss), train_normalized_MAE=train_maes,
            validation=validation, lr=trainer.optimizer.param_groups[0]['lr'],
            epoch_seconds=time.monotonic()-tick, improved=improved,
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        state = dict(model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
            scheduler=scheduler.state_dict(), config=cfg,
            resolved_native_model_config=trainer.config['model'],
            rms_targets=trainer.model.rms_targets.detach().cpu(),
            epoch=epoch+1, best=best, stale=stale, optimizer_steps=updates, record=record,
            elapsed_seconds=prior_seconds+time.monotonic()-started,
            rng=torch.get_rng_state(), loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
        save(args.run/'last.pt', state)
        if improved:
            save(args.run/'best.pt', state)
        with (args.run/'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        write_json(args.run/'status.json', dict(state='paused' if STOP else 'training', **record))
        print(json.dumps(record), flush=True)
        if STOP:
            return
        if stale >= cfg['experiment']['early_stopping_patience']:
            break
    state = torch.load(args.run/'best.pt', map_location='cpu', weights_only=False)
    trainer.model.load_state_dict(state['model'], strict=True)
    test_loader = DataLoader(dataset(Path(cfg['data']['root']), 'test'),
                            batch_size=trainer.batch_size, shuffle=False, num_workers=0)
    result = evaluate(trainer.model, test_loader, trainer.device)
    result.update(checkpoint=str(args.run/'best.pt'), selected_epoch=state['epoch'],
                  selected_optimizer_steps=state['optimizer_steps'], actual_optimizer_steps=updates,
                  training_seconds=prior_seconds+time.monotonic()-started)
    write_json(args.run/'test_metrics.json', result)
    write_json(args.run/'status.json', dict(state='training_and_test_completed', **result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
