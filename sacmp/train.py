"""Train a canonical propagator on 1--16 fs and 16/32/48 fs quartz states."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from reference_data import QuartzSegments, validate_split_manifest
from sacmp.prepare import make_model, prepare_training_anchor
from sacmp.model import SACMPConfig


def _device_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _error(out, truth_q, truth_v, sq, sv, t_axis=-1):
    q_error = ((out["positions"]-truth_q)/sq).square().mean()
    v_error = ((out["velocities"]-truth_v)/sv).square().mean()
    return q_error, v_error


def batch_loss(model, batch, times, scales, do_rollout=False):
    q, v = batch["q"], batch["v"]
    query = list(map(float, times))
    out = model(q[:, 0], v[:, 0], query)
    targets = torch.tensor(list(map(int, times)), device=q.device, dtype=torch.long)
    tq = q.index_select(1, targets).transpose(0, 1)
    tv = v.index_select(1, targets).transpose(0, 1)
    sq, sv = scales["displacement_residual_rms"], scales["velocity_delta_rms"]
    errors_q = ((out["positions"]-tq)/sq).square().mean((1, 2, 3))
    errors_v = ((out["velocities"]-tv)/sv).square().mean((1, 2, 3))
    weights = torch.ones(len(times), device=q.device)
    weights[targets==16] = 2.0
    position_loss = (errors_q*weights).sum()/weights.sum()
    velocity_loss = (errors_v*weights).sum()/weights.sum()
    loss = position_loss+velocity_loss
    result = dict(position_loss=position_loss, velocity_loss=velocity_loss)
    if do_rollout:
        if not bool(batch["rollout_valid"].all()):
            raise ValueError("Rollout batch includes invalid 48 fs continuation")
        endpoint_index = list(map(int, times)).index(16)
        curq = out["positions"][endpoint_index]
        curv = out["velocities"][endpoint_index]
        rollout_errors = []
        for idx in [17, 18]:
            out_next = model(curq, curv, [16.0])
            curq, curv = out_next["positions"][0], out_next["velocities"][0]
            lq = ((curq-q[:, idx])/sq).square().mean()
            lv = ((curv-v[:, idx])/sv).square().mean()
            rollout_errors.append(lq+lv)
        rollout_loss = torch.stack(rollout_errors).mean()
        result["rollout_loss"] = rollout_loss
        loss = .5*loss+.5*rollout_loss
    result["loss"] = loss
    result["endpoint_position_RMSE_A"] = ((out["positions"][list(times).index(16)] -
                                            q[:, 16]).square().mean()).sqrt()
    result["endpoint_velocity_RMSE_A_fs"] = ((out["velocities"][list(times).index(16)] -
                                               v[:, 16]).square().mean()).sqrt()
    return result


def run_epoch(model, loader, optimizer, scales, do_rollout, device, train, seed):
    totals = {}
    observed = 0
    model.train(train)
    for step, raw in enumerate(loader):
        batch = _device_batch(raw, device)
        if train:
            queries = [4, 8, 16, random.Random(seed+step).choice(
                [t for t in range(1, 16) if t not in (4, 8)])]
            queries = sorted(set(queries))
        else:
            queries = [1, 4, 8, 12, 16]
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            losses = batch_loss(model, batch, queries, scales, do_rollout)
        if train:
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError(f"Nonfinite training objective at batch {step}")
            losses["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5., error_if_nonfinite=True)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite parameter gradient norm")
            optimizer.step()
        count = batch["q"].shape[0]
        observed += count
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.)+float(value.detach())*count
    return {key: value/observed for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--segment-epochs", type=int, default=80)
    parser.add_argument("--rollout-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"history.jsonl").exists() and args.resume is None:
        raise FileExistsError("Output already has history: supply --resume")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    manifest = validate_split_manifest(args.data_root)
    trainset = QuartzSegments(args.data_root, "train", args.output/"cache")
    valset = QuartzSegments(args.data_root, "val", args.output/"cache")
    if args.resume:
        old = torch.load(args.resume, weights_only=False, map_location="cpu")
        bootstrap = old["bootstrap"]
        report = old["anchor_report"]
        scales = old["scales"]
    elif args.anchor:
        bootstrap = torch.load(args.anchor, weights_only=False, map_location="cpu")
        report = dict(external_training_anchor=str(args.anchor))
        scales = trainset.scales
    else:
        bootstrap, report = prepare_training_anchor(
            trainset, SACMPConfig(rank=args.rank))
        scales = trainset.scales
        torch.save(bootstrap, args.output/"anchor_train_only.pt")
        (args.output/"anchor_report.json").write_text(json.dumps(report, indent=2))
    model = make_model(bootstrap, device)
    if not args.resume:
        (args.output/"config.json").write_text(json.dumps(
            dict(model=bootstrap["config"], data_manifest=manifest,
                 train_scales=scales, anchor=report,
                 parameter_count=sum(x.numel() for x in model.parameters()),
                 seed=args.seed, independent_splits=True,
                 external_BKS_forces_during_training=False), indent=2))
    phase0, epoch0, best, stale = "segment", 0, float("inf"), 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=4, min_lr=1e-6)
    if args.resume:
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        scheduler.load_state_dict(old["scheduler"])
        phase0, epoch0, best, stale = (old["phase"], old["epoch"]+1,
                                       old["best"], old["stale"])
        torch.set_rng_state(old["torch_rng"])
        random.setstate(old["python_rng"])
        np.random.set_state(old["numpy_rng"])
        if device.type == "cuda" and old["cuda_rng"] is not None:
            torch.cuda.set_rng_state(old["cuda_rng"].cpu(), device)
    started = time.monotonic()
    for phase, max_epochs in [("segment", args.segment_epochs),
                              ("rollout", args.rollout_epochs)]:
        if phase == "segment" and phase0 == "rollout":
            continue
        do_rollout = phase == "rollout"
        if do_rollout and phase0 != "rollout":
            selected = torch.load(args.output/"best_segment.pt",
                                  weights_only=False, map_location=device)
            model.load_state_dict(selected["model"])
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-5, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=.5, patience=4, min_lr=5e-7)
            best, stale, epoch0 = float("inf"), 0, 0
        tr = Subset(trainset, torch.where(trainset.valid)[0].tolist()) if do_rollout else trainset
        va = Subset(valset, torch.where(valset.valid)[0].tolist()) if do_rollout else valset
        loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
        vloader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
        for epoch in range(epoch0, max_epochs):
            training = run_epoch(model, loader, optimizer, scales, do_rollout,
                                 device, train=True, seed=args.seed+epoch*1000000)
            validation = run_epoch(model, vloader, None, scales, do_rollout,
                                   device, train=False, seed=0)
            metric = validation["loss"]
            if not np.isfinite(metric):
                raise FloatingPointError("Nonfinite validation metric")
            scheduler.step(metric)
            improved = metric < best*(1-1e-4)
            if improved:
                best, stale = metric, 0
            else:
                stale += 1
            entry = dict(phase=phase, epoch=epoch, training=training,
                         validation=validation, best=best,
                         lr=optimizer.param_groups[0]["lr"],
                         elapsed_s=time.monotonic()-started)
            with (args.output/"history.jsonl").open("a") as stream:
                stream.write(json.dumps(entry)+"\n")
            print(json.dumps(entry), flush=True)
            checkpoint = dict(bootstrap=bootstrap, anchor_report=report,
                              scales=scales, model=model.state_dict(),
                              optimizer=optimizer.state_dict(),
                              scheduler=scheduler.state_dict(),
                              phase=phase, epoch=epoch, best=best, stale=stale,
                              torch_rng=torch.get_rng_state(),
                              python_rng=random.getstate(),
                              numpy_rng=np.random.get_state(),
                              cuda_rng=torch.cuda.get_rng_state(device)
                              if device.type=="cuda" else None)
            torch.save(checkpoint, args.output/"last.pt")
            if improved:
                torch.save(checkpoint, args.output/("best_"+phase+".pt"))
            if stale >= args.patience and epoch >= 19:
                break
        epoch0 = 0
    (args.output/"training_complete.json").write_text(json.dumps(
        dict(best_checkpoint="best_rollout.pt",
             completed=True, elapsed_s=time.monotonic()-started)))


if __name__ == "__main__":
    main()
