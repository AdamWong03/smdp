"""Composed CSVR rollout training for a force-free propagator.

The target is the stationary measure of the *composed* map (propagator + CSVR),
not the one-step conditional mean.  A short differentiable CSVR rollout is
started from reference NVT states; the ensemble moments at the final state are
penalized against equipartition.  An endpoint anchor on the first step keeps
the one-step dynamics from collapsing.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from motion_loss import decompose_modes
from periodic import PeriodicGraphBuilder
from relational_inference import load_relational, vesin_graph
from relational_transition import endpoint_objective

torch.set_default_dtype(torch.float32)
KINETIC_TO_EV = 103.64269
KB_EV_PER_K = 8.617333262e-5


class NVTPairs(Dataset):
    def __init__(self, path, split, limit=None):
        with h5py.File(path, 'r') as handle:
            group = handle[split]
            self.q0 = group['positions_initial_A'][:limit].astype(np.float32)
            displacement_key = ('displacements_A' if 'displacements_A' in group
                                else 'delta_positions_A')
            self.d = group[displacement_key][:limit].astype(np.float32)
            self.v0 = group['velocities_initial_A_per_fs'][:limit].astype(np.float32)
            self.v1 = group['velocities_final_A_per_fs'][:limit].astype(np.float32)
            cell = np.asarray(handle['cell_A'], dtype=np.float32)
            self.cell = cell[0] if cell.ndim == 3 else cell
            self.z = np.asarray(handle['atomic_numbers'][:], dtype=np.int64)
            self.masses = np.asarray(handle['masses_u'][:], dtype=np.float32)
            self.molecules = np.asarray(handle['molecule_ids'][:], dtype=np.int64)

    def __len__(self):
        return len(self.q0)

    def __getitem__(self, index):
        return dict(q0=self.q0[index], d=self.d[index], v0=self.v0[index],
                    v1=self.v1[index], cell=self.cell, z=self.z,
                    masses=self.masses, molecules=self.molecules)


def make_batch(items, model, builder, device, with_targets=True):
    graphs = []
    for item in items:
        graph = vesin_graph(item['q0'], item['v0'], item['cell'], item['z'],
                            item['masses'], model, builder, device)
        if with_targets:
            graph.displacements = torch.as_tensor(item['d'], device=device)
            graph.update_velocities = torch.as_tensor(item['v1'], device=device)
        graph.molecule_ids = torch.as_tensor(item['molecules'], dtype=torch.long,
                                             device=device)
        graphs.append(graph)
    return Batch.from_data_list(graphs)


def csvr_step(velocities, masses, generator=None, dt=5.0, tau=50.0, temperature=300.0):
    """Deterministic global thermostat for the training rollout.

    The inference protocol uses stochastic CSVR; for the differentiable training
    rollout a deterministic rescaling is more stable and has the same mean
    temperature target.  The mode partition is constrained by the ensemble loss,
    not by this global rescaling.
    """
    dof = 3*len(masses) - 3
    com = (masses[:, None]*velocities).sum(0)/masses.sum()
    internal = velocities - com
    kinetic = 0.5*KINETIC_TO_EV*(masses[:, None]*internal.square()).sum()
    current = 2*kinetic/(dof*KB_EV_PER_K)
    c = math.exp(-dt/tau)
    alpha = torch.sqrt(torch.clamp(
        1.0 + (temperature/current - 1.0)*(1.0-c), min=0.01, max=4.0))
    return com + alpha*internal


def ensemble_loss(model, batch, builder, generator, steps, device, checkpointing=False):
    """Roll out `steps` CSVR steps; return endpoint anchor + ensemble moments."""
    output = model(batch)
    endpoint, _ = endpoint_objective(model, output, batch.displacements,
                                     batch.update_velocities, proposal_weight=0.02)
    q = output['positions']
    v = output['velocities']
    masses = batch.atomic_masses.reshape(-1)
    molecules = batch.molecule_ids.reshape(-1)
    cells = batch.cell
    num_graphs = int(batch.num_graphs)
    v_parts = []
    for index in range(num_graphs):
        mask = batch.batch == index
        v_parts.append(csvr_step(v[mask], masses[mask], generator))
    v = torch.cat(v_parts, 0)
    current_batch = batch
    for _ in range(steps-1):
        graphs = []
        for index in range(num_graphs):
            mask = current_batch.batch == index
            graph = vesin_graph(q[mask], v[mask], cells[index],
                                current_batch.atomic_numbers[mask].reshape(-1).cpu().numpy(),
                                masses[mask].cpu().numpy(), model, builder, device)
            graph.molecule_ids = current_batch.molecule_ids[mask].clone()
            graphs.append(graph)
        next_batch = Batch.from_data_list(graphs)
        if checkpointing:
            def forward_pair(graph):
                out = model(graph)
                return out['positions'], out['velocities']
            q, v = checkpoint_fn(forward_pair, next_batch, use_reentrant=False)
        else:
            output = model(next_batch)
            q = output['positions']
            v = output['velocities']
        v_parts = []
        for index in range(num_graphs):
            mask = next_batch.batch == index
            v_parts.append(csvr_step(v[mask], masses[mask], generator))
        v = torch.cat(v_parts, 0)
        current_batch = next_batch
    ke = {key: [] for key in ('translation', 'rotation', 'internal')}
    temperatures = []
    for index in range(num_graphs):
        mask = current_batch.batch == index
        part = decompose_modes(q[mask], v[mask], masses[mask], molecules[mask],
                               cells[index])
        total = 0.0
        for key in ke:
            value = 0.5*KINETIC_TO_EV*torch.sum(
                masses[mask][:, None]*part[key].double().square())
            ke[key].append(value)
            total = total + value
        dof = 3*int(mask.sum()) - 3
        temperatures.append(2*total/(dof*KB_EV_PER_K))
    targets = {'translation': 0.5*(3*64-3)*KB_EV_PER_K*300.0,
               'rotation': 0.5*(3*64)*KB_EV_PER_K*300.0,
               'internal': 0.5*(3*64)*KB_EV_PER_K*300.0}
    temperatures = torch.stack(temperatures)
    valid = torch.isfinite(temperatures) & (temperatures < 450.0)
    if int(valid.sum()) == 0:
        valid = torch.ones_like(valid, dtype=torch.bool)
    moment = 0.0
    for key in ke:
        mean = torch.stack(ke[key])[valid].mean()
        moment = moment + ((mean-targets[key])/targets[key]).square()
    temperature = temperatures[valid].mean()
    moment = moment + ((temperature-300.0)/300.0).square()
    return endpoint, moment, temperature, ke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--rollout-steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lambda-endpoint', type=float, default=1.0)
    parser.add_argument('--lambda-ensemble', type=float, default=10.0)
    parser.add_argument('--train-limit', type=int, default=2000)
    parser.add_argument('--val-limit', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--checkpoint-rollout', action='store_true')
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--mode-aware', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if (args.run/'last.pt').exists():
        raise FileExistsError(args.run/'last.pt')
    args.run.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    device = torch.device(args.device)
    model, saved = load_relational(args.base_checkpoint, device)
    base_epoch = saved.get('epoch')
    if args.mode_aware:
        from mode_aware import ModeAwareTransition
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = ModeAwareTransition(model).to(device)
        saved = dict(config=dict(architecture='mode_aware_finetune',
                                 base_config=saved['config']), epoch=base_epoch)
        print(json.dumps(dict(trainable_parameters=sum(
            p.numel() for p in model.correction.parameters()))), flush=True)
    elif args.freeze_backbone:
        trainable = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('proposal_head' in name or 'correction_head' in name)
            trainable += parameter.numel() if parameter.requires_grad else 0
        print(json.dumps(dict(trainable_parameters=trainable)), flush=True)
    model.train()
    train = NVTPairs(args.pairs, 'train', args.train_limit)
    val = NVTPairs(args.pairs, 'val', args.val_limit)
    builder = PeriodicGraphBuilder(float(model.cfg.cutoff_angstrom))
    if args.mode_aware:
        parameters = list(model.correction.parameters())
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    metadata = dict(method='composed_csrv_rollout', base_checkpoint=str(args.base_checkpoint),
                    base_epoch=saved.get('epoch'), pairs=str(args.pairs),
                    rollout_steps=args.rollout_steps, batch_size=args.batch_size,
                    lr=args.lr, lambda_endpoint=args.lambda_endpoint,
                    lambda_ensemble=args.lambda_ensemble,
                    train_pairs=len(train), val_pairs=len(val), seed=args.seed,
                    hardware=torch.cuda.get_device_name())
    (args.run/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    order_rng = np.random.default_rng(args.seed)
    best = float('inf'); best_step = -1; started = time.time()
    for step in range(1, args.steps+1):
        indices = order_rng.integers(0, len(train), size=args.batch_size)
        items = [train[int(i)] for i in indices]
        batch = make_batch(items, model, builder, device)
        optimizer.zero_grad(set_to_none=True)
        endpoint, moment, temperature, ke = ensemble_loss(
            model, batch, builder, generator, args.rollout_steps, device,
            checkpointing=args.checkpoint_rollout)
        loss = args.lambda_endpoint*endpoint + args.lambda_ensemble*moment
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 10.0,
            error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0 or step == 1:
            record = dict(step=step, loss=float(loss.detach()),
                          endpoint=float(endpoint.detach()),
                          moment=float(moment.detach()),
                          temperature_K=float(temperature.detach()),
                          ke={k: float(torch.stack(v).mean()) for k, v in ke.items()},
                          elapsed=time.time()-started)
            print(json.dumps(record), flush=True)
        if step % 50 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                v_rng = np.random.default_rng(args.seed + 1)
                v_indices = v_rng.choice(len(val), size=min(len(val), 16), replace=False)
                v_items = [val[int(i)] for i in v_indices]
                v_batch = make_batch(v_items, model, builder, device)
                _, v_moment, v_temperature, v_ke = ensemble_loss(
                    model, v_batch, builder, generator, args.rollout_steps, device,
                    checkpointing=args.checkpoint_rollout)
            model.train()
            improved = float(v_moment) < best
            if improved:
                best = float(v_moment); best_step = step
            payload = dict(model=model.state_dict(), config=saved['config'],
                           base_checkpoint=str(args.base_checkpoint), step=step,
                           best=best, best_step=best_step,
                           validation_moment=float(v_moment),
                           validation_temperature=float(v_temperature),
                           elapsed_seconds=time.time()-started)
            torch.save(payload, args.run/'last.pt.tmp')
            (args.run/'last.pt.tmp').replace(args.run/'last.pt')
            if improved:
                torch.save(payload, args.run/'best.pt.tmp')
                (args.run/'best.pt.tmp').replace(args.run/'best.pt')
            with (args.run/'history.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=step, val_moment=float(v_moment),
                                             val_temperature=float(v_temperature),
                                             improved=improved))+'\n')
            print(json.dumps(dict(step=step, val_moment=float(v_moment),
                                  val_temperature=float(v_temperature),
                                  improved=improved)), flush=True)
    print(json.dumps(dict(state='completed', best_step=best_step, best=best)), flush=True)


if __name__ == '__main__':
    main()
