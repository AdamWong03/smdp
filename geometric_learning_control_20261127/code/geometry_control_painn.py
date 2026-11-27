"""Geometry-only PaiNN-style control for Q_even/P_even.

Inputs are local periodic geometry, species and masses only.  No global
Cartesian position, momentum, force, energy, or additional physical loss.
The model predicts Q_even (A) and P_even (u A / fs) directly.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from e3nn import o3

from control_data import load_states
from data import ParityFlowDataset


def rbf_features(r, centers, width, cutoff):
    diff = (r[:, None] - centers[None, :]) / width
    env = torch.where(r < cutoff,
                      0.5 * (torch.cos(math.pi * r / cutoff) + 1.0),
                      torch.zeros_like(r))
    return torch.exp(-diff * diff) * env[:, None]


class PaiNNLayer(nn.Module):
    def __init__(self, scalar_channels, vector_channels, rbf_channels):
        super().__init__()
        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels
        self.scalar_message = nn.Sequential(
            nn.Linear(scalar_channels + rbf_channels, 128),
            nn.SiLU(),
            nn.Linear(128, scalar_channels),
        )
        self.vector_filter = nn.Sequential(
            nn.Linear(scalar_channels + rbf_channels, 128),
            nn.SiLU(),
            nn.Linear(128, vector_channels),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * scalar_channels + vector_channels, 128),
            nn.SiLU(),
            nn.Linear(128, 2 * scalar_channels),
        )
        self.gate = nn.Linear(2 * scalar_channels, vector_channels)
        self.norm = nn.LayerNorm(scalar_channels)

    def forward(self, s, v, edge_index, edge_vec, rbf):
        sender, receiver = edge_index
        r = torch.linalg.vector_norm(edge_vec, dim=-1).clamp_min(1e-8)
        unit = edge_vec / r[:, None]
        features = torch.cat([s[sender], rbf], dim=-1)
        ds = self.scalar_message(features)
        coeff = self.vector_filter(features)
        dv = coeff[:, :, None] * unit[:, None, :]
        agg_s = torch.zeros_like(s)
        agg_s.index_add_(0, receiver, ds)
        agg_v = torch.zeros_like(v)
        agg_v.index_add_(0, receiver, dv)
        degree = torch.zeros(len(s), device=s.device)
        degree.index_add_(0, receiver, torch.ones_like(receiver, dtype=s.dtype))
        agg_s = agg_s / degree.clamp_min(1.0)[:, None]
        agg_v = agg_v / degree.clamp_min(1.0)[:, None, None]
        context = torch.cat([s, agg_s,
                             (v * agg_v).sum(-1)], dim=-1)
        update = self.update(context)
        delta_s, gate_in = update.split(self.scalar_channels, dim=-1)
        s = self.norm(s + delta_s)
        gate = torch.sigmoid(self.gate(torch.cat([gate_in, s], dim=-1)))
        v = v + gate[:, :, None] * agg_v
        return s, v


class GeometryOnlyPaiNN(nn.Module):
    def __init__(self, species, scalar_channels=64, vector_channels=32,
                 layers=3, rbf_channels=16, cutoff=6.0):
        super().__init__()
        self.cutoff = float(cutoff)
        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels
        self.embedding = nn.Embedding(119, scalar_channels)
        self.register_buffer("centers", torch.linspace(0.8, cutoff, rbf_channels))
        self.register_buffer("width", torch.tensor(cutoff / rbf_channels))
        self.layers = nn.ModuleList([
            PaiNNLayer(scalar_channels, vector_channels, rbf_channels)
            for _ in range(layers)
        ])
        self.q_head = o3.Linear(f"{vector_channels}x1o", "1o")
        self.p_head = o3.Linear(f"{vector_channels}x1o", "1o")

    def forward(self, z, masses, edge_index, edge_vec):
        s = self.embedding(z) + torch.log(masses)[:, None]
        v = torch.zeros(len(z), self.vector_channels, 3,
                        dtype=s.dtype, device=s.device)
        r = torch.linalg.vector_norm(edge_vec, dim=-1)
        rbf = rbf_features(r, self.centers, self.width, self.cutoff)
        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_vec, rbf)
        flat = v.reshape(len(z), -1)
        return self.q_head(flat), self.p_head(flat)


def collate(states, ds, indices, device):
    qs, ps, zs, ms, eis, evs = [], [], [], [], [], []
    offset = 0
    for i in indices:
        s = {k: (v[i] if torch.is_tensor(v) and v.shape[0] == len(states["q0"])
                 else v) for k, v in states.items()}
        dataset_index = int(states["indices"][i])
        e0, e1 = int(ds.edge_offsets[dataset_index]), \
            int(ds.edge_offsets[dataset_index + 1])
        edges = torch.as_tensor(ds.edge_index[:, e0:e1], dtype=torch.long,
                                device=device) + offset
        ev = torch.as_tensor(ds.edge_vec[e0:e1], dtype=torch.float32,
                             device=device)
        n = s["q0"].shape[0]
        qs.append(s["Q_even"].to(device))
        ps.append(s["P_even"].to(device))
        zs.append(s["atomic_numbers"].to(device))
        ms.append(s["masses"].to(device))
        eis.append(edges)
        evs.append(ev)
        offset += n
    return (torch.cat(qs), torch.cat(ps), torch.cat(zs), torch.cat(ms),
            torch.cat(eis, 1), torch.cat(evs, 0))


@torch.no_grad()
def evaluate(model, states, ds, indices, device, scales):
    model.eval()
    q, p, z, m, ei, ev = collate(states, ds, indices, device)
    qp, pp = model(z, m, ei, ev)
    out = {}
    for name, pred, target, scale in (
            ("Q_even", qp, q, scales[0]), ("P_even", pp, p, scales[1])):
        pr = (pred / scale).reshape(-1).double()
        tr = (target / scale).reshape(-1).double()
        mse = float(((pr - tr) ** 2).mean())
        zero = float((tr * tr).mean())
        pc, tc = pr - pr.mean(), tr - tr.mean()
        out[name] = dict(
            mse=mse, zero=zero, mse_over_zero=mse / max(zero, 1e-30),
            r2=1.0 - mse / max(zero, 1e-30),
            corr=float((pc * tc).sum() / (pc.norm() * tc.norm()).clamp_min(1e-30)),
            pred_rms=float(pr.pow(2).mean().sqrt()),
            target_rms=float(tr.pow(2).mean().sqrt()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["water", "quartz"], required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20261127)
    device = torch.device(args.device)
    train = load_states(args.index, "train", 64, 20261127, torch.float32)
    test = load_states(args.index, "test", 64, 20261127, torch.float32)
    ds = ParityFlowDataset(args.index, "train")
    ds_test = ParityFlowDataset(args.index, "test")
    train_idx = list(range(len(train["q0"])))
    test_idx = list(range(len(test["q0"])))
    cutoff = float(train["cutoff"])

    def q_scale(states):
        return float(states["Q_even"].pow(2).mean().sqrt())

    def p_scale(states):
        return float(states["P_even"].pow(2).mean().sqrt())

    scales = [q_scale(train), p_scale(train)]
    model = GeometryOnlyPaiNN(2, cutoff=cutoff).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                            amsgrad=True)

    def factor(step):
        if step < args.warmup:
            return (step + 1) / max(args.warmup, 1)
        frac = (step - args.warmup) / max(args.updates - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    log = open(args.out.with_suffix(".jsonl"), "w")
    t0 = time.time()
    history = []
    order = torch.randperm(len(train_idx))
    cursor = 0
    for update in range(args.updates):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(train_idx))
            cursor = 0
        batch_idx = [train_idx[int(i)] for i in order[cursor:cursor + args.batch_size]]
        cursor += args.batch_size
        q, p, z, m, ei, ev = collate(train, ds, batch_idx, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        qp, pp = model(z, m, ei, ev)
        loss = (((qp - q) / scales[0]) ** 2).mean() + \
               (((pp - p) / scales[1]) ** 2).mean()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if update % args.eval_every == 0 or update == args.updates - 1:
            tr = evaluate(model, train, ds, train_idx, device, scales)
            te = evaluate(model, test, ds_test, test_idx, device, scales)
            rec = dict(update=update, loss=float(loss), grad_norm=float(grad),
                       lr=opt.param_groups[0]["lr"],
                       elapsed_s=time.time() - t0, train=tr, test=te)
            history.append(rec)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    log.close()
    result = dict(system=args.system, status="completed", params=params,
                  cutoff=cutoff, updates=args.updates, lr=args.lr,
                  batch_size=args.batch_size, scales=scales,
                  history=history, final=history[-1],
                  elapsed_s=time.time() - t0,
                  source="geometry-only PaiNN-style control; no global position, "
                         "no momentum, no force/energy labels")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("final", json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()
