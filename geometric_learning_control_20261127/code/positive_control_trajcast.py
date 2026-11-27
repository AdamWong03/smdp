"""Bounded TRAIN-only positive control using the official TrajCast model.

This is not a formal external-baseline reproduction.  It fixes the 64 centered
TRAIN states used by the ParityFlow diagnostics and asks whether the official
EfficientTrajCast architecture can overfit both physical successors:
  (q,  v) -> (q_plus,  v_plus)
  (q, -v) -> (q_minus, -v_minus)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch_geometric.data import Batch

from control_data import load_states, state_i, trajcast_graph

WATER_CFG = dict(
    precision=64, num_chem_elements=2, edge_cutoff=6.0,
    num_edge_rbf=8, num_edge_poly_cutoff=6, num_vel_rbf=8, vel_max=0.135,
    max_rotation_order=2, num_hidden_channels=64, num_mp_layers=4,
    edge_mlp_kwargs=dict(n_neurons=[64, 64, 64], activation="silu"),
    vel_mlp_kwargs=dict(n_neurons=[64, 64, 64], activation="silu"),
    nl_gate_kwargs=dict(activation_scalars=dict(o="tanh", e="silu"),
                        activation_gates=dict(e="silu")),
    conserve_ang_mom=False, o3_backend="e3nn",
    net_lin_mom=[0.0, 0.0, 0.0], units="real",
)


def make_pairs(states, cutoff, horizon, dtype):
    pairs = []
    for i in range(len(states["q0"])):
        s = state_i(states, i)
        gf = trajcast_graph(s, cutoff, horizon, device="cpu", dtype=dtype)
        sr = dict(s)
        sr["v0"] = -s["v0"]
        gr = trajcast_graph(sr, cutoff, horizon, device="cpu", dtype=dtype)
        pairs.append(dict(
            forward=gf, reverse=gr,
            disp_plus=s["disp_plus"], disp_minus=s["disp_minus"],
            v_plus=s["v_plus"], v_minus=s["v_minus"],
            atomic_numbers=s["atomic_numbers"],
        ))
    return pairs


def flatten_pairs(pairs):
    rows = []
    for p in pairs:
        rows.append(dict(graph=p["forward"], disp=p["disp_plus"],
                         vel=p["v_plus"], direction="forward"))
        rows.append(dict(graph=p["reverse"], disp=p["disp_minus"],
                         vel=p["v_minus"], direction="reverse"))
    return rows


def batch_rows(rows, device, dtype):
    graphs = [r["graph"].to(device) for r in rows]
    batch = Batch.from_data_list(graphs).to(device)
    disp = torch.stack([r["disp"] for r in rows], 0).to(device, dtype)
    vel = torch.stack([r["vel"] for r in rows], 0).to(device, dtype)
    return batch, disp, vel


def decode(output, rms, dtype):
    pred = output.target
    return pred[:, :3] * rms[0], pred[:, 3:6] * rms[1]


def split_graph(tensor, ptr):
    return [tensor[ptr[i]:ptr[i + 1]] for i in range(len(ptr) - 1)]


@torch.no_grad()
def evaluate(model, pairs, rms, device, dtype, batch_pairs=2,
             return_arrays=False):
    model.eval()
    out = {k: [] for k in ("df", "dr", "vf", "vr",
                           "df_t", "dr_t", "vf_t", "vr_t")}
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start:start + batch_pairs]
        rows = []
        for p in chunk:
            rows.append(dict(graph=p["forward"], disp=p["disp_plus"],
                             vel=p["v_plus"]))
            rows.append(dict(graph=p["reverse"], disp=p["disp_minus"],
                             vel=p["v_minus"]))
        batch, disp_t, vel_t = batch_rows(rows, device, dtype)
        pred_disp, pred_vel = decode(model(batch), rms, dtype)
        ptr = batch.ptr
        n = len(chunk)
        for j, p in enumerate(chunk):
            df = pred_disp[ptr[2 * j]:ptr[2 * j + 1]]
            dr = pred_disp[ptr[2 * j + 1]:ptr[2 * j + 2]]
            vf = pred_vel[ptr[2 * j]:ptr[2 * j + 1]]
            vr = pred_vel[ptr[2 * j + 1]:ptr[2 * j + 2]]
            out["df"].append(df.cpu())
            out["dr"].append(dr.cpu())
            out["vf"].append(vf.cpu())
            out["vr"].append(vr.cpu())
            out["df_t"].append(p["disp_plus"].cpu())
            out["dr_t"].append(p["disp_minus"].cpu())
            out["vf_t"].append(p["v_plus"].cpu())
            out["vr_t"].append(p["v_minus"].cpu())
    cat = {k: torch.cat(v) for k, v in out.items()}
    qe = (cat["df"] + cat["dr"]) / 2
    qo = (cat["df"] - cat["dr"]) / 2
    pe = (cat["vf"] + cat["vr"]) / 2
    po = (cat["vf"] - cat["vr"]) / 2
    qe_t = (cat["df_t"] + cat["dr_t"]) / 2
    qo_t = (cat["df_t"] - cat["dr_t"]) / 2
    pe_t = (cat["vf_t"] - cat["vr_t"]) / 2
    po_t = (cat["vf_t"] + cat["vr_t"]) / 2

    def stat(pred, target):
        p = pred.reshape(-1).double()
        t = target.reshape(-1).double()
        mse = float(((p - t) ** 2).mean())
        zero = float((t * t).mean())
        pc = p - p.mean()
        tc = t - t.mean()
        corr = float((pc * tc).sum() / (pc.norm() * tc.norm()).clamp_min(1e-30))
        return dict(mse=mse, zero=zero, mse_over_zero=mse / max(zero, 1e-30),
                    r2=1.0 - mse / max(zero, 1e-30), corr=corr,
                    target_rms=float(t.pow(2).mean().sqrt()),
                    pred_rms=float(p.pow(2).mean().sqrt()))

    fwd_disp = float(((cat["df"] - cat["df_t"]) ** 2).mean())
    rev_disp = float(((cat["dr"] - cat["dr_t"]) ** 2).mean())
    fwd_vel = float(((cat["vf"] - cat["vf_t"]) ** 2).mean())
    rev_vel = float(((cat["vr"] - cat["vr_t"]) ** 2).mean())
    parity_disp = self_mse = 0.5 * (fwd_disp + rev_disp)
    parity_vel = 0.5 * (fwd_vel + rev_vel)
    parity_sum_disp = stat(qe, qe_t)["mse"] + stat(qo, qo_t)["mse"]
    parity_sum_vel = stat(pe, pe_t)["mse"] + stat(po, po_t)["mse"]
    result = dict(
        forward=dict(disp_mse=fwd_disp, vel_mse=fwd_vel),
        reverse=dict(disp_mse=rev_disp, vel_mse=rev_vel),
        paired=dict(disp_half_sum=self_mse, vel_half_sum=parity_vel),
        parity=dict(Q_even=stat(qe, qe_t), Q_odd=stat(qo, qo_t),
                    P_even=stat(pe, pe_t), P_odd=stat(po, po_t)),
        equivalence=dict(
            disp_abs_diff=abs(parity_sum_disp - self_mse),
            vel_abs_diff=abs(parity_sum_vel - parity_vel),
            disp_identity=parity_sum_disp,
            disp_half_pair=self_mse,
            vel_identity=parity_sum_vel,
            vel_half_pair=parity_vel,
        ),
    )
    if return_arrays:
        result["arrays"] = dict(
            qe=qe, qo=qo, pe=pe, po=po,
            qe_t=qe_t, qo_t=qo_t, pe_t=pe_t, po_t=po_t)
    return result


def group_grad_norm(model):
    groups = {}
    for name, module in model.named_children():
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum().item()
        groups[name] = total ** 0.5
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    groups["total"] = total ** 0.5
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["water", "quartz"], required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--parity", action="store_true",
                        help="run the optional ParityFlow optimizer/precision arm")
    args = parser.parse_args()

    if args.parity:
        dtype = torch.float32
        torch.set_default_dtype(torch.float32)
        args.lr = 2e-4
        # Micro-batch 4 on the shared GPU; documented as a memory-bounded
        # deviation from the formal ParityFlow batch of 8.
        args.batch_size = 4
    else:
        dtype = torch.float64
        torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_states = load_states(args.index, "train", 64, args.seed, dtype)
    test_states = load_states(args.index, "test", 64, args.seed, dtype)
    cutoff = float(train_states["cutoff"])
    horizon = float(train_states["horizon_fs"])
    train_pairs = make_pairs(train_states, cutoff, horizon, dtype)
    test_pairs = make_pairs(test_states, cutoff, horizon, dtype)
    train_rows = flatten_pairs(train_pairs)

    # TRAIN-only normalization from both physical successors.
    disp_all = torch.cat([r["disp"].reshape(-1) for r in train_rows])
    vel_all = torch.cat([r["vel"].reshape(-1) for r in train_rows])
    rms = [float(disp_all.pow(2).mean().sqrt()),
           float(vel_all.pow(2).mean().sqrt())]
    print("rms", rms, flush=True)

    cfg = deepcopy(WATER_CFG)
    cfg["edge_cutoff"] = cutoff
    cfg["num_chem_elements"] = 2
    cfg["precision"] = 32 if args.parity else 64
    if args.system == "quartz":
        vmax = float(torch.cat([
            s["v0"].norm(dim=-1) for s in
            [state_i(train_states, i) for i in range(len(train_states["q0"]))]])
            .max())
        cfg["vel_max"] = max(0.01, 1.2 * vmax)
    cfg["avg_num_neighbors"] = float(
        sum(p["forward"].num_edges / p["forward"].num_nodes
            for p in train_pairs) / len(train_pairs))

    from trajcast.model.models import EfficientTrajCastModel
    model = EfficientTrajCastModel(
        config=cfg, predicted_fields=["displacements", "update_velocities"],
        rms_targets=rms, mean_targets=[0.0, 0.0]).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = (torch.optim.AdamW(model.parameters(), lr=args.lr,
                             amsgrad=True, weight_decay=0.0)
           if args.parity else
           torch.optim.Adam(model.parameters(), lr=args.lr,
                            amsgrad=True, weight_decay=0.0))
    clip = 10.0 if args.parity else float("inf")
    start_update = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_update = int(ckpt["updates"])
        rms = ckpt["rms"]
        print("resumed", start_update, flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".jsonl")
    log_f = open(log_path, "a")
    t0 = time.time()
    running = 0.0
    running_n = 0
    exposed = dict(forward=0, reverse=0, atoms=0, updates=0)
    history = []
    perm = torch.randperm(len(train_rows))
    cursor = 0
    for update in range(start_update, args.updates):
        if cursor + args.batch_size > len(perm):
            perm = torch.randperm(len(train_rows))
            cursor = 0
        rows = [train_rows[int(i)] for i in perm[cursor:cursor + args.batch_size]]
        cursor += args.batch_size
        batch, disp_t, vel_t = batch_rows(rows, device, dtype)
        ref = torch.cat([disp_t.reshape(-1, 3) / rms[0],
                         vel_t.reshape(-1, 3) / rms[1]], dim=1)
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(batch).target
        loss = torch.nn.functional.mse_loss(pred, ref)
        loss.backward()
        grad = group_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        exposed["updates"] += 1
        for r in rows:
            exposed[r["direction"]] += 1
            exposed["atoms"] += int(r["graph"].num_nodes)
        running += float(loss)
        running_n += 1

        if update % args.eval_every == 0 or update == args.updates - 1:
            train_eval = evaluate(model, train_pairs, rms, device, dtype)
            test_eval = evaluate(model, test_pairs, rms, device, dtype)
            rec = dict(update=update, loss=running / max(running_n, 1),
                       lr=args.lr, elapsed_s=time.time() - t0,
                       updates_per_s=(update - start_update + 1) /
                       max(time.time() - t0, 1e-9),
                       grad_norm=grad["total"], grad_groups=grad,
                       train=train_eval, test=test_eval,
                       exposed=deepcopy(exposed))
            history.append(rec)
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            running = 0.0
            running_n = 0
            if (update + 1) % 500 == 0 or update == args.updates - 1:
                torch.save(dict(model=model.state_dict(),
                                optimizer=opt.state_dict(), updates=update + 1,
                                rms=rms, config=cfg), out_path.with_suffix(".pt"))
    log_f.close()
    result = dict(
        system=args.system, status="completed", params=params,
        config=cfg, rms=rms, updates=args.updates,
        seed=args.seed, precision=32 if args.parity else 64,
        optimizer=("AdamW(amsgrad=True,lr=2e-4,wd=0)" if args.parity
                   else "Adam(amsgrad=True,lr=0.01,wd=0)"),
        batch_size=args.batch_size,
        grad_clip=(10.0 if args.parity else None),
        train_states=64, test_states=64,
        cutoff=cutoff, horizon_fs=horizon,
        history=history, final=history[-1],
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
        elapsed_s=time.time() - t0,
        source="official EfficientTrajCastModel architecture; random init; "
               "TRAIN-only overfit, not a formal external baseline",
    )
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print("final", json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()
