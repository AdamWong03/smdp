"""Interaction-dependent odd residuals for archived ParityFlow checkpoints.

Physical convention:
  q, q_plus, q_minus : A
  p, p_plus, p_minus : u A / fs
  Q_odd = (q_plus - q_minus)/2
  P_odd = (p_plus + p_minus)/2
  residual_Q = Q_odd - h * p/m
  residual_P = P_odd - p

ParityFlow internally predicts mass-weighted components.  We convert to
physical absolute endpoints using the analytic COM terms stored in the
centered index.  The same network is evaluated at p and -p, and the direct
head reconstruction is checked against the paired evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from data import ParityFlowDataset, collate
from model import ParityFlow


def stats(pred, target):
    p = pred.reshape(-1).double()
    t = target.reshape(-1).double()
    mse = float(((p - t) ** 2).mean())
    zero = float((t ** 2).mean())
    pc = p - p.mean()
    tc = t - t.mean()
    corr = float((pc * tc).sum() / (pc.norm() * tc.norm()).clamp_min(1e-30))
    return dict(mse=mse, zero=zero, mse_over_zero=mse / max(zero, 1e-30),
                r2=1.0 - mse / max(zero, 1e-30), corr=corr,
                target_rms=float(t.pow(2).mean().sqrt()),
                pred_rms=float(p.pow(2).mean().sqrt()))


def load_model(checkpoint, norm, num_species, cutoff, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ParityFlow(
        num_species=cfg["num_species"], hidden=32, bottleneck=16, blocks=4,
        edge_rbf=4, cutoff=cfg["cutoff"],
        q_scale=cfg["norm"]["q_scale"], p_scale=cfg["norm"]["p_scale"],
        mass_mean=cfg["norm"]["mass_mean"],
        use_edge_message=bool(cfg.get("use_edge_message", False)),
        use_global_q=bool(cfg.get("use_global_q", True)),
        use_tp_readout=bool(cfg.get("use_tp_readout", False)),
        use_receiver_gate=bool(cfg.get("use_receiver_gate", False)),
        use_two_endpoint=bool(cfg.get("use_two_endpoint", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def reconstruct_absolute(out, batch, norm, horizon):
    """Absolute q_plus/q_minus and p_plus/p_minus from normalized heads."""
    qs = float(norm["Q_scale"])
    ps = float(norm["P_scale"])
    sqrt_m = torch.sqrt(batch["masses"])[:, None]
    qe, qo = out["Q_even"] * qs, out["Q_odd"] * qs
    pe, po = out["P_even"] * ps, out["P_odd"] * ps
    q_rel = batch["q_rel"]
    q_com = batch["q_com"]
    v_com = batch["v_com"]
    m = batch["masses"][:, None]
    q_plus = q_com + horizon * v_com + q_rel + (qe + qo) / sqrt_m
    q_minus = q_com - horizon * v_com + q_rel + (qe - qo) / sqrt_m
    p_plus = sqrt_m * (pe + po) + m * v_com
    p_minus = sqrt_m * (po - pe) + m * v_com
    return q_plus, q_minus, p_plus, p_minus


def load_targets(index_dir, split, sample_indices):
    """Reference physical endpoints for the selected index rows.

    The centered index stores Q/P targets; reconstruct absolute endpoints with
    the analytic COM convention.  The transition file is used only for a
    provenance check against q_plus/p_plus.
    """
    idx = Path(index_dir)
    with np.load(idx / f"{split}.npz", allow_pickle=False) as f:
        q_rel = torch.as_tensor(f["q_rel"][sample_indices], dtype=torch.float64)
        q_com = torch.as_tensor(f["q_com"][sample_indices], dtype=torch.float64)
        v_com = torch.as_tensor(f["v_com"][sample_indices], dtype=torch.float64)
        masses = torch.as_tensor(f["masses"], dtype=torch.float64)
        qe = torch.as_tensor(f["Q_even"][sample_indices], dtype=torch.float64)
        qo = torch.as_tensor(f["Q_odd"][sample_indices], dtype=torch.float64)
        pe = torch.as_tensor(f["P_even"][sample_indices], dtype=torch.float64)
        po = torch.as_tensor(f["P_odd"][sample_indices], dtype=torch.float64)
        source_path = np.asarray(f["source_path"])[sample_indices]
        source_frame = np.asarray(f["source_frame"], dtype=np.int64)[sample_indices]
        sample_index = np.asarray(f["sample_index"], dtype=np.int64)[sample_indices]
        cell = torch.as_tensor(f["cell"][sample_indices], dtype=torch.float64)
        z_idx = torch.as_tensor(f["z_idx"][sample_indices], dtype=torch.long)
        q0 = torch.as_tensor(f["q0"][sample_indices], dtype=torch.float64)
        p0 = torch.as_tensor(f["p0"][sample_indices], dtype=torch.float64)
    manifest = json.loads((idx / "index_manifest.json").read_text())
    horizon = float(manifest["horizon_fs"])
    sqrt_m = torch.sqrt(masses)[None, :, None]
    m = masses[None, :, None]
    q_plus = q_com[:, None, :] + horizon * v_com[:, None, :] + q_rel + (qe + qo) / sqrt_m
    q_minus = q_com[:, None, :] - horizon * v_com[:, None, :] + q_rel + (qe - qo) / sqrt_m
    p_plus = sqrt_m * (pe + po) + m * v_com[:, None, :]
    p_minus = sqrt_m * (po - pe) + m * v_com[:, None, :]
    return dict(q0=q0, p0=p0, cell=cell, z_idx=z_idx, masses=masses,
                q_plus=q_plus, q_minus=q_minus, p_plus=p_plus,
                p_minus=p_minus, horizon=horizon,
                source_path=source_path, source_frame=source_frame,
                sample_index=sample_index, q_rel=q_rel, q_com=q_com,
                v_com=v_com, Q_even=qe, Q_odd=qo, P_even=pe, P_odd=po)


def evaluate(index_dir, split, checkpoint, device, limit=64, seed=20261127):
    ds = ParityFlowDataset(index_dir, split)
    norm = ds.norm
    n = min(limit, len(ds))
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(len(ds), generator=g)[:n].tolist()
    model, ckpt = load_model(checkpoint, norm, int(ds.z_idx.max()) + 1,
                             float(json.loads(
                                 (Path(index_dir) / "index_manifest.json")
                                 .read_text())["cutoff_A"]), device)
    targets = load_targets(index_dir, split, sel)
    horizon = targets["horizon"]

    # Accumulators, physical absolute units.
    preds = {k: [] for k in ("q_plus", "q_minus", "p_plus", "p_minus",
                             "q_plus_rev", "q_minus_rev", "p_plus_rev",
                             "p_minus_rev")}
    head_recon = {k: [] for k in ("q_plus", "q_minus", "p_plus", "p_minus")}
    head_rev = {k: [] for k in ("q_plus", "q_minus", "p_plus", "p_minus")}
    target_chunks = {k: [] for k in ("q_plus", "q_minus", "p_plus", "p_minus")}
    z_chunks = []
    neighbor_counts = []

    for start in range(0, n, 8):
        chunk_sel = sel[start:start + 8]
        samples = [ds[i] for i in chunk_sel]
        batch = collate(samples)
        for k in ("q_rel", "p_tilde", "masses", "z_idx", "edge_vec",
                  "edge_index", "q_com", "v_com", "q0", "p0"):
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            out_f = model(batch["q_rel"], batch["p_tilde"], batch["masses"],
                          batch["z_idx"], batch["edge_vec"],
                          batch["edge_index"], batch["num_nodes"])
            out_r = model(batch["q_rel"], -batch["p_tilde"],
                          batch["masses"], batch["z_idx"],
                          batch["edge_vec"], batch["edge_index"],
                          batch["num_nodes"])
        qpf, qmf, ppf, pmf = reconstruct_absolute(
            out_f, batch, norm, horizon)
        qpr, qmr, ppr, pmr = reconstruct_absolute(
            out_r, batch, norm, horizon)
        for name, value in (("q_plus", qpf), ("q_minus", qmf),
                            ("p_plus", ppf), ("p_minus", pmf),
                            ("q_plus_rev", qpr), ("q_minus_rev", qmr),
                            ("p_plus_rev", ppr), ("p_minus_rev", pmr)):
            preds[name].append(value.cpu())
        # Direct head reconstruction in the p branch.
        sqrt_m = torch.sqrt(batch["masses"])[:, None]
        qs, ps = norm["Q_scale"], norm["P_scale"]
        qe, qo = out_f["Q_even"] * qs, out_f["Q_odd"] * qs
        pe, po = out_f["P_even"] * ps, out_f["P_odd"] * ps
        head_recon["q_plus"].append(
            (batch["q_com"] + horizon * batch["v_com"] + batch["q_rel"] +
             (qe + qo) / sqrt_m).cpu())
        head_recon["q_minus"].append(
            (batch["q_com"] - horizon * batch["v_com"] + batch["q_rel"] +
             (qe - qo) / sqrt_m).cpu())
        head_recon["p_plus"].append(
            (sqrt_m * (pe + po) +
             batch["masses"][:, None] * batch["v_com"]).cpu())
        head_recon["p_minus"].append(
            (sqrt_m * (po - pe) +
             batch["masses"][:, None] * batch["v_com"]).cpu())
        # Reversed branch head reconstruction (should equal q_minus/-p_minus).
        qe_r, qo_r = out_r["Q_even"] * qs, out_r["Q_odd"] * qs
        pe_r, po_r = out_r["P_even"] * ps, out_r["P_odd"] * ps
        head_rev["q_plus"].append(
            (batch["q_com"] - horizon * batch["v_com"] + batch["q_rel"] +
             (qe_r + qo_r) / sqrt_m).cpu())
        head_rev["q_minus"].append(
            (batch["q_com"] + horizon * batch["v_com"] + batch["q_rel"] +
             (qe_r - qo_r) / sqrt_m).cpu())
        head_rev["p_plus"].append(
            (sqrt_m * (pe_r + po_r) +
             batch["masses"][:, None] * batch["v_com"]).cpu())
        head_rev["p_minus"].append(
            (sqrt_m * (po_r - pe_r) +
             batch["masses"][:, None] * batch["v_com"]).cpu())
        for k in ("q_plus", "q_minus"):
            target_chunks[k].append(targets[k][start:start + len(chunk_sel)])
        for k in ("p_plus", "p_minus"):
            target_chunks[k].append(targets[k][start:start + len(chunk_sel)])
        z_chunks.append(targets["z_idx"][start:start + len(chunk_sel)])
        src, dst = batch["edge_index"][0], batch["edge_index"][1]
        counts = torch.zeros(batch["num_nodes"], device=device)
        counts.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        neighbor_counts.append(counts.cpu())

    n_atoms = targets["q0"].shape[1]
    cat = {k: torch.cat(v).reshape(n, n_atoms, 3) for k, v in preds.items()}
    tgt = {k: torch.cat(v) for k, v in target_chunks.items()}
    z = torch.cat(z_chunks)
    q0 = targets["q0"]
    p0 = targets["p0"]
    m = targets["masses"][None, :, None]

    # Paired parity components.
    Q_even = (cat["q_plus"] + cat["q_plus_rev"]) / 2 - q0
    Q_odd = (cat["q_plus"] - cat["q_plus_rev"]) / 2
    P_even = (cat["p_plus"] + cat["p_plus_rev"]) / 2
    P_odd = (cat["p_plus"] - cat["p_plus_rev"]) / 2
    Q_even_t = (tgt["q_plus"] + tgt["q_minus"]) / 2 - q0
    Q_odd_t = (tgt["q_plus"] - tgt["q_minus"]) / 2
    P_even_t = (tgt["p_plus"] - tgt["p_minus"]) / 2
    P_odd_t = (tgt["p_plus"] + tgt["p_minus"]) / 2

    # Direct-head equivalence checks.
    head_recon = {k: torch.cat(v).reshape(n, n_atoms, 3)
                  for k, v in head_recon.items()}
    head_rev = {k: torch.cat(v).reshape(n, n_atoms, 3)
                for k, v in head_rev.items()}
    head_diff = dict(
        q_plus=float((head_recon["q_plus"] - cat["q_plus"]).abs().max()),
        q_minus=float((head_recon["q_minus"] - cat["q_minus"]).abs().max()),
        p_plus=float((head_recon["p_plus"] - cat["p_plus"]).abs().max()),
        p_minus=float((head_recon["p_minus"] - cat["p_minus"]).abs().max()),
        rev_q_plus_to_minus=float(
            (head_rev["q_plus"] - cat["q_minus"]).abs().max()),
        rev_p_plus_to_neg_minus=float(
            (head_rev["p_plus"] + cat["p_minus"]).abs().max()),
    )

    # Odd residuals in physical absolute units.
    q_ballistic = horizon * p0 / m
    p_ballistic = p0
    Q_odd_res = Q_odd - q_ballistic
    Q_odd_res_t = Q_odd_t - q_ballistic
    P_odd_res = P_odd - p_ballistic
    P_odd_res_t = P_odd_t - p_ballistic

    result = dict(
        status="ok", index=str(index_dir), split=split,
        checkpoint=str(checkpoint), checkpoint_update=int(ckpt["update"]),
        samples=n, horizon_fs=horizon,
        units=dict(q="A", p="u A / fs", Q="A", P="u A / fs"),
        head_equivalence=head_diff,
        components={},
        odd_residual={},
        com=dict(
            q_com_rms=float(targets["q_com"].pow(2).mean().sqrt()),
            v_com_rms=float(targets["v_com"].pow(2).mean().sqrt()),
        ),
        per_species={},
        neighbor_count=dict(
            mean=float(torch.cat(neighbor_counts).mean()),
            min=float(torch.cat(neighbor_counts).min()),
            max=float(torch.cat(neighbor_counts).max()),
        ),
    )
    for name, pred, target in (
            ("Q_even", Q_even, Q_even_t), ("Q_odd", Q_odd, Q_odd_t),
            ("P_even", P_even, P_even_t), ("P_odd", P_odd, P_odd_t)):
        result["components"][name] = stats(pred, target)
    for name, pred, target in (
            ("Q_odd_residual", Q_odd_res, Q_odd_res_t),
            ("P_odd_residual", P_odd_res, P_odd_res_t)):
        result["odd_residual"][name] = stats(pred, target)
    # Per-species physical component metrics.
    for zz in sorted(set(int(v) for v in z.flatten())):
        mask = z == zz
        result["per_species"][str(zz)] = {}
        for name, pred, target in (
                ("Q_even", Q_even, Q_even_t), ("Q_odd", Q_odd, Q_odd_t),
                ("P_even", P_even, P_even_t), ("P_odd", P_odd, P_odd_t),
                ("Q_odd_residual", Q_odd_res, Q_odd_res_t),
                ("P_odd_residual", P_odd_res, P_odd_res_t)):
            result["per_species"][str(zz)][name] = stats(
                pred[mask], target[mask])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-index", type=Path,
                        default=Path("experiments/parityflow_20261127/indices/water5"))
    parser.add_argument("--quartz-index", type=Path,
                        default=Path("experiments/parityflow_20261127/indices/quartz30"))
    parser.add_argument("--water-checkpoint", type=Path,
                        default=Path("experiments/parityflow_20261127/paused_v1/water5/last.pt"))
    parser.add_argument("--quartz-checkpoint", type=Path,
                        default=Path("experiments/parityflow_20261127/paused_v1/quartz30/last.pt"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/parityflow_20261127/odd_residual_diagnostics.json"))
    args = parser.parse_args()
    result = dict(
        water=evaluate(args.water_index, args.split, args.water_checkpoint,
                       args.device, args.limit),
        quartz=evaluate(args.quartz_index, args.split, args.quartz_checkpoint,
                        args.device, args.limit),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {
        "components": v["components"],
        "odd_residual": v["odd_residual"],
        "head_equivalence": v["head_equivalence"],
    } for k, v in result.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
