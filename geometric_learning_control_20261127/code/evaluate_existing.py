"""Paired p/-p evaluation of existing trained propagators on fixed states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from control_data import load_states
from periodic import PeriodicGraphBuilder
from relational_inference import load_relational, vesin_graph


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


def metrics(df, dr, vf, vr, disp_p, disp_m, vel_p, vel_m, masses):
    m = masses[None, :, None]
    pf, pr = m * vf, m * vr
    pp, pm = m * vel_p, m * vel_m
    qe, qo = (df + dr) / 2, (df - dr) / 2
    pe, po = (pf + pr) / 2, (pf - pr) / 2
    qe_t, qo_t = (disp_p + disp_m) / 2, (disp_p - disp_m) / 2
    pe_t, po_t = (pp - pm) / 2, (pp + pm) / 2
    comps = dict(Q_even=stat(qe, qe_t), Q_odd=stat(qo, qo_t),
                 P_even=stat(pe, pe_t), P_odd=stat(po, po_t))
    # Interaction-dependent odd residuals in physical absolute units.
    v0 = (vf - vr) * 0  # placeholder; caller supplies via velocity target
    return dict(
        components=comps,
        forward=dict(disp_mse=float(((df - disp_p) ** 2).mean()),
                     vel_mse=float(((vf - vel_p) ** 2).mean())),
        reverse=dict(disp_mse=float(((dr - disp_m) ** 2).mean()),
                     vel_mse=float(((vr - vel_m) ** 2).mean())),
        equivalence=dict(
            disp_half=0.5 * (float(((df - disp_p) ** 2).mean()) +
                             float(((dr - disp_m) ** 2).mean())),
            disp_parity=stat(qe, qe_t)["mse"] + stat(qo, qo_t)["mse"],
            vel_half=0.5 * (float(((vf - vel_p) ** 2).mean()) +
                            float(((vr - vel_m) ** 2).mean())),
            vel_parity=stat(pe, pe_t)["mse"] + stat(po, po_t)["mse"],
        ),
    )


def eval_model(kind, system, states, device, batch_pairs=4):
    if kind == "r3":
        base = Path("/data1/wyd/research/foundation_mlip_adaptation/"
                    "experiments/relational_transition")
        path = (base / ("water5_20261103_r3_opt" if system == "water"
                        else "quartz30_20261103_r3_opt") / "best.pt")
    elif kind == "phase":
        base = Path("/data1/wyd/research/foundation_mlip_adaptation/"
                    "experiments/phase_space_20261120")
        path = base / ("water5" if system == "water" else "quartz30") / "best.pt"
    else:
        raise ValueError(kind)
    model, saved = load_relational(path, device=device)
    model.eval()
    builder = PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    n = len(states["q0"])
    out = {k: [] for k in ("df", "dr", "vf", "vr")}
    with torch.no_grad():
        for start in range(0, n, batch_pairs):
            stop = min(start + batch_pairs, n)
            gf, gr = [], []
            for i in range(start, stop):
                s = {k: (v[i] if torch.is_tensor(v) and
                        v.shape[0] == n else v) for k, v in states.items()}
                gf.append(vesin_graph(s["q0"], s["v0"], s["cell"],
                                      s["atomic_numbers"], s["masses"],
                                      model, builder, device))
                gr.append(vesin_graph(s["q0"], -s["v0"], s["cell"],
                                      s["atomic_numbers"], s["masses"],
                                      model, builder, device))
            bf = Batch.from_data_list(gf).to(device)
            br = Batch.from_data_list(gr).to(device)
            of = model(bf)
            orv = model(br)
            ptr = bf.ptr
            for j in range(stop - start):
                q0 = states["q0"][start + j].to(device)
                df = of["positions"][ptr[j]:ptr[j + 1]] - q0
                dr = orv["positions"][ptr[j]:ptr[j + 1]] - q0
                vf = of["velocities"][ptr[j]:ptr[j + 1]]
                vr = orv["velocities"][ptr[j]:ptr[j + 1]]
                out["df"].append(df.cpu())
                out["dr"].append(dr.cpu())
                out["vf"].append(vf.cpu())
                out["vr"].append(vr.cpu())
    n_atoms = states["q0"].shape[1]
    cat = {k: torch.cat(v).reshape(n, n_atoms, 3) for k, v in out.items()}
    res = metrics(cat["df"], cat["dr"], cat["vf"], cat["vr"],
                  states["disp_plus"], states["disp_minus"],
                  states["v_plus"], states["v_minus"], states["masses"])
    # Odd residuals (physical velocity units), then momentum.
    v0 = states["v0"]
    q_odd = (cat["df"] - cat["dr"]) / 2
    p_odd = states["masses"][None, :, None] * (cat["vf"] - cat["vr"]) / 2
    q_odd_t = (states["disp_plus"] - states["disp_minus"]) / 2
    p_odd_t = states["masses"][None, :, None] * \
        (states["v_plus"] + states["v_minus"]) / 2
    h = float(states["horizon_fs"])
    res["odd_residual"] = dict(
        Q_odd_residual=stat(q_odd - h * v0, q_odd_t - h * v0),
        P_odd_residual=stat(p_odd - states["p0"],
                            p_odd_t - states["p0"]),
    )
    # Per-species and error covariance.
    z = states["atomic_numbers"]
    res["per_species"] = {}
    for zz in sorted(set(int(v) for v in z.tolist())):
        mask = z == zz
        res["per_species"][str(zz)] = {}
        for name, pred, target in (
                ("Q_even", (cat["df"] + cat["dr"]) / 2,
                 (states["disp_plus"] + states["disp_minus"]) / 2),
                ("Q_odd", q_odd, q_odd_t),
                ("P_even", states["masses"][None, :, None] *
                 (cat["vf"] + cat["vr"]) / 2,
                 states["masses"][None, :, None] *
                 (states["v_plus"] - states["v_minus"]) / 2),
                ("P_odd", p_odd, p_odd_t)):
            res["per_species"][str(zz)][name] = stat(
                pred[:, mask], target[:, mask])
    errs = torch.stack([
        ((cat["df"] + cat["dr"]) / 2 -
         (states["disp_plus"] + states["disp_minus"]) / 2).reshape(-1),
        (q_odd - q_odd_t).reshape(-1),
        (states["masses"][None, :, None] *
         (cat["vf"] + cat["vr"]) / 2 -
         states["masses"][None, :, None] *
         (states["v_plus"] - states["v_minus"]) / 2).reshape(-1),
        (p_odd - p_odd_t).reshape(-1),
    ], 0).double()
    cov = torch.cov(errs)
    std = cov.diag().sqrt()
    corr = cov / (std[:, None] * std[None, :]).clamp_min(1e-30)
    res["error_covariance"] = dict(
        covariance=cov.tolist(), correlation=corr.tolist(),
        names=["Q_even", "Q_odd", "P_even", "P_odd"])
    res.update(kind=kind, system=system, checkpoint=str(path),
               params=sum(p.numel() for p in model.parameters()),
               cutoff=float(model.cfg.cutoff_angstrom), horizon_fs=h,
               saved_epoch=int(saved.get("epoch", -1)) if isinstance(saved, dict) else None)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-index", type=Path,
                        default=Path("experiments/parityflow_20261127/indices/water5"))
    parser.add_argument("--quartz-index", type=Path,
                        default=Path("experiments/parityflow_20261127/indices/quartz30"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/parityflow_20261127/existing_propagators_results.json"))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    water = load_states(args.water_index, "test", args.samples,
                        20261127, torch.float32)
    quartz = load_states(args.quartz_index, "test", args.samples,
                         20261127, torch.float32)
    result = dict(
        water_r3=eval_model("r3", "water", water, args.device),
        water_phase=eval_model("phase", "water", water, args.device),
        quartz_r3=eval_model("r3", "quartz", quartz, args.device),
        quartz_phase=eval_model("phase", "quartz", quartz, args.device),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {
        "components": v["components"],
        "odd_residual": v["odd_residual"],
        "equivalence": v["equivalence"],
    } for k, v in result.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
