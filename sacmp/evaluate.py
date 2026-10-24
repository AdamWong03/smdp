"""Once-only held-out quartz segment evaluation; no checkpoint tuning."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from reference_data import QuartzSegments
from sacmp.rollout import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep held-out evaluations immutable")
    args.output.mkdir(parents=True)
    device = torch.device(args.device)
    model, saved = load_checkpoint(args.checkpoint, device)
    test = QuartzSegments(args.data_root, "test", args.output/"cache")
    total = 0
    mse = {k:np.zeros(17, dtype=np.float64) for k in ("q", "v")}
    mae = {k:np.zeros(17, dtype=np.float64) for k in ("q", "v")}
    other = dict(finite_diff=0., ref_finite_diff=0., backward=0.,
                 semigroup=0., cm_momentum=0., learned_energy_relative=0.)
    with torch.no_grad():
        for raw in DataLoader(test, batch_size=4, shuffle=False):
            q = raw["q"].to(device)
            v = raw["v"].to(device)
            bs = len(q)
            out = model(q[:, 0], v[:, 0], list(range(17)))
            predq, predv = out["positions"], out["velocities"]
            refq, refv = q[:, :17].transpose(0, 1), v[:, :17].transpose(0, 1)
            for name, error in (("q", predq-refq), ("v", predv-refv)):
                mse[name] += (error.square().mean((1, 2, 3))*bs).cpu().numpy()
                mae[name] += (error.abs().mean((1, 2, 3))*bs).cpu().numpy()
            other["finite_diff"] += float(
                (((predq[2:]-predq[:-2])/2.-predv[1:-1]).square().mean()))*bs
            other["ref_finite_diff"] += float(
                (((refq[2:]-refq[:-2])/2.-refv[1:-1]).square().mean()))*bs
            reverse = model(predq[-1], predv[-1], [-16.])
            other["backward"] += float(
                (reverse["positions"][0]-q[:, 0]).square().mean()
                +(reverse["velocities"][0]-v[:, 0]).square().mean())*bs
            eight = model(q[:, 0], v[:, 0], [8.])
            split = model(eight["positions"][0], eight["velocities"][0], [8.])
            other["semigroup"] += float(
                (split["positions"][0]-predq[-1]).square().mean()
                +(split["velocities"][0]-predv[-1]).square().mean())*bs
            m = model.mass.view(1, -1, 1)
            cm0 = (v[:, 0]*m).sum(1)/m.sum()
            cm1 = (predv[-1]*m).sum(1)/m.sum()
            other["cm_momentum"] += float((cm1-cm0).square().mean())*bs
            h0 = model.learned_energy(q[:, 0], v[:, 0])
            h1 = model.learned_energy(predq[-1], predv[-1])
            other["learned_energy_relative"] += float(
                ((h1-h0).square()/(h0.abs()+1e-8).square()).mean())*bs
            total += bs
    rows = [dict(time_fs=t,
                 position_MAE_A=float(mae["q"][t]/total),
                 position_RMSE_A=float(np.sqrt(mse["q"][t]/total)),
                 velocity_MAE_A_fs=float(mae["v"][t]/total),
                 velocity_RMSE_A_fs=float(np.sqrt(mse["v"][t]/total)))
            for t in range(17)]
    with (args.output/"query_errors.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    metrics = dict(
        test_samples=total, checkpoint=str(args.checkpoint),
        endpoint=rows[-1],
        finite_difference_velocity_RMSE_A_fs=float(
            np.sqrt(other["finite_diff"]/total)),
        reference_finite_difference_RMSE_A_fs=float(
            np.sqrt(other["ref_finite_diff"]/total)),
        reverse_state_RMSE_mixed_units=float(
            np.sqrt(other["backward"]/total)),
        semigroup_RMSE_mixed_units=float(
            np.sqrt(other["semigroup"]/total)),
        COM_velocity_RMSE_A_fs=float(
            np.sqrt(other["cm_momentum"]/total)),
        learned_energy_relative_RMSE=float(
            np.sqrt(other["learned_energy_relative"]/total)),
        physical_BKS_energy_evaluated=False,
        momentum_structure="exact COM and canonical mass-weighted modes",
        anchor=saved.get("anchor_report", {}))
    (args.output/"test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
