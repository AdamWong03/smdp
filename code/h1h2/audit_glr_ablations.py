"""Audit the existing GLR mode/memory/rescale ablations without rerunning them.

Reads the already-produced 100 ps trajectories and analysis JSONs, computes the
same MSD/GK/kinematic metrics used in the position-velocity audit, and prints a
compact necessity table.  Missing analyses are marked, not rerun.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from position_velocity_audit import block_metrics

ROOT = Path("/data2/wyd/research/foundation_mlip_adaptation")
RUNS = [
    ("glr_100ps", "glr_100ps_20261122"),
    ("glr_noise_off", "glr_noise_off_100ps_20261122"),
    ("glr_stoch_full", "glr_stoch_full_100ps_20261122"),
    ("glr_rescale_off", "glr_rescale_off_100ps_20261122"),
    ("glr_ltrans0.3", "glr_ltrans0.3_100ps_20261122"),
    ("glr_ltrans0.5", "glr_ltrans0.5_100ps_20261122"),
    ("glr_p1", "glr_p1_100ps_20261122"),
    ("glr_p5", "glr_p5_100ps_20261122"),
    ("glr_p20", "glr_p20_100ps_20261122"),
    ("glr_seed2", "glr_seed2_100ps_20261122"),
    ("glr_seed3", "glr_seed3_100ps_20261122"),
]


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def main():
    rows = []
    for label, directory in RUNS:
        run = ROOT / "experiments" / directory
        traj = run / "trajectory.h5"
        if not traj.exists():
            rows.append(dict(label=label, directory=directory, available=False,
                             reason="trajectory missing"))
            continue
        motion = read_json(run / "analysis/motion.json")
        rdf = read_json(run / "analysis/rdf.json")
        vdos = read_json(run / "analysis/vdos.json")
        try:
            metrics = block_metrics(traj, 0.0, 100.0)
        except Exception as error:
            metrics = dict(error=repr(error))
        row = dict(label=label, directory=directory, available=True)
        if motion:
            part = motion["metrics"]["kinetic_partition"]
            temp = motion["metrics"]["temperature_K"]
            row["T_mean"] = temp["SMDP"]["mean"] if "SMDP" in temp else None
            row["T_std"] = temp["SMDP"]["std"] if "SMDP" in temp else None
            row["partition"] = {
                key: dict(mean=value["SMDP"]["mean"], std=value["SMDP"]["std"])
                for key, value in part.items()
                if isinstance(value, dict) and "SMDP" in value
            }
        else:
            row["analysis_missing"] = True
        if rdf:
            row["rdf_rmse"] = {k: v["RMSE"] for k, v in rdf["metrics"].items()}
        if vdos:
            row["vdos_overlap"] = {k: v["overlap"]
                                   for k, v in vdos["metrics"]["elements"].items()}
        row["D_msd_m2_s"] = metrics.get("D_msd_m2_s")
        row["D_gk_m2_s"] = metrics.get("D_gk_m2_s")
        row["GK_over_MSD"] = metrics.get("D_gk_over_msd")
        row["res_mid_over_hv"] = metrics.get("res_midpoint_over_hv")
        rows.append(row)
        print(label, "T", row.get("T_mean"), "Tstd", row.get("T_std"),
              "Dmsd", row.get("D_msd_m2_s"), "Dgk", row.get("D_gk_m2_s"),
              "ratio", row.get("GK_over_MSD"), "res", row.get("res_mid_over_hv"),
              flush=True)

    payload = dict(
        note=("Audit of existing 100 ps GLR ablations; no rollout was rerun. "
              "D_MSD is the 10-100 ps molecular-COM fit; D_GK is 0-5 ps. "
              "The missing glr_int10_100ps directory means the internal-mode "
              "blend 1.0 ablation is unavailable."),
        runs=rows,
    )
    out = ROOT / "experiments/glr_ablation_audit_20261127.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
