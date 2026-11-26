"""Analyze the 5 ps velocity-feedback intervention trajectories.

Computes, for each feedback mode:
  * temperature and instantaneous translation/rotation/internal KE partition;
  * early MSD (1-5 ps slope, not a formal long-time D);
  * position/velocity divergence from the full-feedback trajectory;
  * first-step backbone/readout diagnostics.

The 5 ps runs are a mechanism test, not a D measurement.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from physics.units import KINETIC_TO_EV
from physics.water_motion import motion_partition
from position_velocity_audit import molecular_com, remove_global_com

ROOT = Path("/data2/wyd/research/foundation_mlip_adaptation")
FULL = ROOT / "experiments/feedback_full_5ps_20261127/trajectory.h5"
CONTROLS = {
    "no_feedback": ROOT / "experiments/feedback_no_feedback_5ps_20261127/trajectory.h5",
    "all_no_feedback": ROOT / "experiments/feedback_all_no_feedback_5ps_20261127/trajectory.h5",
    "thermostat_only": ROOT / "experiments/feedback_thermostat_only_5ps_20261127/trajectory.h5",
}
PARTIAL_FULL = ROOT / "experiments/feedback_full_100ps_20261127/trajectory.h5"


def load(path):
    with h5py.File(path, "r") as handle:
        q = np.asarray(handle["positions"], dtype=np.float64)
        v = np.asarray(handle["velocities"], dtype=np.float64)
        cell = np.asarray(handle["cell"], dtype=np.float64)
        masses = np.asarray(handle["masses_u"], dtype=np.float64)
        molecules = np.asarray(handle["molecule_ids"], dtype=np.int64)
        z = np.asarray(handle["atomic_numbers"], dtype=np.int64)
        v_pred = np.asarray(handle["backbone_velocity"], dtype=np.float64)
        v_fd = np.asarray(handle["readout_velocity"], dtype=np.float64)
        dq = np.asarray(handle["backbone_displacement"], dtype=np.float64)
        T = np.asarray(handle["temperature_K"], dtype=np.float64)
    return dict(q=q, v=v, cell=cell, masses=masses, molecules=molecules, z=z,
                v_pred=v_pred, v_fd=v_fd, dq=dq, T=T)


def partition_stats(q, v, cell, z, masses, molecules, stride=5):
    rows = {"translation_eV": [], "rotation_eV": [], "internal_eV": [],
            "total_KE_eV": []}
    for index in range(0, len(q), stride):
        part = motion_partition(q[index], v[index], cell, z, masses, molecules)
        for key in rows:
            rows[key].append(float(part[key]))
    return {key: dict(mean=float(np.mean(val)), std=float(np.std(val)),
                      n=len(val)) for key, val in rows.items()}


def com_msd(q, masses, molecules):
    q_com, v_com, mass = molecular_com(q, np.zeros_like(q), masses, molecules)
    q_rel, _ = remove_global_com(q_com, v_com, mass)
    n = len(q_rel)
    lags = np.arange(1, n)
    msd = np.array([np.mean(np.sum((q_rel[lag:] - q_rel[:-lag]) ** 2, axis=-1))
                    for lag in lags])
    t_ps = lags * 5.0 / 1000.0
    mask = (t_ps >= 1.0) & (t_ps <= 5.0)
    slope = float(np.polyfit(t_ps[mask], msd[mask], 1)[0]) if mask.sum() > 2 else None
    return dict(msd_1ps_A2=float(np.interp(1.0, t_ps, msd)),
                msd_5ps_A2=float(np.interp(5.0, t_ps, msd)),
                early_slope_A2_ps=slope,
                early_D_m2_s=(slope / 6.0 * 1e-8 if slope is not None else None),
                note="early 1-5 ps slope; not a diffusive D")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    full = load(FULL)
    full_part = partition_stats(full["q"], full["v"], full["cell"], full["z"],
                                full["masses"], full["molecules"])
    full_msd = com_msd(full["q"], full["masses"], full["molecules"])
    payload = dict(
        full=dict(T_mean=float(np.mean(full["T"][1:])),
                  T_std=float(np.std(full["T"][1:])),
                  partition=full_part, msd=full_msd),
        controls={},
    )
    for name, path in CONTROLS.items():
        if not path.exists():
            payload["controls"][name] = dict(available=False)
            continue
        ctrl = load(path)
        part = partition_stats(ctrl["q"], ctrl["v"], ctrl["cell"], ctrl["z"],
                               ctrl["masses"], ctrl["molecules"])
        msd = com_msd(ctrl["q"], ctrl["masses"], ctrl["molecules"])
        n = min(len(full["q"]), len(ctrl["q"]))
        dq = full["q"][:n] - ctrl["q"][:n]
        dv = full["v"][:n] - ctrl["v"][:n]
        # First-step diagnostics: both runs share q0,v0, so the first backbone
        # displacement should match; the second displacement is the feedback.
        first_dq_diff = rms(full["dq"][0] - ctrl["dq"][0])
        second_dq_diff = rms(full["dq"][1] - ctrl["dq"][1]) if n > 1 else None
        payload["controls"][name] = dict(
            available=True, T_mean=float(np.mean(ctrl["T"][1:])),
            T_std=float(np.std(ctrl["T"][1:])),
            partition=part, msd=msd,
            final_rms_position_diff_A=rms(dq[-1]),
            final_rms_velocity_diff_A_fs=rms(dv[-1]),
            max_rms_position_diff_A=float(max(rms(dq[i]) for i in range(n))),
            first_backbone_dq_diff_A=first_dq_diff,
            second_backbone_dq_diff_A=second_dq_diff,
            correction_over_v_pred=float(
                rms(full["v_fd"][0] - full["v_pred"][0])
                / (rms(full["v_pred"][0]) + 1e-30)),
        )
    # Validate the 5 ps full against the first 5 ps of the partial 100 ps full.
    if PARTIAL_FULL.exists():
        partial = load(PARTIAL_FULL)
        n = min(len(full["q"]), len(partial["q"]))
        payload["partial_full_validation"] = dict(
            common_frames=n,
            max_position_diff_A=float(np.max(np.abs(full["q"][:n] - partial["q"][:n]))),
            max_velocity_diff_A_fs=float(np.max(np.abs(full["v"][:n] - partial["v"][:n]))),
        )
    out = ROOT / "experiments/feedback_intervention_analysis_20261127.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
