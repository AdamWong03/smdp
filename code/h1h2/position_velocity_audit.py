"""Position-velocity consistency audit for water autonomous trajectories.

For each trajectory and each 100 ps block, compute:

  * molecular COM position increments dr and COM velocities v;
  * discrete kinematic residuals dr - h v_t (forward) and dr - h (v_t+v_{t+1})/2;
  * residual RMS relative to h|v|, parallel/perpendicular components, and ACF;
  * MSD diffusion D_MSD (global COM removed) and Green-Kubo D_GK;
  * block-to-block uncertainty.

The production GLR uses positions from the backbone displacement and velocities
from the GLR readout (decoupled scheme).  This script measures the resulting
kinematic mismatch; it does not assume that a GK/MSD difference alone proves a
physical violation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

KEY_LAGS_FS = (0.0, 5.0, 25.0, 100.0, 500.0)


def load_block(path: Path, start_ps: float, duration_ps: float,
               stride_fs: float = 5.0, position_key: str | None = None,
               velocity_key: str | None = None):
    with h5py.File(path, "r") as handle:
        dt = float(handle.attrs.get("timestep_fs",
                                    handle.attrs.get("output_interval_fs", stride_fs)))
        stride = max(1, int(round(stride_fs / dt)))
        start = int(round(start_ps * 1000.0 / dt))
        # Include both endpoints so a 100 ps block can be fitted to 100 ps.
        n = int(round(duration_ps * 1000.0 / stride_fs)) + 1
        selection = slice(start, start + n * stride, stride)
        pos_key = position_key or ("positions_unwrapped_A" if "positions_unwrapped_A" in handle
                                   else "positions")
        vel_key = velocity_key or ("velocities_A_per_fs" if "velocities_A_per_fs" in handle
                                   else "velocities")
        q = np.asarray(handle[pos_key][selection], dtype=np.float64)
        v = np.asarray(handle[vel_key][selection], dtype=np.float64)
        masses = np.asarray(handle["masses_u"][:], dtype=np.float64)
        molecules = np.asarray(handle["molecule_ids"][:], dtype=np.int64)
        alpha = None
        if "thermostat_alpha" in handle:
            alpha = np.asarray(handle["thermostat_alpha"][selection], dtype=np.float64)
    return q, v, masses, molecules, alpha, stride_fs


def molecular_com(q, v, masses, molecules):
    n_frames = len(q)
    molecules = np.unique(molecules, return_inverse=True)[1]
    n_mol = int(molecules.max()) + 1
    onehot = np.zeros((len(masses), n_mol), dtype=np.float64)
    onehot[np.arange(len(masses)), molecules] = 1.0
    mass = np.bincount(molecules, weights=masses, minlength=n_mol)
    q_com = np.einsum("fak,am->fmk", masses[None, :, None] * q, onehot)
    v_com = np.einsum("fak,am->fmk", masses[None, :, None] * v, onehot)
    q_com /= mass[None, :, None]
    v_com /= mass[None, :, None]
    return q_com, v_com, mass


def remove_global_com(q_com, v_com, mass):
    total = mass.sum()
    q_global = np.einsum("m,fmk->fk", mass, q_com) / total
    v_global = np.einsum("m,fmk->fk", mass, v_com) / total
    return q_com - q_global[:, None, :], v_com - v_global[:, None, :]


def rms_vector(x):
    return float(np.sqrt(np.mean(np.sum(x * x, axis=-1))))


def vector_acf(x, lags):
    n = len(x)
    var = float(np.mean(np.sum(x * x, axis=-1)))
    out = {}
    for lag in lags:
        lag = int(lag)
        if lag == 0:
            out[lag] = 1.0
        elif lag >= n:
            out[lag] = None
        else:
            out[lag] = float(np.mean(np.sum(x[:-lag] * x[lag:], axis=-1)) / var)
    return out


def msd_diffusion(q_com, stride_fs, fit_start_ps=10.0, fit_end_ps=100.0,
                  sample_fs=50.0):
    """D from the time-origin-averaged molecular COM MSD."""
    h = stride_fs
    n = len(q_com)
    max_lag = int(round(fit_end_ps * 1000.0 / h))
    step = max(1, int(round(sample_fs / h)))
    lags = np.arange(0, max_lag + 1, step)
    origins = np.arange(0, n - max_lag, step)
    if len(origins) < 2:
        origins = np.arange(0, max(1, n - max_lag))
    msd = []
    for lag in lags:
        d = q_com[origins + lag] - q_com[origins]
        msd.append(float(np.mean(np.sum(d * d, axis=-1))))
    msd = np.asarray(msd)
    times_ps = lags * h / 1000.0
    mask = (times_ps >= fit_start_ps) & (times_ps <= fit_end_ps)
    if mask.sum() < 3:
        mask = times_ps >= fit_start_ps
    slope = float(np.polyfit(times_ps[mask], msd[mask], 1)[0])  # A^2/ps
    # 1 A^2/ps = 1e-8 m^2/s; D = slope/6 in 3D
    return slope / 6.0 * 1e-8, np.interp(10.0, times_ps, msd), times_ps, msd


def gk_diffusion(v_com, stride_fs, max_lag_fs=5000.0):
    """Green-Kubo D from the molecular COM velocity ACF."""
    h = stride_fs
    n = len(v_com)
    maxlag = int(round(max_lag_fs / h))
    acf = np.empty(maxlag + 1)
    for lag in range(maxlag + 1):
        if lag == 0:
            acf[lag] = np.mean(np.sum(v_com * v_com, axis=-1))
        else:
            acf[lag] = np.sum(v_com[:-lag] * v_com[lag:]) / (n - lag) / v_com.shape[1]
    t = np.arange(maxlag + 1) * h
    d = float(np.trapezoid(acf, t) / 3.0 * 1e-5)
    return d, acf, t


def block_metrics(path: Path, start_ps: float, duration_ps: float,
                  stride_fs: float = 5.0) -> dict:
    q, v, masses, molecules, alpha, h = load_block(
        path, start_ps, duration_ps, stride_fs)
    q_com, v_com, mass = molecular_com(q, v, masses, molecules)
    q_rel, v_rel = remove_global_com(q_com, v_com, mass)
    dr = np.diff(q_rel, axis=0)
    v0 = v_rel[:-1]
    v1 = v_rel[1:]
    v_mid = 0.5 * (v0 + v1)
    res_fwd = dr - h * v0
    res_mid = dr - h * v_mid
    hv = h * v0
    hv_rms = rms_vector(hv)
    res_fwd_rms = rms_vector(res_fwd)
    res_mid_rms = rms_vector(res_mid)
    # Parallel/perpendicular decomposition relative to v0.
    denom = np.sum(v0 * v0, axis=-1)
    proj = np.sum(res_fwd * v0, axis=-1) / np.maximum(denom, 1e-30)
    res_par = proj[..., None] * v0
    res_perp = res_fwd - res_par
    cos = float(np.sum(res_fwd * v0) /
                np.sqrt(np.sum(res_fwd * res_fwd) * np.sum(v0 * v0)))
    # Center the residual for the ACF.
    res_centered = res_mid - res_mid.mean(axis=(0, 1), keepdims=True)
    v_centered = v0 - v0.mean(axis=(0, 1), keepdims=True)
    lags = [int(round(x / h)) for x in KEY_LAGS_FS]
    res_acf = vector_acf(res_centered, lags)
    v_acf = vector_acf(v_centered, lags)
    d_msd, _, _, _ = msd_diffusion(q_rel, h)
    d_gk, gk_acf, _ = gk_diffusion(v_com, h)
    return dict(
        start_ps=float(start_ps), duration_ps=float(duration_ps),
        frames=int(len(q)), thermostat_alpha_mean=(
            float(alpha.mean()) if alpha is not None else None),
        thermostat_alpha_std=(
            float(alpha.std()) if alpha is not None else None),
        hv_rms_A=hv_rms,
        res_forward_rms_A=res_fwd_rms,
        res_midpoint_rms_A=res_mid_rms,
        res_forward_over_hv=float(res_fwd_rms / hv_rms),
        res_midpoint_over_hv=float(res_mid_rms / hv_rms),
        res_parallel_rms_A=rms_vector(res_par),
        res_perp_rms_A=rms_vector(res_perp),
        cos_res_velocity=cos,
        res_acf_lags_fs=list(KEY_LAGS_FS),
        res_acf=[res_acf[int(round(x / h))] for x in KEY_LAGS_FS],
        velocity_acf_lags_fs=list(KEY_LAGS_FS),
        velocity_acf=[v_acf[int(round(x / h))] for x in KEY_LAGS_FS],
        zero_lag_com_A2_fs2=float(np.mean(np.sum(v_com * v_com, axis=-1))),
        D_msd_m2_s=float(d_msd),
        D_gk_m2_s=float(d_gk),
        D_gk_over_msd=(float(d_gk / d_msd) if abs(d_msd) > 1e-12 else None),
    )


def summarize(blocks):
    keys = ("D_msd_m2_s", "D_gk_m2_s", "D_gk_over_msd",
            "res_forward_over_hv", "res_midpoint_over_hv",
            "res_parallel_rms_A", "res_perp_rms_A", "cos_res_velocity")
    out = {}
    for key in keys:
        values = np.asarray([b[key] for b in blocks if b[key] is not None],
                            dtype=float)
        if len(values) == 0:
            out[key] = dict(mean=None, std=None, n=0)
            continue
        out[key] = dict(mean=float(values.mean()),
                        std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        n=int(len(values)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/position_velocity_audit_20261127.json"))
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--block-ps", type=float, default=100.0)
    args = parser.parse_args()

    root = Path("/data2/wyd/research/foundation_mlip_adaptation")
    data = Path("/data1/wyd/research/foundation_mlip_adaptation")
    trajectories = [
        dict(label="reference_2p5ns",
             path=data / "data/smdp_reference_md_v1/water/physical_reference/nvt_2p5ns/production.h5",
             starts=[i * 300.0 for i in range(args.blocks)], duration=args.block_ps),
        dict(label="glr_deterministic_2p5ns",
             path=root / "experiments/glr_2p5ns_20261122/trajectory.h5",
             starts=[i * 300.0 for i in range(args.blocks)], duration=args.block_ps),
        dict(label="glr_stochastic_2p5ns",
             path=root / "experiments/glr_stoch_2p5ns_20261122/trajectory.h5",
             starts=[i * 300.0 for i in range(args.blocks)], duration=args.block_ps),
        dict(label="raw_backbone_100ps",
             path=root / "experiments/nvt_screen_250k_full_20261120/trajectory.h5",
             starts=[50.0], duration=100.0),
    ]

    payload = dict(
        definition=dict(
            discrete_rule=("dr = r_com(t+1)-r_com(t); residual_forward = dr - h*v_com(t); "
                           "residual_midpoint = dr - h*(v_com(t)+v_com(t+1))/2; "
                           "h = 5 fs; molecular COM; global COM removed for MSD; "
                           "GK uses the raw molecular COM velocity ACF"),
            msd_fit_ps=[10.0, 100.0],
            gk_max_lag_fs=5000.0,
            note=("The production GLR uses positions from the backbone displacement and "
                  "velocities from the GLR readout. This audit measures that decoupling; "
                  "a GK/MSD difference alone is not a physical-violation proof."),
        ),
        trajectories=[],
    )
    for spec in trajectories:
        path = spec["path"]
        if not path.exists():
            print("missing", path, flush=True)
            continue
        blocks = []
        for start in spec["starts"]:
            try:
                block = block_metrics(path, start, spec["duration"])
            except Exception as error:
                print("failed", spec["label"], start, repr(error), flush=True)
                continue
            blocks.append(block)
            print(json.dumps(dict(trajectory=spec["label"], **{
                k: block[k] for k in ("start_ps", "D_msd_m2_s", "D_gk_m2_s",
                                      "D_gk_over_msd", "res_midpoint_over_hv",
                                      "res_forward_over_hv", "cos_res_velocity")})),
                flush=True)
        if blocks:
            entry = dict(label=spec["label"], path=str(path),
                         block_ps=spec["duration"], blocks=blocks,
                         summary=summarize(blocks))
            payload["trajectories"].append(entry)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out, flush=True)
    for entry in payload["trajectories"]:
        s = entry["summary"]
        ratio = s["D_gk_over_msd"]["mean"]
        print(entry["label"],
              "D_MSD", round(s["D_msd_m2_s"]["mean"] * 1e9, 4),
              "±", round(s["D_msd_m2_s"]["std"] * 1e9, 4),
              "D_GK", round(s["D_gk_m2_s"]["mean"] * 1e9, 4),
              "±", round(s["D_gk_m2_s"]["std"] * 1e9, 4),
              "GK/MSD", (round(ratio, 3) if ratio is not None else None),
              "res/hv", round(s["res_midpoint_over_hv"]["mean"], 3),
              flush=True)


if __name__ == "__main__":
    main()
