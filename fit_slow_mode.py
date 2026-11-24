"""Feasibility: add one slow OU mode to the AR(10) translation ACF and
constrain the Green-Kubo integral to the reference value.

If a small-weight slow mode can match the reference ACF at 5-5000 fs and
D_ref, it is a cheap Markovian embedding.  If not, the D fix needs a full
damped-oscillator kernel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from motion_loss import decompose_modes

KINETIC_TO_EV = 103.64269


def reference_acf(trajectory, duration_ps=100.0, stride_fs=5.0,
                  max_lag_fs=5000.0):
    with h5py.File(trajectory, "r") as handle:
        dt = float(handle.attrs.get("timestep_fs",
                                    handle.attrs.get("output_interval_fs")))
        stride = int(round(stride_fs / dt))
        frames = int(round(duration_ps * 1000 / stride_fs))
        pos_key = "positions" if "positions" in handle else "positions_unwrapped_A"
        vel_key = "velocities" if "velocities" in handle else "velocities_A_per_fs"
        q = handle[pos_key][:frames * stride:stride].astype(np.float64)
        v = handle[vel_key][:frames * stride:stride].astype(np.float64)
        cell = handle["cell_A"][0] if "cell_A" in handle else handle["cell"][:]
        masses = handle["masses_u"][:].astype(np.float64)
        molecules = handle["molecule_ids"][:].astype(int)
    fields = []
    for index in range(len(q)):
        part = decompose_modes(torch.as_tensor(q[index]),
                               torch.as_tensor(v[index]),
                               torch.as_tensor(masses),
                               torch.as_tensor(molecules),
                               torch.as_tensor(cell))
        fields.append(part["translation"].numpy())
    fields = np.stack(fields)
    weights = masses[None, :, None]
    maxlag = int(round(max_lag_fs / stride_fs))
    acf = np.empty(maxlag + 1)
    for lag in range(maxlag + 1):
        if lag == 0:
            acf[lag] = np.sum(weights * fields * fields) / len(fields)
        else:
            acf[lag] = np.sum(weights * fields[:-lag] * fields[lag:]) / (len(fields) - lag)
    return acf / acf[0], stride_fs


def ar_acf(coefficients, maxlag, dt_fs=5.0):
    from scipy.linalg import solve_discrete_lyapunov
    c = np.asarray(coefficients, dtype=float)
    p = len(c)
    F = np.zeros((p, p))
    F[0, :] = c
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)
    e1 = np.zeros((p, 1))
    e1[0, 0] = 1.0
    sigma = solve_discrete_lyapunov(F, e1 @ e1.T)
    rho = np.empty(maxlag + 1)
    rho[0] = 1.0
    rho[1:p + 1] = sigma[0, :p] / sigma[0, 0]
    for lag in range(p + 1, maxlag + 1):
        rho[lag] = np.dot(c, rho[lag - np.arange(1, p + 1)])
    return rho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model-trajectory", type=Path, default=None)
    parser.add_argument("--coefficients", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-ps", type=float, default=100.0)
    parser.add_argument("--max-lag-fs", type=float, default=5000.0)
    args = parser.parse_args()

    ref, dt = reference_acf(args.trajectory, args.duration_ps,
                            max_lag_fs=args.max_lag_fs)
    if args.model_trajectory is not None:
        model, _ = reference_acf(args.model_trajectory, args.duration_ps,
                                 max_lag_fs=args.max_lag_fs)
        model = model[:len(ref)]
    else:
        if args.coefficients is None:
            raise ValueError("--coefficients or --model-trajectory is required")
        data = json.loads(args.coefficients.read_text())
        c = data["modes"]["translation"]["coefficients"]
        model = ar_acf(c, len(ref) - 1, dt)
    t = np.arange(len(ref)) * dt
    i_ref = np.trapezoid(ref, t)
    i_ar = np.trapezoid(model, t)
    print(f"integral ref={i_ref:.3f} fs  AR10={i_ar:.3f} fs  "
          f"delta={i_ref - i_ar:.3f} fs")
    print(f"ACF ref  5/25/100/200/500/1000/2000/5000 fs: "
          f"{np.interp([5,25,100,200,500,1000,2000,5000], t, ref).round(4)}")
    print(f"ACF AR10 5/25/100/200/500/1000/2000/5000 fs: "
          f"{np.interp([5,25,100,200,500,1000,2000,5000], t, model).round(4)}")

    lags = np.array([5, 25, 100, 200, 500, 1000, 2000, 5000])
    ref_at = np.interp(lags, t, ref)
    best = None
    for tau in np.geomspace(200.0, 10000.0, 200):
        denom = tau - i_ar
        if abs(denom) < 1e-9:
            continue
        w = (i_ref - i_ar) / denom
        if not (0.0 < w < 0.5):
            continue
        combined = (1.0 - w) * model + w * np.exp(-t / tau)
        err = np.sqrt(np.mean((np.interp(lags, t, combined) - ref_at) ** 2))
        short = np.max(np.abs(np.interp([5, 25, 100], t, combined)
                              - np.interp([5, 25, 100], t, ref)))
        score = err + 0.5 * short
        if best is None or score < best["score"]:
            best = dict(tau=float(tau), w=float(w), rmse=float(err),
                        short_max=float(short), score=float(score))
    if best is None:
        print("no admissible slow mode")
        return
    tau, w = best["tau"], best["w"]
    combined = (1.0 - w) * model + w * np.exp(-t / tau)
    result = dict(
        integral_ref=i_ref, integral_ar=i_ar, **best,
        acf_ref_at=ref_at.tolist(), acf_combined_at=np.interp(lags, t, combined).tolist(),
        acf_ar_at=np.interp(lags, t, model).tolist(),
        lags_fs=lags.tolist(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
