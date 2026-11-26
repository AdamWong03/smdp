"""Fit a positive-sum damped-oscillator + slow-mode ACF to the reference
translation velocity ACF, with the Green-Kubo integral constrained.

This is a feasibility test for a velocity-only GLE readout. No rollout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import least_squares

from motion_loss import decompose_modes

KEYS = ("translation", "rotation", "internal")
KEY_LAGS = np.array([5, 25, 100, 200, 500, 1000, 2000, 5000])


def reference_acf(path: Path, duration_ps: float, stride_fs: float,
                  max_lag_fs: float):
    with h5py.File(path, "r") as handle:
        dt = float(handle.attrs.get("timestep_fs",
                                    handle.attrs.get("output_interval_fs", stride_fs)))
        stride = max(1, int(round(stride_fs / dt)))
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
        part = decompose_modes(
            torch.as_tensor(q[index]), torch.as_tensor(v[index]),
            torch.as_tensor(masses), torch.as_tensor(molecules),
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


def dho(t, gamma, omega_d):
    return np.exp(-0.5 * gamma * t) * (
        np.cos(omega_d * t) + 0.5 * gamma / omega_d * np.sin(omega_d * t))


def unpack(x, modes):
    logits = x[:modes + 1]
    logits = logits - logits.max()
    weights = np.exp(logits) / np.exp(logits).sum()
    gamma = np.exp(x[modes + 1:2 * modes + 1])
    omega_d = np.exp(x[2 * modes + 1:3 * modes + 1])
    tau = np.exp(x[3 * modes + 1])
    return weights, gamma, omega_d, tau


def model_acf(t, x, modes):
    weights, gamma, omega_d, tau = unpack(x, modes)
    out = np.zeros_like(t)
    for j in range(modes):
        out += weights[j] * dho(t, gamma[j], omega_d[j])
    out += weights[-1] * np.exp(-t / tau)
    return out


def model_integral(x, modes):
    weights, gamma, omega_d, tau = unpack(x, modes)
    value = weights[-1] * tau
    for j in range(modes):
        value += weights[j] * gamma[j] / (
            omega_d[j] ** 2 + (0.5 * gamma[j]) ** 2)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-ps", type=float, default=100.0)
    parser.add_argument("--stride-fs", type=float, default=5.0)
    parser.add_argument("--max-lag-fs", type=float, default=5000.0)
    parser.add_argument("--modes", type=int, default=3)
    args = parser.parse_args()

    ref, dt = reference_acf(args.reference, args.duration_ps,
                            args.stride_fs, args.max_lag_fs)
    t = np.arange(len(ref)) * dt
    integral_ref = float(np.trapezoid(ref, t))
    print(f"reference ACF integral = {integral_ref:.4f} fs")

    # Fit only the first 2 ps in the residual to keep the fit conditioned,
    # then check the full 5 ps ACF.
    fit_mask = t <= 2000.0
    t_fit = t[fit_mask]
    ref_fit = ref[fit_mask]
    scale = 0.02 + np.abs(ref_fit)

    def residual(x):
        pred = model_acf(t_fit, x, args.modes)
        r = (pred - ref_fit) / scale
        r_int = 20.0 * (model_integral(x, args.modes) - integral_ref) / integral_ref
        return np.concatenate([r, [r_int]])

    rng = np.random.default_rng(20261123)
    best = None
    for attempt in range(12):
        gamma0 = np.exp(rng.uniform(-5.0, -2.5, args.modes))
        omega0 = np.exp(rng.uniform(-4.0, -1.5, args.modes))
        tau0 = np.exp(rng.uniform(np.log(200.0), np.log(5000.0)))
        x0 = np.concatenate([
            np.zeros(args.modes + 1),
            np.log(gamma0),
            np.log(omega0),
            [np.log(tau0)],
        ])
        lower = np.concatenate([
            np.full(args.modes + 1, -15.0),
            np.full(args.modes, np.log(1e-5)),
            np.full(args.modes, np.log(1e-4)),
            [np.log(20.0)],
        ])
        upper = np.concatenate([
            np.full(args.modes + 1, 15.0),
            np.full(args.modes, np.log(2.0)),
            np.full(args.modes, np.log(2.0)),
            [np.log(20000.0)],
        ])
        try:
            fit = least_squares(residual, x0, bounds=(lower, upper),
                                max_nfev=4000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        except Exception:
            continue
        cost = float(np.sum(fit.fun ** 2))
        if best is None or cost < best[0]:
            best = (cost, fit.x)
    if best is None:
        raise SystemExit("all fits failed")

    x = best[1]
    pred = model_acf(t, x, args.modes)
    weights, gamma, omega_d, tau = unpack(x, args.modes)
    result = {
        "modes": args.modes,
        "weights": weights.tolist(),
        "gamma_per_fs": gamma.tolist(),
        "omega_d_rad_per_fs": omega_d.tolist(),
        "tau_fs": float(tau),
        "integral_ref_fs": integral_ref,
        "integral_model_fs": float(model_integral(x, args.modes)),
        "fit_cost": float(best[0]),
        "rmse_fit": float(np.sqrt(np.mean((pred[fit_mask] - ref_fit) ** 2))),
        "rmse_full": float(np.sqrt(np.mean((pred - ref) ** 2))),
        "key_lags_fs": KEY_LAGS.tolist(),
        "acf_ref": np.interp(KEY_LAGS, t, ref).tolist(),
        "acf_model": np.interp(KEY_LAGS, t, pred).tolist(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
