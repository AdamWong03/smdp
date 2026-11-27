"""Offline 4-6 mode DHO + slow-mode fit study for the water translation ACF.

CPU only. Calibration = first 100 ps of the 2.5 ns reference NVT.
Validation = a disjoint 100-200 ps block of the same reference (and, if
available, the independent NVT dump). No rollout, no training.

The fit uses the existing continuous DHO+slow model from fit_velocity_gle.py.
It additionally checks:
  - PSD of the continuous kernel;
  - discrete AR(2)/AR(1) state-space realizability and sampled ACF;
  - held-out ACF/integral;
  - parameter count, condition number, weight degeneracy;
  - comparison to AR(10) coefficients and the failed 3-mode fit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import least_squares

from fit_velocity_gle import KEY_LAGS, dho, model_acf, model_integral, unpack


def translation_acf(path: Path, start_ps: float, duration_ps: float,
                    stride_fs: float = 5.0, max_lag_fs: float = 5000.0):
    with h5py.File(path, "r") as handle:
        dt = float(handle.attrs.get("output_interval_fs",
                                    handle.attrs.get("timestep_fs", stride_fs)))
        stride = max(1, int(round(stride_fs / dt)))
        start = int(round(start_ps * 1000 / dt))
        n = int(round(duration_ps * 1000 / stride_fs))
        selection = slice(start, start + n * stride, stride)
        pos_key = "positions" if "positions" in handle else "positions_unwrapped_A"
        vel_key = "velocities" if "velocities" in handle else "velocities_A_per_fs"
        q = np.asarray(handle[pos_key][selection], dtype=np.float64)
        v = np.asarray(handle[vel_key][selection], dtype=np.float64)
        masses = np.asarray(handle["masses_u"][:], dtype=np.float64)
        molecules = np.asarray(handle["molecule_ids"][:], dtype=np.int64)
    unique = np.unique(molecules)
    v_com = np.zeros((n, len(unique), 3), dtype=np.float64)
    for k, molecule in enumerate(unique):
        mask = molecules == molecule
        v_com[:, k] = (v[:, mask] * masses[mask, None]).sum(1) / masses[mask].sum()
    norm = np.sum(v_com * v_com) / n
    maxlag = int(round(max_lag_fs / stride_fs))
    acf = np.empty(maxlag + 1, dtype=np.float64)
    for lag in range(maxlag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.sum(v_com[:-lag] * v_com[lag:]) / (n - lag) / norm
    return acf, stride_fs


def model_psd(x, modes, max_lag_fs=20000.0, points=20001):
    t = np.linspace(0.0, max_lag_fs, points)
    c = model_acf(t, x, modes)
    spectrum = np.fft.rfft(c)
    return float(np.min(spectrum.real)), float(np.max(np.abs(spectrum.imag)))


def discrete_embedding(x, modes, horizon_fs=5.0):
    """Return sampled ACF of the AR(2)/AR(1) state-space realization.

    Each continuous DHO is sampled at horizon_fs with poles
    exp((-gamma/2 +/- i omega_d) * horizon).  The innovation variance is
    chosen so the stationary variance equals the fitted weight; the slow OU
    mode is sampled exactly.  This checks discrete-time FDT consistency.
    """
    weights, gamma, omega_d, tau = unpack(x, modes)
    # time grid for the sampled ACF
    maxlag = int(round(5000.0 / horizon_fs))
    acf = np.zeros(maxlag + 1, dtype=np.float64)
    min_eig = 1.0
    min_innov = 1e9
    for j in range(modes):
        r = math.exp(-0.5 * gamma[j] * horizon_fs)
        theta = omega_d[j] * horizon_fs
        a1 = 2.0 * r * math.cos(theta)
        a2 = -r * r
        A = np.array([[a1, a2], [1.0, 0.0]], dtype=np.float64)
        eig = np.max(np.abs(np.linalg.eigvals(A)))
        min_eig = min(min_eig, 1.0 - eig)
        # unit innovation q=1, velocity stationary variance
        P = solve_discrete_lyapunov(A, np.array([[1.0, 0.0], [0.0, 0.0]]))
        p00 = float(P[0, 0])
        q = float(weights[j] / p00) if p00 > 0 else 0.0
        min_innov = min(min_innov, q)
        P = P * q
        # Autocovariance at lag k is e1^T A^k P e1 (not (A^k e1)^T P (A^k e1)).
        cov = P.copy()
        e1 = np.array([1.0, 0.0])
        for lag in range(maxlag + 1):
            acf[lag] += float(e1 @ cov @ e1)
            cov = A @ cov
    # slow OU mode
    a = math.exp(-horizon_fs / tau)
    min_eig = min(min_eig, 1.0 - abs(a))
    q = float(weights[-1] * (1.0 - a * a))
    min_innov = min(min_innov, q)
    for lag in range(maxlag + 1):
        acf[lag] += float(weights[-1] * a ** lag)
    acf /= max(acf[0], 1e-30)
    return acf, float(min_eig), float(min_innov)


def fit_model(ref, t, modes, n_starts=16, seed=20261126):
    integral_ref = float(np.trapezoid(ref, t))
    fit_mask = t <= 2000.0
    t_fit = t[fit_mask]
    ref_fit = ref[fit_mask]
    scale = 0.02 + np.abs(ref_fit)

    def residual(x):
        pred = model_acf(t_fit, x, modes)
        r = (pred - ref_fit) / scale
        r_int = 20.0 * (model_integral(x, modes) - integral_ref) / integral_ref
        return np.concatenate([r, [r_int]])

    rng = np.random.default_rng(seed + modes)
    best = None
    costs = []
    for _ in range(n_starts):
        gamma0 = np.exp(rng.uniform(-5.0, -2.5, modes))
        omega0 = np.exp(rng.uniform(-4.0, -1.5, modes))
        tau0 = np.exp(rng.uniform(np.log(200.0), np.log(5000.0)))
        x0 = np.concatenate([np.zeros(modes + 1), np.log(gamma0),
                             np.log(omega0), [np.log(tau0)]])
        lower = np.concatenate([np.full(modes + 1, -15.0),
                                np.full(modes, np.log(1e-5)),
                                np.full(modes, np.log(1e-4)),
                                [np.log(20.0)]])
        upper = np.concatenate([np.full(modes + 1, 15.0),
                                np.full(modes, np.log(2.0)),
                                np.full(modes, np.log(2.0)),
                                [np.log(20000.0)]])
        try:
            fit = least_squares(residual, x0, bounds=(lower, upper),
                                max_nfev=4000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        except Exception:
            continue
        cost = float(np.sum(fit.fun ** 2))
        costs.append(cost)
        if best is None or cost < best[0]:
            best = (cost, fit.x, fit.jac)
    if best is None:
        raise RuntimeError("all fits failed")
    return best, integral_ref, np.asarray(costs)


def evaluate_fit(x, modes, ref, t, val_ref, val_t, horizon_fs=5.0):
    pred = model_acf(t, x, modes)
    weights, gamma, omega_d, tau = unpack(x, modes)
    integral_model = float(model_integral(x, modes))
    integral_ref = float(np.trapezoid(ref, t))
    integral_val = float(np.trapezoid(val_ref, val_t))
    key = KEY_LAGS
    acf_ref_key = np.interp(key, t, ref)
    acf_model_key = np.interp(key, t, pred)
    acf_val_key = np.interp(key, val_t, val_ref)
    val_pred_key = np.interp(key, t, pred)
    psd_min, psd_imag = model_psd(x, modes)
    disc_acf, min_eig, min_innov = discrete_embedding(x, modes, horizon_fs)
    disc_key = np.interp(key, np.arange(len(disc_acf)) * horizon_fs, disc_acf)
    return dict(
        modes=modes,
        weights=weights.tolist(),
        gamma_per_fs=gamma.tolist(),
        omega_d_rad_per_fs=omega_d.tolist(),
        tau_fs=float(tau),
        integral_ref_fs=integral_ref,
        integral_model_fs=integral_model,
        integral_validation_fs=integral_val,
        acf_key_fs=key.tolist(),
        acf_ref_key=acf_ref_key.tolist(),
        acf_model_key=acf_model_key.tolist(),
        acf_validation_key=acf_val_key.tolist(),
        acf_model_on_validation_grid=val_pred_key.tolist(),
        rmse_calibration=float(np.sqrt(np.mean((pred - ref) ** 2))),
        rmse_validation=float(np.sqrt(np.mean((val_pred_key - acf_val_key) ** 2))),
        max_abs_validation=float(np.max(np.abs(val_pred_key - acf_val_key))),
        psd_min_real=psd_min,
        psd_imag_max=psd_imag,
        discrete_min_eig_gap=min_eig,
        discrete_min_innovation_variance=min_innov,
        discrete_acf_key=disc_key.tolist(),
        discrete_rmse_vs_model=float(np.sqrt(np.mean((disc_key - acf_model_key) ** 2))),
        n_parameters=int(3 * modes + 1),
        weight_min=float(weights.min()),
        weight_max=float(weights.max()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-start-ps", type=float, default=0.0)
    parser.add_argument("--calibration-ps", type=float, default=100.0)
    parser.add_argument("--validation-start-ps", type=float, default=100.0)
    parser.add_argument("--validation-ps", type=float, default=100.0)
    parser.add_argument("--modes", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--n-starts", type=int, default=16)
    args = parser.parse_args()

    ref, dt = translation_acf(args.reference, args.calibration_start_ps,
                              args.calibration_ps, 5.0, 5000.0)
    val, val_dt = translation_acf(args.reference, args.validation_start_ps,
                                  args.validation_ps, 5.0, 5000.0)
    t = np.arange(len(ref)) * dt
    val_t = np.arange(len(val)) * val_dt
    print(f"calibration ACF integral {np.trapezoid(ref, t):.4f} fs; "
          f"validation integral {np.trapezoid(val, val_t):.4f} fs", flush=True)

    results = []
    for modes in args.modes:
        (cost, x, jac), integral_ref, costs = fit_model(ref, t, modes,
                                                         args.n_starts)
        result = evaluate_fit(x, modes, ref, t, val, val_t)
        result["fit_cost"] = float(cost)
        result["fit_cost_spread"] = float(np.std(costs)) if len(costs) > 1 else 0.0
        result["fit_cost_min"] = float(np.min(costs)) if len(costs) else float(cost)
        result["jacobian_condition"] = float(np.linalg.cond(jac))
        results.append(result)
        print(json.dumps({k: result[k] for k in
                          ["modes", "integral_model_fs", "integral_validation_fs",
                           "rmse_calibration", "rmse_validation",
                           "psd_min_real", "discrete_min_eig_gap",
                           "discrete_rmse_vs_model", "n_parameters",
                           "jacobian_condition"]}, indent=2), flush=True)

    payload = dict(
        reference=str(args.reference),
        calibration=dict(start_ps=args.calibration_start_ps,
                         duration_ps=args.calibration_ps, stride_fs=5.0),
        validation=dict(start_ps=args.validation_start_ps,
                        duration_ps=args.validation_ps, stride_fs=5.0),
        results=results,
        note="Calibration and validation are disjoint time blocks of the same reference NVT. "
             "A truly independent trajectory validation should be added before a paper claim.",
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
