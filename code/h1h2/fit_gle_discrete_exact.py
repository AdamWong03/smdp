"""Constrained discrete-time DHO+slow fit for a 5 fs GLE readout.

The previous unconstrained 3/5/6-mode fits put weight on an omega~1.25 rad/fs
mode whose period is ~5 fs.  At a 5 fs sampling stride that mode is aliased:
the continuous Green-Kubo formula gives ~14.3 fs but the discrete ACF integral
is 22-27 fs.  This script therefore

  * bounds the DHO frequency to omega <= 0.5 rad/fs (period >= 12.6 fs);
  * fits the sampled ACF on the same 5 fs grid used by the rollout;
  * constrains the *discrete* trapezoid integral to the reference value;
  * reports the exact 2x2 complex-OU embedding and its discrete integral.

No rollout.  CPU only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fit_velocity_gle import KEY_LAGS, model_acf, unpack


def discrete_embedding(x, modes, horizon_fs=5.0, max_lag_fs=5000.0):
    """Exact 2x2 complex-OU realization of each underdamped DHO.

    For one mode the continuous state matrix is

        A = [[-g, -w], [w, -g]],  g = gamma/2,

    and the stationary covariance that gives the DHO velocity ACF is

        Sigma = [[q, -q*g/w], [-q*g/w, q*(1 + 2*(g/w)^2)]].

    The sampled process is x_{n+1} = A_d x_n + eta_n with
    A_d = exp(A*dt), eta ~ N(0, Q_d), Q_d = Sigma - A_d Sigma A_d^T.
    The first component has exactly the DHO ACF at integer lags.
    """
    weights, gamma, omega_d, tau = unpack(x, modes)
    maxlag = int(round(max_lag_fs / horizon_fs))
    acf = np.zeros(maxlag + 1, dtype=np.float64)
    info = []
    for j in range(modes):
        g = 0.5 * gamma[j]
        w = omega_d[j]
        q = weights[j]
        sigma = np.array([[q, -q * g / w],
                          [-q * g / w, q * (1.0 + 2.0 * (g / w) ** 2)]])
        r = math.exp(-g * horizon_fs)
        theta = w * horizon_fs
        a_d = r * np.array([[math.cos(theta), -math.sin(theta)],
                            [math.sin(theta), math.cos(theta)]])
        q_d = sigma - a_d @ sigma @ a_d.T
        q_d = 0.5 * (q_d + q_d.T)
        eig_q = np.linalg.eigvalsh(q_d)
        eig_a = np.abs(np.linalg.eigvals(a_d))
        # Autocovariance at lag k is e1^T A_d^k Sigma e1.
        p = sigma.copy()
        e1 = np.array([1.0, 0.0])
        for lag in range(maxlag + 1):
            acf[lag] += float(e1 @ p @ e1)
            p = a_d @ p
        info.append(dict(
            mode=j, weight=float(q), gamma_per_fs=float(gamma[j]),
            omega_d_rad_per_fs=float(w),
            period_fs=float(2.0 * math.pi / w),
            decay_fs=float(2.0 / gamma[j]),
            a_d=a_d.tolist(), q_d=q_d.tolist(),
            q_d_eigenvalues=eig_q.tolist(),
            q_d_min_eigenvalue=float(eig_q.min()),
            a_d_spectral_radius=float(eig_a.max()),
        ))
    a = math.exp(-horizon_fs / tau)
    q_slow = weights[-1] * (1.0 - a * a)
    acf += weights[-1] * a ** np.arange(maxlag + 1)
    acf /= weights.sum()
    info.append(dict(
        mode="slow", weight=float(weights[-1]), tau_fs=float(tau),
        a=float(a), q=float(q_slow),
    ))
    return acf, info


def fit(mean, t, modes, n_starts=16, seed=20261126, omega_max=0.5,
        fit_min_fs=5.0):
    integral_ref = float(np.trapezoid(mean, t))
    fit_mask = (t >= fit_min_fs) & (t <= 5000.0)
    t_fit = t[fit_mask]
    ref_fit = mean[fit_mask]
    scale = 0.02 + np.abs(ref_fit)
    t_full = t

    def residual(x):
        pred = model_acf(t_fit, x, modes)
        r = (pred - ref_fit) / scale
        full = model_acf(t_full, x, modes)
        r_int = 20.0 * (np.trapezoid(full, t_full) - integral_ref) / integral_ref
        return np.concatenate([r, [r_int]])

    rng = np.random.default_rng(seed + 1000 * modes)
    best = None
    costs = []
    for _ in range(n_starts):
        gamma0 = np.exp(rng.uniform(-5.0, -1.0, modes))
        omega0 = np.exp(rng.uniform(-4.0, np.log(omega_max), modes))
        tau0 = np.exp(rng.uniform(np.log(100.0), np.log(5000.0)))
        x0 = np.concatenate([np.zeros(modes + 1), np.log(gamma0),
                             np.log(omega0), [np.log(tau0)]])
        lower = np.concatenate([np.full(modes + 1, -15.0),
                                np.full(modes, np.log(1e-4)),
                                np.full(modes, np.log(1e-3)),
                                [np.log(20.0)]])
        upper = np.concatenate([np.full(modes + 1, 15.0),
                                np.full(modes, np.log(2.0)),
                                np.full(modes, np.log(omega_max)),
                                [np.log(20000.0)]])
        try:
            result = least_squares(residual, x0, bounds=(lower, upper),
                                   max_nfev=3000, xtol=1e-11, ftol=1e-11,
                                   gtol=1e-11)
        except Exception:
            continue
        cost = float(np.sum(result.fun ** 2))
        costs.append(cost)
        if best is None or cost < best[0]:
            best = (cost, result.x, result.jac)
    if best is None:
        raise RuntimeError("all fits failed")
    return best, integral_ref, np.asarray(costs)


def evaluate(x, modes, mean, t, blocks, key_lags, horizon_fs=5.0):
    acf, info = discrete_embedding(x, modes, horizon_fs, max_lag_fs=t[-1])
    weights, gamma, omega_d, tau = unpack(x, modes)
    integral = float(np.trapezoid(acf, t))
    integral_ref = float(np.trapezoid(mean, t))
    key = np.asarray(key_lags, dtype=float)
    pred_key = np.interp(key, t, acf)
    mean_key = np.interp(key, t, mean)
    block_rmse = []
    block_integrals = []
    for block in blocks:
        block_rmse.append(float(np.sqrt(np.mean(
            (pred_key - np.interp(key, t, block)) ** 2))))
        block_integrals.append(float(np.trapezoid(block, t)))
    return dict(
        modes=modes,
        weights=weights.tolist(),
        gamma_per_fs=gamma.tolist(),
        omega_d_rad_per_fs=omega_d.tolist(),
        tau_fs=float(tau),
        integral_fs=integral,
        integral_ref_fs=integral_ref,
        integral_ratio=float(integral / integral_ref),
        rmse_mean=float(np.sqrt(np.mean((acf - mean) ** 2))),
        rmse_key=float(np.sqrt(np.mean((pred_key - mean_key) ** 2))),
        max_abs_key=float(np.max(np.abs(pred_key - mean_key))),
        acf_key_fs=key.tolist(),
        acf_key_model=pred_key.tolist(),
        acf_key_mean=mean_key.tolist(),
        block_rmse=block_rmse,
        block_integral_fs=block_integrals,
        block_integral_mean=float(np.mean(block_integrals)),
        block_integral_std=float(np.std(block_integrals, ddof=1)),
        discrete_min_eigenvalue=float(min(
            item.get("q_d_min_eigenvalue", 1.0) for item in info)),
        discrete_spectral_radius_max=float(max(
            item.get("a_d_spectral_radius", 0.0) for item in info
            if item["mode"] != "slow")),
        n_parameters=int(3 * modes + 1),
        embedding=info,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path,
                        default=Path("experiments/reference_acf_blocks_20261126.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--n-starts", type=int, default=16)
    parser.add_argument("--omega-max", type=float, default=0.5)
    parser.add_argument("--fit-min-fs", type=float, default=5.0)
    args = parser.parse_args()

    data = np.load(args.npz)
    acfs = data["acfs"]
    t = data["t"]
    mean = acfs.mean(0)
    key_lags = [5, 25, 100, 200, 500, 1000, 2000, 5000]
    results = []
    for modes in args.modes:
        (cost, x, jac), integral_ref, costs = fit(
            mean, t, modes, args.n_starts, omega_max=args.omega_max,
            fit_min_fs=args.fit_min_fs)
        row = evaluate(x, modes, mean, t, acfs, key_lags)
        row["fit_cost"] = float(cost)
        row["fit_cost_spread"] = float(np.std(costs)) if len(costs) > 1 else 0.0
        row["jacobian_condition"] = float(np.linalg.cond(jac))
        row["omega_max"] = float(args.omega_max)
        row["fit_min_fs"] = float(args.fit_min_fs)
        results.append(row)
        print(json.dumps({k: row[k] for k in
                          ["modes", "integral_fs", "integral_ref_fs",
                           "integral_ratio", "rmse_mean", "rmse_key",
                           "max_abs_key", "discrete_min_eigenvalue",
                           "discrete_spectral_radius_max", "n_parameters",
                           "jacobian_condition"]}, indent=2), flush=True)
        print("periods", [round(2 * math.pi / w, 2) for w in row["omega_d_rad_per_fs"]],
              "weights", [round(w, 4) for w in row["weights"]], flush=True)

    payload = dict(
        npz=str(args.npz),
        omega_max=float(args.omega_max),
        fit_min_fs=float(args.fit_min_fs),
        note=("Constrained discrete DHO+slow fit. Frequencies are bounded to "
              "avoid the 5 fs Nyquist aliasing that made the unconstrained "
              "3-mode kernel's discrete integral 1.9x too large."),
        results=results,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
