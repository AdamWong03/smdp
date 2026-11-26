"""Generalized DHO + slow fit, including overdamped modes.

The underdamped DHO used in earlier fits has a fixed functional form
exp(-gamma t/2)(cos(w t) + gamma/(2w) sin(w t)).  The water COM velocity ACF
is smooth and monotonic over the first 60 fs, so the only way the earlier
3-mode fit could reach the 5 fs point was an aliased w~1.25 rad/fs mode
(period ~5 fs); at a 5 fs stride its discrete integral is 1.9x too large.

This script allows the physical DHO to be underdamped or overdamped by
parameterizing the natural frequency Omega and damping gamma:

    w_d^2 = Omega^2 - (gamma/2)^2
    ACF(t) = exp(-gamma t/2) [cos(w_d t) + gamma/(2 w_d) sin(w_d t)]
           = exp(-gamma t/2) [cosh(kappa t) + gamma/(2 kappa) sinh(kappa t)]
             with kappa = sqrt((gamma/2)^2 - Omega^2)

The exact 2x2 state-space is built for both cases (rotation form for
underdamped, diagonal form with a signed observation for overdamped).  The
fit uses the sampled 5 fs ACF and constrains the discrete trapezoid integral.

Still CPU only, no rollout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

KEY_LAGS = np.array([5, 25, 100, 200, 500, 1000, 2000, 5000])


def general_dho(t, gamma, omega):
    """Velocity ACF of a damped harmonic oscillator with natural frequency Omega.

    Omega may be larger than gamma/2 (underdamped) or smaller (overdamped);
    the formula is the same analytic continuation.
    """
    g = 0.5 * gamma
    disc = omega * omega - g * g
    out = np.empty_like(t, dtype=np.float64)
    under = disc > 1e-12
    over = disc < -1e-12
    critical = ~(under | over)
    if np.any(under):
        w = math.sqrt(disc)
        x = t[under]
        out[under] = np.exp(-g * x) * (np.cos(w * x) + g / w * np.sin(w * x))
    if np.any(over):
        k = math.sqrt(-disc)
        x = t[over]
        # Stable exponential form; cosh/sinh overflow before exp(-g t) for
        # strongly overdamped modes.
        c1 = 0.5 * (1.0 + g / k)
        c2 = 0.5 * (1.0 - g / k)
        out[over] = c1 * np.exp(-(g - k) * x) + c2 * np.exp(-(g + k) * x)
    if np.any(critical):
        x = t[critical]
        out[critical] = np.exp(-g * x) * (1.0 + g * x)
    return out


def unpack(x, modes):
    logits = x[:modes + 1]
    logits = logits - logits.max()
    weights = np.exp(logits) / np.exp(logits).sum()
    gamma = np.exp(x[modes + 1:2 * modes + 1])
    omega = np.exp(x[2 * modes + 1:3 * modes + 1])
    tau = np.exp(x[3 * modes + 1])
    return weights, gamma, omega, tau


def model_acf(t, x, modes):
    weights, gamma, omega, tau = unpack(x, modes)
    out = np.zeros_like(t, dtype=np.float64)
    for j in range(modes):
        out += weights[j] * general_dho(t, gamma[j], omega[j])
    out += weights[-1] * np.exp(-t / tau)
    return out


def model_integral(x, modes):
    weights, gamma, omega, tau = unpack(x, modes)
    value = weights[-1] * tau
    for j in range(modes):
        value += weights[j] * gamma[j] / (omega[j] ** 2)
    return value


def discrete_embedding(x, modes, horizon_fs=5.0, max_lag_fs=5000.0):
    """Exact 2x2 state-space for each generalized DHO, sampled at horizon_fs."""
    weights, gamma, omega, tau = unpack(x, modes)
    maxlag = int(round(max_lag_fs / horizon_fs))
    acf = np.zeros(maxlag + 1, dtype=np.float64)
    info = []
    for j in range(modes):
        g = 0.5 * gamma[j]
        Om = omega[j]
        disc = Om * Om - g * g
        if disc > 1e-12:
            w = math.sqrt(disc)
            sigma = np.array([[weights[j], -weights[j] * g / w],
                              [-weights[j] * g / w,
                               weights[j] * (1.0 + 2.0 * (g / w) ** 2)]])
            r = math.exp(-g * horizon_fs)
            theta = w * horizon_fs
            a_d = r * np.array([[math.cos(theta), -math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])
            q_d = sigma - a_d @ sigma @ a_d.T
            q_d = 0.5 * (q_d + q_d.T)
            obs = np.array([1.0, 0.0])
            kind = "underdamped"
            period = 2.0 * math.pi / w
        elif disc < -1e-12:
            k = math.sqrt(-disc)
            lam1, lam2 = g - k, g + k
            c1 = 0.5 * (1.0 + g / k)
            c2 = 0.5 * (1.0 - g / k)
            # Diagonal OU with unit stationary variance; observed process is
            # c1*z1 + c2*z2.  A negative c2 is allowed (physical DHO).
            a_d = np.diag([math.exp(-lam1 * horizon_fs),
                           math.exp(-lam2 * horizon_fs)])
            q_d = np.diag([1.0 - math.exp(-2.0 * lam1 * horizon_fs),
                           1.0 - math.exp(-2.0 * lam2 * horizon_fs)])
            obs = np.array([math.sqrt(max(c1, 0.0)),
                            math.copysign(math.sqrt(abs(c2)), c2)])
            kind = "overdamped"
            period = float("inf")
        else:
            # Critical: Jordan block.  Use a tiny underdamped offset.
            w = 1e-8
            sigma = np.array([[weights[j], -weights[j] * g / w],
                              [-weights[j] * g / w,
                               weights[j] * (1.0 + 2.0 * (g / w) ** 2)]])
            r = math.exp(-g * horizon_fs)
            theta = w * horizon_fs
            a_d = r * np.array([[math.cos(theta), -math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])
            q_d = sigma - a_d @ sigma @ a_d.T
            q_d = 0.5 * (q_d + q_d.T)
            obs = np.array([1.0, 0.0])
            kind = "critical"
            period = float("inf")
        eig_q = np.linalg.eigvalsh(q_d)
        eig_a = np.abs(np.linalg.eigvals(a_d))
        # ACF: obs^T A_d^k Sigma obs (underdamped) or obs^T A_d^k diag(1) obs
        # (overdamped).  Use the stationary covariance that gives the fitted
        # normalized ACF.
        if kind == "overdamped":
            p = np.eye(2)
        else:
            p = sigma
        for lag in range(maxlag + 1):
            acf[lag] += float(obs @ p @ obs)
            p = a_d @ p
        info.append(dict(
            mode=j, kind=kind, weight=float(weights[j]),
            gamma_per_fs=float(gamma[j]), omega_natural_rad_per_fs=float(Om),
            period_fs=float(period), decay_fs=float(2.0 / gamma[j]),
            a_d=a_d.tolist(), q_d=q_d.tolist(), observation=obs.tolist(),
            q_d_eigenvalues=eig_q.tolist(),
            q_d_min_eigenvalue=float(eig_q.min()),
            a_d_spectral_radius=float(eig_a.max()),
        ))
    a = math.exp(-horizon_fs / tau)
    q_slow = weights[-1] * (1.0 - a * a)
    acf += weights[-1] * a ** np.arange(maxlag + 1)
    acf /= weights.sum()
    info.append(dict(mode="slow", weight=float(weights[-1]),
                     tau_fs=float(tau), a=float(a), q=float(q_slow)))
    return acf, info


def fit(mean, t, modes, n_starts=24, seed=20261126, omega_max=1.0,
        fit_min_fs=5.0, key_weight=8.0):
    integral_ref = float(np.trapezoid(mean, t))
    fit_mask = (t >= fit_min_fs) & (t <= 5000.0)
    t_fit = t[fit_mask]
    ref_fit = mean[fit_mask]
    scale = 0.02 + np.abs(ref_fit)
    key = KEY_LAGS
    ref_key = np.interp(key, t, mean)

    def residual(x):
        pred = model_acf(t_fit, x, modes)
        r = (pred - ref_fit) / scale
        pred_key = np.interp(key, t, model_acf(t, x, modes))
        r_key = key_weight * (pred_key - ref_key) / (0.02 + np.abs(ref_key))
        full = model_acf(t, x, modes)
        r_int = 40.0 * (np.trapezoid(full, t) - integral_ref) / integral_ref
        return np.concatenate([r, r_key, [r_int]])

    rng = np.random.default_rng(seed + 1000 * modes)
    best = None
    costs = []
    for _ in range(n_starts):
        gamma0 = np.exp(rng.uniform(-5.0, 0.0, modes))
        # Half of the starts are explicitly overdamped (Omega < gamma/2),
        # where the DHO ACF is a negative-weight sum of two exponentials.
        # The remaining starts are underdamped. A purely log-uniform Omega
        # almost never lands in the narrow overdamped region.
        kind = rng.random(modes) < 0.5
        omega0 = np.empty(modes)
        if np.any(kind):
            omega0[kind] = gamma0[kind] * rng.uniform(0.05, 0.45, kind.sum())
        if np.any(~kind):
            omega0[~kind] = gamma0[~kind] * rng.uniform(0.55, 3.0, (~kind).sum())
        omega0 = np.clip(omega0, 1e-4, omega_max)
        tau0 = np.exp(rng.uniform(np.log(100.0), np.log(5000.0)))
        x0 = np.concatenate([np.zeros(modes + 1), np.log(gamma0),
                             np.log(omega0), [np.log(tau0)]])
        lower = np.concatenate([np.full(modes + 1, -15.0),
                                np.full(modes, np.log(1e-4)),
                                np.full(modes, np.log(1e-4)),
                                [np.log(20.0)]])
        upper = np.concatenate([np.full(modes + 1, 15.0),
                                np.full(modes, np.log(4.0)),
                                np.full(modes, np.log(omega_max)),
                                [np.log(20000.0)]])
        try:
            result = least_squares(residual, x0, bounds=(lower, upper),
                                   max_nfev=4000, xtol=1e-12, ftol=1e-12,
                                   gtol=1e-12)
        except Exception:
            continue
        cost = float(np.sum(result.fun ** 2))
        costs.append(cost)
        if best is None or cost < best[0]:
            best = (cost, result.x, result.jac)
    if best is None:
        raise RuntimeError("all fits failed")
    return best, integral_ref, np.asarray(costs)


def evaluate(x, modes, mean, t, blocks, horizon_fs=5.0):
    acf, info = discrete_embedding(x, modes, horizon_fs, max_lag_fs=t[-1])
    weights, gamma, omega, tau = unpack(x, modes)
    integral = float(np.trapezoid(acf, t))
    integral_ref = float(np.trapezoid(mean, t))
    key = KEY_LAGS
    pred_key = np.interp(key, t, acf)
    mean_key = np.interp(key, t, mean)
    block_rmse, block_integrals = [], []
    for block in blocks:
        block_rmse.append(float(np.sqrt(np.mean(
            (pred_key - np.interp(key, t, block)) ** 2))))
        block_integrals.append(float(np.trapezoid(block, t)))
    return dict(
        modes=modes,
        weights=weights.tolist(),
        gamma_per_fs=gamma.tolist(),
        omega_natural_rad_per_fs=omega.tolist(),
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
    parser.add_argument("--n-starts", type=int, default=24)
    parser.add_argument("--omega-max", type=float, default=1.0)
    parser.add_argument("--fit-min-fs", type=float, default=5.0)
    parser.add_argument("--key-weight", type=float, default=8.0)
    args = parser.parse_args()

    data = np.load(args.npz)
    acfs = data["acfs"]
    t = data["t"]
    mean = acfs.mean(0)
    results = []
    for modes in args.modes:
        (cost, x, jac), integral_ref, costs = fit(
            mean, t, modes, args.n_starts, omega_max=args.omega_max,
            fit_min_fs=args.fit_min_fs, key_weight=args.key_weight)
        row = evaluate(x, modes, mean, t, acfs)
        row["fit_cost"] = float(cost)
        row["fit_cost_spread"] = float(np.std(costs)) if len(costs) > 1 else 0.0
        row["jacobian_condition"] = float(np.linalg.cond(jac))
        row["omega_max"] = float(args.omega_max)
        row["fit_min_fs"] = float(args.fit_min_fs)
        row["key_weight"] = float(args.key_weight)
        results.append(row)
        print(json.dumps({k: row[k] for k in
                          ["modes", "integral_fs", "integral_ref_fs",
                           "integral_ratio", "rmse_mean", "rmse_key",
                           "max_abs_key", "discrete_min_eigenvalue",
                           "discrete_spectral_radius_max", "n_parameters",
                           "jacobian_condition"]}, indent=2), flush=True)
        print("kinds", [e.get("kind", "slow") for e in row["embedding"]],
              "periods", [round(e.get("period_fs", 0.0), 2) for e in row["embedding"]],
              "weights", [round(w, 4) for w in row["weights"]], flush=True)

    payload = dict(
        npz=str(args.npz),
        omega_max=float(args.omega_max),
        fit_min_fs=float(args.fit_min_fs),
        key_weight=float(args.key_weight),
        note=("Generalized DHO+slow fit, underdamped and overdamped. Fits the "
              "sampled 5 fs ACF and constrains the discrete Green-Kubo "
              "integral. The 1 fs reference ACF is smooth, so near-Nyquist "
              "oscillatory modes are treated as aliasing artifacts."),
        results=results,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
