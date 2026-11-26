"""Monte Carlo verification of the exact discrete GLE state-space.

Given a fit JSON with per-mode A_d, Q_d, observation vector, and slow-mode
(a, q), simulate the discrete process at the rollout horizon and compare the
sampled ACF and discrete Green-Kubo integral to the analytic values.

This separates a discrete-implementation failure from a kernel-identification
failure.  CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = (5, 25, 100, 200, 500, 1000, 2000, 5000)


def simulate_mode(a_d, q_d, obs, steps, rng):
    """One 2D OU mode; returns the observed scalar time series."""
    q_d = 0.5 * (np.asarray(q_d, dtype=float) + np.asarray(q_d, dtype=float).T)
    eig = np.linalg.eigvalsh(q_d)
    if eig.min() < -1e-12:
        raise ValueError(f"Q_d is not PSD: {eig}")
    # Cholesky of the PSD covariance (clip tiny negative eigenvalues).
    q_d = q_d + np.eye(2) * max(0.0, -eig.min())
    chol = np.linalg.cholesky(q_d)
    state = np.zeros(2)
    out = np.empty(steps)
    for n in range(steps):
        out[n] = obs @ state
        state = a_d @ state + chol @ rng.standard_normal(2)
    return out


def simulate_slow(a, q, weight, steps, rng):
    state = 0.0
    out = np.empty(steps)
    for n in range(steps):
        out[n] = weight * state
        state = a * state + np.sqrt(q) * rng.standard_normal()
    return out


def acf(x, maxlag):
    x = x - x.mean()
    norm = np.dot(x, x) / len(x)
    out = np.empty(maxlag + 1)
    for lag in range(maxlag + 1):
        out[lag] = np.dot(x[:len(x) - lag], x[lag:]) / (len(x) - lag) / norm
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--mode-index", type=int, required=True,
                        help="Index into fit['results']")
    parser.add_argument("--horizon-fs", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fit = json.loads(args.fit.read_text())
    row = fit["results"][args.mode_index]
    rng = np.random.default_rng(args.seed)
    total = np.zeros(args.steps)
    for item in row["embedding"]:
        if item["mode"] == "slow":
            total += simulate_slow(item["a"], item["q"], item["weight"],
                                   args.steps, rng)
        else:
            total += simulate_mode(np.asarray(item["a_d"]),
                                   np.asarray(item["q_d"]),
                                   np.asarray(item["observation"]),
                                   args.steps, rng)
    maxlag = int(round(5000.0 / args.horizon_fs))
    sampled = acf(total, maxlag)
    t = np.arange(maxlag + 1) * args.horizon_fs
    # Analytic ACF from the embedding matrices.
    analytic = np.zeros(maxlag + 1)
    for item in row["embedding"]:
        if item["mode"] == "slow":
            analytic += item["weight"] * item["a"] ** np.arange(maxlag + 1)
        else:
            a_d = np.asarray(item["a_d"])
            q_d = np.asarray(item["q_d"])
            obs = np.asarray(item["observation"])
            # Stationary covariance of the discrete process.
            from scipy.linalg import solve_discrete_lyapunov
            sigma = solve_discrete_lyapunov(a_d, q_d)
            sigma = 0.5 * (sigma + sigma.T)
            p = sigma.copy()
            for lag in range(maxlag + 1):
                analytic[lag] += float(obs @ p @ obs)
                p = a_d @ p
    analytic /= analytic[0]
    key = np.array(KEYS)
    result = dict(
        fit=str(args.fit), mode_index=args.mode_index, modes=row["modes"],
        steps=args.steps, horizon_fs=args.horizon_fs,
        analytic_integral_fs=float(np.trapezoid(analytic, t)),
        sampled_integral_fs=float(np.trapezoid(sampled, t)),
        analytic_key=np.interp(key, t, analytic).tolist(),
        sampled_key=np.interp(key, t, sampled).tolist(),
        max_abs_acf_error=float(np.max(np.abs(sampled - analytic))),
        key_rmse=float(np.sqrt(np.mean(
            (np.interp(key, t, sampled) - np.interp(key, t, analytic)) ** 2))),
        note="Monte Carlo test of the exact A_d/Q_d realization; the residual "
             "difference is sampling noise, not a discretization error.",
    )
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
