"""Audit the production GLR AR coefficients: stability, extrapolated ACF, D.

The production file `glr_coefficients.json` has no reference path/start
metadata.  This script records what can be recovered from the coefficients
themselves and compares the AR-implied translation ACF integral with the
25-block reference distribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEY_LAGS = np.array([5, 25, 100, 200, 500, 1000, 2000, 5000])


def ar_roots(coefficients):
    # 1 - c1 z - ... - cp z^p = 0; roots of the companion polynomial.
    c = np.asarray(coefficients, dtype=float)
    poly = np.concatenate([[1.0], -c])
    return np.roots(poly)


def extend_acf(coefficients, seed_acf, maxlag):
    c = np.asarray(coefficients, dtype=float)
    order = len(c)
    acf = np.asarray(seed_acf, dtype=float).copy()
    if len(acf) < order + 1:
        raise ValueError("seed ACF shorter than AR order")
    acf = acf[:maxlag + 1]
    for lag in range(len(acf), maxlag + 1):
        value = sum(c[k - 1] * acf[lag - k] for k in range(1, order + 1))
        acf = np.append(acf, value)
    return acf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--reference-blocks", type=Path,
                        default=Path("experiments/reference_acf_blocks_20261126.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride-fs", type=float, default=5.0)
    parser.add_argument("--max-lag-fs", type=float, default=5000.0)
    args = parser.parse_args()

    data = json.loads(args.coefficients.read_text())
    blocks = json.loads(args.reference_blocks.read_text())
    block_integrals = np.array([b["integral_fs"] for b in blocks["blocks"]])
    mean_integral = float(block_integrals.mean())
    maxlag = int(round(args.max_lag_fs / args.stride_fs))
    t = np.arange(maxlag + 1) * args.stride_fs
    key = KEY_LAGS.astype(float)

    result = dict(
        coefficients=str(args.coefficients),
        order=data.get("order"),
        stride_fs=data.get("stride_fs"),
        metadata_present={k: k in data for k in
                          ("reference", "start_ps", "duration_ps", "maxlag")},
        reference_block_mean_fs=mean_integral,
        reference_block_std_fs=float(block_integrals.std(ddof=1)),
        reference_block_min_fs=float(block_integrals.min()),
        reference_block_max_fs=float(block_integrals.max()),
        modes={},
    )
    for mode, values in data.get("modes", {}).items():
        c = np.asarray(values["coefficients"], dtype=float)
        roots = ar_roots(c)
        radius = float(np.max(np.abs(roots)))
        seed = np.asarray(values.get("ar_acf", values.get("acf")), dtype=float)
        acf = extend_acf(c, seed, maxlag)
        acf = acf / acf[0]
        integral = float(np.trapezoid(acf, t))
        result["modes"][mode] = dict(
            coefficients=c.tolist(),
            prediction_error_ratio=values.get("prediction_error_ratio"),
            ar_spectral_radius=radius,
            ar_stable=bool(radius < 1.0),
            integral_fs=integral,
            integral_ratio_to_block_mean=float(integral / mean_integral),
            key_fs=key.tolist(),
            acf_key=np.interp(key, t, acf).tolist(),
            reference_acf_key=(
                [float(values["acf"][int(round(lag / args.stride_fs))])
                 if int(round(lag / args.stride_fs)) < len(values["acf"]) else None
                 for lag in key]
                if "acf" in values else None),
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "modes"}, indent=2))
    for mode, values in result["modes"].items():
        print(mode, "radius", round(values["ar_spectral_radius"], 4),
              "stable", values["ar_stable"],
              "integral", round(values["integral_fs"], 4),
              "ratio", round(values["integral_ratio_to_block_mean"], 4),
              "key", [round(x, 4) for x in values["acf_key"]])


if __name__ == "__main__":
    main()
