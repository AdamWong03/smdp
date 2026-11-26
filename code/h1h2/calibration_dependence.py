"""Calibration-window dependence of the GLR AR(10) fit.

Uses the independent train_nvt_200ps reference only (never the 2.5 ns test
trajectory) to compare:

  * 100 ps block 0 (0-100 ps);
  * 100 ps block 1 (100-200 ps);
  * the full 200 ps;
  * the production glr_fit_20261122 artifact.

For each fit: AR stability, AR-implied 5 ps ACF integral, and the actual
5 ps ACF integral of every window.  Then compares the AR-implied diffusion
with the actual autonomous GLR diffusion.  The goal is to separate a
calibration-data problem from an AR-model/readout-coupling problem.

Note: fit_glr.py's Yule-Walker system uses ACF lags 0..order (0-50 fs at
order 10, stride 5 fs).  --maxlag=40 only extends the reported ACF to 200 fs;
it does not constrain the fit.  This script records that explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gle_kernel_study import translation_acf

ROOT = Path("/data2/wyd/research/foundation_mlip_adaptation")
TRAIN = Path("/data1/wyd/research/foundation_mlip_adaptation/data/"
             "smdp_reference_md_v1/water/physical_reference/train_nvt_200ps/production.h5")
KEY = np.array([5, 25, 100, 200, 500, 1000, 2000, 5000], dtype=float)


def load_audit(path):
    data = json.loads(Path(path).read_text())
    return data["modes"]["translation"]


def main():
    windows = {
        "100a": dict(start=0.0, duration=100.0,
                     fit="experiments/calib_independent_100a_20261127/glr_coefficients.json",
                     audit="experiments/calib_independent_100a_20261127/ar_audit.json"),
        "100b": dict(start=100.0, duration=100.0,
                     fit="experiments/calib_independent_100b_20261127/glr_coefficients.json",
                     audit="experiments/calib_independent_100b_20261127/ar_audit.json"),
        "200": dict(start=0.0, duration=200.0,
                    fit="experiments/calib_independent_200_20261127/glr_coefficients.json",
                    audit="experiments/calib_independent_200_20261127/ar_audit.json"),
    }
    # Actual 5 ps ACF of each independent window.
    actual = {}
    for name, spec in windows.items():
        acf, dt = translation_acf(TRAIN, spec["start"], spec["duration"], 5.0, 5000.0)
        t = np.arange(len(acf)) * dt
        spec["actual_acf"] = acf
        spec["actual_t"] = t
        spec["actual_integral_fs"] = float(np.trapezoid(acf, t))
        spec["actual_key"] = np.interp(KEY, t, acf).tolist()
        actual[name] = spec
    # Production artifact has no recorded window; compare its AR ACF to the
    # independent windows and to the 2.5 ns 25-block mean (evaluation only,
    # not used for fitting).
    production = dict(
        fit="experiments/glr_fit_20261122/glr_coefficients.json",
        audit="experiments/glr_coefficients_audit_20261127.json",
    )
    production_audit = load_audit(ROOT / production["audit"])

    results = {}
    for name, spec in windows.items():
        audit = load_audit(ROOT / spec["audit"])
        ar_key = np.asarray(audit["acf_key"], dtype=float)
        actual_key = np.asarray(spec["actual_key"], dtype=float)
        local = KEY <= 200
        long = KEY >= 500
        results[name] = dict(
            window_ps=[spec["start"], spec["start"] + spec["duration"]],
            fit=str(ROOT / spec["fit"]),
            yule_walker_lags_fs=[0.0, 10 * 5.0],
            reported_maxlag_fs=200.0,
            ar_spectral_radius=audit["ar_spectral_radius"],
            ar_stable=audit["ar_stable"],
            ar_integral_fs=audit["integral_fs"],
            actual_integral_fs=spec["actual_integral_fs"],
            ar_over_actual_integral=float(audit["integral_fs"] / spec["actual_integral_fs"]),
            ar_key=ar_key.tolist(),
            actual_key=actual_key.tolist(),
            key_fs=KEY.tolist(),
            local_rmse=float(np.sqrt(np.mean((ar_key[local] - actual_key[local]) ** 2))),
            long_rmse=float(np.sqrt(np.mean((ar_key[long] - actual_key[long]) ** 2))),
            ar_green_kubo_D_m2_s=audit.get("ar_green_kubo_D_m2_s"),
        )

    # Cross-window prediction: AR ACF from one 100 ps fit vs the other 100 ps
    # actual ACF, local (<=200 fs) and long (>=500 fs).
    cross = {}
    for src in ("100a", "100b", "200"):
        src_key = np.asarray(results[src]["ar_key"], dtype=float)
        for dst in ("100a", "100b", "200"):
            dst_key = np.asarray(results[dst]["actual_key"], dtype=float)
            local = KEY <= 200
            long = KEY >= 500
            cross[f"{src}_to_{dst}"] = dict(
                local_rmse=float(np.sqrt(np.mean((src_key[local] - dst_key[local]) ** 2))),
                long_rmse=float(np.sqrt(np.mean((src_key[long] - dst_key[long]) ** 2))),
            )

    payload = dict(
        calibration_data=str(TRAIN),
        fit_script="code/h1h2/fit_glr.py",
        fit_order=10,
        fit_stride_fs=5.0,
        actual_yule_walker_window_fs=50.0,
        reported_maxlag_fs=200.0,
        note=("fit_glr.py solves Yule-Walker with ACF lags 0..order (0-50 fs for "
              "order 10, stride 5 fs). --maxlag=40 only extends the reported ACF "
              "to 200 fs; it does not constrain the coefficients. The 100-500 fs "
              "negative dip is therefore outside the fit."),
        windows=results,
        cross_window=cross,
        production=dict(
            fit=str(ROOT / production["fit"]),
            audit=str(ROOT / production["audit"]),
            ar_spectral_radius=production_audit["ar_spectral_radius"],
            ar_integral_fs=production_audit["integral_fs"],
            ar_key=production_audit["acf_key"],
            note="No reference path/start/duration recorded in the artifact.",
        ),
        autonomous_glr=dict(
            D_msd_m2_s=1.4756e-09,
            D_msd_std_m2_s=0.228e-09,
            D_gk_m2_s=1.3917e-09,
            D_gk_std_m2_s=0.1266e-09,
            source="experiments/position_velocity_audit_20261127.json, 8 blocks",
            reference_D_msd_m2_s=1.8903e-09,
            reference_D_gk_m2_s=1.9433e-09,
        ),
        interpretation=dict(
            calibration_length_effect=("100 ps vs 200 ps changes the AR(10) "
                                       "coefficients only slightly; the AR "
                                       "spectral radius is 0.853-0.855 and the "
                                       "AR integral is 32.6-33.5 fs in all "
                                       "three independent windows."),
            short_window_inconsistency=("The AR integral is 2.26-2.32x the "
                                        "actual 5 ps ACF integral of the same "
                                        "calibration window; the AR-implied D "
                                        "is ~4.4e-9 vs the actual window D "
                                        "~1.8-2.0e-9 and the autonomous GLR D "
                                        "~1.4e-9."),
            transfer_failure=("The AR fit overpredicts D, while the autonomous "
                              "GLR underpredicts it. The offline AR ACF is not "
                              "the effective velocity/position dynamics of the "
                              "coupled rollout."),
        ),
    )
    out = ROOT / "experiments/calibration_dependence_20261127.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("ar_spectral_radius", "ar_integral_fs",
                                    "actual_integral_fs", "ar_over_actual_integral",
                                    "local_rmse", "long_rmse")}
                      for k, v in results.items()}, indent=2))
    print("cross", json.dumps(cross, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
