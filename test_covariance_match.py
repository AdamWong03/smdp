"""Offline test: can a constant mass-weighted covariance-matching transform
fix the velocity/KE covariance without destroying the ACF and Green-Kubo D?

No rollout and no GPU. Reads two trajectories at a common 5 fs stride,
decomposes molecular modes, estimates the mass-weighted covariances, applies
A = C_ref^{1/2} C_model^{-1/2}, and reports KE, ACF, and D before/after.
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
KEYS = ("translation", "rotation", "internal")


def load_fields(path: Path, frames: int, stride_fs: float, start: int = 0):
    with h5py.File(path, "r") as handle:
        dt = float(handle.attrs.get("timestep_fs",
                                    handle.attrs.get("output_interval_fs", stride_fs)))
        stride = max(1, int(round(stride_fs / dt)))
        pos_key = "positions" if "positions" in handle else "positions_unwrapped_A"
        vel_key = "velocities" if "velocities" in handle else "velocities_A_per_fs"
        cell = handle["cell_A"][0] if "cell_A" in handle else handle["cell"][:]
        masses = handle["masses_u"][:].astype(np.float64)
        molecules = handle["molecule_ids"][:].astype(int)
        indices = start + np.arange(frames) * stride
        q = handle[pos_key][indices].astype(np.float64)
        v = handle[vel_key][indices].astype(np.float64)
    fields = {key: [] for key in KEYS}
    for index in range(len(q)):
        part = decompose_modes(
            torch.as_tensor(q[index]), torch.as_tensor(v[index]),
            torch.as_tensor(masses), torch.as_tensor(molecules),
            torch.as_tensor(cell))
        for key in KEYS:
            fields[key].append(part[key].numpy())
    return {key: np.stack(value) for key, value in fields.items()}, masses, stride


def mass_weight(field, masses):
    return field * np.sqrt(masses)[None, :, None]


def covariance(field):
    x = field.reshape(len(field), -1)
    x = x - x.mean(axis=0, keepdims=True)
    return x.T @ x / len(x)


def psd_sqrt(matrix, floor_ratio=1e-10):
    w, v = np.linalg.eigh((matrix + matrix.T) / 2.0)
    scale = max(float(w.max()), 1e-30)
    w = np.clip(w, floor_ratio * scale, None)
    return v @ np.diag(np.sqrt(w)) @ v.T, w


def inv_psd_sqrt(matrix, floor_ratio=1e-10):
    w, v = np.linalg.eigh((matrix + matrix.T) / 2.0)
    scale = max(float(w.max()), 1e-30)
    w = np.clip(w, floor_ratio * scale, None)
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T, w


def kinetic_weighted(field):
    return 0.5 * KINETIC_TO_EV * np.sum(field ** 2, axis=(1, 2))


def acf(field, maxlag):
    x = field.reshape(len(field), -1)
    x = x - x.mean(axis=0, keepdims=True)
    zero = float(np.mean(np.sum(x * x, axis=1)))
    values = np.empty(maxlag + 1)
    for lag in range(maxlag + 1):
        values[lag] = np.mean(np.sum(x[:len(x) - lag] * x[lag:], axis=1)) / zero
    return values


def green_kubo(field, masses, dt_fs=5.0, maxlag=1000):
    zero = float(np.sum(field * field) / len(field))
    values = np.empty(maxlag + 1)
    values[0] = zero
    for lag in range(1, maxlag + 1):
        values[lag] = np.sum(field[:-lag] * field[lag:]) / (len(field) - lag)
    vacuum = values / masses.sum()
    lags = np.arange(len(values)) * dt_fs
    return float(np.trapezoid(vacuum, lags) / 3.0 * 1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--stride-fs", type=float, default=5.0)
    parser.add_argument("--maxlag", type=int, default=100)
    args = parser.parse_args()

    model, masses, model_stride = load_fields(
        args.model, args.frames, args.stride_fs)
    reference, ref_masses, ref_stride = load_fields(
        args.reference, args.frames, args.stride_fs)
    if not np.allclose(masses, ref_masses):
        raise ValueError("model and reference atom order/masses differ")

    result = {
        "frames": args.frames,
        "stride_fs": args.stride_fs,
        "model_stride": model_stride,
        "reference_stride": ref_stride,
        "groups": {},
    }
    for key in KEYS:
        um = mass_weight(model[key], masses)
        ur = mass_weight(reference[key], masses)
        cm = covariance(um)
        cr = covariance(ur)
        sqrt_ref, w_ref = psd_sqrt(cr)
        inv_sqrt_model, w_model = inv_psd_sqrt(cm)
        transform = sqrt_ref @ inv_sqrt_model
        un = np.einsum("ij,nj->ni", transform, um.reshape(len(um), -1))
        un = un.reshape(ur.shape)

        ke_model = kinetic_weighted(um)
        ke_ref = kinetic_weighted(ur)
        ke_new = kinetic_weighted(un)

        acf_model = acf(um, args.maxlag)
        acf_ref = acf(ur, args.maxlag)
        acf_new = acf(un, args.maxlag)

        entry = {
            "condition_model_floor_ratio": float(w_model.max() / max(w_model.min(), 1e-30)),
            "condition_ref_floor_ratio": float(w_ref.max() / max(w_ref.min(), 1e-30)),
            "covariance_relative_error_after": float(
                np.linalg.norm(covariance(un) - cr) / max(np.linalg.norm(cr), 1e-30)),
            "ke_mean_model": float(ke_model.mean()),
            "ke_std_model": float(ke_model.std()),
            "ke_mean_new": float(ke_new.mean()),
            "ke_std_new": float(ke_new.std()),
            "ke_mean_ref": float(ke_ref.mean()),
            "ke_std_ref": float(ke_ref.std()),
            "acf_model": acf_model[[1, 5, 20, 100]].tolist(),
            "acf_new": acf_new[[1, 5, 20, 100]].tolist(),
            "acf_ref": acf_ref[[1, 5, 20, 100]].tolist(),
        }
        if key == "translation":
            entry["gk_model"] = green_kubo(um, masses)
            entry["gk_new"] = green_kubo(un, masses)
            entry["gk_ref"] = green_kubo(ur, masses)
        result["groups"][key] = entry

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
