"""Mode-resolved diagnostics for quartz phonon-GLR trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

KINETIC_TO_EV = 103.64269
KB_EV_PER_K = 8.617333262e-5
BANDS = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 30), (30, 40)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--basis", type=Path,
                        default=Path("/home/wyd/quartz_phonon/phonon_basis.npz"))
    parser.add_argument("--coefficients", type=Path,
                        default=Path("/home/wyd/quartz_phonon/ar10_diagnostics.npz"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--acf-frames", type=int, default=2000)
    parser.add_argument("--maxlag", type=int, default=100)
    args = parser.parse_args()

    basis = np.load(args.basis, allow_pickle=True)
    ref = np.load(args.coefficients, allow_pickle=True)
    acoustic = basis["acoustic_mask"].astype(bool)
    keep = ~acoustic
    eigenvectors = basis["eigenvectors"].astype(np.float64)[:, keep]
    masses = basis["masses"].astype(np.float64)
    m12 = np.repeat(np.sqrt(masses), 3)
    frequencies = basis["frequencies_thz"].astype(np.float64)[keep]
    ref_acf = ref["acf"].astype(np.float64)
    ref_freq = ref["fthz"].astype(np.float64)
    ref_acoustic = ref["acoustic"].astype(bool)

    with h5py.File(args.trajectory, "r") as handle:
        velocities = handle["velocities"][:].astype(np.float64)
        atomic_numbers = handle["atomic_numbers"][:].astype(int)
        h5_masses = handle["masses_u"][:].astype(np.float64)
    if not np.allclose(h5_masses, masses):
        raise ValueError("basis mass order does not match trajectory")

    amplitudes = (velocities.reshape(len(velocities), -1) * m12) @ eigenvectors
    kinetic = 0.5 * KINETIC_TO_EV * amplitudes ** 2
    target = 0.5 * KB_EV_PER_K * args.temperature
    total_ke = kinetic.sum(axis=1)
    dofs = 3 * len(masses) - 3
    temperature = 2.0 * total_ke / (dofs * KB_EV_PER_K)

    result = {
        "frames": int(len(velocities)),
        "n_modes": int(eigenvectors.shape[1]),
        "temperature_K": {
            "mean": float(temperature.mean()),
            "std": float(temperature.std()),
            "min": float(temperature.min()),
            "max": float(temperature.max()),
        },
        "total_ke_eV": {
            "mean": float(total_ke.mean()),
            "std": float(total_ke.std()),
            "reference_std": float(np.sqrt(2.0 * eigenvectors.shape[1]) * target),
        },
        "mode_ke_eV": {
            "target": float(target),
            "mean": float(kinetic.mean()),
            "fraction_within_10pct": float(np.mean(
                np.abs(kinetic.mean(axis=0) / target - 1.0) < 0.1)),
            "fraction_within_20pct": float(np.mean(
                np.abs(kinetic.mean(axis=0) / target - 1.0) < 0.2)),
            "per_mode_std_median": float(np.median(kinetic.std(axis=0))),
        },
        "species_ke_eV": {},
        "mode_acf_error": {},
    }
    for element in np.unique(atomic_numbers):
        mask = atomic_numbers == element
        species_ke = 0.5 * KINETIC_TO_EV * np.sum(
            masses[mask, None] * velocities[:, mask, :] ** 2, axis=(1, 2))
        result["species_ke_eV"][str(int(element))] = {
            "mean": float(species_ke.mean()),
            "per_atom": float(species_ke.mean() / mask.sum()),
        }

    n = min(args.acf_frames, len(amplitudes))
    sample = amplitudes[:n]
    maxlag = min(args.maxlag, n - 2)
    acf = np.empty((maxlag + 1, sample.shape[1]))
    for lag in range(maxlag + 1):
        acf[lag] = np.sum(sample[:n - lag] * sample[lag:], axis=0) / (n - lag)
    acf /= acf[0]

    for low, high in BANDS:
        model = (np.abs(frequencies) >= low) & (np.abs(frequencies) < high)
        reference = ((~ref_acoustic) & (np.abs(ref_freq) >= low)
                     & (np.abs(ref_freq) < high))
        if not model.any() or not reference.any():
            continue
        result["mode_acf_error"][f"{low}-{high}"] = {
            "n": int(model.sum()),
            "d100fs": float(np.median(np.abs(acf[20, model] - ref_acf[20, reference]))),
            "d250fs": float(np.median(np.abs(acf[50, model] - ref_acf[50, reference]))),
            "d500fs": float(np.median(np.abs(acf[100, model] - ref_acf[100, reference]))),
        }

    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
