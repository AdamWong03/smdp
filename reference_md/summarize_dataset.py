#!/usr/bin/env python3
"""Compute basic physical and integrity statistics for formal reference MD."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Pair-distance validation is small; unrestricted BLAS/OpenMP startup on each
# sampled frame only wastes shared-server CPU.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import numpy as np
from ase.geometry import find_mic


def trajectory_summary(path: Path, distance_samples: int = 101):
    with h5py.File(path, "r") as h5:
        n_frames = len(h5["step"]); n_atoms = len(h5["atomic_numbers"])
        time = np.asarray(h5["thermo/time_fs"]) / 1000.0
        total = np.asarray(h5["thermo/total_energy_eV"])
        temp = np.asarray(h5["thermo/temperature_K"])
        slope = float(np.polyfit(time, total, 1)[0]) if len(time) > 1 else float("nan")
        indices = np.unique(np.linspace(0, n_frames - 1, min(distance_samples, n_frames), dtype=int))
        minimum = np.inf
        for index in indices:
            positions = np.asarray(h5["positions_wrapped_A"][index], dtype=np.float64)
            cell = np.asarray(h5["cell_A"][index], dtype=np.float64)
            delta = positions[:, None, :] - positions[None, :, :]
            rows, cols = np.triu_indices(n_atoms, 1)
            mic, _ = find_mic(delta[rows, cols], cell, pbc=True)
            minimum = min(minimum, float(np.linalg.norm(mic, axis=1).min()))
        return {
            "path": str(path.resolve()), "n_frames": n_frames, "n_atoms": n_atoms,
            "duration_ps": float(h5["time_fs"][-1] - h5["time_fs"][0]) / 1000.0,
            "mean_temperature_K": float(temp.mean()), "std_temperature_K": float(temp.std(ddof=1)),
            "energy_drift_meV_atom_ps": 1000.0 * slope / n_atoms,
            "energy_peak_to_peak_meV_atom": 1000.0 * float(np.ptp(total)) / n_atoms,
            "sampled_minimum_distance_A": minimum,
            "finite": all(np.isfinite(x) for x in (temp.mean(), total.mean(), minimum)),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/raw_nve/*/trajectory.h5"))
    paths += sorted(args.root.glob("*/physical_reference/short_nve/*/trajectory.h5"))
    results = [trajectory_summary(path) for path in paths]
    transitions = []
    for path in sorted(args.root.glob("*/transitions/*fs/*.h5")):
        with h5py.File(path, "r") as h5:
            transitions.append({
                "path": str(path.resolve()),
                "system": path.parts[-4],
                "split": str(h5.attrs["split"]),
                "horizon_fs": float(h5.attrs["horizon_fs"]),
                "n_samples": int(h5.attrs["n_samples"]),
                "n_source_trajectories": len(h5["source_paths"]),
                "size_bytes": path.stat().st_size,
            })
    output = args.root / "physical_summary.json"
    output.write_text(json.dumps({
        "trajectories": results,
        "transitions": transitions,
        "large_scale_quartz": json.loads(
            (args.root / "quartz/large_scale_3x3x3/metadata.json").read_text()
        ),
    }, indent=2) + "\n")
    print(json.dumps({
        "output": str(output), "n_trajectories": len(results),
        "n_transition_files": len(transitions),
        "n_transition_samples": sum(item["n_samples"] for item in transitions),
    }, indent=2))


if __name__ == "__main__":
    main()
