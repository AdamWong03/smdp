#!/usr/bin/env python3
"""Sample trajectory-disjoint finite-time phase-space transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from ase import units


COUNTS = {"train": 5000, "val": 1250, "test": 1000}


def discover(root: Path, split: str) -> list[Path]:
    paths = sorted((root / "raw_nve").glob(f"{split}_*/trajectory.h5"))
    if not paths:
        raise FileNotFoundError(f"no {split} trajectories under {root / 'raw_nve'}")
    return paths


def allocate(count: int, capacities: list[int]) -> list[int]:
    allocation = [count // len(capacities)] * len(capacities)
    for i in range(count % len(capacities)):
        allocation[i] += 1
    if any(n > cap for n, cap in zip(allocation, capacities)):
        raise ValueError("requested transition count exceeds trajectory capacity")
    return allocation


def build(system_root: Path, output: Path, split: str, horizon_fs: float, count: int, seed: int):
    sources = discover(system_root, split)
    with h5py.File(sources[0], "r") as first:
        dt = float(first.attrs["integration_dt_fs"])
        lag = int(round(horizon_fs / dt))
        if not np.isclose(lag * dt, horizon_fs):
            raise ValueError("horizon is not an integer number of MD steps")
        numbers = np.asarray(first["atomic_numbers"])
        masses = np.asarray(first["masses_u"])
        types = np.asarray(first["atom_types"])
        molecules = np.asarray(first["molecule_ids"])
        n_atoms = len(numbers)
    capacities = []
    for source in sources:
        with h5py.File(source, "r") as handle:
            capacities.append(len(handle["step"]) - lag)
    allocations = allocate(count, capacities)
    rng = np.random.default_rng(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(output, "w") as target:
        target.attrs.update({
            "schema": "smdp.reference_md.transitions.v1", "split": split,
            "horizon_fs": horizon_fs, "source_dt_fs": dt, "lag_steps": lag,
            "seed": seed, "n_samples": count,
        })
        target.create_dataset("atomic_numbers", data=numbers)
        target.create_dataset("atom_types", data=types)
        target.create_dataset("masses_u", data=masses)
        target.create_dataset("molecule_ids", data=molecules)
        string_type = h5py.string_dtype("utf-8")
        target.create_dataset("source_paths", data=np.asarray([str(p.resolve()) for p in sources], dtype=object), dtype=string_type)
        shape = (count, n_atoms, 3); chunk = (min(64, count), n_atoms, 3)
        names = [
            "positions_initial_A", "positions_final_A", "positions_midpoint_A", "delta_positions_A",
            "velocities_initial_A_per_fs", "velocities_final_A_per_fs", "velocities_midpoint_A_per_fs",
            "momenta_initial_u_A_per_fs", "momenta_final_u_A_per_fs", "momenta_midpoint_u_A_per_fs", "delta_momenta_u_A_per_fs",
            "momenta_initial_sqrt_eV_u", "momenta_final_sqrt_eV_u", "momenta_midpoint_sqrt_eV_u", "delta_momenta_sqrt_eV_u",
        ]
        datasets = {name: target.create_dataset(name, shape, chunks=chunk, dtype="f4", compression="lzf") for name in names}
        cells = target.create_dataset("cell_A", (count,3,3), chunks=(min(256,count),3,3), dtype="f8", compression="lzf")
        source_trajectory = target.create_dataset("source_trajectory", (count,), dtype="i2")
        source_frame = target.create_dataset("source_frame", (count,), dtype="i8")
        source_step = target.create_dataset("source_step", (count,), dtype="i8")
        source_time = target.create_dataset("source_time_fs", (count,), dtype="f8")
        scalar_names = [
            "potential_energy_initial_eV", "potential_energy_final_eV",
            "kinetic_energy_initial_eV", "kinetic_energy_final_eV",
            "temperature_initial_K", "temperature_final_K",
        ]
        scalars = {name: target.create_dataset(name, (count,), dtype="f8") for name in scalar_names}
        cursor = 0
        for trajectory_index, (source, allocation) in enumerate(zip(sources, allocations)):
            with h5py.File(source, "r") as handle:
                indices = np.sort(rng.choice(len(handle["step"]) - lag, size=allocation, replace=False))
                take0, take1 = indices, indices + lag
                # The raw trajectory intentionally stores all equivalent phase
                # representations.  For sampled transitions, however, reading
                # each momentum representation separately would decompress the
                # same random HDF5 chunks repeatedly.  Read q/v once and derive
                # both momenta exactly using the recorded masses and ASE's time
                # conversion, matching convert_lammps_h5.py.
                all_q = np.asarray(handle["positions_unwrapped_A"])
                all_v = np.asarray(handle["velocities_A_per_fs"])
                q0, q1 = all_q[take0], all_q[take1]
                v0, v1 = all_v[take0], all_v[take1]
                mass_column = masses[None, :, None]
                p0, p1 = mass_column * v0, mass_column * v1
                c0, c1 = p0 / units.fs, p1 / units.fs
                stop = cursor + allocation
                values = {
                    "positions_initial_A":q0, "positions_final_A":q1, "positions_midpoint_A":0.5*(q0+q1), "delta_positions_A":q1-q0,
                    "velocities_initial_A_per_fs":v0, "velocities_final_A_per_fs":v1, "velocities_midpoint_A_per_fs":0.5*(v0+v1),
                    "momenta_initial_u_A_per_fs":p0, "momenta_final_u_A_per_fs":p1, "momenta_midpoint_u_A_per_fs":0.5*(p0+p1), "delta_momenta_u_A_per_fs":p1-p0,
                    "momenta_initial_sqrt_eV_u":c0, "momenta_final_sqrt_eV_u":c1, "momenta_midpoint_sqrt_eV_u":0.5*(c0+c1), "delta_momenta_sqrt_eV_u":c1-c0,
                }
                for name, value in values.items(): datasets[name][cursor:stop] = value
                cells[cursor:stop] = np.asarray(handle["cell_A"][take0])
                source_trajectory[cursor:stop] = trajectory_index
                source_frame[cursor:stop] = indices
                source_step[cursor:stop] = np.asarray(handle["step"][take0])
                source_time[cursor:stop] = np.asarray(handle["time_fs"][take0])
                thermo_steps = np.asarray(handle["thermo/step"])
                if len(thermo_steps) != len(handle["step"]) or not np.array_equal(thermo_steps, np.asarray(handle["step"])):
                    raise ValueError(f"{source}: NVE thermo output is not aligned to every stored frame")
                pe = np.asarray(handle["thermo/potential_energy_eV"])
                ke = np.asarray(handle["thermo/kinetic_energy_eV"])
                temperature = np.asarray(handle["thermo/temperature_K"])
                scalars["potential_energy_initial_eV"][cursor:stop] = pe[take0]
                scalars["potential_energy_final_eV"][cursor:stop] = pe[take1]
                scalars["kinetic_energy_initial_eV"][cursor:stop] = ke[take0]
                scalars["kinetic_energy_final_eV"][cursor:stop] = ke[take1]
                scalars["temperature_initial_K"][cursor:stop] = temperature[take0]
                scalars["temperature_final_K"][cursor:stop] = temperature[take1]
                cursor = stop
    return {"output": str(output), "samples": cursor, "split": split, "horizon_fs": horizon_fs, "sources": [str(p) for p in sources]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system_root", type=Path)
    parser.add_argument("--horizons-fs", type=float, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=2026102001)
    args = parser.parse_args()
    results = []
    for horizon in args.horizons_fs:
        label = f"{horizon:g}fs"
        for offset, split in enumerate(("train","val","test")):
            results.append(build(args.system_root, args.system_root / "transitions" / label / f"{split}.h5", split, horizon, COUNTS[split], args.seed + int(round(horizon*10)) + offset))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
