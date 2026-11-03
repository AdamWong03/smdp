#!/usr/bin/env python3
"""Convert formal LAMMPS custom dumps into random-access SMDP HDF5."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
from ase import units


EV_TO_KCAL_MOL = 23.060548867


def thermo_blocks(path: Path) -> list[dict[str, np.ndarray]]:
    blocks: list[dict[str, np.ndarray]] = []
    names: list[str] | None = None
    rows: list[list[float]] = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if fields and fields[0] == "Step" and "Time" in fields:
            if names is not None and rows:
                values = np.asarray(rows, dtype=np.float64)
                blocks.append({name: values[:, i] for i, name in enumerate(names)})
            names, rows = fields, []
            continue
        if names is not None and len(fields) == len(names):
            try:
                rows.append([float(value) for value in fields])
                continue
            except ValueError:
                pass
        if names is not None and (line.startswith("Loop time") or line.startswith("ERROR")):
            if rows:
                values = np.asarray(rows, dtype=np.float64)
                blocks.append({name: values[:, i] for i, name in enumerate(names)})
            names, rows = None, []
    if names is not None and rows:
        values = np.asarray(rows, dtype=np.float64)
        blocks.append({name: values[:, i] for i, name in enumerate(names)})
    return blocks


def dump_header(handle):
    line = handle.readline()
    if not line:
        return None
    if line.strip() != "ITEM: TIMESTEP":
        raise ValueError(f"unexpected dump marker {line!r}")
    step = int(handle.readline())
    if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
        raise ValueError("missing atom count")
    n_atoms = int(handle.readline())
    bounds_header = handle.readline().split()
    if bounds_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
        raise ValueError("missing box bounds")
    raw_bounds = [list(map(float, handle.readline().split())) for _ in range(3)]
    if len(raw_bounds[0]) == 3:
        bounds = np.asarray(raw_bounds)
        xy, xz, yz = bounds[0, 2], bounds[1, 2], bounds[2, 2]
        xlo = bounds[0, 0] - min(0.0, xy, xz, xy + xz)
        xhi = bounds[0, 1] - max(0.0, xy, xz, xy + xz)
        ylo = bounds[1, 0] - min(0.0, yz)
        yhi = bounds[1, 1] - max(0.0, yz)
        zlo, zhi = bounds[2, :2]
        cell = np.asarray([[xhi-xlo, 0, 0], [xy, yhi-ylo, 0], [xz, yz, zhi-zlo]])
        origin = np.asarray([xlo, ylo, zlo])
    else:
        bounds = np.asarray(raw_bounds)
        origin = bounds[:, 0]
        cell = np.diag(bounds[:, 1] - bounds[:, 0])
    atom_header = handle.readline().split()
    if atom_header[:2] != ["ITEM:", "ATOMS"]:
        raise ValueError("missing atom fields")
    return step, n_atoms, cell, origin, atom_header[2:]


def mappings(system: str):
    if system == "quartz":
        numbers = {1: 8, 2: 14}
        masses = {1: 15.999, 2: 28.0855}
    elif system == "water":
        numbers = {1: 8, 2: 1}
        masses = {1: 15.9994, 2: 1.008}
    else:
        raise ValueError(system)
    return numbers, masses


def choose_thermo(blocks, last_step: int):
    candidates = [block for block in blocks if int(block["Step"][-1]) == last_step]
    if not candidates:
        raise ValueError(f"no thermo block ending at dump step {last_step}")
    return candidates[-1]


def convert(
    run_dir: Path,
    dump_name: str,
    output: Path,
    system: str,
    dt_fs: float,
    buffer: int = 256,
    store_derived_momenta: bool = True,
    store_redundant_coordinates: bool = True,
):
    dump_path = run_dir / dump_name
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    numbers_map, masses_map = mappings(system)
    opener = gzip.open if dump_path.suffix == ".gz" else open
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    first_header = None
    last_step = None
    with opener(dump_path, "rt") as source:
        first_header = dump_header(source)
        if first_header is None:
            raise ValueError("empty dump")
        source.seek(0)
        with h5py.File(output, "w") as target:
            target.attrs.update({
                "schema": "smdp.reference_md.raw.v1", "system": system,
                "integration_dt_fs": dt_fs, "source_dump": str(dump_path.resolve()),
                "ensemble": metadata.get("ensemble", "unknown"),
                "store_derived_momenta": store_derived_momenta,
                "store_redundant_coordinates": store_redundant_coordinates,
            })
            datasets = None
            frame = 0
            buffers: dict[str, list] = {}
            static_types = static_ids = static_molecules = None
            while True:
                header = dump_header(source)
                if header is None:
                    break
                step, n_atoms, cell, origin, fields = header
                index = {name: i for i, name in enumerate(fields)}
                required = ["id", "type", "xu", "yu", "zu", "x", "y", "z", "ix", "iy", "iz", "vx", "vy", "vz"]
                missing = [name for name in required if name not in index]
                if missing:
                    raise ValueError(f"dump missing {missing}")
                raw = np.fromstring("".join(source.readline() for _ in range(n_atoms)), sep=" ").reshape(n_atoms, len(fields))
                raw = raw[np.argsort(raw[:, index["id"]].astype(np.int64))]
                ids = raw[:, index["id"]].astype(np.int64)
                types = raw[:, index["type"]].astype(np.int64)
                molecules = raw[:, index["mol"]].astype(np.int64) if "mol" in index else np.zeros(n_atoms, dtype=np.int64)
                if datasets is None:
                    static_ids, static_types, static_molecules = ids, types, molecules
                    atomic_numbers = np.asarray([numbers_map[int(t)] for t in types], dtype=np.int16)
                    masses = np.asarray([masses_map[int(t)] for t in types], dtype=np.float64)
                    target.create_dataset("atom_ids", data=ids)
                    target.create_dataset("atom_types", data=types)
                    target.create_dataset("atomic_numbers", data=atomic_numbers)
                    target.create_dataset("masses_u", data=masses)
                    target.create_dataset("molecule_ids", data=molecules)
                    shape = (0, n_atoms, 3); maximum = (None, n_atoms, 3)
                    chunk = (min(buffer, 256), n_atoms, 3)
                    datasets = {
                        "step": target.create_dataset("step", (0,), maxshape=(None,), chunks=(4096,), dtype="i8"),
                        "time_fs": target.create_dataset("time_fs", (0,), maxshape=(None,), chunks=(4096,), dtype="f8"),
                        "cell_A": target.create_dataset("cell_A", (0,3,3), maxshape=(None,3,3), chunks=(1024,3,3), dtype="f8", compression="lzf"),
                        "origin_A": target.create_dataset("origin_A", (0,3), maxshape=(None,3), chunks=(1024,3), dtype="f8", compression="lzf"),
                    }
                    dynamic_names = ["positions_unwrapped_A", "velocities_A_per_fs"]
                    if store_redundant_coordinates:
                        dynamic_names += ["positions_wrapped_A", "image_flags"]
                    if store_derived_momenta:
                        dynamic_names += ["momenta_u_A_per_fs", "momenta_sqrt_eV_u"]
                    for name in dynamic_names:
                        dtype = "i4" if name == "image_flags" else "f4"
                        datasets[name] = target.create_dataset(name, shape, maxshape=maximum, chunks=chunk, dtype=dtype, compression="lzf")
                    if all(name in index for name in ("fx","fy","fz")):
                        datasets["forces_eV_per_A"] = target.create_dataset("forces_eV_per_A", shape, maxshape=maximum, chunks=chunk, dtype="f4", compression="lzf")
                    buffers = {name: [] for name in datasets}
                elif not (np.array_equal(ids, static_ids) and np.array_equal(types, static_types) and np.array_equal(molecules, static_molecules)):
                    raise ValueError("atom identity/order changed")
                velocities = raw[:, [index["vx"], index["vy"], index["vz"]]]
                masses = np.asarray([masses_map[int(t)] for t in types])[:, None]
                record = {
                    "step": step, "time_fs": step * dt_fs, "cell_A": cell, "origin_A": origin,
                    "positions_unwrapped_A": raw[:, [index["xu"], index["yu"], index["zu"]]],
                    "velocities_A_per_fs": velocities,
                }
                if store_redundant_coordinates:
                    record["positions_wrapped_A"] = raw[:, [index["x"], index["y"], index["z"]]]
                    record["image_flags"] = raw[:, [index["ix"], index["iy"], index["iz"]]].astype(np.int32)
                if store_derived_momenta:
                    record["momenta_u_A_per_fs"] = masses * velocities
                    record["momenta_sqrt_eV_u"] = masses * velocities / units.fs
                if "forces_eV_per_A" in datasets:
                    record["forces_eV_per_A"] = raw[:, [index["fx"], index["fy"], index["fz"]]] / EV_TO_KCAL_MOL
                for name, value in record.items():
                    buffers[name].append(value)
                frame += 1; last_step = step
                if len(buffers["step"]) >= buffer:
                    start = datasets["step"].shape[0]; stop = start + len(buffers["step"])
                    for name, dataset in datasets.items():
                        dataset.resize(stop, axis=0); dataset[start:stop] = np.asarray(buffers[name]); buffers[name].clear()
                    target.flush()
            if buffers and buffers["step"]:
                start = datasets["step"].shape[0]; stop = start + len(buffers["step"])
                for name, dataset in datasets.items():
                    dataset.resize(stop, axis=0); dataset[start:stop] = np.asarray(buffers[name])
            blocks = thermo_blocks(run_dir / "run.log")
            thermo = choose_thermo(blocks, int(last_step))
            group = target.create_group("thermo")
            key_map = {"Step":"step", "Time":"time_fs", "Temp":"temperature_K", "c_tcom":"temperature_com_K", "PotEng":"potential_energy_kcal_mol", "KinEng":"kinetic_energy_kcal_mol", "TotEng":"total_energy_kcal_mol", "Press":"pressure_atm", "Density":"density_g_cm3", "Volume":"volume_A3"}
            for source_name, values in thermo.items():
                name = key_map.get(source_name, source_name)
                if source_name in ("PotEng","KinEng","TotEng"):
                    name = name.replace("kcal_mol", "eV"); values = values / EV_TO_KCAL_MOL
                group.create_dataset(name, data=values)
            target.attrs["n_frames"] = frame
            target.attrs["output_interval_fs"] = dt_fs * (target["step"][1] - target["step"][0] if frame > 1 else 1)
    return {"output": str(output), "frames": frame, "last_step": int(last_step)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("dump_name")
    parser.add_argument("output", type=Path)
    parser.add_argument("--system", choices=["quartz","water"], required=True)
    parser.add_argument("--dt-fs", type=float, required=True)
    parser.add_argument("--buffer", type=int, default=256)
    parser.add_argument(
        "--omit-derived-momenta", action="store_true",
        help="Do not duplicate momenta that can be reconstructed from stored velocities and masses",
    )
    parser.add_argument(
        "--omit-redundant-coordinates", action="store_true",
        help="Keep unwrapped positions only; wrapped positions/images remain available in the source dump",
    )
    args = parser.parse_args()
    print(json.dumps(convert(
        args.run_dir, args.dump_name, args.output, args.system, args.dt_fs,
        args.buffer,
        store_derived_momenta=not args.omit_derived_momenta,
        store_redundant_coordinates=not args.omit_redundant_coordinates,
    ), indent=2))


if __name__ == "__main__":
    main()
