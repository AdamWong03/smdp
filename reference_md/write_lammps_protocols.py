#!/usr/bin/env python3
"""Write the complete LAMMPS protocols for the formal SMDP reference data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_md.prepare_formal_systems import QUARTZ_BKS, SPCFW


EV_TO_KCAL_MOL = 23.060548867


def bks_commands() -> list[str]:
    p = QUARTZ_BKS
    return [
        f"pair_style hybrid/overlay buck {p['buckingham_cutoff_A']} coul/dsf 0.0 {p['wolf_cutoff_A']}",
        f"pair_coeff 1 1 buck {p['A_OO_eV'] * EV_TO_KCAL_MOL:.14g} {1.0 / p['B_OO_inv_A']:.14g} {p['C_OO_eV_A6'] * EV_TO_KCAL_MOL:.14g}",
        f"pair_coeff 1 2 buck {p['A_SiO_eV'] * EV_TO_KCAL_MOL:.14g} {1.0 / p['B_SiO_inv_A']:.14g} {p['C_SiO_eV_A6'] * EV_TO_KCAL_MOL:.14g}",
        "pair_coeff 2 2 buck 0.0 1.0 0.0",
        "pair_coeff * * coul/dsf",
        "pair_modify pair buck shift yes",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
    ]


def water_commands() -> list[str]:
    p = SPCFW
    return [
        "special_bonds lj/coul 0.0 0.0 0.0",
        f"pair_style lj/cut/coul/long {p['realspace_cutoff_A']}",
        "pair_coeff * * 0.0 1.0",
        f"pair_coeff 1 1 {p['epsilon_OO_kcal_mol']} {p['sigma_OO_A']}",
        "bond_style harmonic",
        f"bond_coeff 1 {p['bond_K_lammps_kcal_mol_A2']} {p['bond_r0_A']}",
        "angle_style harmonic",
        f"angle_coeff 1 {p['angle_K_lammps_kcal_mol_rad2']} {p['angle_theta0_deg']}",
        f"kspace_style pppm {p['pppm_relative_accuracy']}",
        "neighbor 1.0 bin",
        "neigh_modify every 1 delay 0 check yes",
    ]


def common_header(atom_style: str, data: str, timestep_fs: float) -> list[str]:
    return [
        "units real",
        f"atom_style {atom_style}",
        "atom_modify map array",
        "boundary p p p",
        f"read_data {data}",
        f"timestep {timestep_fs}",
        "compute tcom all temp/com",
    ]


def quartz_equilibration(root: Path, seed: int) -> None:
    directory = root / "quartz" / "equilibration"
    (directory / "snapshots").mkdir(parents=True, exist_ok=False)
    data = "../initial/quartz_162.data"
    lines = common_header("charge", data, 1.0) + bks_commands()
    lines += [
        "thermo 100",
        "thermo_style custom step time temp c_tcom pe ke etotal press vol lx ly lz xy xz yz",
        "thermo_modify temp tcom flush yes",
        "min_style fire",
        "minimize 1.0e-8 1.0e-8 5000 50000",
        f"velocity all create 300.0 {seed} mom yes rot no dist gaussian",
        "velocity all zero linear",
        # LAMMPS press/berendsen rejects triclinic cells.  Use the native
        # reversible Nose-Hoover cell dynamics for this 10 ps density
        # relaxation while retaining the published 0.1 ps pressure time.
        "fix thermostat all npt temp 300.0 300.0 100.0 iso 1.0 1.0 100.0 tchain 3 pchain 3",
        "run 10000",
        "unfix thermostat",
        "fix thermostat all nvt temp 300.0 300.0 100.0 tchain 3",
        "run 40000",
        "variable sample loop 10",
        "label sample_loop",
        "run 1000",
        "write_data snapshots/quartz_${sample}.data nocoeff",
        "next sample",
        "jump SELF sample_loop",
        "write_data equilibrated.data nocoeff",
        "write_restart equilibrated.restart",
    ]
    (directory / "in.lammps").write_text("\n".join(lines) + "\n")


def quartz_production(root: Path, seed: int) -> None:
    base = root / "quartz"
    csvr_seed = (seed + 100) % 900_000_000 or 1
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    for index, split in enumerate(splits, start=1):
        directory = base / "raw_nve" / f"{split}_{index:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        relative = f"../../equilibration/snapshots/quartz_{index}.data"
        lines = common_header("charge", relative, 1.0) + bks_commands()
        lines += [
            "reset_timestep 0",
            "fix integrate all nve",
            "thermo 1",
            "thermo_style custom step time temp c_tcom pe ke etotal press",
            "thermo_modify temp tcom flush yes",
            "dump raw all custom/gz 1 trajectory.lammpstrj.gz id type xu yu zu x y z ix iy iz vx vy vz fx fy fz",
            "dump_modify raw sort id",
            "run 100000",
            "write_restart final.restart",
        ]
        (directory / "in.lammps").write_text("\n".join(lines) + "\n")
        (directory / "metadata.json").write_text(json.dumps({
            "system": "alpha_quartz", "split": split, "replica": index,
            "seed": seed + index, "ensemble": "NVE", "dt_fs": 1.0,
            "steps": 100000, "save_every_steps": 1,
            "initial_state": relative, "potential": "BKS/Carre truncated shifted",
        }, indent=2) + "\n")

    directory = base / "physical_reference" / "nvt_15ns"
    directory.mkdir(parents=True, exist_ok=False)
    lines = common_header("charge", "../../equilibration/equilibrated.data", 1.0) + bks_commands()
    lines += [
        "reset_timestep 0",
        "fix integrate all nve",
        f"fix thermostat all temp/csvr 300.0 300.0 100.0 {csvr_seed}",
        "thermo 10",
        "thermo_style custom step time temp c_tcom pe ke etotal press",
        "thermo_modify temp tcom flush yes",
        "dump thermal all custom/gz 10 thermalization.lammpstrj.gz id type xu yu zu x y z ix iy iz vx vy vz",
        "dump_modify thermal sort id",
        "run 1500000",
        "undump thermal",
        "reset_timestep 0",
        "dump production all custom/gz 1 production.lammpstrj.gz id type xu yu zu x y z ix iy iz vx vy vz",
        "dump_modify production sort id",
        "run 15000000",
        "write_restart final.restart",
    ]
    (directory / "in.lammps").write_text("\n".join(lines) + "\n")
    (directory / "metadata.json").write_text(json.dumps({
        "system": "alpha_quartz", "ensemble": "NVT-CSVR", "temperature_K": 300.0,
        "dt_fs": 1.0, "thermalization_steps": 1500000, "production_steps": 15000000,
        "thermalization_save_every_steps": 10, "production_save_every_steps": 1,
        "thermostat_tau_fs": 100.0, "seed": csvr_seed,
    }, indent=2) + "\n")

    nve = base / "physical_reference" / "short_nve"
    nve.mkdir(parents=True, exist_ok=False)
    for index in (8, 9, 10):
        directory = nve / f"replica_{index:02d}"
        directory.mkdir()
        relative = f"../../../equilibration/snapshots/quartz_{index}.data"
        lines = common_header("charge", relative, 1.0) + bks_commands()
        lines += [
            "reset_timestep 0", "fix integrate all nve", "thermo 1",
            "thermo_style custom step time temp c_tcom pe ke etotal press",
            "thermo_modify temp tcom flush yes",
            "dump raw all custom/gz 1 trajectory.lammpstrj.gz id type xu yu zu x y z ix iy iz vx vy vz fx fy fz",
            "dump_modify raw sort id", "run 10000", "write_restart final.restart",
        ]
        (directory / "in.lammps").write_text("\n".join(lines) + "\n")


def water_equilibration(root: Path, seed: int) -> None:
    directory = root / "water" / "equilibration"
    (directory / "snapshots").mkdir(parents=True, exist_ok=False)
    lines = common_header("full", "../initial/water_64_spcfw.data", 0.5) + water_commands()
    lines += [
        "thermo 100",
        "thermo_style custom step time temp c_tcom pe ke etotal press density",
        "thermo_modify temp tcom flush yes",
        "min_style fire",
        "minimize 1.0e-8 1.0e-8 5000 50000",
        f"velocity all create 300.0 {seed} mom yes rot no dist gaussian",
        "velocity all zero linear",
        "fix thermostat all nvt temp 300.0 300.0 500.0 tchain 3",
        "run 15000",
        "variable sample loop 5",
        "label sample_loop",
        "run 1000",
        "write_data snapshots/water_${sample}.data nocoeff",
        "next sample",
        "jump SELF sample_loop",
        "write_data equilibrated.data nocoeff",
        "write_restart equilibrated.restart",
    ]
    (directory / "in.lammps").write_text("\n".join(lines) + "\n")


def water_production(root: Path, seed: int) -> None:
    base = root / "water"
    csvr_seed = (seed + 100) % 900_000_000 or 1
    splits = ["train", "train", "train", "val", "test"]
    for index, split in enumerate(splits, start=1):
        directory = base / "raw_nve" / f"{split}_{index:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        relative = f"../../equilibration/snapshots/water_{index}.data"
        lines = common_header("full", relative, 0.5) + water_commands()
        lines += [
            "reset_timestep 0", "fix integrate all nve", "thermo 1",
            "thermo_style custom step time temp c_tcom pe ke etotal press density",
            "thermo_modify temp tcom flush yes",
            "dump raw all custom/gz 1 trajectory.lammpstrj.gz id type mol xu yu zu x y z ix iy iz vx vy vz fx fy fz",
            "dump_modify raw sort id", "run 100000", "write_restart final.restart",
        ]
        (directory / "in.lammps").write_text("\n".join(lines) + "\n")
        (directory / "metadata.json").write_text(json.dumps({
            "system": "liquid_water_spcfw", "split": split, "replica": index,
            "seed": seed + index, "ensemble": "NVE", "dt_fs": 0.5,
            "steps": 100000, "save_every_steps": 1, "initial_state": relative,
        }, indent=2) + "\n")

    directory = base / "physical_reference" / "nvt_2p5ns"
    directory.mkdir(parents=True, exist_ok=False)
    lines = common_header("full", "../../equilibration/equilibrated.data", 0.5) + water_commands()
    lines += [
        "reset_timestep 0", "fix integrate all nve",
        f"fix thermostat all temp/csvr 300.0 300.0 50.0 {csvr_seed}",
        "thermo 10", "thermo_style custom step time temp c_tcom pe ke etotal press density",
        "thermo_modify temp tcom flush yes",
        "dump thermal all custom/gz 10 thermalization.lammpstrj.gz id type mol xu yu zu x y z ix iy iz vx vy vz",
        "dump_modify thermal sort id", "run 500000", "undump thermal", "reset_timestep 0",
        "dump production all custom/gz 1 production.lammpstrj.gz id type mol xu yu zu x y z ix iy iz vx vy vz",
        "dump_modify production sort id", "run 5000000", "write_restart final.restart",
    ]
    (directory / "in.lammps").write_text("\n".join(lines) + "\n")
    (directory / "metadata.json").write_text(json.dumps({
        "system": "liquid_water_spcfw", "ensemble": "NVT-CSVR", "temperature_K": 300.0,
        "dt_fs": 0.5, "thermalization_steps": 500000, "production_steps": 5000000,
        "thermalization_save_every_steps": 10, "production_save_every_steps": 1,
        "thermostat_tau_fs": 50.0, "seed": csvr_seed,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, default=2026102001)
    args = parser.parse_args()
    quartz_equilibration(args.root, args.seed + 10)
    water_equilibration(args.root, args.seed + 20)
    quartz_production(args.root, args.seed + 1000)
    water_production(args.root, args.seed + 2000)


if __name__ == "__main__":
    main()
