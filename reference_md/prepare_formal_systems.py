#!/usr/bin/env python3
"""Create independent alpha-quartz and SPC/Fw water starting systems.

No previously generated trajectory or equilibrated state is read by this file.
The quartz crystal is constructed from its crystallographic asymmetric unit;
the liquid-water box is constructed from a regular oxygen grid with freshly
sampled molecular orientations and is subsequently equilibrated by LAMMPS.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from ase.spacegroup import crystal
from ase.io import write
from scipy.spatial.transform import Rotation


QUARTZ_BKS = {
    "wolf_cutoff_A": 10.17,
    "buckingham_cutoff_A": 5.5,
    "q_Si_e": 2.4,
    "q_O_e": -1.2,
    "A_SiO_eV": 18003.7572,
    "B_SiO_inv_A": 4.87318,
    "C_SiO_eV_A6": 133.5381,
    "A_OO_eV": 1388.7730,
    "B_OO_inv_A": 2.76,
    "C_OO_eV_A6": 175.0,
}

SPCFW = {
    "box_length_A": 12.4172,
    "q_O_e": -0.82,
    "q_H_e": 0.41,
    "epsilon_OO_kcal_mol": 0.1554253,
    "sigma_OO_A": 3.165492,
    "bond_K_lammps_kcal_mol_A2": 529.581,
    "bond_r0_A": 1.012,
    "angle_K_lammps_kcal_mol_rad2": 37.95,
    "angle_theta0_deg": 113.24,
    "realspace_cutoff_A": 5.9,
    "pppm_relative_accuracy": 1.0e-5,
}


def write_quartz(root: Path) -> None:
    destination = root / "quartz" / "initial"
    destination.mkdir(parents=True, exist_ok=False)
    # Room-temperature alpha-quartz, right-handed P3_121 setting.  The
    # crystallographic primitive hexagonal cell has 3 Si and 6 O atoms.
    primitive = crystal(
        symbols=["Si", "O"],
        # In ASE's standard P3_121 origin choice the Si 3a site is at z=1/3.
        basis=[
            (0.4697, 0.0, 1.0 / 3.0),
            # Shift the published oxygen z coordinate into ASE's standard
            # origin choice together with the silicon 3a site.
            (0.4133, 0.2667, (0.1188 - 1.0 / 3.0) % 1.0),
        ],
        spacegroup=152,
        cellpar=[4.9134, 4.9134, 5.4052, 90.0, 90.0, 120.0],
        primitive_cell=False,
    )
    atoms = primitive.repeat((3, 2, 3))
    if len(atoms) != 162:
        raise RuntimeError(f"expected 162 quartz atoms, constructed {len(atoms)}")
    charges = np.where(atoms.numbers == 14, QUARTZ_BKS["q_Si_e"], QUARTZ_BKS["q_O_e"])
    atoms.set_initial_charges(charges)
    write(
        destination / "quartz_162.data",
        atoms,
        format="lammps-data",
        atom_style="charge",
        units="real",
        masses=True,
        specorder=["O", "Si"],
    )
    write(destination / "quartz_162.extxyz", atoms)
    metadata = {
        "construction": "alpha-quartz crystallographic asymmetric unit, P3_121 (#152), repeat 3x2x3",
        "lattice_A": {"a": 4.9134, "c": 5.4052},
        "fractional_basis": {
            "Si": [0.4697, 0.0, 1.0 / 3.0],
            "O": [0.4133, 0.2667, (0.1188 - 1.0 / 3.0) % 1.0],
        },
        "n_atoms": len(atoms),
        "composition": "Si54O108",
        "potential": "BKS with shifted Buckingham and alpha=0 damped-shifted-force Coulomb following Carre et al.",
        "parameters": QUARTZ_BKS,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def water_geometry() -> np.ndarray:
    r = SPCFW["bond_r0_A"]
    half_angle = math.radians(SPCFW["angle_theta0_deg"] / 2.0)
    # Centre molecular coordinates on the oxygen; molecule COM drift is
    # removed globally by LAMMPS after velocity initialization.
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [r * math.sin(half_angle), 0.0, r * math.cos(half_angle)],
            [-r * math.sin(half_angle), 0.0, r * math.cos(half_angle)],
        ],
        dtype=np.float64,
    )


def write_water(root: Path, seed: int) -> None:
    destination = root / "water" / "initial"
    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    length = SPCFW["box_length_A"]
    spacing = length / 4.0
    centres = np.asarray(
        [((i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing)
         for i in range(4) for j in range(4) for k in range(4)],
        dtype=np.float64,
    )
    rotations = Rotation.random(64, random_state=rng)
    local = water_geometry()
    positions = np.concatenate([rotation.apply(local) + centre for rotation, centre in zip(rotations, centres)])
    path = destination / "water_64_spcfw.data"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("64 flexible SPC/Fw water molecules generated independently for SMDP\n\n")
        handle.write("192 atoms\n128 bonds\n64 angles\n\n")
        handle.write("2 atom types\n1 bond types\n1 angle types\n\n")
        handle.write(f"0.0 {length:.10f} xlo xhi\n0.0 {length:.10f} ylo yhi\n0.0 {length:.10f} zlo zhi\n\n")
        handle.write("Masses\n\n1 15.9994 # O\n2 1.008 # H\n\n")
        handle.write("Atoms # full\n\n")
        atom_id = 1
        for molecule in range(64):
            for site, atom_type, charge in ((0, 1, SPCFW["q_O_e"]), (1, 2, SPCFW["q_H_e"]), (2, 2, SPCFW["q_H_e"])):
                x, y, z = positions[3 * molecule + site]
                handle.write(f"{atom_id} {molecule + 1} {atom_type} {charge:.8f} {x:.12f} {y:.12f} {z:.12f}\n")
                atom_id += 1
        handle.write("\nBonds\n\n")
        bond_id = 1
        for molecule in range(64):
            oxygen = 3 * molecule + 1
            for hydrogen in (oxygen + 1, oxygen + 2):
                handle.write(f"{bond_id} 1 {oxygen} {hydrogen}\n")
                bond_id += 1
        handle.write("\nAngles\n\n")
        for molecule in range(64):
            oxygen = 3 * molecule + 1
            handle.write(f"{molecule + 1} 1 {oxygen + 1} {oxygen} {oxygen + 2}\n")
    metadata = {
        "construction": "4x4x4 oxygen grid with independent Haar-random molecular orientations",
        "seed": seed,
        "n_molecules": 64,
        "n_atoms": 192,
        "model": "SPC/Fw (Wu, Tepper and Voth, JCP 124, 024503, 2006)",
        "parameters": SPCFW,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, default=2026102001)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    write_quartz(args.root)
    write_water(args.root, args.seed)
    (args.root / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "schema": "smdp.reference_md.v1",
                "independent_generation": True,
                "historical_trajectory_inputs_used": False,
                "base_seed": args.seed,
                "research_objective": "Learn a computationally efficient, physically credible long-stride atomistic phase-space propagator that bypasses repeated small-step force integration while retaining long-time dynamical and statistical observables.",
                "algorithm_hypothesis": "A Type-III finite-time generating action S_total = S_free + S_structured + S_residual can combine analytically structured local motion with a learned non-harmonic and cross-mode residual while producing coupled position-momentum updates.",
                "primary_success_criteria": [
                    "stable autonomous trajectories at strides substantially larger than reference MD",
                    "VDOS, potential-energy distribution, RDF, MSD and diffusion consistent with reference MD where applicable",
                    "short NVE trajectory and energy behavior evaluated separately from long NVT statistics",
                    "measured physical-fidelity versus wall-time advantage, not single-step error alone",
                ],
                "data_policy": "All formal structures, equilibration and production trajectories are generated independently in this directory. No historical or TrajCast trajectory is an input.",
                "systems": ["alpha_quartz_300K", "liquid_water_spcfw_300K"],
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
