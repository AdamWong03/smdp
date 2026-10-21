"""TrajCast quartz BKS/Wolf reference implementation through LAMMPS.

Carré et al. use a shifted-force Wolf Coulomb term with a 10.17 Angstrom
cutoff and a shifted 5.5 Angstrom BKS Buckingham term. LAMMPS ``coul/dsf``
at alpha=0 has the same forces but includes a composition-dependent self
energy. ``carre_energy`` removes that constant so the returned value follows
the pair-potential convention of Eq. (4) in Carré et al. (JCP 127, 114512).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.lammpslib import LAMMPSlib


EV_TO_KCAL_MOL = 23.060548867
COULOMB_EV_ANGSTROM = 14.3996454784255


@dataclass(frozen=True)
class BKSParameters:
    wolf_cutoff_a: float = 10.17
    short_cutoff_a: float = 5.5
    q_si: float = 2.4
    q_o: float = -1.2
    a_sio_ev: float = 18003.7572
    b_sio_inv_a: float = 4.87318
    c_sio_ev_a6: float = 133.5381
    a_oo_ev: float = 1388.7730
    b_oo_inv_a: float = 2.76
    c_oo_ev_a6: float = 175.0


def set_bks_charges(atoms: Atoms, parameters: BKSParameters = BKSParameters()) -> None:
    charges = np.where(atoms.numbers == 14, parameters.q_si, parameters.q_o)
    if not np.all(np.isin(atoms.numbers, [8, 14])):
        raise ValueError("The quartz BKS reference only accepts O and Si")
    atoms.set_initial_charges(charges)


def lammps_commands(parameters: BKSParameters = BKSParameters()) -> list[str]:
    p = parameters
    return [
        (
            "pair_style hybrid/overlay "
            f"buck {p.short_cutoff_a} coul/dsf 0.0 {p.wolf_cutoff_a}"
        ),
        (
            "pair_coeff 1 1 buck "
            f"{p.a_oo_ev * EV_TO_KCAL_MOL:.14g} "
            f"{1.0 / p.b_oo_inv_a:.14g} "
            f"{p.c_oo_ev_a6 * EV_TO_KCAL_MOL:.14g}"
        ),
        (
            "pair_coeff 1 2 buck "
            f"{p.a_sio_ev * EV_TO_KCAL_MOL:.14g} "
            f"{1.0 / p.b_sio_inv_a:.14g} "
            f"{p.c_sio_ev_a6 * EV_TO_KCAL_MOL:.14g}"
        ),
        "pair_coeff 2 2 buck 0.0 1.0 0.0",
        "pair_coeff * * coul/dsf",
        "pair_modify pair buck shift yes",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
    ]


def make_calculator(
    parameters: BKSParameters = BKSParameters(), *, keep_alive: bool = True
) -> LAMMPSlib:
    return LAMMPSlib(
        lmpcmds=lammps_commands(parameters),
        atom_types={"O": 1, "Si": 2},
        lammps_header=[
            "units real",
            "atom_style charge",
            "atom_modify map array",
            "boundary p p p",
        ],
        keep_alive=keep_alive,
        log_file=None,
    )


def dsf_self_energy_ev(
    atoms: Atoms, parameters: BKSParameters = BKSParameters()
) -> float:
    charges = np.where(atoms.numbers == 14, parameters.q_si, parameters.q_o)
    return -COULOMB_EV_ANGSTROM * float(np.square(charges).sum()) / parameters.wolf_cutoff_a


def evaluate(
    atoms: Atoms,
    calculator: LAMMPSlib,
    parameters: BKSParameters = BKSParameters(),
) -> tuple[float, np.ndarray, float]:
    """Return Carré-convention energy, forces and raw LAMMPS DSF energy."""
    set_bks_charges(atoms, parameters)
    atoms.calc = calculator
    atoms.get_potential_energy()
    # Match the formal dataset converter, not ASE's slightly different CODATA
    # kcal/mol -> eV constant. No integration is performed in this evaluation.
    raw_energy = float(calculator.lmp.get_thermo('pe')) / EV_TO_KCAL_MOL
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    carre_energy = raw_energy - dsf_self_energy_ev(atoms, parameters)
    return carre_energy, forces, raw_energy


def energy(
    atoms: Atoms,
    calculator: LAMMPSlib,
    parameters: BKSParameters = BKSParameters(),
) -> tuple[float, float]:
    """Return Carré-convention and raw LAMMPS DSF energies without forces."""
    set_bks_charges(atoms, parameters)
    atoms.calc = calculator
    atoms.get_potential_energy()
    raw_energy = float(calculator.lmp.get_thermo('pe')) / EV_TO_KCAL_MOL
    return raw_energy - dsf_self_energy_ev(atoms, parameters), raw_energy
