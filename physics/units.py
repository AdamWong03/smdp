"""Physical-unit conversions used by SMDP.

Positions are Angstrom, time is femtosecond, masses are atomic mass units,
velocities are Angstrom/fs, forces are eV/Angstrom and energies are eV.
"""

from __future__ import annotations

import scipy.constants


# q_ddot [A/fs^2] = FORCE_TO_ACCEL * F [eV/A] / m [u]
FORCE_TO_ACCEL = (
    scipy.constants.electron_volt
    / 1.0e-10
    / scipy.constants.atomic_mass
    * 1.0e-20
)

# K [eV] = 0.5 * m [u] * |v [A/fs]|^2 * KINETIC_TO_EV
KINETIC_TO_EV = 1.0 / FORCE_TO_ACCEL

KB_EV_PER_K = scipy.constants.Boltzmann / scipy.constants.electron_volt
