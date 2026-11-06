"""Offline SPC/Fw evaluation using the exact formal reference LAMMPS setup.

No MD integration is performed here. Bond/molecule identities are read from
the retained reference data file; energies remain in the raw dataset convention.
"""
import ctypes
from pathlib import Path

import numpy as np

from reference_md.write_lammps_protocols import EV_TO_KCAL_MOL, water_commands


class WaterReference:
    components = ('pe', 'ebond', 'eangle', 'evdwl', 'ecoul', 'elong')

    def __init__(self, topology, cell, atomic_numbers):
        from lammps import lammps
        self.lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
        try:
            self.lmp.commands_list([
                'units real', 'atom_style full', 'atom_modify map array',
                'boundary p p p', f'read_data "{Path(topology).resolve()}"',
                *water_commands(),
                'thermo_style custom step pe ebond eangle evdwl ecoul elong',
                'run 0',
            ])
            self.n = int(self.lmp.get_natoms())
            self.cell = np.asarray(cell, dtype=float)
            if not np.allclose(self.cell, np.diag(np.diag(self.cell)), atol=1e-10):
                raise ValueError('Formal water reference is orthorhombic')
            lo, hi, xy, yz, xz, _, _ = self.lmp.extract_box()
            self.origin = np.asarray(lo, dtype=float)
            np.testing.assert_allclose(np.asarray(hi)-lo, np.diag(self.cell), atol=1e-7)
            types = np.ctypeslib.as_array(self.lmp.gather_atoms('type', 0, 1), shape=(self.n,))
            np.testing.assert_array_equal(np.where(types == 1, 8, 1), atomic_numbers)
        except Exception:
            self.lmp.close()
            raise

    def evaluate(self, positions, forces=False):
        q = np.ascontiguousarray(positions, dtype=np.float64)
        if q.shape != (self.n, 3) or not np.isfinite(q).all():
            raise ValueError('Invalid water coordinates')
        # Long diffusive trajectories can span many boxes. Supply wrapped x
        # AND their consistent image flags, rather than expecting run 0 to
        # repair arbitrary unwrapped coordinates after a large frame jump.
        fractional = (q-self.origin) @ np.linalg.inv(self.cell)
        image = np.floor(fractional).astype(np.int64)
        wrapped = np.ascontiguousarray(self.origin+(fractional-image) @ self.cell)
        self.lmp.scatter_atoms('x', 1, 3, wrapped.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        nlocal = int(self.lmp.extract_setting('nlocal'))
        if nlocal != self.n:
            raise ValueError('Offline WaterReference requires a single MPI rank')
        ids = self.lmp.numpy.extract_atom('id')[:nlocal]
        encoded = np.array([self.lmp.encode_image_flags(*map(int, flags)) for flags in image],
                           dtype=self.lmp.c_imageint)
        self.lmp.numpy.extract_atom('image')[:nlocal] = encoded[ids-1]
        self.lmp.command('run 0 pre yes post no')
        result = {name: float(self.lmp.get_thermo(name))/EV_TO_KCAL_MOL
                  for name in self.components}
        result['nonbonded'] = result['evdwl']+result['ecoul']+result['elong']
        if forces:
            result['forces_eV_A'] = np.ctypeslib.as_array(
                self.lmp.gather_atoms('f', 1, 3), shape=(self.n*3,)).copy().reshape(self.n, 3)/EV_TO_KCAL_MOL
        return result

    def close(self):
        self.lmp.close()

    def short_trajectory(self, positions, velocities, dt_fs=.5, steps=10):
        """Diagnostic-only true continuation from the supplied state itself."""
        self.evaluate(positions)
        velocity = np.ascontiguousarray(velocities, dtype=np.float64)
        self.lmp.scatter_atoms('v', 1, 3,
            velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.lmp.command(f'timestep {float(dt_fs)}')
        self.lmp.command('fix diagnostic_nve all nve')
        try:
            self.lmp.command(f'run {int(steps)} pre yes post no')
            q = np.ctypeslib.as_array(self.lmp.gather_atoms('x', 1, 3),
                                     shape=(self.n*3,)).copy().reshape(self.n, 3)
            v = np.ctypeslib.as_array(self.lmp.gather_atoms('v', 1, 3),
                                     shape=(self.n*3,)).copy().reshape(self.n, 3)
            # Each diagnostic step is only 5 fs. Reconstruct its continuous
            # endpoint around the supplied input, not around the original NVE.
            dq = (q-np.asarray(positions)) @ np.linalg.inv(self.cell)
            q = np.asarray(positions)+(dq-np.rint(dq)) @ self.cell
        finally:
            self.lmp.command('unfix diagnostic_nve')
        return q, v


def verify_reference(raw, calculator, frames):
    """Compare raw NVE energies/forces; return measured errors before acceptance."""
    # Long reference trajectories retain coordinates every .5 fs but thermo
    # every 5 fs. Match by the recorded MD step, never by array row number.
    thermo_steps = raw['thermo/step'][:]
    raw_steps = raw['step'][:]
    records = []
    for requested_frame in frames:
        step = raw_steps[requested_frame]
        ti = min(int(np.searchsorted(thermo_steps, step)), len(thermo_steps)-1)
        if ti > 0 and abs(thermo_steps[ti-1]-step) < abs(thermo_steps[ti]-step):
            ti -= 1
        frame = int(np.searchsorted(raw_steps, thermo_steps[ti]))
        if frame >= len(raw_steps) or raw_steps[frame] != thermo_steps[ti]:
            raise ValueError('No coordinate frame matches the saved thermo MD step')
        has_forces = 'forces_eV_per_A' in raw
        result = calculator.evaluate(raw['positions_unwrapped_A'][frame], forces=has_forces)
        record = dict(frame=frame, requested_frame=int(requested_frame), thermo_index=ti,
            energy_error_eV=float(result['pe']-raw['thermo/potential_energy_eV'][ti]),
            force_available=has_forces)
        if has_forces:
            force_error = result['forces_eV_A'] - raw['forces_eV_per_A'][frame]
            record.update(force_RMSE_eV_A=float(np.sqrt(np.mean(force_error**2))),
                          force_max_error_eV_A=float(abs(force_error).max()))
        records.append(record)
    return records
