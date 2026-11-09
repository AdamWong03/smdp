"""Original LAMMPS finite-time continuation for evaluation only."""
import ctypes

import numpy as np

from dynamics_diagnosis.common import DATA
from physics.spcfw import WaterReference, verify_reference
from reference_md.write_lammps_protocols import bks_commands, EV_TO_KCAL_MOL


class QuartzReference:
    def __init__(self, cell, numbers, masses):
        from lammps import lammps
        self.lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
        self.lmp.commands_list(['units real', 'atom_style charge', 'atom_modify map array',
            'boundary p p p', f'read_data "{DATA}/quartz/equilibration/snapshots/quartz_7.data"',
            *bks_commands(), 'thermo_style custom step pe', 'run 0'])
        self.n = int(self.lmp.get_natoms())
        self.cell = np.asarray(cell, dtype=float)
        lo, hi, xy, yz, xz, _, _ = self.lmp.extract_box()
        self.origin = np.asarray(lo)
        l = np.asarray(hi)-lo
        np.testing.assert_allclose([[l[0],0,0],[xy,l[1],0],[xz,yz,l[2]]], self.cell, atol=1e-7)
        types = np.ctypeslib.as_array(self.lmp.gather_atoms('type',0,1),shape=(self.n,))
        np.testing.assert_array_equal(np.where(types==1,8,14),numbers)
        actual = self.lmp.numpy.extract_atom('mass')[types]
        # Never overwrite original MD masses with converted HDF5 metadata.
        # Record any converter discrepancy and test the actual continuation.
        self.integration_masses = actual.copy()

    def evaluate(self, positions, forces=False):
        q = np.asarray(positions,dtype=np.float64)
        frac = (q-self.origin)@np.linalg.inv(self.cell)
        image = np.floor(frac).astype(np.int64)
        wrapped = np.ascontiguousarray(self.origin+(frac-image)@self.cell)
        self.lmp.scatter_atoms('x',1,3,wrapped.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        nlocal = int(self.lmp.extract_setting('nlocal'))
        if nlocal!=self.n:
            raise ValueError('Evaluation calculator must use one MPI rank')
        ids = self.lmp.numpy.extract_atom('id')[:nlocal]
        encoded = np.asarray([self.lmp.encode_image_flags(*map(int,x)) for x in image],
                             dtype=self.lmp.c_imageint)
        self.lmp.numpy.extract_atom('image')[:nlocal] = encoded[ids-1]
        self.lmp.command('run 0 pre yes post no')
        result = dict(pe=float(self.lmp.get_thermo('pe'))/EV_TO_KCAL_MOL)
        if forces:
            result['forces_eV_A'] = np.ctypeslib.as_array(self.lmp.gather_atoms('f',1,3),
                shape=(self.n*3,)).copy().reshape(self.n,3)/EV_TO_KCAL_MOL
        return result

    short_trajectory = WaterReference.short_trajectory

    def close(self):
        self.lmp.close()


def calculator(system, raw):
    cell, numbers, masses = raw['cell_A'][0],raw['atomic_numbers'][:],raw['masses_u'][:]
    if system=='water':
        return WaterReference(DATA/'water/equilibration/snapshots/water_5.data', cell, numbers)
    return QuartzReference(cell,numbers,masses)


def validate_integrator(system, raw, calc):
    dt = float(raw.attrs['integration_dt_fs'])
    horizon = 30. if system=='quartz' else 5.
    steps = round(horizon/dt)
    frames = [0, 1000, 20000]
    verification = verify_reference(raw,calc,frames)
    types=np.ctypeslib.as_array(calc.lmp.gather_atoms('type',0,1),shape=(calc.n,))
    integration_masses=calc.lmp.numpy.extract_atom('mass')[types].copy()
    stored_masses=raw['masses_u'][:]
    sampled_velocity=raw['velocities_A_per_fs'][::1000].astype(float)
    from physics.units import KINETIC_TO_EV
    mass_ke_bias=.5*KINETIC_TO_EV*np.sum(sampled_velocity**2*
        (stored_masses-integration_masses)[None,:,None],axis=(1,2))
    records = []
    for frame in frames:
        q,v = raw['positions_unwrapped_A'][frame], raw['velocities_A_per_fs'][frame]
        q1,v1 = calc.short_trajectory(q,v,dt_fs=dt,steps=steps)
        rq,rv = raw['positions_unwrapped_A'][frame+steps], raw['velocities_A_per_fs'][frame+steps]
        records.append(dict(frame=frame, horizon_fs=horizon,
            q_max_error_A=float(abs(q1-rq).max()), v_max_error_A_fs=float(abs(v1-rv).max())))
    passed = (all(r['q_max_error_A']<5e-4 and r['v_max_error_A_fs']<5e-5 for r in records)
        and all(abs(r['energy_error_eV'])<1e-3 and r.get('force_max_error_eV_A',0)<.02 for r in verification))
    return dict(passed=passed, potential_force_checks=verification, continuation_checks=records,
        integration_masses_by_element={str(int(z)):float(integration_masses[raw['atomic_numbers'][:]==z][0])
            for z in np.unique(raw['atomic_numbers'][:])},
        stored_masses_by_element={str(int(z)):float(stored_masses[raw['atomic_numbers'][:]==z][0])
            for z in np.unique(raw['atomic_numbers'][:])},
        maximum_mass_metadata_difference_u=float(abs(integration_masses-stored_masses).max()),
        metadata_vs_integrator_KE_bias_eV=dict(mean=float(mass_ke_bias.mean()),
            max_abs=float(abs(mass_ke_bias).max()),sample_stride_reference_frames=1000),
        integration_dt_fs=dt, horizon_fs=horizon, reference_for_training=False,
        energy_convention='raw original LAMMPS eV; no recentering')
