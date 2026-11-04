"""Offline molecular motion decomposition; no constraints on the predictor."""
import numpy as np

from physics.units import KINETIC_TO_EV


def minimum_image(vector, cell):
    fractional = np.asarray(vector) @ np.linalg.inv(cell)
    return (fractional-np.rint(fractional)) @ cell


def molecular_geometry(positions, cell, atomic_numbers, molecule_ids):
    molecules = []
    for mid in np.unique(molecule_ids):
        ids = np.flatnonzero(molecule_ids == mid)
        oxygen = ids[atomic_numbers[ids] == 8]
        hydrogen = ids[atomic_numbers[ids] == 1]
        if len(oxygen) != 1 or len(hydrogen) != 2:
            raise ValueError('Expected OHH topology per molecule')
        molecules.append([int(oxygen[0]), *hydrogen.tolist()])
    molecules = np.asarray(molecules)
    q = np.asarray(positions)[..., molecules, :]
    bonds = minimum_image(q[..., 1:, :] - q[..., :1, :], cell)
    lengths = np.linalg.norm(bonds, axis=-1)
    cosine = np.sum(bonds[..., 0, :]*bonds[..., 1, :], axis=-1)/lengths.prod(axis=-1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return molecules, bonds, lengths, angle


def motion_partition(positions, velocities, cell, numbers, masses, molecule_ids):
    """Exact instantaneous translation/rotation/internal kinetic partition."""
    ids, bonds, lengths, angle = molecular_geometry(positions, cell, numbers, molecule_ids)
    mass = masses[ids]
    local = np.concatenate([np.zeros_like(bonds[..., :1, :]), bonds], axis=-2)
    com = (local * mass[..., None]).sum(axis=-2)/mass.sum(axis=-1)[..., None]
    r = local-com[..., None, :]
    v = np.asarray(velocities)[..., ids, :]
    cv = (v*mass[..., None]).sum(axis=-2)/mass.sum(axis=-1)[..., None]
    relative = v-cv[..., None, :]
    inertia = np.sum(mass[..., None, None]*(
        (r*r).sum(axis=-1)[..., None, None]*np.eye(3)-r[..., :, None]*r[..., None, :]), axis=-3)
    angular = np.sum(mass[..., None]*np.cross(r, relative), axis=-2)
    omega = np.linalg.solve(inertia, angular[..., None])[..., 0]
    rotational_v = np.cross(omega[..., None, :], r)
    internal_v = relative-rotational_v
    factor = .5*KINETIC_TO_EV
    translation = factor*np.sum(mass.sum(axis=-1)*(cv*cv).sum(axis=-1), axis=-1)
    rotation = factor*np.sum(mass*(rotational_v**2).sum(axis=-1), axis=(-2, -1))
    internal = factor*np.sum(mass*(internal_v**2).sum(axis=-1), axis=(-2, -1))
    total = factor*np.sum(masses*(np.asarray(velocities)**2).sum(axis=-1), axis=-1)
    np.testing.assert_allclose(translation+rotation+internal, total, rtol=1e-9, atol=1e-9)
    return dict(translation_eV=translation, rotation_eV=rotation, internal_eV=internal,
                total_KE_eV=total, bond_lengths_A=lengths, angles_deg=angle)


def signed_kinetic_error(predicted, reference, masses):
    error = np.asarray(predicted, dtype=float)-reference
    cross = KINETIC_TO_EV*np.sum(masses[..., None]*reference*error, axis=(-2, -1))
    quadratic = .5*KINETIC_TO_EV*np.sum(masses[..., None]*error**2, axis=(-2, -1))
    return cross, quadratic
