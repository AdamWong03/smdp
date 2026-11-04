import numpy as np

from physics.water_motion import motion_partition, signed_kinetic_error
from physics.units import KINETIC_TO_EV
from physics.diffusion import time_averaged_msd


def test_rigid_motion_partition_and_rotation_invariance():
    q = np.array([[0., 0., 0.], [.8, 0., .6], [-.8, 0., .6]])
    masses = np.array([16., 1., 1.]); z = np.array([8, 1, 1]); mol = np.ones(3)
    q -= np.sum(q*masses[:, None], axis=0)/masses.sum()
    v = np.array([.001, -.002, .003])+np.cross([.02, -.01, .03], q)
    out = motion_partition(q, v, np.eye(3)*10, z, masses, mol)
    assert abs(out['internal_eV']) < 1e-20
    rot, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    rotated = motion_partition(q@rot, v@rot, np.eye(3)*10@rot, z, masses, mol)
    for key in ['translation_eV', 'rotation_eV', 'internal_eV']:
        np.testing.assert_allclose(out[key], rotated[key], atol=1e-12)
    shifted = q.copy(); shifted[1] += [10., 0., 0.]
    wrapped = motion_partition(shifted, v, np.eye(3)*10, z, masses, mol)
    np.testing.assert_allclose(out['bond_lengths_A'], wrapped['bond_lengths_A'])


def test_signed_kinetic_error_identity():
    rng = np.random.default_rng(9)
    truth = rng.normal(size=(4, 3, 3))*.01
    pred = truth+rng.normal(size=truth.shape)*.001
    masses = np.array([16., 1., 1.])
    cross, quadratic = signed_kinetic_error(pred, truth, masses)
    actual = .5*KINETIC_TO_EV*np.sum(masses[:, None]*(pred**2-truth**2), axis=(1, 2))
    np.testing.assert_allclose(cross+quadratic, actual, atol=1e-14)


def test_fft_msd_matches_all_time_origin_definition():
    rng=np.random.default_rng(19)
    q=np.cumsum(rng.normal(size=(50,4,3)),axis=0)
    expected=[0.]+[np.mean(np.sum((q[k:]-q[:-k])**2,axis=-1)) for k in range(1,len(q))]
    np.testing.assert_allclose(time_averaged_msd(q),expected,atol=1e-12)
