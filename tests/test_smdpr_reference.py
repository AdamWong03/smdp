import h5py
import numpy as np

from smdpr.reference import aligned_reference


def test_reference_groups_different_cadence_and_interrupted_copy(tmp_path):
    src, out = tmp_path/'source.h5', tmp_path/'aligned.h5'
    q = np.arange(30).reshape(10, 1, 3).astype(float)
    with h5py.File(src, 'w') as f:
        f['time_fs'] = np.arange(10)*.5
        f['step'] = np.arange(10)
        f['positions_unwrapped_A'] = q
        f['velocities_A_per_fs'] = q*.01
        f['atomic_numbers'] = [8]
        f['cell_A'] = np.tile(np.eye(3), (10, 1, 1))
        f['scalar'] = 5.
        group = f.create_group('thermo')
        group['step'] = np.arange(0, 10, 2)
        group['time_fs'] = np.arange(0, 10, 2)*.5
        group['potential_energy_eV'] = [-5., -4., -3., -2., -1.]
    with h5py.File(out, 'w') as f:
        f['positions_unwrapped_A'] = q[:1]
    aligned_reference(src, out, 4)
    with h5py.File(out) as f:
        assert f.attrs['aligned_reference_completed']
        np.testing.assert_array_equal(f['time_fs'][:], np.arange(6)*.5)
        np.testing.assert_array_equal(f['positions_unwrapped_A'][:], q[4:])
        np.testing.assert_array_equal(f['thermo/step'][:], [4, 6, 8])
        np.testing.assert_array_equal(f['thermo/potential_energy_eV'][:], [-3., -2., -1.])
        assert f['scalar'][()] == 5.
    assert len(list(tmp_path.glob('aligned.h5.interrupted-*'))) == 1
    previous = out.stat().st_mtime_ns
    aligned_reference(src, out, 4)
    assert out.stat().st_mtime_ns == previous
