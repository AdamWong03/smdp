"""Align a retained reference trajectory without changing physical values."""
from pathlib import Path
import os
import time

import h5py
import numpy as np


def aligned_reference(source, destination, first_frame):
    """Crop coordinates AND the independently sampled thermo group by MD step.

    Publication is atomic. Old interrupted intermediate copies are retained,
    never mistaken for a complete reference merely because the file exists.
    """
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists():
        with h5py.File(destination, 'r') as existing:
            if (existing.attrs.get('aligned_reference_completed', False)
                    and existing.attrs.get('reference_source') == str(source)
                    and existing.attrs.get('reference_initial_frame') == first_frame):
                return destination
        backup = destination.with_name(destination.name+f'.interrupted-{time.time_ns()}')
        destination.rename(backup)
    temporary = destination.with_name(destination.name+'.partial')
    if temporary.exists():
        temporary.rename(temporary.with_name(temporary.name+f'.interrupted-{time.time_ns()}'))
    with h5py.File(source, 'r') as src, h5py.File(temporary, 'x') as dst:
        frames = len(src['time_fs'])
        if not 0 <= first_frame < frames:
            raise IndexError('Reference initial frame outside trajectory')
        initial_step = src['step'][first_frame]
        initial_time = src['time_fs'][first_frame]

        def copy_group(original, target):
            target.attrs.update(dict(original.attrs))
            group_steps = original.get('step')
            local_start = None
            if isinstance(group_steps, h5py.Dataset) and group_steps.ndim == 1:
                local_start = int(np.searchsorted(group_steps[:], initial_step))
            for key, value in original.items():
                if isinstance(value, h5py.Group):
                    copy_group(value, target.create_group(key))
                    continue
                if value.ndim and value.shape[0] == frames:
                    data = value[first_frame:]
                elif (local_start is not None and value.ndim
                      and value.shape[0] == len(group_steps)):
                    data = value[local_start:]
                else:
                    data = value[()]
                if key == 'time_fs':
                    data = data-initial_time
                dataset = target.create_dataset(key, data=data)
                dataset.attrs.update(dict(value.attrs))

        copy_group(src, dst)
        dst.attrs.update(n_frames=frames-first_frame,
                         reference_initial_frame=first_frame, reference_source=str(source))
        np.testing.assert_array_equal(dst['positions_unwrapped_A'][0],
                                      src['positions_unwrapped_A'][first_frame])
        np.testing.assert_array_equal(dst['velocities_A_per_fs'][0],
                                      src['velocities_A_per_fs'][first_frame])
        if 'thermo/step' in src:
            thermo_steps = dst['thermo/step'][:]
            if not len(thermo_steps) or thermo_steps[0] < initial_step:
                raise ValueError('Reference thermodynamic series is misaligned')
        dst.attrs['aligned_reference_completed'] = True
        dst.flush()
    os.replace(temporary, destination)
    return destination
