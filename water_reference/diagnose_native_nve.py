"""Interim validation NVE, using the official Forecast step unchanged.

The checkpoint is an immutable validation-selected snapshot, not a final test
model. Only HDF5 recording and continuous-image reconstruction are added.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.model.forecast import Forecast
from trajcast.model.models import EfficientTrajCastModel
from trajcast.utils.atomic_computes import wrap_positions_back_to_box_torch

from physics.units import KB_EV_PER_K, KINETIC_TO_EV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_default_dtype(torch.float64)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert state['config']['training']['model_type'] == 'EfficientTrajCastModel'
    model = EfficientTrajCastModel(config=state['resolved_native_model_config'],
        predicted_fields=['displacements', 'update_velocities'],
        rms_targets=state['rms_targets'].tolist(), mean_targets=[0., 0.])
    model.load_state_dict(state['model'], strict=True)
    with h5py.File(args.initial) as source:
        q = source['positions_unwrapped_A'][0].astype(np.float64)
        v = source['velocities_A_per_fs'][0].astype(np.float64)
        cell = source['cell_A'][0]
        z, masses, mol = source['atomic_numbers'][:], source['masses_u'][:], source['molecule_ids'][:]
    graph = AtomicGraph.from_atoms_dict(dict(
        positions=wrap_positions_back_to_box_torch(torch.from_numpy(q), torch.from_numpy(cell)),
        velocities=torch.from_numpy(v), cell=torch.from_numpy(cell),
        atomic_numbers=torch.as_tensor(z, dtype=torch.long).reshape(-1, 1),
        atomic_masses=torch.from_numpy(masses).reshape(-1, 1),
        pbc=torch.tensor([True, True, True])), r_cut=6., atom_type_mapper={1: 0, 8: 1})
    steps, dt = 4000, 5.
    engine = Forecast(dict(device='cuda', seed=2026111601, model=model,
        configuration=graph, timestep=dt, run=steps, units='real'))
    frame = engine.start_graph
    # Verify native Forecast's normalization/update against its own readout.
    with torch.no_grad():
        original_q = frame.pos.clone()
        raw = model(frame.clone()).target
        expected_q = wrap_positions_back_to_box_torch(original_q+raw[:, :3]*model.rms_targets[0], frame.cell)
        expected_v = raw[:, 3:]*model.rms_targets[1]
        trial = engine._make_timestep(frame.clone(), 1)
        q_error = float((trial.pos-expected_q).abs().max())
        v_error = float((trial.velocities-expected_v).abs().max())
        assert q_error < 1e-10 and v_error < 1e-12
    torch.cuda.reset_peak_memory_stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    completed, failure = 0, None
    with h5py.File(args.output, 'x') as output:
        output.attrs.update(ensemble='nve', timestep_fs=dt, requested_steps=steps,
            thermalization_steps=0, completed_steps=0, stable=False, failure_reason='',
            checkpoint=str(args.checkpoint), selected_epoch=state['epoch'],
            initial_trajectory=str(args.initial), initial_frame=0,
            interim_validation_diagnosis=True, final_test_result=False,
            reference_forces_used_during_propagation=False,
            propagation_implementation='official Forecast._make_timestep',
            execution_note='Shared L40S; no inference efficiency claim')
        for key, values in [('cell', cell), ('atomic_numbers', z), ('masses_u', masses), ('molecule_ids', mol)]:
            output[key] = values
        positions = output.create_dataset('positions', (steps+1, len(z), 3), dtype='f8', chunks=(1, len(z), 3))
        velocities = output.create_dataset('velocities', (steps+1, len(z), 3), dtype='f8', chunks=(1, len(z), 3))
        temperatures = output.create_dataset('temperature_K', (steps+1,), dtype='f8')
        output['time_fs'] = np.arange(steps+1)*dt
        def record(index):
            positions[index], velocities[index] = q, v
            temperatures[index] = KINETIC_TO_EV*np.sum(masses[:, None]*v*v)/((3*len(z)-3)*KB_EV_PER_K)
        record(0)
        try:
            with torch.no_grad():
                for index in range(1, steps+1):
                    previous = frame.pos.clone()
                    frame = engine._make_timestep(frame, index)
                    wrapped_delta = (frame.pos-previous).cpu().numpy()
                    fractional = wrapped_delta@np.linalg.inv(cell)
                    q = q+(fractional-np.rint(fractional))@cell
                    v = frame.velocities.cpu().numpy()
                    if not np.isfinite(q).all() or not np.isfinite(v).all():
                        raise FloatingPointError('Nonfinite autonomous state')
                    record(index)
                    completed = index
                    output.attrs['completed_steps'] = index
                    if index % 500 == 0:
                        output.flush()
                        print(json.dumps(dict(step=index, temperature_K=float(temperatures[index]),
                            elapsed_seconds=time.monotonic()-start)), flush=True)
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
        output.attrs.update(stable=failure is None, failure_reason=failure or '')
    result = dict(completed_steps=completed, requested_steps=steps, selected_epoch=state['epoch'],
        checkpoint=str(args.checkpoint), stable=failure is None, failure_reason=failure,
        interim_validation_diagnosis=True, final_test_result=False,
        native_forecast_position_check_A=q_error, native_forecast_velocity_check_A_fs=v_error,
        elapsed_seconds=time.monotonic()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    args.output.with_suffix('.summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
