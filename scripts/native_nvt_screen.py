"""100 ps NVT screen for the native TrajCast validation-best checkpoint.

This is a root-cause control, not a SOTA claim: the native protocol was stopped
before completion, but the validation-best checkpoint is immutable and can be
run under the same 50 ps thermalization + 100 ps production protocol used for
our models.
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


def temperature(velocities, masses):
    return (KINETIC_TO_EV * np.sum(masses[:, None] * velocities**2)
            / ((3 * len(masses) - 3) * KB_EV_PER_K))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thermalization-steps", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026110401)
    parser.add_argument("--ensemble", choices=["nvt", "nve"], default="nvt")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_default_dtype(torch.float64)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert state["config"]["training"]["model_type"] == "EfficientTrajCastModel"
    model = EfficientTrajCastModel(
        config=state["resolved_native_model_config"],
        predicted_fields=["displacements", "update_velocities"],
        rms_targets=state["rms_targets"].tolist(),
        mean_targets=[0.0, 0.0])
    model.load_state_dict(state["model"], strict=True)
    with h5py.File(args.initial) as source:
        q = source["positions_unwrapped_A"][0].astype(np.float64)
        v = source["velocities_A_per_fs"][0].astype(np.float64)
        cell = source["cell_A"][0].astype(np.float64)
        z = source["atomic_numbers"][:]
        masses = source["masses_u"][:]
        molecules = source["molecule_ids"][:]
    graph = AtomicGraph.from_atoms_dict(dict(
        positions=wrap_positions_back_to_box_torch(torch.from_numpy(q), torch.from_numpy(cell)),
        velocities=torch.from_numpy(v), cell=torch.from_numpy(cell),
        atomic_numbers=torch.as_tensor(z, dtype=torch.long).reshape(-1, 1),
        atomic_masses=torch.from_numpy(masses).reshape(-1, 1),
        pbc=torch.tensor([True, True, True])), r_cut=6.0, atom_type_mapper={1: 0, 8: 1})
    total = args.thermalization_steps + args.steps
    protocol = dict(
        device="cuda", seed=args.seed, model=model, configuration=graph,
        timestep=5.0, run=total, units="real")
    if args.ensemble == "nvt":
        protocol.update(thermostat={"Tdamp": 50.0}, temperature=300.0)
    engine = Forecast(protocol)
    frame = engine.start_graph
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed, failure = 0, None
    start = time.monotonic()
    with h5py.File(args.output, "x") as output:
        output.attrs.update(
            ensemble=args.ensemble, timestep_fs=5.0, requested_steps=args.steps,
            thermalization_steps=args.thermalization_steps, completed_steps=0,
            stable=False, failure_reason="", checkpoint=str(args.checkpoint),
            selected_epoch=state.get("epoch"), seed=args.seed,
            initial_trajectory=str(args.initial), initial_frame=0,
            native_protocol_completed=False,
            execution_note="root-cause control; native validation-best checkpoint; 100 ps NVT screen",
            reference_forces_used_during_propagation=False)
        for key, values in [("cell", cell), ("atomic_numbers", z),
                            ("masses_u", masses), ("molecule_ids", molecules)]:
            output[key] = values
        positions = output.create_dataset(
            "positions", (total + 1, len(z), 3), dtype="f8",
            chunks=(1, len(z), 3))
        velocities = output.create_dataset(
            "velocities", (total + 1, len(z), 3), dtype="f8",
            chunks=(1, len(z), 3))
        temperatures = output.create_dataset("temperature_K", (total + 1,), dtype="f8")
        output["time_fs"] = np.arange(total + 1) * 5.0

        def record(index):
            positions[index], velocities[index] = q, v
            temperatures[index] = temperature(v, masses)

        record(0)
        try:
            with torch.no_grad():
                for index in range(1, total + 1):
                    previous = frame.pos.clone()
                    frame = engine._make_timestep(frame, index)
                    wrapped_delta = (frame.pos - previous).cpu().numpy()
                    fractional = wrapped_delta @ np.linalg.inv(cell)
                    q = q + (fractional - np.rint(fractional)) @ cell
                    v = frame.velocities.cpu().numpy()
                    if not np.isfinite(q).all() or not np.isfinite(v).all():
                        raise FloatingPointError("nonfinite autonomous state")
                    record(index)
                    completed = index
                    output.attrs["completed_steps"] = index
                    if index % 1000 == 0:
                        output.flush()
                        print(json.dumps(dict(step=index, total=total,
                                              temperature_K=float(temperatures[index]),
                                              elapsed_s=time.monotonic() - start)),
                              flush=True)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        output.attrs.update(stable=failure is None, failure_reason=failure or "")
    result = dict(completed_steps=completed, requested_steps=args.steps,
                  production_steps=max(0, completed - args.thermalization_steps),
                  simulated_ps=completed * 5.0 / 1000,
                  stable=failure is None, failure_reason=failure,
                  selected_epoch=state.get("epoch"), checkpoint=str(args.checkpoint),
                  elapsed_seconds=time.monotonic() - start,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated())
    args.output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
