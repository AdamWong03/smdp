"""Autonomous 30 fs endpoint rollout for a trained relational model."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from periodic import PeriodicGraphBuilder
from physics.units import KB_EV_PER_K, KINETIC_TO_EV
from relational_inference import load_relational, step


def temperature(velocities, masses):
    return KINETIC_TO_EV * np.sum(masses[:, None] * velocities**2) / ((3*len(masses)-3)*KB_EV_PER_K)


class CSVR:
    def __init__(self, masses, dt_fs, tau_fs, seed):
        self.masses = masses
        self.dof = 3 * len(masses) - 3
        self.c = math.exp(-dt_fs / tau_fs)
        self.rng = np.random.default_rng(seed)

    def __call__(self, velocities):
        com = (velocities * self.masses[:, None]).sum(0) / self.masses.sum()
        internal = velocities - com
        kinetic = .5 * KINETIC_TO_EV * np.sum(self.masses[:, None] * internal**2)
        if not math.isfinite(kinetic) or kinetic <= 0:
            raise FloatingPointError("Invalid pre-thermostat kinetic energy")
        normal = self.rng.normal()
        chi = self.rng.chisquare(self.dof - 1)
        ratio = (1-self.c) * (.5*KB_EV_PER_K*300) / kinetic
        alpha2 = self.c + ratio*(normal*normal+chi) + 2*normal*math.sqrt(self.c*ratio)
        alpha = math.copysign(math.sqrt(max(alpha2, 0)), math.sqrt(self.c)+math.sqrt(ratio)*normal)
        return com + alpha*internal, alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ensemble", choices=["nve", "nvt"], required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--thermalization-steps", type=int, default=0)
    parser.add_argument("--thermostat-metadata", type=Path)
    parser.add_argument("--seed", type=int, default=2026110301)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, saved = load_relational(args.checkpoint, device)
    dt = model.cfg.horizon_fs
    with h5py.File(args.initial) as source:
        if not 0 <= args.initial_frame < len(source["positions_unwrapped_A"]):
            raise IndexError("initial-frame outside source trajectory")
        q = source["positions_unwrapped_A"][args.initial_frame].astype(np.float32)
        v = source["velocities_A_per_fs"][args.initial_frame].astype(np.float32)
        cell = source["cell_A"][args.initial_frame]
        z, masses = source["atomic_numbers"][:], source["masses_u"][:]
    thermostat = None
    thermostat_metadata = {}
    if args.ensemble == "nvt":
        thermostat_metadata = json.loads(args.thermostat_metadata.read_text())
        if thermostat_metadata["ensemble"] != "NVT-CSVR" or thermostat_metadata["temperature_K"] != 300:
            raise ValueError("Expected the frozen 300 K NVT-CSVR protocol")
        thermostat = CSVR(masses, dt, thermostat_metadata["thermostat_tau_fs"], args.seed)
    total_steps = args.thermalization_steps + args.steps
    neighbors = PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    near = PeriodicGraphBuilder(3.0)
    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    start = time.perf_counter()
    model_seconds = graph_seconds = 0.0
    failure = None
    completed = 0
    with h5py.File(args.output, "x") as output:
        output.attrs.update(ensemble=args.ensemble, timestep_fs=dt, requested_steps=args.steps,
                            thermalization_steps=args.thermalization_steps, completed_steps=0,
                            stable=False, failure_reason="", checkpoint=str(args.checkpoint),
                            checkpoint_sha256=checksum, seed=args.seed,
                            initial_trajectory=str(args.initial), initial_frame=args.initial_frame,
                            thermostat_metadata=json.dumps(thermostat_metadata),
                            reference_forces_used_during_propagation=False)
        output["atomic_numbers"] = z; output["masses_u"] = masses; output["cell"] = cell
        shape = (total_steps+1, len(z), 3)
        positions = output.create_dataset("positions", shape, dtype="f4", chunks=(128, len(z), 3), compression="lzf")
        velocities = output.create_dataset("velocities", shape, dtype="f4", chunks=(128, len(z), 3), compression="lzf")
        scalars = {name: output.create_dataset(name, (total_steps+1,), dtype="f8") for name in
                   ["time_fs", "temperature_K", "pre_thermostat_temperature_K",
                    "thermostat_alpha", "minimum_distance_A"]}
        positions[0], velocities[0] = q, v
        scalars["temperature_K"][0] = scalars["pre_thermostat_temperature_K"][0] = temperature(v, masses)
        scalars["thermostat_alpha"][0] = 1.0
        edges, shifts = near.one(q, cell)
        scalars["minimum_distance_A"][0] = np.linalg.norm(q[edges[0]]+shifts-q[edges[1]], axis=1).min()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for index in range(1, total_steps+1):
                torch.cuda.synchronize(device); tic = time.perf_counter()
                qn, vn = step(model, q, v, cell, z, masses, device, builder=neighbors)
                torch.cuda.synchronize(device); model_seconds += time.perf_counter()-tic
                q, v = qn.cpu().numpy(), vn.cpu().numpy()
                pre = temperature(v, masses)
                edges, shifts = near.one(q, cell)
                minimum = float(np.linalg.norm(q[edges[0]]+shifts-q[edges[1]], axis=1).min())
                if not np.isfinite(q).all() or not np.isfinite(v).all():
                    raise FloatingPointError("Non-finite predicted phase space")
                if minimum < .65 or pre > 3000 or pre < 1:
                    raise RuntimeError(f"physical safety stop: min={minimum}, preT={pre}")
                alpha = 1.0
                if thermostat is not None:
                    v, alpha = thermostat(v)
                post = temperature(v, masses)
                positions[index], velocities[index] = q, v
                scalars["time_fs"][index] = index*dt
                scalars["pre_thermostat_temperature_K"][index] = pre
                scalars["temperature_K"][index] = post
                scalars["thermostat_alpha"][index] = alpha
                scalars["minimum_distance_A"][index] = minimum
                completed = index; output.attrs["completed_steps"] = completed
                if index % 1000 == 0:
                    output.flush()
                    print(json.dumps(dict(step=index, total=total_steps, temperature_K=post,
                                          minimum_A=minimum, elapsed_s=time.perf_counter()-start)), flush=True)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        elapsed = time.perf_counter()-start
        output.attrs.update(stable=failure is None, failure_reason=failure or "")
        output.flush()
    result = dict(completed_steps=completed, production_steps=max(0, completed-args.thermalization_steps),
                  simulated_ps=completed*dt/1000, stable=failure is None, failure_reason=failure,
                  elapsed_seconds=elapsed, model_and_graph_seconds=model_seconds,
                  ns_day_end_to_end=completed*dt*1e-6/max(elapsed, 1e-9)*86400,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                  hardware=torch.cuda.get_device_name(device), parameters=sum(p.numel() for p in model.parameters()),
                  selected_epoch=saved["epoch"]+1, checkpoint_sha256=checksum)
    args.output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
