"""Evaluate 1/2/4/8 autonomous 30 fs transitions on independent test starts."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

from physics.units import KB_EV_PER_K, KINETIC_TO_EV
from relational_inference import atomic_graph, load_relational


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    model, saved = load_relational(args.checkpoint, device)
    with h5py.File(args.transitions) as source:
        q0 = source["positions_initial_A"][:]
        v0 = source["velocities_initial_A_per_fs"][:]
        cells = source["cell_A"][:]
        trajectories = source["source_trajectory"][:].astype(int)
        frames = source["source_frame"][:].astype(int)
        paths = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in source["source_paths"][:]]
        z, masses = source["atomic_numbers"][:], source["masses_u"][:]
        lag_steps = int(source.attrs["lag_steps"])
        source_dt_fs = float(source.attrs["source_dt_fs"])
        if not np.isclose(lag_steps * source_dt_fs, model.cfg.horizon_fs):
            raise ValueError("Transition lag/source timestep differs from model horizon")
    horizons = (1, 2, 4, 8)
    valid = np.ones(len(q0), dtype=bool)
    handles = [h5py.File(path, "r") for path in paths]
    try:
        for i, handle in enumerate(handles):
            mask = trajectories == i
            valid[mask] &= frames[mask] + max(horizons) * lag_steps < len(handle["time_fs"])
        selected = np.flatnonzero(valid)
        references = {}
        for horizon in horizons:
            rq, rv = np.empty_like(q0[selected]), np.empty_like(v0[selected])
            for trajectory, handle in enumerate(handles):
                local = np.flatnonzero(trajectories[selected] == trajectory)
                if not len(local):
                    continue
                indices = frames[selected[local]] + horizon * lag_steps
                unique, inverse = np.unique(indices, return_inverse=True)
                rq[local] = handle["positions_unwrapped_A"][unique][inverse]
                rv[local] = handle["velocities_A_per_fs"][unique][inverse]
            references[horizon] = (rq, rv)
    finally:
        for handle in handles:
            handle.close()

    sums = {h: np.zeros(5, dtype=np.float64) for h in horizons}
    model_seconds = graph_seconds = 0.0
    start = time.perf_counter()
    for begin in range(0, len(selected), args.batch_size):
        ids = selected[begin:begin + args.batch_size]
        q = torch.as_tensor(q0[ids], device=device)
        v = torch.as_tensor(v0[ids], device=device)
        for horizon in range(1, max(horizons) + 1):
            if device.type == "cuda": torch.cuda.synchronize(device)
            tic = time.perf_counter()
            graphs = [atomic_graph(q[i].cpu(), v[i].cpu(), cells[j], z, masses, model, device="cpu")
                      for i, j in enumerate(ids)]
            graph = Batch.from_data_list(graphs).to(device)
            if device.type == "cuda": torch.cuda.synchronize(device)
            graph_seconds += time.perf_counter() - tic
            tic = time.perf_counter()
            with torch.inference_mode():
                output = model(graph)
            if device.type == "cuda": torch.cuda.synchronize(device)
            model_seconds += time.perf_counter() - tic
            q = output["positions"].reshape(len(ids), len(z), 3)
            v = output["velocities"].reshape(len(ids), len(z), 3)
            if horizon in horizons:
                rq, rv = references[horizon]
                dq = q.cpu().numpy() - rq[begin:begin + len(ids)]
                dv = v.cpu().numpy() - rv[begin:begin + len(ids)]
                dof = 3 * len(z) - 3
                temperature = KINETIC_TO_EV * (masses[None, :, None] * v.cpu().numpy()**2).sum((1, 2)) / (dof * KB_EV_PER_K)
                ref_temperature = KINETIC_TO_EV * (masses[None, :, None] * rv[begin:begin + len(ids)]**2).sum((1, 2)) / (dof * KB_EV_PER_K)
                sums[horizon] += [np.square(dq).sum(), np.abs(dq).sum(),
                                  np.square(dv).sum(), np.abs(dv).sum(),
                                  np.square(temperature - ref_temperature).sum()]
    elapsed = time.perf_counter() - start
    count = len(selected) * len(z) * 3
    result = {"samples": int(len(selected)), "checkpoint": str(args.checkpoint),
              "selected_epoch": saved["epoch"] + 1, "batch_size": args.batch_size,
              "elapsed_seconds": elapsed, "model_seconds": model_seconds,
              "neighbor_graph_seconds": graph_seconds, "horizons": {}}
    for horizon in horizons:
        value = sums[horizon]
        result["horizons"][str(horizon)] = {
            "time_fs": horizon * model.cfg.horizon_fs,
            "position_RMSE_A": float(np.sqrt(value[0] / count)),
            "position_MAE_A": float(value[1] / count),
            "velocity_RMSE_A_fs": float(np.sqrt(value[2] / count)),
            "velocity_MAE_A_fs": float(value[3] / count),
            "temperature_RMSE_K": float(np.sqrt(value[4] / len(selected)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
