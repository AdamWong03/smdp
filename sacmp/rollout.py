"""Autonomous 16 fs quartz NVE/NVT rollouts using the canonical propagator.

Output preserves the existing analyze_quartz.py HDF5 contract. No external
physical force is called during propagation. The CSVR thermostat is applied
after each macrostep in NVT only; the deterministic map remains distinct.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch

from periodic import PeriodicGraphBuilder
from physics.units import KINETIC_TO_EV, KB_EV_PER_K
from rollout_quartz import CSVR, temperature
from sacmp.prepare import make_model


def load_checkpoint(path, device):
    saved = torch.load(path, weights_only=False, map_location="cpu")
    model = make_model(saved["bootstrap"], device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def _initial(path):
    with h5py.File(path, "r") as source:
        return (
            source["positions_unwrapped_A"][0].astype("f4"),
            source["velocities_A_per_fs"][0].astype("f4"),
            source["cell_A"][0].astype("f4"),
            source["atomic_numbers"][:],
            source["masses_u"][:],
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ensemble", choices=["nve", "nvt"], required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--thermalization-steps", type=int, default=0)
    p.add_argument("--thermostat-metadata", type=Path)
    p.add_argument("--seed", type=int, default=2026102201)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dense", action="store_true",
                   help="Query 1..16 fs for reference-compatible dense VDOS")
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError("Preserve any previous rollout; choose new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, saved = load_checkpoint(args.checkpoint, device)
    q0, v0, cell, z, mass = _initial(args.initial)
    if not np.array_equal(z, model.atomic_numbers.cpu().numpy()):
        raise ValueError("Checkpoint and trajectory have different atom ordering")
    np.testing.assert_allclose(mass, model.mass.cpu().numpy(), rtol=1e-5)
    np.testing.assert_allclose(
        cell, np.asarray(saved["anchor_report"].get("common_cell", cell)),
        atol=1e-4)
    near = PeriodicGraphBuilder(3.)
    def minimum(q):
        e, shift = near.one(q, cell)
        if not e.shape[1]:
            return 3.
        return float(np.linalg.norm(q[e[0]]+shift-q[e[1]], axis=-1).min())
    thermo = None
    metadata = {}
    if args.ensemble == "nvt":
        if args.thermostat_metadata is None:
            raise ValueError("NVT requires reference thermostat metadata")
        metadata = json.loads(args.thermostat_metadata.read_text())
        if metadata["ensemble"] != "NVT-CSVR" or metadata["temperature_K"] != 300:
            raise ValueError("Only matched 300K NVT-CSVR reference is supported")
        thermo = CSVR(mass, 16., metadata["thermostat_tau_fs"], args.seed)
    total_steps = args.steps+args.thermalization_steps
    initial_cm = np.sum(v0*mass[:, None], axis=0)/mass.sum()
    q = torch.from_numpy(q0).to(device).unsqueeze(0)
    v = torch.from_numpy(v0).to(device).unsqueeze(0)
    times = list(range(1, 17)) if args.dense else [16]
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    f = h5py.File(args.output, "x")
    f.attrs.update(ensemble=args.ensemble, timestep_fs=16.,
                   requested_steps=args.steps,
                   thermalization_steps=args.thermalization_steps,
                   checkpoint=str(args.checkpoint),
                   checkpoint_sha256=checkpoint_hash, seed=args.seed,
                   initial_trajectory=str(args.initial),
                   dense_sample_dt_fs=1. if args.dense else -1.,
                   dense_endpoint_velocity="post-CSVR at macro boundaries; pre-CSVR otherwise",
                   thermostat_metadata=json.dumps(metadata))
    f.create_dataset("atomic_numbers", data=z)
    f.create_dataset("masses_u", data=mass)
    f.create_dataset("cell", data=cell)
    def storage(name, shape):
        return f.create_dataset(name, shape=(0, *shape),
                                maxshape=(None, *shape),
                                chunks=(min(256, total_steps+1), *shape),
                                compression="lzf", dtype="f4")
    arrays = {field:storage(field, (len(z), 3))
              for field in ["positions", "velocities"]}
    for field in ["time_fs", "temperature_K", "pre_thermostat_temperature_K",
                  "minimum_distance_A", "thermostat_alpha",
                  "segment_join_derivative_RMS_A_fs"]:
        arrays[field] = storage(field, ())
    dense = storage("dense/velocities", (len(z), 3)) if args.dense else None
    buffers = {field:[] for field in arrays}
    dense_buffer = []
    def append(qn, vn, elapsed, pre, mind, alpha, join):
        vals = [qn, vn, elapsed, temperature(vn, mass), pre, mind, alpha, join]
        for key, val in zip(arrays, vals):
            buffers[key].append(val)
    def flush():
        for name, dset in arrays.items():
            if buffers[name]:
                block = np.asarray(buffers[name], dtype="f4")
                old = len(dset)
                dset.resize(old+len(block), axis=0)
                dset[old:] = block
                buffers[name].clear()
        if dense is not None and dense_buffer:
            block = np.concatenate(dense_buffer, axis=0)
            old = len(dense)
            dense.resize(old+len(block), axis=0)
            dense[old:] = block
            dense_buffer.clear()
        f.flush()
    append(q0, v0, 0., temperature(v0, mass), minimum(q0), 1., 0.)
    if args.dense:
        dense_buffer.append(v0[None])
    flush()
    completed = 0
    failure = None
    prev_left = None
    kernel_seconds = 0.
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(1, total_steps+1):
            tic = time.monotonic()
            # model._kick locally re-enables autograd only for energy gradients.
            with torch.no_grad():
                predicted = model(q, v, times)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            kernel_seconds += time.monotonic()-tic
            qsegment = predicted["positions"][:, 0].detach().cpu().numpy()
            vsegment = predicted["velocities"][:, 0].detach().cpu().numpy()
            if not np.isfinite(qsegment).all() or not np.isfinite(vsegment).all():
                failure = "nonfinite segment"
                f.create_dataset("failure/positions", data=qsegment)
                f.create_dataset("failure/velocities", data=vsegment)
                break
            nq, nv = qsegment[-1], vsegment[-1].copy()
            pre = temperature(nv, mass)
            mind = minimum(nq)
            if abs(np.sum(nv*mass[:, None], axis=0)/mass.sum()-initial_cm).max() > 2e-5:
                failure = "center-of-mass momentum not conserved"
            if not np.isfinite(pre) or pre < 20 or pre > 2000 or mind < .8:
                failure = f"invalid state: T={pre:.6g}K; minimum={mind:.6g} A"
            current_right = qsegment[0]-q[0].detach().cpu().numpy()
            if args.dense:
                left = qsegment[-1]-qsegment[-2]
            else:
                current_right = current_right/16.
                left = (nq-q[0].detach().cpu().numpy())/16.
            join = 0. if prev_left is None else float(
                np.sqrt(np.mean((current_right-prev_left)**2)))
            prev_left = left
            alpha = 1.
            if thermo is not None and failure is None:
                nv, alpha = thermo(nv)
            append(nq, nv, step*16., pre, mind, alpha, join)
            if args.dense:
                vsegment[-1] = nv
                dense_buffer.append(vsegment)
            completed = step
            q = torch.from_numpy(nq.copy()).to(device).unsqueeze(0)
            v = torch.from_numpy(nv.copy()).to(device).unsqueeze(0)
            if step % 256 == 0 or failure:
                flush()
                f.attrs["completed_steps"] = completed
                print(json.dumps(dict(step=step, temperature_K=pre,
                                      elapsed_s=time.monotonic()-start,
                                      minimum_distance_A=mind,
                                      failure=failure)), flush=True)
            if failure:
                break
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        flush()
        elapsed = time.monotonic()-start
        f.attrs.update(completed_steps=completed,
                       stable=(failure is None), failure_reason=failure or "")
        f.close()
        result = dict(completed_steps=completed,
                      production_steps=max(0, completed-args.thermalization_steps),
                      stable=failure is None, failure_reason=failure,
                      simulated_ps=completed*.016,
                      production_ps=max(0, completed-args.thermalization_steps)*.016,
                      requested_production_ps=args.steps*.016,
                      checkpoint_sha256=checkpoint_hash,
                      hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
                      elapsed_s=elapsed, model_seconds=kernel_seconds,
                      ns_day_end_to_end=completed*16e-6/max(elapsed, 1e-9)*86400,
                      dense_queries=args.dense,
                      parameters=sum(x.numel() for x in model.parameters()),
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(device)
                      if device.type=="cuda" else 0)
        args.output.with_suffix(".summary.json").write_text(
            json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
