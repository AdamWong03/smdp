"""Same-process paired velocity-feedback intervention.

Separate processes are not a clean control: CUDA non-determinism grows in this
chaotic system (two full runs from the same state diverged by ~5 A in 5 ps).
This script runs branches in one process, sharing the frozen backbone and the
initial state:

  full              : v_new feedback (production)
  full_control      : second full branch; measures the numerical chaos floor
  no_feedback       : v_pred feedback; readout history still has v_new
  thermostat_only   : rescaled v_pred; no AR/blend

The full-vs-full_control divergence is the floor.  The feedback effect is the
excess divergence of no_feedback/thermostat_only over that floor.  The branches
share the base model object; only the feedback velocity and readout history
differ.  No training, no algorithm change.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from fd_readout import FDTransition
from feedback_probe import CaptureBase, FeedbackTransition, graph_at, load_initial, load_readout
from motion_loss import decompose_modes
from periodic import PeriodicGraphBuilder
from physics.units import KB_EV_PER_K, KINETIC_TO_EV
from physics.water_motion import motion_partition
from relational_inference import load_relational


def temperature(v, masses):
    return (KINETIC_TO_EV * np.sum(masses[:, None] * v**2)
            / ((3 * len(masses) - 3) * KB_EV_PER_K))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def com_msd(q, masses, molecules):
    n_mol = int(molecules.max()) + 1
    onehot = np.zeros((len(masses), n_mol), dtype=np.float64)
    onehot[np.arange(len(masses)), molecules] = 1.0
    mass = np.bincount(molecules, weights=masses, minlength=n_mol)
    q_com = np.einsum("fak,am->fmk", masses[None, :, None] * q, onehot) / mass[None, :, None]
    # remove global COM
    q_global = np.einsum("m,fmk->fk", mass, q_com) / mass.sum()
    q_com = q_com - q_global[:, None, :]
    n = len(q_com)
    lags = np.arange(1, n)
    msd = np.array([np.mean(np.sum((q_com[lag:] - q_com[:-lag]) ** 2, axis=-1))
                    for lag in lags])
    t_ps = lags * 5.0 / 1000.0
    max_t = float(t_ps[-1])
    mask = (t_ps >= 0.5) & (t_ps <= max_t)
    slope = float(np.polyfit(t_ps[mask], msd[mask], 1)[0]) if mask.sum() > 2 else None
    return dict(msd_0p5ps_A2=float(np.interp(0.5, t_ps, msd)),
                msd_end_A2=float(msd[-1]), msd_end_ps=max_t,
                early_slope_A2_ps=slope,
                early_D_m2_s=(slope / 6.0 * 1e-8 if slope is not None else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--glr-coefficients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026110401)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    base, _ = load_relational(args.checkpoint, device)
    capture = CaptureBase(base)
    builder = PeriodicGraphBuilder(float(base.cfg.cutoff_angstrom))
    q0, v0, cell, z, masses, molecules = load_initial(args.initial, args.initial_frame)

    def make_branch(name, mode, blend):
        readout = load_readout(args.glr_coefficients, float(base.cfg.horizon_fs))
        fd = FDTransition(capture, readout=readout, blend=blend, rescale=True,
                          temperature=300.0).to(device)
        return dict(name=name, mode=mode,
                    model=FeedbackTransition(fd, mode),
                    q=q0.copy(), v=v0.copy(), T=[], qs=[], vs=[])

    branches = [
        make_branch("full", "full", 0.1),
        make_branch("full_control", "full", 0.1),
        make_branch("no_feedback", "no_feedback", 0.1),
        make_branch("thermostat_only", "thermostat_only", 0.0),
    ]
    start = time.perf_counter()
    for step in range(args.steps + 1):
        for branch in branches:
            if step == 0:
                branch["T"].append(temperature(branch["v"], masses))
                branch["qs"].append(branch["q"].copy())
                branch["vs"].append(branch["v"].copy())
                continue
            graph = graph_at(branch["q"], branch["v"], cell, z, masses, molecules,
                             branch["model"], builder, device)
            with torch.inference_mode():
                out = branch["model"](graph)
            branch["q"] = out["positions"].detach().cpu().numpy()
            branch["v"] = out["velocities"].detach().cpu().numpy()
            branch["T"].append(temperature(branch["v"], masses))
            branch["qs"].append(branch["q"].copy())
            branch["vs"].append(branch["v"].copy())
        if step and step % 100 == 0:
            print(json.dumps(dict(step=step, total=args.steps,
                                  elapsed_s=time.perf_counter() - start,
                                  full_T=float(np.mean(branches[0]["T"][-20:])),
                                  nofb_T=float(np.mean(branches[2]["T"][-20:])))),
                  flush=True)

    full = branches[0]
    payload = dict(
        steps=args.steps, checkpoint=str(args.checkpoint), initial=str(args.initial),
        initial_frame=int(args.initial_frame), seed=int(args.seed),
        branches={},
        note=("Same-process paired branches share the frozen backbone and initial state. "
              "full_control is the numerical chaos floor; feedback effect is the excess "
              "divergence of no_feedback/thermostat_only over that floor."),
    )
    for branch in branches:
        qs = np.stack(branch["qs"])
        vs = np.stack(branch["vs"])
        try:
            partition = motion_partition(branch["q"], branch["v"], cell, z, masses, molecules)
            partition_values = {k: float(partition[k]) for k in
                                ("translation_eV", "rotation_eV", "internal_eV",
                                 "total_KE_eV")}
        except AssertionError:
            # motion_partition has a 1e-9 internal consistency assert; a
            # 1e-8 float32 roundoff can trip it.  Fall back to the same mode
            # decomposition used by the GLR readout.
            m = torch.as_tensor(masses, dtype=torch.float64)
            part = decompose_modes(
                torch.as_tensor(branch["q"], dtype=torch.float64),
                torch.as_tensor(branch["v"], dtype=torch.float64), m,
                torch.as_tensor(molecules, dtype=torch.long),
                torch.as_tensor(cell, dtype=torch.float64))
            keys = ("translation", "rotation", "internal")
            vals = {key: float(0.5 * KINETIC_TO_EV
                               * torch.sum(m[:, None] * part[key].double() ** 2))
                    for key in keys}
            partition_values = {
                "translation_eV": vals["translation"],
                "rotation_eV": vals["rotation"],
                "internal_eV": vals["internal"],
                "total_KE_eV": vals["translation"] + vals["rotation"] + vals["internal"],
            }
        entry = dict(
            mode=branch["mode"], T_mean=float(np.mean(branch["T"][1:])),
            T_std=float(np.std(branch["T"][1:])),
            final_partition=partition_values,
            msd=com_msd(qs, masses, molecules),
            final_rms_position_from_full_A=None,
            final_rms_velocity_from_full_A_fs=None,
            max_rms_position_from_full_A=None,
        )
        if branch is not full:
            n = len(qs)
            dq = qs - np.stack(full["qs"])[:n]
            dv = vs - np.stack(full["vs"])[:n]
            entry["final_rms_position_from_full_A"] = rms(dq[-1])
            entry["final_rms_velocity_from_full_A_fs"] = rms(dv[-1])
            entry["max_rms_position_from_full_A"] = float(max(rms(dq[i]) for i in range(n)))
        payload["branches"][branch["name"]] = entry
    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
