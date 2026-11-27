"""Velocity-feedback intervention for the frozen R3-Opt + GLR loop.

The production loop is closed:

    q_t, v_t -> backbone -> dq_backbone, v_pred
             -> GLR readout -> v_new (blend + rescale)
             -> q_{t+1} = q_t + dq_backbone
             -> v_{t+1} = v_new
             -> next backbone sees q_{t+1}, v_{t+1}

This script isolates the feedback of v_new to the next backbone without changing
the production GLR algorithm:

  full              : v_new fed back (production)
  no_feedback       : v_pred fed back; readout history still contains v_new
  all_no_feedback   : v_pred fed back and readout history reset to v_pred
  thermostat_only   : v_pred rescaled to the target KE, no AR/blend
  raw               : frozen backbone only (no GLR readout/rescale)

`--probe` performs a one-step intervention at the supplied state:
compare the next backbone displacement with v_pred vs v_new.

No training, no new kernel, no new method.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from glr_readout import GLRReadout
from fd_readout import FDTransition
from motion_loss import decompose_modes
from periodic import PeriodicGraphBuilder
from physics.units import KB_EV_PER_K, KINETIC_TO_EV
from relational_inference import load_relational, vesin_graph


def temperature(velocities, masses):
    return (KINETIC_TO_EV * np.sum(masses[:, None] * velocities**2)
            / ((3 * len(masses) - 3) * KB_EV_PER_K))


class CaptureBase(nn.Module):
    """Wrap the frozen backbone and expose its velocity prediction."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rms_targets = base.rms_targets

    def forward(self, data, return_context=False):
        output = self.base(data, return_context=True)
        output["backbone_velocity"] = output["velocities"]
        return output

    def validate_graph(self, data):
        return self.base.validate_graph(data) if hasattr(self.base, "validate_graph") else None

    def set_diagnostics(self, enabled):
        if hasattr(self.base, "set_diagnostics"):
            self.base.set_diagnostics(enabled)

    def diagnostics(self):
        return self.base.diagnostics() if hasattr(self.base, "diagnostics") else []


class RescaleOnlyTransition(nn.Module):
    """Fair no-AR-memory baseline: frozen backbone + per-mode canonical rescale.

    Uses the same mode basis, mode degrees of freedom, temperature target and
    COM projection as the deterministic GLR, but no AR coefficients, no ACF
    calibration and no readout history.
    """

    def __init__(self, base, temperature=300.0, stochastic_rescale=False):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rms_targets = base.rms_targets
        self.temperature = float(temperature)
        self.stochastic_rescale = bool(stochastic_rescale)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def reset(self):
        return None

    def forward(self, data, return_context=False):
        output = self.base(data, return_context=True)
        v_pred = output["velocities"]
        dq = output["displacements"]
        masses = data.atomic_masses.reshape(-1)
        molecules = data.molecule_ids.reshape(-1)
        part = decompose_modes(data.pos, v_pred, masses, molecules, data.cell)
        n_mol = int(molecules.max()) + 1
        n_atoms = len(masses)
        dofs = (3 * n_mol - 3, 3 * n_mol, 3 * n_atoms - 6 * n_mol)
        keys = ("translation", "rotation", "internal")
        rescaled = torch.zeros_like(v_pred)
        for index, key in enumerate(keys):
            norm = 0.5 * KINETIC_TO_EV * torch.sum(
                masses[:, None] * part[key].square())
            if self.stochastic_rescale:
                concentration = torch.as_tensor(
                    0.5 * dofs[index], dtype=norm.dtype, device=norm.device)
                rate = torch.as_tensor(
                    1.0 / (KB_EV_PER_K * self.temperature),
                    dtype=norm.dtype, device=norm.device)
                target = torch.distributions.Gamma(concentration, rate).sample()
            else:
                target = 0.5 * dofs[index] * KB_EV_PER_K * self.temperature
            scale = torch.sqrt(torch.clamp(target / norm.clamp_min(1e-12), max=1e6))
            rescaled = rescaled + scale * part[key]
        com_v = (masses[:, None] * rescaled).sum(0) / masses.sum()
        rescaled = rescaled - com_v
        com_dq = (masses[:, None] * dq).sum(0) / masses.sum()
        dq = dq - com_dq
        output["positions"] = data.pos + dq
        output["velocities"] = rescaled
        output["displacements"] = dq
        output["rescale_only"] = True
        if return_context:
            output["context"] = output.get("context", {})
        return output

    def validate_graph(self, data):
        return self.base.validate_graph(data) if hasattr(self.base, "validate_graph") else None

    def set_diagnostics(self, enabled):
        if hasattr(self.base, "set_diagnostics"):
            self.base.set_diagnostics(enabled)

    def diagnostics(self):
        return self.base.diagnostics() if hasattr(self.base, "diagnostics") else []


class FeedbackTransition(nn.Module):
    def __init__(self, fd_transition, mode):
        super().__init__()
        self.fd_transition = fd_transition
        self.cfg = fd_transition.cfg
        self.rms_targets = fd_transition.rms_targets
        self.mode = mode

    def reset(self):
        self.fd_transition.reset()

    def validate_graph(self, data):
        return (self.fd_transition.validate_graph(data)
                if hasattr(self.fd_transition, "validate_graph") else None)

    def forward(self, data, return_context=False):
        output = self.fd_transition(data, return_context=True)
        v_pred = output["backbone_velocity"]
        v_new = output["velocities"]
        if self.mode in ("no_feedback", "all_no_feedback"):
            output["velocities"] = v_pred
            if self.mode == "all_no_feedback":
                readout = self.fd_transition.fd
                readout.reset()
                readout.update(data.pos, v_pred, data.atomic_masses,
                               data.molecule_ids, data.cell)
        output["feedback_mode"] = self.mode
        return output

    def set_diagnostics(self, enabled):
        if hasattr(self.fd_transition, "set_diagnostics"):
            self.fd_transition.set_diagnostics(enabled)

    def diagnostics(self):
        return (self.fd_transition.diagnostics()
                if hasattr(self.fd_transition, "diagnostics") else [])


def load_readout(path, horizon_fs):
    data = json.loads(Path(path).read_text())
    keys = ("translation", "rotation", "internal")
    return GLRReadout(
        [data["modes"][key]["coefficients"] for key in keys],
        [data["modes"][key]["prediction_error_ratio"] for key in keys],
        temperature=300.0,
        use_noise=False,
        slow_tau=0.0,
        slow_weight=0.0,
        horizon_fs=horizon_fs,
    )


def build_model(checkpoint, coefficients, device, mode):
    base, _ = load_relational(Path(checkpoint), device)
    if mode == "raw":
        return base
    if mode == "rescale_only":
        return RescaleOnlyTransition(base).to(device)
    if mode == "stochastic_rescale_only":
        return RescaleOnlyTransition(base, stochastic_rescale=True).to(device)
    capture = CaptureBase(base)
    readout = load_readout(coefficients, float(base.cfg.horizon_fs))
    blend = 0.0 if mode == "thermostat_only" else 0.1
    fd = FDTransition(capture, readout=readout, blend=blend, rescale=True,
                      temperature=300.0).to(device)
    return FeedbackTransition(fd, mode)


def load_initial(path, frame):
    with h5py.File(path, "r") as source:
        pos_key = ("positions_unwrapped_A" if "positions_unwrapped_A" in source
                   else "positions")
        vel_key = ("velocities_A_per_fs" if "velocities_A_per_fs" in source
                   else "velocities")
        cell_key = "cell_A" if "cell_A" in source else "cell"
        q = source[pos_key][frame].astype(np.float32)
        v = source[vel_key][frame].astype(np.float32)
        cell = source[cell_key][frame] if source[cell_key].ndim == 3 else source[cell_key][:]
        z = source["atomic_numbers"][:]
        masses = source["masses_u"][:]
        molecules = source["molecule_ids"][:]
    return q, v, cell, z, masses, molecules


def graph_at(q, v, cell, z, masses, molecules, model, builder, device):
    graph = vesin_graph(q, v, cell, z, masses, model, builder, device)
    graph.molecule_ids = torch.as_tensor(molecules, dtype=torch.long,
                                         device=device).reshape(-1)
    return graph


def probe(model, q, v, cell, z, masses, molecules, builder, device):
    graph = graph_at(q, v, cell, z, masses, molecules, model, builder, device)
    with torch.inference_mode():
        if hasattr(model, "fd_transition"):
            base_transition = model.fd_transition
            base0 = base_transition.base(graph)
            fd0 = base_transition(graph)
        else:
            base_transition = model
            base0 = model(graph)
            fd0 = base0
        v_pred0 = base0["velocities"]
        dq0 = base0["displacements"]
        v_new0 = fd0["velocities"]
        v_fd0 = fd0.get("fd_velocity", v_new0)
        graph_new = graph_at(q, v_new0.cpu().numpy(), cell, z, masses, molecules,
                             model, builder, device)
        base1 = base_transition.base(graph_new) if hasattr(base_transition, "base") else base_transition(graph_new)
        v_pred1 = base1["velocities"]
        dq1 = base1["displacements"]
    def rms(x):
        return float(torch.sqrt(torch.mean(x.square())))
    correction = (v_new0 - v_pred0).detach()
    delta_dq = (dq1 - dq0).detach()
    result = dict(
        rms_v_pred_A_fs=rms(v_pred0),
        rms_v_new_A_fs=rms(v_new0),
        rms_correction_A_fs=rms(correction),
        correction_over_v_pred=float(rms(correction) / rms(v_pred0)),
        rms_dq0_A=rms(dq0),
        rms_dq_change_A=rms(delta_dq),
        dq_change_over_dq0=float(rms(delta_dq) / rms(dq0)),
        rms_v_pred_change_A_fs=rms(v_pred1 - v_pred0),
        v_pred_change_over_v_pred=float(rms(v_pred1 - v_pred0) / rms(v_pred0)),
        correction_cosine=float(
            torch.dot(correction.flatten(), v_pred0.flatten())
            / (torch.linalg.vector_norm(correction)
               * torch.linalg.vector_norm(v_pred0) + 1e-30)),
    )
    return result


def rollout(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = build_model(args.checkpoint, args.glr_coefficients, device, args.mode)
    builder = PeriodicGraphBuilder(float(model.cfg.cutoff_angstrom))
    near = PeriodicGraphBuilder(3.0)
    q, v, cell, z, masses, molecules = load_initial(args.initial, args.initial_frame)
    steps = int(args.steps)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completed = 0
    failure = None
    with h5py.File(output_path, "x") as output:
        output.attrs.update(
            ensemble="nvt", timestep_fs=float(model.cfg.horizon_fs),
            requested_steps=steps, completed_steps=0, stable=False,
            failure_reason="", checkpoint=str(args.checkpoint),
            initial_trajectory=str(args.initial), initial_frame=int(args.initial_frame),
            seed=int(args.seed), feedback_mode=args.mode,
            execution_note=args.execution_note,
            reference_forces_used_during_propagation=False,
        )
        output["atomic_numbers"] = z
        output["masses_u"] = masses
        output["molecule_ids"] = molecules
        output["cell"] = cell
        output["time_fs"] = np.arange(steps + 1) * float(model.cfg.horizon_fs)
        positions = output.create_dataset("positions", (steps + 1, len(z), 3),
                                          dtype="f4")
        velocities = output.create_dataset("velocities", (steps + 1, len(z), 3),
                                           dtype="f4")
        v_pred = output.create_dataset("backbone_velocity", (steps, len(z), 3),
                                       dtype="f4")
        v_fd = output.create_dataset("readout_velocity", (steps, len(z), 3),
                                     dtype="f4")
        dq = output.create_dataset("backbone_displacement", (steps, len(z), 3),
                                   dtype="f4")
        temperature_ds = output.create_dataset("temperature_K", (steps + 1,),
                                               dtype="f8")
        minimum_ds = output.create_dataset("minimum_distance_A", (steps + 1,),
                                           dtype="f8")
        positions[0], velocities[0] = q, v
        temperature_ds[0] = temperature(v, masses)
        minimum_ds[0] = 0.0
        try:
            for index in range(1, steps + 1):
                graph = graph_at(q, v, cell, z, masses, molecules, model,
                                 builder, device)
                with torch.inference_mode():
                    out = model(graph)
                qn = out["positions"].detach().cpu().numpy()
                vn = out["velocities"].detach().cpu().numpy()
                if "backbone_velocity" in out:
                    v_pred[index - 1] = out["backbone_velocity"].detach().cpu().numpy()
                    v_fd[index - 1] = out["fd_velocity"].detach().cpu().numpy()
                    dq[index - 1] = out["displacements"].detach().cpu().numpy()
                else:
                    v_pred[index - 1] = vn
                    v_fd[index - 1] = vn
                    dq[index - 1] = (qn - q)
                q, v = qn, vn
                positions[index], velocities[index] = q, v
                temperature_ds[index] = temperature(v, masses)
                edges, shifts = near.one(q, cell)
                minimum = float(np.linalg.norm(q[edges[0]] + shifts - q[edges[1]],
                                               axis=1).min())
                minimum_ds[index] = minimum
                if not np.isfinite(q).all() or not np.isfinite(v).all():
                    raise FloatingPointError("nonfinite autonomous state")
                if minimum < 0.65 or temperature_ds[index] > 3000 or temperature_ds[index] < 1:
                    raise RuntimeError(
                        f"physical safety stop: min={minimum}, T={temperature_ds[index]}")
                completed = index
                output.attrs["completed_steps"] = completed
                if index % 1000 == 0:
                    output.flush()
                    print(json.dumps(dict(mode=args.mode, step=index, total=steps,
                                          T=float(temperature_ds[index]),
                                          elapsed_s=time.perf_counter() - start)),
                          flush=True)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        output.attrs.update(stable=failure is None, failure_reason=failure or "",
                            completed_steps=completed)
    # temperature_ds belongs to the closed HDF5 file; re-open for the summary.
    with h5py.File(output_path, "r") as handle:
        T_values = np.asarray(handle["temperature_K"][1:completed + 1])
    summary = dict(
        mode=args.mode, completed_steps=completed, requested_steps=steps,
        simulated_ps=completed * float(model.cfg.horizon_fs) / 1000.0,
        stable=failure is None, failure_reason=failure,
        elapsed_seconds=time.perf_counter() - start,
        checkpoint=str(args.checkpoint), initial_trajectory=str(args.initial),
        initial_frame=int(args.initial_frame), seed=int(args.seed),
        T_mean=float(np.mean(T_values)) if completed else None,
        T_std=float(np.std(T_values)) if completed else None,
        execution_note=args.execution_note,
    )
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if failure:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--glr-coefficients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "no_feedback", "all_no_feedback",
                                           "thermostat_only", "rescale_only",
                                           "stochastic_rescale_only", "raw"],
                        default="full")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026110401)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-note", default="velocity feedback intervention")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-output", type=Path)
    args = parser.parse_args()
    if args.probe:
        device = torch.device(args.device)
        model = build_model(args.checkpoint, args.glr_coefficients, device, "full")
        builder = PeriodicGraphBuilder(float(model.cfg.cutoff_angstrom))
        q, v, cell, z, masses, molecules = load_initial(args.initial, args.initial_frame)
        result = probe(model, q, v, cell, z, masses, molecules, builder, device)
        result.update(checkpoint=str(args.checkpoint), initial=str(args.initial),
                      initial_frame=int(args.initial_frame))
        print(json.dumps(result, indent=2), flush=True)
        if args.probe_output:
            args.probe_output.write_text(json.dumps(result, indent=2) + "\n")
        return
    rollout(args)


if __name__ == "__main__":
    main()
