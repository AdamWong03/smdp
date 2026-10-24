"""Fit the spectral anchor from TRAIN trajectories ONLY, never from test data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from periodic import PeriodicGraphBuilder
from reference_data import QuartzSegments, validate_split_manifest
from sacmp.model import SACMP, SACMPConfig


def prepare_training_anchor(data: QuartzSegments, cfg: SACMPConfig):
    q = data.q.numpy().astype(np.float64, copy=False)
    v = data.v.numpy().astype(np.float64, copy=False)
    mass = data.m.numpy().astype(np.float64)
    z = data.z.numpy()
    cells = data.cells.numpy().astype(np.float64)
    n = len(mass)
    if cfg.rank > 3*n-3:
        raise ValueError("rank exceeds translationally reduced phase space")
    if float(np.max(np.abs(cells-cells[0]))) > 1e-4:
        raise ValueError("Fixed-reference SACMP requires a common crystal cell")
    # Remove ONLY the rigid translation difference between training samples.
    anchor = q[0, 0]
    shifts = ((q[:, 0]-anchor)*mass[None, :, None]).sum(axis=1)
    shifts /= mass.sum()
    aligned = q[:, 0]-shifts[:, None, :]
    reference = aligned.mean(axis=0)
    registration_rms = float(np.sqrt(np.mean((aligned-reference)**2)))
    if not np.isfinite(registration_rms) or registration_rms > .35:
        raise ValueError(
            f"Reference atom registration failed ({registration_rms:.4f} A): "
            "check PBC image or atom ordering instead of training")
    root = np.sqrt(mass)[None, :, None]
    masses = mass[None, :, None]
    def remove_com(velocity):
        com = (velocity*masses).sum(axis=1, keepdims=True)/mass.sum()
        return ((velocity-com)*root).reshape(len(velocity), -1)
    # Centered-difference 1/fs accelerations: no BKS oracle and no Hessian.
    velocity = remove_com(v[:, 1])
    # t=2fs and t=0fs are separated by exactly 2 fs.
    acceleration = remove_com((v[:, 2]-v[:, 0])/2.0)
    acc_centered = acceleration-acceleration.mean(0, keepdims=True)
    cov = acc_centered.T@acc_centered/len(acc_centered)
    eigenvalue, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigenvalue)[::-1][:cfg.rank]
    W0 = eigvec[:, order].copy()
    # Resolve the arbitrary SVD sign once; QR inside the model has a
    # consistent sign convention, so modal frequency channels remain fixed.
    imax = np.abs(W0).argmax(axis=0)
    W0 *= np.where(W0[imax, np.arange(cfg.rank)] >= 0, 1, -1)[None, :]
    # Correct small numerical COM projection error.
    vec = W0.reshape(n, 3, cfg.rank)
    cm = (vec*np.sqrt(mass)[:, None, None]).sum(0)/np.sqrt(mass.sum())
    vec -= (np.sqrt(mass/mass.sum())[:, None, None]*cm[None])
    W0, _ = np.linalg.qr(vec.reshape(3*n, cfg.rank))
    vmodal = velocity@W0
    amodal = acceleration@W0
    omega = np.sqrt(
        (amodal**2).mean(0)/np.maximum((vmodal**2).mean(0), 1e-12))
    omega = np.clip(omega, .003, .35)
    fast_capture = float(np.sum((acceleration@W0)**2) /
                         max(np.sum(acceleration**2), 1e-20))
    velocity_capture = float(np.sum((velocity@W0)**2) /
                             max(np.sum(velocity**2), 1e-20))
    builder = PeriodicGraphBuilder(cfg.cutoff_a)
    edges, shifts = builder.one(reference, cells[0])
    if not np.isfinite(omega).all():
        raise ValueError("Nonfinite initial modal frequencies")
    bootstrap = dict(
        config=cfg.__dict__,
        reference=torch.tensor(reference, dtype=torch.float32),
        mass=torch.tensor(mass, dtype=torch.float32),
        atomic_numbers=torch.tensor(z, dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_shift=torch.tensor(shifts, dtype=torch.float32),
        initial_basis=torch.tensor(W0.copy(), dtype=torch.float32),
        initial_omega=torch.tensor(omega, dtype=torch.float32)
    )
    metrics = dict(
        training_samples=len(data), n_atoms=n, rank=cfg.rank,
        registration_RMS_A=registration_rms,
        captured_mass_weighted_acceleration_variance=fast_capture,
        captured_mass_weighted_velocity_variance=velocity_capture,
        initial_omega_rad_fs=dict(min=float(omega.min()),
                                  median=float(np.median(omega)),
                                  max=float(omega.max())),
        common_cell=np.asarray(cells[0]).tolist(),
        graph_edges=int(edges.shape[1]),
        train_scaling=data.scales)
    return bootstrap, metrics


def make_model(bootstrap, device="cpu"):
    cfg = SACMPConfig(**bootstrap["config"])
    model = SACMP(
        cfg, bootstrap["reference"], bootstrap["mass"],
        bootstrap["atomic_numbers"], bootstrap["edge_index"],
        bootstrap["edge_shift"], bootstrap["initial_basis"],
        bootstrap["initial_omega"])
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    validate_split_manifest(args.data_root)
    train = QuartzSegments(args.data_root, "train", args.output/"cache")
    cfg = SACMPConfig(rank=args.rank)
    anchor, report = prepare_training_anchor(train, cfg)
    torch.save(anchor, args.output/"anchor_train_only.pt")
    (args.output/"anchor_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
