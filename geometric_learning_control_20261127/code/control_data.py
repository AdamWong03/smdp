"""Shared fixed-state data for the geometric-learning controls.

All controls use the same 64 centered TRAIN states (seed 20261127) and the
same 64 held-out TEST states.  Physical units are A, fs, u.  Velocities and
displacements are mass-centered graph by graph; positions remain absolute.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data import ParityFlowDataset


def _center(value, masses):
    """Mass-center a [B,N,3] tensor graph by graph."""
    m = masses[None, :, None]
    return value - (m * value).sum(1, keepdim=True) / masses.sum()


def load_states(index_dir: str | Path, split: str, limit: int = 64,
                seed: int = 20261127, dtype=torch.float64) -> dict:
    index_dir = Path(index_dir)
    ds = ParityFlowDataset(index_dir, split)
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(len(ds), generator=g)[:limit].tolist()
    with np.load(index_dir / f"{split}.npz", allow_pickle=False) as f:
        q_rel = torch.as_tensor(f["q_rel"][sel], dtype=dtype)
        q_com = torch.as_tensor(f["q_com"][sel], dtype=dtype)
        v_com = torch.as_tensor(f["v_com"][sel], dtype=dtype)
        masses = torch.as_tensor(f["masses"], dtype=dtype)
        qe = torch.as_tensor(f["Q_even"][sel], dtype=dtype)
        qo = torch.as_tensor(f["Q_odd"][sel], dtype=dtype)
        pe = torch.as_tensor(f["P_even"][sel], dtype=dtype)
        po = torch.as_tensor(f["P_odd"][sel], dtype=dtype)
        q0 = torch.as_tensor(f["q0"][sel], dtype=dtype)
        p0 = torch.as_tensor(f["p0"][sel], dtype=dtype)
        cell = torch.as_tensor(f["cell"][sel], dtype=dtype)
        z_idx = torch.as_tensor(f["z_idx"][sel], dtype=torch.long)
        source_path = np.asarray(f["source_path"])[sel]
        source_frame = np.asarray(f["source_frame"], dtype=np.int64)[sel]
        sample_index = np.asarray(f["sample_index"], dtype=np.int64)[sel]
    manifest = json.loads((index_dir / "index_manifest.json").read_text())
    horizon = float(manifest["horizon_fs"])
    species = {int(k): int(v) for k, v in manifest["species"].items()}
    inverse = {v: k for k, v in species.items()}
    atomic_numbers = torch.tensor([inverse[int(v)] for v in z_idx[0]],
                                  dtype=torch.long)
    sqrt_m = torch.sqrt(masses)[None, :, None]
    m = masses[None, :, None]
    q_plus = q_com[:, None, :] + horizon * v_com[:, None, :] + q_rel + \
        (qe + qo) / sqrt_m
    q_minus = q_com[:, None, :] - horizon * v_com[:, None, :] + q_rel + \
        (qe - qo) / sqrt_m
    p_plus = sqrt_m * (pe + po) + m * v_com[:, None, :]
    p_minus = sqrt_m * (po - pe) + m * v_com[:, None, :]
    v0 = _center(p0 / masses[None, :, None], masses)
    v_plus = _center(p_plus / masses[None, :, None], masses)
    v_minus = _center(p_minus / masses[None, :, None], masses)
    disp_plus = _center(q_plus - q0, masses)
    disp_minus = _center(q_minus - q0, masses)
    return dict(
        system=manifest["system"], horizon_fs=horizon,
        cutoff=float(manifest["cutoff_A"]),
        split=split, seed=seed, indices=sel,
        sample_index=sample_index, source_path=source_path,
        source_frame=source_frame,
        q0=q0, v0=v0, cell=cell, masses=masses,
        atomic_numbers=atomic_numbers, z_idx=z_idx,
        q_plus=q0 + disp_plus, q_minus=q0 + disp_minus,
        disp_plus=disp_plus, disp_minus=disp_minus,
        v_plus=v_plus, v_minus=v_minus,
        p0=p0, p_plus=p_plus, p_minus=p_minus,
        Q_even=(q_plus + q_minus) / 2 - q0,
        Q_odd=(q_plus - q_minus) / 2,
        P_even=(p_plus - p_minus) / 2,
        P_odd=(p_plus + p_minus) / 2,
    )


def state_i(states: dict, i: int) -> dict:
    out = {}
    for k, v in states.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == len(states["q0"]):
            out[k] = v[i]
        else:
            out[k] = v
    return out


def trajcast_graph(state: dict, cutoff: float, timestep: float,
                   device="cpu", dtype=torch.float64):
    """Build the official AtomicGraph for one centered state."""
    from trajcast.data.atomic_graph import AtomicGraph
    atoms = dict(
        positions=state["q0"].to(dtype).clone(),
        velocities=state["v0"].to(dtype).clone(),
        cell=state["cell"].to(dtype).clone(),
        pbc=torch.tensor([True, True, True]),
        atomic_numbers=state["atomic_numbers"].to(torch.long).reshape(-1, 1),
        atomic_masses=state["masses"].to(dtype).reshape(-1, 1),
    )
    species = sorted({int(v) for v in state["atomic_numbers"].tolist()})
    mapper = {z: i for i, z in enumerate(species)}
    g = AtomicGraph.from_atoms_dict(atoms, cutoff, mapper)
    g.timestep = torch.tensor([timestep], dtype=dtype)
    return g.to(device)


def relational_graph(state: dict, cutoff: float, timestep: float,
                     device="cpu", dtype=torch.float32):
    """AtomicGraph for R3-Opt / Phase-Space (float32 default)."""
    return trajcast_graph(state, cutoff, timestep, device=device, dtype=dtype)
