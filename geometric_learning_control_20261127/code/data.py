"""In-memory centered-window dataset for ParityFlow.

The index builder stores one centered graph per macrostep (at the central
configuration).  Training batches are disjoint unions of those graphs; edge
indices are offset by the cumulative node count.  No graph is rebuilt here and
no future coordinate enters the central graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


TARGET_KEYS = ("Q_even", "Q_odd", "P_even", "P_odd")


class ParityFlowDataset:
    def __init__(self, index_dir: str | Path, split: str):
        index_dir = Path(index_dir)
        path = index_dir / f"{split}.npz"
        with np.load(path, allow_pickle=False) as f:
            self.q_rel = np.asarray(f["q_rel"], dtype=np.float32)
            self.p_tilde = np.asarray(f["p_tilde"], dtype=np.float32)
            self.q0 = np.asarray(f["q0"], dtype=np.float32)
            self.p0 = np.asarray(f["p0"], dtype=np.float32)
            self.q_com = np.asarray(f["q_com"], dtype=np.float32)
            self.v_com = np.asarray(f["v_com"], dtype=np.float32)
            self.targets = {
                k: np.asarray(f[k], dtype=np.float32) for k in TARGET_KEYS
            }
            self.masses = np.asarray(f["masses"], dtype=np.float32)
            self.z_idx = np.asarray(f["z_idx"], dtype=np.int64)
            self.molecule_ids = np.asarray(f["molecule_ids"], dtype=np.int64)
            self.cell = np.asarray(f["cell"], dtype=np.float64)
            self.edge_index = np.asarray(f["edge_index"], dtype=np.int32)
            self.edge_vec = np.asarray(f["edge_vec"], dtype=np.float32)
            self.edge_offsets = np.asarray(f["edge_offsets"], dtype=np.int64)
            self.source_path = np.asarray(f["source_path"])
            self.source_trajectory = np.asarray(f["source_trajectory"],
                                                dtype=np.int64)
            self.source_frame = np.asarray(f["source_frame"], dtype=np.int64)
            self.sample_index = np.asarray(f["sample_index"], dtype=np.int64)
        with open(index_dir / "normalization.json", "r") as f:
            self.norm = json.load(f)
        self.split = split
        self.num_nodes = int(self.q_rel.shape[1])

    def __len__(self) -> int:
        return int(self.q_rel.shape[0])

    def __getitem__(self, i: int) -> dict:
        e0, e1 = int(self.edge_offsets[i]), int(self.edge_offsets[i + 1])
        return dict(
            q_rel=torch.from_numpy(self.q_rel[i]),
            p_tilde=torch.from_numpy(self.p_tilde[i]),
            q0=torch.from_numpy(self.q0[i]),
            p0=torch.from_numpy(self.p0[i]),
            q_com=torch.from_numpy(self.q_com[i]).expand(self.num_nodes, 3),
            v_com=torch.from_numpy(self.v_com[i]).expand(self.num_nodes, 3),
            targets={k: torch.from_numpy(self.targets[k][i])
                     for k in TARGET_KEYS},
            masses=torch.from_numpy(self.masses),
            z_idx=torch.from_numpy(self.z_idx[i]),
            cell=torch.from_numpy(self.cell[i]),
            edge_index=torch.from_numpy(
                self.edge_index[:, e0:e1].astype(np.int64, copy=False)),
            edge_vec=torch.from_numpy(self.edge_vec[e0:e1]),
            index=i,
            source_frame=int(self.source_frame[i]),
        )

    def state(self, i: int) -> dict:
        """Absolute central state plus analytic center-of-mass quantities."""
        return dict(
            q0=self.q0[i].astype(np.float64, copy=True),
            p0=self.p0[i].astype(np.float64, copy=True),
            q_com=self.q_com[i].astype(np.float64, copy=True),
            v_com=self.v_com[i].astype(np.float64, copy=True),
            masses=self.masses.astype(np.float64, copy=True),
            cell=self.cell[i].astype(np.float64, copy=True),
            z_idx=self.z_idx[i].astype(np.int64, copy=True),
        )


def collate(samples: list[dict]) -> dict:
    """Disjoint-union batch.  Every graph is independent; no cross edges."""
    q = torch.cat([s["q_rel"] for s in samples], dim=0)
    p = torch.cat([s["p_tilde"] for s in samples], dim=0)
    masses = torch.cat([s["masses"] for s in samples], dim=0)
    z = torch.cat([s["z_idx"] for s in samples], dim=0)
    edge_vec = torch.cat([s["edge_vec"] for s in samples], dim=0)
    targets = {k: torch.cat([s["targets"][k] for s in samples], dim=0)
               for k in TARGET_KEYS}

    edge_index = []
    offset = 0
    for s in samples:
        edge_index.append(s["edge_index"] + offset)
        offset += s["q_rel"].shape[0]
    edge_index = torch.cat(edge_index, dim=1)

    return dict(
        q_rel=q,
        p_tilde=p,
        masses=masses,
        z_idx=z,
        edge_vec=edge_vec,
        edge_index=edge_index,
        num_nodes=offset,
        targets=targets,
        q0=torch.cat([s["q0"] for s in samples], dim=0),
        p0=torch.cat([s["p0"] for s in samples], dim=0),
        q_com=torch.cat([s["q_com"] for s in samples], dim=0),
        v_com=torch.cat([s["v_com"] for s in samples], dim=0),
        sample_indices=torch.tensor([s["index"] for s in samples],
                                    dtype=torch.long),
    )


def load_norm(index_dir: str | Path) -> dict:
    with open(Path(index_dir) / "normalization.json", "r") as f:
        return json.load(f)
