"""Inference helpers for autonomous fixed-horizon relational propagation."""
from pathlib import Path

import torch
from trajcast.data.atomic_graph import AtomicGraph
from torch_geometric.data import Data

from periodic import PeriodicGraphBuilder
from relational_transition import RelationalTransition


def load_relational(checkpoint, device="cuda"):
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    model = RelationalTransition(saved["config"]["model"])
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    return model, saved


def atomic_graph(positions, velocities, cell, atomic_numbers, masses, model, device):
    graph = AtomicGraph.from_atoms_dict(
        dict(positions=torch.as_tensor(positions, dtype=torch.float32),
             velocities=torch.as_tensor(velocities, dtype=torch.float32),
             cell=torch.as_tensor(cell, dtype=torch.float32),
             pbc=torch.tensor([True, True, True]),
             atomic_numbers=torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1, 1),
             atomic_masses=torch.as_tensor(masses, dtype=torch.float32).reshape(-1, 1)),
        model.cfg.cutoff_angstrom, {8: 0, 14: 1})
    graph.timestep = torch.tensor([model.cfg.horizon_fs], dtype=torch.float32)
    return graph.to(device)


def vesin_graph(positions, velocities, cell, atomic_numbers, masses, model, builder, device):
    """Equivalent model input using the production Vesin periodic list."""
    edges, shifts = builder.one(positions, cell)  # [sender, receiver]
    graph = Data(pos=torch.as_tensor(positions, dtype=torch.float32),
                 velocities=torch.as_tensor(velocities, dtype=torch.float32),
                 cell=torch.as_tensor(cell, dtype=torch.float32).reshape(1, 3, 3),
                 pbc=torch.tensor([[True, True, True]]),
                 atomic_numbers=torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1, 1),
                 atomic_masses=torch.as_tensor(masses, dtype=torch.float32).reshape(-1, 1),
                 # RelationalTransition normalizes AtomicGraph [central,neighbor]
                 # into [sender,receiver] internally.
                 edge_index=torch.as_tensor(edges[::-1].copy(), dtype=torch.long),
                 shifts=torch.as_tensor(shifts, dtype=torch.float32),
                 timestep=torch.tensor([model.cfg.horizon_fs], dtype=torch.float32))
    return graph.to(device)


@torch.inference_mode()
def step(model, positions, velocities, cell, atomic_numbers, masses, device="cuda", builder=None):
    graph = (atomic_graph(positions, velocities, cell, atomic_numbers, masses, model, device)
             if builder is None else
             vesin_graph(positions, velocities, cell, atomic_numbers, masses, model, builder, device))
    output = model(graph)
    return output["positions"], output["velocities"]
