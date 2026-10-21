"""Vesin periodic graphs; cell rows are lattice vectors, q remains unwrapped."""
import numpy as np
import torch
import vesin


class PeriodicGraphBuilder:
    def __init__(self, cutoff):
        self.neighbors = vesin.NeighborList(cutoff=cutoff, full_list=True, sorted=True)

    def one(self, positions, cell):
        q = np.asarray(positions, dtype=np.float64)
        cell = np.asarray(cell, dtype=np.float64)
        images = np.floor(q @ np.linalg.inv(cell))
        wrapped = q - images @ cell
        # Vesin displacement: wrapped[j]-wrapped[i]+S@cell. Sender=j.
        i, j, shift = self.neighbors.compute(wrapped, cell, True, quantities='ijS')
        translation = (shift - images[j] + images[i]) @ cell
        return np.stack([j, i]).astype(np.int64), translation.astype(np.float32)

    def batch(self, q, cells):
        """Rebuild at predicted configurations; only discrete graph choice is detached."""
        edges, shifts, graphs = [], [], []
        offset = 0
        for g, (positions, cell) in enumerate(zip(q, cells, strict=True)):
            e, s = self.one(positions.detach().cpu().numpy(), cell.detach().cpu().numpy())
            edges.append(torch.as_tensor(e + offset, device=q.device))
            shifts.append(torch.as_tensor(s, device=q.device, dtype=q.dtype))
            graphs.append(torch.full((len(positions),), g, dtype=torch.long, device=q.device))
            offset += len(positions)
        return torch.cat(edges, dim=1), torch.cat(shifts), torch.cat(graphs)


def predict_batch(model, q, v, z, masses, cells, times, builder):
    edges, shifts, graphs = builder.batch(q, cells)
    return model(q.flatten(0, 1), v.flatten(0, 1), z.flatten(), masses.flatten(),
                 edges, shifts, times, graph_index=graphs)
