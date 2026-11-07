"""Real interior labels, preserving the original source-trajectory split."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch


class InteriorLabels:
    def __init__(self, directory, split, graphs):
        self.path = Path(directory) / f'{split}.h5'
        with h5py.File(self.path) as f:
            if f.attrs['split'] != split:
                raise ValueError('Interior split mismatch')
            if len(f['q0']) != len(graphs):
                raise ValueError('Incomplete interior labels')
            self.provenance = dict(path=str(self.path),
                source_paths=json.loads(f.attrs['source_paths']),
                source_transition=f.attrs['source_transition'],
                position_unit=f.attrs['position_unit'], velocity_unit=f.attrs['velocity_unit'])
            # This cache predates SMDP-R. Only real data are reused, no CMIS code.
            self.times = torch.tensor(f['times_fs'][:], dtype=torch.float32)
            self.q = torch.from_numpy(f['q_target'][:].astype(np.float32))
            self.v = torch.from_numpy(f['v_target'][:].astype(np.float32))
            q0, v0 = f['q0'][:], f['v0'][:]
            z, mass, cells = f['z'][:], f['mass'][:], f['cell'][:]
            self.atoms = len(z)
            for i, graph in enumerate(graphs):
                # Verify original cache alignment for every formal sample.
                # Official graph caches wrap q0; real trajectory labels are
                # unwrapped. Carry each atom's SAME lattice-image offset over
                # the whole segment. Never minimum-image endpoint displacements.
                offset = graph.pos.numpy()-q0[i]
                fractional = offset @ np.linalg.inv(cells[i])
                np.testing.assert_allclose(fractional, np.rint(fractional), rtol=0, atol=2e-6)
                np.testing.assert_allclose(graph.cell.numpy().reshape(3, 3), cells[i], rtol=1e-6)
                np.testing.assert_allclose(graph.velocities.numpy(), v0[i], rtol=0, atol=1e-8)
                np.testing.assert_allclose(graph.displacements.numpy(),
                    self.q[i, -1].numpy()-q0[i], rtol=0, atol=3e-6)
                np.testing.assert_allclose(graph.update_velocities.numpy(),
                    self.v[i, -1].numpy(), rtol=0, atol=1e-8)
                np.testing.assert_array_equal(graph.atomic_numbers.reshape(-1).numpy(), z)
                np.testing.assert_allclose(graph.atomic_masses.reshape(-1).numpy(), mass, rtol=1e-6)
                self.q[i] += torch.from_numpy(offset)[None]
                graph.sample_id = torch.tensor([i], dtype=torch.long)

    def batch(self, batch, indices, device):
        ids = batch.sample_id.detach().cpu()
        # [B,T,N,3] -> [T,B*N,3], in exact PyG batch node order.
        q = self.q[ids][:, indices].permute(1, 0, 2, 3).reshape(len(indices), -1, 3)
        v = self.v[ids][:, indices].permute(1, 0, 2, 3).reshape(len(indices), -1, 3)
        return self.times[indices].to(device), q.to(device), v.to(device)


def query_objective(model, graph, output, labels, indices):
    times, q, v = labels.batch(graph, indices, graph.pos.device)
    predicted = model.query(graph, output, times)
    error_q = (predicted['positions']-q)/model.rms_targets[0].detach()
    error_v = (predicted['velocities']-v)/model.rms_targets[1].detach()
    return .5*(error_q.square().mean()+error_v.square().mean()), predicted


class TwoStepLabels:
    """Two true endpoints from each start's own source, never another future."""
    def __init__(self, transitions, graphs):
        with h5py.File(transitions) as f:
            ids = f['source_trajectory'][:].astype(int)
            frames = f['source_frame'][:].astype(int)
            paths = [p.decode() if isinstance(p, bytes) else str(p) for p in f['source_paths'][:]]
            lag = int(f.attrs['lag_steps'])
            q0 = f['positions_initial_A'][:]
        q = np.empty_like(q0, dtype=np.float32)
        v = np.empty_like(q)
        valid = np.zeros(len(graphs), dtype=bool)
        for source, path in enumerate(paths):
            with h5py.File(path) as f:
                selected = np.flatnonzero((ids==source) & (frames+2*lag<len(f['time_fs'])))
                indices, inverse = np.unique(frames[selected]+2*lag, return_inverse=True)
                q[selected] = f['positions_unwrapped_A'][indices][inverse]
                v[selected] = f['velocities_A_per_fs'][indices][inverse]
                valid[selected] = True
        self.graphs = []
        self.excluded = np.flatnonzero(~valid).tolist()
        self.source_paths = paths
        for i in np.flatnonzero(valid):
            graph = graphs[i]
            offset = graph.pos.numpy()-q0[i]
            graph.two_step_positions = torch.from_numpy(q[i]+offset).float()
            graph.two_step_velocities = torch.from_numpy(v[i]).float()
            self.graphs.append(graph)
