"""Finite real-data FP32 gradients and cached/online graph agreement at 6 Å."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from periodic import PeriodicGraphBuilder
from relational_inference import vesin_graph
from relational_transition import RelationalTransition, endpoint_objective
from train_relational import read_graphs
from water_reference.train_native import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads((args.run/'config.json').read_text())
    torch.manual_seed(cfg['seed'])
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    graphs = read_graphs(Path(cfg['cache']), 'train', cfg)
    rms = [torch.cat([g[field] for g in graphs]).square().mean().sqrt().item()
           for field in ['displacements', 'update_velocities']]
    model = RelationalTransition(cfg['model'], rms).cuda()
    batch = next(iter(DataLoader(graphs[:8], batch_size=8))).cuda()
    torch.cuda.reset_peak_memory_stats()
    out = model(batch)
    loss, _ = endpoint_objective(model, out, batch.displacements, batch.update_velocities,
                                 proposal_weight=cfg['proposal_weight'])
    loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated()
    g = graphs[0].cuda()
    with torch.no_grad():
        cached = model(g)
        online = model(vesin_graph(g.pos, g.velocities, g.cell,
            g.atomic_numbers, g.atomic_masses, model, PeriodicGraphBuilder(6.), 'cuda'))
        errors = {key: (cached[key]-online[key]).abs().max().item()
                  for key in ['displacements', 'velocities']}
        assert max(errors.values()) < 2e-7, errors
    # Only cutoff/cache may differ from the historical R3-Opt config.
    original = json.loads(Path('configs/water5_r3_opt.json').read_text())
    expected = json.loads(json.dumps(original))
    expected['cache'] = cfg['cache']
    expected['model']['cutoff_angstrom'] = 6.
    assert cfg == expected
    write_json(args.run/'correctness.json', dict(real_training_samples=len(graphs),
        effective_batch=8, finite_gradients=True, cached_online_max_difference=errors,
        model_changes=['cutoff_angstrom: 4.5 -> 6.0'],
        normalization_train_only=rms, peak_forward_backward_cuda_bytes=peak,
        optimizer_updates=0))
    print((args.run/'correctness.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
