"""Measure real training-graph compute without fitting or loading test data.

Times forward + endpoint loss + backward + AdamW step (lr=0: no fitting).
Graphs are resident on device; this is NOT end-to-end training throughput.
"""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch
from trajcast.data.dataset import AtomicGraphDataset

from relational_transition import RelationalTransition, endpoint_objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[2, 8])
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--precision', choices=['fp32', 'fp64', 'bf16'], default='fp32')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(20261102)
    device = torch.device(args.device)
    dtype = torch.float64 if args.precision == 'fp64' else torch.float32
    # Upstream includes default dtype in its cache filename. Load the existing
    # P:64 cache explicitly, then cast batches; never regenerate training data.
    if not (args.cache / 'quartz30_train;P:64.pt').is_file():
        raise FileNotFoundError('Existing Quartz P:64 training cache required')
    torch.set_default_dtype(torch.float64)
    ds = AtomicGraphDataset(root=str(args.cache), name='quartz30_train',
                            cutoff_radius=4.5, files=[], atom_type_mapper={8:0,14:1})
    model = RelationalTransition().to(device=device, dtype=dtype)
    rows = []
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def context():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=args.precision == 'bf16')
    for size in args.batch_sizes:
        g = Batch.from_data_list([ds[i] for i in range(size)]).to(device)
        g = g.apply(lambda x: x.to(dtype) if x.is_floating_point() else x)
        for factored in [False, True]:
            network = copy.deepcopy(model)
            for module in network.modules():
                if hasattr(module, 'factorized_edges'):
                    module.factorized_edges = factored
            optimizer = torch.optim.AdamW(network.parameters(), lr=0)
            def iteration():
                optimizer.zero_grad(set_to_none=True)
                with context():
                    output = network(g)
                    loss, _ = endpoint_objective(network, output, g.displacements, g.update_velocities)
                loss.backward()
                optimizer.step()
                return loss
            for _ in range(3):
                loss = iteration()
            if not bool(loss.isfinite()) or not all(
                    p.grad is not None and bool(p.grad.isfinite().all()) for p in network.parameters()):
                raise RuntimeError('Non-finite loss/gradient')
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            elapsed = []
            for _ in range(args.iterations):
                sync()
                start = time.perf_counter()
                iteration()
                sync()
                elapsed.append(time.perf_counter()-start)
            row = dict(batch_size=size, atoms=len(g.pos), edges=g.num_edges,
                       factorized=factored, median_seconds=statistics.median(elapsed),
                       samples_per_second=size/statistics.median(elapsed),
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del optimizer, network
    report = dict(device=str(device), precision=args.precision,
                  hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                  threads=torch.get_num_threads(), parameters=sum(p.numel() for p in model.parameters()),
                  fitted=False, data='training graphs only', includes_io=False,
                  optimizer='AdamW lr=0', results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
