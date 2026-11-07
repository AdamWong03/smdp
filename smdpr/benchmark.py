"""Paired inference measurements, not a training/accuracy experiment."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

from relational_transition import RelationalTransition
from smdpr.model import SMDPR
from train_relational import read_graphs
from periodic import PeriodicGraphBuilder
from relational_inference import load_relational, vesin_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--supercell', type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = RelationalTransition(cfg['model']).cuda().eval()
    model = SMDPR(cfg['model'], **cfg['motion']).cuda().eval()
    if args.checkpoint:
        model, _ = load_relational(args.checkpoint)
    graphs = read_graphs(Path(cfg['cache']), 'val', cfg)
    graph = Batch.from_data_list([graphs[0]]).cuda()
    if args.supercell>1:
        import itertools
        source = graphs[0]
        n = args.supercell
        cell = source.cell.reshape(3, 3)
        shifts = torch.tensor(list(itertools.product(range(n), repeat=3)), dtype=torch.float32)@cell
        q = torch.cat([source.pos+shift for shift in shifts])
        graph = vesin_graph(q, source.velocities.repeat(n**3, 1), cell*n,
            source.atomic_numbers.repeat(n**3, 1), source.atomic_masses.repeat(n**3, 1),
            model, PeriodicGraphBuilder(model.cfg.cutoff_angstrom), 'cuda')
    def measure(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(args.iterations//5):
                fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)/(args.iterations//5))
        return dict(mean_seconds=sum(samples)/len(samples), block_seconds=samples)
    with torch.inference_mode():
        original = measure(lambda: baseline(graph))
        endpoint = measure(lambda: model(graph))
        times = torch.arange(1, int(cfg['model']['horizon_fs'])+1, device='cuda')
        dense = measure(lambda: model.query(graph, model(graph, True), times))
    ratio = endpoint['mean_seconds']/original['mean_seconds']
    report = dict(hardware=torch.cuda.get_device_name(), dtype='float32; TF32 disabled',
        measurement='shared-device paired latency; graph prebuilt',
        checkpoint=str(args.checkpoint) if args.checkpoint else 'random initialization',
        supercell=args.supercell,
        scaling_only='replicas for runtime, not independent physical generalization',
        validation_graph_atoms=len(graph.pos), directed_edges=graph.edge_index.shape[1],
        baseline_parameters=sum(p.numel() for p in baseline.parameters()),
        smdpr_parameters=sum(p.numel() for p in model.parameters()),
        baseline=original, smdpr_endpoint=endpoint, smdpr_dense=dense,
        endpoint_overhead_ratio=ratio, latency_target_passed=ratio<=1.5,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        dense_queries=len(times), includes_neighbor_rebuild=False,
        includes_io=False, includes_thermostat=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
