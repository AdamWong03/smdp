"""Same-device cost of the phase-space operator versus the frozen DSRT model.

Times a real TRAIN-batch optimizer iteration (forward + endpoint loss +
backward + AdamW lr=0) and a one-graph autoregressive model call. Graphs are
resident on the device; neighbor-list rebuild and I/O are excluded. This is a
cost measurement, not training and not a physical evaluation.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch
from trajcast.data.dataset import AtomicGraphDataset

from phase_space_transition import PhaseSpaceTransition, train_normalization
from relational_transition import RelationalTransition, endpoint_objective

ROOT = Path('/data1/wyd/research/foundation_mlip_adaptation')
SYSTEMS = {
    'quartz30': dict(
        cache=ROOT/'data/smdp_reference_md_v1/quartz/derived_graphs/trajcast_p64_30fs',
        name='quartz30', mapper={8: 0, 14: 1}, cutoff=4.5, horizon=30.0,
        old_config=Path('configs/quartz30_r3_opt.json'),
        new_config=Path('configs/design/phase_space_quartz30.json')),
    'water5': dict(
        cache=ROOT/'data/smdp_reference_md_v1/water/derived_graphs/water6_reference_20261113',
        name='water5', mapper={1: 0, 8: 1}, cutoff=6.0, horizon=5.0,
        old_config=Path('configs/water6_dsrt_20261113.json'),
        new_config=Path('configs/design/phase_space_water5.json')),
}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def build_pair(system, normalization):
    old_cfg = json.loads(system['old_config'].read_text())
    new_cfg = json.loads(system['new_config'].read_text())
    old = RelationalTransition(old_cfg['model'], normalization['rms_targets'])
    new = PhaseSpaceTransition(new_cfg['model'],
                               rms_targets=normalization['rms_targets'],
                               aggregation_scale=normalization['aggregation_scale'])
    return old, new


def measure(network, graph, device, iterations, warmup, training, proposal_weight):
    network.train(training)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0, amsgrad=True) if training else None
    context = torch.enable_grad if training else torch.inference_mode

    def step():
        if training:
            optimizer.zero_grad(set_to_none=True)
        with context():
            output = network(graph)
            loss = torch.zeros((), device=device)
            if training:
                loss, _ = endpoint_objective(network, output, graph.displacements,
                                             graph.update_velocities,
                                             proposal_weight=proposal_weight)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            optimizer.step()
        return loss

    for _ in range(warmup):
        step()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        loss = step()
        synchronize(device)
        elapsed.append(time.perf_counter()-start)
    if training:
        finite = all(p.grad is not None and bool(p.grad.isfinite().all())
                     for p in network.parameters())
        if not bool(loss.isfinite()) or not finite:
            raise RuntimeError('Non-finite loss or gradient')
    return dict(
        median_seconds=statistics.median(elapsed),
        min_seconds=min(elapsed),
        max_seconds=max(elapsed),
        peak_cuda_bytes=(torch.cuda.max_memory_allocated(device)
                         if device.type == 'cuda' else None),
        iterations=iterations, warmup=warmup, fitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20261120)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(0 if device.index is None else device.index)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    system = SYSTEMS[args.system]
    cache = system['cache']
    file = cache/f"{system['name']}_train;P:64.pt"
    if not file.is_file():
        raise FileNotFoundError(file)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dataset = AtomicGraphDataset(str(cache), f"{system['name']}_train",
                                     system['cutoff'], files=[],
                                     atom_type_mapper=system['mapper'])
    finally:
        torch.set_default_dtype(old_dtype)
    if len(dataset) != 5000:
        raise ValueError('Expected preserved formal TRAIN split')
    normalization = train_normalization(dataset, system['cutoff'])
    graphs = [dataset[i].clone().apply(
        lambda x: x.float() if x.is_floating_point() else x)
        for i in range(args.batch_size)]
    batch = Batch.from_data_list(graphs).to(device)
    single = Batch.from_data_list([graphs[0]]).to(device)
    models = build_pair(system, normalization)
    rows = {}
    for label, model in [('dsrt_frozen', models[0]),
                         ('phase_space_operator_v0', models[1])]:
        model.to(device)
        rows[label] = dict(
            parameters=sum(p.numel() for p in model.parameters()),
            train_batch=measure(model, batch, device, args.iterations, args.warmup,
                                training=True, proposal_weight=0.02),
            single_graph_inference=measure(model, single, device, args.iterations,
                                           args.warmup, training=False,
                                           proposal_weight=0.02),
        )
        print(json.dumps(dict(system=args.system, model=label,
                              **rows[label])), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    reference = rows['dsrt_frozen']['train_batch']['median_seconds']
    for row in rows.values():
        row['train_seconds_ratio_to_dsrt'] = row['train_batch']['median_seconds']/reference
    report = dict(
        purpose=__doc__, device=str(device),
        hardware=(torch.cuda.get_device_name(device)
                  if device.type == 'cuda' else 'CPU'),
        torch=torch.__version__, seed=20261120,
        system=args.system, cutoff_angstrom=system['cutoff'],
        horizon_fs=system['horizon'], batch_size=args.batch_size,
        batch_atoms=len(batch.pos), batch_edges=int(batch.num_edges),
        normalization=normalization, optimizer='AdamW lr=0, amsgrad=True',
        includes_neighbor_rebuild=False, includes_io=False,
        models=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
