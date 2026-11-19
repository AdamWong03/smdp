"""Attribute phase-space operator device time to ATen/GEMM categories.

Profiler instrumentation inflates absolute time; use the ratios only after
comparing with scripts/benchmark_phase_space.py. No fitting or test data.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch_geometric.data import Batch
from trajcast.data.dataset import AtomicGraphDataset

from phase_space_transition import PhaseSpaceTransition, train_normalization
from relational_transition import endpoint_objective

ROOT = Path('/data1/wyd/research/foundation_mlip_adaptation')
SYSTEMS = {
    'quartz30': dict(cache=ROOT/'data/smdp_reference_md_v1/quartz/derived_graphs/trajcast_p64_30fs',
                     name='quartz30', mapper={8: 0, 14: 1}, cutoff=4.5,
                     config=Path('configs/design/phase_space_quartz30.json')),
    'water5': dict(cache=ROOT/'data/smdp_reference_md_v1/water/derived_graphs/water6_reference_20261113',
                   name='water5', mapper={1: 0, 8: 1}, cutoff=6.0,
                   config=Path('configs/design/phase_space_water5.json')),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for a device-time profile')
    system = SYSTEMS[args.system]
    torch.set_default_dtype(torch.float64)
    dataset = AtomicGraphDataset(str(system['cache']), f"{system['name']}_train",
                                 system['cutoff'], files=[],
                                 atom_type_mapper=system['mapper'])
    torch.set_default_dtype(torch.float32)
    normalization = train_normalization(dataset, system['cutoff'])
    graphs = [dataset[i].clone().apply(
        lambda x: x.float() if x.is_floating_point() else x)
        for i in range(args.batch_size)]
    del dataset
    batch = Batch.from_data_list(graphs).cuda()
    cfg = json.loads(system['config'].read_text())['model']
    model = PhaseSpaceTransition(cfg, rms_targets=normalization['rms_targets'],
                                 aggregation_scale=normalization['aggregation_scale']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0, amsgrad=True)

    def step():
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, _ = endpoint_objective(model, output, batch.displacements,
                                     batch.update_velocities)
        loss.backward()
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step()
    torch.cuda.synchronize()
    rows = [(event.key, event.self_device_time_total)
            for event in prof.key_averages() if event.self_device_time_total > 0]
    rows.sort(key=lambda row: -row[1])
    total = sum(row[1] for row in rows)
    report = dict(
        purpose=__doc__, system=args.system, batch_size=args.batch_size,
        steps=args.steps, device=torch.cuda.get_device_name(),
        total_self_device_seconds=total/1e6,
        steps_per_second_inflated=args.steps/(total/1e6),
        note='profiler instrumentation inflates time; compare ratios only',
        top=[dict(operator=key, self_device_seconds=value/1e6,
                  fraction=value/total) for key, value in rows[:20]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['top'][:8], indent=2))


if __name__ == '__main__':
    main()
