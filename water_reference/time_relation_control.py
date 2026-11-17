"""Paired endpoint timing on one shared GPU; no fitting or trajectory claims."""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import h5py
import torch

from periodic import PeriodicGraphBuilder
from relational_inference import load_relational, vesin_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {name: load_relational(path)[0] for name, path in
              [('full', args.full), ('current_only', args.control)]}
    assert models['full'].cfg.relation_conditioning == 'predicted_future'
    assert models['current_only'].cfg.relation_conditioning == 'current_only'
    source = Path('/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1'
                  '/water/raw_nve/val_04/trajectory.h5')
    with h5py.File(source) as f:
        graph = vesin_graph(f['positions_unwrapped_A'][0], f['velocities_A_per_fs'][0],
            f['cell_A'][0], f['atomic_numbers'][:], f['masses_u'][:], models['full'],
            PeriodicGraphBuilder(6.), 'cuda')
    results = {key: [] for key in models}
    with torch.inference_mode():
        for model in models.values():
            for _ in range(20):
                model(graph)
        for repeat in range(4):
            order = list(models) if repeat % 2 == 0 else list(reversed(models))
            for name in order:
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(100):
                    models[name](graph)
                torch.cuda.synchronize()
                seconds = (time.perf_counter()-start)/100
                results[name].append(dict(repeat=repeat, seconds_per_endpoint=seconds,
                    incremental_peak_bytes=torch.cuda.max_memory_allocated()-resident))
    report = dict(hardware=torch.cuda.get_device_name(), torch_version=torch.__version__,
        source=str(source), source_frame=0, atoms=len(graph.pos), directed_edges=graph.num_edges,
        batch_size=1, precision='FP32 / TF32 off', graph_prebuilt=True,
        neighbor_update_thermostat_and_io_excluded=True, endpoint_only=True,
        shared_device=True, optimizer_updates=0, full_checkpoint=str(args.full),
        current_only_checkpoint=str(args.control), results=results,
        gpu_at_timing_end=subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used',
                                         '--format=csv'], capture_output=True, text=True).stdout)
    report['median_seconds_per_endpoint'] = {
        name: statistics.median(row['seconds_per_endpoint'] for row in rows)
        for name, rows in results.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
