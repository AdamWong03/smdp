"""Trace the new operator on existing TRAIN graphs, without fitting a model.

All train graphs determine frozen normalization statistics; eight fixed graphs
per system check the full-size FP32 forward/backward and record tensor shapes.
No optimizer, test/validation access, MD labels or trajectory generation.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get("SMDP_RESEARCH_ROOT", "/data1/wyd/research/foundation_mlip_adaptation"))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT/"upstream/trajcast-c52f107"))

import numpy as np
import torch
from e3nn import __version__ as e3nn_version
from torch_geometric.data import Batch
from trajcast.data.dataset import AtomicGraphDataset

from phase_space_transition import PhaseSpaceTransition, train_normalization
from relational_transition import endpoint_loss_components


def train_statistics(dataset, cutoff):
    stats = train_normalization(dataset, cutoff)
    stats['normalization_source'] = stats.pop('source')
    return stats


def count(module):
    return sum(p.numel() for p in module.parameters())


def trace(name, dataset, config, statistics):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20261120)
    model = PhaseSpaceTransition(config, rms_targets=statistics['rms_targets'],
                                aggregation_scale=statistics['aggregation_scale'])
    ids = np.linspace(0, len(dataset)-1, 8, dtype=int).tolist()
    batch = Batch.from_data_list([dataset[i].clone().apply(
        lambda x: x.float() if x.is_floating_point() else x) for i in ids])
    model.validate_graph(batch)
    activations, hooks = [], []
    def hook(label):
        def capture(_module, inputs, outputs):
            state, edges = outputs
            row = dict(block=label, node_shape=list(state.shape), edge_shape=list(edges.shape),
                node_rms=float(state.detach().square().mean().sqrt()),
                edge_rms=float(edges.detach().square().mean().sqrt()))
            activations.append(row)
        return capture
    for stage, blocks in [('proposal', model.proposal_blocks), ('refinement', model.refinement_blocks)]:
        hooks += [block.register_forward_hook(hook(f'{stage}_{i}')) for i, block in enumerate(blocks)]
    start = time.perf_counter()
    out = model(batch)
    forward_seconds = time.perf_counter()-start
    for handle in hooks:
        handle.remove()
    pieces = endpoint_loss_components(model, out, batch.displacements, batch.update_velocities)
    objective = pieces['final']+.02*pieces['proposal']
    start = time.perf_counter()
    objective.backward()
    backward_seconds = time.perf_counter()-start
    finite = all(p.grad is not None and p.grad.isfinite().all().item() for p in model.parameters())
    if not finite or not torch.isfinite(objective):
        raise RuntimeError(f'Non-finite full-size TRAIN contract for {name}')
    first_edges = dataset[ids[0]].num_edges
    c, m, e, g, r = (model.cfg.node_channels, model.cfg.message_channels,
        model.cfg.edge_channels, model.cfg.geometry_channels, model.relations.scalar_size)
    tensor_widths = dict(geometry_descriptor=18*g, edge_extras=e+r+18*g,
                        transport_gates=5*m, largest_order_message=5*m,
                        node_representation=9*c, message_representation=9*m)
    path_list = [dict(input_orders=[i.i_in1, i.i_in2], output_order=i.i_out,
                      mode=i.connection_mode) for i in model.proposal_blocks[0].coupling.product.instructions]
    return dict(system=name, config=model.config, parameters=count(model),
        parameter_groups={key: count(value) for key, value in model.named_children()},
        block_parameter_groups={key: count(value) for key, value in model.proposal_blocks[0].named_children()},
        cg_paths=path_list, train_statistics=statistics, train_indices=ids,
        batch_atoms=len(batch.pos), batch_edges=batch.num_edges, batch_size=8,
        all_parameter_gradients_finite=finite, initialized_from_pretrained=False,
        optimizer_updates=0, random_initialization_objective=float(objective.detach()),
        cpu_forward_seconds=forward_seconds, cpu_backward_seconds=backward_seconds,
        timing_scope="one cold CPU pass, 2 threads; correctness trace, not a performance benchmark",
        activations=activations, tensor_widths=tensor_widths,
        single_first_graph_fp32_tensor_MiB={key:first_edges*width*4/2**20
            for key, width in tensor_widths.items() if key not in {'node_representation', 'message_representation'}},
        memory_scope="single tensor sizes, not measured peak memory or sum of concurrent allocations")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    rows = []
    definitions = [
        ('quartz30', ROOT/'data/smdp_reference_md_v1/quartz/derived_graphs/trajcast_p64_30fs',
         {8: 0, 14: 1}, dict(cutoff_angstrom=4.5, horizon_fs=30.)),
        ('water5', ROOT/'data/smdp_reference_md_v1/water/derived_graphs/water6_reference_20261113',
         {1: 0, 8: 1}, dict(cutoff_angstrom=6., horizon_fs=5.))]
    for name, cache, mapper, config in definitions:
        file = cache/f'{name}_train;P:64.pt'
        if not file.is_file():
            raise FileNotFoundError(file)
        torch.set_default_dtype(torch.float64)
        dataset = AtomicGraphDataset(str(cache), f'{name}_train', config['cutoff_angstrom'],
                                     files=[], atom_type_mapper=mapper)
        if len(dataset) != 5000:
            raise ValueError('Expected preserved formal train split')
        statistics = train_statistics(dataset, config['cutoff_angstrom'])
        result = trace(name, dataset, config, statistics)
        result['cache'] = str(file)
        rows.append(result)
        print(json.dumps(dict(system=name, parameters=result['parameters'],
            finite=result['all_parameter_gradients_finite'], train_statistics=statistics)), flush=True)
        del dataset
        gc.collect()
    report = dict(purpose=__doc__, torch=torch.__version__, e3nn=e3nn_version,
                  device='cpu', threads=2, seed=20261120, optimizer_updates=0,
                  trained=False, physical_performance_evaluated=False, systems=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
