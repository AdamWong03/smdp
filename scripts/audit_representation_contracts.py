"""Trace implemented tensor/normalization contracts on TRAIN inputs only.

No optimizer, fitting, model selection, force calls or trajectory generation.
Uses existing frozen weights to distinguish actual computation from class names.
"""
import copy
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('SMDP_RESEARCH_ROOT', '/data1/wyd/research/foundation_mlip_adaptation'))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT/'upstream/trajcast-c52f107'))
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import torch
from torch_geometric.data import Batch
from trajcast.data.dataset import AtomicGraphDataset
from trajcast.model.models import EfficientTrajCastModel
from relational_transition import RelationalTransition, endpoint_loss_components, mass_mean

REPORT = REPO/'reports/implementation_audit_20261119'


def stats(tensor):
    a = tensor.detach().double().reshape(-1)
    return dict(shape=list(tensor.shape), rms=float(a.square().mean().sqrt()),
                minimum=float(a.min()), median=float(a.median()), maximum=float(a.max()))


def count(module):
    return sum(p.numel() for p in module.parameters())


def graph_cast(graph, dtype):
    return graph.clone().apply(lambda t: t.to(dtype) if t.is_floating_point() else t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPORT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    full = ROOT/'experiments/water6_reference_20261113/dsrt_radius6'
    state = torch.load(full/'best.pt', map_location='cpu', weights_only=False)
    cfg = state['config']
    cache = Path(cfg['cache'])
    data = AtomicGraphDataset(root=str(cache), name='water5_train', cutoff_radius=6.,
                             files=[], atom_type_mapper={1: 0, 8: 1})
    ids = np.linspace(0, len(data)-1, 16, dtype=int).tolist()
    graphs = [data[i].clone() for i in ids]
    assert len(data) == 5000
    # Preserve stored FP32 weights exactly when comparing arithmetic precision.
    torch.set_default_dtype(torch.float32)
    model = RelationalTransition(cfg['model']).eval()
    model.load_state_dict(state['model'], strict=True)
    model64 = copy.deepcopy(model).double()
    report = dict(purpose=__doc__, optimizer_updates=0, device='cpu',
                  torch_version=torch.__version__, split='train', indices=ids,
                  checkpoint=str(full/'best.pt'), selected_epoch=state['epoch']+1,
                  model_parameters=count(model), parameter_groups={
                      name: count(module) for name, module in model.named_children()},
                  rms_targets=model.rms_targets.tolist(), config=cfg,
                  cache=str(cache), normalization_contract='absolute endpoint velocity; physical displacement',
                  precision_comparison='Same stored FP32 weights; includes input casting and forward arithmetic, not retraining.',
                  samples=[])
    del state
    traces = []

    def before(name):
        def hook(module, args):
            s, v, e, edges, relation = args[:5]
            traces.append(dict(block=name, stage='input', scalar=stats(s), vector=stats(v),
                edge=stats(e), relation_scalars=stats(relation[0]),
                relation_vectors=stats(relation[1]), geometry=stats(module.geometry_features(v, edges, relation[1]))))
        return hook

    def after(name):
        def hook(module, args, output):
            traces.append(dict(block=name, stage='output', scalar=stats(output[0]),
                               vector=stats(output[1]), edge=stats(output[2])))
        return hook

    handles = []
    for name, block in [(f'proposal_{i}', b) for i, b in enumerate(model.proposal_blocks)] + [
            (f'refinement_{i}', b) for i, b in enumerate(model.refinement_blocks)]:
        handles.extend([block.register_forward_pre_hook(before(name)),
                        block.register_forward_hook(after(name))])
    all_degrees, all_weighted_degrees = [], []
    with torch.no_grad():
        for index, original in zip(ids, graphs):
            graph = graph_cast(original, torch.float32)
            double_graph = graph_cast(original, torch.float64)
            out = model(graph)
            if handles:
                for handle in handles:
                    handle.remove()
                handles.clear()
            out64 = model64(double_graph)
            receiver, sender = graph.edge_index
            d = graph.pos[sender]-graph.pos[receiver]+graph.shifts
            r = d.norm(dim=-1)
            weight = 0.5*(1+torch.cos(torch.pi*r/6.))
            degree = torch.bincount(receiver, minlength=len(graph.pos))
            weighted = torch.zeros(len(graph.pos)).index_add(0, receiver, weight)
            all_degrees.append(degree); all_weighted_degrees.append(weighted)
            batch = torch.zeros(len(graph.pos), dtype=torch.long)
            masses = graph.atomic_masses.flatten()
            vrel = graph.velocities-mass_mean(graph.velocities, masses, batch)
            qres = graph.displacements-5*graph.velocities
            vres = graph.update_velocities-graph.velocities
            terms = endpoint_loss_components(model, out, graph.displacements, graph.update_velocities)
            concatenated = torch.cat([(out['displacements']-graph.displacements)/model.rms_targets[0],
                                      (out['velocities']-graph.update_velocities)/model.rms_targets[1]], -1)
            assert torch.allclose(terms['final'], concatenated.square().mean(), rtol=1e-6, atol=1e-9)
            sample = dict(index=index, atoms=len(graph.pos), edges=len(receiver),
                normalized_loss_identity_abs_error=float((terms['final']-concatenated.square().mean()).abs()),
                degree=stats(degree), weighted_degree=stats(weighted),
                true_displacement=stats(graph.displacements), true_velocity=stats(graph.update_velocities),
                ballistic_position_residual=stats(qres), identity_velocity_residual=stats(vres),
                final_position_error=stats(out['displacements']-graph.displacements),
                final_velocity_error=stats(out['velocities']-graph.update_velocities),
                proposal_position_error=stats(out['proposal_displacements']-graph.displacements),
                proposal_velocity_error=stats(out['proposal_velocities']-graph.update_velocities),
                correction_position=stats(out['correction_displacements']),
                correction_velocity=stats(out['correction_velocities']),
                fp32_vs_fp64_same_weights={key:stats(out[key].double()-out64[key])
                    for key in ['displacements', 'velocities']}, species={})
            for z in [1, 8]:
                mask=graph.atomic_numbers.flatten()==z
                sample['species'][str(z)] = dict(speed_A_fs=stats(vrel[mask].norm(dim=-1)),
                    encoded_speed_squared=stats((vrel[mask]/.01).square().sum(-1)))
            report['samples'].append(sample)
    report['dsrt_first_graph_trace'] = traces
    report['degree_over_16_graphs'] = stats(torch.cat(all_degrees))
    report['weighted_degree_over_16_graphs'] = stats(torch.cat(all_weighted_degrees))

    # One real TRAIN forward/backward records actual clipping demand, not a step.
    model.zero_grad(set_to_none=True)
    batch=Batch.from_data_list([graph_cast(g, torch.float32) for g in graphs[:8]])
    out=model(batch)
    pieces=endpoint_loss_components(model, out, batch.displacements, batch.update_velocities)
    loss=pieces['final']+.02*pieces['proposal']
    loss.backward()
    gradients={name:float(sum((p.grad.double().square().sum() for p in module.parameters()
                             if p.grad is not None), torch.tensor(0.,dtype=torch.float64)).sqrt())
               for name,module in model.named_children()}
    report['single_train_batch_gradient'] = dict(batch_size=8, objective=float(loss.detach()),
        parameter_gradient_norms=gradients, all_finite=all(p.grad is None or p.grad.isfinite().all().item()
            for p in model.parameters()), update_applied=False)
    model.zero_grad(set_to_none=True)

    torch.set_default_dtype(torch.float64)
    native_path=ROOT/'experiments/water6_reference_20261113/trajcast_native/best.pt'
    saved=torch.load(native_path, map_location='cpu', weights_only=False)
    native=EfficientTrajCastModel(config=saved['resolved_native_model_config'],
        predicted_fields=['displacements','update_velocities'], rms_targets=saved['rms_targets']).eval()
    native.load_state_dict(saved['model'], strict=True)
    native_trace=[]
    def native_hook(name):
        def hook(module, args, output):
            field=module.output_field if hasattr(module,'output_field') else 'target'
            if field not in output:
                field='node_features'
            native_trace.append(dict(module=name, field=field, values=stats(output[field]),
                irreps=str(module.irreps_out[field]) if hasattr(module,'irreps_out') and field in module.irreps_out else None))
        return hook
    hs=[module.register_forward_hook(native_hook(name)) for name,module in native._int_out.named_children()]
    with torch.no_grad():
        original=graphs[0].clone(); native_out=native(original.clone())
        qlabel=native_out.displacements*native.rms_targets[0]
        vlabel=native_out.update_velocities*native.rms_targets[1]
        torch.testing.assert_close(qlabel,original.displacements,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(vlabel,original.update_velocities,rtol=1e-12,atol=1e-12)
    for h in hs:h.remove()
    native_groups={name:count(m) for name,m in native.named_children()}
    layer_groups={name:{part:count(m) for part,m in layer.named_children()}
                  for name,layer in native._int_out.named_children() if name.startswith('mp_layer')}
    report['native'] = dict(checkpoint=str(native_path), selected_epoch=saved['epoch'],
        protocol_completed=False, parameters=count(native), rms_targets=native.rms_targets.tolist(),
        avg_num_neighbors=float(native.avg_num_neighbors), group_parameters=native_groups,
        per_layer_parameters=layer_groups, first_graph_trace=native_trace,
        label_roundtrip_max_abs=dict(position=float((qlabel-original.displacements).abs().max()),
                                    velocity=float((vlabel-original.update_velocities).abs().max())),
        label_roundtrip_assertion_tolerance=dict(rtol=1e-12, atol=1e-12),
        velocity_target='absolute endpoint velocity',
        graph_fields_mutated_in_forward=['displacements', 'update_velocities'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'representation_contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(args.output/'representation_contracts.json'),
        train_samples=len(ids),optimizer_updates=0,dsrt_parameters=report['model_parameters'],
        native_parameters=report['native']['parameters'],gradient_finite=report['single_train_batch_gradient']['all_finite'],
        weighted_degree=report['weighted_degree_over_16_graphs'],native_trace=native_trace),indent=2))


if __name__ == '__main__':
    main()
