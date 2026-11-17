"""Check the one-path control on real TRAIN graphs, without optimizer steps."""
import argparse
import copy
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
from torch_geometric.data import Batch

from periodic import PeriodicGraphBuilder
from relational_inference import vesin_graph
from relational_transition import RelationalTransition, endpoint_objective
from train_relational import read_graphs
from water_reference.train_native import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.run/'config.json').read_text())
    original = json.loads((args.baseline_run/'config.json').read_text())
    expected = copy.deepcopy(original)
    expected['model']['relation_conditioning'] = 'current_only'
    assert cfg == expected, 'The conditioning selector must be the only config change'
    metadata = json.loads((args.baseline_run/'metadata.json').read_text())
    source = subprocess.run(['git', 'show', 'b3b2f76:relational_transition.py'],
                            check=True, capture_output=True, text=True).stdout
    old = types.ModuleType('frozen_full_relational_transition')
    sys.modules[old.__name__] = old
    exec(compile(source, 'b3b2f76:relational_transition.py', 'exec'), old.__dict__)

    # GPU index_add reductions may otherwise differ at ~1e-8 between identical
    # forwards. Determinism is local to this correctness process, not training.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graphs = read_graphs(Path(cfg['cache']), 'train', cfg)
    scales = [torch.cat([g[field] for g in graphs]).square().mean().sqrt().item()
              for field in ['displacements', 'update_velocities']]
    assert scales == metadata['normalization_train_only']
    models, states = [], []
    for cls, config in [(old.RelationalTransition, original['model']),
                         (RelationalTransition, original['model']),
                         (RelationalTransition, cfg['model'])]:
        torch.manual_seed(cfg['seed'])
        models.append(cls(config, scales))
        states.append(torch.get_rng_state())
    for model, rng in zip(models[1:], states[1:]):
        assert torch.equal(states[0], rng)
        assert models[0].state_dict().keys() == model.state_dict().keys()
        for name, value in models[0].state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
    legacy, full, control = [m.cuda() for m in models]
    batch = Batch.from_data_list(graphs[:cfg['batch_size']]).cuda()
    parameters = sum(p.numel() for p in control.parameters())
    assert parameters == metadata['parameters'] == 2357616
    keys = ['displacements', 'velocities', 'proposal_displacements', 'proposal_velocities']
    with torch.no_grad():
        a, b = legacy(batch), full(batch)
        for key in keys:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        def current_relations(_module, inputs):
            d0, u0, _, _ = inputs
            return d0, u0, d0, u0
        handle = full.relations.register_forward_pre_hook(current_relations)
        b, c = full(batch), control(batch)
        handle.remove()
        for key in keys:
            torch.testing.assert_close(b[key], c[key], rtol=0, atol=0)
        for key in ['proposal_displacements', 'proposal_velocities']:
            torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)

    torch.cuda.reset_peak_memory_stats()
    out = control(batch)
    loss, _ = endpoint_objective(control, out, batch.displacements,
                                 batch.update_velocities, proposal_weight=0)
    loss.backward()
    assert loss.isfinite()
    assert all(p.grad is not None and p.grad.isfinite().all()
               for p in control.parameters())
    head_gradients = {name: sum(float(p.grad.square().sum()) for p in head.parameters())**.5
                      for name, head in [('proposal', control.proposal_head),
                                          ('correction', control.correction_head)]}
    assert all(value > 0 for value in head_gradients.values())
    peak = torch.cuda.max_memory_allocated()
    control.zero_grad(set_to_none=True)
    with torch.no_grad():
        g = graphs[0].cuda()
        cached = control(g)
        online = control(vesin_graph(g.pos, g.velocities, g.cell, g.atomic_numbers,
                         g.atomic_masses, control, PeriodicGraphBuilder(6.), 'cuda'))
        online_errors = {key: float((cached[key]-online[key]).abs().max())
                         for key in ['displacements', 'velocities']}
        assert max(online_errors.values()) < 2e-7, online_errors
        saved = torch.load(args.baseline_run/'best.pt', map_location='cpu', weights_only=False)
        legacy.load_state_dict(saved['model'], strict=True)
        full.load_state_dict(saved['model'], strict=True)
        a, b = legacy(batch), full(batch)
        compatibility = {key: float((a[key]-b[key]).abs().max()) for key in keys}
        assert all(value == 0 for value in compatibility.values()), compatibility
    report = dict(accepted=True, optimizer_updates=0, real_training_samples=len(graphs),
        effective_batch=cfg['batch_size'], parameters=parameters, hardware=torch.cuda.get_device_name(),
        torch_version=torch.__version__, normalization_train_only=scales,
        configuration_change={'model.relation_conditioning': ['predicted_future', 'current_only']},
        deterministic_correctness_process=True, training_determinism_changed=False,
        identical_seeded_parameters_and_rng=True, legacy_random_forward_exact=True,
        control_equals_current_geometry_intervention=True, finite_final_loss_gradients=True,
        auxiliary_loss_used_in_gradient_check=False, head_gradient_norms=head_gradients,
        cached_online_max_difference=online_errors, peak_check_cuda_bytes=peak,
        legacy_selected_checkpoint_forward_difference=compatibility,
        baseline_commit='b3b2f76', baseline_run=str(args.baseline_run),
        checkpoint_loaded_only_for_compatibility_not_training=True)
    write_json(args.run/'correctness.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
