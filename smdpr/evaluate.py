"""Fixed-checkpoint endpoint, real dense-query and causal mode-removal metrics."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from relational_inference import load_relational
from smdpr.model import SMDPR
from smdpr.train import write_json
from train_relational import read_graphs, evaluate


def dense_reference(path, graphs, horizon):
    """Actual 1 fs reference states aligned with each wrapped initial graph."""
    times = np.arange(int(horizon)+1)
    q = np.empty((len(graphs), len(times), len(graphs[0].pos), 3), dtype=np.float32)
    v = np.empty_like(q)
    with h5py.File(path) as f:
        sources = f['source_trajectory'][:].astype(int)
        starts = f['source_frame'][:].astype(int)
        dt = float(f.attrs['source_dt_fs'])
        paths = [s.decode() if isinstance(s, bytes) else str(s) for s in f['source_paths'][:]]
        original = f['positions_initial_A'][:]
    offsets = times/dt
    if not np.allclose(offsets, np.rint(offsets)):
        raise ValueError('No true 1 fs frame in this source')
    for source, name in enumerate(paths):
        local = np.flatnonzero(sources==source)
        indices = starts[local, None]+np.rint(offsets).astype(int)[None]
        with h5py.File(name) as f:
            if indices.max()>=len(f['time_fs']):
                raise ValueError('Missing dense future frame')
            low, high = int(indices.min()), int(indices.max())+1
            # Sequential HDF5 read avoids expensive thousands-index selections.
            q[local] = f['positions_unwrapped_A'][low:high][indices-low]
            v[local] = f['velocities_A_per_fs'][low:high][indices-low]
    for i, g in enumerate(graphs):
        q[i] += g.pos.numpy()-original[i]
        g.sample_id = torch.tensor([i])
    return times, torch.from_numpy(q), torch.from_numpy(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--mode-analysis', action='store_true')
    args = parser.parse_args()
    output = args.run/'evaluation'
    output.mkdir(exist_ok=True)
    dest = output/f'{args.split}_endpoint.json'
    model, saved = load_relational(args.run/'best.pt')
    cfg = saved['config']
    graphs = read_graphs(Path(cfg['cache']), args.split, cfg)
    loader = DataLoader(graphs, batch_size=8, shuffle=False)
    result = evaluate(model, loader, torch.device('cuda'))
    result.update(checkpoint=str(args.run/'best.pt'), selected_epoch=saved['epoch']+1,
                  selected_optimizer_steps=saved['optimizer_steps'], split=args.split)
    if not dest.exists():
        write_json(dest, result)
    if not isinstance(model, SMDPR):
        return
    data = Path('/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1')/cfg['system']
    transitions = data/'transitions'/f'{int(model.cfg.horizon_fs)}fs'/f'{args.split}.h5'
    times, rq, rv = dense_reference(transitions, graphs, model.cfg.horizon_fs)
    loader = DataLoader(graphs, batch_size=8, shuffle=False)
    qsum = np.zeros(len(times))
    vsum = np.zeros(len(times))
    qabs = np.zeros(len(times))
    vabs = np.zeros(len(times))
    kinematic = reference_kinematic = 0.
    modes = [dict(count=0, mean=None, cross=None, products=None, attention=None,
                  within_sum=None, between_sum=None, within_count=0, between_count=0)
             for _ in model.modal_blocks]
    removals = np.zeros((model.mode_count, 4))
    molecule_ids = None
    if cfg['system']=='water':
        with h5py.File(data/'raw_nve/val_04/trajectory.h5') as f:
            molecule_ids = torch.tensor(f['molecule_ids'][:], device='cuda')
    total_components = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.cuda()
            for block in model.modal_blocks:
                block.capture = args.mode_analysis
            full = model(batch, True)
            if args.mode_analysis:
                for aggregate, block in zip(modes, model.modal_blocks):
                    r = block.last_record
                    a = r['affinity'].double()
                    values = [a.sum(0), a.T@a, r['attention'].double().sum(0)]
                    for key, value in zip(['mean', 'products', 'attention'], values):
                        aggregate[key] = value.clone() if aggregate[key] is None else aggregate[key]+value
                    aggregate['count'] += len(a)
                    aggregate['cross'] = aggregate.get('cross') or 0
                    aggregate['cross'] += len(r['attention'])
                    if molecule_ids is not None:
                        src, dst = r['edges']
                        atom_mol = molecule_ids.repeat(batch.num_graphs)+batch.batch*len(molecule_ids)
                        within = atom_mol[src]==atom_mol[dst]
                        for label, mask in [('within', within), ('between', ~within)]:
                            value = a[mask].sum(0)
                            aggregate[label+'_sum'] = value.clone() if aggregate[label+'_sum'] is None else aggregate[label+'_sum']+value
                            aggregate[label+'_count'] += int(mask.sum())
                    block.capture = False
                    block.last_record = None
                for k in range(model.mode_count):
                    if model.mode_count==1:
                        model.modes_enabled = False
                    else:
                        active = [True]*model.mode_count
                        active[k] = False
                        model.set_active_modes(active)
                    removed = model(batch)
                    removals[k] += [float((removed['displacements']-full['displacements']).square().sum()),
                        float((removed['velocities']-full['velocities']).square().sum()),
                        float((removed['displacements']-batch.displacements).square().sum()),
                        float((removed['velocities']-batch.update_velocities).square().sum())]
                    model.modes_enabled = True
                    model.set_active_modes([True]*model.mode_count)
            queried = model.query(batch, full, times)
            ids = batch.sample_id.cpu()
            qt = rq[ids].permute(1, 0, 2, 3).reshape(len(times), -1, 3).cuda()
            vt = rv[ids].permute(1, 0, 2, 3).reshape(len(times), -1, 3).cuda()
            dq, dv = queried['positions']-qt, queried['velocities']-vt
            qsum += dq.square().sum((1, 2)).cpu().numpy()
            vsum += dv.square().sum((1, 2)).cpu().numpy()
            qabs += dq.abs().sum((1, 2)).cpu().numpy()
            vabs += dv.abs().sum((1, 2)).cpu().numpy()
            # Compare identical 1 fs trapezoidal-difference conventions.
            pred_error = torch.diff(queried['positions'], dim=0)-.5*(queried['velocities'][1:]+queried['velocities'][:-1])
            ref_error = torch.diff(qt, dim=0)-.5*(vt[1:]+vt[:-1])
            kinematic += float(pred_error.square().sum())
            reference_kinematic += float(ref_error.square().sum())
            total_components += len(batch.pos)*3
    query_result = dict(split=args.split, times_fs=times.tolist(),
        position_RMSE_A=np.sqrt(qsum/total_components).tolist(),
        velocity_RMSE_A_fs=np.sqrt(vsum/total_components).tolist(),
        position_MAE_A=(qabs/total_components).tolist(), velocity_MAE_A_fs=(vabs/total_components).tolist(),
        kinematic_RMS_A_fs=float(np.sqrt(kinematic/(total_components*(len(times)-1)))),
        reference_kinematic_RMS_A_fs=float(np.sqrt(reference_kinematic/(total_components*(len(times)-1)))),
        kinematic_convention='1 fs position difference minus average endpoint velocities',
        genuine_1fs_reference=True, query_loss_updates_backbone=False)
    path = output/f'{args.split}_query.json'
    if not path.exists():
        write_json(path, query_result)
    if args.mode_analysis:
        summary = []
        for item in modes:
            mean = item['mean']/item['count']
            second = item['products']/item['count']
            variance = (torch.diag(second)-mean.square()).clamp_min(1e-12)
            covariance = second-mean[:, None]*mean[None]
            record = dict(affinity_mean=mean.tolist(),
                affinity_correlation=(covariance/(variance[:, None]*variance[None]).sqrt()).tolist(),
                pairwise_affinity_RMS=(torch.diag(second)[:, None]+torch.diag(second)[None]-2*second).clamp_min(0).sqrt().tolist(),
                cross_mode_attention=(item['attention']/item['cross']).tolist())
            if molecule_ids is not None:
                record.update(intramolecular_affinity=(item['within_sum']/item['within_count']).tolist(),
                              intermolecular_affinity=(item['between_sum']/item['between_count']).tolist())
            summary.append(record)
        path = output/f'{args.split}_modes.json'
        if not path.exists():
            write_json(path, dict(blocks=summary,
                mode_removal=[dict(mode=k, displacement_change_RMS_A=float(np.sqrt(x[0]/total_components)),
                    velocity_change_RMS_A_fs=float(np.sqrt(x[1]/total_components)),
                    position_RMSE_A=float(np.sqrt(x[2]/total_components)),
                    velocity_RMSE_A_fs=float(np.sqrt(x[3]/total_components))) for k, x in enumerate(removals)],
                interpretation='affinity/correlation alone does not establish physical mode identities'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
