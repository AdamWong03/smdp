"""Official graph caches plus true source-indexed 1 fs segments; no new MD."""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import h5py
import torch
import yaml
from torch_geometric.data import InMemoryDataset, Batch
from trajcast.data.dataset import AtomicGraphDataset
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.utils.atomic_computes import wrap_positions_back_to_box_torch
from trajcast_based.acquire import ROOT, RUN as FOUNDATION, COMMIT, sha
from trajcast_based.adapt_data import build
from trajcast_smdp.model import hermite,bubble_basis

DATA=ROOT/'data/smdp_reference_md_v1'
RUN=ROOT/'experiments/trajcast_smdp/quartz30_20261028'


def dataset(run,split):
    return AtomicGraphDataset(root=str(Path(run)/'cache'),name=f'smdp30_{split}',
        cutoff_radius=4.5,files=[],atom_type_mapper={8:0,14:1})


def data_config(run,split):
    return dict(root=str(Path(run)/'cache'),name=f'smdp30_{split}',
                cutoff_radius=4.5,files=[],atom_type_mapper={8:0,14:1})


def fresh_graph(batch,q,v):
    """Differentiable state replacement, official neighbor builder per system.

    The pinned update_edge_index assumes one cell when applying image shifts;
    build each graph independently, then use official PyG collation. No graph
    index/neighbor relation or COM ever mixes separate physical samples.
    """
    items=batch.to_data_list() if isinstance(batch,Batch) else [batch]
    result=[];start=0
    for item in items:
        n=item.num_nodes
        g=AtomicGraph()
        for key in ['cell','pbc','atomic_numbers','atomic_masses','atom_types','total_mass','cutoff_radius','timestep']:
            if key in item: g[key]=item[key]
        # torch_nl's linked-cell API requires wrapped positions. The caller
        # separately retains unwrapped q for displacement/rollout supervision.
        g.pos=wrap_positions_back_to_box_torch(q[start:start+n],g.cell)
        g.velocities=v[start:start+n]
        g.batch=torch.zeros(n,dtype=torch.long,device=q.device)
        g.update_edge_index()
        del g.batch
        result.append(g);start+=n
    return Batch.from_data_list(result)


def prepare(run):
    torch.set_default_dtype(torch.float64)
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    cache=run/'cache';cache.mkdir(exist_ok=True)
    if not (cache/'adaptation.json').exists(): build(DATA,cache,horizon=30)
    offsets=np.array(list(range(31))+[60,90,120,150,180,210,240])
    for split in ['train','val','test']:
        destination=cache/f'smdp30_{split};P:64.pt'
        if destination.exists(): continue
        source=DATA/f'quartz/transitions/30fs/{split}.h5'
        graphs=AtomicGraphDataset(root=str(cache),name=f'quartz30_{split}',cutoff_radius=4.5,
                                 files=[],atom_type_mapper={8:0,14:1})
        with h5py.File(source) as f:
            paths=[p.decode() if isinstance(p,bytes) else p for p in f['source_paths'][:]]
            tids=f['source_trajectory'][:];frames=f['source_frame'][:]
            n=len(frames);nat=len(f['masses_u'])
            q=np.empty((n,len(offsets),nat,3),np.float32);v=np.empty_like(q);valid=np.ones(n,bool)
            for tid,path in enumerate(paths):
                chosen=np.flatnonzero(tids==tid)
                with h5py.File(path) as raw:
                    assert raw.attrs['ensemble']=='NVE' and raw.attrs['output_interval_fs']==1
                    ix=frames[chosen,None]+offsets[None]
                    valid[chosen]=ix[:,-1]<len(raw['time_fs'])
                    ix=np.minimum(ix,len(raw['time_fs'])-1)
                    for key,arr in [('positions_unwrapped_A',q),('velocities_A_per_fs',v)]:
                        values=raw[key][:];arr[chosen]=values[ix];del values
                print('segments',split,path,flush=True)
            np.testing.assert_allclose(q[:,0],f['positions_initial_A'][:],rtol=0,atol=1e-7)
            np.testing.assert_allclose(q[:,30],f['positions_final_A'][:],rtol=0,atol=1e-7)
            np.testing.assert_allclose(v[:,30],f['velocities_final_A_per_fs'][:],rtol=0,atol=1e-9)
        records=[]
        for i in range(n):
            g=graphs[i]
            # Subtract in float64 to avoid cancellation of float32 absolute q.
            g.trajectory_q=torch.tensor((q[i].astype(float)-q[i,0].astype(float)).transpose(1,0,2))
            g.trajectory_v=torch.tensor(v[i].transpose(1,0,2),dtype=torch.float64)
            g.rollout_valid=torch.tensor([valid[i]])
            g.sample_id=torch.tensor([i]);g.source_frame=torch.tensor([int(frames[i])])
            records.append(g)
        torch.save(InMemoryDataset.collate(records),destination)
    # Training-only finite-dimensional oracle; no network fitting or test access.
    train=dataset(run,'train')
    q=train.trajectory_q[:,:31].transpose(0,1);v=train.trajectory_v[:,:31].transpose(0,1)
    sq=float(q[30].square().mean().sqrt());sv=float(v[30].square().mean().sqrt())
    u=torch.arange(31)/30
    hq,hv=hermite(u,30.,q[30],v[0],v[30])
    fits={};selected=None
    for count in [6,8]:
        b,db=bubble_basis(u,count)
        design=torch.cat([b/sq,db/(30*sv)])
        target=torch.cat([(q-hq)/sq,(v-hv)/sv]).reshape(62,-1)
        coefficients=torch.linalg.pinv(design)@target
        errq=(b@coefficients).reshape_as(q)+hq-q
        errv=(db@coefficients/30).reshape_as(v)+hv-v
        rmsq=float(errq.square().mean().sqrt());rmsv=float(errv.square().mean().sqrt())
        fits[str(count)]=dict(position_RMSE_A=rmsq,velocity_RMSE_A_fs=rmsv)
        if selected is None and rmsq<2e-4 and rmsv<2e-5: selected=count
    report=dict(training_only=True,samples=len(train),thresholds=dict(q_A=2e-4,v_A_fs=2e-5),
                fits=fits,selected_basis=selected,rms_displacement_A=sq,rms_velocity_A_fs=sv,
                offsets_fs=offsets.tolist(),upstream_commit=COMMIT,pretrained_weights_used=False)
    (run/'data_preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected is None: raise RuntimeError('Neither candidate basis meets the frozen oracle tolerance; no training started')
    model=yaml.safe_load((FOUNDATION/'official/models/quartz/config_e3nn.yaml').read_text())['model']
    model.update(horizon_fs=30.,segment_basis=selected,net_lin_mom=[])
    config=dict(model=model,data=data_config(run,'train'),training=dict(seed=20261028,device='cuda',
        model_type='EfficientTrajCastModel',reference_fields=['displacements','update_velocities'],target_field='target',
        batch_size=2,num_epochs=600,restart_latest=False,criterion='mse',max_grad_norm=10.,
        optimizer='adam',optimizer_settings=dict(lr=.001,amsgrad=True),scheduler=['ReduceLROnPlateau'],
        scheduler_settings={'ReduceLROnPlateau':dict(factor=.8,patience=20)},
        chained_scheduler_hp=dict(milestones=[10000000],per_epoch=True,monitor_lr_scheduler=False),
        checkpoint_settings=dict(root=str(run/'upstream_checkpoints'),interval=10,keep_latest=True),
        tensorboard_settings=dict(log_dir=str(run/'tensorboard'),loss=True,lr=True,
            loss_validation=dict(data=data_config(run,'val'),batch_size=2))),
        smdp=dict(run=str(run),initialization='random_all_parameters_no_checkpoint_loaded',
            segment_max_epochs=400,segment_patience=60,rollout_stages=[4,8],rollout_stage_epochs=100,
            rollout_patience=30,rollout_every_batches=4,rollout_gradient_steps=2,
            validation_rollout_samples=128,segment_interior_weight=.5,rollout_loss_weight=.5,
            nve_ps=100.,nvt_thermalization_ps=1500.,nvt_production_ps=15000.,
            nvt_tau_fs=100.,reference_root=str(DATA)))
    target=run/'config.yaml'
    if target.exists(): raise FileExistsError('Frozen config exists; do not regenerate')
    target.write_text(yaml.safe_dump(config,sort_keys=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);a=p.parse_args();prepare(a.run)
