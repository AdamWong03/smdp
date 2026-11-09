"""Numerical audit using actual formal graphs and frozen trained models."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

from dynamics_diagnosis.common import DATA, REPORT, checkpoints, save
from periodic import PeriodicGraphBuilder
from relational_inference import load_relational, vesin_graph
from relational_transition import endpoint_objective
from smdpr.data import InteriorLabels, TwoStepLabels
from smdpr.finetune import two_step
from train_relational import read_graphs


def graph_from(g, model, builder, device):
    return vesin_graph(g.pos, g.velocities, g.cell.reshape(3,3),
                       g.atomic_numbers,g.atomic_masses,model,builder,device)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--system',choices=['quartz','water'],required=True)
    args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    system=args.system
    h=30 if system=='quartz' else 5
    first=next(iter(checkpoints(system).values()))
    _,saved=load_relational(first,'cpu')
    cfg=saved['config']
    train=read_graphs(Path(cfg['cache']),'train',cfg)
    # Validate cached lattice offsets/real one-step endpoints for EVERY start.
    interior=DATA/system/'derived_segments'/f'cmis_{h}fs'
    InteriorLabels(interior,'train',train)
    labels=TwoStepLabels(DATA/system/'transitions'/f'{h}fs/train.h5',train)
    max_q=max_v=0.
    with h5py.File(DATA/system/'transitions'/f'{h}fs/train.h5') as tr:
        paths=[p.decode() if isinstance(p,bytes) else str(p) for p in tr['source_paths'][:]]
        ids,frames,lag=tr['source_trajectory'][:],tr['source_frame'][:],int(tr.attrs['lag_steps'])
        for k,path in enumerate(paths):
            with h5py.File(path) as raw:
                for i in np.flatnonzero(ids==k):
                    if frames[i]+2*lag>=len(raw['time_fs']):continue
                    g=train[i]
                    offset=g.pos.numpy()-tr['positions_initial_A'][i]
                    q=raw['positions_unwrapped_A'][int(frames[i]+2*lag)]+offset
                    v=raw['velocities_A_per_fs'][int(frames[i]+2*lag)]
                    max_q=max(max_q,float(abs(g.two_step_positions.numpy()-q).max()))
                    max_v=max(max_v,float(abs(g.two_step_velocities.numpy()-v).max()))
    if max_q>5e-6 or max_v>1e-7:
        raise AssertionError('Two-step labels fail true-source numerical comparison')
    results=[]
    for name,checkpoint in checkpoints(system).items():
        model,_=load_relational(checkpoint)
        builder=PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
        graphs=labels.graphs[:8]
        batch=Batch.from_data_list(graphs).cuda()
        original=model(batch)
        rebuilt=Batch.from_data_list([graph_from(g,model,builder,'cpu') for g in graphs]).cuda()
        predicted=model(rebuilt)
        graph_q=float(abs(original['positions']-predicted['positions']).max())
        graph_v=float(abs(original['velocities']-predicted['velocities']).max())
        # Triclinic image shifts are integer row-lattice translations.
        shifted=[]
        for g in graphs:
            new=g.clone()
            images=torch.arange(len(g.pos))[:,None]%3-1
            new.pos=g.pos+images.to(g.pos.dtype)*g.cell.reshape(3,3).sum(0)
            shifted.append(new)
        reimaged=model(Batch.from_data_list([graph_from(g,model,builder,'cpu') for g in shifted]).cuda())
        pbc_q=float(abs(reimaged['displacements']-original['displacements']).max())
        pbc_v=float(abs(reimaged['velocities']-original['velocities']).max())
        # Capture the actual second forward in the unchanged training function.
        calls=[]
        hook=model.register_forward_hook(lambda module,inputs,output:calls.append((inputs[0],output)))
        loss,proposal,p1,p2=two_step(model,batch,builder,.02)
        hook.remove()
        g2,o2=calls[1]
        absolute=.5*(((o2['positions']-batch.two_step_positions)/model.rms_targets[0]).square().mean()+
                      ((o2['velocities']-batch.two_step_velocities)/model.rms_targets[1]).square().mean())
        increment,_=endpoint_objective(model,o2,batch.two_step_positions-proposal['positions'],
                                       batch.two_step_velocities,proposal_weight=0.)
        equivalent=float(abs(absolute-increment))
        proposal['positions'].retain_grad()
        proposal['velocities'].retain_grad()
        model.zero_grad(set_to_none=True)
        absolute.backward()
        qgrad=proposal['positions'].grad.clone()
        vgrad=proposal['velocities'].grad.clone()
        grad_norm=float(sum(p.grad.square().sum() for p in model.proposal_head.parameters() if p.grad is not None).sqrt())
        # Directional finite difference through rebuilt graph and absolute target.
        direction=qgrad/qgrad.norm()
        qcenter=proposal['positions'].detach()
        vcenter=proposal['velocities'].detach()
        def objective(q):
            pieces=[]
            for i,g in enumerate(graphs):
                sl=slice(i*len(g.pos),(i+1)*len(g.pos))
                pieces.append(vesin_graph(q[sl],vcenter[sl],g.cell.reshape(3,3),
                    g.atomic_numbers,g.atomic_masses,model,builder,'cuda'))
            with torch.no_grad():
                o=model(Batch.from_data_list(pieces))
                return float(.5*(((o['positions']-batch.two_step_positions)/model.rms_targets[0]).square().mean()+
                                 ((o['velocities']-batch.two_step_velocities)/model.rms_targets[1]).square().mean()))
        eps=1e-3
        finite=(objective(qcenter+eps*direction)-objective(qcenter-eps*direction))/(2*eps)
        analytic=float((qgrad*direction).sum())
        relative=abs(finite-analytic)/max(abs(analytic),1e-8)
        passed=bool(graph_q<1e-5 and graph_v<1e-6 and pbc_q<1e-5 and pbc_v<1e-6
            and equivalent<1e-6 and grad_norm>0 and qgrad.norm()>0 and vgrad.norm()>0 and relative<.1)
        row=dict(model=name, passed=passed, graph_q_max_A=graph_q,graph_v_max_A_fs=graph_v,
            reimage_dq_max_A=pbc_q,reimage_v_max_A_fs=pbc_v,
            increment_vs_absolute_loss_difference=equivalent,
            second_loss_to_proposal_head_gradient_norm=grad_norm,
            second_loss_to_first_q_gradient_norm=float(qgrad.norm()),
            second_loss_to_first_v_gradient_norm=float(vgrad.norm()),
            finite_difference_epsilon_A=eps,analytic_directional_derivative=analytic,
            numeric_directional_derivative=finite,finite_difference_relative_error=relative)
        results.append(row)
        print(row,flush=True)
        del model,calls,loss,proposal,original,predicted,reimaged
    report=dict(system=system,formal_training_samples=len(train),two_step_valid=len(labels.graphs),
        excluded=labels.excluded,max_two_step_q_label_error_A=max_q,
        max_two_step_absolute_velocity_label_error_A_fs=max_v,checks=results,
        passed=all(x['passed'] for x in results),weights_updated=False)
    save(REPORT/f'{system}_implementation_audit.json',report)
    if not report['passed']:raise RuntimeError('Real-data implementation audit failed')


if __name__=='__main__':main()
