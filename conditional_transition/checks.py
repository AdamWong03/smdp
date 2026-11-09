"""Necessary numerical checks on formal data; no reduced training experiment."""
import json

import numpy as np
import torch
from torch_geometric.data import Batch

from conditional_transition.common import ROOT,checkpoint,originals,seed_all
from conditional_transition.oracle import create_oracle
from conditional_transition.train import Cycle,new_count
from periodic import PeriodicGraphBuilder
from relational_inference import load_relational,vesin_graph
from relational_transition import endpoint_objective
from smdpr.data import TwoStepLabels
from smdpr.finetune import two_step
from conditional_transition.common import transitions
from smdpr.train import write_json


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    results={}
    assert sum(new_count(i) for i in range(25000))==120000
    for system in ['quartz','water']:
        seed_all(20261102)
        model,saved=load_relational(checkpoint(system))
        train=originals(saved['config'],'train')
        val=originals(saved['config'],'val')
        assert len(train)==5000 and len(val)==1250
        g=train[0];builder=PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
        oracle_graph=vesin_graph(g.pos,g.velocities,g.cell[0],g.atomic_numbers,
            g.atomic_masses,model,builder,'cpu')
        oracle_graph.displacements=g.displacements.clone()
        oracle_graph.update_velocities=g.update_velocities.clone()
        assert set(oracle_graph.keys())==set(g.keys())
        # A real production graph and its cached counterpart must predict alike.
        with torch.no_grad():
            a=model(Batch.from_data_list([g]).cuda())
            b=model(Batch.from_data_list([oracle_graph]).cuda())
        graph_error={k:float((a[k]-b[k]).abs().max()) for k in ['positions','velocities']}
        assert graph_error['positions']<3e-5 and graph_error['velocities']<1e-6,graph_error
        # Same 16 formal samples: accumulation must reproduce the full-batch gradient.
        model.zero_grad(set_to_none=True)
        batch=Batch.from_data_list(train[:16]).cuda();out=model(batch)
        loss,_=endpoint_objective(model,out,batch.displacements,batch.update_velocities,proposal_weight=.02)
        loss.backward();full=torch.cat([p.grad.flatten().cpu() for p in model.parameters() if p.grad is not None])
        full_loss=float(loss.detach());del batch,out,loss
        model.zero_grad(set_to_none=True);acc_loss=0.
        for begin in [0,8]:
            batch=Batch.from_data_list(train[begin:begin+8]).cuda();out=model(batch)
            loss,_=endpoint_objective(model,out,batch.displacements,batch.update_velocities,proposal_weight=.02)
            (.5*loss).backward();acc_loss+=.5*float(loss.detach())
        accum=torch.cat([p.grad.flatten().cpu() for p in model.parameters() if p.grad is not None])
        relative=float(torch.linalg.vector_norm(full-accum)/torch.linalg.vector_norm(full))
        assert relative<.01,relative
        assert abs(full_loss-acc_loss)<1e-6
        model.zero_grad(set_to_none=True)
        labels=TwoStepLabels(transitions(system),train)
        batch=Batch.from_data_list(labels.graphs[:8]).cuda()
        loss,_,_,_=two_step(model,batch,builder,.02);loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        nparams=sum(p.numel() for p in model.parameters())
        del model,batch,out,loss,full,accum,train,val,labels
        torch.cuda.empty_cache()
        calc,cell,z,m,verification=create_oracle(system);calc.close()
        results[system]=dict(passed=True,train_samples=5000,val_samples=1250,
            parameters=nparams,cached_production_graph_max_error=graph_error,
            accumulation_gradient_relative_error=relative,full_batch_loss=full_loss,
            accumulated_loss=acc_loss,reference=verification,
            extra_training_draws=120000,transition_targets_per_arm=400000)
        write_json(ROOT/'numerical_checks.json',results)
        print(json.dumps(dict(system=system,**results[system])),flush=True)


if __name__=='__main__':main()
