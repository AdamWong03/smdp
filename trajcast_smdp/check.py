"""Required implementation identities on actual training graphs, not a pilot fit."""
import argparse,json,time
from pathlib import Path
import torch
import yaml
from torch_geometric.data import Batch
from trajcast.utils.misc import GLOBAL_DEVICE
from trajcast.model.forecast import Forecast
from trajcast.utils.atomic_computes import wrap_positions_back_to_box_torch
from trajcast_smdp.data import RUN,dataset,fresh_graph
from trajcast_smdp.model import SMDP,center_by_mass


def main(run):
    torch.set_default_dtype(torch.float64);torch.manual_seed(20261028)
    GLOBAL_DEVICE.device=torch.device('cpu')
    cfg=yaml.safe_load((run/'config.yaml').read_text())
    ds=dataset(run,'train');scales=json.loads((run/'data_preparation.json').read_text())
    model=SMDP(config=cfg['model'],rms_targets=[scales['rms_displacement_A'],scales['rms_velocity_A_fs']])
    # Nonzero corrections are essential: do not test equivariance on a zero head.
    modes=model._int_out.MotionModes
    with torch.no_grad():
        modes.expand.weight.normal_(std=.01)
        for head in modes.temporal: head.weight.normal_(std=.1)
    model.eval();batch=Batch.from_data_list([ds[0],ds[1]])
    out=model(batch.clone());times=torch.tensor([0.,1.,7.3,15.,29.,30.],requires_grad=True)
    segment=model.query(out,times)
    torch.testing.assert_close(segment['displacements'][0],torch.zeros_like(batch.pos),atol=1e-12,rtol=0)
    torch.testing.assert_close(segment['velocities'][0],batch.velocities,atol=1e-12,rtol=0)
    torch.testing.assert_close(segment['displacements'][-1],out.target[:,:3]*model.rms_targets[0])
    torch.testing.assert_close(segment['velocities'][-1],out.target[:,3:]*model.rms_targets[1])
    _,deriv=torch.autograd.functional.jvp(lambda t:model.query(out,t)['displacements'],times,torch.ones_like(times))
    derivative_error=float((deriv-segment['velocities']).abs().max())
    assert derivative_error<1e-10
    for graph_id in range(2):
        mask=batch.batch==graph_id;m=batch.atomic_masses[mask]
        pin=(m*batch.velocities[mask]).sum(0)
        pout=(segment['velocities'][:,mask]*m).sum(1)
        torch.testing.assert_close(pout,pin[None].expand_as(pout),rtol=1e-9,atol=1e-10)
    rot=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]])
    rotated=batch.clone()
    rotated.pos=rotated.pos@rot.T;rotated.velocities=rotated.velocities@rot.T
    rotated.cell=rotated.cell@rot.T;rotated.shifts=rotated.shifts@rot.T
    rotout=model.query(model(rotated),times)
    equivariance_error=max(float((rotout[k]-segment[k]@rot.T).abs().max()) for k in ['displacements','velocities'])
    assert equivariance_error<1e-9
    with torch.no_grad():
        singles=torch.cat([model(Batch.from_data_list([ds[i]])).target for i in [0,1]])
        batch_error=float((singles-out.target).abs().max());assert batch_error<1e-9
        image=batch.pos.clone();image[::2]+=batch.cell.reshape(2,3,3)[batch.batch[::2],0]
        imaged=model(fresh_graph(batch,image,batch.velocities))
        pbc_error=float((imaged.target-out.target).abs().max());assert pbc_error<1e-8
        # The official Forecast accepts our model directly; no custom integrator.
        initial=ds[0].clone()
        for k in ['trajectory_q','trajectory_v','rollout_valid','sample_id','source_frame','displacements','update_velocities']:
            if k in initial: del initial[k]
        engine=Forecast(dict(device=torch.device('cpu'),seed=20261028,units='real',run=1,timestep=30.,
            configuration=initial,model=model,extra_dof=3,thermostat=False))
        start=engine.start_graph.clone();pred=model(start.clone())
        q=wrap_positions_back_to_box_torch(start.pos+pred.target[:,:3]*model.rms_targets[0],start.cell)
        v=pred.target[:,3:]*model.rms_targets[1]
        actual=engine._make_timestep(start,1)
        torch.testing.assert_close(actual.pos,q,atol=1e-12,rtol=0)
        torch.testing.assert_close(actual.velocities,v,atol=1e-12,rtol=0)
    model.zero_grad()
    segment=model.query(model(batch.clone()),times.detach())
    loss=segment['displacements'][1:].square().mean()+segment['velocities'].square().mean()
    loss.backward()
    gradients={}
    for key in ['inputs','embed','state_gate','state_update','query','key','exchange','exchange_gate','expand','temporal']:
        values=[p.grad for name,p in modes.named_parameters() if name==key or name.startswith(key+'.')]
        assert values and all(g is not None and bool(torch.isfinite(g).all()) for g in values),key
        gradients[key]=sum(float(g.abs().sum()) for g in values)
        assert gradients[key]>0,key
    assert float(torch.diagonal(out.mode_attention,dim1=-2,dim2=-1).abs().max())==0
    from trajcast_smdp.train import rollout_loss
    model.zero_grad()
    loss,_=rollout_loss(model,batch.clone(),2,grad_steps=2)
    loss.backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    report=dict(passed=True,derivative_max_error_A_fs=derivative_error,E3_max_error=equivariance_error,
        batch_single_max_error=batch_error,PBC_reimage_max_error=pbc_error,gradient_L1=gradients,
        endpoint_constraints=True,momentum_per_system=True,official_forecast_identity=True,
        two_step_autoregressive_backward=True,
        pretrained_weights_used=False,parameters=sum(p.numel() for p in model.parameters()))
    (run/'correctness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);a=p.parse_args();main(a.run)
