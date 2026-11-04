"""Signed endpoint/energy error on real and autonomously visited Water states.

New diagnostic labels use the original .5 fs SPC/Fw integrator from each
visited state itself. These labels are evaluation-only and never train a model.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from periodic import PeriodicGraphBuilder
from physics.spcfw import WaterReference, verify_reference
from physics.units import KINETIC_TO_EV, KB_EV_PER_K
from physics.water_motion import signed_kinetic_error
from relational_inference import load_relational, vesin_graph


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def describe(rows):
    frame = pd.DataFrame(rows)
    result = {}
    for group, part in frame.groupby(['origin', 'window'], sort=False):
        result['/'.join(group)] = dict(samples=len(part),
            **{key:dict(mean=float(part[key].mean()), std=float(part[key].std(ddof=0)))
               for key in part.select_dtypes(include='number') if key not in ['sample','frame']})
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ['checkpoint','transitions','trajectory','reference','topology','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--states-per-window',type=int,default=32)
    args = p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'results.json').exists():
        raise FileExistsError('Diagnostic results already exist')
    model, saved = load_relational(args.checkpoint,args.device)
    builder = PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    with h5py.File(args.reference,'r') as raw, h5py.File(args.transitions,'r') as tr:
        z, masses = raw['atomic_numbers'][:], raw['masses_u'][:]
        cell = raw['cell_A'][0]
        calculator = WaterReference(args.topology,cell,z)
        verification = verify_reference(raw,calculator,np.linspace(0,len(raw['time_fs'])-1,8,dtype=int))
        if (max(abs(r['energy_error_eV']) for r in verification) >= 1e-3 or
                max(r['force_max_error_eV_A'] for r in verification) >= .02):
            save(args.output/'reference_verification.json',dict(accepted=False,records=verification))
            raise ValueError('Reference potential mismatch')
        # Check the fresh diagnostic continuation against an existing real NVE
        # endpoint, including original force/mass/time units.
        rq,rv = calculator.short_trajectory(raw['positions_unwrapped_A'][0].astype(float),
                                            raw['velocities_A_per_fs'][0].astype(float))
        qerr = float(abs(rq-raw['positions_unwrapped_A'][10]).max())
        verr = float(abs(rv-raw['velocities_A_per_fs'][10]).max())
        save(args.output/'reference_verification.json',dict(accepted=qerr<5e-4 and verr<5e-5,
            records=verification,continuation_max_position_error_A=qerr,
            continuation_max_velocity_error_A_fs=verr))
        if qerr>=5e-4 or verr>=5e-5:
            raise ValueError('Short continuation does not match existing NVE')
        q0,v0 = tr['positions_initial_A'][:],tr['velocities_initial_A_per_fs'][:]
        q1,v1 = tr['positions_final_A'][:],tr['velocities_final_A_per_fs'][:]
        cells = tr['cell_A'][:]
        rows = []

        def record(sample, origin, window, frame, q, v, truth_q, truth_v, pred_q, pred_v):
            ke = lambda value:.5*KINETIC_TO_EV*np.sum(masses[:,None]*np.asarray(value,dtype=float)**2)
            ei, et, ep = calculator.evaluate(q),calculator.evaluate(truth_q),calculator.evaluate(pred_q)
            cross, quadratic = signed_kinetic_error(pred_v,truth_v,masses)
            result = dict(sample=int(sample),origin=origin,window=window,frame=int(frame),
                input_temperature_K=float(2*ke(v)/((3*len(z)-3)*KB_EV_PER_K)),
                position_RMSE_A=float(np.sqrt(np.mean((pred_q-truth_q)**2))),
                velocity_RMSE_A_fs=float(np.sqrt(np.mean((pred_v-truth_v)**2))),
                kinetic_error_cross_eV=float(cross),kinetic_error_quadratic_eV=float(quadratic),
                kinetic_error_eV=float(ke(pred_v)-ke(truth_v)),
                potential_error_eV=ep['pe']-et['pe'],
                total_energy_error_eV=ep['pe']+ke(pred_v)-et['pe']-ke(truth_v),
                network_energy_change_eV=ep['pe']+ke(pred_v)-ei['pe']-ke(v),
                reference_energy_change_eV=et['pe']+ke(truth_v)-ei['pe']-ke(v))
            for key in ['ebond','eangle','nonbonded']:
                result[key+'_error_eV'] = ep[key]-et[key]
            for element in [1,8]:
                mask = z==element
                a,b = signed_kinetic_error(pred_v[mask],truth_v[mask],masses[mask])
                result[f'Z{element}_kinetic_error_eV'] = float(a+b)
                result[f'Z{element}_velocity_RMSE_A_fs'] = float(np.sqrt(np.mean((pred_v[mask]-truth_v[mask])**2)))
            np.testing.assert_allclose(cross+quadratic,result['kinetic_error_eV'],atol=1e-10)
            rows.append(result)

        for begin in range(0,len(q0),8):
            ids = range(begin,min(begin+8,len(q0)))
            graphs = [vesin_graph(q0[i],v0[i],cells[i],z,masses,model,builder,'cpu') for i in ids]
            with torch.inference_mode():
                out = model(Batch.from_data_list(graphs).to(args.device))
            pq = out['positions'].cpu().numpy().reshape(len(graphs),len(z),3)
            pv = out['velocities'].cpu().numpy().reshape(len(graphs),len(z),3)
            for j,i in enumerate(ids):
                record(i,'reference','test',int(tr['source_frame'][i]),q0[i],v0[i],q1[i],v1[i],
                       pq[j],pv[j])
            if begin % 200 == 0:
                print(json.dumps(dict(origin='reference',samples=len(rows))),flush=True)
        pd.DataFrame(rows).to_csv(args.output/'signed_errors.csv',index=False)
        rng = np.random.default_rng(args.seed)
        with h5py.File(args.trajectory,'r') as trajectory, h5py.File(args.output/'diagnostic_labels.h5','x') as labels:
            labels.attrs.update(purpose='evaluation only; never training',source_checkpoint=str(args.checkpoint),
                source_trajectory=str(args.trajectory),reference_dt_fs=.5,horizon_fs=5.,seed=args.seed)
            dt = float(trajectory.attrs['timestep_fs']); stop=int(trajectory.attrs['completed_steps'])
            if dt != 5 or model.cfg.horizon_fs != 5:
                raise ValueError('This diagnostic uses the formal Water 5 fs horizon')
            selected = []
            for a,b in [(0,1),(1,5),(5,20),(20,50),(50,100)]:
                candidates=np.flatnonzero((np.arange(stop)*dt/1000>=a)&(np.arange(stop)*dt/1000<b))
                selected.extend((int(i),f'{a}-{b}ps') for i in sorted(rng.choice(candidates,
                    size=min(args.states_per_window,len(candidates)),replace=False)))
            for number,(index,window) in enumerate(selected):
                q=trajectory['positions'][index].astype(float); v=trajectory['velocities'][index].astype(float)
                tq,tv=calculator.short_trajectory(q,v)
                graph=vesin_graph(q,v,cell,z,masses,model,builder,args.device)
                with torch.inference_mode():out=model(graph)
                pq=out['positions'].cpu().numpy(); pv=out['velocities'].cpu().numpy()
                record(number,'autonomous',window,index,q,v,tq,tv,pq,pv)
                group=labels.create_group(f'{number:04d}')
                group.attrs.update(source_frame=index,window=window)
                for name,value in [('input_q_A',q),('input_v_A_fs',v),('true_q_A',tq),('true_v_A_fs',tv)]:
                    group[name]=value
                if number%16==0:
                    labels.flush();pd.DataFrame(rows).to_csv(args.output/'signed_errors.csv',index=False)
                    print(json.dumps(dict(origin='autonomous',samples=number+1,total=len(selected))),flush=True)
        calculator.close()
    pd.DataFrame(rows).to_csv(args.output/'signed_errors.csv',index=False)
    save(args.output/'results.json',dict(checkpoint=str(args.checkpoint),selected_epoch=saved['epoch']+1,
        new_labels_used_for_training=False,groups=describe(rows),
        caveat='Reference and autonomous input temperatures differ; temperature/state shifts are confounded.'))


if __name__=='__main__':main()
