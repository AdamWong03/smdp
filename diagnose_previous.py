"""Read-only diagnosis of the frozen first run; write only to a new directory."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from evaluate_quartz import load_model
from periodic import PeriodicGraphBuilder, predict_batch
from train_quartz import diagnostics
from physics.units import KINETIC_TO_EV, KB_EV_PER_K


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.output.resolve()==args.previous.resolve(): raise ValueError('Preserve the original experiment')
    report={}
    for name in ['nve_test_09','nve_test_10']:
        table=np.genfromtxt(args.previous/(name+'_analysis/thermodynamics.csv'),delimiter=',',names=True)
        rows=[]
        for a,b in [(0,1),(1,5),(5,10),(10,25),(25,50),(50,100.01)]:
            selected=(table['time_fs']>=1000*a)&(table['time_fs']<1000*b)
            rows.append(dict(start_ps=a,end_ps=b,temperature_K=float(table['SMDP_temperature_K'][selected].mean()),
                energy_change_from_initial_eV=float(table['SMDP_total_eV'][selected].mean()-table['SMDP_total_eV'][0])))
        report[name]=rows
    with h5py.File(args.previous/'nvt.h5') as f:
        start=int(f.attrs['thermalization_steps'])+1
        post=f['temperature_K'][:].astype(float); pre=f['pre_thermostat_temperature_K'][:].astype(float)
        alpha=f['thermostat_alpha'][:].astype(float); metadata=json.loads(f.attrs['thermostat_metadata'])
        c=np.exp(-16/metadata['thermostat_tau_fs']); selection=slice(start,None)
        expected=c*pre[selection]+(1-c)*300
        report['csvr']=dict(tau_fs=metadata['thermostat_tau_fs'],dt_fs=16,
            post_temperature_mean_K=float(post[selection].mean()),pre_temperature_mean_K=float(pre[selection].mean()),
            alpha_mean=float(alpha[selection].mean()),alpha_std=float(alpha[selection].std()),
            scaling_relation_max_abs_K=float(abs(post[selection]-alpha[selection]**2*pre[selection]).max()),
            conditional_expectation_innovation_mean_K=float((post[selection]-expected).mean()),
            neural_temperature_change_per_step_K=float((pre[start:]-post[start-1:-1]).mean()),
            thermostat_temperature_change_per_step_K=float((post[selection]-pre[selection]).mean()))
        ids=np.linspace(start,len(post)-1,2000,dtype=int)
        v=f['velocities'][ids].astype(float);m=f['masses_u'][:];z=f['atomic_numbers'][:]
        com=(v*m[None,:,None]).sum(1)/m.sum()
        internal=v-com[:,None]; before=internal/alpha[ids,None,None]
        report['csvr']['species_kinetic_eV']={str(int(s)):dict(
            before=float((.5*KINETIC_TO_EV*m[z==s][None,:,None]*before[:,z==s]**2).sum((1,2)).mean()),
            after=float((.5*KINETIC_TO_EV*m[z==s][None,:,None]*internal[:,z==s]**2).sum((1,2)).mean())) for s in np.unique(z)}
    # Real vector contributions, not just softmax weights. Use existing NVE
    # states (diagnosis only), with no retraining and no checkpoint mutation.
    torch.set_num_threads(4)
    model,_=load_model(args.previous/'best_rollout.pt','cuda')
    builder=PeriodicGraphBuilder(model.config.cutoff_angstrom)
    report['mode_contributions']={}
    with h5py.File(args.previous/'nve_test_09.h5') as f, torch.inference_mode():
        for label,indices in [('early',np.arange(0,64)),('late',np.arange(len(f['time_fs'])-64,len(f['time_fs'])) )]:
            sums={}
            for index in indices:
                q=torch.tensor(f['positions'][index],device='cuda')[None]
                v=torch.tensor(f['velocities'][index],device='cuda')[None]
                z=torch.tensor(f['atomic_numbers'][:],device='cuda')[None]
                m=torch.tensor(f['masses_u'][:],device='cuda',dtype=torch.float32)[None]
                cell=torch.tensor(f['cell'][:],device='cuda',dtype=torch.float32)[None]
                out=predict_batch(model,q,v,z,m,cell,[16.],builder)
                for k,x in diagnostics(out).items(): sums[k]=sums.get(k,0.)+np.array(x)
            report['mode_contributions'][label]={k:(v/len(indices)).tolist() for k,v in sums.items()}
    report['attribution_scope']='Exact additive vector-path attribution with scalar decoder coefficients held fixed; not a causal attribution of scalar context.'
    (args.output/'previous_diagnosis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
