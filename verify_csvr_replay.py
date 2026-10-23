"""Replay thermostat random draws against saved scalings; never updates a trajectory."""
import argparse,json,math
from pathlib import Path
import h5py
import numpy as np
from analysis_tasks import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=2026102201)
    a=p.parse_args()
    with h5py.File(a.trajectory,'r') as f:
        pre=f['pre_thermostat_temperature_K'][1:].astype(float);alpha=f['thermostat_alpha'][1:].astype(float)
        post=f['temperature_K'][1:].astype(float);thermal=int(f.attrs['thermalization_steps'])
        metadata=json.loads(f.attrs['thermostat_metadata']);c=math.exp(-16/metadata['thermostat_tau_fs'])
        dof=3*len(f['masses_u'])-3
    rng=np.random.default_rng(a.seed);expected=np.empty(len(pre))
    for i,t in enumerate(pre):
        r=rng.normal();chi=rng.chisquare(dof-1);ratio=(1-c)*300/(dof*t)
        value=c+ratio*(r*r+chi)+2*r*math.sqrt(c*ratio)
        expected[i]=math.copysign(math.sqrt(max(value,0)),math.sqrt(c)+math.sqrt(ratio)*r)
    error=np.abs(alpha-expected)
    innovation=(expected**2*pre-(c*pre+(1-c)*300))[thermal:]
    result=dict(seed=a.seed,seed_source='Default --seed in the frozen rollout_quartz.py used by run_experiment.py; checked by full-sequence replay',
        steps=len(pre),max_alpha_abs_error=float(error.max()),mean_alpha_abs_error=float(error.mean()),
        matched_with_float32_storage=bool(error.max()<3e-7),
        expected_finite_random_sequence_innovation_mean_K=float(innovation.mean()),
        measured_innovation_mean_K=float((post-c*pre-(1-c)*300)[thermal:].mean()),
        caveat='Temperature is stored float32 and includes the preserved, negligible initial COM energy.')
    save_json(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
