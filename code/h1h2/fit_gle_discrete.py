"""Fit a discrete-time 3-6 mode AR(2)+slow state-space ACF to the mean
25-block reference translation ACF, with a Green-Kubo integral constraint.

This is the offline gate for the GLE readout: it fits the sampled ACF that the
readout will actually use, not a continuous kernel followed by discretization.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from gle_kernel_study import discrete_embedding, KEY_LAGS

def fit(mean, t, modes, n_starts=8, seed=20261126):
    integral_ref=float(np.trapezoid(mean,t))
    fit_mask=t<=5000.0
    t_fit=t[fit_mask]; ref=mean[fit_mask]
    scale=0.02+np.abs(ref)
    def residual(x):
        acf,_,_=discrete_embedding(x,modes,5.0)
        pred=acf[:len(ref)]
        r=(pred-ref)/scale
        r_int=20.0*(np.trapezoid(acf,t)-integral_ref)/integral_ref
        return np.concatenate([r,[r_int]])
    rng=np.random.default_rng(seed+modes)
    best=None; costs=[]
    for _ in range(n_starts):
        gamma0=np.exp(rng.uniform(-5.0,-2.5,modes))
        omega0=np.exp(rng.uniform(-4.0,-1.5,modes))
        tau0=np.exp(rng.uniform(np.log(200.0),np.log(5000.0)))
        x0=np.concatenate([np.zeros(modes+1),np.log(gamma0),np.log(omega0),[np.log(tau0)]])
        lower=np.concatenate([np.full(modes+1,-15.0),np.full(modes,np.log(1e-5)),np.full(modes,np.log(1e-4)),[np.log(20.0)]])
        upper=np.concatenate([np.full(modes+1,15.0),np.full(modes,np.log(2.0)),np.full(modes,np.log(2.0)),[np.log(20000.0)]])
        try:
            f=least_squares(residual,x0,bounds=(lower,upper),max_nfev=2000,xtol=1e-10,ftol=1e-10,gtol=1e-10)
        except Exception:
            continue
        cost=float(np.sum(f.fun**2)); costs.append(cost)
        if best is None or cost<best[0]: best=(cost,f.x,f.jac)
    return best,integral_ref,np.asarray(costs)

def main():
    d=np.load('experiments/reference_acf_blocks_20261126.npz')
    acfs=d['acfs']; t=d['t']; mean=acfs.mean(0); block_int=np.array([np.trapezoid(a,t) for a in acfs])
    out=[]; key=np.array(KEY_LAGS); mean_key=np.interp(key,t,mean)
    for modes in [3,4,5,6]:
        (cost,x,jac),integral_ref,costs=fit(mean,t,modes,8)
        acf,min_eig,min_innov=discrete_embedding(x,modes,5.0)
        pred_key=np.interp(key,np.arange(len(acf))*5.0,acf)
        # held-out block 0/1 comparison at key lags
        b0=np.interp(key,t,acfs[0]); b1=np.interp(key,t,acfs[1])
        row=dict(modes=modes,fit_cost=float(cost),cost_spread=float(np.std(costs)),
            jacobian_condition=float(np.linalg.cond(jac)),
            integral_fs=float(np.trapezoid(acf,t)),mean_integral_fs=integral_ref,
            ratio=float(np.trapezoid(acf,t)/integral_ref),
            acf_key=pred_key.tolist(),mean_acf_key=mean_key.tolist(),
            block0_key=b0.tolist(),block1_key=b1.tolist(),
            rmse_mean=float(np.sqrt(np.mean((pred_key-mean_key)**2))),
            rmse_block0=float(np.sqrt(np.mean((pred_key-b0)**2))),
            rmse_block1=float(np.sqrt(np.mean((pred_key-b1)**2))),
            min_eig_gap=float(min_eig),min_innovation_variance=float(min_innov),
            n_parameters=3*modes+1)
        out.append(row)
        print(json.dumps({k:row[k] for k in ['modes','integral_fs','mean_integral_fs','ratio','rmse_mean','rmse_block0','rmse_block1','min_eig_gap','min_innovation_variance','jacobian_condition']},indent=2),flush=True)
    Path('experiments/fit_gle_discrete_20261126.json').write_text(json.dumps(dict(mean_integral_fs=float(np.trapezoid(mean,t)),block_std=float(block_int.std(ddof=1)),results=out),indent=2)+'\n')

if __name__=='__main__': main()
