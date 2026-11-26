"""Fit DHO+slow kernels to the mean 25-block reference translation ACF."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from gle_kernel_study import translation_acf, fit_model, evaluate_fit

def main():
    ref_path = Path('/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/water/physical_reference/nvt_2p5ns/production.h5')
    out = Path('experiments/gle_kernel_study_mean_20261126.json')
    n_blocks, block_ps = 25, 100.0
    acfs = []
    for i in range(n_blocks):
        acf, dt = translation_acf(ref_path, i*block_ps, block_ps, 5.0, 5000.0)
        acfs.append(acf)
        print('block', i, 'integral', float(np.trapezoid(acf, np.arange(len(acf))*dt)), flush=True)
    acfs = np.stack(acfs)
    mean = acfs.mean(axis=0)
    std = acfs.std(axis=0, ddof=1)
    t = np.arange(len(mean))*dt
    np.savez_compressed('experiments/reference_acf_blocks_20261126.npz', acfs=acfs, t=t)
    print('mean integral', float(np.trapezoid(mean, t)), flush=True)
    results=[]
    for modes in [3,4,5,6]:
        (cost,x,jac), integral_ref, costs = fit_model(mean, t, modes, 16)
        result=evaluate_fit(x, modes, mean, t, mean, t)
        result['fit_cost']=float(cost); result['fit_cost_spread']=float(np.std(costs)); result['jacobian_condition']=float(np.linalg.cond(jac))
        result['block_mean_integral_fs']=float(np.trapezoid(mean,t))
        result['block_std_integral_fs']=float(np.std([np.trapezoid(a,t) for a in acfs],ddof=1))
        result['predicted_D_scale_vs_block0']=float(np.trapezoid(mean,t)/np.trapezoid(acfs[0],t))
        results.append(result)
        print(json.dumps({k:result[k] for k in ['modes','integral_model_fs','rmse_calibration','psd_min_real','discrete_min_eig_gap','discrete_rmse_vs_model','n_parameters','jacobian_condition','predicted_D_scale_vs_block0']},indent=2), flush=True)
    out.write_text(json.dumps(dict(reference=str(ref_path),block_ps=block_ps,n_blocks=n_blocks,results=results),indent=2)+'\n')
    print('wrote',out)

if __name__=='__main__': main()
