"""Export final formal CMIS evidence, including failed trajectories and analyses."""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def export(base,out):
    out.mkdir(parents=True,exist_ok=True);all_results={}
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,ax in zip(['quartz30','water5'],axes):
        run=base/(name+'_20261106_r1');target=out/name;target.mkdir(exist_ok=True)
        data={}
        for key in ['config','metadata','status']:
            data[key]=json.loads((run/(key+'.json')).read_text())
        for split in ['val','test']:
            source=run/'evaluation'/(split+'.json')
            data[split]=json.loads(source.read_text())
            shutil.copy2(source,target/(split+'.json'))
        data['physics']={}
        for directory in sorted((run/'physics').iterdir()):
            if not directory.is_dir():continue
            summary=directory/'trajectory.summary.json'
            if not summary.exists():continue
            record=dict(rollout=json.loads(summary.read_text()),analysis={})
            for source in sorted((directory/'analysis').glob('*.json')):
                record['analysis'][source.stem]=json.loads(source.read_text())
            with h5py.File(directory/'trajectory.h5','r') as trajectory:
                completed=int(trajectory.attrs['completed_steps'])
                t=trajectory['time_fs'][:completed+1]/1000
                temp=trajectory['temperature_K'][:completed+1]
                record['saved_temperature_summary']=dict(mean_K=float(temp.mean()),
                    initial_K=float(temp[0]),final_valid_K=float(temp[-1]),maximum_valid_K=float(temp.max()))
                if directory.name.startswith('nve_test'):
                    stride=max(1,len(t)//2000)
                    ax.plot(t[::stride],temp[::stride],label=directory.name)
            copied=target/directory.name;copied.mkdir(exist_ok=True)
            (copied/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
            for source in (directory/'analysis').glob('*'):
                if source.is_file() and source.suffix in ['.png','.csv','.txt']:
                    shutil.copy2(source,copied/source.name)
            data['physics'][directory.name]=record
        ax.set(title=name,xlabel='Time (ps)',ylabel='Saved valid-state temperature (K)')
        ax.legend()
        all_results[name]=data
    fig.tight_layout();fig.savefig(out/'nve_temperature.png',dpi=180);plt.close(fig)
    (out/'results.json').write_text(json.dumps(all_results,indent=2)+'\n')
    lines=['# CMIS: first formal Quartz / Water results','',
        'Upstream model commit d7e20ed; formal integration cdde3bb. Each system',
        'used 350 complete epochs, 218,750 optimizer updates, all 5,000 training',
        'and 1,250 validation samples, seed 20261106, FP32 and batch 8.',
        'Both services finished their prescribed attempt sequences. That does not',
        'mean the requested physical trajectories completed: all NVT attempts',
        'failed before production. Checkpoints and trajectories remain on server.',
        '', '## Independent endpoint test (1,000 samples)','',
        '| System / model | Position RMSE (Å) | Velocity RMSE (Å/fs) |',
        '|---|---:|---:|']
    for name,h,oldq,oldv in [('quartz30','30',.001800432,.000122078),('water5','5',.000654127,.000237643)]:
        q=all_results[name]['test']['queries'][h]
        lines.append(f"| {name} CMIS | {q['position_RMSE_A']:.9f} | {q['velocity_RMSE_A_fs']:.9f} |")
        lines.append(f'| {name} old DSRT R3-Opt | {oldq:.9f} | {oldv:.9f} |')
    lines+=['', 'CMIS endpoint errors are higher on both systems. Architectures, parameter',
        'counts, supervision targets and training schedules differ; this is an',
        'outcome comparison, not an isolated mechanism or matched-budget claim.',
        'CMIS has 431,827 parameters versus 2,357,616 for DSRT R3-Opt.',
        '', '## Selection and training cost','',
        '| System | Selected epoch | Training hours | Val endpoint q RMSE (Å) | Val endpoint v RMSE (Å/fs) |',
        '|---|---:|---:|---:|---:|']
    for name in all_results:
        d=all_results[name];h='30' if name=='quartz30' else '5';v=d['val']['queries'][h]
        lines.append(f"| {name} | {d['test']['selected_epoch']} | {d['status']['elapsed_seconds']/3600:.3f} | {v['position_RMSE_A']:.8f} | {v['velocity_RMSE_A_fs']:.8f} |")
    lines+=['', '## Multistep independent test','',
        '999 starts per system have enough remaining real reference frames for',
        'all eight autonomous steps; no reference states are injected.',
        '', '| System | Time (fs) | Position RMSE (Å) | Velocity RMSE (Å/fs) |',
        '|---|---:|---:|---:|']
    for name,d in all_results.items():
        for h,v in d['test']['multistep']['horizons'].items():
            lines.append(f"| {name} | {v['time_fs']} | {v['position_RMSE_A']:.8f} | {v['velocity_RMSE_A_fs']:.8f} |")
    lines+=['', '## Autonomous physical outcomes','',
        '| System / initial state | Completed valid time (ps) | Completed request? | Outcome |',
        '|---|---:|---|---|']
    for name,d in all_results.items():
        for label,p in d['physics'].items():
            r=p['rollout']
            lines.append(f"| {name} / {label} | {r['simulated_ps']:.3f} | {'yes' if r['stable'] else 'no'} | {r['failure_reason'] or 'No numerical safety stop'} |")
    water=all_results['water5']['physics']['nve_test_05']['analysis']
    motion=water['motion']['metrics'];energy=water['energy']['metrics']
    lines+=['', '### Quartz','',
        'Both independent test NVE attempts fail within 1 ps: 0.66 and 0.48 ps',
        'of valid states. The NVT attempt fails after 0.69 ps during the requested',
        '1.5 ns thermalization, leaving zero production steps. Close contacts and/or',
        'excess temperature trigger the unchanged safety conditions. No full-length',
        'Quartz NVE/NVT, equilibrium RDF or equilibrium VDOS is claimed.',
        'Short failed-trajectory energy curves remain available in the per-run folders.',
        '', '### Water','',
        'The independent NVE trajectory completes 100 ps. Comparisons with the',
        'reference are restricted to its available 50 ps. The matched sampled',
        f"mean temperature is {motion['temperature_K']['SMDP']['mean']:.3f} K versus {motion['temperature_K']['reference']['mean']:.3f} K.",
        f"Total-energy change is {energy['components']['total']['change_eV']['model']:+.5f} eV versus {energy['components']['total']['change_eV']['reference']:+.5f} eV.",
        'Thus the newer model dissipates energy rather than reproducing reference',
        'NVE behavior. Its molecular kinetic partition is less distorted than the',
        'old internally overheated DSRT run, but the system is too cold overall.',
        '', '| Kinetic component, common 50 ps | CMIS (eV) | Reference (eV) |',
        '|---|---:|---:|']
    for key,v in motion['kinetic_partition'].items():
        lines.append(f"| {key} | {v['SMDP']['mean']:.5f} | {v['reference']['mean']:.5f} |")
    lines+=['', 'Water H/O macro-state VDOS overlaps are '+
        '/'.join(f"{water['vdos']['metrics']['elements'][z]['overlap']:.5f}" for z in ['1','8'])+'.',
        'The corresponding real-reference 1 fs versus model queried-1 fs overlaps are '+
        '/'.join(f"{water['dense_vdos']['metrics']['elements'][z]['overlap']:.5f}" for z in ['1','8'])+'.',
        'These improve over the older heated DSRT NVE, but neither the differing',
        'temperature nor nonstationarity permits an equilibrium-fidelity claim.',
        '', 'The Water NVT attempt fails after 113.265 ps during the requested 250 ps',
        'thermalization, with a rejected-step minimum distance 0.5737 Å. There is',
        'no 2.5 ns production trajectory, diffusion coefficient or NVT distribution.',
        'No thermostat parameters or failure criteria were adjusted.',
        '', '### Incomplete validation analysis','',
        'Water validation NVE completes 50 ps, but its offline energy verification',
        'fails the existing reference-accuracy threshold. Other analyses completed.',
        'The failed verification record and traceback are retained. This task is',
        'explicitly incomplete; no energy result is substituted or fabricated.',
        '', '## Mode and segment diagnostics','',
        '| System | Endpoint contribution effective rank (4 modes) | Max qdot–v discrepancy (Å/fs) |',
        '|---|---:|---:|']
    for name,d in all_results.items():
        lines.append(f"| {name} | {d['test']['modes']['endpoint_velocity_contribution']['effective_rank']:.4f} | {d['test']['query_checks']['qdot_v_max_abs_A_fs']:.3g} |")
    lines+=['', 'Individual mode deletion changes predictions, but modes remain strongly',
        'correlated (and some Quartz contributions are anticorrelated). The effective',
        'ranks do not support four independent physical motion modes. Attention,',
        'frequency distributions, deletion effects and dense-query errors are included',
        'in the full validation/test JSON files. Analytic kinematic consistency and',
        'linear momentum handling do not guarantee accurate autonomous dynamics.',
        'Separate versus dense queries agree up to FP32 rounding, not bitwise.',
        '', '## Efficiency and limits','',
        'The complete Water 100 ps NVE run reaches 22.44 ns/day, including two graph',
        'builds, neural updates, dense 1 fs output and file I/O on a shared L40S.',
        'Short failed-run throughput is not treated as long-production speed.',
        'Exclusive-GPU speedup over old methods is not established.',
        '', '## Conclusion','',
        'The first full CMIS experiments are negative overall: supervised accuracy',
        'is worse, Quartz propagation fails early and neither NVT production',
        'completes. Water NVE motion partition and spectrum improve in some respects',
        'but exhibit substantial cooling and energy loss. No model or training',
        'configuration was changed in response to these results.',
        '', '## Reproduction and artifacts','',
        '- Exact training/data/evaluation settings: `PROTOCOL.md` and `configs/cmis_*_20261106.json`.',
        '- Full measured metrics: `results.json`; per-split evaluations: `quartz30/` and `water5/`.',
        '- Curves and tables for each successful/failed physical attempt are retained below those folders.',
        '- Server run root: `/data1/wyd/research/foundation_mlip_adaptation/experiments/cmis`.',
        '- Weights, raw histories and full HDF5 trajectories stay on the server.',
        '- Re-export: `python scripts/export_cmis_results.py --base /path/to/experiments/cmis --out reports/cmis_20261106`.',
        '- The failed NVT requests were not rerun or reported as complete.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();export(a.base,a.out)
