"""Consolidate completed fixed-final runs, including all physical failures."""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conditional_transition.common import ROOT,ARMS
from smdpr.train import write_json

LABELS=dict(original='Original-only',two_step='Two-step',extra_reference='Extra-reference',
            autonomous='Autonomous-successor')


def main():
    report=Path(__file__).resolve().parents[1]/'reports/conditional_transition_20261109'
    figures=report/'figures';figures.mkdir(parents=True,exist_ok=True)
    prediction=[];physics=[];local=[];windows=[];costs=[];curves=[];localsamples=[]
    for system in ['quartz','water']:
        for arm in ARMS:
            run=ROOT/system/arm;ev=run/'evaluation'
            endpoint=json.loads((ev/'validation_endpoint.json').read_text())
            horizons=json.loads((ev/'validation_multistep.json').read_text())
            for h,result in horizons['horizons'].items():
                prediction.append(dict(system=system,arm=arm,steps=int(h),
                    samples=horizons['samples'],**result))
            prediction.append(dict(system=system,arm=arm,steps=1,samples=endpoint['samples'],
                time_fs=30 if system=='quartz' else 5,scope='full_endpoint',
                **{k:v for k,v in endpoint.items() if k in ['position_RMSE_A','velocity_RMSE_A_fs',
                                                         'position_MAE_A','velocity_MAE_A_fs']}))
            for replica in range(2):
                d=ev/f'nve20ps_{replica}'
                p=json.loads((d/'physics.json').read_text())
                summary=json.loads((d/'trajectory.summary.json').read_text())
                df=pd.read_csv(d/'physics.csv');df['system']=system;df['arm']=arm;df['replica']=replica
                df['delta_E_eV']=df.total_E_eV-df.total_E_eV.iloc[0]
                df['reference_delta_E_eV']=df.reference_total_E_eV-df.reference_total_E_eV.iloc[0]
                curves.append(df)
                row=dict(system=system,arm=arm,replica=replica,stable=summary['stable'],
                    failure_reason=summary['failure_reason'],time_ps=p['final_time_ps'],
                    mean_temperature_K=p['mean_temperature_K'],final_temperature_K=p['final_temperature_K'],
                    reference_mean_temperature_K=p['reference_mean_temperature_K'],
                    energy_change_eV=p['energy_change_eV'],reference_energy_change_eV=p['reference_energy_change_eV'],
                    energy_linear_slope_eV_ps=float(np.polyfit(df.time_ps,df.delta_E_eV,1)[0]),
                    ns_day_shared_gpu=summary['ns_day_end_to_end'],peak_cuda_bytes=summary['peak_cuda_bytes'])
                for element,motion in p.get('motion',{}).items():
                    row[element]=motion['mean'];row['reference_'+element]=motion['reference_mean']
                physics.append(row)
                for a,b in [(0,1),(1,5),(5,10),(10,20.01 if system=='quartz' else 20)]:
                    sub=df[(df.time_ps>=a)&(df.time_ps<=b)]
                    if len(sub)<2:continue
                    windows.append(dict(system=system,arm=arm,replica=replica,start_ps=a,end_ps=b,
                        sampled_start_ps=float(sub.time_ps.iloc[0]),sampled_end_ps=float(sub.time_ps.iloc[-1]),
                        mean_temperature_K=float(sub.temperature_K.mean()),
                        delta_KE_eV=float(sub.KE_eV.iloc[-1]-sub.KE_eV.iloc[0]),
                        delta_PE_eV=float(sub.PE_eV.iloc[-1]-sub.PE_eV.iloc[0]),
                        delta_E_eV=float(sub.total_E_eV.iloc[-1]-sub.total_E_eV.iloc[0])))
            for bank in ['reference','r3opt']:
                df=pd.read_csv(ev/f'local_{bank}.csv');df['arm']=arm;localsamples.append(df)
                for category,groups in [('all',[('all',df)]),
                    ('temperature',df.groupby(pd.cut(df.temperature_K,[0,200,250,300,350,500,3000],include_lowest=True),observed=True)),
                    ('window',df.groupby('window'))]:
                    for group,sub in groups:
                        de=sub.signed_energy_discrepancy_eV
                        local.append(dict(system=system,arm=arm,bank=bank,stratification=category,
                            group=str(group),samples=len(sub),signed_energy_mean_eV=float(de.mean()),
                            energy_MAE_eV=float(abs(de).mean()),energy_RMSE_eV=float(np.sqrt(np.mean(de**2))),
                            q_RMSE_mean_A=float(sub.q_RMSE_A.mean()),v_RMSE_mean_A_fs=float(sub.v_RMSE_A_fs.mean())))
            status=json.loads((run/'status.json').read_text());progress=json.loads((run/'progress.json').read_text())
            cost=dict(system=system,arm=arm,updates=status['optimizer_steps'],
                supervised_transition_targets=status['supervised_transition_targets'],
                original_target_draws=status['original_draws'],extra_target_draws=status['extra_draws'],
                training_wall_seconds_at_last_progress=progress['elapsed_seconds'],
                labels=0,reference_microsteps=0,oracle_wall_seconds=0.,neural_collection_steps=0,
                neural_collection_wall_seconds=0.,collection_failed_rollouts=0,reference_failed_queries=0)
            for r in [1,2]:
                pool=run/f'collection_round{r}'
                if not (pool/'labels.h5').exists():continue
                with h5py.File(pool/'labels.h5') as f,h5py.File(pool/'visited_states.h5') as g:
                    assert f.attrs['completed'] and g.attrs['completed']
                    cost['labels']+=len(f['q']);cost['reference_microsteps']+=int(f.attrs['reference_micro_steps'])
                    cost['oracle_wall_seconds']+=float(f.attrs['oracle_seconds'])
                    cost['neural_collection_steps']+=int(g.attrs['neural_macro_steps'])
                    cost['neural_collection_wall_seconds']+=float(g.attrs.get('generation_seconds',0))
                    cost['collection_failed_rollouts']+=int(g.attrs.get('failed_rollouts',0))
                    cost['reference_failed_queries']+=len(json.loads(f.attrs['failures']))
            costs.append(cost)
    frames=dict(prediction=pd.DataFrame(prediction),physics=pd.DataFrame(physics),
        local_errors=pd.DataFrame(local),energy_windows=pd.DataFrame(windows),cost=pd.DataFrame(costs),
        energy_temperature_curves=pd.concat(curves,ignore_index=True),
        local_state_errors=pd.concat(localsamples,ignore_index=True))
    for name,df in frames.items():df.to_csv(report/f'{name}.csv',index=False)
    for system in ['quartz','water']:
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex='col')
        for replica in range(2):
            for arm in ARMS:
                df=frames['energy_temperature_curves'];df=df[(df.system==system)&(df.arm==arm)&(df.replica==replica)]
                axes[0,replica].plot(df.time_ps,df.temperature_K,label=LABELS[arm],lw=1)
                axes[1,replica].plot(df.time_ps,df.delta_E_eV,label=LABELS[arm],lw=1)
            axes[0,replica].plot(df.time_ps,df.reference_temperature_K,'k--',label='Reference MD',lw=.8)
            axes[1,replica].plot(df.time_ps,df.reference_delta_E_eV,'k--',label='Reference MD',lw=.8)
            if system=='quartz':axes[1,replica].set_yscale('symlog',linthresh=1)
            axes[0,replica].set_title(f'{system.capitalize()}, initial state {replica+1}')
            axes[0,replica].set_ylabel('Temperature (K)')
            axes[1,replica].set_ylabel('Total energy change (eV)'+ ('; symlog' if system=='quartz' else ''))
            axes[1,replica].set_xlabel('Time (ps)')
        axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/f'{system}_nve.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for replica,ax in enumerate(axes):
        cols=['translation_eV','rotation_eV','internal_eV'];x=np.arange(3)
        selected=frames['physics'];selected=selected[(selected.system=='water')&(selected.replica==replica)]
        values=[selected.iloc[0][['reference_'+c for c in cols]].to_numpy(float)]+[
            selected[selected.arm==arm].iloc[0][cols].to_numpy(float) for arm in ARMS]
        for j,(label,ys) in enumerate(zip(['Reference MD']+[LABELS[a] for a in ARMS],values)):
            ax.bar(x+(j-2)*.16,ys,width=.15,label=label)
        ax.set_xticks(x,['Translation','Rotation','Internal']);ax.set_ylabel('Mean molecular KE (eV)')
        ax.set_title(f'Water, initial state {replica+1}')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figures/'water_motion.png',dpi=180);plt.close(fig)
    failures=[r for r in physics if not r['stable']]
    write_json(report/'results_status.json',dict(training_arms_completed=8,updates_per_arm=25000,
        validation_analysis_completed=8,nve_trajectories_requested=16,nve_full_duration_completed=16-len(failures),
        failed_nve=failures,test_evaluated=False,long_NVT_started=False,
        scientific_outcome='Autonomous-successor did not improve long-term dynamics under this fixed protocol.'))
    print(report,flush=True)


if __name__=='__main__':main()
