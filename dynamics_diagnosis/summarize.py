"""Finite report builder: measurements first, no inferred model selection."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynamics_diagnosis.common import ARTIFACTS,REPORT,DATA,checkpoints,source,save,initial_sources
from physics.metrics import reference_vdos


def group_summary(frame,keys,values):
    records=[]
    for group,part in frame.groupby(keys,dropna=False):
        if not isinstance(group,tuple):group=(group,)
        row=dict(zip(keys,group));row['samples']=len(part)
        # Four contiguous time blocks per start, not N independent atoms/frames.
        local=part.copy()
        times=local.time_ps
        duration=max(float(times.max()-times.min()),1e-9)
        local['time_block']=np.minimum(3,np.floor(4*(times-times.min())/duration).astype(int))
        for key in values:
            row[key+'_mean']=float(part[key].mean())
            row[key+'_std']=float(part[key].std(ddof=0))
            means=local.groupby(['replica','time_block'])[key].mean()
            row[key+'_block_mean_std']=float(means.std(ddof=0))
        row['starts']=int(part.replica.nunique())
        records.append(row)
    return pd.DataFrame(records)


def spectra():
    rows=[]
    for system in ['quartz','water']:
        h=30 if system=='quartz' else 5
        windows=[(0,1),(1,5),(5,10),(10,20),(20,50)]
        for name in checkpoints(system):
            path=ARTIFACTS/system/name/'validation_50ps.h5'
            if not (path.parent/'completed.json').exists():continue
            with h5py.File(path) as f:
                if not f.attrs.get('completed'):continue
                z=f['z'][:]
                for replica,(ref,initial) in enumerate(initial_sources(system)):
                    with h5py.File(ref) as raw:
                        stride=round(h/float(raw.attrs['output_interval_fs']))
                        for a,b in windows:
                            begin=round(a*1000/h);end=round(b*1000/h)+1
                            end=min(end,(len(raw['time_fs'])-initial-1)//stride+1)
                            if end-begin<round(1000/h):continue
                            v=f['v'][begin:end,replica].astype(float)
                            rv=raw['velocities_A_per_fs'][initial+begin*stride:initial+end*stride:stride].astype(float)
                            for element in np.unique(z):
                                av=v[:,z==element];bv=rv[:,z==element]
                                frequency,cp=reference_vdos(av,h);_,rp=reference_vdos(bv,h)
                                # Uncentered atom-averaged VACF, normalized only
                                # after retaining raw amplitude for dissipation.
                                lags=min(round(500/h),len(av)-1)
                                ca=np.array([(av[:len(av)-k]*av[k:]).sum(-1).mean() for k in range(lags+1)])
                                cr=np.array([(bv[:len(bv)-k]*bv[k:]).sum(-1).mean() for k in range(lags+1)])
                                row=dict(system=system,model=name,replica=replica,window=f'{a}-{b}',element=int(element),
                                    actual_start_ps=begin*h/1000,actual_stop_ps=(end-1)*h/1000,
                                    VDOS_overlap=float(np.minimum(cp,rp).sum()),
                                    normalized_VACF_RMSE=float(np.sqrt(np.mean((ca/ca[0]-cr/cr[0])**2))),
                                    velocity_power_ratio=float(ca[0]/cr[0]),nyquist_THz=500/h,
                                    warning='NVE nonstationary window, not equilibrium density of states')
                                rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(REPORT/'spectral_windows.csv',index=False)
    spectral_figures(frame)


def spectral_figures(frame):
    for system,part in frame.groupby('system'):
        elements=sorted(part.element.unique())
        fig,axes=plt.subplots(2,len(elements),figsize=(10,7),sharex=True)
        for column,element in enumerate(elements):
            for name,g in part[part.element==element].groupby('model',sort=False):
                means=g.groupby('window',sort=False).mean(numeric_only=True)
                axes[0,column].plot(means.index,means.VDOS_overlap,'o-',label=name)
                axes[1,column].plot(means.index,means.velocity_power_ratio,'o-')
            axes[0,column].set(title=f'Z={element}',ylabel='Window VDOS overlap')
            axes[1,column].axhline(1,color='k',lw=.6)
            axes[1,column].set(xlabel='NVE window (ps)',ylabel='Raw velocity power / reference')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(system+' — nonstationary macro-state spectra and amplitude')
        fig.tight_layout();fig.savefig(REPORT/'figures'/f'{system}_spectral_attenuation.png',dpi=180);plt.close(fig)


def main(reuse_spectra=False):
    REPORT.mkdir(exist_ok=True,parents=True)
    figures=REPORT/'figures';figures.mkdir(exist_ok=True)
    comparisons=[];series=[];errors=[]
    for system in ['quartz','water']:
        for name,checkpoint in checkpoints(system).items():
            directory=ARTIFACTS/system/name
            if not (directory/'completed.json').exists():continue
            p=directory/'time_series.csv'
            if p.exists():
                df=pd.read_csv(p);series.append(df)
                for replica,part in df.groupby('replica'):
                    for horizon in [1,5,10,20,50]:
                        stop=part[part.time_ps<=horizon+.02]
                        if stop.empty:continue
                        row=dict(system=system,model=name,replica=replica,horizon_ps=horizon,
                            actual_horizon_ps=float(stop.time_ps.iloc[-1]),checkpoint=str(checkpoint),
                            mean_temperature_K=float(stop.temperature_K.mean()),
                            final_temperature_K=float(stop.temperature_K.iloc[-1]),
                            energy_change_eV=float(stop.total_E_eV.iloc[-1]-stop.total_E_eV.iloc[0]),
                            energy_drift_eV_ps=float(np.polyfit(stop.time_ps,stop.total_E_eV,1)[0]),
                            KE_change_eV=float(stop.KE_eV.iloc[-1]-stop.KE_eV.iloc[0]),
                            PE_change_eV=float(stop.PE_eV.iloc[-1]-stop.PE_eV.iloc[0]))
                        for key in ['position_RMSE_A','velocity_RMSE_A_fs','reference_temperature_K',
                                    'translation_eV','rotation_eV','internal_eV']:
                            if key in stop:
                                row['final_'+key]=float(stop[key].iloc[-1])
                                row['mean_'+key]=float(stop[key].mean())
                        comparisons.append(row)
            for p in directory.glob('local_*.csv'):errors.append(pd.read_csv(p))
    if comparisons:pd.DataFrame(comparisons).to_csv(REPORT/'comparison.csv',index=False)
    if series:
        whole=pd.concat(series,ignore_index=True)
        whole.to_csv(REPORT/'time_series.csv',index=False)
        water=whole[whole.system=='water']
        water.to_csv(REPORT/'water_motion.csv',index=False)
        for system in ['quartz','water']:
            df=whole[whole.system==system]
            colors=dict(zip(checkpoints(system),plt.get_cmap('tab10').colors))
            fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
            for name,part in df.groupby('model',sort=False):
                for replica,line in part.groupby('replica'):
                    style='-' if replica==0 else '--'
                    axes[0].plot(line.time_ps,line.temperature_K,style,color=colors[name],label=f'{name} start{replica}')
                    axes[1].plot(line.time_ps,line.total_E_eV-line.total_E_eV.iloc[0],style,color=colors[name])
                    axes[2].plot(line.time_ps,line.PE_eV-line.PE_eV.iloc[0],style,color=colors[name])
            for replica,line in df[df.model=='r3opt'].groupby('replica'):
                style='-' if replica==0 else '--'
                for ax,key in zip(axes,['temperature_K','total_E_eV','PE_eV']):
                    ref=line['reference_'+key]
                    values=ref if key=='temperature_K' else ref-ref.iloc[0]
                    ax.plot(line.time_ps,values,style,color='k',lw=1.3,alpha=.7,
                            label=f'MD start{replica}' if key=='temperature_K' else None)
            axes[0].set_ylabel('Temperature (K)');axes[0].legend(fontsize=6,ncol=2)
            axes[1].set_ylabel('Total energy change (eV)');axes[2].set_ylabel('PE change (eV)')
            axes[-1].set_xlabel('Time (ps)')
            fig.suptitle(system+' — common validation starts, no thermostat')
            fig.tight_layout();fig.savefig(figures/f'{system}_energy_temperature.png',dpi=180);plt.close(fig)
        if len(water):
            fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
            for name,part in water[water.replica==0].groupby('model',sort=False):
                for ax,key in zip(axes,['translation_eV','rotation_eV','internal_eV']):
                    ax.plot(part.time_ps,part[key],label=name);ax.set_ylabel(key)
            reference=water[(water.replica==0)&(water.model=='r3opt')]
            for ax,key in zip(axes,['translation_eV','rotation_eV','internal_eV']):
                ax.plot(reference.time_ps,reference['reference_'+key],color='k',lw=1.3,label='MD')
            axes[0].legend(fontsize=7,ncol=2);axes[-1].set_xlabel('Time (ps)')
            fig.tight_layout();fig.savefig(figures/'water_molecular_motion.png',dpi=180);plt.close(fig)
        windowrows=[]
        for group,part in whole.groupby(['system','model','replica']):
            for a,b in [(0,1),(1,5),(5,10),(10,20),(20,50)]:
                sub=part[(part.time_ps>=a-.015)&(part.time_ps<=b+.015)]
                if len(sub)<2:continue
                row=dict(zip(['system','model','replica'],group),window=f'{a}-{b}')
                for key in ['KE_eV','PE_eV','total_E_eV']:
                    row[key+'_change']=float(sub[key].iloc[-1]-sub[key].iloc[0])
                row['mean_temperature_K']=float(sub.temperature_K.mean())
                windowrows.append(row)
        pd.DataFrame(windowrows).to_csv(REPORT/'energy_windows.csv',index=False)
    if errors:
        df=pd.concat(errors,ignore_index=True)
        df['signed_KE_discrepancy_eV']=df.model_delta_KE_eV-df.MD_delta_KE_eV
        df['signed_PE_discrepancy_eV']=df.model_delta_PE_eV-df.MD_delta_PE_eV
        np.testing.assert_allclose(df.signed_KE_discrepancy_eV+df.signed_PE_discrepancy_eV,
                                   df.signed_energy_discrepancy_eV,atol=1e-9)
        df.to_csv(REPORT/'energy_transition.csv',index=False)
        water=df[df.system=='water']
        water.to_csv(REPORT/'water_motion_transition.csv',index=False)
        values=['signed_energy_discrepancy_eV','model_delta_E_eV','MD_delta_E_eV',
                'model_delta_KE_eV','model_delta_PE_eV','signed_KE_discrepancy_eV',
                'signed_PE_discrepancy_eV','q_RMSE_A','v_RMSE_A_fs',
                'temperature_K','kinetic_error_cross_eV','kinetic_error_quadratic_eV']
        group_summary(df,['system','model','state_source','origin','window'],values).to_csv(REPORT/'local_by_window.csv',index=False)
        # Uniform sampling inside unequal-duration windows is intentional.
        # Never confuse their equally weighted mean with the trajectory's
        # time-weighted energy drift.
        estimates=[]
        for group,part in df[df.own_state].groupby(['system','model','replica']):
            h=30 if group[0]=='quartz' else 5
            row=dict(zip(['system','model','replica'],group))
            estimate=0.
            for window,local in part.groupby('window'):
                a,b=map(float,window.split('-'))
                contribution=float(local.model_delta_E_eV.mean())*(b-a)*1000/h
                row['window_'+window+'_estimated_delta_E_eV']=contribution
                estimate+=contribution
            row['estimated_50ps_energy_change_eV']=estimate
            estimates.append(row)
        pd.DataFrame(estimates).to_csv(REPORT/'local_time_weighted_energy.csv',index=False)
        df['temperature_bin']=pd.cut(df.temperature_K,[-np.inf,150,225,275,325,400,np.inf],
            labels=['<150','150-225','225-275','275-325','325-400','>=400']).astype(str)
        group_summary(df,['system','model','state_source','temperature_bin'],values).to_csv(REPORT/'local_by_temperature.csv',index=False)
        geom=[]
        for system,part in df.groupby('system'):
            reference=part[part.state_source=='reference'].drop_duplicates(['replica','frame'])
            for feature in ['min_distance_A','mean_edge_distance_A']+(['bond_std_A'] if system=='water' else []):
                quantiles=np.unique(np.quantile(reference[feature],[0,.25,.5,.75,1]))
                bins=np.r_[-np.inf,quantiles[1:-1],np.inf]
                grouped=part.copy();grouped['geometry_bin']=pd.cut(grouped[feature],bins=bins).astype(str)
                s=group_summary(grouped,['system','model','state_source','geometry_bin'],values)
                s['feature']=feature;geom.append(s)
        pd.concat(geom,ignore_index=True).to_csv(REPORT/'local_by_geometry.csv',index=False)
        if len(water):
            group_summary(water,['system','model','state_source','window'],
                ['model_delta_translation_eV','MD_delta_translation_eV','error_translation_eV',
                 'model_delta_rotation_eV','MD_delta_rotation_eV','error_rotation_eV',
                 'model_delta_internal_eV','MD_delta_internal_eV','error_internal_eV']).to_csv(REPORT/'water_local_partition.csv',index=False)
            associations=[]
            for group,g in water.groupby(['model','state_source','window']):
                controls=g[['temperature_K','min_distance_A','bond_std_A']].to_numpy()
                controls=(controls-controls.mean(0))/np.maximum(controls.std(0),1e-12)
                controls=np.c_[np.ones(len(g)),controls]
                target=g.signed_energy_discrepancy_eV.to_numpy()
                residual=target-controls@np.linalg.lstsq(controls,target,rcond=None)[0]
                row=dict(zip(['model','state_source','window'],group),samples=len(g))
                for key in ['error_translation_eV','error_rotation_eV','error_internal_eV']:
                    value=g[key].to_numpy()
                    adjusted=value-controls@np.linalg.lstsq(controls,value,rcond=None)[0]
                    row[key+'_correlation']=float(np.corrcoef(target,value)[0,1])
                    row[key+'_T_geometry_adjusted_correlation']=float(np.corrcoef(residual,adjusted)[0,1])
                associations.append(row)
            pd.DataFrame(associations).to_csv(REPORT/'water_energy_motion_associations.csv',index=False)
        for system,part in df.groupby('system'):
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for name,g in part.groupby('model'):
                own=g[(g.origin=='reference')|(g.own_state)]
                for origin,local in own.groupby('origin'):
                    axes[0].scatter(local.temperature_K,local.signed_energy_discrepancy_eV,
                                    s=5,alpha=.18,label=f'{name} {origin}')
                common=g[g.state_source=='k3_stage2']
                if len(common):
                    window=common.groupby('window',sort=False).signed_energy_discrepancy_eV.mean()
                    axes[1].plot(window.index,window.values,'o-',label=name)
            axes[0].axhline(0,color='k',lw=.6);axes[1].axhline(0,color='k',lw=.6)
            axes[0].set(xlabel='Input temperature (K)',ylabel='Signed energy-increment discrepancy (eV)')
            axes[1].set(xlabel='Common K3 stage2 state window (ps)',ylabel='Same-state mean discrepancy (eV)')
            axes[1].legend(fontsize=7)
            fig.suptitle(system+' — temperature/state associations, not causal layer attribution')
            fig.tight_layout();fig.savefig(figures/f'{system}_local_error.png',dpi=180);plt.close(fig)
    roundtrips=[]
    for system in ['quartz','water']:
        for path in (ARTIFACTS/system).glob('*/roundtrip.json'):
            roundtrips.append(dict(system=system,model=path.parent.name,**json.loads(path.read_text())))
    pd.DataFrame(roundtrips).to_csv(REPORT/'roundtrip.csv',index=False)
    if reuse_spectra:
        spectral_figures(pd.read_csv(REPORT/'spectral_windows.csv'))
    else:
        spectra()
    status={}
    for system in ['quartz','water']:
        p=ARTIFACTS/system/'status.json'
        status[system]=json.loads(p.read_text()) if p.exists() else dict(state='incomplete')
    save(REPORT/'status.json',dict(completed=all(s.get('state')=='diagnosis_completed' for s in status.values()),
        systems=status,no_training=True,no_test_selection=True,
        uncertainty='contiguous time blocks within two starts; not independent frame replicates'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-spectra',action='store_true',
                        help='Replot the already completed frozen-state spectral table')
    args=parser.parse_args()
    main(reuse_spectra=args.reuse_spectra)
