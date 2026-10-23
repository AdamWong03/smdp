"""R1/R2/frozen R3-B/BKS physics tables, retaining all failures and mixed outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_tasks import energy_windows


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def get(obj, *keys):
    for k in keys:
        if not isinstance(obj, dict): return None
        obj=obj.get(k)
    return obj


def main():
    p=argparse.ArgumentParser()
    for name in ['r1','r2','current']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); out=a.current
    roots={'R1':a.r1,'R2':a.r2/'reanalysis_v1','R3-B':a.current}
    trajectories={'R1':a.r1,'R2':a.r2,'R3-B':a.current}
    rows=[]; windows=[]; status={}
    def add(run,metric,values):
        rows.append(dict(run=run,metric=metric,**values))
    for run in ['nve_test_09','nve_test_10','nvt']:
        m={n:read(root/f'{run}_analysis/metrics.json') for n,root in roots.items()}
        summaries={n:read(root/f'{run}.summary.json') for n,root in trajectories.items()}
        status[run]=dict(trajectory_completed=bool(m['R3-B'].get('trajectory_completed',False)),
                        physical_analysis_completed=bool(m['R3-B'].get('physical_analysis_completed',False)),
                        production_ps=summaries['R3-B'].get('production_ps'),
                        failure_reason=summaries['R3-B'].get('failure_reason'),
                        analysis_failure_reason=m['R3-B'].get('analysis_failure_reason'))
        def metric(label,*keys,refkeys=None):
            vals={n:get(v,*keys) for n,v in m.items()}
            vals['BKS']=get(m['R3-B'],*(refkeys or tuple('reference' if k=='SMDP' else k for k in keys)))
            add(run,label,vals)
        for key in ['temperature_K','kinetic_energy_eV','potential_energy_eV']:
            for stat in ['mean','std']:metric(key+'_'+stat,key,'SMDP',stat)
        add(run,'potential_Wasserstein_eV',{**{n:get(v,'potential_energy_eV','wasserstein') for n,v in m.items()},'BKS':0.})
        for key in ['energy_change_eV','total_energy_drift_meV_atom_ps']:metric(key,key,'SMDP')
        add(run,'minimum_distance_A',{**{n:get(v,'minimum_distance_A') for n,v in m.items()},'BKS':None})
        for z in ['8','14']:metric('species_KE_Z'+z,'element_kinetic_energy_eV',z,'SMDP','mean')
        for cadence in ['macro_16fs','queried_1fs']:
            for z in ['8','14']:
                vals={n:get(v,'VDOS',cadence,'elements',z,'overlap') for n,v in m.items()}
                add(run,cadence+'_VDOS_Z'+z,{**vals,'BKS':1.})
        for pair in ['14-8','8-8']:
            add(run,'RDF_RMSE_'+pair,{**{n:get(v,'RDF',pair,'RMSE') for n,v in m.items()},'BKS':0.})
            for key in ['peak_position_A','peak_height']:metric('RDF_'+pair+'_'+key,'RDF',pair,key,'SMDP')
        for key in ['ns_day_end_to_end','model_seconds','neighbor_seconds','elapsed_s','peak_gpu_bytes']:
            add(run,key,{**{n:v.get(key) for n,v in summaries.items()},'BKS':None})
        if run!='nvt':
            fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
            for name,root in roots.items():
                path=root/f'{run}_analysis/thermodynamics.csv'
                if not path.exists():continue
                d=pd.read_csv(path)
                for prefix in ['SMDP','reference']:d[prefix+'_KE_eV']=d[prefix+'_total_eV']-d[prefix+'_PE_eV']
                windows.extend(dict(model=name,run=run,**w) for w in energy_windows(d))
                for prefix,label in [('SMDP',name),('reference','BKS')]:
                    if prefix=='reference' and name!='R3-B':continue
                    e=d[prefix+'_total_eV'].to_numpy(); delta=e-e[0]
                    for key,val in [('total_energy_RMS_change_eV',np.sqrt(np.mean(delta**2))),
                                    ('total_energy_min_change_eV',delta.min()),('total_energy_max_change_eV',delta.max())]:
                        existing=next((r for r in rows if r['run']==run and r['metric']==key),None)
                        if existing is None:existing=dict(run=run,metric=key);rows.append(existing)
                        existing[label]=float(val)
                    for ax,key in zip(axes,['KE','PE','total']):
                        y=d[prefix+'_'+key+'_eV'];ax.plot(d.time_fs/1000,y-y.iloc[0],label=label)
                        ax.set_ylabel('delta '+key+' (eV)')
            for ax in axes:ax.legend();ax.grid(alpha=.2)
            axes[-1].set_xlabel('ps');fig.tight_layout();fig.savefig(out/f'{run}_comparison.png',dpi=160);plt.close(fig)
    for kind in ['macro_16fs','queried_1fs']:
        fig,axes=plt.subplots(2,1,figsize=(8,6))
        for name,root in roots.items():
            path=root/'nvt_analysis'/f'vdos_{kind}.csv'
            if not path.exists():continue
            d=pd.read_csv(path)
            for ax,z in zip(axes,[8,14]):
                s=d[d.element==z];ax.plot(s.frequency_THz,s.SMDP,label=name)
                if name=='R3-B':ax.plot(s.frequency_THz,s.reference,label='BKS',c='k')
                ax.set(title=f'Z={z} {kind}',xlabel='THz',xlim=(0,31.25 if kind=='macro_16fs' else 80));ax.legend()
        fig.tight_layout();fig.savefig(out/f'nvt_{kind}_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for name,root in roots.items():
        pe=root/'nvt_analysis/thermodynamics.csv'; rdf=root/'nvt_analysis/rdf.csv'
        if pe.exists():
            d=pd.read_csv(pe);axes[0].hist(d.SMDP_PE_eV,bins=40,density=True,histtype='step',label=name)
            if name=='R3-B':axes[0].hist(d.reference_PE_eV,bins=40,density=True,histtype='step',label='BKS')
        if rdf.exists():
            d=pd.read_csv(rdf)
            for ax,pair in zip(axes[1:],['14-8','8-8']):
                s=d[d.pair==pair];ax.plot(s.r_A,s.SMDP,label=name)
                if name=='R3-B':ax.plot(s.r_A,s.reference,label='BKS',c='k')
                ax.set(title=pair,xlabel='Angstrom')
    axes[0].set_xlabel('BKS potential energy (eV)')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/'nvt_structure_comparison.png',dpi=160);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'physical_comparison.csv',index=False)
    (out/'physical_comparison.json').write_text(json.dumps(rows,indent=2))
    (out/'physical_run_status.json').write_text(json.dumps(status,indent=2))
    (out/'nve_window_comparison.json').write_text(json.dumps(windows,indent=2))
    complete=all(v['trajectory_completed'] and v['physical_analysis_completed'] for v in status.values())
    lines=['# R3-B 固定 checkpoint 正式物理评价','',f'全部请求完成：{complete}',
        '未重新训练或筛选模型。状态按实际轨迹和后处理记录，而不是目标长度判断。',
        '轨迹/后处理状态：','```json',json.dumps(status,indent=2),'```','',
        '## R1 / R2 / R3-B / BKS 同协议对照','',
        '| Run | Metric | R1 | R2 | R3-B | BKS |','|---|---|---:|---:|---:|---:|']
    for row in rows:
        cells=[row['run'],row['metric']]+['未测/不可用' if row.get(n) is None else f'{row[n]:.8g}' for n in ['R1','R2','R3-B','BKS']]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','## 口径与结论边界','',
        '能量和温度对照沿用 2000 个均匀生产帧，RDF 使用 512 帧；完整时间范围的 KE/PE/总能量曲线同时保留，不能将首末能量抵消视为守恒。时间窗指标见 nve_window_comparison.json。',
        'VDOS 使用既有窗口/归一化，16 fs 自主端点与 1 fs 神经查询分别报告；1 fs 采样本身不证明高频正确。',
        'NVT 维持 300 K、100 fs CSVR 耦合时间、1.5 ns 热化、15 ns 生产；边界恒温间隔16 fs，不是参考MD的1 fs。thermostat.json记录网络与恒温器的动能交换。',
        '所有数值稳定/完成状态仅表示通过原有停止条件，不等于能量守恒或正确平衡分布。若未完成，短轨迹统计不可作为同长度正式对照。',
        '生产吞吐包含图更新、网络、恒温与输出；另存 production_efficiency_benchmark.json 的端点/密集查询短时计时，不混作端到端15 ns吞吐。历史运行共享GPU负载可能不同，不据此宣称严格加速。',
        '两条NVE是独立初态；NVT时间块波动并非独立模拟置信区间。逐项物理改进与退步均在表中保留。本轮不自动进入R4或调节恒温器。']
    (out/'PHYSICS_COMPARISON.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
