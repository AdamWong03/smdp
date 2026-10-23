"""Build bounded report artifacts from completed, immutable R1/R2 evaluations."""
import argparse
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_tasks import energy_windows, save_json
from summarize_experiment import compare_runs


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--r1',type=Path,required=True)
    parser.add_argument('--r2',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--publish',type=Path,help='Optional bounded public report destination; never copies weights or trajectories')
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    code_root=Path(__file__).resolve().parent
    source_files=['analyze_quartz.py','analysis_tasks.py','analyze_training_stages.py','verify_bks_runtime.py',
        'verify_csvr_replay.py','report_r2_reanalysis.py','summarize_experiment.py','physics/bks.py','physics/metrics.py',
        'train_quartz.py','training_objectives.py','smdp_neural.py','rollout_quartz.py']
    save_json(out/'analysis_code.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code_root,text=True).strip(),
        note='Source hashes identify working-tree analysis code, including uncommitted changes at report generation. No model/training changes in this reanalysis.',
        sha256={name:hashlib.sha256((code_root/name).read_bytes()).hexdigest() for name in source_files}))
    checkpoints=[args.r1/'best_rollout.pt']+[args.r2/f'best_{name}.pt' for name in ['segment','rollout4','rollout8','rollout16','rollout']]
    save_json(out/'checkpoint_identity.json',{str(p):dict(size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in checkpoints})
    names=['nve_test_09','nve_test_10','nvt']
    for name in names:
        if not read(out/(name+'_analysis')/'metrics.json')['physical_analysis_completed']:
            raise RuntimeError(f'{name}: analysis is not complete; do not publish a complete comparison')
    compare_runs(args.r1,args.r2,report_output=out,analysis_root=out)
    windows=[];energy_metrics=[]
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    for col,name in enumerate(names[:2]):
        for label,root in [('R1',args.r1),('R2',out)]:
            data=pd.read_csv(root/(name+'_analysis')/'thermodynamics.csv')
            for prefix in ['SMDP','reference']:
                data[prefix+'_KE_eV']=data[prefix+'_total_eV']-data[prefix+'_PE_eV']
            for w in energy_windows(data):
                windows.append(dict(run=name,model=label,**w))
            for prefix,legend in [('SMDP',label),('reference','BKS')]:
                if prefix=='reference' and label=='R1':continue
                e=data[prefix+'_total_eV'].to_numpy();delta=e-e[0]
                energy_metrics.append(dict(run=name,model=legend,end_change_eV=float(delta[-1]),
                    rms_change_eV=float(np.sqrt(np.mean(delta**2))),minimum_change_eV=float(delta.min()),
                    maximum_change_eV=float(delta.max()),range_eV=float(np.ptp(e)),
                    minimum_energy_time_ps=float(data.time_fs.iloc[int(delta.argmin())]/1000),
                    maximum_energy_time_ps=float(data.time_fs.iloc[int(delta.argmax())]/1000),
                    fitted_drift_meV_atom_ps=float(np.polyfit(data.time_fs/1000,e,1)[0]*1000/162)))
                for row,key in enumerate(['KE','PE','total']):
                    y=data[prefix+'_'+key+'_eV'];axes[row,col].plot(data.time_fs/1000,y-y.iloc[0],label=legend,alpha=.8)
                    axes[row,col].set_ylabel(f'delta {key} (eV)');axes[row,col].legend()
        axes[0,col].set_title(name);axes[-1,col].set_xlabel('ps')
    fig.tight_layout();fig.savefig(out/'nve_energy_comparison.png',dpi=170);plt.close(fig)
    save_json(out/'nve_window_comparison.json',windows);save_json(out/'nve_energy_summary.json',energy_metrics)

    stages={k:read(out/'stages'/(k+'.json')) for k in ['R1_final','segment','rollout4','rollout8','rollout16']}
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,d in stages.items():
        for step in ['1','2','16']:
            v=d['steps'][step]
            rows.append(dict(stage=name,epoch_zero_based=d['checkpoint_epoch_zero_based'],step=int(step),
                **{k:v[k] for k in ['samples','position_RMSE_A','velocity_RMSE_A_fs','KE_bias_eV']},
                kinematics_RMSE_A_fs=d['kinematics_RMSE_A_fs'],join_RMS_A_fs=d['join_derivative_jump_RMS_A_fs']))
        t=np.arange(1,17)*16
        for ax,key in zip(axes,['position_RMSE_A','velocity_RMSE_A_fs','KE_bias_eV']):
            ax.plot(t,[d['steps'][str(i)][key] for i in range(1,17)],label=name)
            ax.set(xlabel='autonomous validation horizon (fs)',ylabel=key)
    axes[-1].axhline(0,c='k',lw=.5);axes[0].legend();fig.tight_layout();fig.savefig(out/'stage_validation.png',dpi=170);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'stage_comparison.csv',index=False);save_json(out/'stage_comparison.json',rows)
    history=[json.loads(s) for s in (args.r2/'history.jsonl').read_text().splitlines()]
    records=[]
    for phase in ['segment','rollout4','rollout8','rollout16']:
        group=[v for v in history if v['phase']==phase];best=min(group,key=lambda x:x['validation']['loss'])
        records.append(dict(phase=phase,epochs=len(group),best_epoch_zero_based=best['epoch'],
            first_validation=group[0]['validation'],best_validation=best['validation'],last_validation=group[-1]['validation']))
        if phase!='segment':
            count=int(phase.replace('rollout',''))-1
            records[-1]['gradient_scope']=dict(autonomous_statistic_times=count,differentiable_times=2,
                species_kinetic_connected_fraction=2/count,vacf_connected_entry_fraction=1-(count-2)**2/count**2,
                note='Connected fractions, not measured gradient norms or a causal estimate; earlier prefix tensors are detached.')
    save_json(out/'training_stage_history.json',records)
    fig,ax=plt.subplots(figsize=(8,4))
    for phase in ['rollout4','rollout8','rollout16']:
        vals=[v for v in history if v['phase']==phase]
        ax.plot([v['epoch'] for v in vals],[v['validation']['loss'] for v in vals],marker='.',label=phase)
    ax.set(xlabel='epoch (zero-based within stage)',ylabel='same 16-step validation objective');ax.legend()
    fig.tight_layout();fig.savefig(out/'rollout_objective_history.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,cadence in enumerate(['macro_16fs','queried_1fs']):
        for label,root in [('R1',args.r1),('R2',out)]:
            data=pd.read_csv(root/'nvt_analysis'/f'vdos_{cadence}.csv')
            for row,z in enumerate([8,14]):
                d=data[data.element==z];axes[row,col].plot(d.frequency_THz,d.SMDP,label=label)
                if label=='R2':axes[row,col].plot(d.frequency_THz,d.reference,label='BKS',c='k',alpha=.65)
                axes[row,col].set(xlim=(0,31.25 if col==0 else 80),xlabel='THz',ylabel='normalized VDOS',title=f'Z={z}, {cadence}')
                axes[row,col].legend()
    fig.tight_layout();fig.savefig(out/'nvt_vdos_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for label,root in [('R1',args.r1),('R2',out)]:
        data=pd.read_csv(root/'nvt_analysis'/'thermodynamics.csv')
        axes[0].hist(data.SMDP_PE_eV,bins=40,density=True,histtype='step',label=label)
        if label=='R2':axes[0].hist(data.reference_PE_eV,bins=40,density=True,histtype='step',label='BKS')
        rdf=pd.read_csv(root/'nvt_analysis'/'rdf.csv')
        for ax,pair in zip(axes[1:],['14-8','8-8']):
            d=rdf[rdf.pair==pair];ax.plot(d.r_A,d.SMDP,label=label)
            if label=='R2':ax.plot(d.r_A,d.reference,label='BKS',c='k',alpha=.6)
            ax.set(title=pair,xlabel='Angstrom',ylabel='g(r)')
    axes[0].set(xlabel='BKS potential energy (eV)',ylabel='density')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/'nvt_structure_comparison.png',dpi=170);plt.close(fig)
    balance=pd.read_csv(out/'nvt_analysis/thermostat_blocks.csv')
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for key,label in [('pre_temperature_K','pre CSVR'),('post_temperature_K','post CSVR')]:
        axes[0].plot(balance.time_ps/1000,balance[key],label=label)
    axes[0].axhline(300,c='k',ls='--',label='target');axes[0].set_ylabel('block mean T (K)');axes[0].legend()
    axes[1].plot(balance.time_ps/1000,balance.network_delta_KE_eV,label='network')
    axes[1].plot(balance.time_ps/1000,balance.csvr_delta_KE_eV,label='CSVR')
    axes[1].axhline(0,c='k',lw=.5);axes[1].set(xlabel='production time (ns)',ylabel='mean KE change / step (eV)');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'thermostat_balance.png',dpi=170);plt.close(fig)
    write_report(args,out,stages,records,energy_metrics,windows)
    if args.publish:
        publish(out,args.publish)
    print('Report tables and figures completed:',out)


def write_report(args,out,stages,history,energies,windows):
    old=read(args.r1/'nvt_analysis/metrics.json');new=read(out/'nvt_analysis/metrics.json')
    def f(v):return f'{v:.7g}'
    lines=['# R2 物理评价补全与误差定位（冻结 checkpoint，未启动 R3）','',
        f'R1：`{args.r1}`',f'R2：`{args.r2}`',f'本次新增结果：`{out}`','',
        '原始轨迹、checkpoint、失败日志及服务器旧报告均保留。仅补充离线后处理与冻结模型的验证集推断；没有重新训练或生成新的 MD 轨迹。',
        '两条 100 ps NVE 和 15 ns NVT 生产轨迹均已完成；六项物理后处理也均已完成。原来的 `production_available=0` 是缺失分析被误记为缺失轨迹，并非轨迹生成失败。完整状态见 `run_status.json`。','',
        '## 1. 环境修复与评价口径','',
        '原 traceback 停在 LAMMPS `CDLL(libpath)`，缺少 `libmpi.so.12`。`readelf` 显示 LAMMPS 依赖该 SONAME；已有 MPICH runtime 位于 `/data1/wyd/envs/lammps-runtime/lib`，此前 R2 后处理启动命令未传递此目录。恢复 `LD_LIBRARY_PATH` 即可；没有安装新 MPI、创建跨实现软链接或修改 BKS。',
        '运行时报告 MPICH 5.0.2、ABI 18:2:6、LAMMPS 20250722。24 个固定 R1/参考构型重新计算的 BKS 势能与原报告最大偏差 1.73e-11 eV；详情见 `bks_runtime_verification.json`。',
        '使用与 R1 相同的 raw LAMMPS BKS/DSF 能量零点及单位换算。能量/温度对照表使用 2000 个均匀生产段样本；RDF 使用相同 512 帧、180 bins；VDOS 使用相同 16.384 ps 不重叠窗口、VACF 与归一化，15 ns 生产段包含 915 个完整谱窗口，尾部不足一窗部分不进入谱估计，与 R1 相同。1 fs 查询轨迹和 16 fs 自主端点分别评价。能量不是对每个 MD 帧重新求值；时间窗边界值在这组相同采样帧之间插值。最小原子间距沿用 R1 的全保存轨迹统计，包含热化段。',
        'JSON 中 `block_mean_std` 是八个时间块均值的标准差，不是独立轨迹置信区间。VDOS 的 block overlap 与整体平均谱 overlap 不是同一估计量。两个 NVE 初态是独立重复，NVT 时间块不是独立模拟重复。','',
        '## 2. NVE：平均升温减轻，但不满足能量守恒','',
        '| 轨迹 | 方法 | 末端 ΔE / eV | ΔE RMS / eV | ΔE 最小/最大 / eV | 拟合斜率 meV/(atom·ps) |',
        '|---|---|---:|---:|---:|---:|']
    for d in energies:lines.append(f"| {d['run']} | {d['model']} | {f(d['end_change_eV'])} | {f(d['rms_change_eV'])} | {f(d['minimum_change_eV'])} / {f(d['maximum_change_eV'])} | {f(d['fitted_drift_meV_atom_ps'])} |")
    lines+=['','R2 的能量偏离幅度明显小于 R1，但首尾净变化存在耗散和注能相互抵消。尤其 test_10 的全程线性斜率比 R1 更大，不能以首尾接近或平均温度降低声称 NVE 守恒改善到合格。','',
        '| 轨迹 | 模型 | 时间窗 ps | 平均温度 K | ΔKE eV | ΔPE eV | ΔE eV |','|---|---|---|---:|---:|---:|---:|']
    for d in windows:
        s=d['SMDP'];lines.append(f"| {d['run']} | {d['model']} | {d['start_ps']}–{d['end_ps']} | {f(s['temperature_K'])} | {f(s['KE']['change_eV'])} | {f(s['PE']['change_eV'])} | {f(s['total']['change_eV'])} |")
    lines+=['','前 1 ps 两条轨迹同时损失动能和势能，总能量分别下降约 4.07、4.56 eV；不是动能转移到势能。采样能量最小值分别在 3.344、3.248 ps，总能量最高值分别在 17.248、21.200 ps。5–20 ps 又同时增加动能和势能，总能量增加约 4.86、6.62 eV；不是单纯势能释放。20–100 ps 继续出现不规则交换与净变化。证据支持状态依赖的非物理耗散/注能，而非单一恒定漂移；仅凭现有轨迹不能唯一归因到某层网络。',
        'NVE 的 Si/O 谱并没有随能量偏离缩小而改善：两条 R2 的 16 fs O overlap 为约 0.688/0.661（R1 0.734/0.730），Si 为约 0.715/0.719（R1 0.795/0.796）。因此“能量偏差小一些”和“动力学更真实”必须分开判断。完整 RDF、温度和能量逐项表见 `COMPARISON.md`。','',
        '![NVE energy](nve_energy_comparison.png)','',
        '## 3. NVT：完整生产段对照','',
        '| 指标 | BKS | R1 | R2 | R2 − BKS |','|---|---:|---:|---:|---:|']
    for key,stat,label in [('temperature_K','mean','温度均值 K'),('potential_energy_eV','mean','势能均值 eV'),('potential_energy_eV','std','势能标准差 eV')]:
        ref=new[key]['reference'][stat];r1=old[key]['SMDP'][stat];r2=new[key]['SMDP'][stat]
        lines.append(f'| {label} | {f(ref)} | {f(r1)} | {f(r2)} | {f(r2-ref)} |')
    for z,label in [('8','O 总动能 eV'),('14','Si 总动能 eV')]:
        ref=new['element_kinetic_energy_eV'][z]['reference']['mean'];r1=old['element_kinetic_energy_eV'][z]['SMDP']['mean'];r2=new['element_kinetic_energy_eV'][z]['SMDP']['mean']
        lines.append(f'| {label} | {f(ref)} | {f(r1)} | {f(r2)} | {f(r2-ref)} |')
    for cadence in ['macro_16fs','queried_1fs']:
        for z,label in [('8','O'),('14','Si')]:
            r1=old['VDOS'][cadence]['elements'][z]['overlap'];r2=new['VDOS'][cadence]['elements'][z]['overlap']
            lines.append(f'| {cadence} {label} VDOS overlap | 1 | {f(r1)} | {f(r2)} | {f(r2-1)} |')
    for pair in ['14-8','8-8']:
        r1=old['RDF'][pair]['RMSE'];r2=new['RDF'][pair]['RMSE'];lines.append(f'| {pair} RDF RMSE | 0 | {f(r1)} | {f(r2)} | {f(r2)} |')
    lines+=['',f"势能 Wasserstein 距离：R1 {f(old['potential_energy_eV']['wasserstein'])} eV，R2 {f(new['potential_energy_eV']['wasserstein'])} eV。峰位置、峰高、块波动和完整差值见 `COMPARISON.md` 与 `nvt_analysis/metrics.json`。",
        '16 fs 的 Nyquist 上限为 31.25 THz；1 fs 输出的准确性只由其与同采样率 BKS 谱的比较支持，不能由输出频率推断。密集输出包括宏步终点恒温 kick，段内查询不是独立积分得到的真实微步。','',
        '| 谱 | 元素 | R2 全平均谱 overlap | 配对块 overlap 均值 | 配对块 overlap 标准差 | BKS 奇偶块平均谱 overlap |',
        '|---|---|---:|---:|---:|---:|']
    for cadence in ['macro_16fs','queried_1fs']:
        for z in ['8','14']:
            item=new['VDOS'][cadence]['elements'][z]
            lines.append('| '+' | '.join([cadence,z,f(item['overlap']),f(item['block_overlap']['mean']),f(item['block_overlap']['std']),f(item['reference_split_half_overlap'])])+' |')
    lines+=['', '全平均谱的剩余误差远大于 BKS 奇偶时间块之间的差异，不能用有限采样噪声解释。两种采样率下 NVT 谱相对 R1 均明显改善，但 NVE 谱退步；这说明在恒温补偿下取得更好谱形，并不代表无恒温自主动力学已经正确。',
        'RDF 也不是全面改善：NVT O–O RMSE 从约 0.310 降至 0.0569，而 Si–O 从约 0.240 升至 0.570；后者第一峰高于参考约 18.1%。不能用 O–O 的改善代表全部局部结构。','',
        '![NVT spectra](nvt_vdos_comparison.png)','', '![NVT structure](nvt_structure_comparison.png)','',
        '## 4. 为什么 CSVR 后仍只有约 280.8 K','',
        '全生产段逐步统计（937,500 步；与上述 2000 帧统计有轻微采样差异）：pre 温度 277.4715 K，post 温度 280.7778 K。网络每步 ΔKE = −0.06880678 eV，CSVR 每步 ΔKE = +0.06880606 eV。O 分别为 −0.04667084/+0.04667053 eV，Si 为 −0.02213594/+0.02213553 eV。',
        '保存的 alpha 满足 T_post = alpha² T_pre（最大差 6.7e-5 K）。按原随机种子重放包括热化段的全部 1,031,250 步，alpha 最大差 6.52e-8，与 float32 存储精度相容。随机更新的有限样本均值偏差也被完整重现，未发现 CSVR 实现错误。见 `csvr_replay.json`。',
        '令 c=exp(−16/100)，d=平均(T_pre[n]−T_post[n−1])，则稳态均值关系为 μ_post=300+c·d/(1−c)+ε/(1−c)。忽略有限随机样本 ε 时得到 280.9448 K；加入测得 ε=−0.0247164 K 后得到实际约 280.7778 K。网络持续耗散使有限耦合恒温器在低于目标的状态补偿它；这不是正确平衡分布，也不应通过升高目标温度掩盖。',
        '元素动能分配比 R1 更接近参考，但 Si 与 O 仍均偏低且比例不完全正确。这里比较的是相同速度/质心处理下的分元素总动能，不把每个元素独立强制到 300 K。','',
        '![Thermostat balance](thermostat_balance.png)','',
        '## 5. 冻结阶段 checkpoint 的统一验证','',
        '全部 1,250 个验证起点用于单步；其中 1,245 个有完整 256 fs 参考窗口用于多步。仅验证集、无优化器、无测试集选择、没有生成新的物理轨迹。表中误差单位分别为 Å、Å/fs。','',
        '| 阶段 | 1-step q RMSE | 2-step q RMSE | 16-step q RMSE | 16-step v RMSE | 256 fs KE 偏差 eV | 段内运动学 RMSE Å/fs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,d in stages.items():
        lines.append('| '+' | '.join([label,*[f(d['steps'][s]['position_RMSE_A']) for s in ['1','2','16']],f(d['steps']['16']['velocity_RMSE_A_fs']),f(d['steps']['16']['KE_bias_eV']),f(d['kinematics_RMSE_A_fs'])])+' |')
    lines+=['','参考段内位置差分—速度 RMSE 为 8.697e-6 Å/fs。相邻段导数跳变量的参考 RMS 为 4.8261e-4 Å/fs，R1 为 5.5733e-4、R2 为 4.7721e-4；真实加速度也会造成有限差分跳变，因此 RMS 接近参考不等于逐点导数误差消失。详细指标保存在阶段 JSON/CSV。',
        '短期精度与运动学一致性从 rollout4 阶段就开始付出代价；但 256 fs 预测相对 segment 明显改善，所以不能说长 rollout 完全无效。rollout8 是三个 rollout 阶段中统一验证目标最好的 checkpoint。进入 rollout16 后没有进一步改善，且负动能偏差加重。','',
        '| 阶段 | 最优 epoch（0-based） | 最优同口径 16-step validation loss | 最后 epoch loss |','|---|---:|---:|---:|']
    for r in history[1:]:lines.append(f"| {r['phase']} | {r['best_epoch_zero_based']} | {f(r['best_validation']['loss'])} | {f(r['last_validation']['loss'])} |")
    lines+=['','`train_quartz.py` 在阶段切换时重置 best，并最终复制 `best_rollout16.pt`，而不是所有阶段中的最优 checkpoint。这是模型选择范围的限制，不是本次分析修改了模型。segment 的训练验证 loss 不含相同多步项，不应拿它与 rollout loss 的绝对值直接比较。此轮没有改选模型重跑 NVE/NVT，故不能声称 rollout8 会有更好的 100 ps 或 15 ns 物理结果。',
        '还需注意实际训练的梯度范围：pushforward 只对最后两个 segment 保留梯度，并非 16 步完整反传。`rollout_species_kinetic` 对所有已预测时刻取均值；其中可直接反传的时刻占比由 rollout4 的 2/3 降到 rollout8 的 2/7，再到 rollout16 的 2/15。VACF 全时刻配对中至少一端可反传的比例分别为 8/9、24/49、56/225。已经 detach 的前缀仍进入 loss 数值，但不产生梯度；因此延长展开并不自动加强这些统计目标的有效训练。上述是代码可证的监督范围变化，不等于已经量化其对耗散的因果贡献，本轮不更改权重或梯度方案。',
        '![Stage validation](stage_validation.png)','', '![Stage history](rollout_objective_history.png)','',
        '## 6. 模式是否互补；Recurrent 是否导致错误','',
        'R1 删除 Mode 1–3 最终输出几乎不影响误差。R2 删除任一模式最终贡献都会显著增加误差：验证集 q RMSE 从 0.002777 Å 分别升至 0.008868/0.015575/0.013105/0.027341 Å；速度亦全部变差。这里仅删除最终 decoder 项，保持上游交互不变，不等同于删掉整个模式重新训练或证明物理可辨识性。',
        'R2 位移 signed projection 约 9.77/18.53/18.33/53.36%，速度约 6.60/17.20/11.99/64.21%。Mode 4 仍占主导，但不再是唯一有用输出。速度贡献的两两余弦约 −0.051 到 0.351，并非四份简单重复；这些指标支持多项输出共同参与预测，但不能证明它们承担了不同的物理时间尺度任务。',
        '输入验证状态上的 Recurrent 路由均值约 0.947/0.99994/0.736/0.691。Mode 3 的跨模式注意力约 99.993% 指向 Mode 4，说明交互仍高度集中。按固定预测步分别测量路由与动能增量误差相关性，最终 256 fs 的四个 Pearson r 约 0.039/0.004/0.016/−0.013，未发现 Recurrent 占优导致耗散的证据；尤其路由变化范围很小，不能用低相关性排除影响，也不能用阶段间共变推断因果。逐步/逐元素数值见 `stages/*.json`。','',
        '## 7. 研究判断与停止点','',
        'R2 缩小了 NVE 的能量偏离幅度，且多模式输出确实参与了预测；但是早期双重能量损失、随后注能以及 NVE 频谱退步表明动力学仍不可信。NVT 温度、元素分配和势能分布的改善不等同于正确 NVE；完整频谱与 RDF 数值如上，必须逐项判断。',
        '下一轮最值得优先处理的是已实测的自主速度/能量变化偏差：长展开从 segment 的注能变成 rollout8/16 的耗散，最终模型在 NVT 每步持续抽走动能。具体应优先审视长展开目标与实际截断梯度的对应关系，并避免末阶段强制覆盖更好的跨阶段验证 checkpoint，而不是继续增加 rollout 长度或损失项。这里提出的是误差优先级，不添加损失、网络或新算法。尚无依据优先把问题归咎于恒温器或 Recurrent 架构本身。',
        'R2 生产吞吐约 104.91 ns/day，R1 163.25 ns/day，来自原始运行计时且包括查询与 I/O；未重新跑速度实验，硬件虽同为 L40S但负载条件未严格隔离，不能把差额完全归因于某个模块。',
        '本轮到此停止：未启动 R3、未改传播器/训练目标、未调恒温参数、未删除不理想轨迹。',
        '', '## 复现与文件','',
        '分析与重跑命令见 `REANALYSIS.md`。所有服务器原始 checkpoint 保留；阶段权重路径在 `stages/*.json`。公开仓库只保存代码、汇总 JSON 与关键图表，不上传 HDF5、checkpoint 或大体积逐样本日志。']
    (out/'PHYSICS_REPORT.md').write_text('\n'.join(lines)+'\n')


def publish(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    archive=destination/'historical_incomplete'
    if not archive.exists():
        archive.mkdir()
        for name in ['COMPARISON.md','RESULTS.md','comparison.json']:
            if (destination/name).exists():shutil.copy2(destination/name,archive/name)
        (archive/'README.md').write_text('Historical incomplete R2 postprocessing report. Its production_available=0 entries represented missing analysis, NOT missing generated trajectories. Superseded by ../PHYSICS_REPORT.md and ../COMPARISON.md. Preserved, not used as current physics results.\n')
    files=list(source.glob('*.json'))+list(source.glob('*.png'))+list(source.glob('*.md'))+list(source.glob('stage_comparison.csv'))
    files+=list((source/'stages').glob('*.json'))
    for folder in source.glob('*_analysis'):
        files+=list(folder.glob('*.json'))+list(folder.glob('*.png'))
    for path in files:
        if path.stat().st_size>4_000_000:raise ValueError(f'Unexpected large public artifact: {path}')
        target=destination/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    (destination/'RESULTS.md').write_text('# R2 results: completed physical reanalysis\n\nBoth 100 ps NVE trajectories and the 15 ns NVT production trajectory were already complete. Physical analysis is now complete as well.\n\nSee [PHYSICS_REPORT.md](PHYSICS_REPORT.md), [COMPARISON.md](COMPARISON.md), and [run_status.json](run_status.json). The original incomplete report is preserved in [historical_incomplete](historical_incomplete/README.md).\n\nNo new training, trajectory generation, model changes or thermostat changes were made. Original weights, trajectories and logs remain on the server.\n')


if __name__=='__main__':main()
