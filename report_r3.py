"""Publish bounded artifacts from the completed frozen R3 pair; no inference/training."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data1/wyd/research/foundation_mlip_adaptation'))
    parser.add_argument('--output', type=Path, default=Path('reports/quartz16_r3'))
    args = parser.parse_args()
    ex = args.root/'experiments/motion_modes'
    pair = ex/'quartz16_r3_pair'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if 'R3_PAIR_EXIT_CODE=0' not in (pair/'pipeline.log').read_text():
        raise RuntimeError('Pair has not completed successfully')
    names = ['R2', 'R3_A', 'R3_B']
    runs = dict(zip(names, [ex/'quartz16_20261022_r2_additive',
                            ex/'quartz16_r3a_composition', ex/'quartz16_r3b_spatial_composition']))
    diag = {n: read(pair/'prediction_diagnostics'/f'{n}.json') for n in names}
    bench = {n: read(pair/'benchmark_r2.json' if n == 'R2' else runs[n]/'benchmark.json') for n in names}
    validation = read(ex/'quartz16_r3_pair_validation.json')
    shutil.copy2(ex/'quartz16_r3_pair_validation.json', out/'validation.json')
    rows, provenance, histories = [], {}, {}
    with h5py.File(args.root/'data/smdp_reference_md_v1/quartz/transitions/16fs/val.h5') as f:
        sources = f['source_trajectory'][:]
        source_paths = [p.decode() if isinstance(p, bytes) else str(p) for p in f['source_paths'][:]]
    replicate_rows = []
    for i, n in enumerate(names):
        d, b = diag[n], bench[n]
        for filename, value in [('diagnostics.json', d), ('benchmark.json', b), ('config.json', read(runs[n]/'config.json'))]:
            (out/f'{n}_{filename}').write_text(json.dumps(value, indent=2))
        ckpt = runs[n]/'best_rollout.pt'
        provenance[n] = dict(checkpoint=str(ckpt), sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())
        s1, s16 = d['steps']['1'], d['steps']['16']
        v = validation[i]['metrics']
        rows.append(dict(method=n, q16_RMSE_A=s1['position_RMSE_A'], v16_RMSE_A_fs=s1['velocity_RMSE_A_fs'],
                         q256_RMSE_A=s16['position_RMSE_A'], v256_RMSE_A_fs=s16['velocity_RMSE_A_fs'],
                         KE256_bias_eV=s16['KE_bias_eV'], composition=v['composition'],
                         objective_without_composition=v['reference_objective_without_composition'],
                         kinematics_RMSE_A_fs=d['kinematics_RMSE_A_fs'],
                         O_vacf256_RMSE=s16['species']['8']['vacf_window_RMSE'],
                         Si_vacf256_RMSE=s16['species']['14']['vacf_window_RMSE'],
                         endpoint_ms=b['autonomous_results'][0]['ms_per_macrostep'],
                         dense_ms=b['autonomous_results'][1]['ms_per_macrostep'],
                         endpoint_ns_day=b['autonomous_results'][0]['ns_per_day'],
                         dense_ns_day=b['autonomous_results'][1]['ns_per_day'], parameters=b['parameters']))
        windows = pd.read_csv(pair/'prediction_diagnostics'/f'{n}_windows.csv')
        windows['source_trajectory'] = sources[windows['sample'].to_numpy()]
        for (tid, step), g in windows.groupby(['source_trajectory', 'step']):
            replicate_rows.append(dict(method=n, source=source_paths[int(tid)], step=int(step), samples=len(g),
                q_RMSE_A=np.sqrt(g.q_mse.mean()), v_RMSE_A_fs=np.sqrt(g.v_mse.mean()),
                KE_bias_eV=(g.predicted_KE_eV-g.reference_KE_eV).mean()))
        if n != 'R2':
            history = [json.loads(line) for line in (runs[n]/'history.jsonl').read_text().splitlines()]
            histories[n] = history
            pd.json_normalize(history).to_csv(out/f'{n}_training.csv', index=False)
            shutil.copy2(runs[n]/'training_complete.json', out/f'{n}_training_complete.json')
    pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False)
    pd.DataFrame(replicate_rows).to_csv(out/'trajectory_validation.csv', index=False)
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for n in names:
        s = diag[n]['steps']; t = np.arange(1, 17)*16
        for ax, key in zip(axes.flat[:3], ['position_RMSE_A', 'velocity_RMSE_A_fs', 'KE_bias_eV']):
            ax.plot(t, [s[str(k)][key] for k in range(1, 17)], label=n)
        axes[1, 1].plot(t, [s[str(k)]['species']['8']['vacf_window_RMSE'] for k in range(1, 17)], label=n)
    for ax, title in zip(axes.flat, ['Position RMSE (A)', 'Velocity RMSE (A/fs)', 'Kinetic-energy bias (eV)', 'O normalized VACF window RMSE']):
        ax.set(title=title, xlabel='Autonomous horizon (fs)'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'validation.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for n, history in histories.items():
        for ax, phase in zip(axes, ['segment', 'rollout']):
            rr = [r for r in history if r['phase'].startswith(phase)]
            ax.plot(np.arange(len(rr))+1, [r['validation']['loss'] for r in rr], label=n)
    axes[0].set(title='Segment validation objective', xlabel='Segment epoch', yscale='log')
    axes[1].set(title='Common 16-step validation objective', xlabel='Rollout epoch (4 then 8)')
    axes[1].axvline(12.5, color='gray', linestyle='--')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'training.png', dpi=160); plt.close(fig)
    table = ['| 指标 | R2 | R3-A | R3-B |', '|---|---:|---:|---:|']
    for key, label in [('q16_RMSE_A','16 fs 位置 RMSE / Å'), ('v16_RMSE_A_fs','16 fs 速度 RMSE / Å/fs'),
                       ('q256_RMSE_A','256 fs 位置 RMSE / Å'), ('v256_RMSE_A_fs','256 fs 速度 RMSE / Å/fs'),
                       ('KE256_bias_eV','256 fs 动能偏差 / eV'), ('composition','组合一致性损失'),
                       ('objective_without_composition','去除组合项的统一验证目标'),
                       ('kinematics_RMSE_A_fs','轨迹段运动学 RMSE / Å/fs'),
                       ('O_vacf256_RMSE','256 fs O 速度自相关窗口 RMSE'), ('Si_vacf256_RMSE','256 fs Si 速度自相关窗口 RMSE'),
                       ('endpoint_ms','仅终点耗时 / ms'), ('dense_ms','16 个查询耗时 / ms'),
                       ('endpoint_ns_day','仅终点吞吐 / ns/day'), ('dense_ns_day','密集查询吞吐 / ns/day')]:
        table.append('| '+label+' | '+' | '.join(f'{r[key]:.7g}' for r in rows)+' |')
    report = '''# R3-A/B 正式训练与验证结果

两组均完成 100 个 segment epoch、12 个 rollout4 epoch、12 个 rollout8 epoch；流程退出码 0。
无额外架构/损失修改，无新参考 MD，无新生产 NVE/NVT，无测试集调参。
本报告是验证集结果，不是独立最终测试或长期物理验证。

## 同协议结果

单步诊断覆盖全部 1250 个验证起点，256 fs 覆盖具有完整参考后续的 1245 个起点。
组合损失覆盖这 1245 个起点；其端点误差与全部 1250 样本的诊断误差略有不同。
真实轨迹来自两个独立验证源。逐轨迹分解见 trajectory_validation.csv；
窗口相关，不能把 1245 个窗口当成独立实验重复。仅一个训练种子，不宣称统计显著性。

'''+ '\n'.join(table)+'''

动能偏差不是总能量漂移：本轮没有离线 BKS 势能评估或长期物理轨迹。
256 fs 参考平均动能为 5.975258 eV，R3-B 仍低约 4.99%。

## 判断

1. **A 并未证明时间组合训练改善长跨度相空间预测。** 相比 R2，组合损失降低约 16%，
   16 fs 位置/速度误差改善约 4%，但 256 fs 位置/速度误差分别变差 2.38%/1.89%。
   动能偏差与速度自相关有所改善，不能只按一个误差宣判全部无效，也不能按组合损失宣称成功。
2. **B 有部分实际收益，但不是全面改善。** 相比 R2，16 fs 位置/速度误差下降
   11.06%/11.71%；256 fs 速度误差下降 6.77%，位置误差只下降 0.039%，基本持平。
   相比 A，B 同时改善单步和 256 fs 误差；O/Si 速度相关误差、动能偏差也更小。
3. **成本不能忽略。** A/B 训练耗时 3.70/5.03 小时；参数 900324/998084；
   训练峰值显存 6.36/15.62 GiB。B 端点计时较 A 增加约 8.6%，密集查询增加约 113%。
   密集推理峰值显存约 1.09 GB，而 A 约 0.123 GB。
4. **适合保留为下一步候选的是 B 的跨原子交互，不是宣布已解决长期动力学。**
   端点推理下有可测的精度—成本折中；密集输出是否值得尚不明确。
   A 的组合训练没有独立证明必要性：没有“B 去组合项”对照，本轮不增加第三模型。
5. A/B 预算匹配；历史 R2 还包括 rollout16，且只在最终阶段选 checkpoint。
   因此 A/R2 的差异不能全部归因于组合训练。这里不借此追加新实验。

## 模式与传播机制

B 各模式的 Recurrent 路由均值（跨查询/两阶段平均）约为 55.9%、96.9%、0.44%、81.0%。
Mode 3 主要使用 Temporal，Mode 2 主要使用 Recurrent；每个 epoch 两类传播器及新增空间模块
均有有限非零梯度。但路由变化跨起点较小，权重分工不等于真实物理模式。

B 的位置贡献 signed projection 约 4.15%、26.94%、28.26%、40.65%；速度约
3.29%、23.07%、13.69%、59.95%。Mode 1 明显较弱，不能声称四个模式均衡或充分互补。
删除任意一个最终解码贡献都增加误差；删除 Mode 1 后位置 RMSE 从 0.002470 增至
0.003073 Å。但这是固定上游状态的解码贡献删除，不是整个模式或传播器的因果消融。
Mode 1/4 速度贡献余弦约 0.744，存在明显相关性；Mode 2/3 的跨模式注意力仍几乎全部指向 Mode 4。
不能宣称已经消除所有模式退化，也未通过独立干预证明两种传播器各自不可替代。

B 运动学误差 8.99e-5 Å/fs，虽比 R2 降低约 9.7%，仍是参考 8.70e-6 的约 10.3 倍。
段连接处导数跳变 RMS 为 R2/A/B=4.772e-4/4.506e-4/4.639e-4，参考 4.826e-4 Å/fs；
绝对 RMS 更小不自动代表逐样本连接更准确。统一目标另含参考对齐的连接误差。

## 选择与计时边界

A 选 rollout8 的第 10 个 epoch，B 选第 7 个 epoch，均是预先指定跨 rollout 阶段验证选择，
不是训练最后一个 epoch。后期继续训练并非持续改善，全部历史保留于训练曲线和 CSV。

计时在同一 L40S 上，包括周期邻居图重建、数据传输与网络。表格是重复 16-step
自主验证块的计时，各块重置起点；不是 15 ns 长轨迹吞吐，不含恒温器或输出 I/O。
固定输入计时也单独保存在 benchmark JSON。共享 GPU 存在其他任务，300 步短计时
没有重复置信区间，不与旧的生产 ns/day 直接构造加速比。

## 交付与停止边界

本目录包含配置、训练曲线/CSV、验证 JSON、模式诊断、计时和 checkpoint 哈希。
权重、原始日志和窗口级 CSV 留在 README 所列服务器路径。
复现汇总：`python report_r3.py`；不执行网络推理、训练或 MD。
本轮至此停止，不启动 R4 或长期 NVE/NVT。若下一步只选一个候选做完整物理验证，
建议优先讨论 B；现在不能声称它改善了长期能量漂移、VDOS 或 RDF。
'''
    (out/'RESULTS.md').write_text(report)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
