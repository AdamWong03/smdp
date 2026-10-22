"""Concise report of observed artifacts, never infers success from requested length."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path);args=p.parse_args()
    out=args.output
    history=[json.loads(line) for line in (out/'history.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots(figsize=(8,4))
    for phase in dict.fromkeys(r['phase'] for r in history):
        rows=[(i,r) for i,r in enumerate(history) if r['phase']==phase]
        ax.semilogy([i for i,r in rows],[r['validation']['loss'] for i,r in rows],label=phase+' validation')
        ax.semilogy([i for i,r in rows],[r['train']['loss'] for i,r in rows],alpha=.5,label=phase+' train')
    ax.legend();ax.set(xlabel='epoch',ylabel='normalized supervised loss');fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)
    lines=['# Quartz 16 fs: observed results','',f'Server artifacts: `{out}`','',
           'Model: fixed architecture with learned motion tokens, mode-conditioned recurrent/temporal routing, mid-evolution cross-mode exchange, equivariant segment decoding.',
           'All comparisons below use BKS reference MD. No claim of named physical modes follows from routing weights.','',
           '## Training','',f'Completed epoch records: {len(history)}. Training wall time: {history[-1]["elapsed_s"]/3600:.3f} h.','',
           '## Independent endpoint test','']
    path=out/'test/test_metrics.json'
    if path.exists():
        result=json.loads(path.read_text());lines.extend(['```json',json.dumps(result,indent=2),'```'])
    else:lines.append('Not available.')
    lines+=['','## Autonomous propagation','','| Run | Production ps | Failure | ns/day (including I/O) |','|---|---:|---|---:|']
    for name in ['nve_test_09','nve_test_10','nvt']:
        path=out/f'{name}.summary.json'
        if path.exists():
            r=json.loads(path.read_text());lines.append(f'| {name} | {r["production_ps"]:.6g} | {r["failure_reason"] or "none recorded"} | {r["ns_day_end_to_end"]:.4g} |')
        else:lines.append(f'| {name} | unavailable | not completed | unavailable |')
    for name in ['nve_test_09','nve_test_10','nvt']:
        path=out/f'{name}_analysis/metrics.json'
        if path.exists():lines+=['',f'## {name}: quantitative physical comparison','','```json',json.dumps(json.loads(path.read_text()),indent=2),'```']
    weights=history[-1]['mode_statistics'].get('fusion_weights')
    lines+=['','## Observed mechanism and cost limitations','']
    if weights is not None: lines.append(f'Final validation mean fusion weights: {weights}.')
    else: lines.append('Additive mode readout; no competitive fusion weights. Report actual per-mode displacement/velocity RMS and signed projections from the selected checkpoint test. These attributions are not named physical modes.')
    if weights is not None and max(weights)>.98:
        lines.append('Fusion is dominated by one mode (>98%). This run does not establish useful heterogeneous modal specialization; no balancing regularizer was added to disguise it.')
    lines.append('Soft routing evaluates both recurrent and temporal operators. Inference cost includes all 16 time queries; a learned route does not save expert computation. No matched-hardware BKS speedup is claimed without a measured baseline.')
    lines+=['','## Interpretation boundaries','','Only completed production frames are production evidence. Failed thermalization statistics are diagnostics, not 15 ns NVT results.',
            'The stable field only means no numerical termination threshold was triggered. It is not evidence of physical fidelity or energy conservation.',
            '16 fs spectra compare identically downsampled outputs. Queried 1 fs spectra compare dense velocities separately and include endpoint thermostat kicks; output cadence alone does not establish high-frequency fidelity.',
            'This experiment stops after these evaluations; there is no automatic architecture change or second training run.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if args.previous:
        compare_runs(args.previous,out)


def compare_runs(previous,current):
    """Publish observed numbers and their differences, without hiding failures."""
    rows=[]
    def add(run,metric,old,new,reference=None):
        rows.append(dict(run=run,metric=metric,old=old,new=new,reference=reference,
                         new_minus_old=new-old if old is not None and new is not None else None))
    def read(root,path):
        p=root/path
        return json.loads(p.read_text()) if p.exists() else {}
    oldtest=read(previous,'test/test_metrics.json');newtest=read(current,'test/test_metrics.json')
    for key in ['position_RMSE_A','velocity_RMSE_A_fs']:
        add('test',key,oldtest.get('endpoint',{}).get(key),newtest.get('endpoint',{}).get(key))
    for key in ['position_velocity_consistency_RMSE_A_fs','join_position_derivative_jump_RMS_A_fs','two_step_position_RMSE_A']:
        add('test',key,oldtest.get(key),newtest.get(key))
    for name in ['nve_test_09','nve_test_10','nvt']:
        old=read(previous,f'{name}_analysis/metrics.json');new=read(current,f'{name}_analysis/metrics.json')
        if not new.get('production_available',False):
            add(name,'production_available',int(old.get('production_available',False)),0)
            continue
        for key in ['temperature_K','kinetic_energy_eV','potential_energy_eV']:
            for stat in ['mean','std']:
                add(name,key+'_'+stat,old.get(key,{}).get('SMDP',{}).get(stat),new[key]['SMDP'][stat],new[key]['reference'][stat])
        for key in ['energy_change_eV','total_energy_drift_meV_atom_ps']:
            add(name,key,old.get(key,{}).get('SMDP'),new[key]['SMDP'],new[key]['reference'])
        add(name,'minimum_distance_A',old.get('minimum_distance_A'),new.get('minimum_distance_A'))
        for z in ['8','14']:
            add(name,'species_KE_eV_Z'+z,old.get('element_kinetic_energy_eV',{}).get(z,{}).get('SMDP',{}).get('mean'),
                new['element_kinetic_energy_eV'][z]['SMDP']['mean'],new['element_kinetic_energy_eV'][z]['reference']['mean'])
        for cadence in ['macro_16fs','queried_1fs']:
            for z in ['8','14']:
                add(name,cadence+'_VDOS_overlap_Z'+z,old.get('VDOS',{}).get(cadence,{}).get('elements',{}).get(z,{}).get('overlap'),
                    new.get('VDOS',{}).get(cadence,{}).get('elements',{}).get(z,{}).get('overlap'),1.)
        for pair in ['14-8','8-8']:
            add(name,'RDF_RMSE_'+pair,old.get('RDF',{}).get(pair,{}).get('RMSE'),new.get('RDF',{}).get(pair,{}).get('RMSE'),0.)
            for key in ['peak_position_A','peak_height']:
                add(name,'RDF_'+pair+'_'+key,old.get('RDF',{}).get(pair,{}).get(key,{}).get('SMDP'),
                    new['RDF'][pair][key]['SMDP'],new['RDF'][pair][key]['reference'])
        for key in ['production_ps','ns_day_end_to_end','ms_per_step','peak_gpu_bytes']:
            add(name,key,read(previous,name+'.summary.json').get(key),read(current,name+'.summary.json').get(key))
    (current/'comparison.json').write_text(json.dumps(rows,indent=2))
    lines=['# Previous / additive-mode iteration / BKS comparison','',f'Previous: `{previous}`',f'Current: `{current}`','',
        'Evaluation cadence, stopping thresholds, CSVR parameters and BKS are unchanged. A completed trajectory need not be physically accurate. Partial trajectories are not equivalent 15 ns evidence. Check recorded hardware before interpreting throughput differences.','',
        '| Run | Metric | Previous | Current | BKS | Current − previous |','|---|---|---:|---:|---:|---:|']
    def fmt(x): return 'unavailable' if x is None else f'{x:.7g}'
    for r in rows: lines.append('| '+' | '.join([r['run'],r['metric'],*[fmt(r[x]) for x in ['old','new','reference','new_minus_old']]])+' |')
    (current/'COMPARISON.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
