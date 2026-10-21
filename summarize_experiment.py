"""Concise report of observed artifacts, never infers success from requested length."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output
    history=[json.loads(line) for line in (out/'history.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots(figsize=(8,4))
    for phase in ['segment','rollout']:
        rows=[(i,r) for i,r in enumerate(history) if r['phase']==phase]
        ax.semilogy([i for i,r in rows],[r['validation']['loss'] for i,r in rows],label=phase+' validation')
        ax.semilogy([i for i,r in rows],[r['train']['loss'] for i,r in rows],alpha=.5,label=phase+' train')
    ax.legend();ax.set(xlabel='epoch',ylabel='normalized supervised loss');fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)
    lines=['# Quartz 16 fs: observed first-run results','',f'Server artifacts: `{out}`','',
           'Model: frozen motion tokens, mode-conditioned recurrent/temporal routing, mid-evolution cross-mode exchange, equivariant segment decoding.',
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
    lines+=['','## Interpretation boundaries','','Only completed production frames are production evidence. Failed thermalization statistics are diagnostics, not 15 ns NVT results.',
            '16 fs spectra compare identically downsampled outputs. Queried 1 fs spectra compare dense velocities separately and include endpoint thermostat kicks; output cadence alone does not establish high-frequency fidelity.',
            'This experiment stops after these evaluations; there is no automatic architecture change or second training run.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
