"""Export only small, observed artifacts; never weights, trajectories or raw logs."""
import argparse,json,shutil,subprocess
from pathlib import Path

RUN=Path('/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_20261028')


def main(run,push):
    run=Path(run);repo=Path(__file__).resolve().parents[1]
    out=repo/'reports/smdp_trajcast_quartz30_20261028';out.mkdir(parents=True,exist_ok=True)
    names=['config.yaml','data_preparation.json','correctness.json','training_started.json','training_complete.json',
           'test_metrics.json','source_version.txt','cache/adaptation.json']
    for tag in ['nve_0','nve_1','nvt_0']:
        names.append(f'{tag}/run.json')
        if (run/tag/'analysis').exists():
            names += [str(p.relative_to(run)) for p in (run/tag/'analysis').iterdir() if p.suffix in ['.json','.csv','.png']]
    for name in names:
        source=run/name
        if source.exists():
            if source.stat().st_size>2_000_000: continue
            dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    if (run/'training.jsonl').exists():
        import pandas as pd
        records=[json.loads(s) for s in (run/'training.jsonl').read_text().splitlines() if s.strip()]
        pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in records]).to_csv(out/'training_curves.csv',index=False)
        if records:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for key in ['train_loss','segment_loss','selection_score']:
                axes[0].plot([r[key] for r in records],label=key)
            axes[0].set_yscale('log');axes[0].legend()
            axes[1].plot([r['rollout8_loss'] for r in records],label='validation 240 fs');axes[1].set_yscale('log');axes[1].legend()
            fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)
    lines=['# SMDP / TrajCast infrastructure / Quartz 30 fs','',
        'All SMDP parameters start from random initialization. No pretrained weights or additional baseline training.',
        f'Server artifacts: `{run}`.',
        'Method and commands: [README](../../trajcast_smdp/README.md).','',
        f"Training completed: {(run/'training_complete.json').exists()}.",
        f"Independent test evaluation completed: {(run/'test_metrics.json').exists()}.",'']
    test=run/'test_metrics.json'
    if test.exists():
        d=json.loads(test.read_text())
        lines += [f"Endpoint position RMSE: {d['endpoint_position_RMSE_A']:.7g} Å.",
            f"Endpoint velocity RMSE: {d['endpoint_velocity_RMSE_A_fs']:.7g} Å/fs.",
            f"Eight-step normalized error: {d['rollout8_loss']:.7g}.",'']
    for tag in ['nve_0','nve_1','nvt_0']:
        p=run/tag/'run.json';lines += [f'## {tag}','']
        if not p.exists(): lines += ['Not completed; no physical results claimed.',''];continue
        d=json.loads(p.read_text());lines += [f"Trajectory completed: {d['trajectory_completed']}; duration {d['actual_duration_ps']} ps (includes any thermalization).",
            f"End-to-end throughput: {d['ns_day']:.4g} ns/day.",f"Failure: {d['failure_reason']}.",'']
        for name in ['thermo','energy','macro_30fs','dense_1fs','rdf']:
            p=run/tag/'analysis'/f'{name}.json'
            if p.exists(): lines += [f'### {name}','', '```json',p.read_text().strip(),'```','']
    lines += ['No phonon identification, stability improvement or speedup is inferred from architecture alone.',
        'Official-paper BKS energy discrepancies remain recorded in the foundation report. This experiment compares only against its own independently generated, unchanged BKS reference.','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    if push:
        subprocess.run(['git','add',str(out.relative_to(repo))],cwd=repo,check=True)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode:
            subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m','Report observed SMDP Quartz30 progress and physics'],cwd=repo,check=True)
        subprocess.run(['git','-c','http.proxy=','-c','http.version=HTTP/1.1','push','--ipv4','-u','origin','HEAD'],cwd=repo,check=True,timeout=120)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);p.add_argument('--push',action='store_true')
    a=p.parse_args();main(a.run,a.push)
