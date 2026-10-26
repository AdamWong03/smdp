"""Bounded publication of observed outcomes, including missing/failed stages."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--push',action='store_true');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    dest=repo/'reports'/a.output.name
    names=['basis.json','preflight_cost.json','bks_verification.json','bks_verification_initial_failed.json',
           'correctness.json','acceleration_scale.json','experiment_status.json']
    rows=[]
    for variant in ['direct','independent','full']:
        for stage in range(3):
            names.extend(f'{variant}/stage{stage}/{name}' for name in ['config.json','complete.json'])
        names.extend(f'{variant}/{name}' for name in ['screen_decision.json','efficiency.json','validation/test_metrics.json','validation/query_errors.csv','test/test_metrics.json','test/query_errors.csv'])
        for run in ['nve_screen_val_07','nve_screen_val_08','nve_test_09','nve_test_10','nvt']:
            names.append(f'{variant}/{run}.summary.json')
            names.extend(f'{variant}/{run}_analysis/{name}' for name in ['metrics.json','energy.png','energy_partition.png','rdf.png','rdf.csv','vdos_macro_16fs.png','vdos_queried_1fs.png','thermodynamics.csv','thermostat.json'])
            path=a.output/f'{variant}/{run}_analysis/metrics.json'
            if path.exists():
                m=json.loads(path.read_text());rows.append(f"| {variant} | {run} | {m.get('duration_ps')} | {m.get('trajectory_completed')} | {m.get('physical_analysis_completed')} | {m.get('temperature_K',{}).get('mean_difference')} | {m.get('energy_change_eV',{}).get('SMDP')} |")
    dest.mkdir(parents=True,exist_ok=True)
    for name in names:
        source=a.output/name
        if not source.exists():continue
        if source.stat().st_size>4*1024**2:raise ValueError(name)
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    text='# Neural segment experiment — observed status\n\n'
    text+='Method/configuration: [frozen protocol](../../neural_segment/README.md). No absent result is considered completed.\n\n'
    text+=f'Server artifacts: `{a.output}`. Raw logs, weights and trajectories remain there.\n\n'
    text+='| Model | Run | Actual ps | Trajectory complete | Physics complete | Mean T bias / K | BKS end-start / eV |\n|---|---|---:|---|---|---:|---:|\n'+'\n'.join(rows)+'\n'
    text+='\nCompare matched run lengths and cadence only. See metrics JSON for VDOS, RDF and sampled full energy curves; a thermostat is not evidence of energy conservation.\n'
    (dest/'RESULTS.md').write_text(text)
    if a.push:
        if subprocess.check_output(['git','diff','--name-only'],cwd=repo,text=True).strip():
            raise RuntimeError('Tracked changes present; do not bundle unrelated work')
        subprocess.run(['git','add','--',str(dest.relative_to(repo))],cwd=repo,check=True)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode:
            subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m','Report observed neural segment comparison with completion and failure status'],cwd=repo,check=True)
        subprocess.run(['git','push','origin','HEAD'],cwd=repo,check=True,timeout=120)


if __name__=='__main__':main()
