"""Publish only explicitly allowed small artifacts; never checkpoints or raw logs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report-name',default='quartz16_first');args=p.parse_args()
    repo=Path(__file__).resolve().parent
    # Refuse to bundle unrelated edits made while the experiment was running.
    subprocess.run(['git','diff','--exit-code'],cwd=repo,check=True)
    subprocess.run(['git','diff','--cached','--exit-code'],cwd=repo,check=True)
    if Path(args.report_name).name!=args.report_name: raise ValueError('report-name must be a directory basename')
    target=repo/'reports'/args.report_name
    names=['RESULTS.md','COMPARISON.md','comparison.json','diagnosis/previous_diagnosis.json','config.json','training.png','training_complete.json','pipeline_finished.json',
           'test/test_metrics.json','test/query_errors.csv','PHYSICS_COMPARISON.md','physical_comparison.json',
           'physical_comparison.csv','physical_run_status.json','nve_window_comparison.json',
           'production_efficiency_benchmark.json','evaluation_config.json','runtime_verification/bks_runtime_verification.json',
           'nve_test_09_comparison.png','nve_test_10_comparison.png','nvt_macro_16fs_comparison.png',
           'nvt_queried_1fs_comparison.png','nvt_structure_comparison.png','run_status.json']
    for run in ['nve_test_09','nve_test_10','nvt']:
        names.append(run+'.summary.json')
        for name in ['metrics.json','energy.png','rdf.png','rdf.csv','vdos_macro_16fs.png','vdos_queried_1fs.png',
                     'vdos_macro_16fs.csv','vdos_queried_1fs.csv','nve_trajectory_error.csv',
                     'thermodynamics.csv','energy_partition.png','thermostat.json','thermostat_blocks.csv',
                     'thermo.json','energy.json','rdf.json','vdos_macro.json','vdos_dense.json']:
            names.append(run+'_analysis/'+name)
    copied=[]
    for name in names:
        source=args.output/name
        if not source.exists(): continue
        if source.stat().st_size>4*1024**2: raise ValueError(f'Summary unexpectedly large: {name}')
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        copied.append(str(dest.relative_to(repo)))
    subprocess.run(['git','add','--',*copied],cwd=repo,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m',
                    'Report observed quartz experiment including incomplete or failed rollouts'],cwd=repo,check=True)
    # Normal push only; never pull/rebase/reset or force over another contributor.
    subprocess.run(['git','push','origin','HEAD'],cwd=repo,check=True)


if __name__=='__main__': main()
