"""One frozen experiment, no architecture search or automatic retraining."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--after-training',action='store_true')
    p.add_argument('--wait-training-pid',type=int)
    p.add_argument('--publish-summary',action='store_true')
    p.add_argument('--previous',type=Path)
    p.add_argument('--report-name',default='quartz16_additive_long_rollout')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parent
    failures=[]
    def run(name,script,*options):
        cmd=[sys.executable,str(repo/script),*map(str,options)]
        with (args.output/f'{name}.log').open('a') as f:
            proc=subprocess.run(cmd,cwd=repo,stdout=f,stderr=subprocess.STDOUT)
        if proc.returncode:
            failures.append({'stage':name,'returncode':proc.returncode})
        return proc.returncode==0
    if args.wait_training_pid:
        while not (args.output/'training_complete.json').exists():
            try: os.kill(args.wait_training_pid,0)
            except ProcessLookupError: raise RuntimeError('Training exited without completion; do not evaluate an unfinished checkpoint')
            time.sleep(30)
    if not args.after_training:
        if not run('train','train_quartz.py','--data-root',args.data_root,'--output',args.output):
            raise RuntimeError('Training failed; see train.log')
    if not (args.output/'training_complete.json').exists(): raise RuntimeError('Training must complete before test evaluation')
    checkpoint=args.output/'best_rollout.pt'
    run('test','evaluate_quartz.py','--checkpoint',checkpoint,'--data-root',args.data_root,'--output',args.output/'test')
    for initial_name in ['test_09','test_10']:
        initial=args.data_root/f'quartz/raw_nve/{initial_name}/trajectory.h5'
        output=args.output/f'nve_{initial_name}.h5'
        run('nve_'+initial_name,'rollout_quartz.py','--checkpoint',checkpoint,'--initial',initial,'--output',output,
            '--ensemble','nve','--steps',6250)
        if output.exists(): run('analyze_nve_'+initial_name,'analyze_quartz.py','--candidate',output,'--reference',initial,'--output',args.output/f'nve_{initial_name}_analysis')
    initial=args.data_root/'quartz/physical_reference/nvt_15ns/production.h5'
    output=args.output/'nvt.h5'
    run('nvt','rollout_quartz.py','--checkpoint',checkpoint,'--initial',initial,'--output',output,'--ensemble','nvt',
        '--steps',937500,'--thermalization-steps',93750,'--thermostat-metadata',initial.parent/'metadata.json')
    if output.exists(): run('analyze_nvt','analyze_quartz.py','--candidate',output,'--reference',initial,'--output',args.output/'nvt_analysis')
    report_options=['--output',args.output]
    if args.previous: report_options+=['--previous',args.previous]
    run('report','summarize_experiment.py',*report_options)
    (args.output/'pipeline_finished.json').write_text(json.dumps({'finished':True,'software_failures':failures},indent=2))
    if args.publish_summary:
        run('publish_summary','publish_summary.py','--output',args.output,'--report-name',args.report_name)


if __name__=='__main__': main()
