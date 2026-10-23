"""One frozen experiment, no architecture search or automatic retraining."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import h5py


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--after-training',action='store_true')
    p.add_argument('--wait-training-pid',type=int)
    p.add_argument('--publish-summary',action='store_true')
    p.add_argument('--previous',type=Path)
    p.add_argument('--report-name',default='quartz16_additive_long_rollout')
    p.add_argument('--spectral-workers',type=int,default=1)
    p.add_argument('--r1',type=Path,help='Optional R1 path for the full R1/R2/current physics report')
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
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config_path=args.output/'evaluation_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text())['checkpoint_sha256']!=digest:
            raise ValueError('Selected checkpoint changed since evaluation launch')
    else:
        config_path.write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            argv=sys.argv,python=sys.executable,data_root=str(args.data_root),
            nve_initials=['test_09','test_10'],nve_steps=6250,nvt_steps=937500,
            thermalization_steps=93750,horizon_fs=16,seed=2026102201,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            ld_library_path=os.environ.get('LD_LIBRARY_PATH'),
            analysis_sampling='unchanged 2000 thermodynamic frames / 512 RDF frames / full spectral blocks',
            timestamp_unix=time.time()),indent=2))
    test=args.output/'test/test_metrics.json'
    if test.exists():
        if json.loads(test.read_text()).get('checkpoint_sha256')!=digest:
            raise ValueError('Existing test result does not identify the selected checkpoint; do not overwrite')
    else:
        run('test','evaluate_quartz.py','--checkpoint',checkpoint,'--data-root',args.data_root,'--output',args.output/'test')
    def trajectory(name,initial,output,ensemble,steps,thermal=0,metadata=None):
        if output.exists():
            with h5py.File(output,'r') as f:
                if (f.attrs.get('checkpoint_sha256')!=digest or f.attrs['ensemble']!=ensemble
                    or int(f.attrs['requested_steps'])!=steps or int(f.attrs.get('thermalization_steps',0))!=thermal):
                    raise ValueError(f'Existing trajectory protocol/checkpoint mismatch: {output}')
                complete=int(f.attrs.get('completed_steps',0))>=steps+thermal
                reason=str(f.attrs.get('failure_reason',''))
            if not complete:
                failures.append(dict(stage=name,reason=reason or 'Existing interrupted trajectory retained; not overwritten'))
            print(f'Reusing existing trajectory: {output}; completed={complete}',flush=True)
        else:
            options=['--checkpoint',checkpoint,'--initial',initial,'--output',output,
                     '--ensemble',ensemble,'--steps',steps,'--thermalization-steps',thermal]
            if metadata:options+=['--thermostat-metadata',metadata]
            run(name,'rollout_quartz.py',*options)
        if output.exists():
            run('analyze_'+name,'analysis_tasks.py','--candidate',output,'--reference',initial,
                '--output',args.output/f'{name}_analysis','--spectral-workers',args.spectral_workers)
    for initial_name in ['test_09','test_10']:
        initial=args.data_root/f'quartz/raw_nve/{initial_name}/trajectory.h5'
        output=args.output/f'nve_{initial_name}.h5'
        trajectory('nve_'+initial_name,initial,output,'nve',6250)
    initial=args.data_root/'quartz/physical_reference/nvt_15ns/production.h5'
    output=args.output/'nvt.h5'
    trajectory('nvt',initial,output,'nvt',937500,93750,initial.parent/'metadata.json')
    benchmark=args.output/'production_efficiency_benchmark.json'
    if not benchmark.exists():
        run('efficiency','benchmark_r3.py','--checkpoint',checkpoint,'--data-root',args.data_root,'--output',benchmark)
    report_options=['--output',args.output]
    if args.previous: report_options+=['--previous',args.previous]
    run('report','summarize_experiment.py',*report_options)
    if args.r1 and args.previous:
        run('physical_comparison','report_physical_comparison.py','--r1',args.r1,'--r2',args.previous,'--current',args.output)
    statuses={}
    for name in ['nve_test_09','nve_test_10','nvt']:
        pth=args.output/f'{name}_analysis/metrics.json'
        metrics=json.loads(pth.read_text()) if pth.exists() else {}
        statuses[name]={k:metrics.get(k,False) for k in ['trajectory_completed','physical_analysis_completed']}
    success=not failures and all(all(v.values()) for v in statuses.values())
    (args.output/'pipeline_finished.json').write_text(json.dumps({'finished':True,'all_requested_completed':success,
        'software_failures':failures,'run_status':statuses,'checkpoint_sha256':digest},indent=2))
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=digest:raise RuntimeError('Checkpoint changed during evaluation')
    if args.publish_summary:
        run('publish_summary','publish_summary.py','--output',args.output,'--report-name',args.report_name)


if __name__=='__main__': main()
