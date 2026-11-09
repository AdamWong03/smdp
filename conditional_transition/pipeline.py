"""Execute exactly the accepted four-arm protocol, independent immutable runs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from smdpr.train import write_json


def execute(run,label,module,*args):
    with (run/(label+'.log')).open('a') as log:
        subprocess.run([sys.executable,'-u','-m',module,*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run
    cfg=json.loads((run/'config.json').read_text());state=run/'pipeline_status.json'
    try:
        for phase,target in [(1,12500),(2,25000)]:
            status=run/'status.json'
            current=json.loads(status.read_text()).get('optimizer_steps',0) if status.exists() else 0
            if current>=target:continue
            if cfg['arm'] in ['autonomous','extra_reference']:
                collector=Path(cfg['initial_checkpoint']) if phase==1 else run/'step_12500.pt'
                write_json(state,dict(state='collecting',round=phase,trained_updates=current))
                execute(run,f'collect_round{phase}','conditional_transition.collect',
                    '--run',run,'--round',phase,'--checkpoint',collector)
            write_json(state,dict(state='training',target_step=target))
            execute(run,'train','conditional_transition.train','--run',run,'--target-step',target)
            status=json.loads((run/'status.json').read_text())
            if status['optimizer_steps']!=target or status['state']=='paused':
                write_json(state,dict(state='paused',optimizer_steps=status['optimizer_steps']));return
        write_json(state,dict(state='validation_physics',trained_updates=25000))
        execute(run,'evaluate','conditional_transition.evaluate','--run',run)
        write_json(state,dict(state='completed',optimizer_steps=25000,transition_targets=400000,
            validation_physics_completed=True,test_accessed=False,long_NVT_started=False))
    except Exception as error:
        write_json(state,dict(state='execution_failed',reason=f'{type(error).__name__}: {error}'))
        raise


if __name__=='__main__':main()
