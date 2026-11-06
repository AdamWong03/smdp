"""CMIS validation and formal physical experiments using the existing evaluators."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .train_formal import save_json


def execute(command,log):
    with log.open('a') as stream:
        completed=subprocess.run([sys.executable,'-u',*map(str,command)],stdout=stream,stderr=subprocess.STDOUT)
    return completed.returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--phase',choices=['validation','test'],required=True);a=p.parse_args()
    cfg=json.loads((a.run/'config.json').read_text());system=cfg['system'];h=cfg['model']['horizon_fs']
    root=Path(cfg['reference_root'])/system;physics=a.run/'physics';physics.mkdir(exist_ok=True)
    plans=[]
    if a.phase=='validation':
        source=root/'raw_nve'/('val_07' if system=='quartz' else 'val_04')/'trajectory.h5'
        plans.append(('validation_nve','nve',source,0,int(50000/h),source))
    else:
        for name in (['test_09','test_10'] if system=='quartz' else ['test_05']):
            source=root/'raw_nve'/name/'trajectory.h5'
            plans.append(('nve_'+name,'nve',source,0,int(100000/h),source))
        source=root/'raw_nve'/('test_09' if system=='quartz' else 'test_05')/'trajectory.h5'
        long=root/'physical_reference'/('nvt_15ns' if system=='quartz' else 'nvt_2p5ns')
        plans.append(('nvt_replica_01','nvt',source,50000,500000,long/'production.h5'))
    status={}
    for name,ensemble,initial,thermal,steps,reference in plans:
        directory=physics/name;directory.mkdir(exist_ok=True);trajectory=directory/'trajectory.h5'
        if not trajectory.exists():
            command=['rollout_relational.py','--model-type','cmis','--dense-queries',
                '--checkpoint',a.run/'best.pt','--initial',initial,'--output',trajectory,
                '--ensemble',ensemble,'--steps',str(steps),'--thermalization-steps',str(thermal),
                '--seed','20261106','--execution-note','Shared L40S; graph, thermostat and dense-query output included']
            if ensemble=='nvt':command+=['--thermostat-metadata',reference.parent/'metadata.json']
            if execute(command,directory/'rollout.log')!=0:
                status[name]=dict(trajectory_completed=False,software_failure='rollout failed; see rollout.log')
                save_json(physics/(a.phase+'_status.json'),status);continue
        summary=trajectory.with_suffix('.summary.json')
        if not summary.exists():
            status[name]=dict(trajectory_completed=False,software_failure='No completion summary; existing file was not overwritten')
            save_json(physics/(a.phase+'_status.json'),status);continue
        result=json.loads(summary.read_text())
        status[name]=dict(trajectory_completed=result['completed_steps']==thermal+steps,
                         numerical_safety_passed=result['stable'],failure_reason=result['failure_reason'])
        if result['production_steps']<2:
            status[name]['physical_analysis_completed']=False
            status[name]['analysis_reason']='No valid production interval'
            save_json(physics/(a.phase+'_status.json'),status);continue
        command=['analysis_tasks.py' if system=='quartz' else 'analyze_water.py',
                 '--candidate',trajectory,'--reference',reference,'--output',directory/'analysis']
        if system=='quartz':command+=['--tasks','thermo','rdf','energy','thermostat','vdos_macro','vdos_dense']
        else:
            command+=['--topology',root/'equilibration/snapshots/water_5.data',
                      '--tasks','motion','energy','rdf','vdos','thermostat','diffusion']
            if ensemble=='nve':command+=['full_energy']
        status[name]['physical_analysis_completed']=execute(command,directory/'analysis.log')==0
        if system=='water':
            status[name]['dense_spectrum_execution_succeeded']=execute([
                '-m','cmis.dense_spectrum','--candidate',trajectory,'--reference',reference,
                '--output',directory/'analysis'],directory/'dense_spectrum.log')==0
        save_json(physics/(a.phase+'_status.json'),status)
        print(json.dumps(dict(phase=a.phase,experiment=name,**status[name])),flush=True)
    if any('software_failure' in v for v in status.values()):raise SystemExit(1)


if __name__=='__main__':main()
