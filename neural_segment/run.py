"""One frozen three-model experiment. No automatic redesign or repeated test selection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    def call(label,module,*flags):
        command=[sys.executable,'-u','-m',module,*map(str,flags)]
        print('RUN',label,' '.join(command),flush=True)
        with (root/(label+'.log')).open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (root/'correctness.json').exists():
        with (root/'correctness.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pytest','-q','neural_segment/test_model.py'],stdout=log,stderr=subprocess.STDOUT,check=True)
        (root/'correctness.json').write_text(json.dumps(dict(passed=True,tests='neural_segment/test_model.py')))
    if not (root/'basis.json').exists():
        call('preflight','neural_segment.preflight','--data-root',args.data_root,'--output',root,'--device',args.device)
    if not (root/'acceleration_scale.json').exists():
        call('labels','neural_segment.labels','--data-root',args.data_root,'--output',root)
    statuses={}
    for variant in ['direct','independent','full']:
        out=root/variant;out.mkdir(exist_ok=True)
        for stage in range(3):
            if stage and not (out/f'onpolicy_round{stage}.json').exists():
                call(f'{variant}_onpolicy{stage}','neural_segment.onpolicy','--data-root',args.data_root,
                     '--shared',root,'--checkpoint',out/f'stage{stage-1}/best.pt',
                     '--output',out/f'onpolicy_round{stage}.h5','--round',stage,'--device',args.device)
            if not (out/f'stage{stage}/complete.json').exists():
                call(f'{variant}_train{stage}','neural_segment.train','--data-root',args.data_root,
                     '--shared',root,'--output',out,'--variant',variant,'--stage',stage,'--device',args.device)
        checkpoint=out/'stage2/best.pt'
        if not (out/'validation/test_metrics.json').exists():
            call(variant+'_validation','evaluate_quartz','--data-root',args.data_root,
                 '--checkpoint',checkpoint,'--output',out/'validation','--split','val','--device',args.device)
        call(variant+'_cost','benchmark_r3','--data-root',args.data_root,'--checkpoint',checkpoint,
             '--output',out/'efficiency.json','--device',args.device)
        screen=[]
        for stem in ['val_07','val_08']:
            ref=args.data_root/f'quartz/raw_nve/{stem}/trajectory.h5'
            candidate=out/f'nve_screen_{stem}.h5';analysis=out/f'nve_screen_{stem}_analysis'
            if not candidate.exists():
                call(variant+'_'+stem,'rollout_quartz','--checkpoint',checkpoint,'--initial',ref,
                     '--output',candidate,'--ensemble','nve','--steps',625,'--device',args.device)
            call(variant+'_'+stem+'_analysis','analysis_tasks','--candidate',candidate,'--reference',ref,'--output',analysis)
            metrics=json.loads((analysis/'metrics.json').read_text())
            import numpy as np
            curve=np.genfromtxt(analysis/'thermodynamics.csv',delimiter=',',names=True)
            de=curve['SMDP_total_eV']-curve['SMDP_total_eV'][0]
            passed=(metrics['trajectory_completed'] and metrics['physical_analysis_completed']
                and abs(metrics['temperature_K']['mean_difference'])<=30
                and float(np.sqrt(np.mean(de**2)))<=1.0 and metrics['minimum_distance_A']>=1.2)
            screen.append(dict(source=stem,passed=bool(passed),energy_RMS_eV=float(np.sqrt(np.mean(de**2))),
                               temperature_difference_K=metrics['temperature_K']['mean_difference']))
        approved=all(s['passed'] for s in screen)
        frozen=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    screen=screen,approved=approved,criteria='Both validation 10ps: complete; |mean T error| <=30K; BKS energy-change RMS <=1 eV; min distance >=1.2 A')
        (out/'screen_decision.json').write_text(json.dumps(frozen,indent=2));statuses[variant]=frozen
        if not approved:continue
        if not (out/'test/test_metrics.json').exists():
            call(variant+'_test','evaluate_quartz','--checkpoint',checkpoint,'--data-root',args.data_root,
                 '--output',out/'test','--device',args.device)
        for stem in ['test_09','test_10','nvt']:
            is_nvt=stem=='nvt'
            ref=args.data_root/('quartz/physical_reference/nvt_15ns/production.h5' if is_nvt else f'quartz/raw_nve/{stem}/trajectory.h5')
            candidate=out/(('nvt' if is_nvt else 'nve_'+stem)+'.h5')
            flags=['--thermalization-steps','93750','--thermostat-metadata',str(ref.parent/'metadata.json')] if is_nvt else []
            if not candidate.exists():
                call(variant+'_'+stem,'rollout_quartz','--checkpoint',checkpoint,'--initial',ref,
                     '--output',candidate,'--ensemble','nvt' if is_nvt else 'nve','--steps',937500 if is_nvt else 6250,'--device',args.device,*flags)
            call(variant+'_'+stem+'_physics','analysis_tasks','--candidate',candidate,'--reference',ref,
                 '--output',out/(candidate.stem+'_analysis'),'--spectral-workers',6)
            summary=json.loads(candidate.with_suffix('.summary.json').read_text())
            if not summary['stable'] or summary['production_ps']<summary['requested_production_ps']:
                statuses[variant]['production_failure']=summary;break
    (root/'experiment_status.json').write_text(json.dumps(statuses,indent=2))
    print('Requested pipeline finished; inspect individual completion/failure states',flush=True)


if __name__=='__main__':main()
