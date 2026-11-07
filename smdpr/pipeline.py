"""Resumable frozen post-training protocol; failed physical runs stay failed."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from smdpr.train import write_json

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
DATA = PROJECT/'data/smdp_reference_md_v1'


def command(run, label, *arguments):
    with (run/f'{label}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', *map(str, arguments)], cwd=REPO,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def endpoint(run, split, modes=False):
    required = [run/'evaluation'/f'{split}_endpoint.json']
    cfg = json.loads((run/'config.json').read_text())
    if cfg.get('architecture')=='smdpr':
        required.append(run/'evaluation'/f'{split}_query.json')
        if modes:
            required.append(run/'evaluation'/f'{split}_modes.json')
    if not all(path.exists() for path in required):
        options = ['--mode-analysis'] if modes else []
        command(run, f'evaluate_{split}', '-m', 'smdpr.evaluate', '--run', run,
                '--split', split, *options)
    return json.loads(required[0].read_text())


def multistep(run, cfg, split):
    dest = run/'evaluation'/f'{split}_multistep.json'
    if not dest.exists():
        h = int(cfg['model']['horizon_fs'])
        command(run, f'multistep_{split}', 'evaluate_relational_rollout.py',
            '--checkpoint', run/'best.pt', '--transitions',
            DATA/cfg['system']/'transitions'/f'{h}fs'/f'{split}.h5', '--output', dest)


def trajectory(run, cfg, name, source, ensemble, steps, thermal=0, frame=0, dense=False):
    directory = run/'physics'/name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'trajectory.h5'
    summary = path.with_suffix('.summary.json')
    if summary.exists():
        return path, json.loads(summary.read_text())
    if path.exists():
        raise RuntimeError(f'Partial trajectory retained; manual recovery required: {path}')
    extra = []
    if ensemble=='nvt':
        folder = 'nvt_15ns' if cfg['system']=='quartz' else 'nvt_2p5ns'
        extra += ['--thermostat-metadata', DATA/cfg['system']/'physical_reference'/folder/'metadata.json']
    if dense:
        extra += ['--dense-query-fs', '1']
    command(run, f'rollout_{name}', 'rollout_relational.py', '--checkpoint', run/'best.pt',
        '--initial', source, '--initial-frame', frame, '--output', path,
        '--ensemble', ensemble, '--steps', steps, '--thermalization-steps', thermal,
        '--seed', cfg['seed']+300, '--execution-note',
        'Shared Blackwell GPU; endpoint/query/neighbor/CSVR/storage included in production cost', *extra)
    return path, json.loads(summary.read_text())


def validation_gate(run, cfg, baseline):
    dest = run/'validation_gate.json'
    if dest.exists():
        return json.loads(dest.read_text())['passed']
    metrics = endpoint(run, 'val')
    multistep(run, cfg, 'val')
    historical = torch.load(baseline, map_location='cpu', weights_only=False)
    reference = historical['record']['validation']
    ratios = {key: metrics[key]/reference[key] for key in ['position_RMSE_A', 'velocity_RMSE_A_fs']}
    path = DATA/cfg['system']/'raw_nve'/('val_07' if cfg['system']=='quartz' else 'val_04')/'trajectory.h5'
    trajectory_path, summary = trajectory(run, cfg, 'validation_nve_10ps', path, 'nve',
                                          round(10000/cfg['model']['horizon_fs']))
    with h5py.File(trajectory_path) as f:
        n = int(f.attrs['completed_steps'])
        temperatures = f['temperature_K'][:n+1]
    passed = bool(summary['stable'] and max(ratios.values())<=1.5)
    write_json(dest, dict(passed=passed, criterion='q/v RMSE <=1.5x historical baseline and complete stable ~10 ps NVE',
        validation_ratios=ratios, baseline_checkpoint=str(baseline),
        autonomous_summary=summary, mean_temperature_K=float(np.mean(temperatures)),
        numerical_safety='existing minimum distance and 1..3000K limits; not proof of physical fidelity'))
    return passed


def analyze(run, cfg, trajectory_path, reference, dense):
    out = trajectory_path.parent/'analysis'
    if cfg['system']=='quartz':
        command(run, 'analysis_'+trajectory_path.parent.name, 'analysis_tasks.py',
            '--candidate', trajectory_path, '--reference', reference, '--output', out)
    else:
        tasks = ['motion', 'energy', 'rdf', 'vdos', 'full_energy', 'thermostat', 'diffusion']
        if dense:
            tasks += ['vdos_dense']
        command(run, 'analysis_'+trajectory_path.parent.name, 'analyze_water.py',
            '--candidate', trajectory_path, '--reference', reference, '--topology',
            DATA/'water/equilibration/snapshots/water_5.data', '--output', out,
            '--model-label', 'SMDP-R' if cfg.get('architecture')=='smdpr' else 'R3-Opt two-step',
            '--tasks', *tasks)


def formal_physics(run, cfg):
    endpoint(run, 'test', modes=cfg.get('architecture')=='smdpr')
    multistep(run, cfg, 'test')
    system = cfg['system']
    h = cfg['model']['horizon_fs']
    if cfg.get('architecture')=='smdpr':
        for scale in [1, 2, 3]:
            dest = run/'evaluation'/f'efficiency_supercell{scale}.json'
            if not dest.exists():
                command(run, f'benchmark_supercell{scale}', '-m', 'smdpr.benchmark',
                    '--config', run/'config.json', '--checkpoint', run/'best.pt',
                    '--output', dest, '--supercell', scale, '--iterations', 30)
    cases = [('test_09', 0), ('test_10', 0)] if system=='quartz' else [('test_05', 0), ('test_05', 50000)]
    nve_summaries = []
    for i, (source, frame) in enumerate(cases):
        reference = DATA/system/'raw_nve'/source/'trajectory.h5'
        path, summary = trajectory(run, cfg, f'nve_100ps_{i+1:02}', reference, 'nve', round(100000/h), frame=frame)
        nve_summaries.append(summary)
        if frame:
            # The second Water initial state is 25 ps into the same independent
            # test source. Crop the REFERENCE view for correctly aligned errors.
            cropped = path.parent/'aligned_reference.h5'
            if not cropped.exists():
                with h5py.File(reference) as src, h5py.File(cropped, 'x') as dst:
                    dst.attrs.update(dict(src.attrs))
                    for key, value in src.items():
                        data = value[frame:] if len(value.shape)>0 and value.shape[0]==len(src['time_fs']) else value[:]
                        if key=='time_fs':
                            data = data-data[0]
                        dst[key] = data
                    dst.attrs['n_frames'] = len(dst['time_fs'])
                    dst.attrs['reference_initial_frame'] = frame
            reference = cropped
        analyze(run, cfg, path, reference, False)
    if not all(item['stable'] for item in nve_summaries):
        write_json(run/'formal_physics_status.json', dict(trajectory_completed=False,
            physical_analysis_completed=False, nve_summaries=nve_summaries,
            failure_reason='100 ps NVE failed; NVT not launched to mask instability'))
        return False
    dense = cfg.get('architecture')=='smdpr'
    folder = 'nvt_15ns' if system=='quartz' else 'nvt_2p5ns'
    path, summary = trajectory(run, cfg, 'nvt_production', DATA/system/'raw_nve'/cases[0][0]/'trajectory.h5',
        'nvt', 500000, thermal=50000, dense=dense)
    analyze(run, cfg, path, DATA/system/'physical_reference'/folder/'production.h5', dense)
    write_json(run/'formal_physics_status.json', dict(trajectory_completed=summary['stable'] and summary['production_steps']==500000,
        physical_analysis_completed=True, failure_reason=summary['failure_reason'],
        note='Read each independent analysis status; missing/failed tasks are never replaced by scores'))
    return bool(summary['stable'] and summary['production_steps']==500000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline-control', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run/'config.json').read_text())
    baseline_run = PROJECT/'experiments/relational_transition'/f"{cfg['system']}{int(cfg['model']['horizon_fs'])}_20261103_r3_opt"
    state = run/'pipeline_status.json'
    try:
        write_json(state, dict(state='validation', physical_analysis_completed=False))
        if not validation_gate(run, cfg, baseline_run/'best.pt'):
            # Independent stage-1 prediction results are still reported. No
            # further training or production NVT follows a failed safety gate.
            endpoint(run, 'test', modes=True)
            multistep(run, cfg, 'test')
            write_json(state, dict(state='stopped_by_validation_gate', trajectory_completed=False,
                physical_analysis_completed=False, failure_reason='protective validation gate failed'))
            return
        stage2 = run/'two_step'
        status_path = stage2/'status.json'
        completed = status_path.exists() and json.loads(status_path.read_text()).get('state')=='two_step_training_completed'
        if not completed:
            options = ['--resume'] if (stage2/'last.pt').exists() else []
            command(run, 'two_step_training', '-m', 'smdpr.finetune', '--initial-checkpoint',
                run/'best.pt', '--run', stage2, *options)
        # Both frozen stages get independent prediction evaluation. No choice is
        # made using these test scores; stage 2 is the predeclared final method.
        endpoint(run, 'test', modes=True)
        multistep(run, cfg, 'test')
        if validation_gate(stage2, cfg, baseline_run/'best.pt'):
            physics_completed = formal_physics(stage2, cfg)
        else:
            endpoint(stage2, 'test', modes=True)
            write_json(state, dict(state='stage2_stopped_by_validation_gate', physical_analysis_completed=False))
            return
        if args.baseline_control:
            # Only one fine-tuning control per system, used by the K3 pipeline.
            control = run.parent/f"{cfg['system']}{int(cfg['model']['horizon_fs'])}_r3opt_matched_two_step"
            status = control/'status.json'
            if not status.exists() or json.loads(status.read_text()).get('state')!='two_step_training_completed':
                options = ['--resume'] if (control/'last.pt').exists() else []
                command(run, 'baseline_two_step', '-m', 'smdpr.finetune', '--initial-checkpoint',
                        baseline_run/'best.pt', '--run', control, *options)
            control_cfg = json.loads((control/'config.json').read_text())
            control_cfg['system'] = cfg['system']
            # This top-level metadata addition does not modify checkpoint weights.
            write_json(control/'config.json', control_cfg)
            if validation_gate(control, control_cfg, baseline_run/'best.pt'):
                formal_physics(control, control_cfg)
        write_json(state, dict(state='completed' if physics_completed else 'completed_with_physical_failure',
                              physical_analysis_completed=physics_completed))
    except Exception as error:
        write_json(state, dict(state='execution_failed', physical_analysis_completed=False,
                               failure_reason=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
