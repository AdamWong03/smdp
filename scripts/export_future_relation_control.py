"""Publish measured progress and paired outputs; missing evidence stays pending."""
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/data1/wyd/research/foundation_mlip_adaptation')
BASE = ROOT/'experiments/future_relation_control_20261117'
REPORT = Path(__file__).resolve().parents[1]/'reports/future_relation_control_20261117'
RUNS = dict(full=ROOT/'experiments/water6_reference_20261113/dsrt_radius6',
            current_only=BASE/'current_only')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    snapshot = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), experiments={})
    curves, table, motion_table = [], [], []
    for name, run in RUNS.items():
        history = []
        if (run/'history.jsonl').exists():
            lines = (run/'history.jsonl').read_text().splitlines()
            for index, line in enumerate(lines):
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines)-1:
                        raise
        status, metadata = read(run/'status.json'), read(run/'metadata.json')
        best = min(history, key=lambda r:r['validation']['selection_loss']) if history else None
        physical = run/'validation_nve_20ps'
        trajectory = read(physical/'trajectory.summary.json')
        analysis = read(physical/'analysis/status.json')
        motion = read(physical/'analysis/motion.json')
        trajectory_complete = bool(trajectory and trajectory['completed_steps'] == 4000)
        entry = dict(run_directory=str(run), status=status, metadata=metadata,
            completed_epochs=len(history), optimizer_steps=history[-1]['optimizer_steps'] if history else 0,
            best_validation=best['validation'] if best else None,
            best_epoch=best['epoch'] if best else None, test_metrics=read(run/'test_metrics.json'),
            correctness=read(run/'correctness.json'), trajectory_completed=trajectory_complete,
            trajectory_summary=trajectory, physical_analysis_status=analysis,
            physical_analysis_completed=bool(analysis and analysis.get('physical_analysis_completed')),
            validation_multistep=read(run/'evaluation/val_multistep.json'))
        entry['training_completed'] = entry['optimizer_steps'] == 250000
        entry['endpoint_test_completed'] = entry['test_metrics'] is not None
        if name == 'current_only':
            stage = run/'pipeline_stage.txt'
            entry['pipeline_stage'] = stage.read_text().strip() if stage.exists() else 'prepared'
        snapshot['experiments'][name] = entry
        dest = REPORT/name
        dest.mkdir(exist_ok=True)
        for filename in ['config.json', 'metadata.json', 'correctness.json', 'test_metrics.json',
                         'environment.txt', 'gpu_at_launch.csv', 'source_commit.txt', 'pipeline_stages.txt']:
            if (run/filename).exists():
                shutil.copy2(run/filename, dest/filename)
        for folder, filenames in [('evaluation', ['val_multistep.json', 'paired_inference.json']),
                ('validation_nve_20ps', ['trajectory.summary.json']),
                ('validation_nve_20ps/analysis', ['status.json', 'motion.json', 'motion.csv', 'motion.png',
                    'energy.json', 'reference_verification.json', 'rdf.json', 'rdf.csv', 'rdf.png',
                    'vdos.json', 'vdos.csv', 'vdos.png', 'vacf.csv', 'vacf.png', 'diffusion.json'])]:
            for filename in filenames:
                source = run/folder/filename
                if source.exists():
                    target = dest/folder/filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        (dest/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        for record in history:
            curves.append(dict(model=name, epoch=record['epoch'], optimizer_steps=record['optimizer_steps'],
                train_final_loss=record['train_final_loss'], train_proposal_loss=record['train_proposal_loss'],
                val_final_loss=record['validation']['selection_loss'],
                val_proposal_loss=record['validation']['proposal_loss'],
                val_position_RMSE_A=record['validation']['position_RMSE_A'],
                val_velocity_RMSE_A_fs=record['validation']['velocity_RMSE_A_fs'],
                lr=record['lr'], epoch_seconds=record['epoch_seconds']))
        test = entry['test_metrics'] or {}
        table.append(dict(model=name, completed_epochs=len(history), optimizer_steps=entry['optimizer_steps'],
            best_epoch=entry['best_epoch'], test_position_RMSE_A=test.get('position_RMSE_A'),
            test_velocity_RMSE_A_fs=test.get('velocity_RMSE_A_fs'),
            training_completed=entry['training_completed'],
            endpoint_test_completed=entry['endpoint_test_completed'],
            trajectory_completed=trajectory_complete,
            physical_analysis_completed=entry['physical_analysis_completed'],
            analysis_failure_reason=json.dumps((analysis or {}).get('analysis_failure_reason', {}))))
        if motion and motion.get('completed'):
            for component, values in motion['metrics']['kinetic_partition'].items():
                motion_table.append(dict(model=name, component=component,
                    model_mean_eV=values['SMDP']['mean'], reference_mean_eV=values['reference']['mean'],
                    mean_difference_eV=values['mean_difference']))
    for filename, rows in [('training_curves.csv', curves), ('comparison.csv', table),
                           ('molecular_motion.csv', motion_table)]:
        if rows:
            with (REPORT/filename).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader(); writer.writerows(rows)
    (REPORT/'snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    if (BASE/'correctness_tests.txt').exists():
        shutil.copy2(BASE/'correctness_tests.txt', REPORT/'correctness_tests.txt')
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for name in RUNS:
            rows = [r for r in curves if r['model'] == name]
            if rows:
                x = [r['optimizer_steps'] for r in rows]
                axes[0].semilogy(x, [r['train_final_loss'] for r in rows], label=name+' train')
                axes[0].semilogy(x, [r['val_final_loss'] for r in rows], '--', label=name+' val')
                axes[1].semilogy(x, [r['val_position_RMSE_A'] for r in rows], label=name)
        axes[0].set(ylabel='Normalized endpoint MSE', xlabel='Optimizer updates')
        axes[1].set(ylabel='Validation position RMSE (Å)', xlabel='Optimizer updates')
        for axis in axes: axis.legend()
        fig.tight_layout(); fig.savefig(REPORT/'training_curves.png', dpi=160); plt.close(fig)
    current = snapshot['experiments']['current_only']
    lines = ['# Full vs Current-Only: measured status', '',
        'Recorded UTC: '+snapshot['recorded_utc'], '',
        f"Current-Only: {current['pipeline_stage']}; {current['completed_epochs']} epochs / "
        f"{current['optimizer_steps']:,} of 250,000 updates recorded.", '',
        '| Model | Updates | Best epoch | Test q RMSE (Å) | Test v RMSE (Å/fs) |',
        '|---|---:|---:|---:|---:|']
    for row in table:
        q, v = row['test_position_RMSE_A'], row['test_velocity_RMSE_A_fs']
        lines.append(f"| {row['model']} | {row['optimizer_steps']} | {row['best_epoch'] or 'pending'} | "
                     f"{format(q, '.9g') if q is not None else 'pending'} | "
                     f"{format(v, '.9g') if v is not None else 'pending'} |")
    lines += ['', '| Model | Training | Endpoint test | 20 ps trajectory | Physical analysis |',
        '|---|---|---|---|---|']
    for name, entry in snapshot['experiments'].items():
        states = ['completed' if entry[key] else 'incomplete' for key in
                  ['training_completed', 'endpoint_test_completed', 'trajectory_completed',
                   'physical_analysis_completed']]
        lines.append('| '+name+' | '+' | '.join(states)+' |')
        for task, reason in (entry['physical_analysis_status'] or {}).get('analysis_failure_reason', {}).items():
            lines.append(f'\n{name}, {task}: {reason}\n')
    if (REPORT/'RESULTS.md').exists():
        lines += ['', '[Scientific comparison and limitations](RESULTS.md).']
    lines += ['', 'This is one seed and one',
        'paired validation NVE start; no general mechanism claim is made automatically.',
        'Full trained on shared L40S; Current-Only on shared Blackwell. Optimization',
        'budgets match, wall time and bitwise GPU arithmetic do not.', '',
        'The strict Water reference energy verification remains separate. A numerically',
        'completed trajectory or completed motion/RDF/VDOS task does not certify energy',
        'analysis or physical fidelity. Do not use the legacy two-frame full-energy',
        'result to bypass failed eight-frame verification.', '',
        '[Protocol and reproduction](PROTOCOL.md). Full logs and checkpoints remain',
        'at the absolute server paths recorded in snapshot.json.', '']
    (REPORT/'STATUS.md').write_text('\n'.join(lines))
    print(json.dumps({name: {k: entry.get(k) for k in
        ['completed_epochs', 'optimizer_steps', 'best_epoch', 'pipeline_stage',
         'trajectory_completed', 'physical_analysis_completed']}
        for name, entry in snapshot['experiments'].items()}, indent=2))


if __name__ == '__main__':
    main()
