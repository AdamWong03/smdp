"""Post-training evaluation pipeline for phase_space_operator_v0 runs.

Waits for the trainer to finish validation-only selection and its test read,
then runs multistep evaluation, the system's NVE protocol, offline physical
analysis, and a baseline comparison. Stages are idempotent: completed files are
reused, partial trajectories are never overwritten. Analysis failures are
recorded but do not hide other completed tasks.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/data1/wyd/research/foundation_mlip_adaptation')
REPO = ROOT/'smdp-future-control'
UPSTREAM = ROOT/'upstream/trajcast-c52f107'
LAMMPS_LIB = '/data1/wyd/envs/lammps-runtime/lib'
PYTHON = sys.executable
# The model environment has e3nn but not LAMMPS; offline BKS analysis needs the
# separate MLIP environment. Analysis only reads HDF5 artifacts.
ANALYSIS_PYTHON = '/data1/wyd/envs/mlip/bin/python'

SYSTEMS = {
    'water5': dict(
        run=ROOT/'experiments/phase_space_20261120/water5',
        data=ROOT/'data/smdp_reference_md_v1/water',
        transitions='transitions/5fs',
        nve=[dict(
            name='nve_val_20ps',
            initial='raw_nve/val_04/trajectory.h5',
            frame=0,
            steps=4000,
            analysis='water',
            analysis_reference='raw_nve/val_04/trajectory.h5',
            topology='equilibration/snapshots/water_5.data',
            frames=2000,
            label='phase-space Water 5 fs',
            tasks='motion energy full_energy rdf vdos diffusion')],
        nvt=dict(
            name='nvt_2p5ns',
            initial='raw_nve/test_05/trajectory.h5',
            frame=0,
            steps=500000,
            thermalization_steps=50000,
            seed=2026110401,
            thermostat='physical_reference/nvt_2p5ns/metadata.json',
            analysis='water',
            analysis_reference='physical_reference/nvt_2p5ns/production.h5',
            topology='equilibration/snapshots/water_5.data',
            frames=2000,
            label='phase-space Water 5 fs NVT',
            tasks='motion energy rdf vdos thermostat diffusion')),
    'quartz30': dict(
        run=ROOT/'experiments/phase_space_20261120/quartz30',
        data=ROOT/'data/smdp_reference_md_v1/quartz',
        transitions='transitions/30fs',
        nve=[dict(name='nve_test_09', initial='raw_nve/test_09/trajectory.h5',
                  frame=0, steps=3333, analysis='quartz',
                  analysis_reference='raw_nve/test_09/trajectory.h5',
                  label='phase-space Quartz 30 fs'),
             dict(name='nve_test_10', initial='raw_nve/test_10/trajectory.h5',
                  frame=0, steps=3333, analysis='quartz',
                  analysis_reference='raw_nve/test_10/trajectory.h5',
                  label='phase-space Quartz 30 fs')],
        nvt=dict(
            name='nvt_15ns',
            initial='raw_nve/test_09/trajectory.h5',
            frame=0,
            steps=500000,
            thermalization_steps=50000,
            seed=26103101,
            thermostat='physical_reference/nvt_15ns/metadata.json',
            analysis='quartz',
            analysis_reference='physical_reference/nvt_15ns/production.h5',
            label='phase-space Quartz 30 fs NVT')),
}

BASELINES = {
    'water5': [
        dict(name='dsrt_water6_full',
             run=ROOT/'experiments/water6_reference_20261113/dsrt_radius6',
             multistep='evaluation/val_multistep.json',
             nve=['validation_nve_20ps'], analysis='water'),
        dict(name='dsrt_water6_current_only',
             run=ROOT/'experiments/future_relation_control_20261117/current_only',
             multistep='evaluation/val_multistep.json',
             nve=['validation_nve_20ps'], analysis='water'),
    ],
    'quartz30': [
        dict(name='dsrt_quartz_r3opt',
             run=ROOT/'experiments/relational_transition/quartz30_20261103_r3_opt',
             multistep='evaluation/multistep.json',
             nve=['physics/nve_test_09', 'physics/nve_test_10'],
             analysis='quartz'),
    ],
}


class PipelineError(RuntimeError):
    pass


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def log_event(**value):
    print(json.dumps(dict(time=now(), **value)), flush=True)


def child_env():
    env = os.environ.copy()
    env['PYTHONPATH'] = f'{REPO}:{UPSTREAM}'
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['PYTHONWARNINGS'] = 'ignore::FutureWarning'
    env['LD_LIBRARY_PATH'] = LAMMPS_LIB + (
        ':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def stage(run, name, status='started'):
    line = f'{now()} {name} {status}'
    with (run/'pipeline_stages.txt').open('a') as handle:
        handle.write(line+'\n')
    (run/'pipeline_stage.txt').write_text(f'{name} {status}\n')
    log_event(event='stage', stage=name, status=status)


def run_command(args, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as handle:
        handle.write(f'\n[{now()}] $ {" ".join(str(a) for a in args)}\n')
        handle.flush()
        started = time.perf_counter()
        code = subprocess.run([str(a) for a in args], cwd=REPO, env=child_env(),
                              stdout=handle, stderr=subprocess.STDOUT).returncode
    log_event(event='command', argv=[str(a) for a in args], code=code,
              seconds=time.perf_counter()-started, log=str(log_path))
    return code


def training_complete(run):
    status = read_json(run/'status.json') or {}
    return ((run/'test_metrics.json').is_file()
            and status.get('state') == 'training_and_test_completed')


def wait_for_training(run, service, poll_seconds, max_wait_hours):
    if os.environ.get('SMDP_SKIP_WAIT') == '1':
        if not training_complete(run):
            raise PipelineError('Training is not complete and SMDP_SKIP_WAIT=1')
        return
    started = time.time()
    first_check = True
    while not training_complete(run):
        status = read_json(run/'status.json') or {}
        if status.get('state') == 'paused':
            raise PipelineError('Training was paused before validation/test completion')
        if time.time()-started > max_wait_hours*3600:
            raise PipelineError(f'Training did not complete within {max_wait_hours} hours')
        if service and not first_check:
            failed = subprocess.run(
                ['systemctl', '--user', 'is-failed', '--quiet', service],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if failed:
                raise PipelineError(f'Training service {service} failed before completion')
        log_event(event='waiting_for_training', status=status.get('state'),
                  last_epoch=status.get('last_epoch'), service=service)
        first_check = False
        time.sleep(poll_seconds)


def multistep(run, checkpoint, transitions, split, device, failures):
    output = run/'evaluation'/f'{split}_multistep.json'
    if output.is_file():
        return output, 'existing'
    stage(run, f'multistep_{split}')
    code = run_command([
        PYTHON, '-u', 'evaluate_relational_rollout.py',
        '--checkpoint', checkpoint, '--transitions', transitions,
        '--output', output, '--batch-size', '8', '--device', device],
        output.with_suffix('.log'))
    if code or not output.is_file():
        failures.append(dict(stage=f'multistep_{split}', reason=f'exit {code}'))
        stage(run, f'multistep_{split}', 'failed')
        return None, 'failed'
    stage(run, f'multistep_{split}', 'completed')
    return output, 'completed'


def analysis_command(spec, trajectory, reference, output):
    if spec['analysis'] == 'water':
        return [ANALYSIS_PYTHON, '-u', 'analyze_water.py', '--candidate', trajectory,
                '--reference', reference, '--topology', spec['topology'],
                '--output', output, '--tasks', *spec['tasks'].split(),
                '--frames', spec.get('frames', 2000),
                '--model-label', spec.get('label', 'phase-space')]
    tasks = spec.get('tasks', 'thermo rdf energy vdos_macro').split()
    return [ANALYSIS_PYTHON, '-u', 'analysis_tasks.py', '--candidate', trajectory,
            '--reference', reference, '--output', output, '--tasks', *tasks]


def failed_analysis_tasks(directory, style):
    if style == 'water':
        status = read_json(Path(directory)/'status.json') or {}
        task_status = status.get('task_status') or {}
    else:
        metrics = read_json(Path(directory)/'metrics.json') or {}
        task_status = metrics.get('task_status') or {}
    return {key: value.get('failure_reason') for key, value in task_status.items()
            if not value.get('completed')}


def nve_stage(run, data, spec, checkpoint, device, failures):
    out = run/'physics'/spec['name']
    trajectory = out/'trajectory.h5'
    summary = out/'trajectory.summary.json'
    if trajectory.is_file() and not summary.is_file():
        raise PipelineError(f'{trajectory} exists without a completion summary; '
                            'inspect it instead of overwriting')
    if not trajectory.is_file():
        out.mkdir(parents=True, exist_ok=True)
        stage(run, spec['name'])
        code = run_command([
            PYTHON, '-u', 'rollout_relational.py', '--checkpoint', checkpoint,
            '--initial', data/spec['initial'], '--initial-frame', spec['frame'],
            '--output', trajectory, '--ensemble', 'nve', '--steps', spec['steps'],
            '--device', device,
            '--execution-note', 'phase-space formal pipeline; validation-selected '
                                'checkpoint; shared GPU; no checkpoint selection'],
            out/'rollout.log')
        if code or not summary.is_file():
            failures.append(dict(stage=spec['name'], reason=f'exit {code}'))
            stage(run, spec['name'], 'failed')
            return out
        stage(run, spec['name'], 'completed')
    result = read_json(summary) or {}
    if not result.get('stable'):
        failures.append(dict(stage=spec['name'],
                             reason=result.get('failure_reason') or 'unstable'))
    marker = out/'analysis'/'.pipeline_attempted'
    if not marker.is_file():
        analysis = out/'analysis'
        analysis.mkdir(parents=True, exist_ok=True)
        stage(run, f"{spec['name']}_analysis")
        code = run_command(analysis_command(
            spec, trajectory, data/spec['analysis_reference'], analysis),
            out/'analysis.log')
        marker.write_text(f'{now()} exit={code}\n')
        failed_tasks = failed_analysis_tasks(analysis, spec['analysis'])
        if code or failed_tasks:
            failures.append(dict(stage=f"{spec['name']}_analysis",
                                 reason=failed_tasks or f'exit {code}'))
            stage(run, f"{spec['name']}_analysis", 'incomplete')
        else:
            stage(run, f"{spec['name']}_analysis", 'completed')
    return out


def nvt_stage(run, data, spec, checkpoint, device, failures):
    out = run/'physics'/spec['name']
    trajectory = out/'trajectory.h5'
    summary = out/'trajectory.summary.json'
    if trajectory.is_file() and not summary.is_file():
        raise PipelineError(f'{trajectory} exists without a completion summary; '
                            'inspect it instead of overwriting')
    if not trajectory.is_file():
        out.mkdir(parents=True, exist_ok=True)
        stage(run, spec['name'])
        code = run_command([
            PYTHON, '-u', 'rollout_relational.py', '--checkpoint', checkpoint,
            '--initial', data/spec['initial'], '--initial-frame', spec['frame'],
            '--output', trajectory, '--ensemble', 'nvt',
            '--steps', spec['steps'],
            '--thermalization-steps', spec['thermalization_steps'],
            '--thermostat-metadata', data/spec['thermostat'],
            '--seed', spec['seed'], '--device', device,
            '--execution-note', 'phase-space formal pipeline NVT; validation-selected '
                                'checkpoint; shared GPU; no checkpoint selection'],
            out/'rollout.log')
        if code or not summary.is_file():
            failures.append(dict(stage=spec['name'], reason=f'exit {code}'))
            stage(run, spec['name'], 'failed')
            return out
        stage(run, spec['name'], 'completed')
    result = read_json(summary) or {}
    if not result.get('stable'):
        failures.append(dict(stage=spec['name'],
                             reason=result.get('failure_reason') or 'unstable'))
    marker = out/'analysis'/'.pipeline_attempted'
    if not marker.is_file():
        analysis = out/'analysis'
        analysis.mkdir(parents=True, exist_ok=True)
        stage(run, f"{spec['name']}_analysis")
        code = run_command(analysis_command(
            spec, trajectory, data/spec['analysis_reference'], analysis),
            out/'analysis.log')
        marker.write_text(f'{now()} exit={code}\n')
        failed_tasks = failed_analysis_tasks(analysis, spec['analysis'])
        if code or failed_tasks:
            failures.append(dict(stage=f"{spec['name']}_analysis",
                                 reason=failed_tasks or f'exit {code}'))
            stage(run, f"{spec['name']}_analysis", 'incomplete')
        else:
            stage(run, f"{spec['name']}_analysis", 'completed')
    return out


def endpoint_metrics(path):
    value = read_json(path)
    if not value:
        return None
    keys = ('position_RMSE_A', 'position_MAE_A', 'velocity_RMSE_A_fs',
            'velocity_MAE_A_fs', 'selection_loss', 'selected_epoch',
            'selected_optimizer_steps', 'samples')
    return {key: value[key] for key in keys if key in value}


def multistep_metrics(path):
    value = read_json(path)
    if not value:
        return None
    return dict(samples=value.get('samples'), selected_epoch=value.get('selected_epoch'),
                horizons={key: {name: row[name] for name in
                                ('time_fs', 'position_RMSE_A', 'velocity_RMSE_A_fs',
                                 'temperature_RMSE_K') if name in row}
                          for key, row in value.get('horizons', {}).items()})


def water_analysis(directory):
    directory = Path(directory)
    result = {}
    motion = read_json(directory/'motion.json')
    if motion and motion.get('completed'):
        metrics = motion['metrics']
        temperature = metrics.get('temperature_K', {})
        result['temperature_K'] = {
            'mean': temperature.get('SMDP', {}).get('mean'),
            'reference_mean': temperature.get('reference', {}).get('mean'),
            'mean_difference': temperature.get('mean_difference')}
        result['kinetic_partition'] = {
            name: {'mean': value.get('SMDP', {}).get('mean'),
                   'reference_mean': value.get('reference', {}).get('mean'),
                   'mean_difference': value.get('mean_difference')}
            for name, value in metrics.get('kinetic_partition', {}).items()}
    energy = read_json(directory/'full_energy.json')
    if energy and energy.get('completed'):
        components = energy['metrics'].get('components', {})
        result['full_energy'] = {
            name: {'change_eV': value.get('change_eV'),
                   'slope_eV_ps': value.get('slope_eV_ps')}
            for name, value in components.items() if isinstance(value, dict)}
        result['full_energy']['duration_ps'] = energy['metrics'].get('duration_ps')
    rdf = read_json(directory/'rdf.json')
    if rdf and rdf.get('completed'):
        result['rdf_RMSE'] = {pair: value.get('RMSE')
                              for pair, value in rdf['metrics'].items()
                              if isinstance(value, dict) and 'RMSE' in value}
    vdos = read_json(directory/'vdos.json')
    if vdos and vdos.get('completed'):
        metrics = vdos['metrics']
        result['vdos_overlap'] = {element: value.get('overlap')
                                  for element, value in metrics.get('elements', {}).items()}
        result['vdos_nyquist_THz'] = metrics.get('nyquist_THz')
    diffusion = read_json(directory/'diffusion.json')
    if diffusion and diffusion.get('completed'):
        result['diffusion'] = diffusion['metrics']
    status = read_json(directory/'status.json') or {}
    result['task_status'] = status.get('task_status')
    result['analysis_failure_reason'] = status.get('analysis_failure_reason')
    return result


def quartz_analysis(directory):
    metrics = read_json(Path(directory)/'metrics.json')
    if not metrics:
        return None
    result = {}
    for key in ('stable', 'failure_reason', 'duration_ps', 'minimum_distance_A',
                'temperature_K', 'total_energy_drift_meV_atom_ps',
                'energy_change_eV', 'kinetic_energy_eV', 'potential_energy_eV',
                'element_kinetic_energy_eV', 'task_status'):
        if key in metrics:
            result[key] = metrics[key]
    result['rdf_RMSE'] = {pair: value.get('RMSE')
                          for pair, value in metrics.get('RDF', {}).items()}
    elements = metrics.get('VDOS', {}).get('macro_30fs', {}).get('elements', {})
    result['vdos_overlap'] = {element: value.get('overlap')
                              for element, value in elements.items()}
    return result


def analysis_for(directory, style):
    return water_analysis(directory) if style == 'water' else quartz_analysis(directory)


def nve_record(directory, style):
    directory = Path(directory)
    summary = read_json(directory/'trajectory.summary.json')
    if not summary:
        return None
    keys = ('completed_steps', 'production_steps', 'simulated_ps', 'stable',
            'failure_reason', 'elapsed_seconds', 'model_and_graph_seconds',
            'ns_day_end_to_end', 'peak_cuda_bytes', 'hardware', 'parameters',
            'selected_epoch', 'checkpoint_sha256')
    return dict(summary={key: summary[key] for key in keys if key in summary},
                analysis=analysis_for(directory/'analysis', style))


def baseline_record(definition):
    run = definition['run']
    record = dict(name=definition['name'], run=str(run),
                  endpoint=endpoint_metrics(run/'test_metrics.json'),
                  multistep=multistep_metrics(run/definition['multistep']))
    record['nve'] = {}
    for name in definition['nve']:
        value = nve_record(run/name, definition['analysis'])
        if value:
            record['nve'][Path(name).name] = value
    return record


def collect_summary(system_name, run, failures):
    system = SYSTEMS[system_name]
    test = read_json(run/'test_metrics.json') or {}
    checkpoint = Path(test.get('checkpoint', run/'best.pt'))
    summary = dict(
        system=system_name, run=str(run), generated=now(),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint) if checkpoint.is_file() else None,
        selected_epoch=test.get('selected_epoch'),
        selected_optimizer_steps=test.get('selected_optimizer_steps'),
        endpoint=endpoint_metrics(run/'test_metrics.json'),
        multistep={
            'val': multistep_metrics(run/'evaluation/val_multistep.json'),
            'test': multistep_metrics(run/'evaluation/test_multistep.json')},
        nve={}, nvt=None,
        baselines=[baseline_record(item) for item in BASELINES[system_name]],
        failures=failures)
    for spec in system['nve']:
        value = nve_record(run/'physics'/spec['name'], spec['analysis'])
        if value:
            summary['nve'][spec['name']] = value
    nvt_out = run/'physics'/system['nvt']['name']
    if (nvt_out/'trajectory.summary.json').is_file():
        summary['nvt'] = nve_record(nvt_out, system['nvt']['analysis'])
    return summary


def fmt(value, digits=4):
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.{digits}g}'
    return str(value)


def write_markdown(summary, path):
    water = summary['system'] == 'water5'
    lines = [f"# Phase-space evaluation: {summary['system']}", '',
             f"Run: `{summary['run']}`  ",
             f"Checkpoint: `{summary['checkpoint']}` (epoch {summary['selected_epoch']})  ",
             f"SHA-256: `{summary['checkpoint_sha256']}`  ",
             f"Generated: {summary['generated']}", '']
    lines += ['## Endpoint test', '',
              '| model | position RMSE (Å) | velocity RMSE (Å/fs) | epoch |',
              '|---|---:|---:|---:|']
    lines.append(f"| phase-space | {fmt(summary['endpoint'] and summary['endpoint'].get('position_RMSE_A'))} "
                 f"| {fmt(summary['endpoint'] and summary['endpoint'].get('velocity_RMSE_A_fs'))} "
                 f"| {fmt(summary['selected_epoch'])} |")
    for baseline in summary['baselines']:
        endpoint = baseline['endpoint'] or {}
        lines.append(f"| {baseline['name']} | {fmt(endpoint.get('position_RMSE_A'))} "
                     f"| {fmt(endpoint.get('velocity_RMSE_A_fs'))} "
                     f"| {fmt(endpoint.get('selected_epoch'))} |")
    lines += ['', '## Multistep (autonomous rollout)', '',
              '| split | horizon | position RMSE (Å) | velocity RMSE (Å/fs) | T RMSE (K) |',
              '|---|---:|---:|---:|---:|']
    for split in ('val', 'test'):
        value = summary['multistep'][split]
        if not value:
            continue
        for horizon, row in value['horizons'].items():
            lines.append(f"| {split} | {horizon} x {fmt(row.get('time_fs'), 4)} fs "
                         f"| {fmt(row.get('position_RMSE_A'))} "
                         f"| {fmt(row.get('velocity_RMSE_A_fs'))} "
                         f"| {fmt(row.get('temperature_RMSE_K'))} |")
    baseline_ms = summary['baselines'][0]['multistep'] if summary['baselines'] else None
    if baseline_ms:
        lines.append('')
        lines.append(f"Baseline `{summary['baselines'][0]['name']}` "
                     f"({baseline_ms.get('samples')} samples):")
        lines.append('')
        lines.append('| horizon | position RMSE (Å) | velocity RMSE (Å/fs) |')
        lines.append('|---:|---:|---:|')
        for horizon, row in baseline_ms['horizons'].items():
            lines.append(f"| {horizon} | {fmt(row.get('position_RMSE_A'))} "
                         f"| {fmt(row.get('velocity_RMSE_A_fs'))} |")
    lines += ['', '## NVE physics', '']
    records = [(name, value) for name, value in summary['nve'].items()]
    for baseline in summary['baselines']:
        records += [(f"{baseline['name']}/{name}", value)
                    for name, value in baseline['nve'].items()]
    if not records:
        lines.append('No completed NVE trajectory.')
    for name, record in records:
        trajectory, analysis = record['summary'], record['analysis'] or {}
        lines.append(f"### {name}")
        lines.append('')
        lines.append(f"stable={trajectory.get('stable')} "
                     f"steps={trajectory.get('completed_steps')} "
                     f"simulated_ps={fmt(trajectory.get('simulated_ps'))} "
                     f"ns/day={fmt(trajectory.get('ns_day_end_to_end'))} "
                     f"failure={trajectory.get('failure_reason')}")
        lines.append('')
        if water:
            temperature = analysis.get('temperature_K') or {}
            energy = analysis.get('full_energy') or {}
            partition = analysis.get('kinetic_partition') or {}
            lines.append(f"- temperature: model {fmt(temperature.get('mean'), 6)} K, "
                         f"reference {fmt(temperature.get('reference_mean'), 6)} K, "
                         f"difference {fmt(temperature.get('mean_difference'), 4)} K")
            total = energy.get('total') or {}
            lines.append(f"- total energy: change {fmt(total.get('change_eV'), 5)} eV, "
                         f"slope {fmt(total.get('slope_eV_ps'), 5)} eV/ps")
            for key in ('translation_eV', 'rotation_eV', 'internal_eV'):
                value = partition.get(key) or {}
                lines.append(f"- {key}: model {fmt(value.get('mean'), 5)}, "
                             f"reference {fmt(value.get('reference_mean'), 5)}, "
                             f"difference {fmt(value.get('mean_difference'), 5)} eV")
            vdos = analysis.get('vdos_overlap') or {}
            lines.append(f"- VDOS overlap: {', '.join(f'{k}={fmt(v, 4)}' for k, v in vdos.items())}")
            rdf = analysis.get('rdf_RMSE') or {}
            lines.append(f"- RDF RMSE: {', '.join(f'{k}={fmt(v, 4)}' for k, v in rdf.items())}")
        else:
            temperature = analysis.get('temperature_K') or {}
            smdp = temperature.get('SMDP') or {}
            reference = temperature.get('reference') or {}
            lines.append(f"- temperature: model {fmt(smdp.get('mean'), 6)} K, "
                         f"reference {fmt(reference.get('mean'), 6)} K")
            drift = analysis.get('total_energy_drift_meV_atom_ps') or {}
            change = analysis.get('energy_change_eV') or {}
            lines.append(f"- total energy drift: model {fmt(drift.get('SMDP'), 5)} "
                         f"meV/atom/ps, reference {fmt(drift.get('reference'), 5)}")
            lines.append(f"- total energy change: model {fmt(change.get('SMDP'), 5)} eV, "
                         f"reference {fmt(change.get('reference'), 5)} eV")
            vdos = analysis.get('vdos_overlap') or {}
            lines.append(f"- VDOS overlap: {', '.join(f'{k}={fmt(v, 4)}' for k, v in vdos.items())}")
            rdf = analysis.get('rdf_RMSE') or {}
            lines.append(f"- RDF RMSE: {', '.join(f'{k}={fmt(v, 4)}' for k, v in rdf.items())}")
        failures = analysis.get('analysis_failure_reason')
        if failures:
            lines.append(f"- analysis failures: `{json.dumps(failures)}`")
        lines.append('')
    if summary['nvt']:
        record = summary['nvt']
        lines += ['## NVT', '',
                  f"stable={(record['summary'] or {}).get('stable')} "
                  f"simulated_ps={fmt((record['summary'] or {}).get('simulated_ps'))}",
                  '']
    if summary['failures']:
        lines += ['## Pipeline failures', '']
        for failure in summary['failures']:
            lines.append(f"- {failure['stage']}: {failure['reason']}")
        lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    parser.add_argument('--service', help='training systemd user service to watch')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--max-wait-hours', type=float, default=48.)
    parser.add_argument('--no-wait', action='store_true',
                        help='require completed training immediately')
    parser.add_argument('--include-nvt', action='store_true',
                        help='also run the long NVT protocol after NVE')
    parser.add_argument('--summary-only', action='store_true',
                        help='regenerate summary from completed artifacts')
    args = parser.parse_args()
    system = SYSTEMS[args.system]
    run = system['run']
    run.mkdir(parents=True, exist_ok=True)
    lock = (run/'pipeline.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'Pipeline already running for {run}')
    if args.no_wait:
        if not training_complete(run):
            raise PipelineError('Training is not complete')
    elif not args.summary_only:
        wait_for_training(run, args.service, args.poll_seconds, args.max_wait_hours)
    failures = []
    if not args.summary_only:
        checkpoint = Path(read_json(run/'test_metrics.json')['checkpoint'])
        if not checkpoint.is_file():
            raise PipelineError(f'Selected checkpoint is missing: {checkpoint}')
        data = system['data']
        transitions = data/system['transitions']
        multistep(run, checkpoint, transitions/'val.h5', 'val', args.device, failures)
        multistep(run, checkpoint, transitions/'test.h5', 'test', args.device, failures)
        for spec in system['nve']:
            nve_stage(run, data, spec, checkpoint, args.device, failures)
        if args.include_nvt:
            nvt_stage(run, data, system['nvt'], checkpoint, args.device, failures)
    summary = collect_summary(args.system, run, failures)
    summary['complete'] = (not failures
                           and summary['endpoint'] is not None
                           and summary['multistep']['test'] is not None
                           and len(summary['nve']) == len(system['nve']))
    atomic_json(run/'summary.json', summary)
    write_markdown(summary, run/'SUMMARY.md')
    log_event(event='summary_written', path=str(run/'summary.json'),
              complete=summary['complete'], failures=len(failures))
    raise SystemExit(0 if summary['complete'] else 2)


if __name__ == '__main__':
    main()
