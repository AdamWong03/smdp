"""Export completed Water endpoint/NVE evidence without copying large artifacts."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def export(run, reference, output):
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    endpoint = json.loads((run/'test_metrics.json').read_text())
    multistep = json.loads((run/'evaluation/multistep_test.json').read_text())
    rollout = json.loads((run/'physics/nve_test_05/trajectory.summary.json').read_text())
    with h5py.File(run/'physics/nve_test_05/trajectory.h5', 'r') as f:
        completed = int(f.attrs['completed_steps']) + 1
        time = f['time_fs'][:completed] / 1000
        temperature = f['temperature_K'][:completed]
        minimum = f['minimum_distance_A'][:completed]
    with h5py.File(reference, 'r') as f:
        rt = f['time_fs'][:] / 1000
        rtemp = f['thermo/temperature_K'][:]
        renergy = f['thermo/total_energy_eV'][:]
    common = min(time[-1], rt[-1])
    summary = dict(
        endpoint=endpoint, multistep=multistep, rollout=rollout,
        training=dict(epochs=len(records), optimizer_updates=records[-1]['optimizer_steps'],
                      final_train_loss=records[-1]['train_final_loss'],
                      final_validation=records[-1]['validation']),
        nve=dict(trajectory_completed=True, physical_analysis_completed=False,
                 pending=['potential/total energy', 'RDF', 'VACF/VDOS', 'MSD/diffusion'],
                 initial_temperature_K=float(temperature[0]),
                 mean_temperature_K=float(temperature.mean()),
                 final_temperature_K=float(temperature[-1]),
                 maximum_temperature_K=float(temperature.max()),
                 minimum_distance_A=float(minimum.min()), common_duration_ps=float(common),
                 common_candidate_temperature_K=float(temperature[time <= common].mean()),
                 common_reference_temperature_K=float(rtemp[rt <= common].mean()),
                 reference_total_energy_change_eV=float(renergy[-1]-renergy[0]),
                 windows=[]))
    for start, stop in [(0, 1), (1, 5), (5, 20), (20, 50), (50, 100)]:
        mask = (time >= start) & (time <= stop)
        summary['nve']['windows'].append(dict(start_ps=start, stop_ps=stop,
            mean_temperature_K=float(temperature[mask].mean()),
            std_temperature_K=float(temperature[mask].std())))
    (output/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    epoch = [record['epoch'] for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(epoch, [r['train_final_loss'] for r in records], label='Train')
    axes[0].semilogy(epoch, [r['validation']['selection_loss'] for r in records], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Normalized endpoint loss'); axes[0].legend()
    axes[1].plot(time[::10], temperature[::10], label='DSRT (100 ps)')
    axes[1].plot(rt[::100], rtemp[::100], label='Reference MD (50 ps)', alpha=.7)
    axes[1].set(xlabel='Time (ps)', ylabel='NVE temperature (K)'); axes[1].legend()
    fig.tight_layout(); fig.savefig(output/'training_and_nve.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.reference, args.output)
