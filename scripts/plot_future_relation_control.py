"""Paired figure and window table from existing analysis; no simulation or fitting."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]/'reports/future_relation_control_20261117'
COMPONENTS = ['translation', 'rotation', 'internal']
MODELS = {'full': ('Full DSRT', '#c74836'),
          'current_only': ('Current-Only', '#2679b5')}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    reference_motion = None
    reference_rdf = None
    windows = []
    for name, (label, color) in MODELS.items():
        directory = REPORT/name/'validation_nve_20ps/analysis'
        motion = np.genfromtxt(directory/'motion.csv', delimiter=',', names=True)
        # Fixed 0.2 ps bins only for legibility; not independent statistical samples.
        bins = np.floor(motion['time_ps']/0.2).astype(int)
        time = np.array([motion['time_ps'][bins == b].mean() for b in np.unique(bins)])
        ref = np.column_stack([motion['time_ps']] +
            [motion[f'reference_{component}_eV'] for component in COMPONENTS])
        if reference_motion is not None:
            np.testing.assert_array_equal(ref, reference_motion)
        for axis, component in zip(axes.flat, COMPONENTS):
            means = np.array([motion[f'model_{component}_eV'][bins == b].mean()
                              for b in np.unique(bins)])
            axis.plot(time, means, color=color, label=label)
            if reference_motion is None:
                means = np.array([motion[f'reference_{component}_eV'][bins == b].mean()
                                  for b in np.unique(bins)])
                axis.plot(time, means, color='black', label='Reference MD', linewidth=1.2)
            axis.set(xlabel='Time (ps)', ylabel='Kinetic energy (eV)',
                     title=component.capitalize()+' (64 molecules)')
        reference_motion = ref
        with (directory/'rdf.csv').open() as f:
            rdf = [row for row in csv.DictReader(f) if row['pair'] == '8-8']
        radius = np.array([float(row['r_A']) for row in rdf])
        reference = np.array([float(row['reference']) for row in rdf])
        if reference_rdf is None:
            axes[1, 1].plot(radius, reference, color='black', label='Reference MD')
        else:
            np.testing.assert_array_equal(np.column_stack((radius, reference)), reference_rdf)
        axes[1, 1].plot(radius, [float(row['model']) for row in rdf], color=color, label=label)
        reference_rdf = np.column_stack((radius, reference))
        metrics = json.loads((directory/'motion.json').read_text())['metrics']
        for window in metrics['windows']:
            for component in COMPONENTS + ['total_KE']:
                value = window[component+'_eV']
                windows.append(dict(model=name, start_ps=window['start_ps'], stop_ps=window['stop_ps'],
                    component=component, model_mean_eV=value['SMDP']['mean'],
                    reference_mean_eV=value['reference']['mean'],
                    difference_eV=value['mean_difference']))
    axes[1, 1].set(xlabel='Distance (Å)', ylabel='g(r)', title='O–O RDF, 0–20 ps', xlim=(2, 6))
    for axis in axes.flat:
        axis.legend(fontsize=9)
        axis.grid(alpha=0.15)
    fig.suptitle('Matched Water validation NVE: one initial state, 20 ps\n'
                 'Nonstationary trajectories; motion curves use 0.2 ps bins', fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORT/'paired_motion_structure.png', dpi=180)
    plt.close(fig)
    with (REPORT/'motion_windows.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(windows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(windows)


if __name__ == '__main__':
    main()
