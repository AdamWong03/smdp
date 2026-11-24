"""Figure 1 — pointwise accuracy and short-horizon stability do not predict NVT fidelity.

Panels
  a  one-step position MAE vs mean absolute mode-partition error
  b  25 ps NVE temperature traces (no thermostat)
  c  mode kinetic-energy ratio to the reference NVT (heat map)
  d  longest stable NVT run vs partition verdict

All values are from our own screens; see
method_development/MULTISTEP_STABILITY_TABLE_20261122.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data2/wyd/research/foundation_mlip_adaptation')
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# 2.5 ns NVT reference mean (the first 100 ps block is an outlier; see text).
REFERENCE = np.array([2.459, 2.502, 2.450])

# name, one-step position MAE (A), partition (trans, rot, int) eV, longest stable NVT (ps),
# partition pass, note
MODELS = [
    ('v0', 3.590e-4, (0.111, 0.808, 6.338), 2750.0, False, 'stable, wrong partition'),
    ('250k', 2.970e-4, (0.538, 0.786, 6.044), 150.0, False, 'same bias'),
    ('Current-Only', 3.200e-4, (0.646, 0.535, 6.384), 150.0, False, 'same bias'),
    ('Native ep145', 1.037e-4, (3.256, 3.105, 1.041), 150.0, False, 'opposite bias'),
    ('M0', 2.060e-4, (3.373, 3.270, 1.000), 150.0, False, 'independent start fails'),
    ('FDR v1', 2.970e-4, (2.482, 2.533, 2.392), 2500.0, True, 'AR(1)'),
    ('GLR', 2.970e-4, (2.482, 2.526, 2.399), 2500.0, True, 'D −26%'),
    ('Published', 4.754e-3, (2.505, 2.463, 2.444), 2500.0, True, 'published checkpoint'),
]

COLORS = {
    'v0': '#D55E00',
    '250k': '#0072B2',
    'Current-Only': '#56B4E9',
    'Native ep145': '#CC79A7',
    'M0': '#E69F00',
    'FDR v1': '#009E73',
    'GLR': '#000000',
    'Published': '#8B4513',
}


def partition_error(partition):
    return float(np.mean(np.abs(np.asarray(partition) - REFERENCE)))


def load_trace(path, stride=50):
    with h5py.File(path, 'r') as handle:
        t = handle['time_fs'][::stride]
        temperature = handle['temperature_K'][::stride]
    return np.asarray(t) / 1000.0, np.asarray(temperature)


def main():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 7.5,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'legend.frameon': False,
        'pdf.fonttype': 42,
    })
    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.05],
                            wspace=0.34, hspace=0.42)

    # ---------------------------------------------------------------- a
    ax = fig.add_subplot(grid[0, 0])
    for name, mae, partition, _, passed, note in MODELS:
        marker = 'o' if passed else 'X'
        ax.scatter(mae, partition_error(partition), s=42, marker=marker,
                   color=COLORS[name], edgecolor='black', linewidth=0.4, zorder=3,
                   label=name)
    ax.annotate('', xy=(2.970e-4, partition_error((2.482, 2.526, 2.399))),
                xytext=(2.970e-4, partition_error((0.538, 0.786, 6.044))),
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color='#555555',
                                shrinkA=2, shrinkB=2))
    ax.text(3.6e-4, 0.72, 'GLR moves the\nsame backbone', fontsize=6.2, color='#555555')
    ax.set_xscale('log')
    ax.set_xlabel('one-step position MAE (Å)')
    ax.set_ylabel('mean |mode partition − reference| (eV)')
    ax.set_xlim(7e-5, 9e-3)
    ax.set_ylim(-0.15, 3.5)
    ax.axhspan(-0.15, 0.25, color='#009E73', alpha=0.08, lw=0)
    ax.text(8e-5, 0.33, 'partition correct', fontsize=6.2, color='#00735A')
    ax.grid(True, which='major', ls=':', lw=0.4, color='#cccccc', zorder=0)
    ax.set_title('pointwise accuracy does not order ensemble fidelity', fontsize=7.8,
                 loc='left', pad=4)
    # selective labels: only the points needed for the argument
    offsets = {'250k': (-26, -13), 'Native ep145': (4, 6),
               'GLR': (6, 4), 'Published': (-36, 7)}
    for name, mae, partition, _, _, _ in MODELS:
        if name in offsets:
            ax.annotate(name, (mae, partition_error(partition)),
                        xytext=offsets[name], textcoords='offset points',
                        fontsize=6.2, color=COLORS[name])

    # ---------------------------------------------------------------- b
    ax = fig.add_subplot(grid[0, 1])
    traces = [
        ('published', 'Published', 'nve_published_25ps_20261121'),
        ('m0', 'M0', 'nve_m0_25ps_20261121'),
        ('250k', '250k', 'nve_250k_25ps_20261121'),
    ]
    for key, label, directory in traces:
        t, temperature = load_trace(ROOT / 'experiments' / directory / 'trajectory.h5')
        ax.plot(t, temperature, lw=1.0, color=COLORS[label], label=label)
        ax.annotate(f'{temperature[-1]:.0f} K', (t[-1], temperature[-1]),
                    xytext=(2, 0), textcoords='offset points', fontsize=5.8,
                    color=COLORS[label], va='center')
    ax.axhline(300, color='#777777', ls=':', lw=0.6)
    ax.set_xlabel('NVE time (ps)')
    ax.set_ylabel('temperature (K)')
    ax.set_xlim(0, 25)
    ax.set_ylim(250, 510)
    ax.set_title('25 ps NVE energy drift is not the criterion', fontsize=7.8, loc='left', pad=4)
    ax.legend(loc='upper left', fontsize=6.2, handlelength=1.4)
    ax.grid(True, ls=':', lw=0.4, color='#cccccc')

    # ---------------------------------------------------------------- c
    ax = fig.add_subplot(grid[1, 0])
    order = ['v0', '250k', 'Current-Only', 'Native ep145', 'M0', 'FDR v1', 'GLR', 'Published']
    lookup = {row[0]: row for row in MODELS}
    matrix = np.array([np.asarray(lookup[name][2]) / REFERENCE for name in order])
    image = ax.imshow(matrix, cmap='RdBu_r', vmin=0.3, vmax=1.7, aspect='auto')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['translation', 'rotation', 'internal'])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = 'white' if value < 0.55 or value > 1.55 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=5.6,
                    color=color)
    ax.set_title('mode kinetic energy / reference', fontsize=7.8, loc='left', pad=4)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=6, width=0.5)
    cbar.set_label('ratio', fontsize=6.2)

    # ---------------------------------------------------------------- d
    ax = fig.add_subplot(grid[1, 1])
    entries = [
        ('CSRV', 4.3, False, 'blow-up'),
        ('250k', 150.0, False, ''),
        ('Current-Only', 150.0, False, ''),
        ('Native ep145', 150.0, False, ''),
        ('M0', 150.0, False, 'indep. 130 ps'),
        ('GLR', 2500.0, True, 'D −26%'),
        ('FDR v1', 2500.0, True, ''),
        ('Published', 2500.0, True, ''),
        ('v0', 2750.0, False, ''),
    ]
    y = np.arange(len(entries))
    for index, (name, length, passed, note) in enumerate(entries):
        color = '#009E73' if passed else '#D55E00'
        ax.barh(index, length, height=0.55, color=color, alpha=0.75, edgecolor='none')
        ax.scatter(length, index, s=16, color=color, edgecolor='black', linewidth=0.3, zorder=3)
        if note:
            ax.text(length * 1.12, index, note, va='center', fontsize=5.4, color='#444444')
    ax.set_yticks(y)
    ax.set_yticklabels([entry[0] for entry in entries])
    ax.set_xscale('log')
    ax.set_xlim(3, 9000)
    ax.axvline(100, color='#777777', ls=':', lw=0.7)
    ax.text(108, -0.45, '100 ps screen', fontsize=5.6, color='#555555')
    ax.set_xlabel('longest stable NVT (ps)')
    ax.set_title('stability alone is not correctness', fontsize=7.8, loc='left', pad=4)
    ax.grid(True, axis='x', ls=':', lw=0.4, color='#cccccc')
    ax.set_ylim(-0.6, len(entries) - 0.2)

    fig.savefig(OUT / 'fig1_diagnostic.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig1_diagnostic.png', dpi=300, bbox_inches='tight')
    payload = [dict(name=name, one_step_position_mae_A=mae, partition_eV=list(partition),
                    partition_error_eV=partition_error(partition),
                    longest_stable_nvt_ps=length, partition_pass=passed, note=note)
               for name, mae, partition, length, passed, note in MODELS]
    (OUT / 'fig1_data.json').write_text(json.dumps(dict(reference_eV=REFERENCE.tolist(),
                                                         models=payload), indent=2) + '\n')
    print(json.dumps({'output': str(OUT / 'fig1_diagnostic.pdf')}))


if __name__ == '__main__':
    main()
