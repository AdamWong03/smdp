"""Figure 4 — ablations: memory order, AR(1) structure, displacement coupling.

Panel d is a placeholder until the rescale/noise/seed runs finish.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/data2/wyd/research/foundation_mlip_adaptation')
OUT = ROOT / 'paper' / 'figures'
FIT = ROOT / 'experiments'
GLR = ROOT / 'experiments' / 'glr_100ps_20261122' / 'analysis'
FDR = ROOT / 'experiments' / 'nvt_screen_fd_rescale_100ps_20261121' / 'analysis'
CD = ROOT / 'experiments' / 'glr_cd_100ps_20261122'

C_REF = '#6E6E6E'
C_AR1 = '#D55E00'
C_AR5 = '#E69F00'
C_AR10 = '#000000'
C_AR20 = '#0072B2'
C_GLR = '#000000'
C_FDR = '#D55E00'


def fit_acf(order):
    path = FIT / ('glr_fit_20261122' if order == 10 else f'glr_fit_p{order}_20261122')
    data = json.loads((path / 'glr_coefficients.json').read_text())
    mode = data['modes']['internal']
    return np.asarray(mode['acf']), np.asarray(mode['ar_acf'])


def main():
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.0,
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'legend.frameon': False,
                         'pdf.fonttype': 42})
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.48)

    # ---------------------------------------------------------------- a order
    ax = fig.add_subplot(grid[0, 0])
    reference, _ = fit_acf(20)
    lags = np.arange(len(reference)) * 5.0
    ax.plot(lags, reference, color=C_REF, lw=1.6, label='reference', zorder=2)
    for order, color, ls in [(1, C_AR1, '--'), (10, C_AR10, '-')]:
        _, ar = fit_acf(order)
        ax.plot(lags, ar, color=color, lw=0.9, ls=ls, label=f'AR({order})')
    ax.axhline(0, color='#999999', lw=0.5)
    ax.set_xlim(0, 200); ax.set_ylim(-0.75, 0.55)
    ax.set_xlabel('lag (fs)'); ax.set_ylabel(r'$\alpha_{\rm int}$')
    ax.set_title('internal memory kernel', loc='left', fontsize=7.4, pad=3)
    ax.text(26, 0.50, 'reference', color=C_REF, fontsize=5.8, ha='center')
    ax.text(7, 0.39, 'AR(1)', color=C_AR1, fontsize=5.8, ha='left')
    ax.text(148, 0.40, 'AR(10)', color=C_AR10, fontsize=5.8, ha='center')
    ax.text(-0.27, 1.02, 'a', fontsize=9, weight='bold', transform=ax.transAxes)

    # ---------------------------------------------------------------- b O-H RDF
    ax = fig.add_subplot(grid[0, 1])
    for name, path, color, ls in [('GLR', GLR, C_GLR, '-'),
                                  ('FDR v1 (AR1)', FDR, C_FDR, '--')]:
        frame = pd.read_csv(path / 'rdf.csv')
        subset = frame[frame['pair'] == '8-1']
        ax.plot(subset['r_A'], subset['model'], color=color, ls=ls, lw=0.9, label=name)
    frame = pd.read_csv(GLR / 'rdf.csv')
    subset = frame[frame['pair'] == '8-1']
    ax.plot(subset['r_A'], subset['reference'], color=C_REF, lw=0.8, label='reference')
    ax.set_xlim(0.8, 2.4); ax.set_ylim(0, 33)
    ax.set_xlabel('r (Å)'); ax.set_ylabel('g$_{OH}$(r)')
    ax.set_title('O–H RDF: memory order matters', loc='left', fontsize=7.4, pad=3)
    ax.legend(fontsize=5.6, loc='upper right', handlelength=1.3)
    ax.text(1.35, 18, 'RMSE\n0.104 (GLR)\n0.301 (AR1)', fontsize=5.6,
            color='#333333', va='top')
    ax.text(-0.27, 1.02, 'b', fontsize=9, weight='bold', transform=ax.transAxes)

    # ---------------------------------------------------------------- c displacement coupling
    ax = fig.add_subplot(grid[1, 0])
    with h5py.File(CD / 'trajectory.h5') as handle:
        t = handle['time_fs'][:] / 1000
        minimum = handle['minimum_distance_A'][:]
    ax.plot(t, minimum, color=C_FDR, lw=0.9)
    ax.axhline(0.65, color='#999999', ls=':', lw=0.6)
    ax.scatter([t[-1]], [minimum[-1]], color=C_FDR, s=18, zorder=3)
    ax.annotate('safety stop\n24.8 ps', (t[-1], minimum[-1]), xytext=(-48, 28),
                textcoords='offset points', fontsize=6.0, color=C_FDR,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=C_FDR))
    ax.set_xlim(0, 30); ax.set_ylim(0.6, 1.15)
    ax.set_xlabel('time (ps)'); ax.set_ylabel('minimum distance (Å)')
    ax.set_title('displacement correction is unstable', loc='left', fontsize=7.4, pad=3)
    ax.text(-0.27, 1.02, 'c', fontsize=9, weight='bold', transform=ax.transAxes)

    # ---------------------------------------------------------------- d component ablations
    ax = fig.add_subplot(grid[1, 1])

    def oh_rmse(name):
        path = ROOT / 'experiments' / f'{name}_20261122' / 'analysis' / 'rdf.json'
        data = json.loads(path.read_text())
        return float(data['metrics']['8-1']['RMSE'])

    names = ['p1', 'p5', 'p10', 'p20', 'noise off', 'rescale off']
    values = [oh_rmse('glr_p1_100ps'), oh_rmse('glr_p5_100ps'),
              oh_rmse('glr_100ps'), oh_rmse('glr_p20_100ps'),
              oh_rmse('glr_noise_off_100ps'), oh_rmse('glr_rescale_off_100ps')]
    colors = [C_AR1, C_AR5, C_AR10, C_AR20, '#56B4E9', C_FDR]
    x = np.arange(len(names))
    ax.bar(x, values, color=colors, width=0.62, edgecolor='none')
    ax.axhline(0.118, color=C_REF, ls=':', lw=0.7)
    ax.text(1.5, 0.123, 'published', fontsize=5.2, color=C_REF, ha='left')
    for index, value in enumerate(values):
        ax.text(index, value * 1.08, f'{value:.3f}', ha='center', va='bottom',
                fontsize=5.4, color='#333333')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=6.0)
    ax.set_ylabel('O–H RDF RMSE')
    ax.set_ylim(0.08, 0.75)
    ax.set_title('memory order and rescale', loc='left', fontsize=7.4, pad=3)
    ax.text(-0.27, 1.02, 'd', fontsize=9, weight='bold', transform=ax.transAxes)

    fig.savefig(OUT / 'fig4_ablations.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig4_ablations.png', dpi=300, bbox_inches='tight')
    print(OUT / 'fig4_ablations.pdf')


if __name__ == '__main__':
    main()
