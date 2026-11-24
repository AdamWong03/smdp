"""Figure 3 — water NVT: partition, mode ACFs, RDFs, VDOS, and MSD.

Panels a–c and d–g use the 100 ps runs; panel h uses the 2.5 ns MSD when available.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/data2/wyd/research/foundation_mlip_adaptation')
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
GAIN = ROOT / 'method_development' / 'mode_gain'
GLR = ROOT / 'experiments' / 'glr_100ps_20261122' / 'analysis'
GLRI = ROOT / 'experiments' / 'glr_independent_100ps_20261122' / 'analysis'
PUB = ROOT / 'experiments' / 'nvt_screen_published_trajcast_20261121' / 'analysis'
PUBV = ROOT / 'experiments' / 'nvt_screen_published_trajcast_20261121' / 'analysis_vdos'

C_GLR = '#000000'
C_IND = '#0072B2'
C_PUB = '#D55E00'
C_REF = '#6E6E6E'
LAGS = [5, 10, 25, 50, 100, 500, 1000]


def acf(path, mode):
    data = json.loads((GAIN / path).read_text())
    values = data[mode]['autocorrelation']
    return np.array([values[str(lag)] for lag in LAGS])


def motion(path):
    data = json.loads((path / 'motion.json').read_text())
    return {key: data['metrics']['kinetic_partition'][key]['SMDP']['mean']
            for key in ('translation_eV', 'rotation_eV', 'internal_eV')}


def reference_motion(path):
    data = json.loads((path / 'motion.json').read_text())
    return {key: data['metrics']['kinetic_partition'][key]['reference']['mean']
            for key in ('translation_eV', 'rotation_eV', 'internal_eV')}


def main():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7.0,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'legend.frameon': False, 'pdf.fonttype': 42,
    })
    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(2, 4, wspace=0.45, hspace=0.55)

    # ---------------------------------------------------------------- a partition
    ax = fig.add_subplot(grid[0, 0])
    glr = motion(GLR)
    ind = motion(GLRI)
    pub = motion(PUB)
    # 2.5 ns NVT reference mean; the first 100 ps block is an outlier.
    ref = {'translation_eV': 2.4594, 'rotation_eV': 2.5021, 'internal_eV': 2.4502}
    labels = ['trans', 'rot', 'int']
    keys = ['translation_eV', 'rotation_eV', 'internal_eV']
    x = np.arange(3)
    width = 0.2
    for offset, values, label, color in [
            (-1.5, ref, 'reference', C_REF),
            (-0.5, glr, 'GLR', C_GLR),
            (0.5, ind, 'GLR indep.', C_IND),
            (1.5, pub, 'published', C_PUB)]:
        ax.bar(x + offset * width, [values[key] for key in keys], width,
               color=color, alpha=0.85, label=label, edgecolor='none')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('mode KE (eV)')
    ax.set_ylim(0, 3.2)
    ax.set_title('canonical partition', loc='left', fontsize=7.4, pad=3)
    ax.legend(fontsize=5.4, ncol=2, handlelength=1.0, columnspacing=0.6,
              loc='upper right', borderpad=0.1)
    ax.text(-0.25, 1.02, 'a', fontsize=9, weight='bold', transform=ax.transAxes)

    # ---------------------------------------------------------------- b translation ACF
    ax = fig.add_subplot(grid[0, 1])
    for path, label, color, ls in [('reference.json', 'reference', C_REF, '-'),
                                   ('glr_100ps.json', 'GLR', C_GLR, '-'),
                                   ('glr_independent.json', 'GLR indep.', C_IND, '--')]:
        ax.plot(LAGS, acf(path, 'translation'), marker='o', ms=2.2, lw=0.8,
                color=color, ls=ls, label=label)
    ax.axhline(0, color='#999999', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('lag (fs)'); ax.set_ylabel(r'$\alpha_{\rm trans}$')
    ax.set_ylim(-0.08, 1.02)
    ax.set_title('translation ACF', loc='left', fontsize=7.4, pad=3)
    ax.text(-0.25, 1.02, 'b', fontsize=9, weight='bold', transform=ax.transAxes)
    ax.legend(fontsize=5.2, loc='lower left', handlelength=1.2)

    # ---------------------------------------------------------------- c internal ACF
    ax = fig.add_subplot(grid[0, 2])
    for path, label, color, ls in [('reference.json', 'reference', C_REF, '-'),
                                   ('glr_100ps.json', 'GLR', C_GLR, '-'),
                                   ('glr_independent.json', 'GLR indep.', C_IND, '--')]:
        ax.plot(LAGS, acf(path, 'internal'), marker='o', ms=2.2, lw=0.8,
                color=color, ls=ls, label=label)
    ax.axhline(0, color='#999999', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('lag (fs)'); ax.set_ylabel(r'$\alpha_{\rm int}$')
    ax.set_ylim(-0.75, 0.45)
    ax.set_title('internal ACF', loc='left', fontsize=7.4, pad=3)
    ax.text(-0.25, 1.02, 'c', fontsize=9, weight='bold', transform=ax.transAxes)

    # ---------------------------------------------------------------- d VDOS
    ax = fig.add_subplot(grid[0, 3])
    for directory, label, color, ls in [(GLR, 'GLR', C_GLR, '-'),
                                        (PUBV, 'published', C_PUB, ':')]:
        frame = pd.read_csv(directory / 'vdos.csv')
        for element, style in [('8', '-'), ('1', '--')]:
            subset = frame[frame['element'] == int(element)].copy()
            subset['model'] = subset['model'] / subset['model'].max()
            subset['reference'] = subset['reference'] / subset['reference'].max()
            ax.plot(subset['frequency_THz'], subset['model'], color=color, ls=style,
                    lw=0.8, label=f'{label} {"O" if element=="8" else "H"}')
        for element, style in [('8', '-'), ('1', '--')]:
            subset = frame[frame['element'] == int(element)].copy()
            subset['reference'] = subset['reference'] / subset['reference'].max()
            ax.plot(subset['frequency_THz'], subset['reference'], color=C_REF, lw=0.7,
                    ls=style, alpha=0.7,
                    label=f'reference {"O" if element=="8" else "H"}')
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.08)
    ax.set_xlabel('frequency (THz)'); ax.set_ylabel('VDOS (arb.)')
    ax.set_title('vibrational DOS', loc='left', fontsize=7.4, pad=3)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C_GLR, lw=0.8, label='GLR'),
               Line2D([0], [0], color=C_PUB, lw=0.8, ls=':', label='published'),
               Line2D([0], [0], color=C_REF, lw=0.7, label='reference')]
    ax.legend(handles=handles, fontsize=5.0, loc='upper right', handlelength=1.2)
    ax.text(-0.25, 1.02, 'd', fontsize=9, weight='bold', transform=ax.transAxes)

    # ---------------------------------------------------------------- e–g RDFs
    rdf = {name: pd.read_csv(path / 'rdf.csv') for name, path in
           [('GLR', GLR), ('published', PUB)]}
    for index, (pair, title) in enumerate([('8-8', 'O–O'), ('8-1', 'O–H'), ('1-1', 'H–H')]):
        ax = fig.add_subplot(grid[1, index])
        for name, frame, color, ls in [('GLR', rdf['GLR'], C_GLR, '-'),
                                       ('published', rdf['published'], C_PUB, ':')]:
            subset = frame[frame['pair'] == pair]
            ax.plot(subset['r_A'], subset['model'], color=color, ls=ls, lw=0.9,
                    label=name)
        subset = rdf['GLR'][rdf['GLR']['pair'] == pair]
        ax.plot(subset['r_A'], subset['reference'], color=C_REF, lw=0.8, label='reference')
        ax.set_xlabel('r (Å)'); ax.set_ylabel('g(r)')
        ax.set_title(f'{title} RDF', loc='left', fontsize=7.4, pad=3)
        ax.set_xlim(0.8 if pair != '8-8' else 2.2, 5.0 if pair != '1-1' else 2.4)
        ax.text(-0.25, 1.02, 'efg'[index], fontsize=9, weight='bold',
                transform=ax.transAxes)
        if index == 0:
            ax.legend(fontsize=5.4, loc='upper right', handlelength=1.2)

    # ---------------------------------------------------------------- h diffusion / MSD
    ax = fig.add_subplot(grid[1, 3])
    val_msd = ROOT / 'experiments' / 'glr_2p5ns_20261122' / 'analysis' / 'msd.csv'
    indep_msd = ROOT / 'experiments' / 'glr_independent_2p5ns_20261122' / 'analysis' / 'msd.csv'
    if val_msd.exists():
        reference_frame = None
        for path, label, color, ls in [(val_msd, 'GLR', C_GLR, '-'),
                                       (indep_msd, 'GLR indep.', C_IND, '--')]:
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            subset = frame[frame['source'] == 'model']
            ax.plot(subset['time_ps'], subset['mean_MSD_A2'], color=color, ls=ls,
                    lw=0.9, label=label)
            if reference_frame is None:
                reference_frame = frame[frame['source'] == 'reference']
        if reference_frame is not None:
            ax.plot(reference_frame['time_ps'], reference_frame['mean_MSD_A2'],
                    color=C_REF, lw=0.9, label='reference')
        ax.set_xlim(0, 100)
        ax.set_xlabel('lag (ps)'); ax.set_ylabel(r'MSD (Å$^2$)')
        ax.set_title('diffusion', loc='left', fontsize=7.4, pad=3)
        ax.legend(fontsize=5.0, loc='upper left')
        values = []
        for path, label in [(val_msd, 'GLR'), (indep_msd, 'indep.')]:
            diffusion_path = path.parent / 'diffusion.json'
            if diffusion_path.exists():
                results = json.loads(diffusion_path.read_text())['metrics']['results']
                values.append(f"{label} {results['model']['diffusion_m2_s']['mean']*1e9:.2f}")
        if values:
            reference_d = json.loads((val_msd.parent / 'diffusion.json').read_text())['metrics']['results']['reference']['diffusion_m2_s']['mean']
            ax.text(0.05, 0.62,
                    '$D$: ' + ', '.join(values) + f'\nref {reference_d*1e9:.2f}'
                    '\n($10^{-9}$ m$^2$ s$^{-1}$)',
                    transform=ax.transAxes, fontsize=5.0, color='#333333', va='top')
    else:
        ax.text(0.5, 0.55, '2.5 ns diffusion\nrunning', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='#666666')
        ax.set_axis_off()
    ax.text(-0.25, 1.02, 'h', fontsize=9, weight='bold', transform=ax.transAxes)

    fig.savefig(OUT / 'fig3_water.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig3_water.png', dpi=300, bbox_inches='tight')
    print(OUT / 'fig3_water.pdf')


if __name__ == '__main__':
    main()
