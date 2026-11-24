"""Figure 2 — GLR architecture schematic (no data).

Two separated paths:
  a  structure:  frozen backbone -> displacement -> next position
  b  thermodynamics: mode decomposition -> AR(p) memory + FD noise -> variance rescale -> velocity
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path('/data2/wyd/research/foundation_mlip_adaptation/paper/figures')
OUT.mkdir(parents=True, exist_ok=True)

INK = '#222222'
GREY = '#6E6E6E'
BLUE = '#0072B2'
GREEN = '#009E73'
ORANGE = '#D55E00'
FILL_BACKBONE = '#E8EEF4'
FILL_READOUT = '#E6F2EC'
FILL_STATE = '#F4F4F4'


def box(ax, x, y, w, h, text, facecolor='white', edgecolor=INK, fontsize=7.2,
        lw=0.7, weight='normal', textcolor=INK, rounding=0.02):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.008,rounding_size={rounding}',
                           facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
                           zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, weight=weight, zorder=3, linespacing=1.25)
    return patch


def arrow(ax, start, end, color=INK, lw=0.9, style='-|>', rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=7,
                                 linewidth=lw, color=color,
                                 connectionstyle=f'arc3,rad={rad}',
                                 linestyle=ls, zorder=1, shrinkA=0, shrinkB=0))


def main():
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.9)
    ax.axis('off')

    # ================================================================ a: structure
    ax.text(0.05, 5.62, 'a', fontsize=12, weight='bold', color=INK)
    ax.text(0.38, 5.55, 'Structure path', fontsize=8.4, color=BLUE, va='top', weight='bold')

    box(ax, 0.10, 4.25, 1.25, 0.72, 'state\n$q_t,\\ v_t$', facecolor=FILL_STATE, fontsize=7.4)
    box(ax, 1.75, 4.10, 1.85, 1.00,
        'frozen force-free\nbackbone $f_\\theta$\n(no forces)',
        facecolor=FILL_BACKBONE, edgecolor=GREY, fontsize=7.1)
    box(ax, 4.05, 4.35, 1.10, 0.48, '$\\Delta q_t$', facecolor='white', fontsize=8.2)
    box(ax, 5.60, 4.10, 1.90, 0.95,
        'position update\n$q_{t+1}=q_t+\\Delta q_t$\n$\\rightarrow$ next step',
        facecolor=FILL_BACKBONE, edgecolor=BLUE, fontsize=6.8)

    arrow(ax, (1.35, 4.61), (1.75, 4.61))
    arrow(ax, (3.60, 4.60), (4.05, 4.60))
    arrow(ax, (5.15, 4.60), (5.60, 4.60), color=BLUE)

    # ================================================================ b: thermodynamics
    ax.text(0.05, 3.38, 'b', fontsize=12, weight='bold', color=INK)
    ax.text(0.38, 3.31, 'Thermodynamics path', fontsize=8.4, color=GREEN, va='top',
            weight='bold')

    # dashed link from backbone to the readout branch
    arrow(ax, (2.68, 4.10), (3.48, 3.10), color=GREY, lw=0.8, ls='--')
    ax.text(2.78, 3.55, '$v_t^{\\mathrm{pred}}$', fontsize=6.8, color=GREY)

    box(ax, 3.10, 2.62, 1.05, 0.46, '$v^{\\mathrm{pred}}_t$', facecolor='white', fontsize=7.0)
    box(ax, 4.55, 2.52, 1.00, 0.66, 'mode\ndecomp.', facecolor=FILL_READOUT,
        edgecolor=GREEN, fontsize=6.6)
    arrow(ax, (4.15, 2.85), (4.55, 2.85), color=GREEN)

    modes = [('translation', 2.72), ('rotation', 2.16), ('internal', 1.60)]
    for label, y in modes:
        box(ax, 5.95, y, 1.05, 0.42, label, facecolor='white', edgecolor=GREEN, fontsize=6.4)
        arrow(ax, (5.55, 2.85), (5.95, y + 0.21), color=GREEN, lw=0.7)

    box(ax, 7.40, 1.62, 1.30, 0.98,
        'AR($p$) memory\n+ FD noise\n($\\rho_a$-scaled)',
        facecolor=FILL_READOUT, edgecolor=GREEN, fontsize=6.7)
    for _, y in modes:
        arrow(ax, (7.00, y + 0.21), (7.40, 2.11), color=GREEN, lw=0.7)

    box(ax, 8.95, 1.70, 0.95, 0.55, 'per-mode\nrescale', facecolor=FILL_READOUT,
        edgecolor=GREEN, fontsize=6.4)
    arrow(ax, (8.70, 2.11), (8.95, 2.02), color=GREEN)

    box(ax, 8.45, 0.78, 1.45, 0.62,
        '$v_{t+1}^{\\mathrm{GLR}}$\n$K_a=N_a k_BT/2$', facecolor='white',
        edgecolor=GREEN, fontsize=6.8)
    arrow(ax, (9.42, 1.70), (9.42, 1.40), color=GREEN)
    arrow(ax, (9.18, 0.78), (9.18, 0.45), color=GREEN)
    box(ax, 8.45, 0.10, 1.45, 0.35, 'to next state', facecolor=FILL_STATE,
        edgecolor=GREY, fontsize=6.3)

    # blend annotation
    box(ax, 3.10, 1.62, 2.45, 0.62,
        '$v^{\\mathrm{blend}}=(1-\\lambda)v^{\\mathrm{pred}}+\\lambda\\tilde v$\n'
        '$\\lambda=0.1$',
        facecolor='white', edgecolor=ORANGE, fontsize=5.9)
    arrow(ax, (3.62, 2.62), (3.62, 2.24), color=ORANGE, lw=0.7, ls='--')
    ax.plot([5.55, 5.55, 7.40], [1.78, 1.32, 1.32], color=ORANGE, lw=0.7, ls='--',
            zorder=1)
    arrow(ax, (7.40, 1.32), (7.40, 1.62), color=ORANGE, lw=0.7, ls='--')

    # what is guaranteed
    ax.text(0.05, 1.12,
            'The readout is not integrated as a second position update: it supplies the velocity\n'
            'that enters the next backbone call and the kinetic diagnostics. The rescale enforces\n'
            '$K_a=N_a k_BT/2$ per mode; the AR($p$) stationarity matches the reference ACF.',
            fontsize=6.3, color=GREY, va='top', linespacing=1.35)

    fig.savefig(OUT / 'fig2_architecture.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig2_architecture.png', dpi=300, bbox_inches='tight')
    print(OUT / 'fig2_architecture.pdf')


if __name__ == '__main__':
    main()
