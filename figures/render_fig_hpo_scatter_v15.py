#!/usr/bin/env python3
"""Fig S: HPO objective vs 5-seed validated AUROC scatter.
Top-10 HPO V2 configs from results/hpo_v2_validation.json. Diagonal y=x grey
dashed reference, RF+Morgan baseline (0.668) horizontal dotted reference,
rank-1 highlighted, ranks 2-10 in a uniform neutral colour. Style matched to
fig2_triple_anchored_v15.py.
"""
import json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams.update({
    'font.size': 11, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'xtick.major.width': 1, 'ytick.major.width': 1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})
OUTDIR = Path(__file__).parent
RDIR = Path(__file__).resolve().parent.parent / 'results'

C_DBLUE = '#2166AC'
C_LBLUE = '#92C5DE'
C_ORANGE = '#E08214'
C_LORANGE = '#FDD49E'
C_GREY = '#999999'

BASELINE_RF_MORGAN = 0.668


def load_configs():
    with open(RDIR / 'hpo_v2_validation.json') as f:
        d = json.load(f)
    return d['configs']


def generate_fig():
    configs = load_configs()
    ranks = np.array([c['rank'] for c in configs])
    x = np.array([c['hpo_auroc'] for c in configs])
    y = np.array([c['validated_auroc'] for c in configs])
    yerr = np.array([c['std'] for c in configs])

    fig, ax = plt.subplots(figsize=(6.3, 5.4))

    lo, hi = 0.55, 0.80
    ax.plot([lo, hi], [lo, hi], ls='--', color=C_GREY, lw=1.0,
            alpha=0.7, label='y = x', zorder=1)
    ax.axhline(BASELINE_RF_MORGAN, ls=':', color='black', lw=1.0, alpha=0.7,
               label=f'RF+Morgan baseline ({BASELINE_RF_MORGAN:.3f})', zorder=1)

    is_top = ranks == 1
    other = ~is_top

    ax.errorbar(x[other], y[other], yerr=yerr[other], fmt='o',
                color=C_DBLUE, ecolor=C_DBLUE, mfc=C_DBLUE, mec='black',
                mew=0.6, ms=8, capsize=3, elinewidth=0.9, alpha=0.9,
                label='Ranks 2-10', zorder=3)
    ax.errorbar(x[is_top], y[is_top], yerr=yerr[is_top], fmt='o',
                color=C_ORANGE, ecolor=C_ORANGE, mfc=C_ORANGE, mec='black',
                mew=0.8, ms=10, capsize=3, elinewidth=1.0,
                label='Rank 1 (HPO winner)', zorder=4)

    for r, xi, yi in zip(ranks, x, y):
        dx, dy = 0.006, 0.006
        ha, va = 'left', 'bottom'
        if r == 1:
            dx, dy = 0.008, -0.012
            ha, va = 'left', 'top'
        color = C_ORANGE if r == 1 else 'black'
        weight = 'bold' if r == 1 else 'normal'
        ax.text(xi + dx, yi + dy, str(int(r)), fontsize=9,
                ha=ha, va=va, color=color, fontweight=weight, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('HPO objective AUROC', fontsize=11)
    ax.set_ylabel('5-seed validated AUROC', fontsize=11)
    ax.tick_params(axis='both', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, ls=':', lw=0.5, color=C_GREY, alpha=0.4, zorder=0)
    ax.legend(loc='upper left', fontsize=8.5, frameon=True,
              framealpha=0.95, edgecolor=C_GREY)

    plt.tight_layout()
    outpath = OUTDIR / 'figS_hpo_scatter_v15.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')
    for r, xi, yi, e in zip(ranks, x, y, yerr):
        print(f'  rank {int(r):2d}  hpo={xi:.4f}  val={yi:.4f} +/- {e:.4f}')


if __name__ == '__main__':
    generate_fig()
    print('Done.')
