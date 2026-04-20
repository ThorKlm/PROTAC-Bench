#!/usr/bin/env python3
"""Fig 5: Gap decomposition — target novelty vs lab leakage.
Half-width (6.3 x 3.8). Dotted horizontal lines, labels on right side.
"""
import json, warnings
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
RDIR = Path(__file__).parent.parent / 'results'
C_DBLUE   = '#2166AC'
C_LBLUE   = '#92C5DE'
C_ORANGE  = '#E08214'
C_LORANGE = '#FDD49E'


def _load(name):
    for base in [RDIR, Path('/workspace/results'), Path('/workspace/PROTAC-Bench/results')]:
        p = base / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def generate_fig5():
    d0_mean = 0.8637
    d2_mean = 0.8236
    d1_mean = 0.6356
    d7_mean = 0.6239
    # Seed stds (not per-target)
    d0_std = 0.0075
    d2_std = 0.0102
    d1_std = 0.0080
    d7_std = 0.0268

    labels = ['Random CV', 'Random CV\n(no lab leak)', 'LOTO\n(cold target)',
              'Lab positive\nrate only\n(random CV)']
    means = [d0_mean, d2_mean, d1_mean, d7_mean]
    stds = [d0_std, d2_std, d1_std, d7_std]
    colors = [C_DBLUE, C_LBLUE, C_ORANGE, C_LORANGE]

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.bar(range(4), means, yerr=stds, capsize=4,
           color=colors, edgecolor='black', linewidth=0.6, width=0.65,
           error_kw={'linewidth': 0.8})
    # Dotted horizontal reference lines at first 3 bar heights
    for i in range(3):
        ax.axhline(means[i], color=colors[i], ls=':', lw=0.8, alpha=0.5)
    # Gap labels on right side
    lab_gap = d0_mean - d2_mean
    tgt_gap = d2_mean - d1_mean
    total = d0_mean - d1_mean
    ax.text(3.8, (d0_mean + d2_mean) / 2,
            f'Lab: {lab_gap:.3f} ({lab_gap/total:.0%})',
            ha='left', va='center', fontsize=7.5, fontweight='bold',
            color=C_DBLUE)
    ax.text(3.8, (d2_mean + d1_mean) / 2,
            f'Target: {tgt_gap:.3f} ({tgt_gap/total:.0%})',
            ha='left', va='center', fontsize=7.5, fontweight='bold',
            color=C_ORANGE)
    # Value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.012, f'{m:.3f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('AUROC', fontsize=10)
    ax.set_ylim(0.45, 0.95)
    ax.set_xlim(-0.5, 4.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0.5, color='grey', ls='--', lw=0.5, alpha=0.3)
    plt.tight_layout()
    outpath = OUTDIR / 'fig5_gap_decomposition_v14.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


if __name__ == '__main__':
    generate_fig5()
    print('Done.')
