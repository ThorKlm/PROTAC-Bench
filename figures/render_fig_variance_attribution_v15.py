#!/usr/bin/env python3
"""Supplementary figure: variance attribution for the RS-LOTO gap.

Two-panel layout matched to the visual conventions of
fig2_triple_anchored_v15.py:
  - Panel A: bounded-anchor presentation (total gap + four component anchors).
  - Panel B: orthogonalised partial R^2 and Shapley attributions as grouped
    bars per anchor (two methods adjacent within each anchor group).

Inputs: results/variance_attribution.json
Output: figures/figS_variance_attribution_v15.pdf at 300 dpi
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
RDIR = Path(__file__).parent.parent / 'results'

C_DBLUE = '#2166AC'
C_LBLUE = '#92C5DE'
C_ORANGE = '#E08214'
C_LORANGE = '#FDD49E'
C_GREY = '#999999'

ANCHOR_KEYS = ['lab', 'binar', 'conflict', 'residual']
ANCHOR_COLORS = [C_DBLUE, C_LBLUE, C_LORANGE, C_ORANGE]


def load_data():
    p = RDIR / 'variance_attribution.json'
    with open(p) as f:
        return json.load(f)


def panel_a(ax, d):
    """Bounded-anchor presentation, mirroring fig2_triple_anchored_v15.py."""
    total_m = d['total_gap']['mean']
    total_s = d['total_gap']['std']
    means = [total_m] + [d['bounded_anchors'][k]['mean'] for k in ANCHOR_KEYS]
    stds = [total_s] + [d['bounded_anchors'][k]['std'] for k in ANCHOR_KEYS]
    labels = ['Total\nRS-LOTO gap'] + [d['bounded_anchors'][k]['label']
                                       for k in ANCHOR_KEYS]
    colors = [C_GREY] + ANCHOR_COLORS

    ax.bar(range(5), means, yerr=stds, capsize=4, color=colors,
           edgecolor='black', linewidth=0.6, width=0.65,
           error_kw={'linewidth': 0.8})
    ax.axhline(means[0], color=C_GREY, ls=':', lw=0.8, alpha=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.004, f'{m:.3f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')
    ax.axhline(0.0, color='black', ls='-', lw=0.6, alpha=0.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r'$\Delta$ AUROC contribution', fontsize=10)
    ax.set_ylim(-0.01, max(means) + max(stds) + 0.025)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('A  Bounded anchors', loc='left', fontsize=11,
                 fontweight='bold', pad=6)


def panel_b(ax, d):
    """Grouped bars: orthogonalised partial R^2 vs Shapley per anchor."""
    pr2_m = [d['orth_partial_r2'][k]['mean'] for k in ANCHOR_KEYS]
    pr2_s = [d['orth_partial_r2'][k]['std'] for k in ANCHOR_KEYS]
    sh_m = [d['shapley'][k]['mean'] for k in ANCHOR_KEYS]
    sh_s = [d['shapley'][k]['std'] for k in ANCHOR_KEYS]
    labels = [d['bounded_anchors'][k]['label'] for k in ANCHOR_KEYS]

    x = np.arange(len(ANCHOR_KEYS))
    w = 0.36
    ax.bar(x - w / 2, pr2_m, w, yerr=pr2_s, capsize=3, color=C_DBLUE,
           edgecolor='black', linewidth=0.6,
           error_kw={'linewidth': 0.8}, label=r'Orth. partial $R^{2}$')
    ax.bar(x + w / 2, sh_m, w, yerr=sh_s, capsize=3, color=C_ORANGE,
           edgecolor='black', linewidth=0.6,
           error_kw={'linewidth': 0.8}, label='Shapley')

    for xi, m, s in zip(x - w / 2, pr2_m, pr2_s):
        ax.text(xi, m + s + 0.003, f'{m:.3f}', ha='center', va='bottom',
                fontsize=7.5)
    for xi, m, s in zip(x + w / 2, sh_m, sh_s):
        ax.text(xi, m + s + 0.003, f'{m:.3f}', ha='center', va='bottom',
                fontsize=7.5)

    ax.axhline(0.0, color='black', ls='-', lw=0.6, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r'$\Delta$ AUROC attribution', fontsize=10)
    ymax = max(max(np.array(pr2_m) + np.array(pr2_s)),
               max(np.array(sh_m) + np.array(sh_s)))
    ax.set_ylim(-0.005, ymax + 0.025)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right',
              handlelength=1.2, borderaxespad=0.3)
    ax.set_title('B  Orthogonalised partial $R^{2}$ vs Shapley',
                 loc='left', fontsize=11, fontweight='bold', pad=6)


def generate_figure():
    d = load_data()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 3.8))
    panel_a(axA, d)
    panel_b(axB, d)
    plt.tight_layout()
    outpath = OUTDIR / 'figS_variance_attribution_v15.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')

    print('  Bounded anchors:')
    print(f'    {"Total":15s} {d["total_gap"]["mean"]:+.4f} '
          f'+/- {d["total_gap"]["std"]:.4f}')
    for k in ANCHOR_KEYS:
        a = d['bounded_anchors'][k]
        print(f'    {k:15s} {a["mean"]:+.4f} +/- {a["std"]:.4f}')
    print('  Orthogonalised partial R^2 / Shapley:')
    for k in ANCHOR_KEYS:
        p = d['orth_partial_r2'][k]
        s = d['shapley'][k]
        print(f'    {k:15s} pR2 {p["mean"]:+.4f} +/- {p["std"]:.4f}   '
              f'Shapley {s["mean"]:+.4f} +/- {s["std"]:.4f}')
    return outpath


if __name__ == '__main__':
    generate_figure()
    print('Done.')
