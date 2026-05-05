#!/usr/bin/env python3
"""Fig S: Per-family Leave-One-Family-Out (LOFO) AUROC.
Horizontal bar chart with 26 family rows (sorted by mean_lofo descending).
X-axis: AUROC, 0 to 1. Bars coloured by a green->orange gradient
(green at high AUROC, orange at low). Vertical reference line at the
canonical LOTO baseline of 0.668. Family-level n_targets shown next to
each family label.

Style matched to fig2_triple_anchored_v15.py.
Input: results/lofo.json (per_family_summary list).
"""
import json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
C_GREEN = '#4DAC4A'
C_GREY = '#999999'

LOTO_BASELINE = 0.668


def load_per_family():
    with open(RDIR / 'lofo.json') as f:
        d = json.load(f)
    return d['per_family_summary']


def generate_fig():
    rows = load_per_family()
    rows = sorted(rows, key=lambda r: r['mean_lofo'], reverse=True)
    families = [r['family'] for r in rows]
    aurocs = np.array([r['mean_lofo'] for r in rows], dtype=float)
    n_targets = [r['n_targets'] for r in rows]

    cmap = LinearSegmentedColormap.from_list(
        'lofo_grad', [C_ORANGE, C_LORANGE, C_GREEN]
    )
    norm_vals = (aurocs - aurocs.min()) / max(1e-9, (aurocs.max() - aurocs.min()))
    colors = [cmap(v) for v in norm_vals]

    n = len(rows)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(6.3, 7.6))
    ax.barh(y, aurocs, color=colors, edgecolor='black', linewidth=0.6,
            height=0.72)

    ax.axvline(LOTO_BASELINE, color=C_DBLUE, ls='--', lw=1.0, alpha=0.85,
               zorder=3)
    ax.text(LOTO_BASELINE, n - 0.3, f' LOTO baseline = {LOTO_BASELINE:.3f}',
            color=C_DBLUE, fontsize=8, ha='left', va='bottom')

    for yi, m in zip(y, aurocs):
        ax.text(m + 0.008, yi, f'{m:.3f}', ha='left', va='center',
                fontsize=7.5, fontweight='bold', color='black')

    labels = [f'{fam}  (n={nt})' for fam, nt in zip(families, n_targets)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.7, n - 0.2)
    ax.set_xlabel('LOFO AUROC', fontsize=10)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.tick_params(axis='x', labelsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', ls=':', lw=0.5, color=C_GREY, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    outpath = OUTDIR / 'figS_lofo_per_family_v15.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')
    for fam, m, nt in zip(families, aurocs, n_targets):
        print(f'  {fam:30s} n={nt:<3d} LOFO={m:.4f}')
    return outpath


if __name__ == '__main__':
    p = generate_fig()
    size = p.stat().st_size
    print(f'  PDF size: {size} bytes')
    print('Done.')
