#!/usr/bin/env python3
"""Render structure-ladder LOTO AUROC across geometric and structural approaches.

Six methodologically distinct 3D approaches plus pocket-shuffle and zero-pocket
controls evaluated under matched LOTO conditions on the 30 PDB-eligible target
cohort. Bars report mean AUROC; error bars where 10-seed std is available.
Two reference lines mark the matched 30-target RF+Morgan baseline and the full
65-target canonical RF+Morgan ceiling.

No interpretive annotations; bar values, axis labels, reference lines.
The caption in the manuscript carries the interpretation.
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10, 'font.family': 'sans-serif',
    'axes.linewidth': 1.0, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

# Sequential blue palette consistent with v15 colour discipline
C_HYBRID    = '#2166AC'  # dark blue: hybrid configurations
C_CONTROL   = '#92C5DE'  # light blue: pocket-shuffle and zero-pocket controls
C_STANDALONE= '#4393C3'  # mid blue: standalone geometric approaches
C_BASELINE  = '#999999'  # neutral grey: matched RF+Morgan reference

OUT_DIR = Path('figures')

# Canonical numbers from /workspace/PROTAC-Bench/results/{egnn_10seed,structure_ladder}.json
# Each entry: (label, mean, std_or_None, color, n_targets)
APPROACHES = [
    ('Matched RF+Morgan',        0.652, 0.011, C_BASELINE,  30),
    ('EGNN alone',               0.658, 0.014, C_STANDALONE, 30),
    ('EGNN hybrid',              0.820, 0.012, C_HYBRID,     30),
    ('Pocket-shuffle control',   0.814, 0.018, C_CONTROL,    30),
    ('Zero-pocket control',      0.807, 0.012, C_CONTROL,    30),
    ('Boltz-2 alone',            0.595, None,  C_STANDALONE, 60),
    ('Morgan + Boltz-2',         0.664, None,  C_HYBRID,     60),
    ('AF2-predicted pockets',    0.547, None,  C_STANDALONE, 60),
    ('smina docking score',      0.661, None,  C_STANDALONE, 30),
    ('Morgan + IFP54',           0.615, None,  C_HYBRID,     30),
    ('IFP54 alone',              0.489, None,  C_STANDALONE, 30),
    ('Morgan + pocket descr.',   0.667, None,  C_HYBRID,     30),
    ('Cocrystal binding modes',  0.497, None,  C_STANDALONE, 30),
]

REF_30T_BASELINE = 0.652
REF_FULL_COHORT  = 0.668

def main():
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    n = len(APPROACHES)
    y_pos = np.arange(n)
    means = [a[1] for a in APPROACHES]
    stds = [a[2] for a in APPROACHES]
    colors = [a[3] for a in APPROACHES]
    labels = [a[0] for a in APPROACHES]
    n_targets = [a[4] for a in APPROACHES]

    bars = ax.barh(y_pos, means, xerr=[s if s is not None else 0 for s in stds],
                   color=colors, edgecolor='black', linewidth=0.4, height=0.65,
                   error_kw={'elinewidth': 0.8, 'capsize': 2.5, 'capthick': 0.8})

    # Bar labels: AUROC value at end of bar, n_targets below
    for i, (mean, std, n_tgt) in enumerate(zip(means, stds, n_targets)):
        if std is not None:
            label_text = f'{mean:.3f}±{std:.3f}'
        else:
            label_text = f'{mean:.3f}'
        ax.text(mean + 0.025, i, label_text, va='center', ha='left',
                fontsize=8, color='black')
        ax.text(mean + 0.025, i - 0.32, f'n={n_tgt}', va='center', ha='left',
                fontsize=6.5, color='gray', style='italic')

    # Reference lines: thicker linewidth, included in legend instead of axis text
    ax.axvline(REF_30T_BASELINE, color=C_BASELINE, linestyle='--',
               linewidth=1.4, alpha=0.85, zorder=1)
    ax.axvline(REF_FULL_COHORT, color='black', linestyle=':',
               linewidth=1.4, alpha=0.75, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.40, 0.95)
    ax.set_xlabel('LOTO AUROC', fontsize=10)
    ax.set_xticks(np.arange(0.40, 1.00, 0.10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.5, zorder=0)

    # Combined legend with bar groups + reference lines, one entry per line
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=C_HYBRID,     label='Hybrid (3D + 2D)'),
        plt.Rectangle((0, 0), 1, 1, color=C_STANDALONE, label='Standalone 3D'),
        plt.Rectangle((0, 0), 1, 1, color=C_CONTROL,    label='Pocket controls'),
        plt.Rectangle((0, 0), 1, 1, color=C_BASELINE,   label='Matched RF+Morgan'),
        Line2D([0], [0], color=C_BASELINE, linestyle='--', linewidth=1.4,
               label=f'30-target matched\nbaseline ({REF_30T_BASELINE:.3f})'),
        Line2D([0], [0], color='black', linestyle=':', linewidth=1.4,
               label=f'Full 65-target\nceiling ({REF_FULL_COHORT:.3f})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7.5,
              frameon=False, ncol=1)

    fig.subplots_adjust(top=0.96, bottom=0.10, left=0.28, right=0.95)
    out = OUT_DIR / 'fig_structure_ladder_v15.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

if __name__ == '__main__':
    main()