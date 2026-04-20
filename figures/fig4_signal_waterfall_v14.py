#!/usr/bin/env python3
"""Fig 4: Signal progression waterfall (0.668 -> 0.743).
Half-width (6.3 x 3.8). Dotted reference lines span full figure width.
Note: "+Cross-target transfer" uses whole-PROTAC similarity, not warhead-only.
"""
import warnings
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
C_DBLUE   = '#2166AC'
C_LBLUE   = '#92C5DE'
C_ORANGE  = '#E08214'
C_LORANGE = '#FDD49E'


def generate_fig4():
    stages = [
        ('Morgan\nbaseline',          0.668, 0.005, None),
        ('+ Cross-target\ntransfer',  0.711, 0.008, 0.042),
        ('+ ADMET\ncascade',          0.714, 0.005, 0.003),
        ('+ k=5\nfew-shot',          0.743, 0.012, 0.029),
    ]
    names = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    stds = [s[2] for s in stages]
    deltas = [s[3] for s in stages]
    colors = [C_DBLUE, C_ORANGE, C_LBLUE, C_LORANGE]

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.bar(range(len(vals)), vals, width=0.65, color=colors,
           edgecolor='black', linewidth=0.6, yerr=stds, capsize=4,
           error_kw={'linewidth': 0.8})
    # Full-width dotted reference lines at each bar height
    for i, v in enumerate(vals[:-1]):
        ax.axhline(v, color='black', ls=':', lw=0.7, alpha=0.4)
    # Value labels
    for i, (v, s) in enumerate(zip(vals, stds)):
        ax.text(i, v + s + 0.012, f'{v:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    # Delta labels between bars
    for i, d in enumerate(deltas):
        if d is not None:
            y_ref = vals[i - 1]
            y_label = y_ref + 0.006
            ax.text(i - 0.5, y_label, f'+{d:.3f}', ha='center', va='bottom',
                    fontsize=7.5, color='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0.62, 0.80)
    ax.set_ylabel('LOTO AUROC (10-seed)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    outpath = OUTDIR / 'fig4_signal_waterfall_v14.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


if __name__ == '__main__':
    generate_fig4()
    print('Done.')
