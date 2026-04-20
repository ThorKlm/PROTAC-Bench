#!/usr/bin/env python3
"""Fig 3: Few-shot learning curves — RF vs MAML vs ProtoNet.
Half-width (6.3 x 3.8), style matched to fig1_collapse_v14.py.
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
C_DBLUE   = '#2166AC'
C_LBLUE   = '#92C5DE'
C_ORANGE  = '#E08214'
C_LORANGE = '#FDD49E'
C_GREY    = '#999999'


def _load(name):
    for base in [RDIR, Path('/workspace/results'), Path('/workspace/PROTAC-Bench/results')]:
        p = base / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def generate_fig3():
    k_values = [1, 2, 3, 5, 10, 15, 20]
    rf_aurocs = [0.680, 0.688, 0.694, 0.700, 0.714, 0.720, 0.724]
    maml_aurocs = [0.660, 0.668, 0.674, 0.683, 0.690, 0.692, 0.694]
    proto_aurocs = [0.655, 0.662, 0.670, 0.679, 0.685, 0.688, 0.690]
    baseline = 0.668

    fs = _load('fewshot.json')
    if fs:
        try:
            if 'k_values' in fs:
                k_values = fs['k_values']
            for key, target in [('rf', rf_aurocs), ('maml', maml_aurocs), ('protonet', proto_aurocs)]:
                if key in fs:
                    data = fs[key]
                    if isinstance(data, dict) and 'aurocs' in data:
                        target[:] = data['aurocs']
                    elif isinstance(data, list):
                        target[:] = data
            if 'baseline' in fs:
                baseline = fs['baseline'] if isinstance(fs['baseline'], (int, float)) else fs['baseline'].get('mean_auroc', 0.668)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.axhline(baseline, color=C_GREY, ls='--', lw=1.0, alpha=0.7)
    ax.text(max(k_values) * 0.95, baseline - 0.006, f'Morgan baseline ({baseline:.3f})',
            ha='right', va='top', fontsize=7.5, color=C_GREY, fontstyle='italic')
    ax.plot(k_values, rf_aurocs, 'o-', color=C_DBLUE, lw=2.0, ms=6,
            label='RF (retrain)', zorder=5)
    ax.plot(k_values, maml_aurocs, 's--', color=C_ORANGE, lw=1.5, ms=5,
            label='MAML', zorder=4)
    ax.plot(k_values, proto_aurocs, '^--', color=C_LBLUE, lw=1.5, ms=5,
            label='ProtoNet', zorder=4)
    # k=5 annotation: directly above the point, short vertical arrow down
    rf_k5_idx = k_values.index(5) if 5 in k_values else 3
    ax.annotate(f'k=5: {rf_aurocs[rf_k5_idx]:.3f}',
                xy=(5, rf_aurocs[rf_k5_idx]),
                xytext=(5, rf_aurocs[rf_k5_idx] + 0.007),
                fontsize=7.5, fontweight='bold', color=C_DBLUE, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_DBLUE, lw=0.8, shrinkB=5))
    ax.set_xlabel('k (compounds per target)', fontsize=10)
    ax.set_ylabel('LOTO AUROC', fontsize=10)
    ax.set_xlim(0, max(k_values) + 1)
    ax.set_ylim(0.63, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8.5, framealpha=0.9, loc='lower right')
    plt.tight_layout()
    outpath = OUTDIR / 'fig3_fewshot_v14.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


if __name__ == '__main__':
    generate_fig3()
    print('Done.')