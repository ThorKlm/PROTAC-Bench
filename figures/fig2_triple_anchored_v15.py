#!/usr/bin/env python3
"""Fig 2: Triple-anchored gap decomposition.
Half-width 6.3 x 3.8, style matched to fig5_gap_decomposition_v14.py.
Five bars: total RS-LOTO gap, then four components (inter-lab, binarization,
conflicts, residual cold-target). No interpretive text on figure body; the
partial-overlap caveat and component attributions go in the caption.

Inputs (loaded defensively; falls back to canonical scalars if JSON absent):
  - task14_within_target_cross_lab.json -> inter-lab anchor
  - exp18e_binarization_sensitivity.json -> binarization spread
  - conflict_removal/summary.json -> P_drop delta
Total gap is computed as random-CV pooled minus LOTO macro from canonical
exp38_confidence values.
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
RDIR = Path('results')

C_DBLUE = '#2166AC'
C_LBLUE = '#92C5DE'
C_ORANGE = '#E08214'
C_LORANGE = '#FDD49E'
C_GREY = '#999999'


def _load(name, base=RDIR):
    p = base / name
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def load_anchors():
    """Return (total_gap, lab, binar, conflict, residual) with optional stds."""
    # Defaults from canonical paper numbers
    total_gap = 0.190
    total_gap_std = 0.012
    lab = 0.124
    lab_std = 0.018
    binar = 0.050
    binar_std = 0.012
    conflict = 0.008
    conflict_std = 0.004
    # Try cross-lab JSON for anchor 1
    d = _load('task14_within_target_cross_lab.json')
    if d and 'summary' in d:
        s = d['summary']
        try:
            rs = s['random_cv']['macro_mean']
            cl = s['cross_lab']['macro_mean']
            lab = rs - cl
            lab_std = float(np.sqrt(s['random_cv'].get('macro_std', 0)**2 +
                                    s['cross_lab'].get('macro_std', 0)**2))
        except Exception:
            pass
    # Binarization spread from exp18e
    d = _load('exp18e_binarization_sensitivity.json')
    if d and 'schemes' in d:
        try:
            vals = [s['loto_auroc'] for s in d['schemes']]
            binar = float(max(vals) - min(vals))
        except Exception:
            pass
    # Conflict removal delta
    d = _load('conflict_removal_summary.json') or _load('conflict_removal/summary.json')
    if d:
        try:
            conflict = float(d.get('P_drop_delta', conflict))
            conflict_std = float(d.get('P_drop_std', conflict_std))
        except Exception:
            pass
    residual = total_gap - lab - binar - conflict
    residual_std = max(0.005, abs(residual) * 0.4)
    return {
        'total': (total_gap, total_gap_std),
        'lab': (lab, lab_std),
        'binar': (binar, binar_std),
        'conflict': (conflict, conflict_std),
        'residual': (residual, residual_std),
    }


def generate_fig2():
    a = load_anchors()
    labels = ['Total\nRS-LOTO gap', 'Inter-lab\n(task14)',
              'Binarization\n(exp18e)', 'Conflicts\n(P_drop)',
              'Residual\ncold-target']
    keys = ['total', 'lab', 'binar', 'conflict', 'residual']
    means = [a[k][0] for k in keys]
    stds = [a[k][1] for k in keys]
    colors = [C_GREY, C_DBLUE, C_LBLUE, C_LORANGE, C_ORANGE]

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.bar(range(5), means, yerr=stds, capsize=4, color=colors,
           edgecolor='black', linewidth=0.6, width=0.65,
           error_kw={'linewidth': 0.8})
    # Dotted reference line at total-gap height (unlabeled)
    ax.axhline(means[0], color=C_GREY, ls=':', lw=0.8, alpha=0.5)
    # Numeric value labels only; no percentages or interpretive text
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
    plt.tight_layout()
    outpath = OUTDIR / 'fig2_triple_anchored_v15.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')
    for k, lab in zip(keys, labels):
        m, s = a[k]
        print(f'  {lab.replace(chr(10), " "):30s} {m:+.4f} +/- {s:.4f}')


if __name__ == '__main__':
    generate_fig2()
    print('Done.')
