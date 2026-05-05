#!/usr/bin/env python3
"""Fig 3: Factorial marginal contributions (delta_K, delta_A, delta_W).
Half-width 6.3 x 3.8, style matched to fig5_gap_decomposition_v14.py and figS_new_v14.py.
Reads results/factorial_16cell.json. No interpretive
annotations; per-pair dots and error bars carry the signal. Caption explains.
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
FACTORIAL_JSON = Path('results/factorial_16cell.json')

C_DBLUE = '#2166AC'
C_LBLUE = '#92C5DE'
C_ORANGE = '#E08214'
C_LORANGE = '#FDD49E'
C_GREY = '#999999'

# 2x2x2x2 factorial: cell_id is 4-char binary string M-W-A-K.
# delta_K paired comparisons: 1000->1001, 1010->1011, 1100->1101, 1110->1111
# delta_A paired comparisons: 1000->1010, 1001->1011, 1100->1110, 1101->1111
# delta_W paired comparisons: 1000->1100, 1001->1101, 1010->1110, 1011->1111
PAIRS = {
    'K': [('1000', '1001'), ('1010', '1011'), ('1100', '1101'), ('1110', '1111')],
    'A': [('1000', '1010'), ('1001', '1011'), ('1100', '1110'), ('1101', '1111')],
    'W': [('1000', '1100'), ('1001', '1101'), ('1010', '1110'), ('1011', '1111')],
}


def load_cells():
    """Map cell_id -> mean AUROC. Tries common field names defensively."""
    d = json.load(open(FACTORIAL_JSON))
    out = {}
    for c in d['cells']:
        cid = c.get('cell_id') or c.get('id') or c.get('cell')
        if cid is None:
            continue
        mean = c.get('mean_auroc') or c.get('auroc_mean') or c.get('mean')
        if mean is None and 'auroc' in c:
            v = c['auroc']
            mean = np.mean(v) if isinstance(v, list) else v
        if mean is not None:
            out[str(cid).zfill(4)] = float(mean)
    return out


def compute_marginals(cells):
    """For each factor, return list of per-pair deltas."""
    deltas = {}
    for factor, pairlist in PAIRS.items():
        ds = []
        for off, on in pairlist:
            if off in cells and on in cells:
                ds.append(cells[on] - cells[off])
        deltas[factor] = np.array(ds)
    return deltas


def generate_fig3():
    cells = load_cells()
    deltas = compute_marginals(cells)
    factors = ['K', 'A', 'W']
    labels = ['Few-shot k=5\n(K)', 'ADMET 7-feat\n(A)', 'Warhead transfer\n(W)']
    means = [deltas[f].mean() for f in factors]
    # 95% CI from per-pair distribution; 4 pairs is small so use t-based interval
    from scipy import stats
    cis = []
    for f in factors:
        v = deltas[f]
        if len(v) > 1:
            sem = stats.sem(v)
            half = sem * stats.t.ppf(0.975, len(v) - 1)
            cis.append(half)
        else:
            cis.append(0.0)
    colors = [C_DBLUE, C_LBLUE, C_ORANGE]

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.bar(range(3), means, yerr=cis, capsize=4, color=colors,
           edgecolor='black', linewidth=0.6, width=0.65,
           error_kw={'linewidth': 0.8})
    # Per-pair dots with jitter
    for i, f in enumerate(factors):
        v = deltas[f]
        jitter = np.random.RandomState(i + 7).uniform(-0.18, 0.18, len(v))
        ax.scatter([i + j for j in jitter], v, s=22, color='black',
                   alpha=0.55, edgecolors='none', zorder=5)
    # Value labels above bars (numeric only, no interpretive text)
    for i, (m, c) in enumerate(zip(means, cis)):
        y = m + c + 0.0025 if m + c > 0 else 0.0025
        ax.text(i, y, f'{m:+.4f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')
    ax.axhline(0.0, color='black', ls='-', lw=0.6, alpha=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r'$\Delta$ AUROC (paired, Morgan-anchored)', fontsize=10)
    ymax = max(m + c for m, c in zip(means, cis)) + 0.012
    ymin = min(min(deltas[f].min() for f in factors), 0) - 0.008
    ax.set_ylim(ymin, ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    outpath = OUTDIR / 'fig3_factorial_v15.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')
    print(f'  Means: K={means[0]:+.4f} A={means[1]:+.4f} W={means[2]:+.4f}')
    print(f'  95%CI half-widths: K={cis[0]:.4f} A={cis[1]:.4f} W={cis[2]:.4f}')


if __name__ == '__main__':
    generate_fig3()
    print('Done.')
