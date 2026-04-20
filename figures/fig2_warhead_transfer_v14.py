#!/usr/bin/env python3
"""Fig 2: Cross-target molecular transfer.
  Panel A: Ablation (transfer rate vs warhead fragment identity)
  Panel B: LOTO vs LOFO limitation
Full-width (14 x 3.8). 10-seed data from summary_fullstack_settings.json.

Key finding: whole-PROTAC similarity transfer works (+0.042 in full stack),
but warhead-fragment-only matching does not (+0.006 n.s. in ablation).
The signal is molecular similarity, not warhead-specific pharmacology.
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


def plot_panel_a(ax):
    """Ablation: cross-target transfer rate vs warhead fragment identity."""
    # 10-seed data from summary_fullstack_settings.json
    conditions = ['C0\nMorgan', 'C1\n+Warhead FP', 'C2\n+Cross-target\ntransfer', 'C3\n+Both']
    vals = [0.677, 0.663, 0.682, 0.670]
    stds = [0.010, 0.011, 0.005, 0.010]
    colors = [C_DBLUE, C_LBLUE, C_ORANGE, C_LORANGE]
    # Try loading real data
    d = _load('exp37_warhead_ablation/summary_fullstack_settings.json')
    if d:
        conds = d.get('conditions', {})
        for i, key in enumerate(['C0', 'C1', 'C2', 'C3']):
            if key in conds:
                vals[i] = conds[key].get('mean_auroc', vals[i])
                stds[i] = conds[key].get('std_auroc', stds[i])
    bars = ax.bar(range(len(vals)), vals, width=0.65, color=colors,
                  edgecolor='black', linewidth=0.6, yerr=stds,
                  capsize=4, error_kw={'linewidth': 0.8})
    for i, (v, s) in enumerate(zip(vals, stds)):
        ax.text(i, v + s + 0.004, f'{v:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')
    # Significance bracket C1 vs C0 (hurts, p=0.012)
    y_ns = max(vals[0], vals[1]) + max(stds[0], stds[1]) + 0.016
    ax.plot([0, 0, 1, 1], [y_ns - 0.003, y_ns, y_ns, y_ns - 0.003], 'k-', lw=0.8)
    ax.text(0.5, y_ns + 0.002, 'p=0.012', ha='center', va='bottom',
            fontsize=7, fontweight='bold')
    # Significance bracket C2 vs C0 (n.s., p=0.17)
    y_sig = max(vals[0], vals[2]) + max(stds[0], stds[2]) + 0.030
    ax.plot([0, 0, 2, 2], [y_sig - 0.003, y_sig, y_sig, y_sig - 0.003], 'k-', lw=0.8)
    ax.text(1.0, y_sig + 0.002, 'n.s.', ha='center', va='bottom',
            fontsize=7, color=C_GREY)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylim(0.62, 0.73)
    ax.set_ylabel('LOTO AUROC', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('A  Ablation: Fragment vs Whole-Molecule Transfer',
                 fontsize=11, fontweight='bold', loc='left', pad=8, x=-0.05)


def plot_panel_b(ax):
    """LOTO vs LOFO."""
    # 10-seed data from summary_fullstack_settings.json
    baseline_loto = 0.677
    with_tr_loto = 0.682
    baseline_lofo = 0.576
    with_tr_lofo = 0.536
    stds_loto = [0.010, 0.005]
    stds_lofo = [0.015, 0.012]
    # Try loading real data
    d_abl = _load('exp37_warhead_ablation/summary_fullstack_settings.json')
    if d_abl:
        conds = d_abl.get('conditions', {})
        if 'C0' in conds:
            baseline_loto = conds['C0'].get('mean_auroc', baseline_loto)
            stds_loto[0] = conds['C0'].get('std_auroc', stds_loto[0])
        if 'C2' in conds:
            with_tr_loto = conds['C2'].get('mean_auroc', with_tr_loto)
            stds_loto[1] = conds['C2'].get('std_auroc', stds_loto[1])
    d_lofo = _load('exp37_warhead_lofo/summary_fullstack_settings.json')
    if d_lofo:
        conds = d_lofo.get('conditions', {})
        if 'C0_lofo' in conds:
            baseline_lofo = conds['C0_lofo'].get('mean_auroc', baseline_lofo)
            stds_lofo[0] = conds['C0_lofo'].get('std_auroc', stds_lofo[0])
        if 'C1_lofo' in conds:
            with_tr_lofo = conds['C1_lofo'].get('mean_auroc', with_tr_lofo)
            stds_lofo[1] = conds['C1_lofo'].get('std_auroc', stds_lofo[1])
    conditions = ['LOTO\n(cross-target)', 'LOFO\n(cross-family)']
    baseline = [baseline_loto, baseline_lofo]
    with_tr = [with_tr_loto, with_tr_lofo]
    all_stds_base = [stds_loto[0], stds_lofo[0]]
    all_stds_tr = [stds_loto[1], stds_lofo[1]]
    x = np.arange(len(conditions))
    w = 0.32
    ax.bar(x - w / 2, baseline, w, yerr=all_stds_base, capsize=3,
           color=C_LBLUE, edgecolor='black', linewidth=0.6,
           label='Morgan only', error_kw={'linewidth': 0.8})
    ax.bar(x + w / 2, with_tr, w, yerr=all_stds_tr, capsize=3,
           color=C_ORANGE, edgecolor='black', linewidth=0.6,
           label='Morgan + transfer', error_kw={'linewidth': 0.8})
    for i in range(len(conditions)):
        ax.text(x[i] - w / 2, baseline[i] + all_stds_base[i] + 0.004,
                f'{baseline[i]:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')
        ax.text(x[i] + w / 2, with_tr[i] + all_stds_tr[i] + 0.004,
                f'{with_tr[i]:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0.48, 0.74)
    ax.set_ylabel('AUROC', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
    ax.set_title('B  LOTO vs LOFO', fontsize=11, fontweight='bold',
                 loc='left', pad=8, x=-0.05)


def generate_fig2():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 3.8))
    fig.subplots_adjust(wspace=0.30)
    plot_panel_a(ax_a)
    plot_panel_b(ax_b)
    outpath = OUTDIR / 'fig2_warhead_transfer_v14.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


if __name__ == '__main__':
    generate_fig2()
    print('Done.')
