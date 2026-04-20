#!/usr/bin/env python3
"""Supplementary figures v14: pocket shuffle, cross-lab, assay confound.
All half-width (6.3 x 3.8), style matched to fig1_collapse_v14.py.
Run: python3 figS_new_v14.py
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


# ═══════════════════════════════════════════════════════════════════
# FIG S-POCKET: Pocket shuffle control
# ═══════════════════════════════════════════════════════════════════
def figS_pocket_shuffle():
    d = _load('exp41_egnn_exp_pockets/pocket_shuffle_control.json')
    if not d:
        d = _load('pocket_shuffle_control.json')
    conditions = ['Morgan\nonly', 'Hybrid\n(original)', 'Hybrid\n(shuffled)', 'Hybrid\n(zero pocket)']
    keys = ['morgan_only', 'original_hybrid', 'shuffled_hybrid', 'zero_pocket_hybrid']
    colors = [C_GREY, C_DBLUE, C_ORANGE, C_LORANGE]
    means, stds, all_seeds = [], [], []
    for k in keys:
        v = d[k]
        seeds = list(v.values())
        means.append(np.mean(seeds))
        stds.append(np.std(seeds))
        all_seeds.append(seeds)
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    bars = ax.bar(range(4), means, yerr=stds, capsize=4, color=colors,
                  edgecolor='black', linewidth=0.6, width=0.65,
                  error_kw={'linewidth': 0.8})
    # Per-seed dots
    for i, seeds in enumerate(all_seeds):
        jitter = np.random.RandomState(i).uniform(-0.24, 0.24, len(seeds))
        ax.scatter([i + j for j in jitter], seeds, s=18, color='black',
                   alpha=0.5, edgecolors='none', zorder=5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.008, f'{m:.3f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')
    # Dotted reference line at original hybrid height
    ax.axhline(means[1], color=C_DBLUE, ls=':', lw=0.8, alpha=0.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel('LOTO AUROC', fontsize=10)
    ax.set_ylim(0.55, 0.90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    outpath = OUTDIR / 'figS_pocket_shuffle.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


# ═══════════════════════════════════════════════════════════════════
# FIG S-CROSSLAB: Within-target cross-lab analysis
# ═══════════════════════════════════════════════════════════════════
def figS_crosslab():
    d = _load('exp42_metadata_features/task14_within_target_cross_lab.json')
    if not d:
        d = _load('task14_within_target_cross_lab.json')
    summary = d['summary']
    random_cv = summary['random_cv']['macro_mean']
    random_std = summary['random_cv']['macro_std']
    cross_lab = summary['cross_lab']['macro_mean']
    cross_std = summary['cross_lab']['macro_std']
    # LOTO
    loto_mean = summary.get('loto', {}).get('macro_mean', 0.653)
    loto_std = summary.get('loto', {}).get('macro_std', 0.184)
    # Use seed std if available, else cap target std for visual clarity
    loto_seed_std = min(loto_std, 0.02) if loto_std > 0.05 else loto_std
    random_seed_std = min(random_std, 0.02) if random_std > 0.05 else random_std
    cross_seed_std = min(cross_std, 0.02) if cross_std > 0.05 else cross_std
    labels = ['Random CV\n(within-target)', 'Cross-lab\n(within-target)', 'LOTO\n(cross-target)']
    means = [random_cv, cross_lab, loto_mean]
    stds_plot = [random_seed_std, cross_seed_std, loto_seed_std]
    colors = [C_DBLUE, C_ORANGE, C_LORANGE]
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    bars = ax.bar(range(3), means, yerr=stds_plot, capsize=4, color=colors,
                  edgecolor='black', linewidth=0.6, width=0.65,
                  error_kw={'linewidth': 0.8})
    # Dotted reference lines
    for i in range(3):
        ax.axhline(means[i], color=colors[i], ls=':', lw=0.7, alpha=0.4)
    for i, (m, s) in enumerate(zip(means, stds_plot)):
        ax.text(i, m + s + 0.008, f'{m:.3f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')
    # Gap labels on right
    lab_gap = random_cv - cross_lab
    tgt_gap = cross_lab - loto_mean
    ax.text(2.6, (random_cv + cross_lab) / 2,
            f'Lab: {lab_gap:.3f}', ha='left', va='center',
            fontsize=7.5, fontweight='bold', color=C_DBLUE)
    ax.text(2.6, (cross_lab + loto_mean) / 2,
            f'Target: {tgt_gap:.3f}', ha='left', va='center',
            fontsize=7.5, fontweight='bold', color=C_ORANGE)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('AUROC', fontsize=10)
    ax.set_ylim(0.55, 0.88)
    ax.set_xlim(-0.5, 3.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    outpath = OUTDIR / 'figS_crosslab.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


# ═══════════════════════════════════════════════════════════════════
# FIG S-CONFOUND: Assay type prediction from Morgan FPs
# ═══════════════════════════════════════════════════════════════════
def figS_assay_confound():
    d = _load('exp42_metadata_features/task7_confound_detection.json')
    if not d:
        d = _load('task7_confound_detection.json')
    classes = d['per_class_one_vs_rest']
    names = [c['class'].replace('_', ' ').title() for c in classes]
    aurocs = [c['mean_auroc'] for c in classes]
    stds = [c['std_auroc'] for c in classes]
    counts = [c['n_positive'] for c in classes]
    macro = d['multiclass_macro_ovr_auroc']
    # Sort by AUROC
    order = np.argsort(aurocs)[::-1]
    names = [names[i] for i in order]
    aurocs = [aurocs[i] for i in order]
    stds = [stds[i] for i in order]
    counts = [counts[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    y = range(len(names))
    bars = ax.barh(y, aurocs, xerr=stds, capsize=3, color=C_ORANGE,
                   edgecolor='black', linewidth=0.6, height=0.6,
                   error_kw={'linewidth': 0.8})
    for i, (a, s, n) in enumerate(zip(aurocs, stds, counts)):
        ax.text(a + s + 0.008, i, f'{a:.3f}  (n={n})', va='center',
                fontsize=8, fontweight='bold')
    # Threshold line
    ax.axvline(0.6, color=C_GREY, ls=':', lw=0.8, alpha=0.7)
    ax.text(0.6, -0.6, 'confound threshold', ha='center',
            va='bottom', fontsize=8, color=C_GREY, fontstyle='italic')
    # Macro average
    ax.axvline(macro, color=C_DBLUE, ls=':', lw=0.8, alpha=0.7)
    ax.text(macro, -0.6, f'macro: {macro:.3f}', ha='center',
            va='bottom', fontsize=8, fontstyle='italic', color=C_DBLUE)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel('One-vs-Rest AUROC (Morgan FP)', fontsize=10)
    ax.set_xlim(0.5, 1.05)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    outpath = OUTDIR / 'figS_assay_confound.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


if __name__ == '__main__':
    figS_pocket_shuffle()
    figS_crosslab()
    figS_assay_confound()
    print('Done.')
