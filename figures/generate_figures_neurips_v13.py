#!/usr/bin/env python3
"""Generate NeurIPS publication figures v13 — incorporates all final results.

Changes from v12:
- Fig 4: Add STAN attention (0.714) + warhead transfer (0.713); waterfall to 0.743
- Fig 1A: Verify DM LOTO bar (0.702) included (unchanged from v12)
- Fig S12: Add Chemprop HPO (0.600) alongside GIN (0.608) and D-MPNN (0.620)
- NEW Fig S15: STAN ablation (C0-C3: full, zeroPOI, randomPOI, RF+ESM)
- NEW Fig S16: Temporal prospective (random/LOTO/temporal/temporal+stack)
- NEW Fig S17: Pseudo-labeling (LOTO vs random with/without pseudo-labels)
- NEW Fig S18: exp39 STAN combined progression

Data sources:
  exp38_confidence/summary.json        → 10-seed stack (0.668→0.743)
  exp38_stan_ablation/summary.json     → STAN ablation (C0-C3)
  exp38_stan_full/summary.json         → STAN attention (0.718)
  exp38_temporal/summary.json          → Temporal split results
  exp38_pseudolabel/summary.json       → Pseudo-labeling results
  exp38_chemprop_hpo/summary.json      → Chemprop HPO (0.600)
  exp38_e3_warhead/summary.json        → Warhead transfer (0.713)
  exp39_stan_combined/summary.json     → STAN combined progression

Run: python scripts/generate_figures_neurips_v13.py
"""
import json, warnings, sys, shutil
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

# ── Output dir ────────────────────────────────────────────────────────────────
RDIR = Path('/workspace/results/neurips_figures_v13')
RDIR.mkdir(parents=True, exist_ok=True)
V12_DIR = Path('/workspace/results/neurips_figures_v12')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'xtick.major.width': 1, 'ytick.major.width': 1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})
CB_BLUE    = '#0072B2'
CB_ORANGE  = '#E69F00'
CB_GREEN   = '#009E73'
CB_RED     = '#D55E00'
CB_PURPLE  = '#CC79A7'
CB_GREY    = '#999999'
CB_LBLUE   = '#56B4E9'
CB_YELLOW  = '#F0E442'
CB_TEAL    = '#44AA99'
CB_DBLUE   = '#003F5C'

def _load_json(path):
    with open(path) as f:
        return json.load(f)

def _save(fig, name):
    fig.savefig(RDIR / f'{name}.pdf')
    fig.savefig(RDIR / f'{name}.png', dpi=300)
    plt.close(fig)
    print(f'  saved: {name}')

def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def copy_unchanged_figures():
    """Copy unchanged figures from v12."""
    unchanged = [
        'fig2_perfold', 'fig3_hpo_ceiling', 'fig5_fewshot',
        'fig6_egnn_scatter', 'fig7_plm_scaling', 'fig8_hpo_dimensions',
        'figS1_fragments', 'figS2_threshold', 'figS3_temporal',
        'figS4_cross_dataset', 'figS5_docking', 'figS6_feature_corr',
        'figS7_scaffold', 'figS8_egnn_smina', 'figS9_protein_family',
        'figS10_calibration', 'figS11_learning_curve',
        'fig1a_dm_loto_collapse', 'fig1a_collapse_v12',
        'figS13_warhead_transfer', 'figS14_selective_prediction',
    ]
    copied = 0
    for name in unchanged:
        for ext in ['.pdf', '.png']:
            src = V12_DIR / f'{name}{ext}'
            if src.exists():
                shutil.copy2(src, RDIR / f'{name}{ext}')
                copied += 1
    print(f'  Copied {copied} unchanged files from v12')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1A: COLLAPSE — unchanged from v12, just copy (DM LOTO 0.702 already in)
# ═══════════════════════════════════════════════════════════════════════════════
# (Copied in copy_unchanged_figures)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4: BREAKTHROUGHS — v13: STAN attention + warhead transfer + full stack
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_breakthroughs():
    """Waterfall chart: full stack progression to 0.743.
    v13: Adds STAN attention (0.714) and warhead transfer (0.713) as
    separate signals. Shows complete RF → full stack progression."""
    fig, ax = plt.subplots(figsize=(9, 8))

    # Data from exp38_confidence (10-seed validated) and exp38_e3_warhead
    conf = _load_json('results/exp38_confidence/summary.json')
    c0 = conf['C0_morgan_only']['mean_auroc']         # 0.668
    c1 = conf['C1_morgan+warhead']['mean_auroc']       # 0.711
    c2 = conf['C2_morgan+warhead+admet7']['mean_auroc'] # 0.714
    c3 = conf['C3_morgan+warhead+admet7+k5']['mean_auroc'] # 0.743

    c0_std = conf['C0_morgan_only']['std']
    c3_std = conf['C3_morgan+warhead+admet7+k5']['std']

    # STAN attention from exp38_stan_full
    stan = _load_json('results/exp38_stan_full/summary.json')
    stan_auroc = stan['mean_auroc']  # 0.718

    # Warhead transfer from exp38_e3_warhead
    wh = _load_json('results/exp38_e3_warhead/summary.json')
    wh_auroc = wh['C1_morgan+pooled_transfer']['mean_auroc']  # 0.713

    entries = [
        ('RF+Morgan\n(baseline, 65 tgts)', c0, CB_GREY, False, None, c0_std),
        ('+Warhead transfer\n(65 tgts)', c1, CB_TEAL, False,
         f'p={conf["C1_morgan+warhead"]["ci95_low"]:.3f}-{conf["C1_morgan+warhead"]["ci95_high"]:.3f}', 0),
        ('+ADMET descriptors\n(65 tgts)', c2, CB_BLUE, False, None, 0),
        ('+k=5 few-shot\n(full stack, 65 tgts)', c3, CB_GREEN, False,
         f'p={conf["C3_morgan+warhead+admet7+k5"]["ci95_low"]:.3f}-{conf["C3_morgan+warhead+admet7+k5"]["ci95_high"]:.3f}', c3_std),
        ('', 0, 'white', False, None, 0),  # spacer
        ('STAN attention\n(64 tgts, indep.)', stan_auroc, CB_PURPLE, False,
         f'p={stan["wilcoxon_p"]:.4f}', 0),
        ('Warhead transfer\n(65 tgts, indep.)', wh_auroc, CB_TEAL, True,
         f'p={wh["C1_morgan+pooled_transfer"]["wilcoxon_p"]:.4f}', 0),
    ]

    y_pos = np.arange(len(entries))[::-1]
    for i, (name, val, color, hatched, ptext, std) in enumerate(entries):
        if not name:  # spacer
            continue
        hatch = '///' if hatched else None
        ec = color if hatched else 'white'
        ax.barh(y_pos[i], val, color=color, edgecolor=ec,
                linewidth=0.8, height=0.55, hatch=hatch, alpha=0.85)
        if std and std > 0:
            ax.errorbar(val, y_pos[i], xerr=std, fmt='none', ecolor='black',
                        capsize=3, capthick=0.8, linewidth=0.8, zorder=6)
        offset_x = (std + 0.015) if (std and std > 0) else 0.01
        ax.text(val + offset_x, y_pos[i], f'{val:.3f}',
                va='center', fontsize=10, fontweight='bold')
        if ptext:
            ax.text(val + offset_x, y_pos[i] - 0.22,
                    ptext, va='top', fontsize=8, color='#555', style='italic')

    # Separator line between stack and independent signals
    spacer_y = y_pos[4]
    ax.axhline(spacer_y, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.text(0.02, spacer_y + 0.15, 'Independent signals (not stacked)',
            fontsize=8, color='#555', style='italic')

    # Bracket showing the stack progression
    stack_top = y_pos[0]
    stack_bot = y_pos[3]
    ax.annotate('', xy=(0.78, stack_top + 0.3), xytext=(0.78, stack_bot - 0.3),
                arrowprops=dict(arrowstyle='<->', color=CB_DBLUE, lw=1.5))
    delta = c3 - c0
    ax.text(0.80, (stack_top + stack_bot) / 2,
            f'$\\Delta$={delta:+.3f}\nfull stack',
            fontsize=9, fontweight='bold', color=CB_DBLUE, va='center')

    ax.set_yticks([y for i, y in enumerate(y_pos) if entries[i][0]])
    ax.set_yticklabels([e[0] for e in entries if e[0]], fontsize=9)
    ax.set_xlabel('LOTO AUROC (error bars: seed std)')
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.3)
    ax.axvline(c0, color=CB_GREY, ls=':', lw=1, alpha=0.4)
    _despine(ax)
    ax.set_title('Incremental Improvements Over Baseline (v13)', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig4_breakthroughs')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG S12: GNN BASELINES — v13: Add Chemprop HPO (0.600)
# ═══════════════════════════════════════════════════════════════════════════════
def figS12_gnn_baselines():
    """Compare GIN, D-MPNN, and Chemprop HPO against RF+Morgan."""
    gin = _load_json('results/exp13_gnn_baselines/summary.json')
    gin_auroc = gin['gin_mean_auroc']  # 0.608

    dmpnn = _load_json('results/exp33_chemprop/summary.json')
    dmpnn_auroc = dmpnn['dmpnn_mean_auroc']  # 0.620

    hpo = _load_json('results/exp38_chemprop_hpo/summary.json')
    hpo_auroc = hpo['final_evaluation']['mean_auroc']  # 0.600

    dmpnn_per = dmpnn['dmpnn_per_target']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: Grouped bar chart — now 4 methods
    methods = ['RF+Morgan\n(65 tgts)', 'GIN\n(10 tgts)', 'D-MPNN\n(15 tgts)',
               'D-MPNN HPO\n(15 tgts)']
    aurocs = [0.666, gin_auroc, dmpnn_auroc, hpo_auroc]
    colors = [CB_GREY, CB_GREEN, CB_PURPLE, CB_ORANGE]

    bars = ax1.bar(range(len(methods)), aurocs, color=colors, edgecolor='white',
                    width=0.55, alpha=0.85)
    for i, v in enumerate(aurocs):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    ax1.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax1.axhline(0.666, color=CB_GREY, ls=':', lw=1, alpha=0.5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel('Mean LOTO AUROC')
    ax1.set_ylim(0, 0.85)
    _despine(ax1)
    ax1.set_title('A  GNN Baselines vs RF+Morgan', fontweight='bold', loc='left')

    # Annotation: HPO doesn't help
    ax1.annotate('HPO $\\rightarrow$ no gain',
                 xy=(3, hpo_auroc), xytext=(3.3, 0.70),
                 fontsize=9, fontweight='bold', color=CB_RED,
                 arrowprops=dict(arrowstyle='->', color=CB_RED, lw=1.2))

    # Right: D-MPNN per-target scatter vs RF
    targets = list(dmpnn_per.keys())
    dmpnn_vals = [dmpnn_per[t]['mean'] for t in targets]
    rf_per = dmpnn.get('rf_per_target', {})
    rf_vals = [rf_per[t]['mean'] for t in targets] if rf_per else dmpnn_vals

    ax2.scatter(rf_vals, dmpnn_vals, s=60, color=CB_PURPLE, edgecolors='white',
                linewidth=0.5, alpha=0.8, zorder=5)
    ax2.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax2.set_xlabel('RF+Morgan AUROC (per target)')
    ax2.set_ylabel('D-MPNN AUROC (per target)')
    ax2.set_xlim(0.3, 1.0)
    ax2.set_ylim(0.3, 1.0)
    ax2.set_aspect('equal')
    _despine(ax2)
    ax2.set_title('B  D-MPNN vs RF (per-target)', fontweight='bold', loc='left')

    wins = sum(1 for d, r in zip(dmpnn_vals, rf_vals) if d > r)
    ax2.text(0.95, 0.05, f'D-MPNN wins: {wins}/{len(targets)}',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    _save(fig, 'figS12_gnn_baselines')


# ═══════════════════════════════════════════════════════════════════════════════
# NEW FIG S15: STAN ABLATION — attention helps, PLM does not
# ═══════════════════════════════════════════════════════════════════════════════
def figS15_stan_ablation():
    """Bar chart: C0 (full STAN), C1 (zeroPOI), C2 (randomPOI), C3 (RF+ESM).
    Key message: attention mechanism helps; PLM embeddings alone do not."""
    abl = _load_json('results/exp38_stan_ablation/summary.json')
    baseline = abl['baseline_rf_morgan']  # 0.666

    conditions = [
        ('C0: Full STAN\n(Morgan + ESM\n+ attention)', abl['conditions']['C0']['mean_auroc'], CB_PURPLE),
        ('C1: STAN\nzeroed POI', abl['conditions']['C1']['mean_auroc'], CB_BLUE),
        ('C2: STAN\nrandom POI', abl['conditions']['C2']['mean_auroc'], CB_TEAL),
        ('C3: RF+Morgan\n+ESM (no attn)', abl['conditions']['C3']['mean_auroc'], CB_ORANGE),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    x = np.arange(len(conditions))
    bars = ax.bar(x, [c[1] for c in conditions], color=[c[2] for c in conditions],
                  edgecolor='white', width=0.6, alpha=0.85)

    for i, (name, val, _) in enumerate(conditions):
        ax.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # Baseline reference
    ax.axhline(baseline, color=CB_GREY, ls=':', lw=1.2, alpha=0.7)
    ax.text(len(conditions) - 0.5, baseline + 0.005, f'RF+Morgan baseline ({baseline:.3f})',
            fontsize=8, color=CB_GREY, ha='right')

    # Chance level
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.3)

    # Annotations
    # C0 ≈ C1 ≈ C2 — attention helps regardless of POI content
    ax.annotate('', xy=(0, conditions[0][1] + 0.025),
                xytext=(2, conditions[2][1] + 0.025),
                arrowprops=dict(arrowstyle='<->', color=CB_DBLUE, lw=1.5))
    ax.text(1, conditions[0][1] + 0.04, 'n.s. — attention helps\nregardless of POI content',
            ha='center', fontsize=8, fontweight='bold', color=CB_DBLUE)

    # C3 below baseline — PLM alone doesn't help
    ax.annotate('PLM alone\n= no gain',
                xy=(3, conditions[3][1] - 0.01), xytext=(3.4, 0.60),
                fontsize=9, fontweight='bold', color=CB_RED,
                arrowprops=dict(arrowstyle='->', color=CB_RED, lw=1.2))

    # p-values from ablation
    pvals = abl['conditions']
    ax.text(0, conditions[0][1] - 0.025,
            f'p={pvals["C0"]["wilcoxon_p_vs_baseline"]:.3f}',
            ha='center', fontsize=7, color='#555', style='italic')
    ax.text(3, conditions[3][1] - 0.025,
            f'p={pvals["C3"]["wilcoxon_p_vs_baseline"]:.2f}',
            ha='center', fontsize=7, color='#555', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=9)
    ax.set_ylabel('Mean LOTO AUROC (64 targets)')
    ax.set_ylim(0.55, 0.80)
    _despine(ax)
    ax.set_title('STAN Ablation: Attention Helps, PLM Does Not', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'figS15_stan_ablation')


# ═══════════════════════════════════════════════════════════════════════════════
# NEW FIG S16: TEMPORAL PROSPECTIVE
# ═══════════════════════════════════════════════════════════════════════════════
def figS16_temporal():
    """Bar chart: random (0.757), LOTO (0.665), temporal (0.560),
    temporal+stack (0.689). Shows evaluation stringency matters."""
    tmp = _load_json('results/exp38_temporal/summary.json')

    conditions = [
        ('Random\nsplit', 0.757, CB_LBLUE),
        ('LOTO', 0.665, CB_BLUE),
        ('Temporal', tmp['C0_morgan_only']['mean_auroc'], CB_ORANGE),
        ('Temporal\n+ full stack', tmp['C2_fullstack_fewshot']['mean_auroc'], CB_GREEN),
    ]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    x = np.arange(len(conditions))
    bars = ax.bar(x, [c[1] for c in conditions], color=[c[2] for c in conditions],
                  edgecolor='white', width=0.55, alpha=0.85)

    for i, (name, val, _) in enumerate(conditions):
        ax.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # Error bars for temporal conditions (3-seed std)
    temporal_std_c0 = tmp['C0_morgan_only']['std_auroc']
    temporal_std_c2 = tmp['C2_fullstack_fewshot']['std_auroc']
    ax.errorbar(2, conditions[2][1], yerr=temporal_std_c0, fmt='none',
                ecolor='black', capsize=4, capthick=0.8, zorder=6)
    ax.errorbar(3, conditions[3][1], yerr=temporal_std_c2, fmt='none',
                ecolor='black', capsize=4, capthick=0.8, zorder=6)

    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.3)

    # Annotation: drop from random → temporal
    drop = 0.757 - tmp['C0_morgan_only']['mean_auroc']
    ax.annotate(f'$\\Delta$={drop:-.3f}',
                xy=(2, conditions[2][1]), xytext=(0.5, 0.55),
                fontsize=10, fontweight='bold', color=CB_RED,
                arrowprops=dict(arrowstyle='->', color=CB_RED, lw=1.5))

    # Annotation: stack recovery
    recovery = tmp['C2_fullstack_fewshot']['mean_auroc'] - tmp['C0_morgan_only']['mean_auroc']
    ax.annotate(f'+{recovery:.3f}\nstack recovery',
                xy=(3, conditions[3][1]), xytext=(3.3, 0.62),
                fontsize=9, fontweight='bold', color=CB_GREEN,
                arrowprops=dict(arrowstyle='->', color=CB_GREEN, lw=1.2))

    # Split info
    info = tmp['split_info']
    ax.text(0.02, 0.98,
            f'Train: TPDDB ({info["n_train"]:,})\n'
            f'Test: PROTAC-8K ({info["n_test"]:,})\n'
            f'Novel targets: {info["novel_targets"]}',
            transform=ax.transAxes, ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=10)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.45, 0.85)
    _despine(ax)
    ax.set_title('Evaluation Stringency: Random vs LOTO vs Temporal', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'figS16_temporal_prospective')


# ═══════════════════════════════════════════════════════════════════════════════
# NEW FIG S17: PSEUDO-LABELING
# ═══════════════════════════════════════════════════════════════════════════════
def figS17_pseudolabel():
    """Bar chart: LOTO vs random with/without pseudo-labels.
    Key message: pseudo-labels do NOT help under LOTO."""
    pl = _load_json('results/exp38_pseudolabel/summary.json')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                                    gridspec_kw={'width_ratios': [1, 1]})

    # Left: LOTO split
    loto_conds = [
        ('Baseline', pl['LOTO_results']['C0_baseline']['mean_auroc'], CB_GREY),
        ('+ Pseudo\n(0.8/0.2)', pl['LOTO_results']['C1_pseudo_0.8_0.2']['mean_auroc'], CB_BLUE),
        ('+ Pseudo\n(0.9/0.1)', pl['LOTO_results']['C2_pseudo_0.9_0.1']['mean_auroc'], CB_TEAL),
    ]

    x = np.arange(len(loto_conds))
    bars1 = ax1.bar(x, [c[1] for c in loto_conds], color=[c[2] for c in loto_conds],
                     edgecolor='white', width=0.5, alpha=0.85)
    for i, (name, val, _) in enumerate(loto_conds):
        ax1.text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    # p-values
    p1 = pl['LOTO_results']['C1_pseudo_0.8_0.2']['p_value']
    p2 = pl['LOTO_results']['C2_pseudo_0.9_0.1']['p_value']
    ax1.text(1, loto_conds[1][1] - 0.015, f'p={p1:.2f}', ha='center',
             fontsize=8, color='#555', style='italic')
    ax1.text(2, loto_conds[2][1] - 0.015, f'p={p2:.2f}', ha='center',
             fontsize=8, color='#555', style='italic')

    ax1.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[0] for c in loto_conds], fontsize=9)
    ax1.set_ylabel('Mean LOTO AUROC')
    ax1.set_ylim(0.55, 0.75)
    _despine(ax1)
    ax1.set_title('A  LOTO Split (cold-target)', fontweight='bold', loc='left')

    # "n.s." annotation
    ax1.text(1.5, max(c[1] for c in loto_conds) + 0.025,
             'n.s. — pseudo-labels\nare noise for novel targets',
             ha='center', fontsize=9, fontweight='bold', color=CB_RED)

    # Right: Random split
    rand_conds = [
        ('Baseline', pl['random_split_results']['C0_baseline']['mean_auroc'], CB_GREY),
        ('+ Pseudo\n(0.8/0.2)', pl['random_split_results']['C1_pseudo_0.8_0.2']['mean_auroc'], CB_BLUE),
        ('+ Pseudo\n(0.9/0.1)', pl['random_split_results']['C2_pseudo_0.9_0.1']['mean_auroc'], CB_TEAL),
    ]

    x2 = np.arange(len(rand_conds))
    bars2 = ax2.bar(x2, [c[1] for c in rand_conds], color=[c[2] for c in rand_conds],
                     edgecolor='white', width=0.5, alpha=0.85)
    for i, (name, val, _) in enumerate(rand_conds):
        ax2.text(i, val + 0.002, f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    ax2.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([c[0] for c in rand_conds], fontsize=9)
    ax2.set_ylabel('Mean Random-Split AUROC')
    ax2.set_ylim(0.85, 0.95)
    _despine(ax2)
    ax2.set_title('B  Random Split (target leakage)', fontweight='bold', loc='left')

    ax2.text(1.5, 0.86,
             'No gain under random split either\n(ceiling already saturated)',
             ha='center', fontsize=9, color='#555', style='italic')

    fig.tight_layout()
    _save(fig, 'figS17_pseudolabel')


# ═══════════════════════════════════════════════════════════════════════════════
# NEW FIG S18: EXP39 STAN COMBINED PROGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
def figS18_stan_combined():
    """Bar chart: exp39 STAN combined — shows progression table.
    Key message: STAN + features = redundant; RF stack dominates."""
    sc = _load_json('results/exp39_stan_combined/summary.json')
    conds = sc['conditions']

    labels = []
    aurocs = []
    colors_list = []
    pvals = []

    order = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    color_map = {
        'C0': CB_GREY, 'C1': CB_PURPLE, 'C2': CB_BLUE,
        'C3': CB_TEAL, 'C4': CB_ORANGE, 'C5': CB_GREEN, 'C6': CB_RED,
    }

    for key in order:
        c = conds[key]
        labels.append(c['name'])
        aurocs.append(c['mean_auroc'])
        colors_list.append(color_map[key])
        pvals.append(c['wilcoxon_p_vs_C0'])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    bars = ax.bar(x, aurocs, color=colors_list, edgecolor='white',
                  width=0.6, alpha=0.85)

    for i, (val, p) in enumerate(zip(aurocs, pvals)):
        ax.text(i, val + 0.008, f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        if p < 0.05:
            ax.text(i, val + 0.030, '*', ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color=CB_RED)

    # Baseline reference
    baseline = aurocs[0]
    ax.axhline(baseline, color=CB_GREY, ls=':', lw=1, alpha=0.5)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.3)

    # Highlight best RF-based (C2) vs best STAN-based (C5)
    best_rf = aurocs[2]  # C2
    best_stan = aurocs[5]  # C5
    ax.annotate(f'RF stack wins\n({best_rf:.3f} vs {best_stan:.3f})',
                xy=(2, best_rf + 0.005), xytext=(4, 0.75),
                fontsize=9, fontweight='bold', color=CB_DBLUE,
                arrowprops=dict(arrowstyle='->', color=CB_DBLUE, lw=1.2))

    # Wrap long labels
    short_labels = [
        'RF+Morgan\n(baseline)', 'STAN\nattention', 'RF+Morgan\n+wh+ADMET',
        'STAN\n+warhead', 'STAN+wh\n+ADMET', 'STAN+wh\n+ADMET+k5',
        'RF+Morgan\n*E3+wh+ADMET'
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Mean LOTO AUROC (64 targets)')
    ax.set_ylim(0.55, 0.80)
    _despine(ax)
    ax.set_title('STAN Combined: Feature Stacking Progression', fontweight='bold')

    # Legend for significance
    ax.text(0.98, 0.02, '* p < 0.05 vs baseline',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color=CB_RED, style='italic')

    fig.tight_layout()
    _save(fig, 'figS18_stan_combined')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('  NeurIPS Figures v13')
    print('=' * 60)

    copy_unchanged_figures()

    figs = [
        ('Fig 4 (breakthroughs + full stack)', fig4_breakthroughs),
        ('Fig S12 (GNN baselines + HPO)', figS12_gnn_baselines),
        ('Fig S15 (STAN ablation)', figS15_stan_ablation),
        ('Fig S16 (temporal prospective)', figS16_temporal),
        ('Fig S17 (pseudo-labeling)', figS17_pseudolabel),
        ('Fig S18 (STAN combined)', figS18_stan_combined),
    ]

    results = []
    for name, func in figs:
        try:
            func()
            results.append((name, 'OK'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  {name}: ERROR - {e}')
            results.append((name, f'ERROR: {e}'))

    # Print manifest
    print('\n' + '=' * 60)
    print('  MANIFEST: neurips_figures_v13/')
    print('=' * 60)
    all_files = sorted(RDIR.glob('*.png'))
    new_names = {
        'fig4_breakthroughs', 'figS12_gnn_baselines',
        'figS15_stan_ablation', 'figS16_temporal_prospective',
        'figS17_pseudolabel', 'figS18_stan_combined',
    }
    for f in all_files:
        stem = f.stem
        tag = ' <-- NEW/UPDATED' if stem in new_names else ''
        print(f'  {f.name}{tag}')

    print(f'\n  Total PNG files: {len(all_files)}')
    print(f'  Total PDF files: {len(list(RDIR.glob("*.pdf")))}')

    print('\n' + '-' * 60)
    print(f'  {"Figure":<45} {"Status":<30}')
    print('-' * 60)
    for name, status in results:
        print(f'  {name:<45} {status:<30}')
    print('=' * 60)


if __name__ == '__main__':
    main()
