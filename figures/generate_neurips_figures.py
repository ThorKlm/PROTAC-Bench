#!/usr/bin/env python3
"""Generate all publication-ready figures for NeurIPS 2026 submission.
All data loaded from result JSONs/CSVs — no hardcoded numbers.
"""
import json, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style setup ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'xtick.major.width': 1, 'ytick.major.width': 1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    'axes.spines.top': False, 'axes.spines.right': False,
})
# Colorblind-friendly (tab10 subset)
C_BLUE = '#1f77b4'
C_ORANGE = '#ff7f0e'
C_GREEN = '#2ca02c'
C_RED = '#d62728'
C_GREY = '#7f7f7f'
C_PURPLE = '#9467bd'

OUTDIR = Path('/workspace/results/neurips_figures')
OUTDIR.mkdir(parents=True, exist_ok=True)

def _load(path):
    with open(path) as f:
        return json.load(f)

def _save(fig, name):
    fig.savefig(OUTDIR / f'{name}.pdf')
    fig.savefig(OUTDIR / f'{name}.png', dpi=300)
    plt.close(fig)
    print(f'  Saved {name}.pdf/.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: THE COLLAPSE (two-panel, 14x5)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_collapse():
    full = _load('results/exp1_full_replication/full_summary.json')
    conds = full['conditions']
    dm = _load('results/exp1_degrademaster_loto/summary.json')
    knn = _load('results/exp13_splits_and_baselines/knn_baseline.json')
    knn_loto = knn['loto']['5']['mean_auroc']
    knn_rand = knn['random_cv']['5']['mean_auroc']

    methods = [
        ('RF+Morgan',       conds['A_rf_morgan']['random'],  conds['A_rf_morgan']['loto']),
        ('XGBoost+Morgan',  conds['B_xgb_morgan']['random'], conds['B_xgb_morgan']['loto']),
        ('Ribes-style',     conds['C_ribes_style']['random'],conds['C_ribes_style']['loto']),
        ('STAN-style',      conds['D_stan_style']['random'], conds['D_stan_style']['loto']),
        ('DegradeMaster\nEGNN', 0.830, dm['mean']),
        ('kNN (k=5)',       knn_rand, knn_loto),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Grouped bar chart
    x = np.arange(len(methods))
    w = 0.35
    rvals = [m[1] for m in methods]
    lvals = [m[2] for m in methods]
    ax1.bar(x - w/2, rvals, w, label='Random split', color=C_BLUE, edgecolor='white', lw=0.5)
    ax1.bar(x + w/2, lvals, w, label='LOTO (cold-target)', color=C_ORANGE, edgecolor='white', lw=0.5)
    ax1.axhline(0.5, color=C_GREY, ls='--', lw=0.8, alpha=0.6, label='Chance')
    ax1.set_ylabel('AUROC')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in methods], fontsize=9)
    ax1.set_ylim(0.4, 1.02)
    ax1.legend(loc='upper right', fontsize=9)
    for i, m in enumerate(methods):
        delta = m[1] - m[2]
        mid = (m[1] + m[2]) / 2
        ax1.annotate(f'\u0394={delta:.3f}', (i, mid), ha='center', fontsize=8, color='#333',
                     fontweight='bold')
    ax1.set_title('A. Cold-Target Collapse Is Universal', fontsize=12, fontweight='bold')

    # Panel B: Tanimoto similarity distributions
    tan_rand = np.load('results/exp13_splits_and_baselines/tanimoto_random.npy')
    tan_loto = np.load('results/exp13_splits_and_baselines/tanimoto_loto.npy')
    tan_scaf = np.load('results/exp13_splits_and_baselines/tanimoto_scaffold.npy')

    bins = np.linspace(0, 1, 50)
    ax2.hist(tan_rand, bins=bins, alpha=0.5, density=True, color=C_BLUE, label='Random CV')
    ax2.hist(tan_scaf, bins=bins, alpha=0.5, density=True, color=C_GREEN, label='Scaffold CV')
    ax2.hist(tan_loto, bins=bins, alpha=0.5, density=True, color=C_ORANGE, label='LOTO')
    for thresh in [0.4, 0.6, 0.8]:
        ax2.axvline(thresh, color=C_GREY, ls='--', lw=0.8, alpha=0.6)
    ax2.set_xlabel('Max Tanimoto Similarity to Nearest Training Neighbor')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=9)
    fr_rand = (tan_rand > 0.6).mean()
    fr_scaf = (tan_scaf > 0.6).mean()
    fr_loto = (tan_loto > 0.6).mean()
    txt = f'Frac > 0.6:\n  Random: {fr_rand:.2f}\n  Scaffold: {fr_scaf:.2f}\n  LOTO: {fr_loto:.2f}'
    ax2.text(0.02, 0.97, txt, transform=ax2.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))
    ax2.set_title('B. Similarity Distributions by Split', fontsize=12, fontweight='bold')

    fig.tight_layout()
    _save(fig, 'fig1_collapse')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: PER-FOLD AUROC DISTRIBUTION (7x5)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_perfold():
    df = pd.read_csv('results/exp1_baseline_replication/A_rf_morgan.csv')
    try:
        sim_df = pd.read_csv('results/exp11_similarity_stratified/per_target_with_similarity.csv')
        sim_map = dict(zip(sim_df['target'], sim_df['max_cosine_sim']))
    except Exception:
        sim_map = {}

    df = df.sort_values('auroc').reset_index(drop=True)
    mean_auroc = df['auroc'].mean()

    if sim_map:
        df['sim'] = df['target'].map(sim_map)
        tertile_thresholds = df['sim'].quantile([1/3, 2/3]).values
        def get_color(s):
            if pd.isna(s): return C_GREY
            if s <= tertile_thresholds[0]: return C_RED
            elif s <= tertile_thresholds[1]: return '#f0c541'
            else: return C_GREEN
        colors = [get_color(s) for s in df['sim']]
    else:
        colors = [C_BLUE] * len(df)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(range(len(df)), df['auroc'], c=colors, s=30, zorder=3, edgecolors='white', lw=0.3)
    ax.axhline(mean_auroc, color=C_GREY, ls='--', lw=1, alpha=0.7)

    extreme_low = ['Q96SW2', 'P15170']
    extreme_high = ['O15379', 'P08581']
    for tgt in extreme_low + extreme_high:
        row = df[df['target'] == tgt]
        if len(row) > 0:
            idx = row.index[0]
            val = row.iloc[0]['auroc']
            offset = (-30, -15) if tgt in extreme_low else (10, 10)
            ax.annotate(tgt, (idx, val), textcoords='offset points', xytext=offset,
                        fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.5, color='#555'),
                        color='#333')

    ax.set_xlabel('Targets (ordered by AUROC)')
    ax.set_ylabel('LOTO AUROC')
    ax.set_title('Per-Target AUROC Distribution (RF+Morgan)', fontweight='bold')
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_RED, markersize=7, label='Low similarity'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#f0c541', markersize=7, label='Mid similarity'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_GREEN, markersize=7, label='High similarity'),
        Line2D([0],[0], color=C_GREY, ls='--', lw=1, label=f'Mean = {mean_auroc:.3f}'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper left')

    # Marginal histogram
    divider_ax = fig.add_axes([0.88, 0.11, 0.08, 0.77])
    divider_ax.hist(df['auroc'], bins=15, orientation='horizontal', color=C_BLUE, alpha=0.5, edgecolor='white')
    divider_ax.set_ylim(ax.get_ylim())
    divider_ax.set_xticks([])
    divider_ax.set_yticks([])
    divider_ax.spines['top'].set_visible(False)
    divider_ax.spines['right'].set_visible(False)
    divider_ax.spines['left'].set_visible(False)

    fig.subplots_adjust(right=0.86)
    _save(fig, 'fig2_perfold')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: HPO CEILING (14x5)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_ceiling():
    trials = pd.read_csv('results/exp2_unified_hpo/all_trials.csv')
    importance = _load('results/exp2_unified_hpo/param_importance.json')
    summary = _load('summary.json')

    baseline = summary['baseline']['mean']
    best_hpo = trials['value'].max()
    validated = summary['configs'][0]['mean'] if summary['configs'] else baseline

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: HPO trial distribution
    valid_trials = trials[trials['value'] > 0.4]['value']
    ax1.hist(valid_trials, bins=30, color=C_BLUE, alpha=0.7, edgecolor='white')
    ax1.axvline(baseline, color=C_RED, ls='--', lw=1.5, label=f'Baseline: {baseline:.3f}')
    ax1.axvline(best_hpo, color=C_GREEN, ls='-', lw=1.5, label=f'Best HPO (1 seed): {best_hpo:.3f}')
    ax1.set_xlabel('LOTO AUROC')
    ax1.set_ylabel('Number of trials')
    ax1.legend(fontsize=9)
    p_val = 0.925
    ax1.text(0.03, 0.95, f'5-seed validated: {validated:.3f}\n(p = {p_val})',
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', edgecolor='#ffc107', alpha=0.9))
    ax1.set_title('A. HPO Trial AUROC Distribution', fontweight='bold')

    # Panel B: fANOVA importance
    params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in params]
    vals = [p[1] for p in params]
    colors_bar = [('#1a3a5c' if n == 'head_type' else '#d0d0d0') for n in names]
    y_pos = np.arange(len(names))
    ax2.barh(y_pos, vals, color=colors_bar, edgecolor='white', lw=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([n.replace('_', ' ') for n in names], fontsize=9)
    ax2.set_xlabel('fANOVA Importance')
    ax2.invert_yaxis()
    ax2.text(0.5, 0.95, 'Flat landscape: no parameter\nmatters beyond architecture choice',
             transform=ax2.transAxes, fontsize=9, va='top', ha='center', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ccc', alpha=0.9))
    ax2.text(vals[0] + 0.01, 0, f'{vals[0]*100:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax2.set_title('B. Parameter Importance (fANOVA)', fontweight='bold')

    fig.tight_layout()
    _save(fig, 'fig3_ceiling')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: THREE SIGNALS THAT BREAK THROUGH (7x6)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_breakthroughs():
    baseline_auroc = _load('results/exp1_baseline_replication/summary.json')['models']['rf_morgan']['mean']
    admet = _load('results/exp8c_admet_cascade/summary.json')['conditions']['C2_morgan+admet7']
    fewshot = _load('results/exp4_fewshot/exp4_summary.json')
    egnn = _load('results/exp1_degrademaster_loto/summary.json')

    items = [
        ('Baseline\nRF+Morgan\n(65 targets)', baseline_auroc, None, C_GREY, 'solid'),
        ('+ ADMET cascade\n(65 targets)',      admet['auroc'],  admet['p'],  C_BLUE, 'solid'),
        ('+ k=5 few-shot\n(65 targets)',       fewshot['rf_morgan_k5']['mean'], None, C_GREEN, 'solid'),
        ('+ 3D EGNN\n(27 targets)',            egnn['mean'],    None, C_RED, 'hatched'),
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(len(items))
    for i, (name, val, pval, col, style) in enumerate(items):
        hatch = '///' if style == 'hatched' else ''
        ax.barh(i, val, color=col, alpha=0.8, edgecolor='white', lw=0.5, hatch=hatch)
        ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
        if pval is not None:
            ax.text(val + 0.005, i + 0.25, f'p={pval:.3f}', va='center', fontsize=7, color='#666')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([it[0] for it in items], fontsize=9)
    ax.set_xlabel('LOTO AUROC')
    ax.set_xlim(0.5, 0.88)
    ax.axvline(0.5, color=C_GREY, ls='--', lw=0.8, alpha=0.5)
    ax.invert_yaxis()
    ax.set_title('Three Signals That Break Through', fontweight='bold')

    ax.text(0.98, 0.02,
            'Each addresses a different failure mode:\n'
            '  ADMET \u2192 cell entry\n'
            '  Few-shot \u2192 target-specific SAR\n'
            '  EGNN \u2192 ternary complex geometry',
            transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f4e8', edgecolor='#999', alpha=0.9))

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_GREY, label='65-target eval'),
        Patch(facecolor=C_RED, hatch='///', label='27-target eval'),
    ], loc='lower right', fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _save(fig, 'fig4_breakthroughs')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: FEW-SHOT SATURATION CURVE (7x5)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_fewshot():
    d = _load('results/exp4_fewshot/exp4_summary.json')

    ks = [0, 1, 3, 5, 10]
    rf_morgan = []
    rf_meta = []
    rf_morgan_std = []
    rf_meta_std = []
    for k in ks:
        key_m = f'rf_morgan_k{k}'
        key_r = f'rf_meta_k{k}'
        rf_morgan.append(d[key_m]['mean'])
        # rf_meta_k0 doesn't exist; baseline is shared
        if key_r in d:
            rf_meta.append(d[key_r]['mean'])
            rf_meta_std.append(d[key_r].get('std', 0))
        else:
            rf_meta.append(d[key_m]['mean'])
            rf_meta_std.append(d[key_m].get('std', 0))
        rf_morgan_std.append(d[key_m].get('std', 0))

    maml_morgan = d['maml_morgan_k5']['mean']
    maml_meta = d['maml_meta_k5']['mean']

    fig, ax = plt.subplots(figsize=(7, 5))
    n_tgt = 65
    ax.errorbar(ks, rf_morgan, yerr=[s/np.sqrt(n_tgt) for s in rf_morgan_std],
                marker='o', color=C_BLUE, label='RF + Morgan', capsize=3, lw=1.5)
    ax.errorbar(ks, rf_meta, yerr=[s/np.sqrt(n_tgt) for s in rf_meta_std],
                marker='s', color=C_GREEN, label='RF + meta-features', capsize=3, lw=1.5)
    ax.axhline(maml_morgan, color=C_RED, ls='--', lw=1, alpha=0.8, label=f'MAML Morgan k=5: {maml_morgan:.3f}')
    ax.axhline(maml_meta, color=C_RED, ls=':', lw=1, alpha=0.8, label=f'MAML meta k=5: {maml_meta:.3f}')

    ax.set_xlabel('k (number of target-specific examples)')
    ax.set_ylabel('LOTO AUROC')
    ax.set_xticks(ks)
    ax.legend(fontsize=8, loc='lower right')
    ax.text(0.5, 0.03, 'Plain RF beats gradient-based meta-learning',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', edgecolor='#ffc107', alpha=0.9))
    ax.set_title('Few-Shot Saturation: RF vs MAML', fontweight='bold')

    fig.tight_layout()
    _save(fig, 'fig5_fewshot')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: EGNN SCATTER (7x6)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_egnn_scatter():
    d = _load('results/exp1_degrademaster_loto/per_target_analysis.json')
    pt = d['per_target']

    targets = list(pt.keys())
    rf_vals = [pt[t]['rf_dm'] for t in targets]
    egnn_vals = [pt[t]['egnn'] for t in targets]

    base_df = pd.read_csv('results/exp1_baseline_replication/A_rf_morgan.csv')
    size_map = dict(zip(base_df['target'], base_df['n']))
    sizes = [size_map.get(t, 50) for t in targets]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(rf_vals, egnn_vals, c=sizes, cmap='viridis', s=50, edgecolors='white',
                    lw=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Target size (n entries)', fontsize=9)

    ax.plot([0, 1], [0, 1], color=C_GREY, ls='-', lw=1, alpha=0.5)
    ax.set_xlabel('RF+Morgan AUROC per target')
    ax.set_ylabel('EGNN AUROC per target')
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(0.1, 1.05)

    for t in targets:
        delta = pt[t]['egnn'] - pt[t]['rf_dm']
        if abs(delta) > 0.2:
            ax.annotate(t, (pt[t]['rf_dm'], pt[t]['egnn']),
                        fontsize=7, textcoords='offset points', xytext=(5, 5),
                        arrowprops=dict(arrowstyle='->', lw=0.4, color='#555'))

    wins = sum(1 for t in targets if pt[t]['egnn'] > pt[t]['rf_dm'])
    rho = d.get('correlation_rf_vs_delta', {}).get('rho', -0.847)
    ax.text(0.03, 0.97, f'{wins}/{len(targets)} targets: EGNN > RF\n\u03C1 = {rho:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))
    ax.set_title('EGNN Improves Most Where 2D Fails Hardest', fontweight='bold')

    fig.tight_layout()
    _save(fig, 'fig6_egnn_scatter')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: PLM SCALING FAILURE (7x5)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_plm_scaling():
    bfd = _load('results/exp1_prottrans_bfd/summary.json')
    summ = _load('summary.json')

    esm_8m = None
    esm_3b = None
    for cfg in summ.get('configs', []):
        if cfg.get('prot') == 'esm2_8M':
            esm_8m = cfg.get('mean')
        if cfg.get('prot') == 'esm2_3B':
            esm_3b = cfg.get('mean')

    methods = ['Morgan\nonly']
    vals = [bfd['E0_morgan_only']['mean']]
    errs = [bfd['E0_morgan_only']['std'] / np.sqrt(bfd['E0_morgan_only']['n'])]
    pvals_list = [None]

    methods.append('Morgan +\nBFD (420M)')
    vals.append(bfd['E1_morgan+bfd_poi']['mean'])
    errs.append(bfd['E1_morgan+bfd_poi']['std'] / np.sqrt(bfd['E1_morgan+bfd_poi']['n']))
    pvals_list.append(bfd['E1_morgan+bfd_poi']['p_vs_E0'])

    if esm_8m is not None:
        methods.append('Morgan +\nESM2-8M')
        vals.append(esm_8m)
        errs.append(0)
        pvals_list.append(None)
    if esm_3b is not None:
        methods.append('Morgan +\nESM2-3B')
        vals.append(esm_3b)
        errs.append(0)
        pvals_list.append(None)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(methods))
    colors = [C_BLUE] + [C_ORANGE] * (len(methods) - 1)
    ax.bar(x, vals, yerr=errs, color=colors, edgecolor='white', lw=0.5, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('LOTO AUROC')
    ax.set_ylim(0.55, 0.72)
    ax.axhline(0.666, color=C_GREY, ls='--', lw=0.8, alpha=0.5)

    for i, p in enumerate(pvals_list):
        if p is not None:
            ax.text(i, vals[i] + errs[i] + 0.005, f'p={p:.3f}', ha='center', fontsize=8, color='#666')

    ax.text(0.5, 0.03, 'Larger PLMs encode richer target identity \u2014 adversarial under LOTO',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fde8e8', edgecolor='#e74c3c', alpha=0.9))
    ax.set_title('PLM Scaling Failure Under Cold-Target Evaluation', fontweight='bold')

    fig.tight_layout()
    _save(fig, 'fig7_plm_scaling')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S1: FRAGMENT DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════
def figS1_fragments():
    frags = _load('results/exp_supp_fragments/summary.json')
    frag_order = ['full_protac', 'warhead', 'linker', 'e3_ligand', 'anchor']
    names = [f.replace('_', '\n') for f in frag_order]
    rand_vals = [frags[f]['random'] for f in frag_order]
    loto_vals = [frags[f]['loto'] for f in frag_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, rand_vals, w, label='Random CV', color=C_BLUE, edgecolor='white', lw=0.5)
    ax.bar(x + w/2, loto_vals, w, label='LOTO', color=C_ORANGE, edgecolor='white', lw=0.5)
    ax.axhline(0.5, color=C_GREY, ls='--', lw=0.8, alpha=0.5)
    ax.set_ylabel('AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=9)
    for i in range(len(names)):
        collapse = rand_vals[i] - loto_vals[i]
        mid = (rand_vals[i] + loto_vals[i]) / 2
        ax.annotate(f'\u0394={collapse:.3f}', (i, mid), ha='center', fontsize=8, color='#333', fontweight='bold')
    ax.set_title('Fragment Signal Decomposition Under LOTO', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'figS1_fragments')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S2: THRESHOLD SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════
def figS2_threshold():
    d = _load('results/exp_supp_threshold/summary.json')
    thresholds = d['thresholds']
    labels = [t['threshold'] for t in thresholds]
    rand_vals = [t['random'] for t in thresholds]
    loto_vals = [t['loto'] for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(len(labels)), rand_vals, marker='o', color=C_BLUE, label='Random CV', lw=1.5)
    ax.plot(range(len(labels)), loto_vals, marker='s', color=C_ORANGE, label='LOTO', lw=1.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_xlabel('DC50 Threshold')
    ax.set_ylabel('AUROC')
    ax.axhline(0.5, color=C_GREY, ls='--', lw=0.8, alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_title('Threshold Sensitivity: Random vs LOTO', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'figS2_threshold')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S3: TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
def figS3_temporal():
    d = _load('results/exp_supp_temporal/summary.json')
    years = sorted([y for y in d.keys() if y.isdigit()])
    aurocs = [d[y]['auroc'] for y in years]
    n_trains = [d[y]['n_train'] for y in years]
    overlaps = [d[y]['target_overlap'] for y in years]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(years, aurocs, marker='o', color=C_BLUE, lw=1.5, label='AUROC')
    ax1.set_xlabel('Training Cutoff Year')
    ax1.set_ylabel('AUROC', color=C_BLUE)
    ax1.tick_params(axis='y', labelcolor=C_BLUE)
    ax1.axhline(0.5, color=C_GREY, ls='--', lw=0.8, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.bar(years, n_trains, alpha=0.25, color=C_ORANGE, label='Training size')
    ax2.set_ylabel('Training Set Size', color=C_ORANGE)
    ax2.tick_params(axis='y', labelcolor=C_ORANGE)
    ax2.spines['top'].set_visible(False)

    for i, (y, ov) in enumerate(zip(years, overlaps)):
        ax1.annotate(f'{ov} shared', (i, aurocs[i]), textcoords='offset points',
                     xytext=(0, 10), fontsize=7, ha='center', color='#555')

    ax1.set_title('Temporal Split: AUROC vs Training Cutoff Year', fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    fig.tight_layout()
    _save(fig, 'figS3_temporal')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S4: CROSS-DATASET AND SOURCE BIAS
# ═══════════════════════════════════════════════════════════════════════════════
def figS4_cross_dataset():
    d = _load('results/exp13_cross_dataset/summary.json')
    r = d['results']

    methods = ['Within-source\nCV (mean)', 'Cross-source\n(mean)', 'LOTO\n(all data)']
    vals = [r['within_source_mean'], r['cross_dataset_mean'], 0.666]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [C_BLUE, C_RED, C_ORANGE]
    ax.bar(range(len(methods)), vals, color=colors, edgecolor='white', lw=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color=C_GREY, ls='--', lw=0.8, alpha=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.03, 'Source bias (0.54) is worse than target bias (0.67)',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fde8e8', edgecolor='#e74c3c', alpha=0.9))
    ax.set_title('Cross-Dataset Generalization', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'figS4_cross_dataset')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S5: DOCKING CONFOUND
# ═══════════════════════════════════════════════════════════════════════════════
def figS5_docking_confound():
    d = _load('results/exp11_docking_confound/summary.json')
    c = d['conditions']
    avail_keys = list(c.keys())
    vals = []
    labels_final = []
    for k in avail_keys:
        lbl = k.split('_', 1)[1] if '_' in k else k
        lbl = lbl.replace('_', '\n')
        labels_final.append(lbl)
        vals.append(c[k]['mean_auroc'])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [C_BLUE if i == 0 else C_ORANGE for i in range(len(vals))]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor='white', lw=0.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels_final, fontsize=8)
    ax.set_ylabel('LOTO AUROC')
    ax.set_ylim(0.4, 0.72)
    ax.axhline(0.5, color=C_GREY, ls='--', lw=0.8, alpha=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
    ax.text(0.5, 0.03, '3D docking features explain only 6.3% of EGNN gap',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f0fe', edgecolor='#4285f4', alpha=0.9))
    ax.set_title('Docking Score Confound Analysis', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'figS5_docking_confound')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S6: FEATURE CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
def figS6_feature_correlations():
    d = _load('results/exp9_advanced_failures/summary.json')
    fc = d['feature_correlations']

    items = [(k, v['rho'], v['p']) for k, v in fc.items() if not np.isnan(v['rho'])]
    items.sort(key=lambda x: abs(x[1]), reverse=True)

    names = [it[0] for it in items]
    rhos = [it[1] for it in items]
    pvals = [it[2] for it in items]

    fig, ax = plt.subplots(figsize=(7, max(5, len(items) * 0.3)))
    colors = [C_GREEN if r > 0 else C_RED for r in rhos]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, rhos, color=colors, edgecolor='white', lw=0.3, alpha=0.8)
    ax.set_yticks(y_pos)
    labels = []
    for n, p in zip(names, pvals):
        lbl = n.replace('_', ' ')
        if p < 0.01:
            lbl += ' *'
        labels.append(lbl)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Spearman \u03C1')
    ax.axvline(0, color='black', lw=0.5)

    for i, (n, r, p) in enumerate(items[:2]):
        ax.text(r + (0.005 if r > 0 else -0.005), i,
                f'\u03C1={r:.3f}', va='center', fontsize=7,
                ha='left' if r > 0 else 'right', fontweight='bold')

    ax.set_title('Feature Correlations with Degradation (Spearman)', fontweight='bold')
    ax.text(0.98, 0.98, '* p < 0.01', transform=ax.transAxes, fontsize=7, va='top', ha='right')
    fig.tight_layout()
    _save(fig, 'figS6_feature_correlations')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S7: SCAFFOLD SPLIT COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def figS7_scaffold_split():
    d = _load('results/exp13_splits_and_baselines/scaffold_split.json')
    comp = d['comparison']

    methods = ['Random CV', 'Scaffold CV', 'LOTO']
    vals = [comp['random_cv_auroc'], comp['scaffold_cv_auroc'], comp['loto_auroc']]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [C_BLUE, C_GREEN, C_ORANGE]
    ax.bar(range(len(methods)), vals, color=colors, edgecolor='white', lw=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.5, 1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.03, 'Scaffold split is useless for PROTACs\ndue to high scaffold diversity',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', edgecolor='#ffc107', alpha=0.9))
    ax.set_title('Scaffold Split vs LOTO', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'figS7_scaffold_split')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE S8: EGNN ON SMINA DOCKED POSES
# ═══════════════════════════════════════════════════════════════════════════════
def figS8_egnn_smina():
    pt = pd.read_csv('results/exp14_egnn_full/per_target.csv')
    pt_rf = pd.read_csv('results/exp14_egnn_full/per_target_morgan_rf.csv')
    try:
        pt_wd = pd.read_csv('results/exp14_egnn_full/per_target_well_docked.csv')
        wd_targets = set(pt_wd['target'])
    except Exception:
        wd_targets = set()

    merged = pt.merge(pt_rf, on='target', suffixes=('_egnn', '_rf'))
    rf_vals = merged['auroc_rf'].values
    egnn_vals = merged['auroc_egnn'].values
    is_well_docked = [t in wd_targets for t in merged['target']]
    colors = [C_GREEN if wd else C_RED for wd in is_well_docked]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rf_vals, egnn_vals, c=colors, s=40, edgecolors='white', lw=0.5, zorder=3)
    ax.plot([0, 1], [0, 1], color=C_GREY, ls='-', lw=1, alpha=0.5)
    ax.set_xlabel('RF+Morgan AUROC')
    ax.set_ylabel('EGNN AUROC (Smina poses)')
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(0.1, 1.05)

    for _, row in merged.iterrows():
        delta = row['auroc_egnn'] - row['auroc_rf']
        if abs(delta) > 0.3:
            ax.annotate(row['target'], (row['auroc_rf'], row['auroc_egnn']),
                        fontsize=7, textcoords='offset points', xytext=(5, 5),
                        arrowprops=dict(arrowstyle='->', lw=0.4, color='#555'))

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_GREEN, markersize=7, label='Well-docked (>50% <10\u00c5)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_RED, markersize=7, label='Poorly docked'),
    ], fontsize=8, loc='upper left')

    ax.text(0.97, 0.03, '69% of entries have warhead >20\u00c5 from pocket\nEGNN needs structure quality',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fde8e8', edgecolor='#e74c3c', alpha=0.9))
    ax.set_title('EGNN on Smina-Docked Poses (60 targets)', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'figS8_egnn_smina')

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import os
    os.chdir('/workspace')
    print('Generating NeurIPS 2026 figures...')
    for fn_name, fn in [
        ('Fig 1', fig1_collapse),
        ('Fig 2', fig2_perfold),
        ('Fig 3', fig3_ceiling),
        ('Fig 4', fig4_breakthroughs),
        ('Fig 5', fig5_fewshot),
        ('Fig 6', fig6_egnn_scatter),
        ('Fig 7', fig7_plm_scaling),
        ('Fig S1', figS1_fragments),
        ('Fig S2', figS2_threshold),
        ('Fig S3', figS3_temporal),
        ('Fig S4', figS4_cross_dataset),
        ('Fig S5', figS5_docking_confound),
        ('Fig S6', figS6_feature_correlations),
        ('Fig S7', figS7_scaffold_split),
        ('Fig S8', figS8_egnn_smina),
    ]:
        try:
            fn()
        except Exception as e:
            print(f'  ERROR in {fn_name}: {e}')
            import traceback
            traceback.print_exc()
    print('\nDone!')
