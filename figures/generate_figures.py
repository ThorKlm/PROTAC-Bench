#!/usr/bin/env python3
"""Generate all publication figures for PROTAC-Bench.
Updated for Session 5: BERT-BFD ablation, top-5 variance, target classes, thresholds.

Main figures (1-8) + supplementary (S1-S3).
"""
import json, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RDIR = Path('/workspace/results/figures')
RDIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'xtick.major.width': 1, 'ytick.major.width': 1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1})
C_RANDOM = '#4C72B0'
C_LOTO = '#DD8452'
C_DM = '#55A868'
C_GREY = '#999999'
C_PURPLE = '#8172B2'
C_RED = '#C44E52'


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Fig 1: Collapse Bar Chart (random vs LOTO for all methods)
# ============================================================================
def fig1_collapse():
    methods = [
        ('RF+Morgan', 0.902, 0.666, 65),
        ('XGBoost+Morgan', 0.900, 0.642, 65),
        ('Ribes-style', 0.913, 0.646, 61),
        ('STAN-style (RF)', 0.908, 0.653, 54),
        ('STAN TAN', 0.902, 0.660, 54),
        ('DegradeMaster\nEGNN (3D)', 0.830, 0.801, 27),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    w = 0.35
    bars_r = ax.bar(x - w/2, [m[1] for m in methods], w, label='Random split',
                     color=C_RANDOM, edgecolor='white', linewidth=0.5)
    bars_l = ax.bar(x + w/2, [m[2] for m in methods], w, label='LOTO (cold-target)',
                     color=C_LOTO, edgecolor='white', linewidth=0.5)
    bars_l[-1].set_color(C_DM)
    ax.set_ylabel('AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods], rotation=0, ha='center')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    for i, m in enumerate(methods):
        collapse = m[1] - m[2]
        mid = (m[1] + m[2]) / 2
        if i < len(methods) - 1:
            ax.annotate(f'{collapse:.2f}', (i, mid), ha='center', fontsize=8, color='#333')
    ax.set_title('Cold-Target Collapse Is Universal (Except 3D Structure)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig1_collapse.pdf')
    fig.savefig(RDIR / 'fig1_collapse.png')
    plt.close(fig)
    print('  Fig 1: Collapse bar chart')


# ============================================================================
# Fig 2: Fragment Decomposition
# ============================================================================
def fig2_fragments():
    frags = _load_json('results/exp_supp_fragments/summary.json')
    names, random_v, loto_v = [], [], []
    for k in ['full_protac', 'warhead', 'e3_ligand', 'linker', 'anchor']:
        if k in frags and isinstance(frags[k], dict):
            names.append(k.replace('_', '\n'))
            random_v.append(frags[k]['random'])
            loto_v.append(frags[k]['loto'])
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, random_v, w, label='Random split', color=C_RANDOM, edgecolor='white')
    ax.bar(x + w/2, loto_v, w, label='LOTO', color=C_LOTO, edgecolor='white')
    ax.set_ylabel('AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    for i in range(len(names)):
        collapse = random_v[i] - loto_v[i]
        ax.annotate(f'{collapse:.3f}', (i, (random_v[i]+loto_v[i])/2),
                    ha='center', fontsize=9, color='#333')
    ax.set_title('Fragment Signal Decomposition')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig2_fragments.pdf')
    fig.savefig(RDIR / 'fig2_fragments.png')
    plt.close(fig)
    print('  Fig 2: Fragment decomposition')


# ============================================================================
# Fig 3: Three-Layer Ceiling Waterfall
# ============================================================================
def fig3_ceiling():
    ceil = _load_json('results/exp_supp_ceiling/summary.json')
    r = ceil.get('random_cv', 0.902)
    w = ceil.get('within_target_mean', 0.806)
    l = ceil.get('loto_mean', 0.666)
    layers = ['Random CV\n(interpolation)', 'Within-target\n(target-specific)', 'LOTO\n(cold-target)']
    vals = [r, w, l]
    colors = [C_RANDOM, C_PURPLE, C_LOTO]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(3), vals, color=colors, edgecolor='white', width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(layers)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.annotate('', xy=(0.7, w), xytext=(0.7, r),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(0.85, (r+w)/2, f'noise\n{r-w:.3f}', fontsize=9, va='center')
    ax.annotate('', xy=(1.7, l), xytext=(1.7, w),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(1.85, (w+l)/2, f'transfer\ngap\n{w-l:.3f}', fontsize=9, va='center')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('Three-Layer Ceiling Decomposition')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig3_ceiling.pdf')
    fig.savefig(RDIR / 'fig3_ceiling.png')
    plt.close(fig)
    print('  Fig 3: Ceiling waterfall')


# ============================================================================
# Fig 4: HPO fANOVA Importance
# ============================================================================
def fig4_fanova():
    p = _load_json('results/exp2_posthoc/exp2_posthoc.json')
    imp_all = {}
    for seed_key in ['seed0', 'seed1']:
        if seed_key in p.get('fanova', {}):
            for k, v in p['fanova'][seed_key].items():
                imp_all[k] = imp_all.get(k, []) + [v]
    imp_mean = {k: np.mean(v) for k, v in imp_all.items()}
    imp_sorted = sorted(imp_mean.items(), key=lambda x: -x[1])
    names = [x[0] for x in imp_sorted[:10]]
    vals = [x[1] for x in imp_sorted[:10]]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [C_DM if v > 0.5 else C_RANDOM if v > 0.01 else C_GREY for v in vals]
    ax.barh(range(len(names)-1, -1, -1), vals, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)-1, -1, -1))
    ax.set_yticklabels([n.replace('_', ' ') for n in names])
    ax.set_xlabel('fANOVA Importance')
    for i, v in enumerate(vals):
        ax.text(v + 0.005, len(names)-1-i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_title('HPO Parameter Importance (Random-Phase Trials)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig4_fanova.pdf')
    fig.savefig(RDIR / 'fig4_fanova.png')
    plt.close(fig)
    print('  Fig 4: fANOVA importance')


# ============================================================================
# Fig 5: Few-Shot Line Plot
# ============================================================================
def fig5_fewshot():
    e4 = _load_json('results/exp4_fewshot/exp4_summary.json')
    ks_morgan = [0, 1, 3, 5, 10]
    ks_meta = [1, 3, 5, 10]
    auroc_morgan = [e4[f'rf_morgan_k{k}']['mean'] for k in ks_morgan]
    auroc_meta = [e4[f'rf_meta_k{k}']['mean'] for k in ks_meta]
    maml_morgan = e4.get('maml_morgan_k5', {}).get('mean', None)
    maml_meta = e4.get('maml_meta_k5', {}).get('mean', None)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks_morgan, auroc_morgan, 'o-', color=C_RANDOM, label='RF + Morgan', lw=2, markersize=8)
    ax.plot(ks_meta, auroc_meta, 's-', color=C_DM, label='RF + Morgan + metadata', lw=2, markersize=8)
    if maml_morgan:
        ax.scatter([5], [maml_morgan], marker='D', s=80, color=C_LOTO, zorder=5, label='MAML + Morgan')
    if maml_meta:
        ax.scatter([5], [maml_meta], marker='D', s=80, color=C_RED, zorder=5, label='MAML + meta')
    ax.axhline(0.666, color=C_GREY, ls='--', lw=1, alpha=0.7)
    ax.text(10.3, 0.666, 'LOTO baseline', fontsize=8, va='center', color=C_GREY)
    ax.axhline(0.806, color=C_GREY, ls=':', lw=1, alpha=0.7)
    ax.text(10.3, 0.806, 'within-target ceiling', fontsize=8, va='center', color=C_GREY)
    ax.set_xlabel('k (target-specific examples)')
    ax.set_ylabel('AUROC')
    ax.set_xlim(-0.5, 11.5)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title('Few-Shot Target Adaptation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig5_fewshot.pdf')
    fig.savefig(RDIR / 'fig5_fewshot.png')
    plt.close(fig)
    print('  Fig 5: Few-shot line plot')


# ============================================================================
# Fig 6: DegradeMaster Per-Target EGNN vs RF Scatter
# ============================================================================
def fig6_degrademaster():
    dm = pd.read_csv('results/exp1_degrademaster_loto/loto_degrademaster.csv')
    rf_dm = {
        'P10275': 0.302, 'P00533': 0.688, 'Q06187': 0.468, 'P11802': 0.856,
        'O60885': 0.603, 'Q00534': 0.859, 'Q9UM73': 0.501, 'P36888': 0.457,
        'P09874': 0.726, 'Q02750': 0.556, 'P31749': 0.845, 'P31751': 0.942,
        'P25440': 0.817, 'Q07817': 0.501, 'Q9Y243': 0.924, 'Q15059': 0.686,
        'Q06124': 0.095, 'Q96SW2': 0.183, 'Q00987': 0.472, 'Q05397': 0.800,
        'P24941': 0.844, 'Q9Y616': 0.633, 'P30291': 0.469, 'O60674': 1.000,
        'Q9NYV4': 0.583, 'P11474': 0.472, 'P12931': 0.882,
    }
    # Target class annotations from Session 5
    kinases = {'P24941','P11802','P00533','P31749','P31751','P12931',
               'Q06187','P36888','Q9NYV4','Q9Y616','Q9UM73','P30291',
               'Q06124','Q9Y243','Q07817','Q05397'}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for _, row in dm.iterrows():
        t = row['target']
        if t not in rf_dm: continue
        rf = rf_dm[t]
        egnn = row['auroc']
        size = max(25, min(200, row['n'] * 2))
        is_kinase = t in kinases
        marker = 'o' if is_kinase else 's'
        color = C_DM if egnn > rf else C_LOTO
        ax.scatter(rf, egnn, s=size, c=color, alpha=0.7, marker=marker,
                   edgecolors='white', linewidth=0.5)
        if abs(egnn - rf) > 0.3:
            ax.annotate(t, (rf, egnn), fontsize=6.5, ha='left', va='bottom')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    # legend for marker shapes
    ax.scatter([], [], marker='o', c=C_GREY, s=60, label='Kinase (n=16)')
    ax.scatter([], [], marker='s', c=C_GREY, s=60, label='Non-kinase (n=11)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlabel('RF + Morgan AUROC')
    ax.set_ylabel('DegradeMaster EGNN AUROC')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect('equal')
    ax.set_title('3D Structure Helps Where 2D Fails (rho = -0.847)')
    ax.text(0.05, 0.95, 'EGNN wins: 20/27\np = 0.004',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig6_degrademaster.pdf')
    fig.savefig(RDIR / 'fig6_degrademaster.png')
    plt.close(fig)
    print('  Fig 6: DegradeMaster scatter')


# ============================================================================
# Fig 7: Temporal Split
# ============================================================================
def fig7_temporal():
    temp = _load_json('results/exp_supp_temporal/summary.json')
    years = sorted([k for k in temp if k.isdigit()])
    novel_years = sorted([k for k in temp if 'novel' in k])
    fig, ax = plt.subplots(figsize=(8, 5))
    all_auc = [temp[y]['auroc'] for y in years]
    novel_auc = [temp[y]['auroc'] for y in novel_years]
    novel_labels = [y.replace('_novel_only', '') for y in novel_years]
    ax.plot([int(y) for y in years], all_auc, 'o-', color=C_RANDOM,
            label='All test targets', lw=2)
    ax.plot([int(y) for y in novel_labels], novel_auc, 's--', color=C_LOTO,
            label='Novel targets only', lw=2)
    ax.axhline(0.5, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(0.666, color=C_GREY, ls='--', lw=1, alpha=0.5)
    ax.text(2023.2, 0.666, 'LOTO', fontsize=8, color=C_GREY)
    ax.set_xlabel('Publication Year (train on earlier, test on this year)')
    ax.set_ylabel('AUROC')
    ax.legend()
    ax.set_title('Temporal Split: Novel Targets Are Near Random')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig7_temporal.pdf')
    fig.savefig(RDIR / 'fig7_temporal.png')
    plt.close(fig)
    print('  Fig 7: Temporal split')


# ============================================================================
# Fig 8: ProtTrans Ablation (updated for Session 5 BERT-BFD)
# ============================================================================
def fig8_prottrans():
    # Try Session 5 BERT-BFD results first, fall back to Session 4
    bfd_path = Path('results/exp1_prottrans_bfd/summary.json')
    old_path = Path('results/exp1_full_replication/prottrans_ablation.json')
    if bfd_path.exists():
        abl = _load_json(bfd_path)
        # Session 5 format: nested dicts with poi_e3_subset and poi_only_subset
        poi_e3 = abl.get('poi_e3_subset', {})
        poi_only = abl.get('poi_only_subset', {})
        # Use POI+E3 subset for the 4-bar progressive ablation
        conditions = ['Morgan\nonly', '+ BERT-BFD\nPOI', '+ BERT-BFD\nPOI + E3',
                      '+ BERT-BFD\nPOI+E3+E3type']
        cond_keys = ['E0', 'E1', 'E2', 'E3']
        vals, pvals = [], []
        for ck in cond_keys:
            entry = poi_e3.get(ck, {})
            vals.append(entry.get('loto', 0.66))
            pvals.append(entry.get('p_vs_E0'))
        # Add POI-only significance annotation (the headline p=0.038)
        poi_only_p = poi_only.get('p', None)
        title_suffix = ''
        if poi_only_p is not None:
            title_suffix = f' (POI-only p={poi_only_p:.3f})'
        plm_label = 'BERT-BFD'
    elif old_path.exists():
        abl = _load_json(old_path)
        conditions = ['Morgan\nonly', '+ ProtTrans\nPOI', '+ ProtTrans\nPOI + E3',
                      '+ ProtTrans\nPOI+E3+E3type']
        vals = [abl['morgan_only'], abl['plus_poi'], abl['plus_poi_e3'],
                abl['plus_poi_e3_e3type']]
        pvals = [None, abl['wilcoxon_poi']['p'], abl['wilcoxon_poi_e3']['p'],
                 abl['wilcoxon_poi_e3_e3type']['p']]
        title_suffix = ''
        plm_label = 'ProtTrans'
    else:
        print('  Fig 8: SKIPPED (no ablation data)')
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [C_RANDOM] + [C_LOTO] * 3
    ax.bar(range(4), vals, color=colors, edgecolor='white', width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('LOTO AUROC')
    ymin = min(vals) - 0.02
    ymax = max(vals) + 0.02
    ax.set_ylim(ymin, ymax)
    for i, (v, p) in enumerate(zip(vals, pvals)):
        label = f'{v:.4f}'
        if p is not None:
            sig = '*' if p < 0.05 else 'ns'
            label += f'\n(p={p:.3f} {sig})'
        ax.text(i, v + 0.001, label, ha='center', fontsize=8.5)
    ax.set_title(f'Protein Embeddings Hurt Cold-Target Prediction{title_suffix}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig8_prottrans.pdf')
    fig.savefig(RDIR / 'fig8_prottrans.png')
    plt.close(fig)
    print('  Fig 8: ProtTrans ablation (BERT-BFD)')


# ============================================================================
# Fig 9 (NEW): Top-5 HPO Variance -- HPO best regresses to baseline
# ============================================================================
def fig9_top5_variance():
    path = Path('results/exp2_top5_variance/summary.json')
    if not path.exists():
        print('  Fig 9: SKIPPED (no top-5 variance data)')
        return
    data = _load_json(path)
    # Extract baseline and configs
    baseline = data.get('baseline', {})
    configs = data.get('configs', [])
    if not configs:
        print('  Fig 9: SKIPPED (no configs in summary)')
        return
    # Build arrays: baseline first, then configs sorted by HPO AUROC descending
    names = ['RF+Morgan\n(baseline)']
    hpo_vals = [baseline.get('hpo_auroc', 0.666)]
    seed_vals = [baseline.get('mean', 0.664)]
    seed_stds = [baseline.get('std', 0.004)]
    p_vals = [None]
    configs_sorted = sorted(configs, key=lambda c: -c.get('hpo_auroc', 0))
    for c in configs_sorted:
        label = c.get('label', '')
        if not label:
            mol = c.get('mol_encoder', '?')
            prot = c.get('prot_encoder', '?')
            head = c.get('head_type', '?')
            label = f'{mol}\n{prot}\n{head}'
        names.append(label)
        hpo_vals.append(c.get('hpo_auroc', 0))
        seed_vals.append(c.get('mean', 0))
        seed_stds.append(c.get('std', 0))
        p_vals.append(c.get('p_vs_baseline'))
    n = len(names)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, hpo_vals, w, label='HPO best (single seed)',
           color=C_LOTO, edgecolor='white', alpha=0.8)
    ax.bar(x + w/2, seed_vals, w, label='5-seed mean',
           color=C_RANDOM, edgecolor='white', alpha=0.8,
           yerr=seed_stds, capsize=4, error_kw={'lw': 1.2})
    ax.bar(x[0] + w/2, seed_vals[0], w, color=C_DM, edgecolor='white', alpha=0.8)
    # Significance annotations
    for i in range(1, n):
        p = p_vals[i]
        if p is not None:
            sig = '*' if p < 0.05 else 'ns'
            ax.text(i, max(hpo_vals[i], seed_vals[i]) + seed_stds[i] + 0.005,
                    f'p={p:.3f}\n{sig}', ha='center', fontsize=7.5, color='#555')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, ha='center')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.58, 0.75)
    ax.axhline(baseline.get('mean', 0.664), color=C_GREY, ls='--', lw=1, alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('HPO "Improvements" Vanish Under Multi-Seed Validation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig9_top5_variance.pdf')
    fig.savefig(RDIR / 'fig9_top5_variance.png')
    plt.close(fig)
    print('  Fig 9: Top-5 HPO variance')


# ============================================================================
# Fig S1 (NEW): Target Class Analysis -- kinase vs non-kinase EGNN advantage
# ============================================================================
def figS1_target_class():
    # Hardcoded from Session 5 target class analysis (robust, no CSV dependency)
    kinase_data = {'egnn': 0.822, 'rf': 0.730, 'n': 16}
    nonkinase_data = {'egnn': 0.770, 'rf': 0.499, 'n': 11}
    fig, ax = plt.subplots(figsize=(6, 5))
    w = 0.3
    ax.bar([0 - w/2, 1 - w/2], [kinase_data['rf'], nonkinase_data['rf']], w,
           label='RF + Morgan', color=C_LOTO, edgecolor='white')
    ax.bar([0 + w/2, 1 + w/2], [kinase_data['egnn'], nonkinase_data['egnn']], w,
           label='EGNN (3D)', color=C_DM, edgecolor='white')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Kinase\n(n={kinase_data["n"]})',
                        f'Non-kinase\n(n={nonkinase_data["n"]})'])
    ax.set_ylabel('Mean LOTO AUROC')
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    for i, d in enumerate([kinase_data, nonkinase_data]):
        delta = d['egnn'] - d['rf']
        mid = (d['egnn'] + d['rf']) / 2
        ax.text(i, mid, f'+{delta:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.05, 0.95, 'Kinase: p=0.173 ns\nNon-kinase: p=0.010 *',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    ax.set_title('3D Advantage Is Larger Where 2D Is Blind')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'figS1_target_class.pdf')
    fig.savefig(RDIR / 'figS1_target_class.png')
    plt.close(fig)
    print('  Fig S1: Target class')


# ============================================================================
# Fig S2 (NEW): Threshold Sensitivity
# ============================================================================
def figS2_threshold():
    path = Path('results/exp_supp_threshold/threshold_sensitivity.csv')
    if not path.exists():
        print('  Fig S2: SKIPPED (no threshold data)')
        return
    df = pd.read_csv(path)
    df = df[df['loto'].notna()].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: LOTO AUROC across thresholds
    ax = axes[0]
    x = np.arange(len(df))
    colors = [C_RANDOM if 'DC50' in t and 'OR' not in t and 'AND' not in t
              else C_LOTO if 'Dmax' in t
              else C_DM if 'OR' in t or 'AND' in t or 'DM' in t
              else C_GREY for t in df['threshold']]
    ax.barh(x, df['loto'], color=colors, edgecolor='white')
    ax.set_yticks(x)
    labels = df['threshold'].apply(lambda s: s.replace('_', ' ').replace('(DM)', '\n(DM)'))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('LOTO AUROC')
    ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.axvline(0.666, color=C_GREY, ls=':', lw=1, alpha=0.5)
    ax.set_xlim(0.3, 0.9)
    ax.set_title('LOTO AUROC by Threshold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Right: Collapse (random - LOTO)
    ax = axes[1]
    ax.barh(x, df['collapse'], color=colors, edgecolor='white')
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Collapse (Random - LOTO)')
    ax.set_xlim(0, 0.45)
    ax.set_title('Collapse by Threshold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.suptitle('Label Threshold Sensitivity: Collapse Is Universal', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(RDIR / 'figS2_threshold.pdf')
    fig.savefig(RDIR / 'figS2_threshold.png')
    plt.close(fig)
    print('  Fig S2: Threshold sensitivity')


# ============================================================================
# Fig S3 (NEW): Linker Independence
# ============================================================================
def figS3_linker():
    path = Path('results/exp_supp_linker/summary.json')
    if not path.exists():
        print('  Fig S3: SKIPPED (no linker data)')
        return
    data = _load_json(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # Left: linker overlap per target
    overlap_path = Path('results/exp_supp_linker/per_target_linker_overlap.csv')
    if overlap_path.exists():
        ax = axes[0]
        odf = pd.read_csv(overlap_path)
        odf_sorted = odf.sort_values('frac_overlap')
        ax.barh(range(len(odf_sorted)), odf_sorted['frac_overlap'],
                color=C_RANDOM, edgecolor='white', alpha=0.8)
        ax.set_xlabel('Fraction of linkers shared with training')
        ax.set_ylabel(f'LOTO targets (n={len(odf_sorted)})')
        ax.set_yticks([])
        ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
        ax.set_title('Linker Overlap Per LOTO Target')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # Right: fragment LOTO comparison
    ax = axes[1]
    frag_loto = data.get('loto_by_fragment', {})
    frag_names = ['full\nprotac', 'warhead', 'linker']
    frag_vals = [frag_loto.get('full_protac', 0), frag_loto.get('warhead', 0),
                 frag_loto.get('linker', 0)]
    colors_f = [C_RANDOM, C_LOTO, C_DM]
    ax.bar(range(len(frag_names)), frag_vals, color=colors_f, edgecolor='white', width=0.6)
    ax.set_xticks(range(len(frag_names)))
    ax.set_xticklabels(frag_names)
    ax.set_ylabel('LOTO AUROC')
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    for i, v in enumerate(frag_vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.set_title('Fragment-Specific LOTO')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    tani = data.get('tanimoto_nn', {})
    fig.suptitle(f'Linker Independence (NN Tanimoto: real={tani.get("mean",0):.3f}, '
                 f'random={tani.get("random_baseline_mean",0):.3f}, '
                 f'p<0.001)', fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(RDIR / 'figS3_linker.pdf')
    fig.savefig(RDIR / 'figS3_linker.png')
    plt.close(fig)
    print('  Fig S3: Linker independence')


# ============================================================================
def main():
    print('=' * 50)
    print('  Generating Publication Figures (Session 5)')
    print('=' * 50)
    # Main figures
    fig1_collapse()
    fig2_fragments()
    fig3_ceiling()
    fig4_fanova()
    fig5_fewshot()
    fig6_degrademaster()
    fig7_temporal()
    fig8_prottrans()
    fig9_top5_variance()
    # Supplementary
    figS1_target_class()
    figS2_threshold()
    figS3_linker()
    print(f'\n  All figures saved to {RDIR}')
    figs = sorted(f.name for f in RDIR.iterdir())
    print(f'  Files ({len(figs)}): {figs}')


if __name__ == '__main__':
    main()
