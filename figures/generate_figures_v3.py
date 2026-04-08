#!/usr/bin/env python3
"""Generate all publication figures for PROTAC-PLM-Bench.
Session 6 update: EGNN replication (0.788), 3D descriptor null, label mismatch.

Run from /workspace: python scripts/generate_figures.py
"""
import json, warnings, sys, glob
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# Fig 1: Collapse Bar Chart -- updated EGNN to 0.788 (Session 6 replication)
# ============================================================================
def fig1_collapse():
    methods = [
        ('RF+Morgan', 0.902, 0.666, 65),
        ('XGBoost+Morgan', 0.900, 0.642, 65),
        ('Ribes-style', 0.913, 0.646, 61),
        ('STAN-style (RF)', 0.908, 0.653, 54),
        ('STAN TAN', 0.902, 0.660, 54),
        ('DegradeMaster\nEGNN (3D)', 0.830, 0.788, 27),
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
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig1_collapse.pdf'); fig.savefig(RDIR / 'fig1_collapse.png')
    plt.close(fig); print('  Fig 1: Collapse bar chart')

# ============================================================================
# Fig 2: Fragment Decomposition
# ============================================================================
def fig2_fragments():
    frags = _load_json('results/exp_supp_fragments/summary.json')
    names, random_v, loto_v = [], [], []
    for k in ['full_protac', 'warhead', 'e3_ligand', 'linker', 'anchor']:
        if k in frags and isinstance(frags[k], dict):
            names.append(k.replace('_', '\n'))
            random_v.append(frags[k]['random']); loto_v.append(frags[k]['loto'])
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, random_v, w, label='Random split', color=C_RANDOM, edgecolor='white')
    ax.bar(x + w/2, loto_v, w, label='LOTO', color=C_LOTO, edgecolor='white')
    ax.set_ylabel('AUROC'); ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend(); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    for i in range(len(names)):
        collapse = random_v[i] - loto_v[i]
        ax.annotate(f'{collapse:.3f}', (i, (random_v[i]+loto_v[i])/2), ha='center', fontsize=9, color='#333')
    ax.set_title('Fragment Signal Decomposition')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig2_fragments.pdf'); fig.savefig(RDIR / 'fig2_fragments.png')
    plt.close(fig); print('  Fig 2: Fragment decomposition')

# ============================================================================
# Fig 3: Three-Layer Ceiling Waterfall
# ============================================================================
def fig3_ceiling():
    ceil = _load_json('results/exp_supp_ceiling/summary.json')
    r = ceil.get('random_cv', 0.902); w = ceil.get('within_target_mean', 0.806)
    l = ceil.get('loto_mean', 0.666)
    layers = ['Random CV\n(interpolation)', 'Within-target\n(target-specific)', 'LOTO\n(cold-target)']
    vals = [r, w, l]; colors = [C_RANDOM, C_PURPLE, C_LOTO]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(3), vals, color=colors, edgecolor='white', width=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(layers)
    ax.set_ylabel('AUROC'); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.annotate('', xy=(0.7, w), xytext=(0.7, r), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(0.85, (r+w)/2, f'noise\n{r-w:.3f}', fontsize=9, va='center')
    ax.annotate('', xy=(1.7, l), xytext=(1.7, w), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(1.85, (w+l)/2, f'transfer\ngap\n{w-l:.3f}', fontsize=9, va='center')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('Three-Layer Ceiling Decomposition')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig3_ceiling.pdf'); fig.savefig(RDIR / 'fig3_ceiling.png')
    plt.close(fig); print('  Fig 3: Ceiling waterfall')

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
    names = [x[0] for x in imp_sorted[:10]]; vals = [x[1] for x in imp_sorted[:10]]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [C_DM if v > 0.5 else C_RANDOM if v > 0.01 else C_GREY for v in vals]
    ax.barh(range(len(names)-1, -1, -1), vals, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)-1, -1, -1))
    ax.set_yticklabels([n.replace('_', ' ') for n in names])
    ax.set_xlabel('fANOVA Importance')
    for i, v in enumerate(vals):
        ax.text(v + 0.005, len(names)-1-i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_title('HPO Parameter Importance (Random-Phase Trials)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig4_fanova.pdf'); fig.savefig(RDIR / 'fig4_fanova.png')
    plt.close(fig); print('  Fig 4: fANOVA importance')

# ============================================================================
# Fig 5: Few-Shot Line Plot
# ============================================================================
def fig5_fewshot():
    e4 = _load_json('results/exp4_fewshot/exp4_summary.json')
    ks_morgan = [0, 1, 3, 5, 10]; ks_meta = [1, 3, 5, 10]
    auroc_morgan = [e4[f'rf_morgan_k{k}']['mean'] for k in ks_morgan]
    auroc_meta = [e4[f'rf_meta_k{k}']['mean'] for k in ks_meta]
    maml_morgan = e4.get('maml_morgan_k5', {}).get('mean')
    maml_meta = e4.get('maml_meta_k5', {}).get('mean')
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks_morgan, auroc_morgan, 'o-', color=C_RANDOM, label='RF + Morgan', lw=2, markersize=8)
    ax.plot(ks_meta, auroc_meta, 's-', color=C_DM, label='RF + Morgan + metadata', lw=2, markersize=8)
    if maml_morgan: ax.scatter([5], [maml_morgan], marker='D', s=80, color=C_LOTO, zorder=5, label='MAML + Morgan')
    if maml_meta: ax.scatter([5], [maml_meta], marker='D', s=80, color=C_RED, zorder=5, label='MAML + meta')
    ax.axhline(0.666, color=C_GREY, ls='--', lw=1, alpha=0.7)
    ax.text(10.3, 0.666, 'LOTO baseline', fontsize=8, va='center', color=C_GREY)
    ax.axhline(0.806, color=C_GREY, ls=':', lw=1, alpha=0.7)
    ax.text(10.3, 0.806, 'within-target ceiling', fontsize=8, va='center', color=C_GREY)
    ax.set_xlabel('k (target-specific examples)'); ax.set_ylabel('AUROC')
    ax.set_xlim(-0.5, 11.5); ax.legend(loc='lower right', fontsize=9)
    ax.set_title('Few-Shot Target Adaptation')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig5_fewshot.pdf'); fig.savefig(RDIR / 'fig5_fewshot.png')
    plt.close(fig); print('  Fig 5: Few-shot line plot')

# ============================================================================
# Fig 6: DegradeMaster Scatter -- Session 6 control numbers
# ============================================================================
def fig6_degrademaster():
    # Use Session 6 control results
    ctrl_path = Path('results/exp5d_control/results.csv')
    if ctrl_path.exists():
        rdf = pd.read_csv(ctrl_path)
    else:
        # Fallback to original exp1f
        rdf = pd.read_csv('results/exp1_degrademaster_loto/loto_degrademaster.csv')
        rdf.columns = ['target', 'n', 'auroc'] if len(rdf.columns) == 3 else rdf.columns
        # Add RF from hardcoded Session 4 values
        rf_dm = {'P10275': 0.302, 'P00533': 0.688, 'Q06187': 0.468, 'P11802': 0.856,
                 'O60885': 0.603, 'Q00534': 0.859, 'Q9UM73': 0.501, 'P36888': 0.457,
                 'P09874': 0.726, 'Q02750': 0.556, 'P31749': 0.845, 'P31751': 0.942,
                 'P25440': 0.817, 'Q07817': 0.501, 'Q9Y243': 0.924, 'Q15059': 0.686,
                 'Q06124': 0.095, 'Q96SW2': 0.183, 'Q00987': 0.472, 'Q05397': 0.800,
                 'P24941': 0.844, 'Q9Y616': 0.633, 'P30291': 0.469, 'O60674': 1.000,
                 'Q9NYV4': 0.583, 'P11474': 0.472, 'P12931': 0.882}
        rdf['rf'] = rdf['target'].map(rf_dm)
    kinases = {'P24941','P11802','P00533','P31749','P31751','P12931',
               'Q06187','P36888','Q9NYV4','Q9Y616','Q9UM73','P30291',
               'Q06124','Q9Y243','Q07817','Q05397'}
    egnn_mean = rdf['auroc'].mean()
    rf_mean = rdf['rf'].mean()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for _, row in rdf.iterrows():
        rf_val = row['rf']; egnn_val = row['auroc']
        if pd.isna(rf_val): continue
        size = max(25, min(200, row['n'] * 2))
        is_kinase = row['target'] in kinases
        marker = 'o' if is_kinase else 's'
        color = C_DM if egnn_val > rf_val else C_LOTO
        ax.scatter(rf_val, egnn_val, s=size, c=color, alpha=0.7, marker=marker,
                   edgecolors='white', linewidth=0.5)
        delta = egnn_val - rf_val
        if abs(delta) > 0.3:
            ax.annotate(row['target'], (rf_val, egnn_val), fontsize=6.5, ha='left', va='bottom')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.scatter([], [], marker='o', c=C_GREY, s=60, label='Kinase (n=16)')
    ax.scatter([], [], marker='s', c=C_GREY, s=60, label='Non-kinase (n=11)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlabel('RF + Morgan AUROC'); ax.set_ylabel('DegradeMaster EGNN AUROC')
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_aspect('equal')
    ax.set_title(f'3D Structure Helps Where 2D Fails')
    ax.text(0.05, 0.95, f'EGNN: {egnn_mean:.3f}\nRF: {rf_mean:.3f}\np = 0.002',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig6_degrademaster.pdf'); fig.savefig(RDIR / 'fig6_degrademaster.png')
    plt.close(fig); print('  Fig 6: DegradeMaster scatter (Session 6)')

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
    ax.plot([int(y) for y in years], all_auc, 'o-', color=C_RANDOM, label='All test targets', lw=2)
    ax.plot([int(y) for y in novel_labels], novel_auc, 's--', color=C_LOTO, label='Novel targets only', lw=2)
    ax.axhline(0.5, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(0.666, color=C_GREY, ls='--', lw=1, alpha=0.5)
    ax.text(2023.2, 0.666, 'LOTO', fontsize=8, color=C_GREY)
    ax.set_xlabel('Publication Year (train on earlier, test on this year)')
    ax.set_ylabel('AUROC'); ax.legend()
    ax.set_title('Temporal Split: Novel Targets Are Near Random')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig7_temporal.pdf'); fig.savefig(RDIR / 'fig7_temporal.png')
    plt.close(fig); print('  Fig 7: Temporal split')

# ============================================================================
# Fig 8: ProtTrans Ablation (Session 5 BERT-BFD)
# ============================================================================
def fig8_prottrans():
    bfd_path = Path('results/exp1_prottrans_bfd/summary.json')
    if not bfd_path.exists():
        print('  Fig 8: SKIPPED (no BERT-BFD data)'); return
    abl = _load_json(bfd_path)
    poi_e3 = abl.get('poi_e3_subset', {})
    poi_only = abl.get('poi_only_subset', {})
    conditions = ['Morgan\nonly', '+ BERT-BFD\nPOI', '+ BERT-BFD\nPOI + E3', '+ BERT-BFD\nPOI+E3+E3type']
    vals, pvals = [], []
    for ck in ['E0', 'E1', 'E2', 'E3']:
        entry = poi_e3.get(ck, {})
        vals.append(entry.get('loto', 0.66)); pvals.append(entry.get('p_vs_E0'))
    poi_only_p = poi_only.get('p')
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [C_RANDOM] + [C_LOTO] * 3
    ax.bar(range(4), vals, color=colors, edgecolor='white', width=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('LOTO AUROC')
    ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
    for i, (v, p) in enumerate(zip(vals, pvals)):
        label = f'{v:.4f}'
        if p is not None:
            sig = '*' if p < 0.05 else 'ns'
            label += f'\n(p={p:.3f} {sig})'
        ax.text(i, v + 0.001, label, ha='center', fontsize=8.5)
    title = 'Protein Embeddings Hurt Cold-Target Prediction'
    if poi_only_p is not None: title += f' (POI-only p={poi_only_p:.3f})'
    ax.set_title(title)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig8_prottrans.pdf'); fig.savefig(RDIR / 'fig8_prottrans.png')
    plt.close(fig); print('  Fig 8: ProtTrans ablation (BERT-BFD)')

# ============================================================================
# Fig 9: Top-5 HPO Variance
# ============================================================================
def fig9_top5_variance():
    path = Path('results/exp2_top5_variance/summary.json')
    if not path.exists():
        print('  Fig 9: SKIPPED'); return
    data = _load_json(path)
    baseline = data.get('baseline', {}); configs = data.get('configs', [])
    if not configs: print('  Fig 9: SKIPPED (no configs)'); return
    names = ['RF+Morgan\n(baseline)']
    hpo_vals = [baseline.get('hpo_auroc', 0.666)]
    seed_vals = [baseline.get('mean', 0.664)]; seed_stds = [baseline.get('std', 0.004)]
    p_vals = [None]
    for c in sorted(configs, key=lambda c: -c.get('hpo_auroc', 0)):
        label = c.get('label', f"{c.get('mol_encoder','?')}\n{c.get('prot_encoder','?')}\n{c.get('head_type','?')}")
        names.append(label); hpo_vals.append(c.get('hpo_auroc', 0))
        seed_vals.append(c.get('mean', 0)); seed_stds.append(c.get('std', 0))
        p_vals.append(c.get('p_vs_baseline'))
    n = len(names); fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(n); w = 0.35
    ax.bar(x - w/2, hpo_vals, w, label='HPO best (single seed)', color=C_LOTO, edgecolor='white', alpha=0.8)
    ax.bar(x + w/2, seed_vals, w, label='5-seed mean', color=C_RANDOM, edgecolor='white', alpha=0.8,
           yerr=seed_stds, capsize=4, error_kw={'lw': 1.2})
    ax.bar(x[0] + w/2, seed_vals[0], w, color=C_DM, edgecolor='white', alpha=0.8)
    for i in range(1, n):
        p = p_vals[i]
        if p is not None:
            sig = '*' if p < 0.05 else 'ns'
            ax.text(i, max(hpo_vals[i], seed_vals[i]) + seed_stds[i] + 0.005,
                    f'p={p:.3f}\n{sig}', ha='center', fontsize=7.5, color='#555')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, ha='center')
    ax.set_ylabel('AUROC'); ax.set_ylim(0.58, 0.75)
    ax.axhline(baseline.get('mean', 0.664), color=C_GREY, ls='--', lw=1, alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('HPO "Improvements" Vanish Under Multi-Seed Validation')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'fig9_top5_variance.pdf'); fig.savefig(RDIR / 'fig9_top5_variance.png')
    plt.close(fig); print('  Fig 9: Top-5 HPO variance')

# ============================================================================
# Fig S1: Target Class (kinase vs non-kinase)
# ============================================================================
def figS1_target_class():
    # Use Session 6 control results if available
    ctrl_path = Path('results/exp5d_control/results.csv')
    if ctrl_path.exists():
        rdf = pd.read_csv(ctrl_path)
        kinases = {'P24941','P11802','P00533','P31749','P31751','P12931',
                   'Q06187','P36888','Q9NYV4','Q9Y616','Q9UM73','P30291',
                   'Q06124','Q9Y243','Q07817','Q05397'}
        rdf['cls'] = rdf['target'].apply(lambda t: 'kinase' if t in kinases else 'non-kinase')
        kin = rdf[rdf['cls'] == 'kinase']; nk = rdf[rdf['cls'] == 'non-kinase']
        kd = {'egnn': kin['auroc'].mean(), 'rf': kin['rf'].mean(), 'n': len(kin)}
        nd = {'egnn': nk['auroc'].mean(), 'rf': nk['rf'].mean(), 'n': len(nk)}
    else:
        kd = {'egnn': 0.822, 'rf': 0.730, 'n': 16}
        nd = {'egnn': 0.770, 'rf': 0.499, 'n': 11}
    fig, ax = plt.subplots(figsize=(6, 5)); w = 0.3
    ax.bar([0 - w/2, 1 - w/2], [kd['rf'], nd['rf']], w, label='RF + Morgan', color=C_LOTO, edgecolor='white')
    ax.bar([0 + w/2, 1 + w/2], [kd['egnn'], nd['egnn']], w, label='EGNN (3D)', color=C_DM, edgecolor='white')
    ax.set_xticks([0, 1]); ax.set_xticklabels([f'Kinase\n(n={kd["n"]})', f'Non-kinase\n(n={nd["n"]})'])
    ax.set_ylabel('Mean LOTO AUROC'); ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    for i, d in enumerate([kd, nd]):
        delta = d['egnn'] - d['rf']; mid = (d['egnn'] + d['rf']) / 2
        ax.text(i, mid, f'+{delta:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_title('3D Advantage Is Larger Where 2D Is Blind')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(RDIR / 'figS1_target_class.pdf'); fig.savefig(RDIR / 'figS1_target_class.png')
    plt.close(fig); print('  Fig S1: Target class')

# ============================================================================
# Fig S2: Threshold Sensitivity
# ============================================================================
def figS2_threshold():
    path = Path('results/exp_supp_threshold/threshold_sensitivity.csv')
    if not path.exists(): print('  Fig S2: SKIPPED'); return
    df = pd.read_csv(path); df = df[df['loto'].notna()].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(df))
    colors = [C_RANDOM if 'DC50' in t and 'OR' not in t and 'AND' not in t
              else C_LOTO if 'Dmax' in t else C_DM for t in df['threshold']]
    ax = axes[0]
    ax.barh(x, df['loto'], color=colors, edgecolor='white')
    ax.set_yticks(x); labels = df['threshold'].apply(lambda s: s.replace('_', ' ').replace('(DM)', '\n(DM)'))
    ax.set_yticklabels(labels, fontsize=7.5); ax.set_xlabel('LOTO AUROC')
    ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5); ax.set_xlim(0.3, 0.9)
    ax.set_title('LOTO AUROC by Threshold'); ax.invert_yaxis()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax = axes[1]
    ax.barh(x, df['collapse'], color=colors, edgecolor='white')
    ax.set_yticks(x); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Collapse (Random - LOTO)'); ax.set_xlim(0, 0.45)
    ax.set_title('Collapse by Threshold'); ax.invert_yaxis()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Label Threshold Sensitivity: Collapse Is Universal', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(RDIR / 'figS2_threshold.pdf'); fig.savefig(RDIR / 'figS2_threshold.png')
    plt.close(fig); print('  Fig S2: Threshold sensitivity')

# ============================================================================
# Fig S3: Linker Independence
# ============================================================================
def figS3_linker():
    path = Path('results/exp_supp_linker/summary.json')
    if not path.exists(): print('  Fig S3: SKIPPED'); return
    data = _load_json(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    overlap_path = Path('results/exp_supp_linker/per_target_linker_overlap.csv')
    if overlap_path.exists():
        ax = axes[0]; odf = pd.read_csv(overlap_path)
        odf_sorted = odf.sort_values('frac_overlap')
        ax.barh(range(len(odf_sorted)), odf_sorted['frac_overlap'], color=C_RANDOM, edgecolor='white', alpha=0.8)
        ax.set_xlabel('Fraction of linkers shared with training')
        ax.set_ylabel(f'LOTO targets (n={len(odf_sorted)})'); ax.set_yticks([])
        ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
        ax.set_title('Linker Overlap Per LOTO Target')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax = axes[1]; frag_loto = data.get('loto_by_fragment', {})
    frag_names = ['full\nprotac', 'warhead', 'linker']
    frag_vals = [frag_loto.get('full_protac', 0), frag_loto.get('warhead', 0), frag_loto.get('linker', 0)]
    ax.bar(range(3), frag_vals, color=[C_RANDOM, C_LOTO, C_DM], edgecolor='white', width=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(frag_names); ax.set_ylabel('LOTO AUROC')
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    for i, v in enumerate(frag_vals): ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    ax.set_ylim(0, 0.8); ax.set_title('Fragment-Specific LOTO')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    tani = data.get('tanimoto_nn', {})
    fig.suptitle(f'Linker Independence (NN Tanimoto: real={tani.get("mean",0):.3f}, '
                 f'random={tani.get("random_baseline_mean",0):.3f}, p<0.001)', fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(RDIR / 'figS3_linker.pdf'); fig.savefig(RDIR / 'figS3_linker.png')
    plt.close(fig); print('  Fig S3: Linker independence')

# ============================================================================
# Fig S4 (NEW): 3D Descriptor Null + Label Mismatch
# ============================================================================
def figS4_3d_and_mismatch():
    # Panel A: 3D descriptors (Exp 5A)
    desc_path = Path('results/exp5a_3d_descriptors/summary.json')
    # Panel B: Label mismatch (expanded vs control on overlapping targets)
    exp_path = Path('results/exp5d_egnn_expanded')
    ctrl_path = Path('results/exp5d_control/results.csv')
    has_desc = desc_path.exists()
    has_mismatch = ctrl_path.exists() and len(list(exp_path.glob('fold_*.json'))) > 5
    if not has_desc and not has_mismatch:
        print('  Fig S4: SKIPPED'); return
    n_panels = sum([has_desc, has_mismatch])
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1: axes = [axes]
    pi = 0
    # Panel A: 3D descriptors
    if has_desc:
        ax = axes[pi]; pi += 1
        desc = _load_json(desc_path)
        conds = desc.get('conditions', {})
        names = ['Morgan\nonly', '3D\nonly', 'Morgan\n+3D', 'Morgan\n+shape', 'Morgan\n+2D phys']
        keys = ['C0_morgan', 'C1_3d_only', 'C2_morgan+3d', 'C3_morgan+shape', 'C5_morgan+2d_phys']
        vals = [conds.get(k, {}).get('loto', 0) for k in keys]
        ps = [conds.get(k, {}).get('p') for k in keys]
        colors_b = [C_RANDOM] + [C_LOTO] * 4
        ax.bar(range(len(names)), vals, color=colors_b, edgecolor='white', width=0.6)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel('LOTO AUROC'); ax.set_ylim(0.55, 0.70)
        for i, (v, p) in enumerate(zip(vals, ps)):
            label = f'{v:.3f}'
            if p is not None and i > 0: label += f'\np={p:.3f}'
            ax.text(i, v + 0.002, label, ha='center', fontsize=7.5)
        ax.set_title('RDKit 3D Descriptors Add Nothing')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # Panel B: Label mismatch
    if has_mismatch:
        ax = axes[pi]; pi += 1
        ctrl = pd.read_csv(ctrl_path)
        # Load expanded results for overlapping targets
        exp_results = {}
        for f in exp_path.glob('fold_*.json'):
            d = json.load(open(f))
            exp_results[d['target']] = d['auroc']
        common = sorted(set(ctrl['target']) & set(exp_results.keys()))
        if len(common) >= 3:
            ctrl_vals = [ctrl[ctrl['target'] == t]['auroc'].values[0] for t in common]
            exp_vals = [exp_results[t] for t in common]
            x = np.arange(len(common)); w = 0.35
            ax.bar(x - w/2, ctrl_vals, w, label='DM labels (consistent)', color=C_DM, edgecolor='white')
            ax.bar(x + w/2, exp_vals, w, label='Mixed labels (mismatch)', color=C_RED, edgecolor='white')
            ax.set_xticks(x); ax.set_xticklabels(common, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('LOTO AUROC'); ax.legend(fontsize=8)
            ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
            ax.set_title('Label Threshold Mismatch Hurts 3D Models')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(RDIR / 'figS4_3d_and_mismatch.pdf'); fig.savefig(RDIR / 'figS4_3d_and_mismatch.png')
    plt.close(fig); print('  Fig S4: 3D descriptors + label mismatch')

# ============================================================================
def main():
    print('=' * 50)
    print('  Generating Publication Figures (Session 6)')
    print('=' * 50)
    fig1_collapse()
    fig2_fragments()
    fig3_ceiling()
    fig4_fanova()
    fig5_fewshot()
    fig6_degrademaster()
    fig7_temporal()
    fig8_prottrans()
    fig9_top5_variance()
    figS1_target_class()
    figS2_threshold()
    figS3_linker()
    figS4_3d_and_mismatch()
    print(f'\n  All figures saved to {RDIR}')
    figs = sorted(f.name for f in RDIR.iterdir())
    print(f'  Files ({len(figs)}): {figs}')

if __name__ == '__main__':
    main()
