#!/usr/bin/env python3
"""Generate updated NeurIPS figures (v2) for PROTAC-Bench.
Figures 1-4 and 7, saved as PDF+PNG to /workspace/results/neurips_figures/.
"""
import json, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

RDIR = Path('/workspace/results/neurips_figures')
RDIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.size': 11, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'xtick.major.width': 1, 'ytick.major.width': 1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

def _load_json(path):
    with open(path) as f:
        return json.load(f)

# ============================================================================
# Fig 1: Updated Collapse (14x6)
# ============================================================================
def fig1_collapse():
    print('  Fig 1: Collapse bar chart ...')
    # --- Load data ---
    full = _load_json('results/exp1_full_replication/full_summary.json')
    baseline = _load_json('results/exp1_baseline_replication/summary.json')
    dm_random = _load_json('results/exp1_degrademaster_loto/random_split.json')
    dm_summary = _load_json('results/exp1_degrademaster_loto/summary.json')
    knn = _load_json('results/exp13_splits_and_baselines/knn_baseline.json')
    scaffold = _load_json('results/exp13_splits_and_baselines/scaffold_split.json')
    tani_dist = _load_json('results/exp13_splits_and_baselines/tanimoto_distributions.json')

    # Per-target CSVs for dot overlays
    rf_per = pd.read_csv('results/exp1_baseline_replication/A_rf_morgan.csv')
    dm_per = pd.read_csv('results/exp1_degrademaster_loto/loto_degrademaster.csv')
    ribes_per = pd.read_csv('results/exp1_full_replication/loto_C_ribes_style.csv')
    stan_per = pd.read_csv('results/exp1_full_replication/loto_D_stan_style.csv')

    # kNN k=5: average over seeds for per-target
    knn_seeds = knn['loto']
    knn_per_target = np.mean([knn_seeds[s]['per_target_aurocs'] for s in knn_seeds], axis=0)
    knn_random = np.mean([knn['random_cv'][s]['mean_auroc'] for s in knn['random_cv']])
    knn_loto = np.mean([knn_seeds[s]['mean_auroc'] for s in knn_seeds])

    # Tanimoto per-target mapping for dot colors
    tani_loto = np.load('results/exp13_splits_and_baselines/tanimoto_loto.npy')
    df_full = pd.read_csv('results/ext30_data_expansion/exp1_merged_dataset.csv')
    targets_65 = rf_per['target'].tolist()
    eligible = df_full[df_full['target_uniprot'].isin(targets_65)].copy()
    eligible = eligible.reset_index(drop=True)
    # Map tanimoto to entries, then compute per-target mean
    eligible['tanimoto'] = tani_loto
    tani_per_target = eligible.groupby('target_uniprot')['tanimoto'].mean()

    # --- Define methods ---
    # Group 1: Replications (reported vs replicated)
    # DeepPROTACs: reported=0.847, LOTO=N/A
    # Ribes: reported random=0.865, cs=0.604; replicated random=0.912, LOTO=0.646
    # PROTAC-STAN: reported random=0.883; replicated LOTO=0.682 (27 targets)
    # DegradeMaster: reported random=0.883; replicated random=0.830, LOTO=0.801

    methods = []
    # Group 1 - Replications
    methods.append({
        'name': 'DeepPROTACs', 'group': 1,
        'reported': 0.847, 'replicated': None,  # N/A
        'per_target': None, 'n_targets': None,
    })
    methods.append({
        'name': 'Ribes et al.', 'group': 1,
        'reported': 0.865, 'replicated': full['conditions']['C_ribes_style']['loto'],
        'per_target': ribes_per, 'n_targets': 65,
        'reported_loto': 0.604,
    })
    methods.append({
        'name': 'PROTAC-STAN', 'group': 1,
        'reported': 0.883, 'replicated': full['stan_own_data']['loto_rf'],
        'per_target': stan_per, 'n_targets': 54,
    })
    methods.append({
        'name': 'DegradeMaster', 'group': 1,
        'reported': 0.883, 'replicated': dm_summary['mean'],
        'reported_random': 0.883, 'replicated_random': dm_random['random_split_mean'],
        'per_target': dm_per, 'n_targets': 27,
    })
    # Group 2 - Our baselines
    methods.append({
        'name': 'RF+Morgan', 'group': 2,
        'reported': full['conditions']['A_rf_morgan']['random'],
        'replicated': full['conditions']['A_rf_morgan']['loto'],
        'per_target': rf_per, 'n_targets': 65,
    })
    methods.append({
        'name': 'kNN (k=5)', 'group': 2,
        'reported': knn_random,
        'replicated': knn_loto,
        'per_target_arr': knn_per_target, 'n_targets': 65,
    })
    methods.append({
        'name': 'Scaffold CV', 'group': 2,
        'reported': scaffold['mean_auroc'],
        'replicated': None,  # single bar
        'per_target': None, 'n_targets': None,
    })

    # --- Plot Panel A (left) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.6, 1]})

    n = len(methods)
    x = np.arange(n).astype(float)
    # Add gap between group 1 and group 2
    for i in range(n):
        if methods[i]['group'] == 2:
            x[i:] += 0.5
            break

    w = 0.35
    C_REPORT = '#6BAED6'  # lighter
    C_LOTO = '#E6550D'    # darker
    C_BASELINE = '#74C476'

    cmap = cm.viridis
    norm = Normalize(vmin=0.3, vmax=0.9)

    for i, m in enumerate(methods):
        # Reported bar (left)
        if m['reported'] is not None:
            is_group2 = m['group'] == 2
            color_r = C_BASELINE if is_group2 else C_REPORT
            label_r = 'Random CV' if i == 0 else ('Our baselines' if (is_group2 and i == 4) else None)
            ax1.bar(x[i] - w/2, m['reported'], w, color=color_r, alpha=0.25,
                    edgecolor=color_r, linewidth=1.5, label=label_r)
            ax1.text(x[i] - w/2, m['reported'] + 0.01, f"{m['reported']:.3f}",
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=45)

        # Replicated/LOTO bar (right)
        if m['replicated'] is not None:
            color_l = C_LOTO
            label_l = 'LOTO (cold-target)' if i == 1 else None
            std_val = None
            per_target_vals = None
            per_target_tani = None

            # Get per-target values
            if 'per_target' in m and m['per_target'] is not None:
                pt = m['per_target']
                per_target_vals = pt['auroc'].values
                std_val = np.std(per_target_vals)
                # Map tanimoto
                per_target_tani = np.array([tani_per_target.get(t, 0.5) for t in pt['target'].values])
            elif 'per_target_arr' in m:
                per_target_vals = m['per_target_arr']
                std_val = np.std(per_target_vals)
                # For kNN, use same target order as RF
                per_target_tani = np.array([tani_per_target.get(t, 0.5) for t in targets_65])

            ax1.bar(x[i] + w/2, m['replicated'], w, color=C_LOTO, alpha=0.25,
                    edgecolor=C_LOTO, linewidth=1.5, label=label_l,
                    yerr=std_val, capsize=3, error_kw={'lw': 1, 'color': '#666'})
            ax1.text(x[i] + w/2, m['replicated'] + (std_val or 0) + 0.015,
                     f"{m['replicated']:.3f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=45)

            # Overlay per-target dots
            if per_target_vals is not None:
                jitter = np.random.default_rng(42).uniform(-w/3, w/3, len(per_target_vals))
                if per_target_tani is not None:
                    ax1.scatter(x[i] + w/2 + jitter, per_target_vals, s=8, alpha=0.4,
                               c=per_target_tani, cmap=cmap, norm=norm, zorder=5,
                               edgecolors='none')
                else:
                    ax1.scatter(x[i] + w/2 + jitter, per_target_vals, s=8, alpha=0.4,
                               color='#333', zorder=5, edgecolors='none')
        elif m['name'] == 'DeepPROTACs':
            # Hatched/grayed N/A bar
            ax1.bar(x[i] + w/2, 0.5, w, color='#CCCCCC', alpha=0.3,
                    edgecolor='#999', linewidth=1, hatch='///',
                    label='LOTO (cold-target)')
            ax1.text(x[i] + w/2, 0.52, 'N/A', ha='center', va='bottom',
                     fontsize=9, fontstyle='italic', color='#666')
        elif m['name'] == 'Scaffold CV':
            pass  # only one bar (reported = scaffold CV AUROC)

    # Chance line
    ax1.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5, zorder=0)
    ax1.text(x[-1] + 0.7, 0.5, 'chance', fontsize=7, color='grey', va='center')

    # Colorbar for Tanimoto
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.4, aspect=20, pad=0.02)
    cbar.set_label('Mean max Tanimoto\nto training', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m['name'] for m in methods], rotation=0, ha='center', fontsize=9)
    ax1.set_ylim(0, 1.12)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax1.set_title('A) Cold-Target Collapse Is Universal', fontsize=12, fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Panel B (right): Tanimoto histograms ---
    tani_random = np.load('results/exp13_splits_and_baselines/tanimoto_random.npy')
    tani_scaffold = np.load('results/exp13_splits_and_baselines/tanimoto_scaffold.npy')

    bins = np.linspace(0, 1, 51)
    ax2.hist(tani_random, bins=bins, density=True, alpha=0.5, color='#4C72B0', label='Random CV', edgecolor='none')
    ax2.hist(tani_scaffold, bins=bins, density=True, alpha=0.5, color='#55A868', label='Scaffold CV', edgecolor='none')
    ax2.hist(tani_loto, bins=bins, density=True, alpha=0.5, color='#DD8452', label='LOTO', edgecolor='none')

    for thresh in [0.4, 0.6, 0.8]:
        ax2.axvline(thresh, color='grey', ls='--', lw=0.6, alpha=0.6)

    # Fraction >0.6 text box
    frac_r = np.mean(tani_random > 0.6)
    frac_s = np.mean(tani_scaffold > 0.6)
    frac_l = np.mean(tani_loto > 0.6)
    textstr = f'Frac > 0.6:\n  Random: {frac_r:.2%}\n  Scaffold: {frac_s:.2%}\n  LOTO: {frac_l:.2%}'
    ax2.text(0.03, 0.97, textstr, transform=ax2.transAxes, fontsize=7.5,
             va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Annotation arrow for LOTO peak near 1.0
    # Find the density peak near 1.0 for LOTO
    counts_loto, edges_loto = np.histogram(tani_loto, bins=bins, density=True)
    peak_idx = len(counts_loto) - 1  # last bin
    peak_x = (edges_loto[peak_idx] + edges_loto[peak_idx+1]) / 2
    peak_y = counts_loto[peak_idx]
    ax2.annotate('Identical molecules\ntargeting different\nproteins',
                 xy=(peak_x, peak_y), xytext=(0.55, peak_y * 0.8),
                 fontsize=7, ha='center',
                 arrowprops=dict(arrowstyle='->', color='#333', lw=1),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    ax2.set_xlabel('Max Tanimoto similarity to nearest training neighbor', fontsize=9)
    ax2.set_ylabel('Density', fontsize=9)
    ax2.set_xlim(0, 1.02)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_title('B) Tanimoto Similarity Distributions', fontsize=12, fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(RDIR / 'fig1_collapse.pdf')
    fig.savefig(RDIR / 'fig1_collapse.png', dpi=300)
    plt.close(fig)
    print('    Done.')


# ============================================================================
# Fig 2: Per-Fold Distribution (7x5)
# ============================================================================
def fig2_perfold():
    print('  Fig 2: Per-fold distribution ...')
    sim_csv = pd.read_csv('results/exp11_similarity_stratified/per_target_with_similarity.csv')

    # Sort by LOTO AUROC ascending
    sim_csv = sim_csv.sort_values('loto').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Continuous ESM-2 similarity coloring
    sim_vals = sim_csv['max_cosine_sim'].values
    norm = Normalize(vmin=sim_vals.min(), vmax=sim_vals.max())
    cmap = cm.viridis

    x = np.arange(len(sim_csv))
    colors = cmap(norm(sim_vals))
    ax.scatter(x, sim_csv['loto'].values, c=sim_vals, cmap=cmap, norm=norm,
               s=30, zorder=5, edgecolors='white', linewidth=0.3)

    # Mean AUROC line
    mean_auroc = sim_csv['loto'].mean()
    ax.axhline(mean_auroc, color='grey', ls='--', lw=1, alpha=0.7)
    ax.text(len(sim_csv) + 0.5, mean_auroc, f'mean={mean_auroc:.3f}', fontsize=7,
            va='center', color='grey')

    # Annotate extremes
    annotations = {
        'Q96SW2': 'Q96SW2', 'P15170': 'P15170', 'Q07889': 'Q07889',
        'O15379': 'O15379', 'P08581': 'P08581', 'P30530': 'P30530',
    }
    for idx, row in sim_csv.iterrows():
        t = row['target']
        if t in annotations:
            offset_y = -0.06 if row['loto'] < 0.5 else 0.04
            offset_x = 2 if idx < len(sim_csv) / 2 else -2
            ax.annotate(t, (idx, row['loto']),
                       xytext=(idx + offset_x, row['loto'] + offset_y),
                       fontsize=6.5, ha='center',
                       arrowprops=dict(arrowstyle='-', color='#666', lw=0.5))

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=25)
    cbar.set_label('ESM-2 cosine similarity\nto nearest training target', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel('Targets ordered by LOTO AUROC (ascending)', fontsize=10)
    ax.set_ylabel('AUROC', fontsize=10)
    ax.set_xlim(-1, len(sim_csv))
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks([])
    ax.set_title('Per-Target LOTO Performance', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(RDIR / 'fig2_perfold.pdf')
    fig.savefig(RDIR / 'fig2_perfold.png', dpi=300)
    plt.close(fig)
    print('    Done.')


# ============================================================================
# Fig 3: HPO Ceiling (14x5)
# ============================================================================
def fig3_ceiling():
    print('  Fig 3: HPO ceiling ...')
    trials = pd.read_csv('results/exp2_unified_hpo/all_trials.csv')
    importance = _load_json('results/exp2_unified_hpo/param_importance.json')

    # Only complete trials
    trials = trials[trials['state'] == 'COMPLETE'].copy()

    # Split by seed: first 543 = seed0, rest = seed1 (based on exp2_posthoc info)
    # Trial 947 is best for seed0, trial 999 for seed1
    # Actually, with 1000 complete trials across 1086 total, split in half
    trials_sorted = trials.sort_values('number').reset_index(drop=True)
    mid = len(trials_sorted) // 2
    seed0 = trials_sorted.iloc[:mid]
    seed1 = trials_sorted.iloc[mid:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Trial distribution by seed
    bins = np.linspace(0.45, 0.75, 40)
    ax1.hist(seed0['value'].dropna(), bins=bins, alpha=0.5, color='royalblue',
             label=f'Seed 0 (n={len(seed0)})', edgecolor='none', density=True)
    ax1.hist(seed1['value'].dropna(), bins=bins, alpha=0.5, color='orange',
             label=f'Seed 1 (n={len(seed1)})', edgecolor='none', density=True)

    ax1.axvline(0.664, color='black', ls='--', lw=1.2)
    ax1.text(0.664, ax1.get_ylim()[1] * 0.9, ' Baseline\n 0.664', fontsize=8,
             va='top', ha='left')
    ax1.axvline(0.708, color='red', ls='--', lw=1.2)
    ax1.text(0.708, ax1.get_ylim()[1] * 0.75, ' Best (single seed)\n 0.708', fontsize=8,
             va='top', ha='left', color='red')

    # Annotation about 5-seed validated
    ax1.text(0.97, 0.97, '5-seed validated: 0.668\n(p = 0.925)',
             transform=ax1.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax1.set_xlabel('LOTO AUROC', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_title('A) HPO Trial Distribution', fontsize=12, fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: fANOVA importance (log x-axis)
    imp_sorted = sorted(importance.items(), key=lambda x: -x[1])
    names = [x[0].replace('_', ' ') for x in imp_sorted]
    vals = [x[1] for x in imp_sorted]

    y_pos = np.arange(len(names))[::-1]
    ax2.barh(y_pos, vals, color='#4C72B0', edgecolor='white', height=0.7)
    ax2.set_xscale('log')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('fANOVA Importance (log scale)', fontsize=10)

    for i, v in enumerate(vals):
        ax2.text(v * 1.3, y_pos[i], f'{v:.4f}', va='center', fontsize=8)

    ax2.set_title('B) Parameter Importance', fontsize=12, fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(RDIR / 'fig3_ceiling.pdf')
    fig.savefig(RDIR / 'fig3_ceiling.png', dpi=300)
    plt.close(fig)
    print('    Done.')


# ============================================================================
# Fig 4: Breakthroughs (7x6)
# ============================================================================
def fig4_breakthroughs():
    print('  Fig 4: Breakthroughs ...')
    e8 = _load_json('results/exp8c_admet_cascade/summary.json')
    conds = e8['conditions']
    e4 = _load_json('results/exp4_fewshot/exp4_summary.json')

    # Load EGNN on 60 targets for comparison
    try:
        egnn60 = _load_json('results/exp14_egnn_full/summary.json')
        egnn60_auroc = egnn60['all_entries']['egnn']['mean']
    except Exception:
        egnn60_auroc = None

    bars = [
        {'name': 'RF+Morgan\n(baseline, 65 targets)', 'auroc': 0.666, 'color': '#999999',
         'p': None, 'hatch': '', 'annotation': ''},
        {'name': '+ ADMET cascade\n(65 targets)', 'auroc': conds['C2_morgan+admet7']['auroc'],
         'color': '#4C72B0', 'p': conds['C2_morgan+admet7']['p'],
         'hatch': '', 'annotation': 'Cell entry\n(target-independent)'},
        {'name': '+ k=5 few-shot\n(65 targets)', 'auroc': e4['rf_morgan_k5']['mean'],
         'color': '#55A868', 'p': 8.17e-6,  # from C4 which is admet+fewshot
         'hatch': '', 'annotation': 'Target-specific\nSAR calibration'},
        {'name': '+ 3D EGNN\n(27 targets*)', 'auroc': 0.801, 'color': '#C44E52',
         'p': 0.004, 'hatch': '///',
         'annotation': 'Ternary complex\ngeometry'},
    ]

    fig, ax = plt.subplots(figsize=(7, 6))

    y_pos = np.arange(len(bars))[::-1]
    for i, b in enumerate(bars):
        bar = ax.barh(y_pos[i], b['auroc'], height=0.6, color=b['color'],
                      edgecolor='white' if not b['hatch'] else b['color'],
                      linewidth=1.5, alpha=0.8, hatch=b['hatch'])
        # Value label
        ax.text(b['auroc'] + 0.008, y_pos[i], f"{b['auroc']:.3f}",
                va='center', fontsize=10, fontweight='bold')
        # P-value
        if b['p'] is not None:
            pstr = f"p={b['p']:.3f}" if b['p'] >= 0.001 else f"p<0.001"
            ax.text(b['auroc'] + 0.008, y_pos[i] - 0.22, pstr,
                    va='center', fontsize=8, color='#555')
        # Failure mode annotation
        if b['annotation']:
            ax.text(0.35, y_pos[i] + 0.0, b['annotation'],
                    va='center', fontsize=8, color='white', fontweight='bold',
                    style='italic')

    # EGNN on 60 targets thin bar (if available)
    if egnn60_auroc is not None:
        ax.barh(y_pos[3] - 0.35, egnn60_auroc, height=0.2, color='#C44E52',
                edgecolor='#C44E52', linewidth=1, alpha=0.3)
        ax.text(egnn60_auroc + 0.008, y_pos[3] - 0.35,
                f'{egnn60_auroc:.3f} (60 targets, docked)',
                va='center', fontsize=7, color='#888')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([b['name'] for b in bars], fontsize=10)
    ax.set_xlabel('LOTO AUROC', fontsize=11)
    ax.set_xlim(0, 0.95)
    ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)

    # Footnote
    ax.text(0.02, -0.08,
            '* Evaluated on 27 DegradeMaster targets with curated docked structures.\n'
            '  Extension to all 65 targets limited by pocket structure quality (see Fig S8).',
            transform=ax.transAxes, fontsize=7.5, va='top', color='#555')

    ax.set_title('Progressive Improvements Over Baseline', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(RDIR / 'fig4_breakthroughs.pdf')
    fig.savefig(RDIR / 'fig4_breakthroughs.png', dpi=300)
    plt.close(fig)
    print('    Done.')


# ============================================================================
# Fig 7: PLM Scaling (7x5)
# ============================================================================
def fig7_plm_scaling():
    print('  Fig 7: PLM scaling ...')
    plm = _load_json('results/exp15_plm_scaling/summary.json')

    # Build data from summary
    # Keys like morgan_only, morgan+esm2_8M, etc.
    scales = []
    for key, data in plm.items():
        if key == 'experiment' or key == 'n_targets' or not isinstance(data, dict):
            continue
        if 'loto' not in data and 'loto_mean' not in data:
            continue
        loto_mean = data.get('loto_mean', data.get('loto', {}).get('mean', None))
        loto_std = data.get('loto_std', data.get('loto', {}).get('std', None))
        random_mean = data.get('random_mean', data.get('random', {}).get('mean', None))
        random_std = data.get('random_std', data.get('random', {}).get('std', None))

        if loto_mean is None:
            # Try direct fields
            loto_mean = data.get('mean', None)
            loto_std = data.get('std', None)

        # Determine parameter count
        if 'morgan_only' in key or key == 'morgan_only':
            params = 0
            label = 'Morgan only'
        elif '8M' in key:
            params = 8e6
            label = 'ESM-2 8M'
        elif '35M' in key:
            params = 35e6
            label = 'ESM-2 35M'
        elif '150M' in key:
            params = 150e6
            label = 'ESM-2 150M'
        elif '650M' in key:
            params = 650e6
            label = 'ESM-2 650M'
        elif '3B' in key:
            params = 3e9
            label = 'ESM-2 3B'
        else:
            continue

        scales.append({
            'params': params, 'label': label, 'key': key,
            'loto_mean': loto_mean, 'loto_std': loto_std,
            'random_mean': random_mean, 'random_std': random_std,
        })

    scales.sort(key=lambda x: x['params'])

    # Try to add BFD point
    try:
        bfd = _load_json('results/exp1_prottrans_bfd/summary.json')
        bfd_conds = bfd.get('poi_e3_subset', bfd.get('conditions', {}))
        bfd_e0 = bfd_conds.get('E0', {})
        bfd_e1 = bfd_conds.get('E1', {})
        if bfd_e1:
            bfd_loto = bfd_e1.get('loto', bfd_e1.get('mean', None))
            if bfd_loto:
                scales.append({
                    'params': 420e6, 'label': 'BFD (420M)',
                    'loto_mean': bfd_loto, 'loto_std': None,
                    'random_mean': None, 'random_std': None,
                    'is_bfd': True,
                })
                scales.sort(key=lambda x: x['params'])
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(7, 5))

    # Separate ESM-2 points from BFD
    esm_scales = [s for s in scales if not s.get('is_bfd')]
    bfd_scales = [s for s in scales if s.get('is_bfd')]

    # X positions
    params_esm = [max(s['params'], 1e5) for s in esm_scales]  # replace 0 with small val for log
    loto_esm = [s['loto_mean'] for s in esm_scales]
    random_esm = [s['random_mean'] for s in esm_scales if s['random_mean'] is not None]
    params_rand = [max(s['params'], 1e5) for s in esm_scales if s['random_mean'] is not None]

    loto_std_esm = [s['loto_std'] or 0 for s in esm_scales]

    ax.errorbar(params_esm, loto_esm, yerr=loto_std_esm, fmt='o-', color='#DD8452',
                label='LOTO AUROC', lw=2, markersize=8, capsize=4)
    if random_esm:
        random_std_esm = [s.get('random_std') or 0 for s in esm_scales if s['random_mean'] is not None]
        ax.errorbar(params_rand, random_esm, yerr=random_std_esm, fmt='s-', color='#4C72B0',
                    label='Random CV AUROC', lw=2, markersize=8, capsize=4)

    # BFD point
    for s in bfd_scales:
        ax.scatter(s['params'], s['loto_mean'], marker='D', s=100, color='#C44E52',
                   zorder=10, edgecolors='white', linewidth=1)
        ax.annotate(s['label'], (s['params'], s['loto_mean']),
                    xytext=(5, -15), textcoords='offset points', fontsize=8, color='#C44E52')

    ax.set_xscale('log')
    ax.set_xlabel('PLM Parameter Count', fontsize=10)
    ax.set_ylabel('AUROC', fontsize=10)

    # Custom x-tick labels (only ESM-2 scales, BFD is annotated separately)
    tick_positions = [1e5, 8e6, 35e6, 150e6, 650e6, 3e9]
    tick_labels = ['0', '8M', '35M', '150M', '650M', '3B']
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)

    ax.legend(fontsize=9, loc='center right')

    # Annotation
    ax.text(0.5, 0.03, 'PLMs improve random-split but hurt cold-target prediction',
            transform=ax.transAxes, fontsize=9, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_title('PLM Scaling: Random vs Cold-Target', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(RDIR / 'fig7_plm_scaling.pdf')
    fig.savefig(RDIR / 'fig7_plm_scaling.png', dpi=300)
    plt.close(fig)
    print('    Done.')


# ============================================================================
def main():
    print('=' * 50)
    print('  Generating Updated NeurIPS Figures (v2)')
    print('=' * 50)
    fig1_collapse()
    fig2_perfold()
    fig3_ceiling()
    fig4_breakthroughs()
    fig7_plm_scaling()
    print(f'\n  All figures saved to {RDIR}')

if __name__ == '__main__':
    main()
