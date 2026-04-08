#!/usr/bin/env python3
"""Generate all NeurIPS publication figures for PROTAC-PLM-Bench v6.

Changes from v5:
- Fig1A: TWO versions (tanimoto + simple coloring), random dots use same viridis Tanimoto,
         add sample/target counts to x-tick labels
- Fig1B: legend upper left, remove fraction text box, add mean to legend labels
- Fig2: add linear regression line with confidence band
- Fig3A/3B: unchanged
- Fig4: use seed std (not cross-target), remove asterisks/footnote
- Fig5: fix baseline text y-position, MAML no error bars
- Fig6: unchanged
- Fig7: circles instead of squares, handle per-seed std properly
- Supplementary: unchanged from v5

Run from /workspace: python scripts/generate_figures_neurips_v6.py
"""
import json, warnings, sys, sqlite3
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
import seaborn as sns

# ── Output dir ────────────────────────────────────────────────────────────────
RDIR = Path('/workspace/results/neurips_figures_v6')
RDIR.mkdir(parents=True, exist_ok=True)

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

def _load_json(path):
    with open(path) as f:
        return json.load(f)

def _save(fig, name):
    fig.savefig(RDIR / f'{name}.pdf')
    fig.savefig(RDIR / f'{name}.png', dpi=300)
    plt.close(fig)
    print(f'  {name}')

def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Compute per-target Tanimoto from exp13 (single consistent source) ────────
def _compute_per_target_tanimoto():
    """Return dict: target_uniprot -> mean(max_tanimoto) under LOTO."""
    tani = np.load('results/exp13_splits_and_baselines/tanimoto_loto.npy')
    dataset = pd.read_csv('results/ext30_data_expansion/exp1_merged_dataset.csv')
    rf_csv = pd.read_csv('results/exp1_baseline_replication/A_rf_morgan.csv')
    eligible_targets = rf_csv['target'].tolist()
    sub = dataset[dataset['target_uniprot'].isin(eligible_targets)].copy()
    result = {}
    idx = 0
    for target in eligible_targets:
        mask = sub['target_uniprot'] == target
        n_entries = mask.sum()
        if idx + n_entries <= len(tani):
            target_tani = tani[idx:idx + n_entries]
            result[target] = float(np.mean(target_tani))
            idx += n_entries
        else:
            result[target] = 0.5
    return result


def _get_sample_target_counts():
    """Load sample and target counts from experiment JSONs for each method."""
    counts = {}
    # DeepPROTACs
    dp_path = Path('results/exp20_deepprotacs_aligned/summary.json')
    if dp_path.exists():
        dp = _load_json(dp_path)
        n_tgt = dp.get('loto', {}).get('n_targets', 26)
        pt = dp.get('loto', {}).get('per_target', [])
        n_samples = sum(t.get('n', 0) for t in pt) if pt else 852
        counts['DeepPROTACs'] = (n_samples, n_tgt)
    else:
        counts['DeepPROTACs'] = (852, 26)
    # Ribes: uses the full merged dataset for LOTO over 65 targets
    rf_csv = pd.read_csv('results/exp1_baseline_replication/A_rf_morgan.csv')
    n_rf_targets = len(rf_csv)
    n_rf_samples = int(rf_csv['n'].sum()) if 'n' in rf_csv.columns else 9428
    counts['Ribes et al.'] = (n_rf_samples, n_rf_targets)
    # PROTAC-STAN
    stan_path = Path('results/exp1_stan_loto/summary.json')
    if stan_path.exists():
        stan = _load_json(stan_path)
        n_stan_tgt = stan.get('n', 27)
        # Try to get sample count from CSV
        stan_csv = Path('results/exp1_stan_loto/E0_rf_morgan_subset.csv')
        if stan_csv.exists():
            sdf = pd.read_csv(stan_csv)
            n_stan_samples = int(sdf['n'].sum()) if 'n' in sdf.columns else len(sdf) * 30
        else:
            n_stan_samples = 8754
        counts['PROTAC-STAN'] = (n_stan_samples, n_stan_tgt)
    else:
        counts['PROTAC-STAN'] = (8754, 27)
    # DegradeMaster
    dm_table = Path('results/exp1_degrademaster_loto/target_class_table.csv')
    if dm_table.exists():
        dm_df = pd.read_csv(dm_table)
        n_dm_tgt = len(dm_df)
        n_dm_samples = int(dm_df['n'].sum()) if 'n' in dm_df.columns else 966
    else:
        dm_path = Path('results/exp1_degrademaster_loto/summary.json')
        dm = _load_json(dm_path) if dm_path.exists() else {}
        n_dm_tgt = dm.get('n', 27)
        n_dm_samples = 966
    counts['DegradeMaster'] = (n_dm_samples, n_dm_tgt)
    # RF+Morgan and kNN: same dataset
    dataset = pd.read_csv('results/ext30_data_expansion/exp1_merged_dataset.csv')
    n_total = len(dataset)
    counts['RF+Morgan'] = (n_total, n_rf_targets)
    counts['kNN (k=5)'] = (n_total, n_rf_targets)
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1A: COLLAPSE — version A (tanimoto coloring) and version B (simple)
# ═══════════════════════════════════════════════════════════════════════════════
def _fig1a_core(ax_a, methods, tani_map, within_map, counts, version='tanimoto'):
    """Draw Fig1A panel. version='tanimoto' or 'simple'."""
    n_methods = len(methods)
    gap_pos = 4
    positions = list(range(gap_pos)) + [gap_pos + 0.6, gap_pos + 1.6]
    bar_w = 0.18
    offsets = [-1.5*bar_w, -0.5*bar_w, 0.5*bar_w, 1.5*bar_w]
    colors_bar = [CB_LBLUE, CB_BLUE, '#FFB347', CB_ORANGE]
    alphas_bar = [1.0, 0.25, 1.0, 0.25]
    sc_handle = None

    for i, m in enumerate(methods):
        x = positions[i]
        vals = [m['rep_rs'], m['repl_rs'], m['rep_cs'], m['repl_cs']]
        errs = [None, m.get('repl_rs_err'), None, m.get('repl_cs_err')]
        for j, (v, err) in enumerate(zip(vals, errs)):
            bx = x + offsets[j]
            if v is None:
                ax_a.bar(bx, 0.5, bar_w, color='#DDDDDD', edgecolor='#999999',
                         linewidth=0.8, hatch='///', alpha=0.6)
                ax_a.text(bx, 0.25, 'N/A', ha='center', va='center',
                          fontsize=7, color='#666666', rotation=90)
                continue
            fill_alpha = alphas_bar[j]
            if fill_alpha == 1.0:
                ax_a.bar(bx, v, bar_w, color=colors_bar[j], edgecolor='white', linewidth=0.5)
            else:
                ax_a.bar(bx, v, bar_w, color=colors_bar[j], alpha=fill_alpha,
                         edgecolor=colors_bar[j], linewidth=0.8)
            if err and err > 0:
                ax_a.errorbar(bx, v, yerr=err, fmt='none', ecolor='black',
                              capsize=2, capthick=0.8, linewidth=0.8)
            ax_a.text(bx, v + 0.015, f'{v:.3f}', ha='center', va='bottom',
                      fontsize=8, fontweight='bold')

            # Per-target LOTO dots (j==3)
            if j == 3 and m.get('per_target_cs') is not None:
                pt_df = m['per_target_cs']
                aurocs = pt_df['auroc'].values
                rng = np.random.default_rng(42)
                jitter = rng.uniform(-bar_w*0.35, bar_w*0.35, len(aurocs))
                if version == 'tanimoto':
                    if 'target' in pt_df.columns:
                        colors_dot = [tani_map.get(t, 0.5) for t in pt_df['target']]
                    else:
                        colors_dot = [0.5] * len(aurocs)
                    sc_handle = ax_a.scatter(bx + jitter, aurocs, s=8, alpha=1.0,
                                              c=colors_dot, cmap='viridis', vmin=0, vmax=1,
                                              edgecolors='none', zorder=5, marker='o')
                else:
                    ax_a.scatter(bx + jitter, aurocs, s=8, alpha=0.7,
                                 color='darkorange', edgecolors='none', zorder=5, marker='o')

            # Per-target RANDOM-SPLIT dots (j==1) — v6: same Tanimoto coloring or simple
            if j == 1 and m.get('per_target_cs') is not None:
                pt_df = m['per_target_cs']
                if 'target' in pt_df.columns:
                    wt_aurocs = [within_map.get(t, np.nan) for t in pt_df['target']]
                    valid_mask = [not np.isnan(a) for a in wt_aurocs]
                    wt_arr = np.array([a for a, v in zip(wt_aurocs, valid_mask) if v])
                    valid_targets = [t for t, v in zip(pt_df['target'], valid_mask) if v]
                    if len(wt_arr) > 0:
                        rng2 = np.random.default_rng(99)
                        wt_jitter = rng2.uniform(-bar_w*0.35, bar_w*0.35, len(wt_arr))
                        if version == 'tanimoto':
                            colors_wt = [tani_map.get(t, 0.5) for t in valid_targets]
                            sc_wt = ax_a.scatter(bx + wt_jitter, wt_arr, s=8, alpha=0.8,
                                                  c=colors_wt, cmap='viridis', vmin=0, vmax=1,
                                                  edgecolors='none', zorder=4, marker='o')
                            if sc_handle is None:
                                sc_handle = sc_wt
                        else:
                            ax_a.scatter(bx + wt_jitter, wt_arr, s=8, alpha=0.5,
                                         color='royalblue', edgecolors='none', zorder=4, marker='o')

            elif j == 3 and m['name'] == 'PROTAC-STAN':
                ax_a.text(bx, 0.35, 'per-target\nN/A', ha='center', va='center',
                          fontsize=6, color='#888888', rotation=90)

    # Colorbar only for tanimoto version
    if version == 'tanimoto':
        if sc_handle is not None:
            cbar = plt.gcf().colorbar(sc_handle, ax=ax_a, fraction=0.02, pad=0.01, aspect=30)
        else:
            cbar = plt.gcf().colorbar(plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(0, 1)),
                                ax=ax_a, fraction=0.02, pad=0.01, aspect=30)
        cbar.set_label('Max Tanimoto to\nnearest training', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    ax_a.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)

    # v6: Multi-line x-tick labels with sample/target counts
    tick_labels = []
    for m in methods:
        name = m['name']
        n_samp, n_tgt = counts.get(name, (0, 0))
        tick_labels.append(f"{name}\nsamples={n_samp}\ntargets={n_tgt}")

    ax_a.set_xticks(positions)
    ax_a.set_xticklabels(tick_labels, fontsize=9.5)
    # Make the samples/targets lines smaller
    for label in ax_a.get_xticklabels():
        label.set_fontsize(9.5)
        # Use a custom text with mixed sizes via set_fontproperties won't work,
        # so we handle it differently below
    # Actually override with smaller sub-labels using ax.text
    ax_a.set_xticklabels([m['name'] for m in methods], fontsize=9.5)
    # Add sample/target counts as separate text below
    for idx_m, m in enumerate(methods):
        name = m['name']
        n_samp, n_tgt = counts.get(name, (0, 0))
        xpos = positions[idx_m]
        ax_a.text(xpos, -0.08, f'samples={n_samp}\ntargets={n_tgt}',
                  ha='center', va='top', fontsize=7, color='#555555',
                  transform=ax_a.get_xaxis_transform())

    ax_a.set_ylabel('AUROC')
    ax_a.set_ylim(0, 1.12)
    _despine(ax_a)
    ax_a.set_title('A  Reported vs. Replicated Performance', fontweight='bold', loc='left')

    # Legend
    if version == 'tanimoto':
        handles = [
            mpatches.Patch(facecolor=CB_LBLUE, edgecolor='white', label='Reported RS'),
            mpatches.Patch(facecolor=CB_BLUE, alpha=0.25, edgecolor=CB_BLUE, label='Replicated RS'),
            mpatches.Patch(facecolor='#FFB347', edgecolor='white', label='Reported CS'),
            mpatches.Patch(facecolor=CB_ORANGE, alpha=0.25, edgecolor=CB_ORANGE, label='Replicated CS'),
            mpatches.Patch(facecolor='#DDDDDD', edgecolor='#999999', hatch='///', label='N/A'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=CB_GREY, markersize=5, label='LOTO per-target'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=CB_LBLUE, markersize=5,
                   alpha=0.5, label='Random per-target'),
        ]
    else:
        handles = [
            mpatches.Patch(facecolor=CB_LBLUE, edgecolor='white', label='Reported RS'),
            mpatches.Patch(facecolor=CB_BLUE, alpha=0.25, edgecolor=CB_BLUE, label='Replicated RS'),
            mpatches.Patch(facecolor='#FFB347', edgecolor='white', label='Reported CS'),
            mpatches.Patch(facecolor=CB_ORANGE, alpha=0.25, edgecolor=CB_ORANGE, label='Replicated CS'),
            mpatches.Patch(facecolor='#DDDDDD', edgecolor='#999999', hatch='///', label='N/A'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=5, label='LOTO per-target'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markersize=5,
                   alpha=0.5, label='Random per-target'),
        ]
    ax_a.legend(handles=handles, loc='upper right', fontsize=7, ncol=1)

    return sc_handle


def _fig1b(ax_b):
    """Draw Fig1B panel — Tanimoto histograms with mean in legend."""
    tani_random = np.load('results/exp13_splits_and_baselines/tanimoto_random.npy')
    tani_scaffold = np.load('results/exp13_splits_and_baselines/tanimoto_scaffold.npy')
    tani_loto = np.load('results/exp13_splits_and_baselines/tanimoto_loto.npy')

    # Load mean values from JSON
    tani_dist = _load_json('results/exp13_splits_and_baselines/tanimoto_distributions.json')
    mean_random = tani_dist['random_cv']['mean']
    mean_scaffold = tani_dist['scaffold_cv']['mean']
    mean_loto = tani_dist['loto']['mean']

    bins = np.linspace(0, 1, 51)
    ax_b.hist(tani_random, bins=bins, alpha=0.5, color=CB_BLUE,
              label=f'Random CV\nmean: {mean_random:.3f}', density=True)
    ax_b.hist(tani_scaffold, bins=bins, alpha=0.5, color=CB_GREEN,
              label=f'Scaffold CV\nmean: {mean_scaffold:.3f}', density=True)
    ax_b.hist(tani_loto, bins=bins, alpha=0.5, color=CB_ORANGE,
              label=f'LOTO\nmean: {mean_loto:.3f}', density=True)

    for thresh in [0.4, 0.6, 0.8]:
        ax_b.axvline(thresh, color='grey', ls='--', lw=0.8, alpha=0.6)

    # v6: NO fraction text box — removed

    ax_b.set_xlabel('Max Tanimoto to Nearest Training Neighbor')
    ax_b.set_ylabel('Density')
    # v6: Legend upper left
    ax_b.legend(fontsize=8, loc='upper left')
    _despine(ax_b)
    ax_b.set_title('B  Train\u2013Test Molecular Overlap', fontweight='bold', loc='left')


def _load_methods_data():
    """Load all method data for Fig1A."""
    full = _load_json('results/exp1_full_replication/full_summary.json')
    dp_path = Path('results/exp20_deepprotacs_aligned/summary.json')
    dp = _load_json(dp_path) if dp_path.exists() else None
    knn = _load_json('results/exp13_splits_and_baselines/knn_baseline.json')
    baseline_summary = _load_json('/workspace/summary.json')
    rf_csv = pd.read_csv('results/exp1_baseline_replication/A_rf_morgan.csv')
    tani_map = _compute_per_target_tanimoto()
    within_csv = pd.read_csv('results/exp_supp_ceiling/within_target.csv')
    within_map = dict(zip(within_csv['target'], within_csv['auroc']))

    methods = []
    # 1. DeepPROTACs
    if dp:
        dp_pt = pd.DataFrame(dp['loto']['per_target']).rename(columns={'mean_auroc': 'auroc'})
        methods.append({
            'name': 'DeepPROTACs', 'rep_rs': 0.847,
            'repl_rs': dp['random_split']['linker']['mean'],
            'repl_rs_err': dp['random_split']['linker']['std'],
            'rep_cs': None, 'has_rep_cs': False,
            'repl_cs': dp['loto']['mean_auroc'],
            'repl_cs_err': dp['loto']['std_auroc'],
            'per_target_cs': dp_pt,
        })
    else:
        methods.append({
            'name': 'DeepPROTACs', 'rep_rs': 0.847,
            'repl_rs': None, 'repl_rs_err': None,
            'rep_cs': None, 'has_rep_cs': False,
            'repl_cs': None, 'repl_cs_err': None,
            'per_target_cs': None,
        })
    # 2. Ribes
    ribes_csv = pd.read_csv('results/exp1_full_replication/loto_C_ribes_style.csv')
    methods.append({
        'name': 'Ribes et al.', 'rep_rs': 0.865,
        'repl_rs': full['conditions']['C_ribes_style']['random'],
        'repl_rs_err': None,
        'rep_cs': 0.604, 'has_rep_cs': True,
        'repl_cs': full['conditions']['C_ribes_style']['loto'],
        'repl_cs_err': None,
        'per_target_cs': ribes_csv if 'auroc' in ribes_csv.columns else None,
    })
    # 3. PROTAC-STAN
    methods.append({
        'name': 'PROTAC-STAN', 'rep_rs': 0.883,
        'repl_rs': full['stan_own_data']['their_split'],
        'repl_rs_err': None,
        'rep_cs': None, 'has_rep_cs': False,
        'repl_cs': full['stan_own_data']['loto_rf'],
        'repl_cs_err': None,
        'per_target_cs': None,
    })
    # 4. DegradeMaster
    dm = _load_json('results/exp1_degrademaster_loto/per_target_analysis.json')
    dm_per = dm['per_target']
    dm_targets = list(dm_per.keys())
    dm_aurocs = [dm_per[t]['egnn'] for t in dm_targets]
    dm_df = pd.DataFrame({'target': dm_targets, 'auroc': dm_aurocs})
    methods.append({
        'name': 'DegradeMaster', 'rep_rs': 0.883,
        'repl_rs': 0.830,
        'repl_rs_err': None,
        'rep_cs': None, 'has_rep_cs': False,
        'repl_cs': np.mean(dm_aurocs),
        'repl_cs_err': np.std(dm_aurocs) / np.sqrt(len(dm_aurocs)),
        'per_target_cs': dm_df,
    })
    # 5. RF+Morgan
    seed_std = baseline_summary['baseline']['std']
    methods.append({
        'name': 'RF+Morgan', 'rep_rs': None,
        'repl_rs': full['conditions']['A_rf_morgan']['random'],
        'repl_rs_err': seed_std,
        'rep_cs': None, 'has_rep_cs': False,
        'repl_cs': full['conditions']['A_rf_morgan']['loto'],
        'repl_cs_err': seed_std,
        'per_target_cs': rf_csv,
    })
    # 6. kNN k=5
    knn_loto = knn['loto']['5']
    knn_random = knn['random_cv']['5']
    knn_targets = rf_csv['target'].tolist()
    knn_df = pd.DataFrame({'target': knn_targets, 'auroc': knn_loto['per_target_aurocs']})
    methods.append({
        'name': 'kNN (k=5)', 'rep_rs': None,
        'repl_rs': knn_random['mean_auroc'],
        'repl_rs_err': None,
        'rep_cs': None, 'has_rep_cs': False,
        'repl_cs': knn_loto['mean_auroc'],
        'repl_cs_err': None,
        'per_target_cs': knn_df,
    })
    return methods, tani_map, within_map


def fig1_collapse():
    """Generate both versions of Fig1."""
    methods, tani_map, within_map = _load_methods_data()
    counts = _get_sample_target_counts()

    # Version A: Tanimoto coloring
    fig_a, (ax_a1, ax_b1) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={'width_ratios': [2.2, 1]})
    _fig1a_core(ax_a1, methods, tani_map, within_map, counts, version='tanimoto')
    _fig1b(ax_b1)
    fig_a.tight_layout()
    _save(fig_a, 'fig1a_collapse_tanimoto')

    # Version B: Simple coloring
    fig_b, (ax_a2, ax_b2) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={'width_ratios': [2.2, 1]})
    _fig1a_core(ax_a2, methods, tani_map, within_map, counts, version='simple')
    _fig1b(ax_b2)
    fig_b.tight_layout()
    _save(fig_b, 'fig1a_collapse_simple')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2: PER-FOLD DISTRIBUTION — v6: add regression line with confidence band
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_perfold():
    rf = pd.read_csv('results/exp1_baseline_replication/A_rf_morgan.csv')
    sim = pd.read_csv('results/exp11_similarity_stratified/per_target_with_similarity.csv')
    df = rf.merge(sim[['target', 'max_cosine_sim']], on='target', how='left')

    fig, ax = plt.subplots(figsize=(7, 5))
    x_data = df['max_cosine_sim'].values
    y_data = df['auroc'].values

    ax.scatter(x_data, y_data,
               color='steelblue', s=30, edgecolors='white', linewidth=0.3, zorder=5)

    # v6: Linear regression with prediction interval band
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

    x_line = np.linspace(x_clean.min(), x_clean.max(), 200)
    y_line = slope * x_line + intercept

    # Prediction interval
    n = len(x_clean)
    x_mean = np.mean(x_clean)
    residuals = y_clean - (slope * x_clean + intercept)
    s_res = np.sqrt(np.sum(residuals**2) / (n - 2))
    # Standard error of prediction at each x point
    se_pred = s_res * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / np.sum((x_clean - x_mean)**2))
    t_crit = stats.t.ppf(0.975, n - 2)
    y_upper = y_line + t_crit * se_pred
    y_lower = y_line - t_crit * se_pred

    ax.plot(x_line, y_line, '-', color='steelblue', lw=1.5, alpha=0.8)
    ax.fill_between(x_line, y_lower, y_upper, alpha=0.15, color='steelblue')

    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('ESM-2 Cosine Similarity to Nearest Training Target')
    ax.set_ylabel('LOTO AUROC')
    ax.set_ylim(-0.05, 1.1)
    _despine(ax)
    ax.set_title('Per-Target LOTO Performance vs. Target Similarity')
    fig.tight_layout()
    _save(fig, 'fig2_perfold')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3: HPO CEILING (14×5, 2 panels) — unchanged from v5
# ═══════════════════════════════════════════════════════════════════════════════
def _load_optuna_values(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('''
        SELECT t.number, tv.value
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY t.number
    ''', conn)
    conn.close()
    return df

def fig3_hpo():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    db0 = Path('results/exp2_unified_hpo/exp2_hpo_seed0.db')
    db1 = Path('results/exp2_unified_hpo/exp2_hpo_seed1.db')
    seed0 = _load_optuna_values(db0)
    seed1 = _load_optuna_values(db1)
    explore_idx = 750
    s0_explore = seed0[seed0['number'] < explore_idx]
    s0_exploit = seed0[seed0['number'] >= explore_idx]
    s1_explore = seed1[seed1['number'] < explore_idx]
    s1_exploit = seed1[seed1['number'] >= explore_idx]
    bins = np.linspace(0.3, 0.85, 40)
    ax_a.hist(s0_explore['value'], bins=bins, alpha=0.5, color='royalblue',
              label=f'Seed 0 explore (n={len(s0_explore)})',
              edgecolor='royalblue', linewidth=0.5, histtype='stepfilled')
    if len(s0_exploit) > 0:
        ax_a.hist(s0_exploit['value'], bins=bins, alpha=0.4, color='#87CEEB',
                  label=f'Seed 0 exploit (n={len(s0_exploit)})',
                  edgecolor='#87CEEB', linewidth=0.5, histtype='stepfilled')
    ax_a.hist(s1_explore['value'], bins=bins, alpha=0.5, color='darkorange',
              label=f'Seed 1 explore (n={len(s1_explore)})',
              edgecolor='darkorange', linewidth=0.5, histtype='stepfilled')
    if len(s1_exploit) > 0:
        ax_a.hist(s1_exploit['value'], bins=bins, alpha=0.4, color='#FFD699',
                  label=f'Seed 1 exploit (n={len(s1_exploit)})',
                  edgecolor='#FFD699', linewidth=0.5, histtype='stepfilled')
    ax_a.axvline(0.664, color='black', ls='--', lw=1.2, label='Baseline (0.664)')
    ax_a.axvline(0.708, color=CB_RED, ls='--', lw=1.2, label='Best trial (0.708)')
    ax_a.text(0.708, ax_a.get_ylim()[1] * 0.95, 'Best (single seed): 0.708',
              fontsize=8, ha='left', va='top', color=CB_RED)
    ax_a.text(0.708, ax_a.get_ylim()[1] * 0.85, '5-seed validated: 0.668 (p=0.925)',
              fontsize=8, ha='left', va='top', color='#555')
    ax_a.set_xlabel('LOTO AUROC')
    ax_a.set_ylabel('Count')
    ax_a.legend(fontsize=7, loc='upper left')
    _despine(ax_a)
    ax_a.set_title('A  HPO Trial Distribution', fontweight='bold', loc='left')

    imp = _load_json('results/exp2_unified_hpo/param_importance.json')
    sorted_imp = sorted(imp.items(), key=lambda x: x[1])
    names = [x[0].replace('_', ' ') for x in sorted_imp]
    vals = [x[1] for x in sorted_imp]
    ax_b.barh(range(len(names)), vals, color=CB_BLUE, edgecolor='white',
              linewidth=0.5, alpha=0.8)
    ax_b.set_yticks(range(len(names)))
    ax_b.set_yticklabels(names, fontsize=8)
    ax_b.set_xscale('log')
    ax_b.set_xlabel('fANOVA Importance')
    for i, v in enumerate(vals):
        ax_b.text(v * 1.3, i, f'{v:.4f}', va='center', fontsize=7)
    _despine(ax_b)
    ax_b.set_title('B  Parameter Importance (fANOVA)', fontweight='bold', loc='left')
    fig.tight_layout()
    _save(fig, 'fig3_hpo_ceiling')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4: BREAKTHROUGHS — v6: seed std, remove asterisks/footnote
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_breakthroughs():
    fig, ax = plt.subplots(figsize=(7, 6))

    # Seed-level std values
    baseline_summary = _load_json('/workspace/summary.json')
    rf_seed_std = baseline_summary['baseline']['std']  # 0.0036

    # ADMET cascade: check for seed-level std
    admet_path = Path('results/exp8c_admet_cascade/summary.json')
    admet_seed_std = 0  # default
    if admet_path.exists():
        admet_s = _load_json(admet_path)
        # Look for seed-level std in the summary
        if 'C2_morgan+admet7' in admet_s:
            admet_seed_std = admet_s['C2_morgan+admet7'].get('seed_std', admet_s['C2_morgan+admet7'].get('std', 0))
        elif 'conditions' in admet_s:
            for k, v in admet_s['conditions'].items():
                if 'admet' in k.lower():
                    admet_seed_std = v.get('seed_std', v.get('std', 0))
                    break

    # Few-shot: check for seed-level std
    fewshot_summary = _load_json('results/exp4_fewshot/exp4_summary.json')
    fewshot_seed_std = fewshot_summary.get('rf_morgan_k5', {}).get('seed_std', 0)
    # If no seed_std, use std/sqrt(n) as SEM or just use cross-target std with note
    if fewshot_seed_std == 0:
        # cross-target std is available; for paper we want seed std
        # Check if seed_means is available
        seed_means = fewshot_summary.get('rf_morgan_k5', {}).get('seed_means', None)
        if seed_means:
            fewshot_seed_std = np.std(seed_means)
        else:
            fewshot_seed_std = 0  # no seed data -> no error bar

    # EGNN 27 targets: single architecture, check for seed std
    egnn_summary = _load_json('results/exp1_degrademaster_loto/summary.json')
    egnn_seed_std = egnn_summary.get('seed_std', 0)
    if egnn_seed_std == 0:
        seed_means = egnn_summary.get('seed_means', None)
        if seed_means:
            egnn_seed_std = np.std(seed_means)

    # EGNN Smina 60 targets: single seed, no error bar
    egnn_smina_seed_std = 0  # single seed -> no error bar

    entries = [
        ('RF+Morgan\n(65 targets)', 0.666, CB_GREY, False, None, rf_seed_std),
        ('+ADMET\n(65 targets)', 0.687, CB_BLUE, False, 'p=0.014', admet_seed_std),
        ('+k=5 few-shot\n(65 targets)', 0.700, CB_GREEN, False, 'p<0.001', fewshot_seed_std),
        # v6: removed asterisk from EGNN labels
        ('+3D EGNN\n(27 targets)', 0.801, CB_RED, True, 'p=0.004', egnn_seed_std),
        ('+3D EGNN Smina\n(60 targets, approx. pockets)', 0.542, '#FF9999', True, None, egnn_smina_seed_std),
    ]

    y_pos = np.arange(len(entries))[::-1]
    for i, (name, val, color, hatched, ptext, std) in enumerate(entries):
        hatch = '///' if hatched else None
        ax.barh(y_pos[i], val, color=color, edgecolor='white' if not hatched else color,
                linewidth=0.8, height=0.6, hatch=hatch, alpha=0.85)
        if std and std > 0:
            ax.errorbar(val, y_pos[i], xerr=std, fmt='none', ecolor='black',
                        capsize=3, capthick=0.8, linewidth=0.8, zorder=6)
        offset_x = std + 0.015 if std and std > 0 else 0.01
        ax.text(val + offset_x, y_pos[i], f'{val:.3f}',
                va='center', fontsize=10, fontweight='bold')
        if ptext:
            ax.text(val + offset_x, y_pos[i] - 0.25,
                    ptext, va='top', fontsize=8, color='#555', style='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e[0] for e in entries], fontsize=9)
    ax.set_xlabel('LOTO AUROC')
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)

    # v6: removed asterisk footnote; just note error bars meaning
    ax.text(0.02, -0.06, 'Hatched = structural input | Error bars = seed std',
            transform=ax.transAxes, fontsize=8, style='italic', color='#666')

    _despine(ax)
    ax.set_title('Incremental Improvements Over Baseline')
    fig.tight_layout()
    _save(fig, 'fig4_breakthroughs')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5: FEW-SHOT — v6: fix baseline y, MAML no error bars
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_fewshot():
    e4 = _load_json('results/exp4_fewshot/exp4_summary.json')
    maml_ext = _load_json('results/exp20_maml_full_k/summary.json')

    fig, ax = plt.subplots(figsize=(7, 5))

    # RF+Morgan — with error bars (seed std available)
    ks_rf = [0, 1, 3, 5, 10]
    auroc_rf = [e4[f'rf_morgan_k{k}']['mean'] for k in ks_rf]
    std_rf = [e4[f'rf_morgan_k{k}'].get('std', 0) / np.sqrt(e4[f'rf_morgan_k{k}']['n'])
              for k in ks_rf]
    ax.errorbar(ks_rf, auroc_rf, yerr=std_rf, fmt='o-', color=CB_BLUE,
                label='RF + Morgan', lw=2, markersize=7, capsize=3)

    # RF+meta — with error bars
    ks_meta = [1, 3, 5, 10]
    auroc_meta = [e4[f'rf_meta_k{k}']['mean'] for k in ks_meta]
    std_meta = [e4[f'rf_meta_k{k}'].get('std', 0) / np.sqrt(e4[f'rf_meta_k{k}']['n'])
                for k in ks_meta]
    ax.errorbar(ks_meta, auroc_meta, yerr=std_meta, fmt='s-', color=CB_GREEN,
                label='RF + meta', lw=2, markersize=7, capsize=3)

    # MAML+Morgan — v6: NO error bars (no per-seed std available)
    ks_maml_m = [1, 3, 5, 10]
    auroc_maml_m = [
        maml_ext['conditions']['maml_morgan_k1']['mean_auroc'],
        maml_ext['conditions']['maml_morgan_k3']['mean_auroc'],
        e4['maml_morgan_k5']['mean'],
        maml_ext['conditions']['maml_morgan_k10']['mean_auroc'],
    ]
    ax.plot(ks_maml_m, auroc_maml_m, 'D--', color=CB_RED,
            label='MAML + Morgan', lw=1.5, markersize=6, alpha=0.8)

    # MAML+meta — v6: NO error bars
    auroc_maml_meta = [
        maml_ext['conditions']['maml_meta_k1']['mean_auroc'],
        maml_ext['conditions']['maml_meta_k3']['mean_auroc'],
        e4['maml_meta_k5']['mean'],
        maml_ext['conditions']['maml_meta_k10']['mean_auroc'],
    ]
    ax.plot(ks_maml_m, auroc_maml_meta, 'D--', color=CB_ORANGE,
            label='MAML + meta', lw=1.5, markersize=6, alpha=0.8)

    # v6: Baseline line at correct y=0.666, text just above
    ax.axhline(0.666, color=CB_GREY, ls='--', lw=1, alpha=0.5)
    ax.text(5.0, 0.666 + 0.008, 'LOTO Baseline', fontsize=8, va='bottom', ha='center', color=CB_GREY)

    ax.set_xlabel('k (target-specific examples)')
    ax.set_ylabel('LOTO AUROC')
    ax.set_xlim(-0.5, 11.5)
    ax.legend(loc='lower right', fontsize=9)
    _despine(ax)
    ax.set_title('Few-Shot Target Adaptation')
    fig.tight_layout()
    _save(fig, 'fig5_fewshot')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6: EGNN SCATTER — unchanged from v5
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_egnn_scatter():
    dm = _load_json('results/exp1_degrademaster_loto/per_target_analysis.json')
    pt = dm['per_target']
    fig, ax = plt.subplots(figsize=(7, 6))
    targets = list(pt.keys())
    rf_vals = [pt[t]['rf_merged'] if pt[t]['rf_merged'] is not None else pt[t]['rf_dm']
               for t in targets]
    egnn_vals = [pt[t]['egnn'] for t in targets]
    dm_csv_path = Path('results/exp1_degrademaster_loto/target_class_table.csv')
    n_map = {}
    if dm_csv_path.exists():
        dm_csv = pd.read_csv(dm_csv_path)
        if 'target' in dm_csv.columns:
            for col in ['n', 'n_entries', 'count']:
                if col in dm_csv.columns:
                    n_map = dict(zip(dm_csv['target'], dm_csv[col]))
                    break
    if not n_map:
        rf_csv = pd.read_csv('results/exp1_baseline_replication/A_rf_morgan.csv')
        n_map = dict(zip(rf_csv['target'], rf_csv['n']))
    sizes = [n_map.get(t, 50) or 50 for t in targets]
    log_sizes = np.log10(np.array([float(s) if s is not None else 50.0 for s in sizes]) + 1)
    sc = ax.scatter(rf_vals, egnn_vals, c=log_sizes, cmap='viridis',
                     s=50, edgecolors='white', linewidth=0.5, zorder=5)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    deltas = [(t, pt[t]['egnn'] - (pt[t]['rf_merged'] if pt[t]['rf_merged'] is not None else pt[t]['rf_dm']))
              for t in targets]
    deltas_sorted = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)
    for t, d in deltas_sorted[:4]:
        rf_v = pt[t]['rf_merged'] if pt[t]['rf_merged'] is not None else pt[t]['rf_dm']
        eg_v = pt[t]['egnn']
        ax.annotate(t, (rf_v, eg_v), fontsize=7, ha='left',
                    xytext=(5, 5), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', lw=0.5, color='#555'))
    rho = dm['correlation_rf_vs_delta']['rho']
    wins = dm['egnn_wins']
    n_tgt = len(targets)
    ax.text(0.05, 0.95, f'{wins}/{n_tgt}: EGNN > RF\n$\\rho$={rho:.3f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('log$_{10}$(n\\_entries)', fontsize=9)
    ax.set_xlabel('RF + Morgan AUROC')
    ax.set_ylabel('EGNN AUROC')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect('equal')
    _despine(ax)
    ax.set_title('3D Structure Helps Where 2D Fails')
    fig.tight_layout()
    _save(fig, 'fig6_egnn_scatter')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7: PLM SCALING — v6: circles, handle per-seed std
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_plm_scaling():
    plm = _load_json('results/exp15_plm_scaling/summary.json')
    fig, ax = plt.subplots(figsize=(7, 5))

    plm_configs = [
        ('morgan+esm2_8M', 8e6),
        ('morgan+esm2_35M', 35e6),
        ('morgan+esm2_150M', 150e6),
        ('morgan+esm2_650M', 650e6),
        ('morgan+esm2_3B', 3e9),
    ]

    params = [p[1] for p in plm_configs]
    random_vals = [plm[p[0]]['random_mean'] for p in plm_configs]
    loto_vals = [plm[p[0]]['loto_mean'] for p in plm_configs]

    # v6: Check for per-seed std. The loto_std values are per-fold (cross-target ~0.17-0.20),
    # which is too large for meaningful error bars on this plot.
    # Check if seed_means or seed_std fields exist
    has_seed_std = False
    loto_seed_stds = []
    random_seed_stds = []
    for config_name, _ in plm_configs:
        entry = plm[config_name]
        if 'seed_std' in entry or 'seed_means' in entry:
            has_seed_std = True
            if 'seed_std' in entry:
                loto_seed_stds.append(entry['seed_std'])
            elif 'seed_means' in entry:
                loto_seed_stds.append(np.std(entry['seed_means']))
            random_seed_stds.append(entry.get('random_seed_std', entry.get('random_std', 0)))
        else:
            loto_seed_stds.append(entry.get('loto_std', 0))
            random_seed_stds.append(entry.get('random_std', 0))

    if not has_seed_std:
        # Per-fold std is too large (~0.19). Show NO error bars for LOTO.
        # Random CV std is small (~0.005), keep those.
        print('  [fig7] NOTE: Only per-fold (cross-target) std available for LOTO (~0.19).')
        print('         Showing no LOTO error bars. To get per-seed std, re-run with 5 seeds.')
        print('         Estimated time: ~30 min/condition x 5 conditions = 2.5 hours.')
        # v6: circles (marker='o') with alpha=0.8
        ax.plot(params, random_vals, 'o-', color=CB_BLUE, label='Random CV',
                markersize=7, lw=0.8, alpha=0.8)
        # Add small random_std error bars for random CV
        random_err = [plm[p[0]]['random_std'] for p in plm_configs]
        ax.errorbar(params, random_vals, yerr=random_err, fmt='none', ecolor=CB_BLUE,
                    capsize=3, lw=0.8, alpha=0.5)
        # LOTO: no error bars
        ax.plot(params, loto_vals, 'o-', color=CB_ORANGE, label='LOTO',
                markersize=7, lw=0.8, alpha=0.8)
    else:
        ax.errorbar(params, random_vals, yerr=random_seed_stds, fmt='o', color=CB_BLUE,
                    label='Random CV', markersize=7, capsize=3, lw=0.8, alpha=0.8)
        ax.errorbar(params, loto_vals, yerr=loto_seed_stds, fmt='o', color=CB_ORANGE,
                    label='LOTO', markersize=7, capsize=3, lw=0.8, alpha=0.8)
        ax.plot(params, random_vals, '-', color=CB_BLUE, lw=0.5, alpha=0.4)
        ax.plot(params, loto_vals, '-', color=CB_ORANGE, lw=0.5, alpha=0.4)

    bfd_path = Path('results/exp1_prottrans_bfd/summary.json')
    if bfd_path.exists():
        bfd = _load_json(bfd_path)
        bfd_loto = bfd.get('poi_e3_subset', {}).get('E2', {}).get('loto',
                   bfd.get('poi_e3_subset', {}).get('E1', {}).get('loto'))
        if bfd_loto:
            ax.scatter([420e6], [bfd_loto], marker='*', s=120, color=CB_GREEN,
                       zorder=10, label='BFD (420M)')

    ax.set_xscale('log')
    ax.set_xlabel('PLM Parameters')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    _despine(ax)
    ax.set_title('PLM Scaling: Larger Models Widen the Gap')
    fig.tight_layout()
    _save(fig, 'fig7_plm_scaling')


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES — unchanged from v5
# ═══════════════════════════════════════════════════════════════════════════════

def figS1_fragments():
    frags = _load_json('results/exp_supp_fragments/summary.json')
    names_keys = [('Full\nPROTAC', 'full_protac'), ('Warhead', 'warhead'),
                  ('E3 ligand', 'e3_ligand'), ('Linker', 'linker'), ('Anchor', 'anchor')]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(names_keys))
    w = 0.35
    random_v = [frags[k]['random'] for _, k in names_keys]
    loto_v = [frags[k]['loto'] for _, k in names_keys]
    ax.bar(x - w/2, random_v, w, label='Random split', color=CB_BLUE, edgecolor='white')
    ax.bar(x + w/2, loto_v, w, label='LOTO', color=CB_ORANGE, edgecolor='white')
    for i in range(len(names_keys)):
        ax.text(x[i] - w/2, random_v[i] + 0.015, f'{random_v[i]:.3f}',
                ha='center', fontsize=8, fontweight='bold')
        ax.text(x[i] + w/2, loto_v[i] + 0.015, f'{loto_v[i]:.3f}',
                ha='center', fontsize=8, fontweight='bold')
    ax.set_ylabel('AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in names_keys])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    _despine(ax)
    ax.set_title('Fragment Signal Decomposition')
    fig.tight_layout()
    _save(fig, 'figS1_fragments')


def figS2_threshold():
    ms_path = Path('results/exp20_threshold_multiseed/summary.json')
    old_path = Path('results/exp_supp_threshold/threshold_sensitivity.csv')
    fig, ax = plt.subplots(figsize=(7, 5))
    if ms_path.exists():
        ms = _load_json(ms_path)
        results_list = ms['results']
        entries = []
        for item in results_list:
            label = item.get('condition', item.get('label', ''))
            loto = item['mean']
            std = item.get('std_seed', item.get('std', 0))
            n_tgt = item.get('n_targets', 0)
            ctype = item.get('condition_type', '')
            if 'DC50<5000' in label or 'DC50<10000' in label:
                continue
            if not ctype:
                if 'DC50' in label and 'Dmax' not in label and 'OR' not in label:
                    ctype = 'DC50-only'
                elif 'Dmax' in label and 'DC50' not in label and 'OR' not in label:
                    ctype = 'Dmax-only'
                else:
                    ctype = 'Combined'
            if 'DC50' in ctype and 'Dmax' not in ctype:
                ctype = 'DC50'
            elif 'Dmax' in ctype and 'DC50' not in ctype:
                ctype = 'Dmax'
            else:
                ctype = 'Combined'
            entries.append((label, loto, std, n_tgt, ctype))
        dc50_colors = ['#B3D9FF', '#4D94FF', '#003399']
        dmax_colors = ['#FFD9B3', '#FF944D', '#CC5500']
        comb_colors = ['#B3FFB3', '#4DCC4D', '#006600']
        markers = {'DC50': 'o', 'Dmax': 's', 'Combined': '^'}
        dc50_entries = sorted([e for e in entries if e[4] == 'DC50'], key=lambda x: x[1])
        dmax_entries = sorted([e for e in entries if e[4] == 'Dmax'], key=lambda x: x[1])
        comb_entries = sorted([e for e in entries if e[4] == 'Combined'], key=lambda x: x[1])
        x_pos = 0
        for group, colors, group_entries in [
            ('DC50', dc50_colors, dc50_entries),
            ('Dmax', dmax_colors, dmax_entries),
            ('Combined', comb_colors, comb_entries),
        ]:
            for i, (label, loto, std, n_tgt, ctype) in enumerate(group_entries):
                ci = min(i, len(colors) - 1)
                ax.errorbar(x_pos, loto, yerr=std, fmt=markers[ctype],
                            color=colors[ci], markersize=8, capsize=3,
                            markeredgecolor='white', markeredgewidth=0.5)
                ax.text(x_pos, loto - 0.04, label.replace('_', '\n'),
                        ha='center', fontsize=6, rotation=45)
                x_pos += 1
            x_pos += 0.5
    else:
        df = pd.read_csv(old_path)
        df = df[df['loto'].notna()].reset_index(drop=True)
        for i, row in df.iterrows():
            label = row['threshold']
            if 'DC50' in label and 'OR' not in label and 'AND' not in label:
                color, marker = CB_BLUE, 'o'
            elif 'Dmax' in label:
                color, marker = CB_ORANGE, 's'
            else:
                color, marker = CB_GREEN, '^'
            ax.scatter(i, row['loto'], c=color, marker=marker, s=60,
                       edgecolors='white', linewidth=0.5, zorder=5)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.set_ylabel('LOTO AUROC')
    ax.set_xlabel('Threshold Condition')
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CB_BLUE,
               markersize=8, label='DC50-only'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CB_ORANGE,
               markersize=8, label='Dmax-only'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CB_GREEN,
               markersize=8, label='Combined'),
    ]
    ax.legend(handles=handles, fontsize=9)
    _despine(ax)
    ax.set_title('Threshold Sensitivity')
    fig.tight_layout()
    _save(fig, 'figS2_threshold')


def figS3_temporal():
    temp = _load_json('results/exp_supp_temporal/summary.json')
    fig, ax = plt.subplots(figsize=(7, 5))
    years = sorted([k for k in temp if k.isdigit()])
    novel_years = sorted([k for k in temp if 'novel' in k])
    all_auc = [temp[y]['auroc'] for y in years]
    novel_auc = [temp[y]['auroc'] for y in novel_years]
    novel_labels = [int(y.replace('_novel_only', '')) for y in novel_years]
    int_years = [int(y) for y in years]
    train_sizes = [temp[y]['n_train'] for y in years]
    ax.plot(int_years, all_auc, 'o-', color=CB_BLUE, label='All test targets', lw=2, markersize=7)
    ax.plot(novel_labels, novel_auc, 's--', color=CB_ORANGE, label='Novel targets only', lw=2, markersize=7)
    ax.axhline(0.5, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(0.666, color=CB_GREY, ls='--', lw=1, alpha=0.5)
    ax.text(max(int_years) + 0.2, 0.666, 'LOTO', fontsize=8, color=CB_GREY)
    ax.set_xlabel('Publication Year (train on earlier, test on this year)')
    ax.set_ylabel('AUROC')
    ax.legend(fontsize=9)
    _despine(ax)
    ax2 = ax.twinx()
    ax2.bar(int_years, train_sizes, alpha=0.15, color=CB_BLUE, width=0.4)
    ax2.set_ylabel('Training size', color=CB_GREY, fontsize=9)
    ax2.tick_params(axis='y', labelcolor=CB_GREY, labelsize=8)
    ax2.spines['top'].set_visible(False)
    ax.set_title('Temporal Split')
    fig.tight_layout()
    _save(fig, 'figS3_temporal')


def figS4_cross_dataset():
    path = Path('results/exp13_cross_dataset/summary.json')
    if not path.exists():
        print('  figS4_cross_dataset: SKIPPED (no data)')
        return
    data = _load_json(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    res = data['results']
    conditions = [
        ('TPDDB\nCV', res['within_tpddb_cv']['mean_auroc'], CB_BLUE),
        ('PROTAC-8K\nCV', res['within_protac8k_cv']['mean_auroc'], CB_BLUE),
        ('TPDDB\u2192\nPROTAC-8K', res['train_tpddb_test_protac8k']['mean_auroc'], CB_ORANGE),
        ('PROTAC-8K\u2192\nTPDDB', res['train_protac8k_test_tpddb']['mean_auroc'], CB_ORANGE),
    ]
    x = np.arange(len(conditions))
    vals = [c[1] for c in conditions]
    colors = [c[2] for c in conditions]
    names = [c[0] for c in conditions]
    ax.bar(x, vals, color=colors, edgecolor='white', width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    _despine(ax)
    ax.set_title('Cross-Dataset Generalization')
    fig.tight_layout()
    _save(fig, 'figS4_cross_dataset')


def figS5_docking():
    path = Path('results/exp11_docking_confound/summary.json')
    if not path.exists():
        print('  figS5_docking: SKIPPED (no data)')
        return
    data = _load_json(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    conditions = [
        ('Morgan\nonly', data['conditions']['C0_morgan_only']['mean_auroc'], CB_BLUE),
        ('Morgan+\nDocking', data['conditions']['C1_morgan_plus_dock']['mean_auroc'], CB_GREEN),
        ('Docking\nonly', data['conditions']['C2_dock_only']['mean_auroc'], CB_ORANGE),
        ('Dock+\nADMET', data['conditions']['C3_dock_plus_admet']['mean_auroc'], CB_PURPLE),
    ]
    x = np.arange(len(conditions))
    vals = [c[1] for c in conditions]
    colors = [c[2] for c in conditions]
    names = [c[0] for c in conditions]
    ax.bar(x, vals, color=colors, edgecolor='white', width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(0.801, color=CB_RED, ls='--', lw=1, alpha=0.7)
    ax.text(len(conditions) - 0.5, 0.81, 'DegradeMaster EGNN (0.801)',
            fontsize=8, color=CB_RED, ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('LOTO AUROC (27 targets)')
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    _despine(ax)
    ax.set_title('Docking Features Do Not Explain 3D Advantage')
    fig.tight_layout()
    _save(fig, 'figS5_docking')


def figS6_feature_corr():
    path = Path('results/exp17_feature_importance/summary.json')
    csv_path = Path('results/exp17_feature_importance/importances.csv')
    if not path.exists() or not csv_path.exists():
        print('  figS6_feature_corr: SKIPPED (no data)')
        return
    data = _load_json(path)
    imp = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(imp['importance_random'], imp['importance_loto'],
               s=15, alpha=0.5, color=CB_BLUE, edgecolors='none')
    max_v = max(imp['importance_random'].max(), imp['importance_loto'].max())
    ax.plot([0, max_v], [0, max_v], 'k--', lw=0.8, alpha=0.5)
    rho = data['rank_correlation']['spearman_rho']
    ax.text(0.05, 0.95, f'Spearman $\\rho$={rho:.4f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Feature Importance (Random CV)')
    ax.set_ylabel('Feature Importance (LOTO)')
    _despine(ax)
    ax.set_title('Feature Importance: Random vs LOTO')
    fig.tight_layout()
    _save(fig, 'figS6_feature_corr')


def figS7_scaffold():
    scaffold = _load_json('results/exp13_splits_and_baselines/scaffold_split.json')
    tani_dist = _load_json('results/exp13_splits_and_baselines/tanimoto_distributions.json')
    fig, ax = plt.subplots(figsize=(7, 5))
    conditions = [
        ('Random CV', scaffold['comparison']['random_cv_auroc'], CB_BLUE),
        ('Scaffold CV', scaffold['comparison']['scaffold_cv_auroc'], CB_GREEN),
        ('LOTO', scaffold['comparison']['loto_auroc'], CB_ORANGE),
    ]
    x = np.arange(len(conditions))
    vals = [c[1] for c in conditions]
    colors = [c[2] for c in conditions]
    names = [c[0] for c in conditions]
    ax.bar(x, vals, color=colors, edgecolor='white', width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    for i, split in enumerate(['random_cv', 'scaffold_cv', 'loto']):
        frac = tani_dist[split]['frac_gt_0.6']
        ax.text(i, 0.1, f'>{0.6} Tani:\n{frac:.1%}', ha='center', fontsize=8,
                color='white', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    _despine(ax)
    ax.set_title('Scaffold Split Does Not Approximate Cold-Target')
    fig.tight_layout()
    _save(fig, 'figS7_scaffold')


def figS8_egnn_smina():
    egnn_csv = Path('results/exp14_egnn_full/per_target.csv')
    morgan_csv = Path('results/exp14_egnn_full/per_target_morgan_rf.csv')
    if not egnn_csv.exists() or not morgan_csv.exists():
        print('  figS8_egnn_smina: SKIPPED (no data)')
        return
    egnn = pd.read_csv(egnn_csv)
    morgan = pd.read_csv(morgan_csv)
    df = egnn.merge(morgan, on='target', suffixes=('_egnn', '_morgan'))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df['auroc_morgan'], df['auroc_egnn'], s=30, color=CB_BLUE,
               edgecolors='white', linewidth=0.5, alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    summary = _load_json('results/exp14_egnn_full/summary.json')
    egnn_mean = summary['all_entries']['egnn']['mean']
    morgan_mean = summary['all_entries']['morgan']['mean']
    n_tgt = summary['all_entries']['egnn']['n']
    pct_gt20 = summary['docking_quality']['pct_warhead_gt20A']
    ax.text(0.05, 0.95,
            f'EGNN: {egnn_mean:.3f}\nMorgan: {morgan_mean:.3f}\n'
            f'n={n_tgt} targets\n{pct_gt20:.0f}% entries >20\u00C5\nfrom pocket',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Morgan RF AUROC')
    ax.set_ylabel('EGNN AUROC (Smina pockets)')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect('equal')
    _despine(ax)
    ax.set_title('EGNN with Approximate (Smina) Pockets')
    fig.tight_layout()
    _save(fig, 'figS8_egnn_smina')


def figS9_protein_family():
    path = Path('results/exp19_protein_families/summary.json')
    if not path.exists():
        print('  figS9_protein_family: SKIPPED (no data)')
        return
    data = _load_json(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    per_target = data.get('per_target', [])
    if not per_target:
        print('  figS9_protein_family: SKIPPED (no per_target data)')
        return
    family_sizes = []
    aurocs = []
    families = []
    for t in per_target:
        fs = t.get('family_size_in_training', t.get('n_same_family', 0))
        auroc = t.get('loto_auroc', t.get('auroc', None))
        family = t.get('family', 'Unknown')
        if auroc is not None:
            family_sizes.append(fs)
            aurocs.append(auroc)
            families.append(family)
    unique_families = sorted(set(families))
    n_fam = len(unique_families)
    cmap = plt.cm.get_cmap('tab20', n_fam)
    fam_to_color = {f: cmap(i) for i, f in enumerate(unique_families)}
    colors = [fam_to_color[f] for f in families]
    ax.scatter(family_sizes, aurocs, c=colors, s=40, edgecolors='white',
               linewidth=0.5, alpha=0.8, zorder=5)
    fs_arr = np.array(family_sizes, dtype=float)
    au_arr = np.array(aurocs)
    z = np.polyfit(fs_arr, au_arr, 1)
    x_line = np.linspace(0, max(fs_arr), 100)
    ax.plot(x_line, np.polyval(z, x_line), '--', color=CB_RED, lw=1, alpha=0.7)
    rho = data['family_size_vs_loto_auroc']['spearman_r']
    p = data['family_size_vs_loto_auroc']['spearman_p']
    ax.text(0.05, 0.05, f'Spearman $\\rho$={rho:.3f}\np={p:.1e}',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('N same-family targets in training')
    ax.set_ylabel('LOTO AUROC')
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    _despine(ax)
    major = [f for f in unique_families if families.count(f) >= 3]
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=fam_to_color[f],
                       markersize=7, label=f) for f in major[:8]]
    if handles:
        ax.legend(handles=handles, fontsize=7, loc='upper left', ncol=2)
    ax.set_title('Protein Family Size vs. LOTO Performance')
    fig.tight_layout()
    _save(fig, 'figS9_protein_family')


def figS10_calibration():
    path = Path('results/exp17_calibration/summary.json')
    if not path.exists():
        print('  figS10_calibration: SKIPPED (no data)')
        return
    data = _load_json(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    rf = data['results']['rf_uncalibrated']
    cal = rf['calibration_curve']
    pred = [b['mean_pred'] for b in cal]
    true = [b['mean_true'] for b in cal]
    counts = [b['count'] for b in cal]
    sizes = [max(20, c / 30) for c in counts]
    ax.scatter(pred, true, s=sizes, color=CB_BLUE, edgecolors='white',
               linewidth=0.5, zorder=5, alpha=0.8)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
    ax.plot(pred, true, '-', color=CB_BLUE, lw=1, alpha=0.5)
    ece = rf['ece']
    ax.text(0.05, 0.95, f'ECE = {ece:.4f}\nBrier = {rf["brier"]:.4f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='lower right')
    _despine(ax)
    ax.set_title('RF Calibration (LOTO Predictions)')
    fig.tight_layout()
    _save(fig, 'figS10_calibration')


def figS11_learning_curve():
    path = Path('results/exp17_learning_curve/summary.json')
    if not path.exists():
        print('  figS11_learning_curve: SKIPPED (no data)')
        return
    data = _load_json(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    curve = data['learning_curve']
    fracs = sorted(curve.keys(), key=float)
    frac_vals = [float(f) for f in fracs]
    means = [curve[f]['mean_auroc'] for f in fracs]
    stds = [curve[f]['std_auroc'] for f in fracs]
    ax.errorbar(frac_vals, means, yerr=stds, fmt='o-', color=CB_ORANGE,
                lw=2, markersize=8, capsize=4, label='LOTO AUROC')
    reg = data.get('regression', {})
    slope = reg.get('slope', 0)
    r2 = reg.get('r_squared', 0)
    ax.text(0.05, 0.95, f'slope={slope:.4f}\n$R^2$={r2:.3f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Training Data Fraction')
    ax.set_ylabel('LOTO AUROC')
    ax.set_xlim(0, 1.1)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    _despine(ax)
    ax.set_title('Learning Curve')
    fig.tight_layout()
    _save(fig, 'figS11_learning_curve')


def figS12_seed_variance():
    path = Path('results/exp18_seed_variance/summary.json')
    if not path.exists():
        print('  figS12_seed_variance: SKIPPED (no data)')
        return
    data = _load_json(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    seeds_list = data['per_seed']
    seed_ids = [s['seed'] for s in seeds_list]
    means = [s['mean_auroc'] for s in seeds_list]
    x_pos = np.arange(len(seed_ids))
    ax.scatter(x_pos, means, s=80, color=CB_BLUE, edgecolors='white',
               linewidth=0.5, zorder=5)
    overall_mean = data['overall_mean']
    overall_std = data['overall_std']
    ax.axhline(overall_mean, color=CB_GREY, ls='--', lw=1)
    ax.axhspan(overall_mean - overall_std, overall_mean + overall_std,
               alpha=0.15, color=CB_BLUE)
    ax.text(len(seed_ids) - 0.5, overall_mean + overall_std + 0.002,
            f'Mean={overall_mean:.4f}\nStd={overall_std:.4f}',
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Seed {s}' for s in seed_ids], fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Mean LOTO AUROC')
    _despine(ax)
    ax.set_title('10-Seed Variance')
    fig.tight_layout()
    _save(fig, 'figS12_seed_variance')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('  Generating NeurIPS Figures v6')
    print('=' * 60)

    figs = [
        ('Fig 1 (2 versions)', fig1_collapse),
        ('Fig 2', fig2_perfold),
        ('Fig 3', fig3_hpo),
        ('Fig 4', fig4_breakthroughs),
        ('Fig 5', fig5_fewshot),
        ('Fig 6', fig6_egnn_scatter),
        ('Fig 7', fig7_plm_scaling),
        ('Fig S1', figS1_fragments),
        ('Fig S2', figS2_threshold),
        ('Fig S3', figS3_temporal),
        ('Fig S4', figS4_cross_dataset),
        ('Fig S5', figS5_docking),
        ('Fig S6', figS6_feature_corr),
        ('Fig S7', figS7_scaffold),
        ('Fig S8', figS8_egnn_smina),
        ('Fig S9', figS9_protein_family),
        ('Fig S10', figS10_calibration),
        ('Fig S11', figS11_learning_curve),
        ('Fig S12', figS12_seed_variance),
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

    print('\n' + '=' * 60)
    print(f'  {"Figure":<25} {"Status":<50}')
    print('-' * 60)
    for name, status in results:
        print(f'  {name:<25} {status:<50}')
    print('=' * 60)

    pdfs = list(RDIR.glob('*.pdf'))
    pngs = list(RDIR.glob('*.png'))
    print(f'\n  Output: {len(pdfs)} PDFs, {len(pngs)} PNGs in {RDIR}')

    # Print change summary
    print('\n' + '=' * 60)
    print('  v6 CHANGE SUMMARY')
    print('=' * 60)
    print('  Fig 1A: TWO versions (tanimoto + simple). Random dots now use same')
    print('          Tanimoto coloring as LOTO (version A) or royalblue/darkorange')
    print('          (version B). Sample/target counts added below method names.')
    print('  Fig 1B: Legend moved to upper left. Fraction text box removed.')
    print('          Mean max-similarity added to legend labels.')
    print('  Fig 2:  Linear regression line + 95% prediction interval band added.')
    print('  Fig 3:  Unchanged from v5.')
    print('  Fig 4:  Error bars now show seed std (not cross-target std).')
    print('          Asterisks and footnote removed from EGNN labels.')
    print('  Fig 5:  Baseline text at correct y=0.666+0.008. MAML lines have')
    print('          no error bars (no per-seed std available).')
    print('  Fig 6:  Unchanged from v5.')
    print('  Fig 7:  Markers changed from squares to circles (alpha=0.8).')
    print('          LOTO error bars removed (only per-fold std ~0.19 available).')
    print('  Supplementary: All unchanged from v5.')


if __name__ == '__main__':
    main()
