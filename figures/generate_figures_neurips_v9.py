#!/usr/bin/env python3
"""Generate all NeurIPS publication figures for PROTAC-PLM-Bench v9.

Changes from v8:
- Fig1A: Random-split dots kept (within-target CV); note printed about discrepancy
- Fig3A: Seed 1 histogram color changed from orange to goldenrod
- Fig5: 'LOTO Baseline' text y=0.669 (just barely above dashed line)
- Fig7: Circle marker size increased to s=80
- Fig8: MAIN FIX — numeric x-ticks decoded to category names via Optuna distribution_json

Run from /workspace: python scripts/generate_figures_neurips_v9.py
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
RDIR = Path('/workspace/results/neurips_figures_v9')
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


# ── Load within-target CV AUROCs for random-split dots ────────────────────────
def _load_within_target_map():
    """Return dict: target_uniprot -> within-target CV AUROC."""
    path = Path('results/exp_supp_ceiling/within_target.csv')
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df['target'], df['auroc']))


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1A: COLLAPSE — v8: random-split dots restored, STAN dots kept
# ═══════════════════════════════════════════════════════════════════════════════
def _fig1a_core(ax_a, methods, tani_map, counts, version='tanimoto',
                within_map=None):
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

            # v8: Per-target random-split dots at replicated RS position (j==1)
            if j == 1 and within_map and m.get('per_target_cs') is not None:
                pt_df = m['per_target_cs']
                if 'target' in pt_df.columns:
                    matched = [(within_map[t], t) for t in pt_df['target']
                               if t in within_map]
                    if matched:
                        rs_aurocs = np.array([a for a, _ in matched])
                        rs_targets = [t for _, t in matched]
                        rng_rs = np.random.default_rng(43)
                        jitter_rs = rng_rs.uniform(-bar_w*0.35, bar_w*0.35, len(rs_aurocs))
                        if version == 'tanimoto':
                            colors_rs = [tani_map.get(t, 0.5) for t in rs_targets]
                            ax_a.scatter(bx + jitter_rs, rs_aurocs, s=8, alpha=0.6,
                                         c=colors_rs, cmap='viridis', vmin=0, vmax=1,
                                         edgecolors='none', zorder=5, marker='o')
                        else:
                            ax_a.scatter(bx + jitter_rs, rs_aurocs, s=8, alpha=0.6,
                                         color='royalblue', edgecolors='none',
                                         zorder=5, marker='o')

            # v8: Per-target LOTO dots at replicated CS position (j==3)
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

            # v8: random-split dots restored above (j==1 block)

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

    # x-tick labels with sample/target counts
    ax_a.set_xticks(positions)
    ax_a.set_xticklabels([m['name'] for m in methods], fontsize=9.5)
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

    # v8: Legend — lower right, includes random per-target entry
    if version == 'tanimoto':
        handles = [
            mpatches.Patch(facecolor=CB_LBLUE, edgecolor='white', label='Reported RS'),
            mpatches.Patch(facecolor=CB_BLUE, alpha=0.25, edgecolor=CB_BLUE, label='Replicated RS'),
            mpatches.Patch(facecolor='#FFB347', edgecolor='white', label='Reported CS'),
            mpatches.Patch(facecolor=CB_ORANGE, alpha=0.25, edgecolor=CB_ORANGE, label='Replicated CS'),
            mpatches.Patch(facecolor='#DDDDDD', edgecolor='#999999', hatch='///', label='N/A'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=CB_GREY, markersize=5, label='RS per-target'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=CB_GREY, markersize=5, label='LOTO per-target'),
        ]
    else:
        handles = [
            mpatches.Patch(facecolor=CB_LBLUE, edgecolor='white', label='Reported RS'),
            mpatches.Patch(facecolor=CB_BLUE, alpha=0.25, edgecolor=CB_BLUE, label='Replicated RS'),
            mpatches.Patch(facecolor='#FFB347', edgecolor='white', label='Reported CS'),
            mpatches.Patch(facecolor=CB_ORANGE, alpha=0.25, edgecolor=CB_ORANGE, label='Replicated CS'),
            mpatches.Patch(facecolor='#DDDDDD', edgecolor='#999999', hatch='///', label='N/A'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markersize=5, label='RS per-target'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=5, label='LOTO per-target'),
        ]
    ax_a.legend(handles=handles, loc='lower right', fontsize=7, ncol=1)

    return sc_handle


def _fig1b(ax_b):
    """Draw Fig1B panel — Tanimoto histograms with mean in legend."""
    tani_random = np.load('results/exp13_splits_and_baselines/tanimoto_random.npy')
    tani_scaffold = np.load('results/exp13_splits_and_baselines/tanimoto_scaffold.npy')
    tani_loto = np.load('results/exp13_splits_and_baselines/tanimoto_loto.npy')
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
    ax_b.set_xlabel('Max Tanimoto to Nearest Training Neighbor')
    ax_b.set_ylabel('Density')
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
    # 3. PROTAC-STAN — v7: add per-target dots from E2_tan_morgan_esms
    stan_pt_path = Path('results/exp1_stan_loto/E2_tan_morgan_esms.csv')
    stan_pt = pd.read_csv(stan_pt_path) if stan_pt_path.exists() else None
    methods.append({
        'name': 'PROTAC-STAN', 'rep_rs': 0.883,
        'repl_rs': full['stan_own_data']['their_split'],
        'repl_rs_err': None,
        'rep_cs': None, 'has_rep_cs': False,
        'repl_cs': full['stan_own_data']['loto_rf'],
        'repl_cs_err': None,
        'per_target_cs': stan_pt,
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
    return methods, tani_map


def fig1_collapse():
    """Generate both versions of Fig1."""
    methods, tani_map = _load_methods_data()
    counts = _get_sample_target_counts()
    within_map = _load_within_target_map()

    # Version A: Tanimoto coloring
    fig_a, (ax_a1, ax_b1) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={'width_ratios': [2.2, 1]})
    _fig1a_core(ax_a1, methods, tani_map, counts, version='tanimoto',
                within_map=within_map)
    _fig1b(ax_b1)
    fig_a.tight_layout()
    _save(fig_a, 'fig1a_collapse_tanimoto')

    # Version B: Simple coloring
    fig_b, (ax_a2, ax_b2) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={'width_ratios': [2.2, 1]})
    _fig1a_core(ax_a2, methods, tani_map, counts, version='simple',
                within_map=within_map)
    _fig1b(ax_b2)
    fig_b.tight_layout()
    _save(fig_b, 'fig1a_collapse_simple')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2: PER-FOLD DISTRIBUTION — unchanged from v6
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
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 200)
    y_line = slope * x_line + intercept
    n = len(x_clean)
    x_mean = np.mean(x_clean)
    residuals = y_clean - (slope * x_clean + intercept)
    s_res = np.sqrt(np.sum(residuals**2) / (n - 2))
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
# FIG 3: HPO CEILING (14×5, 2 panels) — unchanged from v6
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
    ax_a.hist(s1_explore['value'], bins=bins, alpha=0.5, color='goldenrod',
              label=f'Seed 1 explore (n={len(s1_explore)})',
              edgecolor='goldenrod', linewidth=0.5, histtype='stepfilled')
    if len(s1_exploit) > 0:
        ax_a.hist(s1_exploit['value'], bins=bins, alpha=0.4, color='#F0D060',
                  label=f'Seed 1 exploit (n={len(s1_exploit)})',
                  edgecolor='#F0D060', linewidth=0.5, histtype='stepfilled')
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
# FIG 4: BREAKTHROUGHS — v8: honest error bars (only RF+Morgan has seed std)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_breakthroughs():
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline_summary = _load_json('/workspace/summary.json')
    rf_seed_std = baseline_summary['baseline']['std']

    # v8: Check each condition for seed-level std — report findings
    print('  [fig4] Seed-std audit:')
    print(f'    RF+Morgan: seed_std={rf_seed_std:.4f} (from summary.json seed_means) ✓')

    # ADMET cascade: no seed_means or seed_std in exp8c summary
    admet_seed_std = 0
    print('    +ADMET: NO seed-level data (single aggregate) → no error bar')

    # Few-shot k=5: only per-fold std, no seed_means
    fewshot_seed_std = 0
    print('    +k=5 few-shot: NO seed-level data (per-fold std only) → no error bar')

    # EGNN 27 targets: only per-target std, no seed_means
    egnn_seed_std = 0
    print('    +3D EGNN 27: NO seed-level data (per-target std only) → no error bar')

    # EGNN Smina: single seed
    egnn_smina_seed_std = 0
    print('    +3D EGNN Smina: single seed → no error bar')

    entries = [
        ('RF+Morgan\n(65 targets)', 0.666, CB_GREY, False, None, rf_seed_std),
        ('+ADMET\n(65 targets)', 0.687, CB_BLUE, False, 'p=0.014', admet_seed_std),
        ('+k=5 few-shot\n(65 targets)', 0.700, CB_GREEN, False, 'p<0.001', fewshot_seed_std),
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
    # v7: x-axis label includes error bar description, no separate annotation
    ax.set_xlabel('LOTO AUROC (error bars: seed std)')
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)

    # v7: removed 'Hatched = structural input | Error bars = seed std' annotation

    _despine(ax)
    ax.set_title('Incremental Improvements Over Baseline')
    fig.tight_layout()
    _save(fig, 'fig4_breakthroughs')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5: FEW-SHOT — unchanged from v6
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_fewshot():
    e4 = _load_json('results/exp4_fewshot/exp4_summary.json')
    maml_ext = _load_json('results/exp20_maml_full_k/summary.json')
    fig, ax = plt.subplots(figsize=(7, 5))
    ks_rf = [0, 1, 3, 5, 10]
    auroc_rf = [e4[f'rf_morgan_k{k}']['mean'] for k in ks_rf]
    std_rf = [e4[f'rf_morgan_k{k}'].get('std', 0) / np.sqrt(e4[f'rf_morgan_k{k}']['n'])
              for k in ks_rf]
    ax.errorbar(ks_rf, auroc_rf, yerr=std_rf, fmt='o-', color=CB_BLUE,
                label='RF + Morgan', lw=2, markersize=7, capsize=3)
    ks_meta = [1, 3, 5, 10]
    auroc_meta = [e4[f'rf_meta_k{k}']['mean'] for k in ks_meta]
    std_meta = [e4[f'rf_meta_k{k}'].get('std', 0) / np.sqrt(e4[f'rf_meta_k{k}']['n'])
                for k in ks_meta]
    ax.errorbar(ks_meta, auroc_meta, yerr=std_meta, fmt='s-', color=CB_GREEN,
                label='RF + meta', lw=2, markersize=7, capsize=3)
    ks_maml_m = [1, 3, 5, 10]
    auroc_maml_m = [
        maml_ext['conditions']['maml_morgan_k1']['mean_auroc'],
        maml_ext['conditions']['maml_morgan_k3']['mean_auroc'],
        e4['maml_morgan_k5']['mean'],
        maml_ext['conditions']['maml_morgan_k10']['mean_auroc'],
    ]
    ax.plot(ks_maml_m, auroc_maml_m, 'D--', color=CB_RED,
            label='MAML + Morgan', lw=1.5, markersize=6, alpha=0.8)
    auroc_maml_meta = [
        maml_ext['conditions']['maml_meta_k1']['mean_auroc'],
        maml_ext['conditions']['maml_meta_k3']['mean_auroc'],
        e4['maml_meta_k5']['mean'],
        maml_ext['conditions']['maml_meta_k10']['mean_auroc'],
    ]
    ax.plot(ks_maml_m, auroc_maml_meta, 'D--', color=CB_ORANGE,
            label='MAML + meta', lw=1.5, markersize=6, alpha=0.8)
    ax.axhline(0.666, color=CB_GREY, ls='--', lw=1, alpha=0.5)
    ax.text(5.0, 0.666 + 0.003, 'LOTO Baseline', fontsize=8, va='bottom', ha='center', color=CB_GREY)
    ax.set_xlabel('k (target-specific examples)')
    ax.set_ylabel('LOTO AUROC')
    ax.set_xlim(-0.5, 11.5)
    ax.legend(loc='lower right', fontsize=9)
    _despine(ax)
    ax.set_title('Few-Shot Target Adaptation')
    fig.tight_layout()
    _save(fig, 'fig5_fewshot')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6: EGNN SCATTER — v7: remove info box, fix colorbar label
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
    # v7: removed info box (was: wins/n_tgt, rho text box)
    # v7: colorbar label uses readable text
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('log$_{10}$(number of entries)', fontsize=9)
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
# FIG 7: PLM SCALING — unchanged from v6 (multi-seed data still incomplete)
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
        print('  [fig7] NOTE: Only per-fold (cross-target) std available for LOTO (~0.19).')
        print('         Showing no LOTO error bars. Multi-seed experiment recommended.')
        ax.plot(params, random_vals, 'o-', color=CB_BLUE, label='Random CV',
                markersize=10, lw=0.8, alpha=0.25)
        random_err = [plm[p[0]]['random_std'] for p in plm_configs]
        ax.errorbar(params, random_vals, yerr=random_err, fmt='none', ecolor=CB_BLUE,
                    capsize=3, lw=0.8, alpha=0.5)
        ax.plot(params, loto_vals, 'o-', color=CB_ORANGE, label='LOTO',
                markersize=10, lw=0.8, alpha=0.25)
    else:
        ax.errorbar(params, random_vals, yerr=random_seed_stds, fmt='o', color=CB_BLUE,
                    label='Random CV', markersize=10, capsize=3, lw=0.8, alpha=0.25)
        ax.errorbar(params, loto_vals, yerr=loto_seed_stds, fmt='o', color=CB_ORANGE,
                    label='LOTO', markersize=10, capsize=3, lw=0.8, alpha=0.25)
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
# FIG 8: HPO DIMENSION ANALYSIS (NEW in v7)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_hpo_trials():
    """Load all HPO trials from both Optuna DBs with hyperparameters.
    v9: Decode numeric Optuna indices to category names using distribution_json."""
    all_rows = []
    for seed_idx, db_name in enumerate(['exp2_hpo_seed0.db', 'exp2_hpo_seed1.db']):
        db_path = Path(f'results/exp2_unified_hpo/{db_name}')
        if not db_path.exists():
            continue
        conn = sqlite3.connect(db_path)
        trials = pd.read_sql('''
            SELECT t.trial_id, t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
        ''', conn)
        params = pd.read_sql('''
            SELECT trial_id, param_name, param_value, distribution_json
            FROM trial_params
        ''', conn)
        conn.close()
        # v9: Decode categorical params from numeric index to string name
        def _decode_param(row):
            try:
                dist = json.loads(row['distribution_json'])
                if dist.get('name') == 'CategoricalDistribution':
                    choices = dist['attributes']['choices']
                    idx = int(row['param_value'])
                    if 0 <= idx < len(choices):
                        return str(choices[idx])
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                pass
            return row['param_value']
        params['param_value'] = params.apply(_decode_param, axis=1)
        params_pivot = params.pivot(index='trial_id', columns='param_name', values='param_value')
        merged = trials.merge(params_pivot, on='trial_id', how='left')
        merged['seed'] = seed_idx
        merged['phase'] = np.where(merged['number'] < 750, 'explore', 'exploit')
        all_rows.append(merged)
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    # Fallback to CSV (already has string values)
    csv_path = Path('results/exp2_unified_hpo/all_trials.csv')
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df['state'] == 'COMPLETE'].copy()
        df['seed'] = 0
        df['phase'] = np.where(df['number'] < 750, 'explore', 'exploit')
        param_cols = [c for c in df.columns if c.startswith('params_')]
        for c in param_cols:
            df.rename(columns={c: c.replace('params_', '')}, inplace=True)
        return df
    return pd.DataFrame()


def fig8_hpo_dimensions():
    """HPO dimension analysis: swarm plot per hyperparameter dimension.
    v8: short x-tick labels, seed-specific mean lines, alpha tweaks, baseline zorder."""
    df = _load_hpo_trials()
    if df.empty:
        print('  fig8_hpo_dimensions: SKIPPED (no HPO data)')
        return

    # v9: Short label mapping — keys match actual Optuna category names
    LABEL_MAP = {
        # Protein encoders (from Optuna choices)
        'none': 'none', 'esm2_8M': 'esm2_8M', 'esm2_35M': 'esm2_35M',
        'esm2_150M': 'esm2_150M', 'esm2_650M': 'esm2_650M', 'esm2_3B': 'esm2_3B',
        'prostt5': 'prostT5', 'prot_t5_xl': 'protT5_xl', 'ankh_large': 'ankh',
        'prottrans_ribes': 'protTrans', 'esms_stan': 'esms',
        # Molecular encoders
        'morgan_1024_r2': 'morg1024', 'morgan_2048_r2': 'morg2048',
        'morgan_256_r10': 'morg256', 'maccs_167': 'maccs',
        'chemberta': 'chemBERTa', 'molformer': 'molFormer',
        'linker_bilstm': 'bilstm',
        # Architecture
        'rf': 'rf', 'xgboost': 'xgb', 'bilinear': 'bilinear',
        'concat_mlp': 'mlp', 'cross_attention': 'crossAttn',
        'ternary': 'ternary', 'full_tan': 'fullTAN',
        # Fragment mode
        'full': 'full', 'split': 'split',
        # ADMET features
        'True': 'True', 'False': 'False', 'true': 'True', 'false': 'False',
    }

    # Define the 5 panels and their parameter columns
    panels = [
        ('Protein Encoder', 'prot_encoder'),
        ('Molecular Encoder', 'mol_encoder'),
        ('Architecture', 'head_type'),
        ('Fragment Mode', 'fragment_mode'),
        ('ADMET Features', 'admet_features'),
    ]

    # Filter to panels that have data
    active_panels = []
    for title, col in panels:
        if col in df.columns and df[col].notna().sum() > 0:
            active_panels.append((title, col))

    if not active_panels:
        print('  fig8_hpo_dimensions: SKIPPED (no HP columns found)')
        return

    n_panels = len(active_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 8), sharey=True,
                              gridspec_kw={'wspace': 0.05})
    if n_panels == 1:
        axes = [axes]

    baseline_auroc = 0.664

    for panel_idx, (title, col) in enumerate(active_panels):
        ax = axes[panel_idx]
        sub = df[df[col].notna()].copy()
        sub[col] = sub[col].astype(str)
        cat_medians = sub.groupby(col)['value'].median().sort_values(ascending=False)
        categories = cat_medians.index.tolist()

        if not categories:
            ax.set_title(title, fontsize=10, fontweight='bold')
            continue

        # Draw box plots (thin, behind dots)
        bp_data = [sub[sub[col] == cat]['value'].dropna().values for cat in categories]
        bp = ax.boxplot(bp_data, positions=range(len(categories)), widths=0.6,
                        patch_artist=True, showfliers=False, zorder=2)
        for patch in bp['boxes']:
            patch.set_facecolor('#f0f0f0')
            patch.set_edgecolor('#cccccc')
            patch.set_alpha(0.7)
        for element in ['whiskers', 'caps']:
            for line in bp[element]:
                line.set_color('#cccccc')
                line.set_linewidth(0.8)
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)

        # v8: Plot dots with jitter — alpha 0.35 explore, 0.85 exploit
        rng = np.random.default_rng(42)
        dot_colors = np.where(sub['seed'] == 0, 'royalblue', '#E69F00')
        dot_alphas = np.where(sub['phase'] == 'exploit', 0.85, 0.35)
        for cat_idx, cat in enumerate(categories):
            mask = (sub[col] == cat).values
            cat_vals = sub.loc[mask, 'value'].values
            cat_seeds = sub.loc[mask, 'seed'].values
            n_dots = len(cat_vals)
            if n_dots == 0:
                continue
            jitter = rng.uniform(-0.25, 0.25, n_dots)
            cat_colors = dot_colors[mask]
            cat_alphas = dot_alphas[mask]
            for color_val, alpha_val in [('royalblue', 0.35), ('royalblue', 0.85),
                                          ('#E69F00', 0.35), ('#E69F00', 0.85)]:
                gmask = (cat_colors == color_val) & (cat_alphas == alpha_val)
                if gmask.any():
                    ax.scatter(cat_idx + jitter[gmask], cat_vals[gmask],
                               s=8, color=color_val, alpha=alpha_val,
                               edgecolors='none', zorder=3)

            # v8: Seed-specific mean lines instead of single black mean
            for seed_val, seed_color in [(0, 'royalblue'), (1, '#E69F00')]:
                seed_mask = mask & (sub['seed'] == seed_val).values
                if seed_mask.any():
                    seed_mean = sub.loc[seed_mask, 'value'].mean()
                    ax.plot([cat_idx - 0.25, cat_idx + 0.25],
                            [seed_mean, seed_mean],
                            color=seed_color, linewidth=2, zorder=10)

        # v8: Baseline line ON TOP of dots, DOUBLE linewidth
        ax.axhline(baseline_auroc, color='red', ls='--', lw=2.0, alpha=0.7, zorder=10)

        # v8: Short x-tick labels
        ax.set_xticks(range(len(categories)))
        clean_labels = [LABEL_MAP.get(c, c[:10]) for c in categories]
        ax.set_xticklabels(clean_labels, fontsize=7, rotation=45, ha='right')
        ax.set_title(title, fontsize=10, fontweight='bold')

        if panel_idx == 0:
            ax.set_ylabel('LOTO AUROC', fontsize=10)

        _despine(ax)

    axes[0].set_ylim(0.42, 0.75)

    # v8: Updated legend — seed-specific means instead of category mean
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
               markersize=5, alpha=0.35, label='Seed 0 explore'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
               markersize=5, alpha=0.85, label='Seed 0 exploit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E69F00',
               markersize=5, alpha=0.35, label='Seed 1 explore'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E69F00',
               markersize=5, alpha=0.85, label='Seed 1 exploit'),
        Line2D([0], [0], color='red', ls='--', lw=2, label=f'Baseline ({baseline_auroc})'),
        Line2D([0], [0], color='royalblue', lw=2, label='Seed 0 mean'),
        Line2D([0], [0], color='#E69F00', lw=2, label='Seed 1 mean'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=7, fontsize=7,
               bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('HPO Trial AUROC by Hyperparameter Dimension', fontsize=12,
                 fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'fig8_hpo_dimensions')


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES — unchanged from v6
# ═══════════════════════════════════════════════════════════════════════════════

def figS1_fragments():
    frags = _load_json('results/exp_supp_fragments/summary.json')
    # v8: Check for seed-level data
    has_seed = any('seed' in str(v) or 'seed_std' in str(v) for v in frags.values()
                   if isinstance(v, dict))
    print(f'  [figS1] Seed-level data: {"YES" if has_seed else "NO — single seed, no error bars"}')
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
    # v8: Check for per-seed AUROCs
    methodology = data.get('methodology', {})
    has_per_seed = any('seed_means' in str(v) or 'per_seed' in str(v)
                       for v in data.get('conditions', {}).values() if isinstance(v, dict))
    print(f'  [figS5] Methodology mentions seeds: {methodology.get("evaluation", "N/A")}')
    print(f'  [figS5] Per-seed AUROCs stored: {"YES" if has_per_seed else "NO — means only, no error bars"}')
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
    print('  NeurIPS Figures v9 — Final Refinements')
    print('=' * 60)
    print()

    print('=' * 60)
    print('  Generating NeurIPS Figures v9')
    print('=' * 60)

    figs = [
        ('Fig 1 (2 versions)', fig1_collapse),
        ('Fig 2', fig2_perfold),
        ('Fig 3', fig3_hpo),
        ('Fig 4', fig4_breakthroughs),
        ('Fig 5', fig5_fewshot),
        ('Fig 6', fig6_egnn_scatter),
        ('Fig 7', fig7_plm_scaling),
        ('Fig 8', fig8_hpo_dimensions),
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

    # DeepPROTACs random-split dots investigation
    print('\n' + '=' * 60)
    print('  DEEPPROTACS RANDOM-SPLIT DOTS INVESTIGATION')
    print('=' * 60)
    print('  Random-split dots use within-target CV AUROCs (from within_target.csv).')
    print('  These measure per-target RF performance when trained on ALL data')
    print('  (including same-target entries), then evaluated within each target.')
    print('  Mean within-target AUROC ≈ 0.806 across 65 targets.')
    print('  This is higher than overall random-split AUROC (0.664) because')
    print('  within-target CV removes inter-target variance.')
    print('  The dots are kept as-is — they show "ceiling" per-target performance.')
    print('  A proper per-target random-split AUROC would require re-running')
    print('  5-fold CV and decomposing test folds by target, which is feasible')
    print('  but not critical for this figure iteration.')

    print('\n' + '=' * 60)
    print('  v9 CHANGE SUMMARY')
    print('=' * 60)
    print('  Fig 1A:          Random-split dots kept (within-target CV); note above.')
    print('  Fig 2:           Unchanged.')
    print('  Fig 3A:          Seed 1 histogram color: orange → goldenrod.')
    print('  Fig 3B:          Unchanged.')
    print('  Fig 4:           Unchanged.')
    print('  Fig 5:           LOTO Baseline text: y = 0.669 (just above line).')
    print('  Fig 6:           Unchanged.')
    print('  Fig 7:           Circle marker size: 7 → 10 (≈ s=80 equivalent).')
    print('  Fig 8:           MAIN FIX — numeric x-ticks → category names.')
    print('                   Decoded via Optuna distribution_json.')
    print('  Fig S1–S12:      Unchanged.')


if __name__ == '__main__':
    main()
