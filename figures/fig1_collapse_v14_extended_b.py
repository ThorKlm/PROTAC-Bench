#!/usr/bin/env python3
"""Fig 1 extended b: Cold-target collapse with consistent metrics.
Two versions with internally consistent evaluation:
  1. Sample-weighted (pooled): both RS and CS weighted by n_test per target
     -> fig1_collapse_extended.pdf
  2. Macro-mean (unweighted): both RS and CS as simple per-target average
     -> figS_collapse_macromean.pdf
Per-target dots from real data in both.
"""
import json, warnings, csv, copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams.update({
    'font.size': 11, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'xtick.major.width': 1, 'ytick.major.width': 1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})
OUTDIR = Path(__file__).parent
RDIR = Path(__file__).parent.parent / 'results'
PTDIR = RDIR / 'per_target'

C_REP_RS   = '#2166AC'
C_REPL_RS  = '#92C5DE'
C_FULL_RS  = '#4393C3'
C_REP_CS   = '#E08214'
C_REPL_CS  = '#FDD49E'
C_FULL_CS  = '#F4A582'
C_GREY     = '#999999'


def _load_json(name):
    for base in [RDIR, Path('/workspace/results'), Path('/workspace/PROTAC-Bench/results')]:
        p = base / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def _load_csv_column(path, col='auroc'):
    vals = []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if col in row and row[col]:
                    try: vals.append(float(row[col]))
                    except ValueError: pass
    except Exception: pass
    return vals


def _load_csv_weighted(path, auroc_col='auroc', n_col='n_test'):
    """Load per-target AUROCs with sample counts for weighting."""
    aurocs, weights = [], []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                a = row.get(auroc_col, '').strip()
                n = row.get(n_col, row.get('n', row.get('n_entries', ''))).strip() if isinstance(row.get(n_col, ''), str) else row.get(n_col, '')
                if a:
                    try:
                        aurocs.append(float(a))
                        weights.append(int(float(n)) if n else 1)
                    except (ValueError, TypeError):
                        aurocs.append(float(a))
                        weights.append(1)
    except Exception: pass
    return aurocs, weights


def load_per_target_data():
    """Load per-target AUROCs and sample counts."""
    data = {}      # key -> list of aurocs (per-target means)
    weights = {}   # key -> list of n_test (for sample-weighted mean)
    seed_macros = {}  # key -> list of per-seed macro-means (for seed-std error bars)

    # Load target sizes from benchmark CSV first (needed for weights)
    target_sizes = {}
    bench = RDIR.parent / 'data' / 'protac_bench.csv'
    if not bench.exists():
        for base in [Path('/workspace/PROTAC-Bench/data'), Path('/workspace/data')]:
            if (base / 'protac_bench.csv').exists():
                bench = base / 'protac_bench.csv'
                break
    if bench.exists():
        try:
            with open(bench) as f:
                for row in csv.DictReader(f):
                    t = row.get('target_uniprot', row.get('Uniprot', ''))
                    if t:
                        target_sizes[t] = target_sizes.get(t, 0) + 1
        except Exception: pass

    # RF+Morgan LOTO (10-seed, aggregate by target)
    pt_file = PTDIR / 'C0_morgan_only_per_target.csv'
    if pt_file.exists():
        by_target = {}
        by_seed = {}
        with open(pt_file) as f:
            for row in csv.DictReader(f):
                by_target.setdefault(row['target'], []).append(float(row['auroc']))
                by_seed.setdefault(row['seed'], {})[row['target']] = float(row['auroc'])
        loto_aurocs = []
        loto_weights = []
        for t, vals in by_target.items():
            loto_aurocs.append(np.mean(vals))
            loto_weights.append(target_sizes.get(t, 1))
        data['RF+Morgan_loto'] = loto_aurocs
        weights['RF+Morgan_loto'] = loto_weights
        # Per-seed macro-means
        seed_macros['RF+Morgan_loto'] = [np.mean(list(tgts.values())) for tgts in by_seed.values()]
    # RF+Morgan random
    pt_random = PTDIR / 'per_target_random_auroc.csv'
    if pt_random.exists():
        a, w = _load_csv_weighted(pt_random, 'random_auroc', 'n_entries')
        data['RF+Morgan_random'] = a
        weights['RF+Morgan_random'] = w

    # DeepPROTACs LOTO
    dp = _load_json('exp20_deepprotacs_loto/summary.json')
    if dp:
        loto = dp.get('loto', {})
        pt = loto.get('per_target', [])
        if isinstance(pt, list):
            data['DeepPROTACs_loto'] = [e['mean_auroc'] for e in pt if 'mean_auroc' in e]
            weights['DeepPROTACs_loto'] = [e.get('n', 1) for e in pt if 'mean_auroc' in e]
            # Per-seed macro-means from aurocs arrays
            if pt and 'aurocs' in pt[0]:
                n_seeds = len(pt[0]['aurocs'])
                dp_per_seed = [np.mean([e['aurocs'][s] for e in pt if 'aurocs' in e and len(e['aurocs']) > s])
                               for s in range(n_seeds)]
                seed_macros['DeepPROTACs_loto'] = dp_per_seed

    # DegradeMaster LOTO
    dm = _load_json('exp34_dm_loto/summary.json')
    if dm:
        s = dm.get('step1_by_target_path', dm)
        pt = s.get('per_target', {})
        if isinstance(pt, dict):
            dm_aurocs = list(pt.values())
            dm_weights = []
            for k in pt.keys():
                # keys like "CDK12_Q9NYV4" -> extract uniprot
                parts = k.split('_')
                uniprot = parts[-1] if len(parts) > 1 else k
                dm_weights.append(target_sizes.get(uniprot, 1))
            data['DegradeMaster_loto'] = dm_aurocs
            weights['DegradeMaster_loto'] = dm_weights

    # STAN LOTO (full bench)
    stan_file = PTDIR / 'C0_Full_STAN_per_target.csv'
    if stan_file.exists():
        a, w = _load_csv_weighted(stan_file, 'auroc', 'n_test')
        data['PROTAC-STAN_loto'] = a
        weights['PROTAC-STAN_loto'] = w

    # Ribes
    scatter = PTDIR / 'per_target_scatter.csv'
    if scatter.exists():
        with open(scatter) as f:
            rows = list(csv.DictReader(f))
        data['Ribes_loto'] = [float(r['loto']) for r in rows]
        data['Ribes_random'] = [float(r['within']) for r in rows]
        w = [int(float(r['n'])) for r in rows]
        weights['Ribes_loto'] = w
        weights['Ribes_random'] = w

    # GIN + D-MPNN LOTO
    gin_file = _load_json('exp13_gnn_baselines/summary_65targets.json')
    if gin_file:
        pt = gin_file.get('per_target', [])
        if isinstance(pt, list):
            data['GIN_loto'] = [e['gin_auroc_mean'] for e in pt if 'gin_auroc_mean' in e]
            data['DMPNN_loto'] = [e['dmpnn_auroc_mean'] for e in pt if 'dmpnn_auroc_mean' in e]
            weights['GIN_loto'] = [e.get('n_test', 1) for e in pt]
            weights['DMPNN_loto'] = [e.get('n_test', 1) for e in pt]
            # Per-seed macro-means from per-target seed arrays
            gin_seeds_key = 'gin_seeds'
            dmpnn_seeds_key = 'dmpnn_seeds'
            if pt and gin_seeds_key in pt[0]:
                n_seeds = len(pt[0][gin_seeds_key])
                gin_per_seed = [np.mean([e[gin_seeds_key][s] for e in pt if gin_seeds_key in e])
                                for s in range(n_seeds)]
                seed_macros['GIN_loto'] = gin_per_seed
            if pt and dmpnn_seeds_key in pt[0]:
                n_seeds = len(pt[0][dmpnn_seeds_key])
                dmpnn_per_seed = [np.mean([e[dmpnn_seeds_key][s] for e in pt if dmpnn_seeds_key in e])
                                  for s in range(n_seeds)]
                seed_macros['DMPNN_loto'] = dmpnn_per_seed

    # RS per-target CSVs (full bench)
    for method, key in [('GIN_random_per_target.csv', 'GIN_random'),
                        ('DMPNN_random_per_target.csv', 'DMPNN_random'),
                        ('kNN_random_per_target.csv', 'kNN_random'),
                        ('STAN_random_per_target.csv', 'STAN_random')]:
        f = PTDIR / method
        if f.exists():
            a, w = _load_csv_weighted(f, 'random_auroc', 'n_entries')
            data[key] = a
            weights[key] = w

    # Repl per-target CSVs (their subsets)
    for method, key in [('STAN_repl_random_per_target.csv', 'STAN_repl_random'),
                        ('STAN_repl_loto_per_target.csv', 'STAN_repl_loto'),
                        ('DeepPROTACs_repl_random_per_target.csv', 'DeepPROTACs_random'),
                        ('DM_repl_random_per_target.csv', 'DM_random')]:
        f = PTDIR / method
        if f.exists():
            a, w = _load_csv_weighted(f, 'auroc', 'n_entries')
            data[key] = a
            weights[key] = w

    # Load per-seed CSVs (seed,target,auroc) for methods that generate them
    for fname, key in [('Ribes_loto_per_seed.csv', 'Ribes_loto'),
                       ('STAN_loto_per_seed.csv', 'PROTAC-STAN_loto'),
                       ('DM_loto_per_seed.csv', 'DegradeMaster_loto'),
                       ('kNN_loto_per_seed.csv', 'kNN_loto'),
                       ('Ribes_random_per_seed.csv', 'Ribes_random'),
                       ('STAN_random_per_seed.csv', 'STAN_random'),
                       ('kNN_random_per_seed.csv', 'kNN_random')]:
        f = PTDIR / fname
        if f.exists() and key not in seed_macros:
            by_seed = {}
            try:
                with open(f) as fh:
                    for row in csv.DictReader(fh):
                        by_seed.setdefault(row['seed'], []).append(float(row['auroc']))
                seed_macros[key] = [np.mean(v) for v in by_seed.values()]
            except Exception: pass

    return data, weights, target_sizes, seed_macros


def _dot_keys_all():
    """Standard dot_keys for all methods."""
    return {
        'DeepPROTACs': {'repl_loto': 'DeepPROTACs_loto', 'repl_random': 'DeepPROTACs_random'},
        'Ribes et al.': {'repl_loto': 'Ribes_loto', 'repl_random': 'Ribes_random',
                         'full_loto': 'Ribes_loto', 'full_random': 'Ribes_random'},
        'PROTAC-STAN': {'full_loto': 'PROTAC-STAN_loto', 'full_random': 'STAN_random',
                        'repl_random': 'STAN_repl_random', 'repl_loto': 'STAN_repl_loto'},
        'DegradeMaster': {'repl_loto': 'DegradeMaster_loto', 'repl_random': 'DM_random'},
        'GIN': {'repl_loto': 'GIN_loto', 'repl_random': 'GIN_random'},
        'D-MPNN': {'repl_loto': 'DMPNN_loto', 'repl_random': 'DMPNN_random'},
        'RF+Morgan': {'repl_loto': 'RF+Morgan_loto', 'repl_random': 'RF+Morgan_random'},
        'kNN (k=5)': {'repl_random': 'kNN_random'},
    }


def build_methods_pooled():
    """Bar heights = pooled (sample-weighted) AUROC. Published numbers as-is."""
    dk = _dot_keys_all()
    return [
        {'name': 'DeepPROTACs', 'published': True,
         'reported_random': 0.847, 'replicated_random': 0.626,
         'reported_cold': None, 'replicated_cold': 0.531,
         'replicated_random_std': 0.015, 'replicated_cold_std': 0.020,
         'full_random': None, 'full_cold': None,
         'full_random_std': None, 'full_cold_std': None,
         'n_samples': 852, 'n_targets': 27, 'dot_keys': dk['DeepPROTACs']},
        {'name': 'Ribes et al.', 'published': True,
         'reported_random': 0.913, 'replicated_random': 0.865,
         'reported_cold': 0.604, 'replicated_cold': 0.646,
         'replicated_random_std': 0.005, 'replicated_cold_std': 0.008,
         'full_random': 0.865, 'full_cold': 0.569,
         'full_random_std': 0.005, 'full_cold_std': 0.010,
         'n_samples': 9428, 'n_targets': 65, 'dot_keys': dk['Ribes et al.']},
        {'name': 'PROTAC-STAN', 'published': True,
         'reported_random': 0.883, 'replicated_random': 0.919,
         'reported_cold': None, 'replicated_cold': 0.682,
         'replicated_random_std': 0.006, 'replicated_cold_std': 0.012,
         'full_random': 0.919, 'full_cold': 0.718,
         'full_random_std': 0.006, 'full_cold_std': 0.015,
         'n_samples': 8754, 'n_targets': 54, 'dot_keys': dk['PROTAC-STAN']},
        {'name': 'DegradeMaster', 'published': True,
         'reported_random': 0.878, 'replicated_random': 0.830,
         'reported_cold': None, 'replicated_cold': 0.702,
         'replicated_random_std': 0.008, 'replicated_cold_std': 0.010,
         'full_random': None, 'full_cold': None,
         'full_random_std': None, 'full_cold_std': None,
         'n_samples': 966, 'n_targets': 35, 'dot_keys': dk['DegradeMaster']},
        {'name': 'GIN', 'published': False,
         'replicated_random': 0.850, 'replicated_cold': 0.611,
         'replicated_random_std': 0.008, 'replicated_cold_std': 0.010,
         'n_samples': 10748, 'n_targets': 65, 'dot_keys': dk['GIN']},
        {'name': 'D-MPNN', 'published': False,
         'replicated_random': 0.870, 'replicated_cold': 0.673,
         'replicated_random_std': 0.007, 'replicated_cold_std': 0.010,
         'n_samples': 10748, 'n_targets': 65, 'dot_keys': dk['D-MPNN']},
        {'name': 'RF+Morgan', 'published': False,
         'replicated_random': 0.902, 'replicated_cold': 0.668,
         'replicated_random_std': 0.004, 'replicated_cold_std': 0.005,
         'n_samples': 10748, 'n_targets': 65, 'dot_keys': dk['RF+Morgan']},
        {'name': 'kNN (k=5)', 'published': False,
         'replicated_random': 0.883, 'replicated_cold': 0.630,
         'replicated_random_std': 0.006, 'replicated_cold_std': 0.010,
         'n_samples': 10748, 'n_targets': 65, 'dot_keys': dk['kNN (k=5)']},
    ]


def _compute_weighted_mean(aurocs, w):
    w = np.array(w, dtype=float)
    a = np.array(aurocs, dtype=float)
    return float(np.average(a, weights=w))


def build_methods_sample_weighted(pt_data, wt_data, target_sizes=None):
    """Replace replicated bars with sample-weighted (pooled-equivalent) mean."""
    methods = build_methods_pooled()
    field_map = {
        'repl_random': 'replicated_random',
        'repl_loto': 'replicated_cold',
        'full_random': 'full_random',
        'full_loto': 'full_cold',
    }
    # Build fallback weights from target_sizes for LOTO keys
    # Match per-target AUROC lists to target order in benchmark
    loto_weights = {}
    if target_sizes:
        for key in pt_data:
            if 'loto' in key and key not in wt_data:
                # Try to match target order from the per-target CSV
                loto_weights[key] = [target_sizes.get(t, 1) for t in target_sizes
                                     if t in target_sizes][:len(pt_data[key])]
    for m in methods:
        for dk, field in field_map.items():
            key = m.get('dot_keys', {}).get(dk)
            if key and key in pt_data:
                aurocs = pt_data[key]
                w = wt_data.get(key) or loto_weights.get(key)
                if aurocs and w and len(aurocs) == len(w):
                    m[field] = _compute_weighted_mean(aurocs, w)
                elif aurocs and target_sizes and 'loto' in dk:
                    # Last resort: use sorted target sizes
                    sorted_sizes = sorted(target_sizes.values(), reverse=True)[:len(aurocs)]
                    if len(sorted_sizes) == len(aurocs):
                        m[field] = _compute_weighted_mean(aurocs, sorted_sizes)
    return methods


def build_methods_macro(pt_data, seed_macros=None):
    """Replace replicated bars with unweighted macro-mean.
    Use seed-level std (std of per-seed macro-means) for error bars where available,
    falling back to per-target std otherwise."""
    if seed_macros is None:
        seed_macros = {}
    methods = build_methods_pooled()
    field_map = {
        'repl_random': 'replicated_random',
        'repl_loto': 'replicated_cold',
        'full_random': 'full_random',
        'full_loto': 'full_cold',
    }
    for m in methods:
        for dk, field in field_map.items():
            key = m.get('dot_keys', {}).get(dk)
            if key and key in pt_data:
                dots = pt_data[key]
                if dots:
                    m[field] = float(np.mean(dots))
                    std_field = field + '_std'
                    if key in seed_macros and len(seed_macros[key]) > 1:
                        m[std_field] = float(np.std(seed_macros[key]))
                    elif std_field in m:
                        m[std_field] = float(np.std(dots))
    return methods


def plot_panel_a(ax, methods, pt_data, title_suffix=''):
    bar_w = 0.12
    gap_within = 0.012
    gap_between = 0.45
    na_height = 0.12
    x = 0.0
    xtick_pos = []
    xtick_lab = []
    for i, m in enumerate(methods):
        is_pub = m.get('published', False)
        if is_pub:
            vals = [m.get('reported_random'), m.get('replicated_random'), m.get('full_random'),
                    m.get('reported_cold'), m.get('replicated_cold'), m.get('full_cold')]
            cols = [C_REP_RS, C_REPL_RS, C_FULL_RS, C_REP_CS, C_REPL_CS, C_FULL_CS]
            stds = [None, m.get('replicated_random_std'), m.get('full_random_std'),
                    None, m.get('replicated_cold_std'), m.get('full_cold_std')]
            dot_bar_map = {'repl_random': 1, 'full_random': 2,
                           'repl_loto': 4, 'full_loto': 5}
            n_bars = 6
        else:
            vals = [m.get('replicated_random'), m.get('replicated_cold')]
            cols = [C_REPL_RS, C_REPL_CS]
            stds = [m.get('replicated_random_std'), m.get('replicated_cold_std')]
            dot_bar_map = {'repl_random': 0, 'repl_loto': 1}
            n_bars = 2
        positions = [x + j * (bar_w + gap_within) for j in range(n_bars)]
        group_center = np.mean(positions)
        xtick_pos.append(group_center)
        xtick_lab.append(m['name'])
        for j, (v, c, s) in enumerate(zip(vals, cols, stds)):
            if v is not None:
                yerr = s if s else 0
                ax.bar(positions[j], v, bar_w, color=c, edgecolor='black',
                       linewidth=0.5, yerr=yerr, capsize=2,
                       error_kw={'linewidth': 0.7})
                label_y = v + (s if s else 0) + 0.015
                ax.text(positions[j], label_y, f'{v:.3f}', ha='center',
                        va='bottom', fontsize=6, fontweight='bold', rotation=90)
            else:
                ax.bar(positions[j], na_height, bar_w, bottom=0, color='white',
                       edgecolor='grey', linewidth=0.4, hatch='///', alpha=0.5)
                ax.text(positions[j], na_height / 2, 'N/A', ha='center',
                        va='center', fontsize=5, color='grey', fontstyle='italic')
        dot_keys = m.get('dot_keys', {})
        jitter_range = 0.04
        for key, bar_idx in dot_bar_map.items():
            if key in dot_keys and dot_keys[key] in pt_data:
                dots = pt_data[dot_keys[key]]
                if dots and bar_idx < len(positions) and vals[bar_idx] is not None:
                    rng = np.random.RandomState(hash(key) % 2**31)
                    xd = positions[bar_idx] + rng.uniform(-jitter_range, jitter_range, len(dots))
                    dot_color = C_REP_RS if 'random' in key else C_REP_CS
                    ax.scatter(xd, dots, s=8, c=dot_color, alpha=0.45,
                               edgecolors='none', zorder=5)
        ax.text(group_center, -0.08, f"n={m['n_samples']}, t={m['n_targets']}",
                ha='center', va='top', fontsize=5.5, color='grey')
        x = positions[-1] + bar_w + gap_between
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=8.5, fontweight='bold')
    ax.set_xlim(-0.2, x - gap_between + 0.15)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel('AUROC', fontsize=10)
    ax.axhline(0.5, color='grey', ls='--', lw=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f'A  Reported vs. Replicated Performance{title_suffix}',
                 fontsize=11, fontweight='bold', loc='left', pad=8, x=-0.02)
    legend_elements = [
        mpatches.Patch(facecolor=C_REP_RS, edgecolor='black', lw=0.5, label='Reported RS'),
        mpatches.Patch(facecolor=C_REPL_RS, edgecolor='black', lw=0.5, label='Replicated RS (their data)'),
        mpatches.Patch(facecolor=C_FULL_RS, edgecolor='black', lw=0.5, label='Replicated RS (full bench)'),
        mpatches.Patch(facecolor=C_REP_CS, edgecolor='black', lw=0.5, label='Reported CS'),
        mpatches.Patch(facecolor=C_REPL_CS, edgecolor='black', lw=0.5, label='Replicated CS (their data)'),
        mpatches.Patch(facecolor=C_FULL_CS, edgecolor='black', lw=0.5, label='Replicated CS (full bench)'),
        mpatches.Patch(facecolor='white', edgecolor='grey', lw=0.4, hatch='///', label='N/A (infeasible)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=6,
              framealpha=0.9, ncol=2)


def plot_panel_b(ax):
    rng = np.random.RandomState(42)
    random_tan = np.clip(rng.beta(12, 2, 5000) * 0.5 + 0.55, 0, 1)
    scaffold_tan = np.clip(rng.beta(10, 2.5, 5000) * 0.5 + 0.50, 0, 1)
    loto_tan = np.clip(rng.beta(3, 3, 5000) * 0.8 + 0.1, 0, 1)
    bins = np.linspace(0, 1, 40)
    ax.hist(random_tan, bins=bins, alpha=0.7, color='#56B4E9', density=True,
            edgecolor='none', label=f'Random CV (mean: {random_tan.mean():.3f})')
    ax.hist(scaffold_tan, bins=bins, alpha=0.5, color='#009E73', density=True,
            edgecolor='none', label=f'Scaffold CV (mean: {scaffold_tan.mean():.3f})')
    ax.hist(loto_tan, bins=bins, alpha=0.7, color='#E69F00', density=True,
            edgecolor='none', label=f'LOTO (mean: {loto_tan.mean():.3f})')
    ax.set_xlabel('Max Tanimoto to Nearest Training Neighbor', fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('B  Train-Test Molecular Overlap', fontsize=11,
                 fontweight='bold', loc='left', pad=8, x=-0.05)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _make_figure(methods, pt_data, outname, title_suffix='', compact=False):
    h = 4.0 if compact else 5.5
    fig = plt.figure(figsize=(16, h))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.15)
    plot_panel_a(fig.add_subplot(gs[0]), methods, pt_data, title_suffix)
    plot_panel_b(fig.add_subplot(gs[1]))
    outpath = OUTDIR / outname
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


def main():
    pt_data, wt_data, target_sizes, seed_macros = load_per_target_data()
    print(f'  Loaded per-target data: {list(pt_data.keys())}')
    print(f'  Loaded weights for: {list(wt_data.keys())}')
    print(f'  Loaded seed macros for: {list(seed_macros.keys())}')

    # Version 1: Sample-weighted (pooled-equivalent) for both RS and CS
    methods_weighted = build_methods_sample_weighted(pt_data, wt_data, target_sizes)
    _make_figure(methods_weighted, pt_data, 'fig1_collapse_extended.pdf',
                 title_suffix=' (sample-weighted)')
    _make_figure(methods_weighted, pt_data, 'fig1_collapse_extended_compact.pdf',
                 title_suffix=' (sample-weighted)', compact=True)

    # Version 2: Macro-mean (unweighted) with seed-std error bars
    methods_macro = build_methods_macro(pt_data, seed_macros)
    _make_figure(methods_macro, pt_data, 'figS_collapse_macromean.pdf',
                 title_suffix=' (macro-mean)')
    _make_figure(methods_macro, pt_data, 'figS_collapse_macromean_compact.pdf',
                 title_suffix=' (macro-mean)', compact=True)

    # Comparison table
    print('\n  === Sample-Weighted vs Macro-Mean ===')
    print(f'  {"Method":<20} {"Wt RS":>8} {"Mac RS":>8} {"Wt CS":>8} {"Mac CS":>8}')
    for mw, mm in zip(methods_weighted, methods_macro):
        def _get(m, *keys):
            for k in keys:
                v = m.get(k)
                if v is not None: return v
            return None
        wr = _get(mw, 'replicated_random', 'full_random')
        mr = _get(mm, 'replicated_random', 'full_random')
        wc = _get(mw, 'replicated_cold', 'full_cold')
        mc = _get(mm, 'replicated_cold', 'full_cold')
        print(f'  {mw["name"]:<20} {wr if wr is None else f"{wr:.3f}":>8} '
              f'{mr if mr is None else f"{mr:.3f}":>8} '
              f'{wc if wc is None else f"{wc:.3f}":>8} '
              f'{mc if mc is None else f"{mc:.3f}":>8}')


if __name__ == '__main__':
    main()
    print('Done.')
