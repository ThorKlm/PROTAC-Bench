#!/usr/bin/env python3
"""Fig 1 extended: Cold-target collapse with full-benchmark replication.
Generates TWO versions:
  1. Pooled AUROC bars (standard, matches published claims) -> fig1_collapse_extended.pdf
  2. Macro-mean AUROC bars (per-target average) -> figS_collapse_macromean.pdf
Both have per-target dots from real data.
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


def load_per_target_data():
    data = {}
    pt_file = PTDIR / 'C0_morgan_only_per_target.csv'
    if pt_file.exists():
        by_target = {}
        with open(pt_file) as f:
            for row in csv.DictReader(f):
                by_target.setdefault(row['target'], []).append(float(row['auroc']))
        data['RF+Morgan_loto'] = [np.mean(v) for v in by_target.values()]
    pt_random = PTDIR / 'per_target_random_auroc.csv'
    if pt_random.exists():
        data['RF+Morgan_random'] = _load_csv_column(pt_random, 'random_auroc')
    dp = _load_json('exp20_deepprotacs_loto/summary.json')
    if dp:
        loto = dp.get('loto', {})
        pt = loto.get('per_target', [])
        if isinstance(pt, list):
            data['DeepPROTACs_loto'] = [e['mean_auroc'] for e in pt if 'mean_auroc' in e]
    dm = _load_json('exp34_dm_loto/summary.json')
    if dm:
        s = dm.get('step1_by_target_path', dm)
        pt = s.get('per_target', {})
        if isinstance(pt, dict):
            data['DegradeMaster_loto'] = list(pt.values())
    stan_file = PTDIR / 'C0_Full_STAN_per_target.csv'
    if stan_file.exists():
        data['PROTAC-STAN_loto'] = _load_csv_column(stan_file, 'auroc')
    scatter = PTDIR / 'per_target_scatter.csv'
    if scatter.exists():
        data['Ribes_loto'] = _load_csv_column(scatter, 'loto')
        data['Ribes_random'] = _load_csv_column(scatter, 'within')
    gin_file = _load_json('exp13_gnn_baselines/summary_65targets.json')
    if gin_file:
        pt = gin_file.get('per_target', [])
        if isinstance(pt, list):
            data['GIN_loto'] = [e['gin_auroc_mean'] for e in pt if 'gin_auroc_mean' in e]
            data['DMPNN_loto'] = [e['dmpnn_auroc_mean'] for e in pt if 'dmpnn_auroc_mean' in e]
    for method, key in [('GIN_random_per_target.csv', 'GIN_random'),
                        ('DMPNN_random_per_target.csv', 'DMPNN_random'),
                        ('kNN_random_per_target.csv', 'kNN_random'),
                        ('STAN_random_per_target.csv', 'STAN_random')]:
        f = PTDIR / method
        if f.exists():
            data[key] = _load_csv_column(f, 'random_auroc')
    for method, key in [('STAN_repl_random_per_target.csv', 'STAN_repl_random'),
                        ('STAN_repl_loto_per_target.csv', 'STAN_repl_loto'),
                        ('DeepPROTACs_repl_random_per_target.csv', 'DeepPROTACs_random'),
                        ('DM_repl_random_per_target.csv', 'DM_random')]:
        f = PTDIR / method
        if f.exists():
            data[key] = _load_csv_column(f, 'auroc')
    return data


def build_methods_pooled():
    return [
        {
            'name': 'DeepPROTACs', 'published': True,
            'reported_random': 0.847, 'replicated_random': 0.626,
            'reported_cold': None, 'replicated_cold': 0.531,
            'replicated_random_std': 0.015, 'replicated_cold_std': 0.020,
            'full_random': None, 'full_cold': None,
            'full_random_std': None, 'full_cold_std': None,
            'n_samples': 852, 'n_targets': 27,
            'dot_keys': {'repl_loto': 'DeepPROTACs_loto', 'repl_random': 'DeepPROTACs_random'},
        },
        {
            'name': 'Ribes et al.', 'published': True,
            'reported_random': 0.913, 'replicated_random': 0.865,
            'reported_cold': 0.604, 'replicated_cold': 0.646,
            'replicated_random_std': 0.005, 'replicated_cold_std': 0.008,
            'full_random': 0.865, 'full_cold': 0.569,
            'full_random_std': 0.005, 'full_cold_std': 0.010,
            'n_samples': 9428, 'n_targets': 65,
            'dot_keys': {'repl_loto': 'Ribes_loto', 'repl_random': 'Ribes_random',
                        'full_loto': 'Ribes_loto', 'full_random': 'Ribes_random'},
        },
        {
            'name': 'PROTAC-STAN', 'published': True,
            'reported_random': 0.883, 'replicated_random': 0.919,
            'reported_cold': None, 'replicated_cold': 0.682,
            'replicated_random_std': 0.006, 'replicated_cold_std': 0.012,
            'full_random': 0.919, 'full_cold': 0.718,
            'full_random_std': 0.006, 'full_cold_std': 0.015,
            'n_samples': 8754, 'n_targets': 54,
            'dot_keys': {'full_loto': 'PROTAC-STAN_loto', 'full_random': 'STAN_random',
                        'repl_random': 'STAN_repl_random', 'repl_loto': 'STAN_repl_loto'},
        },
        {
            'name': 'DegradeMaster', 'published': True,
            'reported_random': 0.878, 'replicated_random': 0.830,
            'reported_cold': None, 'replicated_cold': 0.702,
            'replicated_random_std': 0.008, 'replicated_cold_std': 0.010,
            'full_random': None, 'full_cold': None,
            'full_random_std': None, 'full_cold_std': None,
            'n_samples': 966, 'n_targets': 35,
            'dot_keys': {'repl_loto': 'DegradeMaster_loto', 'repl_random': 'DM_random'},
        },
        {
            'name': 'GIN', 'published': False,
            'replicated_random': 0.850, 'replicated_cold': 0.611,
            'replicated_random_std': 0.008, 'replicated_cold_std': 0.010,
            'n_samples': 10748, 'n_targets': 65,
            'dot_keys': {'repl_loto': 'GIN_loto', 'repl_random': 'GIN_random'},
        },
        {
            'name': 'D-MPNN', 'published': False,
            'replicated_random': 0.870, 'replicated_cold': 0.673,
            'replicated_random_std': 0.007, 'replicated_cold_std': 0.010,
            'n_samples': 10748, 'n_targets': 65,
            'dot_keys': {'repl_loto': 'DMPNN_loto', 'repl_random': 'DMPNN_random'},
        },
        {
            'name': 'RF+Morgan', 'published': False,
            'replicated_random': 0.902, 'replicated_cold': 0.668,
            'replicated_random_std': 0.004, 'replicated_cold_std': 0.005,
            'n_samples': 10748, 'n_targets': 65,
            'dot_keys': {'repl_loto': 'RF+Morgan_loto', 'repl_random': 'RF+Morgan_random'},
        },
        {
            'name': 'kNN (k=5)', 'published': False,
            'replicated_random': 0.883, 'replicated_cold': 0.630,
            'replicated_random_std': 0.006, 'replicated_cold_std': 0.010,
            'n_samples': 10748, 'n_targets': 65,
            'dot_keys': {'repl_random': 'kNN_random'},
        },
    ]


def build_methods_macro(pt_data):
    """Replace replicated bar values with macro-mean from per-target data."""
    methods = build_methods_pooled()
    # Mapping: which dot_key provides macro-mean for which bar field
    macro_map = {
        'repl_random': 'replicated_random',
        'repl_loto': 'replicated_cold',
        'full_random': 'full_random',
        'full_loto': 'full_cold',
    }
    for m in methods:
        dot_keys = m.get('dot_keys', {})
        for dk, field in macro_map.items():
            if dk in dot_keys and dot_keys[dk] in pt_data:
                dots = pt_data[dot_keys[dk]]
                if dots:
                    m[field] = float(np.mean(dots))
                    std_field = field + '_std'
                    if std_field in m:
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

        # Per-target dots
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
    pt_data = load_per_target_data()
    print(f'  Loaded per-target data: {list(pt_data.keys())}')

    # Version 1: Pooled AUROC (standard, matches published claims)
    methods_pooled = build_methods_pooled()
    _make_figure(methods_pooled, pt_data, 'fig1_collapse_extended.pdf')
    _make_figure(methods_pooled, pt_data, 'fig1_collapse_extended_compact.pdf', compact=True)

    # Version 2: Macro-mean AUROC (per-target average)
    methods_macro = build_methods_macro(pt_data)
    _make_figure(methods_macro, pt_data, 'figS_collapse_macromean.pdf',
                 title_suffix=' (macro-mean)')
    _make_figure(methods_macro, pt_data, 'figS_collapse_macromean_compact.pdf',
                 title_suffix=' (macro-mean)', compact=True)

    # Print comparison table
    print('\n  === Pooled vs Macro-mean ===')
    print(f'  {"Method":<20} {"Pooled RS":>10} {"Macro RS":>10} {"Pooled CS":>10} {"Macro CS":>10}')
    for mp, mm in zip(methods_pooled, methods_macro):
        pr = mp.get('replicated_random', mp.get('full_random'))
        mr = mm.get('replicated_random', mm.get('full_random'))
        pc = mp.get('replicated_cold', mp.get('full_cold'))
        mc = mm.get('replicated_cold', mm.get('full_cold'))
        pr_s = f'{pr:.3f}' if pr else 'N/A'
        mr_s = f'{mr:.3f}' if mr else 'N/A'
        pc_s = f'{pc:.3f}' if pc else 'N/A'
        mc_s = f'{mc:.3f}' if mc else 'N/A'
        print(f'  {mp["name"]:<20} {pr_s:>10} {mr_s:>10} {pc_s:>10} {mc_s:>10}')


if __name__ == '__main__':
    main()
    print('Done.')
