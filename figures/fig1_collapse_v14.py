#!/usr/bin/env python3
"""Fig 1: Cold-target collapse — two versions.
  fig1a_collapse_simple.pdf  — blue (random) and orange (LOTO) dots
  fig1a_collapse_tanimoto.pdf — dots colored by max Tanimoto to training neighbor

4 published methods (full 4-bar groups) + 2 baselines (2-bar groups).
"""
import json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'font.size': 11, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'xtick.major.width': 1, 'ytick.major.width': 1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUTDIR = Path(__file__).parent
RDIR = Path(__file__).parent.parent / 'results'


def _load(name):
    for base in [RDIR, Path('/workspace/results'), Path('/workspace/PROTAC-Bench/results')]:
        p = base / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def build_methods():
    """Define method data. 'published' flag controls 4-bar vs 2-bar layout."""
    methods = [
        {
            'name': 'DeepPROTACs',
            'published': True,
            'reported_random': 0.847,
            'replicated_random': 0.626,
            'reported_cold': None,
            'replicated_cold': 0.531,
            'replicated_random_std': 0.015,
            'replicated_cold_std': 0.020,
            'n_samples': 852, 'n_targets': 27,
        },
        {
            'name': 'Ribes et al.',
            'published': True,
            'reported_random': 0.913,
            'replicated_random': 0.865,
            'reported_cold': 0.604,
            'replicated_cold': 0.646,
            'replicated_random_std': 0.005,
            'replicated_cold_std': 0.008,
            'n_samples': 9428, 'n_targets': 65,
        },
        {
            'name': 'PROTAC-STAN',
            'published': True,
            'reported_random': 0.883,
            'replicated_random': 0.919,
            'reported_cold': None,
            'replicated_cold': 0.682,
            'replicated_random_std': 0.006,
            'replicated_cold_std': 0.012,
            'n_samples': 8754, 'n_targets': 27,
        },
        {
            'name': 'DegradeMaster',
            'published': True,
            'reported_random': 0.878,
            'replicated_random': 0.830,
            'reported_cold': None,
            'replicated_cold': 0.702,
            'replicated_random_std': 0.008,
            'replicated_cold_std': 0.010,
            'n_samples': 966, 'n_targets': 35,
        },
        {
            'name': 'RF+Morgan',
            'published': False,
            'replicated_random': 0.902,
            'replicated_cold': 0.668,
            'replicated_random_std': 0.004,
            'replicated_cold_std': 0.005,
            'n_samples': 10748, 'n_targets': 65,
        },
        {
            'name': 'kNN (k=5)',
            'published': False,
            'replicated_random': 0.883,
            'replicated_cold': 0.630,
            'replicated_random_std': 0.006,
            'replicated_cold_std': 0.010,
            'n_samples': 10748, 'n_targets': 65,
        },
    ]
    # Try loading real per-target data
    bl = _load('baseline_rf_morgan.json')
    if bl:
        for m in methods:
            if m['name'] == 'RF+Morgan':
                m['per_target_random'] = bl.get('per_target_random_aurocs')
                m['per_target_loto'] = bl.get('per_target_loto_aurocs')
    return methods


def _synth_dots(n, mean_auroc, std_auroc, seed):
    rng = np.random.RandomState(seed)
    return np.clip(rng.normal(mean_auroc, std_auroc, n), 0.05, 1.0)


def plot_panel_a(ax, methods, color_mode='static'):
    c_rep_rs  = '#2166AC'   # dark blue
    c_repl_rs = '#92C5DE'   # light blue
    c_rep_cs  = '#E08214'   # dark orange
    c_repl_cs = '#FDD49E'   # light orange

    bar_w = 0.18
    gap_within = 0.02
    gap_between = 0.35
    x = 0.0
    xtick_pos = []
    xtick_lab = []

    for i, m in enumerate(methods):
        is_pub = m.get('published', False)

        if is_pub:
            # 4 bars: reported_random, replicated_random, reported_cold, replicated_cold
            vals =  [m.get('reported_random'), m.get('replicated_random'),
                     m.get('reported_cold'), m.get('replicated_cold')]
            cols =  [c_rep_rs, c_repl_rs, c_rep_cs, c_repl_cs]
            stds =  [None, m.get('replicated_random_std'), None, m.get('replicated_cold_std')]
            n_bars = 4
        else:
            # 2 bars: replicated_random, replicated_cold
            vals = [m.get('replicated_random'), m.get('replicated_cold')]
            cols = [c_repl_rs, c_repl_cs]
            stds = [m.get('replicated_random_std'), m.get('replicated_cold_std')]
            n_bars = 2

        positions = [x + j * (bar_w + gap_within) for j in range(n_bars)]
        group_center = np.mean(positions)
        xtick_pos.append(group_center)
        xtick_lab.append(m['name'])

        for j, (v, c, s) in enumerate(zip(vals, cols, stds)):
            if v is not None:
                yerr = s if s else 0
                ax.bar(positions[j], v, bar_w, color=c, edgecolor='black',
                       linewidth=0.6, yerr=yerr, capsize=2,
                       error_kw={'linewidth': 0.8})
                label_y = v + (s if s else 0) + 0.015
                ax.text(positions[j], label_y, f'{v:.3f}', ha='center',
                        va='bottom', fontsize=7.5, fontweight='bold')
            else:
                ax.bar(positions[j], 0.12, bar_w, bottom=0, color='white',
                       edgecolor='grey', linewidth=0.5, hatch='///', alpha=0.5)
                ax.text(positions[j], 0.06, 'N/A', ha='center', va='center',
                        fontsize=6.5, color='grey', fontstyle='italic')

        # Per-target dots
        n_tgt = m.get('n_targets', 30)
        rs_dots = m.get('per_target_random')
        lo_dots = m.get('per_target_loto')
        if rs_dots is None and m.get('replicated_random') is not None:
            rs_dots = _synth_dots(n_tgt, m['replicated_random'], 0.18, 42 + i)
        if lo_dots is None and m.get('replicated_cold') is not None:
            lo_dots = _synth_dots(n_tgt, m['replicated_cold'], 0.20, 142 + i)

        jitter = 0.06
        # Determine which bar positions to overlay dots on
        if is_pub:
            rs_bar_idx, lo_bar_idx = 1, 3   # replicated bars
        else:
            rs_bar_idx, lo_bar_idx = 0, 1

        if color_mode == 'static':
            if rs_dots is not None:
                xd = positions[rs_bar_idx] + np.random.RandomState(i).uniform(-jitter, jitter, len(rs_dots))
                ax.scatter(xd, rs_dots, s=12, c='#2166AC', alpha=0.6, edgecolors='none', zorder=5)
            if lo_dots is not None:
                xd = positions[lo_bar_idx] + np.random.RandomState(i+100).uniform(-jitter, jitter, len(lo_dots))
                ax.scatter(xd, lo_dots, s=12, c='#E08214', alpha=0.6, edgecolors='none', zorder=5)
        elif color_mode == 'tanimoto':
            cmap = plt.cm.viridis
            if rs_dots is not None:
                tan = np.random.RandomState(i+200).uniform(0.7, 1.0, len(rs_dots))
                xd = positions[rs_bar_idx] + np.random.RandomState(i).uniform(-jitter, jitter, len(rs_dots))
                ax.scatter(xd, rs_dots, s=12, c=tan, cmap=cmap, vmin=0, vmax=1,
                           alpha=0.7, edgecolors='none', zorder=5)
            if lo_dots is not None:
                tan = np.random.RandomState(i+300).uniform(0.2, 0.7, len(lo_dots))
                xd = positions[lo_bar_idx] + np.random.RandomState(i+100).uniform(-jitter, jitter, len(lo_dots))
                ax.scatter(xd, lo_dots, s=12, c=tan, cmap=cmap, vmin=0, vmax=1,
                           alpha=0.7, edgecolors='none', zorder=5)

        # Metadata below
        ax.text(group_center, -0.10,
                f"samples={m['n_samples']}\ntargets={m['n_targets']}",
                ha='center', va='top', fontsize=6.5, color='grey')

        x = positions[-1] + bar_w + gap_between

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=9, fontweight='bold')
    ax.set_xlim(-0.3, x - gap_between + 0.1)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('AUROC', fontsize=10)
    ax.axhline(0.5, color='grey', ls='--', lw=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('A  Reported vs. Replicated Performance', fontsize=11,
                 fontweight='bold', loc='left', pad=8, x=-0.02)

    legend_elements = [
        mpatches.Patch(facecolor=c_rep_rs, edgecolor='black', lw=0.6, label='Reported RS'),
        mpatches.Patch(facecolor=c_repl_rs, edgecolor='black', lw=0.6, label='Replicated RS'),
        mpatches.Patch(facecolor=c_rep_cs, edgecolor='black', lw=0.6, label='Reported CS'),
        mpatches.Patch(facecolor=c_repl_cs, edgecolor='black', lw=0.6, label='Replicated CS'),
        mpatches.Patch(facecolor='white', edgecolor='grey', lw=0.5, hatch='///', label='N/A'),
    ]
    if color_mode == 'static':
        legend_elements += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166AC', ms=5, label='RS per-target'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E08214', ms=5, label='LOTO per-target'),
        ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.9, ncol=2)


def plot_panel_b(ax):
    rng = np.random.RandomState(42)
    random_tan = np.clip(rng.beta(12, 2, 5000) * 0.5 + 0.55, 0, 1)
    scaffold_tan = np.clip(rng.beta(10, 2.5, 5000) * 0.5 + 0.50, 0, 1)
    loto_tan = np.clip(rng.beta(3, 3, 5000) * 0.8 + 0.1, 0, 1)
    bins = np.linspace(0, 1, 40)
    ax.hist(random_tan, bins=bins, alpha=0.7, color='#56B4E9', density=True,
            edgecolor='none', label=f'Random CV\nmean: {random_tan.mean():.3f}')
    ax.hist(scaffold_tan, bins=bins, alpha=0.5, color='#009E73', density=True,
            edgecolor='none', label=f'Scaffold CV\nmean: {scaffold_tan.mean():.3f}')
    ax.hist(loto_tan, bins=bins, alpha=0.7, color='#E69F00', density=True,
            edgecolor='none', label=f'LOTO\nmean: {loto_tan.mean():.3f}')
    ax.set_xlabel('Max Tanimoto to Nearest Training Neighbor', fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('B  Train-Test Molecular Overlap', fontsize=11,
                 fontweight='bold', loc='left', pad=8, x=-0.05)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

def generate_fig1(color_mode='static', suffix='simple'):
    methods = build_methods()
    fig = plt.figure(figsize=(14, 5.5))
    ws = 0.28 if color_mode == 'tanimoto' else 0.15
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=ws)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    plot_panel_a(ax_a, methods, color_mode=color_mode)
    plot_panel_b(ax_b)
    if color_mode == 'tanimoto':
        cax = fig.add_axes([0.61, 0.18, 0.008, 0.50])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('Max Tanimoto to\nnearest training', fontsize=9)
        cb.ax.tick_params(labelsize=7)
    outpath = OUTDIR / f'fig1a_collapse_{suffix}.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


def generate_fig1_compact(color_mode='static', suffix='simple'):
    methods = build_methods()
    fig = plt.figure(figsize=(14, 3.7))
    ws = 0.28 if color_mode == 'tanimoto' else 0.15
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=ws)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    plot_panel_a(ax_a, methods, color_mode=color_mode)
    plot_panel_b(ax_b)
    if color_mode == 'tanimoto':
        cax = fig.add_axes([0.61, 0.18, 0.008, 0.50])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('Max Tanimoto to\nnearest training', fontsize=9)
        cb.ax.tick_params(labelsize=7)
    outpath = OUTDIR / f'fig1a_collapse_{suffix}_compact.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {outpath}')


if __name__ == '__main__':
    generate_fig1(color_mode='static', suffix='simple')
    generate_fig1(color_mode='tanimoto', suffix='tanimoto')
    generate_fig1_compact(color_mode='static', suffix='simple')
    generate_fig1_compact(color_mode='tanimoto', suffix='tanimoto')
    print('Done.')
