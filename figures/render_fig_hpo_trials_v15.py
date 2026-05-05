#!/usr/bin/env python3
"""Render per-trial AUROC scatters across major HP dimensions from post-fix HPO V2.

Reads results/exp40_hpo_v2/hpo_v2_seed{0,1}.db. For each canonical
HP dimension, scatters trial-level AUROC against the categorical or numerical
HP value. Four-color scheme (distinct hues per phase, uniform alpha):
  orange       : seed 0 random-phase exploration (trial < N_RANDOM)
  gold         : seed 0 TPE-driven exploitation (trial >= N_RANDOM)
  darker blue  : seed 1 random-phase exploration
  lighter blue : seed 1 TPE-driven exploitation
Dense scatter layers are rasterized; axes/labels/legend remain vector.
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams.update({
    'font.size': 13, 'font.family': 'sans-serif',
    'axes.linewidth': 1.0, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})
C_S0_RAND = '#FFD932'  # orange (seed 0 exploration; preserved verbatim from v15)
C_S0_TPE  = '#FFA411'  # gold (seed 0 exploitation)
C_S1_RAND = '#0FB7FF'  # darker blue (seed 1 exploration)
C_S1_TPE  = '#4470FF'  # lighter blue (seed 1 exploitation)

N_RANDOM = 750
DB_DIR = Path('results/exp40_hpo_v2')
OUT_DIR = Path('figures')
PARAMS = ['head_type', 'mol_encoder', 'prot_encoder', 'fragment_mode',
          'ternary_attention', 'warhead_transfer']

def load_trials(seed):
    db = DB_DIR / f'hpo_v2_seed{seed}.db'
    s = optuna.load_study(study_name=f'hpo_v2_seed{seed}', storage=f'sqlite:///{db}')
    out = []
    for t in s.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        out.append({
            'number': t.number, 'auroc': t.value, 'params': dict(t.params),
            'phase': 'rand' if t.number < N_RANDOM else 'tpe',
        })
    return out

def panel(ax, trials_s0, trials_s1, param):
    cats = sorted({str(t['params'].get(param, 'NA')) for t in trials_s0 + trials_s1})
    cat_to_x = {c: i for i, c in enumerate(cats)}
    rng = np.random.RandomState(7)
    def plot(trials, color, label):
        xs, ys = [], []
        for t in trials:
            v = str(t['params'].get(param, 'NA'))
            if v not in cat_to_x:
                continue
            xs.append(cat_to_x[v] + rng.uniform(-0.22, 0.22))
            ys.append(t['auroc'])
        sc = ax.scatter(xs, ys, s=4, c=color, alpha=0.75, edgecolors='none', label=label, zorder=3)
        sc.set_rasterized(True)
    plot([t for t in trials_s0 if t['phase'] == 'rand'], C_S0_RAND, 'seed 0 rand')
    plot([t for t in trials_s0 if t['phase'] == 'tpe'],  C_S0_TPE,  'seed 0 TPE')
    plot([t for t in trials_s1 if t['phase'] == 'rand'], C_S1_RAND, 'seed 1 rand')
    plot([t for t in trials_s1 if t['phase'] == 'tpe'],  C_S1_TPE,  'seed 1 TPE')
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=10, rotation=30, ha='right')
    ax.set_title(param, fontsize=12)
    ax.set_ylim(0.40, 0.80)
    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.5)

def main():
    trials_s0 = load_trials(0)
    trials_s1 = load_trials(1)
    print(f'Loaded {len(trials_s0)} (seed 0) and {len(trials_s1)} (seed 1) completed trials')
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 6.0), sharey=True)
    for ax, p in zip(axes.flatten(), PARAMS):
        panel(ax, trials_s0, trials_s1, p)
    for r in range(n_rows):
        axes[r, 0].set_ylabel('LOTO AUROC (15-target)', fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    for handle in legend.legend_handles:
        handle.set_sizes([60])
        handle.set_alpha(0.8)
    fig.subplots_adjust(top=0.94, bottom=0.10, left=0.06, right=0.99,
                        hspace=0.55, wspace=0.10)
    out = OUT_DIR / 'fig8_hpo_dimensions_v15.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

if __name__ == '__main__':
    main()
