#!/usr/bin/env python3
"""Render fANOVA importance bar chart from post-fix HPO V2 study databases.

Reads results/exp40_hpo_v2/hpo_v2_seed{0,1}.db, computes fANOVA
on random-phase trials (first 200 per seed, pre-TPE), averages across seeds,
and writes the half-width bar chart to figures/.
No interpretive annotations.
"""
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'xtick.major.width': 1, 'ytick.major.width': 1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})
C_DBLUE = '#2166AC'
C_LBLUE = '#92C5DE'
C_GREY  = '#999999'
N_RANDOM = 200
DB_DIR = Path('results/exp40_hpo_v2')
OUT_DIR = Path('figures')

def fanova_per_seed(seed):
    db = DB_DIR / f'hpo_v2_seed{seed}.db'
    s = optuna.load_study(study_name=f'hpo_v2_seed{seed}', storage=f'sqlite:///{db}')
    completed = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rand = sorted(completed, key=lambda t: t.number)[:N_RANDOM]
    sub = optuna.create_study(direction='maximize')
    for t in rand:
        sub.add_trial(t)
    ev = FanovaImportanceEvaluator(seed=42)
    return get_param_importances(sub, evaluator=ev)

def main():
    per_seed = {s: fanova_per_seed(s) for s in (0, 1)}
    all_keys = sorted(set().union(*[d.keys() for d in per_seed.values()]))
    means = {k: np.mean([per_seed[s].get(k, 0.0) for s in (0, 1)]) for k in all_keys}
    items = sorted(means.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    vals_pct = [v * 100 for _, v in items]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    colors = [C_DBLUE if v >= 10 else (C_LBLUE if v >= 5 else C_GREY) for v in vals_pct]
    ax.barh(range(len(labels)), vals_pct, color=colors,
            edgecolor='black', linewidth=0.6)
    for i, v in enumerate(vals_pct):
        ax.text(v + 0.4, i, f'{v:.1f}%', va='center', fontsize=8.5, fontweight='bold')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('fANOVA variance attribution (%)', fontsize=10)
    ax.set_xlim(0, max(vals_pct) + 5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / 'fig4_fanova_v15.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    print('Mean importances (across 2 seeds, random-phase n=200 each):')
    for k, v in zip(labels, vals_pct):
        print(f'  {k:24s} {v:5.1f}%  per-seed: {per_seed[0].get(k,0)*100:.1f}% / {per_seed[1].get(k,0)*100:.1f}%')

if __name__ == '__main__':
    main()
