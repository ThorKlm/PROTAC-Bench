#!/usr/bin/env python
"""Maximum Tanimoto similarity to the nearest training compound, by protocol.

Reads results/v13/similarity.json. Left panel the splits as conventionally
run, right panel the same splits with the deduplication rule applied, so the
comparison across protocols and the effect of the rule are both visible in one
figure. Densities come from the stored hundred-bin histograms.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = 'results/v13/similarity.json'
OUT = Path(__file__).parent / 'fig_similarity.pdf'
CONDS = [('random_cv', 'random CV', '#2166AC'),
         ('scaffold_cv', 'scaffold CV', '#92C5DE'),
         ('loto', 'LOTO', '#E08214')]
plt.rcParams.update({'font.size': 9, 'figure.dpi': 300,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1})

d = json.load(open(SRC))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True, sharey=True)
rows = []
for ax, (tag, title) in zip(axes, (('retained', 'as conventionally run'),
                                   ('dedup', 'deduplicated'))):
    for base, lab, col in CONDS:
        s = d[f'{base}_{tag}']
        c = np.asarray(s['hist_counts'], float)
        e = np.asarray(s['hist_edges'], float)
        mid, wid = (e[:-1] + e[1:]) / 2, e[1] - e[0]
        ax.plot(mid, c / c.sum() / wid, lw=1.3, color=col, label=lab)
        rows.append((tag, lab, s['n'], s['median'], s['q95'],
                     s['frac_ge_090'], s['frac_eq_1']))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('max Tanimoto to nearest training compound')
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
axes[0].set_ylabel('density')
axes[0].legend(frameon=False, fontsize=8, loc='upper left')
plt.tight_layout()
fig.savefig(OUT)
plt.close(fig)
print(f'  Saved {OUT}')
for tag, lab, n, med, q95, ge, eq in rows:
    print(f'  {tag:9s} {lab:12s} n {n:6d} median {med:.4f} q95 {q95:.4f} '
          f'ge090 {ge:.4f} eq1 {eq:.4f}')
