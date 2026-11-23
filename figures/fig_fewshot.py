#!/usr/bin/env python
"""Few-shot curve across k for the full feature set.

Reads the cells and arms that carry each k: cell_1110 for k of zero and
cell_1111 for k of five, the fewshot arms for one, three and ten. Error bars
are the target-clustered standard error each artefact records, which is the
scale the rest of the project reports effects against. The k of ten point sits
on a smaller cohort, because ten shots leave too few rows in some folds, and is
drawn open with its cohort size stated rather than silently alongside the rest.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = {0: 'results/v13/cell_1110.json', 1: 'results/v13/fewshot_k1.json',
       3: 'results/v13/fewshot_k3.json', 5: 'results/v13/cell_1111.json',
       10: 'results/v13/fewshot_k10.json'}
OUT = Path(__file__).parent / 'fig_fewshot.pdf'
C_AUC, C_AP, C_GREY = '#2166AC', '#E08214', '#999999'
plt.rcParams.update({'font.size': 9, 'figure.dpi': 300,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1})

rows = []
for k, p in SRC.items():
    d = json.load(open(p))
    rows.append((k, d['macro_auroc'], d['mean_ap'], d['target_se'],
                 d['per_target_sd'], len(d['per_target'])))
rows.sort()
k = np.array([r[0] for r in rows], float)
auc = np.array([r[1] for r in rows])
ap = np.array([r[2] for r in rows])
se = np.array([r[3] for r in rows])
nt = np.array([r[5] for r in rows])
full = nt == nt.max()

fig, ax = plt.subplots(figsize=(4.4, 3.2))
for y, col, lab in ((auc, C_AUC, 'macro AUROC'), (ap, C_AP, 'mean average precision')):
    ax.plot(k, y, lw=1.2, color=col, zorder=2)
    ax.errorbar(k, y, yerr=se, fmt='none', ecolor=col, elinewidth=0.8,
                capsize=2, alpha=0.8, zorder=2)
    ax.scatter(k[full], y[full], s=34, color=col, zorder=4, label=lab)
    ax.scatter(k[~full], y[~full], s=34, facecolors='white', edgecolors=col,
               linewidths=1.2, zorder=4)
for i in np.flatnonzero(~full):
    ax.annotate(f'{int(nt[i])} of {int(nt.max())} targets',
                (k[i], min(auc[i], ap[i])), textcoords='offset points',
                xytext=(-4, -14), ha='right', fontsize=7, color=C_GREY)
ax.set_xlabel('labelled compounds from the held-out target (k)')
ax.set_ylabel('score')
ax.set_xticks(k)
ax.legend(frameon=False, fontsize=8, loc='lower right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(OUT)
plt.close(fig)
print(f'  Saved {OUT}')
for kk, a, p_, s_, pts, n in rows:
    print(f'  k {kk:2d} n_targets {n:2d} macro {a:.4f} mean_ap {p_:.4f} '
          f'target_se {s_:.4f} per_target_sd {pts:.4f}')
