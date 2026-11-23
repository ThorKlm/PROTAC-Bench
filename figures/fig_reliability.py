#!/usr/bin/env python
"""Reliability of the reference arm, raw against Platt-rescaled.

Reads results/v13/calibration_C0.json. Bins are the ones the reported ECE is
computed over, so the curve and the metric cannot disagree. Points are the
across-seed mean of each bin's mean prediction and observed frequency, the
vertical bars the across-seed spread, and the lower panel the bin occupancy
that weights the ECE.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = 'results/v13/calibration_C0.json'
OUT = Path(__file__).parent / 'fig_reliability.pdf'
C_RAW, C_CAL, C_GREY = '#2166AC', '#E08214', '#999999'
plt.rcParams.update({'font.size': 9, 'figure.dpi': 300,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1})

d = json.load(open(SRC))
ps = d['per_seed']
seeds = list(ps)
nb = len(ps[seeds[0]]['reliability_raw'])

def series(key):
    mp = np.full((len(seeds), nb), np.nan)
    ob = np.full((len(seeds), nb), np.nan)
    nn = np.zeros((len(seeds), nb))
    for i, s in enumerate(seeds):
        for j, b in enumerate(ps[s][key]):
            nn[i, j] = b['n']
            if b['mean_p'] is not None:
                mp[i, j], ob[i, j] = b['mean_p'], b['obs']
    return mp, ob, nn

fig, (ax, axn) = plt.subplots(2, 1, figsize=(4.4, 5.0), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
plotted = {}
ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color=C_GREY, zorder=1)
for key, col, lab in (('reliability_raw', C_RAW, 'raw'),
                      ('reliability_platt', C_CAL, 'Platt')):
    mp, ob, nn = series(key)
    # Platt rescaling empties some bins in every seed; take means only over the
    # columns that have data, and a spread only where at least two seeds do
    valid = np.isfinite(mp).sum(0)
    m, m2 = valid > 0, valid > 1
    x = np.full(nb, np.nan); y = np.full(nb, np.nan); e = np.zeros(nb)
    x[m] = np.nanmean(mp[:, m], 0); y[m] = np.nanmean(ob[:, m], 0)
    e[m2] = np.nanstd(ob[:, m2], 0, ddof=1)
    plotted[lab] = (x, y, e, nn.mean(0), valid)
    # marker area tracks bin occupancy, so sparsely populated bins carry the
    # visual weight they carry in the ECE rather than equal weight
    nm = nn.mean(0)[m]
    ax.plot(x[m], y[m], lw=1.0, color=col, alpha=0.8, zorder=2)
    ax.errorbar(x[m], y[m], yerr=e[m], fmt='none', ecolor=col, elinewidth=0.8,
                capsize=2, alpha=0.7, zorder=2)
    ax.scatter(x[m], y[m], s=6 + 90 * nm / max(nm.max(), 1), color=col,
               edgecolors='white', linewidths=0.4, label=lab, zorder=4)
    if key == 'reliability_raw':
        axn.bar(np.arange(nb)[m] / nb + 0.5 / nb, nn.mean(0)[m],
                width=0.9 / nb, color=C_GREY, edgecolor='none')
ece = d['aggregate']
ax.set_ylabel('observed frequency')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(frameon=False, loc='upper left', fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
axn.set_xlabel('mean predicted probability')
axn.set_ylabel('rows per bin')
axn.spines['top'].set_visible(False); axn.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(OUT)
plt.close(fig)
print(f'  Saved {OUT}')
print(f'  ECE-10 raw {ece["ece10_raw"]["mean"]:.4f} '
      f'platt {ece["ece10_platt"]["mean"]:.4f} '
      f'temperature {ece["ece10_temperature"]["mean"]:.4f}')
print(f'  Brier raw {ece["brier_raw"]["mean"]:.4f}  bins {nb}  seeds {len(seeds)}')
for lab, (x, y, e, n, valid) in plotted.items():
    print(f'  {lab}:')
    for j in range(nb):
        if valid[j]:
            print(f'    bin {j} pred {x[j]:.4f} obs {y[j]:.4f} '
                  f'spread {e[j]:.4f} rows {n[j]:.0f} seeds {int(valid[j])}')
        else:
            print(f'    bin {j} empty in all seeds')
