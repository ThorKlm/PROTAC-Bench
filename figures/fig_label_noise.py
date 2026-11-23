#!/usr/bin/env python
"""Label-noise sensitivity, with the cross-publication bound as a reference.

Reads a label-noise artefact and results/cascade_intervals.json. Series labels
come from each artefact's own note field. The fit is drawn only over the range
actually measured; the bound is drawn as a horizontal reference at the fitted
intercept minus the bound, and no intersection is projected onto the axis,
because the linear form cannot hold towards a flip rate of fifty percent where
training labels become random. The projected rate is printed with the measured
range beside it.

Usage: fig_label_noise.py [noise.json ...]
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = sys.argv[1:] or ['results/label_noise.json', 'results/label_noise_both.json']
CASC = 'results/cascade_intervals.json'
OUT = Path(__file__).parent / 'fig_label_noise.pdf'
COLS = ['#2166AC', '#E08214']
plt.rcParams.update({'font.size': 9, 'figure.dpi': 300,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1})

bound = json.load(open(CASC))['bound']['point']
fig, ax = plt.subplots(figsize=(4.6, 3.3))
report = []
for i, p in enumerate(SRC):
    d = json.load(open(p))
    x = np.array([q['flip'] for q in d['points']]) * 100
    y = np.array([q['macro'] for q in d['points']])
    e = np.array([q['sample_sd'] for q in d['points']])
    fit = d['linear_fit']
    col = COLS[i % len(COLS)]
    ax.errorbar(x, y, yerr=e, fmt='o', ms=4, lw=0, elinewidth=0.8, capsize=2,
                color=col, label=d.get('note', Path(p).stem), zorder=4)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, fit['intercept'] + fit['slope_per_percent'] * xs, lw=1.1,
            color=col, alpha=0.9, zorder=3)
    ax.axhline(fit['intercept'] - bound, ls='--', lw=0.9, color=col, alpha=0.6,
               zorder=2)
    report.append((p, fit, x.min(), x.max(), bound))
ax.set_xlabel('training labels flipped (percent)')
ax.set_ylabel('LOTO macro AUROC')
ax.legend(frameon=False, fontsize=7.5, loc='lower left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(OUT)
plt.close(fig)
print(f'  Saved {OUT}')
print(f'  cascade bound {bound:.4f} from {CASC}')
for p, fit, lo, hi, b in report:
    proj = b / abs(fit['slope_per_percent'])
    print(f'  {p}')
    print(f'    intercept {fit["intercept"]:.4f} slope_per_percent '
          f'{fit["slope_per_percent"]:.5f} rms_residual {fit["rms_residual"]:.4f}')
    print(f'    reference level {fit["intercept"] - b:.4f}  '
          f'projected flip rate {proj:.1f} percent  measured range {lo:.0f} to {hi:.0f}')
