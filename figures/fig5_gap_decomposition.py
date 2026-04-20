import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path('/workspace/PROTAC-Bench/results')

d0_mean, d0_std = 0.8637, 0.0075
d2_mean, d2_std = 0.8236, 0.0102
d1_mean, d1_std = 0.6356, 0.1903
d7_mean, d7_std = 0.6239, 0.0268

try:
    d = json.load(open(RESULTS / 'task22_gap_decomposition.json'))
    for entry in d.get('conditions', d.get('results', [])):
        if isinstance(entry, dict):
            tag = entry.get('tag', entry.get('condition', '')).lower()
            a = entry.get('auroc', entry.get('mean_auroc', entry.get('macro_mean_auroc', 0)))
            s = entry.get('std', entry.get('auroc_std', 0))
            if 'd0' in tag or ('random' in tag and 'no' not in tag and 'lab' not in tag): d0_mean, d0_std = a, s
            elif 'd2' in tag or 'no_doi' in tag or 'no-doi' in tag: d2_mean, d2_std = a, s
            elif 'd1' in tag or 'loto' in tag: d1_mean, d1_std = a, s
            elif 'd7' in tag or 'lab_rate_only' in tag or 'lab_only' in tag: d7_mean, d7_std = a, s
except Exception as e:
    print(f'  Using hardcoded values: {e}')

labels = ["Random CV\n(standard)", "Random CV\n(no lab leak)", "LOTO\n(cold target)", "Lab rate only\n(random CV)"]
means = [d0_mean, d2_mean, d1_mean, d7_mean]
stds = [d0_std, d2_std, d1_std, d7_std]
colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']

fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.bar(range(4), means, yerr=stds, capsize=4, color=colors, edgecolor='black', linewidth=0.6, width=0.65, error_kw={'linewidth': 1.0})
lab_gap = d0_mean - d2_mean
tgt_gap = d2_mean - d1_mean
total = d0_mean - d1_mean
ax.annotate('', xy=(0, d2_mean), xytext=(0, d0_mean), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax.text(-0.35, (d0_mean+d2_mean)/2, f'Lab\n{lab_gap:.3f}\n({lab_gap/total:.0%})', ha='right', va='center', fontsize=6.5)
ax.annotate('', xy=(1, d1_mean), xytext=(1, d2_mean), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax.text(1.35, (d2_mean+d1_mean)/2, f'Target\n{tgt_gap:.3f}\n({tgt_gap/total:.0%})', ha='left', va='center', fontsize=6.5)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m+s+0.012, f'{m:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('AUROC', fontsize=9)
ax.set_ylim(0.45, 0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0.5, color='gray', ls='--', lw=0.5, alpha=0.5)
ax.set_title('Gap Decomposition: Target Novelty vs Lab Leakage', fontsize=9, pad=10)
plt.tight_layout()
out = Path('/workspace/PROTAC-Bench/figures/fig5_gap_decomposition.pdf')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out}')
