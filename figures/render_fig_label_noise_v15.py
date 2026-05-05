#!/usr/bin/env python3
"""Synthetic label-noise sensitivity figure.
Single panel: LOTO macro-mean AUROC vs uniform-random label flip rate (%),
with per-seed dots, error bars, fitted regression line, and an inter-laboratory
reference band intersected with the regression to project an equivalent flip
rate onto the x-axis. Visual conventions match fig2_triple_anchored_v15.py.
"""
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams.update({
    'font.size': 11, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'xtick.major.width': 1, 'ytick.major.width': 1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUTDIR = Path(__file__).parent
INPUT = Path('/workspace/protac_plm_bench_2/results_summary/exp18_label_noise_extended.json')

C_DBLUE = '#2166AC'
C_LBLUE = '#92C5DE'
C_ORANGE = '#E08214'
C_LORANGE = '#FDD49E'
C_GREY = '#999999'

INTERLAB_DELTA = 0.124


def generate_fig():
    data = json.load(open(INPUT))
    levels = data['noise_levels']
    slope_per_pct = data['loto_slope_per_pct']

    noise_pct = np.array([lv['noise_fraction'] * 100.0 for lv in levels])
    means = np.array([lv['loto_mean'] for lv in levels])
    stds = np.array([lv['loto_std'] for lv in levels])
    seeds = [lv['loto_seeds'] for lv in levels]

    # Fresh unweighted linear regression through the 7 LOTO mean points
    actual_slope, actual_intercept = np.polyfit(noise_pct, means, 1)

    # Inter-laboratory floor anchored to regression intercept for geometric
    # consistency: the regression line, horizontal floor, and vertical guide
    # all meet at a single intersection point by construction.
    floor = actual_intercept - INTERLAB_DELTA
    actual_intersection_x = INTERLAB_DELTA / abs(actual_slope)

    x_line = np.linspace(0.0, 25.0, 200)
    y_line = actual_slope * x_line + actual_intercept

    # Standard error of prediction for the regression line. Band is narrowest
    # at x_mean and widens at the edges (and far into extrapolation).
    y_pred = actual_slope * noise_pct + actual_intercept
    residuals = means - y_pred
    n = len(means)
    resid_std = float(np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1)))
    x_mean = float(np.mean(noise_pct))
    Sxx = float(np.sum((noise_pct - x_mean) ** 2))
    se_pred = resid_std * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / Sxx)

    fig, ax = plt.subplots(figsize=(6.3, 3.8))

    # Inter-laboratory reference horizontal at regression-anchored floor
    ax.axhline(floor, color=C_ORANGE, ls='--', lw=1.1, alpha=0.85,
               label='inter-laboratory bound (0.124 AUROC equivalent)')

    # Standard error of prediction band around the regression line
    ax.fill_between(x_line, y_line - se_pred, y_line + se_pred,
                    color=C_DBLUE, alpha=0.2, lw=0, zorder=1,
                    label='regression std error of prediction')

    # Regression line through the 7 LOTO points (only line in figure)
    ax.plot(x_line, y_line, color=C_DBLUE, lw=1.2, alpha=0.55, ls='-',
            zorder=2,
            label=f'linear regression (slope {actual_slope:.4f} AUROC per %)')

    # Vertical projection guide at the actual regression/floor intersection
    ax.axvline(actual_intersection_x, color=C_GREY, ls=':', lw=1.0, alpha=0.85,
               zorder=1,
               label=f'equivalent flip rate {actual_intersection_x:.1f}%')

    # Marker at projection intersection (regression line ∩ floor)
    ax.plot([actual_intersection_x], [floor], marker='o',
            markersize=6, mfc='white', mec=C_ORANGE, mew=1.4, zorder=5)

    # Per-seed dots overlaid on each error bar
    for i, sds in enumerate(seeds):
        xs = np.full(len(sds), noise_pct[i])
        ax.scatter(xs, sds, s=14, color=C_LBLUE, edgecolor=C_DBLUE,
                   linewidth=0.5, alpha=0.85, zorder=3)

    # Scatter markers + error bars only (no connecting line)
    ax.errorbar(noise_pct, means, yerr=stds, fmt='o',
                color=C_DBLUE, ecolor=C_DBLUE, elinewidth=1.0,
                capsize=3, capthick=1.0, markersize=5,
                mfc=C_DBLUE, mec='black', markeredgewidth=0.6, zorder=4,
                label='LOTO macro-mean ± std (3 seeds)')

    # Annotation for projection point — placed above the intersection marker,
    # with ~0.02 vertical offset above the regression line
    ax.text(actual_intersection_x, floor + 0.02,
            f'equivalent flip rate $\\approx${actual_intersection_x:.1f}%',
            fontsize=8.5, ha='center', va='bottom', color='black')

    ax.set_xlabel('Uniform-random label flip rate (%)', fontsize=10)
    ax.set_ylabel('LOTO macro-mean AUROC', fontsize=10)
    ax.set_xlim(0.0, 25.0)
    y_min = min(floor, float(np.min(means - stds))) - 0.02
    y_max = float(np.max(means + stds)) + 0.025
    ax.set_ylim(y_min, y_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=9)

    ax.legend(loc='upper right', fontsize=8.5, frameon=False)
    plt.tight_layout()

    outpath = OUTDIR / 'figS_label_noise_v15.pdf'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    size_bytes = outpath.stat().st_size
    se_at_0 = resid_std * np.sqrt(1.0 / n + (0.0 - x_mean) ** 2 / Sxx)
    se_at_proj = resid_std * np.sqrt(
        1.0 / n + (actual_intersection_x - x_mean) ** 2 / Sxx)
    print(f'  Saved {outpath} ({size_bytes} bytes)')
    print(f'  actual_slope     = {actual_slope:.10f}')
    print(f'  actual_intercept = {actual_intercept:.10f}')
    print(f'  floor            = {floor:.10f}')
    print(f'  actual_intersection_x = {actual_intersection_x:.6f}')
    print(f'  resid_std (sigma_resid) = {resid_std:.10f}')
    print(f'  x_mean = {x_mean:.6f}, Sxx = {Sxx:.6f}')
    print(f'  band half-width at x=0%             = {se_at_0:.10f}')
    print(f'  band half-width at x={actual_intersection_x:.2f}% (proj) = {se_at_proj:.10f}')
    print(f'  legend handles = {len(ax.get_legend().get_texts())}')
    return outpath, size_bytes, actual_slope, actual_intercept, floor, actual_intersection_x


if __name__ == '__main__':
    generate_fig()
    print('Done.')
