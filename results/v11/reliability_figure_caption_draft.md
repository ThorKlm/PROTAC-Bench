# Reliability Figure (figS_reliability_v17_lowess_density.pdf) — Metadata & Draft Caption

## 1. Figure provenance

- **Figure file:** `/workspace/PROTAC-Bench/figures/figS_reliability_v17_lowess_density.pdf`
  (vector PDF, 300 DPI, 6.3 in × 3.8 in, ~30 KB; created 2026-05-02 06:22 UTC).
- **Render script:** `/workspace/exp93_reliability_lowess_density/render_fig_reliability_v17_lowess_density.py`
- **Per-grid LOWESS / band / density / linewidth values:**
  `/workspace/exp93_reliability_lowess_density/lowess_density_curves.csv`
- **Lineage:** fourth aesthetic alternative for the v17 reliability diagram
  (after `figS_reliability_v17.pdf` binned, `figS_reliability_v17_hybrid.pdf`
  binned + LOWESS overlay, and `figS_reliability_v17_density.pdf` jittered
  scatter). This variant drops the binned-marker layer entirely so the
  density signal is carried by the LOWESS linewidth alone.

## 2. Substantive figure content

### Curves rendered (three series + bands)

| Layer | What it represents | Visual encoding |
|-------|--------------------|-----------------|
| Diagonal `y = x` | Perfect-calibration reference | dashed gray `#999999`, lw 1.0, α 0.7 |
| **C0 — Morgan baseline** | Reliability curve of the canonical RF/Morgan-2048 baseline | LOWESS polyline in `#2166AC` (dark blue) with per-segment linewidth modulated 0.5–4.0 pt by local sample density; 95 % bootstrap band α 0.20 |
| **C3 — Full-stack** | Reliability curve of the full pipeline: Morgan + warhead transfer + ADMET 7-feat + few-shot k=5 | LOWESS polyline in `#E08214` (orange) with per-segment linewidth modulated 0.5–4.0 pt by local sample density; 95 % bootstrap band α 0.20 |

### Axes

- **x:** "Mean predicted probability" on `[0, 1]`, ticks at `linspace(0,1,6)`.
- **y:** "Empirical fraction positive" on `[0, 1]`, ticks at `linspace(0,1,6)`.
- Equal aspect, top/right spines hidden, frameless legend (lower right).

### Binning / smoothing protocol

- **Not** the equal-width 10-bin binning used by the v17 base figure. This
  variant uses a **continuous LOWESS smooth** of the per-prediction stream:
  `statsmodels.nonparametric.smoothers_lowess.lowess(y, p, frac=…, it=0)`
  evaluated on a 199-point grid in `[0.005, 0.995]`.
- **Bandwidth selection:** Silverman's rule of thumb on the per-prediction
  `p` vector — `h = 0.9 · min(σ̂, IQR/1.34) · n^(-1/5)` — converted to a
  LOWESS `frac` by counting the mean fraction of points within ±h of each
  grid anchor. Numerics:

  | Condition | n  | Silverman h | LOWESS frac |
  |-----------|-----|-------------|-------------|
  | C0 baseline | 94,280 | 0.016706 | 0.034955 |
  | C3 full-stack | 94,280 | 0.018935 | 0.039685 |

  These are byte-identical to the bandwidths used in `exp84_reliability_hybrid`,
  so the centerline overlays the v17-hybrid centerline exactly.

### Density encoding (the distinctive feature of this variant)

- Per-prediction `p` density along the x-axis is estimated as a
  **histogram-smoothed pdf**: 100 equal-width bins on `[0, 1]`, normalized to
  density, then post-smoothed with `gaussian_filter1d(σ = 2 bins ≈ 0.02 in
  p-units)`, interpolated to the LOWESS grid.
- Density is mapped to LOWESS linewidth via **square-root** scaling onto
  `[0.5, 4.0] pt` per condition (mirrors the `s = √count` marker-area
  convention of the v17 binned figure). Each curve is rescaled to its own
  range, so within-curve density variation is visible at a glance; absolute
  cross-curve density comparison is *not* the visual claim.
- **No** raw scatter points, **no** binned circles, **no** jittered
  individual predictions — density is conveyed by linewidth alone.

### Bootstrap confidence band

- 1,000 non-parametric paired `(p_i, y_i)` resamples, n = 94,280 per replicate.
- LOWESS frac fixed to the point estimate, `it = 0` (no robustifying
  re-weighting); 2.5 / 97.5 percentile band, fill alpha 0.20, line color
  matching the curve. RNG seed `numpy.random.default_rng(20260501)`.

## 3. Underlying data source

- **NPZ files:** `/workspace/overnight/results/calibration/probas/{C0,C3}_seed{7,13,29,42,43,44,53,71,89,97}.npz`.
- **Pooling:** 65 LOTO targets × 10 canonical seeds → flat (94,280,) per-condition
  vectors of `(p_i, y_i)` pairs. Same canonical 10-seed LOTO cohort used
  throughout the manuscript's headline-table results.
- **Probability scale:** **raw** RF outputs as predicted probability of the
  positive class — `p = P(degrader = 1)`, **not** `max(p, 1-p)` confidence
  and **not** post-Platt rescaled. The Platt / isotonic / temperature-scaled
  variants are reported separately in Table M-Calibration; this figure is the
  *raw, uncalibrated* reliability curve that motivates the recalibration analysis.
- **Positive class fraction** is ~equal for the two streams (both pool the
  same 65-target × 10-seed test sets, only the model differs).

## 4. Key visual feature being highlighted

Both curves lie **systematically above the diagonal across the low- and
mid-confidence range** (`p ≲ 0.7`) — the model under-predicts the realised
positive rate when it is unsure — and both **invert below the diagonal in
the top decile** (`p ≳ 0.85` for C0, `p ≳ 0.9` for C3) — the model
*over*-predicts in the highest-confidence stratum. The inversion is wider
than the 95 % bootstrap band in the top decile for both curves, so it is
not a sampling artefact. The full-stack curve sits closer to the diagonal
than the Morgan baseline for `p ∈ [0.2, 0.8]` (lower raw ECE-10), but the
two curves cross in the top decile and the full-stack curve is *more*
inverted there — this is the high-confidence inversion that drives the
non-monotonic risk-coverage knee at `coverage ≈ 0.30` reported in the
Appendix M body.

## 5. Relationship to ECE and risk-coverage findings in Appendix M body text

The raw-ECE-10 numbers tabulated in the Appendix M calibration table
(`raw 0.150`, `Platt 0.031`, `isotonic 0.058`, `temperature 0.154`) summarise
the *vertical distance* between these LOWESS curves and the diagonal,
weighted by the bin masses that the linewidth here makes visible. Platt
scaling drops ECE below the conventional `0.05` well-calibrated threshold
because it can re-warp the bulk of the curve where most of the predictive
mass sits (the thick-line region around `p ≈ 0.2`–`0.6`). What Platt
*cannot* fix is the high-confidence inversion: Platt is monotone, so it
preserves the *ranking* of confidences, and the non-monotonic risk–coverage
profile (risk re-rises below `coverage ≈ 0.30`) survives every recalibrator
tested in `exp87_calibration_recovery`. The figure visually anchors that
claim — the inversion is intrinsic to the *ranking* of `p` for the most
confident predictions, not a recalibratable miscalibration of the *scale*.

## 6. Draft caption (~95 words, suitable for inline Appendix M placement)

> **Reliability diagram, baseline vs. full-stack pipeline (raw probabilities,
> 10-seed canonical LOTO, 65-target cohort).** Empirical fraction positive
> against mean predicted probability for the canonical Morgan-2048 baseline
> (blue, C0) and the full-stack pipeline (Morgan + warhead transfer +
> ADMET 7-feat + few-shot $k{=}5$, orange, C3). Each curve is a LOWESS smooth
> ($n = 94{,}280$ per condition; Silverman $h \approx 0.017/0.019$, fractional
> spans $\approx 0.035/0.040$) of the pooled per-prediction $(p_i, y_i)$
> stream, with per-segment linewidth scaled to local sample density along
> the $p$-axis ($\sqrt{\,\cdot\,}$-mapped onto $[0.5, 4.0]\,\text{pt}$).
> Shaded bands are 95\% bootstrap intervals (1{,}000 paired resamples);
> the dashed diagonal is perfect calibration. Both curves under-predict the
> realised positive rate at low to moderate confidence ($p \lesssim 0.7$)
> and invert above the diagonal in the top decile ($p \gtrsim 0.9$) by more
> than the bootstrap band — the high-confidence inversion that drives the
> raw ECE-10 of $0.150$ for C3 (Table~\ref{tab:m_calibration}) and the
> non-monotonic risk--coverage knee at coverage $\approx 0.30$
> (Section~\ref{sec:calibration}); Platt scaling rescales the bulk of the
> curve but, being monotone, cannot remove the inversion.

## 7. Recommended Appendix M figure label

- **LaTeX label:** `\label{fig:m_reliability_lowess_density}`
- **Suggested numbering / display name:** Figure M.1 (first figure in the
  Calibration appendix). The render script's existing block uses
  `fig:m1_reliability_lowess_density`; either form is fine — match the
  capitalisation convention already in use in the manuscript's other
  appendix figure labels.
- **In-text reference template:**
  "Figure~\ref{fig:m_reliability_lowess_density} shows the per-prediction
  reliability of C0 and C3 under raw (uncalibrated) probabilities; Platt-
  rescaled curves are tabulated rather than plotted (see
  Table~\ref{tab:m_calibration})."
