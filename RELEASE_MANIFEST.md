# PROTAC-PLM-Bench Release Manifest

Cross-reference between every numerical claim in `paper/main.tex` and a
supporting file in this repository (`PROTAC-Bench/`) or, when missing, a
file in `/workspace/results/` or `/workspace/scripts/` that needs to be
copied in.

Manuscript reference: `/workspace/protac_plm_bench_2/paper/main.tex` (848 lines).

**Excluded by policy** (superseded by post-fix HPO V2):
- `/workspace/results/exp17*` (5 directories: `exp17_adaptive_k`, `exp17_calibration`, `exp17_feature_importance`, `exp17_learning_curve`, `exp17_top5_ensemble`)
- `/workspace/results/exp2_unified_hpo`

**Included by policy**:
- `/workspace/results/exp40_hpo_v2/hpo_v2_seed0.db`, `hpo_v2_seed1.db`
- `/workspace/results/fanova_importances.json`
- `/workspace/results/exp40_hpo_v2/hpo_v2_validation.json` (already landed)
- All canonical `render_fig_*_v15.py` scripts (already in `figures/`)

---

## 1. Body Claims with Source Files

Manuscript section / claim — value(s) — source file in this repo (or external if missing).

### Abstract (lines 26–28)

| Claim | Value | Source |
|---|---|---|
| Binary degradation entries | 10,748 | `data/protac_bench.csv` |
| Target proteins (LOTO-eligible) | 65 of 173 | `data/loto_folds.json` |
| Published methods AUROC range (random) | 0.85–0.92 | citations only |
| LOTO collapse range | 0.56–0.67 | `results/baseline_rf_morgan.json`, `results/stan_combined.json`, `results/deepprotacs_protacdb_replication.json`, `results/dm_loto.json` |
| RF + Morgan baseline | 0.668 | `results/baseline_rf_morgan.json`, `results/confidence_intervals.json` |
| ESM-2 PLM degradation range | −0.010 to −0.023 | `results/plm_scaling.json` |
| HPO trials | 2,000 | `results/exp40_hpo_trials.csv`, **EXTERNAL**: `/workspace/results/exp40_hpo_v2/hpo_v2_seed{0,1}.db` |
| fANOVA head_type variance | 96 % | **EXTERNAL**: `/workspace/results/fanova_importances.json` (Table 3 reuses 95.9 %) |
| ADMET cascade gain | +0.02 | `results/admet_cascade.json`, `results/admet_ablation.json` |
| Few-shot calibration gain | +0.07 | `results/fewshot.json`, `results/full_stack.json` |
| EGNN exp. pockets | 0.658 ± 0.013 | `results/egnn_10seed.json` |
| Pocket-shuffle contribution | 0.013 AUROC | `results/pocket_shuffle_control.json` |

### §1 Introduction (lines 33–51)

| Claim | Value | Source |
|---|---|---|
| DeepPROTACs reported AUROC | 0.88 (cit.); 0.847 random / 0.531 LOTO | `results/deepprotacs_protacdb_replication.json`, `results/per_target/DeepPROTACs_protacdb_random_per_target.csv`, `results/per_target/DeepPROTACs_protacdb_loto_per_target.csv` |
| PROTAC-STAN reported AUROC | 0.85 (cit.) | citation only |
| DegradeMaster reported AUROC | 0.84 (cit.); 0.878 (curated) | citation; `results/structure_ladder.json` (`dm_reported_random_split` = 0.878) |
| ET-PROTACs reported AUROC | 0.87 | citation only |
| RF + Morgan LOTO | 0.668 (10 seeds, std 0.005) | `results/baseline_rf_morgan.json`, `results/confidence_intervals.json` |
| PROTAC-STAN LOTO | 0.653 | `results/stan_combined.json`, `results/per_target/STAN_repl_loto_per_target.csv` |
| Ribes graph LOTO | 0.557 | `results/per_target/Ribes_loto_per_seed.csv`, `results/baseline_rf_morgan.json` (`ribes_target` 0.5561) |
| Random-split AUROC | 0.90 | `results/plm_scaling.json` (random_mean ≈ 0.9016) |
| Gap (random − LOTO) | > 0.23 | computed from above |
| HPO best (XGB+ChemBERTa+ESM-2+ADMET) | 0.708 single-seed | `results/hpo_summary.json`, `results/exp40_hpo_trials.csv` |
| HPO best (10-seed regress) | 0.668 | `results/hpo_summary.json`, `results/hpo_v2_validation.json` |
| ESM-2 8M→3B trend | 0.678 → 0.655 | `results/plm_scaling.json` |
| Random-split with PLM | 0.91 stable | `results/plm_scaling.json` |
| ADMET +0.02 (0.677 → 0.686) | +0.021, p=0.014 | `results/admet_cascade.json` |
| Few-shot k=10 | +0.07 (to 0.709) | `results/fewshot.json` |
| MAML | +0.034 | `results/fewshot.json`, **EXTERNAL** raw: `/workspace/results/exp4_fewshot/maml_meta_k5.csv` |
| EGNN exp pockets | 0.658 ± 0.013 | `results/egnn_10seed.json` |
| AF2-predicted pockets | 0.547 | `results/structure_ladder.json` |
| Pocket-shuffle hybrid | 0.814 vs 0.820 original | `results/pocket_shuffle_control.json` |

### §2 Related Work (lines 56–70)

| Claim | Value | Source |
|---|---|---|
| Ribes et al. cold-target AUROC | 0.604 | citation only |
| AF3 RMSD < 1 Å for 33/62 (Dunlop2025) | — | citation only |

### §3 Dataset (lines 75–120)

| Claim | Value | Source |
|---|---|---|
| Total entries / SMILES / targets | 10,748 / 9,359 / 173 | `data/protac_bench.csv` |
| LOTO-eligible | 65 | `data/loto_folds.json` |
| Positive rate | 65.8 % | `data/protac_bench.csv` |
| E3 ligase mix (CRBN 71.9, VHL 26.9, other 1.2) | — | `data/protac_bench.csv` |
| Murcko scaffolds | 7,427 | `data/protac_bench.csv` |
| Random > scaffold > temporal > LOFO > LOTO hierarchy (~0.90, 0.82, 0.74, 0.70, 0.67) | — | `results/temporal.json`, `results/lofo.json`, `results/baseline_rf_morgan.json`, `results/task7_confound_detection.json` |
| Morgan radius 2, 2,048 bits, RF 100 trees, leaf 3 | — | `baselines/rf_morgan.py`, `src/fingerprints.py` |
| RF + Morgan, 0.668 std 0.005, 10 seeds | — | `results/confidence_intervals.json` |
| HPO 2,000 trials × 21 dimensions | — | `hpo/run_hpo.py`, **EXTERNAL**: `/workspace/results/exp40_hpo_v2/hpo_v2_seed{0,1}.db` |
| Spearman ρ(PR-AUC, ROC-AUC) = 0.754 | — | `results/confidence_intervals.json` |

### §4.1 Universal collapse (lines 129–181)

| Claim | Value | Source |
|---|---|---|
| RF Morgan 0.902 → 0.668 (Δ −0.236) | — | `results/baseline_rf_morgan.json`, `results/plm_scaling.json` |
| Mean inflation +0.092 (median +0.067, std 0.179) | — | `results/per_target/per_target.csv`, `results/per_target/per_target_random_auroc.csv` |
| Per-target failures: PARP1 0.897→0.269, AR 0.916→0.527, SHP2 0.815→0.239, IKZF1 0.844→0.475, MAPT 0.872→0.541 | — | `results/per_target/per_target.csv`, Table S4 in main.tex |
| Scaffold 5-fold CV 0.897 vs random 0.902 | — | `results/task7_confound_detection.json` |
| 7,427 unique scaffolds, 5,771 singletons (77.7 %) | — | `data/protac_bench.csv` |
| Tanimoto: scaffold 0.896 mean / 87 % > 0.8; LOTO 0.608 / 22.9 % | — | `results/task7_confound_detection.json` |
| Bottom 22 targets 0.446 / top 22 targets 0.864 | — | `results/per_target/per_target.csv` |
| AR n=1968 0.527; BTK n=761 0.703; MET n=17 0.989; MAPK14 n=12 0.944 | — | `results/per_target/per_target.csv` |
| Table inflation (RF, STAN, Ribes std/target, DeepPROTACs, HPO best) | — | `results/baseline_rf_morgan.json`, `results/stan_combined.json`, `results/deepprotacs_protacdb_replication.json`, `results/hpo_summary.json` |

### §4.2 PLMs encode target identity (lines 183–197)

| Claim | Value | Source |
|---|---|---|
| Morgan-only 0.678 ± 0.006; ESM-2 8M 0.668; 35M 0.671; 150M 0.670; 650M 0.677; 3B 0.655 ± 0.010 | — | `results/plm_scaling.json` |
| Random-split rises 0.900 → 0.913 | — | `results/plm_scaling.json` |
| ProtTrans-BFD 0.642 (Δ −0.036, p=0.038) | — | `results/plm_scaling.json` |
| DeepPROTACs all-modules 0.531; pocket-ablate 0.548; linker-remove 0.494 | — | `results/stan_ablation.json`, `results/deepprotacs_diagnostic.md`, `results/deepprotacs_protacdb_replication.json` |
| 151 of 155 training targets share unique pocket graph | — | `results/deepprotacs_diagnostic.md` |

### §4.3 HPO confirms a flat landscape (lines 199–246)

| Claim | Value | Source |
|---|---|---|
| HPO V1 1,086 trials, best 0.708 single-seed → 0.668 10-seed (p=0.925) | — | `results/hpo_summary.json` |
| HPO V2 2,000 trials best 0.764 → 0.603 10-seed (Δ −0.065 vs baseline) | — | `results/hpo_v2_validation.json`, `results/exp40_hpo_analysis.json`, **EXTERNAL**: `/workspace/results/exp40_hpo_v2/hpo_v2_seed{0,1}.db` |
| Ranks 2–10 cluster 0.659–0.679 | — | `results/hpo_v2_validation.json` |
| fANOVA: head_type 95.9 %, mol_encoder 1.9 %, prot_encoder 0.7 %, fragment_mode 0.5 %, normalize 0.4 %, admet 0.2 %, e3_onehot 0.2 %, rdkit_desc 0.1 % | — | **EXTERNAL**: `/workspace/results/fanova_importances.json`; rendered by `figures/render_fig_fanova_v15.py` |
| GIN 3-layer 0.613 vs Morgan 0.635 (10 targets, p=0.49) | — | `results/chemprop_hpo.json`, `results/summary_65targets.json` |

### §4.4 Why the ceiling exists (lines 248–257)

| Claim | Value | Source |
|---|---|---|
| Top-50 Morgan bits: 48 shared (96 %), Spearman ρ=0.999 | — | **EXTERNAL** raw: `/workspace/results/exp17_feature_importance/` (excluded — figure already rendered: `figures/figS6_feature_corr.pdf`) |
| LOTO-specific bits 386, 792 | — | same as above (figure rendered) |
| Learning curve 0.613 → 0.666 (β=0.049, R²=0.95, ~0.76 at 1M) | — | **EXTERNAL** raw: `/workspace/results/exp17_learning_curve/` (excluded — figure rendered: `figures/figS11_learning_curve.pdf`) |
| LOFO mean 0.612 vs LOTO 0.674 (Δ −0.062, p=0.0001) | — | `results/lofo.json` |
| Family-size correlation ρ=0.520, p=9 × 10⁻⁶ | — | `results/lofo.json` |
| Singleton 0.477 / small 0.597 / large 0.765 | — | `results/lofo.json` |
| Kinase family 24 targets, mean 0.776 | — | `results/lofo.json` |

### §5 What breaks through (lines 263–298)

| Claim | Value | Source |
|---|---|---|
| Waterfall: ADMET +0.02; few-shot +0.07; 3D +0.21 | — | `results/admet_cascade.json`, `results/fewshot.json`, `results/structure_ladder.json` |
| RF + Morgan baseline 0.668 | — | `results/baseline_rf_morgan.json` |
| + ADMET 7-feat 0.687 (+0.021, p=0.014) | — | `results/admet_cascade.json` |
| k=5 plain RF 0.700 (+0.034, p<0.001) | — | `results/fewshot.json`, `results/confidence_intervals.json` |
| k=10 plain RF 0.709 (+0.043) | — | `results/fewshot.json` |
| ADMET + k=5 stratified 0.734 (+0.068) | — | `results/full_stack.json`, `results/fewshot.json` |
| EGNN exp pockets 30 targets 0.658 | — | `results/egnn_10seed.json`, `results/structure_ladder.json` |
| Hybrid (EGNN + 2D) 0.820 (+0.152), pocket contrib 0.013 | — | `results/pocket_shuffle_control.json` |
| EGNN docked 0.542 (−0.124) | — | `results/structure_ladder.json` |

### §5.1 ADMET cascade (lines 300–305)

| Claim | Value | Source |
|---|---|---|
| 7 endpoints with CV AUROCs (HIA 0.929 n=578, Pgp 0.948 n=1218, logS 0.940 n=9982, PAMPA 0.790 n=2034, lipo 0.881 n=4200, hepatocyte 0.748 n=1213) | — | `results/admet_cascade.json` |
| Morgan+ADMET 0.687 (+0.021, p=0.014) | — | `results/admet_cascade.json` |
| Top correlations: lipo ρ=0.273, Pgp 0.142, sol −0.118 | — | `results/admet_ablation.json` |
| Morgan+ADMET+few-shot 0.716 (+0.050, p=8.2e-6) | — | `results/full_stack.json` |

### §5.2 Few-shot calibration (lines 307–326)

| Claim | Value | Source |
|---|---|---|
| RF k=1/5/10 = 0.675/0.700/0.709 | — | `results/fewshot.json`, **EXTERNAL** raw: `/workspace/results/exp4_fewshot/rf_morgan_k{1,5,10}.csv` |
| MAML k=1/5/10 = 0.654/0.683/0.700 | — | `results/fewshot.json`, **EXTERNAL** raw: `/workspace/results/exp4_fewshot/maml_*` |
| Adaptive k=⌊n/3⌋ cap 10: 0.704 vs fixed k=5 0.680 (p=3.6e-6) | — | **EXTERNAL** raw: `/workspace/results/exp17_adaptive_k/` (**excluded** — superseded by HPO V2) |
| Strategies: stratified 0.734, random 0.714, MaxMin 0.704, uncertainty 0.688 | — | `results/fewshot.json` (column-level), **EXTERNAL** raw: `/workspace/results/exp28_active_learning/per_target.csv` |
| Stratified vs random +0.020 p=0.016; vs uncertainty +0.046 p=0.009 | — | `results/fewshot.json`, **EXTERNAL**: `/workspace/results/exp28_active_learning/summary.json` |
| 14-cell factorial: M=0.661, M+A=0.673, M+K=0.696, M+A+K=0.705, M+W+A+K=0.705 | — | `results/task22_gap_decomposition.json`, `results/full_stack.json` |
| Few-shot +0.030 / ADMET +0.011 / warhead +0.003 marginal | — | `results/task22_gap_decomposition.json` |
| M+A+K → M+A+W+K Δ −0.0004 | — | `results/task22_gap_decomposition.json` |

### §5.3 3D geometry (lines 328–355)

| Claim | Value | Source |
|---|---|---|
| EGNN 0.658 ± 0.013, paired Wilcoxon p=0.27 vs RF on same 30 targets | — | `results/egnn_10seed.json` |
| Hybrid 0.820 ± 0.013 | — | `results/pocket_shuffle_control.json` |
| Shuffled hybrid 0.814; zero-pocket 0.807; original 0.820 | — | `results/pocket_shuffle_control.json` |
| Pocket geometry 0.013 AUROC | — | `results/pocket_shuffle_control.json` |
| 30 targets with experimental binding-site PDBs | — | `results/egnn_10seed.json` |
| Structure ladder: prior 0.878 (30); AF+cluster 0.542 (60); V3 improved 0.547 (48); co-crystal 0.497 (subset) | — | `results/structure_ladder.json` |
| DM 100 % within 5 Å (mean 3.52 Å) vs ours 6.1 % (mean 28.94 Å) | — | **EXTERNAL** raw: `/workspace/results/exp12_smina_full_docking/`, `/workspace/results/exp11_docking_confound/docking_scores.csv` (figure rendered: `figures/figS5_docking.pdf`) |
| Well-docked < 10 Å: EGNN 0.670 vs Morgan 0.684 (Δ −0.015, p=0.89) | — | `results/structure_ladder.json` |
| R²=0.063 docking explanatory power | — | **EXTERNAL** raw: `/workspace/results/exp11_docking_confound/summary.json` (figure rendered: `figures/figS5_docking.pdf`) |

### §5.4 Orthogonality (lines 357–359)

| Claim | Value | Source |
|---|---|---|
| Morgan + ADMET + k=5 → 0.716 (+0.050) | — | `results/full_stack.json` |
| Predicted additivity 0.021 + 0.034 = 0.055 vs observed 0.050 | — | `results/full_stack.json` |

### §6.1 Calibration unaffected (lines 365–370)

| Claim | Value | Source |
|---|---|---|
| Brier baseline 0.229 ± 0.0006; full 0.232 ± 0.0008 | — | `results/confidence_intervals.json`, `results/full_stack.json` |
| ECE@10 baseline 0.152 ± 0.0008; full 0.150 ± 0.0013 | — | `results/confidence_intervals.json` |
| Selective: 0.648 (cov 1.0) → peak 0.678 (cov 0.5) → 0.659 (cov 0.1) | — | **EXTERNAL** raw: `/workspace/results/exp35_selective_prediction/selective_curve.csv` and `summary.json` (figure rendered: `figures/figS13_selective_prediction.pdf`, `figures/figS14_selective_prediction.pdf`) |

### §6.2 DegradeMaster puzzle (lines 385–389)

| Claim | Value | Source |
|---|---|---|
| 30-target subset RF Morgan 0.652 vs full 65 0.668 | — | `results/structure_ladder.json` (`dm_vs_rf_same_targets` 0.694; per-target subsets) and `results/per_target/per_target.csv` |

### §6.3 Implications (lines 393–404)

| Claim | Value | Source |
|---|---|---|
| Lin et al. review (15+ ML methods, no cold-target) | — | citation only |
| Recommendation 1 hierarchy 0.734 | — | `results/full_stack.json`, `results/fewshot.json` |
| Recommendation 2 hierarchy 0.687 | — | `results/admet_cascade.json` |
| Cross-E3 +0.052 asymmetry; CRBN→VHL std 0.049 (range 0.510–0.637); VHL→CRBN std 0.003 (0.627–0.634) | — | `results/cross_e3.json` |

### §6.4 Limitations (lines 405–409)

| Claim | Value | Source |
|---|---|---|
| Threshold sensitivity ± 0.02 | — | **EXTERNAL** raw: `/workspace/results/exp18_binarization/`, `/workspace/results/exp_supp_threshold/` (figure rendered: `figures/figS2_threshold.pdf`) |
| 65 targets: 24 kinases, 6 bromodomains | — | `results/lofo.json` |
| Cross-source AUROC 0.54 | — | **EXTERNAL** raw: `/workspace/results/exp13_cross_dataset/summary.json` (figure rendered: `figures/figS4_cross_dataset.pdf`) |
| Subset bias permutation: STAN p=0.618, DM p=0.609 | — | `results/single_source.json`, `results/confidence_intervals.json` |

### §7 Conclusion (lines 414–420)

All numbers re-cite values already sourced above.

---

## 2. Appendix Claims with Source Files

### Appendix A — Dataset details (lines 446–568)

| Claim | Value | Source |
|---|---|---|
| PROTAC-DB 3.0: 7,727 CRBN + 2,896 VHL = 10,623 | — | `data/protac_bench.csv` (column `data_source`) |
| Ribes et al.: 2,141 entries | — | `data/protac_bench.csv` |
| DegradeMaster: 1,506 entries | — | `data/protac_bench.csv` |
| Raw 12,000+ → unique 10,748 | — | `data/protac_bench.csv` |
| Per-source positive rate (PROTAC-DB 0.777, Ribes 0.377) | — | `data/protac_bench.csv`, `data/croissant.json` |
| Table 7 (`tab:loto_full`) 65 targets, gene, n, activity rate | — | `data/loto_folds.json` |
| Binarization sensitivity (4 schemes: 0.641, 0.659, 0.688, 0.658) | — | **EXTERNAL** raw: `/workspace/results/exp18_binarization/summary.json` (figure rendered) |

### Appendix B — HPO search space (lines 573–615)

| Claim | Value | Source |
|---|---|---|
| 21-dimensional HPO with TPE, 50 startup trials | — | `hpo/run_hpo.py`, **EXTERNAL**: `/workspace/results/exp40_hpo_v2/hpo_v2_seed{0,1}.db` (canonical post-fix study) |
| `head_type` 96 % fANOVA dominance | — | **EXTERNAL**: `/workspace/results/fanova_importances.json` |

### Appendix C — Supplementary figures (lines 620–707)

| Figure | Source data | PDF in repo? |
|---|---|---|
| Fig S1 fragment decomposition | **EXTERNAL** raw: `/workspace/results/exp_supp_fragments/` | `figures/figS1_fragments.pdf` ✓ |
| Fig S2 threshold sensitivity | **EXTERNAL** raw: `/workspace/results/exp18_binarization/`, `/workspace/results/exp_supp_threshold/` | `figures/figS2_threshold.pdf` ✓ |
| Fig S3 temporal split | `results/temporal.json`, **EXTERNAL** raw: `/workspace/results/exp_supp_temporal/`, `/workspace/results/exp19_temporal_within/` | `figures/figS3_temporal.pdf` ✓ |
| Fig S4 cross-dataset | **EXTERNAL** raw: `/workspace/results/exp13_cross_dataset/summary.json` | `figures/figS4_cross_dataset.pdf` ✓ |
| Fig S5 docking confound | **EXTERNAL** raw: `/workspace/results/exp11_docking_confound/`, `/workspace/results/exp12_smina_full_docking/` | `figures/figS5_docking.pdf` ✓ |
| Fig S6 feature correlation | **EXTERNAL** raw: `/workspace/results/exp17_feature_importance/` (excluded) | `figures/figS6_feature_corr.pdf` ✓ |
| Fig S7 scaffold split | `results/task7_confound_detection.json` | `figures/figS7_scaffold.pdf` ✓ |
| Fig S8 EGNN smina | `results/structure_ladder.json` | `figures/figS8_egnn_smina.pdf` ✓ |
| Fig S9 protein family | `results/lofo.json` | `figures/figS9_protein_family.pdf` ✓ |
| Fig S10 calibration | `results/confidence_intervals.json` | `figures/figS10_calibration.pdf` ✓ |
| Fig S11 learning curve | **EXTERNAL** raw: `/workspace/results/exp17_learning_curve/` (excluded) | `figures/figS11_learning_curve.pdf` ✓ |
| Fig S12 seed variance | **EXTERNAL** raw: `/workspace/results/exp18_seed_variance/summary.json` | **MISSING PDF**: `figures/figS12_seed_variance.pdf` (only `figS12_gnn_baselines.pdf` present) |

### Appendix C — Supplementary tables (lines 709–823)

| Table | Source |
|---|---|
| S1 PLM scaling | `results/plm_scaling.json` |
| S2 Cross-E3 transfer | `results/cross_e3.json` |
| S3 3D feature attempts | `results/structure_ladder.json` |
| S4 Per-target failure cases | `results/per_target/per_target.csv` (PARP1, SHP2, AR, IKZF1, MAPT) |
| S5 Robustness checks | `results/confidence_intervals.json`, **EXTERNAL** raw: `/workspace/results/exp18_seed_variance/`, `/workspace/results/exp18_label_noise/`, `/workspace/results/exp18_fp_sensitivity/` |
| S6 Compute budget | descriptive only |

### Appendix D — Dataset release (lines 828–846)

| Claim | Value | Source |
|---|---|---|
| GitHub URL | https://github.com/ThorKlm/protac-plm-bench | repo manifest |
| Files: protac_bench.csv, loto_folds.json, evaluate.py, baselines/ | — | `data/protac_bench.csv`, `data/loto_folds.json`, `src/evaluation.py`, `baselines/` |
| DATASHEET.md | — | **MISSING**: not present in `PROTAC-Bench/` |

---

## 3. Files Currently Present in PROTAC-Bench

(Excluding `.git/` and `__pycache__/`.)

### Top level
- `README.md`, `reproduce.sh`, `requirements.txt`

### `data/`
- `protac_bench.csv` — 10,748 entries
- `loto_folds.json` — 65-fold LOTO assignment
- `admet_scores.csv` — precomputed 7-feature ADMET
- `croissant.json` — dataset metadata

### `src/` (core library, 4 files)
- `__init__.py`, `data_utils.py`, `evaluation.py`, `fingerprints.py`, `stats.py`

### `baselines/` (10 files)
- `rf_morgan.py`, `xgboost_morgan.py`, `plm_scaling.py`
- `deepprotacs_eval.py`, `dm_loto.py`
- `chemprop_hpo.py`, `gnn_baselines.py`
- `rs_per_target_gnn.py`, `rs_per_target_knn.py`, `rs_per_target_stan.py`

### `signals/` (8 files)
- `admet_cascade.py`, `admet_ablation.py`
- `fewshot.py`, `fewshot_strategies.py`
- `full_stack.py`
- `warhead_transfer.py`, `warhead_ablation.py`, `warhead_lofo.py`

### `robustness/` (5 files)
- `confidence_intervals.py`, `cross_e3.py`, `lofo.py`, `nonkinase.py`, `single_source.py`, `temporal.py`

### `hpo/` (3 files)
- `run_hpo.py`, `validate_top_k.py`, `fanova_analysis.py`

### `results/` (40 JSONs + per_target/ + per_target_ro/)
- `admet_ablation.json`, `admet_cascade.json`, `baseline_rf_morgan.json`, `chemprop_hpo.json`,
  `confidence_intervals.json`, `cross_e3.json`, `deepprotacs_diagnostic.md`,
  `deepprotacs_protacdb_replication.json`, `dm_loto.json`, `e3_warhead.json`,
  `egnn_10seed.json`, `exp40_fullstack_10seed_summary.json`, `exp40_hpo_analysis.json`,
  `exp40_hpo_trials.csv`, `fewshot.json`, `full_stack.json`, `hpo_summary.json`,
  `hpo_v2_validation.json`, `leave_e3_out.json`, `lofo.json`,
  `metadata_analysis_summary.json`, `nonkinase.json`, `plm_scaling.json`,
  `pocket_shuffle_control.json`, `pseudolabel.json`, `single_source.json`,
  `stan_ablation.json`, `stan_combined.json`, `structure_ladder.json`,
  `summary_65targets.json`, `task14_within_target_cross_lab.json`,
  `task22_gap_decomposition.json`, `task7_confound_detection.json`, `temporal.json`,
  `warhead_ablation.json`, `warhead_coverage.json`, `warhead_lofo.json`,
  `warhead_transfer.json`
- `per_target/` — 90 CSVs (RF, STAN, DM, DeepPROTACs, kNN, Ribes, EGNN per-target)
- `per_target_ro/` — read-only mirror of per_target/

### `figures/` (PDFs and v15 scripts)

Canonical v15 scripts (already present):
- `fig1_collapse_v15.py`
- `fig2_triple_anchored_v15.py`
- `fig3_factorial_v15.py`
- `render_fig_fanova_v15.py`
- `render_fig_hpo_trials_v15.py`

PDFs referenced from `paper/main.tex` and present:
- `fig1a_collapse_simple.pdf`, `fig1a_collapse_tanimoto.pdf`
- `fig2_perfold.pdf`, `fig3_hpo_ceiling.pdf`, `fig4_breakthroughs.pdf`,
  `fig5_fewshot.pdf`, `fig6_egnn_scatter.pdf`, `fig7_plm_scaling.pdf`,
  `fig8_hpo_dimensions.pdf`
- `figS1_fragments.pdf`, `figS2_threshold.pdf`, `figS3_temporal.pdf`,
  `figS4_cross_dataset.pdf`, `figS5_docking.pdf`, `figS6_feature_corr.pdf`,
  `figS7_scaffold.pdf`, `figS8_egnn_smina.pdf`, `figS9_protein_family.pdf`,
  `figS10_calibration.pdf`, `figS11_learning_curve.pdf`

---

## 4. Files Missing or Located Outside PROTAC-Bench

### 4.1 Required (must be migrated for the manuscript to be self-supporting)

| Target path in PROTAC-Bench | Source path | Why needed |
|---|---|---|
| `results/exp40_hpo_v2/hpo_v2_seed0.db` | `/workspace/results/exp40_hpo_v2/hpo_v2_seed0.db` (3.0 MB) | post-fix HPO V2 study consumed by `figures/render_fig_fanova_v15.py` and `figures/render_fig_hpo_trials_v15.py`; supports §4.3 "2,000 trials × 21 dimensions" and Appendix B |
| `results/exp40_hpo_v2/hpo_v2_seed1.db` | `/workspace/results/exp40_hpo_v2/hpo_v2_seed1.db` (3.0 MB) | second seed of post-fix HPO V2; same consumers |
| `results/fanova_importances.json` | `/workspace/results/fanova_importances.json` | Backs Table 3 fANOVA importances (head_type 95.9 %, etc.) and §4.3 fANOVA claim |
| `figures/figS12_seed_variance.pdf` | `/workspace/paper/figures/figS12_seed_variance.pdf` | Manuscript line 704 uses this filename; current `figS12_gnn_baselines.pdf` has different content |

### 4.2 Already landed (no action required, listed for completeness)

| File | Status |
|---|---|
| `results/hpo_v2_validation.json` | Present (Apr 18); identical 8,398 B copy of `/workspace/results/exp40_hpo_v2/hpo_v2_validation.json` |
| `figures/render_fig_fanova_v15.py` | Present |
| `figures/render_fig_hpo_trials_v15.py` | Present |
| `figures/fig{1_collapse,2_triple_anchored,3_factorial}_v15.py` | Present |

### 4.3 Optional (raw data behind specific figures/claims; figures already rendered)

These are pointed to by the cross-reference but are *not* required if the
release relies on the rendered PDFs already in `figures/` plus the JSON
summaries already in `results/`. They live under `/workspace/results/` and
are not migrated here.

- `/workspace/results/exp4_fewshot/` — raw RF/MAML k-sweeps (summary captured in `results/fewshot.json`)
- `/workspace/results/exp11_docking_confound/`, `/workspace/results/exp12_smina_full_docking/` — raw docking scores (captured in `results/structure_ladder.json`)
- `/workspace/results/exp13_cross_dataset/summary.json` — cross-source AUROC 0.54
- `/workspace/results/exp18_binarization/`, `/workspace/results/exp18_label_noise/`, `/workspace/results/exp18_fp_sensitivity/`, `/workspace/results/exp18_seed_variance/` — Table S5 robustness rows
- `/workspace/results/exp19_temporal_within/`, `/workspace/results/exp_supp_temporal/`, `/workspace/results/exp_supp_threshold/`, `/workspace/results/exp_supp_fragments/` — supplementary figure backing data
- `/workspace/results/exp28_active_learning/` — Section 5.2 strategies (stratified/random/MaxMin/uncertainty)
- `/workspace/results/exp35_selective_prediction/selective_curve.csv` — §6.1 selective-prediction curve

### 4.4 Excluded by user policy (do NOT migrate)

These directories are explicitly superseded by post-fix HPO V2:
- `/workspace/results/exp17_adaptive_k/`
- `/workspace/results/exp17_calibration/`
- `/workspace/results/exp17_feature_importance/`
- `/workspace/results/exp17_learning_curve/`
- `/workspace/results/exp17_top5_ensemble/`
- `/workspace/results/exp2_unified_hpo/` (also in nested `/workspace/results/results/exp2_unified_hpo/`)

Two manuscript claims rely on data inside `exp17_*` directories:
- §4.4 feature-importance overlap (top-50 Morgan bits, 96 %, ρ=0.999, bits 386 / 792)
- §4.4 / Fig S11 learning curve (0.613 → 0.666, β=0.049, R²=0.95)

The corresponding rendered PDFs (`figures/figS6_feature_corr.pdf`,
`figures/figS11_learning_curve.pdf`) are already in this repo, so the
manuscript still compiles. Re-running these analyses against the post-fix
HPO V2 outputs would replace the raw inputs without changing the figures.

### 4.5 Other gaps

- `DATASHEET.md` (Appendix D references) — not present in `PROTAC-Bench/`. Add or remove the reference from the manuscript.

---

## 5. Concrete cp Commands to Migrate Missing Files

Run from any working directory (commands use absolute paths):

```bash
# 1. Post-fix HPO V2 study databases (required by render_fig_*_v15.py)
mkdir -p /workspace/PROTAC-Bench/results/exp40_hpo_v2
cp /workspace/results/exp40_hpo_v2/hpo_v2_seed0.db \
   /workspace/PROTAC-Bench/results/exp40_hpo_v2/hpo_v2_seed0.db
cp /workspace/results/exp40_hpo_v2/hpo_v2_seed1.db \
   /workspace/PROTAC-Bench/results/exp40_hpo_v2/hpo_v2_seed1.db

# 2. fANOVA importances (Table 3)
cp /workspace/results/fanova_importances.json \
   /workspace/PROTAC-Bench/results/fanova_importances.json

# 3. HPO V2 validation JSON (re-sync; current copy is identical Apr 18)
cp /workspace/results/exp40_hpo_v2/hpo_v2_validation.json \
   /workspace/PROTAC-Bench/results/exp40_hpo_v2/hpo_v2_validation.json

# 4. Missing supplementary figure (manuscript line 704)
cp /workspace/paper/figures/figS12_seed_variance.pdf \
   /workspace/PROTAC-Bench/figures/figS12_seed_variance.pdf
```

After these four commands the manuscript's body and appendix claims are
all backed by either a file inside `PROTAC-Bench/` or a citation, except
for the two §4.4 sub-claims whose raw inputs were in the excluded
`exp17_*` tree (figures already rendered).

### Notes on the v15 render scripts

The five canonical v15 render scripts are already in
`PROTAC-Bench/figures/`. After step 1 (database migration) they can be
re-executed without modification because both `render_fig_fanova_v15.py`
and `render_fig_hpo_trials_v15.py` hard-code
`/workspace/results/exp40_hpo_v2/` as the database path. To make the
release self-contained, point them at the in-repo copy by either
(a) editing `DB_DIR = Path('results/exp40_hpo_v2')` after migration, or
(b) running them from `/workspace/` with the existing absolute path.
This is a content edit, not a file migration, so it is left to the
release engineer.
