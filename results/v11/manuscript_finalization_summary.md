# Manuscript Finalization Summary (May 6 Submission)

Targeted extraction from four v11 result files. Run date 2026-05-02.

---

## File 1 — exp64_v2_tpddb_overlap.json (TPDdb vs PROTAC-Bench overlap)

- smiles_intersection_count: **7,052**
- smiles_overlap_pct_of_tpddb: **32.909%**
- smiles_overlap_pct_of_pb: **75.35%**
- uniprot_intersection_count: **117**
- tpd_ids_with_any_DC50_or_Dmax_measurement: **2,634**
- loto_eligibility_tpddb_only.loto_eligible_target_count: **10**
- comparison_to_pb.pb_loto_eligible_targets_v11_paper: **78**
- comparison_to_pb.pb_loto_eligible_targets_v11_audit_doc: **65**
- augmentation_potential.distinct_targets_added_by_augmentation: **3**
- augmentation_potential.tpddb_measurable_pairs_not_in_pb: **53**

---

## File 2 — exp70_matched_cohort_fewshot.json (matched-cohort few-shot)

- matched_cohort target list size: **14**
- k=0 baseline macro mean (matched cohort, 10 seeds): **0.6634**, across-target std: **0.1199**
- k=100 macro mean: **0.7777**
- within_cohort_improvement_k100_minus_k0 (mean): **+0.1143**
- bootstrap 95% CI for k100−k0: **[0.0788, 0.1523]** (B=10,000, target-clustered)
- within_cohort_tail_slope_auroc_per_doubling_k: **0.0325** (CI95 [0.0211, 0.0453])
- paired_wilcoxon_p_vs_canonical_baseline: **NOT PRESENT in file** — field absent; only bootstrap CIs are reported. Flag for manuscript: either add Wilcoxon analysis or rely on bootstrap CI (excludes 0).

---

## File 3 — exp71b_degrademaster_seed_expansion.json (DegradeMaster seed expansion)

- pilot_3seed.mean: **0.8010**, std_across_targets: **0.1248** (n=27 targets)
- fast_protocol_consistent_multiseed.across_seed_macro_mean: **0.6816**
- fast_protocol_consistent_multiseed.across_seed_macro_std_ddof1: **0.00951**
- fast_protocol_consistent_multiseed.n_seeds: **7** (seeds 42, 43, 44, 53, 71, 89, 97)
- verdict: *"Across-seed std of macro AUROC under the multi-seed fast protocol is 0.0095 (<0.02) -> 3-seed pilot under-sampled seed coverage; the multi-seed result is the correct comparator."*

---

## File 4 — croissant_live_validation.txt (live HF Croissant validation)

- VALIDATOR_EXIT: **0** (pass)
- RAI_FIELD_COUNT (populated, substantive): **20**
- Live URL validated: **https://huggingface.co/datasets/ThorKl/protac-bench/raw/main/croissant.json**
- Schema verdict: **PASS** (HF Croissant API endpoint also returned HTTP 200)
- Errors / warnings: **No errors.** One cosmetic warning only — non-standard `@context` key `equivalentProperty` (rdf.py:89). No blocker for submission.
