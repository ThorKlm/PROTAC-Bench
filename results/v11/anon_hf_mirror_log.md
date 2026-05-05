# Anonymized HF Mirror Creation — BLOCKED

**Date:** 2026-05-02
**Task:** Create double-blind anonymized HuggingFace mirror of PROTAC-Bench for the
NeurIPS 2026 Datasets & Benchmarks submission (deadline 2026-05-06).
**Outcome:** **BLOCKED at artifact verification (Task step 1).** No anonymous
namespace was created and no upload was attempted.

---

## Verification result

The task required confirmation that the PROTAC-Bench subrepository at
`/workspace/PROTAC-Bench/` contains seven publication-scope artifacts. Only six are
present, and one of those is named differently than the task requested. Detail:

| # | Required artifact | Status | Path on disk |
|---|---|---|---|
| 1 | Canonical dataset CSV | PRESENT | `data/protac_bench.csv` (1,504,412 B; sha256 `6d273d…30fa` per croissant) |
| 2 | **All four fold assignment files** | **MISSING** | Only `data/loto_folds.json` exists (one file, 121,394 B). No additional fold/split JSONs are present in `data/` and none are documented in `README.md`, `DATASHEET.md`, or `croissant.json`. |
| 3 | Canonical RF baseline evaluation script | PRESENT | `baselines/rf_morgan.py` (anchor baseline, expected 0.668 ± 0.005) |
| 4 | Few-shot calibration script | AMBIGUOUS | No script with "calibration" in its name. Closest matches: `signals/fewshot.py` (RF vs MAML across k=1,3,5,10) and `signals/fewshot_strategies.py` (selection-strategy comparison). Neither is canonically named a "calibration" script. |
| 5 | Variance-share decomposition script | AMBIGUOUS | No script named "variance-share decomposition." Closest matches: `figures/render_fig_variance_attribution_v15.py` (figure renderer, reads `results/variance_attribution.json`) and `hpo/fanova_analysis.py` (fANOVA importance). The decomposition itself appears to live in the JSON; the producing script is not in the subrepo. |
| 6 | Croissant.json with 20 of 20 RAI fields | PRESENT | `data/croissant.json` contains 20 `rai:*` keys (counted: dataCollection, dataCollectionType, dataCollectionTimeframe, dataCollectionRawData, dataCollectionMissingData, dataAnnotationProtocol, dataAnnotationPlatform, dataAnnotationAnalysis, dataAnnotationPerItemTime, dataAnnotationDemographics, dataAnnotationTools, dataPreprocessingProtocol, dataPreprocessingImputation, dataPreprocessingManipulation, dataUseCases, dataLimitation, dataBiases, dataSocialImpact, personalSensitiveInformation, dataReleaseMaintenancePlan). |
| 7 | README | PRESENT | `README.md` (4,225 B). Note: README is already author-stripped (anonymized), but it documents only the single `loto_folds.json` fold file — corroborating that the four-fold expectation is not met by the current subrepo. |

### Primary blocker

**Item 2: Only one fold assignment file is present where four were required.**

- `data/loto_folds.json` is the sole fold file. Inspection confirms it is a single
  dict keyed by 65 UniProt accessions encoding the LOTO splits.
- `README.md` lists only this file under `data/`.
- `croissant.json` `distribution[]` declares only three FileObjects:
  `protac_bench.csv`, `loto_folds.json`, `admet_scores.csv` — no scaffold-split,
  random-split, temporal-split, or cross-E3-split fold file is registered.
- `DATASHEET.md` files list matches: `protac_bench.csv`, `loto_folds.json`,
  `admet_scores.csv`, `croissant.json`. No additional fold files.

If the four expected folds are LOTO + scaffold + random + temporal (or an
equivalent set used in the manuscript), three of those fold-assignment artifacts
have not been materialized into `data/` and have not been registered in the
Croissant manifest. Releasing the mirror in this state would publish a dataset
that contradicts both its own README and its own machine-readable metadata.

### Secondary uncertainties (would not alone block, but compound the risk)

- The "few-shot calibration script" and "variance-share decomposition script"
  have no exact-name match in the subrepo. The candidates listed above plausibly
  fill those roles, but a reviewer following the task description literally would
  not find them. These should be either renamed or explicitly cross-referenced in
  the README before the anonymized mirror is published.

---

## Why no anonymous namespace was created

Per task spec step 2: *"If any artifact is missing or fails reproducibility
verification, document the blocker and stop."*

Because item 2 above is missing, I did **not**:

- call the HuggingFace Hub API to create `anonymous-neurips2026` or
  `anon-tpd-bench-2026`,
- upload any files,
- run `mlcroissant validate` against a live URL,
- fabricate a URL or a validator verdict.

Additionally, `data/croissant.json` already advertises a public URL of
`https://huggingface.co/datasets/anonymous-neurips2026/protac-bench`. Before
creating that namespace, the contents of `data/` must match what the manifest
claims is being distributed; otherwise the live `mlcroissant validate` call in
step 5 would either fail (if checksums are recomputed against missing files) or
falsely succeed (if it only validates the present three FileObjects), neither of
which produces the audit trail this task is meant to provide.

---

## What needs to happen before this task can be retried

1. Either (a) materialize the three missing fold-assignment files
   (e.g. `scaffold_folds.json`, `random_folds.json`, `temporal_folds.json` —
   exact set determined by the manuscript's evaluation protocol) into
   `data/`, OR (b) reconcile the task description with the actual benchmark
   design if only one fold file is intentional.
2. Register every fold file in `data/croissant.json` `distribution[]` with a
   recomputed sha256 and a one-line description.
3. Update `README.md` and `DATASHEET.md` `Files:` lists to enumerate every
   fold file.
4. Disambiguate the "few-shot calibration" and "variance-share decomposition"
   script names — either rename the existing scripts or add a `Mapping` section
   to `README.md` that explicitly names which file plays which role for
   reviewers.
5. Re-run `bash reproduce.sh` end-to-end against the regenerated `data/` to
   confirm the manuscript's headline numbers still reproduce; capture the run
   log alongside this file.
6. Only after the above land, retry the anonymous mirror task. At that point
   the namespace creation will need an HF token bound to a credential that does
   not deanonymize the authors — none is configured in this environment, so
   that prerequisite must also be supplied at retry time.

---

**Status: BLOCKED — missing 3 of 4 required fold assignment files; ambiguous mapping for 2 of 7 named script artifacts.**
**No live URL produced. No validator output produced.**
