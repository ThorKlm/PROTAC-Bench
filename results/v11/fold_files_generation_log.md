# Fold Files Generation Log

**Date:** 2026-05-02
**Task:** Materialize the three missing fold-assignment artifacts that
complete the four-fold evaluation suite alongside the existing
`data/loto_folds.json`. This unblocks the anonymized HF mirror task.
**Generator:** `/workspace/PROTAC-Bench/scripts/generate_fold_files.py`

---

## Outputs

| File | Bytes | sha256 |
|---|---:|---|
| `data/loto_folds.json` (pre-existing) | 121,394 | `61564e68683db7c46424a5b6b58fe25d7cab319ee3dc7a300f86b7611bb4de3c` |
| `data/lofo_folds.json` (new) | 116,979 | `16d4d2fabe2ac84c0f7528128a49f94832eef92db096edd4f3bf8ee2ed32c8ee` |
| `data/cross_lab_folds.json` (new) | 40,152 | `3beb321d4a2e5e2365501b48a38d885cffc3fecf5b82a99c0c5b268bfc8bd964` |
| `data/temporal_prospective_folds.json` (new) | 19,862 | `a68950af03b1be29df7ddb47b373e234cd72190fb2101433b2f818f8aa6cc67b` |

All fold indices are 0-based row positions into `data/protac_bench.csv`.

## Provenance of metadata used to derive the splits

- `target_uniprot`, `label`, and row order are read from
  `data/protac_bench.csv` (which is the artifact being indexed).
- `doi` and `pub_year` are read from
  `/workspace/results/exp42_metadata_features/master_df.csv`,
  which is built by `/workspace/scripts/exp42_step0_build_master.py` and
  is row-aligned with `data/protac_bench.csv` via the `pb_index` column
  (verified: 10,748 rows, identical `target_uniprot` and `label` columns,
  `pb_index == range(N)`).
- `master_df.csv` is **not** redistributed in the published mirror; only
  the derived row-index splits are released. The three new fold files
  therefore expose grouping information without leaking the underlying
  DOI/year metadata.

---

## 1. `data/lofo_folds.json` — Leave-One-Family-Out

### Eligibility filter
- Targets must pass the LOTO eligibility filter
  (`src/data_utils.get_eligible_targets`: >=10 entries, activity rate in
  [0.1, 0.9]) — gives 65 targets.
- Targets must additionally appear in the named `FAMILY_MAP` defined in
  `robustness/lofo.py`. Four LOTO-eligible targets fall outside any named
  family and are excluded as unmapped singletons:
  `O43924`, `Q00534`, `Q9Y2I7`, `Q9Y616`.
- Result: **22 protein families covering 61 LOFO-eligible targets**.

### Fold structure
Each top-level key is a family name. Each entry contains
`family`, `targets`, `n_targets`, `n_entries`, `activity_rate`,
`test_indices`. The implied training set for a family fold is
`{0..N-1} \ test_indices`, where `N = 10,748`.

### Family roster (n_targets, n_entries, activity_rate)

| Family | n_targets | n_entries | activity_rate |
|---|---:|---:|---:|
| Bromodomain | 6 | 562 | 0.6851 |
| DUB | 1 | 19 | 0.3158 |
| E3_ligase | 1 | 26 | 0.3846 |
| GPX4 | 1 | 76 | 0.1974 |
| HAT_CBP_p300 | 3 | 274 | 0.7299 |
| HCFC1 | 1 | 91 | 0.6484 |
| HDAC | 3 | 143 | 0.3497 |
| IAP | 1 | 39 | 0.8462 |
| IDO1 | 1 | 109 | 0.1743 |
| Kinase | 24 | 2,530 | 0.6953 |
| NAMPT | 1 | 36 | 0.5000 |
| Nuclear_receptor | 2 | 3,259 | 0.7223 |
| PARP | 1 | 57 | 0.3684 |
| PRC2 | 1 | 12 | 0.4167 |
| Phosphatase | 3 | 157 | 0.4586 |
| RAS | 3 | 893 | 0.7368 |
| SMARCA | 3 | 751 | 0.6991 |
| STAT | 1 | 24 | 0.2917 |
| Steroid_enzyme | 1 | 11 | 0.3636 |
| Synuclein | 1 | 30 | 0.5333 |
| Tau | 1 | 210 | 0.3714 |
| Translation_factor | 1 | 17 | 0.2353 |
| **Total (22 families)** | **61** | **9,326** | — |

### Methodological notes
- Fourteen of the 22 families have only one target. They are still
  family-holdouts (not single-target holdouts), because the family name
  is a meaningful biological grouping that may share targets with the
  rest of the corpus that did not pass LOTO eligibility (e.g. PARP1
  alone passes filtering, but PARP2/PARP3/etc. exist in the corpus and
  are excluded from training when the PARP family is held out — that
  exclusion is encoded by holding out the family name, not just one
  UniProt accession). For the 22 families and 61 LOFO-eligible targets
  here, the family-holdout test set equals the union of LOTO-eligible
  targets in the family; rows for non-LOTO-eligible siblings of the same
  family that exist elsewhere in the corpus are not held out by the
  test_indices listing (consistent with `robustness/lofo.py`'s
  `run_lofo`, which masks `df['target_uniprot'].isin(fam_targets)` over
  the eligible set).
- LOTO-eligibility numbers reproduce the existing `data/loto_folds.json`
  (65 targets) and the existing `results/lofo.json` (`n_targets: 65,
  n_families: 26` — that file additionally counts the 4 unmapped
  singletons as their own families, which the published fold file
  excludes).

---

## 2. `data/cross_lab_folds.json` — Within-Target Cross-Lab

### Eligibility filter (mirrors `scripts/exp42_task14_within_target_cross_lab.py`)
- `MIN_ENTRIES_PER_TARGET = 20`
- `MIN_PAPERS_PER_TARGET = 3` (distinct DOIs after dropping rows with
  missing DOI)
- Both classes present in target.
- For each holdout paper within an eligible target:
  `MIN_ENTRIES_PER_PAPER = 5` and both classes present in that paper.
- Result: **36 eligible targets**, **84 qualifying paper holdouts**
  (sum across targets).

### Fold structure
Each top-level key is a UniProt accession. Each entry contains
`target_uniprot`, `n_entries` (rows with non-null DOI for this target),
`n_papers_total`, `n_qualifying_papers`, `activity_rate`, and
`papers[]`. Each paper entry is `{doi, n_entries, activity_rate,
test_indices}`. Training set for each paper holdout is
`{0..N-1} \ test_indices`.

### Target roster (qualifying_papers / total_papers, n_entries, activity_rate)

| Target | qual_papers / total | n_entries | activity_rate |
|---|---:|---:|---:|
| O15379 | 1 / 8 | 39 | 0.2308 |
| O60885 | 8 / 32 | 139 | 0.5324 |
| P00533 | 3 / 18 | 103 | 0.3786 |
| P01116 | 1 / 7 | 59 | 0.0508 |
| P03372 | 2 / 10 | 80 | 0.6125 |
| P09874 | 2 / 6 | 45 | 0.2000 |
| P10275 | 6 / 19 | 311 | 0.3698 |
| P11802 | 2 / 8 | 45 | 0.2444 |
| P15056 | 1 / 5 | 24 | 0.5000 |
| P17706 | 3 / 3 | 40 | 0.2750 |
| P18031 | 2 / 3 | 31 | 0.1290 |
| P24941 | 1 / 3 | 22 | 0.3182 |
| P25440 | 2 / 13 | 31 | 0.3871 |
| P36888 | 2 / 5 | 42 | 0.2857 |
| P36969 | 3 / 7 | 76 | 0.1974 |
| P40337 | 1 / 4 | 26 | 0.3846 |
| P43490 | 3 / 4 | 36 | 0.5000 |
| P49841 | 1 / 3 | 23 | 0.0870 |
| P50750 | 3 / 5 | 33 | 0.3939 |
| P51531 | 4 / 8 | 68 | 0.7500 |
| P51532 | 3 / 6 | 53 | 0.5660 |
| P61964 | 1 / 3 | 24 | 0.0833 |
| Q00534 | 2 / 8 | 46 | 0.4565 |
| Q02750 | 1 / 4 | 46 | 0.1522 |
| Q05397 | 1 / 7 | 70 | 0.9714 |
| Q06124 | 1 / 4 | 30 | 0.5000 |
| Q06187 | 10 / 21 | 149 | 0.5570 |
| Q07817 | 1 / 7 | 23 | 0.8696 |
| Q07889 | 4 / 4 | 57 | 0.1579 |
| Q13547 | 0 / 3 | 26 | 0.0385 |
| Q15059 | 2 / 14 | 31 | 0.4194 |
| Q9BY41 | 1 / 6 | 60 | 0.2667 |
| Q9NWZ3 | 1 / 4 | 32 | 0.0312 |
| Q9NYV4 | 1 / 3 | 28 | 0.6786 |
| Q9UBN7 | 1 / 11 | 39 | 0.6154 |
| Q9UM73 | 3 / 8 | 34 | 0.5294 |
| **Total (36 targets)** | **84 / 268** | **2,019** | — |

### Methodological notes
- The eligibility filter operates only on rows where `doi.notna()`. The
  `n_entries` column above counts only rows with a resolvable DOI
  (because cross-lab logic requires lab provenance). The full corpus
  has more rows for these same targets without DOI metadata; those rows
  participate in training but cannot serve as the held-out paper.
- One target (`Q13547`) qualifies at the target level (>=20 entries,
  >=3 distinct papers, both classes present in target) but has zero
  papers that simultaneously meet >=5 entries and both classes within
  the paper. It is retained in the file with `n_qualifying_papers: 0`
  for parity with `results/task14_within_target_cross_lab.json`, which
  also lists 36 eligible targets including this one.
- The summary numbers in `results/task14_within_target_cross_lab.json`
  (`n_eligible_targets: 36`, paired n=35) correspond to this fold file
  exactly.

---

## 3. `data/temporal_prospective_folds.json` — Temporal Prospective Split

### Eligibility filter
- **Train:** `pub_year < 2023` (1,866 entries)
- **Test:** `pub_year == 2024` (132 entries)
- 2023 entries (1,152) are excluded as a temporal-gap year.
- Rows without a resolvable `pub_year` (7,598 rows, ~70.7%) are
  excluded from both train and test.

### Fold structure
A single dict with `train_indices` and `test_indices` arrays. There is
no per-target subdivision; this is a corpus-level prospective split.

### Statistics

| Quantity | Value |
|---|---:|
| Total rows in corpus | 10,748 |
| Rows with `pub_year` | 3,150 |
| Rows excluded as 2023 gap year | 1,152 |
| Rows excluded for missing `pub_year` | 7,598 |
| Train rows (`pub_year < 2023`) | 1,866 |
| Test rows (`pub_year == 2024`) | 132 |
| Train activity rate | 0.3773 |
| Test activity rate | 0.3561 |
| Train target count | 116 |
| Test target count | 16 |
| Shared targets (train ∩ test) | 12 |
| Novel test targets | 4 |

### Train year distribution

| Year | Rows |
|---:|---:|
| 2015 | 11 |
| 2017 | 1 |
| 2018 | 150 |
| 2019 | 301 |
| 2020 | 476 |
| 2021 | 372 |
| 2022 | 555 |

### Methodological notes
- This split is **stricter** than the `source`-based proxy used in the
  existing `robustness/temporal.py` (which trains on `source == 'tpddb'`,
  7,567 rows, and tests on `source == 'protac8k'`, 3,181 rows). The
  source proxy uses dataset provenance as a temporal proxy and includes
  rows whose underlying publication date is unknown; the
  pub_year-based split released here uses only rows whose actual
  publication year was resolved against
  `/workspace/data/preprocessed/doi_years.json`.
- The pub_year-based split is what the task specification requires
  (train < 2023, test == 2024). The source-based split is retained in
  the codebase under `robustness/temporal.py` for backward
  compatibility with results in `results/temporal.json`, but is not
  exposed as a standalone fold file. Users wishing to reproduce
  `results/temporal.json` should use the source proxy described in
  `robustness/temporal.py`; users wishing to use the pub_year-based
  prospective split should use this fold file.
- Coverage caveat: only 18.6% of the corpus has resolvable
  `pub_year`. The 7,598-row gap is documented in
  `_meta.n_rows_excluded_no_pub_year`. The split is fit-for-purpose
  for prospective evaluation but cannot be expanded without either
  recovering more DOIs or relaxing the temporal definition.

---

## Registration

### `README.md`
The Dataset section now enumerates the four fold files, with one-line
descriptions and a note that all fold files are 0-based row indices
into `data/protac_bench.csv`.

### `data/croissant.json`
The `distribution[]` array now contains six FileObjects (was three):
`protac_bench.csv`, `loto_folds.json`, `lofo_folds.json` (new),
`cross_lab_folds.json` (new), `temporal_prospective_folds.json` (new),
`admet_scores.csv`. Each new entry includes the SHA-256 listed above
and a one-line `description`.

### Validator verdict
```
$ mlcroissant validate --jsonld data/croissant.json
W rdf.py:89] WARNING: The JSON-LD `@context` is not standard. ...
  The different keys are: {'equivalentProperty'}
I validate.py:53] Done.
```
**Verdict: PASS** (one pre-existing non-blocking warning about the
`equivalentProperty` key in `@context`, which is unrelated to the
distribution block changes and was already present in the prior
validated croissant manifest).

---

## Reproducibility

```
cd /workspace/PROTAC-Bench
python3 scripts/generate_fold_files.py
mlcroissant validate --jsonld data/croissant.json
```

The generator script is deterministic: it reads
`data/protac_bench.csv` and the row-aligned `master_df.csv`, applies the
three eligibility filters described above, and writes the JSON files
verbatim. No random sampling is involved.

The script is committed at
`/workspace/PROTAC-Bench/scripts/generate_fold_files.py`. It re-uses
the canonical `FAMILY_MAP` from `robustness/lofo.py` and the canonical
LOTO eligibility filter from `src/data_utils.get_eligible_targets`, so
any future change to the family taxonomy or LOTO eligibility rule will
flow through to the LOFO and (indirectly) cross-lab fold files when
the script is re-run.
