# Datasheet for PROTAC-Bench

This datasheet follows the framework of Gebru et al. (2018), *Datasheets for Datasets*
(arXiv:1803.09010). It documents the PROTAC-Bench dataset distributed with the paper
*Cold-Target Evaluation Exposes Systematic Memorization and Identifies Transferable
Signals in PROTAC Degradation Prediction*.

- Dataset version: 1.0.0
- Date published: 2026-04-07
- Files: `data/protac_bench.csv`, `data/loto_folds.json`, `data/admet_scores.csv`,
  `data/croissant.json`
- License: CC-BY-4.0 (annotations and curation); upstream sources retain their
  original licenses (see §3 and §6).

---

## 1. Motivation

**For what purpose was the dataset created?**
PROTAC-Bench was created to evaluate machine-learning models for PROTAC
(PROteolysis-TArgeting Chimera) degradation activity under a *cold-target*
generalization protocol. Published PROTAC predictors report random-split AUROCs
of 0.85–0.92 but, as demonstrated with this benchmark, collapse by 0.15–0.30
AUROC when no test-target molecule appears in training. The dataset operationalizes
this evaluation by providing 65 pre-defined Leave-One-Target-Out (LOTO) folds and a
unified label schema across previously incompatible source datasets.

**Who created the dataset and on behalf of which entity?**
The PROTAC-Bench authors (see `data/croissant.json`, anonymized for review). The
benchmark was assembled from three publicly released upstream sources (PROTAC-DB
3.0, Ribes et al., DegradeMaster) and is released with the companion paper.

**Who funded the creation of the dataset?**
Funding details are reported in the accompanying paper acknowledgements.

---

## 2. Composition

**What do the instances represent?**
Each instance is one measured PROTAC molecule × target-protein pair, with a binary
degradation activity label and (where available) the underlying potency endpoints
(DC50, Dmax) used to derive that label.

**How many instances are there in total?**
- 10,748 PROTAC × target entries
- 9,359 unique canonical SMILES
- 173 distinct target proteins (UniProt accessions)
- 65 targets meet the LOTO-eligibility criterion (≥10 entries with both classes
  represented)
- 7,427 unique Bemis–Murcko scaffolds (5,771 are singletons, 77.7%)
- Positive (active) rate: 65.8%
- E3 ligase distribution: CRBN 71.9%, VHL 26.9%, other 1.2%

**Does the dataset contain all possible instances or is it a sample?**
It is a curated union of three public PROTAC datasets, not an exhaustive sample of
known PROTACs. It excludes proprietary industrial assays and the long tail of
single-target literature reports outside the upstream sources.

**What data does each instance consist of?**
Columns of `protac_bench.csv`:

| Column | Type | Description |
|---|---|---|
| `smiles` | string | RDKit-canonicalized PROTAC SMILES |
| `target_uniprot` | string | UniProt accession of the target protein |
| `e3_type` | string | E3 ligase recruiter family: `VHL`, `CRBN`, or `Other` |
| `label` | int {0,1} | 1 if active (DC50 < 1 µM **or** Dmax > 50%), else 0 |
| `dc50_nm` | float / NA | Half-maximal degradation concentration, nM (when reported) |
| `dmax_pct` | float / NA | Maximum degradation percentage (when reported) |

`admet_scores.csv` provides 7 precomputed ADMET classifier outputs aligned row-wise
with `protac_bench.csv`: `caco2_wang`, `pampa_ncats`, `hia_hou`, `solubility_aqsoldb`,
`pgp_broccatelli`, `lipophilicity_astrazeneca`, `clearance_hepatocyte_az`.

`loto_folds.json` provides, per LOTO-eligible target, the test-set row indices, the
number of entries, and the activity rate.

**Is there a label or target associated with each instance?**
Yes. The primary label is the binary `label` column. The continuous endpoints
(`dc50_nm`, `dmax_pct`) are provided when the underlying source reported them and
may be missing.

**Is any information missing from individual instances?**
DC50 and Dmax are missing for entries whose source records reported only one of
the two endpoints, or only a binary outcome. The binary `label` is always present.
ADMET scores are precomputed from SMILES and are present for all 10,748 entries.

**Are relationships between individual instances made explicit?**
Yes. The `target_uniprot` and `smiles` columns make group structure explicit
(same molecule on different targets, same target with different molecules).
Bemis–Murcko scaffolds and warhead/linker decompositions can be derived from
SMILES; the LOTO folds in `loto_folds.json` group instances by target.

**Are there recommended data splits?**
Yes — and the choice of split is the central methodological contribution of this
benchmark.

- **LOTO (recommended for headline results):** 65-fold leave-one-target-out
  cross-validation defined in `loto_folds.json`. For each fold, all entries with
  that target are held out for testing; remaining targets form the training set.
- **Random split:** provided as a *diagnostic only* — agreement with random-split
  AUROC indicates target-identity memorization, not generalization. We do not
  recommend reporting random-split numbers as headline performance.
- **Auxiliary splits used in the paper:** scaffold-based, leave-one-family-out
  (LOFO), temporal, and per-source LOTO (TPDdb / PROTAC-8K). Scripts to reproduce
  them live in `robustness/`.

**Are there any errors, sources of noise, or redundancies?**
- **Inter-lab measurement noise.** Sources differ in cell line, treatment time,
  detection method, and quantification protocol. The paper attributes ~18% of
  the random→LOTO gap to lab-specific measurement patterns; the remaining ~82%
  is true target-novelty generalization.
- **Threshold sensitivity.** The DC50 < 1 µM / Dmax > 50% binarization shifts
  AUROC by ~±0.02 under reasonable alternatives.
- **Duplicate molecule×target measurements** from different sources are kept as
  separate rows when they disagree, since aggregation would erase legitimate
  inter-lab variance.
- **Class imbalance per target.** The 65 LOTO-eligible targets vary widely in
  size (n = 12–1968) and activity rate; per-target AUROCs in the paper range
  from 0.24 to 0.99.

**Does the dataset rely on external resources?**
The CSV is self-contained at the data layer. Reproducing the upstream provenance
requires PROTAC-DB 3.0 (http-served by its maintainers), the Ribes et al.
supplementary tables, and DegradeMaster's released splits — all public.

**Does the dataset contain confidential or sensitive data?**
No. PROTAC-Bench contains chemical structures (SMILES), public protein identifiers
(UniProt), and biochemical assay readouts drawn from already-published sources.
There is no human-subject, personal, or otherwise sensitive information.

---

## 3. Collection Process

**How was the data acquired?**
PROTAC-Bench is derived from three public datasets:

1. **PROTAC-DB 3.0** — community-curated PROTAC literature database.
2. **Ribes et al.** — supplementary tables released with their 2024 PROTAC GNN paper.
3. **DegradeMaster** — release accompanying the DegradeMaster paper.

Each source provides primary literature pointers; PROTAC-Bench did not collect
new biochemical measurements. All assay values trace back to the originating
publication via the upstream sources.

**What mechanisms were used to collect the data?**
Programmatic ingestion of the three upstream releases (CSV / supplementary
tables), followed by SMILES standardization, UniProt normalization, and a
unified binarization (see §4).

**Over what timeframe was the data collected?**
The upstream sources span peer-reviewed PROTAC literature from approximately
2010 through early 2024. PROTAC-Bench was assembled and frozen on 2026-04-07.

**Were any ethical review processes conducted?**
Not applicable — no human-subject or animal data was collected.

---

## 4. Preprocessing, Cleaning, and Labeling

**Was any preprocessing or cleaning done?**

- **SMILES canonicalization.** All molecules were re-canonicalized with RDKit.
  Salts and stereochemistry handling follow the upstream `data_utils.py` pipeline
  (`src/data_utils.py`).
- **Target normalization.** Target identifiers from heterogeneous source columns
  (gene symbol, RefSeq, free text) were mapped to UniProt accessions. Entries
  that could not be uniquely resolved to a single UniProt ID were dropped.
- **Label harmonization.** Sources reported activity at different cutoffs and as
  different formats (DC50 only, Dmax only, binary call only, etc.). PROTAC-Bench
  uses a single rule:
  > `label = 1` iff (DC50 reported and < 1 µM) **or** (Dmax reported and > 50%);
  > else `label = 0`.
- **LOTO eligibility.** A target is LOTO-eligible if it has ≥10 entries and at
  least one example of each class. 65 of 173 targets pass this filter.
- **ADMET precomputation.** A 7-endpoint cascade (TDC tasks: Caco-2, PAMPA, HIA,
  AqSolDB solubility, P-gp, AstraZeneca lipophilicity, hepatocyte clearance) was
  evaluated on every SMILES with classifiers trained on TDC data; outputs are
  shipped as `admet_scores.csv` so users do not have to re-run the cascade.

**Was the "raw" data saved?**
The upstream sources are publicly available at their original locations. The
PROTAC-Bench release does not redistribute the raw upstream files; it
distributes the merged, canonicalized, harmonized table and the LOTO fold
definitions. Provenance is documented row-wise in the source-tracking columns
used during curation (used internally and by the per-source LOTO analyses in
`robustness/single_source.py`).

**Is the software used to preprocess/clean the data available?**
Yes. The full ingestion, canonicalization, fold construction, and ADMET
precomputation pipeline is in this repository (`src/`, `signals/admet_cascade.py`).

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes — the companion paper uses PROTAC-Bench for:

- Universal cold-target collapse measurement across RF + Morgan, ESM-2 (8M–3B),
  D-MPNN, GIN, DeepPROTACs, PROTAC-STAN, and DegradeMaster.
- 2,000-trial TPE hyperparameter search with fANOVA decomposition (head type
  explains 95.9% of variance).
- Decomposition of the random→LOTO gap into target-novelty (~82%) and
  lab-specific measurement (~18%) components.
- Three transferable-signal interventions — warhead transfer, ADMET cascade,
  and stratified k-shot calibration — that recover ~+0.075 AUROC additively.
- 3D-geometry ablations (EGNN on experimental vs AF-predicted pockets, pocket
  shuffling, docking confound analysis).

Per-claim provenance for each numerical result is in `RELEASE_MANIFEST.md`.

**Is there a repository that links to publications using the dataset?**
The companion paper is the inaugural use; future entries will be tracked in the
project README on the dataset's distribution URL (Hugging Face Hub).

**What other tasks could the dataset be used for?**
- Cold-target evaluation of other PROTAC predictors and degrader-class models.
- Benchmarking few-shot, meta-learning, and active-learning strategies for
  low-data target adaptation.
- Out-of-distribution / scaffold-novelty studies in chemistry ML.
- E3-ligase generalization studies (cross-E3 transfer is asymmetric — see §6.3
  of the paper).
- Ablations of structural representations (SMILES vs graph vs 3D).

**Are there tasks for which the dataset should NOT be used?**
- **Headline predictive claims under random splits.** Random-split AUROC on
  this dataset is reachable by target-identity memorization and should be
  reported only as a diagnostic.
- **Direct prospective triage without recalibration.** The benchmark measures
  generalization to *unseen targets*; deploying a trained model against
  proprietary or industrial targets should include calibration on a few labeled
  examples (see the few-shot analyses in `signals/fewshot.py`).
- **Inter-lab effect-size estimation.** The dataset is too small per target to
  decouple lab and target effects rigorously; the 82/18 attribution in the
  paper is a population-level estimate, not per-target.
- **Inferring physical mechanism.** Labels are activity calls, not structural
  ternary-complex labels.

**Is there anything that affects future uses?**
The label binarization threshold is a project-level choice and may be revised
in future versions of the dataset; users should pin a version (`v1.0.0`) when
reporting results.

---

## 6. Distribution

**Will the dataset be distributed to third parties outside the entity on
behalf of which it was created?**
Yes. PROTAC-Bench is publicly released.

**How will it be distributed?**
- Repository: this Git repository (containing `data/`, code, and figures).
- Mirrored release: Hugging Face Hub
  (`https://huggingface.co/datasets/anonymous-neurips2026/protac-bench` per
  `data/croissant.json`).
- Croissant metadata: `data/croissant.json` (Croissant 1.0; SHA-256 hashes for
  each file).

**When will the dataset be distributed?**
Initial release: 2026-04-07 (version 1.0.0).

**Will the dataset be distributed under a copyright or other IP license?**
PROTAC-Bench's curation, fold definitions, ADMET precomputations, and code are
released under **CC-BY-4.0**. The underlying assay measurements derive from
publicly released upstream datasets that retain their original licenses:

- PROTAC-DB 3.0 — see PROTAC-DB terms of use.
- Ribes et al. supplementary data — see the publisher's permissions.
- DegradeMaster release — see its repository license.

Users redistributing PROTAC-Bench should preserve attribution to all four
sources.

**Have any third parties imposed IP-based or other restrictions?**
None beyond the upstream licenses noted above.

**Do any export controls or other regulatory restrictions apply?**
No.

---

## 7. Maintenance

**Who is supporting / hosting / maintaining the dataset?**
The PROTAC-Bench authors maintain the dataset; the canonical hosted location is
the Hugging Face Hub URL above, with this repository as the code companion.

**How can the owner / curator / manager of the dataset be contacted?**
Through the issue tracker on the dataset repository (Hugging Face Hub) and the
GitHub repository for this codebase.

**Is there an erratum?**
None at v1.0.0. Errata will be tracked in the release notes of the hosted
distribution.

**Will the dataset be updated?**
Likely yes, on a slow cadence:

- New PROTAC-DB releases will be re-ingested.
- Label-threshold robustness analyses may motivate auxiliary label columns
  (e.g., stricter and looser activity calls) released alongside the canonical
  label.
- Any update will increment the dataset version; older versions remain
  resolvable on the Hub.

**If the dataset relates to people, are there limits on retention?**
Not applicable — no personal data.

**Will older versions continue to be supported?**
Yes. Version `1.0.0` will remain pinned and downloadable; the SHA-256 hashes
in `data/croissant.json` allow integrity verification.

**If others want to extend / augment / contribute, is there a mechanism?**
Yes — issues and pull requests on the code repository, and discussion threads
on the Hugging Face dataset page. Contributions that add new upstream sources
should preserve the LOTO eligibility logic and the binarization rule, or
clearly document divergences and ship under a new version tag.
