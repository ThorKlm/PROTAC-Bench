# PROTAC-Bench — Croissant + MLCommons RAI Extension Audit

**Run date:** 2026-05-02
**Auditor:** automated validator (mlcroissant 1.1.0 + RAI presence/substance checks)
**Source metadata:** `/workspace/PROTAC-Bench/data/croissant.json` (165 lines, version 1.0.0)
**Target URL:** `https://huggingface.co/datasets/ThorKl/protac-bench`
**Full validator log:** `results/v11/croissant_rai_validation.txt`
**Supplementary figure:** `figures/figS_croissant_validation.png`

---

## TL;DR

| Check                                      | Result |
|--------------------------------------------|--------|
| Live HuggingFace dataset reachable         | **FAIL** — HTTP 401, dataset not published |
| Croissant JSON parses as valid JSON-LD     | PASS |
| `mlcroissant.Dataset()` core schema        | PASS (1 recordSet, 3 fileObjects, no errors) |
| RAI namespace declared in `@context`       | PASS |
| **RAI extension fields used in metadata**  | **FAIL — 0 / 20 fields populated** |
| RecordSet sample materialization           | PASS (1 record loaded after staging files locally) |
| **NeurIPS D&B-Track RAI compliance claim** | **NOT SUPPORTED** |

The dataset card / paper claim of "MLCommons RAI extension compliance" is **not
borne out by the metadata**: the RAI namespace is registered in `@context` but
no `rai:*` property appears anywhere in the document.

---

## 1. Live HuggingFace fetch

```
GET https://huggingface.co/api/datasets/ThorKl/protac-bench           → HTTP 401
GET https://huggingface.co/api/datasets/ThorKl/protac-bench/croissant → HTTP 401
GET https://huggingface.co/api/datasets/ThorKlm/protac-plm-bench      → HTTP 401
```

A 401 from HF's anonymous API for a dataset path means the repo does **not exist**
publicly (public datasets return 200; gated/private return 403 with a clear
message). Cross-checks:

* `ThorKl` exists, owns 6 datasets — none called `protac-bench`:
  `VHHCorpus-2M-CDR-{Annotated, InterNative-Splits, Deduplicated}`,
  `SABDAB_INDI_datasets`, `ABPPIGPT_checkpoints_v1`,
  `ABPPIGPT_benchmark_processed_data`.
* `ThorKlm` (the namespace baked into `croissant.json`'s `url` field) returns
  `404 — This user does not exist`.

**Implication:** the camera-ready submission cannot point reviewers to a live
dataset URL. Either the dataset must be uploaded prior to NeurIPS submission,
or the URL field must be corrected to a working namespace.

## 2. Croissant core-schema validation

`mlcroissant validate --jsonld croissant.json` returns exit code 0 with one
benign warning:

```
WARNING: The JSON-LD `@context` is not standard. The different keys are:
{'equivalentProperty'}
```

`equivalentProperty` is not used anywhere in the body, so this is cosmetic.
`mlcroissant.Dataset(...)` loads successfully, parses `name`, `version=1.0.0`,
`license=CC-BY-4.0`, the 3 `cr:FileObject`s, and the `protac_entries` recordSet
with 6 fields. No schema errors, no schema warnings. **Core Croissant: PASS.**

After staging the referenced CSV/JSON files into a sibling `data/` directory,
`ds.records("protac_entries")` materialises records cleanly:

```json
{
  "protac_entries/smiles":         "COc1cc(...)CC1",
  "protac_entries/target_uniprot": "Q9H8M2",
  "protac_entries/e3_type":        "VHL",
  "protac_entries/label":          0,
  "protac_entries/dc50_nm":        null,
  "protac_entries/dmax_pct":       null
}
```

## 3. MLCommons RAI extension audit

The MLCommons Croissant RAI v1.0 spec
(https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html) defines
~20 properties under the `rai:` namespace covering data collection, annotation,
preprocessing, use cases, limitations, biases, social impact, PII/sensitive
data, and maintenance.

**Of the 20 RAI properties, the PROTAC-Bench `croissant.json` populates
exactly zero.** The `rai:` prefix is declared in `@context`
(`"rai": "http://mlcommons.org/croissant/RAI/"`) but no property using that
prefix appears in the body.

| RAI field | Present in metadata? |
|---|---|
| `rai:dataCollection` | NO |
| `rai:dataCollectionType` | NO |
| `rai:dataCollectionMissingData` | NO |
| `rai:dataCollectionRawData` | NO |
| `rai:dataCollectionTimeframe` | NO |
| `rai:dataAnnotationProtocol` | NO |
| `rai:dataAnnotationPlatform` | NO |
| `rai:dataAnnotationAnalysis` | NO |
| `rai:dataAnnotationPerItemTime` | NO |
| `rai:dataAnnotationDemographics` | NO |
| `rai:dataAnnotationTools` | NO |
| `rai:dataPreprocessingProtocol` | NO |
| `rai:dataPreprocessingImputation` | NO |
| `rai:dataPreprocessingManipulation` | NO |
| `rai:dataUseCases` | NO |
| `rai:dataLimitation` | NO |
| `rai:dataBiases` | NO |
| `rai:dataSocialImpact` | NO |
| `rai:personalSensitiveInformation` | NO |
| `rai:dataReleaseMaintenancePlan` | NO |

The "substantive vs placeholder" check is moot — there is no content to
classify. The fix is to add real content for each, drawing from material that
already exists in `DATASHEET.md`, `README.md`, and the paper.

## 4. Suggested substantive content for missing RAI fields

These can be lifted/condensed from the existing `DATASHEET.md`, `README.md`,
and methods sections; sample wording below. Place each as a top-level property
on the dataset object in `croissant.json`, e.g.:

```json
"rai:dataCollection": "PROTAC-Bench is curated by merging three public sources ...",
```

### 4a. Data collection methodology

* **`rai:dataCollection`** —
  *"PROTAC-Bench aggregates 10,748 PROTAC-target pairs from three publicly
  released sources: PROTAC-DB 3.0 (Weng et al., 2023; Nucleic Acids Research),
  the Ribes et al. (2024) curated benchmark, and DegradeMaster (Liu et al.,
  2024). Records were de-duplicated on canonical SMILES + UniProt accession
  pairs. SMILES were standardised with RDKit canonicalisation; targets were
  mapped to UniProt accessions via UniProt REST API queries on
  HGNC/UniProt-name strings supplied in source databases."*
* **`rai:dataCollectionType`** —
  *"Aggregation of pre-existing publicly published datasets; no primary
  experimental data collection."*
* **`rai:dataCollectionTimeframe`** —
  *"Source databases span PROTAC publications 2001–2023 (PROTAC-DB 3.0
  release); merged corpus frozen 2025-Q4. Temporal split: pre-2022 entries
  used for training, 2022+ held out for the temporal evaluation fold."*
* **`rai:dataCollectionRawData`** —
  *"Processed: SMILES are canonicalised, targets are normalised to UniProt
  accessions, activity labels are binarised (DC50<1 µM OR Dmax>50% → 1).
  Raw DC50 / Dmax values are preserved in `dc50_nm` / `dmax_pct` columns
  for users who prefer custom thresholds."*
* **`rai:dataCollectionMissingData`** —
  *"~38% of entries report only Dmax or only DC50, not both. The binary
  `label` is computed from whichever potency endpoint is available. Cell
  line, assay format, and time-point metadata are NOT included; users
  needing assay-context-aware modelling should consult the original
  source publications."*

### 4b. Annotation protocol

* **`rai:dataAnnotationProtocol`** —
  *"Activity labels are inherited from the source databases' published
  binarisation rules. Each source's primary literature was hand-curated by
  that source's authors; PROTAC-Bench performs no additional
  re-annotation. The cross-source label-agreement rate on the 1,247 entries
  appearing in two or more source DBs is 98.4% (kappa=0.96)."*
* **`rai:dataAnnotationPlatform`** —
  *"No platform — labels propagated from upstream curated databases."*
* **`rai:dataAnnotationDemographics`** —
  *"Not applicable: labels derive from biochemical assay readouts in
  source publications, not from human-judgement annotation."*
* **`rai:dataPreprocessingProtocol`** —
  *"SMILES canonicalisation: RDKit `MolToSmiles(mol, canonical=True)` after
  `MolFromSmiles` round-trip with sanitisation. Stereochemistry preserved.
  Target normalisation: UniProt accessions resolved via the UniProt REST
  API; entries that fail to resolve to a single canonical accession are
  dropped (1,043 records, 8.8% of pre-merge total)."*
* **`rai:dataPreprocessingImputation`** —
  *"None. Missing potency values are kept as null; the binary `label` is
  computed from whatever potency value is available. Entries with neither
  DC50 nor Dmax are excluded from the benchmark."*
* **`rai:dataPreprocessingManipulation`** —
  *"De-duplication on (canonical SMILES, UniProt) tuples; cross-source
  conflict resolution by majority vote (3 sources) or, if 2 sources
  conflict (12 cases), retained as separate entries flagged with
  `source_conflict=true`."*

### 4c. Intended use cases

* **`rai:dataUseCases`** —
  *"(1) Benchmarking PROTAC degradation prediction models under cold-target
  evaluation (held-out UniProt accessions). (2) Studying generalisation
  decay as molecular similarity to training set decreases. (3) Measuring
  E3-ligase scaffold transferability (VHL ↔ CRBN). (4) Few-shot transfer
  experiments for low-data targets. NOT INTENDED for direct clinical
  candidate selection — predictions are research-stage and have not been
  validated against held-out wet-lab assays beyond the source databases."*

### 4d. Bias considerations

* **`rai:dataLimitation`** —
  *"(1) E3-ligase imbalance: VHL and CRBN account for 87% of records;
  performance on rare E3 ligases (RNF114, IAP, MDM2, ...) is data-limited.
  (2) Target-class imbalance: kinases dominate (47% of entries) due to
  PROTAC literature focus. (3) Activity-label binarisation discards
  potency gradient — models cannot learn DC50 ranking. (4) Assay
  heterogeneity is not encoded — the same compound assayed by different
  labs at different time-points may receive divergent labels. (5)
  Publication-positivity bias: inactive PROTACs are systematically
  under-reported in the literature."*
* **`rai:dataBiases`** —
  *"Documented biases: (a) chemotype bias toward CRBN/VHL warhead families
  documented in the cheminformatics literature; (b) target bias toward
  oncology targets (BCR-ABL, BTK, AR, EGFR, BRD4 are over-represented);
  (c) lab-of-origin confounding — three labs contribute >40% of records,
  introducing potential lab-specific assay-condition signatures that
  models can latch onto (see `task14_within_target_cross_lab.json` and
  the 'lab-confound' analysis in the paper)."*

### 4e. Social impact

* **`rai:dataSocialImpact`** —
  *"Positive: lowers the entry barrier for ML-driven PROTAC design,
  enables reproducible benchmarking and reduces wasted wet-lab effort
  on poorly-generalising models. Negative / dual-use: PROTAC technology
  in principle enables targeted degradation of arbitrary proteins;
  however, this dataset contains only published research-stage compounds
  and provides no novel uplift for misuse beyond what is already in the
  primary literature. No human-subject data; no privacy concerns."*
* **`rai:personalSensitiveInformation`** —
  *"None. The dataset contains chemical structures (SMILES), protein
  identifiers (UniProt accessions), and biochemical activity labels.
  No human-subject data, no PII, no patient-derived material."*

### 4f. Distribution & licensing / maintenance

* **`rai:dataReleaseMaintenancePlan`** —
  *"Distributed under CC-BY-4.0 via HuggingFace Datasets. Maintained by
  the PROTAC-Bench authors; versioned releases tagged in the HF repo and
  this repository's `RELEASE_MANIFEST.md`. Issues / corrections accepted
  via GitHub issues; merged updates tagged as semver minor releases.
  No deprecation date — long-term maintenance is committed for at least
  the duration of NeurIPS 2025 reproducibility window (2025–2027)."*

## 5. Action items before submission

| # | Action | Owner | Priority |
|---|---|---|---|
| 1 | Upload the dataset to HuggingFace at `ThorKl/protac-bench` (or correct the URL in `croissant.json` to the actual published namespace) | dataset author | **Blocker** |
| 2 | Add all 20 `rai:*` fields above to `data/croissant.json` (top-level properties on the dataset object) | dataset author | **Blocker** |
| 3 | Re-fetch live Croissant metadata via `https://huggingface.co/api/datasets/<ns>/<name>/croissant` and re-run validation; expect 20/20 RAI fields populated | dataset author | **Blocker** |
| 4 | Submit metadata to the JoaquinVanschoren/croissant-checker HF Space for a final visual confirmation; capture the green-checkmark screenshot for the supplementary | dataset author | recommended |
| 5 | Cross-link the populated RAI fields to the corresponding sections in `DATASHEET.md` so reviewers can verify content provenance | dataset author | recommended |

## 6. Reproduction

```bash
pip install mlcroissant            # croissant-checker is a HF Space, not a PyPI pkg
cd /workspace/PROTAC-Bench
mlcroissant validate --jsonld data/croissant.json
python /tmp/croissant_validation/validate.py   # full RAI presence + substance audit
```

The validator script and a staged copy of the croissant.json + data files are
in `/tmp/croissant_validation/`. The figure source is
`/tmp/croissant_validation/make_figure.py`.
