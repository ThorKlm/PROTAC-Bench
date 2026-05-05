# PROTAC-Bench HuggingFace Upload Log

**Run date:** 2026-05-02
**Operator:** sequential HF upload runner
**Authoritative croissant source:** `/workspace/PROTAC-Bench/data/croissant_v2.json`
**Live URL:** https://huggingface.co/datasets/ThorKl/protac-bench
**HF API croissant endpoint:** https://huggingface.co/api/datasets/ThorKl/protac-bench/croissant

---

## 1. Namespace selection

| Attempt | Namespace | Result |
|---|---|---|
| 1 | `anonymous-neurips2026/protac-bench` | **403 Forbidden** — runner credentials lack rights to create a dataset under this namespace, and the namespace cannot be self-registered as an org via the API |
| 2 | `ThorKl/protac-bench` | **CREATED** — `https://huggingface.co/datasets/ThorKl/protac-bench` |

**Actual namespace used:** `ThorKl/protac-bench` (per task fallback rule). The
double-blind concern that motivated `anonymous-neurips2026` is preserved by the
README — it contains no author names — and by the croissant `creator` field
which reads "PROTAC-Bench Authors (anonymized for double-blind review)".
The `ThorKl` username is the runner's HuggingFace identity and is not
associated with the paper's authorship in either README.md or DATASHEET.md.

## 2. Repository visibility

The dataset was created with `private=False` but defaulted to private on the
runner's HF account. It was then explicitly set public via
`HfApi.update_repo_settings(private=False)` so anonymous reviewer fetches
succeed. Final state: `private: False`, `gated: False`, `disabled: False`.

## 3. File upload manifest

A single atomic commit (`a1ce9a48f3c8a309fe8a04f42f64bf16fef13675`) uploaded
43 files.

| Local source | Repo path | Size (bytes) |
|---|---|---|
| `data/protac_bench.csv` | `data/protac_bench.csv` | 1,504,412 |
| `data/admet_scores.csv` | `data/admet_scores.csv` | 677,240 |
| `data/loto_folds.json` | `data/loto_folds.json` | 121,394 |
| `data/croissant_v2.json` | `croissant.json` *(renamed)* | 12,609 |
| `README.md` | `README.md` | 4,225 |
| `DATASHEET.md` | `DATASHEET.md` | 15,243 |
| `RELEASE_MANIFEST.md` | `RELEASE_MANIFEST.md` | 28,332 |
| `requirements.txt` | `requirements.txt` | 188 |
| `reproduce.sh` | `reproduce.sh` | 975 |
| `src/*.py` (5 files) | `src/*.py` | ~8.7 KB |
| `baselines/*.py` (10 files) | `baselines/*.py` | ~62 KB |
| `robustness/*.py` (6 files) | `robustness/*.py` | ~50 KB |
| `signals/*.py` (8 files) | `signals/*.py` | ~73 KB |
| `results/lofo.json` | `results/lofo.json` | 6,034 |
| `results/task14_within_target_cross_lab.json` | `results/task14_within_target_cross_lab.json` | 68,735 |
| `results/temporal.json` | `results/temporal.json` | 1,505 |
| `results/dm_loto.json` | `results/dm_loto.json` | 34,632 |
| `results/warhead_lofo.json` | `results/warhead_lofo.json` | 1,046 |

The live repo lists 46 siblings; the extra 3 are pre-existing
`splits/lofo_folds.json`, `splits/loto_folds.json`, and the auto-generated
`.gitattributes`.

## 4. Fold-assignment file inventory (caveat)

The task description requested upload of fold-assignment files for "LOTO
65-fold, LOFO 61-fold, cross-lab 36-fold, temporal-prospective". Only the
LOTO file exists as a discrete fold-assignment JSON in the repo on disk:

| Requested fold file | Status |
|---|---|
| LOTO 65-fold | **Uploaded** as `data/loto_folds.json` (121 KB) |
| LOFO 61-fold | **No discrete fold-assignment JSON on disk**; the LOFO splitting is implemented procedurally in `robustness/lofo.py`. Result file `results/lofo.json` was uploaded as the closest existing artifact. (A pre-existing `splits/lofo_folds.json` from a prior commit is also visible on the HF repo.) |
| Cross-lab 36-fold | **No discrete fold-assignment JSON on disk**; splitting is in `robustness/cross_e3.py`. Result file `results/task14_within_target_cross_lab.json` (which lists the 36 (target, lab) groupings) was uploaded as the closest existing artifact. |
| Temporal-prospective | **No discrete fold-assignment JSON on disk**; implemented in `robustness/temporal.py`. `results/temporal.json` was uploaded. |

The generating source code for all four split families is uploaded so the
folds remain deterministically reproducible. This is documented here for
transparency rather than fabricated.

## 5. Upload outcome

| Step | Result |
|---|---|
| Repo creation under `ThorKl` | **OK** |
| 43-file atomic commit | **OK** (2.0 s wallclock) |
| Visibility flipped to public | **OK** |
| `api.list_repo_files` confirms upload | **OK** (46 siblings listed) |

Commit URL:
https://huggingface.co/datasets/ThorKl/protac-bench/commit/a1ce9a48f3c8a309fe8a04f42f64bf16fef13675

## 6. Live validation outcome (cross-reference)

See `results/v11/croissant_live_validation.txt` for the raw text. Summary:

- mlcroissant 1.1.0 validate against the live `raw/main/croissant.json`:
  exit code 0 (one cosmetic `equivalentProperty` warning, no schema errors).
- RAI namespace declared in `@context`: PASS.
- `rai:*` properties populated: **20 / 20** (verbatim audit text).
- HF Croissant API endpoint
  `https://huggingface.co/api/datasets/ThorKl/protac-bench/croissant`:
  HTTP 200.

## 7. Failures and follow-ups

- **`anonymous-neurips2026` namespace:** could not be claimed by the runner
  (403). For camera-ready, the authors should either (a) create that org on HF
  and grant write access, then mirror this repo, or (b) accept the
  `ThorKl/protac-bench` URL and rely on the README's textual anonymisation.
- **README YAML metadata warning:** HF emits a warning that `README.md` lacks
  a YAML front-matter block (dataset card metadata). The repo loads and the
  croissant validates without it; adding a `license: cc-by-4.0` YAML header
  is the recommended minor follow-up.
- **Missing per-fold JSONs for LOFO / cross-lab / temporal:** see §4 above —
  documented rather than fabricated. The generating source code is uploaded.
