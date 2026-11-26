# PROTAC-Bench

Benchmark corpus and evaluation suite for PROTAC degradation activity prediction.

## Setup

Python 3.12.13.

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt

## Reproduce

    .venv/bin/python reproduce.py
    .venv/bin/python reproduce.py --ablation

The first command runs the 23 core arms, the derived passes and the figures; the
second adds the 24 ablation arms. An arm whose artefact exists is skipped, so an
interrupted run resumes. Fold files are regenerated from the corpus when absent, and the embedding
matrices by `scripts/embed.py`, which downloads two checkpoints on first run.
Core runtime is about twelve hours on sixteen cores; `PB_NJOBS` sets the number
of worker processes. The two graph baselines in `baselines/gnn_baselines.py`
require a CUDA device and are run separately.

## Outputs

    results/reproduction_<UTC timestamp>.txt   every arm present, one block per experiment
    results/<arm>.json                         aggregate, per seed, per target, provenance
    figures/*.pdf                              generated from those artefacts

## Data

    data/protac_bench.csv            10748 rows
    data/loto_primary_folds.json     61 leave-one-target-out folds
    data/lofo_folds.json             22 leave-one-family-out folds

The leave-one-target-out and leave-one-family-out fold files are built by
`scripts/generate_fold_files.py` from the corpus and the cohort definition in
`src/cohort.py`.

## Layout

    src/          cohort definition, features, metrics, seeds
    evaluation/   fold engine, protocol and calibration passes
    scripts/      fold-file and auxiliary data builders
    baselines/    graph neural network baselines
    figures/      generate_figures.py and its output
    results/      artefacts and the reproduction file
    robustness/   runners for protocols outside the reproduced suite
    signals/      feature builders and runners outside the reproduced suite
    hpo/          hyperparameter search, not part of the suite

## Seeds

42, 0, 1, 2, 3, 4, 5, 6, 7, 8, defined in `src/seeds.py`.

## Preprint

The artefacts and scripts accompanying arXiv:2605.11764 are at tag
`preprint-2026-05`. The suite on `main` has since been rebuilt on one cohort
definition and one seed list, so its values differ from those reported there.
