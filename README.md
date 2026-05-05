# PROTAC-Bench

A leave-one-target-out (LOTO) benchmark for PROTAC activity prediction. PROTAC-Bench provides 10,748 PROTAC-target activity measurements across 173 unique UniProt protein targets, with 65 LOTO-eligible folds, four primary evaluation splits, and a measurement-variance decomposition framework that attributes the apparent random-CV-to-LOTO generalisation gap to inter-laboratory measurement effects.

## Contents

- `data/protac_bench.csv`: canonical 10,748-row dataset with SMILES, UniProt target, E3 ligase, and binary activity labels.
- `data/loto_folds.json`, `data/lofo_folds.json`, `data/cross_lab_folds.json`, `data/temporal_prospective_folds.json`: fold assignment files for the four primary evaluation splits.
- `data/metadata.csv`: metadata-enriched 9,384-row subset with cell-line, readout-method, timepoint, and concentration annotations from the dual-LLM enrichment pipeline.
- `data/croissant.json`: MLCommons Croissant 1.0 metadata with all 20 RAI extension fields populated.
- `baselines/`: canonical baselines including `rf_morgan.py` (the documented 0.668 anchor), `dm_loto.py`, `gnn_baselines.py`, `chemprop_hpo.py`.
- `signals/`: full-stack pipeline including `warhead_transfer.py`, `admet_cascade.py`, `fewshot.py`, and `full_stack.py`.
- `robustness/`: cross-source, non-kinase, cross-E3, and temporal robustness checks.
- `hpo/`: 21-dimensional HPO V2 search space and validation.
- `figures/`: figure generation scripts for the canonical results.
- `src/`: shared utilities (data loading, evaluation, statistics, fingerprints).

## Quick start

```bash
pip install -r requirements.txt
bash reproduce.sh
```

Total runtime: approximately 2 to 3 hours on a single CPU node with 16 cores.

For a fast smoke test (2 seeds, 5 targets, completes in minutes):

```bash
python3 baselines/rf_morgan.py --debug
```

The `--debug` flag is available on all replication entry points.

## Dataset

Hosted on HuggingFace at https://huggingface.co/datasets/anonymous-neurips2026/protac-bench under CC-BY-4.0. The official non-anonymized URL will be published with the camera-ready version.

## Citation

If you use PROTAC-Bench in your research, please cite the accompanying NeurIPS 2026 Evaluations and Datasets Track paper. A BibTeX entry will be added upon acceptance.

## License

Code released under MIT License. Dataset released under CC-BY-4.0. The official non-anonymized URL will be published with the camera-ready version.
