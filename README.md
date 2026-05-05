# PROTAC-Bench

Cold-target evaluation benchmark for PROTAC degradation prediction. Companion code for:

> Cold-Target Evaluation Exposes Systematic Memorization and Identifies Transferable Signals in PROTAC Degradation Prediction

## Quick Start

```bash
pip install -r requirements.txt
bash reproduce.sh
```

Runs core experiments and prints a comparison table. Takes ~2-3h on CPU.

## Dataset

10,748 binary PROTAC degradation entries across 173 targets (65 LOTO-eligible).
- `data/protac_bench.csv`: SMILES, target UniProt, E3 type, label, DC50, Dmax
- `data/loto_folds.json`: 65-fold Leave-One-Target-Out assignments
- `data/lofo_folds.json`: Leave-One-Family-Out (22 protein families, 61 targets)
- `data/cross_lab_folds.json`: within-target cross-lab paper holdouts (36 targets with >=3 publications and >=5 entries per holdout publication)
- `data/temporal_prospective_folds.json`: train pub_year < 2023 (n=1,866), test pub_year == 2024 (n=132)
- `data/admet_scores.csv`: precomputed ADMET classifier outputs (7 features)
- `data/croissant.json`: Croissant metadata for dataset discovery

All fold files index into `data/protac_bench.csv` row positions (0-based).

## Key Results

**The cold-target collapse.** All published methods collapse 0.15-0.30 AUROC under LOTO.
2,000 HPO trials across 21 dimensions confirm the ceiling (fANOVA: head type explains 96%).

| Method | Random | LOTO | Seeds |
|--------|--------|------|-------|
| RF + Morgan (baseline) | 0.902 | 0.668 +/- 0.005 | 10 |
| DeepPROTACs | 0.847 | 0.531 | 5 |
| DegradeMaster (their data) | 0.878 | 0.702 | 5 |
| D-MPNN + 27-config HPO | --- | 0.600 | 5 |

**Gap decomposition.** 82% of the gap is target novelty, 18% is lab-specific measurement patterns.

**Transferable signals.** Three orthogonal signals break through with 95% additivity:

| Condition | AUROC | Delta | 95% CI |
|-----------|-------|-------|--------|
| Morgan baseline | 0.668 +/- 0.005 | --- | [0.664, 0.672] |
| + Warhead transfer | 0.711 +/- 0.008 | +0.042 | [0.703, 0.719] |
| + ADMET cascade | 0.714 +/- 0.005 | +0.003 | [0.709, 0.719] |
| + k=5 stratified few-shot | **0.743 +/- 0.012** | +0.029 | [0.731, 0.755] |

## Repository Structure

```
src/               Core utilities (275 lines)
  data_utils.py      Dataset loading, Morgan FP computation, LOTO folds
  evaluation.py      LOTO evaluation loop, statistical tests
  fingerprints.py    Fingerprint variants
  stats.py           Paired tests, confidence intervals

baselines/         Baseline models and published method replications (1,288 lines)
  rf_morgan.py       RF + Morgan baseline
  plm_scaling.py     ESM-2 8M to 3B under LOTO
  deepprotacs_eval.py  DeepPROTACs replication
  dm_loto.py         DegradeMaster under LOTO
  gnn_baselines.py   GIN, D-MPNN
  chemprop_hpo.py    D-MPNN with 27-config HPO

signals/           Transferable signals (2,048 lines)
  warhead_transfer.py   Cross-target warhead activity rate
  warhead_ablation.py   Ablation: transfer rate vs warhead identity
  warhead_lofo.py       LOFO limitation analysis
  admet_cascade.py      7-endpoint ADMET classifier
  admet_ablation.py     Per-endpoint ablation
  fewshot.py            RF vs MAML vs ProtoNet
  fewshot_strategies.py Stratified vs random vs diverse selection
  full_stack.py         Combined signal evaluation

robustness/        Robustness analyses (1,342 lines)
  single_source.py     Per-source LOTO (TPDdb, PROTAC-8K)
  nonkinase.py         Non-kinase target subset
  lofo.py              Leave-one-family-out
  cross_e3.py          Cross-E3 ligase generalization
  temporal.py          Temporal prospective split

hpo/               Hyperparameter optimization (800 lines)
  run_hpo.py           2,000-trial TPE search across 21 dimensions
  validate_top_k.py    5-seed validation of top configs
  fanova_analysis.py   Feature importance decomposition

results/           Precomputed result JSONs (27 experiments + 22 metadata tasks)
figures/           Figure generation scripts and PDFs
data/              Dataset files
```

## Adding Your Own Method

```python
from src.data_utils import load_dataset, compute_morgan, get_loto_folds
from src.evaluation import loto_evaluate

df = load_dataset()
X = compute_morgan(df['smiles'].tolist())
y = df['label'].values
targets = df['target_uniprot'].values

def my_model(X_train, y_train, X_test):
    # return predicted probabilities for X_test
    ...

results = loto_evaluate(X, y, targets, my_model)
print(f'LOTO AUROC: {results["mean_auroc"]:.3f}')
```

## License

Dataset and code: CC-BY-4.0. Underlying PROTAC data from PROTAC-DB, Ribes et al., and DegradeMaster retain original licenses.
