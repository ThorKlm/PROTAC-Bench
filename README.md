# PROTAC-Bench

Cold-target evaluation benchmark for PROTAC degradation prediction. Companion code for:

> Cold-Target Evaluation Exposes Systematic Memorization and Identifies Transferable Signals in PROTAC Degradation Prediction

## Quick Start

```bash
pip install -r requirements.txt
bash reproduce.sh
```

This runs the core experiments and prints a comparison table. Takes ~2-3h on CPU.

## Dataset

10,748 binary PROTAC degradation entries across 173 targets (65 LOTO-eligible).
- `data/protac_bench.csv`: SMILES, target UniProt, E3 type, label, DC50, Dmax
- `data/loto_folds.json`: 65-fold LOTO assignments
- `data/admet_scores.csv`: precomputed ADMET classifier outputs (7 features)

## Key Results

| Method | AUROC | Seeds |
|--------|-------|-------|
| RF + Morgan (baseline) | 0.668 ± 0.005 | 10 |
| + Warhead transfer | 0.711 ± 0.008 | 10 |
| + ADMET cascade | 0.714 ± 0.005 | 10 |
| + k=5 stratified few-shot | **0.743 ± 0.012** | 10 |

## Repository Structure

- `baselines/` — baseline models and published method replications
- `signals/` — transferable signals that break the ceiling
- `hpo/` — hyperparameter optimization (1,086 trials)
- `robustness/` — single-source, non-kinase, LOFO, temporal validation
- `src/` — shared utilities

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
