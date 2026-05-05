#!/usr/bin/env python
"""ADMET cascade signal: concatenate precomputed ADMET scores with Morgan FP.

Expected: 0.687 (+0.021 over baseline, p=0.014).

Conditions:
  C0: Morgan baseline
  C1: Morgan + ADMET7 (all 7 ADMET descriptors)
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets, load_admet
from src.evaluation import loto_evaluate
from src.stats import paired_wilcoxon, format_result
from src.debug import reduce_for_debug

ADMET_NAMES = [
    "Caco2_Wang", "PAMPA_NCATS", "HIA_Hou", "Solubility_AqSolDB",
    "Pgp_Broccatelli", "Lipophilicity_AstraZeneca", "Clearance_Hepatocyte_AZ",
]


def make_rf_fn(seed=42):
    def model_fn(X_tr, y_tr, X_te):
        clf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3,
            class_weight='balanced', random_state=seed, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]
    return model_fn


def correlation_analysis(admet, y):
    """Print point-biserial correlation between each ADMET feature and labels."""
    from scipy.stats import pointbiserialr
    print('\nADMET-label correlations:')
    for i, name in enumerate(ADMET_NAMES):
        # skip NaN
        mask = ~np.isnan(admet[:, i])
        if mask.sum() < 20:
            print(f'  {name}: too few valid values')
            continue
        r, p = pointbiserialr(y[mask], admet[mask, i])
        print(f'  {name:30s}  r={r:+.3f}  p={p:.4f}')


def main():
    parser = argparse.ArgumentParser(description='ADMET cascade signal')
    parser.add_argument('--seeds', default='42,43,44,53,71')
    parser.add_argument('--output', default='results/admet_cascade.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--debug', action='store_true', help='Reduced seeds, cohort, and trials for smoke test')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading data...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values

    print('Computing Morgan FPs...')
    X_morgan = compute_morgan(smiles)

    print('Loading ADMET scores...')
    X_admet = load_admet()
    print(f'  ADMET shape: {X_admet.shape}')
    assert X_admet.shape[0] == X_morgan.shape[0], "ADMET / dataset size mismatch"

    # replace NaN with 0 (shouldn't happen but just in case)
    X_admet = np.nan_to_num(X_admet, nan=0.0)

    eligible = get_eligible_targets(df)
    seeds, eligible, _ = reduce_for_debug(seeds=seeds, eligible=eligible, debug=args.debug)
    print(f'Eligible targets: {len(eligible)}')

    correlation_analysis(X_admet, y)

    # C0: baseline
    print('\n[C0] Morgan baseline...')
    c0 = loto_evaluate(X_morgan, y, targets, make_rf_fn(), eligible=eligible, seeds=seeds)
    print(f'  {format_result(c0["mean_auroc"], c0["std_auroc"], c0["n_seeds"])}')

    # C1: Morgan + ADMET
    X_combined = np.column_stack([X_morgan, X_admet])
    print(f'\n[C1] Morgan + ADMET7 (shape={X_combined.shape})...')
    c1 = loto_evaluate(X_combined, y, targets, make_rf_fn(), eligible=eligible, seeds=seeds)
    print(f'  {format_result(c1["mean_auroc"], c1["std_auroc"], c1["n_seeds"])}')

    # stats
    a_list, b_list = [], []
    for s in [str(s) for s in seeds]:
        shared = set(c0['per_seed'].get(s, {}).keys()) & set(c1['per_seed'].get(s, {}).keys())
        for tgt in sorted(shared):
            b_list.append(c0['per_seed'][s][tgt])
            a_list.append(c1['per_seed'][s][tgt])
    p_val = paired_wilcoxon(a_list, b_list)
    delta = c1['mean_auroc'] - c0['mean_auroc']

    print(f'\n=== ADMET Cascade ===')
    print(f'C0: {c0["mean_auroc"]:.3f}')
    print(f'C1: {c1["mean_auroc"]:.3f}')
    print(f'Delta: {delta:+.3f}, p={p_val:.4f}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        'method': 'Morgan + ADMET7',
        'seeds': seeds,
        'baseline': c0, 'admet_cascade': c1,
        'delta': float(delta), 'p_value': float(p_val),
        'admet_features': ADMET_NAMES,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
