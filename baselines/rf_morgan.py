#!/usr/bin/env python
"""RF + Morgan fingerprint baseline for PROTAC-Bench.

This is the anchor baseline: Random Forest on 2048-bit Morgan FPs
evaluated under LOTO protocol. Expected: 0.668 +/- 0.005.
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.evaluation import loto_evaluate
from src.stats import format_result
from src.debug import reduce_for_debug


def rf_model_fn(X_train, y_train, X_test):
    """Train RF and return predicted probabilities."""
    clf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight='balanced',
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def main():
    parser = argparse.ArgumentParser(description='RF + Morgan baseline')
    parser.add_argument('--seeds', default='42,43,44,53,71',
                        help='comma-separated random seeds')
    parser.add_argument('--output', default='results/baseline_rf_morgan.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--debug', action='store_true', help='Reduced seeds, cohort, and trials for smoke test')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading dataset...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values

    print('Computing Morgan fingerprints (2048 bits)...')
    X = compute_morgan(smiles)
    print(f'  X shape: {X.shape}')

    eligible = get_eligible_targets(df)
    seeds, eligible, _ = reduce_for_debug(seeds=seeds, eligible=eligible, debug=args.debug)
    print(f'Eligible targets: {len(eligible)}')

    print(f'Running LOTO evaluation with seeds={seeds}...')
    results = loto_evaluate(X, y, targets, rf_model_fn, eligible=eligible, seeds=seeds)

    # print per-target results for the first seed
    first_seed = str(seeds[0])
    print(f'\nPer-target AUROCs (seed={first_seed}):')
    for tgt, auc in sorted(results['per_seed'][first_seed].items(),
                            key=lambda x: x[1], reverse=True):
        print(f'  {tgt}: {auc:.3f}')

    print(f'\n=== RF + Morgan LOTO ===')
    print(format_result(results['mean_auroc'], results['std_auroc'], results['n_seeds']))
    print(f'N targets evaluated: {results["n_targets"]}')

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results['method'] = 'RF + Morgan (2048-bit)'
    results['seeds'] = seeds
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
