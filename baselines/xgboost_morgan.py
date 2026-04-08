#!/usr/bin/env python
"""XGBoost + Morgan FP baseline -- quick comparison to RF."""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import xgboost as xgb
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.evaluation import loto_evaluate
from src.stats import format_result


def make_xgb_model_fn(scale_pos_weight):
    def model_fn(X_train, y_train, X_test):
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]
    return model_fn


def main():
    parser = argparse.ArgumentParser(description='XGBoost + Morgan baseline')
    parser.add_argument('--seeds', default='42,43,44,53,71')
    parser.add_argument('--output', default='results/baseline_xgb_morgan.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading data...')
    df = load_dataset(args.data)
    X = compute_morgan(df['smiles'].tolist())
    y = df['label'].values
    targets = df['target_uniprot'].values

    # compute class balance for scale_pos_weight
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw = n_neg / n_pos
    print(f'scale_pos_weight={spw:.2f} (neg={n_neg}, pos={n_pos})')

    eligible = get_eligible_targets(df)
    print(f'Eligible targets: {len(eligible)}')

    model_fn = make_xgb_model_fn(spw)
    print(f'Running LOTO with seeds={seeds}...')
    results = loto_evaluate(X, y, targets, model_fn, eligible=eligible, seeds=seeds)

    print(f'\n=== XGBoost + Morgan LOTO ===')
    print(format_result(results['mean_auroc'], results['std_auroc'], results['n_seeds']))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results['method'] = 'XGBoost + Morgan'
    results['seeds'] = seeds
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
