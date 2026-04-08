#!/usr/bin/env python
"""Compare k=5 few-shot selection strategies.

Strategies:
  - Random:     np.random.choice  -> 0.714
  - Stratified: bin by predicted probability quintiles -> 0.734
  - Diverse:    MaxMin Tanimoto picking -> 0.704
  - Uncertainty: closest to decision boundary -> 0.688

Two-stage: first train baseline RF -> get predictions -> use those
to guide selection for the second-stage RF.

Written after other scripts were done, references fewshot.py results.
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.stats import paired_wilcoxon, format_result


def _tanimoto_chunk(A, B, chunk_size=512):
    """Pairwise Tanimoto. Copied from warhead_transfer.py"""
    n = A.shape[0]
    sum_B = B.sum(axis=1)
    sims = np.zeros((n, B.shape[0]), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Ac = A[start:end]
        sA = Ac.sum(axis=1)
        AB = Ac @ B.T
        denom = sA[:, None] + sum_B[None, :] - AB
        denom = np.maximum(denom, 1e-8)
        sims[start:end] = AB / denom
    return sims


# ---- Selection strategies ----

def select_random(test_idx, y_test, k, rng, **kwargs):
    """Plain random selection."""
    return rng.choice(test_idx, k, replace=False)


def select_stratified(test_idx, y_test, k, rng, pred_probs=None, **kwargs):
    """Stratified by predicted probability quintiles.
    Bin test samples into 5 quintiles, sample proportionally.
    """
    if pred_probs is None:
        return select_random(test_idx, y_test, k, rng)

    n_bins = 5
    bins = np.digitize(pred_probs, np.linspace(0, 1, n_bins + 1)[1:-1])

    selected = []
    # how many per bin
    bin_counts = np.bincount(bins, minlength=n_bins)
    samples_per_bin = np.zeros(n_bins, dtype=int)
    remaining = k
    for b in range(n_bins):
        if bin_counts[b] > 0:
            samples_per_bin[b] = max(1, int(k * bin_counts[b] / len(test_idx)))
    # adjust to exactly k
    total = samples_per_bin.sum()
    while total < k:
        # add to largest bin
        biggest = np.argmax(bin_counts - samples_per_bin)
        samples_per_bin[biggest] += 1
        total += 1
    while total > k:
        biggest = np.argmax(samples_per_bin)
        samples_per_bin[biggest] -= 1
        total -= 1

    for b in range(n_bins):
        bin_idx = test_idx[bins == b]
        if len(bin_idx) == 0:
            continue
        n_sample = min(samples_per_bin[b], len(bin_idx))
        if n_sample > 0:
            selected.extend(rng.choice(bin_idx, n_sample, replace=False))

    # if we didn't get enough (edge cases), fill randomly
    if len(selected) < k:
        remaining_pool = np.setdiff1d(test_idx, selected)
        extra = rng.choice(remaining_pool, k - len(selected), replace=False)
        selected.extend(extra)

    return np.array(selected[:k])


def select_diverse(test_idx, y_test, k, rng, X_fp=None, **kwargs):
    """MaxMin diversity picking based on Tanimoto distance."""
    if X_fp is None or len(test_idx) <= k:
        return select_random(test_idx, y_test, k, rng)

    X_test = X_fp[test_idx]
    # start with random seed point
    selected_local = [rng.randint(len(test_idx))]

    for _ in range(k - 1):
        sel_fps = X_test[selected_local]
        sims = _tanimoto_chunk(X_test, sel_fps)
        max_sim = sims.max(axis=1)  # max similarity to any selected
        # mask already selected
        max_sim[selected_local] = 2.0  # won't be picked (we want min sim)
        # pick the one with minimum max similarity (most diverse)
        next_pick = np.argmin(max_sim)
        selected_local.append(next_pick)

    return test_idx[np.array(selected_local)]


def select_uncertainty(test_idx, y_test, k, rng, pred_probs=None, **kwargs):
    """Select examples closest to decision boundary (pred ~0.5)."""
    if pred_probs is None:
        return select_random(test_idx, y_test, k, rng)

    # sort by distance to 0.5
    dist = np.abs(pred_probs - 0.5)
    # pick k with smallest distance (most uncertain)
    topk = np.argsort(dist)[:k]
    return test_idx[topk]


# ---- Main evaluation ----

STRATEGIES = {
    'random': select_random,
    'stratified': select_stratified,
    'diverse': select_diverse,
    'uncertainty': select_uncertainty,
}


def evaluate_strategy(X, y, targets, eligible, strategy_fn, k=5,
                      seeds=[42], n_reps=5, X_fp=None):
    """Two-stage LOTO with strategy-guided k-shot selection."""
    unique_targets = [t for t in np.unique(targets) if t in eligible]
    all_aurocs = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        per_target = {}

        for tgt in unique_targets:
            test_mask = targets == tgt
            train_mask = ~test_mask
            test_idx = np.where(test_mask)[0]
            y_test_all = y[test_idx]

            if len(np.unique(y_test_all)) < 2 or len(test_idx) <= k + 2:
                continue

            # Stage 1: train baseline RF to get predictions on test set
            clf_stage1 = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=3,
                class_weight='balanced', random_state=seed, n_jobs=-1,
            )
            clf_stage1.fit(X[train_mask], y[train_mask])
            pred_probs = clf_stage1.predict_proba(X[test_idx])[:, 1]

            rep_aucs = []
            for rep in range(n_reps):
                # select k examples using strategy
                shot_idx = strategy_fn(
                    test_idx, y_test_all, k, rng,
                    pred_probs=pred_probs, X_fp=X_fp,
                )

                remaining_idx = np.setdiff1d(test_idx, shot_idx)
                if len(remaining_idx) < 3 or len(np.unique(y[remaining_idx])) < 2:
                    continue

                # Stage 2: retrain with k shots added
                aug_idx = np.concatenate([np.where(train_mask)[0], shot_idx])
                clf = RandomForestClassifier(
                    n_estimators=200, min_samples_leaf=3,
                    class_weight='balanced', random_state=seed + rep, n_jobs=-1,
                )
                clf.fit(X[aug_idx], y[aug_idx])
                y_pred = clf.predict_proba(X[remaining_idx])[:, 1]

                try:
                    rep_aucs.append(roc_auc_score(y[remaining_idx], y_pred))
                except ValueError:
                    pass

            if rep_aucs:
                per_target[tgt] = float(np.mean(rep_aucs))

        all_aurocs.extend(per_target.values())
        # print(f"    seed {seed}: {np.mean(list(per_target.values())):.3f}")

    return {
        'mean_auroc': float(np.mean(all_aurocs)) if all_aurocs else 0.0,
        'std_auroc': float(np.std(all_aurocs)) if all_aurocs else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='Few-shot selection strategies (k=5)')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--output', default='results/fewshot_strategies.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading data...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values

    print('Computing Morgan FPs...')
    X = compute_morgan(smiles)
    eligible = get_eligible_targets(df)
    print(f'Eligible targets: {len(eligible)}, k={args.k}')

    results = {}
    for name, fn in STRATEGIES.items():
        print(f'\n[{name}] evaluating...')
        res = evaluate_strategy(X, y, targets, eligible, fn,
                                k=args.k, seeds=seeds, X_fp=X)
        results[name] = res
        print(f'  {name}: {res["mean_auroc"]:.3f}')

    # summary
    print(f'\n{"="*40}')
    print(f'{"Strategy":<15s} {"AUROC":>8s}')
    print(f'{"-"*25}')
    for name in ['stratified', 'random', 'diverse', 'uncertainty']:
        if name in results:
            print(f'{name:<15s} {results[name]["mean_auroc"]:>8.3f}')

    # pairwise: stratified vs each
    # (need per-target scores for proper test, here we just report means)
    best = max(results.items(), key=lambda x: x[1]['mean_auroc'])
    print(f'\nBest strategy: {best[0]} ({best[1]["mean_auroc"]:.3f})')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        'method': 'fewshot_strategies',
        'k': args.k, 'seeds': seeds,
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
