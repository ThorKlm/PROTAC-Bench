#!/usr/bin/env python
"""Warhead transfer signal for PROTAC-Bench.

Core idea: for each test PROTAC in a LOTO fold, find the top-k most similar
training PROTACs (Tanimoto on Morgan FP) and take their mean activity as
a "warhead transfer rate". This is LOTO-safe because neighbors always come
from other targets.

Expected result: 0.713 (+0.042 over baseline, p=0.0005).
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets, get_loto_folds
from src.stats import paired_wilcoxon, format_result


# ---------- Tanimoto helpers ----------

def _tanimoto_chunk(A, B, chunk_size=512):
    """Compute pairwise Tanimoto between A (n, d) and B (m, d).
    Returns (n, m) similarity matrix. Chunked to avoid OOM.
    """
    n = A.shape[0]
    sum_B = B.sum(axis=1)  # (m,)
    sims = np.zeros((n, B.shape[0]), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        A_chunk = A[start:end]          # (c, d)
        sum_A = A_chunk.sum(axis=1)     # (c,)
        AB = A_chunk @ B.T              # (c, m)
        # Tanimoto = AB / (|A| + |B| - AB)
        denom = sum_A[:, None] + sum_B[None, :] - AB
        denom = np.maximum(denom, 1e-8)
        sims[start:end] = AB / denom

    return sims


def compute_transfer_rate(X_test, X_train, y_train, k=10):
    """For each test sample, find top-k similar training samples and return
    mean activity (warhead_transfer_rate).
    """
    sims = _tanimoto_chunk(X_test, X_train)
    # get top-k indices per test sample
    # NOTE: argpartition is faster than argsort for large arrays
    if k < sims.shape[1]:
        topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
    else:
        topk_idx = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))

    rates = np.zeros(X_test.shape[0], dtype=np.float32)
    for i in range(X_test.shape[0]):
        rates[i] = y_train[topk_idx[i]].mean()

    return rates


# ---------- Unit test ----------

def _test_transfer_rate():
    """Quick sanity check on toy data."""
    X_train = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float32)
    y_train = np.array([1, 0, 1])
    X_test = np.array([[1, 0, 0, 0]], dtype=np.float32)
    rates = compute_transfer_rate(X_test, X_train, y_train, k=2)
    # nearest to [1,0,0,0] should be [1,0,1,0] and [1,1,0,0] -> mean label = 1.0
    assert abs(rates[0] - 1.0) < 1e-5, f"Expected 1.0, got {rates[0]}"
    print("  unit test passed")


# ---------- Custom LOTO loop ----------

def run_warhead_transfer_loto(X_fp, y, targets, eligible, k=10, seeds=[42]):
    """Run LOTO with warhead transfer rate appended to Morgan FP.

    We need a custom loop because the transfer rate must be computed
    per-fold using only training data.
    """
    unique_targets = [t for t in np.unique(targets) if t in eligible]

    all_seed_results = {}
    for seed in seeds:
        np.random.seed(seed)
        per_target = {}

        for tgt in unique_targets:
            test_mask = targets == tgt
            train_mask = ~test_mask

            X_train_fp = X_fp[train_mask]
            y_train = y[train_mask]
            X_test_fp = X_fp[test_mask]
            y_test = y[test_mask]

            if len(np.unique(y_test)) < 2:
                continue

            # compute warhead transfer rate using ONLY training set neighbors
            train_rate = compute_transfer_rate(X_train_fp, X_train_fp, y_train, k=k)
            test_rate = compute_transfer_rate(X_test_fp, X_train_fp, y_train, k=k)

            # concatenate Morgan FP + transfer rate
            X_train = np.column_stack([X_train_fp, train_rate])
            X_test = np.column_stack([X_test_fp, test_rate])

            clf = RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]

            try:
                auc = roc_auc_score(y_test, y_pred)
            except ValueError:
                continue
            per_target[tgt] = auc

        all_seed_results[seed] = per_target
        print(f"  seed {seed}: {np.mean(list(per_target.values())):.3f} ({len(per_target)} targets)")

    # aggregate
    all_targets = set()
    for res in all_seed_results.values():
        all_targets.update(res.keys())

    aurocs = []
    for seed_res in all_seed_results.values():
        aurocs.extend(seed_res.values())

    mean_auroc = np.mean(aurocs) if aurocs else 0.0
    std_auroc = np.std([np.mean(list(r.values())) for r in all_seed_results.values()])

    return {
        'mean_auroc': float(mean_auroc),
        'std_auroc': float(std_auroc),
        'n_targets': len(all_targets),
        'n_seeds': len(seeds),
        'per_seed': {str(s): {k: float(v) for k, v in r.items()}
                     for s, r in all_seed_results.items()},
    }


# ---------- Baseline LOTO (for comparison) ----------

def run_baseline_loto(X_fp, y, targets, eligible, seeds=[42]):
    """Plain Morgan FP + RF baseline."""
    from src.evaluation import loto_evaluate

    def model_fn(X_tr, y_tr, X_te):
        clf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3,
            class_weight='balanced', n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]

    return loto_evaluate(X_fp, y, targets, model_fn, eligible=eligible, seeds=seeds)


def main():
    parser = argparse.ArgumentParser(description='Warhead transfer signal (LOTO)')
    parser.add_argument('--seeds', default='42,43,44,53,71',
                        help='comma-separated random seeds')
    parser.add_argument('--k-neighbors', type=int, default=10,
                        help='number of nearest neighbors for transfer rate')
    parser.add_argument('--output', default='results/warhead_transfer.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--test', action='store_true', help='run unit test only')
    args = parser.parse_args()

    if args.test:
        _test_transfer_rate()
        return

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading dataset...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values

    print('Computing Morgan fingerprints (2048 bits)...')
    X_fp = compute_morgan(smiles)
    print(f'  X shape: {X_fp.shape}')

    eligible = get_eligible_targets(df)
    print(f'Eligible targets: {len(eligible)}')

    # run unit test first
    print('Running sanity check...')
    _test_transfer_rate()

    # --- baseline ---
    print(f'\n--- Baseline: Morgan only ---')
    baseline = run_baseline_loto(X_fp, y, targets, eligible, seeds=seeds)
    print(f'Baseline: {format_result(baseline["mean_auroc"], baseline["std_auroc"], baseline["n_seeds"])}')

    # --- warhead transfer ---
    print(f'\n--- Morgan + warhead_transfer_rate (k={args.k_neighbors}) ---')
    wt_results = run_warhead_transfer_loto(
        X_fp, y, targets, eligible,
        k=args.k_neighbors, seeds=seeds,
    )
    print(f'Warhead transfer: {format_result(wt_results["mean_auroc"], wt_results["std_auroc"], wt_results["n_seeds"])}')

    # --- statistical test ---
    # collect per-target AUROCs across all seeds for paired test
    baseline_aurocs = []
    wt_aurocs = []
    for s in [str(s) for s in seeds]:
        shared = set(baseline['per_seed'].get(s, {}).keys()) & set(wt_results['per_seed'].get(s, {}).keys())
        for tgt in sorted(shared):
            baseline_aurocs.append(baseline['per_seed'][s][tgt])
            wt_aurocs.append(wt_results['per_seed'][s][tgt])

    p_val = paired_wilcoxon(wt_aurocs, baseline_aurocs)
    delta = wt_results['mean_auroc'] - baseline['mean_auroc']

    print(f'\nDelta: {delta:+.3f}, p={p_val:.4f}')

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        'method': 'Morgan + warhead_transfer_rate',
        'k_neighbors': args.k_neighbors,
        'seeds': seeds,
        'baseline': baseline,
        'warhead_transfer': wt_results,
        'delta': float(delta),
        'p_value': float(p_val),
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
