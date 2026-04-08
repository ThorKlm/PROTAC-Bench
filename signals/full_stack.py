#!/usr/bin/env python
"""Full-stack: combine Morgan + warhead_transfer + ADMET + k=5 stratified.

Progressive conditions:
  C0: Morgan only (baseline)
  C1: Morgan + warhead_transfer
  C2: Morgan + k=5 stratified few-shot
  C3: Morgan + warhead + k=5
  C4: Morgan + warhead + ADMET + k=5 (FULL STACK)

Expected best: 0.743 +/- 0.012 (C4 with 10 seeds).
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets, load_admet
from src.stats import paired_wilcoxon, format_result, confidence_interval


# --- Tanimoto + transfer (from warhead_transfer.py) ---

def _tanimoto_chunk(A, B, chunk_size=512):
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


def compute_transfer_rate(X_test, X_train, y_train, k=10):
    sims = _tanimoto_chunk(X_test, X_train)
    if k < sims.shape[1]:
        topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
    else:
        topk_idx = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
    rates = np.zeros(X_test.shape[0], dtype=np.float32)
    for i in range(X_test.shape[0]):
        rates[i] = y_train[topk_idx[i]].mean()
    return rates


# --- Stratified k-shot selection (from fewshot_strategies.py) ---

def select_stratified(test_idx, y_test, k, rng, pred_probs):
    """Bin by predicted probability quintiles, sample proportionally."""
    n_bins = 5
    bins = np.digitize(pred_probs, np.linspace(0, 1, n_bins + 1)[1:-1])

    bin_counts = np.bincount(bins, minlength=n_bins)
    samples_per_bin = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        if bin_counts[b] > 0:
            samples_per_bin[b] = max(1, int(k * bin_counts[b] / len(test_idx)))

    total = samples_per_bin.sum()
    while total < k:
        biggest = np.argmax(bin_counts - samples_per_bin)
        samples_per_bin[biggest] += 1
        total += 1
    while total > k:
        biggest = np.argmax(samples_per_bin)
        samples_per_bin[biggest] -= 1
        total -= 1

    selected = []
    for b in range(n_bins):
        bin_idx = test_idx[bins == b]
        n_sample = min(samples_per_bin[b], len(bin_idx))
        if n_sample > 0 and len(bin_idx) > 0:
            selected.extend(rng.choice(bin_idx, n_sample, replace=False))

    if len(selected) < k:
        remaining_pool = np.setdiff1d(test_idx, selected)
        if len(remaining_pool) > 0:
            extra = rng.choice(remaining_pool, min(k - len(selected), len(remaining_pool)), replace=False)
            selected.extend(extra)

    return np.array(selected[:k])


# --- Evaluation engine ---

def evaluate_condition(X_morgan, y, targets, eligible, seeds,
                       use_transfer=False, use_admet=False, use_fewshot=False,
                       X_admet=None, k=5, n_reps=5, label=""):
    """Run LOTO (with optional few-shot) for a given feature configuration.

    For C0/C1: standard LOTO.
    For C2-C4: LOTO with k=5 stratified few-shot.
    """
    unique_targets = [t for t in np.unique(targets) if t in eligible]
    all_seed_means = []
    all_seed_results = {}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        per_target = {}

        for tgt in unique_targets:
            test_mask = targets == tgt
            train_mask = ~test_mask
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]

            y_train = y[train_mask]
            y_test_full = y[test_idx]

            if len(np.unique(y_test_full)) < 2:
                continue
            if use_fewshot and len(test_idx) <= k + 2:
                continue

            # build training features
            feats_train = [X_morgan[train_mask]]
            feats_test_base = [X_morgan[test_mask]]

            if use_transfer:
                tr_rate = compute_transfer_rate(X_morgan[train_mask], X_morgan[train_mask], y_train, k=10)
                te_rate = compute_transfer_rate(X_morgan[test_mask], X_morgan[train_mask], y_train, k=10)
                feats_train.append(tr_rate.reshape(-1, 1))
                feats_test_base.append(te_rate.reshape(-1, 1))

            if use_admet and X_admet is not None:
                feats_train.append(X_admet[train_mask])
                feats_test_base.append(X_admet[test_mask])

            X_tr_full = np.column_stack(feats_train)
            X_te_full = np.column_stack(feats_test_base)

            if not use_fewshot:
                # standard LOTO
                clf = RandomForestClassifier(
                    n_estimators=100, min_samples_leaf=3, max_features='sqrt',
                    class_weight='balanced', random_state=seed, n_jobs=-1,
                )
                clf.fit(X_tr_full, y_train)
                y_pred = clf.predict_proba(X_te_full)[:, 1]
                try:
                    per_target[tgt] = roc_auc_score(y_test_full, y_pred)
                except ValueError:
                    pass
            else:
                # few-shot: get stage-1 preds for selection
                clf_s1 = RandomForestClassifier(
                    n_estimators=100, min_samples_leaf=3, max_features='sqrt',
                    class_weight='balanced', random_state=seed, n_jobs=-1,
                )
                clf_s1.fit(X_tr_full, y_train)
                pred_probs = clf_s1.predict_proba(X_te_full)[:, 1]

                rep_aucs = []
                for rep in range(n_reps):
                    shot_local = select_stratified(
                        np.arange(len(test_idx)), y_test_full, k, rng, pred_probs,
                    )
                    remaining_local = np.setdiff1d(np.arange(len(test_idx)), shot_local)

                    if len(remaining_local) < 3 or len(np.unique(y_test_full[remaining_local])) < 2:
                        continue

                    # augment training
                    X_tr_aug = np.vstack([X_tr_full, X_te_full[shot_local]])
                    y_tr_aug = np.concatenate([y_train, y_test_full[shot_local]])

                    clf = RandomForestClassifier(
                        n_estimators=100, min_samples_leaf=3, max_features='sqrt',
                        class_weight='balanced', random_state=seed + rep, n_jobs=-1,
                    )
                    clf.fit(X_tr_aug, y_tr_aug)
                    y_pred = clf.predict_proba(X_te_full[remaining_local])[:, 1]
                    try:
                        rep_aucs.append(roc_auc_score(y_test_full[remaining_local], y_pred))
                    except ValueError:
                        pass

                if rep_aucs:
                    per_target[tgt] = float(np.mean(rep_aucs))

        all_seed_results[seed] = per_target
        if per_target:
            seed_mean = np.mean(list(per_target.values()))
            all_seed_means.append(seed_mean)
            print(f"  {label} seed={seed}: {seed_mean:.3f} ({len(per_target)} targets)")

    # aggregate
    aurocs_flat = []
    for r in all_seed_results.values():
        aurocs_flat.extend(r.values())
    mean_auc = np.mean(aurocs_flat) if aurocs_flat else 0.0
    std_auc = np.std(all_seed_means) if all_seed_means else 0.0

    return {
        'mean_auroc': float(mean_auc),
        'std_auroc': float(std_auc),
        'n_seeds': len(seeds),
        'per_seed': {str(s): {k: float(v) for k, v in r.items()}
                     for s, r in all_seed_results.items()},
        'seed_means': [float(m) for m in all_seed_means],
    }


def main():
    parser = argparse.ArgumentParser(description='Full stack: reproduce best result')
    parser.add_argument('--seeds', default='42,43,44',
                        help='comma-separated seeds (use 10 for CI)')
    parser.add_argument('--k', type=int, default=5, help='few-shot k')
    parser.add_argument('--output', default='results/full_stack.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading data...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values

    print('Computing Morgan FPs (2048 bits)...')
    X_morgan = compute_morgan(smiles)

    print('Loading ADMET scores...')
    X_admet = load_admet()
    X_admet = np.nan_to_num(X_admet, nan=0.0)

    eligible = get_eligible_targets(df)
    print(f'Eligible targets: {len(eligible)}, k={args.k}, seeds={seeds}')

    conditions = {}

    # C0: Morgan only
    print('\n[C0] Morgan only...')
    conditions['C0'] = evaluate_condition(
        X_morgan, y, targets, eligible, seeds, label='C0')

    # C1: Morgan + warhead_transfer
    print('\n[C1] Morgan + warhead_transfer...')
    conditions['C1'] = evaluate_condition(
        X_morgan, y, targets, eligible, seeds,
        use_transfer=True, label='C1')

    # C2: Morgan + k=5 stratified
    print('\n[C2] Morgan + k=5 stratified...')
    conditions['C2'] = evaluate_condition(
        X_morgan, y, targets, eligible, seeds,
        use_fewshot=True, k=args.k, label='C2')

    # C3: Morgan + warhead + k=5
    print('\n[C3] Morgan + warhead + k=5...')
    conditions['C3'] = evaluate_condition(
        X_morgan, y, targets, eligible, seeds,
        use_transfer=True, use_fewshot=True, k=args.k, label='C3')

    # C4: FULL STACK
    print('\n[C4] Morgan + warhead + ADMET + k=5 (FULL STACK)...')
    conditions['C4'] = evaluate_condition(
        X_morgan, y, targets, eligible, seeds,
        use_transfer=True, use_admet=True, use_fewshot=True,
        X_admet=X_admet, k=args.k, label='C4')

    # --- Additive decomposition table ---
    print(f'\n{"="*60}')
    print(f'  Additive Decomposition')
    print(f'{"="*60}')
    print(f'{"Condition":<45s} {"AUROC":>7s} {"Delta":>7s}')
    print(f'{"-"*60}')

    c0_mean = conditions['C0']['mean_auroc']
    labels = {
        'C0': 'Morgan only (baseline)',
        'C1': 'Morgan + warhead_transfer',
        'C2': 'Morgan + k=5 stratified',
        'C3': 'Morgan + warhead + k=5',
        'C4': 'Morgan + warhead + ADMET + k=5 (FULL)',
    }
    prev = c0_mean
    for cond in ['C0', 'C1', 'C2', 'C3', 'C4']:
        mean = conditions[cond]['mean_auroc']
        delta = mean - c0_mean
        print(f'  {labels[cond]:<43s} {mean:>7.3f} {delta:>+7.3f}')

    # CI for full stack
    if conditions['C4']['seed_means']:
        ci_mean, ci_lo, ci_hi = confidence_interval(conditions['C4']['seed_means'])
        print(f'\nFull stack CI: {ci_mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]')

    # total gain
    total_delta = conditions['C4']['mean_auroc'] - c0_mean
    print(f'Total gain over baseline: {total_delta:+.3f}')

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        'method': 'full_stack',
        'seeds': seeds, 'k': args.k,
        'conditions': {k: v for k, v in conditions.items()},
        'total_delta': float(total_delta),
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
