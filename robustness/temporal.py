#!/usr/bin/env python
"""Temporal prospective validation: train on old data, predict new entries.

Source-based temporal proxy:
  tpddb    = PROTAC-DB entries (pre-2023)
  protac8k = Ribes et al. 2024 entries

Conditions:
  C0: Morgan only                              (~0.560)
  C1: Morgan + warhead_transfer + ADMET7       (~0.599)
  C2: Morgan + warhead + ADMET + k=5 few-shot  (~0.689)
"""

import sys, os, json, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, load_admet

K_NEIGHBORS = 10
SIM_THRESHOLD = 0.5
K_SHOT = 5


def auroc_safe(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) >= 2 else 0.5


def compute_warhead_transfer(FP, labels, train_mask, test_mask):
    """Compute warhead_transfer features using only train labels (no leakage)."""
    N = len(FP)
    fp_sum = FP.sum(axis=1)
    feats = np.zeros((N, 3), dtype=np.float32)
    FP_train = FP[train_mask]
    labels_train = labels[train_mask]
    fp_sum_train = fp_sum[train_mask]
    train_indices = np.where(train_mask)[0]

    chunk = 200
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        AB = FP[start:end] @ FP_train.T
        denom = fp_sum[start:end, None] + fp_sum_train[None, :] - AB
        np.maximum(denom, 1e-10, out=denom)
        sim = AB / denom
        # zero self-similarity for train samples
        for i in range(end - start):
            global_idx = start + i
            if train_mask[global_idx]:
                train_pos = np.where(train_indices == global_idx)[0]
                if len(train_pos) > 0:
                    sim[i, train_pos[0]] = 0.0
        k = min(K_NEIGHBORS, sim.shape[1])
        top_k = np.argpartition(sim, -k, axis=1)[:, -k:]
        for i in range(end - start):
            feats[start + i, 0] = labels_train[top_k[i]].mean()
        feats[start:end, 1] = (sim > SIM_THRESHOLD).sum(axis=1)
        feats[start:end, 2] = sim.max(axis=1)
    return feats


def main():
    parser = argparse.ArgumentParser(description='Temporal prospective validation')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--output', default='results/temporal.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--k', type=int, default=K_SHOT, help='few-shot k')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    t0 = time.time()

    df = load_dataset(args.data)
    smiles = df['smiles'].values
    y = df['label'].values.astype(int)
    sources = df['source'].values
    targets = df['target_uniprot'].values

    # temporal split
    train_mask = sources == 'tpddb'
    test_mask = sources == 'protac8k'
    y_train, y_test = y[train_mask], y[test_mask]
    test_targets = targets[test_mask]
    print(f'Train (tpddb): {train_mask.sum()}, Test (protac8k): {test_mask.sum()}')
    print(f'Train activity rate: {y_train.mean():.3f}, Test: {y_test.mean():.3f}')

    shared = set(targets[train_mask]) & set(targets[test_mask])
    novel = set(targets[test_mask]) - set(targets[train_mask])
    print(f'Shared targets: {len(shared)}, Novel: {len(novel)}')

    print('Computing Morgan FP...')
    FP = compute_morgan(list(smiles))
    X_train_morgan = FP[train_mask]
    X_test_morgan = FP[test_mask]

    print('Loading ADMET scores...')
    admet_all = load_admet()
    admet_all = np.nan_to_num(admet_all, nan=0.0)
    admet_train = admet_all[train_mask]
    admet_test = admet_all[test_mask]

    print('Computing warhead_transfer features...')
    wh_feats = compute_warhead_transfer(FP, y, train_mask, test_mask)
    wh_train = wh_feats[train_mask]
    wh_test = wh_feats[test_mask]

    results = {}

    # C0: Morgan only
    print('\nC0: Morgan only')
    c0_aurocs = []
    for seed in seeds:
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                    max_features='sqrt', random_state=seed, n_jobs=-1)
        rf.fit(X_train_morgan, y_train)
        p = rf.predict_proba(X_test_morgan)
        p = p[:, 1] if rf.classes_[1] == 1 else 1 - p[:, 0]
        c0_aurocs.append(float(auroc_safe(y_test, p)))
    results['C0_morgan'] = {
        'aurocs': c0_aurocs,
        'mean_auroc': float(np.mean(c0_aurocs)),
        'std_auroc': float(np.std(c0_aurocs)),
    }
    print(f'  AUROC: {np.mean(c0_aurocs):.4f} +/- {np.std(c0_aurocs):.4f}')

    # C1: Morgan + warhead_transfer + ADMET7
    print('C1: Morgan + warhead + ADMET')
    X_train_c1 = np.hstack([X_train_morgan, wh_train, admet_train])
    X_test_c1 = np.hstack([X_test_morgan, wh_test, admet_test])
    c1_aurocs = []
    for seed in seeds:
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                    max_features='sqrt', random_state=seed, n_jobs=-1)
        rf.fit(X_train_c1, y_train)
        p = rf.predict_proba(X_test_c1)
        p = p[:, 1] if rf.classes_[1] == 1 else 1 - p[:, 0]
        c1_aurocs.append(float(auroc_safe(y_test, p)))
    results['C1_morgan_warhead_admet'] = {
        'aurocs': c1_aurocs,
        'mean_auroc': float(np.mean(c1_aurocs)),
        'std_auroc': float(np.std(c1_aurocs)),
    }
    print(f'  AUROC: {np.mean(c1_aurocs):.4f} +/- {np.std(c1_aurocs):.4f}')

    # C2: C1 + k=5 stratified few-shot
    print(f'C2: C1 + k={args.k} stratified few-shot')
    c2_aurocs = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        train_idx = list(np.where(train_mask)[0])
        test_idx_orig = np.where(test_mask)[0]
        remaining_test = []

        for uid in np.unique(test_targets):
            target_test_idx = test_idx_orig[test_targets == uid]
            target_labels = y[target_test_idx]
            n_t = len(target_test_idx)
            n_pos = target_labels.sum()
            n_neg = n_t - n_pos
            if n_t >= 10 + args.k and n_pos >= 2 and n_neg >= 2:
                pos_idx = target_test_idx[target_labels == 1]
                neg_idx = target_test_idx[target_labels == 0]
                k_pos = max(1, round(args.k * n_pos / n_t))
                k_neg = args.k - k_pos
                k_pos = min(k_pos, len(pos_idx) - 1)
                k_neg = min(k_neg, len(neg_idx) - 1)
                if k_pos + k_neg < 2:
                    remaining_test.extend(target_test_idx)
                    continue
                shot_pos = rng.choice(pos_idx, k_pos, replace=False)
                shot_neg = rng.choice(neg_idx, k_neg, replace=False)
                shot_idx = np.concatenate([shot_pos, shot_neg])
                train_idx.extend(shot_idx)
                remaining_test.extend([i for i in target_test_idx if i not in set(shot_idx)])
            else:
                remaining_test.extend(target_test_idx)

        train_idx = np.array(train_idx)
        remaining_test = np.array(remaining_test)

        # recompute warhead_transfer with augmented train
        aug_train_mask = np.zeros(len(df), dtype=bool)
        aug_train_mask[train_idx] = True
        aug_test_mask = np.zeros(len(df), dtype=bool)
        aug_test_mask[remaining_test] = True
        wh_aug = compute_warhead_transfer(FP, y, aug_train_mask, aug_test_mask)

        X_tr = np.hstack([FP[train_idx], wh_aug[train_idx], admet_all[train_idx]])
        X_te = np.hstack([FP[remaining_test], wh_aug[remaining_test], admet_all[remaining_test]])

        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                    max_features='sqrt', random_state=seed, n_jobs=-1)
        rf.fit(X_tr, y[train_idx])
        p = rf.predict_proba(X_te)
        p = p[:, 1] if rf.classes_[1] == 1 else 1 - p[:, 0]
        c2_aurocs.append(float(auroc_safe(y[remaining_test], p)))
        print(f'  seed={seed}: AUROC={c2_aurocs[-1]:.4f}, '
              f'train={len(train_idx)}, test={len(remaining_test)}')

    results['C2_fullstack_fewshot'] = {
        'aurocs': c2_aurocs,
        'mean_auroc': float(np.mean(c2_aurocs)),
        'std_auroc': float(np.std(c2_aurocs)),
    }
    print(f'  AUROC: {np.mean(c2_aurocs):.4f} +/- {np.std(c2_aurocs):.4f}')

    results['split_info'] = {
        'train_source': 'tpddb',
        'test_source': 'protac8k',
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'shared_targets': len(shared),
        'novel_targets': len(novel),
    }
    results['seeds'] = seeds

    # comparison table
    elapsed = (time.time() - t0) / 60
    print(f'\n{"="*50}')
    print('TEMPORAL VALIDATION')
    print(f'{"="*50}')
    print(f'{"Condition":<35} {"AUROC":>8}')
    print(f'{"-"*45}')
    for k, v in results.items():
        if isinstance(v, dict) and 'mean_auroc' in v:
            print(f'{k:<35} {v["mean_auroc"]:>8.4f}')
    print(f'\nElapsed: {elapsed:.1f} min')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
