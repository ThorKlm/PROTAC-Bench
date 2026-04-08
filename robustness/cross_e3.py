#!/usr/bin/env python
"""Cross-E3 evaluation: does chemistry transfer across E3 ligase types?

Train on one E3 type, test on the other:
  C0: Train CRBN → test VHL  (expected ~0.606)
  C1: Train VHL  → test CRBN (expected ~0.643)
  C2: Standard LOTO reference
  C3: 2-fold E3 cross-validation
"""

import sys, os, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets


def auroc_safe(y_true, y_score):
    if len(set(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


def train_predict_rf(X_train, y_train, X_test, seed):
    clf = RandomForestClassifier(n_estimators=256, max_depth=None,
                                  min_samples_leaf=3, n_jobs=-1, random_state=seed)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def main():
    parser = argparse.ArgumentParser(description='Cross-E3 evaluation')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--output', default='results/cross_e3.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    df = load_dataset(args.data)
    print(f'Total: {len(df)}')

    # split by E3 type
    df['e3_group'] = df['e3_name'].apply(
        lambda x: 'VHL' if x == 'VHL' else ('CRBN' if x == 'CRBN' else 'other'))
    for g, cnt in df['e3_group'].value_counts().items():
        print(f'  {g}: {cnt}')

    vhl = df[df['e3_group'] == 'VHL']
    crbn = df[df['e3_group'] == 'CRBN']
    other_e3 = df[df['e3_group'] == 'other']

    print('Computing fingerprints...')
    all_fps = compute_morgan(df['smiles'].tolist())

    fps_vhl = all_fps[vhl.index.values]
    fps_crbn = all_fps[crbn.index.values]
    fps_other = all_fps[other_e3.index.values] if len(other_e3) > 0 else np.empty((0, 2048))

    # C0: Train CRBN → test VHL
    print('\nC0: Train CRBN -> test VHL')
    c0_aurocs = []
    for seed in seeds:
        probs = train_predict_rf(fps_crbn, crbn['label'].values, fps_vhl, seed)
        c0_aurocs.append(auroc_safe(vhl['label'].values, probs))
        print(f'  seed={seed}: {c0_aurocs[-1]:.4f}')
    c0_mean = np.mean(c0_aurocs)
    print(f'  C0 mean: {c0_mean:.4f}')

    # C1: Train VHL → test CRBN
    print('\nC1: Train VHL -> test CRBN')
    c1_aurocs = []
    for seed in seeds:
        probs = train_predict_rf(fps_vhl, vhl['label'].values, fps_crbn, seed)
        c1_aurocs.append(auroc_safe(crbn['label'].values, probs))
        print(f'  seed={seed}: {c1_aurocs[-1]:.4f}')
    c1_mean = np.mean(c1_aurocs)
    print(f'  C1 mean: {c1_mean:.4f}')

    # C2: Standard LOTO
    print('\nC2: Standard LOTO')
    eligible = get_eligible_targets(df)
    c2_target_aurocs = []
    for tgt in eligible:
        mask_test = df['target_uniprot'] == tgt
        mask_train = ~mask_test
        X_train = all_fps[mask_train.values]
        y_train = df.loc[mask_train, 'label'].values
        X_test = all_fps[mask_test.values]
        y_test = df.loc[mask_test, 'label'].values
        seed_aucs = []
        for seed in seeds:
            probs = train_predict_rf(X_train, y_train, X_test, seed)
            seed_aucs.append(auroc_safe(y_test, probs))
        c2_target_aurocs.append(np.mean(seed_aucs))
    c2_mean = np.mean(c2_target_aurocs)
    print(f'  LOTO mean: {c2_mean:.4f}')

    # C3: 2-fold E3 cross-val
    print('\nC3: 2-fold E3 cross-val')
    y_other = other_e3['label'].values if len(other_e3) > 0 else np.array([])
    c3_fold_aurocs = []
    for seed in seeds:
        # fold 1: train CRBN+other, test VHL
        X_tr1 = np.vstack([fps_crbn, fps_other]) if len(fps_other) > 0 else fps_crbn
        y_tr1 = np.concatenate([crbn['label'].values, y_other]) if len(y_other) > 0 else crbn['label'].values
        a1 = auroc_safe(vhl['label'].values, train_predict_rf(X_tr1, y_tr1, fps_vhl, seed))
        # fold 2: train VHL+other, test CRBN
        X_tr2 = np.vstack([fps_vhl, fps_other]) if len(fps_other) > 0 else fps_vhl
        y_tr2 = np.concatenate([vhl['label'].values, y_other]) if len(y_other) > 0 else vhl['label'].values
        a2 = auroc_safe(crbn['label'].values, train_predict_rf(X_tr2, y_tr2, fps_crbn, seed))
        c3_fold_aurocs.append((a1 + a2) / 2)
        print(f'  seed={seed}: VHL={a1:.4f}, CRBN={a2:.4f}, mean={c3_fold_aurocs[-1]:.4f}')
    c3_mean = np.mean(c3_fold_aurocs)
    print(f'  C3 mean: {c3_mean:.4f}')

    # summary
    print(f'\n{"="*55}')
    print(f'{"Condition":<35} {"AUROC":>7} {"vs LOTO":>8}')
    print(f'{"-"*55}')
    print(f'{"C0: Train CRBN -> Test VHL":<35} {c0_mean:>7.4f} {c0_mean-c2_mean:>+8.4f}')
    print(f'{"C1: Train VHL -> Test CRBN":<35} {c1_mean:>7.4f} {c1_mean-c2_mean:>+8.4f}')
    print(f'{"C2: Standard LOTO (ref)":<35} {c2_mean:>7.4f} {"---":>8}')
    print(f'{"C3: 2-fold E3 CV":<35} {c3_mean:>7.4f} {c3_mean-c2_mean:>+8.4f}')

    result = {
        'dataset_split': {
            'VHL': int(len(vhl)), 'CRBN': int(len(crbn)),
            'other': int(len(other_e3)), 'total': int(len(df)),
        },
        'conditions': {
            'C0_train_CRBN_test_VHL': {
                'aurocs_by_seed': {str(s): round(a, 4) for s, a in zip(seeds, c0_aurocs)},
                'mean_auroc': round(c0_mean, 4),
            },
            'C1_train_VHL_test_CRBN': {
                'aurocs_by_seed': {str(s): round(a, 4) for s, a in zip(seeds, c1_aurocs)},
                'mean_auroc': round(c1_mean, 4),
            },
            'C2_standard_LOTO': {
                'n_targets': len(eligible),
                'mean_auroc': round(c2_mean, 4),
            },
            'C3_2fold_E3_crossval': {
                'aurocs_by_seed': {str(s): round(a, 4) for s, a in zip(seeds, c3_fold_aurocs)},
                'mean_auroc': round(c3_mean, 4),
            },
        },
        'seeds': seeds,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
