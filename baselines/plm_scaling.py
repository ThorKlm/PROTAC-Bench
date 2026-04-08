#!/usr/bin/env python
"""PLM scaling experiment: ESM-2 embeddings at different scales + Morgan FP.

Loads precomputed PLM embeddings from cache/ and concatenates with Morgan FPs.
Evaluates under LOTO with RF. Does NOT run ESM-2 inference -- expects .npy files.

Expected .npy naming: prot_esm2_{scale}_targets.npy, target_ids.npy
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.evaluation import loto_evaluate
from src.stats import format_result

PLM_SCALES = ['8M', '35M', '150M', '650M', '3B']


def rf_model_fn(X_train, y_train, X_test):
    clf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight='balanced',
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def load_plm_embeddings(emb_dir, scale, target_ids_map, targets):
    """Load PLM embeddings and align to dataset targets.

    Returns embedding matrix of shape (n_samples, emb_dim) or None.
    """
    fpath = os.path.join(emb_dir, f'prot_esm2_{scale}_targets.npy')
    if not os.path.exists(fpath):
        return None

    emb_matrix = np.load(fpath)  # (n_unique_targets, emb_dim)
    # target_ids_map: maps index in emb_matrix -> UniProt ID
    # we need to broadcast to per-sample embeddings
    id_to_idx = {tid: i for i, tid in enumerate(target_ids_map)}

    sample_embs = np.zeros((len(targets), emb_matrix.shape[1]), dtype=np.float32)
    for i, t in enumerate(targets):
        if t in id_to_idx:
            sample_embs[i] = emb_matrix[id_to_idx[t]]
        # else: zero vector (unknown target)

    return sample_embs


def main():
    parser = argparse.ArgumentParser(description='PLM scaling experiment')
    parser.add_argument('--embeddings-dir', default='../cache',
                        help='directory with precomputed .npy embeddings')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--output', default='results/plm_scaling.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    emb_dir = args.embeddings_dir

    print('Loading dataset...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values
    eligible = get_eligible_targets(df)

    print('Computing Morgan fingerprints...')
    X_morgan = compute_morgan(smiles)

    # load target ID mapping
    tid_path = os.path.join(emb_dir, 'target_ids.npy')
    if os.path.exists(tid_path):
        target_ids_map = np.load(tid_path, allow_pickle=True)
    else:
        print(f'WARNING: {tid_path} not found, cannot load PLM embeddings')
        target_ids_map = None

    # run Morgan-only baseline first
    print('\n--- Morgan-only (baseline) ---')
    res_morgan = loto_evaluate(X_morgan, y, targets, rf_model_fn,
                                eligible=eligible, seeds=seeds)
    print(format_result(res_morgan['mean_auroc'], res_morgan['std_auroc'],
                        res_morgan['n_seeds']))

    all_results = {
        'morgan_only': {
            'mean_auroc': res_morgan['mean_auroc'],
            'std_auroc': res_morgan['std_auroc'],
        }
    }

    # run each PLM scale
    for scale in PLM_SCALES:
        print(f'\n--- ESM-2 {scale} + Morgan ---')

        if target_ids_map is None:
            print(f'  SKIPPED (no target_ids.npy)')
            all_results[scale] = {'skipped': True, 'reason': 'no target_ids.npy'}
            continue

        plm_embs = load_plm_embeddings(emb_dir, scale, target_ids_map, targets)
        if plm_embs is None:
            print(f'  WARNING: prot_esm2_{scale}_targets.npy not found, skipping')
            all_results[scale] = {'skipped': True, 'reason': 'npy not found'}
            continue

        X_concat = np.hstack([X_morgan, plm_embs])
        print(f'  Feature dim: {X_concat.shape[1]} (Morgan={X_morgan.shape[1]} + PLM={plm_embs.shape[1]})')

        res = loto_evaluate(X_concat, y, targets, rf_model_fn,
                           eligible=eligible, seeds=seeds)
        print(f'  {format_result(res["mean_auroc"], res["std_auroc"], res["n_seeds"])}')

        all_results[scale] = {
            'mean_auroc': res['mean_auroc'],
            'std_auroc': res['std_auroc'],
            'emb_dim': int(plm_embs.shape[1]),
        }

    # print summary table
    print('\n' + '='*50)
    print(f'{"Scale":<12} {"AUROC":<15} {"Dim":<8}')
    print('-'*50)
    print(f'{"Morgan":<12} {res_morgan["mean_auroc"]:.3f} +/- {res_morgan["std_auroc"]:.3f}')
    for scale in PLM_SCALES:
        r = all_results.get(scale, {})
        if r.get('skipped'):
            print(f'ESM-2 {scale:<6} {"SKIPPED":<15}')
        else:
            print(f'ESM-2 {scale:<6} {r["mean_auroc"]:.3f} +/- {r["std_auroc"]:.3f}    {r.get("emb_dim", "?")}')
    print('='*50)

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_results['seeds'] = seeds
    all_results['method'] = 'PLM scaling (ESM-2 + Morgan + RF)'
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
