#!/usr/bin/env python
"""Random 5-fold per-target AUROC for kNN (k=5, Morgan 512-bit, Tanimoto)."""
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

SEEDS = [42, 43, 44]
N_FOLDS = 5
K = 5
NBITS = 512
DATA = '/workspace/PROTAC-Bench/data/protac_bench.csv'
OUT = '/workspace/PROTAC-Bench/results/per_target/kNN_random_per_target.csv'


def compute_morgan_512(smiles_list):
    fps = np.zeros((len(smiles_list), NBITS), dtype=np.uint8)
    valid = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=NBITS)
            fps[i] = np.array(fp, dtype=np.uint8)
            valid[i] = True
    return fps, valid


def main():
    print('Loading data...', flush=True)
    df = pd.read_csv(DATA)
    print(f'N = {len(df)}', flush=True)

    print(f'Computing Morgan-{NBITS} fingerprints...', flush=True)
    X, valid = compute_morgan_512(df['smiles'].values)
    y = df['label'].values.astype(int)
    t = df['target_uniprot'].values
    print(f'Valid FPs: {valid.sum()}/{len(df)}', flush=True)

    # Only use valid rows
    X = X[valid]
    y_v = y[valid]
    t_v = t[valid]
    n = len(X)

    per_target_aurocs = defaultdict(list)

    t0 = time.time()
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(np.arange(n), y_v)):
            # sklearn 'jaccard' on binary vectors = Tanimoto distance
            clf = KNeighborsClassifier(n_neighbors=K, metric='jaccard', algorithm='brute', n_jobs=-1)
            clf.fit(X[tr], y_v[tr])
            proba = clf.predict_proba(X[te])
            # handle case of only one class in train
            if proba.shape[1] == 1:
                only_cls = clf.classes_[0]
                preds = np.full(len(te), float(only_cls))
            else:
                # column for class=1
                idx1 = list(clf.classes_).index(1)
                preds = proba[:, idx1]

            t_te = t_v[te]
            y_te = y_v[te]
            for tgt in np.unique(t_te):
                mask = t_te == tgt
                if mask.sum() < 2 or len(np.unique(y_te[mask])) < 2:
                    continue
                try:
                    auc = roc_auc_score(y_te[mask], preds[mask])
                except ValueError:
                    continue
                per_target_aurocs[tgt].append(auc)

            elapsed = time.time() - t0
            print(f'  [kNN] seed={seed} fold={fold+1}/{N_FOLDS} '
                  f'targets={len(per_target_aurocs)} elapsed={elapsed:.0f}s', flush=True)

    rows = []
    for tgt, aucs in per_target_aurocs.items():
        rows.append({'target': tgt,
                     'random_auroc': round(float(np.mean(aucs)), 4),
                     'n_entries': int((t == tgt).sum()),
                     'n_fold_evals': len(aucs)})

    out_df = pd.DataFrame(rows).sort_values('target').reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f'Saved {OUT}: {len(out_df)} targets, mean random_auroc={out_df.random_auroc.mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
