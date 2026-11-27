#!/usr/bin/env python
"""Random and scaffold cross-validation for the reference on the full corpus,
with and without the deduplication rule, at the canonical seeds.

Scaffold folds pack Murcko scaffolds greedily into the smallest fold. The head
takes random_state explicitly rather than drawing from the global stream, and
deduplication mirrors the leave-one-target-out rule: training rows whose
canonical SMILES occurs in the held-out fold are dropped.
"""
import json, sys, time
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src.seeds import CANONICAL
from src.data_utils import compute_morgan
from src.evaluation import _get_scaffold
from evaluation.engine import model_for, provenance

RF = dict(n_estimators=200, min_samples_leaf=3, class_weight='balanced', n_jobs=6)
HEAD = sys.argv[1] if len(sys.argv) > 1 else 'rf'
OUT = f'results/protocols{"" if HEAD == "rf" else "_" + HEAD}.json'

def scaffold_folds(smiles, n_folds=5):
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[_get_scaffold(s)].append(i)
    sizes, folds = [0] * n_folds, [[] for _ in range(n_folds)]
    for grp in sorted(groups.values(), key=len, reverse=True):
        j = int(np.argmin(sizes))
        folds[j].extend(grp); sizes[j] += len(grp)
    return [np.array(sorted(f)) for f in folds], len(groups)

def one(folds, X, y, smi, tg, cohort, seed, dedup):
    per_fold, kept, yy, pp, ii = [], [], [], [], []
    for te in folds:
        m = np.ones(len(y), bool); m[te] = False
        if dedup: m &= ~np.isin(smi, smi[te])
        kept.append(int(m.sum()))
        p = model_for(dict(model=HEAD), seed, RF).fit(X[m], y[m]).predict_proba(X[te])[:, 1]
        per_fold.append(float(roc_auc_score(y[te], p)))
        yy.append(y[te]); pp.append(p); ii.append(te)
    yy, pp, ii = np.concatenate(yy), np.concatenate(pp), np.concatenate(ii)
    # macro-mean over the primary cohort from out-of-fold predictions, so the
    # random-split value is aggregated exactly as the cold-split one is
    pt = {}
    for t in cohort:
        m2 = tg[ii] == t
        if m2.sum() and len(np.unique(yy[m2])) > 1:
            pt[t] = float(roc_auc_score(yy[m2], pp[m2]))
    return dict(per_fold=per_fold, mean_of_folds=float(np.mean(per_fold)),
                fold_sd=float(np.std(per_fold, ddof=1)),
                pooled=float(roc_auc_score(yy, pp)),
                macro=float(np.mean(list(pt.values()))) if pt else None,
                n_targets_macro=len(pt), per_target=pt,
                median_train_rows=int(np.median(kept)))

t0 = time.time()
df = pd.read_csv('data/protac_bench.csv')
y = df.label.values
smi = df.smiles.astype(str).values
tg = df.target_uniprot.astype(str).values
cohort = [k for k in json.load(open('data/loto_primary_folds.json'))
          if not k.startswith('_')]
X = compute_morgan(df.smiles.tolist())
sfolds, n_scaffolds = scaffold_folds(df.smiles.tolist())
print(f'rows {len(y)} scaffolds {n_scaffolds} scaffold fold sizes '
      f'{[len(f) for f in sfolds]}', flush=True)
res = {k: {} for k in ('random_retained', 'random_dedup',
                       'scaffold_retained', 'scaffold_dedup')}
for seed in CANONICAL:
    rfolds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)]
    for proto, folds in (('random', rfolds), ('scaffold', sfolds)):
        for dedup in (False, True):
            k = f'{proto}_{"dedup" if dedup else "retained"}'
            res[k][str(seed)] = one(folds, X, y, smi, tg, cohort, seed, dedup)
    print(f'seed {seed} ' + ' '.join(
        f'{k}={res[k][str(seed)]["pooled"]:.4f}' for k in res), flush=True)
agg = {}
for k, per in res.items():
    po = [per[s]['pooled'] for s in per]
    agg[k] = dict(pooled_mean=round(float(np.mean(po)), 4),
                  across_seed_sd=round(float(np.std(po, ddof=1)), 4),
                  mean_within_seed_fold_sd=round(
                      float(np.mean([per[s]['fold_sd'] for s in per])), 4),
                  median_train_rows=int(np.median([per[s]['median_train_rows'] for s in per])),
                  macro_mean=round(float(np.mean([per[s]['macro'] for s in per])), 4),
                  macro_across_seed_sd=round(
                      float(np.std([per[s]['macro'] for s in per], ddof=1)), 4),
                  n_targets_macro=per[list(per)[0]]['n_targets_macro'],
                  n_seeds=len(per))
out = dict(per_seed=res, aggregate=agg,
           _meta=dict(seeds=list(CANONICAL), n_rows=int(len(y)), n_folds=5,
                      n_scaffolds=n_scaffolds, corpus='full',
                      dedup_rule='training rows whose canonical SMILES occurs in the held-out fold',
                      elapsed_min=round((time.time() - t0) / 60, 1), **provenance(spec=dict(model=HEAD))))
json.dump(out, open(OUT, 'w'), indent=1)
for k, a in agg.items(): print(k, a)
print('wrote', OUT)
