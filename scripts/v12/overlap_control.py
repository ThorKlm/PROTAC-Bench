"""Reference LOTO split by whether the test compound occurs verbatim in training.

Under LOTO a PROTAC assayed against several targets appears in both splits with
only its target annotation differing, and the reference features carry no target
information, so such a compound can be recalled rather than predicted. The novel
column is the generalisation estimate; the seen column is the recall estimate.
Seeding replicates src/evaluation.py exactly: global stream per fold, no
random_state, so the full column is comparable to the published 0.6614.
"""
import json, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from src.seeds import CANONICAL
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets

RF = dict(n_estimators=200, min_samples_leaf=3, class_weight='balanced', n_jobs=8)

def macro(d):
    v = [x for x in d.values() if x is not None]
    return float(np.mean(v)) if v else float('nan')

d = load_dataset('data/protac_bench.csv')
y = d['label'].values
tg_all = d['target_uniprot'].astype(str).values
smi = d['smiles'].values
X = compute_morgan(d['smiles'].tolist())
elig = sorted(set(get_eligible_targets(d)))
out = {'meta': {'seeds': CANONICAL, 'n_targets': len(elig), 'rf': {k: v for k, v in RF.items()}},
       'per_seed': {}}
for seed in CANONICAL:
    cols = {'full': {}, 'novel': {}, 'seen': {}}
    for tg in elig:
        te = tg_all == tg
        tr = ~te
        train_smi = set(smi[tr])
        np.random.seed(seed)
        clf = RandomForestClassifier(**RF).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        yt = y[te]
        cols['full'][tg] = roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else None
        is_seen = np.fromiter((s in train_smi for s in smi[te]), bool, len(yt))
        for name, m in (('novel', ~is_seen), ('seen', is_seen)):
            if m.sum() >= 5 and len(np.unique(yt[m])) > 1:
                cols[name][tg] = roc_auc_score(yt[m], p[m])
    out['per_seed'][str(seed)] = cols
    print(f'seed {seed:2d}  full {macro(cols["full"]):.4f} (n={len(cols["full"])})  '
          f'novel {macro(cols["novel"]):.4f} (n={len(cols["novel"])})  '
          f'seen {macro(cols["seen"]):.4f} (n={len(cols["seen"])})', flush=True)
for name in ('full', 'novel', 'seen'):
    per = [macro(out['per_seed'][str(s)][name]) for s in CANONICAL]
    out[name] = dict(macro=round(float(np.mean(per)), 4),
                     seed_sd=round(float(np.std(per, ddof=1)), 4),
                     n_folds=len(out['per_seed']['42'][name]))
    print(name, out[name])
json.dump(out, open('results/v12/overlap_control.json', 'w'), indent=1)
