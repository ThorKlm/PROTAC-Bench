#!/usr/bin/env python
"""LOTO macro-mean under synthetic training-label noise, canonical seeds.

A fraction f of labels is flipped uniformly at random; held-out evaluation uses
the original labels, so the measurement isolates training-label noise. One
(flip rate, seed) unit per invocation, then --merge.
"""
import sys, os, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.seeds import CANONICAL
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DIR = 'results/pending'
FLIPS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
RF = dict(n_estimators=200, min_samples_leaf=3, max_features='sqrt',
          class_weight='balanced', n_jobs=5)

MODE = 'train'

def run(flip, seed):
    df = load_dataset('data/protac_bench.csv')
    y, tg = df['label'].values.astype(int), df['target_uniprot'].values
    X = compute_morgan(df['smiles'].values)
    elig = get_eligible_targets(df)
    rng = np.random.RandomState(seed)
    yn = y.copy()
    if flip > 0:
        idx = rng.choice(len(y), int(round(flip * len(y))), replace=False)
        yn[idx] = 1 - yn[idx]
    per = {}
    for t in [x for x in np.unique(tg) if x in elig]:
        te = tg == t; tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(yn[tr])) < 2: continue
        clf = RandomForestClassifier(random_state=seed, **RF).fit(X[tr], yn[tr])
        yev = yn if MODE == 'both' else y
        if len(np.unique(yev[te])) < 2: continue
        per[t] = float(roc_auc_score(yev[te], clf.predict_proba(X[te])[:, 1]))
    macro = float(np.mean(list(per.values()))) if per else None
    os.makedirs(DIR, exist_ok=True)
    p = os.path.join(DIR, 'noise%s_%s_%d.json' % ('' if MODE == 'train' else '_both', flip, seed))
    json.dump({'flip': flip, 'seed': seed, 'macro': macro,
               'n_targets': len(per), 'per_target': per}, open(p, 'w'))
    print('flip', flip, 'seed', seed, 'macro', round(macro, 4) if macro else None, flush=True)

def merge():
    import statistics as st
    acc = {}
    pat = 'noise_*_*.json' if MODE == 'train' else 'noise_both_*_*.json'
    for q in sorted(glob.glob(os.path.join(DIR, pat))):
        d = json.load(open(q))
        if d['macro'] is not None: acc.setdefault(d['flip'], []).append(d['macro'])
    rows = []
    for f in sorted(acc):
        v = acc[f]
        rows.append({'flip': f, 'n_seeds': len(v), 'macro': sum(v) / len(v),
                     'sample_sd': st.stdev(v) if len(v) > 1 else 0.0})
        print(f, len(v), round(rows[-1]['macro'], 4), round(rows[-1]['sample_sd'], 4))
    if len(rows) > 2:
        xs = np.array([r['flip'] * 100 for r in rows]); ys = np.array([r['macro'] for r in rows])
        sl, ic = np.polyfit(xs, ys, 1)
        res = float(np.sqrt(np.mean((ys - (sl * xs + ic)) ** 2)))
        print('slope per percent', round(float(sl), 5), 'intercept', round(float(ic), 4), 'rms resid', round(res, 4))
        json.dump({'points': rows, 'linear_fit': {'slope_per_percent': float(sl),
                   'intercept': float(ic), 'rms_residual': res},
                   'seeds': CANONICAL, 'note': 'training labels flipped; evaluation on original labels'},
                  open('results/label_noise.json' if MODE == 'train' else 'results/label_noise_both.json', 'w'), indent=1)
        print('wrote', 'results/label_noise.json' if MODE == 'train' else 'results/label_noise_both.json')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--flip', type=float, choices=FLIPS)
    ap.add_argument('--mode', choices=('train', 'both'), default='train')
    ap.add_argument('--seed', type=int, choices=CANONICAL)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    MODE = a.mode
    globals()['MODE'] = a.mode
    merge() if a.merge else run(a.flip, a.seed)
