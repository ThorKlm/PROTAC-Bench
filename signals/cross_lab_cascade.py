#!/usr/bin/env python
"""Within-target cross-publication cascade at the canonical seeds.

Arms, all on the same held-out paper rows:
  cross_publication  train = all rows except the held-out paper
  loto               train = all rows except the held-out target entirely
  random             test  = a same-size random sample of that target's rows
Random minus cross_publication is the bounded cross-publication shift.
Folds: data/cross_lab_folds.json, row indices into the release CSV.
Writes after every fold and resumes from whatever is on disk.
"""
import sys, os, json, glob, zlib, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.seeds import CANONICAL
from src.data_utils import load_dataset, compute_morgan
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DIR = 'results/pending'
ARMS = ('random', 'cross_publication', 'loto')
RF = dict(n_estimators=200, min_samples_leaf=3, max_features='sqrt',
          class_weight='balanced', n_jobs=6)

def auc(X, y, tr, te, seed):
    if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2: return None
    clf = RandomForestClassifier(random_state=seed, **RF)
    clf.fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))

def run(seed):
    p = os.path.join(DIR, 'cascade_seed_%d.json' % seed)
    out = json.load(open(p))['folds'] if os.path.exists(p) else {}
    df = load_dataset('data/protac_bench.csv')
    y, tg = df['label'].values, df['target_uniprot'].values
    X = compute_morgan(df['smiles'].values)
    folds = json.load(open('data/cross_lab_folds.json'))
    allidx = np.arange(len(y))
    total = sum(len(v['papers']) for k, v in folds.items() if k != '_meta')
    print('seed %d: %d folds total, %d already done' % (seed, total, len(out)), flush=True)
    os.makedirs(DIR, exist_ok=True)
    for tgt, spec in folds.items():
        if tgt == '_meta': continue
        tmask = tg == spec['target_uniprot']
        trows = allidx[tmask]
        for pap in spec['papers']:
            key = '%s|%s' % (tgt, pap['doi'])
            if key in out: continue
            te = np.asarray(pap['test_indices'])
            rng = np.random.RandomState((seed * 1000003 + zlib.crc32(key.encode())) % (2**31 - 1))
            rec = {'n_test': int(len(te))}
            rec['cross_publication'] = auc(X, y, np.setdiff1d(allidx, te), te, seed)
            rec['loto'] = auc(X, y, allidx[~tmask], te, seed)
            pos, neg = trows[y[trows] == 1], trows[y[trows] == 0]
            npos = int((y[te] == 1).sum()); nneg = int(len(te) - npos)
            npos = min(max(npos, 1), max(len(pos) - 1, 0)); nneg = min(max(nneg, 1), max(len(neg) - 1, 0))
            if npos >= 1 and nneg >= 1 and npos + nneg < len(trows):
                ter = np.concatenate([rng.choice(pos, npos, replace=False),
                                      rng.choice(neg, nneg, replace=False)])
                rec['random'] = auc(X, y, np.setdiff1d(allidx, ter), ter, seed)
            else:
                rec['random'] = None
            out[key] = rec
            json.dump({'seed': seed, 'folds': out, 'status': 'partial'}, open(p, 'w'), indent=1)
            print('%d %d/%d %s %s' % (seed, len(out), total, tgt,
                  {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump({'seed': seed, 'folds': out, 'status': 'completed'}, open(p, 'w'), indent=1)
    print('seed %d completed, %d folds' % (seed, len(out)), flush=True)

def merge():
    import statistics as st
    per_seed, status = {}, {}
    for q in sorted(glob.glob(os.path.join(DIR, 'cascade_seed_*.json'))):
        d = json.load(open(q)); per_seed[d['seed']] = d['folds']; status[d['seed']] = d.get('status')
    if not per_seed: print('no seed files'); return
    common = None
    for f in per_seed.values():
        ok = {k for k, v in f.items() if all(isinstance(v.get(a), float) for a in ARMS)}
        common = ok if common is None else (common & ok)
    s = {}
    for a in ARMS:
        m = [sum(per_seed[k][f][a] for f in common) / len(common) for k in sorted(per_seed)]
        s[a] = {'macro_mean': sum(m) / len(m),
                'sample_sd': st.stdev(m) if len(m) > 1 else 0.0,
                'per_seed': dict(zip(sorted(per_seed), m))}
    s['bound_random_minus_crosspub'] = s['random']['macro_mean'] - s['cross_publication']['macro_mean']
    s['crosspub_minus_loto'] = s['cross_publication']['macro_mean'] - s['loto']['macro_mean']
    s['_meta'] = {'seeds': sorted(per_seed), 'seed_status': status,
                  'n_paired_folds': len(common),
                  'rf': {k: v for k, v in RF.items() if k != 'n_jobs'},
                  'fold_source': 'data/cross_lab_folds.json',
                  'note': 'reference configuration with forest depth capped at 12, reproducing the published cascade configuration'}
    json.dump(s, open(os.path.join(DIR, 'cascade_canonical.json'), 'w'), indent=1)
    for a in ARMS: print(a, round(s[a]['macro_mean'], 4), round(s[a]['sample_sd'], 4))
    print('bound', round(s['bound_random_minus_crosspub'], 4),
          'crosspub minus loto', round(s['crosspub_minus_loto'], 4),
          'paired folds', len(common), 'status', status)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=CANONICAL)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    merge() if a.merge else run(a.seed)
