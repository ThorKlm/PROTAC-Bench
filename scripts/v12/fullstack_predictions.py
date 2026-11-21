#!/usr/bin/env python
"""Stored probabilities for the full-stack cell 1111 at the canonical seeds.

Mirrors signals/full_stack.py's few-shot path, importing its helpers rather
than patching it. Each test compound gets one probability, averaged over the
repetitions in which it was not drawn as a shot. The macro it reports must
match factorial cell 1111 (0.7061); that is the correctness check.
One seed per invocation, then --merge.
"""
import sys, os, json, glob, argparse, importlib.util
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.seeds import CANONICAL
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets, load_admet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DIR, K, N_REPS = 'results/pending', 5, 5
RF = dict(n_estimators=200, min_samples_leaf=3, max_features='sqrt',
          class_weight='balanced', n_jobs=5)

def engine():
    sp = importlib.util.spec_from_file_location(
        'fs', os.path.join(os.path.dirname(__file__), '..', '..', 'signals', 'full_stack.py'))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def run(seed):
    fs = engine()
    transfer, select = fs.compute_transfer_rate, fs.select_stratified
    df = load_dataset('data/protac_bench.csv')
    y, tg = df['label'].values, df['target_uniprot'].values
    Xm = compute_morgan(df['smiles'].values)
    Xa = np.nan_to_num(load_admet(), nan=0.0)
    elig = get_eligible_targets(df)
    rng = np.random.RandomState(seed)
    per_target, store = {}, {}
    for t in [x for x in np.unique(tg) if x in elig]:
        te_m = tg == t; tr_m = ~te_m
        te_i = np.where(te_m)[0]
        y_tr, y_te = y[tr_m], y[te_i]
        if len(np.unique(y_te)) < 2 or len(te_i) <= K + 2: continue
        tr_rate = transfer(Xm[tr_m], Xm[tr_m], y_tr, k=10).reshape(-1, 1)
        te_rate = transfer(Xm[te_m], Xm[tr_m], y_tr, k=10).reshape(-1, 1)
        Xtr = np.column_stack([Xm[tr_m], tr_rate, Xa[tr_m]])
        Xte = np.column_stack([Xm[te_m], te_rate, Xa[te_m]])
        s1 = RandomForestClassifier(random_state=seed, **RF).fit(Xtr, y_tr)
        pp = s1.predict_proba(Xte)[:, 1]
        psum, pcnt, aucs = np.zeros(len(te_i)), np.zeros(len(te_i)), []
        for rep in range(N_REPS):
            shot = select(np.arange(len(te_i)), y_te, K, rng, pp)
            rem = np.setdiff1d(np.arange(len(te_i)), shot)
            if len(rem) < 3 or len(np.unique(y_te[rem])) < 2: continue
            Xa_tr = np.vstack([Xtr, Xte[shot]]); ya_tr = np.concatenate([y_tr, y_te[shot]])
            clf = RandomForestClassifier(random_state=seed + rep, **RF).fit(Xa_tr, ya_tr)
            yp = clf.predict_proba(Xte[rem])[:, 1]
            psum[rem] += yp; pcnt[rem] += 1
            try: aucs.append(roc_auc_score(y_te[rem], yp))
            except ValueError: pass
        if aucs: per_target[t] = float(np.mean(aucs))
        m = pcnt > 0
        if m.sum():
            store[t] = {'p': np.round(psum[m] / pcnt[m], 5).tolist(),
                        'y': y_te[m].astype(int).tolist()}
    macro = float(np.mean(list(per_target.values()))) if per_target else None
    os.makedirs(DIR, exist_ok=True)
    p = os.path.join(DIR, 'fspred_seed_%d.json' % seed)
    json.dump({'seed': seed, 'macro': macro, 'n_targets': len(per_target),
               'per_target_auroc': per_target, 'predictions': store}, open(p, 'w'))
    print('seed', seed, 'macro', round(macro, 4) if macro else None,
          'targets', len(per_target), 'wrote', p, flush=True)

def merge():
    import statistics as st
    preds, macros = {}, {}
    for q in sorted(glob.glob(os.path.join(DIR, 'fspred_seed_*.json'))):
        d = json.load(open(q)); preds[str(d['seed'])] = d['predictions']; macros[d['seed']] = d['macro']
    if not preds: print('no seed files'); return
    json.dump(preds, open('results/v12/fullstack_predictions.json', 'w'))
    v = [macros[s] for s in sorted(macros)]
    print('macro mean', round(sum(v) / len(v), 4), 'sd', round(st.stdev(v), 4) if len(v) > 1 else 0.0,
          'seeds', sorted(macros))
    print('check against factorial cell 1111 = 0.7061')
    print('wrote results/v12/fullstack_predictions.json')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=CANONICAL)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    merge() if a.merge else run(a.seed)
