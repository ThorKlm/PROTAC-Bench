"""One LOTO loop for every arm, over the cohort defined in src.cohort.

Feature blocks are assembled per fold so that nothing derived from labels or
from the held-out rows leaks across the split. The warhead transfer rate is
computed from the training rows and training labels alone, as in
signals/full_stack.py. Few-shot arms move k held-out rows into training and
score the remainder, averaged over repetitions.
"""
import json, time, zlib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src.cohort import masks, meta, usable
from signals.full_stack import (compute_transfer_rate, _tanimoto_chunk,
                                select_stratified)

def transfer_self_excluded(Xtr, ytr, k=10):
    """Training-row transfer rate with the row's own label excluded."""
    s = _tanimoto_chunk(Xtr, Xtr)
    np.fill_diagonal(s, -1.0)
    idx = (np.argpartition(s, -k, axis=1)[:, -k:] if k < s.shape[1]
           else np.tile(np.arange(s.shape[1]), (s.shape[0], 1)))
    return ytr[idx].mean(axis=1).astype(np.float32)

# folds are independent and fold() is order-independent, so parallelism goes
# over folds and each forest stays single-threaded
NJOBS = 24
RF = dict(n_estimators=200, min_samples_leaf=3, class_weight='balanced', n_jobs=1)

def provenance():
    # recorded per artefact: seeds alone do not pin a result across environments
    import subprocess, sys, numpy, sklearn
    try:
        h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True).stdout.strip()
    except Exception:
        h = ''
    return dict(rf={k: x for k, x in RF.items() if k != 'n_jobs'},
                morgan_bits=2048, radius=2, git_commit=h,
                python=sys.version.split()[0], numpy=numpy.__version__,
                sklearn=sklearn.__version__)

def features(c, X, Xa, tr, te, use_morgan, use_warhead, use_admet, self_excl=False):
    ftr, fte = [], []
    if use_morgan:
        ftr.append(X[tr]); fte.append(X[te])
    if use_warhead:
        ytr = c['y'][tr]
        a = (transfer_self_excluded(X[tr], ytr, k=10) if self_excl
             else compute_transfer_rate(X[tr], X[tr], ytr, k=10))
        b = compute_transfer_rate(X[te], X[tr], ytr, k=10)
        ftr.append(np.asarray(a).reshape(-1, 1)); fte.append(np.asarray(b).reshape(-1, 1))
    if use_admet:
        ftr.append(Xa[tr]); fte.append(Xa[te])
    return np.column_stack(ftr), np.column_stack(fte)

def fold(c, X, Xa, t, seed, spec):
    """One fold. Few-shot arms follow signals/full_stack.py: stage-1 model for
    shot selection by predicted-probability band, then one AUROC per repetition
    averaged over repetitions. The selection RNG is keyed on seed and target so
    it does not depend on iteration order."""
    tr, te = masks(t, c)
    Ftr, Fte = features(c, X, Xa, tr, te, spec.get('morgan', True),
                        spec.get('warhead', False), spec.get('admet', False),
                        spec.get('self_excl', False))
    ytr, yte = c['y'][tr], c['y'][te]
    rf = dict(RF); rf.update(spec.get('rf', {}))
    k = spec.get('k', 0)
    if not k:
        clf = RandomForestClassifier(random_state=seed, **rf).fit(Ftr, ytr)
        p = clf.predict_proba(Fte)[:, 1]
        return (float(roc_auc_score(yte, p)), float(average_precision_score(yte, p)),
                int(len(yte)), p)
    s1 = RandomForestClassifier(random_state=seed, **rf).fit(Ftr, ytr)
    probs = s1.predict_proba(Fte)[:, 1]
    rng = np.random.RandomState((seed * 1000003 + zlib.crc32(t.encode())) % (2 ** 32))
    loc = np.arange(len(yte))
    au, ap = [], []
    for rep in range(spec.get('reps', 5)):
        shot = select_stratified(loc, yte, k, rng, probs)
        rest = np.setdiff1d(loc, shot)
        if len(rest) < 3 or len(np.unique(yte[rest])) < 2: continue
        clf = RandomForestClassifier(random_state=seed + rep, **rf).fit(
            np.vstack([Ftr, Fte[shot]]), np.concatenate([ytr, yte[shot]]))
        q = clf.predict_proba(Fte[rest])[:, 1]
        au.append(float(roc_auc_score(yte[rest], q)))
        ap.append(float(average_precision_score(yte[rest], q)))
    if not au: return None
    return float(np.mean(au)), float(np.mean(ap)), int(len(yte)), None

def run(name, spec, c, X, Xa, seeds, cohort=None, out=None, store=True):
    tgts = cohort or c['cohort']
    t0, per_seed, preds = time.time(), {}, {}
    for s in seeds:
        use = [t for t in tgts if usable(t, c)]
        rs = Parallel(n_jobs=NJOBS)(delayed(fold)(c, X, Xa, t, s, spec) for t in use)
        d = {}
        for t, r in zip(use, rs):
            if r is None: continue
            auroc, ap, n, p = r
            d[t] = dict(auroc=auroc, ap=ap, n=n)
            if store and p is not None:
                preds.setdefault(str(s), {})[t] = dict(
                    y=c['y'][masks(t, c)[1]].tolist(), p=p.tolist())
        per_seed[str(s)] = d
        print(f'  {name} seed {s:2d} n={len(d)} macro '
              f'{np.mean([v["auroc"] for v in d.values()]):.4f}', flush=True)
    mac = [float(np.mean([v['auroc'] for v in per_seed[str(s)].values()])) for s in seeds]
    aps = [float(np.mean([v['ap'] for v in per_seed[str(s)].values()])) for s in seeds]
    tg = sorted(set.intersection(*[set(per_seed[str(s)]) for s in seeds]))
    pt = np.array([np.mean([per_seed[str(s)][t]['auroc'] for s in seeds]) for t in tg])
    res = dict(arm=name, spec=spec, macro_auroc=round(float(np.mean(mac)), 4),
               seed_sd=round(float(np.std(mac, ddof=1)), 4),
               mean_ap=round(float(np.mean(aps)), 4),
               ap_seed_sd=round(float(np.std(aps, ddof=1)), 4),
               per_target_sd=round(float(pt.std(ddof=1)), 4),
               target_se=round(float(pt.std(ddof=1) / len(pt) ** 0.5), 4),
               n_targets_paired=len(tg),
               per_seed_macro={str(s): round(m, 4) for s, m in zip(seeds, mac)},
               per_target={t: round(float(v), 4) for t, v in zip(tg, pt)},
               elapsed_min=round((time.time() - t0) / 60, 1), _meta={**meta(c, seeds), **provenance()})
    if out:
        json.dump(res, open(out, 'w'), indent=1)
        if store and preds:
            json.dump(preds, open(out.replace('.json', '_predictions.json'), 'w'))
    return res
