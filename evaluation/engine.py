"""One LOTO loop for every arm, over the cohort defined in src.cohort.

Feature blocks are assembled per fold so that nothing derived from labels or
from the held-out rows leaks across the split. The warhead transfer rate is
computed from the training rows and training labels alone, as in
signals/full_stack.py. Few-shot arms move k held-out rows into training and
score the remainder, averaged over repetitions.
"""
import json, os, time, zlib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src.cohort import masks, meta, train_mask, usable, usable_mask
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
NJOBS = int(os.environ.get('PB_NJOBS', 24))
RF = dict(n_estimators=200, min_samples_leaf=3, class_weight='balanced', n_jobs=1)
# exactly the published head from scripts/v12/xgb_protocols.py, so the arm is
# that model on the primary cohort rather than a new model
XGB = dict(n_estimators=300, max_depth=6, n_jobs=1, tree_method='hist',
           eval_metric='logloss')

def model_for(spec, seed, rf):
    """Estimator for one fit. The forest is the default; another head is
    selected by spec['model'], so an arm changes the head without changing the
    fold loop. XGBoost is unweighted, as the published configuration was."""
    name = spec.get('model', 'rf')
    if name == 'rf':
        return RandomForestClassifier(random_state=seed, **rf)
    if name == 'xgb':
        from xgboost import XGBClassifier
        q = dict(XGB); q.update(spec.get('xgb', {}))
        return XGBClassifier(random_state=seed, **q)
    raise ValueError(f'unknown model {name!r}')

def provenance(bits=2048, radius=2, spec=None):
    # recorded per artefact: seeds alone do not pin a result across environments
    import subprocess, sys, numpy, sklearn
    try:
        h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True).stdout.strip()
    except Exception:
        h = ''
    sp = spec or {}
    name = sp.get('model', 'rf')
    eff = ({**RF, **sp.get('rf', {})} if name == 'rf'
           else {**XGB, **sp.get('xgb', {})})
    return dict(model=name,
                model_params={k: x for k, x in eff.items() if k != 'n_jobs'},
                morgan_bits=bits, radius=radius, git_commit=h,
                python=sys.version.split()[0], numpy=numpy.__version__,
                sklearn=sklearn.__version__)

EMB = {}
def embedding(kind, c):
    """Per-row embedding block. Protein vectors come from the per-target matrix
    through the row's accession, averaging the constituents of a composite key."""
    if kind in EMB: return EMB[kind]
    d = 'data/embeddings'
    if kind == 'chemberta':
        M = np.load(f'{d}/chemberta_rows.npy')
    elif kind == 'esm':
        P = np.load(f'{d}/esm2_650m_targets.npy')
        ids = [str(x) for x in np.load(f'{d}/target_ids.npy', allow_pickle=True)]
        pos = {t: i for i, t in enumerate(ids)}
        M = np.zeros((len(c['tg']), P.shape[1]), dtype=np.float32)
        for i, t in enumerate(c['tg']):
            js = [pos[u] for u in str(t).split('/') if u in pos]
            if js: M[i] = P[js].mean(axis=0)
    else:
        raise ValueError(f'unknown embedding {kind!r}')
    if len(M) != len(c['tg']):
        raise ValueError(f'{kind} has {len(M)} rows, corpus has {len(c["tg"])}')
    EMB[kind] = M
    return M

def features(c, X, Xa, tr, te, use_morgan, use_warhead, use_admet, self_excl=False,
             use_chemberta=False, use_esm=False, emb_dim=0):
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
    for kind, use in (('chemberta', use_chemberta), ('esm', use_esm)):
        if use:
            M = embedding(kind, c)
            A, B = M[tr], M[te]
            if emb_dim:
                # basis fitted on training rows only; held-out rows contribute
                # nothing to it, and the fit does not depend on the seed
                from sklearn.decomposition import PCA
                q = PCA(n_components=emb_dim, svd_solver='randomized',
                        random_state=0).fit(A)
                A, B = q.transform(A), q.transform(B)
            ftr.append(A); fte.append(B)
    return np.column_stack(ftr), np.column_stack(fte)

def fold(c, X, Xa, item, seed, spec):
    """One fold. Few-shot arms follow signals/full_stack.py: stage-1 model for
    shot selection by predicted-probability band, then one AUROC per repetition
    averaged over repetitions. The selection RNG is keyed on seed and target so
    it does not depend on iteration order."""
    t, te = item
    tr = train_mask(te, c, **spec.get('exclusions', {}))
    Ftr, Fte = features(c, X, Xa, tr, te, spec.get('morgan', True),
                        spec.get('warhead', False), spec.get('admet', False),
                        spec.get('self_excl', False),
                        spec.get('chemberta', False), spec.get('esm', False),
                        spec.get('emb_dim', 0))
    ytr, yte = c['y'][tr], c['y'][te]
    rf = dict(RF); rf.update(spec.get('rf', {}))
    k = spec.get('k', 0)
    flip = spec.get('flip', 0.0)
    if flip:
        # training labels only; the held-out labels stay clean so the metric
        # measures degradation of the model rather than of the target
        if k: raise ValueError('flip is defined for k=0 arms only')
        fr = np.random.RandomState(
            (seed * 1000003 + zlib.crc32(('flip' + t).encode())) % (2 ** 32))
        ytr = np.where(fr.random_sample(len(ytr)) < flip, 1 - ytr, ytr)
    if not k:
        clf = model_for(spec, seed, rf).fit(Ftr, ytr)
        p = clf.predict_proba(Fte)[:, 1]
        return (float(roc_auc_score(yte, p)), float(average_precision_score(yte, p)),
                int(len(yte)), p)
    shots = spec.get('shots', 'band')
    if shots == 'band':
        s1 = model_for(spec, seed, rf).fit(Ftr, ytr)
        probs = s1.predict_proba(Fte)[:, 1]
    elif shots == 'random':
        probs = None
    else:
        raise ValueError(f'unknown shot selection {shots!r}')
    rng = np.random.RandomState((seed * 1000003 + zlib.crc32(t.encode())) % (2 ** 32))
    loc = np.arange(len(yte))
    au, ap = [], []
    for rep in range(spec.get('reps', 5)):
        shot = (select_stratified(loc, yte, k, rng, probs) if probs is not None
                else rng.choice(loc, min(k, len(loc)), replace=False))
        rest = np.setdiff1d(loc, shot)
        if len(rest) < 3 or len(np.unique(yte[rest])) < 2: continue
        clf = model_for(spec, seed + rep, rf).fit(
            np.vstack([Ftr, Fte[shot]]), np.concatenate([ytr, yte[shot]]))
        q = clf.predict_proba(Fte[rest])[:, 1]
        au.append(float(roc_auc_score(yte[rest], q)))
        ap.append(float(average_precision_score(yte[rest], q)))
    if not au: return None
    return float(np.mean(au)), float(np.mean(ap)), int(len(yte)), None

def run(name, spec, c, X, Xa, seeds, cohort=None, out=None, store=True,
        folds=None, split=None):
    items = folds if folds is not None else [(t, c['tg'] == t)
                                             for t in (cohort or c['cohort'])]
    ex = spec.get('exclusions', {})
    items = [(nm, te) for nm, te in items if usable_mask(te, c, **ex)]
    t0, per_seed, preds = time.time(), {}, {}
    for s in seeds:
        rs = Parallel(n_jobs=NJOBS)(delayed(fold)(c, X, Xa, it, s, spec) for it in items)
        d = {}
        for (nm, te), r in zip(items, rs):
            if r is None: continue
            auroc, ap, n, p = r
            d[nm] = dict(auroc=auroc, ap=ap, n=n)
            if store and p is not None:
                preds.setdefault(str(s), {})[nm] = dict(
                    y=c['y'][te].tolist(), p=p.tolist())
        per_seed[str(s)] = d
        print(f'  {name} seed {s:2d} n={len(d)} macro '
              f'{np.mean([v["auroc"] for v in d.values()]):.4f}', flush=True)
    mac = [float(np.mean([v['auroc'] for v in per_seed[str(s)].values()])) for s in seeds]
    aps = [float(np.mean([v['ap'] for v in per_seed[str(s)].values()])) for s in seeds]
    tg = sorted(set.intersection(*[set(per_seed[str(s)]) for s in seeds]))
    pt = np.array([np.mean([per_seed[str(s)][t]['auroc'] for s in seeds]) for t in tg])
    res = dict(arm=name, spec=spec, macro_auroc=round(float(np.mean(mac)), 4),
               seed_sd=round(float(np.std(mac, ddof=1)), 4) if len(mac) > 1 else None,
               mean_ap=round(float(np.mean(aps)), 4),
               ap_seed_sd=round(float(np.std(aps, ddof=1)), 4) if len(aps) > 1 else None,
               per_target_sd=round(float(pt.std(ddof=1)), 4),
               target_se=round(float(pt.std(ddof=1) / len(pt) ** 0.5), 4),
               n_targets_paired=len(tg),
               per_seed_macro={str(s): round(m, 4) for s, m in zip(seeds, mac)},
               per_target={t: round(float(v), 4) for t, v in zip(tg, pt)},
               elapsed_min=round((time.time() - t0) / 60, 1), _meta={**meta(c, seeds, split=split, n_folds=len(items), exclusions=ex), **provenance(spec.get('nbits', 2048), spec.get('radius', 2), spec)})
    if out:
        json.dump(res, open(out, 'w'), indent=1)
        if store and preds:
            json.dump(preds, open(out.replace('.json', '_predictions.json'), 'w'))
    return res
