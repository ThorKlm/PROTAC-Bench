"""XGBoost under three protocols on one corpus and feature set, to back the
second model family in Section 3.2. Random CV and scaffold CV report pooled
out-of-fold AUROC; LOTO reports the macro-mean over folds."""
import json, time, numpy as np, pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if __name__ != '__main__':
    raise ImportError('scripts/v12 modules execute on import; run as a script')
from src.seeds import CANONICAL
RDLogger.DisableLog('rdApp.*')
SEEDS = CANONICAL
NJOBS = 24
PARAMS = dict(n_estimators=300, max_depth=6, n_jobs=1, tree_method='hist',
              eval_metric='logloss')
df = pd.read_csv('data/protac_bench.csv')
folds = json.load(open('data/loto_folds.json'))
elig = sorted(folds if isinstance(folds, list) else folds.keys())
tg = df.target_uniprot.astype(str).values
y = df.label.values.astype(int)
mols = [Chem.MolFromSmiles(s) for s in df.smiles]
X = np.zeros((len(mols), 2048), dtype=np.float32)
for i, m in enumerate(mols):
    if m is None: continue
    for b in AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048).GetOnBits():
        X[i, b] = 1.0
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else '' for m in mols]
def fit_pred(tr, te, s):
    clf = XGBClassifier(random_state=s, **PARAMS).fit(X[tr], y[tr])
    return te, clf.predict_proba(X[te])[:, 1]
def pooled_and_macro(splitter_jobs, s):
    """Collect out-of-fold predictions, then score pooled and per target."""
    p = np.full(len(y), np.nan)
    for te, pr in Parallel(n_jobs=NJOBS)(delayed(fit_pred)(tr, te, s)
                                          for tr, te in splitter_jobs):
        p[te] = pr
    ok = ~np.isnan(p)
    pooled = roc_auc_score(y[ok], p[ok])
    per = {}
    for t in elig:
        m = (tg == t) & ok
        if m.sum() and len(np.unique(y[m])) > 1:
            per[t] = float(roc_auc_score(y[m], p[m]))
    return float(pooled), float(np.mean(list(per.values()))), per
out, t0 = {}, time.time()
for name in ('random_cv', 'scaffold_cv'):
    po, ma = [], []
    for s in SEEDS:
        if name == 'random_cv':
            jobs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(X, y))
        else:
            idx = np.arange(len(y))
            rng = np.random.default_rng(s)
            perm = rng.permutation(np.unique(scaf))
            rank = {sc: i for i, sc in enumerate(perm)}
            g = np.array([rank[sc] for sc in scaf])
            jobs = list(GroupKFold(5).split(X, y, groups=g))
        a, b, _ = pooled_and_macro(jobs, s)
        po.append(a); ma.append(b)
    out[name] = dict(pooled_mean=round(float(np.mean(po)), 4),
                     pooled_sd=round(float(np.std(po, ddof=1)), 4),
                     macro_mean=round(float(np.mean(ma)), 4),
                     macro_sd=round(float(np.std(ma, ddof=1)), 4), n_seeds=len(SEEDS))
    print(f'{name:12s} pooled={out[name]["pooled_mean"]:.4f} '
          f'macro={out[name]["macro_mean"]:.4f} [{(time.time()-t0)/60:.1f} min]', flush=True)
def loto_one(t, s):
    te = tg == t
    if len(np.unique(y[te])) < 2: return (s, t, None)
    clf = XGBClassifier(random_state=s, **PARAMS).fit(X[~te], y[~te])
    return (s, t, float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
res = Parallel(n_jobs=NJOBS)(delayed(loto_one)(t, s) for s in SEEDS for t in elig)
ps = defaultdict(dict)
for s, t, a in res:
    if a is not None: ps[str(s)][t] = a
macros = [float(np.mean(list(v.values()))) for v in ps.values()]
out['loto'] = dict(macro_mean=round(float(np.mean(macros)), 4),
                   macro_sd=round(float(np.std(macros, ddof=1)), 4),
                   n_seeds=len(ps), n_targets=len(set.intersection(*[set(v) for v in ps.values()])),
                   per_seed=dict(ps))
print(f'loto         macro={out["loto"]["macro_mean"]:.4f} '
      f'sd={out["loto"]["macro_sd"]:.4f} [{(time.time()-t0)/60:.1f} min]')
out['_meta'] = dict(seeds=SEEDS, params=PARAMS, morgan_bits=2048, radius=2,
                    elapsed_min=round((time.time() - t0) / 60, 1))
json.dump(out, open('results/v12/xgb_protocols.json', 'w'), indent=2)
print(f'\ntriple: random {out["random_cv"]["pooled_mean"]:.4f}  '
      f'scaffold {out["scaffold_cv"]["pooled_mean"]:.4f}  loto {out["loto"]["macro_mean"]:.4f}')
