"""RF random-CV and scaffold-CV on the canonical seeds.
scaffold_cv takes no seeds argument, so both protocols are looped externally
with a seed-matched estimator, which makes the two symmetric and reproducible."""
import json, sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
CANONICAL = [42, 0]
from src.data_utils import compute_morgan
import src.evaluation as ev
t0 = time.time()
df = pd.read_csv('data/protac_bench.csv')
smiles = df.smiles.tolist(); y = df.label.values
X = compute_morgan(smiles)
def mk(seed):
    def model_fn(Xtr, ytr, Xte):
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                     class_weight='balanced', random_state=seed, n_jobs=-1)
        return clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return model_fn
raw = {'random_cv': {}, 'scaffold_cv': {}}
for s in CANONICAL:
    np.random.seed(s)
    raw['random_cv'][str(s)] = ev.random_cv(X, y, mk(s), n_folds=5, seeds=[s])
    np.random.seed(s)
    raw['scaffold_cv'][str(s)] = ev.scaffold_cv(X, y, smiles, mk(s), n_folds=5)
    print(f'seed {s} done [{(time.time()-t0)/60:.1f} min]', flush=True)
print('random_cv return shape:', type(raw['random_cv']['42']),
      list(raw['random_cv']['42']) if isinstance(raw['random_cv']['42'], dict) else '')
print('scaffold_cv return shape:', type(raw['scaffold_cv']['42']),
      list(raw['scaffold_cv']['42']) if isinstance(raw['scaffold_cv']['42'], dict) else '')
def pick(d, *keys):
    if not isinstance(d, dict): return float(d)
    for k in keys:
        if k in d: return float(d[k])
    return float('nan')
for proto in ('random_cv', 'scaffold_cv'):
    pool = [pick(v, 'pooled_auroc', 'pooled', 'mean_auroc') for v in raw[proto].values()]
    mac  = [pick(v, 'macro_auroc', 'macro', 'mean_macro') for v in raw[proto].values()]
    raw[proto + '_summary'] = dict(
        pooled_mean=round(float(np.nanmean(pool)), 4), pooled_sd=round(float(np.nanstd(pool, ddof=1)), 4),
        macro_mean=round(float(np.nanmean(mac)), 4), macro_sd=round(float(np.nanstd(mac, ddof=1)), 4))
    print(proto, raw[proto + '_summary'], flush=True)
raw['_meta'] = dict(seeds=CANONICAL, morgan_bits=int(X.shape[1]), rf_n_estimators=200,
                    min_samples_leaf=3, class_weight='balanced',
                    note='scaffold_cv has no seeds parameter; seeded externally per iteration',
                    elapsed_min=round((time.time() - t0) / 60, 1))
json.dump(raw, open('results/v12/smoke_rcv.json', 'w'), indent=2)
print('written', raw['_meta']['elapsed_min'], 'min')
