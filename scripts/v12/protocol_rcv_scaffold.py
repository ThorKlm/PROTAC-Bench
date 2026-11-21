"""Random-CV pooled and scaffold-CV for the reference, on the canonical seeds.
These two values anchor the collapse contrast and no queued job produces them."""
import json, sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.seeds import CANONICAL
from src.data_utils import compute_morgan
import src.evaluation as ev
t0 = time.time()
df = pd.read_csv('data/protac_bench.csv')
X = compute_morgan(df.smiles.tolist()); y = df.label.values
def model_fn(Xtr, ytr, Xte):
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                 class_weight='balanced', n_jobs=-1)
    return clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
out = {'_meta': {'seeds': CANONICAL, 'morgan_bits': int(X.shape[1]),
                 'rf_n_estimators': 200, 'class_weight': 'balanced'}}
out['random_cv'] = ev.random_cv(X, y, model_fn, n_folds=5, seeds=CANONICAL)
out['scaffold_cv'] = ev.scaffold_cv(X, y, df.smiles.tolist(), model_fn, seeds=CANONICAL)
out['_meta']['elapsed_min'] = round((time.time() - t0) / 60, 1)
json.dump(out, open('results/v12/protocol_rcv_scaffold.json', 'w'), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != '_meta'}, indent=2)[:600])
