"""Temporal-prospective evaluation against data/temporal_prospective_folds.json.
Train on pub_year < 2023, test on pub_year == 2024, per-target macro over the
test targets that carry both classes. The fold file is the sole definition."""
import json, os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if __name__ != '__main__':
    raise ImportError('scripts/v12 modules execute on import; run as a script')
from src.seeds import CANONICAL
from src.data_utils import compute_morgan
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
RF = dict(n_estimators=200, min_samples_leaf=3, class_weight='balanced', n_jobs=-1)
df = pd.read_csv('data/protac_bench.csv')
f = json.load(open('data/temporal_prospective_folds.json'))
tr, te = np.array(f['train_indices']), np.array(f['test_indices'])
X = compute_morgan(df.smiles.tolist())
y = df.label.values.astype(int)
tg = df.target_uniprot.astype(str).values
t0, per_seed = time.time(), {}
for s in CANONICAL:
    clf = RandomForestClassifier(random_state=s, **RF).fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    d = {}
    for t in sorted(set(tg[te])):
        m = tg[te] == t
        if len(np.unique(y[te][m])) == 2:
            d[t] = float(roc_auc_score(y[te][m], p[m]))
    per_seed[str(s)] = d
    print(f'  seed {s}: macro {np.mean(list(d.values())):.4f} over {len(d)} targets', flush=True)
mac = [float(np.mean(list(v.values()))) for v in per_seed.values()]
ev = sorted(set.intersection(*[set(v) for v in per_seed.values()]))
pt = np.array([np.mean([per_seed[str(s)][t] for s in CANONICAL]) for t in ev])
out = dict(macro=round(float(np.mean(mac)), 4), seed_sd=round(float(np.std(mac, ddof=1)), 4),
           per_target_sd=round(float(pt.std(ddof=1)), 4),
           target_se=round(float(pt.std(ddof=1) / len(pt) ** 0.5), 4),
           n_test_targets_total=len(set(tg[te])), n_test_targets_evaluable=len(ev),
           n_train_rows=int(len(tr)), n_test_rows=int(len(te)), per_seed=per_seed,
           _meta=dict(seeds=CANONICAL, rf=RF, morgan_bits=2048, radius=2,
                      fold_file='data/temporal_prospective_folds.json',
                      elapsed_min=round((time.time() - t0) / 60, 1)))
json.dump(out, open('/root/pb/out/temporal_prospective.json', 'w'), indent=2)
print(f"macro {out['macro']} sd {out['seed_sd']} evaluable {out['n_test_targets_evaluable']}"
      f" of {out['n_test_targets_total']} targets, {out['n_test_rows']} test rows")
