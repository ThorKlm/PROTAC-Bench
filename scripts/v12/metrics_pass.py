"""Reference-configuration LOTO pass storing raw predictions, so average
precision, ECE, Brier, temperature and Platt scaling all follow post hoc
without refitting. Ten canonical seeds, 2048-bit Morgan radius 2, 200 trees."""
import json, time, numpy as np, pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if __name__ != '__main__':
    raise ImportError('scripts/v12 modules execute on import; run as a script')
from src.seeds import CANONICAL
RDLogger.DisableLog('rdApp.*')
SEEDS = CANONICAL
NJOBS = 24
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
X = np.ascontiguousarray(X)
def one(t, s):
    te = tg == t
    if len(np.unique(y[te])) < 2: return None
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                 class_weight='balanced', random_state=s, n_jobs=1)
    clf.fit(X[~te], y[~te])
    p = clf.predict_proba(X[te])[:, 1]
    return (s, t, float(roc_auc_score(y[te], p)),
            float(average_precision_score(y[te], p)), float(y[te].mean()),
            [round(float(v), 5) for v in p], [int(v) for v in y[te]])
t0 = time.time()
res = [r for r in Parallel(n_jobs=NJOBS)(delayed(one)(t, s) for s in SEEDS for t in elig)
       if r is not None]
auroc, ap, prior, preds = defaultdict(dict), defaultdict(dict), {}, defaultdict(dict)
for s, t, a, p, pr, probs, labels in res:
    auroc[str(s)][t] = a; ap[str(s)][t] = p; prior[t] = pr
    preds[str(s)][t] = dict(p=probs, y=labels)
tg_set = sorted(set.intersection(*[set(v) for v in auroc.values()]))
mac_auroc = [float(np.mean([auroc[s][t] for t in tg_set])) for s in auroc]
mac_ap = [float(np.mean([ap[s][t] for t in tg_set])) for s in ap]
macro_prior = float(np.mean([prior[t] for t in tg_set]))
mean_ap = float(np.mean(mac_ap))
per_t_ap = {t: float(np.mean([ap[s][t] for s in ap])) for t in tg_set}
lift = {t: per_t_ap[t] - prior[t] for t in tg_set}
out = dict(macro_auroc=round(float(np.mean(mac_auroc)), 4),
           macro_auroc_sd=round(float(np.std(mac_auroc, ddof=1)), 4),
           mean_average_precision=round(mean_ap, 4),
           mean_ap_sd=round(float(np.std(mac_ap, ddof=1)), 4),
           macro_per_target_prior=round(macro_prior, 4),
           mean_lift=round(mean_ap - macro_prior, 4),
           median_lift=round(float(np.median(list(lift.values()))), 4),
           n_positive_lift=int(sum(v > 0 for v in lift.values())),
           n_targets=len(tg_set), seeds=SEEDS,
           per_target_auroc={t: round(float(np.mean([auroc[s][t] for s in auroc])), 4) for t in tg_set},
           per_target_ap={t: round(v, 4) for t, v in per_t_ap.items()},
           per_target_prior={t: round(prior[t], 4) for t in tg_set},
           elapsed_min=round((time.time() - t0) / 60, 1))
json.dump(out, open('results/v12/metrics_pass.json', 'w'), indent=2)
json.dump(preds, open('results/v12/loto_predictions.json', 'w'))
print(f'macro AUROC {out["macro_auroc"]:.4f} sd {out["macro_auroc_sd"]:.4f}')
print(f'mean AP {out["mean_average_precision"]:.4f} vs macro prior '
      f'{out["macro_per_target_prior"]:.4f}  lift {out["mean_lift"]:+.4f}  '
      f'median lift {out["median_lift"]:+.4f}  positive on {out["n_positive_lift"]}/{out["n_targets"]}')
print(f'predictions written, elapsed {out["elapsed_min"]} min')
