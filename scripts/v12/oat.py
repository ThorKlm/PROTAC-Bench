"""One-at-a-time configuration sensitivity for the RF+Morgan baseline under LOTO.
Parallelised over (seed, fold) with single-threaded forests, which is the correct
axis: 195 independent fits per configuration rather than 200 trees over 48 threads.
Emits per-config, per-seed, per-target AUROC, which also fills the per-target
dispersion the release never recorded at the reference configuration."""
import json, time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if __name__ != '__main__':
    raise ImportError('scripts/v12 modules execute on import; run as a script')
from src.seeds import CANONICAL
RDLogger.DisableLog('rdApp.*')
SEEDS = CANONICAL
NJOBS = 12
CENTRE = dict(bits=2048, radius=2, trees=200, msl=3, cw='balanced')
VARIANTS = [('centre', {}),
            ('bits512', dict(bits=512)), ('bits1024', dict(bits=1024)),
            ('radius3', dict(radius=3)),
            ('msl1_sklearn_default', dict(msl=1)),
            ('cw_none_sklearn_default', dict(cw=None)),
            ('trees100_sklearn_default', dict(trees=100)),
            ('all_sklearn_defaults', dict(trees=100, msl=1, cw=None))]
df = pd.read_csv('data/protac_bench.csv')
folds = json.load(open('data/loto_folds.json'))
elig = sorted(folds if isinstance(folds, list) else folds.keys())
tg = df.target_uniprot.astype(str).values
y = df.label.values.astype(int)
mols = [Chem.MolFromSmiles(s) for s in df.smiles]
fpcache = {}
def fp(bits, radius):
    if (bits, radius) not in fpcache:
        X = np.zeros((len(mols), bits), dtype=np.float32)
        for i, m in enumerate(mols):
            if m is None: continue
            for b in AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=bits).GetOnBits():
                X[i, b] = 1.0
        fpcache[(bits, radius)] = np.ascontiguousarray(X)
    return fpcache[(bits, radius)]
def one(X, t, s, cfg):
    """One LOTO fold at one seed, single-threaded forest."""
    te = tg == t
    if len(np.unique(y[te])) < 2: return (s, t, None)
    clf = RandomForestClassifier(n_estimators=cfg['trees'], min_samples_leaf=cfg['msl'],
                                 class_weight=cfg['cw'], random_state=s, n_jobs=1)
    clf.fit(X[~te], y[~te])
    return (s, t, float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
out, t0 = {}, time.time()
for name, delta in VARIANTS:
    cfg = {**CENTRE, **delta}
    X = fp(cfg['bits'], cfg['radius'])
    res = Parallel(n_jobs=NJOBS, verbose=0)(
        delayed(one)(X, t, s, cfg) for s in SEEDS for t in elig)
    per_seed = {str(s): {} for s in SEEDS}
    for s, t, a in res:
        if a is not None: per_seed[str(s)][t] = a
    macros = [float(np.mean(list(v.values()))) for v in per_seed.values()]
    tgts = sorted(set.intersection(*[set(v) for v in per_seed.values()]))
    pt = np.array([np.mean([per_seed[str(s)][t] for s in SEEDS]) for t in tgts])
    out[name] = dict(config=cfg, macro=round(float(np.mean(macros)), 4),
                     seed_sd=round(float(np.std(macros, ddof=1)), 4),
                     per_target_sd=round(float(pt.std(ddof=1)), 4),
                     target_se=round(float(pt.std(ddof=1) / len(pt) ** 0.5), 4),
                     n_targets=len(tgts), per_seed=per_seed)
    o = out[name]
    print(f'{name:26s} macro={o["macro"]:.4f} seed_sd={o["seed_sd"]:.4f} '
          f'per_target_sd={o["per_target_sd"]:.4f} target_se={o["target_se"]:.4f} '
          f'n={o["n_targets"]} [{(time.time()-t0)/60:.1f} min]', flush=True)
c = out['centre']['macro']
for k, v in out.items():
    if k != 'centre': v['delta_vs_centre'] = round(v['macro'] - c, 4)
out['_meta'] = dict(seeds=SEEDS, centre=CENTRE, n_jobs=NJOBS,
                    elapsed_min=round((time.time() - t0) / 60, 1))
json.dump(out, open('/root/pb/out/oat.json', 'w'), indent=2)
d = [abs(v['delta_vs_centre']) for k, v in out.items() if k not in ('centre', '_meta')]
print(f'\ncentre {c:.4f}  per_target_sd {out["centre"]["per_target_sd"]:.4f}  '
      f'target_se {out["centre"]["target_se"]:.4f}')
print(f'largest single-factor shift {max(d):.4f} over {len(d)} variants')
print(f'insensitive at the measurement resolution: {max(d) < out["centre"]["target_se"]}')
