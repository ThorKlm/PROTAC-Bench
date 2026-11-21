"""Configuration-insensitivity sweep for the RF+Morgan baseline under LOTO.
Emits per-config, per-seed, per-target AUROC so spans and target-clustered SEs
are computable from one artefact. Three seeds to map; confirm extremes at ten."""
import json, itertools, time, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog('rdApp.*')
SEEDS = [7, 13, 29]
GRID = dict(bits=[1024, 2048], radius=[2, 3], trees=[100, 200, 500],
            msl=[1, 3], cw=['balanced', None])
df = pd.read_csv('data/protac_bench.csv')
folds = json.load(open('data/loto_folds.json'))
elig = set(folds if isinstance(folds, list) else folds.keys())
tg = df.target_uniprot.astype(str).values
y = df.label.values.astype(int)
mols = [Chem.MolFromSmiles(s) for s in df.smiles]
def fp(bits, radius):
    X = np.zeros((len(mols), bits), dtype=np.uint8)
    for i, m in enumerate(mols):
        if m is None: continue
        for b in AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=bits).GetOnBits():
            X[i, b] = 1
    return X
out, t0 = {}, time.time()
cache = {}
for bits, radius, trees, msl, cw in itertools.product(*GRID.values()):
    key = f'b{bits}_r{radius}_t{trees}_msl{msl}_cw{cw}'
    if (bits, radius) not in cache: cache[(bits, radius)] = fp(bits, radius)
    X = cache[(bits, radius)]
    per_seed = {}
    for s in SEEDS:
        per_tgt = {}
        for t in sorted(elig):
            te = tg == t
            if len(np.unique(y[te])) < 2: continue
            clf = RandomForestClassifier(n_estimators=trees, min_samples_leaf=msl,
                                         class_weight=cw, random_state=s, n_jobs=-1)
            clf.fit(X[~te], y[~te])
            per_tgt[t] = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        per_seed[str(s)] = per_tgt
    macros = [float(np.mean(list(v.values()))) for v in per_seed.values()]
    tgts = sorted(set.intersection(*[set(v) for v in per_seed.values()]))
    pt = np.array([np.mean([per_seed[str(s)][t] for s in SEEDS]) for t in tgts])
    out[key] = dict(macro=round(float(np.mean(macros)), 4),
                    seed_sd=round(float(np.std(macros, ddof=1)), 4),
                    per_target_sd=round(float(pt.std(ddof=1)), 4),
                    target_se=round(float(pt.std(ddof=1) / len(pt) ** 0.5), 4),
                    n_targets=len(tgts), per_seed=per_seed)
    print(f'{key:42s} macro={out[key]["macro"]:.4f} seed_sd={out[key]["seed_sd"]:.4f} '
          f'target_se={out[key]["target_se"]:.4f}  [{(time.time()-t0)/60:.1f} min]', flush=True)
out['_meta'] = dict(seeds=SEEDS, grid={k: list(map(str, v)) for k, v in GRID.items()},
                    elapsed_min=round((time.time() - t0) / 60, 1))
json.dump(out, open('results/v12/grid.json', 'w'), indent=2)
ms = [v['macro'] for k, v in out.items() if k != '_meta']
print(f'\nregion span {min(ms):.4f} to {max(ms):.4f} = {max(ms)-min(ms):.4f} '
      f'over {len(ms)} configurations')
