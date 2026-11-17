"""Cohort cells for the primary protocol, ten canonical seeds, reference
configuration. Deduplication removes training rows whose canonical SMILES
appears in the held-out target; constituent exclusion removes training rows
whose UniProt label shares an accession with the held-out label. Ordered so a
partial run yields the two cells the abstract depends on first."""
import json, time, numpy as np, pandas as pd
from collections import defaultdict
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
NJOBS = 24
OUT = '/root/pb/out/cohort_cells.json'
df = pd.read_csv('data/protac_bench.csv')
folds = json.load(open('data/loto_folds.json'))
elig = sorted(folds if isinstance(folds, list) else folds.keys())
tg = df.target_uniprot.astype(str).values
smi = df.smiles.astype(str).values
y = df.label.values.astype(int)
cons = {t: frozenset(t.split('/')) for t in set(tg)}
single = [t for t in elig if len(cons[t]) == 1]
print(f'cohorts: {len(elig)} folds, {len(single)} single-target; '
      f'composites {[t for t in elig if len(cons[t])>1]}', flush=True)
mols = [Chem.MolFromSmiles(s) for s in df.smiles]
X = np.zeros((len(mols), 2048), dtype=np.float32)
for i, m in enumerate(mols):
    if m is None: continue
    for b in AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048).GetOnBits():
        X[i, b] = 1.0
X = np.ascontiguousarray(X)
def one(t, s, dedup, excl):
    te = tg == t
    if len(np.unique(y[te])) < 2: return None
    if excl:
        held = cons[t]
        tr = np.array([not (cons[u] & held) for u in tg])
    else:
        tr = ~te
    if dedup:
        tr = tr & ~np.isin(smi, smi[te])
    if len(np.unique(y[tr])) < 2: return None
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                 class_weight='balanced', random_state=s, n_jobs=1)
    clf.fit(X[tr], y[tr])
    return (s, t, float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])),
            int(tr.sum()))
CELLS = [('dedup_65', dict(cohort=elig, dedup=True, excl=False)),
         ('dedup_excl_61', dict(cohort=single, dedup=True, excl=True)),
         ('retained_excl_61', dict(cohort=single, dedup=False, excl=True))]
out, t0 = {}, time.time()
for name, spec in CELLS:
    res = [r for r in Parallel(n_jobs=NJOBS)(
        delayed(one)(t, s, spec['dedup'], spec['excl'])
        for s in SEEDS for t in spec['cohort']) if r is not None]
    per_seed, ntr = defaultdict(dict), []
    for s, t, a, n in res:
        per_seed[str(s)][t] = a; ntr.append(n)
    tgts = sorted(set.intersection(*[set(v) for v in per_seed.values()]))
    macros = [float(np.mean([per_seed[s][t] for t in tgts])) for s in per_seed]
    pt = np.array([np.mean([per_seed[s][t] for s in per_seed]) for t in tgts])
    out[name] = dict(macro=round(float(np.mean(macros)), 4),
                     seed_sd=round(float(np.std(macros, ddof=1)), 4),
                     per_target_sd=round(float(pt.std(ddof=1)), 4),
                     target_se=round(float(pt.std(ddof=1) / len(pt) ** 0.5), 4),
                     n_targets=len(tgts), n_seeds=len(per_seed),
                     median_train_rows=int(np.median(ntr)),
                     dedup=spec['dedup'], constituent_excluded=spec['excl'],
                     per_target={t: round(float(np.mean([per_seed[s][t] for s in per_seed])), 4)
                                 for t in tgts},
                     per_seed=dict(per_seed))
    o = out[name]
    print(f'{name:18s} macro={o["macro"]:.4f} seed_sd={o["seed_sd"]:.4f} '
          f'pt_sd={o["per_target_sd"]:.4f} tgt_se={o["target_se"]:.4f} '
          f'n={o["n_targets"]} [{(time.time()-t0)/60:.1f} min]', flush=True)
    out['_meta'] = dict(seeds=SEEDS, morgan_bits=2048, radius=2, trees=200,
                        min_samples_leaf=3, class_weight='balanced',
                        elapsed_min=round((time.time() - t0) / 60, 1))
    json.dump(out, open(OUT, 'w'), indent=2)
print('\nabstract values: dedup_65 ->', out.get('dedup_65', {}).get('macro'),
      ' dedup_excl_61 ->', out.get('dedup_excl_61', {}).get('macro'))
