#!/usr/bin/env python
"""Target-clustered intervals for the cross-publication cascade.

Resamples the target clusters of the stored per-fold cascade values; no model
fitting. Reports the bound (random minus cross_publication), the
cross_publication minus loto contrast, and a leave-one-target-out range.
"""
import json, glob, sys, os
import numpy as np

ARMS = ('random', 'cross_publication', 'loto')
B = 5000
BOOT = 42

def load(pat):
    per = {}
    for q in sorted(glob.glob(pat)):
        d = json.load(open(q)); per[d['seed']] = d['folds']
    if not per: return None, None
    common = None
    for f in per.values():
        ok = {k for k, v in f.items() if all(isinstance(v.get(a), float) for a in ARMS)}
        common = ok if common is None else (common & ok)
    tg = {}
    for k in common: tg.setdefault(k.split('|')[0], []).append(k)
    cube = {}
    for s in per:
        cube[s] = {t: {a: float(np.mean([per[s][k][a] for k in ks])) for a in ARMS}
                   for t, ks in tg.items()}
    return cube, sorted(tg)

def macro(cube, targets):
    out = {}
    for a in ARMS:
        out[a] = float(np.mean([np.mean([cube[s][t][a] for t in targets]) for s in cube]))
    return out

def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else 'results/pending/cascade_seed_*.json'
    out = sys.argv[2] if len(sys.argv) > 2 else 'results/cascade_intervals.json'
    cube, targets = load(pat)
    if cube is None: print('no cascade seed files at', pat); return
    pt = macro(cube, targets)
    rng = np.random.RandomState(BOOT)
    bnd, con = [], []
    for _ in range(B):
        samp = list(rng.choice(targets, len(targets), replace=True))
        m = macro(cube, samp)
        bnd.append(m['random'] - m['cross_publication'])
        con.append(m['cross_publication'] - m['loto'])
    loo = []
    for t in targets:
        m = macro(cube, [u for u in targets if u != t])
        loo.append(m['random'] - m['cross_publication'])
    res = {'seeds': sorted(int(x) for x in cube),
           'n_targets': len(targets), 'n_seeds': len(cube), 'n_bootstrap': B,
           'bootstrap_seed': BOOT,
           'arms': {a: round(pt[a], 4) for a in ARMS},
           'bound': {'point': pt['random'] - pt['cross_publication'],
                     'ci95': [float(np.percentile(bnd, 2.5)), float(np.percentile(bnd, 97.5))],
                     'loo_range': [float(min(loo)), float(max(loo))]},
           'crosspub_minus_loto': {'point': pt['cross_publication'] - pt['loto'],
                                   'ci95': [float(np.percentile(con, 2.5)), float(np.percentile(con, 97.5))]},
           'clustering': 'targets resampled with replacement; seeds averaged within replicate',
           'source': pat}
    json.dump(res, open(out, 'w'), indent=1)
    b, c = res['bound'], res['crosspub_minus_loto']
    print('targets', res['n_targets'], 'seeds', res['n_seeds'], res['arms'])
    print('bound', round(b['point'], 4), 'CI', [round(x, 4) for x in b['ci95']],
          'LOO range', [round(x, 4) for x in b['loo_range']])
    print('xpub-loto', round(c['point'], 4), 'CI', [round(x, 4) for x in c['ci95']])
    print('wrote', out)

if __name__ == '__main__':
    main()
