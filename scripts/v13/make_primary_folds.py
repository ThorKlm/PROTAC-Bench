#!/usr/bin/env python
"""Emit data/loto_primary_folds.json from src.cohort.

The released file and every runner come from one definition. Each fold records
the held-out rows and the rows excluded from training, since deduplication and
constituent exclusion act on the training side and a file carrying only test
indices would let a reader reproduce the splits but not the training sets.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.cohort import DEFINITION, load, masks, usable

OUT = 'data/loto_primary_folds.json'
c = load()
n = len(c['df'])
out = {'_meta': dict(split_type='leave_one_target_out_primary', **DEFINITION,
                     n_targets=len(c['cohort']),
                     eligibility_source='data/loto_folds.json restricted to single-accession labels',
                     exclusion_rule='training rows sharing a UniProt accession with the held-out '
                                    'label, or whose canonical SMILES occurs among the held-out rows',
                     fold_index_basis='row index into data/protac_bench.csv (0-based)')}
for t in c['cohort']:
    tr, te = masks(t, c)
    ex = (~tr) & (~te)
    out[t] = dict(target=t, n_entries=int(te.sum()),
                  activity_rate=round(float(c['y'][te].mean()), 4),
                  n_train=int(tr.sum()), usable=bool(usable(t, c)),
                  test_indices=np.flatnonzero(te).tolist(),
                  train_excluded_indices=np.flatnonzero(ex).tolist())
json.dump(out, open(OUT, 'w'), indent=1)
# the file must be sufficient on its own: rebuild the masks from indices alone
for t in c['cohort']:
    te = np.zeros(n, bool); te[out[t]['test_indices']] = True
    ex = np.zeros(n, bool); ex[out[t]['train_excluded_indices']] = True
    tr0, te0 = masks(t, c)
    assert (te == te0).all(), t
    assert (((~tr0) & (~te0)) == ex).all(), t
ec = [len(out[t]['train_excluded_indices']) for t in c['cohort']]
print(f'{OUT}: {len(c["cohort"])} folds, masks verified')
print(f'excluded training rows per fold: median {int(np.median(ec))} min {min(ec)} max {max(ec)}')
print(f'size {os.path.getsize(OUT) / 1024:.0f} KB')
