"""Evaluation cohort for PROTAC-Bench: one definition, imported by every runner.

Folds are the LOTO-eligible targets carrying a single UniProt accession. For
each fold the training rows exclude any row whose target label shares an
accession with the held-out label, and any row whose SMILES occurs among the
held-out target's rows, since a compound assayed against several targets
otherwise appears in both splits with only its annotation differing.
"""
import json
import numpy as np
import pandas as pd

DEFINITION = dict(cohort='loto_single_accession', min_n=10, min_rate=0.1,
                  max_rate=0.9, constituent_excluded=True, smiles_dedup=True)

def load(data='data/protac_bench.csv', folds='data/loto_folds.json'):
    df = pd.read_csv(data)
    tg = df.target_uniprot.astype(str).values
    smi = df.smiles.astype(str).values
    y = df.label.values.astype(int)
    f = json.load(open(folds))
    eligible = sorted(f if isinstance(f, list) else f.keys())
    cons = {t: frozenset(t.split('/')) for t in set(tg)}
    cohort = [t for t in eligible if len(cons[t]) == 1]
    return dict(df=df, tg=tg, smi=smi, y=y, cons=cons, cohort=cohort,
                eligible=eligible)

def train_mask(te, c, constituent=True, dedup=True):
    """Training rows for an arbitrary held-out mask.

    Both exclusions are relative to the held-out set rather than to a single
    label, so any target-based fold family gets them. They are flags because
    the accession rule is meaningful only when the held-out set is defined by
    target: on an E3-defined fold it holds out rows from most targets and would
    remove nearly the whole training set, inverting the design of the analysis.
    """
    tr = ~te
    if constituent:
        held = frozenset().union(*(c['cons'][u] for u in set(c['tg'][te]))) \
            if te.any() else frozenset()
        tr = ~np.isin(c['tg'], [u for u, a in c['cons'].items() if a & held])
    if dedup:
        tr = tr & ~np.isin(c['smi'], c['smi'][te])
    return tr

def masks(t, c, **kw):
    """Training and test row masks for fold t: the single-target special case."""
    te = c['tg'] == t
    return train_mask(te, c, **kw), te

def usable_mask(te, c, **kw):
    tr = train_mask(te, c, **kw)
    return (len(np.unique(c['y'][te])) > 1) and (len(np.unique(c['y'][tr])) > 1)

def usable(t, c, **kw):
    return usable_mask(c['tg'] == t, c, **kw)

def folds_loto(c):
    return [(t, c['tg'] == t) for t in c['cohort']]

def folds_from_file(c, path, key='test_indices'):
    """Folds from a released fold file whose entries list row indices."""
    d = json.load(open(path))
    out = []
    for nm, v in d.items():
        if nm == '_meta': continue
        te = np.zeros(len(c['tg']), bool)
        te[np.asarray(v[key] if isinstance(v, dict) else v, int)] = True
        out.append((nm, te))
    return out

def folds_e3(c):
    e = c['df'].e3_type.astype(str).values
    return [(g, e == g) for g in sorted(set(e))]

def meta(c, seeds, split=None, n_folds=None, exclusions=None):
    m = dict(seeds=list(seeds), n_targets=len(c['cohort']), **DEFINITION)
    if split: m['cohort'] = split
    if n_folds is not None: m['n_folds'] = n_folds
    if exclusions is not None:
        m['constituent_excluded'] = bool(exclusions.get('constituent', True))
        m['smiles_dedup'] = bool(exclusions.get('dedup', True))
    return m
