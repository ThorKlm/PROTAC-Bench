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

def masks(t, c):
    """Training and test row masks for fold t under the definition above."""
    held = c['cons'][t]
    te = c['tg'] == t
    shared = [u for u, a in c['cons'].items() if a & held]
    tr = ~np.isin(c['tg'], shared)
    tr &= ~np.isin(c['smi'], c['smi'][te])
    return tr, te

def usable(t, c):
    tr, te = masks(t, c)
    return (len(np.unique(c['y'][te])) > 1) and (len(np.unique(c['y'][tr])) > 1)

def meta(c, seeds):
    return dict(seeds=list(seeds), n_targets=len(c['cohort']), **DEFINITION)
