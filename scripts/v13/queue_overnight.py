#!/usr/bin/env python
"""Overnight queue of arms that need no change to the fold loop.

Waits for the factorial to finish, then runs each arm in turn. Every artefact is
written on completion and skipped if already present, so an interruption costs
at most the arm in flight. Feature matrices are cached per width and radius.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.seeds import CANONICAL
from src.cohort import load
from src.data_utils import compute_morgan, load_admet
from scripts.v13.engine import run

OUT, EXPECTED, WAIT_H = 'results/v13', 14, 8

ARMS = [
    # few-shot curve: the C3 feature set held fixed while k varies, so the two
    # existing points on this curve are cell_1110 at k=0 and cell_1111 at k=5
    ('fewshot_k1', dict(morgan=True, warhead=True, admet=True, k=1, reps=5)),
    ('fewshot_k3', dict(morgan=True, warhead=True, admet=True, k=3, reps=5)),
    ('fewshot_k10', dict(morgan=True, warhead=True, admet=True, k=10, reps=5)),
    # the published neighbour configuration at ten seeds is cell_1100, so only
    # the self-excluded variant is missing
    ('warhead_self_excluded', dict(morgan=True, warhead=True, self_excl=True)),
    # one factor at a time around the reference
    ('oat_bits512', dict(morgan=True, nbits=512)),
    ('oat_bits1024', dict(morgan=True, nbits=1024)),
    ('oat_radius3', dict(morgan=True, radius=3)),
    ('oat_msl1', dict(morgan=True, rf=dict(min_samples_leaf=1))),
    ('oat_cw_none', dict(morgan=True, rf=dict(class_weight=None))),
    ('oat_trees100', dict(morgan=True, rf=dict(n_estimators=100))),
    ('oat_all_defaults', dict(morgan=True, rf=dict(n_estimators=100, min_samples_leaf=1,
                                                   class_weight=None))),
]

def n_cells():
    return len([f for f in os.listdir(OUT)
                if f.startswith('cell_') and f.endswith('.json')])

t0 = time.time()
while n_cells() < EXPECTED and time.time() - t0 < WAIT_H * 3600:
    time.sleep(60)
print(f'starting queue with {n_cells()} cells present after '
      f'{(time.time() - t0) / 60:.0f} min of waiting', flush=True)

c = load()
smi = c['df'].smiles.tolist()
Xa = np.nan_to_num(load_admet(), nan=0.0)
cache = {}
for name, spec in ARMS:
    out = f'{OUT}/{name}.json'
    if os.path.exists(out):
        print('skip', name, flush=True); continue
    key = (spec.get('radius', 2), spec.get('nbits', 2048))
    if key not in cache:
        cache[key] = compute_morgan(smi, radius=key[0], nbits=key[1])
    print(f'--- {name} {spec} ---', flush=True)
    r = run(name, spec, c, cache[key], Xa, CANONICAL, out=out, store=False)
    print(name, {k: r[k] for k in ('macro_auroc', 'seed_sd', 'mean_ap',
                                   'target_se', 'elapsed_min')}, flush=True)
print('queue done', flush=True)
