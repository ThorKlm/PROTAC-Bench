#!/usr/bin/env python
"""Factorial cells at the canonical seeds, one cell per invocation, or merge.

Cells are indexed (M, W, A, K). 0000 and 0001 are degenerate: with no base
features there is nothing to stack, which is why the engine cannot run them.
The warhead rate is derived from Morgan space, so X_morgan is always passed
and use_morgan controls only whether the fingerprints enter the matrix.
"""
import sys, os, json, glob, argparse, importlib.util
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.seeds import CANONICAL
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets, load_admet

DIR, K, N_REPS = 'results/pending', 5, 5
CELLS = ('1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111',
         '0010', '0011', '0100', '0101', '0110', '0111')

def engine():
    sp = importlib.util.spec_from_file_location(
        'fs', os.path.join(os.path.dirname(__file__), 'full_stack.py'))
    mo = importlib.util.module_from_spec(sp); sp.loader.exec_module(mo); return mo

def meta(n):
    return {'seeds': CANONICAL, 'k_fewshot': K, 'n_reps_fewshot': N_REPS, 'n_targets': n,
            'fold_construction': 'data/loto_folds.json (LOTO; same folds reused across cells)',
            'factors': {'F1': 'morgan_2048', 'F2': 'warhead_transfer_rate (k_neighbors=10)',
                        'F3': 'admet7', 'F4': 'few_shot_k5_stratified (n_reps=5)'},
            'generator': 'signals/factorial_8cell.py'}

def run(cid):
    m, w, a, kf = (int(x) for x in cid)
    if not (m or w or a):
        print(cid, 'degenerate_no_features'); return
    df = load_dataset('data/protac_bench.csv')
    y, targets = df['label'].values, df['target_uniprot'].values
    X_morgan = compute_morgan(df['smiles'].values)
    X_admet = np.nan_to_num(load_admet(), nan=0.0) if a else None
    eligible = get_eligible_targets(df)
    print(f'[{cid}] morgan={m} transfer={w} admet={a} fewshot={kf} targets={len(eligible)}', flush=True)
    r = engine().evaluate_condition(
        X_morgan, y, targets, eligible, CANONICAL,
        use_transfer=bool(w), use_admet=bool(a), use_fewshot=bool(kf),
        X_admet=X_admet, k=K, n_reps=N_REPS, label=cid, use_morgan=bool(m))
    os.makedirs(DIR, exist_ok=True)
    q = os.path.join(DIR, 'cell_%s.json' % cid)
    json.dump({'id': cid, 'F1': m, 'F2': w, 'F3': a, 'F4': kf,
               'mean_auroc': r.get('mean_auroc'), 'std_auroc': r.get('std_auroc'),
               'seed_means': r.get('seed_means'), 'meta': meta(len(eligible))},
              open(q, 'w'), indent=1)
    print(f'  {cid} -> {r.get("mean_auroc")}  wrote {q}', flush=True)

def merge():
    import statistics as st
    cells, n = [], None
    for cid in CELLS:
        q = os.path.join(DIR, 'cell_%s.json' % cid)
        if not os.path.exists(q):
            print('missing', cid); continue
        e = json.load(open(q)); n = e['meta']['n_targets']
        v = e['seed_means']; v = list(v.values()) if isinstance(v, dict) else v
        e = {k: val for k, val in e.items() if k != 'meta'}
        if v: e['sample_sd'] = st.stdev(v)
        cells.append(e)
    q = os.path.join(DIR, 'factorial_canonical.json')
    json.dump({'experiment': 'factorial_canonical', 'meta': meta(n), 'cells': cells,
               'status': 'completed' if len(cells) == len(CELLS) else 'partial'},
              open(q, 'w'), indent=1)
    print('wrote', q, 'with', len(cells), 'of', len(CELLS), 'cells')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', choices=CELLS)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    merge() if args.merge else run(args.cell)
