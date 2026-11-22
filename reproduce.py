#!/usr/bin/env python
"""Regenerate PROTAC-Bench results and emit results/REPRODUCTION.txt.

Every document line is read or computed from an artefact; the only literal text
is the block header. A missing artefact is reported on stderr and contributes no
lines, so the document is never annotated with its own gaps. Bands: the default
runs the overnight set, --all adds the hyperparameter search and the
architecture panel, --print-only rebuilds the document from what exists.
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

R, DOC, L = 'results', 'results/REPRODUCTION.txt', []
def w(s=''): L.append(s)
def head(d): w(); w(f'--- @@: {d} ---')
def g(p):
    try:
        with open(p) as f: return json.load(f)
    except FileNotFoundError:
        print(f'gap: {p}', file=sys.stderr); return None
def v(x, n=4): return round(float(x), n) if isinstance(x, (int, float)) else x
def pv(x):
    # a p-value rounded until it reads as zero is worse than an exponent
    x = float(x)
    return f'{x:.2e}' if x < 1e-4 else round(x, 4)
BASE, PARAMS = {}, ('cohort', 'min_n', 'min_rate', 'max_rate',
                   'constituent_excluded', 'smiles_dedup', 'rf', 'morgan_bits',
                   'radius', 'git_commit', 'python', 'numpy', 'sklearn')
def meta(d, base=False):
    # the cohort block states the definition once; later blocks print only
    # what deviates from it, so a non-canonical seed set stands out
    m = d.get('_meta') or d.get('meta') or {}
    if m.get('seeds') and (base or list(m['seeds']) != list(BASE.get('seeds', ()))):
        w('seeds ' + ','.join(str(x) for x in m['seeds']))
    if 'n_targets' in m: w(f'n_targets {m["n_targets"]}')
    for k in PARAMS:
        if k in m and (base or m[k] != BASE.get(k)):
            x = m[k]
            w(f'{k} ' + (','.join(f'{a}={b}' for a, b in x.items() if a != 'n_jobs')
                         if isinstance(x, dict) else str(x)))
    if base: BASE.update(m)
def agg(d):
    if d.get('spec'): w('spec ' + ','.join(f'{k}={x}' for k, x in d['spec'].items()))
    for k in ('macro_auroc', 'seed_sd', 'mean_ap', 'ap_seed_sd',
              'per_target_sd', 'target_se', 'elapsed_min'):
        if k in d: w(f'{k} {v(d[k])}')
    if d.get('per_seed_macro'):
        w('per_seed_macro ' + ','.join(f'{k}={v(x)}' for k, x in d['per_seed_macro'].items()))
def paired(a, b):
    t = sorted(set(a) & set(b))
    dl = np.array([b[x] for x in t]) - np.array([a[x] for x in t])
    p = float(wilcoxon(dl)[1]) if np.any(dl != 0) else 1.0
    return t, dl, p
def contrast(ref, cur, label):
    t, dl, p = paired(ref, cur)
    w(f'n_paired {len(t)}')
    w(f'delta_vs_{label} {v(dl.mean())}')
    w(f'wilcoxon_p_vs_{label} {pv(p)}')
    w(f'improved {int((dl > 0).sum())}')
    w(f'moved_gt_0.05 {int((np.abs(dl) > 0.05).sum())}')
    w(f'max_gain {v(dl.max())}'); w(f'max_loss {v(dl.min())}')

# the ladder arms are factorial cells, so one run of one revision supplies both
LADDER = [('cell_1000', 'reference, Morgan 2048 radius 2, random forest'),
          ('cell_1100', 'plus warhead transfer rate'),
          ('cell_1110', 'plus seven ADMET scores'),
          ('cell_1111', 'plus five labelled shots')]

def block_cohort(n):
    d = g(f'{R}/v13/cell_1000.json')
    if not d: return n
    head('cohort definition'); meta(d, base=True)
    import pandas as pd
    df = pd.read_csv('data/protac_bench.csv')
    w(f'n_rows {len(df)}'); w(f'n_unique_targets {df.target_uniprot.nunique()}')
    w(f'n_unique_smiles {df.smiles.nunique()}')
    w(f'activity_rate {v(df.label.mean())}')
    w('e3_type ' + ','.join(f'{k}={int(x)}' for k, x in df.e3_type.value_counts().items()))
    return n + 1

def block_ladder(n):
    ref = prev = None
    for name, desc in LADDER:
        d = g(f'{R}/v13/{name}.json')
        if not d: n += 1; continue
        head(desc); meta(d); agg(d)
        pt = d['per_target']
        if ref is None: ref = pt
        else:
            contrast(ref, pt, 'C0')
            w(f'step_vs_previous {v(np.mean([pt[x] - prev[x] for x in sorted(set(pt) & set(prev))]))}')
        prev = pt; n += 1
    return n

def block_warhead_variants(n):
    for name, desc in (('C1_warhead_published_3seed', 'warhead, published kNN configuration'),
                       ('C1_warhead_self_excluded_3seed', 'warhead, query compound excluded from its own neighbours')):
        d = g(f'{R}/v13/{name}.json')
        if not d: n += 1; continue
        head(desc); meta(d); agg(d); n += 1
    return n

def block_factorial(n):
    import glob
    fs = sorted(glob.glob(f'{R}/v13/cell_*.json'))
    if not fs: print(f'gap: {R}/v13/cell_*.json', file=sys.stderr); return n + 1
    head('factorial over Morgan, warhead, ADMET, few-shot')
    cells = {}
    for p in fs:
        d = json.load(open(p)); cid = d['arm'].split('_')[-1]
        cells[cid] = d
        w(f'cell_{cid} macro {v(d["macro_auroc"])} sd {v(d["seed_sd"])} '
          f'ap {v(d["mean_ap"])} se {v(d["target_se"])}')
    for i, fac in enumerate('MWAK'):
        pairs = [(a, b) for a in cells for b in cells
                 if a[i] == '1' and b[i] == '0' and a[:i] + a[i+1:] == b[:i] + b[i+1:]]
        if pairs:
            w(f'marginal_{fac} {v(np.mean([cells[a]["macro_auroc"] - cells[b]["macro_auroc"] for a, b in pairs]))} '
              f'n_contrasts {len(pairs)}')
    return n + 1

def block_cohort_cells(n):
    d = g(f'{R}/v12/cohort_cells.json')
    if not d: return n + 1
    head('cohort variants, reference configuration')
    meta(d)
    for k, c in d.items():
        if k.startswith('_'): continue
        w(f'{k} macro {v(c["macro"])} seed_sd {v(c["seed_sd"])} '
          f'target_se {v(c["target_se"])} per_target_sd {v(c["per_target_sd"])} '
          f'n_targets {c["n_targets"]} median_train_rows {c["median_train_rows"]} '
          f'dedup {c["dedup"]} constituent_excluded {c["constituent_excluded"]}')
        if c.get('per_seed'):
            w(f'{k}_per_seed_macro ' + ','.join(
                f'{s}={v(np.mean(list(r.values())))}' for s, r in c['per_seed'].items()))
    return n + 1

def block_overlap(n):
    d = g(f'{R}/v12/overlap_control.json')
    if not d: return n + 1
    head('reference split by whether the test compound occurs verbatim in training')
    meta(d)
    for k in ('full', 'novel', 'seen'):
        if k in d: w(f'{k} macro {v(d[k]["macro"])} seed_sd {v(d[k]["seed_sd"])} n_folds {d[k]["n_folds"]}')
    return n + 1

def block_source(n):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    df = pd.read_csv('data/protac_bench.csv')
    pdb = np.arange(len(df)) >= 3181
    y = df.label.values
    head('source partition by row position')
    w(f'ribes_rows {int((~pdb).sum())}'); w(f'ribes_rate {v(y[~pdb].mean())}')
    w(f'protacdb_rows {int(pdb.sum())}'); w(f'protacdb_rate {v(y[pdb].mean())}')
    w(f'source_only_pooled_auroc {v(roc_auc_score(y, pdb.astype(int)))}')
    return n + 1

BLOCKS = [block_cohort, block_ladder, block_warhead_variants, block_factorial,
          block_cohort_cells, block_overlap, block_source]

def build():
    n = 1
    for b in BLOCKS: n = b(n)
    os.makedirs(R, exist_ok=True)
    out, k = [], 0
    for line in L:
        if line.startswith('--- @@: '):
            k += 1; line = line.replace('@@', str(k), 1)
        out.append(line)
    with open(DOC, 'w') as f: f.write('\n'.join(out).lstrip() + '\n')
    print(f'{DOC}: {len(L)} lines', file=sys.stderr)

def run_core(seeds):
    from src.cohort import load
    from src.data_utils import compute_morgan, load_admet
    from scripts.v13.engine import run
    c = load(); X = compute_morgan(c['df'].smiles.tolist())
    Xa = np.nan_to_num(load_admet(), nan=0.0)
    specs = [('C0_morgan', dict(morgan=True)),
             ('C1_morgan_warhead', dict(morgan=True, warhead=True)),
             ('C2_morgan_warhead_admet', dict(morgan=True, warhead=True, admet=True)),
             ('C3_morgan_warhead_admet_k5', dict(morgan=True, warhead=True, admet=True, k=5, reps=5))]
    for name, spec in specs:
        out = f'{R}/v13/{name}.json'
        if os.path.exists(out): print(f'have {name}', file=sys.stderr); continue
        run(name, spec, c, X, Xa, seeds, out=out, store=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='include the hyperparameter search and the architecture panel')
    ap.add_argument('--print-only', action='store_true')
    ap.add_argument('--seeds', default='canonical')
    a = ap.parse_args()
    if not a.print_only:
        from src.seeds import parse
        run_core(parse(a.seeds))
        if a.all: print('--all arms are not yet ported to the cohort module', file=sys.stderr)
    build()
