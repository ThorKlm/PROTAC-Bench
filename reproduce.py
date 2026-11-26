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

R, L = 'results', []
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
                   'constituent_excluded', 'smiles_dedup', 'model', 'model_params', 'morgan_bits',
                   'radius', 'git_commit', 'python', 'numpy', 'sklearn')
def meta(d, base=False):
    # the cohort block states the definition once; later blocks print only
    # what deviates from it, so a non-canonical seed set stands out
    m = d.get('_meta') or d.get('meta') or {}
    if m.get('seeds') and (base or list(m['seeds']) != list(BASE.get('seeds', ()))):
        w('seeds ' + ','.join(str(x) for x in m['seeds']))
    if 'n_targets' in m and (base or m['n_targets'] != BASE.get('n_targets')):
        w(f'n_targets {m["n_targets"]}')
    if 'n_folds' in m: w(f'n_folds {m["n_folds"]}')
    for k in PARAMS:
        if k in m and (base or m[k] != BASE.get(k)):
            x = m[k]
            w(f'{k} ' + (','.join(f'{a}={b}' for a, b in x.items() if a != 'n_jobs')
                         if isinstance(x, dict) else str(x)))
    if base: BASE.update(m)
def agg(d):
    if d.get('spec'): w('spec ' + ','.join(f'{k}={x}' for k, x in d['spec'].items()))
    for k in ('macro_auroc', 'seed_sd', 'mean_ap', 'ap_seed_sd',
              'per_target_sd', 'target_se'):
        if k in d and d[k] is not None: w(f'{k} {v(d[k])}')
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

NBINS = 100
def summarise(vals):
    """Distribution summary plus a histogram, so a figure can draw a density
    without the raw per-compound values bloating the artefact."""
    a = np.asarray(vals, float)
    counts, edges = np.histogram(a, bins=NBINS, range=(0.0, 1.0))
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return dict(n=int(len(a)), mean=round(float(a.mean()), 4),
                median=round(float(q[2]), 4), q05=round(float(q[0]), 4),
                q25=round(float(q[1]), 4), q75=round(float(q[3]), 4),
                q95=round(float(q[4]), 4),
                frac_ge_090=round(float((a >= 0.9).mean()), 4),
                frac_eq_1=round(float((a >= 0.999).mean()), 4),
                hist_counts=counts.tolist(),
                hist_edges=[round(float(e), 4) for e in edges])

# the ladder arms are factorial cells, so one run of one revision supplies both
LADDER = [('cell_1000', 'reference, Morgan 2048 radius 2, random forest'),
          ('cell_1100', 'plus warhead transfer rate'),
          ('cell_1110', 'plus seven ADMET scores'),
          ('cell_1111', 'plus five labelled shots')]

def block_cohort(n):
    d = g(f'{R}/cell_1000.json')
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
        d = g(f'{R}/{name}.json')
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
    # at ten seeds the published configuration is cell 1100 exactly, so only the
    # self-excluded variant is a distinct measurement
    for name, desc in (('warhead_self_excluded',
                        'warhead, query compound excluded from its own neighbours'),):
        d = g(f'{R}/{name}.json')
        if not d: n += 1; continue
        head(desc); meta(d); agg(d); n += 1
    return n

def block_factorial(n):
    import glob
    fs = sorted(glob.glob(f'{R}/cell_[01][01][01][01].json'))
    if not fs: print(f'gap: {R}/cell_*.json', file=sys.stderr); return n + 1
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
    d = g(f'{R}/cohort_cells.json')
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
    d, o = g(f'{R}/retained_65.json'), g(f'{R}/overlap.json')
    if not d or not o:
        print(f'gap: {R}/overlap.json', file=sys.stderr); return n + 1
    head('reference on the retained cohort, split by whether the test compound '
         'occurs verbatim in training')
    meta(d); agg(d)
    for k in ('novel', 'seen'):
        w(f'{k} macro {v(o[k]["macro"])} seed_sd {v(o[k]["seed_sd"])} '
          f'n_folds {o[k]["n_folds"]}')
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

def block_calibration(n):
    d = g(f'{R}/calibration.json')
    if not d: return n + 1
    head('calibration and coverage, reference arm, from stored predictions')
    w('source ' + d['source'])
    w('seeds ' + ','.join(str(x) for x in d['seeds']))
    ps = d['per_seed']; s0 = list(ps)[0]
    w(f'n_pooled {ps[s0]["n"]}'); w(f'n_targets {ps[s0]["n_targets"]}')
    for k, a in d['aggregate'].items():
        w(f'{k} {v(a["mean"])} sample_sd {v(a["sample_sd"])}')
    for k in ('brier_raw', 'ece10_raw', 'ece10_temperature', 'ece10_platt'):
        w(f'{k}_per_seed ' + ','.join(f'{q}={v(ps[q][k])}' for q in ps))
    def mu(rows, key):
        vals = [x[key] for x in rows if x.get(key) is not None]
        return v(np.mean(vals)) if vals else None
    for c in sorted(ps[s0]['coverage_raw'], key=float, reverse=True):
        rows = [ps[q]['coverage_raw'][c] for q in ps]
        w(f'coverage {c} n {int(np.mean([x["n"] for x in rows]))} '
          f'pos_rate {mu(rows, "pos_rate")} error_rate {mu(rows, "error_rate")} '
          f'majority_error {mu(rows, "majority_error")} '
          f'auroc {mu(rows, "auroc")} brier {mu(rows, "brier")}')
    return n + 1

def flat(d, pre=''):
    out = []
    for k, x in d.items():
        if isinstance(x, dict): out += flat(x, f'{pre}{k}.').split(',') if x else []
        else: out.append(f'{pre}{k}={x}')
    return ','.join(out)

def block_fewshot(n):
    pts = [(0, 'cell_1110'), (1, 'fewshot_k1'), (3, 'fewshot_k3'),
           (5, 'cell_1111'), (10, 'fewshot_k10')]
    got = [(k, g(f'{R}/{f}.json')) for k, f in pts]
    if not any(d for _, d in got): return n + 1
    head('few-shot curve across k, full feature set')
    base = dict(got).get(0)
    for k, d in got:
        if not d: continue
        w(f'k {k} n_targets {len(d["per_target"])} macro {v(d["macro_auroc"])} '
          f'seed_sd {v(d["seed_sd"])} mean_ap {v(d["mean_ap"])} '
          f'per_target_sd {v(d["per_target_sd"])} target_se {v(d["target_se"])} '
          f'reps {d["spec"].get("reps")}')
    if base:
        for k, d in got:
            if not d or k == 0: continue
            t, dl, pval = paired(base['per_target'], d['per_target'])
            w(f'k {k} delta_vs_k0 {v(dl.mean())} wilcoxon_p_vs_k0 {pv(pval)} '
              f'improved {int((dl > 0).sum())} n_paired {len(t)}')
    for k, bn in ((1, 'fewshot_k1'), (3, 'fewshot_k3'), (5, 'cell_1111'),
                  (10, 'fewshot_k10')):
        d, b = g(f'{R}/fewshot_k{k}_random.json'), g(f'{R}/{bn}.json')
        if not d: continue
        line = (f'k {k} shots random macro {v(d["macro_auroc"])} '
                f'seed_sd {v(d["seed_sd"])} mean_ap {v(d["mean_ap"])} '
                f'target_se {v(d["target_se"])}')
        if b:
            t, dl, pval = paired(b['per_target'], d['per_target'])
            line += (f' delta_vs_band {v(dl.mean())} '
                     f'wilcoxon_p_vs_band {pv(pval)} n_paired {len(t)}')
        if base:
            t0, dl0, p0 = paired(base['per_target'], d['per_target'])
            line += (f' delta_vs_k0 {v(dl0.mean())} '
                     f'wilcoxon_p_vs_k0 {pv(p0)}')
        w(line)
    return n + 1

def block_sweep(n):
    import glob
    fs = sorted(glob.glob(f'{R}/oat_*.json'))
    if not fs:
        print(f'gap: {R}/oat_*.json', file=sys.stderr); return n + 1
    refs = {'rf': g(f'{R}/cell_1000.json'), 'xgb': g(f'{R}/xgb_morgan.json')}
    head('one-factor-at-a-time configuration sweep around the reference')
    for h, r in refs.items():
        if r:
            w(f'reference {h} macro {v(r["macro_auroc"])} seed_sd {v(r["seed_sd"])} '
              f'mean_ap {v(r["mean_ap"])} target_se {v(r["target_se"])}')
    for p in fs:
        d = json.load(open(p))
        name = os.path.basename(p)[4:-5]
        hd = (d.get('_meta') or {}).get('model', 'rf')
        ref = refs.get(hd)
        line = (f'{name} macro {v(d["macro_auroc"])} seed_sd {v(d["seed_sd"])} '
                f'mean_ap {v(d["mean_ap"])} target_se {v(d["target_se"])}')
        if ref:
            line += f' delta_vs_reference_{hd} {v(d["macro_auroc"] - ref["macro_auroc"])}'
        w(line)
        w(f'{name}_spec ' + flat(d['spec']))
    return n + 1

def block_protocols(n):
    import glob
    fs = sorted(glob.glob(f'{R}/protocols*.json'))
    if not fs: return n + 1
    head('random and scaffold cross-validation, full corpus')
    for i, q in enumerate(fs):
        d = json.load(open(q))
        m = d.get('_meta') or {}
        if not i:
            meta(d)
            for k in ('n_rows', 'n_scaffolds', 'corpus', 'dedup_rule'):
                if k in m: w(f'{k} {m[k]}')
        h = m.get('model', 'rf')
        for k, a in d['aggregate'].items():
            w(f'{h} {k} pooled {v(a["pooled_mean"])} '
              f'macro {v(a.get("macro_mean"))} '
              f'n_targets_macro {a.get("n_targets_macro")} '
              f'across_seed_sd {v(a["across_seed_sd"])} '
              f'within_seed_fold_sd {v(a["mean_within_seed_fold_sd"])} '
              f'median_train_rows {a["median_train_rows"]} n_seeds {a["n_seeds"]}')
        for k, per in d['per_seed'].items():
            w(f'{h} {k}_pooled_per_seed '
              + ','.join(f'{z}={v(per[z]["pooled"])}' for z in per))
    return n + 1

def block_similarity(n):
    d = g(f'{R}/similarity.json')
    if not d: return n + 1
    head('maximum Tanimoto to the nearest training compound, by protocol')
    m = d.get('_meta', {})
    for k in ('cv_seed', 'n_rows', 'loto_folds', 'source_folds', 'nbins'):
        if k in m: w(f'{k} {m[k]}')
    for k, x in d.items():
        if k.startswith('_'): continue
        w(f'{k} n {x["n"]} median {x["median"]} mean {x["mean"]} '
          f'q05 {x["q05"]} q25 {x["q25"]} q75 {x["q75"]} q95 {x["q95"]} '
          f'frac_ge_090 {x["frac_ge_090"]} frac_eq_1 {x["frac_eq_1"]}')
    return n + 1

def block_panel(n):
    """Prediction heads on identical folds. Row labels are read from each
    artefact's own model record rather than written here."""
    names = ('cell_1000', 'xgb_morgan', 'gin', 'dmpnn')
    ds = [(nm, g(f'{R}/{nm}.json')) for nm in names]
    if not any(d for _, d in ds): return n
    head('baseline prediction heads on the primary cohort, identical folds')
    ref = None
    for nm, d in ds:
        if not d: continue
        m = d.get('_meta') or {}
        label = m.get('model') or (d.get('spec') or {}).get('architecture') or nm
        parts = [nm, f'model {label}', f'macro {v(d["macro_auroc"])}']
        for k, lbl in (('seed_sd', 'seed_sd'), ('mean_ap', 'ap'),
                       ('target_se', 'target_se')):
            if d.get(k) is not None: parts.append(f'{lbl} {v(d[k])}')
        parts.append(f'folds {d.get("n_targets_paired")}')
        if ref is not None:
            _, dl, pv = paired(ref[1], d['per_target'])
            parts += [f'delta_vs_{ref[0]} {v(dl.mean())}', f'wilcoxon_p {v(pv)}']
        else:
            ref = (nm, d['per_target'])
        w(' '.join(parts))
        if m.get('model_params'):
            w('  ' + nm + ' model_params '
              + ','.join(f'{a}={b}' for a, b in m['model_params'].items()))
    return n + 1

def block_noise(n):
    import glob
    fs = sorted(glob.glob(f'{R}/noise_*.json'))
    if not fs:
        print(f'gap: {R}/noise_*.json', file=sys.stderr); return n + 1
    head('training-label noise injected at the reference configuration')
    base = None
    for q in fs:
        d = json.load(open(q))
        if base is None: base = d['macro_auroc']
        w(f'flip {d["spec"].get("flip", 0.0)} macro {v(d["macro_auroc"])} '
          f'seed_sd {v(d["seed_sd"])} mean_ap {v(d["mean_ap"])} '
          f'target_se {v(d["target_se"])} '
          f'delta_vs_flip0 {v(d["macro_auroc"] - base)}')
    return n + 1

def block_splits(n):
    ds = [(nm, g(f'{R}/{nm}.json')) for nm in ('lofo', 'cross_e3')]
    if not any(d for _, d in ds):
        print(f'gap: {R}/lofo.json', file=sys.stderr); return n + 1
    head('leave-one-family-out and cross-E3 protocols, reference configuration')
    for nm, d in ds:
        if not d: continue
        m = d.get('_meta') or {}
        w(f'{nm} cohort {m.get("cohort")} n_folds {m.get("n_folds")} '
          f'macro {v(d["macro_auroc"])} seed_sd {v(d["seed_sd"])} '
          f'mean_ap {v(d["mean_ap"])} target_se {v(d["target_se"])} '
          f'per_fold_sd {v(d["per_target_sd"])}')
        w(f'{nm}_per_seed_macro '
          + ','.join(f'{a}={v(b)}' for a, b in d['per_seed_macro'].items()))
    return n + 1

def block_representation(n):
    names = ('cell_1000', 'chemberta', 'morgan_chemberta', 'morgan_esm',
             'morgan_chemberta_esm', 'chemberta_p64', 'morgan_chemberta_p64',
             'morgan_esm_p64', 'morgan_chemberta_esm_p64')
    ds = [(nm, g(f'{R}/{nm}.json')) for nm in names]
    ds = [(nm, d) for nm, d in ds if d]
    if len(ds) < 2: return n
    head('learned representations against the fingerprint reference')
    q = 'data/embeddings/provenance.json'
    if os.path.exists(q):
        prov = json.load(open(q))
        for a, b in (('molecule', 'molecule'), ('protein', 'protein')):
            w(f'{b}_model {prov[a + "_model"]} revision {prov[a + "_revision"]} '
              f'dim {prov[a + "_dim"]}')
    ref = None
    for nm, d in ds:
        parts = [nm, f'macro {v(d["macro_auroc"])}', f'seed_sd {v(d["seed_sd"])}',
                 f'mean_ap {v(d["mean_ap"])}', f'target_se {v(d["target_se"])}']
        if ref is None:
            ref = (nm, d['per_target'])
        else:
            t, dl, pval = paired(ref[1], d['per_target'])
            parts += [f'delta_vs_{ref[0]} {v(dl.mean())}', f'wilcoxon_p {pv(pval)}']
        w(' '.join(parts))
    return n + 1

def block_stats(n):
    d = g(f'{R}/stats.json')
    if not d: return n
    head('panel significance, source composition and the within-source residual')
    q = d.get('panel')
    if q:
        w(f'friedman n_targets {q["n_targets"]} statistic {v(q["friedman_stat"])} '
          f'p {pv(q["friedman_p"])} n_comparisons {q["n_comparisons"]}')
        for z in q['pairwise']:
            w(f'{z["a"]} vs {z["b"]} delta {v(z["delta"])} '
              f'wilcoxon_p {pv(z["p"])} holm {pv(z["p_holm"])}')
    c = d.get('composition')
    if c:
        for k, b in c['blocks'].items():
            w(f'{k} rows {b["rows"]} activity_rate {v(b["activity_rate"])}')
        w(f'boundary_row {c["boundary_row"]} '
          f'overall_activity_rate {v(c["overall_activity_rate"])}')
        w(f'source_only_auroc {v(c["source_only_auroc_empirical"])} '
          f'closed_form {v(c["source_only_auroc_closed_form"])}')
    for k, x in (d.get('within_source') or {}).items():
        w(f'within_source {k} macro {v(x["macro"])} seed_sd {v(x["seed_sd"])} '
          f'n_folds {x["n_folds"]}')
    return n + 1

BLOCKS = [block_cohort, block_ladder, block_warhead_variants, block_factorial,
          block_panel, block_stats, block_representation, block_fewshot, block_sweep, block_noise,
          block_cohort_cells, block_protocols, block_splits,
          block_calibration, block_similarity, block_overlap, block_source]

def build():
    n = 1
    for b in BLOCKS: n = b(n)
    os.makedirs(R, exist_ok=True)
    out, k = [], 0
    for line in L:
        if line.startswith('--- @@: '):
            k += 1; line = line.replace('@@', str(k), 1)
        out.append(line)
    from datetime import datetime, timezone
    doc = f'{R}/reproduction_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.txt'
    with open(doc, 'w') as f: f.write('\n'.join(out).lstrip() + '\n')
    print(f'{doc}: {k} blocks, {len(out)} lines', file=sys.stderr)

def ensure_folds():
    """Fold files are generated from the corpus, so a fresh clone needs no download."""
    need = [f for f in ('data/loto_primary_folds.json', 'data/lofo_folds.json')
            if not os.path.exists(f)]
    if not need: return
    import subprocess
    print('generating', ', '.join(need), file=sys.stderr)
    subprocess.run([sys.executable, 'scripts/generate_fold_files.py'], check=True)

def run_overlap():
    """Novel and seen decomposition of the retained arm, from its own stored
    predictions, written once so the document and the figures read one source."""
    out, pr = f'{R}/overlap.json', f'{R}/retained_65_predictions.json'
    if os.path.exists(out) or not os.path.exists(pr): return
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    P = json.load(open(pr))
    df = pd.read_csv('data/protac_bench.csv')
    tg = df.target_uniprot.astype(str).values
    smi = df.smiles.astype(str).values
    rows = {'novel': {}, 'seen': {}}
    for sd, byt in P.items():
        for t, rec in byt.items():
            te = tg == t
            train = set(smi[~te])
            seen = np.fromiter((x in train for x in smi[te]), bool, int(te.sum()))
            y, q = np.asarray(rec['y']), np.asarray(rec['p'])
            for k, m in (('novel', ~seen), ('seen', seen)):
                if m.sum() >= 5 and len(np.unique(y[m])) > 1:
                    rows[k].setdefault(sd, {})[t] = float(roc_auc_score(y[m], q[m]))
    res = {'_meta': dict(json.load(open(f'{R}/retained_65.json'))['_meta'],
                         source=pr, min_subset_rows=5)}
    for k in ('novel', 'seen'):
        per = {sd: float(np.mean(list(rows[k][sd].values()))) for sd in rows[k]}
        tgs = sorted({t for sd in rows[k] for t in rows[k][sd]})
        res[k] = dict(
            macro=round(float(np.mean(list(per.values()))), 4),
            seed_sd=round(float(np.std(list(per.values()), ddof=1)), 4),
            n_folds=len(tgs),
            per_seed_macro={a: round(b, 4) for a, b in per.items()},
            per_target={t: round(float(np.mean([rows[k][sd][t] for sd in rows[k]
                                                if t in rows[k][sd]])), 4) for t in tgs})
    json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out, file=sys.stderr)

def run_stats():
    """Omnibus and pairwise tests over the architecture panel, the source-stratified
    residual of the reference arm, and the activity composition by source block."""
    out = f'{R}/stats.json'
    if os.path.exists(out): return
    from scipy.stats import friedmanchisquare, wilcoxon
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    res = {}
    heads = ('cell_1000', 'xgb_morgan', 'gin', 'dmpnn')
    ds = {h: g(f'{R}/{h}.json') for h in heads}
    ds = {h: d for h, d in ds.items() if d}
    if len(ds) > 2:
        common = sorted(set.intersection(*(set(d['per_target']) for d in ds.values())))
        arr = {h: np.array([d['per_target'][t] for t in common]) for h, d in ds.items()}
        st, pv = friedmanchisquare(*arr.values())
        nm, pairs = list(arr), []
        for i in range(len(nm)):
            for j in range(i + 1, len(nm)):
                a, b = nm[i], nm[j]
                pairs.append(dict(a=a, b=b,
                                  delta=round(float((arr[b] - arr[a]).mean()), 4),
                                  p=float(wilcoxon(arr[a], arr[b]).pvalue)))
        pairs.sort(key=lambda z: z['p'])
        m, run = len(pairs), 0.0
        for i, q in enumerate(pairs):
            run = max(run, min(1.0, q['p'] * (m - i)))
            q['p_holm'] = round(run, 4)
        res['panel'] = dict(n_targets=len(common), friedman_stat=round(float(st), 4),
                            friedman_p=float(pv), n_comparisons=m, pairwise=pairs)
    df = pd.read_csv('data/protac_bench.csv')
    y, n = df.label.values, len(df)
    tg = df.target_uniprot.astype(str).values
    cut = 3181
    blocks = {'ribes': np.arange(n) < cut, 'protacdb': np.arange(n) >= cut}
    comp = {k: dict(rows=int(b.sum()), activity_rate=round(float(y[b].mean()), 4))
            for k, b in blocks.items()}
    r1, n1 = comp['ribes']['activity_rate'], comp['ribes']['rows']
    r2, n2 = comp['protacdb']['activity_rate'], comp['protacdb']['rows']
    pp = n2 * r2 / (n1 * r1 + n2 * r2)
    pn = n2 * (1 - r2) / (n1 * (1 - r1) + n2 * (1 - r2))
    res['composition'] = dict(
        blocks=comp, boundary_row=cut,
        overall_activity_rate=round(float(y.mean()), 4),
        source_only_auroc_empirical=round(float(roc_auc_score(y, blocks['protacdb'])), 4),
        source_only_auroc_closed_form=round(0.5 * (1 + pp - pn), 4))
    pr = f'{R}/cell_1000_predictions.json'
    if os.path.exists(pr):
        P = json.load(open(pr))
        per = {k: {} for k in blocks}
        for sd, byt in P.items():
            for t, rec in byt.items():
                te = tg == t
                yy, qq = np.asarray(rec['y']), np.asarray(rec['p'])
                for k, b in blocks.items():
                    m2 = b[te]
                    if m2.sum() >= 5 and len(np.unique(yy[m2])) > 1:
                        per[k].setdefault(sd, {})[t] = float(roc_auc_score(yy[m2], qq[m2]))
        res['within_source'] = {}
        for k in per:
            if not per[k]: continue
            ms = [float(np.mean(list(per[k][sd].values()))) for sd in per[k]]
            res['within_source'][k] = dict(
                macro=round(float(np.mean(ms)), 4),
                seed_sd=round(float(np.std(ms, ddof=1)), 4),
                n_folds=len(per[k][next(iter(per[k]))]))
    json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out, file=sys.stderr)

def run_similarity():
    """Only caller of src.evaluation.tanimoto_nn_by_protocol."""
    import pandas as pd
    from src.evaluation import tanimoto_nn_by_protocol
    out = f'{R}/similarity.json'
    if os.path.exists(out):
        print('have similarity', file=sys.stderr); return
    df = pd.read_csv('data/protac_bench.csv')
    folds = json.load(open('data/loto_primary_folds.json'))
    res = {}
    for dedup, tag in ((False, 'retained'), (True, 'dedup')):
        r = tanimoto_nn_by_protocol(df.smiles.tolist(), df.label.values, folds,
                                    n_folds=5, seed=42, dedup=dedup)
        for proto, vals in r.items():
            res[f'{proto}_{tag}'] = summarise(vals)
        print(f'similarity {tag} done', file=sys.stderr)
    from evaluation.engine import provenance
    pv = provenance()
    res['_meta'] = dict(cv_seed=42, n_rows=int(len(df)), nbins=NBINS,
                        loto_folds=len([k for k in folds if not k.startswith('_')]),
                        source_folds='data/loto_primary_folds.json',
                        **{k: pv[k] for k in ('git_commit', 'python', 'numpy', 'sklearn')})
    json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out, file=sys.stderr)

CELLS = ('1111', '1000', '1010', '1100', '1001', '1110', '1101', '1011',
         '0010', '0100', '0110', '0011', '0101', '0111')
NOISE = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)

def _cell(cid):
    m, w, a, k = (int(x) for x in cid)
    return dict(morgan=bool(m), warhead=bool(w), admet=bool(a),
                k=5 if k else 0, reps=5)

# One place says what the suite contains: name, spec, band, fold source, split.
# The ladder arms are factorial cells, so nothing here duplicates anything else.
ARMS = ([(f'cell_{c}', _cell(c), 'core', None, None) for c in CELLS]
        + [(f'fewshot_k{k}', dict(morgan=True, warhead=True, admet=True, k=k, reps=5),
            'core', None, None) for k in (1, 3, 10)]
        + [(f'fewshot_k{k}_random',
            dict(morgan=True, warhead=True, admet=True, k=k, reps=5, shots='random'),
            'core', None, None) for k in (1, 3, 5, 10)]
        + [('warhead_self_excluded',
            dict(morgan=True, warhead=True, self_excl=True), 'core', None, None),
           ('oat_bits512', dict(morgan=True, nbits=512), 'core', None, None),
           ('oat_bits1024', dict(morgan=True, nbits=1024), 'core', None, None),
           ('oat_radius3', dict(morgan=True, radius=3), 'core', None, None),
           ('oat_msl1', dict(morgan=True, rf=dict(min_samples_leaf=1)),
            'core', None, None),
           ('oat_cw_none', dict(morgan=True, rf=dict(class_weight=None)),
            'core', None, None),
           ('oat_trees100', dict(morgan=True, rf=dict(n_estimators=100)),
            'core', None, None),
           ('oat_all_defaults',
            dict(morgan=True, rf=dict(n_estimators=100, min_samples_leaf=1,
                                      class_weight=None)), 'core', None, None),
           ('oat_xgb_depth3', dict(morgan=True, model='xgb', xgb=dict(max_depth=3)),
            'core', None, None),
           ('oat_xgb_depth10', dict(morgan=True, model='xgb', xgb=dict(max_depth=10)),
            'core', None, None),
           ('oat_xgb_lr003',
            dict(morgan=True, model='xgb', xgb=dict(learning_rate=0.03)),
            'core', None, None),
           ('oat_xgb_subsample08',
            dict(morgan=True, model='xgb', xgb=dict(subsample=0.8)),
            'core', None, None)]
        + [(f'noise_{int(round(f * 1000)):03d}', dict(morgan=True, flip=f),
            'core', None, None) for f in NOISE]
        + [('xgb_morgan', dict(morgan=True, model='xgb'), 'core', None, None),
           ('lofo', dict(morgan=True), 'core', 'lofo', 'lofo_family_mapped'),
           ('cross_e3', dict(morgan=True, exclusions=dict(constituent=False)),
            'core', 'e3', 'cross_e3_recruited_ligase'),
           ('retained_65',
            dict(morgan=True, exclusions=dict(constituent=False, dedup=False)),
            'core', 'retained', 'loto_eligible_retained'),
           ('chemberta', dict(morgan=False, chemberta=True), 'core', None, None),
           ('morgan_chemberta', dict(morgan=True, chemberta=True),
            'core', None, None),
           ('morgan_esm', dict(morgan=True, esm=True), 'core', None, None),
           ('morgan_chemberta_esm', dict(morgan=True, chemberta=True, esm=True),
            'core', None, None),
           ('chemberta_p64', dict(morgan=False, chemberta=True, emb_dim=64),
            'core', None, None),
           ('morgan_chemberta_p64',
            dict(morgan=True, chemberta=True, emb_dim=64), 'core', None, None),
           ('morgan_esm_p64', dict(morgan=True, esm=True, emb_dim=64),
            'core', None, None),
           ('morgan_chemberta_esm_p64',
            dict(morgan=True, chemberta=True, esm=True, emb_dim=64),
            'core', None, None)])

# the core band carries the headline measurements; the ablation band carries the
# sweeps, the injected-noise ladder and the feature-off cells of the factorial
def _ablation(nm):
    return (nm.startswith(('oat_', 'noise_')) or nm == 'warhead_self_excluded'
            or (nm.startswith('cell_') and nm[5] == '0'))

ARMS = [(nm, sp, 'ablation' if _ablation(nm) else band, src, split)
        for nm, sp, band, src, split in ARMS]

def verify_against(prev):
    """Compare every registered arm against a previous copy of the suite.

    Reported on stderr, since the document carries results rather than checks.
    An arm whose code path did not change should be bit-identical per target.
    """
    same = changed = unpaired = 0
    for name, *_ in ARMS:
        a, b = f'{R}/{name}.json', f'{prev}/{name}.json'
        if not (os.path.exists(a) and os.path.exists(b)):
            unpaired += 1; continue
        x, y = json.load(open(a)), json.load(open(b))
        pa, pb = x.get('per_target', {}), y.get('per_target', {})
        shared = sorted(set(pa) & set(pb))
        moved = [t for t in shared if pa[t] != pb[t]]
        if not moved and x.get('macro_auroc') == y.get('macro_auroc'):
            same += 1
        else:
            changed += 1
            print(f'{name}: macro {y.get("macro_auroc")} -> {x.get("macro_auroc")}, '
                  f'{len(moved)} of {len(shared)} per-target values moved',
                  file=sys.stderr)
    print(f'verify: identical {same}, changed {changed}, unpaired {unpaired}',
          file=sys.stderr)

def run_core(seeds, bands=('core',), only=None):
    """Run every registered arm in the selected bands, skipping what exists.

    Feature matrices are cached per width and radius. Predictions are stored for
    the reference cell only, since calibration and coverage read them and no
    other block needs per-compound output.
    """
    from src.cohort import load, folds_e3, folds_from_file
    from src.data_utils import compute_morgan, load_admet
    from evaluation.engine import run
    c = load()
    smi = c['df'].smiles.tolist()
    Xa = np.nan_to_num(load_admet(), nan=0.0)
    sources = {'lofo': lambda: folds_from_file(c, 'data/lofo_folds.json'),
               'e3': lambda: folds_e3(c),
               'retained': lambda: [(t, c['tg'] == t) for t in c['eligible']]}
    cache = {}
    sel = [a for a in ARMS if a[2] in bands and not (only and a[0] != only)]
    for b in bands:
        if not any(a[2] == b for a in ARMS):
            print(f'band {b!r} selects no registered arm; the hyperparameter '
                  f'search and the architecture panel are separate runners and '
                  f'are not registered', file=sys.stderr)
    print(f'{len(sel)} arms selected', file=sys.stderr)
    for name, spec, band, src, split in sel:
        out = f'{R}/{name}.json'
        if os.path.exists(out):
            print(f'have {name}', file=sys.stderr); continue
        key = (spec.get('radius', 2), spec.get('nbits', 2048))
        if key not in cache:
            cache[key] = compute_morgan(smi, radius=key[0], nbits=key[1])
        print(f'--- {name} {spec} ---', file=sys.stderr, flush=True)
        run(name, spec, c, cache[key], Xa, seeds, out=out,
            store=(name in ('cell_1000', 'retained_65')), split=split,
            folds=sources[src]() if src else None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation', action='store_true',
                    help='also run the ablation band')
    ap.add_argument('--print-only', action='store_true')
    ap.add_argument('--seeds', default='canonical')
    ap.add_argument('--only', default=None, help='run one registered arm by name')
    ap.add_argument('--verify-against', default=None, metavar='DIR',
                    help='compare each arm against a previous copy of the suite')
    a = ap.parse_args()
    if not a.print_only:
        from src.seeds import parse
        ensure_folds()
        bands = ('core', 'ablation') if a.ablation else ('core',)
        run_core(parse(a.seeds), bands=bands, only=a.only)
        run_similarity()
        run_overlap()
        run_stats()
    build()
    if not a.print_only:
        from figures.generate_figures import main as render
        render()
    if a.verify_against: verify_against(a.verify_against)
