#!/usr/bin/env python
"""Per-source LOTO: does the ~0.666 ceiling persist within each data source?

Tests whether the LOTO ceiling is caused by target novelty (real) or
cross-source batch effects (artifact).

  S0: protac8k only → RF+Morgan LOTO
  S1: tpddb only    → RF+Morgan LOTO
  S2: protac8k only → RF+Morgan+warhead_transfer (leak-free)

Expected: S0 ~0.666, S1 ~0.668 → ceiling is real.
"""

import sys, os, json, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.stats import paired_wilcoxon, format_result
from src.debug import reduce_for_debug

K_NEIGHBORS = 10
SIM_THRESHOLD = 0.5


def auroc_safe(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) >= 2 else 0.5


def eval_loto(FP, y, targets, eligible, seeds, label=""):
    """RF LOTO with seed averaging."""
    results = []
    t0 = time.time()
    for ti, tgt in enumerate(eligible):
        te = targets == tgt
        tr = ~te
        y_te = y[te]
        if len(np.unique(y_te)) < 2:
            results.append({'target': tgt, 'auroc': 0.5})
            continue
        all_probs = []
        for s in seeds:
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                        max_features='sqrt', random_state=s, n_jobs=-1)
            rf.fit(FP[tr], y[tr])
            p = rf.predict_proba(FP[te])
            p = p[:, 1] if rf.classes_[1] == 1 else 1 - p[:, 0]
            all_probs.append(p)
        mean_prob = np.mean(all_probs, axis=0)
        results.append({'target': tgt, 'auroc': float(auroc_safe(y_te, mean_prob))})
        if (ti + 1) % 10 == 0:
            print(f'    {label} fold {ti+1}/{len(eligible)} [{time.time()-t0:.0f}s]')
    return results


def eval_loto_with_transfer(FP, y, targets, eligible, seeds, label=""):
    """RF LOTO with Morgan + leak-free warhead_transfer features."""
    results = []
    fp_sum = FP.sum(axis=1)
    t0 = time.time()
    for ti, tgt in enumerate(eligible):
        te_mask = targets == tgt
        tr_mask = ~te_mask
        te_idx = np.where(te_mask)[0]
        tr_idx = np.where(tr_mask)[0]
        y_te = y[te_idx]
        if len(np.unique(y_te)) < 2:
            results.append({'target': tgt, 'auroc': 0.5})
            continue

        k = min(K_NEIGHBORS, len(tr_idx))

        # test→train similarity
        AB_te = FP[te_idx] @ FP[tr_idx].T
        denom_te = fp_sum[te_idx, None] + fp_sum[tr_idx][None, :] - AB_te
        np.maximum(denom_te, 1e-10, out=denom_te)
        sim_te = AB_te / denom_te
        top_k_te = np.argpartition(sim_te, -k, axis=1)[:, -k:]
        transfer_te = np.zeros((len(te_idx), 3), dtype=np.float32)
        for i in range(len(te_idx)):
            transfer_te[i, 0] = y[tr_idx[top_k_te[i]]].mean()
        transfer_te[:, 1] = (sim_te > SIM_THRESHOLD).sum(axis=1)
        transfer_te[:, 2] = sim_te.max(axis=1)

        # train→train similarity (LOO within train)
        AB_tr = FP[tr_idx] @ FP[tr_idx].T
        denom_tr = fp_sum[tr_idx, None] + fp_sum[tr_idx][None, :] - AB_tr
        np.maximum(denom_tr, 1e-10, out=denom_tr)
        sim_tr = AB_tr / denom_tr
        np.fill_diagonal(sim_tr, 0.0)
        top_k_tr = np.argpartition(sim_tr, -k, axis=1)[:, -k:]
        transfer_tr = np.zeros((len(tr_idx), 3), dtype=np.float32)
        for i in range(len(tr_idx)):
            transfer_tr[i, 0] = y[tr_idx[top_k_tr[i]]].mean()
        transfer_tr[:, 1] = (sim_tr > SIM_THRESHOLD).sum(axis=1)
        transfer_tr[:, 2] = sim_tr.max(axis=1)

        X_tr = np.hstack([FP[tr_idx], transfer_tr])
        X_te = np.hstack([FP[te_idx], transfer_te])

        all_probs = []
        for s in seeds:
            rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                        max_features='sqrt', random_state=s, n_jobs=-1)
            rf.fit(X_tr, y[tr_idx])
            p = rf.predict_proba(X_te)
            p = p[:, 1] if rf.classes_[1] == 1 else 1 - p[:, 0]
            all_probs.append(p)
        mean_prob = np.mean(all_probs, axis=0)
        results.append({'target': tgt, 'auroc': float(auroc_safe(y_te, mean_prob))})
        if (ti + 1) % 10 == 0:
            print(f'    {label} fold {ti+1}/{len(eligible)} [{time.time()-t0:.0f}s]')
    return results


def main():
    parser = argparse.ArgumentParser(description='Per-source LOTO robustness check')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--output', default='results/single_source.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--debug', action='store_true', help='Reduced seeds, cohort, and trials for smoke test')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading dataset...')
    df = load_dataset(args.data)
    print(f'  Total entries: {len(df)}')
    print(f'  Source distribution: {dict(df["source"].value_counts())}')

    summary = {}

    for src_name in df['source'].unique():
        sub = df[df['source'] == src_name].reset_index(drop=True)
        smiles = sub['smiles'].tolist()
        tgts = sub['target_uniprot'].values
        y = sub['label'].values.astype(np.float32)
        eligible = get_eligible_targets(sub)

        print(f'\n  {src_name}: {len(sub)} entries, {len(eligible)} LOTO-eligible targets')

        FP = compute_morgan(smiles)

        # Morgan-only LOTO
        print(f'  Running Morgan LOTO on {src_name}...')
        res = eval_loto(FP, y, tgts, eligible, seeds, label=src_name)
        mean_auc = np.mean([r['auroc'] for r in res])
        print(f'  => {src_name} LOTO AUROC = {mean_auc:.4f} ({len(eligible)} targets)')

        summary[f'{src_name}_morgan'] = {
            'n_entries': len(sub),
            'n_loto_targets': len(eligible),
            'mean_auroc': round(float(mean_auc), 4),
            'per_target': res,
        }

    # S2: protac8k with warhead transfer
    print('\nRunning warhead transfer on protac8k...')
    sub_p8k = df[df['source'] == 'protac8k'].reset_index(drop=True)
    smiles_p8k = sub_p8k['smiles'].tolist()
    tgts_p8k = sub_p8k['target_uniprot'].values
    y_p8k = sub_p8k['label'].values.astype(np.float32)
    eligible_p8k = get_eligible_targets(sub_p8k)
    FP_p8k = compute_morgan(smiles_p8k)

    res_wt = eval_loto_with_transfer(FP_p8k, y_p8k, tgts_p8k, eligible_p8k, seeds,
                                      label='protac8k+transfer')
    mean_wt = np.mean([r['auroc'] for r in res_wt])
    print(f'  => protac8k+transfer LOTO AUROC = {mean_wt:.4f}')

    # Wilcoxon: transfer vs baseline
    res_p8k = summary['protac8k_morgan']['per_target']
    da = {r['target']: r['auroc'] for r in res_wt}
    db = {r['target']: r['auroc'] for r in res_p8k}
    common = sorted(set(da) & set(db))
    if len(common) >= 5:
        a_vals = [da[t] for t in common]
        b_vals = [db[t] for t in common]
        p_val = paired_wilcoxon(a_vals, b_vals)
        delta = np.mean([da[t] - db[t] for t in common])
    else:
        p_val, delta = 1.0, 0.0

    summary['protac8k_morgan_transfer'] = {
        'n_entries': len(sub_p8k),
        'n_loto_targets': len(eligible_p8k),
        'mean_auroc': round(float(mean_wt), 4),
        'delta_vs_baseline': round(float(delta), 4),
        'wilcoxon_p': round(float(p_val), 4),
        'per_target': res_wt,
    }

    # comparison table
    print(f'\n{"="*60}')
    print('SINGLE-SOURCE LOTO COMPARISON')
    print(f'{"="*60}')
    for key, vals in summary.items():
        if key.startswith('_'):
            continue
        print(f'  {key:<30s} AUROC={vals["mean_auroc"]:.4f}  ({vals["n_loto_targets"]} targets)')

    all_source_aucs = [v['mean_auroc'] for k, v in summary.items()
                       if k.endswith('_morgan')]
    mean_ss = np.mean(all_source_aucs)
    print(f'\n  Mean single-source LOTO: {mean_ss:.4f}')
    if all(0.55 <= a <= 0.75 for a in all_source_aucs):
        print('  >> Ceiling is REAL — persists within each source')

    summary['_meta'] = {
        'seeds': seeds,
        'mean_single_source_auroc': round(float(mean_ss), 4),
        'ceiling_is_real': all(0.55 <= a <= 0.75 for a in all_source_aucs),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
