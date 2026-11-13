#!/usr/bin/env python
"""10-seed confidence intervals for headline LOTO conditions.

Seeds: 7, 13, 29, 42, 43, 44, 53, 71, 89, 97
Conditions:
  C0: Morgan only
  C1: Morgan + warhead_transfer
  C2: Morgan + warhead + ADMET7
  C3: Morgan + warhead + ADMET7 + k=5 stratified

Transfer rates are computed per LOTO fold from training rows only.
"""

import sys, os, json, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets, load_admet
from src.stats import confidence_interval

SEEDS_10 = [7, 13, 29, 42, 43, 44, 53, 71, 89, 97]
K_SHOT = 5


def auroc_safe(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) >= 2 else 0.5


def _tanimoto_chunk(A, B, chunk_size=512):
    """Pairwise Tanimoto between A (n, d) and B (m, d). Chunked to bound memory."""
    sum_B = B.sum(axis=1)
    sims = np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    for st in range(0, A.shape[0], chunk_size):
        en = min(st + chunk_size, A.shape[0])
        Ac = A[st:en]
        AB = Ac @ B.T
        denom = np.maximum(Ac.sum(axis=1)[:, None] + sum_B[None, :] - AB, 1e-8)
        sims[st:en] = AB / denom
    return sims


def compute_transfer_rate(X_query, X_pool, y_pool, k=10):
    """Mean label of the k most Tanimoto-similar pool members, per query row."""
    sims = _tanimoto_chunk(X_query, X_pool)
    if k < sims.shape[1]:
        topk = np.argpartition(sims, -k, axis=1)[:, -k:]
    else:
        topk = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
    return np.array([y_pool[topk[i]].mean() for i in range(sims.shape[0])], dtype=np.float32)


def precompute_fold_rates(FP, y, targets, eligible, k=10):
    """Per-fold transfer rates for every row, pooled over that fold's training rows only.

    Seed-independent, so one computation per target serves all seeds and conditions.
    """
    cache = {}
    for i, uid in enumerate(eligible):
        tr = targets != uid
        cache[uid] = compute_transfer_rate(FP, FP[tr], y[tr], k=k).reshape(-1, 1)
        if (i + 1) % 10 == 0:
            print(f'    fold rates {i+1}/{len(eligible)}')
    return cache


N_TREES = 200   # canonical; main() overrides from --trees


def _rf(seed):
    return RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=3,
                                  class_weight='balanced', max_features='sqrt',
                                  random_state=seed, n_jobs=-1)


def _features(FP, admet, rates, rows, use_transfer, use_admet):
    blocks = [FP[rows]]
    if use_transfer:
        blocks.append(rates[rows])
    if use_admet:
        blocks.append(admet[rows])
    return np.hstack(blocks)


def eval_loto_single_seed(FP, admet, rate_cache, y, targets, eligible, seed,
                          use_transfer=False, use_admet=False):
    results = {}
    for uid in eligible:
        te = targets == uid
        tr = ~te
        y_te = y[te]
        if len(np.unique(y_te)) < 2:
            results[uid] = 0.5
            continue
        tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
        rates = rate_cache[uid] if use_transfer else None
        X_tr = _features(FP, admet, rates, tr_idx, use_transfer, use_admet)
        X_te = _features(FP, admet, rates, te_idx, use_transfer, use_admet)
        rf = _rf(seed)
        rf.fit(X_tr, y[tr_idx])
        p = rf.predict_proba(X_te)
        p = p[:, 1] if rf.classes_[1] == 1 else 1 - p[:, 0]
        results[uid] = auroc_safe(y_te, p)
    return results


def eval_loto_fewshot_single_seed(FP, admet, rate_cache, y, targets, eligible, seed,
                                  k=K_SHOT, use_transfer=False, use_admet=False):
    results = {}
    for uid in eligible:
        te_mask = targets == uid
        te_indices = np.where(te_mask)[0]
        y_te_all = y[te_mask]
        rate = y_te_all.mean()
        if len(np.unique(y_te_all)) < 2 or len(te_indices) <= k + 2:
            results[uid] = 0.5
            continue
        rng = np.random.RandomState(seed)
        te_pos = te_indices[y[te_indices] == 1]
        te_neg = te_indices[y[te_indices] == 0]
        k_pos = max(1, int(k * rate))
        k_neg = k - k_pos
        if len(te_pos) >= k_pos and len(te_neg) >= k_neg:
            shot_idx = np.concatenate([
                rng.choice(te_pos, size=k_pos, replace=False),
                rng.choice(te_neg, size=k_neg, replace=False),
            ])
        else:
            shot_idx = rng.choice(te_indices, size=min(k, len(te_indices) - 2), replace=False)
        remaining = np.setdiff1d(te_indices, shot_idx)
        if len(remaining) == 0 or len(np.unique(y[remaining])) < 2:
            results[uid] = 0.5
            continue
        tr_aug = np.concatenate([np.where(~te_mask)[0], shot_idx])
        rates = rate_cache[uid] if use_transfer else None
        X_tr = _features(FP, admet, rates, tr_aug, use_transfer, use_admet)
        X_te = _features(FP, admet, rates, remaining, use_transfer, use_admet)
        rf = _rf(seed)
        rf.fit(X_tr, y[tr_aug])
        p = rf.predict_proba(X_te)
        p = p[:, 1] if rf.classes_[1] == 1 else 1 - p[:, 0]
        results[uid] = auroc_safe(y[remaining], p)
    return results


def main():
    parser = argparse.ArgumentParser(description='10-seed confidence intervals')
    parser.add_argument('--seeds', default=','.join(str(s) for s in SEEDS_10))
    parser.add_argument('--output', default='results/confidence_intervals.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--morgan-bits', type=int, default=2048)
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--max-targets', type=int, default=0,
                        help='smoke test: evaluate only the first N eligible targets')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    global N_TREES
    N_TREES = args.trees
    t_start = time.time()

    print('Loading data...')
    df = load_dataset(args.data)
    targets = df['target_uniprot'].values
    y = df['label'].values.astype(int)
    smiles = df['smiles'].tolist()
    N = len(df)
    print(f'  {N} PROTACs, {y.sum()} active ({y.mean():.1%})')

    # features — use 2048-bit Morgan (consistent with other scripts)
    print('Computing Morgan FP...')
    FP = compute_morgan(smiles, nbits=args.morgan_bits)


    print('Loading ADMET...')
    admet = load_admet()
    admet = np.nan_to_num(admet, nan=0.0)

    eligible = get_eligible_targets(df)
    if args.max_targets:
        eligible = sorted(eligible)[:args.max_targets]
    print(f'{len(eligible)} eligible targets, {len(seeds)} seeds\n')

    print('Computing per-fold warhead transfer rates (train-only pool)...')
    rate_cache = precompute_fold_rates(FP, y, targets, eligible, k=10)

    conditions = {
        'C0_morgan_only':              {'transfer': False, 'admet': False, 'fewshot': False},
        'C1_morgan+warhead':           {'transfer': True,  'admet': False, 'fewshot': False},
        'C2_morgan+warhead+admet7':    {'transfer': True,  'admet': True,  'fewshot': False},
        'C3_morgan+warhead+admet7+k5': {'transfer': True,  'admet': True,  'fewshot': True},
    }

    all_results = {c: {} for c in conditions}

    for si, seed in enumerate(seeds):
        print(f'--- Seed {seed} ({si+1}/{len(seeds)}) ---')
        for cname, cfg in conditions.items():
            tc = time.time()
            if cfg['fewshot']:
                target_aurocs = eval_loto_fewshot_single_seed(
                    FP, admet, rate_cache, y, targets, eligible, seed,
                    use_transfer=cfg['transfer'], use_admet=cfg['admet'])
            else:
                target_aurocs = eval_loto_single_seed(
                    FP, admet, rate_cache, y, targets, eligible, seed,
                    use_transfer=cfg['transfer'], use_admet=cfg['admet'])
            mean_auc = np.mean(list(target_aurocs.values()))
            all_results[cname][seed] = mean_auc
            print(f'  {cname}: {mean_auc:.4f} [{time.time()-tc:.0f}s]')

    # stats
    print(f'\n{"="*75}')
    print(f'  {len(seeds)}-SEED CONFIDENCE INTERVALS')
    print(f'{"="*75}')
    print(f'{"Condition":<35} {"Mean":>7} {"Std":>7} {"95% CI":>18}')
    print(f'{"-"*75}')

    summary = {}
    for cname in conditions:
        vals = np.array([all_results[cname][s] for s in seeds])
        mean, ci_lo, ci_hi = confidence_interval(vals)
        std = vals.std(ddof=1)
        headline = f'{mean:.3f} +/- {std:.3f} ({len(seeds)} seeds)'
        print(f'{cname:<35} {mean:>7.4f} {std:>7.4f} [{ci_lo:.4f}, {ci_hi:.4f}]')
        summary[cname] = {
            'mean_auroc': round(float(mean), 4),
            'std': round(float(std), 4),
            'ci95_low': round(float(ci_lo), 4),
            'ci95_high': round(float(ci_hi), 4),
            'headline': headline,
            'per_seed': {str(s): round(float(all_results[cname][s]), 4) for s in seeds},
        }

    # paired comparisons vs C0
    print(f'\n{"Comparison":<40} {"Delta":>7} {"p-val":>8}')
    print(f'{"-"*55}')
    c0_vals = np.array([all_results['C0_morgan_only'][s] for s in seeds])
    for cname in ['C1_morgan+warhead', 'C2_morgan+warhead+admet7', 'C3_morgan+warhead+admet7+k5']:
        cx_vals = np.array([all_results[cname][s] for s in seeds])
        delta = (cx_vals - c0_vals).mean()
        try:
            _, pval = sp_stats.wilcoxon(cx_vals - c0_vals, alternative='two-sided')
        except:
            pval = 1.0
        print(f'{cname} vs C0  {delta:>+7.4f} {pval:>8.4f}')
        summary[cname]['delta_vs_C0'] = round(float(delta), 4)
        summary[cname]['wilcoxon_p_vs_C0'] = round(float(pval), 4)

    elapsed = (time.time() - t_start) / 60
    summary['_meta'] = {
        'seeds': seeds,
        'n_seeds': len(seeds),
        'n_targets': len(eligible),
        'k_shot': K_SHOT,
        'morgan_bits': int(FP.shape[1]),
        'rf_n_estimators': args.trees,
        'class_weight': 'balanced',
        'transfer_rate_scope': 'per-LOTO-fold, training rows only, k=10',
        'elapsed_min': round(elapsed, 1),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nElapsed: {elapsed:.1f} min')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
