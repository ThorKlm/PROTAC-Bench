#!/usr/bin/env python
"""Leave-One-Family-Out (LOFO) evaluation.

Groups targets by protein family, holds out entire families at once.
Compares LOFO vs LOTO to measure family-level generalization gap.

Expected: LOFO 0.612, LOTO 0.666. Family size correlates with AUROC (rho=0.520).
"""

import sys, os, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, spearmanr
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets

SEEDS = [42, 43, 44]

# Family assignments for eligible targets based on UniProt annotations
# yapf: disable
FAMILY_MAP = {}

# Kinases (largest family)
for t in ['P24941', 'P11802', 'P49336', 'P00533', 'P08581', 'P15056',
          'P06239', 'Q02750', 'P36888', 'P33981', 'P30530', 'Q08881',
          'Q16539', 'Q06187', 'Q07889', 'P36897', 'Q13422', 'P50750',
          'P49840', 'Q14004', 'Q5S007', 'Q9UM73', 'Q9NYV4', 'Q96SW2']:
    FAMILY_MAP[t] = 'Kinase'

# Bromodomains
for t in ['O60885', 'P25440', 'Q15059', 'Q9NWZ3', 'O60674', 'O60760']:
    FAMILY_MAP[t] = 'Bromodomain'

# HDACs
for t in ['O15379', 'Q9BY41', 'Q9UBN7']:
    FAMILY_MAP[t] = 'HDAC'

# Nuclear receptors
for t in ['P10275', 'P03372']:
    FAMILY_MAP[t] = 'Nuclear_receptor'

# SMARCA (SWI/SNF ATPases)
for t in ['P51531', 'P51532', 'P51531/P51532']:
    FAMILY_MAP[t] = 'SMARCA'

# CBP/p300 (HATs)
for t in ['Q92793', 'Q09472', 'Q92793/Q09472']:
    FAMILY_MAP[t] = 'HAT_CBP_p300'

# RAS pathway
for t in ['P01116', 'P01116/P04049', 'P01116/Q07889']:
    FAMILY_MAP[t] = 'RAS'

# Phosphatases
for t in ['P18031', 'Q06124', 'P17706']:
    FAMILY_MAP[t] = 'Phosphatase'

# Singletons / small families
FAMILY_MAP['P14902'] = 'IDO1'
FAMILY_MAP['P09874'] = 'PARP'
FAMILY_MAP['P10636'] = 'Tau'
FAMILY_MAP['P37840'] = 'Synuclein'
FAMILY_MAP['P40763'] = 'STAT'
FAMILY_MAP['P40337'] = 'E3_ligase'
FAMILY_MAP['P98170'] = 'IAP'
FAMILY_MAP['P43490'] = 'NAMPT'
FAMILY_MAP['P11474'] = 'Steroid_enzyme'
FAMILY_MAP['P15170'] = 'Translation_factor'
FAMILY_MAP['P36969'] = 'GPX4'
FAMILY_MAP['Q92918'] = 'HCFC1'
FAMILY_MAP['Q93009'] = 'DUB'
FAMILY_MAP['O75530'] = 'PRC2'
# yapf: enable


def auroc_safe(y_true, y_score):
    if len(set(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


def run_loto(df, eligible, fps, seeds):
    results = {}
    for target in eligible:
        mask = df['target_uniprot'] == target
        test_idx = df.index[mask].values
        train_idx = df.index[~mask].values
        X_train, y_train = fps[train_idx], df.loc[train_idx, 'label'].values
        X_test, y_test = fps[test_idx], df.loc[test_idx, 'label'].values
        aurocs = []
        for seed in seeds:
            rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_prob = rf.predict_proba(X_test)[:, 1]
            aurocs.append(auroc_safe(y_test, y_prob))
        results[target] = np.mean(aurocs)
    return results


def run_lofo(df, eligible, fps, family_map, seeds):
    """Hold out all targets in a family at once."""
    families = {}
    for t in eligible:
        fam = family_map.get(t, f'singleton_{t}')
        families.setdefault(fam, []).append(t)

    results = {}
    for fam, fam_targets in families.items():
        mask = df['target_uniprot'].isin(fam_targets)
        test_idx = df.index[mask].values
        train_idx = df.index[~mask].values
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        X_train = fps[train_idx]
        y_train = df.loc[train_idx, 'label'].values

        for seed in seeds:
            rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_prob_all = rf.predict_proba(fps[test_idx])[:, 1]

            for t in fam_targets:
                t_mask = df.loc[test_idx, 'target_uniprot'] == t
                if t_mask.sum() == 0:
                    continue
                y_t = df.loc[test_idx[t_mask], 'label'].values
                p_t = y_prob_all[t_mask]
                results.setdefault(t, []).append(auroc_safe(y_t, p_t))

    return {t: np.mean(v) for t, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description='LOFO evaluation')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--output', default='results/lofo.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading data...')
    df = load_dataset(args.data)
    df = df.reset_index(drop=True)

    eligible = get_eligible_targets(df)
    print(f'Eligible targets: {len(eligible)}')

    # ensure all assigned
    for t in eligible:
        if t not in FAMILY_MAP:
            FAMILY_MAP[t] = f'singleton_{t}'

    # family summary
    fam_summary = {}
    for t in eligible:
        fam = FAMILY_MAP[t]
        fam_summary.setdefault(fam, []).append(t)
    print(f'Families: {len(fam_summary)}')
    for fam in sorted(fam_summary, key=lambda f: -len(fam_summary[f])):
        print(f'  {fam}: {len(fam_summary[fam])} targets')

    print('\nComputing Morgan FP...')
    fps = compute_morgan(df['smiles'].tolist())

    print('Running LOTO...')
    loto_results = run_loto(df, eligible, fps, seeds)

    print('Running LOFO...')
    lofo_results = run_lofo(df, eligible, fps, FAMILY_MAP, seeds)

    # build results table
    tc = df.groupby('target_uniprot').agg(n=('label', 'size'), rate=('label', 'mean'))
    rows = []
    for target in eligible:
        rows.append({
            'target': target,
            'family': FAMILY_MAP.get(target, 'unknown'),
            'n_samples': int(tc.loc[target, 'n']) if target in tc.index else 0,
            'lofo_auroc': round(lofo_results.get(target, np.nan), 4),
            'loto_auroc': round(loto_results.get(target, np.nan), 4),
        })
    res_df = pd.DataFrame(rows)

    # paired comparison
    paired = res_df.dropna(subset=['lofo_auroc', 'loto_auroc'])
    lofo_arr = paired['lofo_auroc'].values
    loto_arr = paired['loto_auroc'].values
    mean_lofo = np.mean(lofo_arr)
    mean_loto = np.mean(loto_arr)
    delta = mean_lofo - mean_loto

    diffs = lofo_arr - loto_arr
    if np.any(diffs != 0) and len(diffs) >= 5:
        stat, pval = wilcoxon(lofo_arr, loto_arr)
    else:
        stat, pval = 0.0, 1.0

    print(f'\n{"="*50}')
    print(f'LOFO AUROC: {mean_lofo:.4f}')
    print(f'LOTO AUROC: {mean_loto:.4f}')
    print(f'Delta (LOFO-LOTO): {delta:+.4f}')
    print(f'Wilcoxon p: {pval:.4f}')

    # family size vs AUROC correlation
    fam_sizes = []
    fam_lofo_aucs = []
    for fam, targets in fam_summary.items():
        aucs = [lofo_results.get(t, np.nan) for t in targets]
        aucs = [a for a in aucs if not np.isnan(a)]
        if aucs:
            fam_sizes.append(len(targets))
            fam_lofo_aucs.append(np.mean(aucs))
    if len(fam_sizes) >= 5:
        rho, rho_p = spearmanr(fam_sizes, fam_lofo_aucs)
        print(f'Family size vs LOFO AUROC: rho={rho:.3f}, p={rho_p:.4f}')
    else:
        rho, rho_p = 0.0, 1.0

    # per-family summary
    fam_drops = res_df.copy()
    fam_drops['drop'] = fam_drops['lofo_auroc'] - fam_drops['loto_auroc']
    per_fam = fam_drops.groupby('family').agg(
        n_targets=('target', 'size'),
        mean_lofo=('lofo_auroc', 'mean'),
        mean_loto=('loto_auroc', 'mean'),
        mean_drop=('drop', 'mean'),
    ).sort_values('mean_drop')
    print('\nPer-family LOFO vs LOTO:')
    print(per_fam.to_string())

    summary = {
        'n_targets': len(eligible),
        'n_families': len(fam_summary),
        'lofo_mean_auroc': round(float(mean_lofo), 4),
        'loto_mean_auroc': round(float(mean_loto), 4),
        'delta_lofo_minus_loto': round(float(delta), 4),
        'wilcoxon_pval': round(float(pval), 4),
        'family_size_vs_auroc_rho': round(float(rho), 3),
        'seeds': seeds,
        'family_assignments': {fam: tgts for fam, tgts in sorted(fam_summary.items())},
        'per_family_summary': per_fam.reset_index().to_dict(orient='records'),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
