#!/usr/bin/env python
"""ADMET feature ablation study.

Two experiments:
  1. Leave-one-out: remove each ADMET feature from the full set, measure drop.
  2. Individual: add each single ADMET feature to Morgan, measure gain.

Expected: HIA_Hou most important (drop 0.021 when removed).

NOTE: uses single seed by default for speed. The full ADMET cascade
(admet_cascade.py) already established significance with multi-seed.
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets, load_admet
from src.evaluation import loto_evaluate
from src.stats import paired_wilcoxon, format_result

ADMET_NAMES = [
    "Caco2_Wang", "PAMPA_NCATS", "HIA_Hou", "Solubility_AqSolDB",
    "Pgp_Broccatelli", "Lipophilicity_AstraZeneca", "Clearance_Hepatocyte_AZ",
]


def make_rf_fn(seed=42):
    def model_fn(X_tr, y_tr, X_te):
        clf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3,
            class_weight='balanced', random_state=seed, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]
    return model_fn


def main():
    parser = argparse.ArgumentParser(description='ADMET ablation (leave-one-out + individual)')
    parser.add_argument('--seed', type=int, default=42, help='single seed for speed')
    parser.add_argument('--output', default='results/admet_ablation.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    seeds = [args.seed]

    print('Loading data...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values

    print('Computing Morgan FPs...')
    X_morgan = compute_morgan(smiles)

    print('Loading ADMET scores...')
    X_admet = load_admet()
    X_admet = np.nan_to_num(X_admet, nan=0.0)

    eligible = get_eligible_targets(df)
    print(f'Eligible targets: {len(eligible)}')
    print(f'Dataset: {len(df)} PROTACs, {y.sum()} active ({y.mean():.1%})')
    print(f'ADMET features: {ADMET_NAMES}')

    # quick sanity: print ADMET stats
    print('\nADMET feature stats:')
    for i, name in enumerate(ADMET_NAMES):
        col = X_admet[:, i]
        print(f'  {name:30s}  mean={col.mean():.3f}  std={col.std():.3f}  '
              f'min={col.min():.3f}  max={col.max():.3f}')

    model_fn = make_rf_fn(args.seed)

    # --- C0: Morgan-only baseline ---
    print('\n[Baseline] Morgan only...')
    baseline = loto_evaluate(X_morgan, y, targets, model_fn, eligible=eligible, seeds=seeds)
    print(f'  AUROC: {baseline["mean_auroc"]:.3f}')

    # --- Full ADMET ---
    X_full = np.column_stack([X_morgan, X_admet])
    print('\n[Full] Morgan + all 7 ADMET...')
    full_res = loto_evaluate(X_full, y, targets, model_fn, eligible=eligible, seeds=seeds)
    print(f'  AUROC: {full_res["mean_auroc"]:.3f}')

    # ========================================
    # Experiment 1: Leave-one-out ablation
    # ========================================
    print('\n--- Leave-one-out ablation ---')
    loo_results = {}
    for i, name in enumerate(ADMET_NAMES):
        # remove column i from ADMET
        keep = [j for j in range(len(ADMET_NAMES)) if j != i]
        X_loo = np.column_stack([X_morgan, X_admet[:, keep]])

        res = loto_evaluate(X_loo, y, targets, model_fn, eligible=eligible, seeds=seeds)
        drop = full_res['mean_auroc'] - res['mean_auroc']
        loo_results[name] = {
            'auroc': res['mean_auroc'],
            'drop': float(drop),
            'per_seed': res['per_seed'],
        }
        print(f'  remove {name:30s}  AUROC={res["mean_auroc"]:.3f}  drop={drop:+.4f}')

    # rank by drop (positive drop = feature was useful)
    print('\nRanked by importance (largest drop = most important):')
    ranked = sorted(loo_results.items(), key=lambda x: -x[1]['drop'])
    for rank, (name, info) in enumerate(ranked, 1):
        print(f'  {rank}. {name:30s}  drop={info["drop"]:+.4f}')
    # TODO: could also do permutation importance here for comparison

    # ========================================
    # Experiment 2: Individual features
    # ========================================
    print('\n--- Individual ADMET features ---')
    individual_results = {}
    for i, name in enumerate(ADMET_NAMES):
        X_single = np.column_stack([X_morgan, X_admet[:, i]])

        res = loto_evaluate(X_single, y, targets, model_fn, eligible=eligible, seeds=seeds)
        gain = res['mean_auroc'] - baseline['mean_auroc']
        individual_results[name] = {
            'auroc': res['mean_auroc'],
            'gain': float(gain),
            'per_seed': res['per_seed'],
        }
        print(f'  +{name:30s}  AUROC={res["mean_auroc"]:.3f}  gain={gain:+.4f}')

    # rank individual gains
    print('\nRanked by individual gain:')
    ind_ranked = sorted(individual_results.items(), key=lambda x: -x[1]['gain'])
    for rank, (name, info) in enumerate(ind_ranked, 1):
        print(f'  {rank}. {name:30s}  gain={info["gain"]:+.4f}')

    # paired test: full ADMET vs. baseline
    a_list, b_list = [], []
    s = str(args.seed)
    shared = set(full_res['per_seed'].get(s, {}).keys()) & set(baseline['per_seed'].get(s, {}).keys())
    for tgt in sorted(shared):
        a_list.append(full_res['per_seed'][s][tgt])
        b_list.append(baseline['per_seed'][s][tgt])
    p_val = paired_wilcoxon(a_list, b_list)

    print(f'\n=== Summary ===')
    print(f'Baseline (Morgan):     {baseline["mean_auroc"]:.3f}')
    print(f'Full (Morgan+ADMET7):  {full_res["mean_auroc"]:.3f}  p={p_val:.4f}')
    print(f'Most important feature: {ranked[0][0]} (drop={ranked[0][1]["drop"]:.4f})')

    # --- HIA stat test ---
    # test HIA individually vs baseline
    hia_key = "HIA_Hou"
    if hia_key in individual_results:
        hia_a, hia_b = [], []
        hia_per = individual_results[hia_key]['per_seed']
        shared2 = set(hia_per.get(s, {}).keys()) & set(baseline['per_seed'].get(s, {}).keys())
        for tgt in sorted(shared2):
            hia_a.append(hia_per[s][tgt])
            hia_b.append(baseline['per_seed'][s][tgt])
        p_hia = paired_wilcoxon(hia_a, hia_b)
        print(f'HIA individual: gain={individual_results[hia_key]["gain"]:+.4f}, p={p_hia:.4f}')

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        'seed': args.seed,
        'baseline_auroc': baseline['mean_auroc'],
        'full_auroc': full_res['mean_auroc'],
        'p_full_vs_baseline': float(p_val),
        'leave_one_out': {k: {'auroc': v['auroc'], 'drop': v['drop']} for k, v in loo_results.items()},
        'individual': {k: {'auroc': v['auroc'], 'gain': v['gain']} for k, v in individual_results.items()},
        'importance_ranking': [name for name, _ in ranked],
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
