#!/usr/bin/env python
"""Non-kinase LOTO: does the ceiling hold without the dominant protein family?

Subsets:
  C0: all 65 eligible targets (reference)
  C1: non-kinase targets only (41)
  C2: kinase-only targets (24)
  C3: no-bromodomain (59)
  C4: no-kinase-no-bromo (35)

Two models per subset: RF+Morgan vs RF+Morgan+warhead_transfer+ADMET.
Expected: non-kinase 0.647, no-kinase-no-bromo 0.622.
"""

import sys, os, json, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets, load_admet
from src.stats import paired_wilcoxon
from src.debug import reduce_for_debug

# Family assignments — kinases and bromodomains are the two largest
KINASE_TARGETS = {
    'P24941', 'P11802', 'P49336', 'P00533', 'P08581', 'P15056',
    'P06239', 'Q02750', 'P36888', 'P33981', 'P30530', 'Q08881',
    'Q16539', 'Q06187', 'Q07889', 'P36897', 'Q13422', 'P50750',
    'P49840', 'Q14004', 'Q5S007', 'Q9UM73', 'Q9NYV4', 'Q96SW2',
}

BROMO_TARGETS = {
    'O60885', 'P25440', 'Q15059', 'Q9NWZ3', 'O60674', 'O60760',
}


def auroc_safe(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def eval_loto_morgan(FP, y, targets, eligible, seeds):
    results = []
    for tgt in eligible:
        te = targets == tgt
        tr = ~te
        y_te = y[te]
        if len(np.unique(y_te)) < 2:
            results.append({'target': tgt, 'auroc': 0.5})
            continue
        aucs = []
        for s in seeds:
            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                        max_features='sqrt', random_state=s, n_jobs=-1)
            rf.fit(FP[tr], y[tr])
            p = rf.predict_proba(FP[te])
            p = p[:, 1] if rf.classes_[1] == 1 else 1 - p[:, 0]
            aucs.append(auroc_safe(y_te, p))
        results.append({'target': tgt, 'auroc': float(np.mean(aucs))})
    return results


def _tanimoto_transfer(FP, y, tr_idx, te_idx, k=10):
    """Compute leak-free warhead transfer features."""
    fp_sum = FP.sum(axis=1)
    def _transfer_feats(row_idx, col_idx):
        AB = FP[row_idx] @ FP[col_idx].T
        denom = fp_sum[row_idx, None] + fp_sum[col_idx][None, :] - AB
        np.maximum(denom, 1e-10, out=denom)
        sim = AB / denom
        kk = min(k, len(col_idx))
        top_k = np.argpartition(sim, -kk, axis=1)[:, -kk:]
        feats = np.zeros((len(row_idx), 3), dtype=np.float32)
        for i in range(len(row_idx)):
            feats[i, 0] = y[col_idx[top_k[i]]].mean()
        feats[:, 1] = (sim > 0.5).sum(axis=1)
        feats[:, 2] = sim.max(axis=1)
        return feats
    te_feats = _transfer_feats(te_idx, tr_idx)
    # train: LOO (zero self-sim)
    AB_tr = FP[tr_idx] @ FP[tr_idx].T
    denom_tr = fp_sum[tr_idx, None] + fp_sum[tr_idx][None, :] - AB_tr
    np.maximum(denom_tr, 1e-10, out=denom_tr)
    sim_tr = AB_tr / denom_tr
    np.fill_diagonal(sim_tr, 0.0)
    kk = min(k, len(tr_idx))
    top_k_tr = np.argpartition(sim_tr, -kk, axis=1)[:, -kk:]
    tr_feats = np.zeros((len(tr_idx), 3), dtype=np.float32)
    for i in range(len(tr_idx)):
        tr_feats[i, 0] = y[tr_idx[top_k_tr[i]]].mean()
    tr_feats[:, 1] = (sim_tr > 0.5).sum(axis=1)
    tr_feats[:, 2] = sim_tr.max(axis=1)
    return tr_feats, te_feats


def eval_loto_full(FP, admet, y, targets, eligible, seeds):
    """Morgan + warhead_transfer + ADMET under LOTO."""
    results = []
    for tgt in eligible:
        te_mask = targets == tgt
        tr_mask = ~te_mask
        te_idx = np.where(te_mask)[0]
        tr_idx = np.where(tr_mask)[0]
        y_te = y[te_idx]
        if len(np.unique(y_te)) < 2:
            results.append({'target': tgt, 'auroc': 0.5})
            continue
        tr_feats, te_feats = _tanimoto_transfer(FP, y, tr_idx, te_idx)
        X_tr = np.hstack([FP[tr_idx], tr_feats, admet[tr_idx]])
        X_te = np.hstack([FP[te_idx], te_feats, admet[te_idx]])
        aucs = []
        for s in seeds:
            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                        max_features='sqrt', random_state=s, n_jobs=-1)
            rf.fit(X_tr, y[tr_idx])
            p = rf.predict_proba(X_te)
            p = p[:, 1] if rf.classes_[1] == 1 else 1 - p[:, 0]
            aucs.append(auroc_safe(y_te, p))
        results.append({'target': tgt, 'auroc': float(np.mean(aucs))})
    return results


def main():
    parser = argparse.ArgumentParser(description='Non-kinase LOTO robustness')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--output', default='results/nonkinase.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--debug', action='store_true', help='Reduced seeds, cohort, and trials for smoke test')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    targets = df['target_uniprot'].values
    y = df['label'].values.astype(int)

    print('Computing Morgan FP...')
    FP = compute_morgan(smiles)

    print('Loading ADMET scores...')
    admet = load_admet()
    admet = np.nan_to_num(admet, nan=0.0)

    all_eligible = get_eligible_targets(df)
    seeds, eligible, _ = reduce_for_debug(seeds=seeds, eligible=eligible, debug=args.debug)
    kinase_elig = [t for t in all_eligible if t in KINASE_TARGETS]
    bromo_elig = [t for t in all_eligible if t in BROMO_TARGETS]
    nonkinase_elig = [t for t in all_eligible if t not in KINASE_TARGETS]
    nobromo_elig = [t for t in all_eligible if t not in BROMO_TARGETS]
    nokinase_nobromo = [t for t in all_eligible
                        if t not in KINASE_TARGETS and t not in BROMO_TARGETS]

    print(f'Eligible: {len(all_eligible)} total, {len(kinase_elig)} kinase, '
          f'{len(bromo_elig)} bromo, {len(nonkinase_elig)} non-kinase, '
          f'{len(nokinase_nobromo)} no-kinase-no-bromo')

    subsets = {
        'C0_all': (all_eligible, None),
        'C1_nonkinase': (nonkinase_elig, KINASE_TARGETS),
        'C2_kinaseonly': (kinase_elig, None),
        'C3_nobromo': (nobromo_elig, BROMO_TARGETS),
        'C4_nokinase_nobromo': (nokinase_nobromo, KINASE_TARGETS | BROMO_TARGETS),
    }

    results_summary = {'seeds': seeds, 'conditions': {}}

    for name, (eligible, exclude_from_data) in subsets.items():
        print(f'\n{"="*50}')
        print(f'  {name} ({len(eligible)} targets)')
        print(f'{"="*50}')

        if exclude_from_data:
            keep = np.array([t not in exclude_from_data for t in targets])
            FP_sub, admet_sub, y_sub, tgt_sub = FP[keep], admet[keep], y[keep], targets[keep]
        elif name == 'C2_kinaseonly':
            # kinase-only test targets but train on all data
            FP_sub, admet_sub, y_sub, tgt_sub = FP, admet, y, targets
        else:
            FP_sub, admet_sub, y_sub, tgt_sub = FP, admet, y, targets

        print(f'  Data points: {len(y_sub)}')

        res_morgan = eval_loto_morgan(FP_sub, y_sub, tgt_sub, eligible, seeds)
        mean_morgan = np.mean([r['auroc'] for r in res_morgan])
        print(f'  Morgan AUROC: {mean_morgan:.4f}')

        res_full = eval_loto_full(FP_sub, admet_sub, y_sub, tgt_sub, eligible, seeds)
        mean_full = np.mean([r['auroc'] for r in res_full])
        print(f'  Full AUROC:   {mean_full:.4f}')

        # Wilcoxon
        da = {r['target']: r['auroc'] for r in res_full}
        db = {r['target']: r['auroc'] for r in res_morgan}
        common = sorted(set(da) & set(db))
        delta = np.mean([da[t] - db[t] for t in common]) if common else 0
        p_val = paired_wilcoxon([da[t] for t in common], [db[t] for t in common]) if len(common) >= 5 else 1.0

        results_summary['conditions'][name] = {
            'n_targets': len(eligible),
            'n_datapoints': int(len(y_sub)),
            'morgan_auroc': round(mean_morgan, 4),
            'full_auroc': round(mean_full, 4),
            'delta_full_vs_morgan': round(float(delta), 4),
            'wilcoxon_p': round(float(p_val), 4),
        }

    # delta vs C0
    c0_morgan = results_summary['conditions']['C0_all']['morgan_auroc']
    for v in results_summary['conditions'].values():
        v['delta_vs_C0'] = round(v['morgan_auroc'] - c0_morgan, 4)

    # print table
    print(f'\n{"="*70}')
    print(f'{"Subset":<25} {"N":>3} {"Morgan":>7} {"Full":>7} {"Δ vs C0":>8}')
    print('-' * 70)
    for k, v in results_summary['conditions'].items():
        print(f'{k:<25} {v["n_targets"]:>3} {v["morgan_auroc"]:>7.4f} '
              f'{v["full_auroc"]:>7.4f} {v["delta_vs_C0"]:>+8.4f}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
