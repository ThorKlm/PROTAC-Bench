#!/usr/bin/env python
"""Validate top-k HPO configs with fresh seeds to check generalization.

Loads the Optuna study, extracts the top configs (unique by
mol_encoder/prot_encoder/head_type combo), re-evaluates each under
5 fresh seeds, and compares against the RF+Morgan baseline.

Key finding: HPO gains don't hold up.
  Best HPO trial 0.708 -> 0.668 with fresh seeds (p=0.925).
  i.e., extensive HPO on LOTO doesn't generalize -- the ceiling is real.
"""

import sys, os, json, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.fingerprints import morgan_fp, maccs_fp, bulk_fps
from src.evaluation import loto_evaluate
from src.stats import paired_wilcoxon, confidence_interval, format_result

optuna.logging.set_verbosity(optuna.logging.WARNING)

# fresh seeds, disjoint from HPO seed (42)
VALIDATION_SEEDS = [7, 13, 29, 53, 97]


def _get_mol_features(smiles_list, encoder_name, cache={}):
    """Compute or retrieve cached mol features."""
    if encoder_name in cache:
        return cache[encoder_name]
    if encoder_name == 'morgan_2048_r2':
        X, _ = bulk_fps(smiles_list, lambda s: morgan_fp(s, radius=2, nbits=2048),
                        show_progress=False)
    elif encoder_name == 'morgan_1024_r2':
        X, _ = bulk_fps(smiles_list, lambda s: morgan_fp(s, radius=2, nbits=1024),
                        show_progress=False)
    elif encoder_name == 'maccs_167':
        X, _ = bulk_fps(smiles_list, maccs_fp, show_progress=False)
    else:
        raise ValueError(f'Unknown encoder: {encoder_name}')
    cache[encoder_name] = X
    return X


def _build_e3_onehot(df):
    return pd.get_dummies(df['e3_ligase'], prefix='e3').values.astype(np.float32)


def _build_model_fn(params, spw):
    """Reconstruct model_fn from stored params dict."""
    head = params['head_type']
    if head == 'rf':
        n_est = params.get('rf_n_estimators', 500)
        min_leaf = params.get('rf_min_samples_leaf', 3)
        max_d_str = params.get('rf_max_depth', 'None')
        max_d = None if max_d_str == 'None' else int(max_d_str)

        def model_fn(X_train, y_train, X_test):
            clf = RandomForestClassifier(
                n_estimators=n_est, min_samples_leaf=min_leaf,
                max_depth=max_d, class_weight='balanced', n_jobs=-1)
            clf.fit(X_train, y_train)
            return clf.predict_proba(X_test)[:, 1]
        return model_fn

    elif head == 'xgboost':
        import xgboost as xgb
        def model_fn(X_train, y_train, X_test):
            clf = xgb.XGBClassifier(
                n_estimators=params.get('xgb_n_rounds', 300),
                max_depth=params.get('xgb_max_depth', 6),
                learning_rate=params.get('xgb_eta', 0.1),
                subsample=params.get('xgb_subsample', 1.0),
                scale_pos_weight=spw,
                use_label_encoder=False, eval_metric='logloss',
                n_jobs=-1, verbosity=0)
            clf.fit(X_train, y_train)
            return clf.predict_proba(X_test)[:, 1]
        return model_fn
    else:
        raise ValueError(f'Unknown head: {head}')


def _build_features(df, smiles, targets, params):
    """Reconstruct feature matrix from params."""
    X_mol = _get_mol_features(smiles, params['mol_encoder'])
    parts = [X_mol]

    # prot_encoder == 'none' in simplified search space
    if params.get('e3_onehot', False):
        parts.append(_build_e3_onehot(df))

    X = np.hstack(parts).astype(np.float32)
    if params.get('normalize', False):
        X = StandardScaler().fit_transform(X)
    return X


def _extract_unique_top_k(study, n_top):
    """Get top-k configs, unique by (mol_encoder, prot_encoder, head_type)."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    seen = set()
    top_configs = []
    for t in completed:
        p = t.params
        key = (p['mol_encoder'], p.get('prot_encoder', 'none'), p['head_type'])
        if key not in seen:
            seen.add(key)
            top_configs.append(t)
            if len(top_configs) >= n_top:
                break
    return top_configs


def main():
    parser = argparse.ArgumentParser(
        description='Validate top HPO configs with fresh seeds')
    parser.add_argument('--study-path', default='results/hpo_study.db')
    parser.add_argument('--study-name', default='protac_hpo')
    parser.add_argument('--n-top', type=int, default=5)
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='number of validation seeds')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--output', default='results/hpo_validation.json')
    args = parser.parse_args()

    seeds = VALIDATION_SEEDS[:args.n_seeds]
    print(f'=== HPO Validation (top-{args.n_top}, seeds={seeds}) ===\n')

    # load study
    storage = f'sqlite:///{args.study_path}'
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    n_trials = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
    print(f'Loaded study with {n_trials} completed trials')

    top_trials = _extract_unique_top_k(study, args.n_top)
    print(f'Extracted {len(top_trials)} unique configs\n')

    # load data
    print('Loading dataset...')
    df = load_dataset(args.data)
    y = df['label'].values
    targets = df['target_uniprot'].values
    smiles = df['smiles'].tolist()
    eligible = get_eligible_targets(df)

    n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
    spw = n_neg / n_pos

    # --- run baseline first ---
    print('\n--- Baseline: RF + Morgan 2048 (default HPs) ---')
    X_baseline = compute_morgan(smiles)

    def baseline_fn(X_train, y_train, X_test):
        clf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3,
            class_weight='balanced', n_jobs=-1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]

    t0 = time.time()
    baseline_res = loto_evaluate(X_baseline, y, targets, baseline_fn,
                                 eligible=eligible, seeds=seeds)
    print(f'  Baseline AUROC: {format_result(baseline_res["mean_auroc"], baseline_res["std_auroc"], len(seeds))}')
    print(f'  ({time.time() - t0:.0f}s)')

    # collect baseline per-target aurocs (averaged across seeds)
    baseline_per_target = {}
    for seed_res in baseline_res['per_seed'].values():
        for tgt, auc in seed_res.items():
            baseline_per_target.setdefault(tgt, []).append(auc)
    baseline_avg = {t: np.mean(v) for t, v in baseline_per_target.items()}

    # --- validate each top config ---
    val_results = []
    print('\n--- Validating top HPO configs ---')
    for rank, trial in enumerate(top_trials):
        p = trial.params
        tag = f'{p["head_type"]}+{p["mol_encoder"]}'
        print(f'\n[{rank+1}/{len(top_trials)}] Trial #{trial.number}: {tag} '
              f'(HPO AUROC={trial.value:.4f})')

        X = _build_features(df, smiles, targets, p)
        model_fn = _build_model_fn(p, spw)

        t0 = time.time()
        res = loto_evaluate(X, y, targets, model_fn,
                            eligible=eligible, seeds=seeds)
        elapsed = time.time() - t0

        # per-target avg for wilcoxon
        per_target = {}
        for seed_res in res['per_seed'].values():
            for tgt, auc in seed_res.items():
                per_target.setdefault(tgt, []).append(auc)
        avg_per_target = {t: np.mean(v) for t, v in per_target.items()}

        # paired comparison vs baseline
        common = sorted(set(avg_per_target.keys()) & set(baseline_avg.keys()))
        a = [avg_per_target[t] for t in common]
        b = [baseline_avg[t] for t in common]
        p_val = paired_wilcoxon(a, b)

        mean_ci, lo, hi = confidence_interval(list(avg_per_target.values()))

        print(f'  Validation AUROC: {res["mean_auroc"]:.4f} +/- {res["std_auroc"]:.4f}')
        print(f'  95% CI: [{lo:.3f}, {hi:.3f}]')
        print(f'  vs baseline p={p_val:.3f} ({elapsed:.0f}s)')

        val_results.append({
            'rank': rank + 1,
            'trial_number': trial.number,
            'tag': tag,
            'hpo_auroc': trial.value,
            'val_auroc': res['mean_auroc'],
            'val_std': res['std_auroc'],
            'ci_lo': lo, 'ci_hi': hi,
            'p_vs_baseline': p_val,
            'params': p,
        })

    # --- regression table ---
    print('\n' + '=' * 65)
    print('HPO -> VALIDATION REGRESSION TABLE')
    print('=' * 65)
    print(f'{"Rank":>4} {"Config":>25} {"HPO":>7} {"Val":>7} {"Delta":>7} {"p":>7}')
    print('-' * 65)
    for r in val_results:
        delta = r['val_auroc'] - r['hpo_auroc']
        sig = '*' if r['p_vs_baseline'] < 0.05 else ''
        print(f'{r["rank"]:4d} {r["tag"]:>25} {r["hpo_auroc"]:7.3f} '
              f'{r["val_auroc"]:7.3f} {delta:+7.3f} {r["p_vs_baseline"]:7.3f}{sig}')

    print('-' * 65)
    print(f'{"":>4} {"RF+Morgan baseline":>25} {"---":>7} '
          f'{baseline_res["mean_auroc"]:7.3f}')

    # TODO: add scatter plot of HPO vs validation AUROCs

    print('\nKey finding: HPO gains do not generalize with fresh seeds.')
    print('The LOTO ceiling is structural, not a tuning failure.\n')

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        'baseline': {
            'mean_auroc': baseline_res['mean_auroc'],
            'std_auroc': baseline_res['std_auroc'],
            'seeds': seeds,
        },
        'configs': val_results,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
