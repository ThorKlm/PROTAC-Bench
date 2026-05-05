#!/usr/bin/env python
"""Bayesian HPO with Optuna TPE sampler for PROTAC-Bench.

Searches over mol encoders, head types, and their hyperparameters using
LOTO evaluation. Full experiment was 1,086 trials across 18 dimensions;
default here is 100 for quick testing.

Results stored in SQLite so we can resume + analyze later.
"""

import sys, os, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.fingerprints import morgan_fp, maccs_fp, bulk_fps
from src.evaluation import loto_evaluate

# suppress optuna info spam during runs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Feature builders -- cached to avoid recomputing every trial
# ---------------------------------------------------------------------------
_FP_CACHE = {}

def _get_mol_features(smiles_list, encoder_name):
    """Return molecular feature matrix. Cached across trials."""
    if encoder_name in _FP_CACHE:
        return _FP_CACHE[encoder_name]

    print(f'  [cache miss] computing {encoder_name}...')
    if encoder_name == 'morgan_2048_r2':
        X, _ = bulk_fps(smiles_list, lambda s: morgan_fp(s, radius=2, nbits=2048),
                        show_progress=False)
    elif encoder_name == 'morgan_1024_r2':
        X, _ = bulk_fps(smiles_list, lambda s: morgan_fp(s, radius=2, nbits=1024),
                        show_progress=False)
    elif encoder_name == 'maccs_167':
        X, _ = bulk_fps(smiles_list, maccs_fp, show_progress=False)
    # ----------------------------------------------------------------
    # requires cached embeddings, see README
    # elif encoder_name == 'molformer_768':
    #     X = np.load('data/embeddings/molformer_protac.npy')
    # elif encoder_name == 'chemberta_600':
    #     X = np.load('data/embeddings/chemberta_protac.npy')
    # ----------------------------------------------------------------
    else:
        raise ValueError(f'Unknown mol encoder: {encoder_name}')

    _FP_CACHE[encoder_name] = X
    return X


# TODO: add ESM-2 protein embeddings when cached .npy files are available
_PROT_CACHE = {}

def _get_prot_features(targets_array, encoder_name):
    """Return protein feature matrix (or None if 'none')."""
    if encoder_name == 'none':
        return None
    # ----------------------------------------------------------------
    # requires cached embeddings, see README
    # if encoder_name in _PROT_CACHE:
    #     return _PROT_CACHE[encoder_name]
    # if encoder_name == 'esm2_1280':
    #     embs = np.load('data/embeddings/esm2_targets.npy', allow_pickle=True).item()
    #     X = np.stack([embs.get(t, np.zeros(1280)) for t in targets_array])
    #     _PROT_CACHE[encoder_name] = X
    #     return X
    # ----------------------------------------------------------------
    raise ValueError(f'Unknown prot encoder: {encoder_name}')


def _build_e3_onehot(df):
    """One-hot encode E3 ligase type."""
    e3_dummies = pd.get_dummies(df['e3_ligase'], prefix='e3')
    return e3_dummies.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def _make_model_fn(trial, head_type, scale_pos_weight):
    """Build a model_fn from trial params."""
    if head_type == 'rf':
        n_est = trial.suggest_categorical('rf_n_estimators', [100, 200, 500])
        min_leaf = trial.suggest_categorical('rf_min_samples_leaf', [1, 3, 5])
        max_d_str = trial.suggest_categorical('rf_max_depth', ['None', '10', '20', '30'])
        max_d = None if max_d_str == 'None' else int(max_d_str)

        def model_fn(X_train, y_train, X_test):
            clf = RandomForestClassifier(
                n_estimators=n_est,
                min_samples_leaf=min_leaf,
                max_depth=max_d,
                class_weight='balanced',
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            return clf.predict_proba(X_test)[:, 1]
        return model_fn

    elif head_type == 'xgboost':
        import xgboost as xgb
        max_d = trial.suggest_categorical('xgb_max_depth', [3, 6, 8, 12])
        eta = trial.suggest_categorical('xgb_eta', [0.01, 0.05, 0.1, 0.3])
        subsample = trial.suggest_categorical('xgb_subsample', [0.6, 0.8, 1.0])
        n_rounds = trial.suggest_categorical('xgb_n_rounds', [100, 200, 300])

        def model_fn(X_train, y_train, X_test):
            clf = xgb.XGBClassifier(
                n_estimators=n_rounds,
                max_depth=max_d,
                learning_rate=eta,
                subsample=subsample,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0,
            )
            clf.fit(X_train, y_train)
            return clf.predict_proba(X_test)[:, 1]
        return model_fn

    # ----------------------------------------------------------------
    # requires neural network heads + torch, see README
    # elif head_type == 'bilinear':
    #     from src.nn_heads import BilinearHead
    #     ...
    # elif head_type == 'concat_mlp':
    #     from src.nn_heads import ConcatMLP
    #     ...
    # elif head_type == 'cross_attention':
    #     from src.nn_heads import CrossAttentionHead
    #     ...
    # ----------------------------------------------------------------
    else:
        raise ValueError(f'Unknown head type: {head_type}')


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
import pandas as pd  # lazy, used for e3 encoding

def objective(trial, df, y, targets, eligible, scale_pos_weight):
    """Optuna objective: build features, run LOTO, return mean AUROC."""

    # --- search space ---
    mol_encoder = trial.suggest_categorical('mol_encoder', [
        'morgan_2048_r2', 'morgan_1024_r2', 'maccs_167',
        # 'molformer_768', 'chemberta_600',  # requires cached embeddings, see README
    ])
    prot_encoder = trial.suggest_categorical('prot_encoder', [
        'none',
        # 'esm2_1280',  # requires cached embeddings, see README
    ])
    head_type = trial.suggest_categorical('head_type', ['rf', 'xgboost'])
    e3_onehot = trial.suggest_categorical('e3_onehot', [True, False])
    normalize = trial.suggest_categorical('normalize', [True, False])

    # --- build feature matrix ---
    smiles = df['smiles'].tolist()
    X_mol = _get_mol_features(smiles, mol_encoder)

    X_prot = _get_prot_features(targets, prot_encoder)

    parts = [X_mol]
    if X_prot is not None:
        parts.append(X_prot)
    if e3_onehot:
        parts.append(_build_e3_onehot(df))

    X = np.hstack(parts).astype(np.float32)

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # --- build model ---
    model_fn = _make_model_fn(trial, head_type, scale_pos_weight)

    # --- evaluate via LOTO (single seed for speed during HPO) ---
    results = loto_evaluate(X, y, targets, model_fn, eligible=eligible, seeds=[42])

    # store per-target aurocs so we can analyze later
    per_target = results['per_seed']['42']
    trial.set_user_attr('per_target_aurocs', per_target)
    trial.set_user_attr('n_targets_evaluated', results['n_targets'])
    trial.set_user_attr('feature_dim', X.shape[1])

    return results['mean_auroc']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Bayesian HPO for PROTAC-Bench (Optuna TPE)')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='number of Optuna trials (full experiment: 1086)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--study-name', default='protac_hpo')
    parser.add_argument('--db-path', default='results/hpo_study.db')
    parser.add_argument('--n-startup', type=int, default=50,
                        help='random trials before TPE kicks in')
    parser.add_argument('--debug', action='store_true', help='Reduced seeds, cohort, and trials for smoke test')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    print(f'=== PROTAC-Bench HPO ===')
    print(f'n_trials={args.n_trials}, seed={args.seed}, n_startup={args.n_startup}')
    print(f'Study DB: {args.db_path}')

    # load data
    print('\nLoading dataset...')
    df = load_dataset(args.data)
    y = df['label'].values
    targets = df['target_uniprot'].values
    smiles = df['smiles'].tolist()

    eligible = get_eligible_targets(df)
    seeds, eligible, _ = reduce_for_debug(seeds=seeds, eligible=eligible, debug=args.debug)
    print(f'Dataset: {len(df)} samples, {len(np.unique(targets))} targets, '
          f'{len(eligible)} eligible')

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw = n_neg / n_pos
    print(f'Class balance: neg={n_neg}, pos={n_pos}, scale_pos_weight={spw:.2f}')

    # precompute common fingerprints to warm the cache
    print('\nPre-computing fingerprints...')
    t0 = time.time()
    for enc in ['morgan_2048_r2', 'morgan_1024_r2', 'maccs_167']:
        _get_mol_features(smiles, enc)
    print(f'  done in {time.time() - t0:.1f}s')

    # create / load study
    storage = f'sqlite:///{args.db_path}'
    sampler = TPESampler(seed=args.seed, n_startup_trials=args.n_startup)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        direction='maximize',
        load_if_exists=True,
    )

    existing = len(study.trials)
    if existing > 0:
        print(f'\nResuming study with {existing} existing trials')
    remaining = max(0, args.n_trials - existing)

    if remaining == 0:
        print('All trials already complete, skipping optimization')
    else:
        print(f'\nRunning {remaining} trials...\n')
        t_start = time.time()

        def _objective(trial):
            return objective(trial, df, y, targets, eligible, spw)

        # TODO: could use n_jobs=-1 here but LOTO already parallelizes internally
        study.optimize(_objective, n_trials=remaining, show_progress_bar=True)

        elapsed = time.time() - t_start
        print(f'\nOptimization finished in {elapsed:.0f}s '
              f'({elapsed / remaining:.1f}s/trial)')

    # --- summary ---
    print('\n' + '=' * 60)
    print('HPO RESULTS SUMMARY')
    print('=' * 60)

    best = study.best_trial
    print(f'\nBest trial #{best.number}:')
    print(f'  AUROC = {best.value:.4f}')
    print(f'  Params:')
    for k, v in best.params.items():
        print(f'    {k}: {v}')
    if 'n_targets_evaluated' in best.user_attrs:
        print(f'  Targets evaluated: {best.user_attrs["n_targets_evaluated"]}')
    if 'feature_dim' in best.user_attrs:
        print(f'  Feature dim: {best.user_attrs["feature_dim"]}')

    # top-5 overview
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    print(f'\nTop 5 / {len(completed)} completed trials:')
    print(f'{"#":>5} {"AUROC":>7}  {"head":>8} {"mol_enc":>16} {"norm":>5} {"e3":>5}')
    for t in completed[:5]:
        p = t.params
        print(f'{t.number:5d} {t.value:7.4f}  {p["head_type"]:>8} '
              f'{p["mol_encoder"]:>16} {str(p["normalize"]):>5} '
              f'{str(p["e3_onehot"]):>5}')

    # save top configs as json too
    top_configs = []
    for t in completed[:10]:
        top_configs.append({
            'trial': t.number,
            'auroc': t.value,
            'params': t.params,
            'user_attrs': {k: v for k, v in t.user_attrs.items()
                          if k != 'per_target_aurocs'},
        })

    out_json = 'results/hpo_top_configs.json'
    with open(out_json, 'w') as f:
        json.dump(top_configs, f, indent=2)
    print(f'\nTop configs saved to {out_json}')
    print(f'Study DB: {args.db_path}')

    # distribution of head_type in top-20
    top20_heads = [t.params['head_type'] for t in completed[:20]]
    from collections import Counter
    head_dist = Counter(top20_heads)
    print(f'\nHead type distribution in top-20: {dict(head_dist)}')

    # quick sanity: mean auroc across all trials
    all_aurocs = [t.value for t in completed]
    print(f'All trials: mean={np.mean(all_aurocs):.4f}, '
          f'std={np.std(all_aurocs):.4f}, '
          f'min={np.min(all_aurocs):.4f}, max={np.max(all_aurocs):.4f}')


if __name__ == '__main__':
    main()
