"""LOTO, random CV, and scaffold CV evaluation loops."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict


def loto_evaluate(X, y, targets, model_fn, eligible=None, seeds=[42], n_jobs=-1):
    """Leave-one-target-out evaluation.

    model_fn(X_train, y_train, X_test) -> y_pred probabilities
    Returns dict with per-target AUROCs, mean, std.
    """
    unique_targets = np.unique(targets)
    if eligible is not None:
        unique_targets = [t for t in unique_targets if t in eligible]

    all_seed_results = {}
    for seed in seeds:
        per_target = {}
        for tgt in unique_targets:
            test_mask = targets == tgt
            train_mask = ~test_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # skip if only one class in test
            if len(np.unique(y_test)) < 2:
                continue

            np.random.seed(seed)
            y_pred = model_fn(X_train, y_train, X_test)
            try:
                auc = roc_auc_score(y_test, y_pred)
            except ValueError:
                continue
            per_target[tgt] = auc

        all_seed_results[seed] = per_target

    # aggregate across seeds
    all_targets = set()
    for res in all_seed_results.values():
        all_targets.update(res.keys())

    aurocs = []
    for seed_res in all_seed_results.values():
        aurocs.extend(seed_res.values())

    mean_auroc = np.mean(aurocs) if aurocs else 0.0
    std_auroc = np.std([np.mean(list(r.values())) for r in all_seed_results.values()])

    return {
        'mean_auroc': float(mean_auroc),
        'std_auroc': float(std_auroc),
        'n_targets': len(all_targets),
        'n_seeds': len(seeds),
        'per_seed': {str(s): {k: float(v) for k, v in r.items()}
                     for s, r in all_seed_results.items()},
    }


def random_cv(X, y, model_fn, n_folds=5, seeds=[42]):
    """Stratified random cross-validation."""
    all_aurocs = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_aurocs = []
        for train_idx, test_idx in skf.split(X, y):
            y_pred = model_fn(X[train_idx], y[train_idx], X[test_idx])
            try:
                auc = roc_auc_score(y[test_idx], y_pred)
                fold_aurocs.append(auc)
            except ValueError:
                pass
        all_aurocs.append(np.mean(fold_aurocs))

    return {
        'mean_auroc': float(np.mean(all_aurocs)),
        'std_auroc': float(np.std(all_aurocs)),
        'n_folds': n_folds,
        'n_seeds': len(seeds),
    }


def _get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 'none'
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return 'none'


def scaffold_cv(X, y, smiles, model_fn, n_folds=5):
    """Scaffold-based cross-validation using Murcko scaffolds."""
    scaffolds = [_get_scaffold(s) for s in smiles]

    # group by scaffold
    scaffold_to_idx = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        scaffold_to_idx[sc].append(i)

    # assign scaffold groups to folds roughly equally
    groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)
    fold_sizes = [0] * n_folds
    fold_indices = [[] for _ in range(n_folds)]
    for grp in groups:
        smallest = int(np.argmin(fold_sizes))
        fold_indices[smallest].extend(grp)
        fold_sizes[smallest] += len(grp)

    aurocs = []
    for i in range(n_folds):
        test_idx = np.array(fold_indices[i])
        train_idx = np.array([j for j in range(len(y)) if j not in set(test_idx)])

        y_pred = model_fn(X[train_idx], y[train_idx], X[test_idx])
        try:
            auc = roc_auc_score(y[test_idx], y_pred)
            aurocs.append(auc)
        except ValueError:
            pass

    return {
        'mean_auroc': float(np.mean(aurocs)),
        'std_auroc': float(np.std(aurocs)),
        'n_folds': n_folds,
        'per_fold': [float(a) for a in aurocs],
    }
