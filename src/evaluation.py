"""LOTO, random CV, and scaffold CV evaluation loops."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
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
        'n_targets_per_seed': {str(s): len(r) for s, r in all_seed_results.items()},
    }


def random_cv(X, y, model_fn, n_folds=5, seeds=[42]):
    """Stratified random cross-validation."""
    all_aurocs = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_aurocs = []
        for train_idx, test_idx in skf.split(X, y):
            y_pred = model_fn(X[train_idx], y[train_idx], X[test_idx])
            fold_aurocs.append(roc_auc_score(y[test_idx], y_pred))
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


def tanimoto_nn_by_protocol(smiles, y, loto_folds, n_folds=5, seed=42,
                            radius=2, nbits=2048, dedup=False):
    """Maximum Tanimoto to the nearest training compound, per split protocol.

    One value per test compound under random cross-validation, scaffold
    cross-validation and leave-one-target-out. Folds are built exactly as
    random_cv and the released fold file build them, so each
    distribution belongs to the protocol it is compared against. Fingerprints
    match the rest of the project, Morgan at the given radius and width.

    With dedup, training rows whose canonical SMILES occurs in the held-out
    fold are dropped. The LOTO arm prefers the fold file's own
    train_excluded_indices, so its distribution reflects the exclusions the
    deposit declares rather than a second derivation of them.
    """
    mols = [Chem.MolFromSmiles(x) if isinstance(x, str) else None for x in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
           if m is not None else None for m in mols]
    n, y = len(fps), np.asarray(y)
    smi = [str(x) for x in smiles]

    def nn(test_idx, train_idx):
        train = [fps[i] for i in train_idx if fps[i] is not None]
        if not train:
            raise ValueError('empty training set')
        return [max(DataStructs.BulkTanimotoSimilarity(fps[i], train))
                for i in test_idx if fps[i] is not None]

    def complement(test_idx):
        held = set(test_idx)
        return [i for i in range(n) if i not in held]

    def drop(test_idx, train_idx):
        if not dedup:
            return train_idx
        held = {smi[i] for i in test_idx}
        return [i for i in train_idx if smi[i] not in held]

    out = {}
    vals = []
    for tr, te in StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=seed).split(np.zeros((n, 1)), y):
        vals += nn(te, drop(te, tr))
    out['random_cv'] = vals

    groups = defaultdict(list)
    for i, x in enumerate(smiles):
        groups[_get_scaffold(x)].append(i)
    sizes, folds = [0] * n_folds, [[] for _ in range(n_folds)]
    for grp in sorted(groups.values(), key=len, reverse=True):
        j = int(np.argmin(sizes))
        folds[j].extend(grp)
        sizes[j] += len(grp)
    vals = []
    for j in range(n_folds):
        vals += nn(folds[j], drop(folds[j], complement(folds[j])))
    out['scaffold_cv'] = vals

    vals = []
    for tgt, spec in loto_folds.items():
        if tgt.startswith('_'):
            continue
        te = spec['test_indices']
        tr = complement(te)
        if dedup:
            ex = set(spec.get('train_excluded_indices') or [])
            tr = [i for i in tr if i not in ex] if ex else drop(te, tr)
        vals += nn(te, tr)
    out['loto'] = vals
    return out
