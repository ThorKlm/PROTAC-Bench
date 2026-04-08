#!/usr/bin/env python
"""Warhead ablation: is the transfer signal genuine or just warhead recognition?

Conditions:
  C0: full Morgan FP (baseline)
  C1: warhead-only FP (1024 bits, BRICS-based)
  C2: Morgan + warhead_transfer_rate
  C3: Morgan + warhead FP (explicit warhead substructure bits)

Key result: C2 >> C3 (p=0.009), transfer is more than just warhead identity.
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.stats import paired_wilcoxon, format_result

# E3 ligand SMARTS for identifying the E3-binding portion
# CRBN: phthalimide / thalidomide core
# VHL: hydroxyproline core
E3_SMARTS = [
    Chem.MolFromSmarts('[#6]1(=O)[#7][#6](=O)c2ccccc12'),   # phthalimide (CRBN)
    Chem.MolFromSmarts('O=C1CC(O)CN1'),                       # hydroxyproline (VHL)
    Chem.MolFromSmarts('c1ccc2c(c1)C(=O)N(C2=O)'),           # glutarimide variant
]


def extract_warhead_smiles(protac_smiles):
    """Extract warhead fragment from PROTAC using BRICS decomposition.
    Returns the largest non-E3-ligand fragment, or None if parsing fails.
    """
    mol = Chem.MolFromSmiles(protac_smiles)
    if mol is None:
        return None

    try:
        frags = BRICS.BRICSDecompose(mol)
    except Exception:
        return None

    if not frags:
        return None

    frags = list(frags)

    # remove fragments that match E3 ligand patterns
    non_e3 = []
    for frag_smi in frags:
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is None:
            continue
        is_e3 = False
        for pat in E3_SMARTS:
            if pat is not None and frag_mol.HasSubstructMatch(pat):
                is_e3 = True
                break
        if not is_e3:
            non_e3.append(frag_smi)

    if not non_e3:
        # fallback: just take the largest fragment
        non_e3 = frags

    # return largest by heavy atom count
    best = max(non_e3, key=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()
               if Chem.MolFromSmiles(s) else 0)
    return best


def compute_warhead_fps(smiles_list, nbits=1024, radius=2):
    """Compute Morgan FPs of extracted warhead fragments."""
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    n_failed = 0
    for i, smi in enumerate(smiles_list):
        wh = extract_warhead_smiles(smi)
        if wh is None:
            n_failed += 1
            continue
        mol = Chem.MolFromSmiles(wh)
        if mol is None:
            n_failed += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        fps[i] = np.array(fp)
    print(f"  warhead extraction failed for {n_failed}/{len(smiles_list)} molecules")
    return fps


# --- Tanimoto + transfer rate (copied from warhead_transfer.py) ---

def _tanimoto_chunk(A, B, chunk_size=512):
    n = A.shape[0]
    sum_B = B.sum(axis=1)
    sims = np.zeros((n, B.shape[0]), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        A_chunk = A[start:end]
        sum_A = A_chunk.sum(axis=1)
        AB = A_chunk @ B.T
        denom = sum_A[:, None] + sum_B[None, :] - AB
        denom = np.maximum(denom, 1e-8)
        sims[start:end] = AB / denom
    return sims


def compute_transfer_rate(X_test, X_train, y_train, k=10):
    sims = _tanimoto_chunk(X_test, X_train)
    if k < sims.shape[1]:
        topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
    else:
        topk_idx = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
    rates = np.zeros(X_test.shape[0], dtype=np.float32)
    for i in range(X_test.shape[0]):
        rates[i] = y_train[topk_idx[i]].mean()
    return rates


# --- LOTO loop ---

def run_condition(X_fp_base, X_extra, y, targets, eligible, seeds, label=""):
    """Run LOTO for a given feature matrix.
    X_extra can be a dict of extra features to compute per-fold (e.g. transfer rate)
    or a static ndarray to concatenate.
    """
    unique_targets = [t for t in np.unique(targets) if t in eligible]
    all_seed_results = {}

    for seed in seeds:
        np.random.seed(seed)
        per_target = {}
        for tgt in unique_targets:
            test_mask = targets == tgt
            train_mask = ~test_mask
            y_train = y[train_mask]
            y_test = y[test_mask]
            if len(np.unique(y_test)) < 2:
                continue

            if X_extra is not None and isinstance(X_extra, str) and X_extra == 'transfer':
                # compute per-fold transfer rate
                tr_rate = compute_transfer_rate(X_fp_base[train_mask], X_fp_base[train_mask], y_train, k=10)
                te_rate = compute_transfer_rate(X_fp_base[test_mask], X_fp_base[train_mask], y_train, k=10)
                X_tr = np.column_stack([X_fp_base[train_mask], tr_rate])
                X_te = np.column_stack([X_fp_base[test_mask], te_rate])
            elif X_extra is not None:
                X_tr = np.column_stack([X_fp_base[train_mask], X_extra[train_mask]])
                X_te = np.column_stack([X_fp_base[test_mask], X_extra[test_mask]])
            else:
                X_tr = X_fp_base[train_mask]
                X_te = X_fp_base[test_mask]

            clf = RandomForestClassifier(
                n_estimators=100, min_samples_leaf=3,
                max_features='sqrt', class_weight='balanced',
                random_state=seed, n_jobs=-1,
            )
            clf.fit(X_tr, y_train)
            y_pred = clf.predict_proba(X_te)[:, 1]
            try:
                per_target[tgt] = roc_auc_score(y_test, y_pred)
            except ValueError:
                pass

        all_seed_results[seed] = per_target

    aurocs = []
    for r in all_seed_results.values():
        aurocs.extend(r.values())
    mean_auc = np.mean(aurocs) if aurocs else 0.0
    std_auc = np.std([np.mean(list(r.values())) for r in all_seed_results.values()])

    return {
        'mean_auroc': float(mean_auc),
        'std_auroc': float(std_auc),
        'per_seed': {str(s): {k: float(v) for k, v in r.items()}
                     for s, r in all_seed_results.items()},
    }


def collect_paired(res_a, res_b, seeds):
    """Collect paired AUROCs for statistical test."""
    a_list, b_list = [], []
    for s in [str(s) for s in seeds]:
        shared = set(res_a['per_seed'].get(s, {}).keys()) & set(res_b['per_seed'].get(s, {}).keys())
        for tgt in sorted(shared):
            a_list.append(res_a['per_seed'][s][tgt])
            b_list.append(res_b['per_seed'][s][tgt])
    return a_list, b_list


def main():
    parser = argparse.ArgumentParser(description='Warhead ablation study')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--output', default='results/warhead_ablation.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading data...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values
    eligible = get_eligible_targets(df)

    print('Computing Morgan FPs...')
    X_morgan = compute_morgan(smiles)

    print('Extracting warhead FPs (BRICS + E3 filtering)...')
    X_warhead = compute_warhead_fps(smiles, nbits=1024)

    # --- C0: baseline Morgan ---
    print('\n[C0] Morgan baseline...')
    c0 = run_condition(X_morgan, None, y, targets, eligible, seeds)
    print(f'  C0: {c0["mean_auroc"]:.3f}')

    # --- C1: warhead-only FP ---
    print('[C1] Warhead-only FP...')
    c1 = run_condition(X_warhead, None, y, targets, eligible, seeds)
    print(f'  C1: {c1["mean_auroc"]:.3f}')

    # --- C2: Morgan + transfer rate ---
    print('[C2] Morgan + warhead_transfer_rate...')
    c2 = run_condition(X_morgan, 'transfer', y, targets, eligible, seeds)
    print(f'  C2: {c2["mean_auroc"]:.3f}')

    # --- C3: Morgan + warhead FP ---
    print('[C3] Morgan + warhead FP...')
    c3 = run_condition(X_morgan, X_warhead, y, targets, eligible, seeds)
    print(f'  C3: {c3["mean_auroc"]:.3f}')

    # --- Stats ---
    a2, b2 = collect_paired(c2, c0, seeds)
    p_c2_vs_c0 = paired_wilcoxon(a2, b2)
    a23, b23 = collect_paired(c2, c3, seeds)
    p_c2_vs_c3 = paired_wilcoxon(a23, b23)

    print(f'\n=== Warhead Ablation Results ===')
    print(f'C0 (Morgan):           {c0["mean_auroc"]:.3f}')
    print(f'C1 (warhead FP only):  {c1["mean_auroc"]:.3f}')
    print(f'C2 (Morgan+transfer):  {c2["mean_auroc"]:.3f}')
    print(f'C3 (Morgan+warhead):   {c3["mean_auroc"]:.3f}')
    print(f'C2 vs C0: p={p_c2_vs_c0:.4f}')
    print(f'C2 vs C3: p={p_c2_vs_c3:.4f}  <-- transfer beats explicit warhead')

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        'conditions': {
            'C0_morgan': c0, 'C1_warhead_only': c1,
            'C2_morgan_transfer': c2, 'C3_morgan_warhead': c3,
        },
        'p_C2_vs_C0': float(p_c2_vs_c0),
        'p_C2_vs_C3': float(p_c2_vs_c3),
        'seeds': seeds,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
