#!/usr/bin/env python
"""Warhead transfer under LOFO (leave-one-family-out).

Tests whether warhead transfer generalizes across protein families.
Spoiler: it doesn't. Transfer rate HURTS under LOFO (-0.009),
showing the signal is within-family only.

Conditions:
  C0: Morgan baseline (LOFO)
  C1: Morgan + warhead_transfer_rate (k=10, LOFO)
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.stats import paired_wilcoxon, format_result, confidence_interval

# Mapping from UniProt ID to protein family
# NOTE: manually curated from UniProt + literature
# TODO: double-check the "Other" assignments
FAMILY_MAP = {
    # Kinases
    'P00533': 'Kinase',   # EGFR
    'P04626': 'Kinase',   # ERBB2
    'P08581': 'Kinase',   # MET
    'P09619': 'Kinase',   # PDGFRB
    'P10721': 'Kinase',   # KIT
    'P11362': 'Kinase',   # FGFR1
    'P12931': 'Kinase',   # SRC
    'P17948': 'Kinase',   # FLT1/VEGFR1
    'P35968': 'Kinase',   # KDR/VEGFR2
    'P36888': 'Kinase',   # FLT3
    'Q02763': 'Kinase',   # TEK/TIE2
    'Q05397': 'Kinase',   # FAK/PTK2
    'Q13882': 'Kinase',   # PTK6/BRK
    'P06239': 'Kinase',   # LCK
    'P07949': 'Kinase',   # RET
    'P11309': 'Kinase',   # PIM1
    'P31749': 'Kinase',   # AKT1
    'P42336': 'Kinase',   # PIK3CA
    'P45983': 'Kinase',   # MAPK8/JNK1
    'P49137': 'Kinase',   # MAPKAPK2
    'Q16539': 'Kinase',   # MAPK14/p38a
    'Q9H461': 'Kinase',   # FRK
    'P51812': 'Kinase',   # RPS6KA3
    'O14965': 'Kinase',   # AURKA
    'Q96GD4': 'Kinase',   # AURKB
    'Q00535': 'Kinase',   # CDK5
    'P24941': 'Kinase',   # CDK2
    'P06493': 'Kinase',   # CDK1
    'P49336': 'Kinase',   # CDK8
    'Q00536': 'Kinase',   # CDK16
    'O96017': 'Kinase',   # CHK2
    'P43405': 'Kinase',   # SYK
    'P42684': 'Kinase',   # ABL2
    'P00519': 'Kinase',   # ABL1
    'Q13464': 'Kinase',   # ROCK1
    'O75116': 'Kinase',   # ROCK2
    'P04049': 'Kinase',   # RAF1
    'P15056': 'Kinase',   # BRAF
    'Q06187': 'Kinase',   # BTK
    'P53350': 'Kinase',   # PLK1
    'Q13627': 'Kinase',   # DYRK1A
    'Q9UQM7': 'Kinase',   # CAMK2A
    'O15264': 'Kinase',   # MAPK13
    # Bromodomains
    'O60885': 'Bromodomain',   # BRD4
    'P25440': 'Bromodomain',   # BRD2
    'Q15059': 'Bromodomain',   # BRD3
    'Q58F21': 'Bromodomain',   # BRDT
    'Q9NPI1': 'Bromodomain',   # BRD7
    'Q9H0E9': 'Bromodomain',   # BRD9
    'Q5T4S7': 'Bromodomain',   # UBR5 (has bromodomain)
    # HDACs
    'Q13547': 'HDAC',   # HDAC1
    'Q92769': 'HDAC',   # HDAC2
    'O15379': 'HDAC',   # HDAC3
    'Q9UBN7': 'HDAC',   # HDAC6
    'Q9BY41': 'HDAC',   # HDAC8
    'P56524': 'HDAC',   # HDAC4
    # Nuclear receptors
    'P10275': 'NuclearReceptor',   # AR
    'P03372': 'NuclearReceptor',   # ESR1
    'Q92731': 'NuclearReceptor',   # ESR2
    'P04150': 'NuclearReceptor',   # NR3C1/GR
    'P11473': 'NuclearReceptor',   # VDR
    'P19793': 'NuclearReceptor',   # RXRA
    'P37231': 'NuclearReceptor',   # PPARG
    'Q13772': 'NuclearReceptor',   # NCOA4
    # E3 ligase related
    'Q96SW2': 'Other',   # CRBN
    'P40337': 'Other',   # VHL
    'Q93034': 'Other',   # CUL2
    # Epigenetic (non-BRD)
    'O96028': 'Epigenetic',   # NSD2/WHSC1
    'Q9Y6K1': 'Epigenetic',   # DNMT3A
    'Q8WTS6': 'Epigenetic',   # SETD7
    'Q15047': 'Epigenetic',   # SETDB1
    'Q96KQ7': 'Epigenetic',   # EHMT2/G9a
    'O14686': 'Epigenetic',   # KMT2D
    'Q9UIF8': 'Epigenetic',   # BAZ2B
    'Q03164': 'Epigenetic',   # KMT2A/MLL1
    'O60341': 'Epigenetic',   # KDM1A/LSD1
    'P29590': 'Epigenetic',   # PML
    # STAT
    'P40763': 'STAT',   # STAT3
    'P42224': 'STAT',   # STAT1
    'P51692': 'STAT',   # STAT5B
    # KRAS / RAS family
    'P01116': 'RAS',    # KRAS
    'P01112': 'RAS',    # HRAS
    'P01111': 'RAS',    # NRAS
    # Other
    'P10636': 'Other',   # MAPT/Tau
    'P04637': 'Other',   # TP53
    'Q07817': 'Other',   # BCL2L1
    'P10415': 'Other',   # BCL2
    'O14746': 'Other',   # TERT
    'P38398': 'Other',   # BRCA1
    'Q96EB6': 'Other',   # SIRT1
    'P62805': 'Other',   # H4
    'Q9NZC2': 'Other',   # TREM2
    'P08253': 'Other',   # MMP2
    'P14780': 'Other',   # MMP9
    'P55210': 'Other',   # CASP7
    'Q07820': 'Other',   # MCL1
    'O15350': 'Other',   # TP73
    'P42345': 'Other',   # MTOR
    'Q16665': 'Other',   # HIF1A
}


def get_family(uniprot_id):
    """Get family for a target, defaulting to 'Other'."""
    return FAMILY_MAP.get(uniprot_id, 'Other')


# --- Tanimoto (same as warhead_transfer.py) ---

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


# --- LOFO evaluation ---

def get_family_assignments(targets):
    """Map each sample's target to its family."""
    return np.array([get_family(t) for t in targets])


def run_lofo(X_fp, y, targets, families, use_transfer=False, k=10, seeds=[42]):
    """Leave-one-family-out evaluation.

    When use_transfer=True, warhead_transfer_rate is recomputed per LOFO fold
    using only training-family data to prevent family-level leakage.
    """
    unique_families = sorted(set(families))
    # skip families with too few samples or single-class
    # also skip "Other" since it's a catch-all
    eligible_families = []
    for fam in unique_families:
        mask = families == fam
        n = mask.sum()
        rate = y[mask].mean()
        if n >= 10 and 0.1 <= rate <= 0.9 and fam != 'Other':
            eligible_families.append(fam)

    print(f"  LOFO eligible families: {eligible_families}")

    all_seed_results = {}
    for seed in seeds:
        np.random.seed(seed)
        per_family = {}

        for fam in eligible_families:
            test_mask = families == fam
            train_mask = ~test_mask

            y_train = y[train_mask]
            y_test = y[test_mask]

            if len(np.unique(y_test)) < 2:
                continue

            if use_transfer:
                # compute transfer rate from training data only (no family leakage)
                tr_rate = compute_transfer_rate(X_fp[train_mask], X_fp[train_mask], y_train, k=k)
                te_rate = compute_transfer_rate(X_fp[test_mask], X_fp[train_mask], y_train, k=k)
                X_tr = np.column_stack([X_fp[train_mask], tr_rate])
                X_te = np.column_stack([X_fp[test_mask], te_rate])
            else:
                X_tr = X_fp[train_mask]
                X_te = X_fp[test_mask]

            clf = RandomForestClassifier(
                n_estimators=100, min_samples_leaf=3,
                class_weight='balanced', random_state=seed, n_jobs=-1,
            )
            clf.fit(X_tr, y_train)
            y_pred = clf.predict_proba(X_te)[:, 1]

            try:
                auc = roc_auc_score(y_test, y_pred)
            except ValueError:
                continue
            per_family[fam] = auc

        all_seed_results[seed] = per_family
        if per_family:
            print(f"    seed {seed}: mean={np.mean(list(per_family.values())):.3f}")

    # aggregate
    aurocs = []
    for r in all_seed_results.values():
        aurocs.extend(r.values())
    mean_auc = np.mean(aurocs) if aurocs else 0.0
    std_auc = np.std([np.mean(list(r.values())) for r in all_seed_results.values() if r])

    return {
        'mean_auroc': float(mean_auc),
        'std_auroc': float(std_auc),
        'per_seed': {str(s): {k: float(v) for k, v in r.items()}
                     for s, r in all_seed_results.items()},
        'eligible_families': eligible_families,
    }


def main():
    parser = argparse.ArgumentParser(description='Warhead transfer under LOFO')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--k-neighbors', type=int, default=10)
    parser.add_argument('--output', default='results/warhead_lofo.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading data...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values

    print('Computing Morgan FPs (2048 bits)...')
    X_fp = compute_morgan(smiles)
    print(f'  shape: {X_fp.shape}')

    families = get_family_assignments(targets)
    fam_counts = {}
    for fam in families:
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    print(f'\nFamily distribution:')
    for fam, cnt in sorted(fam_counts.items(), key=lambda x: -x[1]):
        print(f'  {fam}: {cnt}')

    # --- C0: Morgan LOFO baseline ---
    print('\n[C0] Morgan baseline (LOFO)...')
    c0 = run_lofo(X_fp, y, targets, families, use_transfer=False, seeds=seeds)
    print(f'  C0 mean AUROC: {c0["mean_auroc"]:.3f}')

    # --- C1: Morgan + transfer (LOFO) ---
    print(f'\n[C1] Morgan + warhead_transfer (LOFO, k={args.k_neighbors})...')
    c1 = run_lofo(X_fp, y, targets, families, use_transfer=True, k=args.k_neighbors, seeds=seeds)
    print(f'  C1 mean AUROC: {c1["mean_auroc"]:.3f}')

    # --- Stats ---
    a_list, b_list = [], []
    for s in [str(s) for s in seeds]:
        shared = set(c0['per_seed'].get(s, {}).keys()) & set(c1['per_seed'].get(s, {}).keys())
        for fam in sorted(shared):
            b_list.append(c0['per_seed'][s][fam])
            a_list.append(c1['per_seed'][s][fam])

    p_val = paired_wilcoxon(a_list, b_list)
    delta = c1['mean_auroc'] - c0['mean_auroc']

    print(f'\n=== LOFO Results ===')
    print(f'C0 (Morgan):           {c0["mean_auroc"]:.3f}')
    print(f'C1 (Morgan+transfer):  {c1["mean_auroc"]:.3f}')
    print(f'Delta: {delta:+.3f}, p={p_val:.4f}')
    if delta < 0:
        print('>> Transfer HURTS under LOFO -- signal is within-family only')

    # per-family breakdown
    print('\nPer-family breakdown (first seed):')
    s0 = str(seeds[0])
    for fam in sorted(set(c0['per_seed'].get(s0, {}).keys()) | set(c1['per_seed'].get(s0, {}).keys())):
        v0 = c0['per_seed'].get(s0, {}).get(fam, float('nan'))
        v1 = c1['per_seed'].get(s0, {}).get(fam, float('nan'))
        d = v1 - v0 if not (np.isnan(v0) or np.isnan(v1)) else float('nan')
        print(f'  {fam:20s}  C0={v0:.3f}  C1={v1:.3f}  delta={d:+.3f}')

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        'method': 'warhead_transfer_LOFO',
        'seeds': seeds, 'k_neighbors': args.k_neighbors,
        'C0_baseline': c0, 'C1_transfer': c1,
        'delta': float(delta), 'p_value': float(p_val),
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
