"""Dataset loading and feature computation for PROTAC-Bench."""

import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def load_dataset(path='data/protac_bench.csv'):
    """Load the PROTAC-Bench dataset."""
    df = pd.read_csv(path)
    return df


def compute_morgan(smiles_list, radius=2, nbits=2048):
    """Compute Morgan fingerprints for a list of SMILES strings.
    Returns ndarray of shape (n, nbits). Invalid SMILES get zero vectors.
    """
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps[i] = np.array(fp)
    return fps


def get_loto_folds(path='data/loto_folds.json'):
    """Load LOTO fold assignments. Returns dict mapping UniProt ID -> list of indices."""
    with open(path) as f:
        folds = json.load(f)
    return folds


def get_eligible_targets(df, min_n=10, min_rate=0.1, max_rate=0.9):
    """Filter targets with enough samples and non-trivial class balance."""
    eligible = []
    for tgt, grp in df.groupby('target_uniprot'):
        n = len(grp)
        rate = grp['label'].mean()
        if n >= min_n and rate >= min_rate and rate <= max_rate:
            eligible.append(tgt)
    return sorted(eligible)


def load_admet(path='data/admet_scores.csv'):
    """Load precomputed ADMET scores. Returns ndarray of shape (n, 7)."""
    df = pd.read_csv(path)
    return df.values.astype(np.float32)
