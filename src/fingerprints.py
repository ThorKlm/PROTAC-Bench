"""Fingerprint computation: Morgan, MACCS, RDKit descriptors."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


def morgan_fp(smiles, radius=2, nbits=2048):
    """Compute Morgan FP for a single SMILES. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def maccs_fp(smiles):
    """Compute MACCS keys for a single SMILES. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float32)


def bulk_fps(smiles_list, fp_fn, show_progress=True):
    """Compute fingerprints in bulk. Returns (fp_array, valid_mask).

    fp_fn should be one of morgan_fp, maccs_fp, etc.
    """
    results = []
    valid = []
    for i, smi in enumerate(smiles_list):
        if show_progress and (i + 1) % 1000 == 0:
            print(f'  computed {i+1}/{len(smiles_list)} fingerprints')
        fp = fp_fn(smi)
        if fp is not None:
            results.append(fp)
            valid.append(True)
        else:
            results.append(np.zeros_like(results[0]) if results else None)
            valid.append(False)

    # handle edge case where first molecule was invalid
    if results and results[0] is None:
        # find first valid to get shape
        for r in results:
            if r is not None:
                shape = r.shape
                break
        else:
            raise ValueError('No valid molecules found')
        results = [r if r is not None else np.zeros(shape, dtype=np.float32) for r in results]

    return np.stack(results), np.array(valid)
