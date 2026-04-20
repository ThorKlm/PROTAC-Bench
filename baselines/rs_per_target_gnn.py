#!/usr/bin/env python
"""Random 5-fold per-target AUROC for GIN and D-MPNN on protac_bench.csv.

3 seeds (42, 43, 44); for each target, AUROC computed on that target's subset
of the test fold, then averaged across folds x seeds.
"""
import sys, os, json, time, argparse
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from gnn_baselines import GIN, DMPNN, mol_to_graph

SEEDS = [42, 43, 44]
N_FOLDS = 5
EPOCHS = 30
BATCH = 128
LR = 1e-3
DATA = '/workspace/PROTAC-Bench/data/protac_bench.csv'
OUTDIR = '/workspace/PROTAC-Bench/results/per_target'


def train_one(model, train_loader, device, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.binary_cross_entropy_with_logits(out, batch.y)
            loss.backward()
            optimizer.step()
    return model


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = torch.sigmoid(model(batch))
            preds.extend(out.cpu().numpy().tolist())
    return np.array(preds)


def run_method(method_name, model_cls, model_kwargs, graphs_valid, valid_idx,
               labels_full, targets_full, device):
    """Run 5-fold x 3-seed random CV.

    graphs_valid: list of valid Data objects (aligned to valid_idx subset).
    valid_idx: ndarray of original row indices corresponding to graphs_valid.
    labels_full, targets_full: full arrays indexed by original row indices.
    """
    y_valid = labels_full[valid_idx].astype(int)
    t_valid = targets_full[valid_idx]
    n = len(graphs_valid)

    # per-target fold AUROCs list
    per_target_fold_aurocs = defaultdict(list)

    t0 = time.time()
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr, te) in enumerate(skf.split(np.arange(n), y_valid)):
            train_graphs = [graphs_valid[i] for i in tr]
            test_graphs = [graphs_valid[i] for i in te]

            train_loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)
            test_loader = DataLoader(test_graphs, batch_size=BATCH, shuffle=False)

            model = model_cls(**model_kwargs).to(device)
            train_one(model, train_loader, device)
            preds = predict(model, test_loader, device)

            # per target in test fold
            t_te = t_valid[te]
            y_te = y_valid[te]
            for tgt in np.unique(t_te):
                mask = t_te == tgt
                if mask.sum() < 2:
                    continue
                if len(np.unique(y_te[mask])) < 2:
                    continue
                try:
                    auc = roc_auc_score(y_te[mask], preds[mask])
                except ValueError:
                    continue
                per_target_fold_aurocs[tgt].append(auc)

            elapsed = time.time() - t0
            print(f'  [{method_name}] seed={seed} fold={fold+1}/{N_FOLDS} '
                  f'n_train={len(tr)} n_test={len(te)} '
                  f'targets_with_auroc={sum(1 for v in per_target_fold_aurocs.values() if len(v)>0)} '
                  f'elapsed={elapsed:.0f}s', flush=True)

    # aggregate
    rows = []
    for tgt, aucs in per_target_fold_aurocs.items():
        n_entries = int((targets_full == tgt).sum())
        rows.append({'target': tgt,
                     'random_auroc': round(float(np.mean(aucs)), 4),
                     'n_entries': n_entries,
                     'n_fold_evals': len(aucs)})

    out_df = pd.DataFrame(rows).sort_values('target').reset_index(drop=True)
    return out_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['GIN', 'DMPNN', 'both'], default='both')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    print('Loading data...', flush=True)
    df = pd.read_csv(DATA)
    smiles_col = df['smiles'].values
    labels_full = df['label'].values
    targets_full = df['target_uniprot'].values
    print(f'N = {len(df)}', flush=True)

    print('Converting SMILES to graphs...', flush=True)
    graphs, valid_idx = [], []
    for i, (smi, lab) in enumerate(zip(smiles_col, labels_full)):
        g = mol_to_graph(smi, y=float(lab))
        if g is not None:
            graphs.append(g)
            valid_idx.append(i)
    valid_idx = np.array(valid_idx)
    print(f'Valid graphs: {len(graphs)}/{len(df)}', flush=True)

    atom_dim = graphs[0].x.size(1)
    bond_dim = graphs[0].edge_attr.size(1) if graphs[0].edge_attr.size(0) > 0 else 6

    os.makedirs(OUTDIR, exist_ok=True)

    if args.method in ('GIN', 'both'):
        print('\n=== GIN ===', flush=True)
        df_out = run_method('GIN', GIN,
                            {'in_dim': atom_dim, 'hidden': 64},
                            graphs, valid_idx, labels_full, targets_full, device)
        out_path = os.path.join(OUTDIR, 'GIN_random_per_target.csv')
        df_out.to_csv(out_path, index=False)
        print(f'Saved {out_path}: {len(df_out)} targets, mean random_auroc={df_out.random_auroc.mean():.4f}', flush=True)

    if args.method in ('DMPNN', 'both'):
        print('\n=== D-MPNN ===', flush=True)
        df_out = run_method('D-MPNN', DMPNN,
                            {'atom_dim': atom_dim, 'bond_dim': bond_dim,
                             'hidden': 64, 'depth': 3},
                            graphs, valid_idx, labels_full, targets_full, device)
        out_path = os.path.join(OUTDIR, 'DMPNN_random_per_target.csv')
        df_out.to_csv(out_path, index=False)
        print(f'Saved {out_path}: {len(df_out)} targets, mean random_auroc={df_out.random_auroc.mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
