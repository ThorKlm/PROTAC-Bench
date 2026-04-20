#!/usr/bin/env python
"""Random 5-fold per-target AUROC for simplified PROTAC-STAN (TernaryAttentionNet).

Follows the architecture in /workspace/scripts/exp38_stan_loto.py:
  Morgan-2048 + POI ESM2-650M + E3 one-hot -> cross-attention -> MLP head.

Uses protac_bench.csv labels/targets (aligned with exp1_merged_dataset.csv),
but takes e3_name from the merged dataset (protac_bench collapses to VHL/CRBN/Other).
"""
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = '/workspace/PROTAC-Bench/data/protac_bench.csv'
MERGED = '/workspace/results/ext30_data_expansion/exp1_merged_dataset.csv'
SMILES_LIST = '/workspace/cache/smiles_list.npy'
MORGAN_FP = '/workspace/cache/mol_morgan_2048_protacs.npy'
TARGET_IDS = '/workspace/cache/target_ids.npy'
ESM_FP = '/workspace/cache/prot_esm2_650M_targets.npy'
OUT = '/workspace/PROTAC-Bench/results/per_target/STAN_random_per_target.csv'

SEEDS = [42, 43, 44]
N_FOLDS = 5
EPOCHS = 30
BATCH = 128
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TernaryAttentionNet(nn.Module):
    def __init__(self, morgan_dim=2048, poi_dim=1280, e3_dim=12, hidden=256):
        super().__init__()
        self.protac_branch = nn.Sequential(nn.Linear(morgan_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.poi_branch = nn.Sequential(nn.Linear(poi_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.e3_branch = nn.Sequential(nn.Linear(e3_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.attn_q = nn.Linear(hidden, hidden)
        self.attn_k = nn.Linear(hidden, hidden)
        self.attn_v = nn.Linear(hidden, hidden)
        self.scale = hidden ** 0.5
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1))

    def forward(self, morgan, poi, e3):
        p = self.protac_branch(morgan)
        poi_h = self.poi_branch(poi)
        e3_h = self.e3_branch(e3)
        kv = torch.stack([poi_h, e3_h], dim=1)
        Q = self.attn_q(p).unsqueeze(1)
        K = self.attn_k(kv)
        V = self.attn_v(kv)
        w = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        att = torch.bmm(w, V).squeeze(1)
        return self.head(torch.cat([p, att], dim=1)).squeeze(1)


def main():
    print('Loading data...', flush=True)
    df = pd.read_csv(DATA)
    dfm = pd.read_csv(MERGED)
    assert (df['smiles'].values == dfm['canonical_smiles'].values).all()
    assert (df['target_uniprot'].values == dfm['target_uniprot'].values).all()
    assert (df['label'].values == dfm['label'].values).all()

    smiles_list = np.load(SMILES_LIST, allow_pickle=True)
    morgan_all = np.load(MORGAN_FP)
    target_ids = np.load(TARGET_IDS, allow_pickle=True)
    esm_all = np.load(ESM_FP)
    smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}
    target_to_idx = {t: i for i, t in enumerate(target_ids)}

    e3_canon = {'VHL': 'VHL', 'CRBN': 'CRBN', 'MDM2': 'MDM2', 'cIAP1': 'cIAP1',
                'XIAP': 'XIAP', 'IAP': 'IAP', 'FEM1B': 'FEM1B', 'Keap1': 'Keap1',
                'KEAP1': 'Keap1', 'DCAF16': 'DCAF16', 'UBR box': 'UBR box',
                'DCAF1': 'DCAF1'}
    e3_types = sorted(set(e3_canon.values()))
    e3_to_idx = {e: i for i, e in enumerate(e3_types)}
    N_E3 = len(e3_types)

    def resolve_target_idx(t):
        if t in target_to_idx:
            return target_to_idx[t]
        for p in t.split('/'):
            if p in target_to_idx:
                return target_to_idx[p]
        return None

    n = len(df)
    # Build features
    print('Building features...', flush=True)
    smi_idx = np.array([smiles_to_idx[s] for s in df['smiles']])
    morgan_feats = morgan_all[smi_idx].astype(np.float32)

    e3_onehot = np.zeros((n, N_E3), dtype=np.float32)
    for i, e3 in enumerate(dfm['e3_name']):
        e3_onehot[i, e3_to_idx[e3_canon[e3]]] = 1.0

    labels = df['label'].values.astype(np.float32)
    targets = df['target_uniprot'].values

    poi_idx = []
    valid_poi = np.ones(n, dtype=bool)
    for i, t in enumerate(targets):
        idx = resolve_target_idx(t)
        if idx is None:
            poi_idx.append(0)
            valid_poi[i] = False
        else:
            poi_idx.append(idx)
    poi_idx = np.array(poi_idx)
    poi_feats = esm_all[poi_idx].astype(np.float32)
    # Mask missing POI to zero
    poi_feats[~valid_poi] = 0.0

    print(f'  morgan: {morgan_feats.shape}, poi: {poi_feats.shape}, e3: {e3_onehot.shape}', flush=True)
    print(f'  targets w/o ESM embedding: {(~valid_poi).sum()} rows', flush=True)

    Xm = torch.tensor(morgan_feats)
    Xp = torch.tensor(poi_feats)
    Xe = torch.tensor(e3_onehot)
    Y = torch.tensor(labels)

    per_target_aurocs = defaultdict(list)
    t0 = time.time()
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(np.arange(n), labels)):
            tr_t = torch.tensor(tr, dtype=torch.long)
            te_t = torch.tensor(te, dtype=torch.long)

            train_ds = TensorDataset(Xm[tr_t], Xp[tr_t], Xe[tr_t], Y[tr_t])
            train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)

            model = TernaryAttentionNet(e3_dim=N_E3).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
            criterion = nn.BCEWithLogitsLoss()

            for epoch in range(EPOCHS):
                model.train()
                for mb, mp, me, my in train_dl:
                    mb, mp, me, my = mb.to(DEVICE), mp.to(DEVICE), me.to(DEVICE), my.to(DEVICE)
                    optimizer.zero_grad()
                    pred = model(mb, mp, me)
                    loss = criterion(pred, my)
                    loss.backward()
                    optimizer.step()

            # predict
            model.eval()
            with torch.no_grad():
                # chunk prediction to keep memory in check
                preds = []
                for start in range(0, len(te), 512):
                    sl = te_t[start:start+512]
                    out = model(Xm[sl].to(DEVICE), Xp[sl].to(DEVICE), Xe[sl].to(DEVICE))
                    preds.append(torch.sigmoid(out).cpu().numpy())
                preds = np.concatenate(preds)

            t_te = targets[te]
            y_te = labels[te].astype(int)
            for tgt in np.unique(t_te):
                mask = t_te == tgt
                if mask.sum() < 2 or len(np.unique(y_te[mask])) < 2:
                    continue
                try:
                    auc = roc_auc_score(y_te[mask], preds[mask])
                except ValueError:
                    continue
                per_target_aurocs[tgt].append(auc)

            elapsed = time.time() - t0
            print(f'  [STAN] seed={seed} fold={fold+1}/{N_FOLDS} '
                  f'targets={len(per_target_aurocs)} elapsed={elapsed:.0f}s', flush=True)

    rows = []
    for tgt, aucs in per_target_aurocs.items():
        rows.append({'target': tgt,
                     'random_auroc': round(float(np.mean(aucs)), 4),
                     'n_entries': int((targets == tgt).sum()),
                     'n_fold_evals': len(aucs)})
    out_df = pd.DataFrame(rows).sort_values('target').reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f'Saved {OUT}: {len(out_df)} targets, mean random_auroc={out_df.random_auroc.mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
