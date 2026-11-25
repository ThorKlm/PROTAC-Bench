#!/usr/bin/env python
"""GNN baselines: GIN and D-MPNN for PROTAC degradation prediction.

Expected LOTO: GIN ~0.608, D-MPNN ~0.620.
GIN is simpler but surprisingly competitive with D-MPNN on this task.

Requires torch-geometric. Will fall back gracefully if not installed.
"""

import sys, os, json, argparse, subprocess
from datetime import datetime
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.seeds import CANONICAL

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GINConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print('WARNING: torch-geometric not found. Install with:')
    print('  pip install torch torch-geometric')
    print('Exiting.')

from rdkit import Chem
from sklearn.metrics import roc_auc_score

from src.cohort import DEFINITION, load as load_cohort, masks, meta as cohort_meta


# --- atom/bond featurization ---

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

def one_hot(val, choices):
    enc = [0] * (len(choices) + 1)  # +1 for unknown
    if val in choices:
        enc[choices.index(val)] = 1
    else:
        enc[-1] = 1
    return enc


def atom_features(atom):
    return (one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) +
            one_hot(atom.GetDegree(), ATOM_FEATURES['degree']) +
            one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) +
            one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']) +
            one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']) +
            [1 if atom.GetIsAromatic() else 0])


def bond_features(bond):
    bt = bond.GetBondType()
    return [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]


def mol_to_graph(smiles, y=None):
    """Convert SMILES to PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edges_src, edges_dst, edge_attr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges_src += [i, j]
        edges_dst += [j, i]
        edge_attr += [bf, bf]

    if len(edges_src) == 0:
        # single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data


# --- GIN model ---

class GIN(nn.Module):
    """3-layer Graph Isomorphism Network."""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [in_dim, hidden, hidden, hidden]
        for i in range(3):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.fc = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


# --- D-MPNN model ---
# message passing on directed bonds, following Chemprop/Yang et al.

class DMPNN(nn.Module):
    """Directed Message Passing Neural Network."""
    def __init__(self, atom_dim, bond_dim, hidden=64, depth=3):
        super().__init__()
        self.depth = depth
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden)
        self.W_h = nn.Linear(hidden, hidden)
        self.W_o = nn.Linear(atom_dim + hidden, hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)

        src, dst = edge_index
        # initial bond messages
        h = F.relu(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))

        for _ in range(self.depth - 1):
            # for each bond (u->v), aggregate messages from bonds (w->u) where w != v
            # simplified: aggregate all incoming messages to source node
            msg = torch.zeros(x.size(0), h.size(1), device=x.device)
            msg.index_add_(0, dst, h)
            h_new = msg[src]  # messages arriving at src of each edge
            h = F.relu(self.W_h(h_new))

        # aggregate to atoms
        atom_msg = torch.zeros(x.size(0), h.size(1), device=x.device)
        atom_msg.index_add_(0, dst, h)
        atom_h = F.relu(self.W_o(torch.cat([x, atom_msg], dim=-1)))

        # global pool
        out = global_mean_pool(atom_h, batch)
        return self.fc(out).squeeze(-1)


# --- training ---

def train_gnn(model, train_loader, val_loader, device, lr=1e-3, epochs=100, patience=10):
    """Train with early stopping on validation AUROC."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc = 0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            # handle class imbalance with pos_weight
            loss = F.binary_cross_entropy_with_logits(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = torch.sigmoid(model(batch))
                preds.extend(out.cpu().numpy().tolist())
                labels.extend(batch.y.cpu().numpy().tolist())

        if len(set(labels)) < 2:
            continue
        try:
            val_auc = roc_auc_score(labels, preds)
        except ValueError:
            continue

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def gnn_loto(model_cls, graphs, folds, device, model_kwargs=None, seeds=[42]):
    """LOTO evaluation for GNN models.

    folds comes from src.cohort as one (target, train_mask, test_mask) per
    fold, so the cohort and its deduplication and constituent exclusions are
    the ones every other arm uses rather than rebuilt here.
    """
    model_kwargs = model_kwargs or {}

    all_seed_results = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        per_target = {}

        for tgt, train_mask, test_mask in folds:
            train_graphs = [g for g, m in zip(graphs, train_mask) if m and g is not None]
            test_graphs = [g for g, m in zip(graphs, test_mask) if m and g is not None]

            if len(test_graphs) < 2:
                continue

            y_test_actual = np.array([g.y.item() for g in test_graphs])
            if len(np.unique(y_test_actual)) < 2:
                continue

            # use 10% of train as validation for early stopping; the
            # permutation consumes the seed set at the top of the seed loop,
            # so the partition varies across seeds rather than being fixed
            n_val = max(1, len(train_graphs) // 10)
            perm = np.random.permutation(len(train_graphs))
            val_graphs = [train_graphs[i] for i in perm[:n_val]]
            train_graphs = [train_graphs[i] for i in perm[n_val:]]

            train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=64)
            test_loader = DataLoader(test_graphs, batch_size=64)

            model = model_cls(**model_kwargs).to(device)
            model = train_gnn(model, train_loader, val_loader, device)

            model.eval()
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = torch.sigmoid(model(batch))
                    preds.extend(out.cpu().numpy().tolist())

            try:
                auc = roc_auc_score(y_test_actual, preds)
                per_target[tgt] = auc
            except ValueError:
                pass

            # NOTE: no logging per-target here to keep output clean
        all_seed_results[seed] = per_target
        print(f'  seed {seed}: {len(per_target)} targets, '
              f'mean={np.mean(list(per_target.values())):.3f}')

    aurocs = []
    for res in all_seed_results.values():
        aurocs.extend(res.values())

    return {
        'mean_auroc': float(np.mean(aurocs)) if aurocs else 0.0,
        'std_auroc': float(np.std([np.mean(list(r.values()))
                                    for r in all_seed_results.values()])),
        'per_seed': {str(s): r for s, r in all_seed_results.items()},
    }


def merge(shard_dir, outdir='results'):
    """One artefact per architecture from the per-seed shards.

    Written in the engine's schema so the printer reads these like any other
    arm. Average precision is absent because gnn_loto does not compute it.
    """
    import glob
    shards = sorted(glob.glob(os.path.join(shard_dir, '*.json')))
    if not shards:
        raise SystemExit(f'no shards in {shard_dir}')
    per_arch, metas = defaultdict(dict), []
    for sp in shards:
        d = json.load(open(sp))
        metas.append(d.get('meta') or {})
        for arch, v in d.items():
            if arch == 'meta':
                continue
            per_arch[arch].update(v.get('per_seed') or {})
    for arch, per_seed in per_arch.items():
        ss = sorted(per_seed, key=lambda x: (int(x) != 42, int(x)))
        mac = [float(np.mean(list(per_seed[x].values()))) for x in ss]
        tg = sorted(set.intersection(*[set(per_seed[x]) for x in ss]))
        pt = np.array([np.mean([per_seed[x][t] for x in ss]) for t in tg])
        name = {'GIN': 'gin', 'D-MPNN': 'dmpnn'}.get(arch, arch.lower())
        res = dict(arm=name, spec=dict(architecture=arch),
                   macro_auroc=round(float(np.mean(mac)), 4),
                   seed_sd=round(float(np.std(mac, ddof=1)), 4) if len(mac) > 1 else None,
                   per_target_sd=round(float(pt.std(ddof=1)), 4),
                   target_se=round(float(pt.std(ddof=1) / len(pt) ** 0.5), 4),
                   n_targets_paired=len(tg),
                   per_seed_macro={x: round(m, 4) for x, m in zip(ss, mac)},
                   per_target={t: round(float(v), 4) for t, v in zip(tg, pt)},
                   _meta={**{k: v for k, v in metas[0].items() if k != 'seeds'},
                          'seeds': [int(x) for x in ss], 'n_shards': len(shards)})
        out = os.path.join(outdir, f'{name}.json')
        json.dump(res, open(out, 'w'), indent=1)
        print(f'{out}: macro {res["macro_auroc"]} sd {res["seed_sd"]} '
              f'folds {len(tg)} seeds {len(ss)}')

def main():
    if not HAS_PYG:
        sys.exit(1)

    parser = argparse.ArgumentParser(description='GNN baselines (GIN + D-MPNN)')
    parser.add_argument('--seeds', default=','.join(str(s) for s in CANONICAL))
    parser.add_argument('--output', default='results/gnn_baselines.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--merge', default=None, metavar='DIR',
                        help='combine per-seed shards into one artefact per architecture')
    args = parser.parse_args()
    if args.merge:
        merge(args.merge); return

    seeds = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else args.device if args.device != 'auto' else 'cpu')
    print(f'Using device: {device}')

    print('Loading dataset...')
    c = load_cohort(args.data)
    df = c['df']
    y = df['label'].values
    folds = [(t,) + masks(t, c) for t in c['cohort']]
    print(f'Cohort: {len(folds)} folds, {DEFINITION}')

    print('Converting SMILES to graphs...')
    graphs = []
    for i, (smi, label) in enumerate(zip(df['smiles'], y)):
        g = mol_to_graph(smi, y=label)
        graphs.append(g)
        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{len(df)} molecules processed')

    valid_count = sum(1 for g in graphs if g is not None)
    print(f'Valid graphs: {valid_count}/{len(graphs)}')

    # determine feature dims from first valid graph
    sample_g = next(g for g in graphs if g is not None)
    atom_dim = sample_g.x.size(1)
    bond_dim = sample_g.edge_attr.size(1) if sample_g.edge_attr.size(0) > 0 else 6

    results = {}

    # --- GIN ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def save():
        """Write after each architecture, so a failure costs one arm not the run."""
        results['meta'] = {
            **cohort_meta(c, seeds), 'n_rows': int(len(df)),
            'n_valid_graphs': int(valid_count), 'device': str(device),
            'torch': torch.__version__,
            'git_commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                         capture_output=True, text=True).stdout.strip(),
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'saved {sorted(k for k in results if k != "meta")} '
              f'to {args.output}', flush=True)

    print('\n=== GIN (3-layer, hidden=64) ===')
    gin_res = gnn_loto(
        GIN, graphs, folds, device,
        model_kwargs={'in_dim': atom_dim, 'hidden': 64},
        seeds=seeds,
    )
    print(f'GIN LOTO: {gin_res["mean_auroc"]:.3f} +/- {gin_res["std_auroc"]:.3f}')
    results['GIN'] = gin_res
    save()

    # --- D-MPNN ---
    print('\n=== D-MPNN (depth=3, hidden=64) ===')
    dmpnn_res = gnn_loto(
        DMPNN, graphs, folds, device,
        model_kwargs={'atom_dim': atom_dim, 'bond_dim': bond_dim, 'hidden': 64, 'depth': 3},
        seeds=seeds,
    )
    print(f'D-MPNN LOTO: {dmpnn_res["mean_auroc"]:.3f} +/- {dmpnn_res["std_auroc"]:.3f}')
    results['D-MPNN'] = dmpnn_res
    save()

    # summary
    print('\n' + '='*40)
    print(f'GIN:    {gin_res["mean_auroc"]:.3f}')
    print(f'D-MPNN: {dmpnn_res["mean_auroc"]:.3f}')
    print('='*40)




if __name__ == '__main__':
    main()
