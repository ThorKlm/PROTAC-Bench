#!/usr/bin/env python
"""D-MPNN hyperparameter optimization for PROTAC degradation prediction.

3-stage HPO:
  1. Screen all configs on 5 representative targets
  2. Refine top-5 configs on 15 targets
  3. Validate best config with 3-seed full LOTO

Expected result: best tuned ~0.600 (tuning actually hurts vs default).
This is an OPTIONAL/SLOW script -- full run takes ~12h on a single GPU.

TODO: could parallelize stage 1 across GPUs
"""

import sys, os, json, argparse, itertools, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_dataset, get_eligible_targets

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# reuse graph construction from gnn_baselines
try:
    from baselines.gnn_baselines import mol_to_graph, DMPNN, train_gnn, gnn_loto
    HAS_GNN = True
except ImportError:
    HAS_GNN = False

from sklearn.metrics import roc_auc_score

# HPO grid
HIDDEN_SIZES = [128, 256, 512]
DEPTHS = [2, 3, 4]
DROPOUTS = [0, 0.1, 0.2]

# representative targets for screening (high sample count, diverse)
N_SCREEN_TARGETS = 5
N_REFINE_TARGETS = 15


def get_representative_targets(df, eligible, n=5):
    """Pick targets with most samples from eligible set."""
    counts = df[df['target_uniprot'].isin(eligible)].groupby('target_uniprot').size()
    return counts.nlargest(n).index.tolist()


def make_dmpnn_cls(atom_dim, bond_dim, hidden, depth, dropout):
    """Factory that returns a DMPNN constructor with given hparams."""
    # NOTE: we wrap DMPNN to inject dropout -- a bit hacky but works
    class DmpnnWithDropout(DMPNN):
        def __init__(self):
            super().__init__(atom_dim, bond_dim, hidden=hidden, depth=depth)
            self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        def forward(self, data):
            x, edge_index, edge_attr, batch = (
                data.x, data.edge_index, data.edge_attr, data.batch)
            from torch_geometric.nn import global_mean_pool

            src, dst = edge_index
            h = F.relu(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))
            h = self.drop(h)

            for _ in range(self.depth - 1):
                msg = torch.zeros(x.size(0), h.size(1), device=x.device)
                msg.index_add_(0, dst, h)
                h_new = msg[src]
                h = F.relu(self.W_h(h_new))
                h = self.drop(h)

            atom_msg = torch.zeros(x.size(0), h.size(1), device=x.device)
            atom_msg.index_add_(0, dst, h)
            atom_h = F.relu(self.W_o(torch.cat([x, atom_msg], dim=-1)))

            out = global_mean_pool(atom_h, batch)
            return self.fc(out).squeeze(-1)

    return DmpnnWithDropout


def evaluate_config(model_cls, graphs, y, targets, eval_targets, device, seed=42):
    """Quick LOTO on a subset of targets. Returns mean AUROC."""
    from baselines.gnn_baselines import gnn_loto
    # single seed for screening
    res = gnn_loto(model_cls, graphs, y, targets, set(eval_targets), device,
                   model_kwargs={}, seeds=[seed])
    return res['mean_auroc']


def main():
    if not HAS_TORCH:
        print('ERROR: PyTorch not installed. Exiting.')
        sys.exit(1)
    if not HAS_GNN:
        print('ERROR: Could not import from gnn_baselines. Run from repo root.')
        sys.exit(1)

    parser = argparse.ArgumentParser(description='D-MPNN HPO (slow, optional)')
    parser.add_argument('--output', default='results/chemprop_hpo.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--quick', action='store_true',
                        help='only run stage 1 (screening)')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}')

    print('Loading data...')
    df = load_dataset(args.data)
    y = df['label'].values
    targets = df['target_uniprot'].values
    eligible = get_eligible_targets(df)

    print('Building graphs...')
    graphs = [mol_to_graph(smi, y=lab) for smi, lab in zip(df['smiles'], y)]
    sample_g = next(g for g in graphs if g is not None)
    atom_dim = sample_g.x.size(1)
    bond_dim = sample_g.edge_attr.size(1) if sample_g.edge_attr.size(0) > 0 else 6

    # generate all configs
    configs = list(itertools.product(HIDDEN_SIZES, DEPTHS, DROPOUTS))
    print(f'Total configs: {len(configs)}')

    # Stage 1: screen on representative targets
    screen_targets = get_representative_targets(df, eligible, N_SCREEN_TARGETS)
    print(f'\n=== Stage 1: Screening on {len(screen_targets)} targets ===')

    stage1_results = []
    for i, (hidden, depth, dropout) in enumerate(configs):
        print(f'  [{i+1}/{len(configs)}] hidden={hidden} depth={depth} dropout={dropout}', end='')
        t0 = time.time()
        model_cls = make_dmpnn_cls(atom_dim, bond_dim, hidden, depth, dropout)
        auc = evaluate_config(model_cls, graphs, y, targets, screen_targets, device)
        elapsed = time.time() - t0
        print(f'  AUROC={auc:.3f} ({elapsed:.0f}s)')
        stage1_results.append({
            'hidden': hidden, 'depth': depth, 'dropout': dropout,
            'stage1_auroc': auc,
        })

    stage1_results.sort(key=lambda r: r['stage1_auroc'], reverse=True)
    top5 = stage1_results[:5]
    print(f'\nTop-5 configs after screening:')
    for r in top5:
        print(f'  h={r["hidden"]} d={r["depth"]} drop={r["dropout"]}: {r["stage1_auroc"]:.3f}')

    output = {'stage1': stage1_results, 'top5_configs': top5}

    if not args.quick:
        # Stage 2: refine on more targets
        refine_targets = get_representative_targets(df, eligible, N_REFINE_TARGETS)
        print(f'\n=== Stage 2: Refining top-5 on {len(refine_targets)} targets ===')

        for r in top5:
            model_cls = make_dmpnn_cls(atom_dim, bond_dim,
                                        r['hidden'], r['depth'], r['dropout'])
            auc = evaluate_config(model_cls, graphs, y, targets, refine_targets, device)
            r['stage2_auroc'] = auc
            print(f'  h={r["hidden"]} d={r["depth"]} drop={r["dropout"]}: {auc:.3f}')

        best = max(top5, key=lambda r: r.get('stage2_auroc', 0))
        print(f'\nBest config: hidden={best["hidden"]} depth={best["depth"]} '
              f'dropout={best["dropout"]}')
        output['best_config'] = best

        # Stage 3: full LOTO with 3 seeds
        print(f'\n=== Stage 3: Full LOTO with best config (3 seeds) ===')
        model_cls = make_dmpnn_cls(atom_dim, bond_dim,
                                    best['hidden'], best['depth'], best['dropout'])
        res = gnn_loto(model_cls, graphs, y, targets, set(eligible), device,
                       model_kwargs={}, seeds=[42, 43, 44])
        print(f'Best tuned D-MPNN LOTO: {res["mean_auroc"]:.3f} +/- {res["std_auroc"]:.3f}')
        # NOTE: tuning typically hurts -- overfits to screening targets
        output['final_loto'] = {
            'mean_auroc': res['mean_auroc'],
            'std_auroc': res['std_auroc'],
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
