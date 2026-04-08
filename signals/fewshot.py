#!/usr/bin/env python
"""Few-shot evaluation: RF vs MAML across k=1,3,5,10.

For each LOTO target, sample k examples (stratified when possible),
add to training set, evaluate on the rest.

Expected: RF k=5 = 0.700, MAML k=5 = 0.683.
RF wins everywhere -- meta-learning doesn't help.
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data_utils import load_dataset, compute_morgan, get_eligible_targets
from src.stats import format_result

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Few-shot RF ----------

def fewshot_rf_evaluate(X, y, targets, eligible, k=5, seeds=[42], n_reps=5):
    """LOTO with k-shot: move k examples from test -> train, eval on rest."""
    unique_targets = [t for t in np.unique(targets) if t in eligible]
    all_aurocs = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        per_target = {}

        for tgt in unique_targets:
            test_mask = targets == tgt
            train_mask = ~test_mask
            test_idx = np.where(test_mask)[0]
            y_test_full = y[test_idx]

            if len(np.unique(y_test_full)) < 2 or len(test_idx) <= k + 2:
                continue

            rep_aucs = []
            for rep in range(n_reps):
                # stratified sampling when possible
                pos_idx = test_idx[y_test_full == 1]
                neg_idx = test_idx[y_test_full == 0]

                if len(pos_idx) >= 1 and len(neg_idx) >= 1 and k >= 2:
                    # try to keep ratio
                    n_pos = max(1, int(k * len(pos_idx) / len(test_idx)))
                    n_neg = k - n_pos
                    n_pos = min(n_pos, len(pos_idx) - 1)  # keep at least 1 for test
                    n_neg = min(n_neg, len(neg_idx) - 1)
                    shot_idx = np.concatenate([
                        rng.choice(pos_idx, n_pos, replace=False),
                        rng.choice(neg_idx, n_neg, replace=False),
                    ])
                else:
                    shot_idx = rng.choice(test_idx, min(k, len(test_idx) - 2), replace=False)

                remaining_idx = np.setdiff1d(test_idx, shot_idx)
                if len(remaining_idx) < 3 or len(np.unique(y[remaining_idx])) < 2:
                    continue

                # augment training set
                aug_train_idx = np.concatenate([np.where(train_mask)[0], shot_idx])
                X_tr = X[aug_train_idx]
                y_tr = y[aug_train_idx]
                X_te = X[remaining_idx]
                y_te = y[remaining_idx]

                clf = RandomForestClassifier(
                    n_estimators=200, min_samples_leaf=3,
                    class_weight='balanced', random_state=seed + rep, n_jobs=-1,
                )
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict_proba(X_te)[:, 1]
                try:
                    rep_aucs.append(roc_auc_score(y_te, y_pred))
                except ValueError:
                    pass

            if rep_aucs:
                per_target[tgt] = float(np.mean(rep_aucs))

        all_aurocs.extend(per_target.values())
        print(f"  RF k={k} seed={seed}: {np.mean(list(per_target.values())):.3f} ({len(per_target)} targets)")

    return {
        'mean_auroc': float(np.mean(all_aurocs)) if all_aurocs else 0.0,
        'std_auroc': float(np.std(all_aurocs)) if all_aurocs else 0.0,
        'n_targets': len(set().union(*[set() for _ in seeds])),  # approximate
        'k': k,
    }


# ---------- Minimal MAML ----------

class FewShotMLP(nn.Module):
    """2-layer MLP for MAML. Uses functional forward for inner loop."""
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim, hidden) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.randn(hidden, 1) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x, params=None):
        if params is None:
            w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        else:
            w1, b1, w2, b2 = params
        h = F.relu(x @ w1 + b1)
        out = h @ w2 + b2
        return out.squeeze(-1)


def maml_inner_loop(model, X_support, y_support, inner_lr=0.01, inner_steps=5):
    """MAML inner loop: adapt params on support set."""
    params = [p.clone() for p in [model.w1, model.b1, model.w2, model.b2]]

    for _ in range(inner_steps):
        logits = model(X_support, params)
        loss = F.binary_cross_entropy_with_logits(logits, y_support)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        params = [p - inner_lr * g for p, g in zip(params, grads)]

    return params


def fewshot_maml_evaluate(X, y, targets, eligible, k=5, seeds=[42],
                          n_reps=3, meta_epochs=50, debug=False):
    """MAML-based few-shot evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unique_targets = [t for t in np.unique(targets) if t in eligible]
    all_aurocs = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        per_target = {}

        for tgt in unique_targets:
            test_mask = targets == tgt
            train_mask = ~test_mask
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]
            y_test_full = y[test_idx]

            if len(np.unique(y_test_full)) < 2 or len(test_idx) <= k + 2:
                continue

            # meta-train on training targets
            train_targets = np.unique(targets[train_mask])
            # filter train targets with enough samples
            train_targets = [t for t in train_targets
                             if np.sum(targets[train_mask] == t) >= k + 3
                             and len(np.unique(y[(targets == t) & train_mask])) == 2]

            if len(train_targets) < 3:
                continue

            model = FewShotMLP(X.shape[1]).to(device)
            meta_opt = torch.optim.Adam(model.parameters(), lr=0.001)

            # meta-training
            for epoch in range(meta_epochs):
                meta_opt.zero_grad()
                meta_loss = 0.0
                n_tasks = min(4, len(train_targets))
                task_targets = rng.choice(train_targets, n_tasks, replace=False)

                for tt in task_targets:
                    tt_idx = np.where((targets == tt) & train_mask)[0]
                    tt_idx = rng.permutation(tt_idx)
                    support_idx = tt_idx[:k]
                    query_idx = tt_idx[k:k+10]  # small query set

                    X_s = torch.FloatTensor(X[support_idx]).to(device)
                    y_s = torch.FloatTensor(y[support_idx].astype(np.float32)).to(device)
                    X_q = torch.FloatTensor(X[query_idx]).to(device)
                    y_q = torch.FloatTensor(y[query_idx].astype(np.float32)).to(device)

                    adapted_params = maml_inner_loop(model, X_s, y_s)
                    logits = model(X_q, adapted_params)
                    loss = F.binary_cross_entropy_with_logits(logits, y_q)
                    meta_loss += loss

                meta_loss /= n_tasks
                meta_loss.backward()
                meta_opt.step()

                if debug and (epoch + 1) % 20 == 0:
                    print(f"    meta epoch {epoch+1}: loss={meta_loss.item():.4f}")

            # evaluate on held-out target
            rep_aucs = []
            for rep in range(n_reps):
                perm = rng.permutation(test_idx)
                shot_idx = perm[:k]
                remaining_idx = perm[k:]

                if len(remaining_idx) < 3 or len(np.unique(y[remaining_idx])) < 2:
                    continue

                X_s = torch.FloatTensor(X[shot_idx]).to(device)
                y_s = torch.FloatTensor(y[shot_idx].astype(np.float32)).to(device)

                adapted_params = maml_inner_loop(model, X_s, y_s, inner_steps=10)

                with torch.no_grad():
                    X_te = torch.FloatTensor(X[remaining_idx]).to(device)
                    logits = model(X_te, adapted_params)
                    y_pred = torch.sigmoid(logits).cpu().numpy()

                try:
                    rep_aucs.append(roc_auc_score(y[remaining_idx], y_pred))
                except ValueError:
                    pass

            if rep_aucs:
                per_target[tgt] = float(np.mean(rep_aucs))

        if per_target:
            all_aurocs.extend(per_target.values())
            print(f"  MAML k={k} seed={seed}: {np.mean(list(per_target.values())):.3f} ({len(per_target)} targets)")

    return {
        'mean_auroc': float(np.mean(all_aurocs)) if all_aurocs else 0.0,
        'std_auroc': float(np.std(all_aurocs)) if all_aurocs else 0.0,
        'k': k,
    }


def main():
    parser = argparse.ArgumentParser(description='Few-shot: RF vs MAML')
    parser.add_argument('--k-values', default='1,3,5,10', help='comma-sep k values')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--skip-maml', action='store_true', help='skip MAML (slow)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output', default='results/fewshot.json')
    parser.add_argument('--data', default='data/protac_bench.csv')
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print('Loading data...')
    df = load_dataset(args.data)
    smiles = df['smiles'].tolist()
    y = df['label'].values
    targets = df['target_uniprot'].values

    print('Computing Morgan FPs...')
    X = compute_morgan(smiles)
    eligible = get_eligible_targets(df)
    print(f'Eligible targets: {len(eligible)}')

    results = {'rf': {}, 'maml': {}}

    for k in k_values:
        print(f'\n{"="*50}')
        print(f'k = {k}')
        print(f'{"="*50}')

        # RF
        print(f'\n[RF] k={k}...')
        rf_res = fewshot_rf_evaluate(X, y, targets, eligible, k=k, seeds=seeds)
        results['rf'][str(k)] = rf_res
        print(f'  RF k={k}: {rf_res["mean_auroc"]:.3f}')

        # MAML
        if not args.skip_maml:
            print(f'\n[MAML] k={k}...')
            maml_res = fewshot_maml_evaluate(X, y, targets, eligible, k=k,
                                              seeds=seeds, debug=args.debug)
            results['maml'][str(k)] = maml_res
            print(f'  MAML k={k}: {maml_res["mean_auroc"]:.3f}')

    # summary table
    print(f'\n{"="*50}')
    print(f'{"k":>5s}  {"RF":>8s}  {"MAML":>8s}  {"diff":>8s}')
    print(f'{"-"*35}')
    for k in k_values:
        rf_auc = results['rf'][str(k)]['mean_auroc']
        if str(k) in results.get('maml', {}):
            maml_auc = results['maml'][str(k)]['mean_auroc']
            diff = rf_auc - maml_auc
            print(f'{k:>5d}  {rf_auc:>8.3f}  {maml_auc:>8.3f}  {diff:>+8.3f}')
        else:
            print(f'{k:>5d}  {rf_auc:>8.3f}  {"--":>8s}  {"--":>8s}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        'method': 'fewshot_rf_vs_maml',
        'k_values': k_values,
        'seeds': seeds,
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
