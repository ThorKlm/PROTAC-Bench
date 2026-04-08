#!/usr/bin/env python
"""fANOVA importance analysis on the Optuna HPO study.

Runs fANOVA on random-phase trials (pre-TPE, unbiased) and on all trials
for comparison. Prints importance table, ASCII bar chart, best/worst
config comparison, and marginal plots for top-3 parameters.

Expected: head_type 95.9%, mol_encoder 1.9%, prot_encoder 0.7%.
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _ascii_bar(value, max_val, width=40):
    """Render a proportional ASCII bar."""
    filled = int(round(value / max(max_val, 1e-9) * width))
    return '#' * filled + '.' * (width - filled)


def _print_importance_table(importances, title):
    """Pretty-print parameter importance table with ASCII bars."""
    print(f'\n{"=" * 60}')
    print(f' {title}')
    print(f'{"=" * 60}')
    max_imp = max(importances.values()) if importances else 0
    print(f'{"Parameter":>25} {"Importance":>10}  Bar')
    print('-' * 60)
    for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = _ascii_bar(imp, max_imp)
        print(f'{param:>25} {imp*100:9.1f}%  |{bar}|')
    print()


def _print_marginals(study, top_params, trials):
    """Print ASCII marginal plots for top parameters."""
    print('\n' + '=' * 60)
    print(' MARGINAL ANALYSIS (top-3 parameters)')
    print('=' * 60)

    for param_name in top_params[:3]:
        # get param values and corresponding objectives
        vals, objs = [], []
        for t in trials:
            if param_name in t.params:
                vals.append(str(t.params[param_name]))
                objs.append(t.value)

        if not vals:
            continue

        # group by value
        groups = {}
        for v, o in zip(vals, objs):
            groups.setdefault(v, []).append(o)

        print(f'\n  {param_name}:')
        print(f'  {"Value":>20} {"Mean":>7} {"Std":>6} {"N":>4}  Distribution')
        print(f'  {"-"*55}')

        for val in sorted(groups.keys()):
            aurocs = groups[val]
            m, s = np.mean(aurocs), np.std(aurocs)
            # mini ascii histogram
            bar_len = int(round(m * 40))  # scale to ~0.7 * 40 = 28
            bar = '#' * bar_len
            print(f'  {val:>20} {m:7.4f} {s:6.4f} {len(aurocs):4d}  |{bar}')


def _print_best_worst(study, trials):
    """Compare best vs worst configs."""
    sorted_t = sorted(trials, key=lambda t: t.value, reverse=True)
    best = sorted_t[0]
    worst = sorted_t[-1]

    print('\n' + '=' * 60)
    print(' BEST vs WORST CONFIG')
    print('=' * 60)
    print(f'  {"":>20} {"BEST":>15} {"WORST":>15}')
    print(f'  {"-"*50}')
    print(f'  {"AUROC":>20} {best.value:15.4f} {worst.value:15.4f}')

    all_params = set(best.params.keys()) | set(worst.params.keys())
    for p in sorted(all_params):
        bv = str(best.params.get(p, '---'))
        wv = str(worst.params.get(p, '---'))
        marker = ' <--' if bv != wv else ''
        print(f'  {p:>20} {bv:>15} {wv:>15}{marker}')
    print()


def main():
    parser = argparse.ArgumentParser(
        description='fANOVA importance analysis on HPO study')
    parser.add_argument('--study-path', default='results/hpo_study.db')
    parser.add_argument('--study-name', default='protac_hpo')
    parser.add_argument('--n-random', type=int, default=200,
                        help='number of initial random trials for unbiased fANOVA')
    args = parser.parse_args()

    storage = f'sqlite:///{args.study_path}'
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    print(f'Loaded study: {len(completed)} completed trials')

    # --- fANOVA on random-phase trials only (unbiased) ---
    random_trials = sorted(completed, key=lambda t: t.number)[:args.n_random]
    n_random_actual = len(random_trials)
    print(f'Using first {n_random_actual} trials as random phase for unbiased fANOVA')

    if n_random_actual < 10:
        print('WARNING: too few trials for reliable fANOVA, need at least 10')
        return

    # create a pruned copy of the study with only random-phase trials
    # (optuna fANOVA works on the full study, so we filter by trial number)
    evaluator = FanovaImportanceEvaluator(seed=42)

    # run on random phase only -- create a sub-study
    random_study = optuna.create_study(direction='maximize')
    for t in random_trials:
        random_study.add_trial(t)

    importances_random = get_param_importances(
        random_study, evaluator=evaluator)
    _print_importance_table(importances_random,
                            f'fANOVA (random phase, n={n_random_actual})')

    # --- fANOVA on ALL trials for comparison ---
    if len(completed) > n_random_actual:
        importances_all = get_param_importances(study, evaluator=evaluator)
        _print_importance_table(importances_all,
                                f'fANOVA (all trials, n={len(completed)})')

        # compare
        print('Note: importance on all trials is biased by TPE sampling.')
        print('Random-phase estimates are more reliable for interpretation.\n')
    else:
        importances_all = importances_random

    # --- marginal analysis ---
    top_params = list(sorted(importances_random.keys(),
                             key=lambda k: -importances_random[k]))
    _print_marginals(study, top_params, completed)

    # --- best vs worst ---
    _print_best_worst(study, completed)

    # --- summary ---
    print('=' * 60)
    print(' SUMMARY')
    print('=' * 60)
    print(f'  Dominant factor: head_type '
          f'({importances_random.get("head_type", 0)*100:.1f}%)')
    print(f'  Mol encoder matters little '
          f'({importances_random.get("mol_encoder", 0)*100:.1f}%)')
    print(f'  Protein encoder irrelevant '
          f'({importances_random.get("prot_encoder", 0)*100:.1f}%)')
    print()
    print('  Interpretation: the LOTO ceiling is driven by the head type')
    print('  (RF vs XGBoost), not by molecular representation. Swapping')
    print('  fingerprints or adding E3 features barely moves the needle.')
    # TODO: add latex table export for paper

    # save importances to json
    os.makedirs('results', exist_ok=True)
    out = {
        'random_phase': {k: float(v) for k, v in importances_random.items()},
        'all_trials': {k: float(v) for k, v in importances_all.items()},
        'n_random': n_random_actual,
        'n_total': len(completed),
    }
    out_path = 'results/fanova_importances.json'
    import json
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nImportances saved to {out_path}')


if __name__ == '__main__':
    main()
