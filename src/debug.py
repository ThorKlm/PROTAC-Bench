"""Debug-mode helper for replication scripts.

Reduces seeds, target cohort, and HPO trial count when --debug is passed,
allowing a smoke test in minutes rather than the full multi-hour run.
"""

def reduce_for_debug(seeds=None, eligible=None, n_trials=None, debug=False):
    """Shrink the workload to 2 seeds, 5 targets, 50 trials when debug=True."""
    if not debug:
        return seeds, eligible, n_trials
    if seeds is not None:
        seeds = list(seeds)[:2]
    if eligible is not None:
        eligible = list(eligible)[:5]
    if n_trials is not None:
        n_trials = min(n_trials, 50)
    n_tgt = len(eligible) if eligible is not None else 'n/a'
    print(f'[debug] running with seeds={seeds}, targets={n_tgt}, n_trials={n_trials}')
    return seeds, eligible, n_trials
