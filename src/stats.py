"""Statistical tests and formatting utilities."""

import numpy as np
from scipy import stats


def paired_wilcoxon(aurocs_a, aurocs_b):
    """Two-sided Wilcoxon signed-rank test on paired AUROC values."""
    aurocs_a = np.array(aurocs_a)
    aurocs_b = np.array(aurocs_b)
    diff = aurocs_a - aurocs_b
    # drop zeros (ties)
    diff = diff[diff != 0]
    if len(diff) < 5:
        return 1.0
    _, p = stats.wilcoxon(diff)
    return float(p)


def confidence_interval(values, confidence=0.95):
    """Compute mean and bootstrap-free CI using t-distribution."""
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, mean, mean
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean), float(mean - h), float(mean + h)


def format_result(mean, std, n_seeds):
    """Format as '0.668 +/- 0.005 (n=10)'."""
    return f'{mean:.3f} +/- {std:.3f} (n={n_seeds})'
