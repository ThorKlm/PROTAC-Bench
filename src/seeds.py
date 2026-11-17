"""Canonical seed list: the only seed declaration in this repository.

Recorded 2026-11-12. The earlier list has no stated derivation, which is
indefensible in a paper whose lead contribution is hyperparameter-selection
bias. CANONICAL is 42, the conventional default, followed by 0 through 8.
Nothing else in the tree may declare a seed set; import from here.
"""
CANONICAL = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
PUBLISHED = [7, 13, 29, 42, 43, 44, 53, 71, 89, 97]  # earlier preprints only

def parse(arg=None, n=None):
    """CLI seed parsing: None or 'canonical' yields CANONICAL; n truncates."""
    s = CANONICAL if arg in (None, "", "canonical") else [int(x) for x in str(arg).split(",")]
    return s[:n] if n is None or n >= len(s) else s[:n]
