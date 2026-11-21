"""Replace the flattened seed-target paired test in admet_cascade.py with one
pair per target averaged over seeds, and record the test's n in the artefact.
Self-verifying: refuses to write unless each anchor matches exactly once."""
import difflib, shutil, sys
from pathlib import Path
P = Path('signals/admet_cascade.py')
EDITS = [(
"""    # stats
    a_list, b_list = [], []
    for s in [str(s) for s in seeds]:
        shared = set(c0['per_seed'].get(s, {}).keys()) & set(c1['per_seed'].get(s, {}).keys())
        for tgt in sorted(shared):
            b_list.append(c0['per_seed'][s][tgt])
            a_list.append(c1['per_seed'][s][tgt])
    p_val = paired_wilcoxon(a_list, b_list)
""",
"""    # stats: one pair per target, averaged over seeds, so the paired test runs
    # over independent units rather than over seed-target cells
    ss = [str(s) for s in seeds]
    sets = [set(c0['per_seed'].get(s, {})) & set(c1['per_seed'].get(s, {})) for s in ss]
    shared = sorted(set.intersection(*sets)) if sets else []
    b_list = [sum(c0['per_seed'][s][t] for s in ss) / len(ss) for t in shared]
    a_list = [sum(c1['per_seed'][s][t] for s in ss) / len(ss) for t in shared]
    p_val = paired_wilcoxon(a_list, b_list)
"""), (
"""        'baseline': c0, 'admet_cascade': c1,
        'delta': float(delta), 'p_value': float(p_val),
""",
"""        'baseline': c0, 'admet_cascade': c1,
        'delta': float(delta), 'p_value': float(p_val),
        'n_paired_targets': len(shared),
        'paired_test': 'Wilcoxon over per-target means across seeds',
""")]
src = P.read_text()
for i, (old, new) in enumerate(EDITS, 1):
    n = src.count(old)
    if n != 1:
        sys.exit(f'anchor {i} matched {n} times, expected 1; nothing written')
    src = src.replace(old, new)
shutil.copy(P, str(P) + '.bak')
P.write_text(src)
print(''.join(difflib.unified_diff(
    Path(str(P) + '.bak').read_text().splitlines(keepends=True),
    src.splitlines(keepends=True), fromfile=str(P), tofile=str(P) + ' (patched)')))
print('patched; backup at', str(P) + '.bak')
