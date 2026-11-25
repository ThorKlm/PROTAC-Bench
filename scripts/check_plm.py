"""Validate the five-scale panel: no skipped scale, no zero-vector target."""
import json, sys, numpy as np
d = json.load(open('results/plm_scaling.json'))
skipped = {k: v.get('reason') for k, v in d.items() if isinstance(v, dict) and v.get('skipped')}
print('skipped scales:', skipped or 'none')
for s in ('8M', '35M', '150M', '650M', '3B'):
    e = np.load(f'cache/prot_esm2_{s}_targets.npy')
    zero_rows = int((np.abs(e).sum(axis=1) == 0).sum())
    print(f'{s:5s} embeddings {e.shape}  all-zero rows: {zero_rows}')
for k, v in d.items():
    if isinstance(v, dict) and not v.get('skipped'):
        print(f'{k:8s}', {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))})
