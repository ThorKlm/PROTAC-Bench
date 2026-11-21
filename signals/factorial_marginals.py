#!/usr/bin/env python
"""Morgan-anchored marginals from an eight- or sixteen-cell factorial artefact."""
import json, sys

PAIRS = {
    'W': (('1000', '1100'), ('1001', '1101'), ('1010', '1110'), ('1011', '1111')),
    'A': (('1000', '1010'), ('1001', '1011'), ('1100', '1110'), ('1101', '1111')),
    'K': (('1000', '1001'), ('1010', '1011'), ('1100', '1101'), ('1110', '1111')),
}

def main():
    p = sys.argv[1] if len(sys.argv) > 1 else 'results/pending/factorial_8cell_canonical.json'
    d = json.load(open(p))
    print('seeds', d.get('meta', {}).get('seeds'))
    c = {e['id']: e['mean_auroc'] for e in d['cells'] if e.get('mean_auroc') is not None}
    for name, pairs in PAIRS.items():
        if any(a not in c or b not in c for a, b in pairs):
            print(name, 'incomplete'); continue
        v = [c[b] - c[a] for a, b in pairs]
        print(name, [round(x, 4) for x in v], 'mean', round(sum(v) / len(v), 4))
    for k in ('1000', '1010', '1011'):
        print('tier', k, round(c[k], 4) if k in c else 'absent')

if __name__ == '__main__':
    main()
