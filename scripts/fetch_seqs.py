"""Fetch UniProt sequences for every target string in the benchmark. Composite
labels of the form A/B use the first accession, matching the convention the
UniRef diagnostic recorded."""
import json, time, urllib.request, pandas as pd
df = pd.read_csv('data/protac_bench.csv')
targets = sorted(set(df.target_uniprot.astype(str)))
out, miss = {}, []
for i, t in enumerate(targets, 1):
    acc = t.split('/')[0]
    url = f'https://rest.uniprot.org/uniprotkb/{acc}.fasta'
    for attempt in range(3):
        try:
            txt = urllib.request.urlopen(url, timeout=30).read().decode()
            seq = ''.join(l.strip() for l in txt.splitlines() if not l.startswith('>'))
            if seq: out[t] = seq
            break
        except Exception as e:
            if attempt == 2: miss.append((t, str(e)))
            time.sleep(2)
    if i % 25 == 0: print(f'{i}/{len(targets)} fetched, {len(miss)} failed', flush=True)
json.dump(out, open('data/target_sequences.json', 'w'))
print(f'sequences: {len(out)} of {len(targets)}; missing {miss[:5]}')
print(f'length: min {min(map(len, out.values()))} median '
      f'{sorted(map(len, out.values()))[len(out)//2]} max {max(map(len, out.values()))}')
