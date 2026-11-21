"""Refetch the accessions that hit 429, with exponential backoff."""
import json, time, urllib.request, pandas as pd
df = pd.read_csv('data/protac_bench.csv')
targets = sorted(set(df.target_uniprot.astype(str)))
seqs = json.load(open('data/target_sequences.json'))
missing = [t for t in targets if t not in seqs]
print(f'{len(missing)} missing: {missing}')
for t in missing:
    acc = t.split('/')[0]
    for attempt in range(6):
        try:
            txt = urllib.request.urlopen(
                f'https://rest.uniprot.org/uniprotkb/{acc}.fasta', timeout=60).read().decode()
            seq = ''.join(l.strip() for l in txt.splitlines() if not l.startswith('>'))
            if seq: seqs[t] = seq; print(f'  {t} ok ({len(seq)} aa)'); break
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f'  {t} attempt {attempt+1} failed ({e}); sleeping {wait}s')
            time.sleep(wait)
json.dump(seqs, open('data/target_sequences.json', 'w'))
still = [t for t in targets if t not in seqs]
print(f'now {len(seqs)} of {len(targets)}; still missing {still}')
