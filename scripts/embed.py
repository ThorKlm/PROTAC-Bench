"""Mean-pooled protein and molecule embeddings, written with the checkpoint
revision each matrix came from. Float32 throughout, so a rebuild on a GPU
matches a rebuild on a CPU."""
import json, pathlib
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer

PROT, MOL = 'facebook/esm2_t33_650M_UR50D', 'seyonec/ChemBERTa-zinc-base-v1'
OUT = pathlib.Path('data/embeddings')

def encode(texts, name, maxlen, batch):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name, dtype=torch.float32).eval()
    from huggingface_hub import model_info
    rev = model_info(name).sha
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], return_tensors='pt', padding=True,
                    truncation=True, max_length=maxlen)
            h = mdl(**b).last_hidden_state
            m = b['attention_mask'].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
            print(f'  {min(i + batch, len(texts))}/{len(texts)}', flush=True)
    return np.vstack(out).astype(np.float32), rev

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = json.load(open('data/target_sequences.json'))
    ids = sorted(seqs)
    print(f'esm2 over {len(ids)} sequences')
    P, prev = encode([seqs[i] for i in ids], PROT, 1022, 4)
    np.save(OUT / 'esm2_650m_targets.npy', P)
    np.save(OUT / 'target_ids.npy', np.array(ids, dtype=object), allow_pickle=True)
    smi = pd.read_csv('data/protac_bench.csv').smiles.astype(str).tolist()
    uniq = sorted(set(smi))
    print(f'chemberta over {len(uniq)} unique smiles')
    M, mrev = encode(uniq, MOL, 512, 64)
    pos = {s: i for i, s in enumerate(uniq)}
    np.save(OUT / 'chemberta_rows.npy', M[[pos[s] for s in smi]])
    json.dump(dict(protein_model=PROT, protein_revision=prev,
                   protein_dim=int(P.shape[1]), protein_rows=int(P.shape[0]),
                   molecule_model=MOL, molecule_revision=mrev,
                   molecule_dim=int(M.shape[1]), molecule_rows=len(smi),
                   pooling='attention-masked mean of final hidden states',
                   dtype='float32', protein_maxlen=1022, molecule_maxlen=512),
              open(OUT / 'provenance.json', 'w'), indent=1)
    print('wrote', OUT)

if __name__ == '__main__':
    main()
