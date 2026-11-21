"""Mean-pooled ESM-2 representations for every benchmark target, written in the
form baselines/plm_scaling.py consumes. Sequences truncated to 1022 residues;
final-layer representations averaged over residue positions, excluding BOS/EOS."""
import json, numpy as np, torch, esm
MODELS = {'8M': 'esm2_t6_8M_UR50D', '35M': 'esm2_t12_35M_UR50D',
          '150M': 'esm2_t30_150M_UR50D', '650M': 'esm2_t33_650M_UR50D',
          '3B': 'esm2_t36_3B_UR50D'}
MAXLEN, OUT = 1022, 'cache'
seqs = json.load(open('data/target_sequences.json'))
ids = sorted(seqs)
np.save(f'{OUT}/target_ids.npy', np.array(ids, dtype=object), allow_pickle=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
for tag, name in MODELS.items():
    print(f'--- {tag} ({name})', flush=True)
    model, alphabet = esm.pretrained.__dict__[name]()
    layer = model.num_layers
    model = model.eval().half().to(dev) if dev == 'cuda' else model.eval()
    bc = alphabet.get_batch_converter()
    embs = []
    with torch.no_grad():
        for i, t in enumerate(ids, 1):
            _, _, toks = bc([(t, seqs[t][:MAXLEN])])
            rep = model(toks.to(dev), repr_layers=[layer])['representations'][layer]
            embs.append(rep[0, 1:-1].float().mean(0).cpu().numpy())
            if i % 50 == 0: print(f'  {i}/{len(ids)}', flush=True)
    M = np.stack(embs).astype(np.float32)
    np.save(f'{OUT}/prot_esm2_{tag}_targets.npy', M)
    print(f'  wrote {M.shape}', flush=True)
    del model; torch.cuda.empty_cache()
json.dump(dict(models=MODELS, max_len=MAXLEN, pooling='mean over residues, final layer',
               n_targets=len(ids)), open(f'{OUT}/embedding_meta.json', 'w'), indent=2)
print('done')
