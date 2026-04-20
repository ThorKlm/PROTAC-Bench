# DeepPROTACs Replication Diagnostic (refreshed 2026-04-19)

## TL;DR

**Observed gap**: paper reports random-split AUROC = **0.847**; our previous best replication
reaches **0.626** (`exp20_deepprotacs_aligned`, linker SMILES, bs=32, 5 seeds) — a
**−0.221 AUROC gap**. A partial bs=1 run (4 seeds) reaches **0.716**, closing ~40 % of the
gap. The remaining ~0.13 is attributable to **upstream data differences** that the
authors' public artefacts do not expose.

**Architecture / training-loop level**: the replication is faithful (matches `model.py`
line-for-line). The gap is dominated by:

| # | Cause | Est. effect on AUROC |
|---|-------|----------------------|
| 1 | Different dataset + class balance (PROTAC-DB 988+988 vs DM PROTAC-8K 495+813) | ≈ −0.08 to −0.12 |
| 2 | Pocket extraction (5 Å distance shell on DM PDB vs fpocket-on-mol2) | ≈ −0.05 to −0.10 |
| 3 | **Batch size 32 vs paper's 1** (measured) | ≈ **−0.09** |
| 4 | Label definition (Good/Bad with explicit DC50≤100 nM ∧ Dmax≥80 % vs DM composite) | ≈ −0.02 to −0.05 |
| 5 | Linker SMILES extraction (SMARTS heuristic vs curated `linker_N.smi`) | ≈ 0.00 to −0.03 |

Top 3 most impactful: **dataset/balance, pocket extraction method, batch size**.

---

## 1. Original implementation (from /workspace/deepprotacs_github/)

GitHub: <https://github.com/fenglei104/DeepPROTACs> — repo cloned in
`/workspace/deepprotacs_github/`. Paper: Li, Hu, Zhang et al., *Nat Commun* 13:7133
(2022) — DOI 10.1038/s41467-022-34807-3.

### Architecture (`model.py`, 75 lines)

```
GraphConv(num_embeddings):
  Embedding(num_embeddings, 64)          # atom-type embedding
  GCNConv(64, 128)                       # uses edge_attr (cast float) as edge_weight
  ReLU
  GCNConv(128, 64)
  global_max_pool

SmilesNet:
  Embedding(41, 64, padding_idx=0)
  LSTM(64, 64, batch_first=True, bidirectional=True)
  Linear(128, 64)                         # readout = last timestep

ProtacModel:
  4 × GraphConv(5 or 10) for {ligase_ligand, ligase_pocket,
                              target_ligand, target_pocket}
  1 × SmilesNet for linker SMILES
  concat 5 × 64 → Linear(320, 64) → LeakyReLU(0.01) → Linear(64, 2)
  output: 2-class logits (CrossEntropyLoss)
```

Note: vanilla PyG `GCNConv` actually **ignores** the `edge_attr` argument unless
`add_self_loops=True` and you call it with `edge_weight`. The paper's call uses positional
args `gcn(x, edge_index, edge_attr)`, which PyG silently treats as `edge_weight`. Effect
is real (weights bond_type 1/2/3/ar=4/am=5).

### Training (`main.py:16-19`, `train_and_test.py`)

```
BATCH_SIZE = 1
EPOCH      = 30
TRAIN_RATE = 0.8                  # 80/20 sequential split over name.pkl
LEARNING_RATE = 1e-4
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss      = CrossEntropyLoss()
shuffle   = True (only inside DataLoader for train)
weight_decay = 0
no early stopping
```

Test-set AUROC reported as **pooled `roc_auc_score(y_true, softmax_pos)`** on the held-out
20 % slice. `case_study.ipynb` only reports accuracy on a 16-entry toy.

### Data preparation (`prepare_data.py`)

- **Pockets**: read pre-extracted mol2 files at `ligase_pocket_5/<name>.mol2` and
  `target_pocket_5/<name>.mol2`. The `_5` suffix denotes a 5 Å pocket built in
  `prepare_data.ipynb` (Schrödinger Maestro–dependent; not reproducible without the
  commercial software).
- **SMILES = LINKER ONLY** (`protacs/<name>/linker_<id>.smi`), canonicalised by RDKit,
  tokenised with the 41-char vocab below. Missing files → `[0]` (single PAD token).
- **Atom types**: `PROTEIN_ATOM_TYPE = ['C','N','O','S']` (4 + "other" → 5 embeddings);
  `LIGAND_ATOM_TYPE = ['C','N','O','S','F','Cl','Br','I','P']` (9 + "other" → 10).
- **Edge attr**: `{'1':1,'2':2,'3':3,'ar':4,'am':5}` from `@<TRIPOS>BOND` bond-type code.
- **Labels**: `Degradation Identification new 1` column → "Good" → 1, "Bad" → 0; "Poor"
  rows are dropped. "Good" = DC50 ≤ 100 nM AND Dmax ≥ 80 % per the paper.
- **SMILES vocab** (41 chars):
  `['[PAD]','C','(','=','O',')','N','[','@','H',']','1','c','n','/','2','#','S','s','+','-','\\','3','4','l','F','o','I','B','r','P','5','6','i','7','8','9','%','0','p']`

### Dataset

The GitHub repo ships **only 16 entries** (`data/processed/` + `protacs/{1_BRD7_VHL,
1_BRD9_VHL, 2_BRD7_VHL}`) — the case-study toy set. The full ~988 + 988 balanced training
set used in the paper was **never released**. `protacs_example.csv` shows just 4 lines.

### Evaluation

- Pooled AUROC on a single sequential 80/20 split.
- No CV, no LOTO, no per-target reporting in the paper.

---

## 2. Our replication (current state)

### Files

- `/workspace/PROTAC-Bench/baselines/deepprotacs_eval.py` — stub that loads cached
  per-target results (used for the benchmark figures).
- `/workspace/results/exp20_deepprotacs_loto/build_data.py` — builds `.pt` files from
  DegradeMaster PROTAC-8K (1308 entries with all four mol2/PDB files present).
- `/workspace/results/exp20_deepprotacs_loto/run_deepprotacs.py` — v1 trainer (full PROTAC
  SMILES, bs=32, 30 epochs random / 20 epochs LOTO).
- `/workspace/results/exp20_deepprotacs_faithful/run_faithful.py` — v2 trainer with linker
  SMILES extraction (SMARTS + graph fallback), bs=8, 30 epochs.
- `/workspace/results/exp20_deepprotacs_aligned/run_aligned_v3.py` — v3 trainer with
  linker vs full SMILES at bs=32 (5 seeds) + bs=1 (4 seeds, partial).

### Data source

DegradeMaster PROTAC-8K (`/workspace/baselines/DegradeMaster/degrademaster/data/PROTAC`):
1308 labelled entries with `protac.json`, `target_pocket/*.pdb`, `ligase_pocket/*.pdb`,
plus pre-docked warhead/E3 ligand mol2 files.

### Pocket extraction

`pdb_to_pocket_graph` (build_data.py:67-117): parses PDB, takes heavy atoms within 5 Å of
the docked warhead/E3-ligand mol2 coordinates, builds edges by 0.1–1.9 Å distance cutoff
with `edge_attr = 1` for every edge. **Atom types match** (C/N/O/S + other), but bond
types are all 1 (no chemistry).

**151/155 targets collapse to a single unique pocket graph** under this representation
(see `exp20_deepprotacs_loto/investigation_report.md`). This makes the GCN a target-ID
lookup, which is the structural cause of the LOTO collapse to 0.43–0.53.

### Linker SMILES extraction

SMARTS-based on CRBN/VHL handles + graph-decomposition fallback (run_faithful.py:43-186).
Coverage: 1104 SMARTS (84.4 %), 185 graph (14.1 %), 19 raw (1.5 %). Mean linker length
15.8 chars (paper's curated linkers: 5–20 chars).

### Labels

`entry['label']` from DegradeMaster PROTAC-8K. Different mapping than the paper: keeps
"Poor" entries as negatives, uses a different DC50 threshold. **495 pos / 813 neg = 37.8 %
positive** vs paper's ~50 % balanced.

---

## 3. Systematic comparison

### Architecture (everything matches)

| Component | Original | Replication | Match |
|-----------|----------|-------------|:-----:|
| GCN: 2 layers (64→128→64) with edge_weight | yes | yes | ✅ |
| Atom-type embedding (pocket / ligand) | 5 / 10 | 5 / 10 | ✅ |
| `global_max_pool` | yes | yes | ✅ |
| SMILES embedding | `Embedding(41, 64, pad=0)` | identical | ✅ |
| BiLSTM(64, 64) → Linear(128, 64) on last timestep | yes | yes | ✅ |
| Fusion: concat 320 → Linear(64) → LeakyReLU(0.01) → Linear(2) | yes | yes | ✅ |
| Loss: 2-class CrossEntropy | yes | yes | ✅ |

### Training

| Param | Original | Replication (best) | Match |
|-------|----------|-------------------:|:-----:|
| Optimizer | Adam | Adam | ✅ |
| Learning rate | 1e-4 | 1e-4 | ✅ |
| **Batch size** | **1** | 32 (with bs=1 partial) | ❌ |
| Epochs | 30 | 30 | ✅ |
| Weight decay | 0 | 0 | ✅ |
| Early stopping | none | none | ✅ |
| Train split | sequential 80/20 | random 80/20 (multi-seed) | ≈ |

### Inputs

| Input | Original | Replication | Match |
|-------|----------|-------------|:-----:|
| Target pocket | fpocket on co-crystal mol2 (5 Å) with mol2 bond types | 5 Å distance shell from PDB, edge_attr=1 | ❌ |
| Ligase pocket | same | same | ❌ |
| Warhead ligand graph | mol2 from PROTAC-DB | pre-docked mol2 (DM) | ≈ |
| E3 ligand graph | mol2 from PROTAC-DB | pre-docked mol2 (DM) | ≈ |
| SMILES | linker only (`linker_N.smi`, 5–20 chars) | linker only (SMARTS + graph fallback, 1–30 chars) | ≈ |

### Data / labels

| Property | Original | Replication | Match |
|----------|----------|-------------|:-----:|
| Source | PROTAC-DB 1.0 (curated) | DegradeMaster PROTAC-8K | ❌ |
| Size | 988 pos + 988 neg (balanced) or 1047 all | 495 pos + 813 neg | ❌ |
| Class balance | ~50 % pos | 37.8 % pos | ❌ |
| Label rule | Good = DC50 ≤ 100 nM ∧ Dmax ≥ 80 % | DegradeMaster `label` field | ❌ |

### Evaluation

| Aspect | Original | Replication |
|--------|----------|-------------|
| Metric | pooled `roc_auc_score(y_true, softmax_pos)` on held-out 20 % | identical |
| Split | single sequential 80/20 | random 80/20, mean ± std over seeds |
| LOTO | not in paper | added (26 eligible targets) |

---

## 4. Results (current replications)

| Condition | Random 80/20 AUROC | LOTO AUROC | Source |
|-----------|-------------------:|-----------:|--------|
| **Paper reported** | **0.847** | N/A | Li et al. 2022 |
| v1 (full SMILES, bs=32, 3 seeds) | 0.629 ± 0.036 | 0.425 ± 0.111 | exp20_deepprotacs_loto |
| v2 faithful (linker SMILES, bs=8, 3 seeds) | 0.664 ± 0.044 | 0.531 ± 0.154 | exp20_deepprotacs_faithful |
| v2 faithful (full SMILES, bs=8, 3 seeds) | 0.661 ± 0.032 | 0.451 ± 0.098 | exp20_deepprotacs_faithful |
| v3 aligned (linker, bs=32, 5 seeds) | **0.626 ± 0.040** | 0.456 ± 0.113 | exp20_deepprotacs_aligned |
| v3 aligned (full, bs=32, 5 seeds) | 0.622 ± 0.032 | — | exp20_deepprotacs_aligned |
| v3 aligned (linker, **bs=1**, 4 seeds, partial) | **0.716** | — | linker_bs1_partial in summary |

Deltas vs paper (0.847):
- best bs=32: −0.221
- bs=1 partial: −0.131 (recovers ~0.09)

LOTO ablation (linker SMILES):
- full model: 0.531
- no POI pocket: 0.548 (+0.016, **improves**)
- no E3 pocket: 0.531 (neutral)
- no warhead: 0.522 (slight hurt)
- no E3 ligand: 0.558 (+0.027, **improves**)
- no linker: 0.494 (−0.038, hurts most)

→ Pocket and E3-ligand graphs **hurt** under LOTO (target-ID adversarial); linker SMILES is
the only generalisable signal.

---

## 5. Top 3 most likely causes of the 0.22 gap

### #1 — Dataset + class balance (≈ −0.08 to −0.12)

- Paper trains on **988 pos + 988 neg balanced** (PROTAC-DB 1.0 with curated
  Good/Bad). We train on **495 pos + 813 neg imbalanced** (DM PROTAC-8K).
- With CrossEntropy without class weights, the imbalance gives a free majority-class
  shortcut. Different molecule set means different intrinsic difficulty.
- Not directly fixable (PROTAC-DB 1.0 + matching pocket data not released).

### #2 — Pocket extraction method (≈ −0.05 to −0.10)

- Paper: fpocket on mol2 → real chemical bonds (types 1/2/3/ar/am cast as `edge_weight`).
- Us: 5 Å heavy-atom shell from PDB, `edge_attr=1` for all edges (distance cutoff
  0.1–1.9 Å). The GCN effectively sees an unweighted graph.
- 151/155 targets share the **same** pocket graph after this extraction → GCN collapses to
  a target-ID lookup.
- Not fixable without (a) the authors' fpocket mol2 files or (b) a Schrödinger licence to
  re-run their `prepare_data.ipynb`.

### #3 — Batch size (≈ −0.09, **measured**)

- Paper: bs=1. Us: bs=32 (for feasibility).
- Partial bs=1 run gave **0.716** vs bs=32 **0.626** on the same data (Δ = +0.09).
- Very small batches give the BiLSTM ~32× more gradient updates and act as implicit
  regularisation.
- **This is the single largest controllable factor.** Re-running the v3 pipeline at bs=1
  for 3 seeds is the cheapest way to close ~0.09 of the gap.

(Smaller contributors: label definition, linker extraction method — see results table.)

---

## 6. What is and isn't fixable in v2

| Issue | Status | Fix |
|-------|:------:|-----|
| Linker vs full SMILES | ✅ fixed | SMARTS + graph extraction |
| Architecture / loss / optimizer | ✅ matches | line-for-line copy |
| Batch size 1 | ⏳ planned | re-run at bs=1 (single largest controllable) |
| Class balance | ⏳ planned | undersample negatives to 495+495 (matches paper's balanced split) |
| Sequential vs random split | minor | use random with 3 seeds (better than the paper's single split) |
| Distance-shell vs fpocket pockets | ❌ unfixable | requires authors' data or Schrödinger |
| Original 1047-row PROTAC-DB dataset | ❌ unfixable | not released |
| 151/155 pockets collapse to one graph | ❌ unfixable | structural property of distance pockets |

---

## 7. Plan for v2 faithful run (Phase 2)

1. Use the **already-extracted linker SMILES** from `run_faithful.py` extraction logic
   (cache to disk so we don't redo).
2. Switch to **batch_size = 1** (paper's setting). Time budget: ~30 min/seed × 3 seeds
   ≈ 90 min on a 4090 for random split; LOTO at bs=1 is too expensive
   (26 targets × 3 seeds × ~30 min ≈ 39 h), so use **bs=4** for LOTO as a compromise.
3. **Balanced sampling**: undersample negatives during training to 495 + 495 (mirrors the
   paper's 988+988 balance). Keep the test set unbalanced to report honest AUROC.
4. Keep the rest of the architecture and hyperparameters identical to the paper.
5. Run 3 seeds [42, 43, 44]; report per-seed and mean random AUROC + per-target LOTO
   AUROCs.
6. Save:
   - `/workspace/results/deepprotacs_faithful_v2/summary.json`
   - `/workspace/PROTAC-Bench/results/per_target/DeepPROTACs_faithful_random_per_target.csv`
   - `/workspace/PROTAC-Bench/results/per_target/DeepPROTACs_faithful_loto_per_target.csv`

Expected ceiling: ~0.72–0.78 random-split AUROC (combining bs=1 effect + class balance).
Remaining 0.07–0.13 gap to 0.847 is attributable to pocket-extraction method and is not
closable without authors' data.
