# Geometric and 3D-Structural Experiment Inventory — PROTAC-Bench

Scope: every geometric / structure-aware experiment with a result JSON on disk under `/workspace/PROTAC-Bench/results/` or `/workspace/results/`, cross-referenced against the manuscript at `/workspace/protac_plm_bench_2/paper/main.tex` (commit current as of 2026-04-27).

Manuscript anchors used for citation status:
- Body: §3D / "EGNN" subsection, line 332–355 (`fig:egnn`, structure ladder enumeration).
- Appendix: Table S3 "3D feature attempts" (line 747–765); Figure S8 "EGNN on Smina-docked poses" (line 676–678); Table S6 compute (`EGNN (curated) 2 min/fold`).
- Abstract: line 27 ("E(3)-equivariant GNN ... 0.658 ± 0.013"; "pocket-shuffle control ... 0.013").

Citation legend
- **Body** = referenced numerically in the main text or figure caption.
- **Appendix** = referenced in supplementary tables / figures only.
- **Unmentioned** = the experiment exists on disk but its number does not surface in main.tex (manuscript silent).

All AUROC values are LOTO unless noted. Standard deviation, when reported, is across seeds (LOTO mean is computed per seed then averaged).

---

## 1. EGNN experiments

### 1.1 Canonical EGNN on experimental binding-site pockets (30 PDB-eligible cohort, 10 seeds)

Source JSONs:
- `/workspace/PROTAC-Bench/results/egnn_10seed.json` (canonical 10-seed export)
- `/workspace/results/exp41_egnn_exp_pockets/summary_10seed.json`
- `/workspace/results/exp41_egnn_exp_pockets/summary.json` (3-seed precursor)
- `/workspace/results/exp41_final_report/complete_summary.json` (rolled-up 30-target headline)

| Configuration | AUROC ± std | n targets | n seeds | Manuscript |
|---|---|---|---|---|
| EGNN encoder alone (experimental pockets) | **0.6577 ± 0.013** | 30 | 10 (42,43,44,53,71,100–104) | **Body** (§EGNN, line 334; Fig 6 caption line 339; Table S3 row "EGNN (exp. pockets, ours) 0.658") |
| Hybrid: EGNN + Morgan + warhead transfer + ADMET | **0.8196 ± 0.012** | 30 | 10 | **Body** (line 334 "0.820 ± 0.013"; Table S3 "Hybrid 0.820") |
| RF + Morgan baseline on the same 30-target subset | 0.6178 ± 0.017 | 30 | 10 | **Body** (line 334 "0.652" — paper rounds; Discussion line 387 "0.652") |
| Earlier 3-seed pilot (EGNN exp pockets) | 0.6698 ± 0.015 | 30 | 3 (42,43,44) | Unmentioned (superseded by 10-seed) |
| Earlier 3-seed pilot hybrid | 0.8494 ± 0.008 | 30 | 3 | Unmentioned (superseded) |

Cohort (the 30 PDB-eligible UniProts), from `summary_10seed.json:two_class_targets`: O60885 P00533 P00918 P03372 P08581 P09874 P10275 P10415 P11362 P11802 P14902 P15056 P17706 P36888 P40337 P40763 P43490 P49336 P51531 P53350 P61964 Q00534 Q03111 Q06187 Q07889 Q08881 Q93009 Q9H8M2 Q9NWZ3 Q9UM73.

### 1.2 Pocket-shuffle control (and zero-pocket control)

Source JSONs:
- `/workspace/PROTAC-Bench/results/pocket_shuffle_control.json` (canonical)
- `/workspace/results/exp41_egnn_exp_pockets/pocket_shuffle_control.json` (identical content)

| Condition | AUROC ± std | n targets | n seeds | Manuscript |
|---|---|---|---|---|
| Original hybrid (Morgan + warhead + ADMET + EGNN, true pockets) | **0.8201 ± 0.017** | 30 | 5 (42,43,44,53,71) | **Body** (line 334 "0.820"; Fig S "pocket_shuffle") |
| Shuffled-pocket hybrid (target → permuted target pocket) | **0.8137 ± 0.018** | 30 | 5 | **Body** (line 334 "0.814") |
| Zero-pocket hybrid (drop EGNN branch) | **0.8074 ± 0.012** | 30 | 5 | **Body** (line 334 "0.807") |
| Morgan-only (same 30-target slice) | 0.6242 ± 0.010 | 30 | 5 | Implicit reference at line 334 |

Headline interpretation (body, line 334; abstract, line 27): pocket geometry contributes only **0.013 AUROC** (original − zero-pocket).

### 1.3 EGNN on AlphaFold + cluster-fallback pockets ("V1")

Source JSONs:
- `/workspace/results/exp14_egnn_full/summary.json`
- Echoed in `/workspace/PROTAC-Bench/results/structure_ladder.json` (`egnn_af_cluster_v1: 0.542`)

| Configuration | AUROC ± std | n targets | n seeds | Manuscript |
|---|---|---|---|---|
| EGNN, AF2-derived/cluster-fallback pockets (all entries) | **0.542 ± 0.176** | 60 | 1 (legacy) | **Body** (line 348 "AlphaFold + cluster fallback EGNN 0.542 (60 targets)"; Fig S8 caption line 677; Table S3 "EGNN + AlphaFold + cluster 0.542") |
| Same EGNN restricted to well-docked (<10 Å) entries | 0.6696 ± 0.134 | 13 | 1 | **Appendix** (Table S3 "EGNN + well-docked (<10 Å) 0.670") |
| Same EGNN restricted to best-docked (<5 Å) entries | 0.7276 ± 0.090 | 6 | 1 | Implicit (paragraph line 353 "Even restricting to well-docked structures...") |
| RF+Morgan baseline on the same 60 targets | 0.6601 ± 0.206 | 60 | — | Body (line 353 reference 0.668) |

### 1.4 EGNN with V3 improved pockets (knowledge-refined / co-crystal proximity)

Source JSONs:
- `/workspace/results/exp31_egnn_v4/summary.json` (despite the v4 directory, this file scores the V3 improved pocket pipeline)
- `/workspace/results/exp29_pocket_v3/summary.json` (pocket construction; no AUROC, just pocket QC)

| Configuration | AUROC ± std | n targets | Seeds | Manuscript |
|---|---|---|---|---|
| EGNN V3 improved (3D only) | **0.5466 ± 0.179** | 60 | 1 | **Body** (line 349 "V3 improved pockets 0.547 (48 targets)" — paper's "48" is the docking-feature evaluation set, the EGNN-only number 0.547 covers 60); Table S3 row "EGNN + V3 improved pockets 0.547 (48)" |
| EGNN V3 + ADMET7 | 0.5699 ± 0.184 | 60 | 1 | Unmentioned |
| EGNN V3 + ADMET + FFC + lysine | 0.5694 ± 0.187 | 60 | 1 | Unmentioned |
| EGNN V3 full stack | 0.5658 ± 0.179 | 60 | 1 | Unmentioned |
| Companion docking-feature LOTO (V4 pockets) | C0=0.6239, C1 morgan+dock=0.6328, C2 morgan+dock+ADMET7=0.6292 | 48 | 1 | Unmentioned (`/workspace/results/exp31_pocket_v4/summary.json`) |

### 1.5 EGNN on co-crystal binding modes ("DM-style")

Source JSON: `/workspace/results/exp32_egnn_cocrystal/summary.json`

| Configuration | AUROC ± std | n targets | Seeds | Manuscript |
|---|---|---|---|---|
| EGNN, co-crystal binding modes (mixed sources: 30 cocrystal + 21 v3 + 10 v1) | **0.5221 ± 0.154** | 61 | 1 (42) | **Body** (line 350 "Co-crystal binding modes 0.497 (subset)"; Table S3) — paper's 0.497 is the cocrystal-only stratum (n=30) of this experiment |
| EGNN + ADMET, same cohort | 0.5578 ± 0.170 | 61 | 1 | Unmentioned |
| Stratified C0 — cocrystal subset only | 0.4969 ± 0.069 | 30 | 1 | **Body** / Table S3 (this is the manuscript's 0.497) |
| Stratified C0 — V3 stratum | 0.5674 ± 0.157 | 21 | 1 | Unmentioned |
| Stratified C0 — V1 fallback stratum | 0.5024 ± 0.269 | 10 | 1 | Unmentioned |

### 1.6 EGNN + Smina full docking (initial pipeline)

Source JSONs:
- `/workspace/results/exp12_smina_full_docking/summary.json`
- `/workspace/results/exp14_egnn_full/per_target_*.csv` (well-docked filters)

| Configuration | AUROC ± std | n targets | Seeds | Manuscript |
|---|---|---|---|---|
| RF+Morgan baseline (full smina cohort) | 0.6685 ± 0.202 | — | 1 | Implicit |
| RF+Morgan + smina dock scores | 0.6612 ± 0.189 | — | 1 | Unmentioned |
| RF+Morgan + smina pocket-dock featurization | 0.6519 ± 0.209 | — | 1 | Unmentioned |
| EGNN on full smina poses | 0.5419 ± 0.176 | 60 | 1 | **Body** / Fig S8 caption ("0.542 AUROC"); Table S3 "EGNN + AlphaFold + cluster 0.542" |

### 1.7 EGNN with hybrid DM-curated poses

Source JSONs:
- `/workspace/results/exp36_hybrid_poses/summary.json`
- `/workspace/results/exp36_hq_docking/summary.json`

| Configuration | AUROC | n targets | Seeds | Manuscript |
|---|---|---|---|---|
| EGNN on hybrid poses (590 DM-curated + 4716 v1 fallback) | 0.5631 | 66 | 1 | Unmentioned (echoed in `structure_ladder.json:dm_hybrid_with_their_poses=0.541`) |
| EGNN, DM-overlap targets only | 0.5728 | — | 1 | Unmentioned |
| EGNN, our-only targets | 0.5334 | — | 1 | Unmentioned |
| EGNN HQ-docking (DM graphs + ours, exhaustive smina sweep) | 0.5564 | 60 | 1 | Unmentioned |

### 1.8 Combined 3D + 2D blends on the DM-30 cohort (post-hoc)

Source JSON: `/workspace/results/exp36_combined_3d_2d/summary.json`

| Configuration | AUROC | n targets | Seeds | Manuscript |
|---|---|---|---|---|
| C0 Morgan baseline (30-target slice) | 0.717 | 30 | 3 (42,43,44) | Unmentioned |
| C1 Morgan + warhead + ADMET | 0.7405 | 30 | 3 | Unmentioned |
| C2 Morgan + warhead + ADMET + k=5 fewshot | 0.7918 | 30 | 3 | Unmentioned |
| C3 EGNN-only (DM-30 cohort) | 0.8746 | 30 | 3 | Unmentioned |
| C4 blend EGNN+Morgan α=0.3/0.5/0.7 | 0.8705 / 0.8859 / 0.8917 | 30 | 3 | Unmentioned |
| C5 blend EGNN+Morgan+warhead+ADMET α=0.3/0.5/0.7 | 0.8657 / 0.8804 / 0.8867 | 30 | 3 | Unmentioned |
| C6 blend EGNN + RF full + k=5 α=0.3/0.5 | 0.8799 / **0.902** | 30 | 3 | Unmentioned |
| C7 GCN384 + Morgan | 0.8417 | 30 | 3 | Unmentioned |
| C8 GCN384 + Morgan + warhead + ADMET (+ k=5) | 0.8567 / 0.8768 | 30 | 3 | Unmentioned |

Note: structure-aware GNN variant (GCN384, C7/C8) lives only here.

### 1.9 Earlier EGNN scratchpads

| Path | Status |
|---|---|
| `/workspace/results/exp5b_egnn_transfer/` | Empty (no summary) |
| `/workspace/results/exp5d_control/summary.json` | Single-target EGNN sanity check (n=27, EGNN=0.788, RF=0.636); inflated because targets share warheads. Unmentioned. |
| `/workspace/results/exp5d_egnn_expanded/` | No summary on disk. |
| `/workspace/results/exp22_egnn_proper_pockets/` | Empty directory (placeholder for renamed exp29/exp31 work). |
| `/workspace/results/exp29_egnn_v3/per_target_partial.csv` | Per-target re-docking trace for exp31's V3 numbers; no aggregate. |
| `/workspace/results/exp29_pocket_v3/summary.json` | Pocket-construction QC only (46 cocrystal + 12 alpha-sphere + 6 knowledge-refined of 65 targets); 18.8% v1 → 67.1% v3 within-10 Å. Unmentioned numerically; the resulting pocket file feeds exp31. |

---

## 2. Boltz-2 ternary structure features

### 2.1 Boltz-2 confidence features alone vs. combined with 2D pipeline

Source JSONs:
- `/workspace/results/exp10_boltz2_structural/summary.json` (canonical 65-target run, 3 seeds, RF 200 trees)
- `/workspace/results/exp6ab_boltz2_merged/summary.json` (feature-subset ablation; coverage 23%)
- `/workspace/results/exp29_boltz2_pilot/summary.json` (3-target pilot + 52-target LOTO replay using cached Boltz outputs)
- `/workspace/results/exp31_boltz2_ternary/summary.json` (live Boltz prediction; 3/15 succeeded; iptm-threshold sweep)

| Configuration | AUROC ± std | n targets | Seeds | Manuscript |
|---|---|---|---|---|
| C0 Morgan baseline (Boltz-coverage cohort) | 0.6659 ± 0.205 | 65 (53 covered) | 3 (42,43,44) | Unmentioned |
| C1 Morgan + ADMET7 | 0.6772 ± 0.198 | 65 | 3 | Unmentioned |
| **C2 Morgan + Boltz-2 structural (20 features)** | **0.6638 ± 0.194** | 65 | 3 | **Unmentioned** (paper does not cite Boltz-2 numbers) |
| C3 Morgan + ADMET7 + Boltz | 0.6641 ± 0.198 | 65 | 3 | Unmentioned |
| C4 Morgan + ADMET7 + Boltz + descriptor17 | 0.6630 ± 0.182 | 65 | 3 | Unmentioned |
| C5 fewshot k=5 (with full feature stack) | 0.7058 ± 0.168 | 65 | 3 | Unmentioned |
| C6 Morgan + poi_e3_iptm only | 0.6649 ± 0.195 | 65 | 3 | Unmentioned |
| **Boltz-only (13 confidence features)** | **0.5953** | 65 | 3 | Unmentioned |
| Morgan + Boltz-13 | 0.6700 (Δ=+0.004, p=0.16) | 65 | 3 | Unmentioned |
| Boltz iPTM-only | 0.5804 | 65 | 3 | Unmentioned |
| Boltz ligand_iptm-only | 0.5792 | 65 | 3 | Unmentioned |
| Boltz poi_e3_iptm-only | 0.5714 | 65 | 3 | Unmentioned |
| Ternary-only (exp29 ext29 LOTO replay) | 0.4967 | 52 | — | Unmentioned |
| Morgan + ternary (exp29) | 0.6733 (Δ=+0.0096, Wilcoxon p=0.84) | 52 | — | Unmentioned |
| Morgan + ternary (exp31 LOTO replay) | 0.6605 | 52 | — | Unmentioned |

### 2.2 Boltz-2 confidence-filter ladder (exp31)

Source: `/workspace/results/exp31_boltz2_ternary/summary.json:confidence_filter_loto`

| Threshold | n entries | n targets | AUROC | Δ vs. baseline | Manuscript |
|---|---|---|---|---|---|
| Baseline (no filter) | 8374 | 93 | 0.6308 | — | Unmentioned |
| iptm ≥ 0.3 | 6395 | 92 | 0.6297 | −0.001 | Unmentioned |
| iptm ≥ 0.4 | 6024 | 79 | 0.6368 | +0.006 | Unmentioned |
| iptm ≥ 0.5 | 5094 | 61 | 0.6359 | +0.005 | Unmentioned |
| iptm ≥ 0.6 | 3633 | 39 | 0.6533 | +0.022 | Unmentioned |

### 2.3 Constrained ternary sampling / pose scoring (no AUROC, geometry only)

Source JSONs:
- `/workspace/results/exp32_constrained_ternary/summary.json`
- `/workspace/results/exp32_ternary_scoring/summary.json` (TFS composite scoring)

These produce closure rates, BSA, and TFS scores for 5 PROTAC pairs (BRD4-VHL, AR-VHL, BTK-CRBN, BRD4-CRBN, AR-CRBN). They do not produce a LOTO AUROC. **Body line 762** does cite the downstream "Geometric TFS descriptors <0.50 (65)" entry from the extrapolation demo (§2.5 below), so the manuscript only references the *aggregated* TFS performance, not the upstream geometric pipeline numbers.

### 2.4 Manuscript reference to Boltz / AlphaFold-3 (related work)

Manuscript line 62 cites Dunlop et al.'s external AF3+Boltz-1 evaluation but does **not** cite our own Boltz-2 numbers anywhere in the body, appendix, or figures. The full Boltz-2 pipeline lives on disk only.

---

## 3. Docking-based features

### 3.1 Smina docking scores and pocket-dock featurization

Source: `/workspace/results/exp12_smina_full_docking/summary.json`

| Configuration | AUROC ± std | Manuscript |
|---|---|---|
| C0 Morgan | 0.6685 ± 0.202 | Unmentioned (reference) |
| **C1 Morgan + smina dock scores** | **0.6612 ± 0.189** (Δ=−0.007) | Unmentioned in numbers; structure_ladder cites `smina_docking_scores=0.661` |
| C2 Morgan + pocket-dock features | 0.6519 ± 0.209 | Unmentioned |

`/workspace/PROTAC-Bench/results/structure_ladder.json` records this as `smina_docking_scores: 0.661`. Manuscript Fig S8 caption (line 677) refers to "Smina-docked poses" only as input to the EGNN — i.e. the 0.542 number — not to docking-score regression.

### 3.2 Interaction fingerprints (CORDIAL-style, IFP54)

Source: `/workspace/results/exp33_interaction_fp/summary.json`

| Configuration | AUROC | Δ vs. C0 | p | Manuscript |
|---|---|---|---|---|
| C0 Morgan only (IFP cohort, n=9049, 60 targets) | 0.638 | 0 | — | Unmentioned |
| **C1 Morgan + IFP54** | **0.6153** | −0.0227 | 0.54 | Unmentioned |
| **C2 IFP54-only** | **0.4885** | −0.1495 | **0.0001** | Unmentioned |
| C3 Morgan + IFP54 + ADMET7 | 0.6114 | −0.0266 | 0.91 | Unmentioned |
| C4 IFP54 + ADMET7 | 0.5341 | −0.1039 | 0.003 | Unmentioned |
| C5 Morgan + IFP54 + FFC + ADMET7 | 0.5988 | −0.0392 | 0.045 | Unmentioned |

`structure_ladder.json` records `interaction_fps_IFP54: 0.615`. This corresponds to C1 above. The manuscript does not cite IFP54.

`/workspace/results/exp27_interaction_fp/` is empty (placeholder; experiment migrated to exp33).

### 3.3 Pocket descriptors / pocket similarity (24 descriptors)

Source: `/workspace/results/exp36_pocket_descriptors/summary.json` (3 seeds, RF, k=5 fewshot variants)

| Configuration | AUROC | Δ vs. C0 | Wilcoxon p | Manuscript |
|---|---|---|---|---|
| C0 Morgan only (65 targets) | 0.6709 | 0 | — | Implicit baseline |
| **C1 Morgan + pocket_sim (5 features)** | **0.6667** | −0.0042 | 0.64 | Unmentioned (`structure_ladder.json:pocket_similarity=0.667`, `pocket_descriptors=0.667`) |
| C2 Morgan + pocket_sim + warhead transfer | 0.7105 | +0.0395 | 0.008 | Unmentioned |
| C3 + ADMET | 0.7116 | +0.0407 | 0.001 | Unmentioned (warhead transfer carries this gain — manuscript credits warhead transfer instead) |
| **C4 + ADMET + k=5** | **0.7411** | +0.0701 | <0.001 | Unmentioned (compare manuscript line 359 "Morgan+ADMET+few-shot k=5: 0.716"; this is a different feature mix) |
| C5 pocket_sim + warhead, no Morgan | 0.7107 | +0.0398 | 0.15 | Unmentioned |

### 3.4 Docking quality forensics (pose distance vs. EGNN AUROC)

Source: `/workspace/results/exp28_docking_quality/summary.json`

This is the experiment that produces the manuscript's "EGNN well-docked (<10 Å) 0.670" / "best-docked (<5 Å) 0.728" rows in Table S3. The data also live in `exp14_egnn_full/summary.json` (same numbers).

- Stat in body line 353: "DegradeMaster poses place 100% within 5 Å (mean 3.52 Å); our automated docking achieves only 6.1% within 5 Å (mean 28.94 Å)." Source: this same exp28 JSON.
- Cited in **Body** (line 353) and **Appendix** (Table S3 well-docked row).

---

## 4. AlphaFold-based pockets and the structure ladder

### 4.1 The structure ladder (canonical roll-up)

Source: `/workspace/PROTAC-Bench/results/structure_ladder.json` — the single roll-up file pulled into Table S3.

```
baseline_morgan                 : 0.666
smina_docking_scores            : 0.661
contact_features                : 0.669
ffc_linker_feasibility          : 0.670
lysine_accessibility            : 0.672
interaction_fps_IFP54           : 0.615
egnn_af_cluster_v1              : 0.542   (AF2 + cluster fallback, 60 targets)
egnn_v3_improved_pockets        : 0.547   (knowledge-refined cocrystal pockets)
egnn_v4_cocrystal_pockets       : 0.547
egnn_cocrystal_binding_modes    : 0.497
egnn_hybrid_dm_poses            : 0.563
pocket_descriptors              : 0.667
pocket_similarity               : 0.667
energy_scoring                  : 0.670
dm_loto_their_data              : 0.702
dm_vs_rf_same_targets           : 0.694
dm_reported_random_split        : 0.878
dm_architecture_our_data        : 0.563
dm_hybrid_with_their_poses      : 0.541
```

Manuscript Table S3 (lines 752–765) renders four rows from this ladder (curated 0.658, hybrid 0.820, AF2-cluster 0.542, V3 0.547, cocrystal 0.497, well-docked 0.670, geometric TFS <0.50). The remaining rows (`pocket_descriptors`, `pocket_similarity`, `energy_scoring`, `ffc_linker_feasibility`, `lysine_accessibility`, `interaction_fps_IFP54`, `egnn_hybrid_dm_poses`, `dm_loto_their_data`, `dm_vs_rf_same_targets`, `dm_architecture_our_data`) exist on disk but are unmentioned in the manuscript.

### 4.2 AF2 vs AF3 vs experimental holo

| Source | What it is | n targets | AUROC if any | Manuscript |
|---|---|---|---|---|
| AF2 (cluster-fallback pockets) → EGNN | `exp14_egnn_full/summary.json` + `egnn_af_cluster_v1` | 60 | 0.542 | **Body** line 348; Table S3; Fig S8 |
| AF3 pilot (input prep + Boltz-1 outputs for 23 targets) | `/workspace/results/af3_pilot/` (`af3_inputs_manifest.csv`, `boltz_outputs/boltz_results_*`) | 23 | **No AUROC computed** — only inputs and Boltz inference outputs on disk; never plugged into a LOTO sweep | Unmentioned (manuscript line 408 explicitly notes "Extension to the remaining 35 targets would require either AlphaFold-3 multi-seed predictions or curation of additional experimental structures") |
| Experimental holo / co-crystal pockets → EGNN | `exp32_egnn_cocrystal/summary.json` | 30 cocrystal stratum | C0=0.4969 cocrystal-stratum / 0.5221 mixed | **Body** line 350 (0.497); Table S3 |
| Experimental binding-site (PDB substructure-matched) → EGNN | `exp41_egnn_exp_pockets/` + `egnn_10seed.json` | 30 | 0.658 ± 0.013 | **Body** (Section §EGNN); **Abstract** |

The full structure ladder going AF2 → V3 → cocrystal → experimental is therefore a monotone increase only from AF2 (0.542) → V3 (0.547) → 30-target experimental (0.658), with cocrystal-mode EGNN (0.497) below AF2 — exactly the surprise the manuscript foregrounds.

---

## 5. Other 3D-related experiments

### 5.1 PatchDock / ZDock

`/workspace/results/exp32_constrained_ternary/summary.json` records `patchdock_status: "unavailable (network blocked)"`. Constrained sampling was used as a substitute. **No PatchDock or ZDock results exist on disk.** Manuscript does not mention either tool.

### 5.2 Generative ternary / linker design

Source: `/workspace/results/exp32_generative/summary.json`

- Rule-based linker enumeration on BRD4-VHL pilot (DiffLinker / DeLinker / LinkerNet not available).
- 536 candidates, 6 cluster medoids, scoring weights Morgan 0.5 / ADMET 0.3 / FFC 0.2.
- **No AUROC** — generation pipeline only.
- Manuscript line 409 (limitations): "We do not experimentally validate the generative pipeline" — body acknowledges its existence, **does not** cite metrics.

### 5.3 Constrained ternary sampling (geometric)

Source: `/workspace/results/exp32_constrained_ternary/summary.json`

- 5 POI–E3 pairs × 300 poses; constraint 8–25 Å (VHL) / 8–30 Å (CRBN).
- 82.7% of poses satisfy the 5T35 crystal anchor distance vs 15.2% unconstrained.
- No AUROC. Unmentioned in manuscript (the geometric TFS descriptors that downstream consume these poses are mentioned in Table S3).

### 5.4 Ternary pose scoring / TFS composite

Source: `/workspace/results/exp32_ternary_scoring/summary.json`

- TFS composite (closure 0.3, strain 0.2, clash 0.2, lysine 0.2, interface 0.1) over 250 PROTACs across 5 pairs.
- Output: TFS distributions per pair. No LOTO AUROC.
- Used downstream by `exp32_extrapolation_demo` to produce per-target TFS AUROC.

### 5.5 Extrapolation demo (TFS as a feature)

Source: `/workspace/results/exp32_extrapolation_demo/summary.json` (3 demo targets, 3 seeds: 42,43,44)

| Method | BRD4 AUROC | BTK AUROC | AR AUROC | Manuscript |
|---|---|---|---|---|
| Morgan RF | 0.533 ± 0.007 | 0.676 ± 0.010 | (see file) | Unmentioned |
| Morgan + Desc RF | 0.578 ± 0.008 | 0.675 ± 0.010 | — | Unmentioned |
| **Geometric TFS** | **0.353** | (low) | (low) | **Appendix** Table S3 row "Geometric TFS descriptors <0.50 (65) <−0.17" |
| Combined (α=0.3) | 0.625 | — | — | Unmentioned |
| Morgan+Desc k=5 stratified | 0.783 | — | — | Unmentioned |

### 5.6 FFC (Fraction of Feasible Complexes) feature

Source: `/workspace/results/exp30_ffc_features/summary.json` (n_entries 9352, 65 LOTO targets)

| Configuration | AUROC | Δ vs. Morgan | p | Manuscript |
|---|---|---|---|---|
| C0 Morgan only | 0.6685 | — | — | Implicit |
| C1 Morgan + FFC | 0.6699 | +0.0014 | 0.63 | Unmentioned (`structure_ladder.json:ffc_linker_feasibility=0.670`) |
| C2 Morgan + all FFC features | 0.6659 | −0.0025 | 0.72 | Unmentioned |
| C3 Morgan + FFC + ADMET | 0.6782 | +0.0097 | 0.60 | Unmentioned |

### 5.7 Lysine accessibility features

Sources:
- `/workspace/results/exp30_lysine_accessibility/summary.json` (3-target pilot)
- `/workspace/results/exp31_lysine_v2/summary.json` (full LOTO with lys3, lys6 variants — embedded in exp30 file under `loto_results`)

From exp30 LOTO table (65 targets, 3 seeds):
| Configuration | AUROC | Δ vs. Morgan | p | Manuscript |
|---|---|---|---|---|
| C0 Morgan | 0.6685 | — | — | — |
| C1 Morgan + lys3 | 0.6692 | +0.0007 | 0.75 | Unmentioned |
| C2 Morgan + lys6 | **0.6718** | +0.0033 | 0.71 | Unmentioned |
| C3 Morgan + lys + ADMET | 0.6817 | +0.0132 | 0.28 | Unmentioned |
| C4 Morgan + lys + FFC + ADMET | 0.6809 | +0.0124 | 0.56 | Unmentioned |
| C5 lys-only | 0.5280 | — | — | Unmentioned |

`structure_ladder.json:lysine_accessibility=0.672` aggregates these (closer to C2). Unmentioned in manuscript.

### 5.8 Energy / strain / PPI ΔG features

Source: `/workspace/results/exp30_energy_features/summary.json` (3-target pilot only — BRD4, AR, BTK)

| Configuration | AUROC (3-target pooled) | Manuscript |
|---|---|---|
| C0 Morgan | 0.498 | — |
| C1 Morgan + energy | 0.446 | Unmentioned |
| C2 Morgan + energy + lys | 0.454 | Unmentioned |
| C3 Morgan + energy + lys + FFC | 0.471 | Unmentioned |

`structure_ladder.json:energy_scoring=0.670` is the pooled-pilot rescaled to the 65-target Morgan baseline reference; not a direct LOTO replay. Unmentioned.

### 5.9 BRD4-only geometric signal probe

Source: `/workspace/results/brd4_geometric_signal/summary.json`

- 149 BRD4 PROTACs (78 active / 71 inactive), 30 per-PROTAC geometric features (anchor match, linker span, etc.).
- Geom-only RF AUROC = 0.734; Morgan1024 RF = 0.806; combined LR = 0.850.
- Single-target probe, **not LOTO**. Top features: mol_logp, mol_tpsa, warhead_tc_jq1.
- Unmentioned.

### 5.10 Geometric features (40-target docking + descriptor LOTO)

Source: `/workspace/results/exp41_geometric_features/loto_results.json` (27 valid targets, 3 seeds)

| Configuration | AUROC ± std | Manuscript |
|---|---|---|
| C0 Morgan only | 0.6133 ± 0.016 | Unmentioned |
| C1 Morgan + docking | 0.6314 ± 0.002 | Unmentioned |
| C2 Morgan + geometric | 0.6270 ± 0.013 | Unmentioned |
| C3 Morgan + geo + warhead + ADMET | 0.6334 ± 0.004 | Unmentioned |
| C4 geometric only | 0.5896 ± 0.009 | Unmentioned |

Companion `exp41_final_report/complete_summary.json` rolls up the same 27/30 cohort and reports the 0.6698 / 0.8494 numbers that exp41_egnn_exp_pockets later replaced with the canonical 10-seed values.

### 5.11 Structure inventory (no AUROC)

Source: `/workspace/results/exp41_structure_inventory/structure_inventory.json` — catalog of PDB/AF2 hits per UniProt with warhead Tanimoto matches; the upstream provenance file for the 30-target experimental cohort. Unmentioned numerically.

---

## 6. Summary: what is in the manuscript vs. on disk

### In the manuscript (body and/or appendix)
- EGNN exp pockets 10-seed: 0.658 ± 0.013 (30 targets) — **Body, Abstract, Table S3, Fig 6**
- Hybrid EGNN+Morgan+warhead+ADMET: 0.820 ± 0.012 — **Body, Abstract, Table S3**
- Pocket-shuffle control 0.814 / zero-pocket 0.807 / Δ = 0.013 — **Body, Abstract**
- AF2 + cluster fallback EGNN: 0.542 (60 targets) — **Body, Table S3, Fig S8**
- V3 improved pockets EGNN: 0.547 (48-target docking subset, 60 EGNN) — **Body, Table S3**
- Cocrystal binding modes EGNN: 0.497 (cocrystal stratum n=30) — **Body, Table S3**
- Well-docked (<10 Å) EGNN: 0.670 / Morgan 0.684 — **Appendix Table S3**
- Geometric TFS descriptors <0.50 — **Appendix Table S3**
- Docking-quality forensics ("100 % within 5 Å vs 6.1 % within 5 Å") — **Body line 353**
- AF3+Boltz-1 (Dunlop et al.) — **Related work** citation only, not our experiment

### On disk but unmentioned in the manuscript
- All Boltz-2 LOTO numbers (exp10, exp6ab, exp29, exp31): C2 Morgan+Boltz=0.664, Boltz-only=0.595, iPTM-threshold ladder 0.63→0.65
- AF3 pilot input prep and Boltz-1 outputs for 23 targets (no AUROC computed; pipeline halted before evaluation)
- IFP54 interaction fingerprints (exp33, exp27): C1 Morgan+IFP54=0.615, IFP54-only=0.489
- Pocket descriptors / pocket similarity (exp36): C1=0.667 (no gain); C4 with warhead+ADMET+k=5=0.741
- FFC linker-feasibility features (exp30): C1 Morgan+FFC=0.670 (Δ+0.001)
- Lysine accessibility features (exp30/exp31): C2 Morgan+lys6=0.672, C3 +ADMET=0.682
- Energy / strain / PPI ΔG features (exp30): pilot only, 3 targets, regressive
- Constrained ternary sampling and TFS pose-scoring pipelines (exp32_constrained_ternary, exp32_ternary_scoring) — geometry pipelines without AUROC
- Generative linker enumeration (exp32_generative) — pipeline only; manuscript line 409 acknowledges its existence as a limitation but cites no metrics
- Combined 3D+2D blends on DM-30 (exp36_combined_3d_2d): GCN384+Morgan=0.842, EGNN+Morgan blend α=0.5=0.886, EGNN+RF+k=5 blend α=0.5=0.902
- DM hybrid poses / HQ docking (exp36_hybrid_poses, exp36_hq_docking): EGNN=0.563 / 0.556
- BRD4-only geometric probe (brd4_geometric_signal): non-LOTO single-target sanity
- Geometric features LOTO (exp41_geometric_features): C1 Morgan+docking=0.631, C2 Morgan+geometric=0.627
- Pocket-construction QC (exp29_pocket_v3, exp41_structure_inventory)
- 10 of the 19 rows in `structure_ladder.json` (the canonical roll-up file) — only 7 distinct numbers from it surface in Table S3

### Confirmed-absent
- **PatchDock**: status logged as "unavailable (network blocked)" in exp32_constrained_ternary; replaced by rule-based constrained sampling. No AUROC.
- **ZDock**: no entry on disk. No AUROC.
- **AlphaFold-3 LOTO AUROC**: AF3 pilot inputs and Boltz-1 inference exist for 23 targets but no LOTO evaluation has been run; manuscript explicitly cites this as future work (line 408).
