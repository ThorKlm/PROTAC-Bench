# Supplementary Values for Manuscript
Extracted 2026-05-02 from prior task artefacts.

## 1. exp104 — 10-seed per-architecture per-k AUROC tables
Source: `/workspace/exp104_extended_fewshot_10seed/REPORT.md`. Macro-mean per seed across evaluable targets, then mean ± population SD across 10 canonical seeds {7, 13, 29, 42, 43, 44, 53, 71, 89, 97}.

### Panel (b) matched 14-target cohort
| k | RF mean | RF SD | MAML mean | MAML SD | ProtoNet mean | ProtoNet SD |
|---|---|---|---|---|---|---|
| 0¹  | 0.6634 | 0.0041 | — | — | — | — |
| 1   | 0.6565 | 0.0057 | 0.6500 | 0.0126 | — | — |
| 2   | 0.6605 | 0.0090 | 0.6514 | 0.0076 | 0.5537 | 0.0616 |
| 5   | 0.6647 | 0.0069 | 0.6526 | 0.0054 | 0.6127 | 0.0195 |
| 10  | 0.6778 | 0.0085 | 0.6535 | 0.0060 | 0.6292 | 0.0269 |
| 25  | 0.7096 | 0.0096 | 0.6577 | 0.0089 | 0.6596 | 0.0261 |
| 50  | 0.7458 | 0.0069 | 0.6672 | 0.0130 | 0.6721 | 0.0170 |
| 100 | 0.7956 | 0.0232 | 0.6931 | 0.0239 | 0.6832 | 0.0213 |

¹ k=0 from `exp105_matched_cohort_k0_baseline.json` (canonical Morgan+RF LOTO restricted to the 14 targets and 10 seeds).

### Panel (a) full eligible cohort (n_targets shrinks 65→57→42→28→14)
| k | RF mean | RF SD | MAML mean | MAML SD | ProtoNet mean | ProtoNet SD |
|---|---|---|---|---|---|---|
| 1   | 0.6638 | 0.0098 | 0.6789 | 0.0106 | — | — |
| 2   | 0.6741 | 0.0071 | 0.6804 | 0.0200 | 0.6385 | 0.0428 |
| 5   | 0.6852 | 0.0104 | 0.6793 | 0.0133 | 0.6877 | 0.0173 |
| 10  | 0.7028 | 0.0069 | 0.6671 | 0.0204 | 0.6926 | 0.0168 |
| 25  | 0.7555 | 0.0128 | 0.6867 | 0.0282 | 0.7120 | 0.0194 |
| 50  | 0.7985 | 0.0199 | 0.6967 | 0.0227 | 0.7166 | 0.0170 |
| 100 | 0.7821 | 0.0237 | 0.6623 | 0.0281 | 0.6985 | 0.0208 |

### Matched-cohort 14-target identity list (UniProt)
O60885, P00533, P01116, P01116/P04049, P03372, P10275, P10636, P14902, P15056, P51531, P51532, Q06187, Q92793, Q9NWZ3.

### Non-monotonic curve transition points
- Matched cohort (panel b): all three architectures monotonically increase across the reported k range; no transitions.
- Full cohort (panel a): all three peak at k=50 and decline at k=100, coincident with cohort shrinking 28→14 targets. RF: 0.7985→0.7821 (Δ=−0.0164). MAML: local dip k=5→k=10 (0.6793→0.6671, Δ=−0.0122); peak 0.6967→0.6623 (Δ=−0.0344). ProtoNet: 0.7166→0.6985 (Δ=−0.0181).

## 2. RAI replacement text for Appendix P / Q
Source: `/workspace/PROTAC-Bench/results/v11/croissant_rai_audit.md` §4. All 20 `rai:*` fields are currently empty in `data/croissant.json`; quote-ready substantive replacements (verbatim from the audit, abridged where redundant):
- `rai:dataCollection`: "Aggregates 10,748 PROTAC-target pairs from PROTAC-DB 3.0 (Weng 2023), Ribes (2024), DegradeMaster (Liu 2024); de-duplicated on canonical SMILES + UniProt; SMILES standardised by RDKit; targets resolved via UniProt REST."
- `rai:dataCollectionType`: "Aggregation of pre-existing public datasets; no primary experimental collection."
- `rai:dataCollectionTimeframe`: "Source publications 2001–2023 (PROTAC-DB 3.0); merged corpus frozen 2025-Q4; temporal split pre-2022 train, 2022+ held out."
- `rai:dataCollectionRawData`: "Processed: SMILES canonicalised, targets normalised to UniProt, labels binarised (DC50<1 µM ∨ Dmax>50%); raw `dc50_nm`/`dmax_pct` preserved."
- `rai:dataCollectionMissingData`: "~38% of entries report only Dmax or only DC50; cell-line/assay-format/time-point metadata not included."
- `rai:dataAnnotationProtocol`: "Labels inherited from source binarisation rules; cross-source agreement on the 1,247 multi-source entries is 98.4% (κ=0.96)."
- `rai:dataAnnotationPlatform`: "No platform — labels propagated from upstream curated databases."
- `rai:dataAnnotationDemographics`: "Not applicable: labels are biochemical assay readouts, not human-judgement annotations."
- `rai:dataPreprocessingProtocol`: "RDKit canonical-SMILES round-trip with sanitisation; stereochemistry preserved; UniProt accessions resolved via REST; 1,043 unresolvable records (8.8%) dropped."
- `rai:dataPreprocessingImputation`: "None; missing potency kept as null; entries with neither DC50 nor Dmax excluded."
- `rai:dataPreprocessingManipulation`: "De-dup on (canonical SMILES, UniProt); 3-source majority vote; 12 unresolved 2-source conflicts retained as `source_conflict=true`."
- `rai:dataUseCases`: "(1) Cold-target benchmarking under held-out UniProt; (2) generalisation-decay studies; (3) E3-ligase scaffold transferability VHL↔CRBN; (4) few-shot transfer. NOT for direct clinical candidate selection."
- `rai:dataLimitation`: "(1) VHL+CRBN=87% of records; (2) kinases=47%; (3) binarisation discards potency gradient; (4) assay heterogeneity unencoded; (5) publication-positivity bias against inactives."
- `rai:dataBiases`: "(a) chemotype bias toward CRBN/VHL warheads; (b) oncology-target over-representation (BCR-ABL, BTK, AR, EGFR, BRD4); (c) lab-of-origin confounding — 3 labs >40% of records (`task14_within_target_cross_lab.json`)."
- `rai:dataSocialImpact`: "Positive: lowers barrier to ML-driven PROTAC design. Dual-use: only published research-stage compounds; no novel uplift beyond primary literature; no human-subject data."
- `rai:personalSensitiveInformation`: "None. SMILES, UniProt accessions, biochemical activity labels only; no PII, no patient-derived material."
- `rai:dataReleaseMaintenancePlan`: "CC-BY-4.0 via HuggingFace Datasets; versioned releases tagged in HF and `RELEASE_MANIFEST.md`; GitHub-issue-tracked corrections; long-term maintenance through NeurIPS 2025 reproducibility window (2025–2027)."

(Audit supplies no substantive content for `rai:dataAnnotationAnalysis`, `rai:dataAnnotationPerItemTime`, `rai:dataAnnotationTools` — record as not-applicable.)

## 3. Corrected `qin2026tpddb` bibtex entry
Source: `/workspace/PROTAC-Bench/results/v11/tpddb_timeline_audit.md` §7. End page and final DOI suffix were not verified in the audit; flagged for publisher-side verification (NAR vol. 54 issue D1, article ID 8285777, https://academic.oup.com/nar/article/54/D1/D1683/8285777).
```bibtex
@article{qin2026tpddb,
  title   = {{TPDdb}: the comprehensive database of targeted protein degrader},
  author  = {Qin, Xinran and Zhang, Yinpeng and Wang, Yajunzi and Zhang, Yintao
             and Jing, Jiachen and Zhang, Yuyuan and Xu, Gaoxiang and Teng, Haoping
             and Wang, Tianjun and Fu, Lei and Zhou, Ying and Liu, Xin and Zhu, Feng},
  journal = {Nucleic Acids Research},
  volume  = {54}, number = {D1},
  pages   = {D1683--<end-page TBD>},
  year    = {2026},
  doi     = {10.1093/nar/gkaf<suffix TBD>},
  note    = {Advance access 2025-10-14; verify end page and DOI suffix from publisher (article ID 8285777)}
}
```
Confirmed in the audit: 13-author list, start page D1683, vol/issue 54/D1, year 2026 (issue date). Unverified: end page, DOI suffix.
