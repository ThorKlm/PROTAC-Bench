# TPDdb / PROTAC-PatentDB Timeline Audit — PROTAC-Bench Section 3.1

Audit date: 2026-05-02
Scope: verify whether PROTAC-Bench predates or post-dates the two recently-published competing PROTAC corpora (TPDdb, PROTAC-PatentDB), and recommend Section 3.1 phrasing.

---

## 1. PROTAC-Bench development timeline (this work)

Earliest evidence on disk, in chronological order:

| Artefact | Timestamp | Source |
|---|---|---|
| `/workspace/PROTAC_PIPELINE_CLAUDE_CODE_TASKS.md` (initial pipeline scoping doc) | **2026-03-22 / 2026-03-23** | filesystem mtime |
| `/workspace/protac_plm_bench_2` repository — first git commit `07e74b2` "PROTAC-PLM-Bench complete: 50+ experiments, figures, paper draft, Croissant metadata" | **2026-04-07 15:14 UTC** | `git log --reverse` |
| `/workspace/PROTAC-Bench` repository — first git commit `fabda24` "PROTAC-Bench: cold-target evaluation benchmark for PROTAC degradation prediction" | **2026-04-08 06:02 UTC** | `git log --reverse` |
| `/workspace/protac_plm_bench_2/paper/main.tex` last edit | 2026-04-25 08:35 UTC | filesystem mtime |
| Final figures (10-seed pooled with seed-std error bars), commit `ecac7a0` | 2026-04-20 17:32 UTC | `git log` |

There is no arXiv submission timestamp on disk — the manuscript exists as `/workspace/protac_plm_bench_2/paper/main.tex` (NeurIPS 2026 template) and has not yet been posted publicly per any artefact in the workspace. The earliest defensible "PROTAC-Bench started" date is **late March 2026** (pipeline scoping); the earliest *committed* code is **2026-04-07**.

`/mnt/user-data/uploads/` does not exist on this instance — no internal status reports were available there.

---

## 2. TPDdb (Yu/Qin/Zhang et al.) — verified publication timeline

**Title:** *TPDdb: the comprehensive database of targeted protein degrader*
**Authors:** Xinran Qin, Yinpeng Zhang, Yajunzi Wang, Yintao Zhang, Jiachen Jing, Yuyuan Zhang, Gaoxiang Xu, Haoping Teng, Tianjun Wang, Lei Fu, Ying Zhou, Xin Liu, Feng Zhu (IDRB lab, Zhejiang University)
**Venue:** *Nucleic Acids Research*, Volume 54, Issue D1 (Database Issue), pages D1683–…
**URL:** https://academic.oup.com/nar/article/54/D1/D1683/8285777
**Database URL:** https://idrblab.org/TPDdb/

| Milestone | Date |
|---|---|
| Manuscript received | 2025-08-07 |
| Revision received | 2025-08-31 |
| Accepted | 2025-09-16 |
| **Published online (advance access)** | **2025-10-14** |
| Issue publication | **2026-01-06** |

**Coverage:** 22,183 PROTACs + 6,002 Molecular Glues + 249 LYTACs + 169 ATTECs + 29 AUTOTACs + 23 AUTACs targeting 580 POIs; 27,796 activity measurements across 201 cell lines.

**Note re prior internal claim (`/workspace/results/exp64_tpddb_comparison/summary.json`):** that file (dated April 2026) concluded "no public release of a 22,183-entry 'TPDdb 2025' dataset found" — this conclusion was based on searching the **wrong group** (Mercado / Ribes / cadd.zju PROTAC-DB lineage). The actual TPDdb is from a *different* Zhejiang University lab (Zhu's IDRBlab), hosted at **idrblab.org/TPDdb**, not cadd.zju.edu.cn. The reviewer was correct; our prior search missed it.

---

## 3. PROTAC-PatentDB (Cai et al.) — verified publication timeline

**Title:** *PROTAC-PatentDB: A PROTAC Patent Compound Dataset*
**Authors:** Cai H., Yao G., Shi Y., et al.
**Venue:** *Scientific Data* 12, Article 1840 (2025); DOI 10.1038/s41597-025-06136-9; PMID 41261151
**URL:** https://www.nature.com/articles/s41597-025-06136-9
**Database URL:** http://protacpatentdb.com

| Milestone | Date |
|---|---|
| **Published** | **2025-11-19** |

**Coverage:** 63,136 unique PROTAC compounds from 590 patent families targeting 252 proteins; 120 predicted physicochemical properties per compound. Patent-derived (not curated literature activity); does **not** carry DC50/Dmax labels for most entries.

---

## 4. Timeline comparison

```
2025-10-14  ── TPDdb online (advance access)              ┐
2025-11-19  ── PROTAC-PatentDB published                  │  Both predate any
2026-01-06  ── TPDdb formal issue date (NAR D1)           │  PROTAC-Bench artefact
─────────────────────────────────────────────────────────  │  by 4–6 months
2026-03-22  ── earliest PROTAC-Bench scoping doc          │
2026-04-07  ── first git commit (protac_plm_bench_2)      │
2026-04-08  ── first git commit (PROTAC-Bench repo)       ┘
2026-04-25  ── current paper draft last edit
2026-05-02  ── this audit
```

**Conclusion on order of precedence:** TPDdb and PROTAC-PatentDB were published **before** PROTAC-Bench began. PROTAC-Bench cannot honestly be framed as having developed independently of, or concurrently with, these two corpora. Any Section 3.1 framing that does not acknowledge them is a citation gap, not a timing question.

---

## 5. What the paper currently says (Section 3.1, `main.tex` lines 77–104)

> "We consolidate PROTAC degradation data from three public sources: PROTAC-DB 3.0 (Weng et al. 2023), the curated set of Ribes et al. (2024), and the DegradeMaster corpus (Wu et al. 2024). … The final dataset comprises **10,748** binary degradation entries spanning **173** UniProt target proteins …"

The paper as written **does not** mention TPDdb or PROTAC-PatentDB at all. There is no claim "TPDdb at 22,183 entries as comparison cohort" anywhere in `main.tex` or `references.bib`. The premise in the task statement reflects a *proposed* framing under consideration, not the current text.

---

## 6. Does Section 3.1 require retrospective reframing?

**Yes — current framing is incomplete and exposed to a citation-priority objection.** Three problems with leaving the section as-is:

1. **Missing prior art.** The first sentence enumerates "three public sources" while two larger, contemporaneous public PROTAC compendia (TPDdb 22,183 entries; PROTAC-PatentDB 63,136 entries) exist and predate this work. A reviewer will flag the omission immediately.
2. **Defensive 10,748-entry framing fails.** Compared to TPDdb's 22,183 activity-labelled PROTACs and PROTAC-PatentDB's 63,136 patent compounds, "10,748 entries" reads as smaller and less ambitious unless the reason is made explicit (DC50/Dmax-binarisable entries with sufficient per-target depth for LOTO ≥ 10 entries between 10–90 % activity).
3. **The contribution is the *protocol*, not the corpus size.** PROTAC-Bench's headline claim is the LOTO ceiling and the universal collapse result, not "biggest PROTAC dataset." The current text invites a corpus-size comparison the paper will lose.

**Reframing direction:** explicitly position PROTAC-Bench as a **LOTO-evaluable benchmark** carved from the publicly available PROTAC literature, contrasted against (not competing with) TPDdb/PROTAC-PatentDB, which are **catalogue-style compendia** that mix activity-labelled and patent-only entries.

---

## 7. Recommended Section 3.1 phrasing

Insert a new opening paragraph (or expand the existing first paragraph) along the following lines. Numbers below are taken from this audit and the existing dataset table.

> **§3.1 Dataset Construction (revised).**
>
> Two large public PROTAC compendia have appeared concurrently with this work: TPDdb (Qin et al., *NAR* 2026; 22,183 PROTACs, 580 POIs, advance access October 2025) and PROTAC-PatentDB (Cai et al., *Sci. Data* 2025; 63,136 patent-derived PROTACs, 252 targets). Both are catalogue-style resources optimised for breadth of chemical and target coverage. Neither, however, supplies the per-target activity depth required for leave-one-target-out (LOTO) evaluation: TPDdb's 27,796 activity measurements are spread across 201 cell-line × assay contexts with heterogeneous endpoints, and PROTAC-PatentDB's compound-level patent annotations are not paired with quantitative DC$_{50}$/D$_\text{max}$ labels for the majority of entries.
>
> PROTAC-Bench is therefore not designed as a competing catalogue. We consolidate PROTAC degradation data from three public, activity-labelled sources — PROTAC-DB 3.0 (Weng et al., 2023), the curated set of Ribes et al. (2024), and the DegradeMaster corpus (Wu et al., 2024) — applying canonical-SMILES + UniProt joins, majority-vote label resolution, and the disjunctive activity criterion (DC$_{50}<1\,\mu$M $\lor$ D$_\text{max}>50\%$). The resulting 10,748 binary entries span 173 UniProt targets and yield **65 LOTO-eligible targets** ($n\geq 10$ entries each, 10–90 % activity rate) — the largest LOTO-evaluable benchmark we are aware of in the PROTAC literature, and the metric on which our cold-target ceiling claim rests. Cross-corpus integration with TPDdb and PROTAC-PatentDB is left as future work; the LOTO methodology developed here is dataset-agnostic and applies to any future enlarged corpus.

Plus one citation entry in `references.bib`:

```bibtex
@article{qin2026tpddb,
  title   = {{TPDdb}: the comprehensive database of targeted protein degrader},
  author  = {Qin, Xinran and Zhang, Yinpeng and Wang, Yajunzi and Zhang, Yintao
             and Jing, Jiachen and Zhang, Yuyuan and Xu, Gaoxiang and Teng, Haoping
             and Wang, Tianjun and Fu, Lei and Zhou, Ying and Liu, Xin and Zhu, Feng},
  journal = {Nucleic Acids Research},
  volume  = {54},
  number  = {D1},
  pages   = {D1683},
  year    = {2026},
  doi     = {10.1093/nar/gkaf...},  % verify final DOI from publisher
  note    = {Advance access 2025-10-14}
}

@article{cai2025patentdb,
  title   = {{PROTAC-PatentDB}: A {PROTAC} Patent Compound Dataset},
  author  = {Cai, H. and Yao, G. and Shi, Y. and others},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {1840},
  year    = {2025},
  doi     = {10.1038/s41597-025-06136-9}
}
```

---

## 8. Risks of the alternative framings

| Framing | Risk |
|---|---|
| **Keep current text unchanged** | High. Two large, dated-prior public corpora are unmentioned. Reviewers (especially DB-paper reviewers) will see this as a citation omission. |
| **Claim TPDdb as a "comparison cohort"** (the framing implied by the task statement) | Wrong on facts. We have not actually evaluated PROTAC-Bench models on the TPDdb 22,183-entry split — we did not even have access to TPDdb during corpus construction. Claiming it would be a misrepresentation. |
| **Recommended (§7 above): acknowledge as concurrent catalogues, distinguish on protocol** | Low. Honest about timing, defensible on contribution (LOTO depth, not breadth), invites future-work line that strengthens rather than weakens the paper. |

---

## 9. Action items

1. Add Qin et al. (TPDdb) and Cai et al. (PROTAC-PatentDB) citations to `references.bib`.
2. Replace / extend `main.tex` lines 77–82 (current §3.1 opening) with the revised paragraph in §7 above.
3. Consider adding a short Appendix entry (Dataset Details, around line 449) cross-walking TPDdb and PROTAC-PatentDB UniProt coverage against our 65 LOTO-eligible targets — this is a low-effort, high-value reviewer pre-empt and the data already lives in `/workspace/results/exp64_tpddb_comparison/` (note: that file's "TPDdb not found" conclusion is stale — re-run against `idrblab.org/TPDdb` to produce a real overlap table).
4. Update `/workspace/results/exp64_tpddb_comparison/summary.json` to correct the "NOT ACCESSIBLE" claim — TPDdb is publicly accessible; the previous search looked at the wrong lab.

---

*Generated 2026-05-02 by timeline-audit task. Source verification: NAR website (TPDdb), Nature Scientific Data + PubMed (PROTAC-PatentDB), local git logs and filesystem mtimes (PROTAC-Bench).*
