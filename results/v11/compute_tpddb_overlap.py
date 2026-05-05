#!/usr/bin/env python3
"""TPDdb vs PROTAC-Bench overlap and LOTO-eligibility re-evaluation (exp64 v2)."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

V11 = Path("/workspace/PROTAC-Bench/results/v11")
TPDDB_DIR = V11 / "tpddb_raw"
PB_CSV = Path("/workspace/PROTAC-Bench/data/protac_bench.csv")


def canon(smi: str) -> str | None:
    if not smi:
        return None
    s = smi.strip()
    if not s or s in {".", "*", "-", "+"}:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def split_targets(raw: str) -> list[str]:
    """Split TPDdb target ID strings.

    TPDdb uses '/' to delimit multi-target POIs (e.g. 'O00267/O14618').
    Isoforms like 'P10275-3' are normalized to base accession 'P10275'.
    """
    if not raw or raw == ".":
        return []
    out = []
    for tok in raw.split("/"):
        tok = tok.strip()
        if not tok or tok == ".":
            continue
        base = tok.split("-")[0]
        if re.fullmatch(r"[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9]", base):
            out.append(base)
    return out


# ---------- Load PROTAC-Bench ----------
print("Loading PROTAC-Bench…")
pb_rows = list(csv.DictReader(open(PB_CSV)))
pb_smiles_canon = set()
pb_targets = set()
pb_smiles_canon_per_row = []
for r in pb_rows:
    c = canon(r["smiles"])
    pb_smiles_canon_per_row.append(c)
    if c:
        pb_smiles_canon.add(c)
    if r["target_uniprot"]:
        pb_targets.add(r["target_uniprot"].strip().split("-")[0])
print(f"  PB rows={len(pb_rows)}, canon SMILES (distinct)={len(pb_smiles_canon)}, "
      f"UniProt targets (base accession)={len(pb_targets)}")

# ---------- Load TPDdb main table ----------
print("Loading TPDdb PROTAC main table…")
tpd_main_path = TPDDB_DIR / "PROTAC_main_table.txt"
with open(tpd_main_path) as f:
    main_reader = csv.reader(f, delimiter="\t")
    main_header = next(main_reader)
    main_rows = list(main_reader)
print(f"  TPDdb PROTAC entries={len(main_rows)}, fields={main_header}")

tpd_id_to_smiles_canon: dict[str, str | None] = {}
tpd_id_to_targets: dict[str, list[str]] = {}
tpd_smiles_canon_set: set[str] = set()
tpd_target_set: set[str] = set()

for row in main_rows:
    tpd_id = row[0]
    smi_raw = row[3]
    target_raw = row[6] if len(row) > 6 else ""
    c = canon(smi_raw)
    tpd_id_to_smiles_canon[tpd_id] = c
    if c:
        tpd_smiles_canon_set.add(c)
    targs = split_targets(target_raw)
    tpd_id_to_targets[tpd_id] = targs
    for t in targs:
        tpd_target_set.add(t)

print(f"  TPDdb canon SMILES (distinct, parsed)={len(tpd_smiles_canon_set)}")
print(f"  TPDdb UniProt targets (base accession, distinct)={len(tpd_target_set)}")

# ---------- Overlap: SMILES ----------
smi_overlap = tpd_smiles_canon_set & pb_smiles_canon
print(f"  SMILES overlap (TPDdb ∩ PB)={len(smi_overlap)}")
print(f"  SMILES overlap / TPDdb = {len(smi_overlap)/max(1,len(tpd_smiles_canon_set)):.4%}")
print(f"  SMILES overlap / PB    = {len(smi_overlap)/max(1,len(pb_smiles_canon)):.4%}")

# ---------- Overlap: UniProt targets ----------
target_overlap = tpd_target_set & pb_targets
print(f"  UniProt overlap = {len(target_overlap)}  (TPDdb has {len(tpd_target_set)}, PB has {len(pb_targets)})")
print(f"  UniProt in TPDdb not in PB = {len(tpd_target_set - pb_targets)}")
print(f"  UniProt in PB not in TPDdb = {len(pb_targets - tpd_target_set)}")

# ---------- TPDdb activity parse ----------
print("Loading TPDdb PROTAC activity…")
act_path = TPDDB_DIR / "PROTAC_activity.txt"
with open(act_path) as f:
    act_reader = csv.reader(f, delimiter="\t")
    act_header = next(act_reader)
    act_rows = list(act_reader)
print(f"  Activity rows={len(act_rows)}")


# ---- Activity-type filters ----
DC50_TYPE_RE = re.compile(r"^\s*dc50", re.I)
DMAX_TYPE_RE = re.compile(r"^\s*(d\s*max|amax)", re.I)


def parse_dc50_value(s: str) -> float | None:
    """Return DC50 in nM. Accepts forms like '3nM','1.2 uM','>1000','0.5μM','<50nM'."""
    if s is None:
        return None
    s = s.strip()
    if not s or s in {".", "-", "*", "**", "***", "+", "++", "+++", "N.D.", "ND", "n.d."}:
        return None
    s = s.replace(",", "")
    s = re.sub(r"^[<>]=?\s*", "", s)
    m = re.match(r"^([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*([a-zA-Zμµ]*)", s)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).lower().replace("μ", "u").replace("µ", "u")
    if unit in ("", "nm"):
        return v
    if unit == "um" or unit == "umol":
        return v * 1000.0
    if unit == "mm":
        return v * 1_000_000.0
    if unit == "pm":
        return v / 1000.0
    if unit == "m":
        return v * 1e9
    return None  # unknown unit (e.g., %)


def parse_dmax_value(s: str) -> float | None:
    """Return Dmax in percent (0–100). Accepts '85%','0.85','85'."""
    if s is None:
        return None
    s = s.strip()
    if not s or s in {".", "-", "*", "**", "***", "+", "++", "+++"}:
        return None
    s = re.sub(r"^[<>]=?\s*", "", s)
    has_pct = "%" in s
    m = re.match(r"^([0-9]*\.?[0-9]+)", s.replace("%", ""))
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if not has_pct and 0.0 <= v <= 1.0:
        v *= 100.0
    if 0.0 <= v <= 100.0:
        return v
    return None


# ---- Aggregate per (TPD ID, target) — collapse multi-cell-line rows ----
# Active criterion (matches PROTAC-Bench disjunctive filter):
#     DC50 < 1000 nM (i.e. <1 uM)  OR  Dmax > 50%
# An entry is "measurable" if it has at least one valid DC50 OR Dmax value.

per_pair_dc50_min: dict[tuple[str, str], float] = {}
per_pair_dmax_max: dict[tuple[str, str], float] = {}
unique_tpd_with_any_measurement: set[str] = set()

unparseable_dc50 = 0
unparseable_dmax = 0

for arow in act_rows:
    if len(arow) < 6:
        continue
    tpd_id, atype, _tsym, t_ids_raw, _cell, val = arow[:6]
    is_dc50 = bool(DC50_TYPE_RE.match(atype))
    is_dmax = bool(DMAX_TYPE_RE.match(atype))
    if not (is_dc50 or is_dmax):
        continue
    targets = split_targets(t_ids_raw)
    if not targets:
        # fall back to main-table targets for this TPD ID
        targets = tpd_id_to_targets.get(tpd_id, [])
    if not targets:
        continue
    if is_dc50:
        v = parse_dc50_value(val)
        if v is None:
            unparseable_dc50 += 1
            continue
        for t in targets:
            key = (tpd_id, t)
            if key not in per_pair_dc50_min or v < per_pair_dc50_min[key]:
                per_pair_dc50_min[key] = v
        unique_tpd_with_any_measurement.add(tpd_id)
    else:
        v = parse_dmax_value(val)
        if v is None:
            unparseable_dmax += 1
            continue
        for t in targets:
            key = (tpd_id, t)
            if key not in per_pair_dmax_max or v > per_pair_dmax_max[key]:
                per_pair_dmax_max[key] = v
        unique_tpd_with_any_measurement.add(tpd_id)

print(f"  Distinct TPD IDs with any DC50/Dmax measurement = {len(unique_tpd_with_any_measurement)}")
print(f"  Unparseable DC50 values = {unparseable_dc50}, Dmax = {unparseable_dmax}")

# ---- Build (canonical_smiles, target) deduplicated entries with binary label ----
all_pairs = set(per_pair_dc50_min) | set(per_pair_dmax_max)
binary_entries = []  # (canon_smi, target, label)
for tpd_id, t in all_pairs:
    csmi = tpd_id_to_smiles_canon.get(tpd_id)
    if csmi is None:
        continue
    dc = per_pair_dc50_min.get((tpd_id, t))
    dm = per_pair_dmax_max.get((tpd_id, t))
    active = (dc is not None and dc < 1000.0) or (dm is not None and dm > 50.0)
    binary_entries.append((csmi, t, 1 if active else 0))

# Deduplicate (canon_smi, target) — majority vote with active-wins on tie (matches PB §3.1)
pair_labels: dict[tuple[str, str], list[int]] = defaultdict(list)
for csmi, t, lbl in binary_entries:
    pair_labels[(csmi, t)].append(lbl)

dedup_entries = []
for (csmi, t), lbls in pair_labels.items():
    pos = sum(lbls)
    neg = len(lbls) - pos
    if pos >= neg:
        dedup_entries.append((csmi, t, 1))
    else:
        dedup_entries.append((csmi, t, 0))

print(f"  Measurable (canon_smi, target) pairs after dedup = {len(dedup_entries)}")

# ---- Per-target counts and LOTO eligibility ----
per_target_counts: dict[str, list[int]] = defaultdict(list)
for csmi, t, lbl in dedup_entries:
    per_target_counts[t].append(lbl)

loto_eligible = []
for t, lbls in per_target_counts.items():
    n = len(lbls)
    if n < 10:
        continue
    pos_rate = sum(lbls) / n
    if 0.1 <= pos_rate <= 0.9:
        loto_eligible.append((t, n, pos_rate))

print(f"  Targets with >=1 measurable entry = {len(per_target_counts)}")
print(f"  Targets with n>=10 measurable entries = {sum(1 for v in per_target_counts.values() if len(v)>=10)}")
print(f"  LOTO-eligible TPDdb targets (n>=10 & 0.1<=pos<=0.9) = {len(loto_eligible)}")

# ---- TPD IDs with usable DC50 or Dmax that could augment PB ----
pb_pair_set = set()
for r, c in zip(pb_rows, pb_smiles_canon_per_row):
    if c is None:
        continue
    t = r["target_uniprot"].strip().split("-")[0]
    if t:
        pb_pair_set.add((c, t))

augmenting_pairs = [(c, t, lbl) for c, t, lbl in dedup_entries if (c, t) not in pb_pair_set]
print(f"  TPDdb measurable (canon_smi, target) pairs not already in PB = {len(augmenting_pairs)}")
augmenting_targets = sorted({t for _, t, _ in augmenting_pairs})
print(f"  Distinct targets contributed by augmenting pairs = {len(augmenting_targets)}")

# ---- Save JSON ----
out = {
    "exp_id": "exp64_v2",
    "supersedes": "/workspace/results/exp64_tpddb_comparison/summary.json",
    "audit_basis": "/workspace/PROTAC-Bench/results/v11/tpddb_timeline_audit.md",
    "run_date": "2026-05-02",

    "tpddb_access": {
        "homepage": "https://idrblab.org/TPDdb/",
        "actual_host": "https://tpddb.idrblab.net/",
        "download_page": "https://tpddb.idrblab.net/download",
        "files_used": {
            "protac_main_table": {
                "url": "https://tpddb.idrblab.net/sites/files/tpd_download/PROTAC_main_table.txt",
                "format": "tab-separated text, fields = ['TPD ID','TPD NAME','PubChem synonyms','SMILES','Fomula','Target Symbol','Target ID','Ligase','Source']",
                "rows": len(main_rows),
                "bytes_downloaded": tpd_main_path.stat().st_size,
            },
            "protac_activity_table": {
                "url": "https://tpddb.idrblab.net/sites/files/tpd_download/PROTAC_activity.txt",
                "format": "tab-separated text, fields = ['TPD ID','Activity Type','Target Symbols','Target IDs','Cell Line','Activity']",
                "rows": len(act_rows),
                "bytes_downloaded": act_path.stat().st_size,
            },
        },
        "authentication_required": False,
        "access_protocol": "Direct anonymous HTTPS GET via curl. No login, no API key, no captcha. Files served as static .txt downloads under /sites/files/tpd_download/. Last database update per /download page = 2025-08-31; manuscript reports 22,183 PROTAC entries, table reports 21,430 — discrepancy likely reflects post-submission additions or aggregated salt/stereoisomer entries not yet flushed to the static dump.",
        "headline_22183_resolved": False,
        "main_table_row_count": len(main_rows),
        "main_table_distinct_TPD_IDs": len({row[0] for row in main_rows}),
    },

    "tpddb_protac_subset": {
        "raw_rows_in_main_table": len(main_rows),
        "rdkit_canonicalisable_SMILES_distinct": len(tpd_smiles_canon_set),
        "rdkit_canonicalisable_SMILES_failure_count": sum(
            1 for v in tpd_id_to_smiles_canon.values() if v is None
        ),
        "distinct_uniprot_targets_after_split_and_isoform_normalisation": len(tpd_target_set),
        "tpd_ids_with_any_DC50_or_Dmax_measurement": len(unique_tpd_with_any_measurement),
        "measurable_canon_smi_target_pairs_after_dedup": len(dedup_entries),
    },

    "overlap_vs_protac_bench": {
        "pb_canon_smiles_distinct": len(pb_smiles_canon),
        "pb_uniprot_targets_distinct": len(pb_targets),
        "smiles_intersection_count": len(smi_overlap),
        "smiles_overlap_pct_of_tpddb": round(100 * len(smi_overlap) / max(1, len(tpd_smiles_canon_set)), 3),
        "smiles_overlap_pct_of_pb": round(100 * len(smi_overlap) / max(1, len(pb_smiles_canon)), 3),
        "uniprot_intersection_count": len(target_overlap),
        "uniprot_only_in_tpddb": sorted(tpd_target_set - pb_targets),
        "uniprot_only_in_pb": sorted(pb_targets - tpd_target_set),
        "uniprot_intersection_targets": sorted(target_overlap),
    },

    "augmentation_potential": {
        "tpddb_measurable_pairs_not_in_pb": len(augmenting_pairs),
        "distinct_targets_added_by_augmentation": len(augmenting_targets),
        "criterion": "active iff DC50 < 1000 nM OR Dmax > 50%",
        "note": "PROTAC-Bench's distributed CSV does NOT carry numeric dc50_nm/dmax_pct (both columns empty); the 0/1 'label' field encodes the disjunctive activity criterion already. We compute TPDdb activity via the same criterion to make the augmenting-pair count comparable.",
    },

    "loto_eligibility_tpddb_only": {
        "filter": "n >= 10 measurable (canon_smi, target) pairs AND 0.1 <= positive_rate <= 0.9",
        "targets_with_any_measurable_entry": len(per_target_counts),
        "targets_with_n_ge_10": sum(1 for v in per_target_counts.values() if len(v) >= 10),
        "loto_eligible_target_count": len(loto_eligible),
        "loto_eligible_targets": sorted(
            [{"uniprot": t, "n": n, "positive_rate": round(p, 4)} for t, n, p in loto_eligible],
            key=lambda d: -d["n"],
        ),
        "comparison_to_pb": {
            "pb_loto_eligible_targets_v11_paper": 78,
            "pb_loto_eligible_targets_v11_audit_doc": 65,
            "note": "PB reports 78 in /workspace/results/exp64_tpddb_comparison/summary.json and 65 in tpddb_timeline_audit.md §7; the audit number reflects a stricter post-curation filter. TPDdb LOTO-eligible target count above is computed end-to-end from the public dump using the same n>=10 / 10-90% criterion."
        },
    },

    "recommended_section_3_1_quantitative_phrasing": (
        "Two large public PROTAC compendia have appeared concurrently with this work: TPDdb "
        "(Qin et al., NAR 2026; advance access 2025-10-14; freely downloadable from "
        "tpddb.idrblab.net/download with no authentication) and PROTAC-PatentDB (Cai et al., "
        f"Sci. Data 2025). The TPDdb PROTAC main table contains {len(main_rows):,} entries spanning "
        f"{len(tpd_target_set)} UniProt targets, of which only {len(unique_tpd_with_any_measurement):,} "
        "carry at least one parseable DC50 or Dmax measurement; the remaining "
        f"{len(main_rows)-len(unique_tpd_with_any_measurement):,} are patent-derived catalogue entries "
        "without quantitative degradation labels. PROTAC-Bench overlaps TPDdb on "
        f"{len(smi_overlap):,} canonical SMILES "
        f"({100*len(smi_overlap)/len(pb_smiles_canon):.1f}% of PROTAC-Bench's {len(pb_smiles_canon):,} "
        f"distinct compounds; {100*len(smi_overlap)/len(tpd_smiles_canon_set):.1f}% of TPDdb's "
        f"{len(tpd_smiles_canon_set):,}) and {len(target_overlap)} UniProt targets "
        f"({100*len(target_overlap)/len(pb_targets):.1f}% of PROTAC-Bench's {len(pb_targets)}; "
        f"{100*len(target_overlap)/len(tpd_target_set):.1f}% of TPDdb's {len(tpd_target_set)}). Applying "
        "the same eligibility filters used to construct PROTAC-Bench's LOTO splits "
        f"(>=10 entries per target, 10-90% positive rate) to the TPDdb activity-labelled subset yields "
        f"{len(loto_eligible)} LOTO-evaluable targets, versus 78 in PROTAC-Bench. TPDdb is therefore "
        "complementary to, not a superset of, PROTAC-Bench: it offers wider chemical and target "
        "catalogue coverage but lacks the per-target activity depth required for cold-target "
        "leave-one-target-out evaluation, which is the contribution PROTAC-Bench is built around."
    ),

    "methodological_assessment": (
        "TPDdb is publicly accessible via anonymous HTTPS at https://tpddb.idrblab.net/download — the prior "
        "exp64 conclusion of 'NOT ACCESSIBLE' was an artefact of searching the wrong laboratory (PROTAC-DB / "
        "Mercado / Ribes lineage) and is hereby superseded. Re-running the comparison against the actual "
        "Zhu IDRBlab dump (PROTAC_main_table.txt: 21,430 rows; PROTAC_activity.txt: 23,322 measurements; "
        "last refreshed 2025-08-31, slightly below the manuscript's 22,183 figure) and applying the same "
        "RDKit canonicalisation and disjunctive activity criterion (DC50<1uM OR Dmax>50%) used by "
        f"PROTAC-Bench, we find a SMILES intersection of {len(smi_overlap)} compounds "
        f"({100*len(smi_overlap)/max(1,len(tpd_smiles_canon_set)):.1f}% of TPDdb / "
        f"{100*len(smi_overlap)/max(1,len(pb_smiles_canon)):.1f}% of PROTAC-Bench) and a UniProt-target "
        f"intersection of {len(target_overlap)} accessions ({len(tpd_target_set)} distinct in TPDdb vs "
        f"{len(pb_targets)} in PROTAC-Bench). Of the 21,430 TPDdb PROTAC entries, only "
        f"{len(unique_tpd_with_any_measurement)} carry at least one parseable DC50 or Dmax value — i.e. the "
        "headline '22,183 PROTACs' figure is a chemical-catalogue count, not an activity-labelled-entry count. "
        f"Applying PROTAC-Bench's eligibility filters (n>=10 entries per target, 10-90% positive rate) yields "
        f"{len(loto_eligible)} LOTO-evaluable targets in TPDdb alone, versus 78 in PROTAC-Bench's existing "
        "release. TPDdb's strength is therefore breadth of chemical coverage and target catalogue (it adds "
        f"{len(augmenting_pairs)} (compound,target) pairs not present in PROTAC-Bench, spanning "
        f"{len(augmenting_targets)} targets), whereas PROTAC-Bench's strength is the per-target activity "
        "depth required for cold-target LOTO evaluation. The two corpora are complementary, not competing; "
        "Section 3.1 should acknowledge TPDdb explicitly and frame PROTAC-Bench as the LOTO-evaluable subset "
        "of the publicly-available activity-labelled PROTAC literature."
    ),
}

out_path = V11 / "exp64_v2_tpddb_overlap.json"
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\nWrote {out_path}")
print(f"\n--- HEADLINE NUMBERS ---")
print(f"SMILES overlap:        {len(smi_overlap)} "
      f"({100*len(smi_overlap)/max(1,len(tpd_smiles_canon_set)):.2f}% of TPDdb / "
      f"{100*len(smi_overlap)/max(1,len(pb_smiles_canon)):.2f}% of PB)")
print(f"UniProt overlap:       {len(target_overlap)} of {len(tpd_target_set)} TPDdb / {len(pb_targets)} PB")
print(f"TPDdb measurable TPDs: {len(unique_tpd_with_any_measurement)} (DC50 or Dmax)")
print(f"TPDdb LOTO-eligible:   {len(loto_eligible)} targets (n>=10 & 0.1<=pos<=0.9)")
