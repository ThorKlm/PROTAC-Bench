#!/usr/bin/env python3
"""Generate the three missing fold-assignment artifacts that complete the
PROTAC-Bench evaluation suite alongside data/loto_folds.json.

Outputs:
  data/lofo_folds.json                  Leave-one-family-out (22 families, 61 targets)
  data/cross_lab_folds.json             Within-target cross-lab (36 targets, paper holdout)
  data/temporal_prospective_folds.json  Train pub_year < 2023, test pub_year == 2024

The fold-generation logic mirrors the eligibility filters used by the
benchmark's evaluation scripts:
  * LOFO uses the FAMILY_MAP defined in robustness/lofo.py and the
    LOTO eligibility filter (>=10 entries, activity rate in [0.1, 0.9]).
    LOFO eligibility is restricted to LOTO-eligible targets that are
    also assigned to a *named* family in FAMILY_MAP (excludes 4
    unmapped singletons O43924, Q00534, Q9Y2I7, Q9Y616).
  * Cross-lab uses the eligibility filter from
    scripts/exp42_task14_within_target_cross_lab.py:
      MIN_ENTRIES_PER_TARGET = 20
      MIN_PAPERS_PER_TARGET  = 3
      MIN_ENTRIES_PER_PAPER  = 5
      Both classes present in target.
  * Temporal uses the pub_year column from the master dataframe to
    define a strict pre-2023 vs 2024 split (2023 entries excluded as
    a temporal gap year).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/workspace/PROTAC-Bench")
DATA_DIR = REPO_ROOT / "data"
PUB_CSV = DATA_DIR / "protac_bench.csv"
MASTER_CSV = Path("/workspace/results/exp42_metadata_features/master_df.csv")

sys.path.insert(0, str(REPO_ROOT))
from src.data_utils import get_eligible_targets  # noqa: E402
from robustness.lofo import FAMILY_MAP  # noqa: E402

# Cross-lab eligibility constants (mirror exp42_task14_within_target_cross_lab.py)
MIN_ENTRIES_PER_TARGET = 20
MIN_PAPERS_PER_TARGET = 3
MIN_ENTRIES_PER_PAPER = 5

# Temporal cutoffs (per task spec)
TRAIN_YEAR_MAX_EXCLUSIVE = 2023  # train: pub_year < 2023
TEST_YEAR = 2024                  # test:  pub_year == 2024


def load_aligned_frames():
    pub = pd.read_csv(PUB_CSV)
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    if len(pub) != len(master):
        raise RuntimeError(
            f"Row count mismatch: protac_bench.csv has {len(pub)} rows, "
            f"master_df.csv has {len(master)} rows — fold indices would not "
            f"align with the published CSV.")
    if not (master["pb_index"].values == np.arange(len(master))).all():
        raise RuntimeError("master_df.csv pb_index is not a 0..N-1 sequence.")
    if not (master["target_uniprot"].fillna("").values
            == pub["target_uniprot"].fillna("").values).all():
        raise RuntimeError(
            "target_uniprot mismatch between master_df.csv and protac_bench.csv "
            "— alignment cannot be trusted.")
    if not (master["label"].values == pub["label"].values).all():
        raise RuntimeError(
            "label mismatch between master_df.csv and protac_bench.csv "
            "— alignment cannot be trusted.")
    return pub, master


def build_lofo_folds(pub: pd.DataFrame) -> dict:
    eligible = set(get_eligible_targets(pub))
    families: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for tgt in sorted(eligible):
        fam = FAMILY_MAP.get(tgt)
        if fam is None:
            unmapped.append(tgt)
            continue
        families.setdefault(fam, []).append(tgt)

    out = {
        "_meta": {
            "split_type": "leave_one_family_out",
            "eligibility_source": "FAMILY_MAP in robustness/lofo.py",
            "loto_eligibility": {
                "min_entries": 10,
                "min_activity_rate": 0.1,
                "max_activity_rate": 0.9,
            },
            "n_families": len(families),
            "n_targets": sum(len(t) for t in families.values()),
            "unmapped_singletons_excluded": sorted(unmapped),
            "fold_index_basis": "row index into data/protac_bench.csv (0-based)",
        },
    }

    n_total = len(pub)
    for fam, targets in sorted(families.items()):
        mask = pub["target_uniprot"].isin(targets).values
        test_idx = np.where(mask)[0].astype(int).tolist()
        out[fam] = {
            "family": fam,
            "targets": sorted(targets),
            "n_targets": len(targets),
            "n_entries": int(mask.sum()),
            "activity_rate": round(float(pub.loc[mask, "label"].mean()), 4),
            "test_indices": test_idx,
        }
        assert len(test_idx) <= n_total
    return out


def build_cross_lab_folds(pub: pd.DataFrame, master: pd.DataFrame) -> dict:
    has_doi = master["doi"].notna()

    eligible_targets = []
    for uid, sub in master[has_doi].groupby("target_uniprot"):
        if pd.isna(uid):
            continue
        n = len(sub)
        npapers = sub["doi"].nunique()
        rate = float(sub["label"].mean())
        if n < MIN_ENTRIES_PER_TARGET or npapers < MIN_PAPERS_PER_TARGET:
            continue
        if rate <= 0.0 or rate >= 1.0:
            continue
        eligible_targets.append(uid)
    eligible_targets.sort()

    out = {
        "_meta": {
            "split_type": "within_target_cross_lab",
            "eligibility": {
                "min_entries_per_target": MIN_ENTRIES_PER_TARGET,
                "min_papers_per_target": MIN_PAPERS_PER_TARGET,
                "min_entries_per_paper": MIN_ENTRIES_PER_PAPER,
                "both_classes_required_in_holdout_paper": True,
            },
            "metadata_source": (
                "doi column from /workspace/results/exp42_metadata_features/"
                "master_df.csv (built by scripts/exp42_step0_build_master.py)"
            ),
            "n_targets": len(eligible_targets),
            "fold_index_basis": "row index into data/protac_bench.csv (0-based)",
            "holdout_strategy": (
                "For each (target, paper) where the paper has >=5 entries and "
                "both classes present, test_indices = rows with that target "
                "AND that doi; train = all remaining rows."
            ),
        }
    }

    for uid in eligible_targets:
        target_mask = (master["target_uniprot"] == uid) & has_doi
        sub = master[target_mask]
        n_entries = int(len(sub))
        n_papers_total = int(sub["doi"].nunique())
        rate = round(float(sub["label"].mean()), 4)

        papers = []
        for doi, pg in sub.groupby("doi"):
            n = int(len(pg))
            n_classes = int(pg["label"].nunique())
            if n >= MIN_ENTRIES_PER_PAPER and n_classes >= 2:
                test_idx = pg["pb_index"].astype(int).tolist()
                papers.append({
                    "doi": str(doi),
                    "n_entries": n,
                    "activity_rate": round(float(pg["label"].mean()), 4),
                    "test_indices": sorted(test_idx),
                })
        papers.sort(key=lambda p: p["doi"])

        out[uid] = {
            "target_uniprot": uid,
            "n_entries": n_entries,
            "n_papers_total": n_papers_total,
            "n_qualifying_papers": len(papers),
            "activity_rate": rate,
            "papers": papers,
        }
    return out


def build_temporal_folds(pub: pd.DataFrame, master: pd.DataFrame) -> dict:
    pub_year = master["pub_year"]
    train_mask = (pub_year < TRAIN_YEAR_MAX_EXCLUSIVE).fillna(False).values
    test_mask = (pub_year == TEST_YEAR).fillna(False).values

    train_idx = np.where(train_mask)[0].astype(int).tolist()
    test_idx = np.where(test_mask)[0].astype(int).tolist()

    train_targets = set(pub.loc[train_mask, "target_uniprot"].dropna().unique())
    test_targets = set(pub.loc[test_mask, "target_uniprot"].dropna().unique())

    train_year_dist = (master.loc[train_mask, "pub_year"]
                       .value_counts().sort_index().to_dict())
    train_year_dist = {int(y): int(c) for y, c in train_year_dist.items()}

    out = {
        "_meta": {
            "split_type": "temporal_prospective",
            "train_filter": f"pub_year < {TRAIN_YEAR_MAX_EXCLUSIVE}",
            "test_filter": f"pub_year == {TEST_YEAR}",
            "metadata_source": (
                "pub_year column from /workspace/results/exp42_metadata_features/"
                "master_df.csv (built by scripts/exp42_step0_build_master.py from "
                "data/preprocessed/doi_years.json)"
            ),
            "fold_index_basis": "row index into data/protac_bench.csv (0-based)",
            "n_rows_total": int(len(master)),
            "n_rows_with_pub_year": int(pub_year.notna().sum()),
            "n_rows_excluded_year_2023_gap": int((pub_year == 2023).sum()),
            "n_rows_excluded_no_pub_year": int(pub_year.isna().sum()),
            "train_year_distribution": train_year_dist,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "train_activity_rate": (round(float(pub.loc[train_mask, "label"].mean()), 4)
                                     if train_mask.any() else None),
            "test_activity_rate": (round(float(pub.loc[test_mask, "label"].mean()), 4)
                                    if test_mask.any() else None),
            "n_train_targets": len(train_targets),
            "n_test_targets": len(test_targets),
            "n_shared_targets": len(train_targets & test_targets),
            "n_novel_test_targets": len(test_targets - train_targets),
        },
        "train_indices": train_idx,
        "test_indices": test_idx,
    }
    return out


def write_json(path: Path, payload: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path.stat().st_size


def main():
    pub, master = load_aligned_frames()
    print(f"Loaded {len(pub)} rows from protac_bench.csv (aligned with master_df.csv)")

    print("\n=== LOFO ===")
    lofo = build_lofo_folds(pub)
    n_fams = lofo["_meta"]["n_families"]
    n_tgts = lofo["_meta"]["n_targets"]
    print(f"Families: {n_fams}, Targets: {n_tgts}, "
          f"Excluded singletons: {lofo['_meta']['unmapped_singletons_excluded']}")
    sz = write_json(DATA_DIR / "lofo_folds.json", lofo)
    print(f"Wrote data/lofo_folds.json ({sz} bytes)")

    print("\n=== CROSS-LAB ===")
    cross = build_cross_lab_folds(pub, master)
    n_targets = cross["_meta"]["n_targets"]
    n_papers = sum(v["n_qualifying_papers"] for k, v in cross.items() if k != "_meta")
    print(f"Eligible targets: {n_targets}, Total qualifying paper holdouts: {n_papers}")
    sz = write_json(DATA_DIR / "cross_lab_folds.json", cross)
    print(f"Wrote data/cross_lab_folds.json ({sz} bytes)")

    print("\n=== TEMPORAL ===")
    temporal = build_temporal_folds(pub, master)
    print(f"Train (pub_year < 2023): {temporal['_meta']['n_train']}")
    print(f"Test  (pub_year == 2024): {temporal['_meta']['n_test']}")
    print(f"Shared targets: {temporal['_meta']['n_shared_targets']}, "
          f"Novel test targets: {temporal['_meta']['n_novel_test_targets']}")
    sz = write_json(DATA_DIR / "temporal_prospective_folds.json", temporal)
    print(f"Wrote data/temporal_prospective_folds.json ({sz} bytes)")


if __name__ == "__main__":
    main()
