"""
check_dataset.py

Pre-validation checker for Stage 1: Dataset Preparation (VinBigData subset).

Purpose:
    Verify that the local VinBigData subset has all required folders, image files,
    annotation CSV files, and correct structure before running the main validation
    script (scripts/01_validate_dataset.py).

What this script does NOT do:
    - Does not train any model
    - Does not create train/val/test splits
    - Does not create labeled/unlabeled splits
    - Does not generate pseudo-labels
    - Does not convert annotations to YOLO, COCO, or Pascal VOC
    - Does not modify any raw data file

Run from the project root:
    python scripts/check_dataset.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data" / "raw" / "vinbigdata"
IMAGES_DIR   = DATA_ROOT / "images"
ANN_DIR      = DATA_ROOT / "annotations"
TRAIN_CSV    = ANN_DIR / "train.csv"

OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_LOGS    = PROJECT_ROOT / "outputs" / "logs"

REPORT_PATH = OUTPUTS_REPORTS / "check_dataset_report.md"
LOG_PATH    = OUTPUTS_LOGS    / "check_dataset.log"

# Columns confirmed by manual inspection of the actual VinBigData annotation files
REQUIRED_ANN_COLUMNS = [
    "image_id", "class_name", "class_id",
    "rad_id", "x_min", "y_min", "x_max", "y_max",
]

NUM_PARTS = 16  # part_001 ... part_016


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure a logger that writes to both stdout and the log file."""
    OUTPUTS_LOGS.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("check_dataset")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Check 1: Folder existence
# ---------------------------------------------------------------------------

def check_folders(logger: logging.Logger) -> dict:
    """
    Verify that the three required directories exist.

    These folders must be present before any other check can proceed:
      - data/raw/vinbigdata             (dataset root)
      - data/raw/vinbigdata/images      (DICOM files)
      - data/raw/vinbigdata/annotations (CSV annotation files)
    """
    logger.info("=== CHECK 1: Folder existence ===")

    folders = {
        "dataset_root": DATA_ROOT,
        "images_dir":   IMAGES_DIR,
        "ann_dir":      ANN_DIR,
    }

    results = {}
    all_pass = True

    for name, path in folders.items():
        exists = path.exists() and path.is_dir()
        status = "PASS" if exists else "FAIL"
        results[name] = {"path": str(path), "status": status}
        logger.info(f"  [{status}] {name}: {path}")
        if not exists:
            all_pass = False

    results["overall"] = "PASS" if all_pass else "FAIL"
    return results


# ---------------------------------------------------------------------------
# Check 2: Image file existence
# ---------------------------------------------------------------------------

def check_images(logger: logging.Logger) -> dict:
    """
    Confirm DICOM image files are present under images/.

    VinBigData uses the .dicom extension (not .dcm).  A WARNING is issued
    if .dcm files are also found, since that could indicate mixed or
    misnamed files that may cause issues in later processing steps.
    """
    logger.info("=== CHECK 2: Image file existence ===")

    if not IMAGES_DIR.exists():
        logger.error("  [FAIL] images/ directory does not exist — skipping image check.")
        return {"status": "FAIL", "dicom_count": 0, "dcm_count": 0, "dicom_stems": []}

    dicom_files = sorted(IMAGES_DIR.glob("*.dicom"))
    dcm_files   = sorted(IMAGES_DIR.glob("*.dcm"))

    dicom_count = len(dicom_files)
    dcm_count   = len(dcm_files)

    # Filename stem (no extension) is the image_id used in annotation files
    dicom_stems = [f.stem for f in dicom_files]

    if dicom_count == 0:
        status = "FAIL"
        logger.error(f"  [FAIL] No .dicom files found in {IMAGES_DIR}")
    else:
        status = "PASS"
        logger.info(f"  [PASS] Found {dicom_count:,} .dicom files")

    if dcm_count > 0:
        logger.warning(
            f"  [WARNING] Also found {dcm_count} .dcm files — "
            "expected only .dicom in this dataset"
        )

    return {
        "status":      status,
        "dicom_count": dicom_count,
        "dcm_count":   dcm_count,
        "dicom_stems": dicom_stems,
    }


# ---------------------------------------------------------------------------
# Check 3: Main annotation file (train.csv)
# ---------------------------------------------------------------------------

def check_train_csv(logger: logging.Logger) -> dict:
    """
    Verify train.csv exists and contains all required columns.

    train.csv is the authoritative annotation source for the full VinBigData
    training set (15,000 images, 67,914 annotation rows).  Every local DICOM
    image ID must appear in this file for the dataset to be usable.
    """
    logger.info("=== CHECK 3: Main annotation file (train.csv) ===")

    if not TRAIN_CSV.exists():
        logger.error(f"  [FAIL] train.csv not found at {TRAIN_CSV}")
        return {
            "status": "FAIL", "df": None, "columns": [],
            "row_count": 0, "unique_ids": 0, "missing_cols": REQUIRED_ANN_COLUMNS,
        }

    df = pd.read_csv(TRAIN_CSV)
    actual_cols  = list(df.columns)
    missing_cols = [c for c in REQUIRED_ANN_COLUMNS if c not in actual_cols]
    unique_ids   = df["image_id"].nunique() if "image_id" in actual_cols else 0

    if missing_cols:
        status = "FAIL"
        logger.error(f"  [FAIL] train.csv missing required columns: {missing_cols}")
    else:
        status = "PASS"
        logger.info(
            f"  [PASS] train.csv — {len(df):,} rows, {unique_ids:,} unique image_ids"
        )
        logger.info(f"  Columns: {actual_cols}")

    return {
        "status":       status,
        "df":           df if status == "PASS" else None,
        "columns":      actual_cols,
        "row_count":    len(df),
        "unique_ids":   unique_ids,
        "missing_cols": missing_cols,
    }


# ---------------------------------------------------------------------------
# Check 4: Part annotation files (part_001 ... part_016)
# ---------------------------------------------------------------------------

def check_part_files(logger: logging.Logger) -> dict:
    """
    Verify that all 16 part_xxx_annotations.csv and part_xxx_image_ids.csv
    files exist and have consistent columns.

    These part files represent the local 1,000-image subset split across 16
    groups.  Each annotation part must have the same 8 columns as train.csv.
    Each image_ids part must contain only the single 'image_id' column.
    """
    logger.info("=== CHECK 4: Part annotation files (part_001 ... part_016) ===")

    missing_ann      = []
    missing_ids      = []
    inconsistent_ann = []
    inconsistent_ids = []
    ann_col_sets     = {}
    ids_col_sets     = {}

    for i in range(1, NUM_PARTS + 1):
        part     = f"{i:03d}"
        ann_file = ANN_DIR / f"part_{part}_annotations.csv"
        ids_file = ANN_DIR / f"part_{part}_image_ids.csv"

        # --- annotation file ---
        if not ann_file.exists():
            missing_ann.append(ann_file.name)
            logger.error(f"  [FAIL] Missing: {ann_file.name}")
        else:
            cols    = tuple(pd.read_csv(ann_file, nrows=0).columns.tolist())
            ann_col_sets[part] = cols
            missing = [c for c in REQUIRED_ANN_COLUMNS if c not in cols]
            extra   = [c for c in cols if c not in REQUIRED_ANN_COLUMNS]
            if missing or extra:
                inconsistent_ann.append(part)
                logger.warning(
                    f"  [WARNING] part_{part}_annotations.csv column mismatch — "
                    f"missing: {missing}, extra: {extra}"
                )

        # --- image_ids file ---
        if not ids_file.exists():
            missing_ids.append(ids_file.name)
            logger.error(f"  [FAIL] Missing: {ids_file.name}")
        else:
            cols = tuple(pd.read_csv(ids_file, nrows=0).columns.tolist())
            ids_col_sets[part] = cols
            if "image_id" not in cols:
                inconsistent_ids.append(part)
                logger.warning(
                    f"  [WARNING] part_{part}_image_ids.csv missing 'image_id' column"
                )

    all_ann_consistent = (
        len(set(ann_col_sets.values())) <= 1 and not inconsistent_ann
    )
    all_ids_consistent = (
        len(set(ids_col_sets.values())) <= 1 and not inconsistent_ids
    )

    status = "PASS" if (
        not missing_ann and not missing_ids
        and all_ann_consistent and all_ids_consistent
    ) else "FAIL"

    logger.info(f"  Annotation columns consistent across all parts: {all_ann_consistent}")
    logger.info(f"  Image_ids columns consistent across all parts:  {all_ids_consistent}")
    if missing_ann:
        logger.error(f"  Missing annotation files: {missing_ann}")
    if missing_ids:
        logger.error(f"  Missing image_ids files:  {missing_ids}")

    logger.info(f"  [{status}] Part files check complete")

    return {
        "status":             status,
        "missing_ann":        missing_ann,
        "missing_ids":        missing_ids,
        "inconsistent_ann":   inconsistent_ann,
        "inconsistent_ids":   inconsistent_ids,
        "all_ann_consistent": all_ann_consistent,
        "all_ids_consistent": all_ids_consistent,
    }


# ---------------------------------------------------------------------------
# Check 5: Image ID matching
# ---------------------------------------------------------------------------

def check_image_id_matching(
    logger: logging.Logger,
    dicom_stems: list,
    train_df: pd.DataFrame,
) -> dict:
    """
    Match local DICOM image IDs against image IDs in train.csv.

    Matching rule:
      - The image_id for a DICOM file is its filename without the .dicom extension.
      - train.csv covers the full 15,000-image VinBigData training set.
        Only 1,000 images are downloaded locally, so most train.csv IDs
        will not be present on disk — this is expected and not an error.
      - FAIL if any local DICOM file has no matching row in train.csv,
        because such an image could not be annotated or validated.
    """
    logger.info("=== CHECK 5: Image ID matching ===")

    if train_df is None:
        logger.error("  [SKIP] train.csv was not loaded — cannot perform matching")
        return {
            "status": "FAIL", "local_dicom_count": 0, "csv_unique_count": 0,
            "matched": 0, "only_on_disk": [], "only_in_csv_count": 0,
        }

    dicom_set = set(dicom_stems)
    csv_ids   = set(train_df["image_id"].unique())

    matched      = dicom_set & csv_ids
    only_on_disk = dicom_set - csv_ids   # local DICOMs with no annotation row
    only_in_csv  = csv_ids  - dicom_set  # annotation IDs not downloaded locally

    logger.info(f"  Local DICOM image IDs:          {len(dicom_set):,}")
    logger.info(f"  Unique image IDs in train.csv:  {len(csv_ids):,}")
    logger.info(f"  Matched (in both):              {len(matched):,}")
    logger.info(f"  Only on disk (no annotation):   {len(only_on_disk):,}")
    logger.info(
        f"  Only in train.csv (not local):  {len(only_in_csv):,} "
        "(expected — full dataset not downloaded)"
    )

    if only_on_disk:
        status = "FAIL"
        logger.error(
            f"  [FAIL] {len(only_on_disk)} local DICOM(s) have no annotation in train.csv"
        )
        for uid in sorted(only_on_disk)[:10]:
            logger.error(f"    Missing: {uid}")
        if len(only_on_disk) > 10:
            logger.error(f"    ... and {len(only_on_disk) - 10} more")
    else:
        status = "PASS"
        logger.info("  [PASS] All local DICOM images have matching records in train.csv")

    return {
        "status":            status,
        "local_dicom_count": len(dicom_set),
        "csv_unique_count":  len(csv_ids),
        "matched":           len(matched),
        "only_on_disk":      sorted(only_on_disk),
        "only_in_csv_count": len(only_in_csv),
    }


# ---------------------------------------------------------------------------
# Check 6: VinBigData label convention
# ---------------------------------------------------------------------------

def check_vinbigdata_labels(
    logger: logging.Logger,
    train_df: pd.DataFrame,
) -> dict:
    """
    Confirm VinBigData-specific label conventions are present in train.csv.

    In VinBigData:
      - class_id = 14 marks a "No finding" / normal image.
      - class_name = "No finding" is used for these rows.
      - Rows with class_id = 14 intentionally have empty bounding box columns.
        This is expected behaviour and must NOT be flagged as invalid here.

    This check only confirms the convention exists; bounding box validation
    for positive findings is handled by scripts/01_validate_dataset.py.
    """
    logger.info("=== CHECK 6: VinBigData label convention check ===")

    if train_df is None:
        logger.error("  [SKIP] train.csv was not loaded")
        return {"status": "SKIP", "has_no_finding_id": False, "has_no_finding_name": False}

    has_class_id_14    = 14 in train_df["class_id"].values
    has_no_finding_str = "No finding" in train_df["class_name"].values

    logger.info(f"  class_id = 14 ('No finding') present: {has_class_id_14}")
    logger.info(f"  class_name = 'No finding' present:    {has_no_finding_str}")

    if has_class_id_14 and has_no_finding_str:
        status = "PASS"
        no_finding_count = (train_df["class_id"] == 14).sum()
        logger.info(
            f"  [PASS] VinBigData label convention confirmed "
            f"({no_finding_count:,} 'No finding' rows in train.csv)"
        )
    else:
        status = "WARNING"
        logger.warning(
            "  [WARNING] Expected VinBigData label conventions not found — "
            "verify that train.csv is the correct VinBigData annotation file"
        )

    return {
        "status":              status,
        "has_no_finding_id":   has_class_id_14,
        "has_no_finding_name": has_no_finding_str,
    }


# ---------------------------------------------------------------------------
# Check 7: Readiness decision
# ---------------------------------------------------------------------------

def decide_readiness(results: dict, logger: logging.Logger):
    """
    Determine whether all blocking conditions pass.

    A dataset is ready for detailed validation only when every blocking
    condition is satisfied.  Warnings (e.g. .dcm files present) do not
    block readiness but are captured in the report.
    """
    logger.info("=== CHECK 7: Readiness decision ===")

    blocking_issues = []

    if results["folders"]["overall"] == "FAIL":
        blocking_issues.append("One or more required folders are missing")
    if results["images"]["status"] == "FAIL":
        blocking_issues.append("No DICOM images found in images/ directory")
    if results["train_csv"]["status"] == "FAIL":
        blocking_issues.append("train.csv is missing or has incorrect columns")
    if results["id_match"]["status"] == "FAIL":
        blocking_issues.append(
            "Some local DICOM images have no matching annotation in train.csv"
        )
    if results["part_files"]["status"] == "FAIL":
        blocking_issues.append(
            "One or more part annotation or image_id files are missing or have inconsistent columns"
        )

    ready = len(blocking_issues) == 0

    if ready:
        logger.info("  READY_FOR_VALIDATION = YES")
    else:
        logger.error("  READY_FOR_VALIDATION = NO")
        for issue in blocking_issues:
            logger.error(f"    BLOCKING: {issue}")

    return ready, blocking_issues


# ---------------------------------------------------------------------------
# Check 8: Save Markdown report
# ---------------------------------------------------------------------------

def save_report(
    results: dict,
    ready: bool,
    blocking_issues: list,
    logger: logging.Logger,
) -> None:
    """Write the Markdown summary report to outputs/reports/check_dataset_report.md."""

    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)

    r_folders = results["folders"]
    r_images  = results["images"]
    r_train   = results["train_csv"]
    r_parts   = results["part_files"]
    r_match   = results["id_match"]
    r_labels  = results["labels"]

    lines = [
        "# VinBigData Subset — Pre-Validation Check Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Folder Existence",
        "",
        "| Folder | Path | Status |",
        "|--------|------|--------|",
    ]
    for name, info in r_folders.items():
        if name == "overall":
            continue
        lines.append(f"| {name} | `{info['path']}` | {info['status']} |")
    lines.append(f"\n**Overall:** {r_folders['overall']}")

    lines += [
        "",
        "## 2. Image Files",
        "",
        f"- `.dicom` files found: **{r_images['dicom_count']:,}**",
        f"- `.dcm` files found: {r_images['dcm_count']}",
        f"- Status: **{r_images['status']}**",
    ]
    if r_images["dcm_count"] > 0:
        lines.append("- WARNING: `.dcm` files detected — expected only `.dicom`")

    lines += [
        "",
        "## 3. train.csv",
        "",
        f"- Rows (excl. header): **{r_train['row_count']:,}**",
        f"- Unique image_ids: **{r_train['unique_ids']:,}**",
        f"- Columns: `{', '.join(r_train['columns'])}`",
    ]
    if r_train["missing_cols"]:
        lines.append(f"- FAIL — Missing columns: `{r_train['missing_cols']}`")
    lines.append(f"- Status: **{r_train['status']}**")

    lines += [
        "",
        "## 4. Part Annotation Files (part_001 ... part_016)",
        "",
        f"- Missing annotation files: {r_parts['missing_ann'] or 'None'}",
        f"- Missing image_ids files:  {r_parts['missing_ids'] or 'None'}",
        f"- Annotation columns consistent across all parts: **{r_parts['all_ann_consistent']}**",
        f"- Image_ids columns consistent across all parts:  **{r_parts['all_ids_consistent']}**",
        f"- Status: **{r_parts['status']}**",
    ]

    lines += [
        "",
        "## 5. Image ID Matching",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Local DICOM image IDs | {r_match.get('local_dicom_count', 0):,} |",
        f"| Unique IDs in train.csv | {r_match.get('csv_unique_count', 0):,} |",
        f"| Matched (in both) | {r_match.get('matched', 0):,} |",
        f"| Only on disk — no annotation | {len(r_match.get('only_on_disk', [])):,} |",
        f"| Only in train.csv — not local | {r_match.get('only_in_csv_count', 0):,} |",
        f"\n- Status: **{r_match['status']}**",
        "- Note: train.csv IDs not present locally are expected — "
        "only 1,000 of 15,000 images are downloaded.",
    ]
    if r_match.get("only_on_disk"):
        lines.append("\nDICOM files missing from train.csv (first 10):")
        for uid in r_match["only_on_disk"][:10]:
            lines.append(f"  - `{uid}`")

    lines += [
        "",
        "## 6. VinBigData Label Convention",
        "",
        f"- `class_id = 14` present: **{r_labels['has_no_finding_id']}**",
        f"- `class_name = 'No finding'` present: **{r_labels['has_no_finding_name']}**",
        f"- Status: **{r_labels['status']}**",
        "- Note: 'No finding' rows intentionally have empty bounding box columns.",
    ]

    lines += [
        "",
        "---",
        "",
        "## Final Readiness Decision",
        "",
    ]
    if ready:
        lines += [
            "```",
            "READY_FOR_VALIDATION = YES",
            "```",
            "",
            "All blocking conditions passed. Proceed to `scripts/01_validate_dataset.py`.",
        ]
    else:
        lines += [
            "```",
            "READY_FOR_VALIDATION = NO",
            "```",
            "",
            "**Blocking issues:**",
        ]
        for issue in blocking_issues:
            lines.append(f"- {issue}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  Report saved → {REPORT_PATH}")
    logger.info(f"  Log saved    → {LOG_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger = setup_logging()
    logger.info("Starting VinBigData dataset pre-validation check")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Dataset root: {DATA_ROOT}")

    results = {}

    results["folders"]    = check_folders(logger)
    results["images"]     = check_images(logger)
    results["train_csv"]  = check_train_csv(logger)
    results["part_files"] = check_part_files(logger)

    results["id_match"] = check_image_id_matching(
        logger,
        dicom_stems=results["images"]["dicom_stems"],
        train_df=results["train_csv"]["df"],
    )

    results["labels"] = check_vinbigdata_labels(
        logger,
        train_df=results["train_csv"]["df"],
    )

    ready, blocking_issues = decide_readiness(results, logger)
    save_report(results, ready, blocking_issues, logger)

    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
