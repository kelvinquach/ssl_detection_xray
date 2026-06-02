"""
01_validate_dataset.py

Stage 1 — Annotation-level and bbox-level validation for the VinBigData
chest X-ray subset before conversion to COCO / YOLO format.

What this script does NOT do:
  - Does not train any model
  - Does not create train/val/test splits
  - Does not create labeled/unlabeled splits
  - Does not generate pseudo-labels
  - Does not convert to YOLO / COCO / Pascal VOC
  - Does not modify or delete raw data

VinBigData annotation format:
  image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max
  class_id = 14  → "No finding" / negative image (bbox columns intentionally empty)
  class_id != 14 → positive finding with a required bounding box

Run from project root:
  python scripts/01_validate_dataset.py \
      --dataset_root data/raw/vinbigdata \
      --images_dir   data/raw/vinbigdata/images \
      --annotation_csv data/raw/vinbigdata/annotations/train.csv \
      --output_dir   outputs/dataset_validation \
      --image_ext    .dicom
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# pydicom is required for reading DICOM dimensions
try:
    import pydicom
except ImportError:
    print("[ERROR] pydicom is not installed.  Run: pip install pydicom")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NO_FINDING_CLASS_ID = 14          # VinBigData convention
REQUIRED_BBOX_COLS  = ["x_min", "y_min", "x_max", "y_max"]
REQUIRED_ANN_COLS   = ["image_id", "class_name", "class_id",
                        "rad_id", "x_min", "y_min", "x_max", "y_max"]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VinBigData subset — annotation & bbox validation (Stage 1)"
    )
    p.add_argument("--dataset_root",   default="data/raw/vinbigdata",
                   help="Root of the VinBigData dataset folder")
    p.add_argument("--images_dir",     default="data/raw/vinbigdata/images",
                   help="Folder containing local .dicom files")
    p.add_argument("--annotation_csv", default="data/raw/vinbigdata/annotations/train.csv",
                   help="Path to train.csv (full VinBigData annotation file)")
    p.add_argument("--output_dir",     default="outputs/dataset_validation",
                   help="Directory where all output files will be saved")
    p.add_argument("--image_ext",      default=".dicom",
                   help="Image file extension (default: .dicom)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "validation.log"

    logger = logging.getLogger("validate_dataset")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

# ---------------------------------------------------------------------------
# Step 1 — Load annotations and filter to local images
# ---------------------------------------------------------------------------

def load_and_filter_annotations(
    annotation_csv: Path,
    local_ids: set[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load train.csv and keep only rows whose image_id is present locally.
    This avoids false errors from the other 14,000 images not downloaded.
    """
    logger.info(f"Loading annotation CSV: {annotation_csv}")
    df = pd.read_csv(annotation_csv, dtype={"class_id": "Int64"})

    missing = [c for c in REQUIRED_ANN_COLS if c not in df.columns]
    if missing:
        logger.error(f"train.csv is missing required columns: {missing}")
        sys.exit(1)

    # Ensure numeric bbox columns are truly numeric (coerce bad values to NaN)
    for col in REQUIRED_BBOX_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    original_rows = len(df)
    df_local = df[df["image_id"].isin(local_ids)].copy()
    logger.info(
        f"  train.csv rows: {original_rows:,} → after filtering to local images: "
        f"{len(df_local):,} rows covering {df_local['image_id'].nunique():,} image_ids"
    )
    return df_local

# ---------------------------------------------------------------------------
# Step 2 — Read DICOM dimensions (with error handling)
# ---------------------------------------------------------------------------

def read_dicom_dimensions(
    image_ids: list[str],
    images_dir: Path,
    image_ext: str,
    logger: logging.Logger,
) -> tuple[dict[str, tuple[int, int]], list[dict]]:
    """
    Attempt to open each local DICOM file and extract (width, height).

    Returns:
        dims_map  : {image_id: (width, height)}  for readable files
        unreadable: list of dicts for the unreadable_images.csv output
    """
    dims_map:   dict[str, tuple[int, int]] = {}
    unreadable: list[dict]                 = []

    total = len(image_ids)
    logger.info(f"Reading DICOM dimensions for {total:,} local images …")

    for idx, iid in enumerate(image_ids, 1):
        if idx % 200 == 0 or idx == total:
            logger.info(f"  Progress: {idx}/{total}")

        path = images_dir / f"{iid}{image_ext}"

        if not path.exists():
            # Missing file — tracked separately in image-level summary
            continue

        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            w  = int(ds.Columns)
            h  = int(ds.Rows)
            dims_map[iid] = (w, h)
        except Exception as exc:
            unreadable.append({
                "image_id":     iid,
                "image_path":   str(path),
                "error_message": str(exc),
            })
            logger.warning(f"  Cannot read DICOM: {path.name} — {exc}")

    logger.info(
        f"  Readable: {len(dims_map):,}  |  Unreadable: {len(unreadable):,}"
    )
    return dims_map, unreadable

# ---------------------------------------------------------------------------
# Step 3 — Image-level summary (one row per local image_id)
# ---------------------------------------------------------------------------

def build_image_level_summary(
    local_ids: set[str],
    df_local: pd.DataFrame,
    dims_map: dict[str, tuple[int, int]],
    unreadable_ids: set[str],
    images_dir: Path,
    image_ext: str,
) -> pd.DataFrame:
    """
    For every local image_id produce a single summary row covering:
      - annotation presence, file presence, readability
      - positive / no-finding classification
      - bbox counts per image
    """
    rows = []

    ann_ids = set(df_local["image_id"].unique())

    for iid in sorted(local_ids):
        img_path = images_dir / f"{iid}{image_ext}"
        has_file = img_path.exists()
        readable = has_file and (iid in dims_map)
        w, h     = dims_map.get(iid, (None, None))

        has_ann  = iid in ann_ids
        sub      = df_local[df_local["image_id"] == iid] if has_ann else pd.DataFrame()

        num_rows      = len(sub)
        is_positive   = False
        is_no_finding = False
        mixed         = False
        n_candidate   = 0   # rows with class_id != 14
        n_valid       = 0
        n_invalid     = 0
        issues        = []

        if has_ann:
            pos_rows  = sub[sub["class_id"] != NO_FINDING_CLASS_ID]
            nf_rows   = sub[sub["class_id"] == NO_FINDING_CLASS_ID]
            is_positive   = len(pos_rows) > 0
            is_no_finding = len(nf_rows) > 0 and len(pos_rows) == 0
            mixed         = len(pos_rows) > 0 and len(nf_rows) > 0

            n_candidate = len(pos_rows)

            if len(pos_rows) > 0:
                valid_mask = _valid_bbox_mask(pos_rows, w, h)
                n_valid    = int(valid_mask.sum())
                n_invalid  = n_candidate - n_valid

        if not has_file:
            issues.append("missing_dicom")
        if iid in unreadable_ids:
            issues.append("unreadable_dicom")
        if not has_ann:
            issues.append("no_annotation")
        if mixed:
            issues.append("mixed_no_finding_and_positive")

        rows.append({
            "image_id":                        iid,
            "has_annotation":                  has_ann,
            "has_image_file":                  has_file,
            "readable_image":                  readable,
            "image_width":                     w,
            "image_height":                    h,
            "is_positive":                     is_positive,
            "is_no_finding":                   is_no_finding,
            "has_mixed_no_finding_and_positive": mixed,
            "num_annotation_rows":             num_rows,
            "num_candidate_bboxes":            n_candidate,
            "num_valid_bboxes":                n_valid,
            "num_invalid_bboxes":              n_invalid,
            "issue":                           "; ".join(issues) if issues else "",
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Step 4 — BBox-level validation
# ---------------------------------------------------------------------------

def _valid_bbox_mask(
    df: pd.DataFrame,
    image_width:  Optional[int],
    image_height: Optional[int],
) -> pd.Series:
    """
    Return a boolean mask indicating which rows have fully valid bboxes.
    A bbox is valid when:
      1. All four coordinates are present (not NaN)
      2. width  = x_max - x_min > 0
      3. height = y_max - y_min > 0
      4. x_min >= 0, y_min >= 0
      5. x_max <= image_width  (only if image_width is known)
      6. y_max <= image_height (only if image_height is known)
    """
    has_all  = df[REQUIRED_BBOX_COLS].notna().all(axis=1)
    pos_mins = (df["x_min"] >= 0) & (df["y_min"] >= 0)
    pos_maxs = (df["x_max"] >= 0) & (df["y_max"] >= 0)
    width_ok  = df["x_max"] > df["x_min"]
    height_ok = df["y_max"] > df["y_min"]

    mask = has_all & pos_mins & pos_maxs & width_ok & height_ok

    if image_width is not None:
        mask = mask & (df["x_max"] <= image_width)
    if image_height is not None:
        mask = mask & (df["y_max"] <= image_height)

    return mask


def validate_bboxes(
    df_local: pd.DataFrame,
    dims_map: dict[str, tuple[int, int]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate every positive (class_id != 14) annotation row.

    Returns:
        df_pos_annotated : positive rows with extra diagnostic columns added
        invalid_df       : rows that failed at least one check (for invalid_bboxes.csv)
    """
    logger.info("Validating bounding boxes …")

    pos = df_local[df_local["class_id"] != NO_FINDING_CLASS_ID].copy()

    # Attach image dimensions
    pos["image_width"]  = pos["image_id"].map(lambda i: dims_map.get(i, (None, None))[0])
    pos["image_height"] = pos["image_id"].map(lambda i: dims_map.get(i, (None, None))[1])

    # Derived geometry
    pos["bbox_width"]  = pos["x_max"] - pos["x_min"]
    pos["bbox_height"] = pos["y_max"] - pos["y_min"]

    # Individual error flags
    pos["err_missing_any"]   = pos[REQUIRED_BBOX_COLS].isna().any(axis=1)
    pos["err_x_min_neg"]     = pos["x_min"].fillna(0) < 0
    pos["err_y_min_neg"]     = pos["y_min"].fillna(0) < 0
    pos["err_x_max_neg"]     = pos["x_max"].fillna(0) < 0
    pos["err_y_max_neg"]     = pos["y_max"].fillna(0) < 0
    pos["err_width_le_zero"] = (pos["bbox_width"].fillna(0) <= 0)
    pos["err_height_le_zero"]= (pos["bbox_height"].fillna(0) <= 0)

    # Boundary — only where image dims are known
    known_w = pos["image_width"].notna()
    known_h = pos["image_height"].notna()
    pos["err_outside_width"]  = False
    pos["err_outside_height"] = False
    pos.loc[known_w, "err_outside_width"]  = (
        pos.loc[known_w, "x_min"] < 0) | (
        pos.loc[known_w, "x_max"] > pos.loc[known_w, "image_width"])
    pos.loc[known_h, "err_outside_height"] = (
        pos.loc[known_h, "y_min"] < 0) | (
        pos.loc[known_h, "y_max"] > pos.loc[known_h, "image_height"])
    pos["err_outside_any"] = pos["err_outside_width"] | pos["err_outside_height"]

    # A row is invalid if it has ANY error flag
    error_cols = [c for c in pos.columns if c.startswith("err_")]
    pos["is_invalid"] = pos[error_cols].any(axis=1)
    pos["is_valid"]   = ~pos["is_invalid"]

    # Build error_type string for the invalid_bboxes.csv output
    def _error_types(row: pd.Series) -> str:
        labels = []
        if row["err_missing_any"]:
            labels.append("missing_coordinate")
        if row["err_x_min_neg"]:
            labels.append("x_min_negative")
        if row["err_y_min_neg"]:
            labels.append("y_min_negative")
        if row["err_x_max_neg"]:
            labels.append("x_max_negative")
        if row["err_y_max_neg"]:
            labels.append("y_max_negative")
        if row["err_width_le_zero"]:
            labels.append("bbox_width_le_zero")
        if row["err_height_le_zero"]:
            labels.append("bbox_height_le_zero")
        if row["err_outside_width"]:
            labels.append("outside_image_width")
        if row["err_outside_height"]:
            labels.append("outside_image_height")
        return "; ".join(labels) if labels else ""

    pos["error_type"] = pos.apply(_error_types, axis=1)

    invalid_df = pos[pos["is_invalid"]][
        ["image_id", "class_id", "class_name", "rad_id",
         "x_min", "y_min", "x_max", "y_max",
         "bbox_width", "bbox_height",
         "image_width", "image_height", "error_type"]
    ].copy()

    logger.info(
        f"  Candidate positive rows: {len(pos):,}  |  "
        f"Valid: {pos['is_valid'].sum():,}  |  "
        f"Invalid: {pos['is_invalid'].sum():,}"
    )
    return pos, invalid_df

# ---------------------------------------------------------------------------
# Step 5 — No-finding anomaly check
# ---------------------------------------------------------------------------

def check_no_finding_bboxes(
    df_local: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    In VinBigData, class_id = 14 rows must have empty bbox columns.
    Flag any No-finding row that has numeric bbox values as a WARNING.
    """
    nf = df_local[df_local["class_id"] == NO_FINDING_CLASS_ID].copy()

    # A row has a numeric bbox if at least one coordinate is non-NaN
    nf["has_numeric_bbox"] = nf[REQUIRED_BBOX_COLS].notna().any(axis=1)
    anomalous = nf[nf["has_numeric_bbox"]]

    if len(anomalous) > 0:
        logger.warning(
            f"  [WARNING] {len(anomalous):,} 'No finding' rows have numeric bbox values "
            f"(affects {anomalous['image_id'].nunique():,} images)"
        )
    else:
        logger.info("  [OK] All 'No finding' rows have empty bbox columns.")

    return anomalous

# ---------------------------------------------------------------------------
# Step 6 — Build validation_summary (key-value)
# ---------------------------------------------------------------------------

def build_summary(
    args:           argparse.Namespace,
    local_ids:      set[str],
    df_local:       pd.DataFrame,
    img_summary:    pd.DataFrame,
    pos_df:         pd.DataFrame,
    invalid_df:     pd.DataFrame,
    nf_anomaly_df:  pd.DataFrame,
    unreadable:     list[dict],
    dims_map:       dict[str, tuple[int, int]],
) -> dict:
    """Assemble every required key-value metric into a single dict."""

    ann_ids   = set(df_local["image_id"].unique())
    disk_only = local_ids - ann_ids
    ann_only  = ann_ids  - local_ids

    total_local    = len(local_ids)
    pos_images     = int(img_summary["is_positive"].sum())
    nf_images      = int(img_summary["is_no_finding"].sum())
    mixed_images   = int(img_summary["has_mixed_no_finding_and_positive"].sum())

    n_pos_rows     = len(pos_df)
    n_valid        = int(pos_df["is_valid"].sum())   if len(pos_df) > 0 else 0
    n_invalid      = int(pos_df["is_invalid"].sum()) if len(pos_df) > 0 else 0

    complete_mask  = (
        pos_df[REQUIRED_BBOX_COLS].notna().all(axis=1)
        if len(pos_df) > 0 else pd.Series(dtype=bool)
    )
    n_complete     = int(complete_mask.sum()) if len(pos_df) > 0 else 0

    avg_valid_all  = round(n_valid / total_local, 4) if total_local > 0 else 0.0
    avg_valid_pos  = round(n_valid / pos_images, 4)  if pos_images  > 0 else 0.0
    avg_cand_pos   = round(n_pos_rows / pos_images, 4) if pos_images > 0 else 0.0

    def _cnt(col: str) -> int:
        return int(pos_df[col].sum()) if col in pos_df.columns and len(pos_df) > 0 else 0

    # Positive rows with any missing coordinate
    pos_missing_bbox = int(
        pos_df[REQUIRED_BBOX_COLS].isna().any(axis=1).sum()
    ) if len(pos_df) > 0 else 0

    pos_imgs_missing = int(
        img_summary.loc[img_summary["is_positive"],
                        "num_invalid_bboxes"].gt(0).sum()
    ) if pos_images > 0 else 0

    # Determine READY_FOR_CONVERSION
    blocking = (
        len([u for u in unreadable]) > 0              # unreadable DICOM
        or len(disk_only) > 0                          # local images with no annotation
        or pos_missing_bbox > 0                        # positive rows missing bbox
        or _cnt("err_width_le_zero")  > 0             # zero-width bbox
        or _cnt("err_height_le_zero") > 0             # zero-height bbox
        or _cnt("err_outside_any")    > 0             # bbox outside image
        or len(nf_anomaly_df) > 0                      # No finding with numeric bbox
    )
    ready = "NO" if blocking else "YES"

    return {
        "total_local_images":                       total_local,
        "total_local_dicom_files":                  len(list(
            Path(args.images_dir).glob(f"*{args.image_ext}"))),
        "annotation_rows_after_filter":             len(df_local),
        "unique_annotation_images_after_filter":    df_local["image_id"].nunique(),
        "matched_images":                           len(local_ids & ann_ids),
        "disk_only_images":                         len(disk_only),
        "annotation_only_images":                   len(ann_only),
        "unreadable_images":                        len(unreadable),
        "positive_images":                          pos_images,
        "negative_no_finding_images":               nf_images,
        "positive_annotation_rows":                 n_pos_rows,
        "no_finding_rows":                          int(
            (df_local["class_id"] == NO_FINDING_CLASS_ID).sum()),
        "images_with_mixed_no_finding_and_positive": mixed_images,
        "candidate_bboxes":                         n_pos_rows,
        "bboxes_with_complete_coordinates":         n_complete,
        "valid_bboxes":                             n_valid,
        "invalid_bboxes":                           n_invalid,
        "avg_valid_bboxes_per_all_images":          avg_valid_all,
        "avg_valid_bboxes_per_positive_images":     avg_valid_pos,
        "avg_candidate_bboxes_per_positive_images": avg_cand_pos,
        "bbox_missing_any_coordinate":              _cnt("err_missing_any"),
        "bbox_x_min_negative":                      _cnt("err_x_min_neg"),
        "bbox_y_min_negative":                      _cnt("err_y_min_neg"),
        "bbox_x_max_le_x_min":                      _cnt("err_width_le_zero"),
        "bbox_y_max_le_y_min":                      _cnt("err_height_le_zero"),
        "bbox_width_le_zero":                       _cnt("err_width_le_zero"),
        "bbox_height_le_zero":                      _cnt("err_height_le_zero"),
        "bbox_outside_width":                       _cnt("err_outside_width"),
        "bbox_outside_height":                      _cnt("err_outside_height"),
        "bbox_outside_image_boundary":              _cnt("err_outside_any"),
        "no_finding_rows_with_numeric_bbox":        len(nf_anomaly_df),
        "positive_rows_missing_bbox":               pos_missing_bbox,
        "positive_rows_invalid_bbox":               n_invalid,
        "ready_for_conversion":                     ready,
    }

# ---------------------------------------------------------------------------
# Step 7 — Class distribution
# ---------------------------------------------------------------------------

def build_class_distribution(
    df_local: pd.DataFrame,
    pos_df:   pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (cid, cname), grp in df_local.groupby(["class_id", "class_name"], dropna=False):
        pos_grp = pos_df[pos_df["class_id"] == cid] if len(pos_df) > 0 else pd.DataFrame()
        rows.append({
            "class_id":             cid,
            "class_name":           cname,
            "num_annotation_rows":  len(grp),
            "num_images":           grp["image_id"].nunique(),
            "num_valid_bboxes":     int(pos_grp["is_valid"].sum())   if len(pos_grp) > 0 else 0,
            "num_invalid_bboxes":   int(pos_grp["is_invalid"].sum()) if len(pos_grp) > 0 else 0,
        })
    return pd.DataFrame(rows).sort_values("class_id").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Step 8 — BBox statistics (valid bboxes only)
# ---------------------------------------------------------------------------

def build_bbox_statistics(pos_df: pd.DataFrame) -> pd.DataFrame:
    if len(pos_df) == 0:
        return pd.DataFrame(columns=["metric", "value"])

    valid = pos_df[pos_df["is_valid"]].copy()
    if len(valid) == 0:
        return pd.DataFrame(columns=["metric", "value"])

    valid["area"] = valid["bbox_width"] * valid["bbox_height"]

    stats = {}
    for dim, col in [("width", "bbox_width"), ("height", "bbox_height"), ("area", "area")]:
        stats[f"min_{dim}"]    = round(float(valid[col].min()),    2)
        stats[f"max_{dim}"]    = round(float(valid[col].max()),    2)
        stats[f"mean_{dim}"]   = round(float(valid[col].mean()),   2)
        stats[f"median_{dim}"] = round(float(valid[col].median()), 2)

    return pd.DataFrame(
        [{"metric": k, "value": v} for k, v in stats.items()]
    )

# ---------------------------------------------------------------------------
# Step 9 — Markdown report
# ---------------------------------------------------------------------------

def _tick(val) -> str:
    """Return PASS / FAIL tick string based on truthiness (0 / False = PASS)."""
    return "✅ PASS" if not val else "❌ FAIL"

def _warn(val) -> str:
    return "✅ OK" if not val else "⚠️ WARNING"


def write_markdown_report(
    output_dir: Path,
    args:       argparse.Namespace,
    summary:    dict,
    img_summary: pd.DataFrame,
    class_dist:  pd.DataFrame,
    bbox_stats:  pd.DataFrame,
    run_time:    str,
) -> None:
    ready = summary["ready_for_conversion"]
    lines = [
        "# VinBigData Subset — Dataset Validation Report",
        f"\n**Generated:** {run_time}",
        f"  \n**Dataset root:** `{args.dataset_root}`",
        f"  \n**Annotation CSV:** `{args.annotation_csv}`",
        f"  \n**Images dir:** `{args.images_dir}`",
        f"  \n**Output dir:** `{args.output_dir}`",
        "",
        "---",
        "",
        "## A. Dataset Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total local DICOM files | {summary['total_local_dicom_files']:,} |",
        f"| Total local image IDs | {summary['total_local_images']:,} |",
        f"| Annotation rows (after filter) | {summary['annotation_rows_after_filter']:,} |",
        f"| Unique annotation image IDs (after filter) | {summary['unique_annotation_images_after_filter']:,} |",
        "",
        "---",
        "",
        "## B. Image–Annotation Matching",
        "",
        "| Metric | Value | Status |",
        "|--------|-------|--------|",
        f"| Matched images | {summary['matched_images']:,} | {_tick(summary['matched_images'] < summary['total_local_images'])} |",
        f"| On disk — no annotation | {summary['disk_only_images']:,} | {_tick(summary['disk_only_images'] > 0)} |",
        f"| In annotation — not on disk | {summary['annotation_only_images']:,} | ℹ️ Expected (subset) |",
        "",
        "---",
        "",
        "## C. Image Readability",
        "",
        "| Metric | Value | Status |",
        "|--------|-------|--------|",
        f"| Unreadable DICOM files | {summary['unreadable_images']:,} | {_tick(summary['unreadable_images'] > 0)} |",
        "",
        "---",
        "",
        "## D. Positive Images",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Positive images | {summary['positive_images']:,} |",
        f"| Positive ratio | {summary['positive_images'] / max(summary['total_local_images'], 1):.1%} |",
        f"| Positive annotation rows | {summary['positive_annotation_rows']:,} |",
        "",
        "---",
        "",
        "## E. No Finding / Negative Images",
        "",
        "| Metric | Value | Status |",
        "|--------|-------|--------|",
        f"| No finding images | {summary['negative_no_finding_images']:,} | ℹ️ |",
        f"| No finding ratio | {summary['negative_no_finding_images'] / max(summary['total_local_images'], 1):.1%} | ℹ️ |",
        f"| No finding annotation rows | {summary['no_finding_rows']:,} | ℹ️ |",
        f"| Images with mixed No finding + positive | {summary['images_with_mixed_no_finding_and_positive']:,} | {_warn(summary['images_with_mixed_no_finding_and_positive'] > 0)} |",
        "",
        "---",
        "",
        "## F–G. BBox Counts & Averages",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Candidate bboxes (positive rows) | {summary['candidate_bboxes']:,} |",
        f"| Bboxes with complete coordinates | {summary['bboxes_with_complete_coordinates']:,} |",
        f"| Valid bboxes | {summary['valid_bboxes']:,} |",
        f"| Invalid bboxes | {summary['invalid_bboxes']:,} |",
        f"| Avg valid bboxes / all images | {summary['avg_valid_bboxes_per_all_images']} |",
        f"| Avg valid bboxes / positive images | {summary['avg_valid_bboxes_per_positive_images']} |",
        f"| Avg candidate bboxes / positive images | {summary['avg_candidate_bboxes_per_positive_images']} |",
    ]

    # Per-image bbox distribution
    if len(img_summary) > 0:
        pos_sub = img_summary[img_summary["is_positive"]]
        lines += [
            "",
            "**Valid bbox distribution (positive images only):**",
            "",
            "| # Valid bboxes | # Images |",
            "|----------------|----------|",
        ]
        for cnt, grp in pos_sub.groupby("num_valid_bboxes"):
            lines.append(f"| {cnt} | {len(grp):,} |")
        lines.append(
            f"\nMax valid bboxes on a single image: "
            f"**{int(pos_sub['num_valid_bboxes'].max()) if len(pos_sub) > 0 else 0}**"
        )

    lines += [
        "",
        "---",
        "",
        "## H–M. BBox Coordinate Errors",
        "",
        "| Check | Count | Status |",
        "|-------|-------|--------|",
        f"| Missing any coordinate (positive rows) | {summary['bbox_missing_any_coordinate']:,} | {_tick(summary['bbox_missing_any_coordinate'] > 0)} |",
        f"| x_min < 0 | {summary['bbox_x_min_negative']:,} | {_tick(summary['bbox_x_min_negative'] > 0)} |",
        f"| y_min < 0 | {summary['bbox_y_min_negative']:,} | {_tick(summary['bbox_y_min_negative'] > 0)} |",
        f"| x_max <= x_min | {summary['bbox_x_max_le_x_min']:,} | {_tick(summary['bbox_x_max_le_x_min'] > 0)} |",
        f"| y_max <= y_min | {summary['bbox_y_max_le_y_min']:,} | {_tick(summary['bbox_y_max_le_y_min'] > 0)} |",
        f"| bbox_width <= 0 | {summary['bbox_width_le_zero']:,} | {_tick(summary['bbox_width_le_zero'] > 0)} |",
        f"| bbox_height <= 0 | {summary['bbox_height_le_zero']:,} | {_tick(summary['bbox_height_le_zero'] > 0)} |",
        "",
        "---",
        "",
        "## N. Boundary Validation",
        "",
        "| Check | Count | Status |",
        "|-------|-------|--------|",
        f"| bbox outside image width | {summary['bbox_outside_width']:,} | {_tick(summary['bbox_outside_width'] > 0)} |",
        f"| bbox outside image height | {summary['bbox_outside_height']:,} | {_tick(summary['bbox_outside_height'] > 0)} |",
        f"| bbox outside image (any) | {summary['bbox_outside_image_boundary']:,} | {_tick(summary['bbox_outside_image_boundary'] > 0)} |",
        "",
        "---",
        "",
        "## O–P. No Finding & Positive Cross-Checks",
        "",
        "| Check | Count | Status |",
        "|-------|-------|--------|",
        f"| No finding rows with numeric bbox | {summary['no_finding_rows_with_numeric_bbox']:,} | {_tick(summary['no_finding_rows_with_numeric_bbox'] > 0)} |",
        f"| Positive rows missing bbox | {summary['positive_rows_missing_bbox']:,} | {_tick(summary['positive_rows_missing_bbox'] > 0)} |",
        f"| Positive rows with invalid bbox | {summary['positive_rows_invalid_bbox']:,} | {_tick(summary['positive_rows_invalid_bbox'] > 0)} |",
        "",
        "---",
        "",
        "## Class Distribution",
        "",
    ]

    if len(class_dist) > 0:
        lines += [
            "| class_id | class_name | annotation_rows | images | valid_bboxes | invalid_bboxes |",
            "|----------|------------|-----------------|--------|--------------|----------------|",
        ]
        for _, row in class_dist.iterrows():
            lines.append(
                f"| {row['class_id']} | {row['class_name']} | "
                f"{row['num_annotation_rows']:,} | {row['num_images']:,} | "
                f"{row['num_valid_bboxes']:,} | {row['num_invalid_bboxes']:,} |"
            )

    lines += [
        "",
        "---",
        "",
        "## BBox Statistics (valid bboxes only)",
        "",
    ]
    if len(bbox_stats) > 0:
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for _, row in bbox_stats.iterrows():
            lines.append(f"| {row['metric']} | {row['value']} |")

    lines += [
        "",
        "---",
        "",
        "## Final Decision",
        "",
        f"```",
        f"READY_FOR_CONVERSION = {ready}",
        f"```",
        "",
    ]

    if ready == "YES":
        lines.append(
            "All blocking conditions passed. "
            "Proceed to annotation conversion (COCO / YOLO)."
        )
    else:
        lines.append("**Blocking issues detected — resolve before conversion:**")
        blocking_map = {
            "unreadable_images":                    "Unreadable DICOM files present",
            "disk_only_images":                     "Local images with no annotation record",
            "positive_rows_missing_bbox":           "Positive rows missing bbox coordinates",
            "bbox_width_le_zero":                   "Bboxes with width <= 0",
            "bbox_height_le_zero":                  "Bboxes with height <= 0",
            "bbox_outside_image_boundary":          "Bboxes outside image boundaries",
            "no_finding_rows_with_numeric_bbox":    "No-finding rows with unexpected numeric bbox",
        }
        for key, desc in blocking_map.items():
            if summary.get(key, 0):
                lines.append(f"- ❌ {desc}: **{summary[key]}**")

    report_path = output_dir / "validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report → {report_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger   = setup_logging(output_dir)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 60)
    logger.info("VinBigData Dataset Validation — Stage 1")
    logger.info(f"Run time     : {run_time}")
    logger.info(f"Dataset root : {args.dataset_root}")
    logger.info(f"Images dir   : {args.images_dir}")
    logger.info(f"Annotation   : {args.annotation_csv}")
    logger.info(f"Output dir   : {args.output_dir}")
    logger.info("=" * 60)

    images_dir     = Path(args.images_dir)
    annotation_csv = Path(args.annotation_csv)

    # ------------------------------------------------------------------
    # A. Enumerate local DICOM files
    # ------------------------------------------------------------------
    logger.info("Step 1/9 — Enumerating local DICOM files")
    dicom_files = sorted(images_dir.glob(f"*{args.image_ext}"))
    local_ids   = {f.stem for f in dicom_files}
    logger.info(f"  Found {len(local_ids):,} local DICOM images")

    if not local_ids:
        logger.error("No local images found. Check --images_dir and --image_ext.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # B. Load and filter annotation
    # ------------------------------------------------------------------
    logger.info("Step 2/9 — Loading and filtering annotation CSV")
    df_local = load_and_filter_annotations(annotation_csv, local_ids, logger)

    # ------------------------------------------------------------------
    # C. Read DICOM dimensions
    # ------------------------------------------------------------------
    logger.info("Step 3/9 — Reading DICOM dimensions")
    dims_map, unreadable = read_dicom_dimensions(
        sorted(local_ids), images_dir, args.image_ext, logger
    )
    unreadable_ids = {u["image_id"] for u in unreadable}

    # ------------------------------------------------------------------
    # D–E. Image-level summary
    # ------------------------------------------------------------------
    logger.info("Step 4/9 — Building image-level summary")
    img_summary = build_image_level_summary(
        local_ids, df_local, dims_map, unreadable_ids,
        images_dir, args.image_ext
    )

    # ------------------------------------------------------------------
    # F–M. BBox-level validation
    # ------------------------------------------------------------------
    logger.info("Step 5/9 — Validating bounding boxes")
    pos_df, invalid_df = validate_bboxes(df_local, dims_map, logger)

    # ------------------------------------------------------------------
    # O. No-finding rows with unexpected numeric bbox
    # ------------------------------------------------------------------
    logger.info("Step 6/9 — Checking No finding bbox anomalies")
    nf_anomaly_df = check_no_finding_bboxes(df_local, logger)

    # ------------------------------------------------------------------
    # Summary + class distribution + bbox statistics
    # ------------------------------------------------------------------
    logger.info("Step 7/9 — Assembling summary metrics")
    summary    = build_summary(
        args, local_ids, df_local, img_summary,
        pos_df, invalid_df, nf_anomaly_df, unreadable, dims_map
    )
    class_dist  = build_class_distribution(df_local, pos_df)
    bbox_stats  = build_bbox_statistics(pos_df)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    logger.info("Step 8/9 — Saving output files")

    # 1. validation_summary.csv
    summary_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in summary.items()]
    )
    summary_df.to_csv(output_dir / "validation_summary.csv", index=False)
    logger.info(f"  Saved validation_summary.csv")

    # 2. invalid_bboxes.csv
    if len(invalid_df) > 0:
        invalid_df.to_csv(output_dir / "invalid_bboxes.csv", index=False)
        logger.info(f"  Saved invalid_bboxes.csv ({len(invalid_df):,} rows)")
    else:
        pd.DataFrame(columns=[
            "image_id", "class_id", "class_name", "rad_id",
            "x_min", "y_min", "x_max", "y_max",
            "bbox_width", "bbox_height", "image_width", "image_height", "error_type"
        ]).to_csv(output_dir / "invalid_bboxes.csv", index=False)
        logger.info("  Saved invalid_bboxes.csv (empty — no invalid bboxes)")

    # 3. unreadable_images.csv
    pd.DataFrame(unreadable if unreadable else
                 [{"image_id": "", "image_path": "", "error_message": ""}]
    ).to_csv(output_dir / "unreadable_images.csv", index=False)
    logger.info(f"  Saved unreadable_images.csv ({len(unreadable):,} rows)")

    # 4. image_level_summary.csv
    img_summary.to_csv(output_dir / "image_level_summary.csv", index=False)
    logger.info(f"  Saved image_level_summary.csv ({len(img_summary):,} rows)")

    # 5. class_distribution.csv
    class_dist.to_csv(output_dir / "class_distribution.csv", index=False)
    logger.info(f"  Saved class_distribution.csv ({len(class_dist):,} rows)")

    # 6. bbox_statistics.csv
    bbox_stats.to_csv(output_dir / "bbox_statistics.csv", index=False)
    logger.info(f"  Saved bbox_statistics.csv")

    # 7. validation_report.md
    logger.info("Step 9/9 — Writing Markdown report")
    write_markdown_report(
        output_dir, args, summary,
        img_summary, class_dist, bbox_stats, run_time
    )

    # ------------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------------
    ready = summary["ready_for_conversion"]
    logger.info("=" * 60)
    logger.info(f"READY_FOR_CONVERSION = {ready}")
    logger.info("=" * 60)

    sys.exit(0 if ready == "YES" else 1)


if __name__ == "__main__":
    main()
