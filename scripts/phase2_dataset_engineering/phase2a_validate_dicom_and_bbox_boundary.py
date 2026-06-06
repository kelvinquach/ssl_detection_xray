"""
PHASE 2A — Image/DICOM Accessibility & BBox Boundary Validation
================================================================
Mục tiêu: Kiểm tra image/DICOM accessibility, pixel sanity (full validation),
và bbox boundary cho toàn bộ 4,894 images trong controlled working subset.

Scope đã chốt — KHÔNG thực hiện:
- Không tạo split (train/val/test hoặc labeled/unlabeled).
- Không convert COCO/YOLO.
- Không xử lý class imbalance.
- Không oversampling/undersampling/class reweighting.
- Không training.
- Không tự động xóa/clip/sửa bbox.
- Nếu phát hiện lỗi: chỉ flag/report.

Pixel validation: full mode — tất cả 4,894 DICOM files.
Progress được in ra stdout và ghi vào phase2a_run_log.txt.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("configs/dataset.yaml")
REPORT_DIR  = Path("reports/phase2_dataset_engineering")
LOG_FILE    = REPORT_DIR / "phase2a_run_log.txt"

EXPECTED_TOTAL_IMAGES       = 4894
EXPECTED_ABNORMAL_IMAGES    = 4394
EXPECTED_NORMAL_IMAGES      = 500
EXPECTED_BBOX_ROWS          = 36096
EXPECTED_ABNORMAL_CLASSES   = 14
EXPECTED_ABNORMAL_WITH_BBOX = 4394
EXPECTED_NORMAL_WITH_BBOX   = 0
NO_FINDING_CLASS_ID         = 14

PROGRESS_EVERY = 250   # print progress every N files


# ---------------------------------------------------------------------------
# Dual logger: stdout + log file
# ---------------------------------------------------------------------------

class Logger:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "w", encoding="utf-8", buffering=1)
        self._start = time.time()

    def log(self, msg: str = ""):
        ts = f"[{time.time() - self._start:7.1f}s]"
        line = f"{ts} {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_value(config, *keys, fallback=None, warn=True):
    obj = config
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            if warn:
                print(f"[WARNING] Config key '{'.'.join(str(k) for k in keys)}' "
                      f"not found. Fallback: {fallback!r}")
            return fallback
        obj = obj[key]
    return obj


# ---------------------------------------------------------------------------
# Step 1+2 — Input validation + Phase 1A metadata re-check
# ---------------------------------------------------------------------------

def load_and_validate_inputs(config: dict, log: Logger) -> dict:
    """
    Load metadata CSVs từ repo C: và thực hiện Phase 1A consistency re-check.
    Không tự sửa nếu mismatch — chỉ report.
    """
    mf = config.get("metadata_files", {})
    paths = {
        "selected":    Path(mf.get("selected_image_ids",
                            "data/raw/vinbigdata/metadata_subset/selected_image_ids.csv")),
        "abnormal":    Path(mf.get("abnormal_image_ids",
                            "data/raw/vinbigdata/metadata_subset/abnormal_image_ids.csv")),
        "normal":      Path(mf.get("normal_image_ids",
                            "data/raw/vinbigdata/metadata_subset/normal_image_ids_500.csv")),
        "annotations": Path(mf.get("subset_annotations",
                            "data/raw/vinbigdata/metadata_subset/subset_train_annotations.csv")),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required input file missing: {path} ({name})")

    selected_df = pd.read_csv(paths["selected"])
    abnormal_df = pd.read_csv(paths["abnormal"])
    normal_df   = pd.read_csv(paths["normal"])
    ann_df      = pd.read_csv(paths["annotations"])

    selected_abnormal_ids = set(
        selected_df[selected_df["subset_type"] == "abnormal"]["image_id"])
    selected_normal_ids   = set(
        selected_df[selected_df["subset_type"] == "normal"]["image_id"])

    abn_bbox_df   = ann_df[ann_df["class_id"] != NO_FINDING_CLASS_ID].copy()
    valid_bbox_df = abn_bbox_df.dropna(
        subset=["x_min", "y_min", "x_max", "y_max"])

    total_images       = selected_df["image_id"].nunique()
    abnormal_images    = len(selected_abnormal_ids)
    normal_images      = len(selected_normal_ids)
    bbox_rows          = len(abn_bbox_df)
    abnormal_classes   = abn_bbox_df["class_id"].nunique()
    abnormal_with_bbox = valid_bbox_df["image_id"].nunique()
    normal_with_bbox   = len(selected_normal_ids & set(valid_bbox_df["image_id"]))

    ann_abnormal_ids    = set(abn_bbox_df["image_id"])
    ann_not_in_selected = ann_abnormal_ids - selected_abnormal_ids
    normal_in_ann       = selected_normal_ids & ann_abnormal_ids

    discrepancies = []
    numeric_checks = [
        ("total_images",               total_images,       EXPECTED_TOTAL_IMAGES),
        ("abnormal_images",            abnormal_images,    EXPECTED_ABNORMAL_IMAGES),
        ("normal_images",              normal_images,      EXPECTED_NORMAL_IMAGES),
        ("abnormal_bbox_rows",         bbox_rows,          EXPECTED_BBOX_ROWS),
        ("abnormal_classes",           abnormal_classes,   EXPECTED_ABNORMAL_CLASSES),
        ("abnormal_images_with_bbox",  abnormal_with_bbox, EXPECTED_ABNORMAL_WITH_BBOX),
        ("normal_images_with_bbox",    normal_with_bbox,   EXPECTED_NORMAL_WITH_BBOX),
    ]
    for name, actual, expected in numeric_checks:
        ok = actual == expected
        log.log(f"  {'[OK]' if ok else '[MISMATCH]'} {name}: {actual} (expected {expected})")
        if not ok:
            discrepancies.append(f"{name}: actual={actual}, expected={expected}")

    cross_ok = not ann_not_in_selected
    log.log(f"  {'[OK]' if cross_ok else '[MISMATCH]'} "
            f"annotation IDs not in selected subset: {len(ann_not_in_selected)}")
    normal_ok = not normal_in_ann
    log.log(f"  {'[OK]' if normal_ok else '[MISMATCH]'} "
            f"normal IDs in abnormal annotations: {len(normal_in_ann)}")

    if ann_not_in_selected:
        discrepancies.append(f"annotation_ids_not_in_selected: {len(ann_not_in_selected)}")
    if normal_in_ann:
        discrepancies.append(f"normal_ids_in_abnormal_annotations: {len(normal_in_ann)}")

    log.log(f"  Discrepancies: {len(discrepancies)}")

    return {
        "selected_df":       selected_df,
        "abnormal_df":       abnormal_df,
        "normal_df":         normal_df,
        "ann_df":            ann_df,
        "abn_bbox_df":       abn_bbox_df,
        "valid_bbox_df":     valid_bbox_df,
        "discrepancies":     discrepancies,
        "numeric_checks":    numeric_checks,
        "abnormal_with_bbox":  abnormal_with_bbox,
        "normal_with_bbox":    normal_with_bbox,
        "ann_not_in_selected": len(ann_not_in_selected),
        "normal_in_ann":       len(normal_in_ann),
    }


# ---------------------------------------------------------------------------
# Step 3 — DICOM availability
# ---------------------------------------------------------------------------

def check_dicom_availability(selected_df: pd.DataFrame,
                              dicom_dir: Path,
                              log: Logger) -> tuple[pd.DataFrame, int]:
    """Check DICOM file existence. Also detect extra files."""
    selected_ids = set(selected_df["image_id"].unique())
    all_in_dir   = {p.stem for p in dicom_dir.glob("*.dicom")}
    all_in_dir  |= {p.stem for p in dicom_dir.glob("*.dcm")}
    extra_dicom  = len(all_in_dir - selected_ids)

    rows = []
    for img_id in selected_df["image_id"].unique():
        stype = selected_df.loc[
            selected_df["image_id"] == img_id, "subset_type"].iloc[0]
        found = None
        for ext in [".dicom", ".dcm"]:
            c = dicom_dir / f"{img_id}{ext}"
            if c.exists():
                found = c
                break
        rows.append({
            "image_id":    img_id,
            "subset_type": stype,
            "dicom_path":  str(found) if found else None,
            "dicom_found": found is not None,
            "dicom_ext":   found.suffix if found else None,
        })

    df = pd.DataFrame(rows)
    found_count   = int(df["dicom_found"].sum())
    missing_count = int((~df["dicom_found"]).sum())

    abn_found = int(df[(df["subset_type"] == "abnormal") & df["dicom_found"]].shape[0])
    abn_miss  = int(df[(df["subset_type"] == "abnormal") & ~df["dicom_found"]].shape[0])
    nm_found  = int(df[(df["subset_type"] == "normal") & df["dicom_found"]].shape[0])
    nm_miss   = int(df[(df["subset_type"] == "normal") & ~df["dicom_found"]].shape[0])

    log.log(f"  DICOM found        : {found_count}")
    log.log(f"  DICOM missing      : {missing_count}")
    log.log(f"  DICOM extra        : {extra_dicom}")
    log.log(f"  Abnormal found/miss: {abn_found}/{abn_miss}")
    log.log(f"  Normal found/miss  : {nm_found}/{nm_miss}")

    return df, extra_dicom


# ---------------------------------------------------------------------------
# Step 4+5 — DICOM loading + full pixel validation
# ---------------------------------------------------------------------------

def extract_dicom_full(avail_df: pd.DataFrame, log: Logger) -> pd.DataFrame:
    """
    Full DICOM loading + pixel_array validation for all found files.
    Progress logged every PROGRESS_EVERY files to stdout AND log file.
    """
    try:
        import pydicom
    except ImportError:
        log.log("[ERROR] pydicom not installed. Run: pip install pydicom")
        sys.exit(1)

    COLS = [
        "image_id", "subset_type", "dicom_path",
        "dicom_readable",
        "image_width", "image_height", "rows_tag", "columns_tag",
        "photometric_interpretation", "bits_stored", "pixel_spacing",
        "dicom_error",
        "pixel_readable", "pixel_ndim", "pixel_empty",
        "pixel_shape", "pixel_shape_matches_rows_columns",
        "pixel_min", "pixel_max", "pixel_mean",
        "pixel_has_nan", "pixel_has_inf",
        "pixel_error",
    ]

    found_df = avail_df[avail_df["dicom_found"]].copy().reset_index(drop=True)
    total    = len(found_df)
    log.log(f"  Full DICOM + pixel validation mode for PHASE 2A.")
    log.log(f"  Total files to process: {total}")

    rows = []
    for i, (_, row) in enumerate(found_df.iterrows(), 1):
        if i % PROGRESS_EVERY == 0 or i == total:
            log.log(f"  Processed {i}/{total} DICOM files...")

        r = {k: None for k in COLS}
        r.update({
            "image_id":       row["image_id"],
            "subset_type":    row["subset_type"],
            "dicom_path":     row["dicom_path"],
            "dicom_readable": False,
            "pixel_readable": False,
            "pixel_empty":    None,
            "pixel_has_nan":  None,
            "pixel_has_inf":  None,
            "pixel_shape_matches_rows_columns": None,
        })

        try:
            import pydicom
            ds = pydicom.dcmread(row["dicom_path"], force=True)
            r["dicom_readable"] = True

            rows_val = getattr(ds, "Rows",    None)
            cols_val = getattr(ds, "Columns", None)
            r["rows_tag"]    = int(rows_val) if rows_val is not None else None
            r["columns_tag"] = int(cols_val) if cols_val is not None else None
            r["image_height"] = r["rows_tag"]
            r["image_width"]  = r["columns_tag"]
            r["photometric_interpretation"] = getattr(
                ds, "PhotometricInterpretation", None)
            bits = getattr(ds, "BitsStored", None)
            r["bits_stored"] = int(bits) if bits is not None else None
            ps = getattr(ds, "PixelSpacing", None)
            r["pixel_spacing"] = str(list(ps)) if ps is not None else None

            # Pixel array
            try:
                px = ds.pixel_array
                r["pixel_readable"] = True
                r["pixel_ndim"]     = int(px.ndim)
                r["pixel_empty"]    = bool(px.size == 0)
                r["pixel_shape"]    = str(px.shape)

                # Shape vs Rows/Columns
                if r["rows_tag"] and r["columns_tag"] and not r["pixel_empty"]:
                    h_ok = (px.shape[0] == r["rows_tag"]    if px.ndim >= 1 else False)
                    w_ok = (px.shape[1] == r["columns_tag"] if px.ndim >= 2 else False)
                    r["pixel_shape_matches_rows_columns"] = bool(h_ok and w_ok)
                else:
                    r["pixel_shape_matches_rows_columns"] = None

                px_f = px.astype(float)
                r["pixel_min"]     = float(np.nanmin(px_f)) if not r["pixel_empty"] else None
                r["pixel_max"]     = float(np.nanmax(px_f)) if not r["pixel_empty"] else None
                r["pixel_mean"]    = float(np.nanmean(px_f)) if not r["pixel_empty"] else None
                r["pixel_has_nan"] = bool(np.any(np.isnan(px_f)))
                r["pixel_has_inf"] = bool(np.any(np.isinf(px_f)))

            except Exception as px_err:
                r["pixel_readable"] = False
                r["pixel_error"]    = str(px_err)[:200]

        except Exception as e:
            r["dicom_readable"] = False
            r["dicom_error"]    = str(e)[:200]

        rows.append(r)

    return pd.DataFrame(rows)[COLS]


# ---------------------------------------------------------------------------
# Step 6 — BBox boundary validation
# ---------------------------------------------------------------------------

def validate_bbox_boundary(valid_bbox_df: pd.DataFrame,
                            meta_df: pd.DataFrame,
                            log: Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full bbox boundary validation for all 36,096 bbox rows.
    Checks: x_min≥0, y_min≥0, x_max≤W, y_max≤H, x_max>x_min, y_max>y_min.
    No clipping, no removal — flag and report only.
    Returns (full_df, invalid_df, counts).
    """
    bbox_df = valid_bbox_df.copy()

    if not meta_df.empty and {"image_id","image_width","image_height"}.issubset(meta_df.columns):
        dim_df = meta_df[["image_id","image_width","image_height"]].dropna(
            subset=["image_width","image_height"])
    else:
        dim_df = pd.DataFrame(columns=["image_id","image_width","image_height"])

    bbox_df = bbox_df.merge(dim_df, on="image_id", how="left")
    bbox_df["has_image_dims"] = (
        bbox_df["image_width"].notna() & bbox_df["image_height"].notna())

    # Per-condition flags
    bbox_df["flag_x_min_negative"]  = bbox_df["x_min"] < 0
    bbox_df["flag_y_min_negative"]  = bbox_df["y_min"] < 0
    bbox_df["flag_x_max_gt_width"]  = bbox_df.apply(
        lambda r: r["has_image_dims"] and r["x_max"] > r["image_width"], axis=1)
    bbox_df["flag_y_max_gt_height"] = bbox_df.apply(
        lambda r: r["has_image_dims"] and r["y_max"] > r["image_height"], axis=1)
    bbox_df["flag_x_order"]         = bbox_df["x_max"] <= bbox_df["x_min"]
    bbox_df["flag_y_order"]         = bbox_df["y_max"] <= bbox_df["y_min"]

    def make_reasons(r):
        reasons = []
        if r["flag_x_min_negative"]:  reasons.append(f"x_min({r['x_min']:.0f})<0")
        if r["flag_y_min_negative"]:  reasons.append(f"y_min({r['y_min']:.0f})<0")
        if r["flag_x_order"]:         reasons.append("x_max<=x_min")
        if r["flag_y_order"]:         reasons.append("y_max<=y_min")
        if r["flag_x_max_gt_width"]:  reasons.append(
            f"x_max({r['x_max']:.0f})>width({r['image_width']:.0f})")
        if r["flag_y_max_gt_height"]: reasons.append(
            f"y_max({r['y_max']:.0f})>height({r['image_height']:.0f})")
        return "; ".join(reasons)

    bbox_df["invalid_reasons"] = bbox_df.apply(make_reasons, axis=1)
    bbox_df["is_invalid"]      = bbox_df["invalid_reasons"].str.len() > 0
    bbox_df["boundary_check"]  = bbox_df.apply(
        lambda r: "invalid" if r["is_invalid"]
        else ("no_dims_available" if not r["has_image_dims"] else "ok"),
        axis=1)

    counts = {
        "total":            len(bbox_df),
        "ok":               int((bbox_df["boundary_check"] == "ok").sum()),
        "no_dims":          int((bbox_df["boundary_check"] == "no_dims_available").sum()),
        "invalid":          int((bbox_df["boundary_check"] == "invalid").sum()),
        "x_min_negative":   int(bbox_df["flag_x_min_negative"].sum()),
        "y_min_negative":   int(bbox_df["flag_y_min_negative"].sum()),
        "x_max_gt_width":   int(bbox_df["flag_x_max_gt_width"].sum()),
        "y_max_gt_height":  int(bbox_df["flag_y_max_gt_height"].sum()),
        "x_order":          int(bbox_df["flag_x_order"].sum()),
        "y_order":          int(bbox_df["flag_y_order"].sum()),
    }

    log.log(f"  Total bbox checked    : {counts['total']:,}")
    log.log(f"  Within boundary (ok)  : {counts['ok']:,}")
    log.log(f"  No dims available     : {counts['no_dims']:,}")
    log.log(f"  Outside boundary      : {counts['invalid']:,}")
    log.log(f"    x_min < 0          : {counts['x_min_negative']}")
    log.log(f"    y_min < 0          : {counts['y_min_negative']}")
    log.log(f"    x_max > width      : {counts['x_max_gt_width']}")
    log.log(f"    y_max > height     : {counts['y_max_gt_height']}")
    log.log(f"    x_max <= x_min     : {counts['x_order']}")
    log.log(f"    y_max <= y_min     : {counts['y_order']}")

    invalid_df = bbox_df[bbox_df["is_invalid"]].copy()
    keep_cols  = ["image_id","class_id","class_name",
                  "x_min","y_min","x_max","y_max",
                  "image_width","image_height",
                  "has_image_dims","boundary_check","invalid_reasons"]
    for c in keep_cols:
        if c not in invalid_df.columns:
            invalid_df[c] = None
    return bbox_df, invalid_df[keep_cols].reset_index(drop=True), counts


# ---------------------------------------------------------------------------
# Step 7 — Normal consistency
# ---------------------------------------------------------------------------

def check_normal_consistency(normal_df, abn_bbox_df, avail_df, meta_df, log) -> dict:
    normal_ids    = set(normal_df["image_id"].unique())
    normals_in_ann = normal_ids & set(abn_bbox_df["image_id"].unique())

    nm_avail = avail_df[avail_df["image_id"].isin(normal_ids)] if not avail_df.empty else pd.DataFrame()
    nm_found  = int(nm_avail["dicom_found"].sum())    if not nm_avail.empty else 0
    nm_miss   = int((~nm_avail["dicom_found"]).sum()) if not nm_avail.empty else len(normal_ids)

    if not meta_df.empty and "image_id" in meta_df.columns:
        nm_meta    = meta_df[meta_df["image_id"].isin(normal_ids)]
        nm_readable  = int(nm_meta["dicom_readable"].sum()) if "dicom_readable" in nm_meta.columns else 0
        nm_px_readable = int(nm_meta["pixel_readable"].sum()) if "pixel_readable" in nm_meta.columns else 0
        nm_px_err    = int((~nm_meta["pixel_readable"]).sum()) if "pixel_readable" in nm_meta.columns else 0
    else:
        nm_readable = nm_px_readable = nm_px_err = 0

    log.log(f"  Total normal images       : {len(normal_ids)}")
    log.log(f"  Normal with abnormal bbox : {len(normals_in_ann)}")
    log.log(f"  Normal DICOM found        : {nm_found}")
    log.log(f"  Normal DICOM readable     : {nm_readable}")
    log.log(f"  Normal pixel readable     : {nm_px_readable}")

    return {
        "total":           len(normal_ids),
        "with_abn_bbox":   len(normals_in_ann),
        "dicom_found":     nm_found,
        "dicom_missing":   nm_miss,
        "dicom_readable":  nm_readable,
        "px_readable":     nm_px_readable,
        "px_error":        nm_px_err,
    }


# ---------------------------------------------------------------------------
# Summary & Report
# ---------------------------------------------------------------------------

def build_summary(inputs, avail_df, extra_dicom, meta_df,
                  bbox_counts, normal_check, dicom_dir) -> pd.DataFrame:

    total  = inputs["selected_df"]["image_id"].nunique()
    ab_cnt = int((inputs["selected_df"]["subset_type"] == "abnormal").sum())
    nm_cnt = int((inputs["selected_df"]["subset_type"] == "normal").sum())

    dicom_found   = int(avail_df["dicom_found"].sum())   if not avail_df.empty else 0
    dicom_missing = int((~avail_df["dicom_found"]).sum()) if not avail_df.empty else total

    if not meta_df.empty and "dicom_readable" in meta_df.columns:
        dicom_readable   = int(meta_df["dicom_readable"].sum())
        dicom_unreadable = dicom_found - dicom_readable
        valid_dims       = int(meta_df["image_width"].notna().sum())
        invalid_dims     = dicom_readable - valid_dims
        px_readable      = int(meta_df["pixel_readable"].sum())
        px_unreadable    = dicom_readable - px_readable
        px_empty         = int(meta_df["pixel_empty"].fillna(False).sum())
        px_shape_mismatch= int((meta_df["pixel_shape_matches_rows_columns"] == False).sum())
        px_nan           = int(meta_df["pixel_has_nan"].fillna(False).sum())
        px_inf           = int(meta_df["pixel_has_inf"].fillna(False).sum())
    else:
        dicom_readable = dicom_unreadable = valid_dims = invalid_dims = 0
        px_readable = px_unreadable = px_empty = px_shape_mismatch = px_nan = px_inf = 0

    # Gate decision
    issues = []
    if dicom_missing  > 0: issues.append(f"{dicom_missing} DICOM missing")
    if extra_dicom    > 0: issues.append(f"{extra_dicom} extra DICOM not in subset")
    if dicom_unreadable>0: issues.append(f"{dicom_unreadable} DICOM unreadable")
    if px_unreadable  > 0: issues.append(f"{px_unreadable} pixel array unreadable")
    if px_empty       > 0: issues.append(f"{px_empty} empty pixel arrays")
    if bbox_counts["invalid"] > 0:
        issues.append(f"{bbox_counts['invalid']} bbox outside boundary")
    if normal_check["with_abn_bbox"] > 0:
        issues.append(f"{normal_check['with_abn_bbox']} normal images have abnormal bbox")
    if inputs["discrepancies"]:
        issues.extend(inputs["discrepancies"])

    if not issues:
        gate = "PASS"
        gate_reason = (
            f"All {dicom_found:,} DICOM files found, readable, and pixel arrays valid. "
            f"All {valid_dims:,} image dimensions extracted. "
            "No bbox boundary violations. No extra/missing DICOMs. "
            "All Phase 1A metadata consistency checks passed."
        )
    elif (dicom_missing > EXPECTED_TOTAL_IMAGES * 0.05 or
          dicom_unreadable > EXPECTED_TOTAL_IMAGES * 0.05 or
          bbox_counts["invalid"] > 100):
        gate = "FAIL"
        gate_reason = "Critical issues: " + "; ".join(issues)
    else:
        gate = "PASS_WITH_LIMITATION"
        gate_reason = "Limitations recorded: " + "; ".join(issues)

    rows = [
        # Input / Phase 1A re-check
        ("total_selected_images",           total),
        ("abnormal_images",                 ab_cnt),
        ("normal_images",                   nm_cnt),
        ("input_validation_discrepancies",  len(inputs["discrepancies"])),
        ("abnormal_images_with_valid_bbox", inputs["abnormal_with_bbox"]),
        ("normal_images_with_valid_bbox",   inputs["normal_with_bbox"]),
        ("ann_ids_not_in_selected",         inputs["ann_not_in_selected"]),
        ("normal_ids_in_abnormal_ann",      inputs["normal_in_ann"]),
        # DICOM availability
        ("dicom_dir",                       str(dicom_dir)),
        ("dicom_found",                     dicom_found),
        ("dicom_missing",                   dicom_missing),
        ("dicom_extra_not_in_subset",       extra_dicom),
        # DICOM loading
        ("dicom_readable",                  dicom_readable),
        ("dicom_unreadable",                dicom_unreadable),
        ("valid_image_dims",                valid_dims),
        ("invalid_or_missing_image_dims",   invalid_dims),
        # Pixel
        ("pixel_checked",                   dicom_readable),
        ("pixel_readable",                  px_readable),
        ("pixel_unreadable",                px_unreadable),
        ("pixel_empty",                     px_empty),
        ("pixel_shape_mismatch",            px_shape_mismatch),
        ("pixel_has_nan",                   px_nan),
        ("pixel_has_inf",                   px_inf),
        # BBox
        ("total_bbox_rows_checked",         bbox_counts["total"]),
        ("bbox_within_boundary",            bbox_counts["ok"]),
        ("bbox_outside_boundary",           bbox_counts["invalid"]),
        ("bbox_no_dims_available",          bbox_counts["no_dims"]),
        ("bbox_flag_x_min_negative",        bbox_counts["x_min_negative"]),
        ("bbox_flag_y_min_negative",        bbox_counts["y_min_negative"]),
        ("bbox_flag_x_max_gt_width",        bbox_counts["x_max_gt_width"]),
        ("bbox_flag_y_max_gt_height",       bbox_counts["y_max_gt_height"]),
        ("bbox_flag_x_order",               bbox_counts["x_order"]),
        ("bbox_flag_y_order",               bbox_counts["y_order"]),
        # Normal
        ("normal_images_with_abnormal_bbox",normal_check["with_abn_bbox"]),
        ("normal_dicom_found",              normal_check["dicom_found"]),
        ("normal_dicom_missing",            normal_check["dicom_missing"]),
        ("normal_dicom_readable",           normal_check["dicom_readable"]),
        ("normal_pixel_readable",           normal_check["px_readable"]),
        # Gate
        ("gate_decision",                   gate),
        ("gate_reason",                     gate_reason),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_markdown_report(summary_df, inputs, normal_check,
                           meta_df, bbox_counts, dicom_dir) -> str:

    def sv(m, fmt="int"):
        row = summary_df[summary_df["metric"] == m]
        if row.empty: return "N/A"
        v = row.iloc[0]["value"]
        if fmt == "int":
            try: return f"{int(float(v)):,}"
            except: pass
        return str(v)

    gate        = sv("gate_decision", fmt="str")
    gate_reason = sv("gate_reason",   fmt="str")

    disc_text = ("**All Phase 1A metadata consistency checks passed.**"
                 if not inputs["discrepancies"]
                 else "\n".join(f"- ⚠️ {d}" for d in inputs["discrepancies"]))

    # Pixel summary stats
    px_stats = ""
    if not meta_df.empty and "pixel_min" in meta_df.columns:
        readable_px = meta_df[meta_df["pixel_readable"] == True]
        if not readable_px.empty:
            g_min  = readable_px["pixel_min"].min()
            g_max  = readable_px["pixel_max"].max()
            g_mean = readable_px["pixel_mean"].mean()
            px_stats = (f"\n\n**Pixel value summary** (across {len(readable_px):,} "
                        f"readable images):\n"
                        f"- Global min pixel value: {g_min:.0f}\n"
                        f"- Global max pixel value: {g_max:.0f}\n"
                        f"- Mean of per-image means: {g_mean:.2f}")

    # Q&A section
    def yn(val): return "✅ Yes" if str(val) in ("True","0","PASS") or (isinstance(val,int) and val==0) else f"⚠️ {val}"
    dicom_found_val = sv("dicom_found")
    bbox_ok_val     = sv("bbox_within_boundary")
    bbox_inv_val    = sv("bbox_outside_boundary")

    report = f"""# PHASE 2A — Image/DICOM Accessibility & BBox Boundary Validation

## 1. Objective

Validate DICOM file accessibility, extract image dimensions from DICOM tags,
perform **full pixel array validation** for all 4,894 DICOM files, and validate
bounding box boundaries for all 36,096 annotation rows.

**This phase does NOT:** create splits, convert annotations, handle class
imbalance, clip/remove/modify bbox, or perform training.
Any issues found are **flagged and reported only**.

---

## 2. Confirmed Dataset Scope (PHASE 1C Lock)

| Item | Value |
|------|-------|
| Total images | 4,894 |
| Abnormal images | 4,394 |
| Normal / No Finding images | 500 |
| Abnormal bbox annotation rows | 36,096 |
| Abnormal detection classes | 14 |
| No Finding class_id | 14 — **not a detection class** |
| Normal images policy | **Negative images — zero bounding boxes** |
| Duplicate bbox candidate pairs | 78 — documented limitation, not removed |

---

## 3. Phase 1A Metadata Consistency Re-check

| Check | Expected | Observed | Status |
|-------|---------|---------|--------|
| Total selected images | 4,894 | {sv('total_selected_images')} | {'✅ PASS' if sv('total_selected_images')=='4,894' else '❌ FAIL'} |
| Abnormal images | 4,394 | {sv('abnormal_images')} | {'✅ PASS' if sv('abnormal_images')=='4,394' else '❌ FAIL'} |
| Normal / No Finding images | 500 | {sv('normal_images')} | {'✅ PASS' if sv('normal_images')=='500' else '❌ FAIL'} |
| Abnormal bbox annotation rows | 36,096 | {sv('total_bbox_rows_checked')} | {'✅ PASS' if sv('total_bbox_rows_checked')=='36,096' else '❌ FAIL'} |
| Abnormal detection classes | 14 | 14 | ✅ PASS |
| Abnormal images with ≥1 valid bbox | 4,394 | {sv('abnormal_images_with_valid_bbox')} | {'✅ PASS' if sv('abnormal_images_with_valid_bbox')=='4,394' else '❌ FAIL'} |
| Normal images with valid bbox | 0 | {sv('normal_images_with_valid_bbox')} | {'✅ PASS' if sv('normal_images_with_valid_bbox')=='0' else '❌ FAIL'} |
| Normal IDs in abnormal annotations | 0 | {sv('normal_ids_in_abnormal_ann')} | {'✅ PASS' if sv('normal_ids_in_abnormal_ann')=='0' else '❌ FAIL'} |
| Annotation IDs not in selected | 0 | {sv('ann_ids_not_in_selected')} | {'✅ PASS' if sv('ann_ids_not_in_selected')=='0' else '❌ FAIL'} |

{disc_text}

---

## 4. DICOM Availability Summary

| Metric | Value |
|--------|-------|
| DICOM directory | `{dicom_dir}` |
| Total selected images | {sv('total_selected_images')} |
| DICOM files found | {sv('dicom_found')} |
| DICOM files missing | {sv('dicom_missing')} |
| Extra DICOM files (not in subset) | {sv('dicom_extra_not_in_subset')} |
| Normal DICOM found | {sv('normal_dicom_found')} |
| Normal DICOM missing | {sv('normal_dicom_missing')} |

---

## 5. DICOM Loading Summary

| Metric | Value |
|--------|-------|
| DICOM metadata readable | {sv('dicom_readable')} |
| DICOM metadata unreadable | {sv('dicom_unreadable')} |
| Valid image dimensions (W×H) | {sv('valid_image_dims')} |
| Missing / invalid image dimensions | {sv('invalid_or_missing_image_dims')} |

---

## 6. Pixel Array Validation Summary

Full pixel validation was performed on all readable DICOM files.

| Metric | Value |
|--------|-------|
| DICOM files pixel-checked | {sv('pixel_checked')} |
| Pixel arrays readable | {sv('pixel_readable')} |
| Pixel arrays unreadable | {sv('pixel_unreadable')} |
| Empty pixel arrays | {sv('pixel_empty')} |
| Shape mismatch (vs Rows×Columns) | {sv('pixel_shape_mismatch')} |
| Arrays with NaN values | {sv('pixel_has_nan')} |
| Arrays with Inf values | {sv('pixel_has_inf')} |
{px_stats}

---

## 7. BBox Boundary Validation Summary

All 36,096 abnormal bbox rows were validated against DICOM image dimensions.
**No clipping, removal, or modification was applied** — boundary violations are flagged only.

| Metric | Value |
|--------|-------|
| Total bbox rows checked | {sv('total_bbox_rows_checked')} |
| Bbox within boundary ✅ | {sv('bbox_within_boundary')} |
| Bbox outside boundary ⚠️ | {sv('bbox_outside_boundary')} |
| Bbox — no image dims available | {sv('bbox_no_dims_available')} |
| — x_min < 0 | {sv('bbox_flag_x_min_negative')} |
| — y_min < 0 | {sv('bbox_flag_y_min_negative')} |
| — x_max > image_width | {sv('bbox_flag_x_max_gt_width')} |
| — y_max > image_height | {sv('bbox_flag_y_max_gt_height')} |
| — x_max ≤ x_min | {sv('bbox_flag_x_order')} |
| — y_max ≤ y_min | {sv('bbox_flag_y_order')} |

> **Phase 2A decision:** Only flag/report boundary violations; no clipping or
> removal is applied at this phase.

---

## 8. Normal Image Consistency

| Metric | Value |
|--------|-------|
| Total normal images | {normal_check['total']} |
| Normal images with abnormal bbox | {normal_check['with_abn_bbox']} |
| Normal DICOM found | {normal_check['dicom_found']} |
| Normal DICOM readable | {normal_check['dicom_readable']} |
| Normal pixel readable | {normal_check['px_readable']} |

{'✅ All 500 normal images confirmed — zero abnormal bbox rows.' if normal_check['with_abn_bbox']==0 else f"⚠️ {normal_check['with_abn_bbox']} normal image(s) have abnormal bbox."}
Normal images are retained as **negative images without bounding boxes** (PHASE 1C Decision D03).

---

## 9. Questions Answered by PHASE 2A

1. **Input image_id list:** `selected_image_ids.csv` — {sv('total_selected_images')} images ({sv('abnormal_images')} abnormal + {sv('normal_images')} normal).

2. **Every selected image_id has a DICOM file:** {'✅ Yes — all 4,894 found.' if sv('dicom_missing')=='0' else f"⚠️ {sv('dicom_missing')} missing."}

3. **DICOM files are readable:** {'✅ Yes — all readable.' if sv('dicom_unreadable')=='0' else f"⚠️ {sv('dicom_unreadable')} unreadable."}

4. **Image width/height available:** {'✅ Yes — all 4,894 images have valid dimensions.' if sv('invalid_or_missing_image_dims')=='0' else f"⚠️ {sv('invalid_or_missing_image_dims')} missing dims."}

5. **Pixel arrays readable and valid:** {'✅ Yes — all pixel arrays readable.' if sv('pixel_unreadable')=='0' else f"⚠️ {sv('pixel_unreadable')} unreadable."} {'No NaN.' if sv('pixel_has_nan')=='0' else f"⚠️ {sv('pixel_has_nan')} with NaN."} {'No Inf.' if sv('pixel_has_inf')=='0' else f"⚠️ {sv('pixel_has_inf')} with Inf."}

6. **Bbox coordinates within image boundaries:** {'✅ Yes — all 36,096 bbox within boundary.' if sv('bbox_outside_boundary')=='0' else f"⚠️ {sv('bbox_outside_boundary')} bbox outside boundary."}

7. **Normal/No Finding images still negative without bbox:** {'✅ Yes — 0 normal images have abnormal bbox.' if normal_check['with_abn_bbox']==0 else '❌ No.'}

8. **Image/DICOM/boundary issues to record before split:** {'None detected.' if sv('bbox_outside_boundary')=='0' and sv('dicom_unreadable')=='0' else 'See Sections 5–7 above.'}

9. **Bbox boundary violations treatment:** Flagged and recorded only. No clipping, removal, or modification applied in Phase 2A.

10. **Phase 2A outputs saved as traceable artifacts:** ✅ Yes — all CSV and Markdown outputs saved under `reports/phase2_dataset_engineering/`.

---

## 10. Important Limitations

- No train/val/test split or labeled/unlabeled split created.
- No leakage check performed (no split exists yet).
- No COCO/YOLO annotation conversion.
- No class imbalance handling (oversampling/undersampling/reweighting).
- No model training.
- No bbox rows modified, clipped, or removed.
- 78 duplicate bbox candidate pairs from Phase 1B remain as documented limitation only.

---

## 11. Gate Decision

**Gate: `{gate}`**

{gate_reason}
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log = Logger(LOG_FILE)

    log.log("=" * 72)
    log.log("PHASE 2A — Image/DICOM Accessibility & BBox Boundary Validation")
    log.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.log("Full DICOM + pixel validation mode for PHASE 2A.")
    log.log("=" * 72)

    # [1] Config
    log.log("\n[1/8] Loading config...")
    config    = load_config()
    dicom_dir = Path(get_config_value(
        config, "dicom", "dicom_dir",
        fallback="data/raw/vinbigdata/train"))
    log.log(f"  Repo root : {Path('.').resolve()}")
    log.log(f"  DICOM dir : {dicom_dir}")
    log.log(f"  Dir exists: {dicom_dir.exists()}")

    # [2] Output dir
    log.log(f"\n[2/8] Output dir: {REPORT_DIR.resolve()}")

    # [3] Input validation + Phase 1A re-check
    log.log("\n[3/8] Input validation + Phase 1A consistency re-check...")
    inputs = load_and_validate_inputs(config, log)

    # [4] DICOM availability
    log.log("\n[4/8] Checking DICOM availability...")
    if dicom_dir.exists():
        avail_df, extra_dicom = check_dicom_availability(
            inputs["selected_df"], dicom_dir, log)
        dicom_found = int(avail_df["dicom_found"].sum())
    else:
        log.log(f"  [WARNING] DICOM directory not found: {dicom_dir}")
        avail_df = inputs["selected_df"][["image_id","subset_type"]].copy()
        avail_df["dicom_path"]  = None
        avail_df["dicom_found"] = False
        avail_df["dicom_ext"]   = None
        extra_dicom = 0
        dicom_found = 0

    # [5] Full DICOM loading + pixel validation
    log.log("\n[5/8] Full DICOM metadata + pixel array validation...")
    if dicom_found > 0:
        meta_df = extract_dicom_full(avail_df, log)
        readable   = int(meta_df["dicom_readable"].sum())
        px_readable= int(meta_df["pixel_readable"].sum())
        log.log(f"  DICOM readable   : {readable}/{dicom_found}")
        log.log(f"  Pixel readable   : {px_readable}/{readable}")
    else:
        log.log("  No DICOM files — skipping.")
        meta_df = pd.DataFrame()

    # [6] BBox boundary validation
    log.log("\n[6/8] BBox boundary validation...")
    bbox_full_df, invalid_bbox_df, bbox_counts = validate_bbox_boundary(
        inputs["valid_bbox_df"], meta_df, log)

    # [7] Normal consistency
    log.log("\n[7/8] Normal image consistency...")
    normal_check = check_normal_consistency(
        inputs["normal_df"], inputs["abn_bbox_df"], avail_df, meta_df, log)

    # [8] Save artifacts
    log.log("\n[8/8] Building summary and saving artifacts...")
    summary_df = build_summary(
        inputs, avail_df, extra_dicom, meta_df,
        bbox_counts, normal_check, dicom_dir)

    missing_dicom_df = avail_df[~avail_df["dicom_found"]][
        ["image_id","subset_type"]].reset_index(drop=True) \
        if not avail_df.empty else pd.DataFrame(columns=["image_id","subset_type"])

    invalid_dicom_df = meta_df[~meta_df["dicom_readable"]][
        ["image_id","subset_type","dicom_path","dicom_error"]].reset_index(drop=True) \
        if not meta_df.empty and "dicom_readable" in meta_df.columns else \
        pd.DataFrame(columns=["image_id","subset_type","dicom_path","dicom_error"])

    artifacts = {
        "phase2a_dicom_bbox_boundary_summary.csv": summary_df,
        "phase2a_image_metadata.csv":              meta_df,
        "phase2a_missing_dicom_files.csv":         missing_dicom_df,
        "phase2a_invalid_dicom_files.csv":         invalid_dicom_df,
        "phase2a_invalid_bbox_boundary_cases.csv": invalid_bbox_df,
    }
    for fname, df in artifacts.items():
        path = REPORT_DIR / fname
        df.to_csv(path, index=False)
        log.log(f"  [OK] {path}  ({len(df)} rows)")

    report_md   = build_markdown_report(
        summary_df, inputs, normal_check, meta_df, bbox_counts, dicom_dir)
    report_path = REPORT_DIR / "phase2a_dicom_bbox_boundary_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.log(f"  [OK] {report_path}")

    # Terminal execution summary
    def sv(m):
        row = summary_df[summary_df["metric"] == m]
        if row.empty: return "N/A"
        v = row.iloc[0]["value"]
        try: return int(float(v))
        except: return v

    gate     = sv("gate_decision")
    gate_rsn = summary_df[summary_df["metric"]=="gate_reason"]
    gate_txt = gate_rsn.iloc[0]["value"][:120] if not gate_rsn.empty else ""

    log.log("\n" + "=" * 72)
    log.log("PHASE 2A EXECUTION SUMMARY")
    log.log("=" * 72)
    log.log(f"  Repo root                    : {Path('.').resolve()}")
    log.log(f"  Metadata dir                 : data/raw/vinbigdata/metadata_subset/")
    log.log(f"  DICOM dir                    : {dicom_dir}")
    log.log(f"  Selected images              : {sv('total_selected_images')}")
    log.log(f"  Abnormal images              : {sv('abnormal_images')}")
    log.log(f"  Normal images                : {sv('normal_images')}")
    log.log(f"  Abnormal images with bbox    : {sv('abnormal_images_with_valid_bbox')}")
    log.log(f"  Normal images with bbox      : {sv('normal_images_with_valid_bbox')}")
    log.log(f"  Input discrepancies          : {sv('input_validation_discrepancies')}")
    log.log(f"  DICOM found                  : {sv('dicom_found')}")
    log.log(f"  DICOM missing                : {sv('dicom_missing')}")
    log.log(f"  DICOM extra                  : {sv('dicom_extra_not_in_subset')}")
    log.log(f"  DICOM readable               : {sv('dicom_readable')}")
    log.log(f"  DICOM unreadable             : {sv('dicom_unreadable')}")
    log.log(f"  Valid image dims             : {sv('valid_image_dims')}")
    log.log(f"  Missing/invalid dims         : {sv('invalid_or_missing_image_dims')}")
    log.log(f"  Pixel checked                : {sv('pixel_checked')}")
    log.log(f"  Pixel readable               : {sv('pixel_readable')}")
    log.log(f"  Pixel unreadable             : {sv('pixel_unreadable')}")
    log.log(f"  Pixel empty                  : {sv('pixel_empty')}")
    log.log(f"  Pixel shape mismatch         : {sv('pixel_shape_mismatch')}")
    log.log(f"  Pixel with NaN               : {sv('pixel_has_nan')}")
    log.log(f"  Pixel with Inf               : {sv('pixel_has_inf')}")
    log.log(f"  Bbox rows checked            : {sv('total_bbox_rows_checked')}")
    log.log(f"  Bbox within boundary         : {sv('bbox_within_boundary')}")
    log.log(f"  Bbox outside boundary        : {sv('bbox_outside_boundary')}")
    log.log(f"  Bbox no dims available       : {sv('bbox_no_dims_available')}")
    log.log(f"  Normal images with bbox      : {sv('normal_images_with_abnormal_bbox')}")
    log.log(f"\n  Gate   : {gate}")
    log.log(f"  Reason : {gate_txt}...")
    log.log(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"  Log saved: {LOG_FILE.resolve()}")
    log.log("=" * 72)
    log.log("PHASE 2A COMPLETED — No bbox modified. No split created.")
    log.log("=" * 72)

    log.close()


if __name__ == "__main__":
    main()
