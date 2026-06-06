"""
PHASE 2B — Canonical Detection Annotation Schema & Class Mapping
=================================================================
Workflow lock: WF-SSL-XRAY-DET-V1

Objective:
Create a canonical, framework-independent and format-agnostic internal
annotation schema for detection. This schema will be used by PHASE 2C
to decide framework-specific formats (COCO / YOLO / Pascal VOC / etc.).

Strict constraints:
- Do NOT assume COCO format.
- Do NOT assume YOLO format.
- Do NOT choose any detection framework.
- Do NOT create train/val/test split.
- Do NOT create labeled/unlabeled split.
- Do NOT train any model.
- Do NOT generate pseudo-labels.
- Do NOT merge multi-reader annotations into consensus boxes.
- Do NOT drop duplicate-looking boxes automatically.
- Do NOT create consensus boxes.

All outputs are written to:
    reports/phase2b_canonical_schema/

PHASE 2C will decide framework-specific formats later.
"""

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEL_PATH = Path("data/raw/vinbigdata/metadata_subset/selected_image_ids.csv")
ANN_PATH = Path("data/raw/vinbigdata/metadata_subset/subset_train_annotations.csv")
REPORT_DIR = Path("reports/phase2b_canonical_schema")
NO_FINDING_CLASS_ID = 14

# Canonical class mapping rules:
# - class_id 0..13 : detection classes (have real bboxes)
# - class_id 14    : "No finding" -> image-level negative label, NOT detection
DETECTION_CLASSES = set(range(0, 14))     # 0..13
IMAGE_LEVEL_CLASSES = {14}                # No finding only


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_inputs():
    """Load selected_image_ids and subset_train_annotations."""
    for p, name in [(SEL_PATH, "selected_image_ids"),
                    (ANN_PATH, "subset_train_annotations")]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p} ({name})")

    sel_df = pd.read_csv(SEL_PATH)
    ann_df = pd.read_csv(ANN_PATH)

    if "image_id" not in sel_df.columns:
        raise ValueError("Missing column 'image_id' in selected_image_ids.csv")
    for col in ["image_id", "class_id", "class_name",
                "rad_id", "x_min", "y_min", "x_max", "y_max"]:
        if col not in ann_df.columns:
            raise ValueError(
                f"Missing column '{col}' in subset_train_annotations.csv")
    return sel_df, ann_df


# ---------------------------------------------------------------------------
# Step 1 — Canonical class mapping
# ---------------------------------------------------------------------------
def build_class_mapping(ann_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build canonical class mapping from all unique (class_id, class_name) pairs.

    Rules:
    - Detection classes (class_id 0..13): canonical_class_id == original_class_id.
    - No finding (class_id 14): image-level negative label, canonical_class_id null.
    """
    pairs = (ann_df[["class_id", "class_name"]]
             .drop_duplicates()
             .sort_values("class_id")
             .reset_index(drop=True))

    rows = []
    for _, row in pairs.iterrows():
        cid = int(row["class_id"])
        name = row["class_name"]
        is_det = cid in DETECTION_CLASSES
        is_img = cid in IMAGE_LEVEL_CLASSES
        has_bbox = bool(ann_df.loc[ann_df["class_id"] == cid, "x_min"].notna().any())

        if is_det:
            can_id = cid
            notes = ("Abnormal detection class with bounding-box annotations. "
                     "canonical_class_id mirrors original_class_id.")
        else:
            can_id = None
            notes = ("Image-level negative label (No finding). "
                     "Not a detection object class. Excluded from bbox-level "
                     "detection table. canonical_class_id is null.")

        rows.append({
            "canonical_class_id": can_id,
            "original_class_id": cid,
            "class_name": name,
            "is_detection_class": is_det,
            "is_image_level_class": is_img,
            "has_bbox": has_bbox,
            "notes": notes,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2 — Canonical bbox-level annotation table
# ---------------------------------------------------------------------------
def build_canonical_bbox(ann_df: pd.DataFrame,
                         class_map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build canonical bbox-level annotation table.

    Rules:
    - Exclude No Finding (class_id == 14).
    - Keep rad_id for multi-reader traceability.
    - Keep source_row_id (original row index in the annotation file).
    - Do NOT merge boxes. Do NOT drop duplicate-looking boxes.
    - Do NOT create consensus boxes. Only flag issues in bbox_status.
    """
    can_id_map = {int(r["original_class_id"]): r["canonical_class_id"]
                  for _, r in class_map_df.iterrows()}

    # source_row_id = original positional row index in the annotation CSV
    bbox_df = ann_df.reset_index().rename(columns={"index": "source_row_id"})
    bbox_df = bbox_df[bbox_df["class_id"] != NO_FINDING_CLASS_ID].copy()

    bbox_df["original_class_id"] = bbox_df["class_id"].astype(int)
    bbox_df["canonical_class_id"] = bbox_df["original_class_id"].map(can_id_map)
    bbox_df["bbox_width"] = bbox_df["x_max"] - bbox_df["x_min"]
    bbox_df["bbox_height"] = bbox_df["y_max"] - bbox_df["y_min"]
    bbox_df["bbox_area"] = bbox_df["bbox_width"] * bbox_df["bbox_height"]

    def check_validity(row):
        if any(pd.isna(row[c]) for c in ("x_min", "y_min", "x_max", "y_max")):
            return False, "missing_coordinates"
        if row["x_max"] <= row["x_min"]:
            return False, "invalid_x_order"
        if row["y_max"] <= row["y_min"]:
            return False, "invalid_y_order"
        if row["x_min"] < 0 or row["y_min"] < 0:
            return False, "negative_coordinate"
        if row["bbox_area"] <= 0:
            return False, "zero_or_negative_area"
        return True, "valid"

    res = bbox_df.apply(check_validity, axis=1)
    bbox_df["is_valid_bbox"] = res.apply(lambda x: x[0])
    bbox_df["bbox_status"] = res.apply(lambda x: x[1])

    out_cols = [
        "image_id", "source_row_id", "rad_id",
        "original_class_id", "canonical_class_id", "class_name",
        "x_min", "y_min", "x_max", "y_max",
        "bbox_width", "bbox_height", "bbox_area",
        "is_valid_bbox", "bbox_status",
    ]
    return bbox_df[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 — Canonical image-level annotation table (vectorized)
# ---------------------------------------------------------------------------
def build_canonical_image(sel_df: pd.DataFrame,
                          ann_df: pd.DataFrame,
                          bbox_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per selected image_id.

    image_label_type:
    - "abnormal"          : >=1 valid abnormal bbox AND no No-Finding label
    - "no_finding"        : No-Finding label AND no valid abnormal bbox
    - "mixed_or_conflict" : both present
    - "unknown"           : neither present
    """
    img_ids = sel_df["image_id"].drop_duplicates().reset_index(drop=True)
    out = pd.DataFrame({"image_id": img_ids})

    num_raw = ann_df.groupby("image_id").size()
    reader_cnt = ann_df.groupby("image_id")["rad_id"].nunique()
    num_bbox = bbox_df.groupby("image_id").size()

    valid = bbox_df[bbox_df["is_valid_bbox"]]
    num_valid = valid.groupby("image_id").size()
    det_classes = valid.groupby("image_id")["original_class_id"].nunique()

    nf_images = set(
        ann_df.loc[ann_df["class_id"] == NO_FINDING_CLASS_ID, "image_id"].unique())

    out["num_raw_rows"] = out["image_id"].map(num_raw).fillna(0).astype(int)
    out["num_bbox_rows"] = out["image_id"].map(num_bbox).fillna(0).astype(int)
    out["num_abnormal_boxes"] = out["image_id"].map(num_valid).fillna(0).astype(int)
    out["num_detection_classes_present"] = (
        out["image_id"].map(det_classes).fillna(0).astype(int))
    out["has_abnormality"] = out["num_abnormal_boxes"] > 0
    out["has_no_finding_label"] = out["image_id"].isin(nf_images)
    out["reader_count"] = out["image_id"].map(reader_cnt).fillna(0).astype(int)

    def label_type(r):
        abn, nf = r["has_abnormality"], r["has_no_finding_label"]
        if abn and not nf:
            return "abnormal"
        if nf and not abn:
            return "no_finding"
        if abn and nf:
            return "mixed_or_conflict"
        return "unknown"

    out["image_label_type"] = out.apply(label_type, axis=1)

    cols = [
        "image_id", "num_raw_rows", "num_bbox_rows", "num_abnormal_boxes",
        "num_detection_classes_present", "has_abnormality",
        "has_no_finding_label", "image_label_type", "reader_count",
    ]
    return out[cols]


# ---------------------------------------------------------------------------
# Step 4 — Export canonical JSONL
# ---------------------------------------------------------------------------
def export_canonical_jsonl(image_df: pd.DataFrame,
                           bbox_df: pd.DataFrame,
                           out_path: Path) -> int:
    """One JSON object per line per image. Internal canonical format only."""
    valid = bbox_df[bbox_df["is_valid_bbox"]]
    bbox_by_image = {}
    for _, row in valid.iterrows():
        bbox_by_image.setdefault(row["image_id"], []).append({
            "class_id": int(row["original_class_id"]),
            "class_name": row["class_name"],
            "bbox_xyxy": [float(row["x_min"]), float(row["y_min"]),
                          float(row["x_max"]), float(row["y_max"])],
            "rad_id": row["rad_id"],
            "source_row_id": int(row["source_row_id"]),
        })

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, img in image_df.iterrows():
            record = {
                "image_id": img["image_id"],
                "image_label_type": img["image_label_type"],
                "boxes": bbox_by_image.get(img["image_id"], []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Step 5 — Validation
# ---------------------------------------------------------------------------
def validate_schema(sel_df, ann_df, class_map_df, bbox_df, image_df):
    warnings = []
    checks = []

    total_sel = sel_df["image_id"].nunique()
    total_img_rows = len(image_df)
    total_bbox = len(bbox_df)
    total_valid = int(bbox_df["is_valid_bbox"].sum())
    total_invalid = total_bbox - total_valid
    total_nf = int((image_df["image_label_type"] == "no_finding").sum())
    total_abn = int((image_df["image_label_type"] == "abnormal").sum())
    total_mixed = int((image_df["image_label_type"] == "mixed_or_conflict").sum())
    total_unknown = int((image_df["image_label_type"] == "unknown").sum())
    n_det_classes = int(class_map_df["is_detection_class"].sum())
    n_img_classes = int(class_map_df["is_image_level_class"].sum())

    def add(check, expected, observed, ok, level="FAIL", warn=None):
        checks.append({"check": check, "expected": expected,
                       "observed": observed,
                       "status": "PASS" if ok else level})
        if not ok and warn:
            warnings.append(warn)

    add("image_count_consistency", total_sel, total_img_rows,
        total_sel == total_img_rows,
        warn=f"Image count mismatch: selected={total_sel}, image_df={total_img_rows}")

    orig_pairs = set(ann_df[["class_id", "class_name"]]
                     .drop_duplicates().itertuples(index=False, name=None))
    can_pairs = set(class_map_df[["original_class_id", "class_name"]]
                    .itertuples(index=False, name=None))
    add("class_id_name_consistency", f"{len(orig_pairs)} unique pairs",
        f"{len(can_pairs)} canonical pairs", orig_pairs == can_pairs,
        level="WARNING",
        warn="class_id/class_name pairs differ between annotation and canonical table.")

    nf_in_bbox = int((bbox_df["original_class_id"] == NO_FINDING_CLASS_ID).sum())
    add("no_finding_excluded_from_bbox_table", 0, nf_in_bbox, nf_in_bbox == 0,
        warn=f"No finding appears {nf_in_bbox} times in bbox table!")

    add("invalid_bbox_count", 0, total_invalid, total_invalid == 0,
        level="WARNING",
        warn=f"{total_invalid} invalid bbox rows found (flagged, not removed).")

    missing_src = int(bbox_df["source_row_id"].isna().sum())
    add("source_row_id_completeness", 0, missing_src, missing_src == 0)

    missing_rad = int(bbox_df["rad_id"].isna().sum())
    add("rad_id_completeness", 0, missing_rad, missing_rad == 0)

    none_can = int(bbox_df["canonical_class_id"].isna().sum())
    add("canonical_class_id_completeness_in_bbox", 0, none_can, none_can == 0)

    add("mixed_or_conflict_images", 0, total_mixed, total_mixed == 0,
        level="WARNING",
        warn=f"{total_mixed} images have both No-Finding and abnormal bbox labels "
             f"(flagged for PHASE 2C review, not resolved here).")

    add("unknown_label_type_images", 0, total_unknown, total_unknown == 0,
        level="WARNING",
        warn=f"{total_unknown} images have unknown label type.")

    summary_rows = [
        ("total_selected_images", total_sel),
        ("total_canonical_image_rows", total_img_rows),
        ("total_canonical_bbox_rows", total_bbox),
        ("total_valid_abnormal_boxes", total_valid),
        ("total_invalid_bbox_rows", total_invalid),
        ("total_no_finding_images", total_nf),
        ("total_abnormal_images", total_abn),
        ("total_mixed_or_conflict_images", total_mixed),
        ("total_unknown_label_images", total_unknown),
        ("num_detection_classes", n_det_classes),
        ("num_image_level_classes", n_img_classes),
        ("overall_status",
         "PASS" if not warnings else f"WARNING ({len(warnings)} issue(s))"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    check_df = pd.DataFrame(checks)
    return summary_df, check_df, warnings


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------
def build_validation_report(summary_df, check_df, warnings, class_map_df):
    def sv(m):
        row = summary_df[summary_df["metric"] == m]
        if row.empty:
            return "N/A"
        v = row.iloc[0]["value"]
        try:
            return f"{int(float(v)):,}"
        except (ValueError, TypeError):
            return str(v)

    check_rows = "\n".join(
        f"| {r['check']} | {r['expected']} | {r['observed']} | {r['status']} |"
        for _, r in check_df.iterrows())

    cls_rows = "\n".join(
        f"| {'' if pd.isna(r['canonical_class_id']) else int(r['canonical_class_id'])} "
        f"| {int(r['original_class_id'])} | {r['class_name']} "
        f"| {r['is_detection_class']} | {r['is_image_level_class']} "
        f"| {r['has_bbox']} |"
        for _, r in class_map_df.iterrows())

    warn_section = ("\n".join(f"- WARNING: {w}" for w in warnings)
                    if warnings else "No warnings.")

    return f"""# PHASE 2B — Canonical Detection Annotation Schema: Validation Report

**Workflow Lock: WF-SSL-XRAY-DET-V1**

---

## 1. Phase Constraints

This phase is **framework-independent and format-agnostic**.

- Does NOT convert to COCO, YOLO, or Pascal VOC.
- Does NOT create train/val/test split.
- Does NOT create labeled/unlabeled split.
- Does NOT merge multi-reader annotations.
- Does NOT create consensus boxes.
- Does NOT train any model.
- Does NOT generate pseudo-labels.
- All outputs are under `reports/phase2b_canonical_schema/`.
- PHASE 2C will decide framework-specific formats.

---

## 2. Summary Statistics

| Metric | Value |
|--------|-------|
| Total selected images | {sv('total_selected_images')} |
| Total canonical image rows | {sv('total_canonical_image_rows')} |
| Total canonical bbox rows | {sv('total_canonical_bbox_rows')} |
| Total valid abnormal boxes | {sv('total_valid_abnormal_boxes')} |
| Total invalid bbox rows | {sv('total_invalid_bbox_rows')} |
| Total No Finding images | {sv('total_no_finding_images')} |
| Total abnormal images | {sv('total_abnormal_images')} |
| Total mixed/conflict images | {sv('total_mixed_or_conflict_images')} |
| Total unknown label images | {sv('total_unknown_label_images')} |
| Number of detection classes | {sv('num_detection_classes')} |
| Number of image-level classes | {sv('num_image_level_classes')} |
| **Overall status** | **{sv('overall_status')}** |

---

## 3. Class Mapping

| canonical_id | original_id | class_name | is_detection | is_image_level | has_bbox |
|:---:|:---:|---|:---:|:---:|:---:|
{cls_rows}

> `canonical_class_id` is null for No Finding — it is not a detection object class.

---

## 4. Validation Checks

| Check | Expected | Observed | Status |
|-------|---------|---------|--------|
{check_rows}

---

## 5. Warnings

{warn_section}

> NOTE: WARNING items are *flagged only*. PHASE 2B does not delete, merge, or
> resolve them. They are handed to later phases for decision.

---

## 6. Label Type Distribution

| Label Type | Count |
|-----------|-------|
| abnormal | {sv('total_abnormal_images')} |
| no_finding | {sv('total_no_finding_images')} |
| mixed_or_conflict | {sv('total_mixed_or_conflict_images')} |
| unknown | {sv('total_unknown_label_images')} |

---

## 7. Multi-Reader Note

Every bbox row preserves `rad_id` for multi-reader traceability and
`source_row_id` for linkage back to `subset_train_annotations.csv`.
No boxes were merged or dropped. Each reader's annotation is kept as an
independent row in the canonical bbox table. Duplicate-looking boxes
(if any) are preserved as-is and only flagged, never removed.

---

## 8. Next Phase

PHASE 2C will use these canonical outputs to decide the framework-specific
annotation format (COCO / YOLO / Pascal VOC / other).
"""


def build_schema_definition(class_map_df, summary_df):
    def sv(m):
        row = summary_df[summary_df["metric"] == m]
        return "N/A" if row.empty else row.iloc[0]["value"]

    cls_rows = "\n".join(
        f"| {'' if pd.isna(r['canonical_class_id']) else int(r['canonical_class_id'])} "
        f"| {int(r['original_class_id'])} | {r['class_name']} "
        f"| {'Yes' if r['is_detection_class'] else 'No'} |"
        for _, r in class_map_df.iterrows())

    return f"""# PHASE 2B — Canonical Detection Annotation Schema Definition

**Workflow Lock: WF-SSL-XRAY-DET-V1**
**Format: Framework-independent internal canonical format**

---

## Schema Overview

- Phase: PHASE 2B
- Dataset: VinBigData Chest X-ray Abnormalities Detection (metadata-only working subset)
- Total selected images: {sv('total_selected_images')}
- Detection classes: {sv('num_detection_classes')} (class_id 0–13)
- Image-level classes: {sv('num_image_level_classes')} (No finding, class_id 14)

---

## 1. Class Mapping Schema (`canonical_class_mapping.csv`)

| canonical_class_id | original_class_id | class_name | is_detection_class |
|:------------------:|:-----------------:|-----------|:-----------------:|
{cls_rows}

Rules:
- `canonical_class_id` = `original_class_id` for detection classes (0–13).
- No finding (`original_class_id = 14`) has `canonical_class_id = null` and is
  an image-level negative label, NOT a detection object class.

---

## 2. BBox Annotation Schema (`canonical_bbox_annotations.csv`)

| Column | Type | Description |
|--------|------|-------------|
| image_id | string | Unique image identifier |
| source_row_id | int | Original row index in `subset_train_annotations.csv` (traceability) |
| rad_id | string | Radiologist / reader identifier |
| original_class_id | int | Class ID from original annotation (0–13) |
| canonical_class_id | int | Canonical class ID (same as original for detection classes) |
| class_name | string | Human-readable class name |
| x_min, y_min, x_max, y_max | float | BBox corners in pixel space (origin = top-left) |
| bbox_width | float | x_max - x_min |
| bbox_height | float | y_max - y_min |
| bbox_area | float | bbox_width * bbox_height |
| is_valid_bbox | bool | True if all coordinates are valid |
| bbox_status | string | "valid" or an error description |

Rules: No Finding excluded; multi-reader rows preserved; no merging, no
consensus boxes, no automatic duplicate removal.

---

## 3. Image Annotation Schema (`canonical_image_annotations.csv`)

| Column | Type | Description |
|--------|------|-------------|
| image_id | string | Unique image identifier |
| num_raw_rows | int | Total annotation rows (all readers, all classes) |
| num_bbox_rows | int | Rows in bbox table for this image |
| num_abnormal_boxes | int | Count of valid abnormal bbox rows |
| num_detection_classes_present | int | Unique detection classes present |
| has_abnormality | bool | True if >=1 valid abnormal bbox |
| has_no_finding_label | bool | True if any reader labeled No Finding |
| image_label_type | string | abnormal / no_finding / mixed_or_conflict / unknown |
| reader_count | int | Unique readers who annotated this image |

`image_label_type` rules:
- `abnormal`: has_abnormality AND NOT has_no_finding_label
- `no_finding`: has_no_finding_label AND NOT has_abnormality
- `mixed_or_conflict`: both True (requires later review)
- `unknown`: both False (requires later review)

---

## 4. JSONL Schema (`canonical_detection_annotations.jsonl`)

One JSON object per line. Internal canonical format — NOT COCO, NOT YOLO.

```json
{{
  "image_id": "<string>",
  "image_label_type": "abnormal | no_finding | mixed_or_conflict | unknown",
  "boxes": [
    {{
      "class_id": <int>,
      "class_name": "<string>",
      "bbox_xyxy": [x_min, y_min, x_max, y_max],
      "rad_id": "<reader_id>",
      "source_row_id": <int>
    }}
  ]
}}
```

- For No Finding images: `"boxes": []`.
- `bbox_xyxy` is [x_min, y_min, x_max, y_max] in pixel space (origin = top-left).
- Converted to a framework-specific format only in PHASE 2C.

---

## 5. Design Decisions

1. No consensus boxes — each reader's bbox is an independent row.
2. No automatic duplicate removal — flagged only.
3. No framework assumption — no COCO categories, YOLO labels, or VOC XML.
4. Traceability — every row links back via `source_row_id` and keeps `rad_id`.
5. No Finding handling — negative image-level label only; absent from bbox table.
"""


# ---------------------------------------------------------------------------
# CLAUDE.md / README.md update helpers
# ---------------------------------------------------------------------------
def _update_file(path: Path, section_title: str, section_content: str):
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = f"## {section_title}"
    if marker in current:
        before = current[:current.index(marker)]
        rest = current[current.index(marker) + len(marker):]
        nxt = rest.find("\n## ")
        after = rest[nxt:] if nxt >= 0 else ""
        new = before + marker + "\n\n" + section_content.strip() + "\n" + after
    else:
        new = current.rstrip() + "\n\n" + marker + "\n\n" + section_content.strip() + "\n"
    path.write_text(new, encoding="utf-8")


def update_docs():
    note = """PHASE 2B creates a canonical, framework-independent and format-agnostic
detection annotation schema.

- Outputs are stored under `reports/phase2b_canonical_schema/`.
- Does NOT convert to COCO/YOLO/Pascal VOC.
- Does NOT create train/val/test or labeled/unlabeled splits.
- Does NOT train models or generate pseudo-labels.
- Multi-reader annotations are preserved (no consensus/merging).
- PHASE 2C will decide the framework-specific annotation format later.
"""
    _update_file(Path("CLAUDE.md"), "PHASE 2B Note", note)
    _update_file(Path("README.md"), "PHASE 2B Note", note)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("PHASE 2B — Canonical Detection Annotation Schema & Class Mapping")
    print("Workflow Lock: WF-SSL-XRAY-DET-V1 | Framework-independent | Format-agnostic")
    print("=" * 72)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/8] Loading metadata inputs...")
    sel_df, ann_df = load_inputs()
    print(f"      selected_image_ids       : {sel_df.shape}")
    print(f"      subset_train_annotations : {ann_df.shape}")

    print("\n[2/8] Building canonical class mapping...")
    class_map_df = build_class_mapping(ann_df)
    print(f"      Total classes      : {len(class_map_df)}")
    print(f"      Detection classes  : {int(class_map_df['is_detection_class'].sum())}")
    print(f"      Image-level classes: {int(class_map_df['is_image_level_class'].sum())}")

    print("\n[3/8] Building canonical bbox annotation table...")
    bbox_df = build_canonical_bbox(ann_df, class_map_df)
    valid_count = int(bbox_df["is_valid_bbox"].sum())
    print(f"      Total bbox rows  : {len(bbox_df):,}")
    print(f"      Valid bbox rows  : {valid_count:,}")
    print(f"      Invalid bbox rows: {len(bbox_df) - valid_count:,}")

    print("\n[4/8] Building canonical image annotation table...")
    image_df = build_canonical_image(sel_df, ann_df, bbox_df)
    for lt, cnt in sorted(image_df["image_label_type"].value_counts().to_dict().items()):
        print(f"      {lt:<20}: {cnt:,}")

    print("\n[5/8] Exporting canonical JSONL...")
    jsonl_path = REPORT_DIR / "canonical_detection_annotations.jsonl"
    n_written = export_canonical_jsonl(image_df, bbox_df, jsonl_path)
    print(f"      Written {n_written:,} image records -> {jsonl_path}")

    print("\n[6/8] Validating schema...")
    summary_df, check_df, warnings = validate_schema(
        sel_df, ann_df, class_map_df, bbox_df, image_df)
    overall = summary_df.loc[summary_df["metric"] == "overall_status", "value"].values[0]
    print(f"      Checks run: {len(check_df)} | Warnings: {len(warnings)} | Status: {overall}")
    for w in warnings:
        print(f"      WARNING: {w}")

    print("\n[7/8] Saving artifacts...")
    files = {
        "canonical_class_mapping.csv": class_map_df,
        "canonical_bbox_annotations.csv": bbox_df,
        "canonical_image_annotations.csv": image_df,
        "phase2b_validation_summary.csv": summary_df,
    }
    for fname, df in files.items():
        p = REPORT_DIR / fname
        df.to_csv(p, index=False)
        print(f"      [OK] {p}  ({len(df):,} rows)")

    (REPORT_DIR / "phase2b_validation_report.md").write_text(
        build_validation_report(summary_df, check_df, warnings, class_map_df),
        encoding="utf-8")
    print(f"      [OK] {REPORT_DIR / 'phase2b_validation_report.md'}")

    (REPORT_DIR / "phase2b_schema_definition.md").write_text(
        build_schema_definition(class_map_df, summary_df), encoding="utf-8")
    print(f"      [OK] {REPORT_DIR / 'phase2b_schema_definition.md'}")

    print("\n[8/8] Updating CLAUDE.md and README.md...")
    update_docs()
    print("      [OK] CLAUDE.md and README.md updated")

    def sv(m):
        v = summary_df.loc[summary_df["metric"] == m, "value"].values[0]
        try:
            return f"{int(float(v)):,}"
        except (ValueError, TypeError):
            return str(v)

    print("\n" + "=" * 72)
    print("PHASE 2B EXECUTION SUMMARY")
    print("=" * 72)
    print(f"  Total selected images       : {sv('total_selected_images')}")
    print(f"  Canonical image rows        : {sv('total_canonical_image_rows')}")
    print(f"  Canonical bbox rows         : {sv('total_canonical_bbox_rows')}")
    print(f"  Valid abnormal boxes        : {sv('total_valid_abnormal_boxes')}")
    print(f"  Invalid bbox rows           : {sv('total_invalid_bbox_rows')}")
    print(f"  No Finding images           : {sv('total_no_finding_images')}")
    print(f"  Abnormal images             : {sv('total_abnormal_images')}")
    print(f"  Mixed/conflict images       : {sv('total_mixed_or_conflict_images')}")
    print(f"  Unknown-label images        : {sv('total_unknown_label_images')}")
    print(f"  Detection classes           : {sv('num_detection_classes')}")
    print(f"  Image-level classes         : {sv('num_image_level_classes')}")
    print(f"  JSONL records written       : {n_written:,}")
    print(f"  Warnings                    : {len(warnings)}")
    print(f"  Overall status              : {overall}")
    print()
    print("  Framework assumed: NO | Split created: NO | Model trained: NO")
    print("  Pseudo-labels: NO | Boxes merged/dropped: NO")
    print("  Do NOT commit automatically. Wait for user confirmation.")
    print("=" * 72)
    print("PHASE 2B COMPLETED — Canonical schema ready for PHASE 2C.")
    print("=" * 72)


if __name__ == "__main__":
    main()
