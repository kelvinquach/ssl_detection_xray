"""
PHASE 2C — Framework-specific Annotation Format Decision
=========================================================
Workflow lock: WF-SSL-XRAY-DET-V1

Objective:
Evaluate candidate annotation formats (COCO, YOLO, Pascal VOC, custom/JSONL)
against the canonical schema from PHASE 2B and recommend a primary format
direction for the next detection phases.

Strict constraints:
- Do NOT create COCO JSON files.
- Do NOT create YOLO txt label files.
- Do NOT create image splits.
- Do NOT train any model.
- Do NOT generate pseudo-labels.
- Do NOT modify PHASE 2B canonical files.
- This phase is a decision/documentation phase only.

All outputs are written to:
    reports/phase2c_format_decision/
"""

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHASE2B_DIR = Path("reports/phase2b_canonical_schema")
REPORT_DIR  = Path("reports/phase2c_format_decision")

# Evaluation criteria used to score formats
CRITERIA = [
    "multi_class_detection_support",
    "negative_image_support",
    "compatibility_with_canonical_schema",
    "compatibility_with_supervised_detection_baseline",
    "compatibility_with_ssl_object_detection_research",
    "medical_xray_workflow_suitability",
    "ease_of_validation",
    "reproducibility",
    "kaggle_colab_vastai_practicality",
    "rad_id_source_row_traceability_risk",  # lower = less risk
    "paper_thesis_suitability",
]

# Score scale: 1 (poor) → 5 (excellent)
# For traceability_risk: 1 (high risk of losing metadata) → 5 (no risk)
SCORES = {
    "coco_json": {
        "multi_class_detection_support":                   5,
        "negative_image_support":                          5,
        "compatibility_with_canonical_schema":             5,
        "compatibility_with_supervised_detection_baseline": 5,
        "compatibility_with_ssl_object_detection_research": 5,
        "medical_xray_workflow_suitability":               5,
        "ease_of_validation":                              4,
        "reproducibility":                                 5,
        "kaggle_colab_vastai_practicality":                5,
        "rad_id_source_row_traceability_risk":             5,  # custom fields in JSON
        "paper_thesis_suitability":                        5,
    },
    "yolo_txt": {
        "multi_class_detection_support":                   5,
        "negative_image_support":                          2,  # empty file convention
        "compatibility_with_canonical_schema":             3,
        "compatibility_with_supervised_detection_baseline": 4,
        "compatibility_with_ssl_object_detection_research": 3,
        "medical_xray_workflow_suitability":               2,
        "ease_of_validation":                              3,
        "reproducibility":                                 3,
        "kaggle_colab_vastai_practicality":                4,
        "rad_id_source_row_traceability_risk":             1,  # lost in plain txt
        "paper_thesis_suitability":                        3,
    },
    "pascal_voc_xml": {
        "multi_class_detection_support":                   4,
        "negative_image_support":                          3,
        "compatibility_with_canonical_schema":             3,
        "compatibility_with_supervised_detection_baseline": 3,
        "compatibility_with_ssl_object_detection_research": 2,
        "medical_xray_workflow_suitability":               3,
        "ease_of_validation":                              2,
        "reproducibility":                                 3,
        "kaggle_colab_vastai_practicality":                2,
        "rad_id_source_row_traceability_risk":             3,  # possible via XML attrs
        "paper_thesis_suitability":                        3,
    },
    "internal_jsonl": {
        "multi_class_detection_support":                   5,
        "negative_image_support":                          5,
        "compatibility_with_canonical_schema":             5,
        "compatibility_with_supervised_detection_baseline": 2,  # needs custom loader
        "compatibility_with_ssl_object_detection_research": 3,
        "medical_xray_workflow_suitability":               5,
        "ease_of_validation":                              5,
        "reproducibility":                                 5,
        "kaggle_colab_vastai_practicality":                3,
        "rad_id_source_row_traceability_risk":             5,  # native support
        "paper_thesis_suitability":                        4,
    },
}

FORMAT_LABELS = {
    "coco_json":      "COCO JSON",
    "yolo_txt":       "YOLO TXT",
    "pascal_voc_xml": "Pascal VOC XML",
    "internal_jsonl": "Internal JSONL (PHASE 2B)",
}


# ---------------------------------------------------------------------------
# Load & summarise PHASE 2B
# ---------------------------------------------------------------------------

def load_phase2b_summary() -> dict:
    """Read key numbers from PHASE 2B validation summary."""
    summary_path = PHASE2B_DIR / "phase2b_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"PHASE 2B validation summary not found: {summary_path}\n"
            "Please run PHASE 2B first.")

    df = pd.read_csv(summary_path)
    sv = {row["metric"]: row["value"] for _, row in df.iterrows()}

    # Also read a sample from canonical bbox to confirm column set
    bbox_path = PHASE2B_DIR / "canonical_bbox_annotations.csv"
    bbox_cols = list(pd.read_csv(bbox_path, nrows=1).columns)

    # Read class mapping
    cls_path  = PHASE2B_DIR / "canonical_class_mapping.csv"
    cls_df    = pd.read_csv(cls_path)
    det_cls   = cls_df[cls_df["is_detection_class"]]["class_name"].tolist()
    img_cls   = cls_df[~cls_df["is_detection_class"]]["class_name"].tolist()

    return {
        "summary": sv,
        "bbox_cols": bbox_cols,
        "detection_classes": det_cls,
        "image_level_classes": img_cls,
    }


# ---------------------------------------------------------------------------
# Build framework comparison table
# ---------------------------------------------------------------------------

def build_framework_comparison(p2b: dict) -> pd.DataFrame:
    """
    Score each format on each criterion.
    Adds rationale notes per (format, criterion).
    """
    rationale = {
        ("coco_json", "multi_class_detection_support"):
            "Native category list; 14 classes map directly.",
        ("coco_json", "negative_image_support"):
            "Empty 'annotations' list per image_id is standard COCO negative.",
        ("coco_json", "compatibility_with_canonical_schema"):
            "1:1 mapping: each canonical bbox row becomes one COCO annotation. "
            "rad_id and source_row_id stored as extra fields.",
        ("coco_json", "compatibility_with_supervised_detection_baseline"):
            "Detectron2, MMDetection, YOLOv5/v8, DETR, Co-DETR all support COCO natively.",
        ("coco_json", "compatibility_with_ssl_object_detection_research"):
            "STAC, Unbiased Teacher, Soft-Teacher, Semi-DETR, SoftER — all use COCO. "
            "SSL OD benchmark papers universally report on COCO-formatted data.",
        ("coco_json", "medical_xray_workflow_suitability"):
            "JSON supports per-annotation metadata; supports per-image attributes. "
            "Image width/height from PHASE 2A already available.",
        ("coco_json", "ease_of_validation"):
            "pycocotools provides API for loading and validating COCO JSON. "
            "Slightly verbose but well-tooled.",
        ("coco_json", "reproducibility"):
            "Deterministic JSON files under version control. Exact schema documented.",
        ("coco_json", "kaggle_colab_vastai_practicality"):
            "pycocotools pre-installed on Kaggle kernels and most deep-learning Docker images.",
        ("coco_json", "rad_id_source_row_traceability_risk"):
            "Custom extra fields ('rad_id', 'source_row_id') can be added to each "
            "annotation dict without breaking COCO spec. Zero loss of traceability.",
        ("coco_json", "paper_thesis_suitability"):
            "Industry-standard in medical AI and detection papers. "
            "Reviewers expect COCO-format baselines.",

        ("yolo_txt", "multi_class_detection_support"):
            "Supported via class index in txt labels; class names in separate yaml/txt.",
        ("yolo_txt", "negative_image_support"):
            "Non-standard: requires empty .txt file or special yaml flag. "
            "No universal convention — implementation varies by framework version.",
        ("yolo_txt", "compatibility_with_canonical_schema"):
            "Requires coordinate normalisation (pixel → relative 0..1). "
            "Multi-reader: separate file per image cannot natively store rad_id.",
        ("yolo_txt", "compatibility_with_supervised_detection_baseline"):
            "YOLOv5/v8/v9/v10/v11 native. MMDetection supports YOLO via adapter.",
        ("yolo_txt", "compatibility_with_ssl_object_detection_research"):
            "Most SSL OD research codebases (STAC, Unbiased Teacher, SoftER) "
            "use COCO as primary format. YOLO requires conversion wrapper.",
        ("yolo_txt", "medical_xray_workflow_suitability"):
            "Normalised coordinates complicate medical reporting (pixel-space metrics "
            "like IoU at fixed resolution are standard in radiology benchmarks).",
        ("yolo_txt", "ease_of_validation"):
            "Simple line-by-line parsing, but class-count/file-count consistency "
            "hard to validate automatically.",
        ("yolo_txt", "reproducibility"):
            "Normalisation means pixel coordinates are derived; small rounding "
            "differences possible if resolution metadata changes.",
        ("yolo_txt", "kaggle_colab_vastai_practicality"):
            "YOLOv8 (ultralytics) widely available. Fast to train with default CLI.",
        ("yolo_txt", "rad_id_source_row_traceability_risk"):
            "Plain text per-line format has no standard metadata field. "
            "rad_id and source_row_id are lost unless a parallel sidecar file is maintained.",
        ("yolo_txt", "paper_thesis_suitability"):
            "Acceptable for YOLO-specific ablations but less common in academic "
            "detection benchmark comparisons that use COCO metrics.",

        ("pascal_voc_xml", "multi_class_detection_support"):
            "Supported; each <object> tag has <name>.",
        ("pascal_voc_xml", "negative_image_support"):
            "Supported via XML with no <object> tags; somewhat non-standard practice.",
        ("pascal_voc_xml", "compatibility_with_canonical_schema"):
            "One XML file per image; supports extra XML attributes for rad_id. "
            "Verbose and harder to batch-validate.",
        ("pascal_voc_xml", "compatibility_with_supervised_detection_baseline"):
            "Supported by MMDetection and torchvision datasets, but requires adapter.",
        ("pascal_voc_xml", "compatibility_with_ssl_object_detection_research"):
            "SSL OD literature rarely uses Pascal VOC XML as primary format for "
            "new experiments; mostly seen in legacy code (PASCAL VOC 2007/2012).",
        ("pascal_voc_xml", "medical_xray_workflow_suitability"):
            "XML can store rich metadata, but large collections of 4K+ XML files "
            "are cumbersome. No native support for multi-reader provenance.",
        ("pascal_voc_xml", "ease_of_validation"):
            "XML schema validation requires extra tooling. "
            "One-file-per-image makes batch validation slow.",
        ("pascal_voc_xml", "reproducibility"):
            "XML files are reproducible but verbose; large file count.",
        ("pascal_voc_xml", "kaggle_colab_vastai_practicality"):
            "Less common in modern Kaggle competitions and Colab notebooks. "
            "Older format.",
        ("pascal_voc_xml", "rad_id_source_row_traceability_risk"):
            "Possible via custom XML attributes but non-standard; fragile in practice.",
        ("pascal_voc_xml", "paper_thesis_suitability"):
            "Outdated in modern detection research; rarely used in new papers after 2019.",

        ("internal_jsonl", "multi_class_detection_support"):
            "Fully supported; current PHASE 2B JSONL already stores all 14 classes.",
        ("internal_jsonl", "negative_image_support"):
            "Supported via 'boxes': [] for no_finding images. Already implemented.",
        ("internal_jsonl", "compatibility_with_canonical_schema"):
            "Perfect: JSONL IS the canonical format from PHASE 2B.",
        ("internal_jsonl", "compatibility_with_supervised_detection_baseline"):
            "Requires a custom DataLoader; no off-the-shelf framework reads this directly.",
        ("internal_jsonl", "compatibility_with_ssl_object_detection_research"):
            "SSL OD frameworks expect COCO or YOLO; JSONL needs conversion wrapper.",
        ("internal_jsonl", "medical_xray_workflow_suitability"):
            "Ideal for internal validation and provenance tracking.",
        ("internal_jsonl", "ease_of_validation"):
            "One line per image; easy to parse and validate with standard json tools.",
        ("internal_jsonl", "reproducibility"):
            "Deterministic line-by-line JSON. Already version-controlled in repo.",
        ("internal_jsonl", "kaggle_colab_vastai_practicality"):
            "Readable anywhere Python is available; no special library needed. "
            "Not directly usable by training frameworks without adaptation.",
        ("internal_jsonl", "rad_id_source_row_traceability_risk"):
            "Native: rad_id and source_row_id are first-class fields in every box dict.",
        ("internal_jsonl", "paper_thesis_suitability"):
            "Good as supplementary / appendix schema documentation; "
            "primary experiment results should use COCO for comparability.",
    }

    rows = []
    for fmt_key, scores in SCORES.items():
        for criterion, score in scores.items():
            rows.append({
                "format_key":   fmt_key,
                "format_label": FORMAT_LABELS[fmt_key],
                "criterion":    criterion,
                "score":        score,
                "rationale":    rationale.get((fmt_key, criterion), ""),
            })

    df = pd.DataFrame(rows)

    # Add total score per format
    totals = df.groupby("format_key")["score"].sum().rename("total_score").reset_index()
    df = df.merge(totals, on="format_key")
    return df.sort_values(["total_score", "format_key"], ascending=[False, True])


# ---------------------------------------------------------------------------
# Build conversion requirements table
# ---------------------------------------------------------------------------

def build_conversion_requirements(p2b: dict) -> pd.DataFrame:
    """
    Document what conversion steps would be needed from the canonical
    PHASE 2B JSONL to each target format.
    """
    bbox_cols = p2b["bbox_cols"]

    rows = [
        # ── COCO JSON ──────────────────────────────────────────────────
        {
            "target_format":   "COCO JSON",
            "requirement":     "image_metadata",
            "input_source":    "PHASE 2A phase2a_image_metadata.csv",
            "transformation":  "Extract image_id → id, image_width, image_height, filename",
            "custom_fields_needed": "file_name must be constructed (image_id + extension)",
            "traceability_preserved": "YES — rad_id and source_row_id as extra annotation fields",
            "risk":            "LOW — standard JSON field extension",
            "blocker":         "NO",
        },
        {
            "target_format":   "COCO JSON",
            "requirement":     "category_list",
            "input_source":    "canonical_class_mapping.csv (PHASE 2B)",
            "transformation":  "Map canonical_class_id → COCO category id (1-indexed). "
                               "No Finding excluded.",
            "custom_fields_needed": "None — standard COCO 'categories' list",
            "traceability_preserved": "YES",
            "risk":            "LOW — trivial remapping (0-indexed → 1-indexed or keep 0-indexed)",
            "blocker":         "NO",
        },
        {
            "target_format":   "COCO JSON",
            "requirement":     "bbox_format",
            "input_source":    "canonical_bbox_annotations.csv (PHASE 2B)",
            "transformation":  "Convert [x_min, y_min, x_max, y_max] → "
                               "COCO [x_min, y_min, width, height]",
            "custom_fields_needed": "None — computed from existing cols",
            "traceability_preserved": "YES",
            "risk":            "VERY LOW — single arithmetic step",
            "blocker":         "NO",
        },
        {
            "target_format":   "COCO JSON",
            "requirement":     "negative_images",
            "input_source":    "canonical_image_annotations.csv (PHASE 2B)",
            "transformation":  "Images with image_label_type='no_finding' get empty "
                               "'annotations': [] list",
            "custom_fields_needed": "None",
            "traceability_preserved": "YES",
            "risk":            "VERY LOW",
            "blocker":         "NO",
        },
        # ── YOLO TXT ───────────────────────────────────────────────────
        {
            "target_format":   "YOLO TXT",
            "requirement":     "coordinate_normalisation",
            "input_source":    "canonical_bbox_annotations.csv + PHASE 2A image dims",
            "transformation":  "Normalise pixel [x,y,w,h] by image_width/image_height → "
                               "relative [0..1]. Requires image_width/image_height per image.",
            "custom_fields_needed": "None",
            "traceability_preserved": "NO — rad_id and source_row_id lost in plain txt",
            "risk":            "HIGH — metadata loss; requires sidecar solution",
            "blocker":         "PARTIAL — solvable with sidecar JSON",
        },
        {
            "target_format":   "YOLO TXT",
            "requirement":     "negative_images",
            "input_source":    "canonical_image_annotations.csv (PHASE 2B)",
            "transformation":  "Create empty .txt file for no_finding images "
                               "(or use dataset.yaml 'nc' flag with empty files). "
                               "Convention differs by YOLO version.",
            "custom_fields_needed": "Framework-version-specific yaml configuration",
            "traceability_preserved": "N/A",
            "risk":            "MEDIUM — convention varies by YOLOv5 vs v8 vs v9",
            "blocker":         "NO, but requires careful documentation",
        },
        {
            "target_format":   "YOLO TXT",
            "requirement":     "class_names_file",
            "input_source":    "canonical_class_mapping.csv (PHASE 2B)",
            "transformation":  "Write sorted detection class names to data.yaml",
            "custom_fields_needed": "YOLO data.yaml with 'nc', 'names'",
            "traceability_preserved": "YES (at class level)",
            "risk":            "LOW",
            "blocker":         "NO",
        },
        # ── Pascal VOC ─────────────────────────────────────────────────
        {
            "target_format":   "Pascal VOC XML",
            "requirement":     "per_image_xml_generation",
            "input_source":    "canonical_bbox_annotations.csv + PHASE 2A image dims",
            "transformation":  "Generate one XML file per image_id. "
                               "4,894 XML files required.",
            "custom_fields_needed": "Custom XML attributes for rad_id, source_row_id",
            "traceability_preserved": "PARTIAL — possible via custom attributes but non-standard",
            "risk":            "MEDIUM — 4,894 files to manage; no universal reader",
            "blocker":         "NO, but high engineering overhead",
        },
        {
            "target_format":   "Pascal VOC XML",
            "requirement":     "negative_images",
            "input_source":    "canonical_image_annotations.csv (PHASE 2B)",
            "transformation":  "Generate XML with no <object> tags for no_finding images.",
            "custom_fields_needed": "None",
            "traceability_preserved": "YES",
            "risk":            "LOW — standard empty annotation",
            "blocker":         "NO",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build risk assessment markdown
# ---------------------------------------------------------------------------

def build_risk_report(comparison_df: pd.DataFrame,
                      conversion_df: pd.DataFrame,
                      p2b: dict) -> str:

    sv = p2b["summary"]

    # Per-format total scores
    totals = (comparison_df[["format_label","total_score"]]
              .drop_duplicates()
              .sort_values("total_score", ascending=False))

    score_table = "\n".join(
        f"| {r['format_label']} | {int(r['total_score'])} / 55 |"
        for _, r in totals.iterrows()
    )

    # Traceability scores specifically
    trace = comparison_df[
        comparison_df["criterion"] == "rad_id_source_row_traceability_risk"
    ][["format_label","score","rationale"]].sort_values("score", ascending=False)
    trace_table = "\n".join(
        f"| {r['format_label']} | {r['score']} | {r['rationale']} |"
        for _, r in trace.iterrows()
    )

    # SSL OD compatibility scores
    ssl_scores = comparison_df[
        comparison_df["criterion"] == "compatibility_with_ssl_object_detection_research"
    ][["format_label","score","rationale"]].sort_values("score", ascending=False)
    ssl_table = "\n".join(
        f"| {r['format_label']} | {r['score']} | {r['rationale']} |"
        for _, r in ssl_scores.iterrows()
    )

    return f"""# PHASE 2C — Annotation Format Risk Assessment

**Workflow Lock: WF-SSL-XRAY-DET-V1**

---

## 1. Overview

This document assesses risks and tradeoffs of each candidate annotation format
for the VinBigData subset canonical schema (PHASE 2B).

**Dataset facts (from PHASE 2B):**
- Total images: {sv.get('total_selected_images', 'N/A')}
- Abnormal images: {sv.get('total_abnormal_images', 'N/A')} (with bbox)
- No Finding images: {sv.get('total_no_finding_images', 'N/A')} (negatives, no bbox)
- Total bbox rows: {sv.get('total_canonical_bbox_rows', 'N/A')} (all valid)
- Detection classes: {sv.get('num_detection_classes', 'N/A')}
- Readers per image: 3 (rad_id must be preserved)

---

## 2. Overall Score Summary

Scores are 1–5 per criterion × 11 criteria = max 55.

| Format | Total Score |
|--------|------------|
{score_table}

---

## 3. Critical Risk: rad_id / source_row_id Traceability

Multi-reader traceability is a key requirement for this research project
(Kappa analysis in PHASE 1D; future PHASE 6 agreement analysis).

| Format | Score | Notes |
|--------|-------|-------|
{trace_table}

**Decision:** YOLO TXT is the only format that materially risks losing
rad_id and source_row_id traceability. This alone disqualifies it as a
primary format for this research project.

---

## 4. Critical Risk: SSL OD Framework Compatibility

| Format | Score | Notes |
|--------|-------|-------|
{ssl_table}

**Decision:** COCO JSON is the de-facto standard for SSL OD research
(STAC, Unbiased Teacher, Soft-Teacher, Semi-DETR, SoftER, etc.).
Using COCO from the start avoids conversion overhead in training phases.

---

## 5. Negative Image Handling Risk

500 no_finding images must be representable:

| Format | Strategy | Risk |
|--------|---------|------|
| COCO JSON | Empty `"annotations": []` per image | ✅ Zero risk — standard |
| YOLO TXT | Empty .txt file (version-dependent) | ⚠️ Medium — convention varies |
| Pascal VOC XML | XML with no `<object>` tags | ✅ Low risk |
| Internal JSONL | `"boxes": []` already implemented | ✅ Zero risk |

---

## 6. Conversion Effort Risk

| Format | Engineering Effort | Key Dependency |
|--------|-------------------|----------------|
| COCO JSON | Low | image_width/height from PHASE 2A; trivial bbox format change |
| YOLO TXT | Medium | image dims + normalisation + sidecar for traceability |
| Pascal VOC XML | High | 4,894 individual XML files; custom attribute schema |
| Internal JSONL | None | Already done in PHASE 2B |

---

## 7. Risk Summary

| Risk | COCO JSON | YOLO TXT | Pascal VOC | Internal JSONL |
|------|-----------|----------|-----------|----------------|
| Traceability loss | None | **HIGH** | Low | None |
| Negative image handling | Trivial | Medium | Low | Trivial |
| Framework compatibility | Excellent | Good | Poor | Needs adapter |
| Conversion effort (future) | Low | Medium | High | None needed |
| SSL OD research blocker | None | Medium | **High** | Medium |
| Paper/thesis comparability | None | Minor | **High** | Minor |
"""


# ---------------------------------------------------------------------------
# Build format decision report (main report)
# ---------------------------------------------------------------------------

def build_decision_report(comparison_df: pd.DataFrame,
                           conversion_df: pd.DataFrame,
                           p2b: dict) -> str:

    sv = p2b["summary"]

    # Scores pivot for per-format summary
    totals = (comparison_df[["format_label","total_score"]]
              .drop_duplicates()
              .sort_values("total_score", ascending=False))

    # Full comparison table (criteria as rows, formats as columns)
    pivot = comparison_df.pivot_table(
        index="criterion",
        columns="format_label",
        values="score",
        aggfunc="first"
    ).reset_index()

    pivot_rows = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for _, row in pivot.iterrows()
    )
    pivot_header = "| Criterion | " + " | ".join(pivot.columns[1:]) + " |"
    pivot_sep    = "|" + "|".join(["---"] * len(pivot.columns)) + "|"

    # Conversion requirements summary
    conv_coco  = conversion_df[conversion_df["target_format"] == "COCO JSON"]
    conv_risks = conversion_df[conversion_df["risk"].str.contains("HIGH|MEDIUM", na=False)]

    conv_rows = "\n".join(
        f"| {r['target_format']} | {r['requirement']} | {r['risk']} | "
        f"{r['traceability_preserved']} | {r['blocker']} |"
        for _, r in conversion_df.iterrows()
    )

    return f"""# PHASE 2C — Framework-specific Annotation Format Decision

**Workflow Lock: WF-SSL-XRAY-DET-V1**

---

## 1. Objective

Select a primary and secondary annotation format direction for the VinBigData
canonical detection schema (PHASE 2B), based on multi-criteria evaluation.

**This phase does NOT:**
- Create COCO JSON files.
- Create YOLO txt labels.
- Create Pascal VOC XML files.
- Create image splits.
- Train any model.
- Generate pseudo-labels.
- Modify PHASE 2B canonical files.

---

## 2. Canonical Schema Summary (from PHASE 2B)

| Metric | Value |
|--------|-------|
| Total images | {sv.get('total_selected_images', 'N/A')} |
| Abnormal images (with bbox) | {sv.get('total_abnormal_images', 'N/A')} |
| No Finding images (negatives) | {sv.get('total_no_finding_images', 'N/A')} |
| Total bbox rows | {sv.get('total_canonical_bbox_rows', 'N/A')} |
| Detection classes | {sv.get('num_detection_classes', 'N/A')} |
| Image-level classes | {sv.get('num_image_level_classes', 'N/A')} |
| Readers per image | 3 (rad_id preserved) |
| Coordinate system | pixel [x_min, y_min, x_max, y_max] |
| Canonical format | Internal JSONL (PHASE 2B) |

---

## 3. Candidate Formats Evaluated

| # | Format | Key Characteristics |
|---|--------|-------------------|
| 1 | **COCO JSON** | JSON, pixel coords, nested structure, COCO API, 1 file per split |
| 2 | **YOLO TXT** | Per-image txt, normalised coords, class index, ultralytics ecosystem |
| 3 | **Pascal VOC XML** | Per-image XML, pixel coords, legacy Torchvision/MMDetection |
| 4 | **Internal JSONL** | Per-image JSON lines, pixel coords, native rad_id (PHASE 2B) |

---

## 4. Multi-Criteria Evaluation

Scale: 1 (poor) → 5 (excellent).
For `rad_id_source_row_traceability_risk`: 5 = no risk, 1 = high risk of metadata loss.

{pivot_header}
{pivot_sep}
{pivot_rows}

**Total scores (max 55):**

{chr(10).join(f"- {r['format_label']}: **{int(r['total_score'])} / 55**" for _, r in totals.iterrows())}

---

## 5. Conversion Requirements Summary

| Format | Requirement | Risk | Traceability | Blocker |
|--------|------------|------|-------------|---------|
{conv_rows}

---

## 6. Decision

### ✅ Primary Format: COCO JSON

**Rationale:**
1. **SSL OD research standard.** All major SSL OD papers and codebases
   (STAC, Unbiased Teacher, Soft-Teacher, Semi-DETR, SoftER, SemiDETR)
   use COCO-format data. Adopting COCO from PHASE 2D onwards ensures
   zero friction when running or reproducing SSL OD baselines.
2. **Full traceability.** COCO JSON allows arbitrary extra fields in each
   annotation object. `rad_id` and `source_row_id` are preserved as
   custom fields without breaking the COCO spec.
3. **Native negative image support.** `"annotations": []` for no_finding
   images is the standard COCO convention — no ambiguity.
4. **Low conversion cost.** Requires only:
   - image_width/height (already available from PHASE 2A);
   - bbox format change: [x_min, y_min, x_max, y_max] → [x_min, y_min, w, h];
   - category id 0-indexed to COCO 1-indexed (or keep 0-indexed with documentation).
5. **Paper/thesis comparability.** COCO mAP metrics are standard in
   medical AI detection benchmarks.

### 🟡 Secondary Fallback: YOLO TXT (with sidecar JSON)

**Use case:** If a YOLO-native training pipeline is required (e.g., ultralytics
YOLOv8/v9/v11 CLI) and COCO conversion is not available in the target environment.

**Condition:** A sidecar JSON file (`{"{image_id}_meta.json"}`) must be maintained
alongside every .txt file to preserve rad_id and source_row_id traceability.
Without the sidecar, YOLO TXT is disqualified for this research project.

### ❌ Not Recommended: Pascal VOC XML

**Reasons:**
- 4,894 individual XML files — high engineering and storage overhead.
- No modern SSL OD framework uses Pascal VOC XML as primary format.
- Poor paper/thesis comparability (outdated post-2019).

### ✅ Retained: Internal JSONL (PHASE 2B)

The Internal JSONL from PHASE 2B is retained as the canonical reference format.
It is NOT replaced by COCO JSON — both coexist:
- Internal JSONL → source of truth for schema validation, traceability, PHASE 6 analysis.
- COCO JSON → training-pipeline format, generated in PHASE 2D from Internal JSONL.

---

## 7. Recommended Framework Direction

| Category | Recommendation |
|---------|---------------|
| Primary annotation format | **COCO JSON** (with custom fields for rad_id, source_row_id) |
| Secondary fallback | **YOLO TXT** (only with sidecar metadata JSON) |
| Internal canonical | **Internal JSONL** (PHASE 2B, retain as-is) |
| Detection frameworks | Detectron2, MMDetection, or YOLOv8 — all support COCO |
| ViT/attention detector | DINO, Co-DETR, DETR-family — all use COCO format |
| SSL OD framework | STAC, Unbiased Teacher, Soft-Teacher — all use COCO |
| Training data split | **Deferred to PHASE 2E** |
| Labeled/unlabeled split | **Deferred to PHASE 2F** |
| Model training | **Deferred to PHASE 4B+** |

---

## 8. What Is Postponed to the Next Phase

| Item | Phase |
|------|-------|
| COCO JSON file generation | **PHASE 2D** |
| Train/val/test split | **PHASE 2E** |
| Labeled/unlabeled SSL split | **PHASE 2F** |
| ViT/attention detector design | **PHASE 2C / 4B** *(framework selected here)* |
| Model training | **PHASE 4B+** |
| Pseudo-label generation | **PHASE 5** |

---

## 9. Scope Boundary

This report is a decision and documentation phase only.
No annotation files were created, modified, or converted.
No training pipeline was configured.
No split was created.
"""


# ---------------------------------------------------------------------------
# CLAUDE.md / README.md update
# ---------------------------------------------------------------------------

def _update_section(path: Path, title: str, content: str):
    text    = path.read_text(encoding="utf-8") if path.exists() else ""
    marker  = f"## {title}"
    section = f"{marker}\n\n{content.strip()}\n"
    if marker in text:
        before = text[:text.index(marker)]
        rest   = text[text.index(marker) + len(marker):]
        nxt    = rest.find("\n## ")
        after  = rest[nxt:] if nxt >= 0 else ""
        path.write_text(before + section + after, encoding="utf-8")
    else:
        path.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")


def update_docs():
    note = """PHASE 2C — Framework-specific Annotation Format Decision selects COCO JSON
as the primary annotation format for detection training phases.

- Primary format: **COCO JSON** (with rad_id and source_row_id as custom fields).
- Secondary fallback: YOLO TXT (only with sidecar metadata JSON).
- Internal JSONL (PHASE 2B) is retained as canonical reference.
- Decision outputs are under `reports/phase2c_format_decision/`.
- COCO JSON file generation is deferred to **PHASE 2D**.
- Train/val/test split is deferred to **PHASE 2E**.
- Labeled/unlabeled SSL split is deferred to **PHASE 2F**.
"""
    _update_section(Path("CLAUDE.md"), "PHASE 2C Note", note)
    _update_section(Path("README.md"), "PHASE 2C Note", note)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("PHASE 2C — Framework-specific Annotation Format Decision")
    print("Workflow Lock: WF-SSL-XRAY-DET-V1")
    print("=" * 72)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # [1] Load PHASE 2B summary
    print("\n[1/6] Loading PHASE 2B canonical schema summary...")
    p2b = load_phase2b_summary()
    sv  = p2b["summary"]
    print(f"      Images              : {sv.get('total_selected_images')}")
    print(f"      Bbox rows           : {sv.get('total_canonical_bbox_rows')}")
    print(f"      Detection classes   : {sv.get('num_detection_classes')}")
    print(f"      No Finding images   : {sv.get('total_no_finding_images')}")

    # [2] Framework comparison table
    print("\n[2/6] Building framework comparison table...")
    comparison_df = build_framework_comparison(p2b)
    totals = (comparison_df[["format_key","format_label","total_score"]]
              .drop_duplicates()
              .sort_values("total_score", ascending=False))
    for _, r in totals.iterrows():
        print(f"      {r['format_label']:<28}: {int(r['total_score'])} / 55")

    # [3] Conversion requirements
    print("\n[3/6] Building conversion requirements table...")
    conversion_df = build_conversion_requirements(p2b)
    print(f"      Requirements documented: {len(conversion_df)}")

    # [4] Save CSVs
    print("\n[4/6] Saving CSV artifacts...")
    comp_out_path = REPORT_DIR / "phase2c_framework_comparison.csv"
    conv_out_path = REPORT_DIR / "phase2c_conversion_requirements.csv"
    comparison_df.to_csv(comp_out_path, index=False)
    print(f"      [OK] {comp_out_path}  ({len(comparison_df)} rows)")
    conversion_df.to_csv(conv_out_path, index=False)
    print(f"      [OK] {conv_out_path}  ({len(conversion_df)} rows)")

    # [5] Markdown reports
    print("\n[5/6] Writing markdown reports...")
    risk_md   = build_risk_report(comparison_df, conversion_df, p2b)
    risk_path = REPORT_DIR / "phase2c_risk_assessment.md"
    risk_path.write_text(risk_md, encoding="utf-8")
    print(f"      [OK] {risk_path}")

    decision_md   = build_decision_report(comparison_df, conversion_df, p2b)
    decision_path = REPORT_DIR / "phase2c_format_decision_report.md"
    decision_path.write_text(decision_md, encoding="utf-8")
    print(f"      [OK] {decision_path}")

    # [6] Update docs
    print("\n[6/6] Updating CLAUDE.md and README.md...")
    update_docs()
    print("      [OK] CLAUDE.md updated")
    print("      [OK] README.md updated")

    # ── Execution Summary ────────────────────────────────────────────────
    winner = totals.iloc[0]
    print("\n" + "=" * 72)
    print("PHASE 2C EXECUTION SUMMARY")
    print("=" * 72)
    print(f"  Formats evaluated         : {totals['format_label'].tolist()}")
    print(f"  Criteria per format       : {len(CRITERIA)}")
    print()
    for _, r in totals.iterrows():
        marker = " ← PRIMARY" if _ == 0 else (
                 " ← SECONDARY" if r["format_label"] == "YOLO TXT" else "")
        print(f"  {r['format_label']:<28}: {int(r['total_score'])} / 55{marker}")
    print()
    print(f"  Primary format decision   : COCO JSON")
    print(f"  Secondary fallback        : YOLO TXT (with sidecar metadata JSON)")
    print(f"  Internal canonical        : Internal JSONL (PHASE 2B, retained)")
    print(f"  Framework direction       : Detectron2 / MMDetection / YOLOv8 (COCO-compatible)")
    print()
    print(f"  Generated files:")
    for f in [comp_out_path, conv_out_path, risk_path, decision_path]:
        print(f"    {f}")
    print()
    print(f"  CLAUDE.md updated         : YES")
    print(f"  README.md updated         : YES")
    print(f"  COCO JSON created         : NO (deferred to PHASE 2D)")
    print(f"  YOLO TXT created          : NO (deferred to PHASE 2D)")
    print(f"  Split created             : NO")
    print(f"  Model trained             : NO")
    print(f"  Pseudo-labels generated   : NO")
    print(f"  PHASE 2B files modified   : NO")
    print()
    print("  Do NOT commit automatically.")
    print("  Wait for user confirmation after reviewing this summary.")
    print("=" * 72)
    print("PHASE 2C COMPLETED — Format decision documented.")
    print("=" * 72)


if __name__ == "__main__":
    main()
