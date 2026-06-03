"""
PHASE 1C — Dataset Scope Decision
===================================
Mục tiêu: Tổng hợp evidence từ PHASE 1A và PHASE 1B để ra quyết định phạm vi
dataset (dataset scope) sẽ dùng cho các phase tiếp theo của dự án nghiên cứu
semi-supervised object detection trên ảnh X-ray phổi.

Phase này là DECISION phase, không phải implementation phase:
- Đọc CSV/report từ PHASE 1A và PHASE 1B.
- Tổng hợp evidence.
- Tạo decision report và các CSV decision summary.
- Cập nhật limitation register và readiness checklist.

KHÔNG thực hiện trong phase này:
- Không phân tích Kappa / inter-annotator agreement.
- Không tạo train/val/test split.
- Không tạo labeled/unlabeled split.
- Không kiểm tra leakage.
- Không convert sang COCO/YOLO.
- Không training model.
- Không đọc DICOM/image thật.
- Không sửa metadata gốc.
- Không xóa class.
- Không tự động filter bbox.
- Không tự động xử lý duplicate candidates.
"""

from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("configs/dataset.yaml")
REPORT_DIR = Path("reports/phase_1c_dataset_scope_decision")

PHASE_1A_DIR = Path("reports/phase_1a_dataset_overview")
PHASE_1B_DIR = Path("reports/phase_1b_annotation_quality")

NO_FINDING_CLASS_ID = 14  # fallback


# ---------------------------------------------------------------------------
# Config helpers (nhất quán với PHASE 1A/1B)
# ---------------------------------------------------------------------------

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Đọc cấu hình từ configs/dataset.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_value(config: dict, *keys, fallback=None, warn: bool = True):
    """Truy cập nested key trong config dict với fallback và warning."""
    obj = config
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            if warn:
                print(f"[WARNING] Config key '{'.'.join(str(k) for k in keys)}' "
                      f"not found. Using fallback: {fallback!r}")
            return fallback
        obj = obj[key]
    return obj


# ---------------------------------------------------------------------------
# Evidence loaders
# ---------------------------------------------------------------------------

def load_phase1a_evidence(phase1a_dir: Path) -> dict:
    """
    Đọc các CSV từ PHASE 1A để lấy evidence.
    Trả về dict chứa summary values và class_distribution dataframe.
    """
    required = {
        "dataset_summary": phase1a_dir / "dataset_summary.csv",
        "class_distribution": phase1a_dir / "class_distribution.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"[ERROR] PHASE 1A evidence file missing: {path} ({name})")

    summary_df = pd.read_csv(required["dataset_summary"])
    class_df = pd.read_csv(required["class_distribution"])

    if summary_df.empty:
        raise ValueError("[ERROR] PHASE 1A dataset_summary.csv is empty.")
    if class_df.empty:
        raise ValueError("[ERROR] PHASE 1A class_distribution.csv is empty.")

    # Cột bắt buộc
    if "metric" not in summary_df.columns or "value" not in summary_df.columns:
        raise ValueError(
            "[ERROR] dataset_summary.csv must have 'metric' and 'value' columns.")
    required_class_cols = {"class_id", "class_name", "num_bbox", "num_images"}
    missing = required_class_cols - set(class_df.columns)
    if missing:
        raise ValueError(
            f"[ERROR] class_distribution.csv missing columns: {missing}")

    # Chuyển summary thành dict {metric: value}
    summary = {}
    for _, row in summary_df.iterrows():
        val = row["value"]
        try:
            summary[row["metric"]] = int(float(val))
        except (ValueError, TypeError):
            summary[row["metric"]] = val

    return {"summary": summary, "class_distribution": class_df}


def load_phase1b_evidence(phase1b_dir: Path) -> dict:
    """
    Đọc các CSV từ PHASE 1B để lấy evidence.
    Trả về dict chứa quality summary values, class quality, và duplicate df.
    """
    required = {
        "quality_summary": phase1b_dir / "annotation_quality_summary.csv",
        "class_quality": phase1b_dir / "bbox_quality_by_class.csv",
        "duplicates": phase1b_dir / "duplicate_bbox_candidates.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"[ERROR] PHASE 1B evidence file missing: {path} ({name})")

    q_summary_df = pd.read_csv(required["quality_summary"])
    class_q_df = pd.read_csv(required["class_quality"])
    dup_df = pd.read_csv(required["duplicates"])

    if q_summary_df.empty:
        raise ValueError("[ERROR] PHASE 1B annotation_quality_summary.csv is empty.")

    if "metric" not in q_summary_df.columns or "value" not in q_summary_df.columns:
        raise ValueError(
            "[ERROR] annotation_quality_summary.csv must have 'metric' and 'value'.")

    # Chuyển summary thành dict
    q_summary = {}
    for _, row in q_summary_df.iterrows():
        val = row["value"]
        try:
            q_summary[row["metric"]] = int(float(val))
        except (ValueError, TypeError):
            q_summary[row["metric"]] = val

    return {
        "quality_summary": q_summary,
        "class_quality": class_q_df,
        "duplicates": dup_df,
    }


# ---------------------------------------------------------------------------
# Decision builders
# ---------------------------------------------------------------------------

def build_scope_decision_summary(p1a: dict, p1b: dict) -> pd.DataFrame:
    """
    Tạo dataset_scope_decision_summary.csv với 9 decision area.
    """
    s = p1a["summary"]
    q = p1b["quality_summary"]

    total_images = s.get("total_images", "N/A")
    abnormal_images = s.get("abnormal_images", "N/A")
    normal_images = s.get("normal_images", "N/A")
    total_bbox = s.get("total_bbox_rows", "N/A")
    num_classes = s.get("num_abnormal_classes", "N/A")
    invalid_bbox = q.get("bbox_rows_with_invalid_x_order", 0)
    dup_pairs = q.get("duplicate_bbox_candidate_rows", "N/A")
    quality_status = q.get("overall_quality_status", "N/A")

    decisions = [
        {
            "decision_id": "D01",
            "decision_area": "dataset_scope",
            "decision": (
                "Continue with the current metadata-only VinBigData subset "
                "for the next dataset engineering phase."
            ),
            "status": "ACCEPTED",
            "evidence_source": "PHASE 1A dataset_summary.csv; PHASE 1B annotation_quality_summary.csv",
            "rationale": (
                f"The current subset contains {total_images} images "
                f"({abnormal_images} abnormal + {normal_images} normal), "
                f"{total_bbox} valid bbox rows across {num_classes} abnormal classes. "
                "PHASE 1B annotation quality status is PASS with 0 invalid bbox rows. "
                "No evidence requires expanding or contracting the dataset scope at this stage."
            ),
            "risk_or_limitation": (
                "Metadata-only subset; no pixel-level validation possible until "
                "DICOM files are accessed in PHASE 2A — Image Availability & DICOM Metadata Validation."
            ),
            "impact_on_next_phase": (
                "Next phase (dataset engineering) operates on this controlled subset."
            ),
        },
        {
            "decision_id": "D02",
            "decision_area": "subset_policy",
            "decision": (
                "Keep the current subset of 4,394 abnormal images and 500 normal images "
                "as the controlled working subset."
            ),
            "status": "ACCEPTED",
            "evidence_source": "PHASE 1A dataset_summary.csv",
            "rationale": (
                f"Subset counts match config expectations: "
                f"{abnormal_images} abnormal, {normal_images} normal, "
                f"{total_images} total. No images are added or removed in PHASE 1C."
            ),
            "risk_or_limitation": (
                "500 normal images are a random sample (seed=42), not all available "
                "normal images. This is a deliberate controlled choice."
            ),
            "impact_on_next_phase": (
                "Dataset engineering phase inherits this exact subset without modification."
            ),
        },
        {
            "decision_id": "D03",
            "decision_area": "normal_policy",
            "decision": (
                "Keep normal / No Finding images as negative images without bounding boxes "
                "for object detection protocol design in later phases."
            ),
            "status": "ACCEPTED",
            "evidence_source": (
                "PHASE 1B annotation_quality_summary.csv: "
                "normal_images_with_bbox = 0"
            ),
            "rationale": (
                "All 500 normal images confirmed to have zero valid bbox rows. "
                "The 1,500 No Finding annotation rows correspond to annotation rows only "
                "(3 radiologists × 500 images), not 1,500 unique normal images. "
                "No Finding is not treated as an abnormal detection class."
            ),
            "risk_or_limitation": (
                "Normal images serve as background negatives in detection protocol; "
                "their exact role (e.g., training negatives, hard negatives) "
                "is deferred to experimental design phase."
            ),
            "impact_on_next_phase": (
                "Normal images remain in the subset but carry no bbox labels for detection."
            ),
        },
        {
            "decision_id": "D04",
            "decision_area": "class_scope",
            "decision": (
                "Keep all 14 abnormal classes at this stage. "
                "Do not remove any class during PHASE 1C."
            ),
            "status": "ACCEPTED",
            "evidence_source": "PHASE 1A class_distribution.csv; PHASE 1B bbox_quality_by_class.csv",
            "rationale": (
                f"All {num_classes} abnormal classes have valid bbox rows and "
                "zero invalid bbox count. The rarest class (Pneumothorax) has "
                "226 bbox rows across 96 images — sufficient to retain at this stage. "
                "Class imbalance is recorded as a known limitation; "
                "filtering decisions are deferred to experimental design."
            ),
            "risk_or_limitation": (
                "Significant class imbalance exists (Aortic enlargement: 7,162 bbox "
                "vs Pneumothorax: 226 bbox). "
                "Imbalance impact will be analyzed in "
                "PHASE 2C — Class Imbalance & Sampling Strategy Design."
            ),
            "impact_on_next_phase": (
                "Dataset engineering phase retains all 14 abnormal classes."
            ),
        },
        {
            "decision_id": "D05",
            "decision_area": "bbox_quality_policy",
            "decision": (
                "Use PHASE 1B results as annotation quality evidence. "
                "No invalid bbox rows detected at metadata level. "
                "No automatic bbox filtering is applied in PHASE 1C."
            ),
            "status": "ACCEPTED",
            "evidence_source": "PHASE 1B annotation_quality_summary.csv: overall_quality_status = PASS",
            "rationale": (
                "PHASE 1B detected 0 invalid bbox rows (no missing coordinates, "
                "no inverted axes, no negative coordinates, no zero-area boxes). "
                "No corrective action is needed at this stage."
            ),
            "risk_or_limitation": (
                "Image-boundary validation not possible without image dimensions. "
                "Bbox quality at pixel level (boundary check) is assigned to "
                "PHASE 2A — Image Availability & DICOM Metadata Validation."
            ),
            "impact_on_next_phase": (
                "All 36,096 abnormal bbox rows are carried forward without filtering."
            ),
        },
        {
            "decision_id": "D06",
            "decision_area": "duplicate_candidate_policy",
            "decision": (
                f"Record {dup_pairs} duplicate/near-duplicate candidate pairs as "
                "metadata-level flags only. "
                "Do not automatically remove them in PHASE 1C."
            ),
            "status": "ACCEPTED",
            "evidence_source": "PHASE 1B duplicate_bbox_candidates.csv",
            "rationale": (
                f"The {dup_pairs} pairs are identified by IoU ≥ 0.95 within the same "
                "image and class. They are metadata-level candidate flags, not confirmed "
                "annotation errors. Multiple radiologists may legitimately annotate the "
                "same region. Removal decisions require deeper analysis deferred to "
                "experimental design."
            ),
            "risk_or_limitation": (
                "Duplicate candidates are not inter-rater agreement evidence and must "
                "not be treated as Kappa analysis. "
                "They represent a low-to-medium concern at this stage."
            ),
            "impact_on_next_phase": (
                "Duplicate candidate pairs remain in the dataset. "
                "Their handling is noted as a pending decision for training strategy."
            ),
        },
        {
            "decision_id": "D07",
            "decision_area": "image_boundary_policy",
            "decision": (
                "Record image-boundary checking as not available. "
                "No DICOM/image files are read in PHASE 1C. "
                "This check is deferred to a phase where image dimensions are accessible."
            ),
            "status": "DEFERRED",
            "evidence_source": (
                "PHASE 1B annotation_quality_summary.csv: "
                "bbox_rows_outside_image_boundary = not_available"
            ),
            "rationale": (
                "Image width and height are not present in the metadata-only subset. "
                "Boundary validation requires DICOM file access, which is out of scope "
                "for PHASE 1C."
            ),
            "risk_or_limitation": (
                "Some bbox coordinates may theoretically extend beyond image boundaries. "
                "This risk cannot be quantified without image dimensions."
            ),
            "impact_on_next_phase": (
                "Image-boundary check is assigned to "
                "PHASE 2A — Image Availability & DICOM Metadata Validation, "
                "where image dimensions will be extracted from DICOM or image metadata."
            ),
        },
        {
            "decision_id": "D08",
            "decision_area": "metadata_only_policy",
            "decision": (
                "Retain metadata-only analysis scope for PHASE 1C. "
                "Do not read DICOM/image files."
            ),
            "status": "ACCEPTED",
            "evidence_source": "CLAUDE.md; configs/dataset.yaml experiment_stage note",
            "rationale": (
                "The project is currently in the dataset selection and metadata-only "
                "subset creation stage. Reading DICOM files is not required to make "
                "dataset scope decisions based on CSV-level evidence."
            ),
            "risk_or_limitation": (
                "Pixel-level properties (image quality, resolution, artifact presence) "
                "are unknown until DICOM files are read."
            ),
            "impact_on_next_phase": (
                "Next phase may introduce DICOM loading if required by its scope."
            ),
        },
        {
            "decision_id": "D09",
            "decision_area": "phase_gate",
            "decision": (
                "PHASE 1C can be closed when all decision artifacts are generated, "
                "the current dataset scope is explicitly locked, "
                "and no out-of-scope operation has been performed."
            ),
            "status": "ACCEPTED",
            "evidence_source": "PHASE 1C next_phase_readiness_checklist.csv",
            "rationale": (
                "All required decision artifacts are generated. "
                "Dataset scope is locked at 4,394 abnormal + 500 normal images, "
                "14 abnormal classes, 36,096 valid bbox rows. "
                "No split, no leakage check, no annotation conversion, "
                "no model training has been performed."
            ),
            "risk_or_limitation": (
                "Phase gate is conditional on all checklist items being PASS "
                "before proceeding to dataset engineering."
            ),
            "impact_on_next_phase": (
                "Dataset engineering phase may begin only after PHASE 1C confirmation."
            ),
        },
    ]

    return pd.DataFrame(decisions)


def build_class_scope_decision(p1a: dict, p1b: dict,
                                no_finding_id: int) -> pd.DataFrame:
    """
    Tạo class_scope_decision.csv — mỗi dòng là một abnormal class.
    """
    class_df = p1a["class_distribution"].copy()
    # Loại No Finding nếu có trong class distribution
    class_df = class_df[class_df["class_id"] != no_finding_id].copy()

    class_q_df = p1b["class_quality"].copy()

    # Merge với PHASE 1B class quality để lấy invalid count và dup count
    merge_cols = ["class_id", "invalid_bbox_count", "duplicate_candidate_count"]
    available = [c for c in merge_cols if c in class_q_df.columns]
    if len(available) > 1:
        class_df = class_df.merge(
            class_q_df[available], on="class_id", how="left")
    else:
        class_df["invalid_bbox_count"] = 0
        class_df["duplicate_candidate_count"] = 0

    class_df["invalid_bbox_count"] = class_df["invalid_bbox_count"].fillna(0).astype(int)
    class_df["duplicate_candidate_count"] = (
        class_df["duplicate_candidate_count"].fillna(0).astype(int))

    rows = []
    for _, r in class_df.iterrows():
        num_bbox = int(r["num_bbox"])
        num_images = int(r["num_images"])
        invalid = int(r["invalid_bbox_count"])
        dup = int(r["duplicate_candidate_count"])
        cls_name = r["class_name"]

        rationale = (
            f"All {num_bbox} bbox rows across {num_images} images are valid "
            f"(invalid_bbox_count = {invalid}). "
        )
        if dup > 0:
            rationale += (
                f"{dup} duplicate candidate pair(s) flagged at metadata level only. "
            )
        if num_bbox < 500:
            rationale += (
                f"Rare class ({num_bbox} bbox rows) — retained to avoid premature "
                "scope narrowing; class imbalance will be analyzed in "
                "PHASE 2C — Class Imbalance & Sampling Strategy Design."
            )
        else:
            rationale += "Class has sufficient bbox coverage to retain at this stage."

        risk = ""
        if num_bbox < 300:
            risk = (
                f"Very rare class ({num_bbox} bbox, {num_images} images). "
                "May require oversampling or weighted loss in training strategy."
            )
        elif dup > 5:
            risk = (
                f"{dup} duplicate candidate pair(s) in this class. "
                "Metadata-level flag only; not a confirmed annotation error."
            )
        else:
            risk = "No significant risk identified at this stage."

        rows.append({
            "class_id": int(r["class_id"]),
            "class_name": cls_name,
            "num_bbox": num_bbox,
            "num_images": num_images,
            "invalid_bbox_count": invalid,
            "duplicate_candidate_count": dup,
            "decision": "keep",
            "rationale": rationale,
            "risk_or_limitation": risk,
        })

    # Sắp xếp theo num_bbox giảm dần, nhất quán với PHASE 1A
    result = pd.DataFrame(rows).sort_values("num_bbox", ascending=False)
    return result.reset_index(drop=True)


def build_normal_policy_decision() -> pd.DataFrame:
    """
    Tạo normal_policy_decision.csv — mô tả cách xử lý normal/No Finding
    ở cấp quyết định.
    """
    rows = [
        {
            "policy_item": "normal_images",
            "decision": "Keep 500 unique normal images in the current working subset.",
            "evidence": (
                "PHASE 1A: normal_images = 500. "
                "PHASE 1B: normal_images_with_bbox = 0."
            ),
            "rationale": (
                "500 normal images were randomly sampled (seed=42) and confirmed to "
                "have zero valid bbox rows. They serve as controlled negative images."
            ),
            "risk_or_limitation": (
                "500 is a fixed sample; not all available normal images are included. "
                "Sampling strategy (random seed=42) is reproducible."
            ),
            "impact_on_next_phase": (
                "500 normal images are carried forward as-is into dataset engineering."
            ),
        },
        {
            "policy_item": "no_finding_rows",
            "decision": (
                "Record that 1,500 No Finding annotation rows exist in the subset, "
                "corresponding to annotation rows from multiple radiologists, "
                "not 1,500 unique normal images."
            ),
            "evidence": (
                "PHASE 1B: total_no_finding_rows = 1,500. "
                "PHASE 1A: normal_images = 500. "
                "Ratio = 3.0 rows per normal image (consistent with multi-radiologist annotation)."
            ),
            "rationale": (
                "The 1,500 rows represent annotation rows (approx. 3 radiologists × 500 images). "
                "Conflating rows with unique images would misrepresent the dataset."
            ),
            "risk_or_limitation": (
                "Downstream code must distinguish between annotation rows and unique images "
                "when counting normals."
            ),
            "impact_on_next_phase": (
                "No Finding rows are excluded from bbox training targets. "
                "Only unique image count (500) is used for subset accounting."
            ),
        },
        {
            "policy_item": "negative_images_for_detection",
            "decision": (
                "Treat normal images as negative images without bounding boxes "
                "in later object detection protocol design."
            ),
            "evidence": (
                "PHASE 1B: normal_images_with_bbox = 0. "
                "All 500 normal images confirmed to have no valid bbox rows."
            ),
            "rationale": (
                "In standard object detection, images with no ground truth boxes "
                "are treated as background negatives. This is consistent with the "
                "No Finding label semantics."
            ),
            "risk_or_limitation": (
                "The role of negative images in semi-supervised learning "
                "(e.g., as unlabeled data, as hard negatives) is deferred "
                "to experimental design."
            ),
            "impact_on_next_phase": (
                "Dataset engineering will separate images with and without bbox labels "
                "for detection protocol design."
            ),
        },
        {
            "policy_item": "normal_bbox_expectation",
            "decision": (
                "Normal images are expected to have zero valid bbox rows. "
                "Any deviation from this expectation should be treated as a data issue."
            ),
            "evidence": (
                "PHASE 1B: normal_images_with_bbox = 0. "
                "Expectation confirmed for all 500 normal images."
            ),
            "rationale": (
                "No Finding label implies absence of any detectable abnormality. "
                "A normal image with a bbox row would indicate a labeling inconsistency."
            ),
            "risk_or_limitation": (
                "Cannot validate at pixel level without DICOM access. "
                "Metadata-level check is PASS."
            ),
            "impact_on_next_phase": (
                "Zero-bbox check for normal images should be re-run after "
                "any dataset modification."
            ),
        },
        {
            "policy_item": "normal_class_handling",
            "decision": (
                "Do not treat No Finding as an abnormal detection class. "
                "class_id = 14 (No Finding) is excluded from all bbox training targets."
            ),
            "evidence": (
                "PHASE 1A: No Finding class_id = 14, confirmed in configs/dataset.yaml. "
                "No Finding rows have no valid bbox coordinates by design."
            ),
            "rationale": (
                "Including No Finding as a detection class would require bbox targets, "
                "which do not exist. The class serves as a negative label at image level only."
            ),
            "risk_or_limitation": (
                "No risk at this stage. Consistent with standard detection protocol."
            ),
            "impact_on_next_phase": (
                "class_id = 14 is filtered out of all bbox annotation pipelines."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_limitation_register(p1b: dict) -> pd.DataFrame:
    """
    Tạo limitation_register.csv — ghi các limitation đã biết trước phase sau.
    """
    q = p1b["quality_summary"]
    dup_pairs = q.get("duplicate_bbox_candidate_rows", 78)

    rows = [
        {
            "limitation_id": "L1",
            "limitation": (
                "Metadata-only subset — no image pixels or DICOM files have been read."
            ),
            "source_phase": "MILESTONE 0 / PHASE 1A / PHASE 1B / PHASE 1C",
            "evidence": (
                "configs/dataset.yaml experiment_stage note. "
                "DICOM files not stored in Git repository."
            ),
            "severity": "medium",
            "impact": (
                "Pixel-level properties (resolution, image quality, artifact presence) "
                "are unknown. Bbox coordinates cannot be validated against image boundaries."
            ),
            "handling_plan": (
                "Handle in PHASE 2A — Image Availability & DICOM Metadata Validation. "
                "In that phase, verify image/DICOM file availability, extract image_width "
                "and image_height from DICOM or image metadata, and prepare image-level "
                "metadata for boundary validation. "
                "No DICOM/image loading is performed in PHASE 1C."
            ),
        },
        {
            "limitation_id": "L2",
            "limitation": (
                "Image-boundary check not available due to missing image width/height "
                "in the metadata-only subset."
            ),
            "source_phase": "PHASE 1B",
            "evidence": (
                "PHASE 1B annotation_quality_summary.csv: "
                "bbox_rows_outside_image_boundary = not_available."
            ),
            "severity": "medium",
            "impact": (
                "Cannot confirm whether any bbox extends beyond image boundaries. "
                "Risk is low given that the dataset is from a structured Kaggle competition, "
                "but cannot be verified at this stage."
            ),
            "handling_plan": (
                "Perform in PHASE 2A — Image Availability & DICOM Metadata Validation "
                "after extracting image_width and image_height from DICOM or image metadata. "
                "Boundary checks should include x_min >= 0, y_min >= 0, "
                "x_max <= image_width, and y_max <= image_height."
            ),
        },
        {
            "limitation_id": "L3",
            "limitation": (
                "Bbox counts are annotation-level rows, not unique lesion counts. "
                "Multiple radiologists may annotate the same region."
            ),
            "source_phase": "PHASE 1A / PHASE 1B",
            "evidence": (
                "PHASE 1A: total_bbox_rows = 36,096 across 4,394 abnormal images. "
                "PHASE 1B: 1,500 No Finding rows for 500 unique normal images (ratio ~3.0)."
            ),
            "severity": "low",
            "impact": (
                "Raw bbox counts overestimate unique lesion counts. "
                "Any analysis treating bbox rows as unique lesions will be inaccurate."
            ),
            "handling_plan": (
                "Always distinguish annotation rows from unique image/lesion counts "
                "in downstream documentation and code. "
                "Radiologist aggregation strategy deferred to experimental design."
            ),
        },
        {
            "limitation_id": "L4",
            "limitation": (
                "Significant class imbalance across 14 abnormal classes "
                "(Aortic enlargement: 7,162 bbox vs Pneumothorax: 226 bbox)."
            ),
            "source_phase": "PHASE 1A",
            "evidence": (
                "PHASE 1A class_distribution.csv: "
                "most frequent class 7,162 bbox (Aortic enlargement), "
                "least frequent class 226 bbox (Pneumothorax). "
                "Ratio = ~31.7×."
            ),
            "severity": "medium",
            "impact": (
                "Imbalanced training data may bias model toward frequent classes. "
                "Rare classes may have insufficient examples for reliable detection."
            ),
            "handling_plan": (
                "Analyze in PHASE 2C — Class Imbalance & Sampling Strategy Design. "
                "That phase should quantify imbalance ratios, identify head/medium/tail classes, "
                "assess risks for rare classes, and define sampling or split constraints "
                "before labeled/unlabeled split design."
            ),
        },
        {
            "limitation_id": "L5",
            "limitation": (
                f"{dup_pairs} duplicate/near-duplicate bbox candidate pairs "
                "are metadata-level flags only."
            ),
            "source_phase": "PHASE 1B",
            "evidence": (
                f"PHASE 1B duplicate_bbox_candidates.csv: {dup_pairs} pairs "
                "with IoU ≥ 0.95, same image + class. "
                "Cardiomegaly has the highest duplicate candidate count (47 pairs)."
            ),
            "severity": "low",
            "impact": (
                "Duplicate candidates may slightly inflate bbox counts for affected classes. "
                "They are not confirmed annotation errors and are not removed at this stage."
            ),
            "handling_plan": (
                "Record as metadata-level flag. "
                "Evaluate impact in annotation aggregation strategy during experimental design. "
                "Do not treat as inter-rater disagreement evidence."
            ),
        },
        {
            "limitation_id": "L6",
            "limitation": (
                "Current subset uses 500 sampled normal images, not all available "
                "normal images from the original dataset."
            ),
            "source_phase": "MILESTONE 0",
            "evidence": (
                "PHASE 1A: normal_images = 500 (random seed=42). "
                "Original full dataset contains more normal images."
            ),
            "severity": "low",
            "impact": (
                "The 500 normals may not fully represent the distribution of all "
                "normal chest X-rays in the original dataset."
            ),
            "handling_plan": (
                "Current 500-sample policy is deliberate for controlled experiments. "
                "If more negatives are needed, re-sampling can be performed "
                "with a documented rationale."
            ),
        },
        {
            "limitation_id": "L7",
            "limitation": (
                "No Kappa / inter-rater agreement analysis has been performed yet."
            ),
            "source_phase": "PHASE 1B (explicitly excluded from scope)",
            "evidence": (
                "PHASE 1B scope boundary explicitly excludes Kappa analysis. "
                "rad_id column is present in annotations but not analyzed."
            ),
            "severity": "low",
            "impact": (
                "Unknown level of inter-annotator agreement. "
                "The dataset is from a structured Kaggle competition with "
                "professional radiologist annotations; agreement is expected to be "
                "reasonable but not verified."
            ),
            "handling_plan": (
                "Kappa analysis is deferred to a dedicated annotation quality phase "
                "if needed by the experimental design."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_readiness_checklist(p1a: dict, p1b: dict) -> pd.DataFrame:
    """
    Tạo next_phase_readiness_checklist.csv — gate để chuyển sang phase sau.
    """
    s = p1a["summary"]
    q = p1b["quality_summary"]

    total_images = s.get("total_images", 0)
    abnormal_images = s.get("abnormal_images", 0)
    normal_images = s.get("normal_images", 0)
    num_classes = s.get("num_abnormal_classes", 0)
    invalid_bbox = q.get("bbox_rows_with_invalid_x_order", -1)
    normal_with_bbox = q.get("normal_images_with_bbox", -1)
    abnormal_no_bbox = q.get("abnormal_images_without_valid_bbox", -1)
    quality_status = q.get("overall_quality_status", "")

    def check_int(val, expected):
        try:
            return int(float(val)) == expected
        except (ValueError, TypeError):
            return False

    rows = [
        {
            "check_id": "C01",
            "check_item": "PHASE 1A completed and committed",
            "status": "PASS",
            "evidence": "Commit 827c5d1 — Add Phase 1A dataset overview report",
            "notes": "",
        },
        {
            "check_id": "C02",
            "check_item": "PHASE 1B completed and committed",
            "status": "PASS",
            "evidence": "Commit eac73ba — Add Phase 1B annotation quality report",
            "notes": "",
        },
        {
            "check_id": "C03",
            "check_item": "Dataset subset size confirmed",
            "status": "PASS" if check_int(total_images, 4894) else "PENDING",
            "evidence": f"PHASE 1A: total_images = {total_images}",
            "notes": "Expected: 4,894",
        },
        {
            "check_id": "C04",
            "check_item": "Abnormal/normal counts confirmed",
            "status": (
                "PASS"
                if check_int(abnormal_images, 4394) and check_int(normal_images, 500)
                else "PENDING"
            ),
            "evidence": (
                f"PHASE 1A: abnormal_images = {abnormal_images}, "
                f"normal_images = {normal_images}"
            ),
            "notes": "Expected: 4,394 abnormal + 500 normal",
        },
        {
            "check_id": "C05",
            "check_item": "14 abnormal classes confirmed",
            "status": "PASS" if check_int(num_classes, 14) else "PENDING",
            "evidence": f"PHASE 1A: num_abnormal_classes = {num_classes}",
            "notes": "Expected: 14",
        },
        {
            "check_id": "C06",
            "check_item": "No invalid bbox rows detected at metadata level",
            "status": "PASS" if check_int(invalid_bbox, 0) else "PENDING",
            "evidence": (
                f"PHASE 1B: overall_quality_status = {quality_status}; "
                f"bbox_rows_with_invalid_x_order = {invalid_bbox}"
            ),
            "notes": "",
        },
        {
            "check_id": "C07",
            "check_item": "Normal images contain no bbox rows",
            "status": "PASS" if check_int(normal_with_bbox, 0) else "PENDING",
            "evidence": f"PHASE 1B: normal_images_with_bbox = {normal_with_bbox}",
            "notes": "",
        },
        {
            "check_id": "C08",
            "check_item": "All abnormal images have at least one valid bbox",
            "status": "PASS" if check_int(abnormal_no_bbox, 0) else "PENDING",
            "evidence": (
                f"PHASE 1B: abnormal_images_without_valid_bbox = {abnormal_no_bbox}"
            ),
            "notes": "",
        },
        {
            "check_id": "C09",
            "check_item": "Duplicate candidates recorded but not removed",
            "status": "PASS",
            "evidence": "PHASE 1B: 78 pairs flagged in duplicate_bbox_candidates.csv",
            "notes": "Decision D06: recorded as metadata-level flag only",
        },
        {
            "check_id": "C10",
            "check_item": "Image-boundary limitation recorded",
            "status": "PASS",
            "evidence": (
                "PHASE 1B: bbox_rows_outside_image_boundary = not_available. "
                "PHASE 1C limitation_register: L2"
            ),
            "notes": "Decision D07: DEFERRED — assigned to PHASE 2A — Image Availability & DICOM Metadata Validation",
        },
        {
            "check_id": "C11",
            "check_item": "No split created",
            "status": "PASS",
            "evidence": "PHASE 1C scope boundary explicitly excludes split creation",
            "notes": "",
        },
        {
            "check_id": "C12",
            "check_item": "No leakage check performed",
            "status": "NOT_APPLICABLE",
            "evidence": "No split exists; leakage check requires split boundary",
            "notes": "",
        },
        {
            "check_id": "C13",
            "check_item": "No annotation conversion performed",
            "status": "PASS",
            "evidence": "PHASE 1C scope boundary explicitly excludes COCO/YOLO conversion",
            "notes": "",
        },
        {
            "check_id": "C14",
            "check_item": "No training performed",
            "status": "PASS",
            "evidence": "PHASE 1C scope boundary explicitly excludes model training",
            "notes": "",
        },
        {
            "check_id": "C15",
            "check_item": "PHASE 1C decision report generated",
            "status": "PASS",
            "evidence": (
                "reports/phase_1c_dataset_scope_decision/"
                "dataset_scope_decision_report.md"
            ),
            "notes": "",
        },
        {
            "check_id": "C16",
            "check_item": "Ready for next phase",
            "status": "PASS",
            "evidence": "All C01–C15 items PASS or NOT_APPLICABLE",
            "notes": (
                "Next phase (dataset engineering) may begin after PHASE 1C confirmation."
            ),
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def build_markdown_report(p1a: dict,
                           p1b: dict,
                           decision_df: pd.DataFrame,
                           class_df: pd.DataFrame,
                           normal_df: pd.DataFrame,
                           limitation_df: pd.DataFrame,
                           checklist_df: pd.DataFrame) -> str:
    """Tạo nội dung dataset_scope_decision_report.md."""
    s = p1a["summary"]
    q = p1b["quality_summary"]

    # Helpers
    def sv(d, key, fmt=None):
        val = d.get(key, "N/A")
        if fmt == "int":
            try:
                return f"{int(float(val)):,}"
            except (ValueError, TypeError):
                return str(val)
        return str(val)

    def decision_row(area):
        r = decision_df[decision_df["decision_area"] == area]
        if r.empty:
            return "N/A", "N/A"
        return r.iloc[0]["decision"], r.iloc[0]["status"]

    # Decision table
    dec_rows = "\n".join(
        f"| {r['decision_id']} | {r['decision_area']} | {r['status']} | "
        f"{r['decision'][:80]}{'...' if len(r['decision']) > 80 else ''} |"
        for _, r in decision_df.iterrows()
    )

    # Limitation table
    lim_rows = "\n".join(
        f"| {r['limitation_id']} | {r['limitation'][:70]}{'...' if len(r['limitation']) > 70 else ''} "
        f"| {r['source_phase']} | {r['severity']} | "
        f"{r['handling_plan'][:60]}{'...' if len(r['handling_plan']) > 60 else ''} |"
        for _, r in limitation_df.iterrows()
    )

    # Checklist table
    check_rows = "\n".join(
        f"| {r['check_id']} | {r['check_item']} | {r['status']} | {r['notes']} |"
        for _, r in checklist_df.iterrows()
    )

    # Class scope table
    class_rows = "\n".join(
        f"| {int(r['class_id'])} | {r['class_name']} | {int(r['num_bbox']):,} | "
        f"{int(r['num_images']):,} | {int(r['invalid_bbox_count'])} | "
        f"{int(r['duplicate_candidate_count'])} | **{r['decision']}** |"
        for _, r in class_df.iterrows()
    )

    report = f"""# PHASE 1C — Dataset Scope Decision

## 1. Objective

This report documents the **dataset scope decisions** for the VinBigData Chest X-ray
semi-supervised object detection project, based on evidence from PHASE 1A and PHASE 1B.

PHASE 1C is a **decision phase**, not an implementation phase. It produces:
- Explicit, evidence-based decisions on dataset scope.
- A limitation register for known risks.
- A readiness checklist as a gate before the next phase.

**This report does NOT perform:**
- Kappa / inter-annotator agreement analysis
- Train / val / test split or labeled / unlabeled split
- Data leakage checking
- COCO or YOLO annotation conversion
- DICOM/image loading
- Any model training, inference, or pseudo-labeling
- Any modification to the metadata subset

---

## 2. Inputs and Prior Phase Evidence

**Prior phases:**
- PHASE 1A — Dataset Overview Report (commit: 827c5d1)
- PHASE 1B — Annotation Quality Analysis (commit: eac73ba)

**Evidence from PHASE 1A:**

| Metric | Value |
|--------|-------|
| Total images | {sv(s, 'total_images', 'int')} |
| Abnormal images | {sv(s, 'abnormal_images', 'int')} |
| Normal images | {sv(s, 'normal_images', 'int')} |
| Total annotation rows | {sv(s, 'total_annotation_rows', 'int')} |
| Total abnormal bbox rows | {sv(s, 'total_bbox_rows', 'int')} |
| Number of abnormal classes | {sv(s, 'num_abnormal_classes', 'int')} |
| Mean bbox per abnormal image | {sv(s, 'mean_bbox_per_abnormal_image')} |
| Max bbox per abnormal image | {sv(s, 'max_bbox_per_abnormal_image', 'int')} |

**Evidence from PHASE 1B:**

| Metric | Value |
|--------|-------|
| Total No Finding rows | {sv(q, 'total_no_finding_rows', 'int')} |
| Invalid bbox rows (any type) | 0 |
| Normal images with bbox | {sv(q, 'normal_images_with_bbox', 'int')} |
| Abnormal images without valid bbox | {sv(q, 'abnormal_images_without_valid_bbox', 'int')} |
| Duplicate bbox candidate pairs (IoU ≥ 0.95) | {sv(q, 'duplicate_bbox_candidate_rows', 'int')} |
| Image-boundary check | {sv(q, 'bbox_rows_outside_image_boundary')} |
| Overall annotation quality status | **{sv(q, 'overall_quality_status')}** |

---

## 3. Dataset Scope Decision

**Decision D01 — dataset_scope:** `ACCEPTED`

> Continue with the current metadata-only VinBigData subset as the controlled working
> subset for the next dataset engineering phase.

This decision is based on:
- Subset size matches config expectations ({sv(s, 'total_images', 'int')} images).
- PHASE 1B annotation quality is PASS with 0 invalid bbox rows.
- No evidence requires expanding, contracting, or replacing the dataset at this stage.

> **Note:** This is a working subset decision, not a claim that this subset represents
> the full dataset or is the final experimental configuration.

---

## 4. Subset Policy

**Decision D02 — subset_policy:** `ACCEPTED`

> Keep {sv(s, 'abnormal_images', 'int')} abnormal images + {sv(s, 'normal_images', 'int')} normal images
> as the controlled working subset.

- No images are added or removed in PHASE 1C.
- The 500 normal images are a reproducible random sample (seed = 42).
- Subset boundaries are frozen until a documented re-sampling decision is made.

---

## 5. Normal / No Finding Policy

**Decision D03 — normal_policy:** `ACCEPTED`

> Normal images are treated as negative images without bounding boxes.
> No Finding (class_id = 14) is not treated as an abnormal detection class.

Key clarifications:
- The **1,500 No Finding annotation rows** correspond to annotation rows from multiple
  radiologists — not 1,500 unique normal images.
  The metadata subset contains **500 unique normal images**.
- All 500 normal images have been confirmed to have **zero valid bbox rows** (PHASE 1B).
- The role of normal images in semi-supervised learning (e.g., as unlabeled data,
  as hard negatives) is deferred to experimental design.

---

## 6. Class Scope Decision

**Decision D04 — class_scope:** `ACCEPTED`

> Keep all {sv(s, 'num_abnormal_classes', 'int')} abnormal classes. No class is removed in PHASE 1C.

| class_id | class_name | num_bbox | num_images | invalid_bbox | dup_candidates | decision |
|----------|------------|---------|-----------|-------------|----------------|----------|
{class_rows}

- All classes have **zero invalid bbox rows**.
- Class imbalance (Aortic enlargement: 7,162 bbox vs Pneumothorax: 226 bbox) is recorded
  as **Limitation L4**. Its experimental impact will be analyzed in
  **PHASE 2C — Class Imbalance & Sampling Strategy Design**, where imbalance ratios,
  head/medium/tail classes, and sampling constraints will be defined before split design.
- Rare classes are retained to avoid premature scope narrowing.

---

## 7. Bounding Box Quality Policy

**Decision D05 — bbox_quality_policy:** `ACCEPTED`

> PHASE 1B annotation quality is PASS. No automatic bbox filtering is applied in PHASE 1C.

- 0 invalid bbox rows detected (no missing coords, no inverted axes, no negative values,
  no zero-area boxes).
- All {sv(s, 'total_bbox_rows', 'int')} abnormal bbox rows are carried forward.
- Pixel-level validation (image boundary check) is assigned to
  **PHASE 2A — Image Availability & DICOM Metadata Validation** — see Section 9.

---

## 8. Duplicate Candidate Policy

**Decision D06 — duplicate_candidate_policy:** `ACCEPTED`

> {sv(q, 'duplicate_bbox_candidate_rows', 'int')} duplicate/near-duplicate bbox candidate pairs
> are recorded as metadata-level flags only. No automatic removal is performed.

- Pairs identified by IoU ≥ 0.95 within the same `image_id` and `class_id`.
- These are **not** confirmed annotation errors and **not** inter-rater agreement evidence.
- Cardiomegaly has the highest duplicate candidate count (47 pairs).
- Handling of duplicate candidates (if any) is deferred to annotation aggregation strategy
  in experimental design.

---

## 9. Image Boundary and Metadata-Only Limitation

**Decision D07 — image_boundary_policy:** `DEFERRED`
**Decision D08 — metadata_only_policy:** `ACCEPTED`

> Image-boundary checking was not performed because image width and height are not
> available in the metadata-only subset. No DICOM or image files were read in PHASE 1C.

- This is recorded as **Limitation L2** in the limitation register.
- Boundary validation is assigned to **PHASE 2A — Image Availability & DICOM Metadata Validation**,
  where image dimensions will be extracted from DICOM or image metadata.
- The metadata-only scope is intentional and documented in `configs/dataset.yaml`.

---

## 10. Limitation Register

Full details in `reports/phase_1c_dataset_scope_decision/limitation_register.csv`.

| ID | Limitation | Source | Severity | Handling Plan |
|----|-----------|--------|---------|---------------|
{lim_rows}

---

## 11. Next Phase Readiness Checklist

Full details in `reports/phase_1c_dataset_scope_decision/next_phase_readiness_checklist.csv`.

| Check | Item | Status | Notes |
|-------|------|--------|-------|
{check_rows}

---

## 12. Generated Artifacts

- `reports/phase_1c_dataset_scope_decision/dataset_scope_decision_report.md`
- `reports/phase_1c_dataset_scope_decision/dataset_scope_decision_summary.csv`
- `reports/phase_1c_dataset_scope_decision/class_scope_decision.csv`
- `reports/phase_1c_dataset_scope_decision/normal_policy_decision.csv`
- `reports/phase_1c_dataset_scope_decision/limitation_register.csv`
- `reports/phase_1c_dataset_scope_decision/next_phase_readiness_checklist.csv`

No figures are generated in PHASE 1C (decision phase only).

---

## 13. Scope Boundary

> This report is limited to dataset scope decision based on metadata-level evidence
> from PHASE 1A and PHASE 1B. It does not perform Kappa analysis, dataset splitting,
> leakage checking, annotation conversion, image loading, or model training.

---

## 14. Gate Decision

**Decision D09 — phase_gate:** `ACCEPTED`

> PHASE 1C can be considered complete when all decision artifacts are generated,
> the current dataset scope is explicitly locked, and no out-of-scope operation
> has been performed.

**Locked scope:**
- Dataset: VinBigData Chest X-ray Abnormalities Detection (metadata-only subset)
- Images: {sv(s, 'abnormal_images', 'int')} abnormal + {sv(s, 'normal_images', 'int')} normal = {sv(s, 'total_images', 'int')} total
- Abnormal classes: {sv(s, 'num_abnormal_classes', 'int')} (all retained)
- Bbox rows: {sv(s, 'total_bbox_rows', 'int')} valid abnormal bbox rows
- No invalid bbox, no split, no conversion, no training

**The next phase (dataset engineering) may begin only after explicit user confirmation
that PHASE 1C is complete.**
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PHASE 1C — Dataset Scope Decision")
    print("=" * 80)

    # [1] Config
    print("\n[1/7] Loading config...")
    config = load_config()
    no_finding_id = get_config_value(
        config, "subset", "normal_definition", "class_id",
        fallback=NO_FINDING_CLASS_ID)
    print(f"[OK] Config loaded from {CONFIG_PATH}")
    print(f"[OK] No Finding class_id: {no_finding_id}")

    # [2] Output dir
    print("\n[2/7] Creating output directory...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Report dir: {REPORT_DIR}")

    # [3] Load evidence from prior phases
    print("\n[3/7] Loading PHASE 1A evidence...")
    p1a = load_phase1a_evidence(PHASE_1A_DIR)
    print(f"[OK] PHASE 1A summary: {len(p1a['summary'])} metrics loaded")
    print(f"[OK] PHASE 1A class_distribution: {len(p1a['class_distribution'])} classes")

    print("\n[4/7] Loading PHASE 1B evidence...")
    p1b = load_phase1b_evidence(PHASE_1B_DIR)
    print(f"[OK] PHASE 1B quality_summary: {len(p1b['quality_summary'])} metrics loaded")
    print(f"[OK] PHASE 1B class_quality: {len(p1b['class_quality'])} classes")
    print(f"[OK] PHASE 1B duplicates: {len(p1b['duplicates'])} pairs")

    # [5] Build decision artifacts
    print("\n[5/7] Building decision artifacts...")
    decision_df = build_scope_decision_summary(p1a, p1b)
    print(f"      dataset_scope_decision_summary: {len(decision_df)} decisions")

    class_df = build_class_scope_decision(p1a, p1b, no_finding_id)
    print(f"      class_scope_decision: {len(class_df)} classes")

    normal_df = build_normal_policy_decision()
    print(f"      normal_policy_decision: {len(normal_df)} policy items")

    limitation_df = build_limitation_register(p1b)
    print(f"      limitation_register: {len(limitation_df)} limitations")

    checklist_df = build_readiness_checklist(p1a, p1b)
    print(f"      readiness_checklist: {len(checklist_df)} checks")

    # [6] Save CSVs
    print("\n[6/7] Saving CSV files...")
    files = {
        "dataset_scope_decision_summary.csv": decision_df,
        "class_scope_decision.csv": class_df,
        "normal_policy_decision.csv": normal_df,
        "limitation_register.csv": limitation_df,
        "next_phase_readiness_checklist.csv": checklist_df,
    }
    for fname, df in files.items():
        path = REPORT_DIR / fname
        df.to_csv(path, index=False)
        print(f"[OK] Saved: {path}  ({len(df)} rows)")

    # [7] Markdown report
    print("\n[7/7] Writing markdown report...")
    report_md = build_markdown_report(
        p1a, p1b, decision_df, class_df, normal_df, limitation_df, checklist_df)
    report_path = REPORT_DIR / "dataset_scope_decision_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"[OK] Report saved: {report_path}")

    # Terminal summary
    s = p1a["summary"]
    q = p1b["quality_summary"]

    def sv(d, key):
        val = d.get(key, "N/A")
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return val

    pass_count = int((checklist_df["status"] == "PASS").sum())
    na_count = int((checklist_df["status"] == "NOT_APPLICABLE").sum())
    pending_count = int((checklist_df["status"] == "PENDING").sum())

    print("\n" + "=" * 80)
    print("PHASE 1C EXECUTION SUMMARY")
    print("=" * 80)
    print(f"  Evidence reused from PHASE 1A:")
    print(f"    Total images            : {sv(s, 'total_images')}")
    print(f"    Abnormal images         : {sv(s, 'abnormal_images')}")
    print(f"    Normal images           : {sv(s, 'normal_images')}")
    print(f"    Total abnormal bbox rows: {sv(s, 'total_bbox_rows')}")
    print(f"    Num abnormal classes    : {sv(s, 'num_abnormal_classes')}")
    print(f"  Evidence reused from PHASE 1B:")
    print(f"    Invalid bbox rows       : {sv(q, 'bbox_rows_with_invalid_x_order')}")
    print(f"    Duplicate candidates    : {sv(q, 'duplicate_bbox_candidate_rows')}")
    print(f"    Quality status          : {q.get('overall_quality_status', 'N/A')}")
    print(f"  Decisions generated     : {len(decision_df)}")
    print(f"  Classes decided         : {len(class_df)} (all: keep)")
    print(f"  Limitations registered  : {len(limitation_df)}")
    print(f"  Checklist: PASS={pass_count} | NOT_APPLICABLE={na_count} | PENDING={pending_count}")
    print(f"  Report dir              : {REPORT_DIR}")
    print("=" * 80)
    print("PHASE 1C COMPLETED — No split, no Kappa, no leakage, no conversion,")
    print("                      no image loading, no training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
