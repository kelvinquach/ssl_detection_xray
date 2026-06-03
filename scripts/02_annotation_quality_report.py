"""
PHASE 1B — Annotation Quality Analysis
=======================================
Mục tiêu: Kiểm tra chất lượng bounding box annotation trong metadata-only subset
của dataset VinBigData Chest X-ray Abnormalities Detection.

Phase này CHỈ phân tích annotation quality ở mức metadata/bounding box:
- Kiểm tra tọa độ bbox hợp lệ (missing, inverted, negative, zero-area).
- Phân tích phân bố kích thước bbox (width, height, area, aspect ratio).
- Phát hiện duplicate/near-duplicate bbox candidate trong cùng image + class.
- Tóm tắt chất lượng annotation theo class và theo image.

KHÔNG thực hiện trong phase này:
- Không phân tích Kappa / inter-annotator agreement.
- Không tạo train/val/test split.
- Không tạo labeled/unlabeled split.
- Không kiểm tra leakage.
- Không convert sang COCO/YOLO.
- Không training model.
- Không đọc DICOM/image thật (metadata-only).
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("configs/dataset.yaml")
REPORT_DIR = Path("reports/phase_1b_annotation_quality")
FIGURES_DIR = REPORT_DIR / "figures"
PHASE_1A_DIR = Path("reports/phase_1a_dataset_overview")

NO_FINDING_CLASS_ID = 14  # fallback

# Ngưỡng IoU để xác định duplicate candidate
DUPLICATE_IOU_THRESHOLD = 0.95

# Ngưỡng diện tích (pixel²) để cảnh báo bbox "rất nhỏ" — dùng percentile, không hard-code
AREA_GROUP_PERCENTILES = (5, 25, 75, 95)


# ---------------------------------------------------------------------------
# Config helpers (nhất quán với PHASE 1A)
# ---------------------------------------------------------------------------

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Đọc cấu hình từ configs/dataset.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_value(config: dict, *keys, fallback=None, warn: bool = True):
    """
    Truy cập nested key trong config dict.
    Nếu không tìm thấy, trả về fallback và in cảnh báo nếu warn=True.
    """
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
# Data loader
# ---------------------------------------------------------------------------

def load_metadata(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Đọc selected_image_ids.csv và subset_train_annotations.csv.
    Trả về (selected_df, ann_df).
    """
    selected_path = Path(get_config_value(
        config, "metadata_files", "selected_image_ids",
        fallback="data/raw/vinbigdata/metadata_subset/selected_image_ids.csv"))
    ann_path = Path(get_config_value(
        config, "metadata_files", "subset_annotations",
        fallback="data/raw/vinbigdata/metadata_subset/subset_train_annotations.csv"))

    for path, name in [(selected_path, "selected_image_ids"),
                       (ann_path, "subset_annotations")]:
        if not path.exists():
            raise FileNotFoundError(
                f"[ERROR] Required metadata file missing: {path} ({name})")

    selected_df = pd.read_csv(selected_path)
    ann_df = pd.read_csv(ann_path)

    # Kiểm tra cột bắt buộc
    required_sel = {"image_id", "subset_type"}
    required_ann = {"image_id", "class_id", "class_name",
                    "x_min", "y_min", "x_max", "y_max"}
    miss_sel = required_sel - set(selected_df.columns)
    miss_ann = required_ann - set(ann_df.columns)
    if miss_sel:
        raise ValueError(
            f"[ERROR] Missing columns in selected_image_ids.csv: {miss_sel}")
    if miss_ann:
        raise ValueError(
            f"[ERROR] Missing columns in subset_train_annotations.csv: {miss_ann}")
    if selected_df.empty:
        raise ValueError("[ERROR] selected_image_ids.csv is empty.")
    if ann_df.empty:
        raise ValueError("[ERROR] subset_train_annotations.csv is empty.")

    return selected_df, ann_df


# ---------------------------------------------------------------------------
# Bbox computation helpers
# ---------------------------------------------------------------------------

def compute_bbox_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính width, height, area, aspect_ratio cho mỗi bbox row.
    Chỉ áp dụng cho các hàng có đủ 4 tọa độ không null.
    Các hàng thiếu tọa độ sẽ có NaN ở các cột tính toán.
    """
    df = df.copy()
    df["bbox_width"] = df["x_max"] - df["x_min"]
    df["bbox_height"] = df["y_max"] - df["y_min"]
    df["bbox_area"] = df["bbox_width"] * df["bbox_height"]
    # Tránh chia cho 0: nếu height == 0, aspect_ratio = NaN
    df["bbox_aspect_ratio"] = df.apply(
        lambda r: (r["bbox_width"] / r["bbox_height"])
        if (pd.notna(r["bbox_height"]) and r["bbox_height"] != 0)
        else float("nan"),
        axis=1,
    )
    return df


def compute_area_group(series: pd.Series,
                       pcts: tuple = AREA_GROUP_PERCENTILES) -> pd.Series:
    """
    Gán nhóm area theo percentile-based thresholds để tránh hard-code pixel.
    Nhóm: very_small, small, medium, large, very_large.
    Chỉ tính trên giá trị không null.
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series("unknown", index=series.index)
    p5, p25, p75, p95 = np.percentile(valid, list(pcts))

    def label(v):
        if pd.isna(v):
            return "unknown"
        if v <= p5:
            return "very_small"
        if v <= p25:
            return "small"
        if v <= p75:
            return "medium"
        if v <= p95:
            return "large"
        return "very_large"

    return series.apply(label)


def compute_size_group(series: pd.Series, label_prefix: str) -> pd.Series:
    """
    Gán nhóm width hoặc height theo percentile (p5/p25/p75/p95).
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series("unknown", index=series.index)
    p5, p25, p75, p95 = np.percentile(valid, [5, 25, 75, 95])

    def label(v):
        if pd.isna(v):
            return "unknown"
        if v <= p5:
            return f"{label_prefix}_very_small"
        if v <= p25:
            return f"{label_prefix}_small"
        if v <= p75:
            return f"{label_prefix}_medium"
        if v <= p95:
            return f"{label_prefix}_large"
        return f"{label_prefix}_very_large"

    return series.apply(label)


# ---------------------------------------------------------------------------
# Invalid bbox detection
# ---------------------------------------------------------------------------

def detect_invalid_bbox(ann_df: pd.DataFrame,
                        no_finding_id: int) -> pd.DataFrame:
    """
    Phát hiện các dòng bbox không hợp lệ (chỉ trên abnormal rows).
    Trả về DataFrame các dòng lỗi kèm cột 'invalid_reason'.
    """
    # Chỉ xét hàng có class_id != no_finding_id
    abn = ann_df[ann_df["class_id"] != no_finding_id].copy()
    abn = compute_bbox_metrics(abn)

    invalid_rows = []

    for idx, row in abn.iterrows():
        reasons = []
        # 1. Thiếu tọa độ
        for col in ["x_min", "y_min", "x_max", "y_max"]:
            if pd.isna(row[col]):
                reasons.append(f"missing_{col}")
        # Nếu đã thiếu tọa độ thì không check thêm các điều kiện khác
        if reasons:
            invalid_rows.append({**row.to_dict(),
                                  "invalid_reason": "; ".join(reasons)})
            continue
        # 2. x_max <= x_min
        if row["x_max"] <= row["x_min"]:
            reasons.append("x_max_le_x_min")
        # 3. y_max <= y_min
        if row["y_max"] <= row["y_min"]:
            reasons.append("y_max_le_y_min")
        # 4. Non-positive width
        if pd.notna(row["bbox_width"]) and row["bbox_width"] <= 0:
            if "x_max_le_x_min" not in reasons:
                reasons.append("non_positive_width")
        # 5. Non-positive height
        if pd.notna(row["bbox_height"]) and row["bbox_height"] <= 0:
            if "y_max_le_y_min" not in reasons:
                reasons.append("non_positive_height")
        # 6. Zero area
        if pd.notna(row["bbox_area"]) and row["bbox_area"] <= 0:
            if not any(r in reasons for r in
                       ["x_max_le_x_min", "y_max_le_y_min",
                        "non_positive_width", "non_positive_height"]):
                reasons.append("zero_or_negative_area")
        # 7. Negative coordinate
        for col in ["x_min", "y_min", "x_max", "y_max"]:
            if pd.notna(row[col]) and row[col] < 0:
                reasons.append(f"negative_{col}")
        if reasons:
            invalid_rows.append({**row.to_dict(),
                                  "invalid_reason": "; ".join(reasons)})

    cols = ["image_id", "class_id", "class_name",
            "x_min", "y_min", "x_max", "y_max",
            "bbox_width", "bbox_height", "bbox_area", "invalid_reason"]

    if not invalid_rows:
        return pd.DataFrame(columns=cols)

    result = pd.DataFrame(invalid_rows)
    # Chỉ giữ các cột cần thiết (bỏ bbox_aspect_ratio, rad_id, v.v.)
    for c in cols:
        if c not in result.columns:
            result[c] = None
    return result[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Duplicate bbox detection (IoU-based, cùng image + class)
# ---------------------------------------------------------------------------

def compute_iou(box1: tuple, box2: tuple) -> float:
    """
    Tính IoU giữa hai bbox dạng (x_min, y_min, x_max, y_max).
    Trả về 0.0 nếu không overlap.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def detect_duplicate_candidates(bbox_df: pd.DataFrame,
                                 iou_threshold: float = DUPLICATE_IOU_THRESHOLD,
                                 ) -> pd.DataFrame:
    """
    Phát hiện duplicate/near-duplicate bbox candidate:
    - Chỉ so sánh trong cùng image_id và cùng class_id.
    - Đánh dấu pair có IoU >= iou_threshold.
    - KHÔNG phân tích inter-rater agreement, KHÔNG tính Kappa.
    """
    cols = ["image_id", "class_id", "class_name",
            "bbox_index_1", "bbox_index_2", "iou", "duplicate_type"]
    dup_rows = []

    for (img_id, cls_id), grp in bbox_df.groupby(["image_id", "class_id"]):
        grp = grp.reset_index()  # giữ index gốc
        if len(grp) < 2:
            continue
        cls_name = grp.iloc[0]["class_name"]
        for (i, r1), (j, r2) in combinations(grp.iterrows(), 2):
            # Bỏ qua nếu bất kỳ tọa độ nào là NaN
            coords1 = (r1["x_min"], r1["y_min"], r1["x_max"], r1["y_max"])
            coords2 = (r2["x_min"], r2["y_min"], r2["x_max"], r2["y_max"])
            if any(pd.isna(v) for v in coords1 + coords2):
                continue
            # Bỏ qua nếu bbox không hợp lệ (width/height <= 0)
            if (coords1[2] <= coords1[0] or coords1[3] <= coords1[1] or
                    coords2[2] <= coords2[0] or coords2[3] <= coords2[1]):
                continue
            iou_val = compute_iou(coords1, coords2)
            if iou_val >= iou_threshold:
                dup_rows.append({
                    "image_id": img_id,
                    "class_id": int(cls_id),
                    "class_name": cls_name,
                    "bbox_index_1": int(r1["index"]),
                    "bbox_index_2": int(r2["index"]),
                    "iou": round(iou_val, 6),
                    "duplicate_type": "exact_or_near_duplicate_candidate",
                })

    if not dup_rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(dup_rows)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build annotation_quality_summary
# ---------------------------------------------------------------------------

def build_quality_summary(ann_df: pd.DataFrame,
                           selected_df: pd.DataFrame,
                           invalid_df: pd.DataFrame,
                           dup_df: pd.DataFrame,
                           no_finding_id: int) -> pd.DataFrame:
    """
    Tạo annotation_quality_summary.csv với tất cả metric bắt buộc.
    """
    abn_rows = ann_df[ann_df["class_id"] != no_finding_id]
    no_finding_rows = ann_df[ann_df["class_id"] == no_finding_id]

    # Có tọa độ đủ 4 field
    abn_with_coords = abn_rows.dropna(
        subset=["x_min", "y_min", "x_max", "y_max"])
    abn_computed = compute_bbox_metrics(abn_with_coords)

    # Metric counts
    total_annotation_rows = len(ann_df)
    total_abnormal_bbox_rows = len(abn_rows)
    total_no_finding_rows = len(no_finding_rows)
    rows_missing_class_id = int(ann_df["class_id"].isna().sum())
    rows_missing_class_name = int(ann_df["class_name"].isna().sum())
    abn_missing_any_coord = int(abn_rows[
        abn_rows[["x_min", "y_min", "x_max", "y_max"]].isna().any(axis=1)
    ].shape[0])

    # Từ computed bbox
    bbox_invalid_x = int((abn_computed["x_max"] <= abn_computed["x_min"]).sum())
    bbox_invalid_y = int((abn_computed["y_max"] <= abn_computed["y_min"]).sum())
    bbox_nonpos_w = int((abn_computed["bbox_width"] <= 0).sum())
    bbox_nonpos_h = int((abn_computed["bbox_height"] <= 0).sum())
    bbox_zero_area = int((abn_computed["bbox_area"] <= 0).sum())
    bbox_neg_coord = int(
        ((abn_computed["x_min"] < 0) | (abn_computed["y_min"] < 0) |
         (abn_computed["x_max"] < 0) | (abn_computed["y_max"] < 0)).sum()
    )

    # Outside image boundary: không có dimension trong metadata → not_available
    bbox_outside_boundary = "not_available (image dimensions not in metadata)"

    # Extremely small / large: dùng p5 / p95 của area
    valid_area = abn_computed["bbox_area"].dropna()
    if len(valid_area) > 0:
        p5_area = float(np.percentile(valid_area, 5))
        p95_area = float(np.percentile(valid_area, 95))
        bbox_extremely_small = int((abn_computed["bbox_area"] <= p5_area).sum())
        bbox_extremely_large = int((abn_computed["bbox_area"] >= p95_area).sum())
    else:
        bbox_extremely_small = 0
        bbox_extremely_large = 0

    # Duplicate candidates
    dup_candidate_rows = len(dup_df)

    # Normal images with bbox
    normal_ids = set(selected_df[selected_df["subset_type"] == "normal"]["image_id"])
    valid_bbox_image_ids = set(
        abn_computed[
            (abn_computed["bbox_width"] > 0) &
            (abn_computed["bbox_height"] > 0)
        ]["image_id"]
    )
    normal_with_bbox = int(len(normal_ids & valid_bbox_image_ids))

    # Abnormal images without valid bbox
    abnormal_ids = set(
        selected_df[selected_df["subset_type"] == "abnormal"]["image_id"])
    abnormal_without_valid_bbox = int(
        len(abnormal_ids - valid_bbox_image_ids))

    # Overall quality status
    total_issues = (
        abn_missing_any_coord + bbox_invalid_x + bbox_invalid_y +
        bbox_nonpos_w + bbox_nonpos_h + bbox_zero_area +
        bbox_neg_coord + normal_with_bbox + abnormal_without_valid_bbox
    )
    quality_status = "PASS" if total_issues == 0 else f"WARNING ({total_issues} issue(s) detected)"

    rows = [
        ("total_annotation_rows", total_annotation_rows),
        ("total_abnormal_bbox_rows", total_abnormal_bbox_rows),
        ("total_no_finding_rows", total_no_finding_rows),
        ("rows_missing_class_id", rows_missing_class_id),
        ("rows_missing_class_name", rows_missing_class_name),
        ("abnormal_rows_missing_any_bbox_coordinate", abn_missing_any_coord),
        ("bbox_rows_with_invalid_x_order", bbox_invalid_x),
        ("bbox_rows_with_invalid_y_order", bbox_invalid_y),
        ("bbox_rows_with_non_positive_width", bbox_nonpos_w),
        ("bbox_rows_with_non_positive_height", bbox_nonpos_h),
        ("bbox_rows_with_zero_area", bbox_zero_area),
        ("bbox_rows_with_negative_coordinate", bbox_neg_coord),
        ("bbox_rows_outside_image_boundary", bbox_outside_boundary),
        ("bbox_rows_with_extremely_small_area", bbox_extremely_small),
        ("bbox_rows_with_extremely_large_area", bbox_extremely_large),
        ("duplicate_bbox_candidate_rows", dup_candidate_rows),
        ("normal_images_with_bbox", normal_with_bbox),
        ("abnormal_images_without_valid_bbox", abnormal_without_valid_bbox),
        ("overall_quality_status", quality_status),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


# ---------------------------------------------------------------------------
# Build bbox_size_distribution
# ---------------------------------------------------------------------------

def build_bbox_size_distribution(ann_df: pd.DataFrame,
                                  no_finding_id: int) -> pd.DataFrame:
    """
    Mỗi dòng là một bbox hợp lệ (class_id != no_finding_id, có đủ tọa độ,
    width > 0, height > 0).
    Thêm cột area_group, width_group, height_group theo percentile.
    """
    valid = ann_df[
        (ann_df["class_id"] != no_finding_id) &
        ann_df[["x_min", "y_min", "x_max", "y_max"]].notna().all(axis=1)
    ].copy()

    valid = compute_bbox_metrics(valid)

    # Chỉ giữ bbox hợp lệ (width > 0, height > 0)
    valid = valid[
        (valid["bbox_width"] > 0) & (valid["bbox_height"] > 0)
    ].copy()

    valid["bbox_area_group"] = compute_area_group(valid["bbox_area"])
    valid["bbox_width_group"] = compute_size_group(valid["bbox_width"], "width")
    valid["bbox_height_group"] = compute_size_group(valid["bbox_height"], "height")

    cols = ["image_id", "class_id", "class_name",
            "x_min", "y_min", "x_max", "y_max",
            "bbox_width", "bbox_height", "bbox_area", "bbox_aspect_ratio",
            "bbox_area_group", "bbox_width_group", "bbox_height_group"]
    for c in cols:
        if c not in valid.columns:
            valid[c] = None
    return valid[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build bbox_quality_by_class
# ---------------------------------------------------------------------------

def build_quality_by_class(ann_df: pd.DataFrame,
                            size_df: pd.DataFrame,
                            invalid_df: pd.DataFrame,
                            dup_df: pd.DataFrame,
                            no_finding_id: int) -> pd.DataFrame:
    """
    Tổng hợp chất lượng bbox theo class (chỉ abnormal class).
    """
    abn = ann_df[ann_df["class_id"] != no_finding_id].copy()

    # Tổng số bbox per class (kể cả invalid)
    total_per_class = (
        abn.groupby(["class_id", "class_name"])
        .size()
        .reset_index(name="num_bbox")
    )

    # Invalid count per class
    if not invalid_df.empty:
        inv_count = (
            invalid_df.groupby("class_id")
            .size()
            .reset_index(name="invalid_bbox_count")
        )
    else:
        inv_count = pd.DataFrame(columns=["class_id", "invalid_bbox_count"])

    # Statistics from size_df (valid bbox only)
    if not size_df.empty:
        stats = (
            size_df.groupby(["class_id", "class_name"])
            .agg(
                mean_bbox_width=("bbox_width", "mean"),
                median_bbox_width=("bbox_width", "median"),
                mean_bbox_height=("bbox_height", "mean"),
                median_bbox_height=("bbox_height", "median"),
                mean_bbox_area=("bbox_area", "mean"),
                median_bbox_area=("bbox_area", "median"),
                mean_aspect_ratio=("bbox_aspect_ratio", "mean"),
                median_aspect_ratio=("bbox_aspect_ratio", "median"),
                very_small_bbox_count=(
                    "bbox_area_group",
                    lambda x: (x == "very_small").sum(),
                ),
                very_large_bbox_count=(
                    "bbox_area_group",
                    lambda x: (x == "very_large").sum(),
                ),
            )
            .reset_index()
        )
    else:
        stats = pd.DataFrame(columns=[
            "class_id", "class_name",
            "mean_bbox_width", "median_bbox_width",
            "mean_bbox_height", "median_bbox_height",
            "mean_bbox_area", "median_bbox_area",
            "mean_aspect_ratio", "median_aspect_ratio",
            "very_small_bbox_count", "very_large_bbox_count",
        ])

    # Duplicate count per class
    if not dup_df.empty:
        dup_count = (
            dup_df.groupby("class_id")
            .size()
            .reset_index(name="duplicate_candidate_count")
        )
    else:
        dup_count = pd.DataFrame(columns=["class_id", "duplicate_candidate_count"])

    # Merge
    result = total_per_class.copy()
    result = result.merge(inv_count, on="class_id", how="left")
    result = result.merge(stats.drop(columns=["class_name"], errors="ignore"),
                          on="class_id", how="left")
    result = result.merge(dup_count, on="class_id", how="left")

    result["invalid_bbox_count"] = result["invalid_bbox_count"].fillna(0).astype(int)
    result["invalid_bbox_percentage"] = (
        result["invalid_bbox_count"] / result["num_bbox"] * 100
    ).round(4)
    result["duplicate_candidate_count"] = (
        result["duplicate_candidate_count"].fillna(0).astype(int)
    )

    # Round float columns
    float_cols = [
        "mean_bbox_width", "median_bbox_width",
        "mean_bbox_height", "median_bbox_height",
        "mean_bbox_area", "median_bbox_area",
        "mean_aspect_ratio", "median_aspect_ratio",
    ]
    for c in float_cols:
        if c in result.columns:
            result[c] = result[c].round(4)

    result = result.sort_values("num_bbox", ascending=False).reset_index(drop=True)

    out_cols = [
        "class_id", "class_name", "num_bbox",
        "invalid_bbox_count", "invalid_bbox_percentage",
        "mean_bbox_width", "median_bbox_width",
        "mean_bbox_height", "median_bbox_height",
        "mean_bbox_area", "median_bbox_area",
        "mean_aspect_ratio", "median_aspect_ratio",
        "very_small_bbox_count", "very_large_bbox_count",
        "duplicate_candidate_count",
    ]
    for c in out_cols:
        if c not in result.columns:
            result[c] = None
    return result[out_cols]


# ---------------------------------------------------------------------------
# Build bbox_quality_by_image
# ---------------------------------------------------------------------------

def build_quality_by_image(selected_df: pd.DataFrame,
                            ann_df: pd.DataFrame,
                            size_df: pd.DataFrame,
                            invalid_df: pd.DataFrame,
                            dup_df: pd.DataFrame,
                            no_finding_id: int) -> pd.DataFrame:
    """
    Tổng hợp chất lượng bbox theo image.
    """
    result = selected_df[["image_id", "subset_type"]].rename(
        columns={"subset_type": "image_type"})

    # Tổng số bbox row per image (không tính No Finding)
    abn_rows = ann_df[ann_df["class_id"] != no_finding_id]
    total_bbox_per_img = (
        abn_rows.groupby("image_id").size().rename("num_bbox")
    )

    # Invalid bbox per image — reset_index để merge được theo cột
    if not invalid_df.empty:
        inv_per_img = (
            invalid_df.groupby("image_id").size()
            .rename("num_invalid_bbox").reset_index()
        )
    else:
        inv_per_img = pd.DataFrame(columns=["image_id", "num_invalid_bbox"])

    # Unique classes per image (valid bbox only)
    if not size_df.empty:
        cls_per_img = (
            size_df.groupby("image_id")["class_id"]
            .nunique()
            .rename("num_unique_classes").reset_index()
        )
        area_stats = (
            size_df.groupby("image_id")["bbox_area"]
            .agg(mean_bbox_area="mean",
                 median_bbox_area="median",
                 min_bbox_area="min",
                 max_bbox_area="max")
            .reset_index()
        )
    else:
        cls_per_img = pd.DataFrame(columns=["image_id", "num_unique_classes"])
        area_stats = pd.DataFrame(columns=[
            "image_id", "mean_bbox_area", "median_bbox_area",
            "min_bbox_area", "max_bbox_area"])

    # Duplicate candidates per image
    if not dup_df.empty:
        dup_per_img = (
            dup_df.groupby("image_id").size()
            .rename("dup_count").reset_index()
        )
    else:
        dup_per_img = pd.DataFrame(columns=["image_id", "dup_count"])

    result = result.merge(total_bbox_per_img.reset_index(), on="image_id", how="left")
    result = result.merge(inv_per_img, on="image_id", how="left")
    result = result.merge(cls_per_img, on="image_id", how="left")
    result = result.merge(area_stats, on="image_id", how="left")
    result = result.merge(dup_per_img, on="image_id", how="left")

    result["num_bbox"] = result["num_bbox"].fillna(0).astype(int)
    result["num_invalid_bbox"] = result["num_invalid_bbox"].fillna(0).astype(int)
    result["num_valid_bbox"] = result["num_bbox"] - result["num_invalid_bbox"]
    result["num_unique_classes"] = result["num_unique_classes"].fillna(0).astype(int)
    result["has_invalid_bbox"] = result["num_invalid_bbox"] > 0
    result["has_duplicate_candidate"] = result["dup_count"].fillna(0) > 0

    for c in ["mean_bbox_area", "median_bbox_area", "min_bbox_area", "max_bbox_area"]:
        if c in result.columns:
            result[c] = result[c].round(2)

    result = result.drop(columns=["dup_count"], errors="ignore")

    out_cols = [
        "image_id", "image_type", "num_bbox",
        "num_invalid_bbox", "num_valid_bbox", "num_unique_classes",
        "has_invalid_bbox", "has_duplicate_candidate",
        "mean_bbox_area", "median_bbox_area", "min_bbox_area", "max_bbox_area",
    ]
    for c in out_cols:
        if c not in result.columns:
            result[c] = None
    return result[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def create_figures(size_df: pd.DataFrame) -> list[str]:
    """
    Tạo 4 histogram trong figures/:
    bbox_width, bbox_height, bbox_area, bbox_aspect_ratio.
    Trả về list đường dẫn đã tạo thành công.
    """
    created = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[FIGURE ERROR] matplotlib is not installed. Cannot create figures.")
        return []

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if size_df.empty:
        print("[FIGURE WARNING] size_df is empty — no figures will be created.")
        return []

    specs = [
        ("bbox_width", "bbox_width_distribution.png",
         "Bounding Box Width Distribution", "Width (pixels)", False),
        ("bbox_height", "bbox_height_distribution.png",
         "Bounding Box Height Distribution", "Height (pixels)", False),
        ("bbox_area", "bbox_area_distribution.png",
         "Bounding Box Area Distribution (log scale)", "Area (pixels²)", True),
        ("bbox_aspect_ratio", "bbox_aspect_ratio_distribution.png",
         "Bounding Box Aspect Ratio Distribution (width / height)",
         "Aspect Ratio", False),
    ]

    for col, fname, title, xlabel, use_log in specs:
        try:
            data = size_df[col].dropna()
            if data.empty:
                print(f"[FIGURE WARNING] No valid data for {col}. Skipping {fname}.")
                continue
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(data, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
            if use_log:
                ax.set_yscale("log")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count (log scale)" if use_log else "Count")
            plt.tight_layout()
            out = FIGURES_DIR / fname
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"[OK] Figure saved: {out}")
            created.append(str(out))
        except Exception as e:
            print(f"[FIGURE ERROR] {fname} — {e}")

    return created


# ---------------------------------------------------------------------------
# Sanity observations
# ---------------------------------------------------------------------------

def compute_sanity_observations(summary_df: pd.DataFrame) -> list[str]:
    """
    Đọc từ summary_df và tạo danh sách observation dạng text.
    """
    def sv(metric):
        row = summary_df[summary_df["metric"] == metric]
        return row.iloc[0]["value"] if not row.empty else "N/A"

    obs = []

    def check(metric, good_val, good_msg, bad_msg_tpl):
        val = sv(metric)
        try:
            v = int(float(val))
        except (ValueError, TypeError):
            obs.append(f"[INFO] {metric}: {val}")
            return
        if v == good_val:
            obs.append(f"[OK] {good_msg}")
        else:
            obs.append(f"[WARNING] {bad_msg_tpl.format(v)}")

    check("rows_missing_class_id", 0,
          "No annotation rows with missing class_id.",
          "Missing class_id in {} annotation row(s).")
    check("rows_missing_class_name", 0,
          "No annotation rows with missing class_name.",
          "Missing class_name in {} annotation row(s).")
    check("abnormal_rows_missing_any_bbox_coordinate", 0,
          "No abnormal annotation rows have missing bbox coordinates.",
          "{} abnormal annotation row(s) have missing bbox coordinates.")
    check("bbox_rows_with_invalid_x_order", 0,
          "No bbox rows with x_max <= x_min.",
          "{} bbox row(s) have x_max <= x_min.")
    check("bbox_rows_with_invalid_y_order", 0,
          "No bbox rows with y_max <= y_min.",
          "{} bbox row(s) have y_max <= y_min.")
    check("bbox_rows_with_negative_coordinate", 0,
          "No bbox rows with negative coordinates.",
          "{} bbox row(s) have negative coordinates.")
    check("bbox_rows_with_zero_area", 0,
          "No zero-area bbox rows found.",
          "{} zero-area bbox row(s) found.")
    check("normal_images_with_bbox", 0,
          "No normal images have actual bbox rows — consistent with No Finding label.",
          "{} normal image(s) unexpectedly have valid bbox rows.")
    check("abnormal_images_without_valid_bbox", 0,
          "All abnormal images have at least one valid bbox row.",
          "{} abnormal image(s) have no valid bbox rows.")

    dup_val = sv("duplicate_bbox_candidate_rows")
    try:
        dup_int = int(float(dup_val))
    except (ValueError, TypeError):
        dup_int = -1
    if dup_int == 0:
        obs.append("[OK] No duplicate bbox candidates found (IoU >= 0.95 threshold).")
    elif dup_int > 0:
        obs.append(
            f"[INFO] {dup_int} duplicate bbox candidate pair(s) found (IoU >= 0.95). "
            "These are annotation-level duplicates, NOT inter-rater disagreement analysis."
        )

    obs.append(
        "[INFO] bbox_rows_outside_image_boundary: not_available — "
        "image dimensions are not present in the current metadata subset."
    )
    return obs


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def build_markdown_report(summary_df: pd.DataFrame,
                           quality_by_class_df: pd.DataFrame,
                           invalid_df: pd.DataFrame,
                           size_df: pd.DataFrame,
                           dup_df: pd.DataFrame,
                           sanity_obs: list[str],
                           figures_created: list[str],
                           config: dict) -> str:
    no_finding_id = get_config_value(
        config, "subset", "normal_definition", "class_id",
        fallback=NO_FINDING_CLASS_ID)

    INT_METRICS = {
        "total_annotation_rows", "total_abnormal_bbox_rows",
        "total_no_finding_rows", "rows_missing_class_id", "rows_missing_class_name",
        "abnormal_rows_missing_any_bbox_coordinate",
        "bbox_rows_with_invalid_x_order", "bbox_rows_with_invalid_y_order",
        "bbox_rows_with_non_positive_width", "bbox_rows_with_non_positive_height",
        "bbox_rows_with_zero_area", "bbox_rows_with_negative_coordinate",
        "bbox_rows_with_extremely_small_area", "bbox_rows_with_extremely_large_area",
        "duplicate_bbox_candidate_rows",
        "normal_images_with_bbox", "abnormal_images_without_valid_bbox",
    }

    def sv(metric):
        row = summary_df[summary_df["metric"] == metric]
        if row.empty:
            return "N/A"
        val = row.iloc[0]["value"]
        if metric in INT_METRICS:
            try:
                return int(float(val))
            except (ValueError, TypeError):
                pass
        return val

    # --- Class table ---
    class_table_rows = []
    for _, r in quality_by_class_df.iterrows():
        class_table_rows.append(
            f"| {int(r['class_id'])} | {r['class_name']} | {int(r['num_bbox'])} | "
            f"{int(r['invalid_bbox_count'])} | {r['invalid_bbox_percentage']:.2f}% | "
            f"{r['mean_bbox_area']:.1f} | {r['median_bbox_area']:.1f} | "
            f"{r['mean_aspect_ratio']:.4f} | "
            f"{int(r['very_small_bbox_count'])} | {int(r['very_large_bbox_count'])} | "
            f"{int(r['duplicate_candidate_count'])} |"
        )
    class_table = "\n".join(class_table_rows)

    # --- Size stats ---
    if not size_df.empty:
        w_mean = size_df["bbox_width"].mean()
        w_med = size_df["bbox_width"].median()
        h_mean = size_df["bbox_height"].mean()
        h_med = size_df["bbox_height"].median()
        a_mean = size_df["bbox_area"].mean()
        a_med = size_df["bbox_area"].median()
        ar_mean = size_df["bbox_aspect_ratio"].mean()
        ar_med = size_df["bbox_aspect_ratio"].median()

        p5_w, p95_w = np.percentile(size_df["bbox_width"].dropna(), [5, 95])
        p5_h, p95_h = np.percentile(size_df["bbox_height"].dropna(), [5, 95])
        p5_a, p95_a = np.percentile(size_df["bbox_area"].dropna(), [5, 95])
        p5_ar, p95_ar = np.percentile(
            size_df["bbox_aspect_ratio"].dropna(), [5, 95])

        area_group_counts = size_df["bbox_area_group"].value_counts().to_dict()

        size_section = f"""
| Statistic | Width (px) | Height (px) | Area (px²) | Aspect Ratio |
|-----------|-----------|------------|-----------|--------------|
| Mean | {w_mean:.2f} | {h_mean:.2f} | {a_mean:.2f} | {ar_mean:.4f} |
| Median | {w_med:.2f} | {h_med:.2f} | {a_med:.2f} | {ar_med:.4f} |
| p5 | {p5_w:.2f} | {p5_h:.2f} | {p5_a:.2f} | {p5_ar:.4f} |
| p95 | {p95_w:.2f} | {p95_h:.2f} | {p95_a:.2f} | {p95_ar:.4f} |

**Area group distribution** (percentile-based: very_small ≤ p5 < small ≤ p25 < medium ≤ p75 < large ≤ p95 < very_large):

| Group | Count |
|-------|-------|
| very_small | {area_group_counts.get('very_small', 0)} |
| small | {area_group_counts.get('small', 0)} |
| medium | {area_group_counts.get('medium', 0)} |
| large | {area_group_counts.get('large', 0)} |
| very_large | {area_group_counts.get('very_large', 0)} |
"""
    else:
        size_section = "\n_No valid bbox data available for size distribution._\n"

    # --- Invalid section ---
    if invalid_df.empty:
        invalid_section = "No invalid bbox rows were detected in this metadata subset."
    else:
        inv_reasons = (
            invalid_df["invalid_reason"]
            .str.split("; ")
            .explode()
            .value_counts()
        )
        inv_lines = "\n".join(
            f"- `{r}`: {c} row(s)"
            for r, c in inv_reasons.items()
        )
        invalid_section = (
            f"{len(invalid_df)} invalid bbox row(s) detected.\n\n"
            f"Breakdown by reason:\n\n{inv_lines}"
        )

    # --- Duplicate section ---
    if dup_df.empty:
        dup_section = (
            f"No duplicate bbox candidates found at IoU ≥ {DUPLICATE_IOU_THRESHOLD} "
            "within the same image and class."
        )
    else:
        top_dup = dup_df.nlargest(5, "iou")[
            ["image_id", "class_name", "iou"]].to_string(index=False)
        dup_section = (
            f"{len(dup_df)} duplicate bbox candidate pair(s) detected "
            f"(IoU ≥ {DUPLICATE_IOU_THRESHOLD}, same image + class).\n\n"
            f"Top 5 highest IoU pairs:\n```\n{top_dup}\n```"
        )

    # --- Figures ---
    if figures_created:
        fig_lines = "\n".join(f"- `{f}`" for f in figures_created)
    else:
        fig_lines = ("- No figures were created "
                     "(matplotlib not available or data was empty).")

    # --- Sanity list ---
    sanity_lines = "\n".join(f"- {obs}" for obs in sanity_obs)

    report = f"""# PHASE 1B — Annotation Quality Analysis

## 1. Objective

This report analyzes the **quality of bounding box annotations** in the
VinBigData Chest X-ray metadata-only subset at the metadata/CSV level.

**Scope of this report:**
- Validate bbox coordinate completeness and geometric validity.
- Analyze bbox size distribution (width, height, area, aspect ratio).
- Detect potential duplicate / near-duplicate bbox pairs within the same image and class.
- Summarize annotation quality per class and per image.

**This report does NOT perform:**
- Kappa / inter-annotator agreement analysis
- Train / val / test split or labeled / unlabeled split
- Data leakage checking
- COCO or YOLO annotation conversion
- DICOM/image reading (metadata-only)
- Any model training, inference, or pseudo-labeling

---

## 2. Inputs and Scope

**Input files:**
- `configs/dataset.yaml`
- `data/raw/vinbigdata/metadata_subset/selected_image_ids.csv`
- `data/raw/vinbigdata/metadata_subset/subset_train_annotations.csv`
- `reports/phase_1a_dataset_overview/` (PHASE 1A — completed and committed at 827c5d1)

**No Finding class_id:** `{no_finding_id}` (excluded from bbox quality checks)

**Duplicate detection threshold:** IoU ≥ `{DUPLICATE_IOU_THRESHOLD}` (same image + class only)

**Image boundary check:** Image boundary checking was not performed because image
width and height are not available in the metadata-only subset. No DICOM or image
files were read in this phase.

---

## 3. Annotation Quality Summary

| Metric | Value |
|--------|-------|
| Total annotation rows | {sv('total_annotation_rows')} |
| Total abnormal bbox rows | {sv('total_abnormal_bbox_rows')} |
| Total No Finding rows | {sv('total_no_finding_rows')} |
| Rows missing class_id | {sv('rows_missing_class_id')} |
| Rows missing class_name | {sv('rows_missing_class_name')} |
| Abnormal rows missing any bbox coord | {sv('abnormal_rows_missing_any_bbox_coordinate')} |
| Bbox rows with invalid x-order (x_max ≤ x_min) | {sv('bbox_rows_with_invalid_x_order')} |
| Bbox rows with invalid y-order (y_max ≤ y_min) | {sv('bbox_rows_with_invalid_y_order')} |
| Bbox rows with non-positive width | {sv('bbox_rows_with_non_positive_width')} |
| Bbox rows with non-positive height | {sv('bbox_rows_with_non_positive_height')} |
| Bbox rows with zero/negative area | {sv('bbox_rows_with_zero_area')} |
| Bbox rows with negative coordinates | {sv('bbox_rows_with_negative_coordinate')} |
| Bbox rows outside image boundary | {sv('bbox_rows_outside_image_boundary')} |
| Bbox rows with extremely small area (≤ p5) | {sv('bbox_rows_with_extremely_small_area')} |
| Bbox rows with extremely large area (≥ p95) | {sv('bbox_rows_with_extremely_large_area')} |
| Duplicate bbox candidate pairs | {sv('duplicate_bbox_candidate_rows')} |
| Normal images with valid bbox | {sv('normal_images_with_bbox')} |
| Abnormal images without valid bbox | {sv('abnormal_images_without_valid_bbox')} |
| **Overall quality status** | **{sv('overall_quality_status')}** |

> **Note on No Finding rows:** The 1,500 No Finding rows correspond to annotation rows,
> not unique normal images. The metadata subset contains 500 unique normal images.
> Each normal image has multiple annotation rows (one per radiologist) all carrying
> class_id = {no_finding_id} with no bbox coordinates.

> **Note on extremely small/large area:** "extremely small" = area ≤ 5th percentile;
> "extremely large" = area ≥ 95th percentile.
> These thresholds are computed from the distribution of valid bbox areas in this subset —
> they are **not** fixed pixel thresholds.
> Small and large bbox groups are percentile-based relative size flags and should not be
> interpreted as confirmed annotation errors.

---

## 4. Invalid Bounding Box Analysis

{invalid_section}

> Invalid bbox rows are saved to `reports/phase_1b_annotation_quality/invalid_bbox_rows.csv`.
> If the file contains only a header row, no invalid bboxes were found.

---

## 5. Bounding Box Size Distribution

The following statistics are computed on **valid bbox rows only**
(class_id ≠ {no_finding_id}, coordinates present, width > 0, height > 0).

{size_section}

> **Scope note:** Each row reflects one annotation-level bounding-box entry.
> Multiple radiologists may annotate the same lesion, so these counts should not be
> interpreted directly as the number of unique clinical lesions per image.

> **Area grouping logic:** Percentile-based (p5/p25/p75/p95 of valid bbox areas in this subset).
> No fixed pixel thresholds were used.
> Small and large bbox groups are percentile-based relative size flags and should not be
> interpreted as confirmed annotation errors.

---

## 6. Class-Level Annotation Quality

| class_id | class_name | num_bbox | invalid | invalid_pct | mean_area | median_area | mean_AR | very_small | very_large | dup_cand |
|----------|------------|----------|---------|-------------|-----------|-------------|---------|------------|------------|---------|
{class_table}

> AR = aspect ratio (width / height). very_small / very_large = bbox in bottom/top 5th
> percentile of area distribution. dup_cand = duplicate candidate pairs for that class.

---

## 7. Image-Level Annotation Quality

Image-level stats are saved in
`reports/phase_1b_annotation_quality/bbox_quality_by_image.csv`.

Key highlights:
- Images with **has_invalid_bbox = True**: {int(invalid_df['image_id'].nunique()) if not invalid_df.empty else 0}
- Images with **has_duplicate_candidate = True**: {int(dup_df['image_id'].nunique()) if not dup_df.empty else 0}
- Normal images: all have `num_bbox = 0` and `num_valid_bbox = 0` by design.

---

## 8. Duplicate Bounding Box Candidate Analysis

{dup_section}

> **Important:** These pairs are metadata-level duplicate or near-duplicate candidates
> based on IoU ≥ {DUPLICATE_IOU_THRESHOLD} within the same image and class.
> They should not be interpreted as confirmed annotation errors or inter-rater
> agreement results. Duplicate candidates are identified purely by geometric overlap
> within the same `image_id` and `class_id`.
> This is **NOT** Kappa analysis and **NOT** inter-rater agreement analysis.
> Multiple annotators may legitimately annotate the same region — this check only
> flags pairs with near-identical coordinates for awareness.

---

## 9. Basic Sanity Observations

{sanity_lines}

---

## 10. Generated Artifacts

**CSV files:**
- `reports/phase_1b_annotation_quality/annotation_quality_summary.csv`
- `reports/phase_1b_annotation_quality/invalid_bbox_rows.csv`
- `reports/phase_1b_annotation_quality/bbox_size_distribution.csv`
- `reports/phase_1b_annotation_quality/bbox_quality_by_class.csv`
- `reports/phase_1b_annotation_quality/bbox_quality_by_image.csv`
- `reports/phase_1b_annotation_quality/duplicate_bbox_candidates.csv`

**Figures:**
{fig_lines}

**Report:**
- `reports/phase_1b_annotation_quality/annotation_quality_report.md`

---

## 11. Scope Boundary

> This report is limited to annotation quality analysis at the metadata/bounding-box level.
> It does not perform Kappa analysis, dataset splitting, leakage checking,
> annotation conversion, or model training.

---

## 12. Next Step Recommendation

After the user confirms that PHASE 1B is complete and the annotation quality observations
are acceptable, the next phase can be designed separately.

**No further action is taken automatically.**
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PHASE 1B — Annotation Quality Analysis")
    print("=" * 80)

    # [1] Config
    print("\n[1/9] Loading config...")
    config = load_config()
    no_finding_id = get_config_value(
        config, "subset", "normal_definition", "class_id",
        fallback=NO_FINDING_CLASS_ID)
    print(f"[OK] Config loaded from {CONFIG_PATH}")
    print(f"[OK] No Finding class_id: {no_finding_id}")

    # [2] Output dirs
    print("\n[2/9] Creating output directories...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Report dir : {REPORT_DIR}")
    print(f"[OK] Figures dir: {FIGURES_DIR}")

    # [3] Load metadata
    print("\n[3/9] Loading metadata files...")
    selected_df, ann_df = load_metadata(config)
    print(f"[OK] selected_image_ids      : {selected_df.shape}")
    print(f"[OK] subset_train_annotations: {ann_df.shape}")

    # [4] Invalid bbox
    print("\n[4/9] Detecting invalid bbox rows...")
    invalid_df = detect_invalid_bbox(ann_df, no_finding_id)
    print(f"      Invalid bbox rows found : {len(invalid_df)}")

    # [5] Duplicate candidates
    print("\n[5/9] Detecting duplicate bbox candidates (IoU >= "
          f"{DUPLICATE_IOU_THRESHOLD})...")
    abn_valid = ann_df[
        (ann_df["class_id"] != no_finding_id) &
        ann_df[["x_min", "y_min", "x_max", "y_max"]].notna().all(axis=1)
    ].copy()
    dup_df = detect_duplicate_candidates(abn_valid, DUPLICATE_IOU_THRESHOLD)
    print(f"      Duplicate candidate pairs: {len(dup_df)}")

    # [6] Size distribution
    print("\n[6/9] Building bbox size distribution...")
    size_df = build_bbox_size_distribution(ann_df, no_finding_id)
    print(f"      Valid bbox rows for size analysis: {len(size_df)}")

    # [7] Summary + quality tables
    print("\n[7/9] Building quality summary and quality tables...")
    summary_df = build_quality_summary(
        ann_df, selected_df, invalid_df, dup_df, no_finding_id)
    quality_class_df = build_quality_by_class(
        ann_df, size_df, invalid_df, dup_df, no_finding_id)
    quality_image_df = build_quality_by_image(
        selected_df, ann_df, size_df, invalid_df, dup_df, no_finding_id)

    # [8] Save CSVs
    print("\n[8/9] Saving CSV files...")
    files = {
        "annotation_quality_summary.csv": summary_df,
        "invalid_bbox_rows.csv": invalid_df,
        "bbox_size_distribution.csv": size_df,
        "bbox_quality_by_class.csv": quality_class_df,
        "bbox_quality_by_image.csv": quality_image_df,
        "duplicate_bbox_candidates.csv": dup_df,
    }
    for fname, df in files.items():
        path = REPORT_DIR / fname
        df.to_csv(path, index=False)
        print(f"[OK] Saved: {path}  ({len(df)} rows)")

    # [9] Figures + report
    print("\n[9/9] Creating figures...")
    figures_created = create_figures(size_df)

    print("      Writing markdown report...")
    sanity_obs = compute_sanity_observations(summary_df)
    report_md = build_markdown_report(
        summary_df, quality_class_df, invalid_df, size_df, dup_df,
        sanity_obs, figures_created, config)
    report_path = REPORT_DIR / "annotation_quality_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"[OK] Report saved: {report_path}")

    # Terminal summary
    INT_METRICS = {
        "total_annotation_rows", "total_abnormal_bbox_rows",
        "total_no_finding_rows", "rows_missing_class_id",
        "rows_missing_class_name", "abnormal_rows_missing_any_bbox_coordinate",
        "bbox_rows_with_invalid_x_order", "bbox_rows_with_invalid_y_order",
        "bbox_rows_with_non_positive_width", "bbox_rows_with_non_positive_height",
        "bbox_rows_with_zero_area", "bbox_rows_with_negative_coordinate",
        "bbox_rows_with_extremely_small_area", "bbox_rows_with_extremely_large_area",
        "duplicate_bbox_candidate_rows",
        "normal_images_with_bbox", "abnormal_images_without_valid_bbox",
    }

    def sv(metric):
        row = summary_df[summary_df["metric"] == metric]
        if row.empty:
            return "N/A"
        val = row.iloc[0]["value"]
        if metric in INT_METRICS:
            try:
                return int(float(val))
            except (ValueError, TypeError):
                pass
        return val

    print("\n" + "=" * 80)
    print("PHASE 1B EXECUTION SUMMARY")
    print("=" * 80)
    print(f"  Total annotation rows                    : {sv('total_annotation_rows')}")
    print(f"  Total abnormal bbox rows                 : {sv('total_abnormal_bbox_rows')}")
    print(f"  Total No Finding rows                    : {sv('total_no_finding_rows')}")
    print(f"  Missing class_id rows                    : {sv('rows_missing_class_id')}")
    print(f"  Missing class_name rows                  : {sv('rows_missing_class_name')}")
    print(f"  Abnormal rows missing bbox coord         : {sv('abnormal_rows_missing_any_bbox_coordinate')}")
    print(f"  Invalid x-order bbox rows                : {sv('bbox_rows_with_invalid_x_order')}")
    print(f"  Invalid y-order bbox rows                : {sv('bbox_rows_with_invalid_y_order')}")
    print(f"  Non-positive width rows                  : {sv('bbox_rows_with_non_positive_width')}")
    print(f"  Non-positive height rows                 : {sv('bbox_rows_with_non_positive_height')}")
    print(f"  Zero-area bbox rows                      : {sv('bbox_rows_with_zero_area')}")
    print(f"  Negative-coordinate rows                 : {sv('bbox_rows_with_negative_coordinate')}")
    print(f"  Outside-boundary bbox rows               : {sv('bbox_rows_outside_image_boundary')}")
    print(f"  Extremely small area rows (<=p5)         : {sv('bbox_rows_with_extremely_small_area')}")
    print(f"  Extremely large area rows (>=p95)        : {sv('bbox_rows_with_extremely_large_area')}")
    print(f"  Duplicate bbox candidate pairs           : {sv('duplicate_bbox_candidate_rows')}")
    print(f"  Normal images with bbox                  : {sv('normal_images_with_bbox')}")
    print(f"  Abnormal images without valid bbox       : {sv('abnormal_images_without_valid_bbox')}")
    print(f"  Overall quality status                   : {sv('overall_quality_status')}")
    print(f"\n  Report dir : {REPORT_DIR}")
    print(f"  Figures    : {len(figures_created)} file(s) created")
    print("=" * 80)
    print("PHASE 1B COMPLETED — No Kappa, no split, no leakage, no conversion,"
          " no training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
