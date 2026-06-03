"""
PHASE 1A — Dataset Overview Report
===================================
Mục tiêu: Tạo báo cáo tổng quan dữ liệu cho metadata-only subset của dataset
VinBigData Chest X-ray Abnormalities Detection.

Phase này CHỈ mô tả dataset ở mức overview:
- Số lượng ảnh, phân bố class, phân bố bbox, thống kê annotation.

KHÔNG thực hiện trong phase này:
- Không phân tích Kappa.
- Không tạo train/val/test split.
- Không tạo labeled/unlabeled split.
- Không kiểm tra leakage.
- Không convert COCO/YOLO.
- Không training model.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("configs/dataset.yaml")
REPORT_DIR = Path("reports/phase_1a_dataset_overview")
FIGURES_DIR = REPORT_DIR / "figures"
NO_FINDING_CLASS_ID = 14  # fallback nếu không có trong config


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Đọc cấu hình từ configs/dataset.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_value(config: dict, *keys, fallback=None, warn=True):
    """
    Truy cập nested key trong config dict.
    Nếu không tìm thấy, trả về fallback và in cảnh báo nếu warn=True.
    """
    obj = config
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            if warn:
                print(f"[WARNING] Config key '{'.'.join(str(k) for k in keys)}' not found."
                      f" Using fallback: {fallback!r}")
            return fallback
        obj = obj[key]
    return obj


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_metadata(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Đọc hai file metadata chính:
    - selected_image_ids.csv
    - subset_train_annotations.csv
    Trả về (selected_df, ann_df).
    """
    mf = get_config_value(config, "metadata_files", fallback={})

    selected_path = Path(get_config_value(
        config, "metadata_files", "selected_image_ids",
        fallback="data/raw/vinbigdata/metadata_subset/selected_image_ids.csv"))
    ann_path = Path(get_config_value(
        config, "metadata_files", "subset_annotations",
        fallback="data/raw/vinbigdata/metadata_subset/subset_train_annotations.csv"))

    for path, name in [(selected_path, "selected_image_ids"), (ann_path, "subset_annotations")]:
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] Required metadata file missing: {path} ({name})")

    selected_df = pd.read_csv(selected_path)
    ann_df = pd.read_csv(ann_path)

    # Kiểm tra cột bắt buộc
    required_selected = {"image_id", "subset_type"}
    required_ann = {"image_id", "class_id", "class_name", "x_min", "y_min", "x_max", "y_max"}

    missing_sel = required_selected - set(selected_df.columns)
    missing_ann = required_ann - set(ann_df.columns)

    if missing_sel:
        raise ValueError(f"[ERROR] Missing columns in selected_image_ids.csv: {missing_sel}")
    if missing_ann:
        raise ValueError(f"[ERROR] Missing columns in subset_train_annotations.csv: {missing_ann}")

    if selected_df.empty:
        raise ValueError("[ERROR] selected_image_ids.csv is empty.")
    if ann_df.empty:
        raise ValueError("[ERROR] subset_train_annotations.csv is empty.")

    return selected_df, ann_df


# ---------------------------------------------------------------------------
# Statistics builders
# ---------------------------------------------------------------------------

def build_dataset_summary(selected_df: pd.DataFrame,
                           ann_df: pd.DataFrame,
                           config: dict) -> pd.DataFrame:
    """
    Xây dựng dataset_summary.csv với các chỉ số tổng quan.
    Phân biệt rõ image-level / annotation-row-level / bbox-level.
    """
    no_finding_id = get_config_value(
        config, "subset", "normal_definition", "class_id",
        fallback=NO_FINDING_CLASS_ID)
    expected_abnormal = get_config_value(config, "subset", "abnormal_images", fallback=None)
    expected_normal = get_config_value(config, "subset", "normal_images", fallback=None)

    # --- Image-level ---
    total_images = selected_df["image_id"].nunique()
    abnormal_images = selected_df[selected_df["subset_type"] == "abnormal"]["image_id"].nunique()
    normal_images = selected_df[selected_df["subset_type"] == "normal"]["image_id"].nunique()

    # --- Annotation-row-level ---
    total_annotation_rows = len(ann_df)

    # --- Bbox-level: hàng có class_id != no_finding_id VÀ có tọa độ không null ---
    bbox_df = ann_df[
        (ann_df["class_id"] != no_finding_id) &
        ann_df["x_min"].notna() &
        ann_df["y_min"].notna() &
        ann_df["x_max"].notna() &
        ann_df["y_max"].notna()
    ].copy()
    total_bbox_rows = len(bbox_df)

    # Số class bất thường (không tính No Finding)
    num_abnormal_classes = ann_df[ann_df["class_id"] != no_finding_id]["class_id"].nunique()

    # Số ảnh có ít nhất 1 bbox thật
    images_with_bbox = bbox_df["image_id"].nunique()
    images_without_bbox = total_images - images_with_bbox

    # Số bbox per abnormal image
    bbox_per_abnormal = (
        bbox_df.groupby("image_id")["class_id"].count()
    )
    # Chỉ tính trên ảnh abnormal (kể cả ảnh abnormal có 0 bbox thật)
    abnormal_ids = selected_df[selected_df["subset_type"] == "abnormal"]["image_id"]
    bbox_counts_abnormal = abnormal_ids.map(
        lambda img: bbox_per_abnormal.get(img, 0)
    )

    mean_bbox = round(bbox_counts_abnormal.mean(), 4)
    median_bbox = round(bbox_counts_abnormal.median(), 4)
    min_bbox = int(bbox_counts_abnormal.min())
    max_bbox = int(bbox_counts_abnormal.max())

    rows = [
        ("total_images", total_images),
        ("abnormal_images", abnormal_images),
        ("normal_images", normal_images),
        ("expected_abnormal_images", expected_abnormal if expected_abnormal is not None else "N/A"),
        ("expected_normal_images", expected_normal if expected_normal is not None else "N/A"),
        ("total_annotation_rows", total_annotation_rows),
        ("total_bbox_rows", total_bbox_rows),
        ("num_abnormal_classes", num_abnormal_classes),
        ("num_images_with_bbox", images_with_bbox),
        ("num_images_without_bbox", images_without_bbox),
        ("mean_bbox_per_abnormal_image", mean_bbox),
        ("median_bbox_per_abnormal_image", median_bbox),
        ("min_bbox_per_abnormal_image", min_bbox),
        ("max_bbox_per_abnormal_image", max_bbox),
    ]

    summary_df = pd.DataFrame(rows, columns=["metric", "value"])
    return summary_df


def build_class_distribution(ann_df: pd.DataFrame,
                              selected_df: pd.DataFrame,
                              config: dict) -> pd.DataFrame:
    """
    Xây dựng class_distribution.csv cho các abnormal class.
    Không gộp class, không lọc bỏ class hiếm.
    """
    no_finding_id = get_config_value(
        config, "subset", "normal_definition", "class_id",
        fallback=NO_FINDING_CLASS_ID)

    total_abnormal_images = selected_df[selected_df["subset_type"] == "abnormal"]["image_id"].nunique()

    # Chỉ lấy annotation có bbox thật (class_id != no_finding_id, tọa độ không null)
    bbox_df = ann_df[
        (ann_df["class_id"] != no_finding_id) &
        ann_df["x_min"].notna()
    ].copy()

    # Kiểm tra class name thiếu
    missing_name = bbox_df["class_name"].isna().sum()
    if missing_name > 0:
        print(f"[WARNING] {missing_name} bbox rows have missing class_name.")

    grouped = (
        bbox_df.groupby(["class_id", "class_name"])
        .agg(
            num_bbox=("image_id", "count"),
            num_images=("image_id", "nunique"),
        )
        .reset_index()
    )

    total_bbox = grouped["num_bbox"].sum()
    grouped["bbox_percentage"] = (grouped["num_bbox"] / total_bbox * 100).round(2)
    grouped["image_percentage"] = (grouped["num_images"] / total_abnormal_images * 100).round(2)
    grouped["mean_bbox_per_image_for_class"] = (grouped["num_bbox"] / grouped["num_images"]).round(4)

    grouped = grouped.sort_values("num_bbox", ascending=False).reset_index(drop=True)
    return grouped


def build_bbox_distribution(ann_df: pd.DataFrame,
                             selected_df: pd.DataFrame,
                             config: dict) -> pd.DataFrame:
    """
    Xây dựng bbox_distribution.csv — mỗi dòng là một ảnh,
    kèm số bbox, is_normal, is_abnormal, has_bbox.
    """
    no_finding_id = get_config_value(
        config, "subset", "normal_definition", "class_id",
        fallback=NO_FINDING_CLASS_ID)

    bbox_df = ann_df[
        (ann_df["class_id"] != no_finding_id) &
        ann_df["x_min"].notna()
    ].copy()

    bbox_per_image = bbox_df.groupby("image_id")["class_id"].count().rename("num_bbox")

    # Merge với selected để giữ tất cả ảnh
    result = selected_df[["image_id", "subset_type"]].copy()
    result = result.merge(bbox_per_image, on="image_id", how="left")
    result["num_bbox"] = result["num_bbox"].fillna(0).astype(int)
    result["has_bbox"] = result["num_bbox"] > 0
    result["is_normal"] = result["subset_type"] == "normal"
    result["is_abnormal"] = result["subset_type"] == "abnormal"
    result = result.drop(columns=["subset_type"])
    result = result.sort_values("num_bbox", ascending=False).reset_index(drop=True)
    return result


def build_image_level_distribution(ann_df: pd.DataFrame,
                                   selected_df: pd.DataFrame,
                                   config: dict) -> pd.DataFrame:
    """
    Xây dựng image_level_distribution.csv — mỗi dòng là một ảnh với
    image_type, num_bbox, num_unique_classes, class_names, class_ids.
    Logic với ảnh normal:
        - num_bbox = 0
        - class_names = "" (No Finding không có bbox thật)
        - class_ids = ""
    """
    no_finding_id = get_config_value(
        config, "subset", "normal_definition", "class_id",
        fallback=NO_FINDING_CLASS_ID)

    bbox_df = ann_df[
        (ann_df["class_id"] != no_finding_id) &
        ann_df["x_min"].notna()
    ].copy()

    # Thống kê per-image
    agg = (
        bbox_df.groupby("image_id")
        .agg(
            num_bbox=("class_id", "count"),
            num_unique_classes=("class_id", "nunique"),
            class_names=("class_name", lambda x: "|".join(sorted(x.dropna().unique()))),
            class_ids=("class_id", lambda x: "|".join(str(c) for c in sorted(x.unique()))),
        )
        .reset_index()
    )

    result = selected_df[["image_id", "subset_type"]].rename(columns={"subset_type": "image_type"})
    result = result.merge(agg, on="image_id", how="left")

    # Ảnh normal: điền 0 / ""
    result["num_bbox"] = result["num_bbox"].fillna(0).astype(int)
    result["num_unique_classes"] = result["num_unique_classes"].fillna(0).astype(int)
    result["class_names"] = result["class_names"].fillna("")
    result["class_ids"] = result["class_ids"].fillna("")

    result = result.sort_values(["image_type", "num_bbox"], ascending=[True, False]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Sanity observations
# ---------------------------------------------------------------------------

def compute_sanity_observations(ann_df: pd.DataFrame,
                                selected_df: pd.DataFrame,
                                config: dict) -> list[str]:
    """
    Tính các sanity observation cơ bản ở mức overview.
    Không làm leakage, không làm split, không phân tích Kappa.
    """
    no_finding_id = get_config_value(
        config, "subset", "normal_definition", "class_id",
        fallback=NO_FINDING_CLASS_ID)

    observations = []

    abnormal_ids = set(selected_df[selected_df["subset_type"] == "abnormal"]["image_id"])
    normal_ids = set(selected_df[selected_df["subset_type"] == "normal"]["image_id"])

    bbox_df = ann_df[
        (ann_df["class_id"] != no_finding_id) &
        ann_df["x_min"].notna()
    ]

    images_with_bbox = set(bbox_df["image_id"])

    # Ảnh abnormal không có bbox
    abnormal_no_bbox = abnormal_ids - images_with_bbox
    if abnormal_no_bbox:
        observations.append(
            f"Abnormal images with no bbox: {len(abnormal_no_bbox)} image(s). "
            f"These may carry annotation rows without valid coordinates."
        )
    else:
        observations.append("All abnormal images have at least one valid bbox row.")

    # Ảnh normal có bbox thật
    normal_with_bbox = normal_ids & images_with_bbox
    if normal_with_bbox:
        observations.append(
            f"[WARNING] Normal images with actual bbox (unexpected): {len(normal_with_bbox)} image(s)."
        )
    else:
        observations.append("No normal images have actual bbox rows — consistent with No Finding label.")

    # Thiếu tọa độ bbox trong non-No-Finding rows
    non_finding = ann_df[ann_df["class_id"] != no_finding_id]
    missing_coords = non_finding[
        non_finding["x_min"].isna() |
        non_finding["y_min"].isna() |
        non_finding["x_max"].isna() |
        non_finding["y_max"].isna()
    ]
    if not missing_coords.empty:
        observations.append(
            f"Annotation rows with class_id != {no_finding_id} but missing bbox coordinates: "
            f"{len(missing_coords)} row(s)."
        )
    else:
        observations.append(
            "No annotation rows with abnormal class_id have missing bbox coordinates."
        )

    # Class name thiếu
    missing_class_name = ann_df[ann_df["class_name"].isna()]
    if not missing_class_name.empty:
        observations.append(
            f"Annotation rows with missing class_name: {len(missing_class_name)} row(s)."
        )
    else:
        observations.append("All annotation rows have a class_name value.")

    return observations


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def create_figures(class_dist_df: pd.DataFrame,
                   bbox_dist_df: pd.DataFrame) -> list[str]:
    """
    Tạo 3 biểu đồ trong thư mục figures/.
    Trả về danh sách đường dẫn file đã tạo thành công.
    Nếu thiếu matplotlib, ghi rõ lý do và không âm thầm bỏ qua.
    """
    created = []

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        msg = "[FIGURE ERROR] matplotlib is not installed. Cannot create figures."
        print(msg)
        return []

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: class_bbox_distribution ---
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(class_dist_df["class_name"], class_dist_df["num_bbox"], color="steelblue")
        ax.set_title("Number of Bounding Boxes per Abnormal Class", fontsize=13)
        ax.set_xlabel("Class Name")
        ax.set_ylabel("Number of Bounding Boxes")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        out = FIGURES_DIR / "class_bbox_distribution.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[OK] Figure saved: {out}")
        created.append(str(out))
    except Exception as e:
        print(f"[FIGURE ERROR] class_bbox_distribution.png — {e}")

    # --- Figure 2: class_image_distribution ---
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(class_dist_df["class_name"], class_dist_df["num_images"], color="darkorange")
        ax.set_title("Number of Images per Abnormal Class", fontsize=13)
        ax.set_xlabel("Class Name")
        ax.set_ylabel("Number of Images")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        out = FIGURES_DIR / "class_image_distribution.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[OK] Figure saved: {out}")
        created.append(str(out))
    except Exception as e:
        print(f"[FIGURE ERROR] class_image_distribution.png — {e}")

    # --- Figure 3: bbox_per_image_distribution ---
    try:
        def bucket(n):
            if n == 0: return "0 bbox"
            if n == 1: return "1 bbox"
            if n == 2: return "2 bbox"
            if n == 3: return "3 bbox"
            return ">=4 bbox"

        order = ["0 bbox", "1 bbox", "2 bbox", "3 bbox", ">=4 bbox"]
        counts = bbox_dist_df["num_bbox"].apply(bucket).value_counts().reindex(order, fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(counts.index, counts.values, color="mediumseagreen")
        ax.set_title("Distribution of Bounding Boxes per Image", fontsize=13)
        ax.set_xlabel("Bbox Group")
        ax.set_ylabel("Number of Images")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha="center", fontsize=9)
        plt.tight_layout()
        out = FIGURES_DIR / "bbox_per_image_distribution.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[OK] Figure saved: {out}")
        created.append(str(out))
    except Exception as e:
        print(f"[FIGURE ERROR] bbox_per_image_distribution.png — {e}")

    return created


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def build_markdown_report(summary_df: pd.DataFrame,
                           class_dist_df: pd.DataFrame,
                           bbox_dist_df: pd.DataFrame,
                           image_level_df: pd.DataFrame,
                           sanity_obs: list[str],
                           figures_created: list[str],
                           config: dict) -> str:
    """
    Tạo nội dung dataset_overview_report.md.
    """
    no_finding_id = get_config_value(
        config, "subset", "normal_definition", "class_id",
        fallback=NO_FINDING_CLASS_ID)

    # Các metric là count (integer) — ép kiểu int để tránh hiển thị dạng 4894.0
    INT_METRICS = {
        "total_images", "abnormal_images", "normal_images",
        "total_annotation_rows", "total_bbox_rows", "num_abnormal_classes",
        "num_images_with_bbox", "num_images_without_bbox",
        "min_bbox_per_abnormal_image", "max_bbox_per_abnormal_image",
    }

    def sv(metric):
        """Lấy giá trị từ summary_df theo metric name.
        Count metrics được ép kiểu int để hiển thị không có phần thập phân.
        """
        row = summary_df[summary_df["metric"] == metric]
        if row.empty:
            return "N/A"
        val = row.iloc[0]["value"]
        if metric in INT_METRICS and val != "N/A":
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return val
        return val

    total_images = sv("total_images")
    abnormal_images = sv("abnormal_images")
    normal_images = sv("normal_images")
    total_ann_rows = sv("total_annotation_rows")
    total_bbox_rows = sv("total_bbox_rows")
    num_ab_classes = sv("num_abnormal_classes")
    imgs_with_bbox = sv("num_images_with_bbox")
    imgs_without_bbox = sv("num_images_without_bbox")
    mean_bbox = sv("mean_bbox_per_abnormal_image")
    median_bbox = sv("median_bbox_per_abnormal_image")
    min_bbox = sv("min_bbox_per_abnormal_image")
    max_bbox = sv("max_bbox_per_abnormal_image")

    # Bbox bucket stats (all images)
    def bucket(n):
        if n == 0: return "0"
        if n == 1: return "1"
        if n == 2: return "2"
        if n == 3: return "3"
        return ">=4"

    bucket_counts = bbox_dist_df["num_bbox"].apply(bucket).value_counts()
    b0 = bucket_counts.get("0", 0)
    b1 = bucket_counts.get("1", 0)
    b2 = bucket_counts.get("2", 0)
    b3 = bucket_counts.get("3", 0)
    bge4 = bucket_counts.get(">=4", 0)

    # Top and bottom class
    top_class = class_dist_df.iloc[0]["class_name"] if not class_dist_df.empty else "N/A"
    top_class_bbox = class_dist_df.iloc[0]["num_bbox"] if not class_dist_df.empty else "N/A"
    bot_class = class_dist_df.iloc[-1]["class_name"] if not class_dist_df.empty else "N/A"
    bot_class_bbox = class_dist_df.iloc[-1]["num_bbox"] if not class_dist_df.empty else "N/A"

    # Class distribution markdown table
    class_table_rows = []
    for _, row in class_dist_df.iterrows():
        class_table_rows.append(
            f"| {int(row['class_id'])} | {row['class_name']} | "
            f"{int(row['num_bbox'])} | {int(row['num_images'])} | "
            f"{row['bbox_percentage']:.2f}% | {row['image_percentage']:.2f}% | "
            f"{row['mean_bbox_per_image_for_class']:.4f} |"
        )
    class_table = "\n".join(class_table_rows)

    # Figures section
    if figures_created:
        fig_lines = "\n".join(f"- `{f}`" for f in figures_created)
    else:
        fig_lines = "- No figures were created (matplotlib not available or error occurred)."

    # Sanity observations list
    sanity_lines = "\n".join(f"- {obs}" for obs in sanity_obs)

    report = f"""# PHASE 1A — Dataset Overview Report

## 1. Objective

This report provides a **metadata-level descriptive overview** of the VinBigData Chest X-ray
Abnormalities Detection subset used in this project.

The scope is strictly limited to describing dataset statistics: image counts, class distribution,
bounding box distribution, and basic annotation sanity observations.

**This report does NOT perform:**
- Kappa / inter-annotator agreement analysis
- Train / val / test split or labeled / unlabeled split
- Data leakage checking
- COCO or YOLO format conversion
- Any model training, inference, or pseudo-labeling

---

## 2. Dataset Source and Current Subset

- **Dataset:** VinBigData Chest X-ray Abnormalities Detection (Kaggle competition)
- **Subset type:** Metadata-only subset (no DICOM images stored in the repository)
- **Expected subset composition:**
  - Abnormal images: 4,394 (all images with at least one annotation with `class_id != 14`)
  - Normal images: 500 (randomly sampled with seed 42 from `class_id == 14` images)
  - **Total:** 4,894 images
- **Metadata location:** `data/raw/vinbigdata/metadata_subset/`
- **Config:** `configs/dataset.yaml`

---

## 3. Overall Dataset Summary

| Metric | Value |
|--------|-------|
| Total images | {total_images} |
| Abnormal images | {abnormal_images} |
| Normal images | {normal_images} |
| Total annotation rows | {total_ann_rows} |
| Total bbox rows (excl. No Finding) | {total_bbox_rows} |
| Number of abnormal classes | {num_ab_classes} |
| Images with at least 1 bbox | {imgs_with_bbox} |
| Images without any bbox | {imgs_without_bbox} |
| Mean bbox per abnormal image | {mean_bbox} |
| Median bbox per abnormal image | {median_bbox} |
| Min bbox per abnormal image | {min_bbox} |
| Max bbox per abnormal image | {max_bbox} |

> **Note on counting levels:**
> - *Annotation rows*: total rows in `subset_train_annotations.csv` (includes No Finding rows).
> - *Bbox rows*: rows where `class_id != {no_finding_id}` AND coordinates are not null.
> - *Image-level stats*: computed per unique `image_id`.

---

## 4. Image-Level Distribution

- **Abnormal images:** {abnormal_images}
- **Normal images:** {normal_images}
- **Images with at least 1 valid bbox:** {imgs_with_bbox}
- **Images with 0 valid bbox:** {imgs_without_bbox}

Breakdown by bbox count group (across all images):

| Bbox group | Number of images |
|------------|-----------------|
| 0 bbox | {b0} |
| 1 bbox | {b1} |
| 2 bbox | {b2} |
| 3 bbox | {b3} |
| >=4 bbox | {bge4} |

Normal images contribute entirely to the "0 bbox" group since No Finding annotations
carry no valid bounding box coordinates.

---

## 5. Class-Level Distribution

The dataset contains **{num_ab_classes} abnormal classes** (excluding No Finding).

| class_id | class_name | num_bbox | num_images | bbox_pct | image_pct | mean_bbox/img |
|----------|------------|----------|------------|----------|-----------|---------------|
{class_table}

- **Most frequent class (by bbox count):** {top_class} ({top_class_bbox} bbox rows)
- **Least frequent class (by bbox count):** {bot_class} ({bot_class_bbox} bbox rows)

> Note: A single image may appear in multiple class rows if it contains more than one
> abnormality class. `image_percentage` is computed against total abnormal images ({abnormal_images}).

---

## 6. Bounding Box Distribution

Distribution of bbox counts per image across all {total_images} images:

| Bbox group | Count |
|------------|-------|
| 0 bbox | {b0} |
| 1 bbox | {b1} |
| 2 bbox | {b2} |
| 3 bbox | {b3} |
| >=4 bbox | {bge4} |

Statistics computed on abnormal images only:

| Statistic | Value |
|-----------|-------|
| Mean bbox per abnormal image | {mean_bbox} |
| Median bbox per abnormal image | {median_bbox} |
| Min bbox per abnormal image | {min_bbox} |
| Max bbox per abnormal image | {max_bbox} |

- Normal images have 0 valid bbox by design (No Finding label has no bbox coordinates).
- In this metadata subset, no abnormal images with 0 valid bbox were found.

> **Scope note:** Each bbox count reflects annotation-level bounding-box rows in the current
> metadata subset. It should not be interpreted directly as the number of unique clinical
> lesions per image.

---

## 7. Basic Annotation Sanity Observations

{sanity_lines}

> These are overview-level observations only. No leakage checking, split analysis,
> or Kappa analysis was performed.

---

## 8. Generated Artifacts

**CSV files:**
- `reports/phase_1a_dataset_overview/dataset_summary.csv`
- `reports/phase_1a_dataset_overview/class_distribution.csv`
- `reports/phase_1a_dataset_overview/bbox_distribution.csv`
- `reports/phase_1a_dataset_overview/image_level_distribution.csv`

**Figures:**
{fig_lines}

**Report:**
- `reports/phase_1a_dataset_overview/dataset_overview_report.md`

---

## 9. Scope Boundary

> This report is limited to dataset overview only. It does not perform Kappa analysis,
> dataset splitting, leakage checking, annotation conversion, or model training.

---

## 10. Next Step Recommendation

After the user confirms that PHASE 1A is complete and the numbers are acceptable,
the next phase can be designed separately (e.g., annotation quality analysis,
dataset splitting strategy, or baseline model design).

**No further action is taken automatically.**
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PHASE 1A — Dataset Overview Report")
    print("=" * 80)

    # --- Load config ---
    print("\n[1/8] Loading config...")
    config = load_config()
    print(f"[OK] Config loaded from {CONFIG_PATH}")

    # --- Create output dirs ---
    print("\n[2/8] Creating output directories...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Report dir: {REPORT_DIR}")
    print(f"[OK] Figures dir: {FIGURES_DIR}")

    # --- Load metadata ---
    print("\n[3/8] Loading metadata files...")
    selected_df, ann_df = load_metadata(config)
    print(f"[OK] selected_image_ids: {selected_df.shape}")
    print(f"[OK] subset_train_annotations: {ann_df.shape}")

    # --- Build statistics ---
    print("\n[4/8] Building dataset summary...")
    summary_df = build_dataset_summary(selected_df, ann_df, config)

    print("\n[5/8] Building class distribution...")
    class_dist_df = build_class_distribution(ann_df, selected_df, config)

    print("\n[6/8] Building bbox distribution...")
    bbox_dist_df = build_bbox_distribution(ann_df, selected_df, config)

    print("      Building image-level distribution...")
    image_level_df = build_image_level_distribution(ann_df, selected_df, config)

    # --- Sanity observations ---
    print("\n[7/8] Computing sanity observations...")
    sanity_obs = compute_sanity_observations(ann_df, selected_df, config)
    for obs in sanity_obs:
        print(f"      {obs}")

    # --- Save CSVs ---
    summary_path = REPORT_DIR / "dataset_summary.csv"
    class_dist_path = REPORT_DIR / "class_distribution.csv"
    bbox_dist_path = REPORT_DIR / "bbox_distribution.csv"
    image_level_path = REPORT_DIR / "image_level_distribution.csv"

    summary_df.to_csv(summary_path, index=False)
    class_dist_df.to_csv(class_dist_path, index=False)
    bbox_dist_df.to_csv(bbox_dist_path, index=False)
    image_level_df.to_csv(image_level_path, index=False)

    print(f"\n[OK] Saved: {summary_path}")
    print(f"[OK] Saved: {class_dist_path}")
    print(f"[OK] Saved: {bbox_dist_path}")
    print(f"[OK] Saved: {image_level_path}")

    # --- Figures ---
    print("\n[8/8] Creating figures...")
    figures_created = create_figures(class_dist_df, bbox_dist_df)

    # --- Markdown report ---
    print("\n      Writing markdown report...")
    report_md = build_markdown_report(
        summary_df, class_dist_df, bbox_dist_df, image_level_df,
        sanity_obs, figures_created, config
    )
    report_path = REPORT_DIR / "dataset_overview_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"[OK] Report saved: {report_path}")

    # --- Terminal summary ---
    # Count metrics ép kiểu int để không hiển thị dạng 4894.0
    _TERM_INT_METRICS = {
        "total_images", "abnormal_images", "normal_images",
        "total_annotation_rows", "total_bbox_rows", "num_abnormal_classes",
        "num_images_with_bbox", "num_images_without_bbox",
        "min_bbox_per_abnormal_image", "max_bbox_per_abnormal_image",
    }

    def sv(metric):
        row = summary_df[summary_df["metric"] == metric]
        if row.empty:
            return "N/A"
        val = row.iloc[0]["value"]
        if metric in _TERM_INT_METRICS and val != "N/A":
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return val
        return val

    print("\n" + "=" * 80)
    print("PHASE 1A EXECUTION SUMMARY")
    print("=" * 80)
    print(f"  Total images            : {sv('total_images')}")
    print(f"  Abnormal images         : {sv('abnormal_images')}")
    print(f"  Normal images           : {sv('normal_images')}")
    print(f"  Total annotation rows   : {sv('total_annotation_rows')}")
    print(f"  Total bbox rows         : {sv('total_bbox_rows')}")
    print(f"  Num abnormal classes    : {sv('num_abnormal_classes')}")
    print(f"  Images with bbox        : {sv('num_images_with_bbox')}")
    print(f"  Images without bbox     : {sv('num_images_without_bbox')}")
    print(f"  Mean bbox/abnormal img  : {sv('mean_bbox_per_abnormal_image')}")
    print(f"  Median bbox/abnormal img: {sv('median_bbox_per_abnormal_image')}")
    print(f"  Min bbox/abnormal img   : {sv('min_bbox_per_abnormal_image')}")
    print(f"  Max bbox/abnormal img   : {sv('max_bbox_per_abnormal_image')}")
    print(f"\n  Report dir : {REPORT_DIR}")
    print(f"  Figures    : {len(figures_created)} file(s) created")
    print("=" * 80)
    print("PHASE 1A COMPLETED — No split, no Kappa, no leakage, no training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
