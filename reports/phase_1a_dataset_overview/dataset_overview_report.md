# PHASE 1A — Dataset Overview Report

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
| Total images | 4894 |
| Abnormal images | 4394 |
| Normal images | 500 |
| Total annotation rows | 37596 |
| Total bbox rows (excl. No Finding) | 36096 |
| Number of abnormal classes | 14 |
| Images with at least 1 bbox | 4394 |
| Images without any bbox | 500 |
| Mean bbox per abnormal image | 8.2148 |
| Median bbox per abnormal image | 7.0 |
| Min bbox per abnormal image | 3 |
| Max bbox per abnormal image | 57 |

> **Note on counting levels:**
> - *Annotation rows*: total rows in `subset_train_annotations.csv` (includes No Finding rows).
> - *Bbox rows*: rows where `class_id != 14` AND coordinates are not null.
> - *Image-level stats*: computed per unique `image_id`.

---

## 4. Image-Level Distribution

- **Abnormal images:** 4394
- **Normal images:** 500
- **Images with at least 1 valid bbox:** 4394
- **Images with 0 valid bbox:** 500

Breakdown by bbox count group (across all images):

| Bbox group | Number of images |
|------------|-----------------|
| 0 bbox | 500 |
| 1 bbox | 0 |
| 2 bbox | 0 |
| 3 bbox | 370 |
| >=4 bbox | 4024 |

Normal images contribute entirely to the "0 bbox" group since No Finding annotations
carry no valid bounding box coordinates.

---

## 5. Class-Level Distribution

The dataset contains **14 abnormal classes** (excluding No Finding).

| class_id | class_name | num_bbox | num_images | bbox_pct | image_pct | mean_bbox/img |
|----------|------------|----------|------------|----------|-----------|---------------|
| 0 | Aortic enlargement | 7162 | 3067 | 19.84% | 69.80% | 2.3352 |
| 3 | Cardiomegaly | 5427 | 2300 | 15.03% | 52.34% | 2.3596 |
| 11 | Pleural thickening | 4842 | 1981 | 13.41% | 45.08% | 2.4442 |
| 13 | Pulmonary fibrosis | 4655 | 1617 | 12.90% | 36.80% | 2.8788 |
| 8 | Nodule/Mass | 2580 | 826 | 7.15% | 18.80% | 3.1235 |
| 7 | Lung Opacity | 2483 | 1322 | 6.88% | 30.09% | 1.8782 |
| 10 | Pleural effusion | 2476 | 1032 | 6.86% | 23.49% | 2.3992 |
| 9 | Other lesion | 2203 | 1134 | 6.10% | 25.81% | 1.9427 |
| 6 | Infiltration | 1247 | 613 | 3.45% | 13.95% | 2.0343 |
| 5 | ILD | 1000 | 386 | 2.77% | 8.78% | 2.5907 |
| 2 | Calcification | 960 | 452 | 2.66% | 10.29% | 2.1239 |
| 4 | Consolidation | 556 | 353 | 1.54% | 8.03% | 1.5751 |
| 1 | Atelectasis | 279 | 186 | 0.77% | 4.23% | 1.5000 |
| 12 | Pneumothorax | 226 | 96 | 0.63% | 2.18% | 2.3542 |

- **Most frequent class (by bbox count):** Aortic enlargement (7162 bbox rows)
- **Least frequent class (by bbox count):** Pneumothorax (226 bbox rows)

> Note: A single image may appear in multiple class rows if it contains more than one
> abnormality class. `image_percentage` is computed against total abnormal images (4394).

---

## 6. Bounding Box Distribution

Distribution of bbox counts per image across all 4894 images:

| Bbox group | Count |
|------------|-------|
| 0 bbox | 500 |
| 1 bbox | 0 |
| 2 bbox | 0 |
| 3 bbox | 370 |
| >=4 bbox | 4024 |

Statistics computed on abnormal images only:

| Statistic | Value |
|-----------|-------|
| Mean bbox per abnormal image | 8.2148 |
| Median bbox per abnormal image | 7.0 |
| Min bbox per abnormal image | 3 |
| Max bbox per abnormal image | 57 |

- Normal images have 0 valid bbox by design (No Finding label has no bbox coordinates).
- In this metadata subset, no abnormal images with 0 valid bbox were found.

> **Scope note:** Each bbox count reflects annotation-level bounding-box rows in the current
> metadata subset. It should not be interpreted directly as the number of unique clinical
> lesions per image.

---

## 7. Basic Annotation Sanity Observations

- All abnormal images have at least one valid bbox row.
- No normal images have actual bbox rows — consistent with No Finding label.
- No annotation rows with abnormal class_id have missing bbox coordinates.
- All annotation rows have a class_name value.

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
- `reports\phase_1a_dataset_overview\figures\class_bbox_distribution.png`
- `reports\phase_1a_dataset_overview\figures\class_image_distribution.png`
- `reports\phase_1a_dataset_overview\figures\bbox_per_image_distribution.png`

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
