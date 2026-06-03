# PHASE 1B — Annotation Quality Analysis

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

**No Finding class_id:** `14` (excluded from bbox quality checks)

**Duplicate detection threshold:** IoU ≥ `0.95` (same image + class only)

**Image boundary check:** Image boundary checking was not performed because image
width and height are not available in the metadata-only subset. No DICOM or image
files were read in this phase.

---

## 3. Annotation Quality Summary

| Metric | Value |
|--------|-------|
| Total annotation rows | 37596 |
| Total abnormal bbox rows | 36096 |
| Total No Finding rows | 1500 |
| Rows missing class_id | 0 |
| Rows missing class_name | 0 |
| Abnormal rows missing any bbox coord | 0 |
| Bbox rows with invalid x-order (x_max ≤ x_min) | 0 |
| Bbox rows with invalid y-order (y_max ≤ y_min) | 0 |
| Bbox rows with non-positive width | 0 |
| Bbox rows with non-positive height | 0 |
| Bbox rows with zero/negative area | 0 |
| Bbox rows with negative coordinates | 0 |
| Bbox rows outside image boundary | not_available (image dimensions not in metadata) |
| Bbox rows with extremely small area (≤ p5) | 1805 |
| Bbox rows with extremely large area (≥ p95) | 1806 |
| Duplicate bbox candidate pairs | 78 |
| Normal images with valid bbox | 0 |
| Abnormal images without valid bbox | 0 |
| **Overall quality status** | **PASS** |

> **Note on No Finding rows:** The 1,500 No Finding rows correspond to annotation rows,
> not unique normal images. The metadata subset contains 500 unique normal images.
> Each normal image has multiple annotation rows (one per radiologist) all carrying
> class_id = 14 with no bbox coordinates.

> **Note on extremely small/large area:** "extremely small" = area ≤ 5th percentile;
> "extremely large" = area ≥ 95th percentile.
> These thresholds are computed from the distribution of valid bbox areas in this subset —
> they are **not** fixed pixel thresholds.
> Small and large bbox groups are percentile-based relative size flags and should not be
> interpreted as confirmed annotation errors.

---

## 4. Invalid Bounding Box Analysis

No invalid bbox rows were detected in this metadata subset.

> Invalid bbox rows are saved to `reports/phase_1b_annotation_quality/invalid_bbox_rows.csv`.
> If the file contains only a header row, no invalid bboxes were found.

---

## 5. Bounding Box Size Distribution

The following statistics are computed on **valid bbox rows only**
(class_id ≠ 14, coordinates present, width > 0, height > 0).


| Statistic | Width (px) | Height (px) | Area (px²) | Aspect Ratio |
|-----------|-----------|------------|-----------|--------------|
| Mean | 440.94 | 391.40 | 218415.71 | 1.4054 |
| Median | 323.00 | 320.00 | 106471.00 | 0.9853 |
| p5 | 76.00 | 66.00 | 6411.75 | 0.4436 |
| p95 | 1141.00 | 1086.00 | 733590.00 | 3.5290 |

**Area group distribution** (percentile-based: very_small ≤ p5 < small ≤ p25 < medium ≤ p75 < large ≤ p95 < very_large):

| Group | Count |
|-------|-------|
| very_small | 1805 |
| small | 7219 |
| medium | 18048 |
| large | 7220 |
| very_large | 1804 |


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
| 0 | Aortic enlargement | 7162 | 0 | 0.00% | 111786.4 | 96440.0 | 0.9192 | 0 | 10 | 7 |
| 3 | Cardiomegaly | 5427 | 0 | 0.00% | 430588.3 | 396256.0 | 2.8789 | 0 | 301 | 47 |
| 11 | Pleural thickening | 4842 | 0 | 0.00% | 61236.1 | 31626.5 | 1.9173 | 130 | 23 | 2 |
| 13 | Pulmonary fibrosis | 4655 | 0 | 0.00% | 150699.7 | 75744.0 | 1.2945 | 193 | 99 | 3 |
| 8 | Nodule/Mass | 2580 | 0 | 0.00% | 73546.4 | 9972.5 | 0.9927 | 999 | 37 | 1 |
| 7 | Lung Opacity | 2483 | 0 | 0.00% | 251134.1 | 151872.0 | 1.0190 | 49 | 152 | 2 |
| 10 | Pleural effusion | 2476 | 0 | 0.00% | 290881.4 | 62673.0 | 0.8776 | 32 | 318 | 3 |
| 9 | Other lesion | 2203 | 0 | 0.00% | 259795.8 | 119028.0 | 1.0210 | 220 | 193 | 0 |
| 6 | Infiltration | 1247 | 0 | 0.00% | 343015.9 | 244720.0 | 0.8972 | 1 | 141 | 1 |
| 5 | ILD | 1000 | 0 | 0.00% | 653507.9 | 551686.0 | 0.6373 | 2 | 349 | 3 |
| 2 | Calcification | 960 | 0 | 0.00% | 92550.7 | 27255.5 | 0.9361 | 179 | 9 | 2 |
| 4 | Consolidation | 556 | 0 | 0.00% | 325284.2 | 236006.0 | 0.9481 | 0 | 44 | 0 |
| 1 | Atelectasis | 279 | 0 | 0.00% | 392763.1 | 272586.0 | 1.0942 | 0 | 39 | 0 |
| 12 | Pneumothorax | 226 | 0 | 0.00% | 805752.1 | 531347.5 | 0.9336 | 0 | 89 | 7 |

> AR = aspect ratio (width / height). very_small / very_large = bbox in bottom/top 5th
> percentile of area distribution. dup_cand = duplicate candidate pairs for that class.

---

## 7. Image-Level Annotation Quality

Image-level stats are saved in
`reports/phase_1b_annotation_quality/bbox_quality_by_image.csv`.

Key highlights:
- Images with **has_invalid_bbox = True**: 0
- Images with **has_duplicate_candidate = True**: 68
- Normal images: all have `num_bbox = 0` and `num_valid_bbox = 0` by design.

---

## 8. Duplicate Bounding Box Candidate Analysis

78 duplicate bbox candidate pair(s) detected (IoU ≥ 0.95, same image + class).

Top 5 highest IoU pairs:
```
                        image_id         class_name      iou
672ece5866dfa259ad2cede3afd0d41f Aortic enlargement 0.986842
e529c5465cafd94ee9a3b38f7267523a       Cardiomegaly 0.979763
904b3b12d54cf5f6f4ccccbcaffa3714       Cardiomegaly 0.978326
e529c5465cafd94ee9a3b38f7267523a       Cardiomegaly 0.975611
f3e804892343e3b12e2542939d9101a6       Pneumothorax 0.975538
```

> **Important:** These pairs are metadata-level duplicate or near-duplicate candidates
> based on IoU ≥ 0.95 within the same image and class.
> They should not be interpreted as confirmed annotation errors or inter-rater
> agreement results. Duplicate candidates are identified purely by geometric overlap
> within the same `image_id` and `class_id`.
> This is **NOT** Kappa analysis and **NOT** inter-rater agreement analysis.
> Multiple annotators may legitimately annotate the same region — this check only
> flags pairs with near-identical coordinates for awareness.

---

## 9. Basic Sanity Observations

- [OK] No annotation rows with missing class_id.
- [OK] No annotation rows with missing class_name.
- [OK] No abnormal annotation rows have missing bbox coordinates.
- [OK] No bbox rows with x_max <= x_min.
- [OK] No bbox rows with y_max <= y_min.
- [OK] No bbox rows with negative coordinates.
- [OK] No zero-area bbox rows found.
- [OK] No normal images have actual bbox rows — consistent with No Finding label.
- [OK] All abnormal images have at least one valid bbox row.
- [INFO] 78 duplicate bbox candidate pair(s) found (IoU >= 0.95). These are annotation-level duplicates, NOT inter-rater disagreement analysis.
- [INFO] bbox_rows_outside_image_boundary: not_available — image dimensions are not present in the current metadata subset.

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
- `reports\phase_1b_annotation_quality\figures\bbox_width_distribution.png`
- `reports\phase_1b_annotation_quality\figures\bbox_height_distribution.png`
- `reports\phase_1b_annotation_quality\figures\bbox_area_distribution.png`
- `reports\phase_1b_annotation_quality\figures\bbox_aspect_ratio_distribution.png`

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
