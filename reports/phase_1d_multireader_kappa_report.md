# PHASE 1D — Multi-Reader Annotation Agreement Feasibility & Kappa Analysis

**Workflow Lock: WF-SSL-XRAY-DET-V1**

---

## 1. Objective

Check whether the VinBigData subset metadata contains multi-reader annotation
information, assess Kappa feasibility at three levels (image, class, bbox), and
compute Kappa wherever methodologically valid.

**This phase does NOT:** modify any annotation or bbox, create canonical annotation
files, assume any target format, create any split, or train any model.

---

## 2. Workflow Lock

**Keyword: WF-SSL-XRAY-DET-V1**

- Kappa / multi-reader agreement → PHASE 1D and PHASE 6 only.
- Canonical annotation → PHASE 2B (format-agnostic).
- COCO / YOLO / framework decisions → PHASE 2C only.
- Format conversion → PHASE 2D only.
- Fixed split → PHASE 2E only.
- SSL labeled/unlabeled split → PHASE 2F only.
- ViT / attention → PHASE 2C, 4B, 5, 6.
- Do not commit automatically.

---

## 3. Input Metadata Files Inspected

| File | Rows | Columns |
|------|------|---------|
| `data\processed\vinbigdata\phase_1d_kappa\bbox_level_agreement_feasibility.csv` | 6 | 3 |
| `data\processed\vinbigdata\phase_1d_kappa\class_level_kappa.csv` | 42 | 8 |
| `data\processed\vinbigdata\phase_1d_kappa\image_level_kappa.csv` | 1 | 7 |
| `data\processed\vinbigdata\phase_1d_kappa\image_level_pairwise_kappa.csv` | 3 | 5 |
| `data\processed\vinbigdata\phase_1d_kappa\kappa_feasibility_summary.csv` | 4 | 4 |
| `data\processed\vinbigdata\phase_1d_kappa\metadata_column_audit.csv` | 246 | 5 |
| `data\processed\vinbigdata\phase_1d_kappa\reader_column_candidates.csv` | 8 | 5 |
| `data\processed\vinbigdata\phase_1d_kappa\reader_count_per_image.csv` | 4894 | 5 |
| `data\raw\vinbigdata\metadata_subset\abnormal_image_ids.csv` | 4394 | 2 |
| `data\raw\vinbigdata\metadata_subset\normal_image_ids_500.csv` | 500 | 2 |
| `data\raw\vinbigdata\metadata_subset\positive_normal_summary.csv` | 1 | 5 |
| `data\raw\vinbigdata\metadata_subset\selected_image_ids.csv` | 4894 | 2 |
| `data\raw\vinbigdata\metadata_subset\subset_class_distribution.csv` | 15 | 4 |
| `data\raw\vinbigdata\metadata_subset\subset_summary.csv` | 1 | 15 |
| `data\raw\vinbigdata\metadata_subset\subset_train_annotations.csv` | 37596 | 8 |
| `data\raw\vinbigdata\original\train.csv` | 67914 | 8 |
| `reports\phase2_dataset_engineering\phase2a_dicom_bbox_boundary_summary.csv` | 40 | 2 |
| `reports\phase2_dataset_engineering\phase2a_image_metadata.csv` | 4894 | 23 |
| `reports\phase2_dataset_engineering\phase2a_invalid_bbox_boundary_cases.csv` | 0 | 12 |
| `reports\phase2_dataset_engineering\phase2a_invalid_dicom_files.csv` | 0 | 4 |
| `reports\phase2_dataset_engineering\phase2a_missing_dicom_files.csv` | 0 | 2 |
| `reports\phase_1a_dataset_overview\bbox_distribution.csv` | 4894 | 5 |
| `reports\phase_1a_dataset_overview\class_distribution.csv` | 14 | 7 |
| `reports\phase_1a_dataset_overview\dataset_summary.csv` | 14 | 2 |
| `reports\phase_1a_dataset_overview\image_level_distribution.csv` | 4894 | 6 |
| `reports\phase_1b_annotation_quality\annotation_quality_summary.csv` | 19 | 2 |
| `reports\phase_1b_annotation_quality\bbox_quality_by_class.csv` | 14 | 16 |
| `reports\phase_1b_annotation_quality\bbox_quality_by_image.csv` | 4894 | 12 |
| `reports\phase_1b_annotation_quality\bbox_size_distribution.csv` | 36096 | 14 |
| `reports\phase_1b_annotation_quality\duplicate_bbox_candidates.csv` | 78 | 7 |
| `reports\phase_1b_annotation_quality\invalid_bbox_rows.csv` | 0 | 11 |
| `reports\phase_1c_dataset_scope_decision\class_scope_decision.csv` | 14 | 9 |
| `reports\phase_1c_dataset_scope_decision\dataset_scope_decision_summary.csv` | 9 | 8 |
| `reports\phase_1c_dataset_scope_decision\limitation_register.csv` | 7 | 7 |
| `reports\phase_1c_dataset_scope_decision\next_phase_readiness_checklist.csv` | 16 | 5 |
| `reports\phase_1c_dataset_scope_decision\normal_policy_decision.csv` | 5 | 6 |

---

## 4. Metadata Column Audit Summary

Total unique CSV files inspected: **36**
Total column entries audited: **248**

Role distribution:

| Role | Count |
|------|-------|
| unknown | 122 |
| bbox_coordinate | 59 |
| class_name | 27 |
| metadata | 16 |
| image_id | 16 |
| reader_candidate | 8 |

---

## 5. Reader / Annotator / Radiologist Column Candidates

**Reader column candidates found:** 8

| File | Column | Unique Values | Sample Values |\n|------|--------|---------------|---------------|\n| `data\processed\vinbigdata\phase_1d_kappa\class_level_kappa.csv` | `reader_triplet` | 1 | ('R10', 'R8', 'R9') |\n| `data\processed\vinbigdata\phase_1d_kappa\class_level_kappa.csv` | `reader_pair` | 3 | R10_vs_R8, R10_vs_R9, R8_vs_R9 |\n| `data\processed\vinbigdata\phase_1d_kappa\image_level_pairwise_kappa.csv` | `reader_1` | 2 | R10, R8 |\n| `data\processed\vinbigdata\phase_1d_kappa\image_level_pairwise_kappa.csv` | `reader_2` | 2 | R8, R9 |\n| `data\processed\vinbigdata\phase_1d_kappa\reader_count_per_image.csv` | `selected_reader_column` | 1 | rad_id |\n| `data\processed\vinbigdata\phase_1d_kappa\reader_count_per_image.csv` | `num_unique_readers` | 1 | 3 |\n| `data\raw\vinbigdata\metadata_subset\subset_train_annotations.csv` | `rad_id` | 17 | R1, R10, R11, R12, R13, R14, R15, R16 |\n| `data\raw\vinbigdata\original\train.csv` | `rad_id` | 17 | R1, R10, R11, R12, R13, R14, R15, R16 |

**Selected reader column:** `rad_id`
- 17 unique reader IDs: R1–R17
- Exactly **3 readers** per image (consistent across all 4,894 images)
- Most common reader triplet: **(R8, R9, R10)** covering **4,222 images** (86.3%)

---

## 6. Multi-Reader Availability Result

✅ **Multi-reader metadata IS available.**

- Reader column: `rad_id`
- Readers: 17 unique (R1–R17)
- Coverage: 4,894 / 4,894 images have exactly 3 independent reader annotations
- 3 readers per image is sufficient for Fleiss' Kappa and pairwise Cohen's Kappa

---

## 7. Reader Count Per Image Summary

| Metric | Value |
|--------|-------|
| Total images | 4,894 |
| Images with exactly 3 readers | 4,894 |
| Images with < 3 readers | 0 |
| Images with > 3 readers | 0 |
| Mean readers per image | 3.0 |
| Most common reader triplet | (R8, R9, R10) — 4,222 images |

---

## 8. Image-Level Kappa Feasibility and Result

**Feasibility: FEASIBLE**

**Kappa methodology:**
In PHASE 1D, image-level abnormal/normal agreement was evaluated using
**Fleiss' Kappa** because each image had three reader assessments.
Pairwise Cohen's Kappa was additionally reported as a consistency check
between reader pairs within the dominant triplet.
Class-level disease presence/absence agreement was evaluated using
**mean pairwise Cohen's Kappa** across the dominant reader triplet (R8/R9/R10).
**Weighted Kappa was NOT used** because the labels are nominal disease
categories or binary presence/absence decisions rather than ordinal severity scores.

For each (image, reader) pair, an independent binary decision was derived:
- `is_abnormal = 1` if the reader labeled ≥1 abnormal class (class_id ≠ 14)
- `is_abnormal = 0` if the reader labeled only No Finding (class_id = 14)

### Results

| Metric | Value |
|--------|-------|
| Total images evaluated | 4,894 |
| Images with unanimous reader decisions | 4,894 |
| Images with reader disagreement | 0 |
| Fleiss' Kappa (image-level) | **1.0** |

### Pairwise Cohen's Kappa (image-level)

| Reader Pair | Cohen's Kappa |
|-------------|---------------|
| R10 vs R8 | 1.0000 |
| R10 vs R9 | 1.0000 |
| R8 vs R9 | 1.0000 |

### ⚠️ Important Interpretation Note

All images show unanimous reader decisions (is_abnormal). Fleiss' Kappa = 1.0 reflects perfect agreement but is trivially determined by dataset construction: the subset was selected such that all 3 assigned readers agreed on the abnormal/normal status of each image. This result does not provide discriminative information for model training.

---

## 9. Class-Level Kappa Feasibility and Result

**Feasibility: PARTIALLY_FEASIBLE**

**Methodology:** Mean pairwise Cohen's Kappa (unweighted) for binary class
presence (1 = reader labeled this class in this image, 0 = reader did not),
computed within the dominant reader triplet (R8/R9/R10, 4,222 images).
**This is NOT Fleiss' Kappa** — Fleiss' Kappa cannot be applied here because
different images are annotated by different reader triplets, making a unified
rating matrix ill-defined. Mean pairwise Cohen's Kappa within the dominant
triplet is the methodologically valid approximation.
**Weighted Kappa was NOT used** — labels are binary presence/absence, not ordinal.

**Limitation:** Only the dominant triplet (R8/R9/R10, 4,222 images) yields
statistically stable estimates. Other triplets cover < 50 images each and are
excluded from computation.

### Mean Pairwise Cohen's Kappa per Class (dominant triplet R8/R9/R10, unweighted)

| class_id | class_name | mean_pairwise_kappa |
|----------|------------|---------------------|
| 12 | Pneumothorax | 0.7519 |
| 3 | Cardiomegaly | 0.6954 |
| 10 | Pleural effusion | 0.6694 |
| 13 | Pulmonary fibrosis | 0.6140 |
| 0 | Aortic enlargement | 0.6064 |
| 8 | Nodule/Mass | 0.4860 |
| 5 | ILD | 0.4722 |
| 6 | Infiltration | 0.4130 |
| 1 | Atelectasis | 0.3900 |
| 11 | Pleural thickening | 0.3605 |
| 2 | Calcification | 0.3473 |
| 7 | Lung Opacity | 0.3344 |
| 4 | Consolidation | 0.3241 |
| 9 | Other lesion | 0.2969 |

> Kappa interpretation: ≥0.8 Almost Perfect · ≥0.6 Substantial · ≥0.4 Moderate ·
> ≥0.2 Fair · ≥0.0 Slight · <0.0 Poor

---

## 10. BBox-Level Agreement Feasibility

**Feasibility: NOT_FEASIBLE_IN_PHASE_1D**

| Check | Result |
|-------|--------|
| bbox_coordinates_exist | True |
| reader_identity_exists | True |
| images_with_multi_reader_bbox | 4394 / 4394 abnormal images |
| bbox_matching_method_defined | NOT_DEFINED_IN_PHASE_1D |
| risk_of_misleading_kappa | HIGH |
| feasibility_verdict | NOT_FEASIBLE_IN_PHASE_1D |

**Reason not computed:**
BBox-level inter-reader agreement is not feasible in PHASE 1D because an
IoU-based cross-reader bbox matching protocol has not been defined yet.
PHASE 2B will only create framework-independent and format-agnostic canonical
bbox tables; it is **NOT** the phase for computing bbox-level inter-reader
agreement. BBox-level agreement may be revisited later in **PHASE 6** after
canonical annotations and a valid cross-reader bbox matching protocol are
available. Computing bbox Kappa without a defined matching method would risk
producing misleading values by treating unmatched bboxes as false disagreements.

---

## 11. Clear Explanation of Non-Feasible Levels

- **Image-level Kappa = 1.0:** Computed but trivially so. By dataset construction,
  all 3 readers agreed on every image's normal/abnormal status. This reflects
  the subset selection criterion, not a meaningful measure of annotation quality.
- **BBox-level Kappa:** Not computed. Canonical bbox matching method is not yet
  defined. Will be revisited after PHASE 2B.

---

## 12. Limitations

1. Image-level Kappa = 1.0 is a dataset-construction artifact, not an independent
   quality signal.
2. Class-level Kappa is restricted to the dominant triplet (R8/R9/R10); results
   for 14 other triplets are omitted due to small sample sizes.
3. No Finding (class_id=14) is excluded from class-level Kappa; its "presence"
   is already captured by the image-level analysis.
4. BBox-level agreement is NOT computed in PHASE 1D because an IoU-based
   cross-reader bbox matching protocol has not been defined yet. PHASE 2B
   creates canonical bbox tables only. BBox-level agreement may be revisited
   in PHASE 6.
5. 78 duplicate bbox candidate pairs (from PHASE 1B) may slightly inflate apparent
   single-reader over-annotation; these are documented limitations, not removed.
6. No annotation or bbox was modified in this phase.

---

## 13. Gate Decision

**Gate: `PASS`**

- ✅ Metadata files inspected.
- ✅ Reader column `rad_id` identified (17 readers, 3 per image).
- ✅ Multi-reader availability confirmed.
- ✅ Image-level Kappa: FEASIBLE → COMPUTED (Fleiss' κ = 1.0).
- ✅ Class-level Kappa: PARTIALLY_FEASIBLE → COMPUTED for dominant triplet.
- ✅ BBox-level: NOT_FEASIBLE_IN_PHASE_1D → clearly documented. PHASE 2B creates
  canonical bbox tables only; bbox agreement deferred to PHASE 6.
- ✅ No annotation or bbox modified.
- ✅ No target format or framework assumed.
- ✅ No split or training performed.
- ✅ CLAUDE.md updated.
- ✅ README.md updated.
