# PHASE 2B — Canonical Detection Annotation Schema: Validation Report

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
| Total selected images | 4,894 |
| Total canonical image rows | 4,894 |
| Total canonical bbox rows | 36,096 |
| Total valid abnormal boxes | 36,096 |
| Total invalid bbox rows | 0 |
| Total No Finding images | 500 |
| Total abnormal images | 4,394 |
| Total mixed/conflict images | 0 |
| Total unknown label images | 0 |
| Number of detection classes | 14 |
| Number of image-level classes | 1 |
| **Overall status** | **PASS** |

---

## 3. Class Mapping

| canonical_id | original_id | class_name | is_detection | is_image_level | has_bbox |
|:---:|:---:|---|:---:|:---:|:---:|
| 0 | 0 | Aortic enlargement | True | False | True |
| 1 | 1 | Atelectasis | True | False | True |
| 2 | 2 | Calcification | True | False | True |
| 3 | 3 | Cardiomegaly | True | False | True |
| 4 | 4 | Consolidation | True | False | True |
| 5 | 5 | ILD | True | False | True |
| 6 | 6 | Infiltration | True | False | True |
| 7 | 7 | Lung Opacity | True | False | True |
| 8 | 8 | Nodule/Mass | True | False | True |
| 9 | 9 | Other lesion | True | False | True |
| 10 | 10 | Pleural effusion | True | False | True |
| 11 | 11 | Pleural thickening | True | False | True |
| 12 | 12 | Pneumothorax | True | False | True |
| 13 | 13 | Pulmonary fibrosis | True | False | True |
|  | 14 | No finding | False | True | False |

> `canonical_class_id` is null for No Finding — it is not a detection object class.

---

## 4. Validation Checks

| Check | Expected | Observed | Status |
|-------|---------|---------|--------|
| image_count_consistency | 4894 | 4894 | PASS |
| class_id_name_consistency | 15 unique pairs | 15 canonical pairs | PASS |
| no_finding_excluded_from_bbox_table | 0 | 0 | PASS |
| invalid_bbox_count | 0 | 0 | PASS |
| source_row_id_completeness | 0 | 0 | PASS |
| rad_id_completeness | 0 | 0 | PASS |
| canonical_class_id_completeness_in_bbox | 0 | 0 | PASS |
| mixed_or_conflict_images | 0 | 0 | PASS |
| unknown_label_type_images | 0 | 0 | PASS |

---

## 5. Warnings

No warnings.

> NOTE: WARNING items are *flagged only*. PHASE 2B does not delete, merge, or
> resolve them. They are handed to later phases for decision.

---

## 6. Label Type Distribution

| Label Type | Count |
|-----------|-------|
| abnormal | 4,394 |
| no_finding | 500 |
| mixed_or_conflict | 0 |
| unknown | 0 |

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
