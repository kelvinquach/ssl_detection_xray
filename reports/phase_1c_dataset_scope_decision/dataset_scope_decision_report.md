# PHASE 1C — Dataset Scope Decision

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
| Total images | 4,894 |
| Abnormal images | 4,394 |
| Normal images | 500 |
| Total annotation rows | 37,596 |
| Total abnormal bbox rows | 36,096 |
| Number of abnormal classes | 14 |
| Mean bbox per abnormal image | 8 |
| Max bbox per abnormal image | 57 |

**Evidence from PHASE 1B:**

| Metric | Value |
|--------|-------|
| Total No Finding rows | 1,500 |
| Invalid bbox rows (any type) | 0 |
| Normal images with bbox | 0 |
| Abnormal images without valid bbox | 0 |
| Duplicate bbox candidate pairs (IoU ≥ 0.95) | 78 |
| Image-boundary check | not_available (image dimensions not in metadata) |
| Overall annotation quality status | **PASS** |

---

## 3. Dataset Scope Decision

**Decision D01 — dataset_scope:** `ACCEPTED`

> Continue with the current metadata-only VinBigData subset as the controlled working
> subset for the next dataset engineering phase.

This decision is based on:
- Subset size matches config expectations (4,894 images).
- PHASE 1B annotation quality is PASS with 0 invalid bbox rows.
- No evidence requires expanding, contracting, or replacing the dataset at this stage.

> **Note:** This is a working subset decision, not a claim that this subset represents
> the full dataset or is the final experimental configuration.

---

## 4. Subset Policy

**Decision D02 — subset_policy:** `ACCEPTED`

> Keep 4,394 abnormal images + 500 normal images
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

> Keep all 14 abnormal classes. No class is removed in PHASE 1C.

| class_id | class_name | num_bbox | num_images | invalid_bbox | dup_candidates | decision |
|----------|------------|---------|-----------|-------------|----------------|----------|
| 0 | Aortic enlargement | 7,162 | 3,067 | 0 | 7 | **keep** |
| 3 | Cardiomegaly | 5,427 | 2,300 | 0 | 47 | **keep** |
| 11 | Pleural thickening | 4,842 | 1,981 | 0 | 2 | **keep** |
| 13 | Pulmonary fibrosis | 4,655 | 1,617 | 0 | 3 | **keep** |
| 8 | Nodule/Mass | 2,580 | 826 | 0 | 1 | **keep** |
| 7 | Lung Opacity | 2,483 | 1,322 | 0 | 2 | **keep** |
| 10 | Pleural effusion | 2,476 | 1,032 | 0 | 3 | **keep** |
| 9 | Other lesion | 2,203 | 1,134 | 0 | 0 | **keep** |
| 6 | Infiltration | 1,247 | 613 | 0 | 1 | **keep** |
| 5 | ILD | 1,000 | 386 | 0 | 3 | **keep** |
| 2 | Calcification | 960 | 452 | 0 | 2 | **keep** |
| 4 | Consolidation | 556 | 353 | 0 | 0 | **keep** |
| 1 | Atelectasis | 279 | 186 | 0 | 0 | **keep** |
| 12 | Pneumothorax | 226 | 96 | 0 | 7 | **keep** |

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
- All 36,096 abnormal bbox rows are carried forward.
- Pixel-level validation (image boundary check) is assigned to
  **PHASE 2A — Image Availability & DICOM Metadata Validation** — see Section 9.

---

## 8. Duplicate Candidate Policy

**Decision D06 — duplicate_candidate_policy:** `ACCEPTED`

> 78 duplicate/near-duplicate bbox candidate pairs
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
| L1 | Metadata-only subset — no image pixels or DICOM files have been read. | MILESTONE 0 / PHASE 1A / PHASE 1B / PHASE 1C | medium | Handle in PHASE 2A — Image Availability & DICOM Metadata Val... |
| L2 | Image-boundary check not available due to missing image width/height i... | PHASE 1B | medium | Perform in PHASE 2A — Image Availability & DICOM Metadata Va... |
| L3 | Bbox counts are annotation-level rows, not unique lesion counts. Multi... | PHASE 1A / PHASE 1B | low | Always distinguish annotation rows from unique image/lesion ... |
| L4 | Significant class imbalance across 14 abnormal classes (Aortic enlarge... | PHASE 1A | medium | Analyze in PHASE 2C — Class Imbalance & Sampling Strategy De... |
| L5 | 78 duplicate/near-duplicate bbox candidate pairs are metadata-level fl... | PHASE 1B | low | Record as metadata-level flag. Evaluate impact in annotation... |
| L6 | Current subset uses 500 sampled normal images, not all available norma... | MILESTONE 0 | low | Current 500-sample policy is deliberate for controlled exper... |
| L7 | No Kappa / inter-rater agreement analysis has been performed yet. | PHASE 1B (explicitly excluded from scope) | low | Kappa analysis is deferred to a dedicated annotation quality... |

---

## 11. Next Phase Readiness Checklist

Full details in `reports/phase_1c_dataset_scope_decision/next_phase_readiness_checklist.csv`.

| Check | Item | Status | Notes |
|-------|------|--------|-------|
| C01 | PHASE 1A completed and committed | PASS |  |
| C02 | PHASE 1B completed and committed | PASS |  |
| C03 | Dataset subset size confirmed | PASS | Expected: 4,894 |
| C04 | Abnormal/normal counts confirmed | PASS | Expected: 4,394 abnormal + 500 normal |
| C05 | 14 abnormal classes confirmed | PASS | Expected: 14 |
| C06 | No invalid bbox rows detected at metadata level | PASS |  |
| C07 | Normal images contain no bbox rows | PASS |  |
| C08 | All abnormal images have at least one valid bbox | PASS |  |
| C09 | Duplicate candidates recorded but not removed | PASS | Decision D06: recorded as metadata-level flag only |
| C10 | Image-boundary limitation recorded | PASS | Decision D07: DEFERRED — assigned to PHASE 2A — Image Availability & DICOM Metadata Validation |
| C11 | No split created | PASS |  |
| C12 | No leakage check performed | NOT_APPLICABLE |  |
| C13 | No annotation conversion performed | PASS |  |
| C14 | No training performed | PASS |  |
| C15 | PHASE 1C decision report generated | PASS |  |
| C16 | Ready for next phase | PASS | Next phase (dataset engineering) may begin after PHASE 1C confirmation. |

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
- Images: 4,394 abnormal + 500 normal = 4,894 total
- Abnormal classes: 14 (all retained)
- Bbox rows: 36,096 valid abnormal bbox rows
- No invalid bbox, no split, no conversion, no training

**The next phase (dataset engineering) may begin only after explicit user confirmation
that PHASE 1C is complete.**
