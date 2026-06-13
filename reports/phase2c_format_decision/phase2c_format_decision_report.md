# PHASE 2C — Framework-specific Annotation Format Decision

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
| Total images | 4894 |
| Abnormal images (with bbox) | 4394 |
| No Finding images (negatives) | 500 |
| Total bbox rows | 36096 |
| Detection classes | 14 |
| Image-level classes | 1 |
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

| Criterion | COCO JSON | Internal JSONL (PHASE 2B) | Pascal VOC XML | YOLO TXT |
|---|---|---|---|---|
| compatibility_with_canonical_schema | 5 | 5 | 3 | 3 |
| compatibility_with_ssl_object_detection_research | 5 | 3 | 2 | 3 |
| compatibility_with_supervised_detection_baseline | 5 | 2 | 3 | 4 |
| ease_of_validation | 4 | 5 | 2 | 3 |
| kaggle_colab_vastai_practicality | 5 | 3 | 2 | 4 |
| medical_xray_workflow_suitability | 5 | 5 | 3 | 2 |
| multi_class_detection_support | 5 | 5 | 4 | 5 |
| negative_image_support | 5 | 5 | 3 | 2 |
| paper_thesis_suitability | 5 | 4 | 3 | 3 |
| rad_id_source_row_traceability_risk | 5 | 5 | 3 | 1 |
| reproducibility | 5 | 5 | 3 | 3 |

**Total scores (max 55):**

- COCO JSON: **54 / 55**
- Internal JSONL (PHASE 2B): **47 / 55**
- YOLO TXT: **33 / 55**
- Pascal VOC XML: **31 / 55**

---

## 5. Conversion Requirements Summary

| Format | Requirement | Risk | Traceability | Blocker |
|--------|------------|------|-------------|---------|
| COCO JSON | image_metadata | LOW — standard JSON field extension | YES — rad_id and source_row_id as extra annotation fields | NO |
| COCO JSON | category_list | LOW — trivial remapping (0-indexed → 1-indexed or keep 0-indexed) | YES | NO |
| COCO JSON | bbox_format | VERY LOW — single arithmetic step | YES | NO |
| COCO JSON | negative_images | VERY LOW | YES | NO |
| YOLO TXT | coordinate_normalisation | HIGH — metadata loss; requires sidecar solution | NO — rad_id and source_row_id lost in plain txt | PARTIAL — solvable with sidecar JSON |
| YOLO TXT | negative_images | MEDIUM — convention varies by YOLOv5 vs v8 vs v9 | N/A | NO, but requires careful documentation |
| YOLO TXT | class_names_file | LOW | YES (at class level) | NO |
| Pascal VOC XML | per_image_xml_generation | MEDIUM — 4,894 files to manage; no universal reader | PARTIAL — possible via custom attributes but non-standard | NO, but high engineering overhead |
| Pascal VOC XML | negative_images | LOW — standard empty annotation | YES | NO |

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

**Condition:** A sidecar JSON file (`{image_id}_meta.json`) must be maintained
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
