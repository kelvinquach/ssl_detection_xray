# PHASE 2B — Canonical Detection Annotation Schema Definition

**Workflow Lock: WF-SSL-XRAY-DET-V1**
**Format: Framework-independent internal canonical format**

---

## Schema Overview

- Phase: PHASE 2B
- Dataset: VinBigData Chest X-ray Abnormalities Detection (metadata-only working subset)
- Total selected images: 4894
- Detection classes: 14 (class_id 0–13)
- Image-level classes: 1 (No finding, class_id 14)

---

## 1. Class Mapping Schema (`canonical_class_mapping.csv`)

| canonical_class_id | original_class_id | class_name | is_detection_class |
|:------------------:|:-----------------:|-----------|:-----------------:|
| 0 | 0 | Aortic enlargement | Yes |
| 1 | 1 | Atelectasis | Yes |
| 2 | 2 | Calcification | Yes |
| 3 | 3 | Cardiomegaly | Yes |
| 4 | 4 | Consolidation | Yes |
| 5 | 5 | ILD | Yes |
| 6 | 6 | Infiltration | Yes |
| 7 | 7 | Lung Opacity | Yes |
| 8 | 8 | Nodule/Mass | Yes |
| 9 | 9 | Other lesion | Yes |
| 10 | 10 | Pleural effusion | Yes |
| 11 | 11 | Pleural thickening | Yes |
| 12 | 12 | Pneumothorax | Yes |
| 13 | 13 | Pulmonary fibrosis | Yes |
|  | 14 | No finding | No |

Rules:
- `canonical_class_id` = `original_class_id` for detection classes (0–13).
- No finding (`original_class_id = 14`) has `canonical_class_id = null` and is
  an image-level negative label, NOT a detection object class.

---

## 2. BBox Annotation Schema (`canonical_bbox_annotations.csv`)

| Column | Type | Description |
|--------|------|-------------|
| image_id | string | Unique image identifier |
| source_row_id | int | Original row index in `subset_train_annotations.csv` (traceability) |
| rad_id | string | Radiologist / reader identifier |
| original_class_id | int | Class ID from original annotation (0–13) |
| canonical_class_id | int | Canonical class ID (same as original for detection classes) |
| class_name | string | Human-readable class name |
| x_min, y_min, x_max, y_max | float | BBox corners in pixel space (origin = top-left) |
| bbox_width | float | x_max - x_min |
| bbox_height | float | y_max - y_min |
| bbox_area | float | bbox_width * bbox_height |
| is_valid_bbox | bool | True if all coordinates are valid |
| bbox_status | string | "valid" or an error description |

Rules: No Finding excluded; multi-reader rows preserved; no merging, no
consensus boxes, no automatic duplicate removal.

---

## 3. Image Annotation Schema (`canonical_image_annotations.csv`)

| Column | Type | Description |
|--------|------|-------------|
| image_id | string | Unique image identifier |
| num_raw_rows | int | Total annotation rows (all readers, all classes) |
| num_bbox_rows | int | Rows in bbox table for this image |
| num_abnormal_boxes | int | Count of valid abnormal bbox rows |
| num_detection_classes_present | int | Unique detection classes present |
| has_abnormality | bool | True if >=1 valid abnormal bbox |
| has_no_finding_label | bool | True if any reader labeled No Finding |
| image_label_type | string | abnormal / no_finding / mixed_or_conflict / unknown |
| reader_count | int | Unique readers who annotated this image |

`image_label_type` rules:
- `abnormal`: has_abnormality AND NOT has_no_finding_label
- `no_finding`: has_no_finding_label AND NOT has_abnormality
- `mixed_or_conflict`: both True (requires later review)
- `unknown`: both False (requires later review)

---

## 4. JSONL Schema (`canonical_detection_annotations.jsonl`)

One JSON object per line. Internal canonical format — NOT COCO, NOT YOLO.

```json
{
  "image_id": "<string>",
  "image_label_type": "abnormal | no_finding | mixed_or_conflict | unknown",
  "boxes": [
    {
      "class_id": <int>,
      "class_name": "<string>",
      "bbox_xyxy": [x_min, y_min, x_max, y_max],
      "rad_id": "<reader_id>",
      "source_row_id": <int>
    }
  ]
}
```

- For No Finding images: `"boxes": []`.
- `bbox_xyxy` is [x_min, y_min, x_max, y_max] in pixel space (origin = top-left).
- Converted to a framework-specific format only in PHASE 2C.

---

## 5. Design Decisions

1. No consensus boxes — each reader's bbox is an independent row.
2. No automatic duplicate removal — flagged only.
3. No framework assumption — no COCO categories, YOLO labels, or VOC XML.
4. Traceability — every row links back via `source_row_id` and keeps `rad_id`.
5. No Finding handling — negative image-level label only; absent from bbox table.
