# SSL Detection X-ray

This repository contains the research workflow for semi-supervised object detection on chest X-ray images using the VinBigData Chest X-ray Abnormalities Detection dataset.

## Research direction

Vietnamese title:

**Nghiên cứu học bán giám sát cho dò tìm bất thường trên ảnh X-quang phổi với detector dựa trên attention/Transformer trong điều kiện hạn chế nhãn**

## Current stage

The project is currently at the dataset selection and metadata-only subset preparation stage.

Current dataset:

- VinBigData Chest X-ray Abnormalities Detection
- Metadata-only subset
- 500 Normal / No Finding images
- All abnormal images
- No DICOM images are committed to Git

## Current data structure

```text
data/raw/vinbigdata
├── metadata_subset
│   ├── selected_image_ids.csv
│   ├── subset_train_annotations.csv
│   ├── abnormal_image_ids.csv
│   ├── normal_image_ids_500.csv
│   ├── subset_summary.csv
│   ├── subset_class_distribution.csv
│   ├── positive_normal_summary.csv
│   └── README_metadata_subset.md
│
└── original
    └── train.csv

## Current Research Workflow

**Workflow Lock Keyword:** WF-SSL-XRAY-DET-V1

**Main research direction:**
Semi-supervised object detection for chest X-ray abnormality detection.

**Current phase:**
PHASE 1D — Multi-Reader Annotation Agreement Feasibility & Kappa Analysis.

**Reason PHASE 1D was added:**
To address the mentor's requirement to consider Kappa / multi-reader agreement
before proceeding to annotation engineering.

### Completed phases

| Phase | Commit |
|-------|--------|
| PHASE 1A — Dataset Overview Report | 827c5d1 |
| PHASE 1B — Annotation Quality Analysis | eac73ba |
| PHASE 1C — Dataset Scope Decision | 1491550 |
| PHASE 2A — Image/DICOM Accessibility & BBox Boundary Validation | cb8b0d8 |

### Next phase after PHASE 1D

PHASE 2B — Canonical Detection Annotation Schema & Class Mapping.

### Important constraints

- PHASE 2B remains format-agnostic and framework-independent.
- COCO/YOLO/framework decisions are deferred to PHASE 2C.
- ViT/attention is handled in PHASE 2C, PHASE 4B, PHASE 5, and PHASE 6.
- No split, conversion, framework selection, model training, or pseudo-label
  generation is allowed before the corresponding phase.

## PHASE 2B Note

PHASE 2B creates a canonical, framework-independent and format-agnostic
detection annotation schema.

- Outputs are stored under `reports/phase2b_canonical_schema/`.
- Does NOT convert to COCO/YOLO/Pascal VOC.
- Does NOT create train/val/test or labeled/unlabeled splits.
- Does NOT train models or generate pseudo-labels.
- Multi-reader annotations are preserved (no consensus/merging).
- PHASE 2C will decide the framework-specific annotation format later.
