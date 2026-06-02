# CLAUDE.md

# Stage 1: Dataset Preparation for Chest X-ray Object Detection

## Current Stage

This project is currently in **Stage 1: Dataset Preparation**.

Do not implement model training yet.
Do not create train/validation/test split yet.
Do not create labeled/unlabeled split yet.
Do not implement semi-supervised learning yet.
Do not generate pseudo-labels yet.
Do not convert annotations to YOLO, COCO, or Pascal VOC yet.

The current goal is only to inspect, validate, and summarize the dataset and annotation files.

---

## Research Context

This project studies **semi-supervised object detection for chest X-ray abnormality detection**.

However, the current stage is only dataset preparation.

At this stage, the task is to check whether the dataset is valid for a future object detection pipeline.

This is an object detection project, not image classification and not semantic segmentation.

---

## Dataset

The current dataset is:

**VinBigData chest X-ray object detection subset**

The raw dataset is located at:

```text
data/raw/vinbigdata/
```

Current dataset structure:

```text
data/raw/vinbigdata/
├── annotations/
│   ├── train.csv
│   ├── part_001_annotations.csv
│   ├── part_001_image_ids.csv
│   ├── ...
│   ├── part_016_annotations.csv
│   └── part_016_image_ids.csv
│
└── images/
    ├── *.dicom
```

The image files are DICOM files.

The annotation files are CSV files.

Do not assume annotation columns before inspecting the actual CSV files.

---

## Important Rule

Do not assume this dataset is RSNA.

Do not assume:

* image ID column name,
* class label column name,
* bounding-box column names,
* bounding-box coordinate format,
* positive/negative label definition,
* train/validation/test split,
* labeled/unlabeled split.

Always inspect the actual files before making assumptions.

---

## Object Detection Context

This project is for object detection.

The future model should detect abnormal regions on chest X-ray images using bounding boxes.

Do not convert this project into:

* image classification,
* semantic segmentation,
* report generation,
* image-level disease prediction only.

The current stage is only dataset validation.

---

## What Claude Must Do First

Before creating or modifying code, inspect the project folder and report:

1. existing dataset folders,
2. number of DICOM images,
3. annotation CSV files found,
4. columns of `train.csv`,
5. columns of `part_001_annotations.csv`,
6. columns of `part_001_image_ids.csv`,
7. whether image filenames match annotation image IDs,
8. whether the dataset is ready for Stage 1 validation,
9. what checks should be included in `scripts/01_validate_dataset.py`.

Do not modify files before reporting the plan.

Do not create code until the dataset structure and annotation columns are inspected.

---

## Dataset Inspection Requirements

The inspection should answer:

### Image files

* How many DICOM images exist?
* Are image files directly under `data/raw/vinbigdata/images/`?
* What is the image ID format?
* Do filenames include extensions?
* Are there duplicate image IDs?
* Are there unreadable or corrupted DICOM files?

### Annotation files

* Which CSV files exist?
* What are the columns in `train.csv`?
* What are the columns in each `part_xxx_annotations.csv` file?
* What are the columns in each `part_xxx_image_ids.csv` file?
* Which annotation file should be treated as the main annotation source?
* Are `part_xxx_annotations.csv` files subsets of `train.csv` or separate derived files?

### Matching between images and annotations

* Are image IDs in annotation files matched with DICOM filenames?
* Are there annotation records without corresponding image files?
* Are there image files without annotation records?
* Are image IDs duplicated?

---

## Bounding-box Validation Requirements

After inspecting the annotation columns, the validation script should check bounding boxes.

The checks should include:

1. missing image IDs,
2. missing class labels,
3. missing bounding-box coordinates,
4. non-numeric bounding-box coordinates,
5. invalid bounding-box width or height,
6. negative coordinates,
7. bounding boxes outside image boundaries,
8. multiple bounding boxes for the same image,
9. images with no bounding boxes,
10. annotation rows that do not match any image file.

Do not remove invalid annotations automatically.

Only report issues at this stage.

---

## Required Output Files

When the validation script is created and executed later, it should save:

```text
outputs/metrics/dataset_summary.csv
outputs/metrics/image_validation_summary.csv
outputs/metrics/annotation_summary.csv
outputs/metrics/bbox_validation_summary.csv
outputs/reports/dataset_validation_report.md
outputs/logs/dataset_validation.log
```

All outputs should be generated under `outputs/`.

Do not save generated reports inside `data/raw/`.

---

## Rules

1. Do not train any model.
2. Do not create train/validation/test split yet.
3. Do not create labeled/unlabeled split yet.
4. Do not implement semi-supervised learning yet.
5. Do not generate pseudo-labels.
6. Do not convert annotations to YOLO, COCO, or Pascal VOC yet.
7. Do not delete or modify raw data.
8. Do not invent dataset statistics.
9. Always inspect files before assuming anything.
10. Save outputs as CSV, Markdown, or log files.
11. Explain planned changes before editing code.
12. Keep the project focused on object detection.

---

## First Task

The first task is:

Inspect the local project folder and report the actual VinBigData dataset structure.

Do not create code yet.

After the structure and annotation columns are confirmed, the first coding task will be:

```text
scripts/01_validate_dataset.py
```

This script should only validate the dataset and save reports.

It must not train models, create splits, generate pseudo-labels, or convert annotation formats.
