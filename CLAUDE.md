# CLAUDE.md

# Stage 1: Dataset Preparation for Chest X-ray Object Detection

## 1. Current Project Stage

This project is currently in **Stage 1: Dataset Preparation**.

Do not implement model training yet.
Do not implement semi-supervised learning yet.
Do not implement pseudo-labeling yet.
Do not create teacher-student code yet.
Do not create detection training scripts yet.

The current goal is only to inspect, validate, and summarize the dataset and annotation files.

---

## 2. Research Context

This project studies **semi-supervised object detection for chest X-ray abnormality detection**.

However, the current stage is not semi-supervised training yet.

At this stage, the only task is to prepare the dataset correctly for a future object detection pipeline.

The dataset must be checked carefully before any training pipeline is created.

---

## 3. Dataset

The starting dataset is:

RSNA Pneumonia Detection Challenge

The dataset may contain:

* DICOM images
* PNG images
* JPG/JPEG images
* CSV annotation files

Do not assume the file format.
Inspect the actual files inside the local project folder first.

---

## 4. Object Detection Context

This project is an **object detection** project.

The future model will detect abnormal/pneumonia-like regions using bounding boxes.

At this stage, the main focus is to verify whether the dataset and annotations are valid for object detection.

Do not convert the project into image classification.

Do not convert the project into semantic segmentation.

---

## 5. Important Dataset Definitions

### Positive image

A positive image is an image explicitly labeled as abnormal or pneumonia-positive.

In RSNA-style annotations, this is usually indicated by:

Target = 1

A positive image should have at least one valid bounding box.

### Negative image

A negative image is an image explicitly labeled as negative.

In RSNA-style annotations, this is usually indicated by:

Target = 0

A negative image should not have a valid bounding box.

### Bounding box

A bounding box usually contains:

* x
* y
* width
* height

However, the actual annotation file must be inspected before assuming the exact column names.

---

## 6. What Claude Must Do First

Before creating or modifying any code, Claude must:

1. Inspect the current project folder structure.
2. Identify where image files are stored.
3. Identify where annotation files are stored.
4. Report the detected file types.
5. Report the likely dataset structure.
6. Ask for clarification only if the dataset path or annotation file cannot be found.

Do not write code before inspecting the available files.

---

## 7. Dataset Inspection Requirements

Claude should inspect and summarize:

### Folder structure

* project root
* data folder
* raw data folder
* image folder
* annotation folder
* output folder

### Image files

Check:

* total number of image files
* image file extensions
* whether images are DICOM, PNG, JPG, or mixed
* whether image files can be opened
* unreadable or corrupted files
* image width and height
* abnormal image sizes
* duplicate image IDs if detectable

### Annotation files

Check:

* available CSV files
* annotation file names
* annotation columns
* number of annotation rows
* possible image ID column
* possible class/target column
* possible bounding-box columns

---

## 8. Annotation Validation Requirements

The annotation validation must check whether the annotation file has columns equivalent to:

* patientId or image_id
* x
* y
* width
* height
* Target or class label

Claude must not assume the column names.
Claude should inspect the CSV header and then map the detected columns.

The script should check:

1. Missing image IDs
2. Duplicate annotation rows
3. Missing target/class labels
4. Missing bbox coordinates
5. Non-numeric bbox coordinates
6. Positive rows without bbox coordinates
7. Negative rows with unexpected bbox coordinates
8. Multiple boxes for the same image
9. Number of positive images
10. Number of negative images
11. Number of true bounding boxes

---

## 9. Bounding-box Validation Requirements

For each valid positive bounding box, check:

1. x >= 0
2. y >= 0
3. width > 0
4. height > 0
5. x + width <= image_width
6. y + height <= image_height

Also report:

* very small bounding boxes
* very large bounding boxes
* unusual aspect ratios
* boxes outside image boundary
* boxes with missing values
* boxes with invalid width or height

Do not remove invalid rows automatically.
Only report them at this stage.

---

## 10. Image and Annotation Matching

Claude must check consistency between images and annotations.

Check:

1. Images referenced in annotation file but missing from disk
2. Images present on disk but missing from annotation file
3. Annotation rows with image IDs that cannot be matched to image files
4. Duplicate image IDs
5. Number of unique images in annotations
6. Number of image files on disk

If DICOM files use `.dcm` but annotation IDs do not include extension, match by stem name.

Example:

patientId = abc123
image file = abc123.dcm

These should be treated as matching.

---

## 11. Required Output Files

The dataset validation step should save summary outputs.

Create these folders if they do not exist:

outputs/metrics
outputs/logs
outputs/reports

Save the following files:

outputs/metrics/dataset_summary.csv
outputs/metrics/image_validation_summary.csv
outputs/metrics/annotation_summary.csv
outputs/metrics/bbox_validation_summary.csv
outputs/reports/dataset_validation_report.md
outputs/logs/dataset_validation.log

The report should be readable and explain the findings clearly.

---

## 12. Required Summary Statistics

The final dataset validation report must include:

### Dataset summary

* total image files
* image formats found
* total annotation files found
* selected annotation file
* total annotation rows
* total unique image IDs in annotation
* total matched images
* total missing images
* total extra images without annotation

### Image summary

* readable images
* unreadable images
* minimum image width
* maximum image width
* minimum image height
* maximum image height
* most common image size
* abnormal image size count

### Annotation summary

* detected image ID column
* detected target/class column
* detected bbox columns
* positive image count
* negative image count
* positive annotation row count
* negative annotation row count
* total true bounding boxes
* images with multiple bounding boxes

### Bounding-box summary

* valid bbox count
* missing bbox count
* invalid width count
* invalid height count
* negative coordinate count
* out-of-bound bbox count
* suspiciously small bbox count
* suspiciously large bbox count
* unusual aspect ratio count

---

## 13. Important Rules

Claude must follow these rules:

1. Do not train any model.
2. Do not create any train/val/test split yet unless explicitly requested later.
3. Do not create labeled/unlabeled split yet.
4. Do not generate pseudo-labels.
5. Do not convert annotations to YOLO or COCO yet unless explicitly requested later.
6. Do not delete or modify raw data.
7. Do not remove invalid annotations automatically.
8. Do not invent dataset statistics.
9. Always inspect files before assuming anything.
10. Save every summary to outputs.
11. Explain the plan before creating or editing code.

---

## 14. First Coding Task

The first coding task is:

Create a script:

scripts/01_validate_dataset.py

This script should:

1. Inspect the dataset folder.
2. Detect image files.
3. Detect annotation CSV files.
4. Read the selected annotation file.
5. Validate annotation columns.
6. Validate image readability.
7. Validate bounding boxes.
8. Match image files with annotation rows.
9. Save CSV summaries.
10. Save a Markdown validation report.

Before writing the script, explain:

* which folders will be inspected,
* which files will be created,
* what checks will be performed,
* what output files will be saved.

---

## 15. Expected Next Stage

After this dataset validation stage is completed, the next stage will be:

Stage 2: Create fixed train/validation/test split.

Do not move to Stage 2 until Stage 1 output files are created and reviewed.
