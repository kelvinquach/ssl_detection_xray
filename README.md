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