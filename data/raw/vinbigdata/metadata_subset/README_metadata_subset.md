
# VinBigData Chest X-ray Metadata-only Subset

This subset contains metadata only. No DICOM images are copied.

## Dataset source

Kaggle competition:
VinBigData Chest X-ray Abnormalities Detection

## Subset rule

- Keep all abnormal images: class_id != 14
- Randomly sample 500 normal / No Finding images: class_id == 14
- Random seed: 42

## Main files

1. selected_image_ids.csv
   - image_id
   - subset_type: abnormal or normal

2. subset_train_annotations.csv
   - annotation rows from original train.csv for selected images only

3. subset_summary.csv
   - summary statistics of the subset

4. subset_class_distribution.csv
   - class distribution in the subset

5. positive_normal_summary.csv
   - abnormal vs normal image count

6. annotation_validation_summary_basic.csv
   - basic annotation validation from CSV only

7. missing_images.csv
   - selected image ids whose DICOM file was not found in the Kaggle input folder

## Important note

DICOM images are not stored in this subset. To access images during training or validation,
use image_id to read files from:

/kaggle/input/competitions/vinbigdata-chest-xray-abnormalities-detection/train/<image_id>.dicom

This design avoids Kaggle working directory storage errors and keeps the subset reproducible.
