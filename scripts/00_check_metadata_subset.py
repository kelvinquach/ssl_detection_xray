from pathlib import Path
import pandas as pd
import yaml


def load_config(config_path: str = "configs/dataset.yaml") -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_file_exists(path: Path, name: str) -> bool:
    if path.exists():
        print(f"[OK] {name}: {path}")
        return True
    else:
        print(f"[MISSING] {name}: {path}")
        return False


def main():
    config = load_config()

    print("=" * 80)
    print("CHECK METADATA-ONLY SUBSET")
    print("=" * 80)

    metadata_files = config["metadata_files"]

    required_files = {
        "selected_image_ids": Path(metadata_files["selected_image_ids"]),
        "subset_annotations": Path(metadata_files["subset_annotations"]),
        "abnormal_image_ids": Path(metadata_files["abnormal_image_ids"]),
        "normal_image_ids": Path(metadata_files["normal_image_ids"]),
        "subset_summary": Path(metadata_files["subset_summary"]),
        "class_distribution": Path(metadata_files["class_distribution"]),
        "positive_normal_summary": Path(metadata_files["positive_normal_summary"]),
    }

    print("\n1. Checking required metadata files")
    print("-" * 80)

    all_ok = True
    for name, path in required_files.items():
        exists = check_file_exists(path, name)
        all_ok = all_ok and exists

    if not all_ok:
        raise FileNotFoundError("Some required metadata files are missing.")

    print("\n2. Loading metadata files")
    print("-" * 80)

    selected_df = pd.read_csv(required_files["selected_image_ids"])
    ann_df = pd.read_csv(required_files["subset_annotations"])
    abnormal_df = pd.read_csv(required_files["abnormal_image_ids"])
    normal_df = pd.read_csv(required_files["normal_image_ids"])

    print("selected_image_ids.csv:", selected_df.shape)
    print("subset_train_annotations.csv:", ann_df.shape)
    print("abnormal_image_ids.csv:", abnormal_df.shape)
    print("normal_image_ids_500.csv:", normal_df.shape)

    print("\n3. Checking required columns")
    print("-" * 80)

    required_selected_cols = {"image_id", "subset_type"}
    required_ann_cols = {"image_id", "class_id", "class_name", "x_min", "y_min", "x_max", "y_max"}

    missing_selected_cols = required_selected_cols - set(selected_df.columns)
    missing_ann_cols = required_ann_cols - set(ann_df.columns)

    if missing_selected_cols:
        raise ValueError(f"Missing columns in selected_image_ids.csv: {missing_selected_cols}")

    if missing_ann_cols:
        raise ValueError(f"Missing columns in subset_train_annotations.csv: {missing_ann_cols}")

    print("[OK] selected_image_ids.csv columns")
    print("[OK] subset_train_annotations.csv columns")

    print("\n4. Checking subset counts")
    print("-" * 80)

    total_images = selected_df["image_id"].nunique()
    abnormal_images = selected_df[selected_df["subset_type"] == "abnormal"]["image_id"].nunique()
    normal_images = selected_df[selected_df["subset_type"] == "normal"]["image_id"].nunique()

    print("Total selected images:", total_images)
    print("Abnormal images:", abnormal_images)
    print("Normal images:", normal_images)

    expected_total = config["subset"]["total_images_expected"]
    expected_normal = config["subset"]["normal_images"]

    if total_images != expected_total:
        print(f"[WARNING] Expected total images = {expected_total}, but found {total_images}")
    else:
        print("[OK] Total image count matches expected value")

    if normal_images != expected_normal:
        print(f"[WARNING] Expected normal images = {expected_normal}, but found {normal_images}")
    else:
        print("[OK] Normal image count matches expected value")

    print("\n5. Checking annotation consistency")
    print("-" * 80)

    selected_ids = set(selected_df["image_id"].unique())
    ann_ids = set(ann_df["image_id"].unique())

    ann_not_in_selected = ann_ids - selected_ids
    selected_not_in_ann = selected_ids - ann_ids

    print("Annotation image_ids not in selected subset:", len(ann_not_in_selected))
    print("Selected image_ids without annotation:", len(selected_not_in_ann))

    if len(ann_not_in_selected) == 0:
        print("[OK] All annotation image_ids belong to selected subset")
    else:
        print("[WARNING] Some annotations do not belong to selected subset")

    if len(selected_not_in_ann) == 0:
        print("[OK] All selected images have annotation rows")
    else:
        print("[WARNING] Some selected images do not have annotation rows")

    print("\n6. Basic bbox validation")
    print("-" * 80)

    no_finding_class_id = config["subset"]["normal_definition"]["class_id"]
    bbox_df = ann_df[ann_df["class_id"] != no_finding_class_id].copy()

    invalid_bbox = bbox_df[
        (bbox_df["x_min"].isna()) |
        (bbox_df["y_min"].isna()) |
        (bbox_df["x_max"].isna()) |
        (bbox_df["y_max"].isna()) |
        (bbox_df["x_max"] <= bbox_df["x_min"]) |
        (bbox_df["y_max"] <= bbox_df["y_min"]) |
        (bbox_df["x_min"] < 0) |
        (bbox_df["y_min"] < 0)
    ]

    print("Total bbox rows excluding No finding:", len(bbox_df))
    print("Invalid bbox rows basic check:", len(invalid_bbox))

    if len(invalid_bbox) == 0:
        print("[OK] Basic bbox validation passed")
    else:
        print("[WARNING] Invalid bbox rows found")

    print("\n7. Class distribution")
    print("-" * 80)

    class_dist = (
        ann_df.groupby(["class_id", "class_name"])
        .agg(
            annotation_rows=("image_id", "count"),
            num_images=("image_id", "nunique")
        )
        .reset_index()
        .sort_values("annotation_rows", ascending=False)
    )

    print(class_dist)

    print("\n" + "=" * 80)
    print("METADATA SUBSET CHECK COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()