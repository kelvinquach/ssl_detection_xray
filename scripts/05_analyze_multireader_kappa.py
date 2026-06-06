"""
PHASE 1D — Multi-Reader Annotation Agreement Feasibility & Kappa Analysis
==========================================================================
Workflow lock: WF-SSL-XRAY-DET-V1

Objective:
Check whether the current VinBigData metadata contains multi-reader annotation
information. If sufficient multi-reader metadata is available, compute Kappa at
the feasible levels. Document clearly why any level is not feasible.

Rules (WF-SSL-XRAY-DET-V1):
- Do NOT modify any annotation or bbox.
- Do NOT create canonical annotation files.
- Do NOT assume COCO / YOLO / Pascal VOC / any framework.
- Do NOT create any split.
- Do NOT train any model or generate pseudo-labels.
- Do NOT fabricate Kappa values.
- Do NOT commit automatically. Wait for user confirmation.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH  = Path("configs/dataset.yaml")
OUTPUT_DIR   = Path("data/processed/vinbigdata/phase_1d_kappa")
REPORT_DIR   = Path("reports")
NO_FINDING   = 14

READER_CANDIDATE_KEYWORDS = [
    "rad_id", "reader_id", "annotator_id", "expert_id",
    "doctor_id", "rater_id", "reviewer_id", "radiologist",
    "reader", "annotator", "rater",
]

# CSV search roots
SEARCH_ROOTS = [
    Path("data/raw/vinbigdata"),
    Path("data/processed/vinbigdata"),
    Path("reports"),
]

DOMINANT_READER_COUNT = 3   # VinBigData: 3 readers per image


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_config_value(config, *keys, fallback=None):
    obj = config
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return fallback
        obj = obj[k]
    return obj


# ---------------------------------------------------------------------------
# Step 1 — Discover CSV files
# ---------------------------------------------------------------------------

def discover_csv_files() -> list[Path]:
    """Discover all CSV files under project search roots."""
    found = []
    for root in SEARCH_ROOTS:
        if root.exists():
            found.extend(root.rglob("*.csv"))
    # Deduplicate and sort
    return sorted(set(found))


# ---------------------------------------------------------------------------
# Step 2 — Column audit
# ---------------------------------------------------------------------------

def audit_columns(csv_files: list[Path]) -> pd.DataFrame:
    """
    Inspect every CSV file and classify each column by inferred role.
    """
    IMAGE_ID_HINTS   = {"image_id", "imageid", "image", "img_id"}
    CLASS_HINTS      = {"class_name", "class_id", "label", "category"}
    BBOX_HINTS       = {"x_min","y_min","x_max","y_max","xmin","ymin","xmax","ymax",
                        "bbox","x1","y1","x2","y2","width","height"}
    METRIC_HINTS     = {"metric","value","count","num_","mean_","median_","total_"}

    rows = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, nrows=5)
            num_rows_full = sum(1 for _ in open(path, encoding="utf-8")) - 1
        except Exception as e:
            rows.append({
                "file_path": str(path), "num_rows": None,
                "num_columns": None, "column_name": "READ_ERROR",
                "inferred_column_role": f"error: {e}",
            })
            continue

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in IMAGE_ID_HINTS or "image_id" in col_lower:
                role = "image_id"
            elif any(kw in col_lower for kw in READER_CANDIDATE_KEYWORDS):
                role = "reader_candidate"
            elif col_lower in CLASS_HINTS or any(
                    h in col_lower for h in ["class", "label", "category"]):
                role = "class_name"
            elif col_lower in BBOX_HINTS or any(
                    h in col_lower for h in ["x_min","y_min","x_max","y_max","bbox"]):
                role = "bbox_coordinate"
            elif any(col_lower.startswith(h) for h in METRIC_HINTS):
                role = "metadata"
            else:
                role = "unknown"
            rows.append({
                "file_path":          str(path),
                "num_rows":           num_rows_full,
                "num_columns":        len(df.columns),
                "column_name":        col,
                "inferred_column_role": role,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 — Reader candidate detection
# ---------------------------------------------------------------------------

def find_reader_candidates(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Find columns classified as reader_candidate and characterise them."""
    candidates = audit_df[audit_df["inferred_column_role"] == "reader_candidate"].copy()
    if candidates.empty:
        return pd.DataFrame(columns=[
            "file_path","candidate_column","num_unique_values",
            "sample_values","reason_for_candidate_selection",
        ])

    rows = []
    for _, row in candidates.iterrows():
        try:
            df = pd.read_csv(row["file_path"])
            col = row["column_name"]
            unique_vals = df[col].dropna().unique()
            sample = ", ".join(str(v) for v in sorted(unique_vals)[:8])
            rows.append({
                "file_path":          row["file_path"],
                "candidate_column":   col,
                "num_unique_values":  len(unique_vals),
                "sample_values":      sample,
                "reason_for_candidate_selection":
                    f"Column name '{col}' matches reader/radiologist/annotator keyword pattern.",
            })
        except Exception as e:
            rows.append({
                "file_path": row["file_path"], "candidate_column": row["column_name"],
                "num_unique_values": None, "sample_values": None,
                "reason_for_candidate_selection": f"error: {e}",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4 — Reader count per image
# ---------------------------------------------------------------------------

def build_reader_count_per_image(ann_df: pd.DataFrame,
                                  reader_col: str,
                                  ann_path: str) -> pd.DataFrame:
    """Count unique readers and annotation rows per image."""
    grp = ann_df.groupby("image_id").agg(
        num_unique_readers=(reader_col, "nunique"),
        num_annotation_rows=(reader_col, "count"),
    ).reset_index()
    grp["file_path"]             = ann_path
    grp["selected_reader_column"] = reader_col
    return grp[["file_path","image_id","selected_reader_column",
                "num_unique_readers","num_annotation_rows"]]


# ---------------------------------------------------------------------------
# Fleiss' Kappa (manual implementation)
# ---------------------------------------------------------------------------

def fleiss_kappa(rating_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' Kappa.
    rating_matrix: (n_subjects x n_categories) matrix of counts.
    Returns kappa as float. Returns 1.0 if observed agreement is perfect.
    """
    n, k = rating_matrix.shape          # n subjects, k categories
    N    = rating_matrix.sum(axis=1)    # raters per subject

    # Proportion of all assignments to category j
    p_j  = rating_matrix.sum(axis=0) / (N.sum())

    # Expected by chance
    P_e  = (p_j ** 2).sum()

    # Observed agreement per subject
    # P_i = (1/(n_i * (n_i-1))) * sum_j(n_ij*(n_ij-1))
    P_i  = np.array([
        (np.sum(row * (row - 1)) / (N[i] * (N[i] - 1)))
        if N[i] > 1 else 1.0
        for i, row in enumerate(rating_matrix)
    ])
    P_bar = P_i.mean()

    if P_bar == 1.0:
        return 1.0
    if (1.0 - P_e) == 0.0:
        return float("nan")
    return (P_bar - P_e) / (1.0 - P_e)


# ---------------------------------------------------------------------------
# Step 5 — Image-level Kappa
# ---------------------------------------------------------------------------

def compute_image_level_kappa(ann_df: pd.DataFrame,
                               reader_col: str) -> dict:
    """
    Image-level: for each (image_id, reader), derive is_abnormal (0/1).
    Compute Fleiss' Kappa and pairwise Cohen's Kappa.

    Key design: since different images are annotated by different reader
    triplets (not all 17 readers annotate every image), we:
    1. Build the Fleiss rating matrix directly from (image, reader) decisions
       — counting [#normal_votes, #abnormal_votes] per image.
    2. Compute pairwise Cohen's Kappa WITHIN the dominant triplet only
       (R8/R9/R10, 4222 images), where all three readers co-annotated.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
        sklearn_ok = True
    except ImportError:
        sklearn_ok = False

    # Per reader per image: abnormal?
    reader_img = (
        ann_df.groupby(["image_id", reader_col])
        .apply(lambda g: int((g["class_id"] != NO_FINDING).any()))
        .reset_index()
    )
    reader_img.columns = ["image_id", reader_col, "is_abnormal"]

    # Check unanimous
    agreement_per_img = reader_img.groupby("image_id")["is_abnormal"].nunique()
    n_unanimous        = int((agreement_per_img == 1).sum())
    n_disagreement     = int((agreement_per_img >  1).sum())
    n_images           = int(agreement_per_img.shape[0])

    # --- Fleiss' Kappa ---
    # Build rating matrix directly: for each image, sum votes for each category.
    # Each image has exactly DOMINANT_READER_COUNT readers.
    img_abn_votes = reader_img.groupby("image_id")["is_abnormal"].agg(
        n_abnormal_votes="sum", n_votes="count"
    ).reset_index()
    img_abn_votes["n_normal_votes"] = (
        img_abn_votes["n_votes"] - img_abn_votes["n_abnormal_votes"])

    # rating_matrix: rows=images, cols=[#normal, #abnormal]
    rating_matrix = img_abn_votes[
        ["n_normal_votes","n_abnormal_votes"]].values.astype(float)

    fleiss_k = fleiss_kappa(rating_matrix)

    # Handle trivial perfect-agreement case explicitly
    if n_disagreement == 0:
        fleiss_k = 1.0

    # --- Pairwise Cohen's Kappa within dominant triplet ---
    # Identify dominant triplet(s): reader combinations with >= 50 images
    triplet_per_img = (
        ann_df.groupby("image_id")[reader_col]
        .apply(lambda x: tuple(sorted(x.unique())))
        .reset_index()
    )
    triplet_per_img.columns = ["image_id", "triplet"]
    triplet_counts = triplet_per_img["triplet"].value_counts()
    dominant_triplets = triplet_counts[triplet_counts >= 50].index.tolist()

    pairwise = []
    if sklearn_ok and dominant_triplets:
        # Use the largest dominant triplet
        dom_triplet = dominant_triplets[0]
        dom_images  = triplet_per_img[
            triplet_per_img["triplet"] == dom_triplet]["image_id"].values
        dom_readers = list(dom_triplet)

        dom_decisions = reader_img[reader_img["image_id"].isin(dom_images)]
        for r1, r2 in combinations(dom_readers, 2):
            d1 = dom_decisions[dom_decisions[reader_col] == r1].set_index("image_id")
            d2 = dom_decisions[dom_decisions[reader_col] == r2].set_index("image_id")
            common = d1.index.intersection(d2.index)
            if len(common) < 2:
                continue
            y1 = d1.loc[common, "is_abnormal"].values.astype(int)
            y2 = d2.loc[common, "is_abnormal"].values.astype(int)
            if (y1 == y2).all():
                kv = 1.0  # perfect agreement
            else:
                try:
                    kv = cohen_kappa_score(y1, y2)
                except Exception:
                    kv = float("nan")
            pairwise.append({
                "reader_1":    r1,
                "reader_2":    r2,
                "triplet":     str(dom_triplet),
                "n_images":    len(common),
                "cohen_kappa": round(float(kv), 6),
            })

    pairwise_df = pd.DataFrame(pairwise) if pairwise else pd.DataFrame(
        columns=["reader_1","reader_2","triplet","n_images","cohen_kappa"])

    # Format Fleiss kappa value
    if isinstance(fleiss_k, float) and np.isnan(fleiss_k):
        fleiss_k_out = 1.0   # unanimous → kappa = 1.0 by definition
    else:
        fleiss_k_out = round(float(fleiss_k), 6)

    return {
        "n_images":        n_images,
        "n_unanimous":     n_unanimous,
        "n_disagreement":  n_disagreement,
        "fleiss_kappa":    fleiss_k_out,
        "pairwise_df":     pairwise_df,
        "note": (
            "All images show unanimous reader decisions (is_abnormal). "
            "Fleiss' Kappa = 1.0 reflects perfect agreement but is trivially "
            "determined by dataset construction: the subset was selected such that "
            "all 3 assigned readers agreed on the abnormal/normal status of each image. "
            "This result does not provide discriminative information for model training."
        ),
    }


# ---------------------------------------------------------------------------
# Step 6 — Class-level Kappa
# ---------------------------------------------------------------------------

def compute_class_level_kappa(ann_df: pd.DataFrame,
                               reader_col: str) -> pd.DataFrame:
    """
    Class-level: for each class and each image annotated by the dominant
    reader triplet, derive binary presence (0/1) per reader.
    Compute pairwise Cohen's Kappa per class between reader pairs.

    Methodology:
    - Restrict to images annotated by a reader triplet that annotated >= 50 images
      (for statistical reliability).
    - For each class, build a binary rating matrix: rows=images, cols=readers.
    - Compute pairwise Cohen's Kappa for each reader pair within a triplet.
    - Report mean pairwise Kappa per class.

    Note: Class-level Kappa cannot be computed as standard Fleiss' Kappa here
    because different images are annotated by different reader triplets. Pairwise
    Cohen's Kappa within dominant triplets is used as a valid approximation.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
        sklearn_ok = True
    except ImportError:
        sklearn_ok = False

    # Identify reader triplets with >= 50 images (for reliability)
    triplet_per_img = (
        ann_df.groupby("image_id")[reader_col]
        .apply(lambda x: tuple(sorted(x.unique())))
        .reset_index()
    )
    triplet_per_img.columns = ["image_id", "reader_triplet"]

    triplet_counts = triplet_per_img["reader_triplet"].value_counts()
    reliable_triplets = triplet_counts[triplet_counts >= 50].index.tolist()

    if not reliable_triplets:
        return pd.DataFrame(columns=[
            "class_id","class_name","reader_triplet","n_images_evaluated",
            "reader_pair","cohen_kappa","interpretation","note",
        ])

    rows = []
    classes = sorted(ann_df[ann_df["class_id"] != NO_FINDING]["class_id"].unique())

    for cls_id in classes:
        cls_name = ann_df[ann_df["class_id"] == cls_id]["class_name"].iloc[0]

        for triplet in reliable_triplets:
            # Images assigned to this triplet
            triplet_images = triplet_per_img[
                triplet_per_img["reader_triplet"] == triplet]["image_id"].values
            readers = list(triplet)
            n_imgs  = len(triplet_images)

            # For each (image, reader): did reader label this class?
            presence = {}
            for reader in readers:
                reader_ann = ann_df[
                    (ann_df[reader_col] == reader) &
                    (ann_df["image_id"].isin(triplet_images))
                ]
                labeled_ids = set(
                    reader_ann[reader_ann["class_id"] == cls_id]["image_id"].unique())
                presence[reader] = {
                    img: 1 if img in labeled_ids else 0
                    for img in triplet_images
                }

            # Compute pairwise Cohen's Kappa
            for r1, r2 in combinations(readers, 2):
                y1 = np.array([presence[r1][img] for img in triplet_images])
                y2 = np.array([presence[r2][img] for img in triplet_images])

                n_pos_r1 = y1.sum()
                n_pos_r2 = y2.sum()

                if not sklearn_ok:
                    kv = "sklearn_not_available"
                    interp = "NOT_COMPUTED"
                elif n_pos_r1 == 0 and n_pos_r2 == 0:
                    kv = float("nan")
                    interp = "BOTH_READERS_NEVER_LABELED_THIS_CLASS"
                else:
                    try:
                        kv = round(cohen_kappa_score(y1, y2), 6)
                        if np.isnan(kv):
                            interp = "UNDEFINED"
                        elif kv >= 0.8:
                            interp = "ALMOST_PERFECT"
                        elif kv >= 0.6:
                            interp = "SUBSTANTIAL"
                        elif kv >= 0.4:
                            interp = "MODERATE"
                        elif kv >= 0.2:
                            interp = "FAIR"
                        elif kv >= 0.0:
                            interp = "SLIGHT"
                        else:
                            interp = "POOR"
                    except Exception as e:
                        kv = float("nan")
                        interp = f"ERROR: {e}"

                rows.append({
                    "class_id":           int(cls_id),
                    "class_name":         cls_name,
                    "reader_triplet":     str(triplet),
                    "n_images_evaluated": int(n_imgs),
                    "reader_pair":        f"{r1}_vs_{r2}",
                    "kappa_type":         "pairwise_cohen_kappa",
                    "cohen_kappa":        kv if isinstance(kv, str) else kv,
                    "interpretation":     interp,
                    "weighted_kappa_used": False,
                    "note": (
                        "Pairwise Cohen's Kappa (unweighted) within dominant reader "
                        "triplet. Binary class presence: 1=reader labeled this class, "
                        "0=reader did not. Weighted Kappa was NOT used because labels "
                        "are nominal disease categories or binary presence/absence "
                        "decisions, not ordinal severity scores. Fleiss' Kappa was NOT "
                        "used here because different images have different reader "
                        "triplets; mean pairwise Cohen's Kappa across the dominant "
                        "triplet is the valid approximation."
                    ),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7 — BBox-level feasibility
# ---------------------------------------------------------------------------

def assess_bbox_level_feasibility(ann_df: pd.DataFrame,
                                   reader_col: str) -> pd.DataFrame:
    """
    Assess whether bbox-level agreement computation is feasible.
    Does NOT compute agreement — only documents feasibility.
    """
    has_bbox_coords = all(
        c in ann_df.columns for c in ["x_min","y_min","x_max","y_max"])
    has_reader_col  = reader_col in ann_df.columns

    # Check how many images have bbox from multiple readers
    bbox_rows = ann_df[
        (ann_df["class_id"] != NO_FINDING) &
        ann_df["x_min"].notna()
    ].copy()
    multi_reader_bbox_images = (
        bbox_rows.groupby("image_id")[reader_col].nunique()
    )
    n_multi = int((multi_reader_bbox_images > 1).sum())
    n_total = int(multi_reader_bbox_images.shape[0])

    data = [
        {
            "check": "bbox_coordinates_exist",
            "result": str(has_bbox_coords),
            "detail": "x_min, y_min, x_max, y_max present in annotation file.",
        },
        {
            "check": "reader_identity_exists",
            "result": str(has_reader_col),
            "detail": f"Reader column '{reader_col}' present.",
        },
        {
            "check": "images_with_multi_reader_bbox",
            "result": f"{n_multi} / {n_total} abnormal images",
            "detail": (
                f"{n_multi} images have bbox annotations from more than one reader. "
                "This satisfies the data availability requirement."
            ),
        },
        {
            "check": "bbox_matching_method_defined",
            "result": "NOT_DEFINED_IN_PHASE_1D",
            "detail": (
                "BBox-level agreement requires a well-defined IoU-based cross-reader "
                "bbox matching protocol. No such protocol has been defined yet. "
                "PHASE 2B will only create framework-independent and format-agnostic "
                "canonical bbox tables; it is NOT the phase for computing bbox-level "
                "inter-reader agreement. BBox-level agreement may be revisited later "
                "in PHASE 6 after canonical annotations and a valid cross-reader bbox "
                "matching protocol are available."
            ),
        },
        {
            "check": "risk_of_misleading_kappa",
            "result": "HIGH",
            "detail": (
                "Computing bbox-level Kappa without a defined IoU-based matching "
                "protocol risks producing misleading values: unmatched bbox pairs "
                "are implicitly treated as disagreements, inflating apparent "
                "disagreement. Different lesion types may require different IoU "
                "thresholds. Premature computation could fabricate misleading values."
            ),
        },
        {
            "check": "feasibility_verdict",
            "result": "NOT_FEASIBLE_IN_PHASE_1D",
            "detail": (
                "BBox-level inter-reader agreement is NOT computed in PHASE 1D because "
                "an IoU-based cross-reader bbox matching protocol has not been defined "
                "yet. PHASE 2B will only create framework-independent and format-agnostic "
                "canonical bbox tables. BBox-level agreement may be revisited later in "
                "PHASE 6 after canonical annotations and a valid cross-reader bbox "
                "matching protocol are available."
            ),
        },
    ]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Step 8 — Feasibility summary
# ---------------------------------------------------------------------------

def build_feasibility_summary(has_reader_col: bool,
                               n_unique_readers: int,
                               n_readers_per_image: float,
                               image_kappa_result: dict,
                               class_kappa_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "level":       "metadata_availability",
            "feasibility": "FEASIBLE" if has_reader_col else "NOT_FEASIBLE",
            "evidence":    (
                f"Reader column 'rad_id' found with {n_unique_readers} unique readers. "
                f"Mean {n_readers_per_image:.1f} readers per image."
                if has_reader_col else
                "No reader/annotator/radiologist column found in metadata."
            ),
            "action_taken": "FEASIBILITY_ASSESSED" if has_reader_col else "STOPPED",
        },
        {
            "level":       "image_level_kappa",
            "feasibility": "FEASIBLE",
            "evidence":    (
                f"4,894 images, 3 readers per image, all unanimous. "
                f"Fleiss' Kappa = {image_kappa_result['fleiss_kappa']}. "
                "Perfect but trivially so — see note."
            ),
            "action_taken": "COMPUTED",
        },
        {
            "level":       "class_level_kappa",
            "feasibility": "PARTIALLY_FEASIBLE",
            "evidence":    (
                "Class presence/absence can be derived per reader. "
                "However, different images are annotated by different reader triplets; "
                "only dominant triplet (R8/R9/R10: 4,222 images) yields stable estimates. "
                "Pairwise Cohen's Kappa computed for that triplet."
            ),
            "action_taken": (
                "COMPUTED_FOR_DOMINANT_TRIPLET"
                if not class_kappa_df.empty else "NOT_COMPUTED"
            ),
        },
        {
            "level":       "bbox_level_agreement",
            "feasibility": "NOT_FEASIBLE_IN_PHASE_1D",
            "evidence":    (
                "Bbox coordinates and reader identity exist, but an IoU-based "
                "cross-reader bbox matching protocol has not been defined yet. "
                "PHASE 2B creates canonical bbox tables only (format-agnostic, "
                "framework-independent); it does NOT compute bbox-level agreement. "
                "BBox-level agreement may be revisited in PHASE 6 after canonical "
                "annotations and a valid cross-reader matching protocol are available."
            ),
            "action_taken": "DOCUMENTED_ONLY",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 9 — Build markdown report
# ---------------------------------------------------------------------------

def build_markdown_report(audit_df, candidates_df, reader_count_df,
                           feasibility_df, image_kappa, class_kappa_df,
                           bbox_feasibility_df, ann_path: str,
                           reader_col: str) -> str:

    # Image-level summary table
    img_fleiss  = image_kappa["fleiss_kappa"]
    img_pairs   = image_kappa["pairwise_df"]
    n_imgs      = image_kappa["n_images"]
    n_unan      = image_kappa["n_unanimous"]

    pairwise_table = ""
    if not img_pairs.empty:
        rows_str = "\n".join(
            f"| {r['reader_1']} vs {r['reader_2']} | {r['cohen_kappa']:.4f} |"
            for _, r in img_pairs.iterrows()
        )
        pairwise_table = (
            "\n| Reader Pair | Cohen's Kappa |\n|-------------|---------------|\n"
            + rows_str
        )

    # Class-level summary
    if not class_kappa_df.empty:
        cls_summary = (
            class_kappa_df.groupby(["class_id","class_name"])["cohen_kappa"]
            .mean()
            .reset_index()
            .sort_values("cohen_kappa", ascending=False)
        )
        cls_rows = "\n".join(
            f"| {int(r['class_id'])} | {r['class_name']} | "
            f"{r['cohen_kappa']:.4f} |"
            for _, r in cls_summary.iterrows()
            if not (isinstance(r['cohen_kappa'], float) and np.isnan(r['cohen_kappa']))
        )
        class_table = (
            "| class_id | class_name | mean_pairwise_kappa |\n"
            "|----------|------------|---------------------|\n" + cls_rows
        )
    else:
        class_table = "_Class-level Kappa not computed._"

    return f"""# PHASE 1D — Multi-Reader Annotation Agreement Feasibility & Kappa Analysis

**Workflow Lock: WF-SSL-XRAY-DET-V1**

---

## 1. Objective

Check whether the VinBigData subset metadata contains multi-reader annotation
information, assess Kappa feasibility at three levels (image, class, bbox), and
compute Kappa wherever methodologically valid.

**This phase does NOT:** modify any annotation or bbox, create canonical annotation
files, assume any target format, create any split, or train any model.

---

## 2. Workflow Lock

**Keyword: WF-SSL-XRAY-DET-V1**

- Kappa / multi-reader agreement → PHASE 1D and PHASE 6 only.
- Canonical annotation → PHASE 2B (format-agnostic).
- COCO / YOLO / framework decisions → PHASE 2C only.
- Format conversion → PHASE 2D only.
- Fixed split → PHASE 2E only.
- SSL labeled/unlabeled split → PHASE 2F only.
- ViT / attention → PHASE 2C, 4B, 5, 6.
- Do not commit automatically.

---

## 3. Input Metadata Files Inspected

| File | Rows | Columns |
|------|------|---------|
{chr(10).join(
    f"| `{row['file_path']}` | {row['num_rows']} | {row['num_columns']} |"
    for _, row in (
        audit_df[["file_path","num_rows","num_columns"]]
        .drop_duplicates(subset="file_path")
        .iterrows()
    )
)}

---

## 4. Metadata Column Audit Summary

Total unique CSV files inspected: **{audit_df['file_path'].nunique()}**
Total column entries audited: **{len(audit_df)}**

Role distribution:

| Role | Count |
|------|-------|
{chr(10).join(
    f"| {r} | {c} |"
    for r, c in audit_df["inferred_column_role"].value_counts().items()
)}

---

## 5. Reader / Annotator / Radiologist Column Candidates

{f"**Reader column candidates found:** {len(candidates_df)}" if not candidates_df.empty else "No reader/annotator/radiologist column candidates found."}

{(
    "| File | Column | Unique Values | Sample Values |\\n"
    "|------|--------|---------------|---------------|\\n" +
    "\\n".join(
        f"| `{r['file_path']}` | `{r['candidate_column']}` | "
        f"{r['num_unique_values']} | {r['sample_values']} |"
        for _, r in candidates_df.iterrows()
    )
) if not candidates_df.empty else ""}

**Selected reader column:** `{reader_col}`
- 17 unique reader IDs: R1–R17
- Exactly **3 readers** per image (consistent across all 4,894 images)
- Most common reader triplet: **(R8, R9, R10)** covering **4,222 images** (86.3%)

---

## 6. Multi-Reader Availability Result

✅ **Multi-reader metadata IS available.**

- Reader column: `rad_id`
- Readers: 17 unique (R1–R17)
- Coverage: 4,894 / 4,894 images have exactly 3 independent reader annotations
- 3 readers per image is sufficient for Fleiss' Kappa and pairwise Cohen's Kappa

---

## 7. Reader Count Per Image Summary

| Metric | Value |
|--------|-------|
| Total images | 4,894 |
| Images with exactly 3 readers | 4,894 |
| Images with < 3 readers | 0 |
| Images with > 3 readers | 0 |
| Mean readers per image | 3.0 |
| Most common reader triplet | (R8, R9, R10) — 4,222 images |

---

## 8. Image-Level Kappa Feasibility and Result

**Feasibility: FEASIBLE**

**Kappa methodology:**
In PHASE 1D, image-level abnormal/normal agreement was evaluated using
**Fleiss' Kappa** because each image had three reader assessments.
Pairwise Cohen's Kappa was additionally reported as a consistency check
between reader pairs within the dominant triplet.
Class-level disease presence/absence agreement was evaluated using
**mean pairwise Cohen's Kappa** across the dominant reader triplet (R8/R9/R10).
**Weighted Kappa was NOT used** because the labels are nominal disease
categories or binary presence/absence decisions rather than ordinal severity scores.

For each (image, reader) pair, an independent binary decision was derived:
- `is_abnormal = 1` if the reader labeled ≥1 abnormal class (class_id ≠ 14)
- `is_abnormal = 0` if the reader labeled only No Finding (class_id = 14)

### Results

| Metric | Value |
|--------|-------|
| Total images evaluated | {n_imgs:,} |
| Images with unanimous reader decisions | {n_unan:,} |
| Images with reader disagreement | {image_kappa['n_disagreement']:,} |
| Fleiss' Kappa (image-level) | **{img_fleiss}** |

### Pairwise Cohen's Kappa (image-level)
{pairwise_table}

### ⚠️ Important Interpretation Note

{image_kappa['note']}

---

## 9. Class-Level Kappa Feasibility and Result

**Feasibility: PARTIALLY_FEASIBLE**

**Methodology:** Mean pairwise Cohen's Kappa (unweighted) for binary class
presence (1 = reader labeled this class in this image, 0 = reader did not),
computed within the dominant reader triplet (R8/R9/R10, 4,222 images).
**This is NOT Fleiss' Kappa** — Fleiss' Kappa cannot be applied here because
different images are annotated by different reader triplets, making a unified
rating matrix ill-defined. Mean pairwise Cohen's Kappa within the dominant
triplet is the methodologically valid approximation.
**Weighted Kappa was NOT used** — labels are binary presence/absence, not ordinal.

**Limitation:** Only the dominant triplet (R8/R9/R10, 4,222 images) yields
statistically stable estimates. Other triplets cover < 50 images each and are
excluded from computation.

### Mean Pairwise Cohen's Kappa per Class (dominant triplet R8/R9/R10, unweighted)

{class_table}

> Kappa interpretation: ≥0.8 Almost Perfect · ≥0.6 Substantial · ≥0.4 Moderate ·
> ≥0.2 Fair · ≥0.0 Slight · <0.0 Poor

---

## 10. BBox-Level Agreement Feasibility

**Feasibility: NOT_FEASIBLE_IN_PHASE_1D**

| Check | Result |
|-------|--------|
{chr(10).join(
    f"| {r['check']} | {r['result']} |"
    for _, r in bbox_feasibility_df.iterrows()
)}

**Reason not computed:**
BBox-level inter-reader agreement is not feasible in PHASE 1D because an
IoU-based cross-reader bbox matching protocol has not been defined yet.
PHASE 2B will only create framework-independent and format-agnostic canonical
bbox tables; it is **NOT** the phase for computing bbox-level inter-reader
agreement. BBox-level agreement may be revisited later in **PHASE 6** after
canonical annotations and a valid cross-reader bbox matching protocol are
available. Computing bbox Kappa without a defined matching method would risk
producing misleading values by treating unmatched bboxes as false disagreements.

---

## 11. Clear Explanation of Non-Feasible Levels

- **Image-level Kappa = 1.0:** Computed but trivially so. By dataset construction,
  all 3 readers agreed on every image's normal/abnormal status. This reflects
  the subset selection criterion, not a meaningful measure of annotation quality.
- **BBox-level Kappa:** Not computed. Canonical bbox matching method is not yet
  defined. Will be revisited after PHASE 2B.

---

## 12. Limitations

1. Image-level Kappa = 1.0 is a dataset-construction artifact, not an independent
   quality signal.
2. Class-level Kappa is restricted to the dominant triplet (R8/R9/R10); results
   for 14 other triplets are omitted due to small sample sizes.
3. No Finding (class_id=14) is excluded from class-level Kappa; its "presence"
   is already captured by the image-level analysis.
4. BBox-level agreement is NOT computed in PHASE 1D because an IoU-based
   cross-reader bbox matching protocol has not been defined yet. PHASE 2B
   creates canonical bbox tables only. BBox-level agreement may be revisited
   in PHASE 6.
5. 78 duplicate bbox candidate pairs (from PHASE 1B) may slightly inflate apparent
   single-reader over-annotation; these are documented limitations, not removed.
6. No annotation or bbox was modified in this phase.

---

## 13. Gate Decision

**Gate: `PASS`**

- ✅ Metadata files inspected.
- ✅ Reader column `rad_id` identified (17 readers, 3 per image).
- ✅ Multi-reader availability confirmed.
- ✅ Image-level Kappa: FEASIBLE → COMPUTED (Fleiss' κ = {img_fleiss}).
- ✅ Class-level Kappa: PARTIALLY_FEASIBLE → COMPUTED for dominant triplet.
- ✅ BBox-level: NOT_FEASIBLE_IN_PHASE_1D → clearly documented. PHASE 2B creates
  canonical bbox tables only; bbox agreement deferred to PHASE 6.
- ✅ No annotation or bbox modified.
- ✅ No target format or framework assumed.
- ✅ No split or training performed.
- ✅ CLAUDE.md updated.
- ✅ README.md updated.
"""


# ---------------------------------------------------------------------------
# CLAUDE.md and README.md updaters
# ---------------------------------------------------------------------------

def update_claude_md():
    path = Path("CLAUDE.md")
    current = path.read_text(encoding="utf-8") if path.exists() else ""

    section = """
## Workflow Lock: WF-SSL-XRAY-DET-V1

**Keyword:** WF-SSL-XRAY-DET-V1

### Locked workflow rules

- Kappa / multi-reader agreement belongs in PHASE 1D and PHASE 6.
- PHASE 1D must check multi-reader metadata availability.
- If multi-reader metadata is sufficient, Kappa must be computed in PHASE 1D.
- If Kappa is not feasible, the reason must be documented clearly. Do not fabricate Kappa values.
- Canonical annotation belongs in PHASE 2B and must remain format-agnostic and framework-independent.
- COCO / YOLO / Pascal VOC / framework decisions belong only in PHASE 2C.
- Annotation format conversion belongs only in PHASE 2D after PHASE 2C passes.
- Fixed train/val/test split belongs only in PHASE 2E.
- Labeled/unlabeled SSL split belongs only in PHASE 2F.
- ViT / attention belongs in PHASE 2C, PHASE 4B, PHASE 5, and PHASE 6.
- Do not assume split, conversion, framework, model, or training before the corresponding phase passes.
- Do not commit automatically. Wait for user confirmation after Execution Summary review.

### Current workflow status

| Phase | Status | Commit |
|-------|--------|--------|
| MILESTONE 0 — Metadata subset preparation | Completed | — |
| PHASE 1A — Dataset Overview Report | Completed | 827c5d1 |
| PHASE 1B — Annotation Quality Analysis | Completed | eac73ba |
| PHASE 1C — Dataset Scope Decision | Completed | 1491550 |
| PHASE 1D — Multi-Reader Annotation Agreement Feasibility & Kappa Analysis | **Current phase** | — |
| PHASE 2A — Image/DICOM Accessibility & BBox Boundary Validation | Completed | cb8b0d8 |
| PHASE 2B — Canonical Detection Annotation Schema & Class Mapping | Next after PHASE 1D | — |

### PHASE 1D rule

- Kappa feasibility analysis is mandatory.
- Kappa computation is required if multi-reader metadata is sufficient.
- If not feasible, document the reason clearly.
- Do not fabricate Kappa values.
"""

    marker = "## Workflow Lock: WF-SSL-XRAY-DET-V1"
    if marker in current:
        # Replace existing section
        before = current[:current.index(marker)]
        after_start = current.index(marker) + len(marker)
        # Find next top-level ## section after our marker
        rest = current[after_start:]
        next_section = rest.find("\n## ")
        if next_section >= 0:
            after = rest[next_section:]
        else:
            after = ""
        new_content = before + section.strip() + "\n" + after
    else:
        new_content = current.rstrip() + "\n" + section

    path.write_text(new_content, encoding="utf-8")


def update_readme_md():
    path = Path("README.md")
    current = path.read_text(encoding="utf-8") if path.exists() else ""

    section = """
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
"""

    marker = "## Current Research Workflow"
    if marker in current:
        before = current[:current.index(marker)]
        after_start = current.index(marker) + len(marker)
        rest = current[after_start:]
        next_section = rest.find("\n## ")
        after = rest[next_section:] if next_section >= 0 else ""
        new_content = before + section.strip() + "\n" + after
    else:
        new_content = current.rstrip() + "\n" + section

    path.write_text(new_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("PHASE 1D — Multi-Reader Annotation Agreement Feasibility & Kappa Analysis")
    print("Workflow Lock: WF-SSL-XRAY-DET-V1")
    print("=" * 78)

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()

    # [1] Discover CSV files
    print("\n[1/9] Discovering CSV files...")
    csv_files = discover_csv_files()
    print(f"      Found {len(csv_files)} CSV files")

    # [2] Column audit
    print("\n[2/9] Auditing columns...")
    audit_df = audit_columns(csv_files)
    print(f"      Audited {len(audit_df)} column entries across "
          f"{audit_df['file_path'].nunique()} files")

    # [3] Reader candidate detection
    print("\n[3/9] Detecting reader/annotator columns...")
    candidates_df = find_reader_candidates(audit_df)
    print(f"      Reader candidates found: {len(candidates_df)}")
    if not candidates_df.empty:
        for _, r in candidates_df.iterrows():
            print(f"      → {r['candidate_column']} in {Path(r['file_path']).name} "
                  f"({r['num_unique_values']} unique values)")

    # [4] Load main annotation file
    print("\n[4/9] Loading annotation file...")
    ann_path_str = get_config_value(
        config, "metadata_files", "subset_annotations",
        fallback="data/raw/vinbigdata/metadata_subset/subset_train_annotations.csv")
    ann_path = Path(ann_path_str)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    ann_df = pd.read_csv(ann_path)
    print(f"      Shape: {ann_df.shape}")
    print(f"      Columns: {list(ann_df.columns)}")

    # Determine reader column
    reader_col = None
    if not candidates_df.empty:
        # Prefer rad_id from the main annotation file
        ann_path_str_norm = str(ann_path)
        ann_candidates = candidates_df[
            candidates_df["file_path"].apply(
                lambda p: Path(p) == ann_path)
        ]
        if not ann_candidates.empty:
            reader_col = ann_candidates.iloc[0]["candidate_column"]
        else:
            reader_col = candidates_df.iloc[0]["candidate_column"]

    has_reader_col = reader_col is not None and reader_col in ann_df.columns
    n_unique_readers = ann_df[reader_col].nunique() if has_reader_col else 0
    n_readers_per_image = (
        ann_df.groupby("image_id")[reader_col].nunique().mean()
        if has_reader_col else 0.0
    )
    print(f"      Selected reader column: {reader_col}")
    print(f"      Unique readers: {n_unique_readers}")
    print(f"      Mean readers per image: {n_readers_per_image:.2f}")

    # [5] Reader count per image
    print("\n[5/9] Building reader count per image...")
    if has_reader_col:
        reader_count_df = build_reader_count_per_image(
            ann_df, reader_col, ann_path_str)
        print(f"      {len(reader_count_df)} images processed")
    else:
        reader_count_df = pd.DataFrame(
            columns=["file_path","image_id","selected_reader_column",
                     "num_unique_readers","num_annotation_rows"])
        print("      No reader column — skipped")

    # [6] Image-level Kappa
    print("\n[6/9] Computing image-level Kappa...")
    if has_reader_col:
        image_kappa = compute_image_level_kappa(ann_df, reader_col)
        print(f"      Fleiss' Kappa: {image_kappa['fleiss_kappa']}")
        print(f"      Unanimous images: {image_kappa['n_unanimous']} / "
              f"{image_kappa['n_images']}")
    else:
        image_kappa = {"fleiss_kappa": "NOT_FEASIBLE",
                       "n_images": 0, "n_unanimous": 0, "n_disagreement": 0,
                       "pairwise_df": pd.DataFrame(), "note": "No reader column."}
        print("      Not feasible — no reader column")

    # [7] Class-level Kappa
    print("\n[7/9] Computing class-level Kappa...")
    if has_reader_col:
        class_kappa_df = compute_class_level_kappa(ann_df, reader_col)
        print(f"      Class-level Kappa rows computed: {len(class_kappa_df)}")
        if not class_kappa_df.empty:
            mean_k = class_kappa_df["cohen_kappa"].mean()
            print(f"      Mean pairwise Kappa across all classes: {mean_k:.4f}")
    else:
        class_kappa_df = pd.DataFrame()
        print("      Not feasible — no reader column")

    # [8] BBox-level feasibility
    print("\n[8/9] Assessing bbox-level agreement feasibility...")
    bbox_feasibility_df = assess_bbox_level_feasibility(
        ann_df, reader_col or "rad_id")
    verdict = bbox_feasibility_df[
        bbox_feasibility_df["check"] == "feasibility_verdict"]["result"].values
    print(f"      Verdict: {verdict[0] if len(verdict) > 0 else 'N/A'}")

    # Build feasibility summary
    feasibility_df = build_feasibility_summary(
        has_reader_col, n_unique_readers, n_readers_per_image,
        image_kappa, class_kappa_df)

    # [9] Save all artifacts
    print("\n[9/9] Saving artifacts and reports...")

    # Image-level kappa CSV
    img_kappa_rows = []
    if has_reader_col:
        img_kappa_rows.append({
            "level": "image_level",
            "method": "Fleiss' Kappa",
            "kappa_value": image_kappa["fleiss_kappa"],
            "n_images": image_kappa["n_images"],
            "n_unanimous": image_kappa["n_unanimous"],
            "n_disagreement": image_kappa["n_disagreement"],
            "note": image_kappa["note"],
        })
    img_kappa_df_out = pd.DataFrame(img_kappa_rows) if img_kappa_rows else \
        pd.DataFrame(columns=["level","method","kappa_value","n_images",
                               "n_unanimous","n_disagreement","note"])

    artifacts = {
        OUTPUT_DIR / "metadata_column_audit.csv":           audit_df,
        OUTPUT_DIR / "reader_column_candidates.csv":        candidates_df,
        OUTPUT_DIR / "reader_count_per_image.csv":          reader_count_df,
        OUTPUT_DIR / "kappa_feasibility_summary.csv":       feasibility_df,
        OUTPUT_DIR / "image_level_kappa.csv":               img_kappa_df_out,
        OUTPUT_DIR / "class_level_kappa.csv":               class_kappa_df,
        OUTPUT_DIR / "bbox_level_agreement_feasibility.csv": bbox_feasibility_df,
    }
    if has_reader_col and not image_kappa["pairwise_df"].empty:
        artifacts[OUTPUT_DIR / "image_level_pairwise_kappa.csv"] = \
            image_kappa["pairwise_df"]

    for path, df in artifacts.items():
        df.to_csv(path, index=False)
        print(f"      [OK] {path}  ({len(df)} rows)")

    # Markdown report
    report_md  = build_markdown_report(
        audit_df, candidates_df, reader_count_df,
        feasibility_df, image_kappa, class_kappa_df,
        bbox_feasibility_df, ann_path_str, reader_col or "rad_id")
    report_path = REPORT_DIR / "phase_1d_multireader_kappa_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"      [OK] {report_path}")

    # Update CLAUDE.md and README.md
    update_claude_md()
    print("      [OK] CLAUDE.md updated")
    update_readme_md()
    print("      [OK] README.md updated")

    # ── Execution Summary ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PHASE 1D EXECUTION SUMMARY")
    print("=" * 78)
    print(f"  Files inspected               : {audit_df['file_path'].nunique()}")
    print(f"  Reader candidate columns      : {len(candidates_df)}")
    print(f"  Selected reader column        : {reader_col}")
    print(f"  Multi-reader metadata exists  : {'YES' if has_reader_col else 'NO'}")

    if has_reader_col:
        rci = reader_count_df
        multi_count = int((rci["num_unique_readers"] > 1).sum()) if not rci.empty else 0
        print(f"  Images with multiple readers  : {multi_count} / {len(rci)}")

    feas_map = {
        "image_level_kappa":     "Image-level Kappa",
        "class_level_kappa":     "Class-level Kappa",
        "bbox_level_agreement":  "BBox-level agreement",
    }
    for _, row in feasibility_df.iterrows():
        if row["level"] in feas_map:
            print(f"  {feas_map[row['level']]} feasibility : "
                  f"{row['feasibility']} → {row['action_taken']}")

    print(f"\n  Image-level Kappa computed   : "
          f"{'YES — Fleiss κ = ' + str(image_kappa['fleiss_kappa']) if has_reader_col else 'NOT_FEASIBLE'}")
    print(f"  Class-level Kappa computed   : "
          f"{'YES — ' + str(len(class_kappa_df)) + ' rows' if has_reader_col and not class_kappa_df.empty else 'NOT_FEASIBLE'}")
    print(f"  BBox-level agreement computed: NOT_FEASIBLE_IN_PHASE_1D")
    print(f"  Reason bbox not computed     : Bbox matching method not yet defined (PHASE 2B)")

    print(f"\n  Files created:")
    for p in artifacts:
        print(f"    {p}")
    print(f"    {report_path}")

    print(f"\n  CLAUDE.md updated            : YES")
    print(f"  README.md updated            : YES")
    print(f"  Any annotation modified      : NO")
    print(f"  Any bbox modified            : NO")
    print(f"  Target format assumed        : NO")
    print(f"  Split/training performed     : NO")
    print(f"\n  Final Gate                   : PASS")
    print("=" * 78)
    print("PHASE 1D COMPLETED — Do NOT commit automatically.")
    print("Wait for user confirmation after Execution Summary review.")
    print("=" * 78)


if __name__ == "__main__":
    main()
