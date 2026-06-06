# CLAUDE.md

## Project context

This project is about semi-supervised object detection for chest X-ray abnormality detection.

The main research direction is:

**Semi-supervised learning for detecting abnormalities on chest X-ray images using attention/Transformer-based detectors under limited bounding box annotation.**

The current dataset is:

- VinBigData Chest X-ray Abnormalities Detection
- Metadata-only subset
- 500 Normal / No Finding images
- All abnormal images
- No DICOM images are stored in Git

## Current project stage

The project is currently at:

```text
Dataset selection and metadata-only subset creation

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
