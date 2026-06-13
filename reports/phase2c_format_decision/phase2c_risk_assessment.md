# PHASE 2C — Annotation Format Risk Assessment

**Workflow Lock: WF-SSL-XRAY-DET-V1**

---

## 1. Overview

This document assesses risks and tradeoffs of each candidate annotation format
for the VinBigData subset canonical schema (PHASE 2B).

**Dataset facts (from PHASE 2B):**
- Total images: 4894
- Abnormal images: 4394 (with bbox)
- No Finding images: 500 (negatives, no bbox)
- Total bbox rows: 36096 (all valid)
- Detection classes: 14
- Readers per image: 3 (rad_id must be preserved)

---

## 2. Overall Score Summary

Scores are 1–5 per criterion × 11 criteria = max 55.

| Format | Total Score |
|--------|------------|
| COCO JSON | 54 / 55 |
| Internal JSONL (PHASE 2B) | 47 / 55 |
| YOLO TXT | 33 / 55 |
| Pascal VOC XML | 31 / 55 |

---

## 3. Critical Risk: rad_id / source_row_id Traceability

Multi-reader traceability is a key requirement for this research project
(Kappa analysis in PHASE 1D; future PHASE 6 agreement analysis).

| Format | Score | Notes |
|--------|-------|-------|
| COCO JSON | 5 | Custom extra fields ('rad_id', 'source_row_id') can be added to each annotation dict without breaking COCO spec. Zero loss of traceability. |
| Internal JSONL (PHASE 2B) | 5 | Native: rad_id and source_row_id are first-class fields in every box dict. |
| Pascal VOC XML | 3 | Possible via custom XML attributes but non-standard; fragile in practice. |
| YOLO TXT | 1 | Plain text per-line format has no standard metadata field. rad_id and source_row_id are lost unless a parallel sidecar file is maintained. |

**Decision:** YOLO TXT is the only format that materially risks losing
rad_id and source_row_id traceability. This alone disqualifies it as a
primary format for this research project.

---

## 4. Critical Risk: SSL OD Framework Compatibility

| Format | Score | Notes |
|--------|-------|-------|
| COCO JSON | 5 | STAC, Unbiased Teacher, Soft-Teacher, Semi-DETR, SoftER — all use COCO. SSL OD benchmark papers universally report on COCO-formatted data. |
| Internal JSONL (PHASE 2B) | 3 | SSL OD frameworks expect COCO or YOLO; JSONL needs conversion wrapper. |
| YOLO TXT | 3 | Most SSL OD research codebases (STAC, Unbiased Teacher, SoftER) use COCO as primary format. YOLO requires conversion wrapper. |
| Pascal VOC XML | 2 | SSL OD literature rarely uses Pascal VOC XML as primary format for new experiments; mostly seen in legacy code (PASCAL VOC 2007/2012). |

**Decision:** COCO JSON is the de-facto standard for SSL OD research
(STAC, Unbiased Teacher, Soft-Teacher, Semi-DETR, SoftER, etc.).
Using COCO from the start avoids conversion overhead in training phases.

---

## 5. Negative Image Handling Risk

500 no_finding images must be representable:

| Format | Strategy | Risk |
|--------|---------|------|
| COCO JSON | Empty `"annotations": []` per image | ✅ Zero risk — standard |
| YOLO TXT | Empty .txt file (version-dependent) | ⚠️ Medium — convention varies |
| Pascal VOC XML | XML with no `<object>` tags | ✅ Low risk |
| Internal JSONL | `"boxes": []` already implemented | ✅ Zero risk |

---

## 6. Conversion Effort Risk

| Format | Engineering Effort | Key Dependency |
|--------|-------------------|----------------|
| COCO JSON | Low | image_width/height from PHASE 2A; trivial bbox format change |
| YOLO TXT | Medium | image dims + normalisation + sidecar for traceability |
| Pascal VOC XML | High | 4,894 individual XML files; custom attribute schema |
| Internal JSONL | None | Already done in PHASE 2B |

---

## 7. Risk Summary

| Risk | COCO JSON | YOLO TXT | Pascal VOC | Internal JSONL |
|------|-----------|----------|-----------|----------------|
| Traceability loss | None | **HIGH** | Low | None |
| Negative image handling | Trivial | Medium | Low | Trivial |
| Framework compatibility | Excellent | Good | Poor | Needs adapter |
| Conversion effort (future) | Low | Medium | High | None needed |
| SSL OD research blocker | None | Medium | **High** | Medium |
| Paper/thesis comparability | None | Minor | **High** | Minor |
