# Dataset Card

## 1. Object selection
| Property | Detail |
|----------|--------|
| **Object** | Flat, rigid **phone cover** |
| **Approx. real dimensions** | _TBD_ mm (width) × _TBD_ mm (height), measured by calliper |
| **Reference object** | ISO/IEC 7810 **ID-1 card** (85.60 × 53.98 mm) in-frame |

**Why this object (justification):**
- **Availability** — a single, consistent, always-on-hand item.
- **Geometry** — thin and flat, so its top surface is ~coplanar with the
  reference card; this keeps a single pixels-per-mm scale valid (a thick object
  would sit above the card and photograph magnified). Distinct width ≠ height
  gives a clean two-dimension measurement demo.
- **Labelling ease** — high contrast against a plain background enables accurate
  automatic mask labelling.

## 2. Collection strategy
- **Camera:** iPhone 12 Pro Max, main lens fixed at **1×** (same camera as
  calibration), AE/AF locked, JPEG.
- **View:** top-down (camera parallel to the surface), ~30–50 cm.
- **Scene:** object on a plain contrasting surface with the reference card in
  frame, on the same plane.
- **Diversity:** varied object position (center/edges/corners), rotation
  (0–90°), distance, background surface, and lighting direction.
- **Undistortion:** every image is undistorted with the Step-1 calibration
  **before** labelling and training.
- **Count:** **70+** images (spec minimum: 70).

## 3. Labelling
| Property | Detail |
|----------|--------|
| **Method** | Automatic instance-mask labelling via **classical CV** (no YOLO, no Roboflow) — [`dataset/auto_label.py`](../dataset/auto_label.py) |
| **How** | Background colour estimated from image border → foreground by colour distance (Otsu) → morphological cleanup → contour → polygon. The reference card is detected by its ISO aspect ratio (~1.585) and **excluded**, so only the object is labelled. |
| **QA** | A labelled overlay is written for every image and visually reviewed before training; any failures are re-checked. |
| **Export** | COCO instance-segmentation JSON (`dataset/exports/annotations.json`). |

> An optional SAM (Segment Anything) labeller with a centre-point prompt is
> documented in SETUP.md for low-contrast cases.

## 4. Statistics
> Filled after labelling + split.

| Split | Images | Instances |
|-------|--------|-----------|
| Train (70%) | _TBD_ | _TBD_ |
| Val (20%)   | _TBD_ | _TBD_ |
| Test (10%)  | _TBD_ | _TBD_ |
| **Total**   | _TBD_ | _TBD_ |

- **Classes:** 1 (`phone_cover`) + background. **Class balance:** single-class,
  one instance per image (balanced by construction).
- **Split method:** seeded random split (`--seed 42`) via
  [`dataset/split_dataset.py`](../dataset/split_dataset.py) — reproducible.

## 5. Data quality notes
- Images with no detected object are excluded from the splits (reported by the
  split script).
- Auto-generated masks are contour-accurate for high-contrast scenes; edge
  precision is discussed in the Measurement Report (mask boundary → error).

## 6. Hosting
Raw images, undistorted images, and the COCO export are on **Google Drive**
(links in README). Only code and this card are committed to GitHub.
