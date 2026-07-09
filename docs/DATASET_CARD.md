# Dataset Card

## 1. Object selection
| Property | Detail |
|----------|--------|
| **Object** | Flat, rigid **phone cover** (silicone, black with copper accents) |
| **Real dimensions (ground truth)** | **160 mm (height) × 79 mm (width)**, ruler-measured |
| **Reference object** | ISO/IEC 7810 **ID-1 card** (85.60 × 53.98 mm) in-frame |

**Why this object (justification):**
- **Availability** — a single consistent item, always on hand; the ID card is a
  precise, internationally standardised scale reference.
- **Geometry** — thin and flat, so its top surface is ~coplanar with the
  reference card, keeping a single pixels-per-mm scale valid. Its **2.03 aspect
  ratio is clearly distinct from the card's 1.585**, so the two never get
  confused. Distinct width ≠ height gives a clean two-dimension measurement demo.
- **Labelling ease** — a solid dark object; on a plain background it is the
  dominant dark region, enabling reliable automatic mask labelling.

## 2. Collection strategy
- **Camera:** iPhone 12 Pro Max, main lens fixed at **1×** (same camera as
  calibration), JPEG (converted from HEIC), full resolution **3024 × 4032**.
- **View:** top-down (camera parallel to the surface), ~30–50 cm.
- **Background:** plain **mid-grey towel** — contrasts with both the black cover
  and the white card, and (unlike the initial wood-table attempt) carries no
  colour that confuses the segmenter. Only the cover + card in frame, small gap.
- **Diversity:** varied object position, rotation, distance, and lighting.
- **Undistortion:** every image is undistorted with the Step-1 calibration
  **before** labelling and training.
- **Count:** **79** images (spec minimum: 70).

## 3. Labelling
| Property | Detail |
|----------|--------|
| **Method** | Automatic instance-mask labelling via **classical CV** (no YOLO, no Roboflow) — [`dataset/grabcut_label.py`](../dataset/grabcut_label.py) |
| **How** | The cover is located as the largest **dark** blob (grey background and white card excluded by brightness), then its exact silhouette is refined with **GrabCut** (rectangle-initialised). Card and background are excluded by construction. |
| **QA** | A labelled overlay is written for every image (`dataset/exports/overlays/`) and reviewed; masks confirmed to trace the cover silhouette. |
| **Export** | COCO instance-segmentation JSON (`dataset/exports/annotations.json`). |

> Approach history (documented for transparency): a first attempt on a cluttered
> wood table failed — the wood's orange hue and background clutter defeated
> colour/brightness cues. Re-shooting on a plain grey background made the
> dark-object + GrabCut method reliable. This is a real illustration of how data
> collection conditions drive segmentation robustness.

## 4. Statistics
Seeded split (`--seed 42`) via [`dataset/split_dataset.py`](../dataset/split_dataset.py).

| Split | Images |
|-------|--------|
| Train (70%) | _filled after split_ |
| Val (20%)   | _filled after split_ |
| Test (10%)  | _filled after split_ |
| **Total**   | 79 |

- **Classes:** 1 (`phone_cover`) + background. **Class balance:** single-class,
  one instance per image (balanced by construction).

## 5. Hosting
Raw images, undistorted images, and the COCO export are on **Google Drive**
(links in README). Only code and this card are committed to GitHub.
