# XIS Assessment — Calibrated Object Measurement Pipeline

An end-to-end computer-vision system that measures the **real-world width and
height (in millimetres)** of a custom object from a single photograph, using
camera calibration, deep-learning instance segmentation, and a reference object
for scale.

> **Target object:** a flat, rigid **phone cover**.
> **Reference object:** a standard **ID-1 bank/ID card** (85.60 × 53.98 mm, ISO/IEC 7810).
> **Segmentation model:** **Mask R-CNN (ResNet-50-FPN)** via torchvision —
> deliberately *not* Ultralytics YOLO or a Roboflow model, per the assessment.

---

## Key capabilities
- **Intrinsic camera calibration** (OpenCV) from checkerboard images, with
  reprojection-error reporting and lens-distortion removal.
- **Automatic dataset labelling** (classical CV, no YOLO/Roboflow) → COCO export.
- **Instance segmentation** with Mask R-CNN, fully evaluated (mAP, IoU, P/R/F1).
- **Pixel→mm measurement** from a calibrated, undistorted image using a reference
  card, validated against physical calliper measurements (MAE / MPE).
- **Reproducible**: seeded splits, config-driven training, one-command scripts.

## Pipeline architecture

```
        iPhone 12 Pro Max @ 1x (single calibrated camera)
        │                                   │
        │ checkerboard photos               │ phone-cover photos (+ card in frame)
        ▼                                   │
 calibration/calibrate.py                   │
        │  → calibration.json (K, dist)     │
        ▼                                   ▼
   camera_utils.undistort()  ◄──────────────┘        (undistortion: mandatory)
        │  (undistorted images)
        ├─► dataset/auto_label.py  → COCO annotations
        │        └─► dataset/split_dataset.py → train / val / test (70/20/10)
        │                                   │
        │                          models/train.py  (Mask R-CNN, Colab GPU)
        │                                   │  → models/weights/maskrcnn_best.pth
        ▼                                   ▼
 measurement/measure.py  ◄──────────────────┘
   1. detect reference card → pixels_per_mm
   2. model mask → min-area rectangle → pixel W×H
   3. convert → millimetres
        ├─► annotated image (mask overlay + W/H mm + confidence)
        └─► measurement/validate_accuracy.py → MAE / MPE vs calliper ground truth
```

## Repository guide

| Path | Contents |
|------|----------|
| `calibration/` | `calibrate.py`, `camera_utils.py`, checkerboard target, (images on Drive) |
| `dataset/`     | `auto_label.py`, `split_dataset.py`, train/val/test (images on Drive) |
| `models/`      | `train.py`, Colab notebook, configs (weights on Drive) |
| `inference/`   | `model.py` (Mask R-CNN), `infer.py`, demo outputs |
| `measurement/` | `measure.py`, `validate_accuracy.py`, accuracy report |
| `docs/`        | all reports (calibration, dataset, training, measurement), setup, capture guide |

## Quick start
See **[docs/SETUP.md](docs/SETUP.md)** for full details.
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 1. Calibrate the camera
python calibration/calibrate.py --images calibration/images --save-detections
# 2. Undistort object images
python calibration/camera_utils.py --input dataset/raw --output dataset/undistorted
# 3. Auto-label + split
python dataset/auto_label.py --images dataset/undistorted --category phone_cover
python dataset/split_dataset.py --images dataset/undistorted
# 4. Train (Colab GPU recommended — models/Train_MaskRCNN_Colab.ipynb)
python models/train.py --data-root dataset --epochs 25
# 5. Measure a new image (end-to-end demo)
python measurement/measure.py --image path/to/new.jpg --method model \
    --weights models/weights/maskrcnn_best.pth
# 6. Validate accuracy vs calliper ground truth
python measurement/validate_accuracy.py --images measurement/eval_images \
    --method model --weights models/weights/maskrcnn_best.pth \
    --gt-width <W_mm> --gt-height <H_mm>
```

## 📦 Large files (Google Drive — per Section 2.2)
Large files are **not** stored in this repo. All are on Google Drive with
"Anyone with the link can view":

| Contents | Link |
|----------|------|
| Calibration images (25) | _TBD — add shareable link_ |
| Full raw dataset (70+ images) | _TBD — add shareable link_ |
| Labelled dataset export (COCO) | _TBD — add shareable link_ |
| Trained model weights (`maskrcnn_best.pth`) | _TBD — add shareable link_ |

## Documentation
- [SETUP.md](docs/SETUP.md) — installation, environment, run instructions
- [CALIBRATION_REPORT.md](docs/CALIBRATION_REPORT.md) — method, intrinsics, reprojection error
- [DATASET_CARD.md](docs/DATASET_CARD.md) — object, collection, labelling, statistics
- [TRAINING_REPORT.md](docs/TRAINING_REPORT.md) — architecture, hyperparameters, metrics, curves
- [MEASUREMENT_REPORT.md](docs/MEASUREMENT_REPORT.md) — pixel→mm derivation, accuracy, limitations
- [CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md) — how the images were captured

## Module / API reference (summary)
| Module | Key function | Input → Output |
|--------|--------------|----------------|
| `calibration/calibrate.py` | CLI | checkerboard images → `calibration.json` (K, dist, error) |
| `calibration/camera_utils.py` | `undistort(img, K, dist)` | BGR image → undistorted BGR image |
| `dataset/auto_label.py` | CLI | object images → COCO annotations + QA overlays |
| `inference/model.py` | `predict(model, img)` | BGR image → `[{box, score, mask}]` |
| `measurement/measure.py` | `run(image, ...)` | image → `{width_mm, height_mm, confidence}` + annotated image |
| `measurement/validate_accuracy.py` | CLI | images + GT → MAE / MPE table |

## Design decisions (summary — full rationale in the reports)
- **Mask R-CNN over YOLO/Roboflow** (excluded): mature COCO-pretrained instance
  segmenter that transfer-learns well on a small custom set and yields precise masks.
- **Flat, thin object + coplanar reference card**: keeps the object surface level
  with the card so a single pixels-per-mm scale is valid (see MEASUREMENT_REPORT).
- **Classical CV auto-labelling**: the high-contrast object on a plain background is
  segmentable without a model, giving fast, consistent ground-truth masks.

## Assumptions & limitations (summary)
- The object and reference card are approximately **coplanar** and viewed
  **top-down**; large tilt or object thickness introduces perspective error.
- One calibrated camera at a **fixed 1× lens**; switching lenses invalidates calibration.
- Measurement scale is **isotropic** (square pixels, fronto-parallel view).
- Full discussion in [MEASUREMENT_REPORT.md](docs/MEASUREMENT_REPORT.md).
