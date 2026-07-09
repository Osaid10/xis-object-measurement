#!/usr/bin/env python3
"""
Pixel-to-millimetre measurement of a segmented object using a reference card.

Pipeline (per the spec's conceptual flow):
  1. Undistort the image with the stored intrinsics (mandatory before measuring).
  2. Detect the reference card (ISO 7810 ID-1, 85.60 x 53.98 mm) -> pixels_per_mm.
  3. Obtain the object mask (trained Mask R-CNN, or classical CV for validation).
  4. Fit a min-area rectangle to the mask -> object pixel dimensions.
  5. Convert to millimetres via pixels_per_mm -> width_mm, height_mm.
  6. Annotate the image with the mask overlay and metric labels.

Usage:
    python measurement/measure.py --image path.jpg --method cv
    python measurement/measure.py --image path.jpg --method model \
        --weights models/weights/maskrcnn_best.pth --out out.jpg

Why undistortion matters: lens distortion bends straight edges and changes the
apparent pixel size non-uniformly across the frame, so pixels_per_mm measured
from the card would not apply correctly to the object. Run with --no-undistort
to reproduce the (incorrect) distorted measurement for the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# make repo-root packages importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibration.camera_utils import load_calibration, undistort  # noqa: E402

CARD_LONG_MM = 85.60
CARD_SHORT_MM = 53.98
CARD_ASPECT = CARD_LONG_MM / CARD_SHORT_MM  # ~1.585


# ----------------------------- segmentation (CV) ----------------------------- #
def _foreground_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    b = 12
    edges = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                            img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    bg = np.median(edges, axis=0)
    diff = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)


def _rect_metrics(contour):
    (cx, cy), (w, h), ang = cv2.minAreaRect(contour)
    if w < 1 or h < 1:
        return None
    long_px, short_px = max(w, h), min(w, h)
    aspect = long_px / short_px
    fill = cv2.contourArea(contour) / (w * h)
    box = cv2.boxPoints(((cx, cy), (w, h), ang))
    return dict(long_px=long_px, short_px=short_px, aspect=aspect, fill=fill,
               box=box.astype(int), area=cv2.contourArea(contour))


# ----------------------------- reference card ------------------------------- #
def detect_reference_card(img: np.ndarray, card_long=CARD_LONG_MM,
                          card_short=CARD_SHORT_MM, min_frac=0.004, max_frac=0.6):
    """Find the reference card and derive pixels_per_mm. Returns dict or None."""
    mask = _foreground_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    target_aspect = card_long / card_short
    best, best_score = None, 1e9
    for c in contours:
        m = _rect_metrics(c)
        if m is None:
            continue
        frac = m["area"] / img_area
        if not (min_frac <= frac <= max_frac) or m["fill"] < 0.85:
            continue
        score = abs(m["aspect"] - target_aspect)
        if score < best_score:
            best_score, best = score, m
    if best is None or best_score > 0.30:      # reject if nothing card-shaped
        return None
    ppm = (best["long_px"] / card_long + best["short_px"] / card_short) / 2.0
    return {"pixels_per_mm": float(ppm), "box": best["box"],
            "long_px": best["long_px"], "short_px": best["short_px"]}


# ------------------------------ object dims --------------------------------- #
def object_mask_cv(img: np.ndarray, card_box=None):
    """Largest foreground blob that is not the reference card (classical CV)."""
    mask = _foreground_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    card_c = None
    if card_box is not None:
        cx, cy = card_box.mean(axis=0)
        card_c = (cx, cy)
    def is_card(c):
        if card_c is None:
            return False
        M = cv2.moments(c)
        if M["m00"] == 0:
            return False
        return abs(M["m10"] / M["m00"] - card_c[0]) < 20 and abs(M["m01"] / M["m00"] - card_c[1]) < 20
    cands = [c for c in contours if not is_card(c)] or contours
    biggest = max(cands, key=cv2.contourArea)
    out = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(out, [biggest], -1, 255, cv2.FILLED)
    return out


def measure_object(object_mask: np.ndarray, ppm: float):
    """Min-area-rectangle dimensions of a binary mask, converted to mm."""
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    m = _rect_metrics(max(contours, key=cv2.contourArea))
    short_mm = m["short_px"] / ppm
    long_mm = m["long_px"] / ppm
    return {"width_mm": round(short_mm, 2), "height_mm": round(long_mm, 2),
            "short_mm": round(short_mm, 2), "long_mm": round(long_mm, 2),
            "long_px": round(m["long_px"], 1), "short_px": round(m["short_px"], 1),
            "box": m["box"]}


# ------------------------------- annotation --------------------------------- #
def annotate(img, card, obj_mask, dims, score=None):
    vis = img.copy()
    overlay = vis.copy()
    overlay[obj_mask > 0] = (0, 200, 0)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    if card is not None:
        cv2.drawContours(vis, [card["box"]], -1, (0, 165, 255), 3)
        cv2.putText(vis, "REF 85.6x54.0mm", tuple(card["box"][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    if dims is not None:
        cv2.drawContours(vis, [dims["box"]], -1, (0, 255, 255), 3)
        label = f"W: {dims['width_mm']:.1f} mm  H: {dims['height_mm']:.1f} mm"
        if score is not None:
            label += f"  conf: {score:.2f}"
        org = (int(dims["box"][:, 0].min()), int(dims["box"][:, 1].min()) - 12)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis


# ------------------------------- driver ------------------------------------- #
def get_object_mask(img, method, weights, category, device, card_box, model=None):
    if method == "cv":
        return object_mask_cv(img, card_box), None
    # model method (lazy import so CV path needs no torch)
    from inference.model import load_model, predict
    if model is None:
        model = load_model(weights, num_classes=2, device=device)
    dets = predict(model, img, device=device, score_thresh=0.5)
    if not dets:
        return None, None
    top = dets[0]
    return (top["mask"] * 255).astype(np.uint8), top["score"]


def run(image_path, calib="calibration/calibration.json", method="cv",
        weights=None, category="object", do_undistort=True, device="cpu", model=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Cannot read {image_path}")
    if do_undistort:
        K, dist, _ = load_calibration(calib)
        img = undistort(img, K, dist)
    card = detect_reference_card(img)
    if card is None:
        raise SystemExit("Reference card not detected — check contrast/placement.")
    obj_mask, score = get_object_mask(img, method, weights, category, device,
                                      card["box"], model=model)
    if obj_mask is None:
        raise SystemExit("Object not detected.")
    dims = measure_object(obj_mask, card["pixels_per_mm"])
    vis = annotate(img, card, obj_mask, dims, score)
    result = {
        "image": str(image_path), "undistorted": do_undistort, "method": method,
        "pixels_per_mm": round(card["pixels_per_mm"], 4),
        "width_mm": dims["width_mm"], "height_mm": dims["height_mm"],
        "confidence": round(score, 3) if score is not None else None,
    }
    return result, vis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--calib", default="calibration/calibration.json")
    ap.add_argument("--method", choices=["cv", "model"], default="cv")
    ap.add_argument("--weights", default="models/weights/maskrcnn_best.pth")
    ap.add_argument("--category", default="object")
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result, vis = run(args.image, args.calib, args.method, args.weights,
                      args.category, args.undistort, args.device)
    print(json.dumps(result, indent=2))
    out = args.out or str(Path("inference/demo_outputs") /
                          (Path(args.image).stem + "_measured.jpg"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, vis)
    print(f"Annotated image -> {out}")


if __name__ == "__main__":
    main()
