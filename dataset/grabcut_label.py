#!/usr/bin/env python3
"""
Automatic instance-mask labelling of the phone cover (dark-object cue + GrabCut).

On a plain mid-tone background the cover is the dominant DARK object; the white
reference card and the background are excluded by construction. Steps:
  1. Threshold dark pixels -> take the largest dark blob as the cover's location.
  2. Refine the exact silhouette with GrabCut (rectangle-initialised).
  3. Export COCO instance-segmentation polygons + a QA overlay per image.

No YOLO, no Roboflow — classical CV only.

Usage:
    python dataset/grabcut_label.py --images dataset/undistorted \
        --out-json dataset/exports/annotations.json \
        --overlays dataset/exports/overlays --category phone_cover
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def cover_box(img, dark_thr=None, min_frac=0.02):
    """Bounding box of the largest dark blob (the cover)."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = dark_thr if dark_thr else int(min(90, np.percentile(gray, 20)))
    dark = (gray < thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, w, h, a = stats[i][:5]
        if a >= min_frac * H * W and a > best_area:
            best_area, best = a, (int(x), int(y), int(w), int(h))
    return best


def grabcut_contour(img, box, pad=20, iters=5):
    """Rectangle-initialised GrabCut -> largest refined foreground contour."""
    x, y, w, h = box
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = min(img.shape[1] - x, w + 2 * pad), min(img.shape[0] - y, h + 2 * pad)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, (x, y, w, h), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def contour_to_polygon(contour, eps_frac=0.0015):
    eps = eps_frac * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 3:
        approx = contour
    return approx.reshape(-1).astype(float).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out-json", default="dataset/exports/annotations.json", type=Path)
    ap.add_argument("--overlays", default="dataset/exports/overlays", type=Path)
    ap.add_argument("--category", default="phone_cover")
    ap.add_argument("--dark-thr", type=int, default=None, help="fixed dark threshold (0-255)")
    ap.add_argument("--min-area-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0, help="only process first N (testing)")
    args = ap.parse_args()

    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        files = files[:args.limit]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)

    coco = {"info": {"description": f"Auto-labelled {args.category} (GrabCut)"},
            "images": [], "annotations": [],
            "categories": [{"id": 1, "name": args.category, "supercategory": "object"}]}
    ann_id, ok, miss = 1, 0, []
    for img_id, p in enumerate(files, start=1):
        img = cv2.imread(str(p))
        if img is None:
            miss.append(p.name); continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        box = cover_box(img, args.dark_thr, args.min_area_frac)
        c = grabcut_contour(img, box) if box else None
        overlay = img.copy()
        if c is None or cv2.contourArea(c) < args.min_area_frac * h * w:
            miss.append(p.name)
        else:
            x, y, bw, bh = cv2.boundingRect(c)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": [contour_to_polygon(c)],
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": float(cv2.contourArea(c)), "iscrowd": 0})
            ann_id += 1; ok += 1
            cv2.drawContours(overlay, [c], -1, (0, 255, 0), 4)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 128, 255), 3)
        cv2.imwrite(str(args.overlays / p.name), overlay)

    with open(args.out_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Labelled {ok}/{len(files)} -> {args.out_json}")
    print(f"QA overlays -> {args.overlays}")
    if miss:
        print(f"MISSED {len(miss)}: {', '.join(miss[:12])}" + (" ..." if len(miss) > 12 else ""))


if __name__ == "__main__":
    main()
