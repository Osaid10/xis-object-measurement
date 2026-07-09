# Camera Calibration Report

## 1. Objective
Estimate the **intrinsic parameters** (camera matrix and lens-distortion
coefficients) of the capture camera so that all downstream images can be
**undistorted** before measurement. Undistortion is mandatory: uncorrected
radial/tangential distortion bends straight edges and makes the pixel-to-mm
scale vary across the frame, which would corrupt any measurement.

## 2. Equipment
| Item | Detail |
|------|--------|
| Camera | Apple iPhone 12 Pro Max, **main wide lens fixed at 1×** |
| Format | JPEG ("Most Compatible"), AE/AF locked |
| Target | Checkerboard, **8 × 10 squares → 7 × 9 inner corners** (`calibration/checkerboard.html`) |
| Display | Checkerboard shown full-screen on a laptop LCD, photographed with the phone |

**Why a screen instead of print:** no printer was available. An LCD is flat and
rigid, so it is a valid calibration surface. Crucially, the intrinsic matrix and
distortion coefficients are **independent of the physical square size** (scaling
the board only scales the unused extrinsic translation), so undistortion is
unaffected by the exact on-screen square dimensions. Known minor caveats
(screen glare, possible moiré) were mitigated by high brightness, a dim room,
and moderate shooting distance.

## 3. Method
Implemented in [`calibration/calibrate.py`](../calibration/calibrate.py) with OpenCV:
1. Detect inner corners per image with `cv2.findChessboardCornersSB`
   (robust detector), falling back to `findChessboardCorners` + `cornerSubPix`.
2. Build 3-D object points on the `z = 0` plane for the 7 × 9 grid.
3. Run `cv2.calibrateCamera` to estimate `K` and distortion coefficients
   `(k1, k2, p1, p2, k3)`.
4. Compute the RMS reprojection error and a per-image error breakdown.

Reproduce:
```bash
python calibration/calibrate.py --images calibration/images \
    --cols 7 --rows 9 --square-size 20 --save-detections
```

## 4. Calibration set
- **Images captured:** 25 (spec minimum: 20), varied angle, distance, position.
- **Images with a valid detection:** _TBD_ / 25.
- Corner-overlay detections saved to `calibration/detected/` (sample in `docs/figures/`).

## 5. Results
> Filled from `calibration/calibration.json` after running calibration.

**Camera matrix `K` (pixels):**
```
[ fx   0  cx ]     [ _TBD    0    _TBD ]
[  0  fy  cy ]  =  [  0    _TBD   _TBD ]
[  0   0   1 ]     [  0      0      1  ]
```

**Distortion coefficients** `(k1, k2, p1, p2, k3)`: _TBD_

| Metric | Value | Spec |
|--------|-------|------|
| RMS reprojection error | _TBD_ px | < 0.5 acceptable, < 0.3 excellent |
| Mean reprojection error | _TBD_ px | — |
| Image resolution | _TBD_ × _TBD_ | — |

## 6. Undistortion verification
`calibration/camera_utils.py` applies `cv2.undistort` with the stored intrinsics.
A before/after comparison (straight screen edges become straight) is shown in
`docs/figures/undistort_before_after.jpg` _(TBD)_.

```bash
python calibration/camera_utils.py --input dataset/raw --output dataset/undistorted
```

## 7. Discussion
Modern phone main-cameras apply internal correction, so the residual distortion
is typically small; the estimated coefficients are expected to be modest and the
undistortion close to (but not exactly) identity. Calibration is still mandatory
and is applied to **every** image used for measurement. The reprojection error is
the primary quality indicator and is reported above.

## 8. Artifacts
- `calibration/calibration.json` — K, distortion, errors (in repo, small)
- `calibration/calibration.npz` — same, NumPy format
- Calibration images — **Google Drive** (see README), not committed.
