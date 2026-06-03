"""Keypoint projection + overlay helpers for previewing 3D bouts.

These are used by ``scripts/preview_bouts.py`` to render multi-camera montages
with the 3D keypoints reprojected onto each raw video frame, so tracking quality
can be inspected by eye.
"""
import cv2

import numpy as np

from .calibration import disassemble_extrinsics


def project_points(K, dist, extrinsics, X, offset=None):
    """Reproject 3D points ``X`` (N, 3) to 2D pixels via ``cv2.projectPoints``.

    Honors the anipose distortion coefficients (unlike the ortho pinhole path).
    ``extrinsics`` is the 4x4 [R|t] matrix; ``offset`` (len-2, the crop's
    top-left corner in the full frame) is subtracted so points land in the
    stored raw-video frame coordinates. NaN inputs map to NaN outputs.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    out = np.full((X.shape[0], 2), np.nan)

    finite = np.isfinite(X).all(axis=1)
    if finite.any():
        rvec, tvec = disassemble_extrinsics(extrinsics)
        pts, _ = cv2.projectPoints(
            X[finite].reshape(-1, 1, 3), rvec, tvec,
            np.asarray(K, dtype=np.float64),
            np.asarray(dist, dtype=np.float64).reshape(-1))
        pts = pts.reshape(-1, 2)
        if offset is not None:
            pts = pts - np.asarray(offset, dtype=np.float64).reshape(1, 2)
        out[finite] = pts

    return out


def draw_keypoints(frame, pts2d, vis=None, radius=3, thickness=-1):
    """Draw one circle per keypoint on a copy of ``frame`` (BGR uint8).

    Points outside the frame or with non-finite coordinates are skipped. When
    ``vis`` (length-N bool) is given, visible points are green and occluded
    points are red.
    """
    frame = frame.copy()
    h, w = frame.shape[:2]

    for i, (x, y) in enumerate(np.asarray(pts2d, dtype=np.float64)):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        visible = True if vis is None else bool(vis[i])
        color = (0, 255, 0) if visible else (0, 0, 255)   # BGR: green / red
        cv2.circle(frame, (xi, yi), radius, color, thickness)

    return frame


def make_montage(images, rows, cols, pad=2, bg=0):
    """Tile ``images`` (list of HxWx3 BGR) into a single ``rows`` x ``cols`` grid.

    Cells are sized to the largest image; smaller images are top-left aligned.
    Missing cells (when ``len(images) < rows*cols``) stay background-colored.
    """
    images = [im for im in images if im is not None]
    if not images:
        return None

    H = max(im.shape[0] for im in images)
    W = max(im.shape[1] for im in images)

    canvas = np.full((rows * H + (rows + 1) * pad,
                      cols * W + (cols + 1) * pad, 3), bg, dtype=np.uint8)

    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        if r >= rows:
            break
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        h, w = im.shape[:2]
        canvas[y0:y0 + h, x0:x0 + w] = im

    return canvas
