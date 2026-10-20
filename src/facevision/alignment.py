"""Face alignment from five landmarks, in NumPy.

The detector gives five points per face: both eyes, the tip of the nose
and both corners of the mouth. A similarity transform (rotation, uniform
scale and translation) is fitted that carries them as close as possible,
in the least-squares sense, to fixed positions in a 112x112 crop; the
crop is then resampled through that transform. After this, the eyes sit
at the same pixels in every face whatever the head tilt, the distance to
the camera or the jitter of the detector box.

The fit is Umeyama's closed form; the reference positions are the 5-point
template of ArcFace for 112x112 crops (InsightFace).

    S. Umeyama. "Least-squares estimation of transformation parameters
    between two point patterns". IEEE TPAMI, 13(4), 1991.
"""

from __future__ import annotations

import numpy as np

from .imaging import resize

ALIGNED_SIZE = 112          # output crop is ALIGNED_SIZE x ALIGNED_SIZE

# Image-left eye, image-right eye, nose tip, image-left and image-right
# mouth corners, in a 112x112 crop.
TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
])


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 matrix M minimizing sum ||M [src_i, 1] - dst_i||^2 over rotations,
    uniform scales and translations (Umeyama 1991)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (s ** 2).sum(axis=1).mean()
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]])


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample ``img`` at real coordinates (x, y); outside points take the
    value of the nearest border pixel."""
    h, w = img.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(int), w - 2) if w > 1 else np.zeros_like(x, int)
    y0 = np.minimum(np.floor(y).astype(int), h - 2) if h > 1 else np.zeros_like(y, int)
    fx, fy = x - x0, y - y0
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def warp_similarity(gray: np.ndarray, M: np.ndarray, size: int) -> np.ndarray:
    """Resample ``gray`` into a ``size`` x ``size`` image through the 2x3
    similarity ``M`` (source -> output coordinates).

    When the face is much larger than the crop, the source region is
    first shrunk with area averaging (``imaging.resize``), so that the
    bilinear sampling does not alias.
    """
    gray = np.asarray(gray, dtype=np.float64)
    A, t = M[:, :2], M[:, 2]
    scale = np.sqrt(abs(np.linalg.det(A)))        # output pixels per source pixel
    Ainv = np.linalg.inv(A)

    if scale < 0.7:
        # Source bounding box of the output square, with a margin.
        corners = np.array([[0, 0], [size, 0], [0, size], [size, size]], float)
        src = (corners - t) @ Ainv.T
        x0, y0 = np.floor(src.min(0)).astype(int) - 2
        x1, y1 = np.ceil(src.max(0)).astype(int) + 2
        H, W = gray.shape
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            region = gray[y0:y1, x0:x1]
            nw = max(2, int(round((x1 - x0) * scale)))
            nh = max(2, int(round((y1 - y0) * scale)))
            small = resize(region, (nw, nh))
            # source -> small:  p' = (p - offset) * (n / extent)
            sx, sy = nw / (x1 - x0), nh / (y1 - y0)
            S = np.diag([sx, sy])
            # output = A p + t = A (S^-1 p' + offset) + t
            A2 = A @ np.linalg.inv(S)
            t2 = t + A @ np.array([x0, y0], float)
            gray, A, t, Ainv = small, A2, t2, np.linalg.inv(A2)

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    out_pts = np.stack([xs.ravel(), ys.ravel()], axis=1)    # same convention as OpenCV
    src = (out_pts - t) @ Ainv.T
    return _bilinear(gray, src[:, 0], src[:, 1]).reshape(size, size)


def align_face(gray: np.ndarray, landmarks: np.ndarray, size: int = ALIGNED_SIZE) -> np.ndarray:
    """The aligned ``size`` x ``size`` uint8 crop of the face whose five
    landmarks (image-left eye, image-right eye, nose, image-left and
    image-right mouth corners) are ``landmarks`` (5, 2)."""
    M = similarity_transform(landmarks, TEMPLATE * (size / 112.0))
    return np.clip(np.rint(warp_similarity(gray, M, size)), 0, 255).astype(np.uint8)
