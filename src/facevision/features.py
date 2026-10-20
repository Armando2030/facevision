"""Feature vectors computed from aligned 112x112 crops.

pixels  Histogram-equalized gray levels inside the ellipse inscribed in
        the crop (the corners, where background shows, are set to 0),
        scaled to [0, 1]: 12,544 values. Used by Eigenfaces and Fisherfaces.
lbp     Tan-Triggs illumination normalization, then uniform LBP histograms
        (P = 8, radii 1 and 2) over an 8x8 grid of cells: 7,552 values.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from .alignment import ALIGNED_SIZE
from .imaging import equalize_hist, tan_triggs
from .lbp import lbp_histogram


def _ellipse(n: int) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    return ((xx - c) / (n / 2)) ** 2 + ((yy - c) / (n / 2)) ** 2 <= 1.0


FACE_MASK = _ellipse(ALIGNED_SIZE)


def pixel_features(crop: np.ndarray) -> np.ndarray:
    return equalize_hist(np.asarray(crop, np.uint8), mask=FACE_MASK).astype(np.float64).ravel() / 255.0


def lbp_features(crop: np.ndarray) -> np.ndarray:
    return lbp_histogram(tan_triggs(crop))


EXTRACTORS = {"pixels": pixel_features, "lbp": lbp_features}


def compute(kind: str, crops) -> np.ndarray:
    """Features of every crop, shape (n, dim)."""
    f = EXTRACTORS[kind]
    return np.array([f(c) for c in crops]) if len(crops) else np.empty((0, 0))


def _cache_dir() -> str:
    return os.environ.get("FACEVISION_CACHE",
                          os.path.join(os.path.expanduser("~"), ".cache", "facevision"))


def compute_cached(kind: str, crops: np.ndarray) -> np.ndarray:
    """Like ``compute``, but saved on disk keyed by the crops' content, so
    the generic faces are only processed once per machine."""
    key = hashlib.sha1(np.ascontiguousarray(crops).tobytes()).hexdigest()[:16]
    path = os.path.join(_cache_dir(), f"features_{kind}_{key}.npy")
    if os.path.exists(path):
        try:
            return np.load(path)
        except (OSError, ValueError):
            pass
    out = compute(kind, crops).astype(np.float32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, out)
    except OSError:
        pass
    return out
