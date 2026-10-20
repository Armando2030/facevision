"""Image operations in plain NumPy: grayscale, resizing, histogram
equalization, Gaussian blur and Tan-Triggs illumination normalization.

These replace the corresponding OpenCV calls. OpenCV is used only for
reading the camera, decoding image files, showing windows and running
the pre-trained face detector (YuNet).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# ITU-R BT.601 luma weights, in BGR order (OpenCV's channel order).
_LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR (H, W, 3) -> grayscale (H, W) uint8. Gray input is returned as is."""
    if img.ndim == 2:
        return img
    gray = img[..., :3].astype(np.float32) @ _LUMA_BGR
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


@lru_cache(maxsize=256)
def _resample_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix that resamples a 1-D signal of length n_in.

    Shrinking uses area averaging: each output pixel is the mean of the
    input interval it covers, counting fractional overlap at the edges,
    which avoids aliasing. Enlarging interpolates linearly between the
    two nearest input pixels.
    """
    M = np.zeros((n_out, n_in))
    scale = n_in / n_out
    if n_out <= n_in:
        for i in range(n_out):
            start, end = i * scale, (i + 1) * scale
            for j in range(int(np.floor(start)), min(int(np.ceil(end)), n_in)):
                overlap = min(end, j + 1) - max(start, j)
                if overlap > 0:
                    M[i, j] = overlap / scale
    else:
        for i in range(n_out):
            src = min(max((i + 0.5) * scale - 0.5, 0.0), n_in - 1.0)
            j0 = int(np.floor(src))
            j1 = min(j0 + 1, n_in - 1)
            t = src - j0
            M[i, j0] += 1.0 - t
            M[i, j1] += t
    return M


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D image to ``size = (width, height)``.

    Separable: rows and columns are resampled independently with two
    small matrices, ``out = Ry @ img @ Rx.T``. uint8 in, uint8 out.
    """
    if img.ndim != 2:
        raise ValueError("resize expects a 2-D grayscale image")
    w_out, h_out = int(size[0]), int(size[1])
    h_in, w_in = img.shape
    if min(w_out, h_out, h_in, w_in) <= 0:
        raise ValueError(f"cannot resize {img.shape} to {size}")
    out = _resample_matrix(h_in, h_out) @ img.astype(np.float64) @ _resample_matrix(w_in, w_out).T
    if img.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


def equalize_hist(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Histogram equalization of a uint8 image.

    Maps each gray level through the normalized cumulative histogram so
    the output spans 0-255 with a roughly flat histogram. Same formula
    as OpenCV: the lowest occupied level maps to 0, the highest to 255.

    With ``mask`` (boolean, same shape), the histogram is computed from
    the pixels inside the mask only and the pixels outside are set to 0,
    so whatever lies outside cannot change how the inside is mapped.
    """
    if img.dtype != np.uint8:
        raise ValueError("equalize_hist expects uint8")
    values = img if mask is None else img[mask]
    hist = np.bincount(values.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    first = hist[np.nonzero(hist)[0][0]]
    if values.size == first:                    # a single gray level: nothing to spread
        out = img.copy()
    else:
        lut = np.rint((cdf - first) * (255.0 / (values.size - first)))
        out = np.clip(lut, 0, 255).astype(np.uint8)[img]
    if mask is not None:
        out[~mask] = 0
    return out


@lru_cache(maxsize=16)
def _gaussian_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma + 0.5))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-x ** 2 / (2 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a 2-D float image (reflected borders)."""
    k = _gaussian_kernel(float(sigma))
    r = len(k) // 2
    h, w = img.shape
    p = np.pad(np.asarray(img, dtype=np.float64), r, mode="reflect")
    p = sum(k[i] * p[i : i + h, :] for i in range(len(k)))
    return sum(k[i] * p[:, i : i + w] for i in range(len(k)))


def tan_triggs(img: np.ndarray, gamma: float = 0.2, sigma0: float = 1.0, sigma1: float = 2.0,
               alpha: float = 0.1, tau: float = 10.0) -> np.ndarray:
    """Illumination normalization of Tan and Triggs (2010).

    1. Gamma correction (``gamma``): compresses the dynamic range, so that
       shadows and highlights weigh less.
    2. Difference of Gaussians (``sigma0``, ``sigma1``): removes the slow
       shading that a light source casts across the face and the finest
       noise, keeping the structure in between (eyes, brows, mouth, nose).
    3. Two-stage contrast equalization with a robust mean (``alpha``) and
       a tanh compression to (-``tau``, ``tau``), so that a few extreme
       pixels cannot dominate.

    X. Tan, B. Triggs. "Enhanced local texture feature sets for face
    recognition under difficult lighting conditions". IEEE Transactions
    on Image Processing, 19(6), 2010.
    """
    I = (np.asarray(img, dtype=np.float64) / 255.0 + 1e-3) ** gamma
    I = gaussian_blur(I, sigma0) - gaussian_blur(I, sigma1)
    I = I / (np.mean(np.abs(I) ** alpha) ** (1 / alpha) + 1e-12)
    I = I / (np.mean(np.minimum(tau, np.abs(I)) ** alpha) ** (1 / alpha) + 1e-12)
    return tau * np.tanh(I / tau)
