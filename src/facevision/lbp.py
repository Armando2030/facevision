"""Local Binary Patterns, in NumPy.

Each pixel is compared with P neighbours on a circle of radius R; the
P comparisons (neighbour >= centre) form a P-bit code. Only the order of
gray levels matters, so the code does not change when brightness or
contrast change monotonically (exactly for neighbours on the pixel grid,
approximately for the diagonal ones, which are interpolated). Codes with at most two 0/1 transitions
around the circle ("uniform" patterns: edges, corners, spots) get one bin
each and all the rest share a single bin: 59 bins for P = 8.

The face is divided into a grid of cells and the histogram of codes is
computed per cell, so the descriptor keeps where each texture occurs
while tolerating small shifts inside a cell.

    T. Ojala, M. Pietikainen, T. Maenpaa. "Multiresolution gray-scale and
    rotation invariant texture classification with local binary patterns".
    IEEE TPAMI, 24(7), 2002.
    T. Ahonen, A. Hadid, M. Pietikainen. "Face description with local
    binary patterns: application to face recognition". IEEE TPAMI,
    28(12), 2006.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def uniform_lut(P: int = 8) -> np.ndarray:
    """Code -> bin: uniform patterns get bins 0..P*(P-1)+1, the rest the last one."""
    lut = np.empty(2 ** P, dtype=np.int64)
    nxt = 0
    for code in range(2 ** P):
        bits = [(code >> i) & 1 for i in range(P)]
        transitions = sum(bits[i] != bits[(i + 1) % P] for i in range(P))
        if transitions <= 2:
            lut[code] = nxt
            nxt += 1
        else:
            lut[code] = -1
    lut[lut == -1] = nxt
    return lut


def lbp_codes(img: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    """Uniform LBP bin of every pixel of a 2-D image (border pixels use
    replicated neighbours)."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    pad = int(np.ceil(R)) + 1
    p = np.pad(img, pad, mode="edge")
    yy, xx = np.mgrid[0:h, 0:w]
    code = np.zeros((h, w), dtype=np.int64)
    for k in range(P):
        a = 2 * np.pi * k / P
        dy, dx = -R * np.sin(a), R * np.cos(a)
        # bilinear sample of the neighbour (exact at integer offsets)
        y = yy + pad + dy
        x = xx + pad + dx
        y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
        fy, fx = y - y0, x - x0
        v = (p[y0, x0] * (1 - fx) * (1 - fy) + p[y0, x0 + 1] * fx * (1 - fy)
             + p[y0 + 1, x0] * (1 - fx) * fy + p[y0 + 1, x0 + 1] * fx * fy)
        code |= (v >= img - 1e-9).astype(np.int64) << k
    return uniform_lut(P)[code]


def lbp_histogram(img: np.ndarray, grid: int = 8, P: int = 8, radii=(1, 2)) -> np.ndarray:
    """Concatenated per-cell histograms (each summing to 1) of uniform LBP
    codes, for every radius in ``radii``. Length: len(radii) * grid^2 * 59."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    nb = P * (P - 1) + 3
    rows = np.minimum(np.arange(h) * grid // h, grid - 1)
    cols = np.minimum(np.arange(w) * grid // w, grid - 1)
    cell = (rows[:, None] * grid + cols[None, :]).ravel()
    out = []
    for R in radii:
        codes = lbp_codes(img, P, R).ravel()
        hist = np.bincount(cell * nb + codes, minlength=grid * grid * nb).reshape(grid * grid, nb)
        out.append((hist / hist.sum(axis=1, keepdims=True)).ravel())
    return np.concatenate(out)


def chi2_distances(G: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Chi-square distance between each row of G and q (histograms)."""
    num = (G - q) ** 2
    den = G + q
    return (num / np.where(den > 0, den, 1.0)).sum(axis=1)
