"""From a camera frame or an image file to aligned face crops.

    frame -> detect (YuNet: box + 5 landmarks) -> align (alignment.py) -> 112x112 gray crop

Aligned crops are what gets stored for each registered person and what
the recognition methods receive; each method computes its own features
from them (features.py).
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

import numpy as np

from .alignment import ALIGNED_SIZE, align_face
from .detection import detect
from .imaging import to_gray

FACE_SIZE = (ALIGNED_SIZE, ALIGNED_SIZE)        # (w, h) of every crop

# Saved in model.npz; a model built with another preprocessing is refused.
PREPROCESSING = "yunet-aligned-v2"

IMG_EXTS = (".pgm", ".png", ".jpg", ".jpeg", ".bmp")


def read_image(path: str) -> np.ndarray | None:
    """Read an image file (BGR, or gray if it has one channel), or None."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = (img / max(1, img.max()) * 255).astype(np.uint8)
    return img


def extract_faces(frame: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Every face in a frame as ``(aligned crop, box)`` pairs, largest first."""
    gray = to_gray(frame)
    return [(align_face(gray, d.landmarks), d.box) for d in detect(frame)]


def extract_face(frame: np.ndarray):
    """The largest face as ``(aligned crop, box)``, or ``(None, None)``."""
    faces = extract_faces(frame)
    return faces[0] if faces else (None, None)


def store_crop(faces_dir: str, name: str, crop: np.ndarray) -> str:
    """Write an aligned crop to ``faces_dir/name/<timestamp>.png``."""
    import cv2

    out = os.path.join(faces_dir, name)
    os.makedirs(out, exist_ok=True)
    stamp = int(time.time() * 1000)
    path = os.path.join(out, f"{stamp}.png")
    k = 1
    while os.path.exists(path):           # several crops in the same millisecond
        path = os.path.join(out, f"{stamp}_{k}.png")
        k += 1
    cv2.imwrite(path, crop)
    return path


def save_capture(faces_dir: str, name: str, frame: np.ndarray):
    """Align the largest face of ``frame`` and store it under
    ``faces_dir/name/``. Returns its box, or None if there is no face."""
    crop, box = extract_face(frame)
    if crop is not None:
        store_crop(faces_dir, name, crop)
    return box


def as_aligned(img: np.ndarray) -> np.ndarray | None:
    """An image file's content as an aligned crop.

    A 112x112 image is taken as a crop already aligned by this library
    (what ``store_crop`` writes). Anything else is a photo: its largest
    face is detected and aligned, or None if there is none.
    """
    if img.shape[:2] == (ALIGNED_SIZE, ALIGNED_SIZE):
        return to_gray(img)
    crop, _ = extract_face(img)
    return crop


def count_identities(root: str) -> dict[str, int]:
    """``{identity: number of images}`` for a ``root/<identity>/`` tree."""
    if not os.path.isdir(root):
        return {}
    out = {}
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if name.startswith(".") or not os.path.isdir(d):
            continue
        out[name] = sum(1 for f in os.listdir(d)
                        if not f.startswith(".") and f.lower().endswith(IMG_EXTS))
    return out


def photo_time(path: str) -> float:
    """When a photo was taken, in seconds: from the name ``store_crop``
    gives it (milliseconds since 1970), or else the file's modification time."""
    stem = os.path.basename(path).split(".")[0].split("_")[0]
    if stem.isdigit() and int(stem) > 10 ** 12:
        return int(stem) / 1000.0
    return os.path.getmtime(path)


def load_dataset(root: str, with_times: bool = False):
    """Load a ``root/<identity>/<image>`` tree as aligned crops.

    Returns ``(crops, labels)``: crops is (n, 112, 112) uint8 and labels
    holds the directory names. Images where no face is found are skipped.
    With ``with_times``, also the time of each photo (``photo_time``).
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset directory not found: {root}")
    crops: list[np.ndarray] = []
    labels: list[str] = []
    times: list[float] = []
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if person.startswith(".") or not os.path.isdir(pdir):
            continue
        for fname in sorted(os.listdir(pdir)):
            if fname.startswith(".") or not fname.lower().endswith(IMG_EXTS):
                continue
            path = os.path.join(pdir, fname)
            img = read_image(path)
            crop = None if img is None else as_aligned(img)
            if crop is not None:
                crops.append(crop)
                labels.append(person)
                times.append(photo_time(path))
    out = np.array(crops, dtype=np.uint8) if crops else np.empty((0, *FACE_SIZE[::-1]), np.uint8)
    return (out, labels, np.array(times)) if with_times else (out, labels)


GENERIC_FACES = os.path.join(os.path.dirname(__file__), "assets", "generic_faces.npz")


@lru_cache(maxsize=1)
def load_generic() -> tuple[np.ndarray, tuple[str, ...]]:
    """The bundled generic faces: aligned crops of people from ORL, Yale
    and Extended Yale B (scripts/build_generic_faces.py). They are used to
    learn the face space, never as people to recognize."""
    z = np.load(GENERIC_FACES, allow_pickle=False)
    return z["faces"], tuple(str(s) for s in z["labels"])
