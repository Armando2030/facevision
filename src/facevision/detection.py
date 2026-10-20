"""Face detection with YuNet, a pre-trained detector distributed with OpenCV.

Detection is the one part of the pipeline that is not implemented here.
YuNet is a small convolutional network (Wu, Peng, Yu, 2023) that returns,
for each face, a box, a confidence and five landmarks (both eyes, the
nose tip and both mouth corners), which ``alignment.py`` uses to align
the face. The model file (``assets/face_detection_yunet_2023mar.onnx``,
BSD-3 license, from github.com/ShiqiYu/libfacedetection.train) runs
through ``cv2.FaceDetectorYN`` (OpenCV >= 4.5.4).

    W. Wu, H. Peng, S. Yu. "YuNet: A Tiny Millisecond-level Face
    Detector". Machine Intelligence Research, 20(5), 2023.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

MODEL = os.path.join(os.path.dirname(__file__), "assets", "face_detection_yunet_2023mar.onnx")

SCORE_THRESHOLD = 0.7   # minimum confidence of a face
NMS_THRESHOLD = 0.3     # overlap above which two boxes are the same face
MIN_SIZE = 40           # faces narrower than this (pixels) are ignored
SMALL_IMAGE = 160       # images smaller than this are enlarged before detecting
MAX_SIDE = 640          # larger images are shrunk to this for detection (landmarks
                        # are mapped back, so alignment still uses full resolution)

_detector = None


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]      # (x, y, w, h)
    landmarks: np.ndarray               # (5, 2): image-left eye, image-right eye, nose,
                                        #         image-left and image-right mouth corners
    score: float


def _get_detector():
    global _detector
    if _detector is None:
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(f"OpenCV {cv2.__version__} no tiene FaceDetectorYN; "
                               "actualiza con: pip install -U opencv-python")
        try:            # silence an informational warning some OpenCV builds print here
            level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            level = None
        try:
            _detector = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), SCORE_THRESHOLD,
                                                  NMS_THRESHOLD, 5000)
        finally:
            if level is not None:
                cv2.utils.logging.setLogLevel(level)
    return _detector


def _as_bgr(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img[..., :3])


def detect(image: np.ndarray) -> list[Detection]:
    """Every face in ``image`` (BGR or gray), largest first.

    Small images, such as an already cropped face, are enlarged and given
    a border first, because the network needs some context around a face.
    """
    import cv2

    img = _as_bgr(image)
    h, w = img.shape[:2]
    scale, pad = 1.0, 0
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    elif min(h, w) < SMALL_IMAGE:
        scale = SMALL_IMAGE * 2 / min(h, w)
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_CUBIC)
        pad = int(0.25 * max(img.shape[:2]))
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    det = _get_detector()
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    out = []
    for f in [] if faces is None else faces:
        box = (f[:4] - [pad, pad, 0, 0]) / scale
        lm = (f[4:14].reshape(5, 2) - pad) / scale
        x, y, bw, bh = box
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
        # MIN_SIZE is in original pixels, except for an enlarged small image,
        # where the whole image may be a single small face.
        min_w = MIN_SIZE / scale if scale > 1 else MIN_SIZE
        if x1 - x0 < min_w or y1 <= y0:
            continue
        out.append(Detection((x0, y0, x1 - x0, y1 - y0), lm, float(f[14])))
    return sorted(out, key=lambda d: d.box[2] * d.box[3], reverse=True)


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Boxes (x, y, w, h) of every face, largest first."""
    return [d.box for d in detect(image)]


def detect_face(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """The largest face's box, or None."""
    faces = detect(image)
    return faces[0].box if faces else None
