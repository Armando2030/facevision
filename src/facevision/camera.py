"""Camera loops shared by FaceRecognizer and the command line.

OpenCV is used here only to read the camera and to show a window.
"""

from __future__ import annotations

import time
from collections import Counter, deque

import numpy as np

from .preprocessing import extract_face, store_crop

CAPTURE_INTERVAL_S = 0.4    # seconds between two stored captures

# A face is drawn only after it shows up in the same place for several
# frames in a row, which drops detector false positives lasting one frame.
MIN_FRAMES = 3      # frames in a row before a face is shown
MAX_MISSES = 3      # frames without it before it is forgotten
VOTE_FRAMES = 7     # the name shown is the most voted over these frames


def open_camera(index: int = 0):
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir la cámara {index}. Prueba con otra (camera=1), "
                           "o revisa que ninguna otra aplicación la esté usando.")
    return cap


def read_frame(cap, retries: int = 30):
    """Read a frame, tolerating the warm-up failures AVFoundation (macOS)
    produces for the first reads after opening the device. None if the
    camera gives nothing."""
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.05)
    return None


def _iou(a, b) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class Tracker:
    """Follows each face across frames and smooths its name by voting."""

    def __init__(self):
        self.tracks = []

    def update(self, detections):
        """``detections``: list of (box, label, dist) for the current frame.
        Returns the faces to draw, as (box, label, dist)."""
        for t in self.tracks:
            t["seen"] = False
        for box, label, dist in sorted(detections, key=lambda d: -d[0][2] * d[0][3]):
            free = [t for t in self.tracks if not t["seen"]]
            best = max(free, key=lambda t: _iou(t["box"], box), default=None)
            if best is None or _iou(best["box"], box) < 0.3:
                best = {"hits": 0, "votes": deque(maxlen=VOTE_FRAMES),
                        "dists": deque(maxlen=VOTE_FRAMES)}
                self.tracks.append(best)
            best.update(box=box, seen=True, misses=0, hits=best["hits"] + 1)
            best["votes"].append(label)
            best["dists"].append(dist)
        for t in self.tracks:
            if not t["seen"]:
                t["misses"] += 1
        self.tracks = [t for t in self.tracks if t["misses"] <= MAX_MISSES]

        shown = []
        for t in self.tracks:
            if t["seen"] and t["hits"] >= MIN_FRAMES:
                label = Counter(t["votes"]).most_common(1)[0][0]
                d = [x for l, x in zip(t["votes"], t["dists"]) if l == label]
                shown.append((t["box"], label, float(np.median(d))))
        return shown


def draw_faces(frame, faces) -> None:
    """Draw (box, label, dist) triples on ``frame`` in place."""
    import cv2

    for (x, y, w, h), label, dist in faces:
        color = (0, 255, 0) if label is not None else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label or 'desconocido'} ({dist:.2f})", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def capture_faces(faces_dir: str, name: str, shots: int = 20, camera: int = 0) -> int:
    """Store ``shots`` face crops of ``name`` from the camera, one every
    CAPTURE_INTERVAL_S. A window shows progress; ``q`` stops early.
    Returns how many were stored."""
    import cv2

    cap = open_camera(camera)
    saved, last = 0, 0.0
    try:
        while saved < shots:
            frame = read_frame(cap)
            if frame is None:
                raise RuntimeError("la cámara no entrega imágenes; prueba otra cámara o revisa "
                                   "el permiso de cámara de tu terminal.")
            crop, box = extract_face(frame)     # aligned before anything is drawn on the frame
            if box is not None:
                if time.time() - last > CAPTURE_INTERVAL_S:
                    store_crop(faces_dir, name, crop)
                    saved += 1
                    last = time.time()
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}: {saved}/{shots}  (q para terminar)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("facevision - captura", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return saved


def live(predict, camera: int = 0, window: str = "facevision") -> None:
    """Show the camera with every face recognized until ``q`` is pressed.

    ``predict(frame)`` returns a list of (box, label, dist).
    """
    import cv2

    cap = open_camera(camera)
    tracker = Tracker()
    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                raise RuntimeError("la cámara no entrega imágenes; prueba otra cámara o revisa "
                                   "el permiso de cámara de tu terminal.")
            draw_faces(frame, tracker.update(predict(frame)))
            cv2.putText(frame, "q para salir", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
