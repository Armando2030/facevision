"""High-level API: register people and recognize them, from Python.

    from facevision import FaceRecognizer

    rec = FaceRecognizer("mi_base")             # carpeta donde se guarda todo
    rec.capture("armando")                      # fotos con la cámara
    rec.add_person("ana", ["ana1.jpg", "ana2.jpg"])
    for face in rec.recognize("foto.jpg"):
        print(face.name, face.distance, face.box)
    rec.live()                                  # cámara en vivo

The model is trained automatically the first time it is needed and
again whenever the registered photos change, so there is no separate
training step to remember. ``train()`` forces it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass

import numpy as np

from .models import METHODS, FaceModel, train_model
from .preprocessing import IMG_EXTS, count_identities, extract_faces, load_dataset, save_capture


@dataclass(frozen=True)
class Face:
    """One face found in an image."""

    name: str | None                    # None: not a registered person
    distance: float                     # to the closest registered photo; lower is closer
    box: tuple[int, int, int, int]      # (x, y, w, h) in the image

    @property
    def known(self) -> bool:
        return self.name is not None


def _as_image(image) -> np.ndarray:
    """A path or an array (BGR or gray) -> array."""
    if isinstance(image, (str, os.PathLike)):
        import cv2

        img = cv2.imread(os.fspath(image))
        if img is None:
            raise FileNotFoundError(f"no se pudo abrir la imagen {image}")
        return img
    img = np.asarray(image)
    if img.ndim not in (2, 3) or img.size == 0:
        raise ValueError(f"imagen no válida, forma {img.shape}")
    return img


def _check_name(name: str) -> str:
    name = str(name).strip()
    if not name or name != os.path.basename(name) or name.startswith("."):
        raise ValueError(f"nombre no válido: {name!r}")
    return name


class FaceRecognizer:
    """Registers people and recognizes them.

    Parameters
    ----------
    folder : str
        Where everything is kept: ``folder/faces/<name>/`` holds each
        person's photos and ``folder/model.npz`` the trained model. It is
        created if missing; an existing one is reused as is.
    method : {"combined", "lbp", "fisherfaces", "eigenfaces"}
        "combined" (default) adds the Fisherfaces and LBP distances; it was
        the most accurate in every test (see the README).
    threshold : float, optional
        Rejection threshold. By default the one calibrated at training.
    generic : bool
        Learn the face space with the bundled generic faces too
        (required by every method except eigenfaces).
    """

    def __init__(self, folder: str = "facevision_data", method: str = "combined",
                 threshold: float | None = None, generic: bool = True):
        if method not in METHODS:
            raise ValueError(f"método desconocido: {method!r}; usa uno de {', '.join(METHODS)}")
        self.folder = folder
        self.faces_dir = os.path.join(folder, "faces")
        self.model_path = os.path.join(folder, "model.npz")
        self.method, self.generic = method, generic
        self._threshold = threshold
        self._model: FaceModel | None = None
        os.makedirs(self.faces_dir, exist_ok=True)

    # ------------------------------------------------------------- people

    @property
    def people(self) -> dict[str, int]:
        """``{name: number of photos}`` of everyone registered."""
        return count_identities(self.faces_dir)

    def add_person(self, name: str, images) -> int:
        """Register ``name`` from photos: paths or arrays, one or a list.

        The largest face of each photo is stored; photos without a face
        are skipped. Adding photos to someone already registered keeps
        the previous ones. Returns how many were stored.
        """
        name = _check_name(name)
        if isinstance(images, (str, os.PathLike, np.ndarray)):
            images = [images]
        saved = sum(save_capture(self.faces_dir, name, _as_image(img)) is not None
                    for img in images)
        if saved == 0:
            raise ValueError(f"no se encontró ninguna cara en las fotos de {name!r}")
        return saved

    def add_folder(self, folder: str) -> dict[str, int]:
        """Register everyone in ``folder/<name>/<photos>`` at once.
        Returns ``{name: photos stored}``."""
        out = {}
        for name in sorted(os.listdir(folder)):
            d = os.path.join(folder, name)
            if name.startswith(".") or not os.path.isdir(d):
                continue
            files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                     if f.lower().endswith(IMG_EXTS)]
            if files:
                out[name] = self.add_person(name, files)
        return out

    def capture(self, name: str, shots: int = 20, camera: int = 0) -> int:
        """Register ``name`` with the camera: a window opens and stores a
        photo every 0.4 s until ``shots`` (``q`` stops early). Run it
        again under other lighting to add more. Returns how many were stored."""
        from .camera import capture_faces

        return capture_faces(self.faces_dir, _check_name(name), shots, camera)

    def remove_person(self, name: str) -> None:
        """Delete ``name`` and all their photos."""
        d = os.path.join(self.faces_dir, _check_name(name))
        if not os.path.isdir(d):
            raise KeyError(f"{name!r} no está registrado")
        shutil.rmtree(d)
        self._model = None

    # -------------------------------------------------------------- model

    def _fingerprint(self) -> str:
        """Changes whenever a registered photo is added, removed or replaced,
        or the training settings change."""
        h = hashlib.sha1(f"{self.method}|{self.generic}".encode())
        for root, _, files in sorted(os.walk(self.faces_dir)):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS) and not f.startswith("."):
                    st = os.stat(os.path.join(root, f))
                    rel = os.path.relpath(os.path.join(root, f), self.faces_dir)
                    h.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}".encode())
        return h.hexdigest()

    def train(self) -> dict:
        """Train now with everyone registered and save the model.

        Returns a summary: people, photos, threshold and any note."""
        if not self.people:
            raise RuntimeError("no hay nadie registrado; usa capture() o add_person() primero")
        X, y, times = load_dataset(self.faces_dir, with_times=True)
        if len(X) == 0:
            raise RuntimeError(f"no se encontró ninguna cara en las fotos de {self.faces_dir}")
        model, note = train_model(X, y, self.method, generic=self.generic, times=times)
        model.save(self.model_path, meta={"fingerprint": self._fingerprint()})
        model.meta_ = {"fingerprint": self._fingerprint()}
        self._model = model
        cal = model.calibration_ or {}
        return {"people": self.people, "photos": len(X), "threshold": model.suggested_threshold(),
                "rule": cal.get("rule"), "sessions": cal.get("sessions"),
                "accepts_registered": cal.get("genuine_accept"), "note": note}

    @property
    def model(self) -> FaceModel:
        """The trained model, (re)trained first if the photos changed."""
        fp = self._fingerprint()
        if self._model is not None and self._model.meta_.get("fingerprint") == fp:
            return self._model
        if os.path.exists(self.model_path):
            try:
                m = FaceModel.load(self.model_path)
                if m.meta_.get("fingerprint") == fp:
                    self._model = m
                    return m
            except ValueError:          # saved by an older version
                pass
        self.train()
        return self._model

    @property
    def threshold(self) -> float:
        """Distance above which a face is reported as unknown."""
        return self._threshold if self._threshold is not None else self.model.suggested_threshold()

    @threshold.setter
    def threshold(self, value: float | None) -> None:
        self._threshold = value

    # -------------------------------------------------------- recognizing

    def recognize(self, image) -> list[Face]:
        """Every face in ``image`` (path or array), largest first."""
        model, threshold = self.model, self.threshold
        faces = extract_faces(_as_image(image))
        preds = model.predict_many([c for c, _ in faces], threshold=threshold)
        return [Face(None if name is None else str(name), float(dist), tuple(int(v) for v in box))
                for (name, dist), (_, box) in zip(preds, faces)]

    def draw(self, image, faces: list[Face] | None = None) -> np.ndarray:
        """A copy of ``image`` with each face boxed and labeled
        (green: registered, red: unknown). Recognizes it if ``faces`` is None."""
        from .camera import draw_faces

        img = _as_image(image).copy()
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        faces = self.recognize(img) if faces is None else faces
        draw_faces(img, [(f.box, f.name, f.distance) for f in faces])
        return img

    def live(self, camera: int = 0) -> None:
        """Recognize from the camera in a window until ``q`` is pressed."""
        from .camera import live

        model, threshold = self.model, self.threshold

        def predict(frame):
            faces = extract_faces(frame)
            preds = model.predict_many([c for c, _ in faces], threshold=threshold)
            return [(box, name, dist) for (_, box), (name, dist) in zip(faces, preds)]

        live(predict, camera)

    def __repr__(self) -> str:
        people = ", ".join(f"{k} ({v})" for k, v in self.people.items()) or "nadie"
        return f"FaceRecognizer({self.folder!r}, {self.method}, registrados: {people})"
