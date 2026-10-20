"""Recognition methods: Eigenfaces, Fisherfaces, LBP and their combination.

All of them map a face to a vector (an *embedding*) and compare
embeddings with the cosine distance; a face is recognized as the
registered photo at the smallest distance, if that distance is below the
rejection threshold.

eigenfaces   PCA of the pixel features (Turk and Pentland 1991).
fisherfaces  PCA, then LDA, of the pixel features (Belhumeur et al. 1997).
lbp          sqrt of the LBP histograms (Tan-Triggs normalized), whitened
             PCA, then LDA.
combined     fisherfaces + lbp: the two distances, each divided by its
             typical value between different people, are added. Default.

PCA is always learned from the generic faces (people who are never
recognized), so it describes how faces vary in general. LDA is learned
from the generic faces and the registered people together: it looks for
the directions that separate people while ignoring what changes within a
person (lighting, expression), and the generic faces supply many
examples of those changes.

References
----------
[1] M. Turk, A. Pentland. "Eigenfaces for Recognition". Journal of
    Cognitive Neuroscience, 3(1), 1991.
[2] P. Belhumeur, J. Hespanha, D. Kriegman. "Eigenfaces vs. Fisherfaces:
    Recognition Using Class Specific Linear Projection". IEEE TPAMI,
    19(7), 1997.
[3] T. Ahonen, A. Hadid, M. Pietikainen. "Face description with local
    binary patterns: application to face recognition". IEEE TPAMI,
    28(12), 2006.
"""

from __future__ import annotations

import numpy as np

from .features import compute, compute_cached
from .preprocessing import PREPROCESSING

EPS = 1e-12
METHODS = ("combined", "lbp", "fisherfaces", "eigenfaces")

# Rejection thresholds used with fewer than 5 registered people, where the
# threshold cannot be measured from the user's own data. Each is the
# equal-error point on ORL (different people, same camera and lighting),
# the hardest case for rejecting strangers; see the README.
DEFAULT_THRESHOLDS = {"combined": 0.45, "lbp": 0.21, "fisherfaces": 0.17, "eigenfaces": 0.17}


# ------------------------------------------------------------------ algebra

def cosine_distances(G: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """1 - cos between every row of Q and every row of G: shape (len(Q), len(G))."""
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + EPS)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + EPS)
    return 1.0 - Q @ G.T


def pca(X: np.ndarray, k: int):
    """Principal components of the rows of X.

    Returns ``(mean, components, variances)``: components is (d, k) with
    orthonormal columns. When there are fewer samples than dimensions
    (n << d), the (n, n) Gram matrix A A^T is decomposed instead of the
    (d, d) covariance: if v is an eigenvector of A A^T, A^T v is one of
    A^T A with the same eigenvalue [1].
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mean = X.mean(axis=0)
    A = X - mean
    k = max(1, min(k, n - 1, d))
    if n < d:
        w, V = np.linalg.eigh(A @ A.T)
        order = np.argsort(w)[::-1][:k]
        w, V = w[order], V[:, order]
        C = A.T @ V
        C /= np.linalg.norm(C, axis=0) + EPS
    else:
        w, V = np.linalg.eigh(A.T @ A)
        order = np.argsort(w)[::-1][:k]
        w, C = w[order], V[:, order]
    return mean, C, np.maximum(w, EPS) / n


def lda(P: np.ndarray, y, k: int) -> np.ndarray:
    """Fisher's linear discriminant [2]: up to ``k`` (and at most c - 1)
    directions maximizing between-class over within-class scatter.

    The generalized problem S_b v = l S_w v is solved by whitening S_w
    with ``eigh`` (with a floor on its eigenvalues) rather than
    ``eig(inv(S_w) S_b)``: S_w is badly conditioned, and the non-symmetric
    route returns complex eigenpairs.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    mu = P.mean(axis=0)
    dim = P.shape[1]
    Sw = np.zeros((dim, dim))
    Sb = np.zeros((dim, dim))
    for c in classes:
        Pc = P[y == c]
        m = Pc.mean(axis=0)
        D = Pc - m
        Sw += D.T @ D
        diff = (m - mu)[:, None]
        Sb += len(Pc) * (diff @ diff.T)
    w, U = np.linalg.eigh(Sw)
    w = np.maximum(w, max(w.max(), EPS) * 1e-10)
    Wh = U / np.sqrt(w)
    S = Wh.T @ Sb @ Wh
    S = 0.5 * (S + S.T)
    ev, V = np.linalg.eigh(S)
    order = np.argsort(ev)[::-1][: max(1, min(k, len(classes) - 1))]
    W = Wh @ V[:, order]
    return W / (np.linalg.norm(W, axis=0) + EPS)


# ------------------------------------------------------------------- parts

class Subspace:
    """One projection: features -> (sqrt) -> PCA (whitened or not) -> (LDA)."""

    def __init__(self, feature: str, pca_k: int, whiten: bool = False,
                 use_lda: bool = False, sqrt: bool = False):
        self.feature, self.pca_k, self.whiten = feature, pca_k, whiten
        self.use_lda, self.sqrt = use_lda, sqrt
        self.mean_ = self.components_ = self.var_ = self.W_ = None

    def _pre(self, F):
        F = np.asarray(F, dtype=np.float64)
        return np.sqrt(np.maximum(F, 0)) if self.sqrt else F

    def fit_pca(self, F_generic):
        self.mean_, self.components_, self.var_ = pca(self._pre(F_generic), self.pca_k)
        return self

    def project(self, F):
        P = (self._pre(F) - self.mean_) @ self.components_
        return P / np.sqrt(self.var_) if self.whiten else P

    def fit_lda(self, P, y):
        self.W_ = lda(P, y, 1000) if self.use_lda else None
        return self

    def embed(self, F):
        P = self.project(F)
        return P @ self.W_ if self.W_ is not None else P

    def state(self, i: int) -> dict:
        s = {f"p{i}_mean": self.mean_, f"p{i}_components": self.components_.astype(np.float32),
             f"p{i}_var": self.var_}
        if self.W_ is not None:
            s[f"p{i}_W"] = self.W_
        return s

    def load_state(self, z, i: int):
        self.mean_ = z[f"p{i}_mean"]
        self.components_ = z[f"p{i}_components"].astype(np.float64)
        self.var_ = z[f"p{i}_var"]
        self.W_ = z[f"p{i}_W"] if f"p{i}_W" in z.files else None
        return self


def make_parts(method: str) -> list[Subspace]:
    fisher = Subspace("pixels", 300, use_lda=True)
    lbp = Subspace("lbp", 300, whiten=True, use_lda=True, sqrt=True)
    return {"eigenfaces": [Subspace("pixels", 80)], "fisherfaces": [fisher],
            "lbp": [lbp], "combined": [fisher, lbp]}[method]


# ------------------------------------------------------------------- model

class FaceModel:
    """A trained recognizer: projections, gallery and threshold."""

    def __init__(self, method: str, parts: list[Subspace], scales=None):
        if method not in METHODS:
            raise ValueError(f"método desconocido: {method!r}")
        self.method, self.parts = method, parts
        self.scales = list(scales) if scales is not None else [1.0] * len(parts)
        self.gallery_: list[np.ndarray] = []
        self.labels_: list[str] = []
        self.threshold_: float | None = None
        self.calibration_: dict | None = None
        self.n_generic_ = 0
        self.meta_: dict[str, str] = {}

    @property
    def kinds(self) -> list[str]:
        return sorted({p.feature for p in self.parts})

    def features(self, crops) -> dict[str, np.ndarray]:
        return {k: compute(k, crops) for k in self.kinds}

    def enroll_features(self, feats: dict, labels) -> None:
        self.gallery_ = [p.embed(feats[p.feature]) for p in self.parts]
        self.labels_ = [str(s) for s in labels]

    def distances_features(self, feats: dict) -> np.ndarray:
        """Distances (n_queries, n_gallery) from feature dicts."""
        return sum(cosine_distances(G, p.embed(feats[p.feature])) / s
                   for p, G, s in zip(self.parts, self.gallery_, self.scales))

    def distances(self, crops) -> np.ndarray:
        return self.distances_features(self.features(crops))

    def suggested_threshold(self) -> float:
        return self.threshold_ if self.threshold_ is not None else DEFAULT_THRESHOLDS[self.method]

    def predict_many(self, crops, threshold: float | None = None):
        """``[(label or None, distance), ...]`` for aligned crops."""
        if not self.labels_:
            raise RuntimeError("empty gallery")
        if len(crops) == 0:
            return []
        t = self.suggested_threshold() if threshold is None else threshold
        D = self.distances(crops)
        out = []
        for row in D:
            i = int(np.argmin(row))
            d = float(row[i])
            out.append((None if d > t else self.labels_[i], d))
        return out

    def predict(self, crop, threshold: float | None = None):
        return self.predict_many([crop], threshold)[0]

    # ------------------------------------------------------- persistence

    def save(self, path: str, meta: dict | None = None) -> None:
        arrays = {}
        for i, (p, G) in enumerate(zip(self.parts, self.gallery_)):
            arrays.update(p.state(i))
            arrays[f"p{i}_gallery"] = G
        np.savez_compressed(
            path, method=self.method, scales=np.array(self.scales),
            labels=np.array(self.labels_, dtype=np.str_),
            threshold=np.nan if self.threshold_ is None else self.threshold_,
            preprocessing=PREPROCESSING, n_generic=self.n_generic_,
            **arrays, **{f"meta_{k}": str(v) for k, v in (meta or {}).items()},
        )

    @classmethod
    def load(cls, path: str) -> "FaceModel":
        z = np.load(path, allow_pickle=False)
        used = str(z["preprocessing"]) if "preprocessing" in z.files else "anterior a 2.0"
        if used != PREPROCESSING or "method" not in z.files:
            raise ValueError(f"{path} es de otra versión ({used}); hay que volver a entrenar")
        method = str(z["method"])
        parts = [p.load_state(z, i) for i, p in enumerate(make_parts(method))]
        m = cls(method, parts, z["scales"])
        m.gallery_ = [z[f"p{i}_gallery"] for i in range(len(parts))]
        m.labels_ = [str(s) for s in z["labels"]]
        if np.isfinite(z["threshold"]):
            m.threshold_ = float(z["threshold"])
        m.n_generic_ = int(z["n_generic"])
        m.meta_ = {k[5:]: str(z[k]) for k in z.files if k.startswith("meta_")}
        return m


# ---------------------------------------------------------------- training

class Trainer:
    """Fits the PCA part of a method once on the generic faces, then
    builds models for any set of registered people (cheap: LDA only).
    Calibration uses it to refit many times."""

    def __init__(self, method: str, generic_crops: np.ndarray, generic_labels):
        self.method = method
        self.yg = np.asarray(generic_labels)
        self.crops_g = generic_crops
        self.Fg = {k: compute_cached(k, generic_crops) for k in
                   {p.feature for p in make_parts(method)}} if len(generic_crops) else {}
        self.parts = make_parts(method)
        self.exclude: set[str] = set()
        for p in self.parts:
            p.fit_pca(self.Fg[p.feature])
        self.Pg = [p.project(self.Fg[p.feature]) for p in self.parts]
        self.scales = [1.0] * len(self.parts)
        if len(self.parts) > 1:
            self.scales = self._scales()

    def _scales(self):
        """Typical distance between different people for each part: the
        median over pairs of generic photos of different people, with LDA
        fitted on the generic faces."""
        rng = np.random.default_rng(0)
        idx = rng.choice(len(self.yg), size=min(400, len(self.yg)), replace=False)
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
        diff = self.yg[b][:, None] != self.yg[a][None, :]
        out = []
        for p, Pg in zip(self.parts, self.Pg):
            W = lda(Pg, self.yg, 1000) if p.use_lda else None
            E = Pg @ W if W is not None else Pg
            out.append(float(np.median(cosine_distances(E[a], E[b])[diff])))
        return out

    def same_people(self, feats: dict, labels, share: float = 0.6) -> dict[str, str]:
        """Generic identities that are the same person as someone registered
        (for example, when testing with photos from ORL or Yale).

        The generic faces alone are used as a gallery; a generic identity is
        flagged when it is the closest one for at least ``share`` of a
        registered person's photos, at a median distance below 1.5 times the
        method's default threshold (a photo processed from a different
        framing of the same picture can land somewhat above the threshold;
        a stranger lands far above it). Otherwise LDA would be asked to
        separate that person from themselves.
        """
        import copy

        if not len(self.yg):
            return {}
        parts = [copy.copy(p) for p in self.parts]
        for p, Pg in zip(parts, self.Pg):
            if p.use_lda:
                p.fit_lda(Pg, self.yg)
        g = FaceModel(self.method, parts, self.scales)
        g.enroll_features(self.Fg, self.yg)
        D = g.distances_features(feats)
        best, dist = self.yg[D.argmin(axis=1)], D.min(axis=1)
        labels = np.asarray(labels)
        out = {}
        for person in np.unique(labels):
            mine = labels == person
            ids, counts = np.unique(best[mine], return_counts=True)
            top = ids[counts.argmax()]
            if counts.max() >= share * mine.sum():
                if np.median(dist[mine][best[mine] == top]) <= 1.5 * DEFAULT_THRESHOLDS[self.method]:
                    out[str(top)] = str(person)
        return out

    def without(self, identities) -> "Trainer":
        """A trainer whose generic faces leave out ``identities`` entirely,
        PCA included (not only LDA): the PCA must not have seen the photos
        of anyone who will be registered or evaluated."""
        keep = ~np.isin(self.yg, list(identities))
        return Trainer(self.method, self.crops_g[keep], self.yg[keep])

    def build(self, feats: dict, labels, exclude=None) -> FaceModel:
        """Model for the registered people whose features are ``feats``;
        the generic identities in ``exclude`` (default: ``self.exclude``)
        are left out."""
        import copy

        labels = np.asarray(labels)
        exclude = self.exclude if exclude is None else exclude
        keep = ~np.isin(self.yg, list(exclude)) if len(exclude) else np.ones(len(self.yg), bool)
        parts = [copy.copy(p) for p in self.parts]
        for p, Pg in zip(parts, self.Pg):
            if p.use_lda:
                Pe = p.project(feats[p.feature])
                p.fit_lda(np.vstack([Pg[keep], Pe]), np.r_[self.yg[keep], labels])
        model = FaceModel(self.method, parts, self.scales)
        model.enroll_features(feats, labels)
        model.n_generic_ = int(keep.sum())
        return model


def train_model(crops, labels, method: str = "combined", generic=True, trainer: Trainer | None = None,
                times=None):
    """Train and calibrate a model for the registered people.

    ``crops``: aligned crops (n, 112, 112); ``labels``: their names.
    ``generic``: True uses the bundled generic faces; False none (only
    possible for eigenfaces, which needs no other people); a pair
    (crops, labels) other ones. ``times``: when each photo was taken
    (seconds); photos of a person more than a few seconds apart belong to
    different capture sessions, which calibration uses. Returns ``(model, note)``.
    """
    from .calibration import calibrate
    from .preprocessing import load_generic

    crops = np.asarray(crops, dtype=np.uint8)
    labels = np.asarray([str(s) for s in labels])
    if len(crops) == 0:
        raise ValueError("no hay fotos para entrenar")
    if method not in METHODS:
        raise ValueError(f"método desconocido: {method!r}")
    if generic is True:
        gc, gl = load_generic()
    elif generic is False or generic is None:
        gc, gl = np.empty((0, *crops.shape[1:]), np.uint8), ()
    else:
        gc, gl = generic
    gl = np.asarray(gl)
    note = None
    if not len(gc) and method != "eigenfaces":
        raise ValueError(f"{method} necesita las caras genéricas (generic=True)")
    if trainer is None or trainer.method != method or len(trainer.yg) != len(gl):
        if not len(gc):
            # Eigenfaces without generic faces: PCA of the registered photos themselves.
            trainer = Trainer.__new__(Trainer)
            trainer.method, trainer.yg, trainer.crops_g = method, np.empty(0, str), gc
            trainer.parts = make_parts(method)
            Fe = compute("pixels", crops)
            trainer.parts[0].fit_pca(Fe)
            trainer.Fg, trainer.Pg, trainer.scales = {}, [np.empty((0, trainer.parts[0].components_.shape[1]))], [1.0]
            trainer.exclude = set()
        else:
            trainer = Trainer(method, gc, gl)
    feats = {k: compute(k, crops) for k in {p.feature for p in trainer.parts}}
    same = trainer.same_people(feats, labels)
    if same:
        trainer = trainer.without(same)
        note = ("Se quitaron de las caras genéricas " +
                ", ".join(f"{g} (es {p})" for g, p in sorted(same.items())) + ".")
    model = trainer.build(feats, labels)
    from .calibration import sessions_from_times

    sessions = None if times is None else sessions_from_times(labels, times)
    model.calibration_ = calibrate(trainer, feats, labels, sessions)
    model.threshold_ = model.calibration_["threshold"]
    if model.calibration_["rule"] == "default":
        extra = ("El umbral no se pudo medir con tus fotos (hace falta capturar en al menos 2 "
                 f"sesiones, o registrar a 5 personas); se usa {model.threshold_:.2f}.")
        note = f"{note} {extra}" if note else extra
    return model, note
