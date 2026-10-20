"""Choose the rejection threshold from the registered photos.

Distances are always measured on photos the model did not see:

* genuine: each registered photo against a model built without it.
  When a person was captured in several sessions (photos more than
  SESSION_GAP_S seconds apart), a whole session is held out at a time,
  which imitates seeing that person again in new conditions; otherwise,
  stratified k-fold.
* impostor:
  - with 5 or more people, leave-one-person-out: each person is removed
    and their photos are compared with the others. This imitates a
    stranger in front of the same camera, so the threshold is the
    equal-error point of genuine and impostor distances.
  - with fewer, generic people left out of the model are compared with
    the registered photos. They come from other cameras, so they are a
    looser reference: the threshold is the 90th percentile of the genuine
    distances (accepts ~90% of photos in new conditions), but never above
    0.85 of the closest generic stranger (1st percentile) nor below the
    method's default.
  With a single session and fewer than 5 people, the default is used.

Rebuilding is cheap: the PCA part of every method is learned once from
the generic faces (models.Trainer) and only LDA is refitted.
"""

from __future__ import annotations

import numpy as np

from .models import DEFAULT_THRESHOLDS

MIN_IDENTITIES = 5
SESSION_GAP_S = 5.0


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray, steps: int = 400):
    """``(eer, threshold)``: where false accepts and false rejects coincide.

    FAR(t) = share of impostor distances <= t
    FRR(t) = share of genuine distances  >  t
    """
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    grid = np.linspace(0, max(genuine.max(), impostor.max()), steps)
    far = np.array([(impostor <= t).mean() for t in grid])
    frr = np.array([(genuine > t).mean() for t in grid])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(grid[i])


def sessions_from_times(labels, times, gap: float = SESSION_GAP_S) -> np.ndarray:
    """Session number of each photo: within a person, a new session starts
    after a pause of more than ``gap`` seconds between photos."""
    labels, times = np.asarray(labels), np.asarray(times, dtype=float)
    out = np.zeros(len(labels), dtype=int)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(times[idx])]
        out[idx] = np.r_[0, np.cumsum(np.diff(times[idx]) > gap)]
    return out


def _take(feats: dict, idx) -> dict:
    return {k: v[idx] for k, v in feats.items()}


def _nearest_same(model, feats_test, labels_test):
    D = model.distances_features(feats_test)
    gl = np.asarray(model.labels_)
    out = []
    for row, lab in zip(D, labels_test):
        same = gl == lab
        if same.any():
            out.append(row[same].min())
    return out


def genuine_distances(trainer, feats: dict, labels, sessions=None, folds: int = 5, seed: int = 0):
    """Genuine distances, and whether they came from held-out sessions."""
    labels = np.asarray(labels)
    out, by_session = [], False
    if sessions is not None:
        sessions = np.asarray(sessions)
        for c in np.unique(labels):
            mine = labels == c
            for s in np.unique(sessions[mine]):
                if len(np.unique(sessions[mine])) < 2:
                    break
                test = np.where(mine & (sessions == s))[0]
                train = np.where(~(mine & (sessions == s)))[0]
                out += _nearest_same(trainer.build(_take(feats, train), labels[train]),
                                     _take(feats, test), labels[test])
                by_session = True
    if by_session:
        return np.array(out), True
    rng = np.random.default_rng(seed)
    fold = np.full(len(labels), -1)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % folds
    for f in range(folds):
        test, train = np.where(fold == f)[0], np.where(fold != f)[0]
        if len(test) and len(train):
            out += _nearest_same(trainer.build(_take(feats, train), labels[train]),
                                 _take(feats, test), labels[test])
    return np.array(out), False


def impostor_distances(trainer, feats: dict, labels) -> np.ndarray:
    """Leave-one-person-out distances (strangers in front of the same camera)."""
    labels = np.asarray(labels)
    out = []
    for c in np.unique(labels):
        others, mine = np.where(labels != c)[0], np.where(labels == c)[0]
        model = trainer.build(_take(feats, others), labels[others])
        out += list(model.distances_features(_take(feats, mine)).min(axis=1))
    return np.array(out)


def generic_impostor_distances(trainer, feats: dict, labels, groups: int = 5) -> np.ndarray:
    """Generic people left out of LDA, against the registered photos."""
    from .features import compute_cached

    ids = np.unique(trainer.yg)
    if len(ids) < groups:
        return np.array([])
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    Fg = {k: compute_cached(k, trainer.crops_g) for k in feats}
    out = []
    for part in np.array_split(ids, groups):
        model = trainer.build(feats, labels, exclude=set(part) | set(trainer.exclude))
        held = np.isin(trainer.yg, part)
        out += list(model.distances_features(_take(Fg, held)).min(axis=1))
    return np.array(out)


def calibrate(trainer, feats: dict, labels, sessions=None) -> dict:
    """Returns the threshold, the rule used ("eer", "sessions" or "default"),
    whether genuine distances came from held-out sessions, and the share of
    held-out registered photos (and of strangers, when measured) accepted."""
    labels = np.asarray(labels)
    gen, by_session = genuine_distances(trainer, feats, labels, sessions)
    n_people = len(np.unique(labels))
    default = DEFAULT_THRESHOLDS[trainer.method]
    imp_acc = None
    if n_people >= MIN_IDENTITIES and gen.size:
        imp = impostor_distances(trainer, feats, labels)
        t = equal_error_rate(gen, imp)[1]
        rule, imp_acc = "eer", float(np.mean(imp <= t))
    elif by_session and gen.size and len(trainer.yg):
        gimp = generic_impostor_distances(trainer, feats, labels)
        t = float(np.percentile(gen, 90))
        if gimp.size:
            t = min(t, 0.85 * float(np.percentile(gimp, 1)))
            imp_acc = float(np.mean(gimp <= max(t, default)))
        t, rule = max(t, default), "sessions"
    else:
        t, rule = default, "default"
    return {"threshold": float(t), "rule": rule, "sessions": by_session,
            "genuine_accept": float(np.mean(gen <= t)) if gen.size else None,
            "impostor_accept": imp_acc}
