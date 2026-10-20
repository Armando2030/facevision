"""Evaluate every method on a dataset folder (``root/<person>/<photos>``).

* Identification (rank-1): half of each person's photos are registered,
  the other half must be recognized among everyone registered.
* Verification (equal error rate): 80% of the people are registered with
  half their photos; the other half are genuine queries and every photo
  of the remaining 20% is a stranger. The EER is where the share of
  strangers accepted equals the share of registered photos rejected.

Each measurement is repeated over ``seeds`` random splits. If the dataset
contains people who are also among the generic faces, they are taken out
of the generic faces first (otherwise the test would be contaminated).
"""

from __future__ import annotations

import numpy as np

from .calibration import equal_error_rate
from .features import compute
from .models import METHODS, Trainer
from .preprocessing import load_dataset, load_generic


def _half_split(y, rng, frac=0.5):
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        k = max(1, int(round(len(idx) * frac)))
        tr += list(idx[:k])
        te += list(idx[k:])
    return np.array(tr), np.array(te)


def evaluate(root: str, methods=METHODS, seeds: int = 5, log=print) -> dict:
    crops, y = load_dataset(root)
    y = np.asarray(y)
    if len(np.unique(y)) < 5:
        raise ValueError("hacen falta al menos 5 personas para evaluar")
    gc, gl = load_generic()
    gl = np.asarray(gl)
    log(f"{len(crops)} fotos, {len(np.unique(y))} personas")

    results = {}
    for method in methods:
        trainer = Trainer(method, gc, gl)
        feats = {k: compute(k, crops) for k in {p.feature for p in trainer.parts}}
        same = trainer.same_people(feats, y)
        if same:
            trainer = trainer.without(same)
            log(f"  ({method}: {len(same)} personas de la base también estaban "
                "entre las genéricas y se quitaron de ellas)")
        take = lambda idx: {k: v[idx] for k, v in feats.items()}
        rank1, eers = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            tr, te = _half_split(y, rng)
            model = trainer.build(take(tr), y[tr])
            D = model.distances_features(take(te))
            rank1.append(np.mean(np.asarray(model.labels_)[D.argmin(1)] == y[te]))

            people = rng.permutation(np.unique(y))
            unk = people[: max(1, len(people) // 5)]
            known = ~np.isin(y, unk)
            ktr, kte = _half_split(y[known], rng)
            kidx = np.where(known)[0]
            model = trainer.build(take(kidx[ktr]), y[kidx[ktr]])
            gl_ = np.asarray(model.labels_)
            Dg = model.distances_features(take(kidx[kte]))
            gen = np.where(gl_[None, :] == y[kidx[kte]][:, None], Dg, np.inf).min(axis=1)
            imp = model.distances_features(take(np.where(~known)[0])).min(axis=1)
            eers.append(equal_error_rate(gen, imp)[0])
        results[method] = {"rank1": np.array(rank1), "eer": np.array(eers)}
        r = results[method]
        log(f"  {method:12} rank-1 {r['rank1'].mean():.3f} ± {r['rank1'].std():.3f}   "
            f"EER {r['eer'].mean():5.1%} ± {r['eer'].std():.1%}")
    return results
