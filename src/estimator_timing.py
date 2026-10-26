"""Fit and inference time of the ridge model of the predictions API against XGBoost.

For each of the 14 sample tasks (``src/tasks.py``) the documents are featurized once with the
task's columns plus the whole document embedded at 1024 dimensions (voyage-4-nano's output size;
the hashing embedder stands in for it, since fit and predict time depend on the matrix shape, not
on where the embedding came from). Then, on the same matrices:

* ridge: the predictions API's ``estimator="ridge"`` (``RidgeClassifierCV`` / ``RidgeCV`` over 13
  alphas, efficient leave-one-out CV);
* xgboost: ``XGBClassifier`` / ``XGBRegressor`` with default settings (100 trees, depth 6, hist).

Fit is timed once; inference (``predict``, plus ``predict_proba`` / ``decision_function`` for
classifiers) is the median of 3 runs on the test set.

    uv run python -m src.estimator_timing --uri mongodb://127.0.0.1:27117 --threads 16
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import time

import numpy as np
from pymongo import MongoClient

from pymongopredict import DocumentFeaturizer, MongoDataset
from pymongopredict.predictions_api.models import _estimator
from src._split import split
from src.tasks import TASKS

ROOT = pathlib.Path(__file__).resolve().parents[1]


def xgboost(kind: str, threads: int):
    import xgboost as xgb

    cls = xgb.XGBClassifier if kind == "clf" else xgb.XGBRegressor
    return cls(n_jobs=threads, tree_method="hist")


def timed(fn, repeats: int = 1) -> float:
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--uri", default="mongodb://127.0.0.1:27117")
    p.add_argument("--threads", type=int, default=16, help="threads for BLAS and XGBoost")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--out", default=str(ROOT / "results" / "estimator_timing.json"))
    a = p.parse_args()
    client = MongoClient(a.uri)
    rows = []
    for t in TASKS:
        ds = MongoDataset(client[t.database][t.collection], target=t.target, filter=t.filter)
        tr, te = split(ds, **t.split)
        fz = DocumentFeaturizer(
            columns=t.columns,
            embedding_model="hashing",
            embedding_kwargs={"n_features": a.dim},
            cache_dir=None,
        )
        Xtr, Xte = fz.fit_transform(tr), fz.transform(te)
        ytr = tr.y
        task = "classification" if t.kind == "clf" else "regression"
        if t.kind == "clf":  # xgboost needs classes 0..k-1
            classes, ytr_codes = np.unique(ytr, return_inverse=True)
        else:
            ytr_codes = ytr.astype(float)
        row = {
            "task": t.name,
            "kind": t.kind,
            "n_train": len(ytr),
            "n_test": len(te),
            "n_features": Xtr.shape[1],
            "n_classes": int(len(np.unique(ytr))) if t.kind == "clf" else None,
        }
        for name, est, y in (
            ("ridge", _estimator("ridge", task), ytr),
            ("xgboost", xgboost(t.kind, a.threads), ytr_codes),
        ):
            row[f"{name}_fit_s"] = timed(lambda: est.fit(Xtr, y))

            def infer():
                est.predict(Xte)
                if t.kind == "clf":
                    (est.predict_proba if hasattr(est, "predict_proba") else est.decision_function)(
                        Xte
                    )

            row[f"{name}_predict_s"] = timed(infer, 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = {"threads": a.threads, "dim": a.dim, "cpu_count": os.cpu_count(), "tasks": rows}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
