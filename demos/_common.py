"""Shared helpers for the demo scripts."""

from __future__ import annotations

import argparse
import os
import time

from pymongo import MongoClient
from sklearn.base import BaseEstimator, ClassifierMixin


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--uri", default=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"))
    p.add_argument(
        "--embedding-model",
        default="voyage-4-nano",
        help='"voyage-4-nano" (local, default), a Hugging Face id, a Voyage API model or "hashing"',
    )
    p.add_argument("--dimension", type=int, default=None, help="Matryoshka truncation, e.g. 512")
    p.add_argument("--limit", type=int, default=None, help="use only the first N documents")
    p.add_argument("--cache-dir", default=".mongopredict_cache")
    p.add_argument(
        "--streaming",
        action="store_true",
        help="fetch documents in _id batches as needed instead of holding them all in memory",
    )
    p.add_argument("--batch-size", type=int, default=512)
    return p


def embedding_kwargs(args) -> dict | None:
    if args.embedding_model == "hashing":
        return None
    return {"dimension": args.dimension} if args.dimension else None


def markdown_table(rows: list[dict], floatfmt: str = "{:.3f}") -> str:
    cols = list(rows[0])
    fmt = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t


def collection(uri: str, database: str, name: str):
    coll = MongoClient(uri)[database][name]
    if coll.estimated_document_count() == 0:
        raise SystemExit(
            f"{database}.{name} is empty at {uri}. Load it with\n"
            f"  python -m demos.load_sample_data --uri {uri} {database}"
        )
    return coll


def xgb_estimator(kind: str):
    """XGBoost with the benchmark settings (200 trees, depth 6), or None when xgboost is not
    installed. The classifier label-encodes string targets."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    params = dict(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.5,
        tree_method="hist",
        random_state=0,
    )
    if kind == "reg":
        return XGBRegressor(**params)

    return LabelEncodedXGB(params)


class LabelEncodedXGB(ClassifierMixin, BaseEstimator):
    """XGBClassifier on label-encoded targets, so string classes work (a proper sklearn
    estimator: Pipeline and sklearn's fitted-state checks need its tags)."""

    def __init__(self, params: dict | None = None):
        self.params = params

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.model_ = XGBClassifier(**(self.params or {})).fit(X, self.le_.transform(y))
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.model_.predict(X))
