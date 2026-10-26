"""Metric helpers shared by the benchmark scripts (the wrapper no longer scores anything)."""

from __future__ import annotations

import numpy as np
from sklearn import metrics
from sklearn.linear_model import RidgeClassifierCV, RidgeCV


def class_scores(est, X):
    """(scores, classes) of a fitted classifier: predict_proba when available, otherwise decision_function
    turned into probabilities (sigmoid for two classes, softmax for more)."""
    classes = np.asarray(est.classes_)
    if hasattr(est, "predict_proba"):
        try:
            return np.asarray(est.predict_proba(X)), classes
        except (AttributeError, NotImplementedError):
            pass
    d = np.asarray(est.decision_function(X), dtype=float)
    if d.ndim == 1:  # binary: score of classes[1]
        p1 = 1 / (1 + np.exp(-d))
        return np.column_stack([1 - p1, p1]), classes
    d = d - d.max(axis=1, keepdims=True)
    e = np.exp(d)
    return e / e.sum(axis=1, keepdims=True), classes


def auroc(y_true, scores, classes) -> float:
    """Macro one-vs-rest AUROC over the classes seen in training (binary: AUROC of classes[1]); test rows whose
    class the model never saw are left out. NaN when the test part has a single class."""
    y = np.asarray(y_true).astype(str)
    classes = np.asarray(classes).astype(str)
    keep = np.isin(y, classes)
    y, scores = y[keep], np.asarray(scores)[keep]
    if len(np.unique(y)) < 2:
        return float("nan")
    if len(classes) == 2:
        return float(metrics.roc_auc_score(y == classes[1], scores[:, 1]))
    present = [k for k, c in enumerate(classes) if (y == c).any()]
    aucs = [metrics.roc_auc_score(y == classes[k], scores[:, k]) for k in present if (y != classes[k]).any()]
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate(kind: str, y_true, y_pred, scores=None, classes=None) -> dict:
    """Classification: AUROC (the headline metric; needs ``scores`` and ``classes`` from :func:`class_scores`), plus
    accuracy, balanced accuracy and macro-F1. Regression: R², MAE, RMSE."""
    if kind == "clf":
        out = {
            "n": len(y_true),
            "accuracy": metrics.accuracy_score(y_true, y_pred),
            "balanced_accuracy": metrics.balanced_accuracy_score(y_true, y_pred),
            "macro_f1": metrics.f1_score(y_true, y_pred, average="macro"),
        }
        if scores is not None:
            out["auroc"] = auroc(y_true, scores, classes)
        return out
    return {
        "n": len(y_true),
        "r2": metrics.r2_score(y_true, y_pred),
        "mae": metrics.mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
    }


ALPHAS = np.logspace(-3, 3, 10)  # 10 values from 1e-3 to 1e3 (13 before 2026-09-24)


class _SVDFallback:
    """RidgeCV / RidgeClassifierCV whose leave-one-out fit falls back to the eigendecomposition of X X^T when LAPACK's SVD of X
    does not converge (a LinAlgError on a few embedding matrices, 2026-09-30); the same ridge fit by another decomposition, so
    only fits that used to fail change."""

    def fit(self, X, y, sample_weight=None, **kw):
        try:
            return super().fit(X, y, sample_weight=sample_weight, **kw)
        except np.linalg.LinAlgError:
            self.gcv_mode = "eigen"
            return super().fit(X, y, sample_weight=sample_weight, **kw)


class RidgeCVFallback(_SVDFallback, RidgeCV):
    pass


class RidgeClassifierCVFallback(_SVDFallback, RidgeClassifierCV):
    pass


def linear(kind: str, balanced: bool = False):
    """The linear model used everywhere: RidgeCV (regression) / RidgeClassifierCV (classification),
    alpha chosen by efficient leave-one-out CV over 10 values from 1e-3 to 1e3. ``balanced`` weights
    classes inversely to their frequency (used by the dataset screening, where imbalance is common)."""
    if kind == "clf":
        return RidgeClassifierCVFallback(alphas=ALPHAS, class_weight="balanced" if balanced else None)
    return RidgeCVFallback(alphas=ALPHAS)
