"""The demo helpers work with the installed sklearn (the demos themselves need a MongoDB server)."""

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from demos._common import xgb_estimator


def test_xgb_classifier_on_string_labels_in_a_pipeline():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = np.where(X[:, 0] > 0, "yes", "no")
    pipe = make_pipeline(StandardScaler(), xgb_estimator("clf")).fit(X, y)
    check_is_fitted(pipe[-1])
    assert set(pipe.predict(X)) <= {"yes", "no"} and pipe.score(X, y) > 0.9
