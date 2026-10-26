"""MongoDataset works with scikit-learn's own splitters and model selection."""

import datetime as dt

import mongomock
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupShuffleSplit,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline

from pymongopredict import DocumentFeaturizer, MongoDataset
from pymongopredict.embeddings_api._time import _as_datetime


@pytest.fixture
def coll():
    c = mongomock.MongoClient().db.items
    c.insert_many(
        [
            {"t": i, "g": i % 7, "text": f"word{i % 3}", "y": "a" if i % 3 else "b"}
            for i in range(60)
        ]
    )
    return c


@pytest.mark.parametrize("streaming", [False, True])
def test_train_test_split_returns_datasets(coll, streaming):
    ds = MongoDataset(coll, target="y", streaming=streaming, batch_size=7)
    tr, te = train_test_split(ds, test_size=0.25, stratify=ds.y, random_state=0)
    assert isinstance(tr, MongoDataset) and isinstance(te, MongoDataset)
    assert len(tr) == 45 and len(te) == 15 and not set(tr.ids) & set(te.ids)
    assert tr.streaming == streaming and (tr._docs is None) == streaming
    assert abs((te.y == "b").mean() - (ds.y == "b").mean()) < 0.05  # stratified
    a, b, ya, yb = train_test_split(ds, ds.y, test_size=10, random_state=0)  # X and y together
    assert list(ya) == list(a.y) and list(yb) == list(b.y)


def test_time_order_split_and_splitters(coll):
    ds = MongoDataset(coll, target="y", sort=[("t", -1)])  # newest first
    old, new = train_test_split(
        MongoDataset(coll, target="y", sort=[("t", 1)]), test_size=0.2, shuffle=False
    )
    assert max(old.field_values("t")) < min(new.field_values("t"))
    tr, te = next(
        GroupShuffleSplit(test_size=0.3, random_state=0).split(ds, groups=ds.field_values("g"))
    )
    assert not set(np.asarray(ds.field_values("g"))[tr]) & set(np.asarray(ds.field_values("g"))[te])
    for tr, te in TimeSeriesSplit(n_splits=3, gap=2).split(ds):
        assert len(ds[tr]) == len(tr) and tr.max() + 2 < te.min()


def test_cross_val_score_with_a_pipeline(coll):
    ds = MongoDataset(coll, target="y")
    pipe = make_pipeline(
        DocumentFeaturizer(embedding_model="hashing", cache_dir=None), LogisticRegression()
    )
    scores = cross_val_score(pipe, ds, ds.y, cv=3)
    assert len(scores) == 3 and scores.min() > 0.6


def test_indexing(coll):
    ds = MongoDataset(coll, target="y")
    assert ds.shape == (60,)
    assert len(ds[[0, 5, 5]]) == 3 and len(ds[2:10]) == 8 and len(ds.iloc[::2]) == 30
    assert len(ds[ds.y == "b"]) == 20 and len(ds.take(np.arange(4))) == 4
    assert ds[[3]].ids == [ds.ids[3]] and len(ds[np.array([1, 2]), ...]) == 2
    with pytest.raises(TypeError, match="ds.documents"):
        ds[0]
    with pytest.raises(ValueError, match="axis=0"):
        ds.take([0], axis=1)
    with pytest.raises(IndexError):
        ds[[60]]


def test_time_values_parse_as_naive_utc():
    assert _as_datetime("2020-01-01T05:00:00+05:00") == dt.datetime(2020, 1, 1, 0, 0)
    assert _as_datetime(1_600_000_000_000) == _as_datetime(1_600_000_000)  # epoch ms and s
    assert _as_datetime(float("nan")) is None and _as_datetime(True) is None
    assert _as_datetime(np.datetime64("NaT", "s")) is None
    assert _as_datetime("Jan-2015").year == 2015 and _as_datetime("not a date") is None
