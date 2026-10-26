"""Missing and unusual data as real collections contain it (staff review, round 1)."""

import datetime as dt
import warnings

import mongomock
import numpy as np
import pytest
from bson import Binary, Decimal128

from pymongopredict import DocumentFeaturizer, MongoDataset, MongoSeries
from pymongopredict.embeddings_api.documents import _document_to_json, _projection_for, _scalar


def featurizer(**kw):
    return DocumentFeaturizer(embed_fields=False, cache_dir=None, **kw)


def test_dates_far_in_the_past_or_future_stay_finite_and_ordered():
    years = [1, 1677, 1970, 2262, 9999]
    f = featurizer(columns=["x"], scale_numeric=False).fit(
        [{"x": dt.datetime(y, 1, 1)} for y in years]
    )
    x = f.transform([{"x": dt.datetime(y, 1, 1)} for y in years])[:, 0]
    assert np.isfinite(x).all() and (np.diff(x) > 0).all() and x[2] == 0


def test_numbers_and_dates_in_the_wrong_column_kind_do_not_depend_on_the_batch():
    num = featurizer(columns=["n"]).fit([{"n": float(i)} for i in range(20)])
    alone = num.transform([{"n": dt.datetime(2020, 1, 1)}])
    with_other = num.transform([{"n": dt.datetime(2020, 1, 1)}, {"n": 3.0}])[:1]
    assert np.allclose(alone, with_other)
    dates = featurizer(columns=["t"], scale_numeric=False).fit(
        [{"t": dt.datetime(2000 + i, 1, 1)} for i in range(5)]
    )
    as_date = dates.transform([{"t": dt.datetime(2020, 1, 1)}])
    as_seconds = dates.transform([{"t": 1577836800}])  # 2020-01-01 as epoch seconds
    assert np.allclose(as_date, as_seconds)


def test_signalling_nan_decimals_and_binary_blobs():
    assert _document_to_json({"a": Decimal128("sNaN"), "b": 1}) == '{"a":null,"b":1}'
    assert np.isnan(_scalar(Decimal128("sNaN")))
    blob = {"thumb": Binary(b"x" * 50_000), "title": "Inception"}
    text = _document_to_json(blob, max_chars=8000)
    assert "Inception" in text and "<binary 50000 bytes>" in text


def test_placeholder_infinite_and_container_targets():
    docs = [{"y": float(i)} for i in range(10)] + [
        {"y": v} for v in ("N/A", "", " null ", "-", float("inf"))
    ]
    assert len(MongoDataset.from_documents(docs, target="y").y) == 10
    with pytest.raises(ValueError, match="single values"):
        MongoDataset.from_documents([{"y": ["a", "b"]}, {"y": ["c"]}], target="y").load()


def test_label_spellings_and_huge_integer_codes():
    y = MongoDataset.from_documents(
        [{"y": v} for v in (1, 1.0, "1", "x", True, "TRUE")], target="y"
    ).y
    assert sorted(set(y)) == ["1", "true", "x"]
    big = MongoDataset.from_documents([{"y": 2**53}, {"y": 2**53 + 1}], target="y").y
    assert len(set(big)) == 2


def test_target_listed_as_a_column_and_empty_columns():
    ds = MongoDataset.from_documents([{"t": i, "x": i % 3} for i in range(10)], target="t")
    with pytest.raises(ValueError, match="leak"):
        featurizer(columns=["t", "x"]).fit(ds)
    with pytest.warns(UserWarning, match="no value"):
        featurizer(columns=["x", "a.b"]).fit([{"x": 1, "a.b": 2}, {"x": 2}])


def test_dollar_field_names_fetch_whole_documents():
    assert _projection_for(["$price", "x"]) is None
    assert _projection_for(["x"]) == {"x": 1}


def test_series_skips_infinite_and_boolean_values_and_refuses_huge_spans():
    c = mongomock.MongoClient().db.s
    c.insert_many([{"t": dt.datetime(2020, 1, d), "v": float(d)} for d in range(1, 8)])
    c.insert_many(
        [
            {"t": dt.datetime(2020, 1, 8), "v": float("inf")},
            {"t": dt.datetime(2020, 1, 9), "v": True},
        ]
    )
    s = MongoSeries(c, time_field="t", value_field="v", freq="D")
    assert np.isfinite(s.series.to_numpy()).all() and len(s.series) == 7
    c.insert_one({"t": dt.datetime(9999, 12, 31), "v": 1.0})
    with pytest.raises(ValueError, match="periods"):
        MongoSeries(c, time_field="t", value_field="v", freq="h").load()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MongoSeries(c, time_field="t", value_field="v", freq="YE").load()  # calendar freq: small
