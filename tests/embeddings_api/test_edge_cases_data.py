"""Regression tests for edge cases of datasets, splits, document helpers and series."""

import datetime as dt
import threading
import time

import mongomock
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from pymongopredict import MongoDataset, MongoSeries
from pymongopredict.embeddings_api.documents import (
    _compile_path,
    _document_to_json,
    _get_compiled,
    _get_path,
    _select_fields,
)

UTC_MINUS_5 = dt.timezone(dt.timedelta(hours=-5))


def coll_of(docs):
    c = mongomock.MongoClient().db.c
    c.insert_many(docs)
    return c


# ---------------------------------------------------------------- documents
def test_select_fields_removes_excluded_paths_inside_and_above_fields():
    doc = {"address": {"market": "Porto", "city": "x"}, "r": [{"o": 1, "k": 2}], "a": {"b": 1}}
    assert _select_fields(doc, ["address"], ["address.market"]) == {"address": {"city": "x"}}
    assert _select_fields(doc, ["r"], ["r.o"]) == {"r": [{"k": 2}]}
    assert _select_fields(doc, ["a.b", "address.city"], ["a"]) == {"address.city": "x"}
    assert doc["address"]["market"] == "Porto"  # input untouched
    assert _document_to_json(doc, ["a"], exclude=["a.b"]) == '{"a":{}}'


def test_compiled_paths_read_numeric_dict_keys_like_get_path():
    d = {"a": {"0": 5}, "b": [7, 8]}
    for path in ("a.0", "b.1", "b.5", "a.1"):
        assert _get_compiled(d, _compile_path(path)) == _get_path(d, path)


def test_json_budget_with_a_prefix_longer_than_the_budget():
    out = _document_to_json({"a": "x" * 50}, prefix="p" * 40, max_chars=30)
    assert out.startswith("p" * 40) and len(out) <= 42  # never a negative budget


# ---------------------------------------------------------------- datasets
def test_streaming_with_embedded_document_ids():
    c = coll_of([{"_id": {"k": i}, "y": i % 2, "x": i} for i in range(6)])
    ds = MongoDataset(c, "y", streaming=True, batch_size=4)
    assert [d["x"] for d in ds.documents] == list(range(6))
    assert [d["x"] for d in ds.subset([5, 0]).documents] == [5, 0]


def test_projection_without_id_in_memory():
    c = coll_of([{"x": i, "y": i} for i in range(4)])
    ds = MongoDataset(c, "y", projection={"_id": 0, "x": 1, "y": 1})
    assert len(ds) == 4 and ds.ids == [0, 1, 2, 3]


def test_subset_cannot_be_reloaded_into_the_full_collection():
    c = coll_of([{"y": i} for i in range(5)])
    tr, te = train_test_split(MongoDataset(c, "y"), test_size=0.4, random_state=0)
    assert len(te) == 2
    with pytest.raises(ValueError, match="cannot be reloaded"):
        te.load(force=True)


def test_prefetch_thread_stops_when_the_consumer_stops_early():
    c = coll_of([{"y": i} for i in range(200)])
    ds = MongoDataset(c, "y", streaming=True, batch_size=5, prefetch=2)
    before = threading.active_count()
    for _ in range(5):
        for _batch in ds.iter_batches():
            break
    deadline = time.monotonic() + 3
    while threading.active_count() > before + 1 and time.monotonic() < deadline:
        time.sleep(0.02)  # the workers notice the stop within their 0.1 s put timeout
    assert threading.active_count() <= before + 1


# ---------------------------------------------------------------- time splits
def hours(n, **kw):
    return [{"t": dt.datetime(2020, 1, 1) + dt.timedelta(hours=h), **kw} for h in range(n)]


# ---------------------------------------------------------------- series
def test_series_epoch_seconds_offsets_and_too_few_periods():
    c = coll_of([{"t": 1600000000 + 86400 * i, "v": i} for i in range(5)])
    s = MongoSeries(c, time_field="t", value_field="v", freq="D")
    assert len(s) == 5 and str(s.series.index[0]) == "2020-09-13"
    y_tr, y_te = s.split(np.int64(2))
    assert len(y_te) == 2
    c = coll_of(
        [
            {"t": f"2020-01-0{i + 1}T00:00:00{off}", "v": i}
            for i, off in enumerate(["+02:00", "+05:00", "+00:00", "Z"])
        ]
    )
    s = MongoSeries(
        c, time_field="t", value_field="v", freq="D"
    )  # 00:00+02:00 is 22:00 UTC the day before
    assert [str(p) for p in s.series.index] == [
        "2019-12-31",
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
    ]
    c = coll_of([{"t": dt.datetime(2020, 1, 1, h), "v": h} for h in range(3)])
    with pytest.raises(ValueError, match="periods"):
        MongoSeries(c, time_field="t", value_field="v", freq="D").load()


# ---------------------------------------------------------------- round 2
def test_projection_without_the_target_and_streaming_without_id():
    c = coll_of([{"a": i, "y": i % 2} for i in range(5)])
    assert len(MongoDataset(c, target="y", projection={"a": 1})) == 5
    docs = MongoDataset(c, target="y", projection={"_id": 0, "a": 1}, streaming=True).documents
    assert [d["a"] for d in docs] == list(range(5)) and "_id" not in docs[0]


def test_uuid_ids_and_overlapping_array_excludes():
    import uuid

    from pymongopredict.embeddings_api.documents import _id_key, _without_paths

    u = uuid.uuid4()
    assert _id_key(u) == _id_key(uuid.UUID(str(u))) != _id_key(str(u))
    # (mongomock cannot store native UUIDs: streaming over them is in the server test)
    assert _without_paths({"arr": [10, 11, 12, 13]}, ["arr.0", "arr.1"]) == {"arr": [12, 13]}


def test_missing_targets_of_every_kind_and_bad_inputs():
    docs = [{"y": v} for v in (np.float32("nan"), pd.NA, pd.NaT, None, 1.0, 2.0)]
    assert list(MongoDataset.from_documents(docs, target="y").y) == [1.0, 2.0]
    with pytest.raises(ValueError, match="one dimensional"):
        MongoDataset.from_documents([{}, {}], y=np.zeros((2, 2)))
    for bs in (0, -1):
        with pytest.raises(ValueError, match="batch_size"):
            MongoDataset(coll_of([{"y": 1}]), "y", batch_size=bs)


# ---------------------------------------------------------------- round 3
@pytest.mark.parametrize(
    "projection",
    [
        {"_id": 1},
        {"a": 1.0},
        {"a": 1, "l": {"$slice": 1}},
        {"a": 1, "y.z": 1},
    ],  # (mongomock lacks {"a": {"b": 1}})
)
def test_inclusion_projections_always_fetch_the_target(projection):
    c = coll_of([{"a": {"b": i}, "l": [1, 2], "y": i % 2} for i in range(5)])
    assert len(MongoDataset(c, target="y", projection=projection)) == 5


def test_exclusion_projections_are_left_alone():
    from pymongopredict.embeddings_api.dataset import _with_paths

    for proj in ({"a": 0}, {"a": {"b": 0}}, {"l": {"$slice": 1}}, {"_id": 0}):
        assert _with_paths(proj, ["y"]) == proj


# ---------------------------------------------------------------- round 4
def test_projection_targets_with_indices_elem_match_and_nested_ancestors():
    from pymongopredict.embeddings_api.dataset import _with_paths

    docs = [
        {"scores": [{"score": i}], "z": 1, "l": [{"k": 9}], "a": {"b": i, "c": i}, "y": i % 2}
        for i in range(5)
    ]
    c = coll_of(docs)
    assert len(MongoDataset(c, "scores.0.score", projection={"z": 1})) == 5
    assert len(MongoDataset(c, "y", projection={"l": {"$elemMatch": {"k": 9}}})) == 5
    assert len(MongoDataset(c, "a.c", projection={"a.b": 1})) == 5
    assert _with_paths({"a": {"b": 1}}, ["a.c"]) == {"a.b": 1, "a.c": 1}
    assert _with_paths({"a": {"b": 1}}, ["a"]) == {"a": 1}
    assert _with_paths({"_id": 1}, ["_id.x"]) == {"_id": 1}
    with pytest.raises(ValueError, match="computes"):
        _with_paths({"a": "$z"}, ["a.b"])


# ---------------------------------------------------------------- round 5
def test_in_memory_and_streaming_agree_when_the_projection_leaves_fields_out():
    c = coll_of([{"_id": i, "y": i % 2, "g": i % 3, "x": i} for i in range(6)])
    for proj in ({"x": 1}, {"y": 0}, {"g": 0}):
        a = MongoDataset(c, "y", projection=proj)
        b = MongoDataset(c, "y", projection=proj, streaming=True)
        assert len(a) == len(b) == 6 and list(a.y) == list(b.y)
        assert a.field_values("g") == b.field_values("g") == [0, 1, 2, 0, 1, 2]
    with pytest.raises(ValueError, match="leaves out"):
        MongoDataset(c, "y", projection={"_id": 0, "x": 1}).field_values("g")


def test_excludes_through_array_indices_and_overlapping_removals():
    from pymongopredict.embeddings_api.documents import _json_pieces, _select_fields, _without_paths

    d = {"a": [{"b": 1, "c": 1}, {"b": 2, "c": 2}]}
    assert _select_fields(d, ["a.0"], ["a.b"]) == {"a.0": {"c": 1}}
    assert _select_fields(d, ["a.0.b"], ["a.b"]) == {}
    assert _select_fields(d, ["a.1"], ["a.0"]) == {"a.1": {"b": 2, "c": 2}}
    assert _json_pieces(d, ["a.0"], first={"a.b": "?"}) == ['"a.b":"?"', '"a.0":{"c":1}']
    assert _without_paths(d, ["a.0", "a.0.b"]) == {"a": [{"b": 2, "c": 2}]}


def test_projection_paths_json_budget_and_unicode_digits():
    from pymongopredict.embeddings_api.documents import (
        _assemble_json,
        _get_path,
        _projection_for,
        _without_paths,
    )

    assert _projection_for(["2020", "x"]) == {"2020": 1, "x": 1}
    assert _projection_for(["_id.k", "a.0.b"]) == {"_id": 1, "a": 1}
    assert _assemble_json(['"k":1'], max_units=7) == '{"k":1}'
    assert _assemble_json(['"a":1', '"b":2'], max_units=13) == '{"a":1,"b":2}'
    assert _get_path({"a": [1]}, "a.²") is None and _without_paths({"a": 1}, ["a.²"]) == {"a": 1}


@pytest.mark.parametrize("freq", ["ME", "MS", "QE", "M", "Y"])
def test_series_monthly_and_longer_frequencies(freq):
    c = coll_of(
        [{"t": dt.datetime(2021 + m // 12, m % 12 + 1, 3), "v": float(m)} for m in range(40)]
    )
    assert len(MongoSeries(c, time_field="t", value_field="v", freq=freq).series) >= 3


def test_series_skips_values_that_are_not_numbers():
    docs = [{"t": dt.datetime(2020, 1, d), "v": d} for d in range(1, 6)] + [
        {"t": dt.datetime(2020, 1, 6), "v": "n/a"}
    ]
    assert len(MongoSeries(coll_of(docs), time_field="t", value_field="v", freq="D")) == 5


# ---------------------------------------------------------------- round 6
def test_projections_with_computed_nested_or_id_only_fields():
    from pymongopredict.embeddings_api.dataset import _covers, _with_paths

    assert (
        _covers({"g2": "$g"}, "g2")
        and _covers({"a": {"b": 1}}, "a.b")
        and _covers({"l": {"$slice": 1}}, "l")
    )
    assert not _covers({"x": 1}, "g")
    assert _with_paths({"_id": 1, "t": 0}, ["t"]) is None
    c = coll_of([{"_id": i, "t": i % 2, "a": {"b": i}} for i in range(6)])
    assert len(MongoDataset(c, "t", projection={"_id": 1, "t": 0})) == 6
    assert MongoDataset(c, "t", projection={"_id": 0, "t": 1, "a.b": 1}).field_values(
        "a.b"
    ) == list(range(6))


def test_removals_covered_through_array_wildcards():
    from pymongopredict.embeddings_api.documents import _select_fields, _without_paths

    assert _without_paths({"a": [{"b": [10, 11, 12]}]}, ["a.b.0", "a.0.b.0"]) == {
        "a": [{"b": [11, 12]}]
    }
    doc = {"a": [{"b": 1, "c": 1}, {"b": 2, "c": 2}, {"b": 3}]}
    assert _select_fields(doc, ["a"], ["a.0", "a.0.b"]) == {
        "a": _without_paths(doc, ["a.0", "a.0.b"])["a"]
    }


def test_anchored_start_frequencies_label_their_periods():
    c = coll_of([{"t": dt.datetime(2020, m, 15), "v": 1.0} for m in range(1, 13)])
    s = MongoSeries(c, time_field="t", value_field="v", freq="QS-FEB", agg="sum").series
    assert str(s.index[0].start_time.date()) == "2019-11-01"  # the Nov-Jan bin


def test_streaming_keeps_the_dataset_projection_with_explicit_embed_fields():
    from pymongopredict import DocumentFeaturizer

    c = coll_of(
        [
            {"_id": i, "a": f"w{i}", "b": f"s{i}", "n": i, "y": "x" if i % 2 else "z"}
            for i in range(6)
        ]
    )
    for proj in ({"b": 0}, {"a": 1, "n": 1}):
        X = [
            DocumentFeaturizer(
                columns=["n"], embed_fields=["a", "b"], embedding_model="hashing", cache_dir=None
            ).fit_transform(MongoDataset(c, target="y", projection=proj, streaming=s))
            for s in (False, True)
        ]
        assert np.allclose(X[0], X[1])


# ---------------------------------------------------------------- round 7
def test_numeric_dict_keys_are_not_array_indices():
    from pymongopredict import DocumentFeaturizer
    from pymongopredict.embeddings_api.documents import _select_fields, _without_paths

    assert _without_paths({"price": 5, "2020": {"price": 7}}, ["price", "2020.price"]) == {
        "2020": {}
    }
    assert _select_fields({"name": "x", "2020": {"name": "y"}}, ["2020"], ["name"]) == {
        "2020": {"name": "y"}
    }
    assert _without_paths({"a": [{"b": 1, "c": 2}]}, ["b", "a.b"]) == {
        "a": [{"c": 2}]
    }  # arrays still map
    docs = [{"price": i, "2020": {"price": 2 * i}} for i in range(4)]
    f = DocumentFeaturizer(exclude=["price"], embedding_model="hashing", cache_dir=None)
    f.fit(MongoDataset.from_documents(docs, target="2020.price"))
    assert f._texts(docs[1:2]) == ['{"2020":{}}']


def test_id_with_slice_projection_is_an_inclusion():
    from pymongopredict.embeddings_api.dataset import _with_paths

    assert _with_paths({"_id": 1, "c": {"$slice": 1}}, ["a"]) == {
        "_id": 1,
        "c": {"$slice": 1},
        "a": 1,
    }
    assert _with_paths({"c": {"$slice": 1}}, ["a"]) == {
        "c": {"$slice": 1}
    }  # alone: keeps every field


# ---------------------------------------------------------------- round 8
def test_streaming_keeps_real_ids_with_id_excluded_and_a_sliced_target():
    c = coll_of([{"_id": i, "name": f"doc{i}", "arr": [10 * i]} for i in range(6)])
    proj = {"_id": 0, "name": 1, "arr": {"$slice": 1}}
    ds = MongoDataset(c, "arr.0", filter={"_id": {"$gte": 3}}, projection=proj, streaming=True)
    assert ds.ids == [3, 4, 5] and list(ds.y) == [30.0, 40.0, 50.0]
    assert [d["name"] for d in ds.documents] == ["doc3", "doc4", "doc5"]


def test_tuples_and_nested_array_coverage_in_removals():
    from pymongopredict.embeddings_api.documents import _json_pieces, _without_paths

    assert _json_pieces({"a": ({"t": 1, "u": 2},), "b": 3}, None, ["a.0.t"]) == [
        '"a":[{"u":2}]',
        '"b":3',
    ]
    doc = {"x": [{"a": [{"b": [1, 2]}]}]}
    assert _without_paths(doc, ["x.a.0.b.0", "x.a.b.0"]) == _without_paths(doc, ["x.a.b.0"])


def test_field_values_refreshed_after_a_forced_reload():
    for streaming in (False, True):
        c = coll_of([{"_id": i, "y": i % 2, "g": "a"} for i in range(4)])
        ds = MongoDataset(c, "y", projection={"y": 1}, streaming=streaming)
        assert ds.field_values("g") == ["a"] * 4
        c.delete_one({"_id": 0})
        c.insert_one({"_id": 9, "y": 1, "g": "b"})
        ds.load(force=True)
        assert ds.field_values("g") == ["a", "a", "a", "b"]


@pytest.mark.parametrize("freq", ["2M", "2MS", "2QS", "2QS-FEB", "QE-FEB", "2YS"])
def test_series_frequency_multiples(freq):
    c = coll_of(
        [
            {"t": dt.datetime(2015, 1, 1) + dt.timedelta(days=15 * i), "v": float(i)}
            for i in range(400)
        ]
    )
    s = MongoSeries(c, time_field="t", value_field="v", freq=freq).series
    assert len(s) >= 3 and s.index[1] == s.index[0] + 1  # consecutive bins: consecutive periods
