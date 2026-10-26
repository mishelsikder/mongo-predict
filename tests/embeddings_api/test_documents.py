import datetime as dt
import json

from bson import Decimal128, ObjectId

from pymongopredict.embeddings_api.documents import _document_to_json, _get_path, _select_fields


def test_get_path_nested_and_list():
    doc = {"a": {"b": [10, {"c": 3}]}}
    assert _get_path(doc, "a.b.0") == 10
    assert _get_path(doc, "a.b.1.c") == 3
    assert _get_path(doc, "a.x", default="d") == "d"


def test_document_to_json_converts_bson_and_selects_fields():
    doc = {
        "_id": ObjectId("5553a998e4b02cf7151190b8"),
        "price": Decimal128("80.00"),
        "ts": dt.datetime(1984, 3, 5),
        "address": {"market": "Porto"},
        "secret": 1,
    }
    text = _document_to_json(doc, fields=["price", "address.market", "ts"])
    assert json.loads(text) == {
        "price": 80.0,
        "address.market": "Porto",
        "ts": "1984-03-05T00:00:00",
    }
    all_but = json.loads(_document_to_json(doc, exclude=["secret"]))
    assert set(all_but) == {"price", "ts", "address"}  # _id and secret dropped


def test_select_fields_keeps_missing_out():
    assert _select_fields({"a": 1}, ["a", "b"]) == {"a": 1}


def test_nested_exclude_removes_target_without_mutating_doc():
    doc = {"airTemperature": {"value": 1.5, "quality": "1"}, "st": "x"}
    text = _document_to_json(doc, exclude=["airTemperature.value", "nope.deeper"])
    assert json.loads(text) == {"airTemperature": {"quality": "1"}, "st": "x"}
    assert doc["airTemperature"]["value"] == 1.5


def test_iter_batches_and_projection_helpers():
    import pandas as pd

    from pymongopredict.embeddings_api.documents import _iter_batches, _projection_for

    assert [len(b) for b in _iter_batches(iter(range(10)), 4)] == [4, 4, 2]
    assert list(_iter_batches({"a": 1}, 3)) == [[{"a": 1}]]
    df = pd.DataFrame({"a": range(5)})
    assert [len(b) for b in _iter_batches(df, 2)] == [2, 2, 1]
    assert _projection_for(["position.coordinates.0", "address", "address.market", "x.y"]) == {
        "address": 1,
        "position.coordinates": 1,
        "x.y": 1,
    }


def test_nested_exclude_through_arrays():
    doc = {"scores": [{"type": "exam", "score": 1.0}, {"type": "quiz", "score": 2.0}], "k": 1}
    out = json.loads(_document_to_json(doc, exclude=["scores.0.score"]))
    assert out == {"scores": [{"type": "exam"}, {"type": "quiz", "score": 2.0}], "k": 1}
    assert doc["scores"][0]["score"] == 1.0  # original untouched
    assert (
        json.loads(_document_to_json(doc, exclude=["scores.5.score", "scores.0.nope.x"]))["scores"][
            0
        ]["score"]
        == 1.0
    )


def test_document_to_json_budget_skips_trailing_fields():
    doc = {"a": "x" * 50, "reviews": [{"t": "long" * 100}] * 50, "z": 1}
    full = _document_to_json(doc)
    assert json.loads(full)["z"] == 1
    short = _document_to_json(doc, max_chars=200)
    assert len(short) <= 200 and short.startswith('{"a":"xxx')
    # nothing beyond the budget is serialised, so the output is a prefix of the full JSON
    assert full.startswith(short.rstrip("}").rstrip(","))
    tiny = _document_to_json({"k": "v" * 10}, max_chars=8)
    assert tiny == "{}"


def test_dumps_handles_bson_and_numpy():
    import numpy as np

    from pymongopredict.embeddings_api.documents import _dumps

    out = json.loads(
        _dumps(
            {
                "o": ObjectId("5553a998e4b02cf7151190b8"),
                "d": Decimal128("1.5"),
                "n": np.float32(2.5),
                "t": dt.datetime(2020, 1, 2, 3),
                "b": b"ab",
                3: "int key",
            }
        )
    )
    assert out == {
        "o": "5553a998e4b02cf7151190b8",
        "d": 1.5,
        "n": 2.5,
        "t": "2020-01-02T03:00:00",
        "b": "<binary 2 bytes>",  # binary data is not embedded as base64
        "3": "int key",
    }


def test_without_paths_through_arrays():
    from pymongopredict.embeddings_api.documents import _without_paths

    doc = {
        "patient": {"reaction": [{"name": "a", "outcome": 1}, {"name": "b", "outcome": 2}]},
        "scores": [{"score": 5}, {"score": 6}],
        "k": 1,
    }
    out = _without_paths(doc, ["patient.reaction.outcome", "scores.0.score", "missing.path"])
    reactions = [{"name": "a"}, {"name": "b"}]
    assert out == {"patient": {"reaction": reactions}, "scores": [{}, {"score": 6}], "k": 1}
    assert doc["patient"]["reaction"][0] == {"name": "a", "outcome": 1}  # input untouched
    assert _without_paths(doc, ["k"]) == {"patient": doc["patient"], "scores": doc["scores"]}
