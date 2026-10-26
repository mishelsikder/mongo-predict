"""Model building and $predict on messy data and real deployments (staff review, round 1)."""

import datetime as dt
import os

import mongomock
import pytest
from pymongo.errors import OperationFailure

from pymongopredict import PredictionModel
from pymongopredict import predictions_api as P
from pymongopredict.predictions_api.stage import cleanup_temporary_collections


def coll_of(docs):
    c = mongomock.MongoClient().db.c
    c.insert_many(docs)
    return c


def model(c, definition, name="m", **kw):
    P.create_prediction_model(
        c,
        PredictionModel(
            {"embedFields": False, "featurizerOptions": {"cache_dir": None}, **definition}, name
        ),
        **kw,
    )
    return P.list_prediction_models(c, name)[0]


def test_numeric_targets_with_placeholders_currency_and_infinities():
    docs = [{"x": i, "y": float(i % 50) + 0.5} for i in range(100)]
    docs += [{"x": 1, "y": "N/A"}, {"x": 2, "y": ""}, {"x": 3, "y": "null"}]
    rec = model(coll_of(docs), {"target": "y", "columns": ["x"]})
    assert rec["task"] == "regression" and rec["numTrainingDocuments"] == 100
    money = [{"x": i, "y": f"${1000 + i * 37:,}"} for i in range(60)]
    assert model(coll_of(money), {"target": "y", "columns": ["x"]})["task"] == "regression"
    rec = model(coll_of(docs + [{"x": 4, "y": float("inf")}]), {"target": "y", "columns": ["x"]})
    assert rec["status"] == "READY" and rec["numTrainingDocuments"] == 100


def test_a_rare_class_in_the_holdout_does_not_fail_the_build():
    docs = [{"x": i, "y": "a"} for i in range(99)]
    docs.insert(7, {"x": 7, "y": "b"})
    with pytest.warns(UserWarning):
        rec = model(coll_of(docs), {"target": "y", "columns": ["x"], "testSize": 0.5})
    assert rec["status"] == "READY"


def test_definitions_rejected_up_front():
    for d in (
        {"target": "$price"},
        {"target": "t", "columns": ["t"]},
        {"target": "t", "columns": ["t.x"]},
        {"target": "a..b"},
    ):
        with pytest.raises(ValueError):
            PredictionModel(d, "m")


def test_stale_builds_and_cleanup_of_temporary_collections():
    c = coll_of([{"x": i, "y": "a" if i % 2 else "b"} for i in range(20)])
    model(c, {"target": "y", "columns": ["x"]})
    records = c.database[P.MODELS_COLLECTION]
    records.update_one(
        {"_id": "c.m"}, {"$set": {"status": "BUILDING", "buildStartedAt": dt.datetime(2020, 1, 1)}}
    )
    assert P.list_prediction_models(c, "m")[0]["stale"] is True
    db = c.database
    db.create_collection(f"{P.TMP_PREFIX}1000000000_abc")  # 2001: a leftover
    import time

    db.create_collection(f"{P.TMP_PREFIX}{int(time.time())}_def")  # a running call
    db.create_collection(f"{P.TMP_PREFIX}0123456789abcdef0123456789abcdef")  # older versions' names
    dropped = cleanup_temporary_collections(db)
    assert sorted(dropped) == sorted(
        [f"{P.TMP_PREFIX}1000000000_abc", f"{P.TMP_PREFIX}0123456789abcdef0123456789abcdef"]
    )


def test_failing_cleanup_keeps_the_result_and_documents_without_id_raise(monkeypatch):
    c = coll_of([{"x": i, "y": "a" if i % 2 else "b"} for i in range(20)])
    model(c, {"target": "y", "columns": ["x"]})

    def refuse(self, name, *a, **k):
        raise OperationFailure("not authorized on db to execute command { drop: ... }", 13)

    monkeypatch.setattr(type(c.database), "drop_collection", refuse)
    with pytest.warns(UserWarning, match="could not drop"):
        out = list(P.aggregate(c, [{"$predict": {"model": "m"}}]))
    assert len(out) == 20
    monkeypatch.undo()
    monkeypatch.setattr(
        type(c),
        "options",
        lambda self: (_ for _ in ()).throw(OperationFailure("denied", 13)),
        raising=False,
    )
    with pytest.raises(PermissionError, match="listCollections"):
        P.aggregate(c, [{"$predict": {"model": "m"}}])


@pytest.mark.skipif(
    not os.environ.get("MONGOPREDICT_TEST_URI"),
    reason="set MONGOPREDICT_TEST_URI to a mongod to run",
)
def test_server_codec_options_secondary_preference_and_dollar_columns():
    import uuid

    from bson.codec_options import CodecOptions, UuidRepresentation
    from pymongo import MongoClient, ReadPreference

    client = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])
    db = client[f"mongopredict_test_staff_{os.getpid()}"]
    try:
        c = db.get_collection(
            "u", codec_options=CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        )
        c.insert_many(
            [
                {"_id": uuid.uuid4(), "x": i, "$p": i % 3, "y": "a" if i % 2 else "b"}
                for i in range(30)
            ]
        )
        model(c, {"target": "y", "columns": ["x", "$p"]})
        assert len(list(P.aggregate(c, [{"$predict": {"model": "m"}}]))) == 30
        sec = c.with_options(read_preference=ReadPreference.SECONDARY_PREFERRED)
        assert len(list(P.aggregate(sec, [{"$predict": {"model": "m"}}]))) == 30
    finally:
        client.drop_database(db.name)


def test_documents_without_id_raise_a_clear_error():
    c = coll_of([{"x": i, "y": "a" if i % 2 else "b"} for i in range(20)])
    model(c, {"target": "y", "columns": ["x"]})
    real_find = type(c).find

    def find_without_id(self, *a, **k):  # what a view that leaves out _id returns
        return iter(
            [{k2: v for k2, v in d.items() if k2 != "_id"} for d in real_find(self, *a, **k)]
        )

    c.find = find_without_id.__get__(c)
    with pytest.raises(ValueError, match="_id"):
        P.aggregate(c, [{"$predict": {"model": "m"}}])
