"""Interactions between the staff-review changes (round 5): streamed training, reversed joins,
class values and signatures."""

import os

import mongomock
import pytest

from pymongopredict import PredictionModel
from pymongopredict import predictions_api as P
from pymongopredict.embeddings_api import dataset as D
from pymongopredict.predictions_api import models as M

server = pytest.mark.skipif(
    not os.environ.get("MONGOPREDICT_TEST_URI"), reason="set MONGOPREDICT_TEST_URI to a mongod"
)
DEFINITION = {"embedFields": False, "featurizerOptions": {"cache_dir": None}}


def coll_of(docs):
    c = mongomock.MongoClient().db.s
    c.insert_many(docs)
    return c


def build(c, definition, name="m"):
    P.create_prediction_model(c, PredictionModel({**DEFINITION, **definition}, name))
    return P.list_prediction_models(c, name)[0]


def change_after_the_id_scan(monkeypatch, change):
    real = D.MongoDataset.load

    def load(self, force=False):
        first = self._ids is None
        out = real(self, force)
        if first and not self._view:
            change()
        return out

    monkeypatch.setattr(D.MongoDataset, "load", load)


@pytest.mark.parametrize("change", ["relabel", "delete"])
def test_documents_changed_during_training(monkeypatch, change):
    c = coll_of([{"_id": i, "x": float(i % 2), "y": "cat" if i % 2 else "dog"} for i in range(100)])
    if change == "relabel":
        change_after_the_id_scan(
            monkeypatch, lambda: c.update_one({"_id": 0}, {"$set": {"y": "cat"}})
        )
        rec = build(c, {"target": "y", "columns": ["x"]})
    else:
        change_after_the_id_scan(monkeypatch, lambda: c.delete_one({"_id": 0}))
        with pytest.warns(UserWarning, match="deleted during training"):
            rec = build(c, {"target": "y", "columns": ["x"]})
    assert rec["status"] == "READY" and rec["classes"] == rec["classValues"] == ["cat", "dog"]


def test_a_rebuild_removes_fields_of_the_previous_build(monkeypatch):
    c = coll_of([{"x": i, "y": "ab"[i % 2], "n": float(i)} for i in range(40)])
    monkeypatch.setenv(M.MODEL_KEY_ENV, "k")
    assert "classValues" in build(c, {"target": "y", "columns": ["x"]})
    monkeypatch.delenv(M.MODEL_KEY_ENV)
    P.update_prediction_model(c, "m", {**DEFINITION, "target": "n", "columns": ["x"]})
    rec = P.list_prediction_models(c, "m")[0]
    assert rec["task"] == "regression" and "classValues" not in rec and "modelSignature" not in rec


def test_the_signature_covers_the_class_values(monkeypatch):
    c = coll_of([{"x": i % 2, "y": "ab"[i % 2]} for i in range(40)])
    monkeypatch.setenv(M.MODEL_KEY_ENV, "k")
    build(c, {"target": "y", "columns": ["x"]})
    M._loaded.clear()
    c.database[M.MODELS_COLLECTION].update_one(
        {"_id": "s.m"}, {"$set": {"classValues": ["b", "a"]}}
    )
    with pytest.raises(PermissionError):
        P._load_prediction_model(c, "m")


@server
def test_server_reversed_join_uses_the_predicted_documents_only():
    import datetime as dt

    from pymongo import MongoClient

    db = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])[f"mongopredict_test_rev_{os.getpid()}"]
    try:
        # the filter is not run again: $rand and the user's collation would select other documents
        db.create_collection("ts", timeseries={"timeField": "t", "metaField": "m"})
        t0 = dt.datetime(2024, 1, 1)
        db.ts.insert_many(
            [
                {"t": t0 + dt.timedelta(minutes=i), "m": {"k": "Aa"[i % 2]}, "x": i, "y": i % 2}
                for i in range(60)
            ]
        )
        build(db.ts, {"target": "y", "columns": ["x"]})
        ci = {"locale": "en", "strength": 2}
        stage = {"$predict": {"model": "m", "filter": {"m.k": "a"}}}
        if tuple(db.client.server_info()["versionArray"][:2]) < (6, 0):  # time series are views
            with pytest.raises(ValueError, match="cannot apply collation"):
                P.aggregate(db.ts, [stage], collation=ci)
        else:
            assert len(P.aggregate(db.ts, [stage], collation=ci).to_list()) == 60
        rand = {"$predict": {"model": "m", "filter": {"$expr": {"$lt": [{"$rand": {}}, 0.5]}}}}
        out = P.aggregate(db.ts, [rand])
        assert len(out.to_list()) == out.stats["nDocuments"]

        # a case-insensitive view joins "a" to its own prediction only
        db.base.insert_many(
            [{"_id": "a", "x": 0.0, "y": "dog"}, {"_id": "A", "x": 1.0, "y": "cat"}]
        )
        db.base.insert_many(
            [{"_id": f"k{i}", "x": float(i % 2), "y": "cd"[i % 2]} for i in range(20)]
        )
        db.create_collection("v", viewOn="base", pipeline=[], collation=ci)
        build(db.v, {"target": "y", "columns": ["x"]})
        rows = P.aggregate(
            db.v, [{"$predict": {"model": "m", "filter": {"_id": {"$in": ["a", "A"]}}}}]
        )
        assert sorted(d["_id"] for d in rows) == ["A", "a"]
    finally:
        db.client.drop_database(db.name)
