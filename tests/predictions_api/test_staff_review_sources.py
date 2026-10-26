"""Sources, labels and inputs that production deployments hit (staff review, round 3)."""

import datetime as dt
import os
from types import SimpleNamespace

import mongomock
import pytest
from bson import ObjectId

from pymongopredict import PredictionModel
from pymongopredict import predictions_api as P
from pymongopredict.predictions_api import stage as S

server = pytest.mark.skipif(
    not os.environ.get("MONGOPREDICT_TEST_URI"), reason="set MONGOPREDICT_TEST_URI to a mongod"
)


def coll_of(docs, name="c"):
    c = mongomock.MongoClient().db[name]
    c.insert_many(docs)
    return c


def build(c, definition, name="m"):
    definition = {"embedFields": False, "featurizerOptions": {"cache_dir": None}, **definition}
    P.create_prediction_model(c, PredictionModel(definition, name))
    return P.list_prediction_models(c, name)[0]


def predict(c, name="m", **spec):
    pipeline = [{"$predict": {"model": name, **spec}}, {"$set": {"p": {"$meta": "prediction"}}}]
    return list(P.aggregate(c, pipeline))


def test_placeholder_text_is_a_label_for_categories_and_missing_for_numbers():
    c = coll_of([{"x": i, "continent": ["NA", "EU", "none"][i % 3]} for i in range(60)])
    assert sorted(build(c, {"target": "continent", "columns": ["x"]})["classes"]) == [
        "EU",
        "NA",
        "none",
    ]
    c = coll_of([{"x": i, "y": float(i)} for i in range(30)] + [{"x": 1, "y": "N/A"}])
    assert build(c, {"target": "y", "columns": ["x"]})["numTrainingDocuments"] == 30


def test_predictions_keep_the_stored_label_type():
    users = [ObjectId() for _ in range(3)]
    day = dt.datetime(2020, 1, 1)
    c = coll_of(
        [{"x": i % 3, "u": users[i % 3], "d": day + dt.timedelta(days=i % 2)} for i in range(60)]
    )
    build(c, {"target": "u", "columns": ["x"]})
    assert {d["p"] for d in predict(c)} <= set(users)
    build(c, {"target": "d", "columns": ["x"]}, "md")
    assert all(isinstance(d["p"], dt.datetime) for d in predict(c, "md"))


def test_model_names_cannot_reach_another_collections_model():
    db = mongomock.MongoClient().db
    db["d.a"].insert_many([{"x": i, "y": i % 2} for i in range(20)])
    build(db["d.a"], {"target": "y", "columns": ["x"]}, "b")
    for call in (
        lambda: P._load_prediction_model(db["d"], "a.b"),
        lambda: P.drop_prediction_model(db["d"], "a.b"),
        lambda: predict(db["d"], "a.b"),
    ):
        with pytest.raises(ValueError, match="model names"):
            call()
    assert P.list_prediction_models(db["d.a"])


def test_bad_inputs_raise_clear_errors():
    c = coll_of([{"x": i, "y": i % 2} for i in range(20)])
    bad_models = [
        {"name": "z"},
        ["target"],
        {"definition": ["target"]},
    ]
    for m in bad_models:
        with pytest.raises(ValueError):
            P.create_prediction_model(c, m)
    for d in (
        {"target": "y", "columns": [None]},
        {"target": "y", "columns": "x"},
        {"target": "y", "featurizerOptions": {"bogus": 1}},
        {"target": "y", "featurizerOptions": ["x"]},
    ):
        with pytest.raises(ValueError):
            PredictionModel(d, "z")
    with pytest.raises(ValueError, match="fit"):
        P.create_prediction_model(
            c, PredictionModel({"target": "y", "columns": ["x"]}, "z"), estimator=object()
        )
    build(c, {"target": "y", "columns": ["x"]})
    for pipeline in (
        {"$predict": {"model": "m"}},
        [{"$predict": {"model": "m", "limit": 2**63}}],
        [{"$predict": {"model": "m", "filter": []}}],
    ):
        with pytest.raises(ValueError):
            P.aggregate(c, pipeline)
    snapshot = SimpleNamespace(in_transaction=False, options=SimpleNamespace(snapshot=True))
    with pytest.raises(ValueError, match="snapshot"):
        P.aggregate(c, [{"$predict": {"model": "m"}}], session=snapshot)


def test_a_dropped_temporary_collection_raises_instead_of_losing_rows(monkeypatch):
    c = coll_of([{"x": i, "y": i % 2} for i in range(30)])
    build(c, {"target": "y", "columns": ["x"]})
    real, calls = S._predict_rows, []

    def predict_then_cleanup(*a, **kw):
        calls.append(1)
        if len(calls) == 2:  # another process cleans up while this call runs
            S.cleanup_temporary_collections(c.database, older_than=dt.timedelta(0))
        return real(*a, **kw)

    monkeypatch.setattr(S, "_predict_rows", predict_then_cleanup)
    with pytest.raises(RuntimeError, match="lost predictions"):
        predict(c, batchSize=10)


@server
def test_server_views_time_series_raw_documents_and_duplicate_ids():
    from bson.codec_options import CodecOptions
    from bson.raw_bson import RawBSONDocument
    from pymongo import MongoClient

    db = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])[f"mongopredict_test_src_{os.getpid()}"]
    try:
        base = db.base
        base.insert_many([{"_id": i, "x": i, "y": i % 2, "tags": [1, -1]} for i in range(200)])
        build(base, {"target": "y", "columns": ["x"]})
        expected = {d["_id"]: d["p"] for d in predict(base)}

        # a raw-BSON codec must not read every field as missing
        raw = db.get_collection("base", codec_options=CodecOptions(document_class=RawBSONDocument))
        assert {d["_id"]: d["p"] for d in predict(raw)} == expected
        build(raw, {"target": "y", "columns": ["x"]}, "m2")
        assert P.list_prediction_models(raw, "m2")[0]["numTrainingDocuments"] == 200

        # a view and a time-series collection join through the temporary collection's index
        db.create_collection("v", viewOn="base", pipeline=[{"$match": {"x": {"$gte": 0}}}])
        build(db.v, {"target": "y", "columns": ["x"]})
        got = {d["_id"]: d["p"] for d in predict(db.v, filter={"x": {"$lt": 50}})}
        assert got == {k: v for k, v in expected.items() if k < 50}
        db.create_collection("ts", timeseries={"timeField": "t"})
        t0 = dt.datetime(2024, 1, 1)
        db.ts.insert_many(
            [{"t": t0 + dt.timedelta(minutes=i), "x": i, "y": i % 2} for i in range(50)]
        )
        build(db.ts, {"target": "y", "columns": ["x"]})
        assert len(predict(db.ts, limit=20)) == 20

        # a view that repeats _id values can be neither trained on nor joined
        db.create_collection("u", viewOn="base", pipeline=[{"$unwind": "$tags"}])
        with pytest.raises(ValueError, match="unique _id"):
            build(db.u, {"target": "y", "columns": ["x"]})
        db.command("collMod", "u", viewOn="base", pipeline=[])
        build(db.u, {"target": "y", "columns": ["x"]})
        db.command("collMod", "u", viewOn="base", pipeline=[{"$unwind": "$tags"}])
        with pytest.raises(ValueError, match="unique _id"):
            predict(db.u)
        assert not [n for n in db.list_collection_names() if n.startswith(S.TMP_PREFIX)]
    finally:
        db.client.drop_database(db.name)
