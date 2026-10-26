import os

import mongomock
import numpy as np
import pytest

from pymongopredict import (
    PredictionModel,
    create_prediction_model,
    drop_prediction_model,
    list_prediction_models,
)
from pymongopredict import predictions_api as P


def room_model(tmp_path, **extra):
    return PredictionModel(
        {
            "target": "room_type",
            "columns": ["beds", "price", "address.market"],
            "embedFields": ["name", "summary"],
            "embeddingModel": "hashing",
            "embeddingKwargs": {"n_features": 32},
            "featurizerOptions": {"cache_dir": str(tmp_path), "min_frequency": 1},
            **extra,
        },
        name="room_type",
    )


def test_create_list_drop(listings, tmp_path):
    assert create_prediction_model(listings, room_model(tmp_path)) == "room_type"
    (rec,) = list_prediction_models(listings)
    assert rec["status"] == "READY" and rec["queryable"] and rec["task"] == "classification"
    assert rec["numTrainingDocuments"] == 120 and "model" not in rec
    assert rec["metrics"]["accuracy"] > 0.8 and rec["metrics"]["numTestDocuments"] == 24
    assert set(rec["classes"]) == {"Entire home/apt", "Private room", "Shared room"}
    create_prediction_model(listings, room_model(tmp_path))  # rebuild bumps the version
    assert list_prediction_models(listings, "room_type")[0]["version"] == 2
    drop_prediction_model(listings, "room_type")
    assert list_prediction_models(listings) == []


def test_meta_in_project_sort_and_limit(listings, tmp_path):
    create_prediction_model(listings, room_model(tmp_path))
    out = P.aggregate(
        listings,
        [
            {
                "$predict": {
                    "model": "room_type",
                    "filter": {"address.market": "Porto"},
                    "limit": 10,
                    "batchSize": 3,
                }
            },
            {
                "$project": {
                    "room_type": 1,
                    "pred": {"$meta": "prediction"},
                    "score": {"$meta": "predictScore"},
                    "proba": {"$meta": "predictProbabilities"},
                }
            },
            {"$sort": {"score": -1}},
        ],
    ).to_list()
    assert len(out) == 10
    assert all(set(d) == {"_id", "room_type", "pred", "score", "proba"} for d in out)
    assert [d["score"] for d in out] == sorted((d["score"] for d in out), reverse=True)
    assert all(
        abs(sum(d["proba"].values()) - 1) < 1e-6 and d["score"] == max(d["proba"].values())
        for d in out
    )
    assert sum(d["pred"] == d["room_type"] for d in out) >= 8
    porto = {d["_id"] for d in listings.find({"address.market": "Porto"})}
    assert {d["_id"] for d in out} <= porto


def test_sort_by_meta_hides_internal_field(listings, tmp_path):
    create_prediction_model(listings, room_model(tmp_path))
    out = list(
        P.aggregate(
            listings,
            [
                {"$predict": {"model": "room_type"}},
                {"$sort": {"s": {"$meta": "predictScore"}}},
                {"$limit": 5},
            ],
        )
    )
    assert len(out) == 5 and all(P.META_FIELD not in d for d in out)
    assert set(out[0]) == set(listings.find_one())  # whole documents come back


def test_group_and_out(listings, tmp_path):
    create_prediction_model(listings, room_model(tmp_path))
    counts = list(
        P.aggregate(
            listings,
            [
                {"$predict": {"model": "room_type"}},
                {"$group": {"_id": {"$meta": "prediction"}, "n": {"$sum": 1}}},
            ],
        )
    )
    assert sum(c["n"] for c in counts) == 120
    P.aggregate(
        listings,
        [
            {"$predict": {"model": "room_type"}},
            {"$addFields": {"predicted_room_type": {"$meta": "prediction"}}},
            {"$out": "scored"},
        ],
    )
    db = listings.database
    assert db.scored.count_documents({}) == 120 and P.META_FIELD not in db.scored.find_one()
    assert not [c for c in db.list_collection_names() if c.startswith(P.TMP_PREFIX)]


def test_regression(listings, tmp_path):
    create_prediction_model(
        listings,
        PredictionModel(
            {
                "target": "beds",
                "task": "regression",  # (fewer than 8 values: "auto" would pick classification)
                "columns": ["price"],
                "embedFields": False,
                "featurizerOptions": {"cache_dir": str(tmp_path)},
            },
            name="beds",
        ),
    )
    rec = list_prediction_models(listings, "beds")[0]
    assert rec["task"] == "regression" and rec["metrics"]["r2"] > 0.5
    out = list(
        P.aggregate(
            listings,
            [
                {"$predict": {"model": "beds", "limit": 4}},
                {"$project": {"p": {"$meta": "prediction"}, "s": {"$meta": "predictScore"}}},
            ],
        )
    )
    assert len(out) == 4 and all(isinstance(d["p"], float) and "s" not in d for d in out)


def test_training_filter_on_target_and_non_numeric_targets(listings, tmp_path):
    listings.update_many({"beds": 1}, {"$set": {"beds": ""}})  # non-numeric targets are dropped
    create_prediction_model(
        listings,
        PredictionModel(
            {
                "target": "beds",
                "task": "regression",
                "filter": {"beds": {"$ne": 4}},  # a filter on the target itself is kept
                "columns": ["price"],
                "embedFields": False,
                "featurizerOptions": {"cache_dir": str(tmp_path)},
            },
            name="beds",
        ),
    )
    n = listings.count_documents({"beds": {"$nin": ["", 4]}})
    assert list_prediction_models(listings, "beds")[0]["numTrainingDocuments"] == n


def test_errors(listings, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        P.aggregate(listings, [{"$predict": {"model": "nope"}}])
    with pytest.raises(ValueError, match="first stage"):
        P.aggregate(listings, [{"$match": {}}, {"$predict": {"model": "room_type"}}])
    with pytest.raises(ValueError, match="first stage"):
        P.aggregate(listings, [{"$match": {}}, {"$facet": {"a": [{"$predict": {"model": "x"}}]}}])
    with pytest.raises(ValueError, match="unknown fields"):
        P.aggregate(listings, [{"$predict": {"model": "x", "index": "y"}}])
    with pytest.raises(ValueError, match="target"):
        PredictionModel({"columns": ["beds"]})
    assert len(list(P.aggregate(listings, [{"$match": {"beds": 1}}]))) > 0  # untouched


def test_enable_patches_collection(listings, tmp_path):
    P.enable(mongomock.Collection)
    try:
        listings.create_prediction_model(room_model(tmp_path))
        assert listings.list_prediction_models()[0]["status"] == "READY"
        out = list(
            listings.aggregate(
                [
                    {"$predict": {"model": "room_type", "limit": 3}},
                    {"$project": {"p": {"$meta": "prediction"}}},
                ]
            )
        )
        assert len(out) == 3 and all("p" in d for d in out)
        assert len(list(listings.aggregate([{"$match": {"beds": 1}}]))) > 0
    finally:
        P.disable()
    assert not hasattr(mongomock.Collection, "create_prediction_model")


@pytest.mark.skipif(
    not os.environ.get("MONGOPREDICT_TEST_URI"),
    reason="set MONGOPREDICT_TEST_URI to a mongod to run",
)
def test_merge_back_on_real_server(listings, tmp_path):
    from pymongo import MongoClient

    client = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])
    db = client[f"mongopredict_test_{os.getpid()}"]
    try:
        db.listings.insert_many(list(listings.find()))
        P.enable()
        db.listings.create_prediction_model(room_model(tmp_path))
        db.listings.aggregate(
            [
                {"$predict": {"model": "room_type", "filter": {"address.market": "Porto"}}},
                {
                    "$project": {
                        "predicted_room_type": {"$meta": "prediction"},
                        "p": {"$meta": "predictScore"},
                    }
                },
                {
                    "$merge": {
                        "into": "listings",
                        "on": "_id",
                        "whenMatched": "merge",
                        "whenNotMatched": "discard",
                    }
                },
            ]
        )
        n_porto = db.listings.count_documents({"address.market": "Porto"})
        assert db.listings.count_documents({"predicted_room_type": {"$exists": True}}) == n_porto
        assert db.listings.count_documents({P.META_FIELD: {"$exists": True}}) == 0
        top = list(
            db.listings.aggregate(
                [
                    {"$predict": {"model": "room_type"}},
                    {"$sort": {"s": {"$meta": "predictScore"}}},
                    {"$limit": 3},
                ]
            )
        )
        assert len(top) == 3
    finally:
        P.disable()
        client.drop_database(db.name)


def test_progress_bars_show_each_step(listings, tmp_path, capsys):
    from pymongopredict.embeddings_api.features import _on_documents

    create_prediction_model(listings, room_model(tmp_path), progress=True)
    err = capsys.readouterr().err
    steps = [
        "[1/5] embedding the training part",
        "[2/5] fitting on the training part",
        "[3/5] embedding and scoring the holdout part",
        "[4/5] embedding all documents",
        "[5/5] fitting the final model",
    ]
    assert [err.index(s) for s in steps] == sorted(err.index(s) for s in steps)
    assert "96/96" in err and "24/24" in err and "120/120" in err  # documents counted
    assert _on_documents.get() is None  # the hook is reset after training

    create_prediction_model(listings, room_model(tmp_path, testSize=0), progress=True)
    err = capsys.readouterr().err
    assert "[1/2] embedding all documents" in err and "[2/2] fitting the final model" in err


def test_progress_off(listings, tmp_path, capsys):
    P.enable(mongomock.collection.Collection)
    try:
        listings.create_prediction_model(room_model(tmp_path), progress=False)
        P.update_prediction_model(
            listings, "room_type", room_model(tmp_path).definition, progress=False
        )
    finally:
        P.disable()
    assert "embedding" not in capsys.readouterr().err


def test_default_estimator_is_xgboost(listings, tmp_path):
    from xgboost import XGBRegressor

    from pymongopredict.predictions_api.models import _estimator, _XGBLabelClassifier

    assert isinstance(_estimator("auto", "classification"), _XGBLabelClassifier)
    assert isinstance(_estimator("auto", "regression"), XGBRegressor)
    create_prediction_model(listings, room_model(tmp_path), progress=False)
    fz, est, _ = P._load_prediction_model(listings, "room_type")
    assert isinstance(est, _XGBLabelClassifier)
    assert set(est.classes_) == {
        "Entire home/apt",
        "Private room",
        "Shared room",
    }  # labels, not 0..k-1
    rows = P.aggregate(
        listings,
        [
            {"$predict": {"model": "room_type"}},
            {"$project": {"p": {"$meta": "prediction"}, "s": {"$meta": "predictProbabilities"}}},
        ],
    ).to_list()
    assert {r["p"] for r in rows} <= set(est.classes_)
    assert all(set(r["s"]) == set(est.classes_) for r in rows)


def test_review_fixes(listings, tmp_path):
    import pickle

    from pymongopredict.embeddings_api.embed import VoyageEmbedder
    from pymongopredict.predictions_api import models as M

    base = {"target": "room_type", "embeddingModel": "voyage-4-lite"}
    for bad in (
        {**base, "embeddingKwargs": {"api_key": "k"}},
        {**base, "featurizerOptions": {"embedding_kwargs": {"api_key": "k"}}},
        {**base, "embeddingModel": VoyageEmbedder(api_key="k")},
    ):
        with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
            PredictionModel(bad)
    # a key never reaches the stored blob, even past the definition check
    fz = M.DocumentFeaturizer(embedding_model=VoyageEmbedder(api_key="secret-key"))
    assert b"secret-key" not in M._dump(fz, None)
    # models pickled under the old class name still load
    X = [[float(i % 2)] for i in range(40)]
    est = M._XGBLabelClassifier().fit(X, ["a" if x[0] == 0 else "b" for x in X])
    blob = pickle.dumps(est, protocol=0)  # (text protocol: class names are plain lines)
    blob = blob.replace(b"\n_XGBLabelClassifier\n", b"\nXGBLabelClassifier\n")
    assert b"_XGBLabelClassifier" not in blob
    assert list(pickle.loads(blob).predict(X)) == list(est.predict(X))
    assert est.score([[1.0], [0.0]], ["b", "a"]) == 1.0
    # a testSize count must leave training documents
    with pytest.raises(ValueError, match="no training documents"):
        create_prediction_model(listings, room_model(tmp_path, testSize=120), progress=False)
    assert "xgboost" in M._versions()


def test_float_positions_are_refused(listings):
    from pymongopredict import MongoDataset

    with pytest.raises(IndexError, match="integers"):
        MongoDataset(listings)[np.array([1.7, 2.2])]


def test_currency_parsing_is_linear_and_unchanged():
    import time

    from pymongopredict.predictions_api.models import _as_numbers

    values = ["$1,100", "€ 5", "5 £", "¥ 7 ¥", "-3.5", "", "$", "$$", "12,345.5$"]
    got = _as_numbers(np.array(values, dtype=object))
    assert got is not None and list(got[:5]) == [1100.0, 5.0, 5.0, 7.0, -3.5]
    assert np.isnan(got[5]) and np.isnan(got[6]) and np.isnan(got[7]) and got[8] == 12345.5
    assert _as_numbers(np.array(["a b", "$ x"], dtype=object)) is None  # text stays text
    long = np.array(["a" + " " * 200_000 + "b", "c"], dtype=object)
    t = time.perf_counter()
    assert _as_numbers(long) is None
    assert time.perf_counter() - t < 2  # (the old regex took minutes here)
