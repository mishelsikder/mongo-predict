"""Regression tests for edge cases of the prediction model store and the $predict stage."""

import os
import pathlib

import mongomock
import numpy as np
import pytest

from pymongopredict import predictions_api as P
from pymongopredict.embeddings_api.embed import _make_embedder
from pymongopredict.predictions_api import PredictionModel


def definition(tmp_path, **extra):
    return {
        "target": "room_type",
        "columns": ["beds", "address.market"],
        "embedFields": False,
        "featurizerOptions": {"cache_dir": str(tmp_path), "min_frequency": 1},
        **extra,
    }


def regression(tmp_path):
    return {
        "target": "beds",
        "task": "regression",  # (beds has fewer than 8 values: "auto" would pick classification)
        "columns": ["price"],
        "embedFields": False,
        "featurizerOptions": {"cache_dir": str(tmp_path)},
    }


def copy_of(listings):
    c = mongomock.MongoClient().db.listings
    c.insert_many(list(listings.find()))
    return c


def predictions(coll, name="m", n=3):
    return [
        d["p"]
        for d in P.aggregate(
            coll,
            [
                {"$predict": {"model": name, "limit": n}},
                {"$project": {"p": {"$meta": "prediction"}}},
            ],
        )
    ]


def test_same_names_on_another_client_are_separate_models(listings, tmp_path):
    other = copy_of(listings)  # same database / collection names, another client
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    P.create_prediction_model(other, PredictionModel(regression(tmp_path), "m"))
    assert all(isinstance(p, str) for p in predictions(listings))
    assert all(isinstance(p, float) for p in predictions(other))


def test_recreated_model_is_not_served_from_a_stale_cache(listings, tmp_path):
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    predictions(listings)
    rec = P.list_prediction_models(listings, "m")[0]
    P.drop_prediction_model(listings, "m")
    P.create_prediction_model(listings, PredictionModel(regression(tmp_path), "m"))
    assert P.list_prediction_models(listings, "m")[0]["buildId"] != rec["buildId"]
    assert all(isinstance(p, float) for p in predictions(listings))


def test_model_names_and_definitions(listings, tmp_path):
    for bad in ("b.c", "$x", ""):
        with pytest.raises(ValueError, match="model names"):
            PredictionModel(definition(tmp_path), bad)
    d = definition(
        tmp_path,
        embedFields=["name"],
        embeddingModel=_make_embedder("hashing", None, n_features=8),
        featurizerOptions={"cache_dir": pathlib.Path(tmp_path), "min_frequency": 1},
    )
    P.create_prediction_model(listings, PredictionModel(d, "m"))
    stored = P.list_prediction_models(listings, "m")[0]["latestDefinition"]
    assert "HashingEmbedder" in stored["embeddingModel"] and isinstance(
        stored["featurizerOptions"]["cache_dir"], str
    )
    with pytest.raises(ValueError, match="not found"):
        P.update_prediction_model(listings, "nope", definition(tmp_path))


def test_rewrite_of_sort_specs_facets_projections_and_lookups():
    f = P.META_FIELD
    out = P.stage._rewrite(
        [
            {
                "$setWindowFields": {
                    "sortBy": {"s": {"$meta": "predictScore"}},
                    "output": {"r": {"$rank": {}}},
                }
            },
            {
                "$group": {
                    "_id": None,
                    "top": {"$top": {"sortBy": {"s": {"$meta": "predictScore"}}, "output": "$x"}},
                    "docs": {"$push": "$$ROOT"},
                }
            },
            {"$project": {"name": 1}},
            {"$facet": {"a": [{"$limit": 1}]}},
        ]
    )
    assert out[0]["$setWindowFields"]["sortBy"] == {f"{f}.score": -1}
    assert out[1]["$group"]["top"]["$top"]["sortBy"] == {f"{f}.score": -1}
    assert out[1]["$group"]["docs"]["$push"] == {"$unsetField": {"field": f, "input": "$$ROOT"}}
    assert out[2] == {"$project": {"name": 1, f: 1}}
    assert out[3]["$facet"]["a"][-1] == {"$project": {f: 0}}
    with pytest.raises(ValueError, match="sub-pipelines"):
        P.stage._rewrite(
            [
                {
                    "$lookup": {
                        "from": "x",
                        "pipeline": [{"$match": {"$expr": {"$meta": "prediction"}}}],
                        "as": "y",
                    }
                }
            ]
        )


def test_meta_after_an_inclusion_project_and_inside_a_facet(listings, tmp_path):
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    out = list(
        P.aggregate(
            listings,
            [
                {"$predict": {"model": "m", "limit": 4}},
                {"$project": {"name": 1}},
                {"$addFields": {"p": {"$meta": "prediction"}}},
            ],
        )
    )
    assert len(out) == 4 and all("p" in d and P.META_FIELD not in d for d in out)
    (facet,) = P.aggregate(
        listings, [{"$predict": {"model": "m", "limit": 2}}, {"$facet": {"a": [{"$limit": 2}]}}]
    )
    assert all(P.META_FIELD not in d for d in facet["a"])


def test_stage_arguments_are_validated(listings, tmp_path):
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    for spec in ({"limit": 0}, {"limit": -3}, {"limit": 2.5}, {"batchSize": 0}):
        with pytest.raises(ValueError, match="positive integer"):
            P.aggregate(listings, [{"$predict": {"model": "m", **spec}}])


def test_enable_with_positional_session_and_subclasses(listings, tmp_path):
    class Sub(mongomock.Collection):
        pass

    P.enable(mongomock.Collection)
    P.enable(Sub)
    try:
        listings.create_prediction_model(PredictionModel(definition(tmp_path), "m"))
        assert len(list(listings.aggregate([{"$predict": {"model": "m", "limit": 2}}], None))) == 2
    finally:
        P.disable()
    assert "aggregate" not in Sub.__dict__ and not hasattr(
        mongomock.Collection, "create_prediction_model"
    )
    assert mongomock.Collection.aggregate.__name__ == "aggregate"


@pytest.mark.skipif(
    not os.environ.get("MONGOPREDICT_TEST_URI"),
    reason="set MONGOPREDICT_TEST_URI to a mongod to run",
)
def test_server_collation_hint_root_window_and_ids(tmp_path):
    from pymongo import MongoClient

    from pymongopredict import MongoDataset

    client = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])
    db = client[f"mongopredict_test_edge_{os.getpid()}"]
    try:
        c = db.create_collection("c", collation={"locale": "en", "strength": 2})
        names = ["b", "A", "c", "B", "a", "C"] * 4
        c.insert_many(
            [
                {"_id": i, "name": n, "x": i % 3, "y": "u" if i % 2 else "v"}
                for i, n in enumerate(names)
            ]
        )
        c.create_index("x")
        P.create_prediction_model(
            c,
            PredictionModel(
                {
                    "target": "y",
                    "columns": ["x"],
                    "embedFields": False,
                    "featurizerOptions": {"cache_dir": str(tmp_path)},
                },
                "m",
            ),
        )
        stage = {"$predict": {"model": "m"}}
        plain = [d["_id"] for d in c.aggregate([{"$match": {"name": "a"}}])]
        assert [
            d["_id"]
            for d in P.aggregate(
                c, [stage, {"$match": {"name": "a"}}, {"$sort": {"_id": 1}}], hint={"x": 1}
            )
        ] == plain
        (g,) = P.aggregate(c, [stage, {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}}])
        assert len(g["docs"]) == len(names) and all(P.META_FIELD not in d for d in g["docs"])
        out = list(
            P.aggregate(
                c,
                [
                    stage,
                    {
                        "$setWindowFields": {
                            "sortBy": {"s": {"$meta": "predictScore"}},
                            "output": {"r": {"$rank": {}}},
                        }
                    },
                ],
            )
        )
        assert len(out) == len(names) and all("r" in d for d in out)
        ids = db.ids
        ids.insert_many(
            [{"_id": True, "y": "bool"}, {"_id": 1, "y": "int"}, {"_id": {"k": 1}, "y": "doc"}]
        )
        ds = MongoDataset(ids, "y", streaming=True)
        assert [d["y"] for d in ds.documents] == list(ds.y)
        import uuid

        uuids = MongoClient(os.environ["MONGOPREDICT_TEST_URI"], uuidRepresentation="standard")[
            db.name
        ].u
        uuids.insert_many([{"_id": uuid.uuid4(), "y": i % 2, "x": i} for i in range(4)])
        assert [d["x"] for d in MongoDataset(uuids, "y", streaming=True).documents] == [0, 1, 2, 3]
    finally:
        client.drop_database(db.name)


# ---------------------------------------------------------------- round 2
def test_rewrite_keeps_literals_query_values_and_replaced_roots():
    f, rw = P.META_FIELD, P.stage._rewrite
    assert rw([{"$project": {"l": {"$literal": "$$ROOT"}}}]) == [
        {"$project": {"l": {"$literal": "$$ROOT"}, f: 1}}
    ]
    assert rw(
        [{"$match": {"txt": "$$ROOT", "$expr": {"$gt": [{"$meta": "predictScore"}, 0.5]}}}]
    ) == [{"$match": {"txt": "$$ROOT", "$expr": {"$gt": [f"$$ROOT.{f}.score", 0.5]}}}]
    (stage,) = rw([{"$replaceRoot": {"newRoot": {"doc": "$$ROOT"}}}])
    merged = stage["$replaceRoot"]["newRoot"]["$mergeObjects"]
    assert merged[1] == {f: f"${f}"}  # the predictions stay available
    assert merged[0]["$cond"][1] == {"doc": {"$unsetField": {"field": f, "input": "$$ROOT"}}}
    with pytest.raises(ValueError, match="sub-pipelines"):
        rw([{"$merge": {"into": "x", "whenMatched": [{"$set": {"p": {"$meta": "prediction"}}}]}}])


@pytest.mark.parametrize(
    "spec, inclusion",
    [
        ({"a": {"b": 0}}, False),
        ({"a": {"b": False}, "t": 0}, False),
        ({"_id": 1}, True),
        ({"_id": "$beds"}, True),
        ({"_id": 0}, False),
        ({"a": "$b"}, True),
        ({"a": {"$literal": 0}}, True),
        ({"a": 0, "_id": 1}, False),
    ],
)
def test_projection_kinds(spec, inclusion):
    assert P.stage._is_inclusion(spec) is inclusion


def test_cursor_api_like_pymongo(listings, tmp_path):
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    cur = P.aggregate(listings, [{"$predict": {"model": "m", "limit": 3}}])
    assert cur.alive and cur.next()["_id"] is not None and cur.try_next() is not None
    with pytest.raises(ValueError, match="greater than 0"):
        cur.to_list(0)
    assert len(cur.to_list()) == 1 and not cur.alive and cur.try_next() is None


def test_model_cache_old_records_and_validation(listings, tmp_path):
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    key = (listings.database.name, "listings.m")
    assert key in P.models._loaded
    listings.database[P.MODELS_COLLECTION].update_one(
        {"_id": "listings.m"}, {"$unset": {"buildId": ""}}
    )
    P.models._loaded.clear()
    assert len(predictions(listings)) == 3  # a record from before build ids
    P.drop_prediction_model(listings, "m")
    assert key not in P.models._loaded
    for n in (0, -3, 2.5):
        with pytest.raises(ValueError, match="numTrainingDocuments"):
            P.create_prediction_model(
                listings, PredictionModel(definition(tmp_path, numTrainingDocuments=n), "m")
            )


def test_transactions_and_the_reserved_field(listings, tmp_path):
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))

    class InTransaction:
        in_transaction = True

    with pytest.raises(ValueError, match="transaction"):
        P.aggregate(listings, [{"$predict": {"model": "m"}}], session=InTransaction())
    listings.update_one({}, {"$set": {P.META_FIELD: "mine"}})
    P.create_prediction_model(
        listings,
        PredictionModel(definition(tmp_path, embedFields=None, embeddingModel="hashing"), "w"),
    )
    with pytest.raises(ValueError, match="reserved"):
        P.aggregate(listings, [{"$predict": {"model": "w"}}])


@pytest.mark.skipif(
    not os.environ.get("MONGOPREDICT_TEST_URI"),
    reason="set MONGOPREDICT_TEST_URI to a mongod to run",
)
def test_server_user_collation_and_replace_with(tmp_path):
    from pymongo import MongoClient

    client = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])
    db = client[f"mongopredict_test_edge2_{os.getpid()}"]
    try:
        c = db.c
        c.insert_many(
            [{"_id": k, "x": i % 2, "y": "u" if i % 2 else "v"} for i, k in enumerate("aAbBcC")]
        )
        P.create_prediction_model(
            c,
            PredictionModel(
                {
                    "target": "y",
                    "columns": ["x"],
                    "embedFields": False,
                    "featurizerOptions": {"cache_dir": str(tmp_path)},
                },
                "m",
            ),
        )
        ci = {"locale": "en", "strength": 2}
        out = list(
            P.aggregate(c, [{"$predict": {"model": "m"}}, {"$sort": {"_id": 1}}], collation=ci)
        )
        assert sorted(d["_id"] for d in out) == sorted("aAbBcC")
        out = list(
            P.aggregate(
                c,
                [
                    {"$predict": {"model": "m"}},
                    {"$replaceWith": {"$mergeObjects": ["$$ROOT", {"z": 1}]}},
                    {"$project": {"s": {"$meta": "prediction"}}},
                ],
            )
        )
        assert len(out) == 6 and all("s" in d for d in out)
        out = list(
            P.aggregate(
                c,
                [
                    {"$predict": {"model": "m", "limit": 2}},
                    {"$replaceWith": {"doc": "$$ROOT", "p": {"$meta": "prediction"}}},
                ],
            )
        )
        assert all(P.META_FIELD not in d["doc"] and "p" in d for d in out)
        with pytest.raises(Exception, match="document"):
            list(
                P.aggregate(
                    c, [{"$predict": {"model": "m"}}, {"$replaceRoot": {"newRoot": "$nope"}}]
                )
            )
        assert not [n for n in db.list_collection_names() if n.startswith(P.TMP_PREFIX)]
    finally:
        client.drop_database(db.name)


def test_reserved_field_with_explicit_fields(listings, tmp_path):
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))  # columns only
    listings.update_one({}, {"$set": {P.META_FIELD: "user data"}})
    with pytest.raises(ValueError, match="reserved"):
        P.aggregate(listings, [{"$predict": {"model": "m"}}])


# ---------------------------------------------------------------- round 5
def test_rewrite_of_current_redact_and_merge_let():
    rw, f = P.stage._rewrite, P.META_FIELD
    (red,) = rw(
        [
            {
                "$redact": {
                    "$cond": [{"$size": {"$objectToArray": "$$CURRENT"}}, "$$DESCEND", "$$PRUNE"]
                }
            }
        ]
    )
    assert "$$ROOT" not in str(red) and red["$redact"]["$cond"][1] == "$$KEEP"
    with pytest.raises(ValueError, match="let"):
        rw(
            [
                {
                    "$merge": {
                        "into": "t",
                        "let": {"p": {"$meta": "prediction"}},
                        "whenMatched": "merge",
                    }
                }
            ]
        )
    assert rw([{"$merge": {"into": "t", "let": {"p": "$x"}}}]) == [
        {"$merge": {"into": "t", "let": {"p": "$x"}}}
    ]
    assert f in str(rw([{"$addFields": {"p": {"$meta": "prediction"}}}]))


def test_auroc_is_left_out_when_a_class_is_missing_from_the_holdout(listings, tmp_path):
    with pytest.warns(UserWarning, match="stratify"):  # a one-document holdout
        P.create_prediction_model(listings, PredictionModel(definition(tmp_path, testSize=1), "m"))
    metrics = P.list_prediction_models(listings, "m")[0]["metrics"]
    assert "auroc" not in metrics and metrics["numTestDocuments"] == 1


@pytest.mark.skipif(
    not os.environ.get("MONGOPREDICT_TEST_URI"),
    reason="set MONGOPREDICT_TEST_URI to a mongod to run",
)
def test_server_redact_let_and_id_subfields(tmp_path):
    from pymongo import MongoClient

    from pymongopredict import MongoDataset

    client = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])
    db = client[f"mongopredict_test_edge5_{os.getpid()}"]
    try:
        c = db.c
        c.insert_many(
            [
                {"_id": i, "x": i, "level": 1, "y": "u" if i % 2 else "v", "sub": {"k": 1}}
                for i in range(6)
            ]
        )
        P.create_prediction_model(
            c,
            PredictionModel(
                {
                    "target": "y",
                    "columns": ["x"],
                    "embedFields": False,
                    "featurizerOptions": {"cache_dir": str(tmp_path)},
                },
                "m",
            ),
        )
        redact = {"$redact": {"$cond": [{"$eq": ["$level", 1]}, "$$DESCEND", "$$PRUNE"]}}
        out = list(
            P.aggregate(
                c,
                [
                    {"$predict": {"model": "m", "limit": 2}},
                    redact,
                    {"$project": {"p": {"$meta": "prediction"}}},
                ],
            )
        )
        assert len(out) == 2 and all("p" in d for d in out)
        size = {
            "$redact": {
                "$cond": [
                    {"$gt": [{"$size": {"$objectToArray": "$$CURRENT"}}, 2]},
                    "$$DESCEND",
                    "$$PRUNE",
                ]
            }
        }
        plain = list(c.aggregate([{"$limit": 1}, size, {"$project": {"sub": 1}}]))
        assert (
            list(
                P.aggregate(
                    c, [{"$predict": {"model": "m", "limit": 1}}, size, {"$project": {"sub": 1}}]
                )
            )
            == plain
        )
        flt = {"$expr": {"$lte": ["$x", "$$k"]}}
        assert (
            len(list(P.aggregate(c, [{"$predict": {"model": "m", "filter": flt}}], let={"k": 1})))
            == 2
        )
        ids = db.ids
        ids.insert_many([{"_id": {"s": 1, "k": i}, "y": i} for i in range(4)])
        assert MongoDataset(ids, "y", streaming=True).field_values("_id.k") == [0, 1, 2, 3]
        assert [d["y"] for d in MongoDataset(ids, "_id.k", streaming=True).documents] == [
            0,
            1,
            2,
            3,
        ]
    finally:
        client.drop_database(db.name)


def test_let_without_a_filter_on_mongomock_and_current_rebinding(listings, tmp_path):
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    out = list(P.aggregate(listings, [{"$predict": {"model": "m", "limit": 2}}], let={"n": 2}))
    assert len(out) == 2
    (stage,) = P.stage._rewrite(
        [{"$project": {"c": {"$let": {"vars": {"CURRENT": "$x"}, "in": "$$CURRENT"}}}}]
    )
    assert "$cond" in str(stage)  # $$CURRENT may be rebound to a non-document


@pytest.mark.parametrize(
    "bad",
    [{"task": "foo"}, {"testSize": 1.0}, {"testSize": -1}, {"testSize": True}, {"testSize": "x"}],
)
def test_task_and_test_size_are_validated(listings, tmp_path, bad):
    with pytest.raises(ValueError, match="task|testSize"):
        P.create_prediction_model(listings, PredictionModel(definition(tmp_path, **bad), "m"))


@pytest.mark.skipif(
    not os.environ.get("MONGOPREDICT_TEST_URI"),
    reason="set MONGOPREDICT_TEST_URI to a mongod to run",
)
def test_server_operator_projections_in_both_modes():
    from pymongo import MongoClient

    from pymongopredict import MongoDataset

    client = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])
    db = client[f"mongopredict_test_edge7_{os.getpid()}"]
    try:
        db.c.insert_many(
            [{"_id": i, "a": i, "b": {"p": i}, "c": [i, 10 + i, 20 + i]} for i in range(3)]
        )
        for target, proj in (("c.0", {"c": {"$slice": -1}}), ("a", {"_id": 1, "c": {"$slice": 1}})):
            mem, stream = (
                MongoDataset(db.c, target, projection=proj, streaming=s) for s in (False, True)
            )
            assert len(mem) == len(stream) == 3 and list(mem.y) == list(stream.y)
        db.g.insert_many([{"_id": i, "y": i % 2, "g": f"x{i}"} for i in range(4)])
        proj = {"y": 1, "gu": {"$toUpper": "$g"}}
        mem, stream = (MongoDataset(db.g, "y", projection=proj, streaming=s) for s in (False, True))
        assert mem.field_values("gu") == stream.field_values("gu") == ["X0", "X1", "X2", "X3"]
        proj = {"a": 1, "c": {"$elemMatch": {"$gt": 10}}}
        mem, stream = (MongoDataset(db.c, "a", projection=proj, streaming=s) for s in (False, True))
        assert mem.field_values("c.0") == stream.field_values("c.0")
    finally:
        client.drop_database(db.name)


# ---------------------------------------------------------------- round 7
def test_a_superseded_build_never_overwrites_a_newer_one(listings, tmp_path, monkeypatch):
    real_fit = P.models._fit
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))  # version 1

    def fit_then_rebuild(*a, fail=False, **k):
        out = real_fit(*a, **k)
        monkeypatch.setattr(P.models, "_fit", real_fit)
        P.create_prediction_model(
            listings, PredictionModel(regression(tmp_path), "m")
        )  # starts later
        if fail:
            raise RuntimeError("old build failed")
        return out

    monkeypatch.setattr(P.models, "_fit", fit_then_rebuild)
    with pytest.warns(UserWarning, match="superseded"):
        P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    rec = P.list_prediction_models(listings, "m")[0]
    assert rec["status"] == "READY" and rec["task"] == "regression" and rec["version"] == 3
    monkeypatch.setattr(P.models, "_fit", lambda *a, **k: fit_then_rebuild(*a, fail=True, **k))
    with pytest.raises(RuntimeError, match="old build failed"):
        P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    rec = P.list_prediction_models(listings, "m")[0]
    assert rec["status"] == "READY" and rec["version"] == 5 and "message" not in rec


def test_interrupted_and_oversized_builds_are_marked_failed(listings, tmp_path, monkeypatch):
    class Interrupted:
        def fit(self, X, y):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        P.create_prediction_model(
            listings, PredictionModel(definition(tmp_path), "m"), estimator=Interrupted()
        )
    rec = P.list_prediction_models(listings, "m")[0]
    assert rec["status"] == "FAILED" and "interrupted" in rec["message"]
    monkeypatch.setattr(P.models, "MAX_RECORD_BYTES", 1000)
    with pytest.raises(ValueError, match="16 MB"):
        P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    assert P.list_prediction_models(listings, "m")[0]["status"] == "FAILED"


def test_records_from_before_version_counters_keep_counting(listings, tmp_path):
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    listings.database[P.MODELS_COLLECTION].update_one(
        {"_id": "listings.m"}, {"$set": {"version": 7}, "$unset": {"nextVersion": ""}}
    )
    P.create_prediction_model(listings, PredictionModel(definition(tmp_path), "m"))
    assert P.list_prediction_models(listings, "m")[0]["version"] == 8


def test_auto_task_counts_distinct_numbers_and_drops_blank_strings(listings, tmp_path):
    def build(target, **extra):
        P.create_prediction_model(
            listings,
            PredictionModel(
                {
                    "target": target,
                    "columns": ["accommodates"] if target != "beds" else ["price"],
                    "embedFields": False,
                    "featurizerOptions": {"cache_dir": str(tmp_path)},
                    **extra,
                },
                "m",
            ),
        )
        return P.list_prediction_models(listings, "m")[0]

    listings.update_many(
        {"beds": 1}, {"$set": {"beds": ""}}
    )  # "" for a missing value, as in sample_mflix
    listings.update_many({}, [{"$set": {"accommodates": "$beds"}}])
    listings.update_one({"beds": 2}, {"$set": {"price": ""}})
    rec = build("beds")  # 3 distinct numbers: categories, without a "" class
    assert rec["task"] == "classification" and "" not in rec["classes"]
    assert rec["numTrainingDocuments"] == listings.count_documents({"beds": {"$ne": ""}})
    # predictions come back as stored (beds are ints), not as the floats the estimator saw
    assert all(type(p) is int for p in predictions(listings))
    rec = build("price")  # many distinct numbers (Decimal128 and a blank string)
    assert rec["task"] == "regression"
    assert build("beds", task="regression")["task"] == "regression"
    flags = P.models._auto_task
    assert flags(np.array([0.0, 1.0] * 10)) == "classification"
    assert (
        flags(np.arange(8, dtype=float)) == "regression"
        and flags(np.arange(7, dtype=float)) == "classification"
    )
