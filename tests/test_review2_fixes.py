"""Regression tests for bugs found by the second review (cache threads, duplicates, $meta rewrites,
view joins with long _ids, MongoSeries frequencies)."""

import datetime as dt
import threading

import mongomock
import numpy as np
import pytest

from pymongopredict import MongoSeries
from pymongopredict import predictions_api as P
from pymongopredict.embeddings_api.embed import BaseEmbedder, _CachedEmbedder


class Echo(BaseEmbedder):
    """Vector = a hash of the text; emits in chunks of 2 and records what it was asked to embed."""

    name = "echo"

    def __init__(self):
        self.asked = []

    def embed(self, texts):
        texts = list(texts)
        self.asked += texts
        out = np.array([[hash(t) % 997, len(t)] for t in texts], np.float32)
        for a in range(0, len(texts), 2):
            self._emit(range(a, min(a + 2, len(texts))), out[a : a + 2])
        return out


def test_threads_sharing_an_embedder_never_cross_cache_rows(tmp_path):
    backend = Echo()
    cache = _CachedEmbedder(backend, str(tmp_path))

    def work(k):
        cache.embed([f"t{k}-{i}" for i in range(40)])

    threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    texts = [f"t{k}-{i}" for k in range(6) for i in range(40)]
    fresh = _CachedEmbedder(Echo(), str(tmp_path)).embed(texts)  # all from the cache
    np.testing.assert_array_equal(fresh, Echo().embed(texts))


def test_duplicate_texts_are_embedded_once(tmp_path):
    backend = Echo()
    E = _CachedEmbedder(backend, str(tmp_path)).embed(["same"] * 5 + ["other"])
    assert backend.asked == ["same", "other"]
    assert (E[:5] == E[0]).all() and not (E[5] == E[0]).all()


def test_meta_reads_the_root_document_and_sortby_fields_are_plain():
    f, rw = P.META_FIELD, P.stage._rewrite
    # $redact with $$DESCEND: the prediction of the document, not of each subdocument
    out = rw(
        [{"$redact": {"$cond": [{"$eq": [{"$meta": "prediction"}, "a"]}, "$$DESCEND", "$$PRUNE"]}}]
    )
    assert f"$$ROOT.{f}.prediction" in str(out)
    # a field named sortBy is an ordinary field; sortBy of $setWindowFields is a sort spec
    assert rw([{"$set": {"sortBy": {"$meta": "prediction"}}}]) == [
        {"$set": {"sortBy": f"$$ROOT.{f}.prediction"}}
    ]
    win = rw([{"$setWindowFields": {"sortBy": {"p": {"$meta": "predictScore"}}, "output": {}}}])
    assert win[0]["$setWindowFields"]["sortBy"] == {f"{f}.score": -1}


@pytest.mark.parametrize(("freq", "ok"), [("BM", True), ("BQ", True), ("MS", True), ("SME", False)])
def test_series_frequencies(freq, ok):
    c = mongomock.MongoClient().db.s
    t0 = dt.datetime(2020, 1, 1)
    c.insert_many([{"ts": t0 + dt.timedelta(days=i), "v": float(i)} for i in range(400)])
    if ok:
        assert len(MongoSeries(c, "ts", "v", freq=freq).series) >= 3
    else:
        with pytest.raises(ValueError, match="no matching pandas period"):
            MongoSeries(c, "ts", "v", freq=freq).load()


@pytest.mark.skipif(
    not __import__("os").environ.get("MONGOPREDICT_TEST_URI"),
    reason="set MONGOPREDICT_TEST_URI to a mongod to run",
)
def test_view_join_with_long_ids_switches_to_a_full_join(monkeypatch, tmp_path):
    import os

    from pymongo import MongoClient

    from pymongopredict import PredictionModel

    db = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])[f"mongopredict_test_ids_{os.getpid()}"]
    try:
        db.base.insert_many(
            [{"_id": f"{i:06d}" + "x" * 200, "t": f"w{i % 4}", "y": i % 2} for i in range(300)]
        )
        db.create_collection("v", viewOn="base", pipeline=[])
        definition = {
            "target": "y",
            "embeddingModel": "hashing",
            "featurizerOptions": {"cache_dir": None},
        }
        P.create_prediction_model(db.v, PredictionModel(definition, name="m"), progress=False)
        monkeypatch.setattr(P.stage, "MAX_JOIN_ID_BYTES", 10_000)  # ~40 long ids
        rows = P.aggregate(
            db.v, [{"$predict": {"model": "m"}}, {"$project": {"p": {"$meta": "prediction"}}}]
        ).to_list()
        assert len(rows) == 300 and {r["p"] for r in rows} <= {0, 1}
        assert not [n for n in db.list_collection_names() if n.startswith(P.TMP_PREFIX)]
    finally:
        db.client.drop_database(db.name)


def test_servers_before_5_0_are_refused(monkeypatch, tmp_path):
    from pymongopredict import PredictionModel

    c = mongomock.MongoClient().db.items
    c.insert_many([{"t": f"w{i % 3}", "y": i % 2} for i in range(30)])
    model = PredictionModel(
        {"target": "y", "embeddingModel": "hashing", "featurizerOptions": {"cache_dir": None}},
        name="m",
    )
    P.create_prediction_model(c, model, progress=False)  # mongomock reports 5.0.5
    old = {"version": "4.4.29", "versionArray": [4, 4, 29, 0]}
    monkeypatch.setattr(type(c.database.client), "server_info", lambda self: old)
    with pytest.raises(
        RuntimeError, match="needs MongoDB 5.0 or later, but this server is MongoDB 4.4.29"
    ):
        P.create_prediction_model(c, model, progress=False)
    with pytest.raises(RuntimeError, match="needs MongoDB 5.0"):
        P.aggregate(c, [{"$predict": {"model": "m"}}])


@pytest.mark.skipif(
    not __import__("os").environ.get("MONGOPREDICT_TEST_URI"),
    reason="set MONGOPREDICT_TEST_URI to a mongod to run",
)
def test_server_version_check_on_a_real_server():
    import os

    from pymongo import MongoClient

    from pymongopredict import PredictionModel

    db = MongoClient(os.environ["MONGOPREDICT_TEST_URI"])[f"mongopredict_test_ver_{os.getpid()}"]
    try:
        db.c.insert_many([{"t": f"w{i % 3}", "y": i % 2} for i in range(30)])
        model = PredictionModel(
            {"target": "y", "embeddingModel": "hashing", "featurizerOptions": {"cache_dir": None}},
            name="m",
        )
        if tuple(db.client.server_info()["versionArray"][:2]) < (5, 0):
            with pytest.raises(RuntimeError, match="needs MongoDB 5.0 or later"):
                P.create_prediction_model(db.c, model, progress=False)
        else:
            P.create_prediction_model(db.c, model, progress=False)
            assert len(P.aggregate(db.c, [{"$predict": {"model": "m"}}]).to_list()) == 30
    finally:
        db.client.drop_database(db.name)
