"""Serving stored models safely (staff review, round 4): signatures, versions, caches, labels."""

import os
import pickle
import sqlite3

import mongomock
import numpy as np
import pytest
from bson.binary import Binary
from sklearn.exceptions import NotFittedError

from pymongopredict import DocumentFeaturizer, MongoDataset, PredictionModel
from pymongopredict import predictions_api as P
from pymongopredict.embeddings_api.embed import HashingEmbedder, _CachedEmbedder
from pymongopredict.predictions_api import models as M


def coll_of(docs):
    c = mongomock.MongoClient().db.c
    c.insert_many(docs)
    return c


def build(c, definition, name="m"):
    definition = {"embedFields": False, "featurizerOptions": {"cache_dir": None}, **definition}
    P.create_prediction_model(c, PredictionModel(definition, name))
    return P.list_prediction_models(c, name)[0]


def run(c, *fields, name="m"):
    project = {f: {"$meta": f} for f in fields}
    return list(P.aggregate(c, [{"$predict": {"model": name}}, {"$project": project}]))


class Evil:
    def __reduce__(self):
        return (os.getpid, ())


def test_a_model_key_signs_models_and_refuses_unsigned_or_changed_ones(monkeypatch):
    c = coll_of([{"x": i, "y": i % 2} for i in range(30)])
    monkeypatch.setenv(M.MODEL_KEY_ENV, "secret")
    build(c, {"target": "y", "columns": ["x"]})
    M._loaded.clear()
    assert len(run(c, "prediction")) == 30  # signed and verified
    records = c.database[M.MODELS_COLLECTION]
    records.update_one({"_id": "c.m"}, {"$set": {"model": Binary(pickle.dumps(Evil()))}})
    M._loaded.clear()
    with pytest.raises(PermissionError, match="not signed"):
        run(c, "prediction")
    records.update_one({"_id": "c.m"}, {"$unset": {"modelSignature": ""}})
    with pytest.raises(PermissionError):
        run(c, "prediction")


def test_versions_are_recorded_and_unloadable_blobs_fail_clearly(monkeypatch):
    c = coll_of([{"x": i, "y": i % 2} for i in range(30)])
    rec = build(c, {"target": "y", "columns": ["x"]})
    assert {"scikit-learn", "numpy", "python"} <= set(rec["versions"])
    records = c.database[M.MODELS_COLLECTION]
    records.update_one({"_id": "c.m"}, {"$set": {"versions.scikit-learn": "0.0.1"}})
    M._loaded.clear()
    with pytest.warns(UserWarning, match="scikit-learn 0.0.1"):
        run(c, "prediction")
    records.update_one({"_id": "c.m"}, {"$set": {"model": Binary(b"not a pickle")}})
    M._loaded.clear()
    with pytest.raises(ValueError, match="cannot be loaded.*update_prediction_model"):
        run(c, "prediction")


def test_a_model_dropped_while_loading_gives_a_clear_error(monkeypatch):
    c = coll_of([{"x": i, "y": i % 2} for i in range(30)])
    build(c, {"target": "y", "columns": ["x"]})
    M._loaded.clear()
    records = M._models(c)
    real = type(records).find_one

    def dropped_after_the_metadata(self, flt=None, projection=None, *a, **kw):
        return real(self, flt, projection, *a, **kw) if projection else None

    monkeypatch.setattr(type(records), "find_one", dropped_after_the_metadata)
    with pytest.raises(ValueError, match="was dropped"):
        P._load_prediction_model(c, "m")


def test_probability_keys_match_the_predictions_and_the_record():
    c = coll_of([{"x": i % 5, "stars": i % 5 + 1, "flag": bool(i % 2)} for i in range(100)])
    rec = build(c, {"target": "stars", "columns": ["x"]})
    assert rec["classes"] == ["1", "2", "3", "4", "5"]
    for d in run(c, "prediction", "predictProbabilities"):
        assert type(d["prediction"]) is int and str(d["prediction"]) in d["predictProbabilities"]
    assert build(c, {"target": "flag", "columns": ["x"]}, "f")["classes"] == ["False", "True"]


def test_serving_uses_no_cache_unless_asked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = coll_of([{"t": f"doc {i}", "y": i % 2} for i in range(30)])
    P.create_prediction_model(
        c, PredictionModel({"target": "y", "embedFields": ["t"], "embeddingModel": "hashing"}, "m")
    )
    (tmp_path / ".mongopredict_cache").rename(tmp_path / "training_cache")
    M._loaded.clear()
    run(c, "prediction")
    assert not (tmp_path / ".mongopredict_cache").exists()
    monkeypatch.setenv(M.PREDICT_CACHE_ENV, str(tmp_path / "serve"))
    M._loaded.clear()
    run(c, "prediction")
    assert (tmp_path / "serve" / "embeddings.sqlite").exists()


def test_a_cache_that_fails_after_opening_warns_and_still_embeds(tmp_path):
    emb = _CachedEmbedder(HashingEmbedder(8), str(tmp_path))
    emb.embed(["a"])
    con = sqlite3.connect(emb.path)
    con.execute("DROP TABLE emb")
    con.commit()
    con.close()
    with pytest.warns(UserWarning, match="embedding cache"):
        out = emb.embed(["a", "b"])
    assert out.shape == (2, 8) and np.isfinite(out).all()


def test_dataset_and_featurizer_inputs():
    ds = MongoDataset.from_documents([{"x": i, "y": i} for i in range(10)], target="y")
    assert len(ds.subset(np.asarray(ds.y) > 6)) == 3
    for bad in ([-1], [10], np.ones(3, bool)):
        with pytest.raises((IndexError, ValueError)):
            ds.subset(bad)
    with pytest.raises(ValueError, match="field path"):
        ds.field_values("")
    fz = DocumentFeaturizer(columns=["x"], embed_fields=False, cache_dir=None)
    with pytest.raises(NotFittedError):
        fz.transform([{"x": 1}])
    with pytest.raises(NotFittedError):
        fz.get_feature_names_out()
    with pytest.raises(ValueError, match="more than once"):
        DocumentFeaturizer(columns=["x", "x"], embed_fields=False, cache_dir=None).fit(ds)
