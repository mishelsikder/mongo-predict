"""Large collections and documents (staff review, round 2): what is fetched, kept and computed."""

import mongomock
import pytest

from pymongopredict import PredictionModel
from pymongopredict import predictions_api as P
from pymongopredict.embeddings_api.documents import _iter_batches
from pymongopredict.embeddings_api.embed import HashingEmbedder, _CachedEmbedder, _make_embedder
from pymongopredict.predictions_api import models as M
from pymongopredict.predictions_api.stage import _meta_used, _predict_rows


def coll_of(docs):
    c = mongomock.MongoClient().db.c
    c.insert_many(docs)
    return c


def build(c, definition, name="m"):
    P.create_prediction_model(
        c, PredictionModel({"featurizerOptions": {"cache_dir": None}, **definition}, name)
    )


def test_batches_are_cut_by_bytes_as_well_as_count():
    docs = [{"t": "x" * 1000} for _ in range(10)]
    assert [len(b) for b in _iter_batches(docs, 100)] == [10]
    assert [len(b) for b in _iter_batches(docs, 100, max_bytes=2500)] == [3, 3, 3, 1]


def test_training_fetches_only_the_fields_the_model_reads(monkeypatch):
    c = coll_of([{"x": i, "big": "z" * 10_000, "y": i % 2} for i in range(40)])
    seen = []
    real = M.MongoDataset

    def spy(*a, **kw):
        seen.append(kw.get("projection"))
        assert kw.get("streaming")  # only _id and the target are held
        return real(*a, **kw)

    monkeypatch.setattr(M, "MongoDataset", spy)
    build(c, {"target": "y", "columns": ["x"], "embedFields": False})
    assert seen and "big" not in (seen[0] or {"big": 1}) and "x" in seen[0]


def test_probabilities_only_when_the_pipeline_reads_them():
    c = coll_of([{"x": i, "y": "ab"[i % 2]} for i in range(40)])
    build(c, {"target": "y", "columns": ["x"], "embedFields": False})
    fz, est, _ = P._load_prediction_model(c, "m")
    docs = list(c.find().limit(3))
    assert all("probabilities" not in r["meta"] for r in _predict_rows(fz, est, docs, set()))
    rows = _predict_rows(fz, est, docs, {"predictProbabilities"})
    assert all("probabilities" in r["meta"] for r in rows)
    pipeline = [{"$project": {"p": {"$meta": "prediction"}, "s": {"$meta": "predictScore"}}}]
    assert _meta_used(pipeline) == {"prediction", "predictScore"}


def test_the_stored_model_is_downloaded_once_per_build(monkeypatch):
    c = coll_of([{"x": i, "y": i % 2} for i in range(40)])
    build(c, {"target": "y", "columns": ["x"], "embedFields": False})
    M._loaded.clear()
    models = M._models(c)
    calls = []
    real = type(models).find_one

    def spy(self, flt=None, projection=None, *a, **kw):
        calls.append(projection)
        return real(self, flt, projection, *a, **kw)

    monkeypatch.setattr(type(models), "find_one", spy)
    P._load_prediction_model(c, "m")
    P._load_prediction_model(c, "m")
    assert calls.count(None) == 1  # the full record (with the blob) once, then only metadata


def test_an_unusable_cache_directory_warns_and_embeds_without_it(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.warns(UserWarning, match="embedding cache not used"):
        emb = _make_embedder("hashing", cache_dir=str(blocker / "cache"), n_features=8)
    assert not isinstance(emb, _CachedEmbedder) and emb.embed(["a"]).shape == (1, 8)


class CountingEmbedder(HashingEmbedder):
    name = "counting-test"
    texts: list = []

    def embed(self, texts, *a, **kw):
        CountingEmbedder.texts.extend(texts)
        return super().embed(texts, *a, **kw)


def test_training_without_a_cache_embeds_each_document_once():
    CountingEmbedder.texts = []
    c = coll_of([{"t": f"doc {i}", "y": i % 2} for i in range(40)])
    definition = {"target": "y", "embedFields": ["t"], "embeddingModel": CountingEmbedder(16)}
    build(c, definition)
    assert len(CountingEmbedder.texts) == len(set(CountingEmbedder.texts)) == 40
    fz, _, _ = P._load_prediction_model(c, "m")
    assert fz.cache_dir is None
