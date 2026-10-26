"""Stopping a run partway (Ctrl-C) keeps the finished work and leaves no mess behind."""

import hashlib

import mongomock
import numpy as np
import pytest

from pymongopredict import DocumentFeaturizer, PredictionModel
from pymongopredict import predictions_api as P
from pymongopredict.embeddings_api.embed import BaseEmbedder, HashingEmbedder, _CachedEmbedder


def vec(text):
    seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=4).astype(np.float32)


class Chunked(BaseEmbedder):
    """Embeds in chunks of 3, reporting each finished chunk; raises Ctrl-C after ``stop`` texts."""

    name = "chunked"

    def __init__(self, stop=None):
        self.stop, self.done = stop, 0

    def embed(self, texts):
        texts = list(texts)
        out = np.zeros((len(texts), 4), np.float32)
        for a in range(0, len(texts), 3):
            if self.stop is not None and self.done >= self.stop:
                raise KeyboardInterrupt
            b = min(a + 3, len(texts))
            out[a:b] = [vec(t) for t in texts[a:b]]
            self.done += b - a
            self._emit(range(a, b), out[a:b])
        return out


def test_an_interrupted_call_keeps_the_finished_rows(tmp_path):
    texts = [f"doc {i}" for i in range(10)]
    first = Chunked(stop=6)
    with pytest.raises(KeyboardInterrupt):
        _CachedEmbedder(first, str(tmp_path)).embed(texts)
    again = Chunked()
    E = _CachedEmbedder(again, str(tmp_path)).embed(texts)
    assert again.done == 4  # only the 4 texts not finished before the interruption
    np.testing.assert_allclose(E, [vec(t) for t in texts])


def test_featurizer_says_what_was_kept(tmp_path, capsys):
    docs = [{"text": f"item {i}"} for i in range(700)]
    fz = DocumentFeaturizer(embedding_model=Chunked(stop=600), cache_dir=str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        fz.fit_transform(docs)
    err = capsys.readouterr().err
    assert "Interrupted after embedding 512 documents" in err and "cached in" in err
    fz = DocumentFeaturizer(embedding_model=Chunked(stop=1), cache_dir=None)
    with pytest.raises(KeyboardInterrupt):
        fz.fit_transform(docs)
    assert "not cached" in capsys.readouterr().err


@pytest.fixture
def items():
    c = mongomock.MongoClient().db.items
    c.insert_many([{"text": f"thing {i}", "y": "a" if i % 2 else "b"} for i in range(80)])
    return c


def model(tmp_path):
    return PredictionModel(
        {
            "target": "y",
            "embeddingModel": "hashing",
            "featurizerOptions": {"cache_dir": tmp_path and str(tmp_path)},
        },
        name="m",
    )


def interrupt(*a, **k):
    raise KeyboardInterrupt


def test_interrupted_refit_keeps_the_previous_version(items, tmp_path, monkeypatch):
    P.create_prediction_model(items, model(tmp_path), progress=False)
    monkeypatch.setattr(HashingEmbedder, "embed", interrupt)
    with pytest.raises(KeyboardInterrupt):  # (no cache: the refit has to embed)
        P.create_prediction_model(items, model(None), progress=False)
    rec = P.list_prediction_models(items, "m")[0]
    assert rec["status"] == "FAILED" and "interrupted" in rec["message"] and rec["queryable"]
    monkeypatch.undo()
    rows = P.aggregate(items, [{"$predict": {"model": "m"}}]).to_list()
    assert len(rows) == 80  # the previous version still answers


def test_interrupted_predict_drops_its_temporary_collection(items, tmp_path, monkeypatch):
    P.create_prediction_model(items, model(tmp_path), progress=False)
    P.models._loaded.clear()
    monkeypatch.setattr(HashingEmbedder, "embed", interrupt)
    with pytest.raises(KeyboardInterrupt):
        P.aggregate(items, [{"$predict": {"model": "m"}}]).to_list()
    assert not [n for n in items.database.list_collection_names() if n.startswith(P.TMP_PREFIX)]


def test_interrupted_fit_without_a_cache_says_nothing_was_kept(items, capsys):
    definition = {
        "target": "y",
        "embeddingModel": Chunked(stop=30),
        "featurizerOptions": {"cache_dir": None},
    }
    with pytest.raises(KeyboardInterrupt):
        P.create_prediction_model(items, PredictionModel(definition, name="n"), progress=False)
    err = capsys.readouterr().err
    assert "temporary cache that is deleted" in err
    assert "marked FAILED and there is no earlier version" in err
    assert P.list_prediction_models(items, "n")[0]["status"] == "FAILED"
