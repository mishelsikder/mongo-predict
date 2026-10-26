"""The Voyage API embedder against a fake ``voyageai`` module: no network and no API key."""

import hashlib
import os
import sys
import types

import mongomock
import numpy as np
import pytest

from pymongopredict import DocumentFeaturizer, PredictionModel
from pymongopredict import predictions_api as P
from pymongopredict.embeddings_api.embed import VoyageEmbedder, _make_embedder


class FakeClient:
    """Records how it was built and every request; returns a vector derived from each text."""

    instances: list = []

    def __init__(self, api_key=None, max_retries=0, timeout=None, base_url=None):
        self.init = {"api_key": api_key, "max_retries": max_retries, "timeout": timeout}
        self.calls = []
        FakeClient.instances.append(self)

    def embed(self, texts, model=None, input_type=None, truncation=True, output_dimension=None):
        assert isinstance(texts, list) and 1 <= len(texts) <= 1000
        self.calls.append(
            {
                "n": len(texts),
                "model": model,
                "input_type": input_type,
                "truncation": truncation,
                "output_dimension": output_dimension,
            }
        )
        dim = output_dimension or 1024
        vecs = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=dim)
            vecs.append((v / np.linalg.norm(v)).tolist())
        return types.SimpleNamespace(embeddings=vecs)


@pytest.fixture
def fake_voyage(monkeypatch):
    FakeClient.instances.clear()
    monkeypatch.setitem(sys.modules, "voyageai", types.SimpleNamespace(Client=FakeClient))
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    return FakeClient


def test_requests_respect_text_and_token_limits(fake_voyage):
    v = VoyageEmbedder("voyage-4-large", output_dimension=256, max_retries=3, timeout=30)
    E = v.embed([f"doc {i}" for i in range(2500)])
    assert E.shape == (2500, 256) and E.dtype == np.float32
    (client,) = fake_voyage.instances
    assert client.init == {"api_key": None, "max_retries": 3, "timeout": 30}
    assert [c["n"] for c in client.calls] == [1000, 1000, 500]  # at most 1,000 texts per request
    assert client.calls[0] == {
        "n": 1000,
        "model": "voyage-4-large",
        "input_type": "document",
        "truncation": True,
        "output_dimension": 256,
    }
    # long texts: voyage-4-large takes 120K tokens per request (texts estimated at characters / 3)
    long = ["x" * 30_000] * 40  # about 10,000 tokens each
    v.embed(long)
    sizes = [c["n"] for c in client.calls[3:]]
    assert sum(sizes) == 40 and max(sizes) * 10_000 <= 0.8 * 120_000


def test_lite_takes_bigger_requests_than_large(fake_voyage):
    texts = ["y" * 3_000] * 400  # about 1,000 tokens each
    VoyageEmbedder("voyage-4-lite").embed(texts)
    VoyageEmbedder("voyage-4-large").embed(texts)
    lite, large = fake_voyage.instances
    assert len(lite.calls) == 1 and len(large.calls) > 1


def test_clear_errors(fake_voyage, monkeypatch):
    with pytest.raises(ValueError, match="not a text embedding model"):
        _make_embedder("voyage-multimodal-3.5")
    with pytest.raises(ValueError, match="not a text embedding model"):
        VoyageEmbedder("voyage-context-3")
    with pytest.raises(ValueError, match="batch_size"):
        VoyageEmbedder(batch_size=5000)
    monkeypatch.delenv("VOYAGE_API_KEY")
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageEmbedder().embed(["x"])
    VoyageEmbedder(api_key="k").embed(["x"])  # a key passed directly is enough
    assert fake_voyage.instances[-1].init["api_key"] == "k"
    monkeypatch.setitem(sys.modules, "voyageai", None)  # not installed
    with pytest.raises(ImportError, match="reinstall pymongopredict"):
        VoyageEmbedder(api_key="k").embed(["x"])


def test_prediction_model_with_an_api_model(fake_voyage, tmp_path):
    coll = mongomock.MongoClient().db.items
    coll.insert_many(
        [
            {"text": f"item {i} {'red' if i % 2 else 'blue'}", "label": "a" if i % 2 else "b"}
            for i in range(60)
        ]
    )
    definition = {
        "target": "label",
        "embeddingModel": "voyage-4-lite",
        "embeddingKwargs": {"output_dimension": 256, "max_retries": 2},
        "featurizerOptions": {"cache_dir": str(tmp_path)},
    }
    P.create_prediction_model(coll, PredictionModel(definition, name="label"), progress=False)
    (rec,) = P.list_prediction_models(coll)
    assert rec["status"] == "READY"
    calls = [c for client in fake_voyage.instances for c in client.calls]
    assert calls and all(
        c["model"] == "voyage-4-lite" and c["output_dimension"] == 256 for c in calls
    )
    n_before = sum(c["n"] for c in calls)
    rows = P.aggregate(
        coll, [{"$predict": {"model": "label"}}, {"$project": {"p": {"$meta": "prediction"}}}]
    ).to_list()
    assert len(rows) == 60 and {r["p"] for r in rows} <= {"a", "b"}
    calls = [c for client in fake_voyage.instances for c in client.calls]
    assert sum(c["n"] for c in calls) >= n_before  # $predict embeds through the API too


def test_api_key_is_never_stored_with_a_model():
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        PredictionModel(
            {
                "target": "y",
                "embeddingModel": "voyage-4-lite",
                "embeddingKwargs": {"api_key": "secret"},
            }
        )


def test_a_stored_api_model_loads_and_predicts_in_a_new_process(fake_voyage, tmp_path):
    """The stored model holds no client: loading rebuilds the embedder, which reads the key from the
    environment when it first embeds."""
    from pymongopredict.predictions_api import models as M

    coll = mongomock.MongoClient().db.items2
    coll.insert_many([{"text": f"n {i % 3}", "y": float(i % 10)} for i in range(40)])
    P.create_prediction_model(
        coll,
        PredictionModel(
            {"target": "y", "embeddingModel": "voyage-4", "featurizerOptions": {"cache_dir": None}},
            name="y",
        ),
        progress=False,
    )
    M._loaded.clear()  # as a fresh process would start
    rows = P.aggregate(
        coll,
        [{"$predict": {"model": "y", "limit": 5}}, {"$project": {"p": {"$meta": "prediction"}}}],
    ).to_list()
    assert len(rows) == 5 and all(isinstance(r["p"], float) for r in rows)
    assert fake_voyage.instances[-1].calls[-1]["model"] == "voyage-4"


def test_the_api_key_never_reaches_a_pickle(fake_voyage, monkeypatch):
    import copy
    import pickle

    from pymongopredict.predictions_api.models import _dump

    secret = "pa-test-secret-key-0123456789"
    monkeypatch.setenv("VOYAGE_API_KEY", secret)

    class EnvClient(FakeClient):  # resolves the key from the environment, as voyageai.Client does
        def __init__(self, api_key=None, **kw):
            super().__init__(api_key=api_key or os.environ["VOYAGE_API_KEY"], **kw)
            self.api_key = api_key or os.environ["VOYAGE_API_KEY"]

    monkeypatch.setitem(sys.modules, "voyageai", types.SimpleNamespace(Client=EnvClient))
    for emb in (VoyageEmbedder("voyage-4"), VoyageEmbedder("voyage-4", api_key=secret)):
        emb.embed(["x"])  # creates and caches the client
        assert emb._client is not None
        fz = DocumentFeaturizer(embedding_model=emb, cache_dir=None)
        assert secret.encode() not in pickle.dumps(emb)
        assert secret.encode() not in _dump(fz, None)  # the stored model
        clone = copy.deepcopy(emb)  # copies in the process keep an explicit key (sklearn clone)
        assert clone.api_key == emb.api_key and copy.copy(emb).api_key == emb.api_key
    restored = pickle.loads(pickle.dumps(VoyageEmbedder("voyage-4", api_key=secret)))
    assert restored.api_key is None and restored.model == "voyage-4"
    restored.embed(["y"])  # the key is read from the environment again
