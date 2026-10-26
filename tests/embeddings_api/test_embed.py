import numpy as np

from pymongopredict.embeddings_api.embed import (
    HashingEmbedder,
    _CachedEmbedder,
    _make_embedder,
    _MatryoshkaTruncate,
)


class Counting(HashingEmbedder):
    calls = 0

    def embed(self, texts):
        Counting.calls += len(texts)
        return super().embed(texts)


def test_hashing_embedder_shape_and_determinism():
    e = HashingEmbedder(n_features=64)
    a = e.embed(['{"a": 1}', '{"a": 2}'])
    assert a.shape == (2, 64) and a.dtype == np.float32
    assert np.allclose(a, e.embed(['{"a": 1}', '{"a": 2}']))


def test_cached_embedder_only_embeds_new_texts(tmp_path):
    inner = Counting(n_features=32)
    e = _CachedEmbedder(inner, str(tmp_path))
    first = e.embed(["x", "y"])
    assert Counting.calls == 2
    again = e.embed(["y", "z", "x"])
    assert Counting.calls == 3
    assert np.allclose(again[2], first[0]) and np.allclose(again[0], first[1])
    assert e.embed([]).shape[0] == 0


def test_make_embedder_dispatch(tmp_path):
    assert _make_embedder(None) is None and _make_embedder(False) is None
    assert isinstance(_make_embedder("hashing"), HashingEmbedder)
    assert isinstance(_make_embedder("hashing", cache_dir=str(tmp_path)), _CachedEmbedder)
    st = _make_embedder("voyage-4-nano")
    assert st.model_name == "voyageai/voyage-4-nano"  # local, no API key needed
    assert _make_embedder("voyage-4-lite").name.startswith("voyage-api")


def test_matryoshka_truncation_shares_the_cache(tmp_path):
    Counting.calls = 0
    inner = Counting(n_features=64)
    e512 = _make_embedder(inner, cache_dir=str(tmp_path), dimension=16)
    e_full = _make_embedder(inner, cache_dir=str(tmp_path))
    assert isinstance(e512, _MatryoshkaTruncate) and e512.name.endswith("@16")
    a = e512.embed(["x", "y"])
    norms = np.linalg.norm(a, axis=1)
    assert a.shape == (2, 16) and np.allclose(norms[norms > 0], 1)
    assert Counting.calls == 2
    e_full.embed(["x", "y"])  # served from the cache written by the truncated embedder
    assert Counting.calls == 2
    assert _make_embedder("voyage-4-nano@512").dimension == 512
    assert _make_embedder("voyage-4-nano@512").embedder.model_name == "voyageai/voyage-4-nano"


def test_sentence_transformer_empty_input_uses_model_dimension(monkeypatch):
    from pymongopredict.embeddings_api.embed import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder("some/model")

    class Fake:
        def get_sentence_embedding_dimension(self):
            return 7

    e._model = Fake()
    assert e.embed([]).shape == (0, 7)


def test_token_groups_cap_batch_tokens():
    from pymongopredict.embeddings_api.embed import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder("some/model", batch_size=4, max_batch_tokens=100)

    class Fake:
        max_seq_length = 60

    e._model = Fake()
    texts = ["x" * 300, "y" * 3, "z" * 3, "w" * 150, "v" * 3, "u" * 3, "t" * 3]
    groups = e._token_groups(texts)
    assert sorted(i for g in groups for i in g) == list(range(7))
    assert groups[0] == [0]  # 100 estimated tokens: alone
    assert all(len(g) <= 4 for g in groups)
