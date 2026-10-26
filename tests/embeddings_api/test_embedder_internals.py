"""Embedder internals with a fake sentence-transformers model (no download)."""

import pickle
import shutil

import numpy as np
import pytest

from pymongopredict import DocumentFeaturizer
from pymongopredict.embeddings_api.embed import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    _CachedEmbedder,
    _make_embedder,
    _MatryoshkaTruncate,
)


class FakeST:
    """Stands in for a SentenceTransformer: vectors from the text length, NaN until float()."""

    max_seq_length = 512

    def __init__(self, nan_until_float=False):
        self.nan = nan_until_float
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 4

    def float(self):
        self.nan = False

    def encode(self, texts, batch_size=32, normalize_embeddings=True, **kw):
        self.calls.append((batch_size, max(len(t) for t in texts)))
        E = np.array([[len(t), 1.0, 2.0, 3.0] for t in texts], np.float32)
        if self.nan:
            E[:] = np.nan
        if normalize_embeddings:
            E /= np.linalg.norm(E, axis=1, keepdims=True)
        return E

    encode_document = encode


def fake(**kw):
    e = SentenceTransformerEmbedder("org/x", **kw)
    e._model = FakeST()
    return e


def test_part_mode_does_not_stick_to_the_embedder(tmp_path, monkeypatch):
    e = fake(batch_size=2)
    monkeypatch.setenv("MONGOPREDICT_EMBED_PART", "0/2")
    c = _CachedEmbedder(e, tmp_path)
    c.embed(["a" * i for i in range(1, 9)])
    assert e.part is None
    monkeypatch.delenv("MONGOPREDICT_EMBED_PART")
    assert (np.abs(e.embed(["a" * i for i in range(1, 9)])).sum(1) > 0).all()
    assert pickle.loads(pickle.dumps(c)).part is None


@pytest.mark.parametrize("bad", ["2/2", "1/0", "x", "1/2/3"])
def test_part_values_are_validated(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("MONGOPREDICT_EMBED_PART", bad)
    with pytest.raises(ValueError, match="MONGOPREDICT_EMBED_PART"):
        _CachedEmbedder(HashingEmbedder(4), tmp_path)


def test_nan_fallback_keeps_token_bounded_batches_and_is_never_cached(tmp_path):
    e = SentenceTransformerEmbedder("org/x", max_batch_tokens=1000, batch_size=64)
    e._model = FakeST(nan_until_float=True)
    texts = ["x" * 1500] * 10 + ["y"] * 40
    with pytest.warns(RuntimeWarning, match="non-finite"):
        E = e.embed(texts)
    assert np.isfinite(E).all()
    assert all(bs == 1 or bs * min(n // 3, 512) <= 1000 for bs, n in e._model.calls)
    bad = SentenceTransformerEmbedder("org/x")
    bad._model = FakeST(nan_until_float=True)
    bad._model.float = lambda: None  # stays broken
    c = _CachedEmbedder(bad, tmp_path)
    with pytest.warns(RuntimeWarning):
        c.embed(["abc"])
    import sqlite3

    assert sqlite3.connect(c.path).execute("select count(*) from emb").fetchone()[0] == 0


def test_extra_cache_paths_with_special_characters(tmp_path, monkeypatch):
    src = _CachedEmbedder(HashingEmbedder(8), tmp_path / "src")
    want = src.embed(["hello", "world"])
    weird = tmp_path / "a#b?c%d"
    weird.mkdir()
    import sqlite3

    sqlite3.connect(src.path).execute("PRAGMA wal_checkpoint(TRUNCATE)")  # (parts use DELETE mode)
    shutil.copy(src.path, weird / "embeddings.sqlite")
    monkeypatch.setenv("MONGOPREDICT_EXTRA_CACHES", str(weird / "embeddings.sqlite"))

    class Never(HashingEmbedder):
        def embed(self, texts):
            assert not texts, "should come from the extra cache"
            return super().embed(texts)

    assert np.allclose(_CachedEmbedder(Never(8), tmp_path / "dst").embed(["hello", "world"]), want)


def test_surrogates_empty_input_and_dimensions(tmp_path):
    c = _CachedEmbedder(HashingEmbedder(8), tmp_path)
    assert c.embed(["a\ud800b"]).shape == (1, 8) and c.embed([]).shape == (0, 8)
    assert _MatryoshkaTruncate(c, 4).embed([]).shape == (0, 4)
    raw = _make_embedder(fake(normalize=False), None, dimension=2, normalize=False)
    assert not np.isclose(np.linalg.norm(raw.embed(["abcdef"])), 1)
    for bad in (True, 1, 2.5):
        with pytest.raises(ValueError, match="unknown"):
            _make_embedder(bad)
    with pytest.raises(ValueError, match="dimension"):
        _make_embedder("hashing@0")


def test_fitted_featurizer_pickles_without_the_model(tmp_path):
    f = DocumentFeaturizer(embedding_model=fake(), cache_dir=None).fit([{"a": "x"}, {"a": "yy"}])
    assert f.embedder_._model is not None
    g = pickle.loads(pickle.dumps(f))
    assert g.embedder_._model is None


def test_dimension_suffix_is_stripped_for_tokenizer_and_positions(monkeypatch):
    import sys
    import types

    transformers = types.ModuleType(
        "transformers"
    )  # (a fake: importing the real one takes seconds)
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=None)
    transformers.AutoConfig = types.SimpleNamespace(from_pretrained=None)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    seen = []

    class Tok:
        model_max_length = 131072

    class Config:
        max_position_embeddings = 32768

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda n, **k: seen.append(n) or Tok()
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda n, **k: seen.append(n) or Config()
    )
    f = DocumentFeaturizer(embedding_model="voyage-4-nano@256", max_tokens="model", cache_dir=None)
    assert f.token_limit() == 32768 and all("@" not in n for n in seen)


def test_old_pickles_with_a_part_left_on_the_embedder(tmp_path):
    e = fake(batch_size=2)
    e.part = (1, 2)  # what an older version left on the embedder
    c = pickle.loads(pickle.dumps(_CachedEmbedder(e, tmp_path)))
    c.embedder._model = FakeST()  # (models are not pickled)
    E = c.embed(["a" * i for i in range(1, 12)])
    assert (np.abs(E).sum(1) > 0).all()
    assert "part" not in e.__getstate__()


def test_cache_init_retries_while_another_process_switches_the_journal(tmp_path, monkeypatch):
    # (several processes opening a new cache at once: switching to WAL fails with "database is
    # locked" without waiting; reproduced with processes, which a fast test cannot afford)
    import sqlite3

    real, calls = sqlite3.connect, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) <= 2:
            raise sqlite3.OperationalError("database is locked")
        return real(*a, **k)

    monkeypatch.setattr(sqlite3, "connect", flaky)
    c = _CachedEmbedder(HashingEmbedder(4), tmp_path)
    assert c.embed(["x"]).shape == (1, 4) and len(calls) > 2


def test_unusable_cache_files_name_their_path(tmp_path):
    import sqlite3

    (tmp_path / "embeddings.sqlite").write_bytes(b"not a database" * 100)
    with pytest.raises(sqlite3.DatabaseError, match="embeddings.sqlite.*cache_dir=None"):
        _CachedEmbedder(HashingEmbedder(4), tmp_path)
