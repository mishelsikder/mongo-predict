"""Regression tests for edge cases of DocumentFeaturizer and the embedders."""

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest

from pymongopredict import DocumentFeaturizer, MongoDataset
from pymongopredict.embeddings_api.embed import (
    BaseEmbedder,
    HashingEmbedder,
    VoyageEmbedder,
    _CachedEmbedder,
    _make_embedder,
    _MatryoshkaTruncate,
)


def columns_only(**kw):
    return DocumentFeaturizer(embed_fields=False, cache_dir=None, scale_numeric=False, **kw)


def test_datetimes_are_epoch_seconds_in_every_batch():
    f = columns_only(columns=["t"]).fit(
        [{"t": pd.Timestamp("2020-01-01 00:00:00.000000001")}, {"t": dt.datetime(2021, 1, 1)}]
    )
    x = f.transform([{"t": dt.datetime(2021, 1, 1)}, {"t": np.datetime64("2021-01-01", "ns")}])
    assert np.allclose(x[:, 0], dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc).timestamp())


def test_infinite_and_huge_numbers_stay_finite():
    docs = [{"x": 1.0}, {"x": float("inf")}, {"x": 3}, {"x": -float("inf")}]
    f = DocumentFeaturizer(columns=["x"], embed_fields=False, cache_dir=None)
    assert np.isfinite(f.fit_transform(docs)).all()
    assert np.isfinite(columns_only(columns=["x"]).fit(docs).transform([{"x": 1e300}])).all()


def test_size_and_bool_columns_with_unexpected_values():
    f = columns_only(columns=["l", "b"]).fit([{"l": [1, 2], "b": True}, {"l": [1], "b": False}])
    assert f.column_kinds_ == {"l": "size", "b": "bool"}
    x = f.transform([{"l": 5, "b": "false"}, {"l": [1, 2, 3], "b": "TRUE"}, {"b": "maybe"}])
    assert np.isfinite(x).all() and x[0, 1] == 0 and x[1, 1] == 1 and x[1, 0] == 3


def test_empty_batches_are_ignored(tmp_path):
    f = DocumentFeaturizer(
        embedding_model="hashing", embedding_kwargs={"n_features": 8}, cache_dir=str(tmp_path)
    )
    X = f.fit_transform_batches([[], [{"a": "x"}, {"a": "y"}], []])
    assert X.shape == (2, 8) == (2, len(f.get_feature_names_out()))
    assert f.transform_batches([[{"a": "x"}], []]).shape == (1, 8)


def test_nested_target_never_reaches_the_embedded_text():
    docs = [{"address": {"city": "P", "price": 123456}, "r": [{"o": 1, "k": 2}]}] * 3
    ds = MongoDataset.from_documents(docs, target="address.price")
    f = DocumentFeaturizer(
        embed_fields=["address", "r"], exclude=["r.o"], embedding_model="hashing", cache_dir=None
    ).fit(ds)
    (text,) = f._texts(docs[:1])
    assert "123456" not in text and '"o"' not in text and '"city"' in text


def test_refit_resets_target_and_tokenizer():
    class Tok:
        def __init__(self, k):
            self.k = k

        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [list(range(len(t) // self.k)) for t in texts]}

        def decode(self, ids):
            return "x" * (len(ids) * self.k)

    docs = [{"price": 3, "a": "word " * 20}] * 3
    f = DocumentFeaturizer(
        embedding_model="hashing", cache_dir=None, max_tokens=40, tokenizer=Tok(1)
    )
    f.fit(MongoDataset.from_documents(docs, target="price"))
    assert f.exclude_ == ["price"]
    f.set_params(tokenizer=Tok(10))
    f.fit_batches([docs])
    assert f.exclude_ == [] and f.tokenizer_.k == 10 and '"price"' in f._texts(docs[:1])[0]


def test_budgets_and_embed_fields_true():
    f = DocumentFeaturizer(
        embedding_model="hashing", cache_dir=None, prefix="p" * 40, max_json_chars=30
    )
    f.fit([{"a": "x" * 50}, {"a": "y"}])
    assert len(f._texts([{"a": "x" * 50}])[0]) <= 42
    g = DocumentFeaturizer(embed_fields=True, embedding_model="hashing", cache_dir=None).fit(
        [{"a": 1}]
    )
    assert g._texts([{"a": 1, "b": 2}]) == ['{"a":1,"b":2}'] and g._projection(None) is None


def test_cache_keys_separate_embedders(tmp_path):
    a = _make_embedder("hashing", str(tmp_path))
    b = _make_embedder("hashing", str(tmp_path), analyzer="char_wb", ngram_range=(3, 5))
    assert a.name != b.name and not np.allclose(a.embed(["hello world"]), b.embed(["hello world"]))

    class Const(BaseEmbedder):
        def __init__(self, d):
            self.d = d

        def embed(self, texts):
            return np.full((len(texts), self.d), float(self.d), np.float32)

    with pytest.warns(UserWarning, match="not cached"):
        e4 = _make_embedder(Const(4), str(tmp_path))
    assert not isinstance(e4, _CachedEmbedder) and e4.embed(["x"]).shape == (1, 4)
    with pytest.raises(ValueError, match="cannot truncate"):
        _MatryoshkaTruncate(HashingEmbedder(n_features=8), 16).embed(["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _make_embedder("hashing@4", None).embed(["x"]).shape == (1, 4)


def test_voyage_embedder_on_empty_input():
    v = VoyageEmbedder("voyage-4-lite", output_dimension=256)
    v._client = object()  # never called
    assert v.embed([]).shape == (0, 256)


# ---------------------------------------------------------------- round 2
def test_string_parameters_and_empty_fits_raise():
    for kw in ({"exclude": "price"}, {"embed_fields": "name"}, {"columns": "price"}):
        with pytest.raises(TypeError, match="list of field paths"):
            DocumentFeaturizer(embedding_model="hashing", cache_dir=None, **kw).fit([{"price": 1}])
    with pytest.raises(ValueError, match="zero documents"):
        DocumentFeaturizer(embedding_model="hashing", cache_dir=None).fit([])


def test_mask_key_on_plain_documents():
    f = DocumentFeaturizer(
        embedding_model="hashing", cache_dir=None, mask_target=True, mask_key="price"
    )
    f.fit([{"a": 1, "price": 5}])
    assert f._texts([{"a": 1, "price": 5}]) == [
        'Predict the value of <MASK>: {"price":"<MASK>","a":1}'
    ]
    g = DocumentFeaturizer(embedding_model="hashing", cache_dir=None, mask_target=True).fit(
        [{"a": 1}]
    )
    assert g._texts([{"a": 1}]) == ['{"a":1}']  # nothing to mask: no prefix


def test_integral_floats_and_ints_are_one_category():
    docs = [{"c": v} for v in (1, "1", 1.0, "a", "a")]
    f = columns_only(columns=["c"], min_frequency=1).fit(docs)
    assert list(f.get_feature_names_out()) == ["cat__c_1", "cat__c_a"]


def test_embedder_keys_for_truncation_and_input_type():
    from pymongopredict.embeddings_api.embed import SentenceTransformerEmbedder

    st = SentenceTransformerEmbedder
    # the plain name is the untruncated embedding; every truncation length is in the key
    assert st().name == "voyageai/voyage-4-nano|len=1024"
    assert st(max_seq_length=None).name == "voyageai/voyage-4-nano"
    assert st(max_seq_length=128).name != st(max_seq_length=4096).name
    assert st(max_seq_length=4096, max_seq_length_in_key=False).name == "voyageai/voyage-4-nano"
    budget = DocumentFeaturizer(max_tokens=4096, cache_dir=None)._embedder().name
    assert budget == "voyageai/voyage-4-nano" != DocumentFeaturizer(cache_dir=None)._embedder().name
    # a budget measured with another tokenizer, or a custom prompt, may still be truncated
    custom = DocumentFeaturizer(max_tokens=4096, tokenizer=object(), cache_dir=None)
    prompt = DocumentFeaturizer(max_tokens=4096, cache_dir=None, embedding_kwargs={"prompt": "Q: "})
    assert custom._embedder().name.endswith("|len=4096")
    assert prompt._embedder().name.endswith("|len=4096")
    assert VoyageEmbedder(input_type="query").name != VoyageEmbedder().name
    assert HashingEmbedder(n_features=8).embed([]).shape == (0, 8)


def test_scaled_huge_numbers_stay_finite():
    f = DocumentFeaturizer(columns=["x"], embed_fields=False, cache_dir=None)
    f.fit([{"x": i / 100} for i in range(5)])
    assert np.isfinite(f.transform([{"x": 1e38}])).all()


def test_json_keys_with_colons_big_ints_and_token_budgets():
    from pymongopredict.embeddings_api.documents import _document_to_json

    out = _document_to_json({"a": 1, "http://example.org/a-long-key": "v" * 100}, max_chars=25)
    assert out == '{"a":1}' or '"http://example.org/a-long-key":' in out
    assert _document_to_json({"n": 2**64}) == '{"n":18446744073709551616}'

    class FourChars:  # 4 characters per token, decodes back to the text
        def __call__(self, texts, add_special_tokens=False):
            chunks = lambda t: [t[i : i + 4] for i in range(0, len(t), 4)]  # noqa: E731
            return {
                "input_ids": chunks(texts) if isinstance(texts, str) else [chunks(t) for t in texts]
            }

        def decode(self, ids):
            return "".join(ids)

    f = DocumentFeaturizer(
        embedding_model="hashing", cache_dir=None, max_tokens=60, tokenizer=FourChars()
    )
    f.fit([{"a_rather_long_field_name_of_39_chars_xx": "v" * 400}])
    assert (
        '"a_rather_long_field_name_of_39_chars_xx":'
        in f._texts([{"a_rather_long_field_name_of_39_chars_xx": "v" * 400}])[0]
    )
