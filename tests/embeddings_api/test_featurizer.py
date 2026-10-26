import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from pymongopredict import DocumentFeaturizer, MongoDataset


def test_featurizer_infers_kinds_and_shapes(listings, tmp_path):
    docs = list(listings.find())
    f = DocumentFeaturizer(
        columns=["beds", "price", "minimum_nights", "address.market", "amenities", "last_review"],
        embed_fields=["name", "summary"],
        embedding_model="hashing",
        embedding_kwargs={"n_features": 16},
        cache_dir=str(tmp_path),
        min_frequency=1,
    ).fit(docs)
    assert f.column_kinds_ == {
        "beds": "numeric",
        "price": "numeric",
        "minimum_nights": "numeric",
        "address.market": "categorical",
        "amenities": "size",
        "last_review": "datetime",
    }
    X = f.transform(docs[:5])
    assert X.shape == (5, f.n_tabular_features_ + 16) and X.dtype == np.float32
    assert len(f.get_feature_names_out()) == X.shape[1]
    # unseen category and missing fields do not break transform
    assert f.transform([{"beds": None, "address": {"market": "Mars"}}]).shape == (1, X.shape[1])


def test_featurizer_without_embeddings_and_without_columns(listings, tmp_path):
    docs = list(listings.find())
    tab = DocumentFeaturizer(columns=["beds"], embed_fields=False).fit(docs)
    assert tab.transform(docs).shape == (120, 1)
    emb = DocumentFeaturizer(
        embedding_model="hashing", embedding_kwargs={"n_features": 8}, cache_dir=None
    ).fit(docs)
    assert emb.transform(docs).shape == (120, 8)
    with pytest.raises(ValueError):
        DocumentFeaturizer(embed_fields=False).fit(docs)


def prep(tmp_path, **kw):
    return DocumentFeaturizer(
        columns=["beds", "price", "address.market"],
        embed_fields=["name", "summary"],
        embedding_model="hashing",
        cache_dir=str(tmp_path),
        **kw,
    )


def test_sklearn_workflow_with_datasets(listings, tmp_path):
    ds = MongoDataset(listings, target="room_type")
    train, test = train_test_split(ds, test_size=0.25, stratify=ds.y, random_state=0)
    p = prep(tmp_path)
    X_train = p.fit_transform(train)
    assert p.target_ == "room_type" and "room_type" in p.exclude_
    assert X_train.shape[0] == len(train) == 90
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, train.y)
    assert clf.score(p.transform(test), test.y) > 0.9  # beds determine room type here
    assert clf.predict_proba(p.transform(test)).shape == (30, 3)
    # cursors, generators and lists all work as X
    assert p.transform(listings.find().limit(7)).shape[0] == 7
    assert p.transform(d for d in test.documents).shape == p.transform(test).shape
    np.testing.assert_array_equal(p.transform(test.documents), p.transform(test))


def test_pipeline_composition(listings, tmp_path):
    ds = MongoDataset(listings, target="room_type", streaming=True, batch_size=17)
    train, test = train_test_split(ds, test_size=0.25, stratify=ds.y, random_state=0)
    pipe = make_pipeline(prep(tmp_path), LogisticRegression(max_iter=2000))
    pipe.fit(train, train.y)
    assert pipe.score(test, test.y) > 0.8
    assert len(pipe.predict(test)) == 30
    assert train._docs is None  # still streaming, nothing materialised


def test_streaming_matches_in_memory(listings, tmp_path):
    a = MongoDataset(listings, target="room_type")
    b = MongoDataset(listings, target="room_type", streaming=True, batch_size=17)
    tr_a, te_a = train_test_split(
        a, test_size=0.25, stratify=a.field_values("address.market"), random_state=0
    )
    tr_b, te_b = train_test_split(
        b, test_size=0.25, stratify=b.field_values("address.market"), random_state=0
    )
    pa, pb = prep(tmp_path), prep(tmp_path)
    np.testing.assert_array_equal(pa.fit_transform(tr_a), pb.fit_transform(tr_b))
    np.testing.assert_array_equal(pa.transform(te_a), pb.transform(te_b))
    np.testing.assert_array_equal(tr_a.y, tr_b.y)
    # streaming fetches only the used fields
    batch = next(te_b.iter_batches(pb._projection(te_b.target)))
    assert set(batch[0]) == {"_id", "beds", "price", "address", "name", "summary", "room_type"}
    assert pb._projection("room_type") == {
        "address.market": 1,
        "beds": 1,
        "name": 1,
        "price": 1,
        "room_type": 1,
        "summary": 1,
    }
    assert DocumentFeaturizer(embed_fields=None)._projection("t") is None


def test_regression_with_time_split_and_typed_targets(observations):
    ds = MongoDataset(
        observations,
        target="airTemperature.value",
        streaming=True,
        batch_size=100,
        filter={"airTemperature.value": {"$lt": 900}},
        sort=[("ts", 1)],
    )
    train, test = train_test_split(ds, test_size=0.2, shuffle=False)  # the newest 20% as test
    assert train.y.dtype == float
    assert max(train.field_values("ts")) < min(test.field_values("ts"))
    p = DocumentFeaturizer(columns=["st", "pressure.value"], embed_fields=False, min_frequency=1)
    reg = Ridge().fit(p.fit_transform(train), train.y)
    pred = reg.predict(p.transform(test))
    assert np.abs(pred - test.y).mean() < 3


def test_from_documents_dataset_roundtrip(listings):
    ds = MongoDataset.from_documents(listings.find(), target="price")
    assert ds.y.dtype == float  # Decimal128 prices become floats
    train, test = train_test_split(ds, test_size=0.2, random_state=0)
    p = DocumentFeaturizer(columns=["beds"], embed_fields=False)
    reg = Ridge().fit(p.fit_transform(train), train.y)
    assert reg.score(p.transform(test), test.y) > 0.5


def test_transform_batches_matches_single_call(listings):
    docs = list(listings.find())
    f = DocumentFeaturizer(
        columns=["beds", "price", "address.market", "last_review"],
        embed_fields=["name"],
        embedding_model="hashing",
        embedding_kwargs={"n_features": 8},
        cache_dir=None,
        min_frequency=1,
    )
    X_fit = f.fit_transform(docs)
    one = f.transform(docs)
    batched = f.transform_batches([docs[i : i + 7] for i in range(0, len(docs), 7)])
    np.testing.assert_array_equal(one, batched)
    np.testing.assert_array_equal(one, X_fit)
    assert f.transform_batches([]).shape == (0, one.shape[1])


class WordTokenizer:
    """Whitespace tokenizer standing in for a Hugging Face one (encode / decode / __call__)."""

    model_max_length = 60

    def __call__(self, texts, add_special_tokens=False):
        one = isinstance(texts, str)
        ids = [t.split(" ") for t in ([texts] if one else texts)]
        return {"input_ids": ids[0] if one else ids}

    def decode(self, ids):
        return " ".join(ids)


def test_mask_target_mode_keeps_target_first_and_prefix(listings):
    import json

    ds = MongoDataset(listings, target="room_type")
    f = DocumentFeaturizer(
        embed_fields=["name", "beds"], embedding_model="hashing", cache_dir=None, mask_target=True
    )
    f.fit(ds)
    text = f._texts(ds.documents[:1])[0]
    assert text.startswith('Predict the value of <MASK>: {"room_type":"<MASK>","name":')
    body = json.loads(text[len("Predict the value of <MASK>: ") :])
    assert body["room_type"] == "<MASK>" and set(body) == {"room_type", "name", "beds"}
    # whole-document mode: real value gone, mask present, even for a nested target
    g = DocumentFeaturizer(
        embedding_model="hashing", cache_dir=None, mask_target=True, prefix="Q: ", mask_token="[?]"
    )
    doc = {"a": 1, "scores": [{"type": "exam", "score": 5.0}], "_id": 1}
    g.fit(MongoDataset.from_documents([doc, doc], target="scores.0.score"))
    t = g._texts([doc])[0]
    assert t == 'Q: {"scores.0.score":"[?]","a":1,"scores":[{"type":"exam"}]}'
    # without mask mode the target is simply removed
    h = DocumentFeaturizer(embedding_model="hashing", cache_dir=None).fit(ds)
    assert "room_type" not in h._texts(ds.documents[:1])[0]


def test_token_budget_with_custom_tokenizer():
    docs = [
        {
            "_id": i,
            "y": i,
            "short": "a b",
            "long": " ".join(str(k) for k in range(100)),
            "tail": "never seen",
        }
        for i in range(3)
    ]
    ds = MongoDataset.from_documents(docs, target="y")
    f = DocumentFeaturizer(
        embedding_model="hashing",
        cache_dir=None,
        mask_target=True,
        max_tokens="model",
        tokenizer=WordTokenizer(),
    )
    f.fit(ds)
    assert f.token_limit() == 60
    text = f._texts(docs[:1])[0]
    tokens = text.split(" ")
    assert len(tokens) <= 60 - f._RESERVED_TOKENS  # budget honoured
    assert text.startswith('Predict the value of <MASK>: {"y":"<MASK>","short":"a b","long":"0')
    assert "tail" not in text  # dropped after the budget was spent
    g = DocumentFeaturizer(
        embedding_model="hashing", cache_dir=None, max_tokens=10**9, tokenizer=WordTokenizer()
    ).fit(ds)
    assert "never seen" in g._texts(docs[:1])[0]
    # the embedder's max_seq_length follows the token budget for local models only
    h = DocumentFeaturizer(
        embedding_model="voyage-4-nano", max_tokens=123, tokenizer=WordTokenizer(), cache_dir=None
    )
    emb = h._embedder()  # not fitted; just inspect the embedder construction
    assert emb.max_seq_length == 123
    assert "max_seq_length" not in (
        DocumentFeaturizer(
            embedding_model="voyage-4-lite", max_tokens=5, tokenizer=WordTokenizer()
        ).embedding_kwargs
        or {}
    )
