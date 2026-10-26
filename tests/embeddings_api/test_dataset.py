import pytest

from pymongopredict import MongoDataset


def test_dataset_loads_target_and_drops_missing(listings):
    listings.insert_one({"_id": "no-target", "beds": 1})
    ds = MongoDataset(listings, target="room_type")
    assert len(ds) == 120 and "no-target" not in ds.ids
    assert set(ds.y) == {"Entire home/apt", "Private room", "Shared room"}
    assert len(ds.documents) == 120
    assert ds.field_values("beds")[:3] == [d["beds"] for d in ds.documents[:3]]
    assert "n=120" in repr(ds)


def test_dataset_filter_limit_projection(listings):
    ds = MongoDataset(
        listings,
        target="room_type",
        filter={"address.market": "Porto"},
        limit=10,
        projection={"room_type": 1, "beds": 1},
    )
    assert len(ds) == 10
    assert set(ds.documents[0]) == {"_id", "room_type", "beds"}


def test_streaming_dataset_batches_and_projection(listings):
    ds = MongoDataset(listings, target="room_type", streaming=True, batch_size=50)
    assert ds._docs is None and len(ds) == 120
    batches = list(ds.iter_batches(projection={"beds": 1}))
    assert [len(b) for b in batches] == [50, 50, 20]
    assert set(batches[0][0]) == {"_id", "beds"}
    assert [d["_id"] for b in batches for d in b] == ds.ids
    sub = ds.subset([5, 1, 7])
    assert sub.ids == [ds.ids[5], ds.ids[1], ds.ids[7]] and list(sub.y) == [
        ds.y[5],
        ds.y[1],
        ds.y[7],
    ]
    assert sub.field_values("beds") == [ds.field_values("beds")[i] for i in (5, 1, 7)]


def test_from_documents_with_target_or_y():
    docs = [{"a": 1, "t": "x"}, {"a": 2}, {"a": 3, "t": "y"}]
    ds = MongoDataset.from_documents(docs, target="t")
    assert len(ds) == 2 and list(ds.y) == ["x", "y"]
    ds2 = MongoDataset.from_documents(docs, y=[0, 1, 0])
    assert len(ds2) == 3 and ds2.target is None and ds2.y.dtype == float
    with pytest.raises(ValueError):
        MongoDataset.from_documents(docs, y=[0, 1])
    with pytest.raises(ValueError):
        MongoDataset(None).load()


def test_streaming_prefetch_matches_sequential(listings):
    a = MongoDataset(listings, target="room_type", streaming=True, batch_size=16, prefetch=3)
    b = MongoDataset(listings, target="room_type", streaming=True, batch_size=16, prefetch=0)
    assert [d["_id"] for x in a.iter_batches() for d in x] == [
        d["_id"] for x in b.iter_batches() for d in x
    ]
    assert a.subset([3, 1]).prefetch == 3


def test_prefetch_propagates_errors():
    from pymongopredict.embeddings_api.dataset import _prefetched

    def boom():
        raise RuntimeError("db went away")

    gen = _prefetched(iter([lambda: [1], boom]), depth=2)
    assert next(gen) == [1]
    with pytest.raises(RuntimeError):
        next(gen)
