import datetime as dt
import os

# tests never download models: a test that would load one fails fast instead of reaching the network
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# one thread per test worker: XGBoost (the default estimator) and BLAS otherwise each use every core
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import random

import mongomock
import pytest
from bson import Decimal128

from pymongopredict.predictions_api import models as _models

# the tests check behaviour, not accuracy: a small default XGBoost keeps the suite fast when many
# workers share a machine
_models.XGB_PARAMS.update(n_estimators=20, max_depth=3, n_jobs=1)


@pytest.fixture
def listings():
    """Small synthetic collection shaped like sample_airbnb.listingsAndReviews."""
    rng = random.Random(0)
    coll = mongomock.MongoClient().db.listings
    docs = []
    for i in range(120):
        room = rng.choice(["Entire home/apt", "Entire home/apt", "Private room", "Shared room"])
        beds = {"Entire home/apt": 3, "Private room": 1, "Shared room": 1}[room] + rng.randint(0, 1)
        docs.append(
            {
                "_id": str(i),
                "name": f"{room} number {i}",
                "summary": f"A lovely {room.lower()} with {beds} beds in the city",
                "amenities": ["Wifi", "Kitchen"][: rng.randint(1, 2)],
                "room_type": room,
                "beds": beds,
                "price": Decimal128(str(50 * beds + rng.randint(0, 20))),
                "minimum_nights": str(rng.randint(1, 5)),
                "address": {"market": rng.choice(["Porto", "Sydney", "New York"])},
                "host": {"host_id": str(i % 10)},
                "last_review": dt.datetime(2019, 1, 1) + dt.timedelta(days=i),
            }
        )
    coll.insert_many(docs)
    return coll


@pytest.fixture
def observations():
    """Synthetic hourly observations shaped like sample_weatherdata.data."""
    coll = mongomock.MongoClient().db.data
    t0 = dt.datetime(1984, 3, 5)
    docs = []
    for h in range(200):
        for station in range(3):
            docs.append(
                {
                    "st": f"s{station}",
                    "ts": t0 + dt.timedelta(hours=h),
                    "airTemperature": {"value": 10 + 5 * ((h % 24) / 24) + station, "quality": "1"},
                    "pressure": {"value": 1010 + station},
                }
            )
    docs.append({"st": "bad", "ts": t0, "airTemperature": {"value": 999.9, "quality": "9"}})
    coll.insert_many(docs)
    return coll


def pytest_collection_modifyitems(config, items):
    """On a MongoDB server older than 5.0 (MONGOPREDICT_TEST_URI), the server tests are skipped,
    except the one that checks the predictions API refuses such servers."""
    uri = os.environ.get("MONGOPREDICT_TEST_URI")
    if not uri:
        return
    from pymongo import MongoClient

    version = MongoClient(uri, serverSelectionTimeoutMS=5000).server_info()["versionArray"]
    if tuple(version[:2]) >= (5, 0):
        return
    skip = pytest.mark.skip(reason="the predictions API needs MongoDB 5.0 or later")
    for item in items:
        server_test = any(
            "MONGOPREDICT_TEST_URI" in str(m.kwargs.get("reason", ""))
            for m in item.iter_markers("skipif")
        )
        if server_test and item.name != "test_server_version_check_on_a_real_server":
            item.add_marker(skip)
