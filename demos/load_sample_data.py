"""Load MongoDB sample datasets into a MongoDB instance.

The official sample datasets (https://www.mongodb.com/docs/manual/sample-data/) ship with
MongoDB Atlas. For a local instance this script downloads the same collections, as MongoDB
extended JSON, from the public mirror at github.com/neelabalan/mongodb-sample-dataset and
inserts them with pymongo.

Example:
    python -m demos.load_sample_data --uri mongodb://localhost:27017 sample_airbnb \
        sample_weatherdata
"""

from __future__ import annotations

import argparse
import pathlib

import requests
from bson import json_util
from pymongo import MongoClient

MIRROR = "https://raw.githubusercontent.com/neelabalan/mongodb-sample-dataset/main"
# database -> list of collections available in the mirror
DATASETS = {
    "sample_airbnb": ["listingsAndReviews"],
    "sample_weatherdata": ["data"],
    "sample_mflix": ["movies", "comments", "theaters", "users", "sessions"],
    "sample_analytics": ["accounts", "customers", "transactions"],
    "sample_geospatial": ["shipwrecks"],
    "sample_supplies": ["sales"],
    "sample_training": [
        "companies",
        "grades",
        "inspections",
        "posts",
        "routes",
        "stories",
        "trips",
        "tweets",
        "zips",
    ],
}


def download(database: str, collection: str, cache_dir: pathlib.Path) -> pathlib.Path:
    path = cache_dir / database / f"{collection}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{MIRROR}/{database}/{collection}.json"
        print(f"downloading {url}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    return path


def load(uri: str, database: str, cache_dir: pathlib.Path, drop: bool = True) -> None:
    client = MongoClient(uri)
    db = client[database]
    for collection in DATASETS[database]:
        path = download(database, collection, cache_dir)
        if drop:
            db[collection].drop()
        batch = []
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:  # one extended-JSON document per line
                line = line.strip()
                if not line:
                    continue
                batch.append(json_util.loads(line))
                if len(batch) == 1000:
                    db[collection].insert_many(batch)
                    n += len(batch)
                    batch = []
        if batch:
            db[collection].insert_many(batch)
            n += len(batch)
        print(f"{database}.{collection}: inserted {n} documents")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "databases",
        nargs="*",
        default=["sample_airbnb", "sample_weatherdata"],
        choices=sorted(DATASETS),
    )
    p.add_argument("--uri", default="mongodb://localhost:27017")
    p.add_argument("--cache-dir", default=pathlib.Path(__file__).parent / "data", type=pathlib.Path)
    p.add_argument(
        "--keep-existing", action="store_true", help="do not drop collections before inserting"
    )
    args = p.parse_args()
    for database in args.databases:
        load(args.uri, database, args.cache_dir, drop=not args.keep_existing)


if __name__ == "__main__":
    main()
