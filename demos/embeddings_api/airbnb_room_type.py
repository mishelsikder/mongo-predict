"""Demo 1 (standard prediction): classify the room type of Airbnb listings.

Data: ``sample_airbnb.listingsAndReviews`` (5,555 listings, 3 classes: Entire home/apt,
Private room, Shared room). Split: 80/20 stratified by the target.

Compares a ridge classifier (RidgeClassifierCV) on (a) tabular columns extracted from the
documents, (b) a voyage-4-nano embedding of the listing's JSON text fields, and (c) both.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from demos._common import Timer, collection, embedding_kwargs, markdown_table, parser, xgb_estimator
from pymongopredict import DocumentFeaturizer, MongoDataset

COLUMNS = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "price",
    "cleaning_fee",
    "guests_included",
    "minimum_nights",
    "number_of_reviews",
    "review_scores.review_scores_rating",
    "address.market",
    "property_type",
    "bed_type",
    "cancellation_policy",
]
TEXT_FIELDS = [
    "name",
    "summary",
    "space",
    "description",
    "neighborhood_overview",
    "notes",
    "house_rules",
    "amenities",
    "property_type",
    "bed_type",
    "address.market",
]


def main():
    args = parser(__doc__).parse_args()
    coll = collection(args.uri, "sample_airbnb", "listingsAndReviews")

    # 1. dataset + split -----------------------------------------------------------------
    ds = MongoDataset(
        coll,
        target="room_type",
        limit=args.limit,
        streaming=args.streaming,
        batch_size=args.batch_size,
    )
    ds_train, ds_test = train_test_split(ds, test_size=0.2, stratify=ds.y, random_state=0)
    print(f"{ds_train!r}\n{ds_test!r}\n")

    # 2. featurizer (the wrapper) + any sklearn estimator ---------------------------------
    common = dict(
        embedding_model=args.embedding_model,
        embedding_kwargs=embedding_kwargs(args),
        cache_dir=args.cache_dir,
    )
    lr = lambda: RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))  # noqa: E731
    configs = {
        "columns only (ridge classifier)": (dict(columns=COLUMNS, embed_fields=False), lr),
        "columns only (gradient boosting)": (
            dict(columns=COLUMNS, embed_fields=False),
            lambda: HistGradientBoostingClassifier(random_state=0),
        ),
        "embeddings only (ridge classifier)": (dict(embed_fields=TEXT_FIELDS), lr),
        "columns + embeddings (ridge classifier)": (
            dict(columns=COLUMNS, embed_fields=TEXT_FIELDS),
            lr,
        ),
    }
    if xgb_estimator("clf") is not None:  # optional: only when xgboost is installed
        configs["columns only (xgboost)"] = (
            dict(columns=COLUMNS, embed_fields=False),
            lambda: xgb_estimator("clf"),
        )
        configs["columns + embeddings (xgboost)"] = (
            dict(columns=COLUMNS, embed_fields=TEXT_FIELDS),
            lambda: xgb_estimator("clf"),
        )
    majority = Counter(ds_test.y).most_common(1)[0][0]
    rows = [
        {
            "features / model": "majority class baseline",
            "accuracy": float((ds_test.y == majority).mean()),
            "balanced_accuracy": 1 / len(set(ds_test.y)),
            "macro_f1": float("nan"),
            "n_features": 0,
            "seconds": 0.0,
        }
    ]
    for name, (feats, make_est) in configs.items():
        pipe = make_pipeline(DocumentFeaturizer(**common, **feats), make_est())
        with Timer() as t:
            pipe.fit(ds_train, ds_train.y)
            pred = pipe.predict(ds_test)
        rows.append(
            {
                "features / model": name,
                "accuracy": accuracy_score(ds_test.y, pred),
                "balanced_accuracy": balanced_accuracy_score(ds_test.y, pred),
                "macro_f1": f1_score(ds_test.y, pred, average="macro"),
                "n_features": len(pipe[0].get_feature_names_out()),
                "seconds": t.seconds,
            }
        )
        print(f"{name}: accuracy={rows[-1]['accuracy']:.3f}", flush=True)
    print(f"\ntrain={len(ds_train)} test={len(ds_test)} embedding_model={args.embedding_model}\n")
    print(markdown_table(rows))

    # 3. predictions on fresh documents straight from a cursor ------------------------------
    porto = list(coll.find({"address.market": "Porto"}).limit(3))
    print("\nexample predictions:", list(zip([d["name"] for d in porto], pipe.predict(porto))))


if __name__ == "__main__":
    main()
