"""Demo 3: a ``$predict`` aggregation stage on sample_airbnb.listingsAndReviews.

Trains a room type model on every market except Porto (like creating a search index), then
scores Porto's listings with ``$predict`` pipelines (like querying with ``$vectorSearch``) and
writes the predictions to ``sample_airbnb.listings_predictions`` with ``$merge``.

    uv run python -m demos.predictions_api.predict_stage --uri mongodb://localhost:27017
"""

from __future__ import annotations

from demos._common import Timer, collection, embedding_kwargs, markdown_table, parser
from demos.embeddings_api.airbnb_room_type import COLUMNS, TEXT_FIELDS
from pymongopredict import PredictionModel
from pymongopredict import predictions_api as P


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    coll = collection(args.uri, "sample_airbnb", "listingsAndReviews")
    P.enable()  # coll.aggregate now understands $predict

    model = PredictionModel(
        {
            "target": "room_type",
            "columns": COLUMNS,
            "embedFields": TEXT_FIELDS,
            "embeddingModel": args.embedding_model,
            "embeddingKwargs": embedding_kwargs(args),
            "filter": {"address.market": {"$ne": "Porto"}},
            "featurizerOptions": {"cache_dir": args.cache_dir},
        },
        name="room_type",
    )
    with Timer() as t:
        coll.create_prediction_model(model)
    rec = coll.list_prediction_models("room_type")[0]
    print(
        f"## create_prediction_model: {t.seconds:.1f} s, status {rec['status']}, "
        f"{rec['numTrainingDocuments']} training documents, holdout metrics {rec['metrics']}\n"
    )

    porto = {"address.market": "Porto"}
    with Timer() as t:
        top = coll.aggregate(
            [
                {"$predict": {"model": "room_type", "filter": porto}},
                {"$match": {"$expr": {"$eq": [{"$meta": "prediction"}, "Private room"]}}},
                {"$sort": {"score": {"$meta": "predictScore"}}},
                {"$limit": 5},
                {
                    "$project": {
                        "_id": 0,
                        "name": 1,
                        "room_type": 1,
                        "predicted": {"$meta": "prediction"},
                        "score": {"$meta": "predictScore"},
                    }
                },
            ]
        ).to_list()
    print(f"## Most confident 'Private room' predictions in Porto ({t.seconds:.1f} s)\n")
    print(markdown_table(top) + "\n")

    with Timer() as t:
        confusion = coll.aggregate(
            [
                {"$predict": {"model": "room_type", "filter": porto}},
                {
                    "$group": {
                        "_id": {"actual": "$room_type", "predicted": {"$meta": "prediction"}},
                        "n": {"$sum": 1},
                    }
                },
                {"$sort": {"n": -1}},
            ]
        ).to_list()
    rows = [
        {"actual": c["_id"]["actual"], "predicted": c["_id"]["predicted"], "n": c["n"]}
        for c in confusion
    ]
    n = sum(r["n"] for r in rows)
    acc = sum(r["n"] for r in rows if r["actual"] == r["predicted"]) / n
    print(f"## Porto, actual vs predicted ({n} listings, accuracy {acc:.3f}; {t.seconds:.1f} s)\n")
    print(markdown_table(rows) + "\n")

    with Timer() as t:
        coll.aggregate(
            [
                {"$predict": {"model": "room_type", "filter": porto}},
                {
                    "$project": {
                        "predicted_room_type": {"$meta": "prediction"},
                        "predicted_room_type_score": {"$meta": "predictScore"},
                    }
                },
                {"$merge": {"into": "listings_predictions", "on": "_id", "whenMatched": "replace"}},
            ]
        )
    out = coll.database.listings_predictions
    print(
        f"## $merge: {out.count_documents({})} predictions in sample_airbnb.listings_predictions "
        f"({t.seconds:.1f} s); e.g. {out.find_one()}"
    )


if __name__ == "__main__":
    main()
