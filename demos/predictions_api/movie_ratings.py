"""Predict a movie's IMDb rating with ``$predict`` on sample_mflix.movies (the README's example).

A prediction model is trained on the movies released before 2010 from the embedding of each movie
(title, year, runtime, age rating, genres, cast, directors and plot, all in ``embedFields``).
``$predict`` then rates the movies released from 2010 on, and the predictions are compared with
their actual IMDb ratings and with always guessing the average rating. The other ratings stored in
the documents (Rotten Tomatoes, Metacritic, awards, IMDb votes) are not used.

    uv run python -m demos.predictions_api.movie_ratings --uri mongodb://localhost:27017
"""

from __future__ import annotations

import json
import statistics

from demos._common import Timer, collection, embedding_kwargs, markdown_table, parser
from pymongopredict import PredictionModel
from pymongopredict import predictions_api as P

TARGET = "imdb.rating"
EMBED_FIELDS = ["title", "year", "runtime", "rated", "genres", "cast", "directors", "plot"]
RATED = {TARGET: {"$type": "number"}}
OLDER = {**RATED, "year": {"$lt": 2010}}  # training movies
NEWER = {**RATED, "year": {"$gte": 2010}}  # movies to rate


def definition(args) -> dict:
    return {
        "target": TARGET,
        "embedFields": EMBED_FIELDS,
        "embeddingModel": args.embedding_model,
        "embeddingKwargs": embedding_kwargs(args),
        "filter": OLDER,
        "featurizerOptions": {"cache_dir": args.cache_dir},
    }


def main():
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--out", default=None, help="write the results as JSON here")
    args = p.parse_args()
    coll = collection(args.uri, "sample_mflix", "movies")
    P.enable()

    results = {"n_train": coll.count_documents(OLDER), "n_rated": coll.count_documents(NEWER)}
    mean_older = next(
        coll.aggregate([{"$match": OLDER}, {"$group": {"_id": None, "m": {"$avg": f"${TARGET}"}}}])
    )["m"]
    actual = {d["_id"]: d["imdb"]["rating"] for d in coll.find(NEWER, {TARGET: 1})}
    with Timer() as t:
        coll.create_prediction_model(PredictionModel(definition(args), name="imdb_rating"))
    with Timer() as q:
        rows = coll.aggregate([
            {"$predict": {"model": "imdb_rating", "filter": NEWER}},
            {"$project": {"predicted": {"$meta": "prediction"}}},
        ]).to_list()
    predicted = {r["_id"]: r["predicted"] for r in rows}
    results["seconds"] = {"train": t.seconds, "predict": q.seconds}
    results["task"] = coll.list_prediction_models("imdb_rating")[0]["task"]

    errors = {
        "Always guess the average": [abs(r - mean_older) for r in actual.values()],
        "$predict (embedding of the movie)": [abs(predicted[i] - actual[i]) for i in actual],
    }
    table = [
        {
            "approach": label,
            "average error": statistics.mean(e),
            "within 1 point": sum(x <= 1 for x in e) / len(e),
        }
        for label, e in errors.items()
    ]
    results["table"] = table
    print(
        f"## {results['n_rated']} movies from 2010 on, "
        f"model trained on {results['n_train']} older ones ({results['task']})\n"
    )
    print(markdown_table(table) + "\n")

    # a few well-known recent movies (most IMDb votes), predicted vs actual
    known = coll.find(NEWER, {"title": 1, "year": 1, "imdb": 1}).sort("imdb.votes", -1).limit(20)
    examples, seen = [], set()
    for d in known:  # (the sample data holds a few movies twice)
        if d["title"] in seen or len(examples) == 8:
            continue
        seen.add(d["title"])
        examples.append({
            "title": d["title"],
            "year": d["year"],
            "actual": d["imdb"]["rating"],
            "predicted": round(predicted[d["_id"]], 1),
        })
    results["examples"] = examples
    print(markdown_table(examples, floatfmt="{:.1f}"))
    # the movies shown in the README's Fig. 1
    figure = ["Inception", "The Hunger Games", "Django Unchained"]
    results["figure"] = {
        d["title"]: round(predicted[d["_id"]], 1)
        for d in coll.find({**NEWER, "title": {"$in": figure}}, {"title": 1})
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main()
