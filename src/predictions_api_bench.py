"""Benchmark of the ``$predict`` stage (report 01): overhead, agreement and a voyage-4-nano example.

Part 1, for every task of ``src/tasks.py`` (one per collection of the MongoDB sample datasets) with
the hashing embedder and no embedding cache:

* ``create_prediction_model`` on up to ``--n`` documents (whole document embedded, the task's
  columns, holdout metrics on 20%);
* the same model on up to ``--n`` documents four ways: a ``$predict`` pipeline that projects the
  prediction (with the per-step ``stats``), the client-side baseline it replaces (``find`` +
  ``transform`` + ``predict``, timed the same way), a ``$group`` on the prediction, and a
  ``$merge`` of the predictions into a scratch database; plus the fraction of documents whose
  ``$predict`` output equals the baseline's.

Part 2: the Airbnb room type model trained on every market except Porto with voyage-4-nano, then
queried on Porto (``demos/predictions_api/predict_stage.py``). Needs a GPU, or a cache that already holds
the embeddings (``--cache-dir``).

    uv run python -m src.predictions_api_bench --uri mongodb://localhost:27017 --cache-dir .mongopredict_cache
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from pymongo import MongoClient

from pymongopredict import predictions_api
from pymongopredict.predictions_api import PredictionModel
from pymongopredict.predictions_api.models import _load_prediction_model
from src.tasks import TASKS

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRATCH_DB = "mongopredict_bench"


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def bench_task(client, task, n: int) -> dict:
    coll = client[task.database][task.collection]
    definition = {
        "target": task.target,
        "task": "classification" if task.kind == "clf" else "regression",
        "columns": task.columns,
        "embeddingModel": "hashing",
        "filter": task.filter or {},
        "numTrainingDocuments": n,
        "featurizerOptions": {"cache_dir": None},
    }
    name = f"bench_{task.name}"
    _, t_create = timed(
        lambda: predictions_api.create_prediction_model(coll, PredictionModel(definition, name))
    )
    rec = predictions_api.list_prediction_models(coll, name)[0]
    fz, est, _ = _load_prediction_model(coll, name)
    stage = {"$predict": {"model": name, "filter": task.filter or {}, "limit": n}}

    cur, t_project = timed(
        lambda: predictions_api.aggregate(
            coll, [stage, {"$project": {"p": {"$meta": "prediction"}}}]
        )
    )
    got = {d["_id"]: d["p"] for d in cur.to_list()}

    # the client-side code $predict replaces, for the same documents
    docs, t_fetch = timed(lambda: list(coll.find(task.filter or {}, fz._projection(None), limit=n)))
    pred, t_pred = timed(lambda: est.predict(fz.transform(docs)))
    # regression outputs differ in the last bits between batch sizes, so floats are compared with a tolerance
    pairs = [(got.get(d["_id"]), p.item() if hasattr(p, "item") else p) for d, p in zip(docs, pred)]
    floats = [(a, b) for a, b in pairs if isinstance(b, float) and a is not None]
    same = sum(
        abs(a - b) <= 1e-5 * (1 + abs(b)) if isinstance(b, float) and a is not None else a == b
        for a, b in pairs
    )
    max_diff = max((abs(a - b) for a, b in floats), default=0.0)

    _, t_group = timed(
        lambda: predictions_api.aggregate(
            coll, [stage, {"$group": {"_id": {"$meta": "prediction"}, "n": {"$sum": 1}}}]
        )
    )
    out = client[SCRATCH_DB][task.name]
    out.drop()
    merge = [
        stage,
        {"$project": {"p": {"$meta": "prediction"}}},
        {"$merge": {"into": {"db": SCRATCH_DB, "coll": task.name}, "on": "_id"}},
    ]
    mcur, t_merge = timed(lambda: predictions_api.aggregate(coll, merge))
    n_merged = out.count_documents({})
    predictions_api.drop_prediction_model(coll, name)
    return {
        "task": task.name,
        "ns": task.ns,
        "kind": task.kind,
        "n_train": rec["numTrainingDocuments"],
        "metrics": rec["metrics"],
        "model_bytes": rec["modelBytes"],
        "t_create": t_create,
        "n_predicted": cur.stats["nDocuments"],
        "t_project": t_project,
        "stats": cur.stats,
        "t_baseline": t_fetch + t_pred,
        "t_baseline_fetch": t_fetch,
        "t_baseline_predict": t_pred,
        "agreement": same / max(len(docs), 1),
        "max_abs_diff": max_diff,
        "t_group": t_group,
        "t_merge": t_merge,
        "merge_stats": mcur.stats,
        "n_merged": n_merged,
    }


def bench_porto(client, cache_dir: str | None) -> dict:
    from demos.embeddings_api.airbnb_room_type import COLUMNS, TEXT_FIELDS

    coll = client.sample_airbnb.listingsAndReviews
    porto = {"address.market": "Porto"}
    model = PredictionModel(
        {
            "target": "room_type",
            "columns": COLUMNS,
            "embedFields": TEXT_FIELDS,
            "embeddingModel": "voyage-4-nano",
            "filter": {"address.market": {"$ne": "Porto"}},
            "featurizerOptions": {"cache_dir": cache_dir},
        },
        name="room_type_bench",
    )
    _, t_create = timed(lambda: predictions_api.create_prediction_model(coll, model))
    rec = predictions_api.list_prediction_models(coll, "room_type_bench")[0]
    stage = {"$predict": {"model": "room_type_bench", "filter": porto}}
    top, t_top = timed(
        lambda: predictions_api.aggregate(
            coll,
            [
                stage,
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
            ],
        )
    )
    conf, t_conf = timed(
        lambda: predictions_api.aggregate(
            coll,
            [
                stage,
                {
                    "$group": {
                        "_id": {"actual": "$room_type", "predicted": {"$meta": "prediction"}},
                        "n": {"$sum": 1},
                    }
                },
                {"$sort": {"n": -1}},
            ],
        )
    )
    predictions_api.drop_prediction_model(coll, "room_type_bench")
    return {
        "n_train": rec["numTrainingDocuments"],
        "metrics": rec["metrics"],
        "t_create": t_create,
        "top": top.to_list(),
        "t_top": t_top,
        "top_stats": top.stats,
        "confusion": [
            {"actual": c["_id"]["actual"], "predicted": c["_id"]["predicted"], "n": c["n"]}
            for c in conf.to_list()
        ],
        "t_confusion": t_conf,
        "confusion_stats": conf.stats,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--uri", default="mongodb://localhost:27017")
    p.add_argument(
        "--n", type=int, default=5000, help="documents per task, for training and for prediction"
    )
    p.add_argument("--cache-dir", default=".mongopredict_cache", help="embedding cache for part 2")
    p.add_argument("--skip-porto", action="store_true")
    args = p.parse_args()
    client = MongoClient(args.uri)
    out = {"n": args.n, "tasks": []}
    for task in TASKS:
        r = bench_task(client, task, args.n)
        print(
            f"{task.name}: n={r['n_predicted']} create={r['t_create']:.2f}s project={r['t_project']:.2f}s "
            f"baseline={r['t_baseline']:.2f}s agreement={r['agreement']:.3f} merge={r['t_merge']:.2f}s",
            flush=True,
        )
        out["tasks"].append(r)
    client.drop_database(SCRATCH_DB)
    if not args.skip_porto:
        out["porto"] = bench_porto(client, args.cache_dir)
        print("porto:", out["porto"]["metrics"], f"create={out['porto']['t_create']:.1f}s")
    path = ROOT / "results" / "predictions_api.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
