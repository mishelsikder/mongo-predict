"""Run one (task, mode) speed measurement in a fresh process and print a JSON line.

Stages (training documents, hashing embedder, no cache):

* ``t_load_split``: dataset load + ``train_test_split``
* ``t_fetch``: iterating the training batches (in memory: slicing; streaming: DB round trips)
* ``t_json``: serialising the documents to the JSON strings that get embedded
* ``t_tabular``: column extraction + type coercion + sklearn preprocessing
* ``t_embed``: the embedding backend on the JSON strings (hashing here, so cheap)
* ``t_fit_total`` / ``t_evaluate_test``: the end to end calls
* ``rss_peak_mb``: peak resident memory of the process
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import numpy as np
from pymongo import MongoClient

from pymongopredict import DocumentFeaturizer, MongoDataset
from src._metrics import evaluate, linear
from src._split import split
from src.tasks import BY_NAME


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--mode", choices=["in_memory", "streaming"], default="in_memory")
    p.add_argument("--uri", default="mongodb://localhost:27017")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--embed", choices=["full", "none"], default="full")
    args = p.parse_args()
    task = BY_NAME[args.task]
    client = MongoClient(args.uri)
    coll = client[task.database][task.collection]
    stats = client[task.database].command("collstats", task.collection)
    out = {
        "task": task.name,
        "mode": args.mode,
        "ns": task.ns,
        "avg_doc_bytes": stats.get("avgObjSize"),
        "n_docs": stats.get("count"),
        "rss_start_mb": rss_mb(),
    }

    t0 = time.perf_counter()
    ds = MongoDataset(
        coll,
        target=task.target,
        filter=task.filter,
        limit=args.limit,
        streaming=args.mode == "streaming",
        batch_size=args.batch_size,
    )
    ds_train, ds_test = split(ds, **task.split)
    out["t_load_split"] = time.perf_counter() - t0
    out["n_train"], out["n_test"] = len(ds_train), len(ds_test)
    out["rss_after_load_mb"] = rss_mb()

    est = linear(task.kind)
    feat_kwargs = dict(
        columns=task.columns,
        embed_fields=None if args.embed == "full" else False,
        embedding_model="hashing",
        embedding_kwargs={"n_features": 256},
        cache_dir=None,
        batch_size=args.batch_size,
    )
    baseline = not hasattr(DocumentFeaturizer, "_projection")  # commit 55b3540: no dataset input
    if baseline:
        feat_kwargs.pop("batch_size")
        feat_kwargs["exclude"] = [task.target]  # the old estimator classes did this
    f = DocumentFeaturizer(**feat_kwargs)

    def fit_transform(ds):
        return f.fit_transform_batches(ds.iter_batches()) if baseline else f.fit_transform(ds)

    def transform(ds):
        return f.transform_batches(ds.iter_batches()) if baseline else f.transform(ds)

    y_train = np.asarray(ds_train.y)
    y_test = np.asarray(ds_test.y)
    if task.kind == "reg":  # baseline datasets return object arrays
        y_train, y_test = y_train.astype(float), y_test.astype(float)
    else:
        y_train, y_test = y_train.astype(str), y_test.astype(str)
    t0 = time.perf_counter()
    est.fit(fit_transform(ds_train), y_train)
    out["t_fit_total"] = time.perf_counter() - t0
    out["n_features"] = len(f.get_feature_names_out())

    # stage breakdown on the training documents (fitted featurizer, so only transform costs)
    proj = f._projection(ds_train.target) if hasattr(f, "_projection") else None
    t_fetch = t_json = t_tab = t_emb = 0.0
    n_json_chars = 0
    frames = []
    t0 = time.perf_counter()
    for docs in ds_train.iter_batches(proj):
        t_fetch += time.perf_counter() - t0
        if f.embedder_ is not None:
            t1 = time.perf_counter()
            texts = f._texts(docs)
            t_json += time.perf_counter() - t1
            n_json_chars += sum(map(len, texts))
            t1 = time.perf_counter()
            f.embedder_.embed(texts)
            t_emb += time.perf_counter() - t1
        if f.tabular_ is not None:
            t1 = time.perf_counter()
            frames.append(f._frame(docs))
            t_tab += time.perf_counter() - t1
        t0 = time.perf_counter()
    if f.tabular_ is not None:  # the featurizer encodes the collected column frames once
        t1 = time.perf_counter()
        if hasattr(f, "_assemble"):
            import pandas as pd

            f.tabular_.transform(f._coerce(pd.concat(frames, ignore_index=True)))
        else:  # baseline code: one sklearn call per batch
            for fr in frames:
                f.tabular_.transform(f._coerce(fr))
        t_tab += time.perf_counter() - t1
    out.update(
        t_fetch=t_fetch,
        t_json=t_json,
        t_tabular=t_tab,
        t_embed=t_emb,
        json_chars_per_doc=n_json_chars / max(out["n_train"], 1),
    )
    t0 = time.perf_counter()
    out["metric"] = evaluate(task.kind, ds_test.y, est.predict(transform(ds_test)))
    out["t_evaluate_test"] = time.perf_counter() - t0
    out["rss_peak_mb"] = rss_mb()
    out["wrapper_docs_per_s"] = out["n_train"] / max(t_fetch + t_json + t_tab, 1e-9)
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
