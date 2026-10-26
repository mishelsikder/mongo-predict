"""Render ``reports/01_predictions_api.html`` (the ``$predict`` stage) from ``results/predictions_api.json``.

uv run python -m src.predictions_api_report
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import statistics

from src._html import CSS, avg_row, esc, f, mean_of, table, tile

HERE = pathlib.Path(__file__).resolve().parents[1] / "reports"  # reports/
RESULTS = HERE.parent / "results" / "predictions_api.json"
DOC = HERE / "01_predictions_api.html"


def metric(r) -> str:
    m = r["metrics"]
    return f"accuracy {m['accuracy']:.3f}" if r["kind"] == "clf" else f"R² {m['r2']:.3f}"


def task_tables(tasks):
    header = [
        "task",
        "n",
        "create_prediction_model s",
        "holdout metric",
        "$predict s",
        "client baseline s",
        "$predict / baseline",
        "fetch s",
        "predict s",
        "upload s",
        "server s",
        "agreement",
        "$group s",
        "$merge s",
    ]
    rows, raw = [], []
    for r in tasks:
        s = r["stats"]
        vals = [
            r["n_predicted"],
            r["t_create"],
            r["t_project"],
            r["t_baseline"],
            r["t_project"] / r["t_baseline"],
            s["fetchSeconds"],
            s["predictSeconds"],
            s["uploadSeconds"],
            s["serverSeconds"],
            r["agreement"],
            r["t_group"],
            r["t_merge"],
        ]
        raw.append(vals)
        rows.append(
            [
                esc(r["task"]),
                f"{r['n_predicted']:,}",
                f(r["t_create"]),
                metric(r),
                f(r["t_project"]),
                f(r["t_baseline"]),
                f"{vals[4]:.2f}x",
                f(s["fetchSeconds"]),
                f(s["predictSeconds"]),
                f(s["uploadSeconds"]),
                f(s["serverSeconds"]),
                f"{100 * r['agreement']:.1f}%",
                f(r["t_group"]),
                f(r["t_merge"]),
            ]
        )
    m = [mean_of(c) for c in zip(*raw)]
    avg = avg_row(
        f"average ({len(tasks)} tasks)",
        [
            f"{m[0]:,.0f}",
            f(m[1]),
            "",
            f(m[2]),
            f(m[3]),
            f"{m[4]:.2f}x",
            f(m[5]),
            f(m[6]),
            f(m[7]),
            f(m[8]),
            f"{100 * m[9]:.1f}%",
            f(m[10]),
            f(m[11]),
        ],
    )
    return table(header, [avg]), table(header, rows)


def timing_section():
    """Sec. 4 tables from ``results/estimator_timing*.json``: ridge against XGBoost, 16 threads and 1 thread."""
    runs = {}
    for label, name in (
        ("16 threads", "estimator_timing.json"),
        ("1 thread", "estimator_timing_1cpu.json"),
    ):
        path = HERE.parent / "results" / name
        if path.exists():
            runs[label] = json.loads(path.read_text())
    med = statistics.median
    summary = []
    for label, d in runs.items():
        t = d["tasks"]
        for est in ("ridge", "xgboost"):
            summary.append(
                [
                    label,
                    "ridge (RidgeCV, 13 alphas)" if est == "ridge" else "XGBoost (defaults)",
                    f(med(r[f"{est}_fit_s"] for r in t)),
                    f(sum(r[f"{est}_fit_s"] for r in t), 1),
                    f(max(r[f"{est}_fit_s"] for r in t), 1),
                    f(med(r[f"{est}_predict_s"] / r["n_test"] * 1e6 for r in t)),
                    f(med(r[f"{est}_predict_s"] * 1e3 for r in t), 1),
                ]
            )
    summary_tab = table(
        [
            "threads",
            "model",
            "fit s (median task)",
            "fit s (all 14 tasks)",
            "fit s (slowest task)",
            "inference ms per 1,000 docs (median)",
            "inference ms per test set (median)",
        ],
        summary,
        text_cols=(0, 1),
    )
    first = next(iter(runs.values()))["tasks"]
    header = ["task", "train / test docs", "classes"]
    for label in runs:
        header += [
            f"ridge fit s ({label})",
            f"XGBoost fit s ({label})",
            f"ridge inference ms ({label})",
            f"XGBoost inference ms ({label})",
        ]
    rows = []
    for i, r in enumerate(first):
        row = [
            esc(r["task"]),
            f"{r['n_train']:,} / {r['n_test']:,}",
            r["n_classes"] or "regression",
        ]
        for d in runs.values():
            q = d["tasks"][i]
            row += [
                f(q["ridge_fit_s"]),
                f(q["xgboost_fit_s"]),
                f(q["ridge_predict_s"] * 1e3, 1),
                f(q["xgboost_predict_s"] * 1e3, 1),
            ]
        rows.append(row)
    return runs, summary_tab, table(header, rows, text_cols=(0,))


def main():
    d = json.loads(RESULTS.read_text())
    tasks, porto = d["tasks"], d.get("porto")
    n_tasks = len(tasks)
    ratio_med = statistics.median(r["t_project"] / r["t_baseline"] for r in tasks)
    per_1k = statistics.median(
        (r["t_project"] - r["t_baseline"]) / r["n_predicted"] * 1000 for r in tasks
    )
    agree_all = all(r["agreement"] == 1.0 for r in tasks)
    worst = max(tasks, key=lambda r: r["max_abs_diff"])
    n_docs = sum(r["n_predicted"] for r in tasks)
    share = {
        k: sum(r["stats"][k] for r in tasks) / sum(r["t_project"] for r in tasks)
        for k in ("fetchSeconds", "predictSeconds", "uploadSeconds", "serverSeconds")
    }
    merged_ok = all(r["n_merged"] == r["n_predicted"] for r in tasks)
    avg_tab, task_tab = task_tables(tasks)
    runs, timing_sum, timing_tasks = timing_section()
    t16, t1 = runs["16 threads"]["tasks"], runs.get("1 thread", {}).get("tasks")
    med = statistics.median
    fit16 = {e: med(r[f"{e}_fit_s"] for r in t16) for e in ("ridge", "xgboost")}
    fit1 = {e: med(r[f"{e}_fit_s"] for r in t1) for e in ("ridge", "xgboost")} if t1 else None
    inf16 = {
        e: med(r[f"{e}_predict_s"] / r["n_test"] * 1e6 for r in t16) for e in ("ridge", "xgboost")
    }
    inf1 = (
        {e: med(r[f"{e}_predict_s"] / r["n_test"] * 1e6 for r in t1) for e in ("ridge", "xgboost")}
        if t1
        else None
    )
    many = max(t16, key=lambda r: r["n_classes"] or 0)
    many1 = next(r for r in t1 if r["task"] == many["task"]) if t1 else None
    biggest = max(t16, key=lambda r: r["n_train"])
    few = [r for r in t16 if (r["n_classes"] or 0) != 1 and (r["n_classes"] or 2) <= 2]
    xgb_wins = [r for r in few if r["xgboost_fit_s"] < r["ridge_fit_s"]]
    speedup = ""
    if t1:
        sp = {
            e: med(q[f"{e}_fit_s"] / r[f"{e}_fit_s"] for r, q in zip(t16, t1))
            for e in ("ridge", "xgboost")
        }
        speedup = (
            f" Going from 1 to 16 threads speeds up fitting a median {sp['ridge']:.1f}x for ridge and "
            f"{sp['xgboost']:.1f}x for XGBoost."
        )
    timing_s = sum(
        r["ridge_fit_s"] + r["xgboost_fit_s"] + 3 * (r["ridge_predict_s"] + r["xgboost_predict_s"])
        for d in runs.values()
        for r in d["tasks"]
    )
    one = f"; on one thread, {fit1['ridge']:.1f} s against {fit1['xgboost']:.1f} s" if fit1 else ""
    total_s = sum(
        r["t_create"] + r["t_project"] + r["t_baseline"] + r["t_group"] + r["t_merge"]
        for r in tasks
    )

    pc = porto["confusion"]
    n_porto = sum(c["n"] for c in pc)
    acc_porto = sum(c["n"] for c in pc if c["actual"] == c["predicted"]) / n_porto
    top_tab = table(
        ["name", "room type", "predicted", "predictScore"],
        [
            [esc(t["name"]), esc(t["room_type"]), esc(t["predicted"]), f(t["score"], 3)]
            for t in porto["top"]
        ],
        text_cols=(0,),
    )
    conf_tab = table(
        ["actual", "predicted", "n"],
        [[esc(c["actual"]), esc(c["predicted"]), c["n"]] for c in pc],
        text_cols=(0, 1),
    )

    design = table(
        ["", "$vectorSearch (Atlas)", "$predict (mongo-predict)"],
        [
            [
                "defined with",
                '<code>create_search_index(SearchIndexModel(definition, name, type="vectorSearch"))</code>',
                "<code>create_prediction_model(PredictionModel(definition, name))</code>",
            ],
            [
                "managed with",
                "<code>list_ / update_ / drop_search_index</code>",
                "<code>list_ / update_ / drop_prediction_model</code>",
            ],
            [
                "stored in",
                "the search index (<code>mongot</code>)",
                "the <code>__mongopredict_models</code> collection (status, definition, metrics, pickled model)",
            ],
            [
                "stage",
                "<code>{$vectorSearch: {index, path, queryVector, filter, limit, numCandidates}}</code>",
                "<code>{$predict: {model, filter, limit, batchSize}}</code>",
            ],
            ["position", "first stage only", "first stage only"],
            [
                "output",
                '<code>{$meta: "vectorSearchScore"}</code>',
                '<code>{$meta: "prediction"}</code>, <code>{$meta: "predictScore"}</code>, <code>{$meta: "predictProbabilities"}</code>',
            ],
            ["runs", "inside Atlas", "on the client, then the rest of the pipeline on the server"],
        ],
        text_cols=(0, 1, 2),
    )

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predictions API $predict</title><style>{CSS}</style></head><body>
<header class="top"><span><a href="index.html">reports</a> &rsaquo; 01 predictions api</span>
<span>generated {dt.date.today().isoformat()} by <code>uv run python -m src.predictions_api_report</code></span></header>

<h1>Predictions API: a <code>$predict</code> aggregation stage, modelled on <code>$vectorSearch</code></h1>
<p class="note">A prototype of predictions inside MongoDB aggregation pipelines that needs no change to the server:
<code>pymongopredict.predictions_api</code> stores a trained model in the database and runs a leading <code>$predict</code> stage on the
client. Measured on the {n_tasks} MongoDB sample tasks of <code>src/tasks.py</code> and on an Airbnb model with voyage-4-nano.</p>

<div class="tiles">
{tile("100%" if agree_all else "below 100%", f"of {n_docs:,} documents get the same prediction as the client-side code")}
{tile(f"{ratio_med:.2f}x", "time of a $predict pipeline relative to find + transform + predict (median)")}
{tile(f"{per_1k * 1000:.0f} ms", "added per 1,000 documents (median)")}
{tile(f"{acc_porto:.3f}", f"accuracy on Porto's {n_porto} listings, model trained on the other markets")}
</div>

<div class="callout"><ul>
<li>On all {n_tasks} tasks a <code>$predict</code> pipeline returns the same predictions as calling the model on the client
(regression outputs differ only by float32 rounding between batch sizes: at most {worst["max_abs_diff"]:.1e} on
{esc(worst["task"])}, whose values are in the thousands), so <code>$match</code>, <code>$sort</code>,
<code>$group</code> and <code>$merge</code> can be applied to predictions on the server (Sec. [2](#sec-2)).</li>
<li>The stage costs a median {ratio_med:.2f}x the time of the client-side code it replaces, about {per_1k * 1000:.0f} ms more per 1,000
documents with the hashing embedder: uploading the predictions and the server join take {100 * (share["uploadSeconds"] + share["serverSeconds"]):.0f}%
of the pipeline time. Neither depends on the embedding model, so the share is smaller with a slower embedder such as voyage-4-nano.</li>
<li>Fitting the default ridge model takes a median {fit16["ridge"]:.1f} s per task with 16 threads and default XGBoost
{fit16["xgboost"]:.1f} s{one} (Sec. [4](#sec-4)). XGBoost builds one set of trees per class, so the {many["n_classes"]}-class task takes it
{many["xgboost_fit_s"]:.0f} s. Inference is under {max(inf16.values()) if not inf1 else max(max(inf16.values()), max(inf1.values())):.0f} ms per 1,000 documents for both, far below the time to embed them.</li>
<li>A room type model trained with voyage-4-nano on the {porto["n_train"]:,} listings outside Porto (holdout accuracy
{porto["metrics"]["accuracy"]:.3f}) labels Porto's {n_porto} listings with accuracy {acc_porto:.3f}, through one pipeline
(Sec. [3](#sec-3)).</li>
</ul></div>

<h2 id="sec-1">1. Design</h2>
<p>A <code>$vectorSearch</code> query needs an index created beforehand and referenced by name; the stage must come first and later
stages read its score through <code>$meta</code>. <code>$predict</code> keeps that shape (Table [1](#tab-1)), with a prediction model in
place of the index.</p>
<div id="tab-1">{design}</div>
<pre>from pymongopredict import PredictionModel, predictions_api
predictions_api.enable()   # pymongo's Collection.aggregate now understands $predict

coll.create_prediction_model(PredictionModel(
    {{"target": "room_type", "columns": [...], "embedFields": [...], "embeddingModel": "voyage-4-nano",
     "filter": {{"address.market": {{"$ne": "Porto"}}}}}},
    name="room_type"))
coll.aggregate([
    {{"$predict": {{"model": "room_type", "filter": {{"address.market": "Porto"}}}}}},
    {{"$sort": {{"score": {{"$meta": "predictScore"}}}}}},
    {{"$project": {{"name": 1, "predicted": {{"$meta": "prediction"}}, "score": {{"$meta": "predictScore"}}}}}},
])</pre>
<p>How a pipeline runs:</p>
<ol>
<li><b>Model</b>: <code>create_prediction_model</code> fits a <code>DocumentFeaturizer</code> (columns plus an embedding of the
document) and an sklearn estimator (logistic regression or ridge by default) on the documents that have the target, records holdout
metrics on 20%, refits on all of them and stores the pickled model with its status in <code>__mongopredict_models</code>.</li>
<li><b>Client</b>: <code>aggregate</code> fetches the documents selected by the stage's <code>filter</code> and <code>limit</code>,
projected to the fields the model reads, featurizes and predicts them in batches, and inserts only
<code>{{_id, prediction, score, probabilities}}</code> rows into a temporary collection.</li>
<li><b>Server</b>: the rest of the pipeline runs on the temporary collection after a <code>$lookup</code> on <code>_id</code> that
brings back each full document; <code>{{$meta: ...}}</code> references are rewritten to the joined fields and the helper field is
removed before the output (or before a final <code>$merge</code> / <code>$out</code>). The temporary collection is dropped
afterwards and the result is returned as a list-backed cursor whose <code>stats</code> hold the time of each step.</li>
</ol>

<h2 id="sec-2">2. Overhead and agreement on the sample tasks</h2>
<p>For each of the {n_tasks} tasks (one per sample collection), a model with the task's columns and the hashing embedder on the whole
document (no embedding cache) is created on up to {d["n"]:,} documents, then applied to up to {d["n"]:,} documents four ways: a
<code>$predict</code> pipeline that projects the prediction, the client-side code it replaces (<code>find</code> +
<code>transform</code> + <code>predict</code>), a <code>$group</code> on the prediction and a <code>$merge</code> of the predictions into
a scratch database (Table [2](#tab-2); per task rows in Table [3](#tab-3)). Agreement is the share of documents whose
<code>$predict</code> output equals the client-side prediction (relative tolerance 1e-5 for regression). The predicted documents
overlap the training documents, so the holdout metric, not these predictions, measures accuracy. The <code>$merge</code> wrote
{"every" if merged_ok else "not every"} prediction on every task.</p>
<div id="tab-2">{avg_tab}</div>
<details id="tab-3"><summary>Table 3: per task results</summary>{task_tab}</details>
<p>Fetching and predicting take {100 * (share["fetchSeconds"] + share["predictSeconds"]):.0f}% of the <code>$predict</code> time, as in
the client-side code; the upload of the predictions takes {100 * share["uploadSeconds"]:.0f}% and the server join and remaining
stages {100 * share["serverSeconds"]:.0f}%.</p>

<h2 id="sec-3">3. Example: Airbnb room types in Porto with voyage-4-nano</h2>
<p><code>demos/predictions_api/predict_stage.py</code>: the room type model is trained on every market except Porto
({porto["n_train"]:,} listings, the 14 columns of the embeddings API demo and the embedded text fields; holdout accuracy
{porto["metrics"]["accuracy"]:.3f}, AUROC {porto["metrics"]["auroc"]:.3f}). Two pipelines on Porto's {n_porto} listings:
the most confident "Private room" predictions (<code>$match</code> on the prediction, <code>$sort</code> on
<code>predictScore</code>; Table [4](#tab-4)) and actual against predicted room type (<code>$group</code>; Table [5](#tab-5)).
With the embeddings already in the cache, creating the model took {porto["t_create"]:.1f} s and each pipeline
{porto["t_top"]:.1f} s; without the cache, embedding the 5,555 listings took about 2 minutes on a shared H100.</p>
<div id="tab-4">{top_tab}</div>
<div id="tab-5">{conf_tab}</div>
<p>Most errors are entire homes predicted as private rooms; the 5 shared rooms are all predicted as private rooms, as shared rooms
are 1.5% of the training listings.</p>

<h2 id="sec-4">4. Estimator fit and inference time: ridge and XGBoost</h2>
<p><code>src/estimator_timing.py</code> featurizes each of the {len(t16)} tasks once (the task's columns plus the whole document
embedded at 1,024 dimensions, voyage-4-nano's output size; the hashing embedder stands in for it, since fit and inference time
depend on the size of the feature matrix, not on where the embedding came from) and then times, on the same train and test
matrices, the predictions API's <code>estimator="ridge"</code> (<code>RidgeClassifierCV</code> / <code>RidgeCV</code> over 13
alphas with leave-one-out CV) and <code>XGBClassifier</code> / <code>XGBRegressor</code> with default settings (100 trees per class,
depth 6, hist). Fit is timed once and inference (<code>predict</code>, plus class probabilities or scores for classifiers) is the
median of 3 runs on the test documents. Both runs share the login node with other jobs; "16 threads" caps BLAS and XGBoost at 16
threads, "1 thread" at one (Table [6](#tab-6); per task in Table [7](#tab-7)). Embedding time is not included.</p>
<div id="tab-6">{timing_sum}</div>
<details id="tab-7"><summary>Table 7: fit and inference time per task</summary>{timing_tasks}</details>
<p>Ridge time grows with the number of training documents (the largest task, {esc(biggest["task"])} with
{biggest["n_train"]:,} documents, takes {biggest["ridge_fit_s"]:.1f} s with 16 threads), while XGBoost time grows with the number of
classes: {esc(many["task"])} has {many["n_classes"]} classes and takes {many["xgboost_fit_s"]:.0f} s with 16 threads{f" and {many1['xgboost_fit_s']:.0f} s with one" if many1 else ""}.
With one class or two, or a regression target, XGBoost builds only 100 trees, and it is faster than ridge on
{len(xgb_wins)} of these {len(few)} tasks ({", ".join(esc(r["task"]) for r in xgb_wins)}); ridge fits all 13 alphas
whatever the number of classes.{speedup} routes_codeshare has a single class in its training split, so XGBoost has nothing to fit there. Inference
takes {inf16["ridge"]:.1f} ms (ridge) and {inf16["xgboost"]:.1f} ms (XGBoost) per 1,000 documents with 16 threads{f", {inf1['ridge']:.1f} ms and {inf1['xgboost']:.1f} ms with one" if inf1 else ""}.</p>

<h2 id="sec-5">5. Limitations</h2>
<ul>
<li>The documents travel to the client and the model runs there; a server stage would avoid that transfer.</li>
<li>Results are materialised before they are returned, and the predictions are written to a temporary collection, so the client
needs write access to the database.</li>
<li><code>$predict</code> cannot appear inside <code>$lookup</code>, <code>$unionWith</code> or <code>$facet</code> sub-pipelines, and it
works from Python only (not from mongosh or other drivers).</li>
<li>The stored model is a pickle of at most 15 MB: only users trusted to run code on the client should be able to write
<code>__mongopredict_models</code>, unless <code>MONGOPREDICT_MODEL_KEY</code> is set, which signs stored models and refuses
unsigned ones.</li>
<li>A native stage would follow <code>$vectorSearch</code>: a thin stage in <code>mongod</code> that hands documents to a model
service. This prototype fixes the user-facing syntax such a stage could keep.</li>
</ul>

<details><summary>Computational cost</summary>
<ul>
<li>Sec. [2](#sec-2): {total_s / 60:.1f} minutes of measured time on the login node's CPUs (MongoDB 7.0.14 on the same machine), no GPU
(0 H100 node hours). Throughput of the <code>$predict</code> pipelines: {n_docs / sum(r["t_project"] for r in tasks):,.0f}
documents per second with the hashing embedder.</li>
<li>Sec. [4](#sec-4): {timing_s / 60:.1f} minutes of fit and inference on the login node's CPUs over the {len(runs)} thread settings
(0 H100 node hours), plus featurizing the {len(t16)} tasks with the hashing embedder for each run.</li>
<li>Sec. [3](#sec-3): voyage-4-nano embeddings of the 5,555 listings, computed once on one shared H100 (about 2 minutes, under 0.01 H100
node hours) and then read from the embedding cache.</li>
</ul></details>

<details><summary>Reproducibility</summary>
<pre># MongoDB with the sample data
uv run python -m demos.load_sample_data --uri mongodb://localhost:27017 \\
    sample_airbnb sample_weatherdata sample_analytics sample_geospatial sample_mflix sample_supplies sample_training
# benchmark (part 2 embeds with voyage-4-nano: a GPU, or a cache that holds the embeddings)
uv run python -m src.predictions_api_bench --uri mongodb://localhost:27017 --cache-dir .mongopredict_cache
# Sec. 4: ridge and XGBoost timing, 16 threads and 1 thread
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 uv run python -m src.estimator_timing --threads 16
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m src.estimator_timing --threads 1 \\
    --out results/estimator_timing_1cpu.json
# this document
uv run python -m src.predictions_api_report</pre>
<p>Prompt for a coding agent: "In the mongo-predict repo (github.com/voyage-ai/mongo-predict), load the MongoDB sample datasets into a
local MongoDB, run <code>python -m src.predictions_api_bench</code> (part 1: for each task of <code>src/tasks.py</code>, create a prediction
model with the hashing embedder on up to 5,000 documents and time a <code>$predict</code> projection, the client-side
find + transform + predict, a <code>$group</code> and a <code>$merge</code>, checking that the predictions agree; part 2: the Airbnb room
type model trained outside Porto with voyage-4-nano and queried on Porto), which writes <code>results/predictions_api.json</code>; run
<code>python -m src.estimator_timing</code> with 16 threads and with 1 thread (BLAS and XGBoost capped through the environment and
<code>--threads</code>; the second with <code>--out results/estimator_timing_1cpu.json</code>) to time fitting and inference of the
ridge estimator against default XGBoost on the same 1,024-dimension embedding plus column features of each task; then
render <code>reports/01_predictions_api.html</code> with <code>python -m src.predictions_api_report</code>. The stage is in
<code>pymongopredict/predictions_api/</code>."</p>
</details>
<footer>mongo-predict reports. Result files: <code>results/predictions_api.json</code>, <code>results/estimator_timing.json</code>,
<code>results/estimator_timing_1cpu.json</code>.</footer>
</body></html>
"""
    # references link on the number only: "Sec. [2](#sec-2)" -> Sec. <a href='#sec-2'>2</a>
    page = re.sub(r"\[(\d+)\]\((#[\w-]+)\)", r"<a href='\2'>\1</a>", page)
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text(page)
    print(f"wrote {DOC} ({len(page)} chars)")


if __name__ == "__main__":
    main()
