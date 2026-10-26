"""Render ``reports/00_embeddings_api.html`` (embeddings API wrapper speed, before vs after) from ``results/speed_*.json``.

    uv run python -m src.embeddings_api_report
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import statistics

from src._html import CSS, avg_row, esc, f, mean_of, ratio, table, tile
from src.tasks import TASKS

HERE = pathlib.Path(__file__).resolve().parents[1] / "reports"  # reports/
RESULTS = HERE.parent / "results"
DOC = HERE / "00_embeddings_api.html"


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else []


def speed_section(before, after):
    key = lambda r: (r["task"], r["mode"])  # noqa: E731
    b = {key(r): r for r in before if "error" not in r}
    a = {key(r): r for r in after if "error" not in r}
    wrapper = lambda r: r["t_fetch"] + r["t_json"] + r["t_tabular"]  # noqa: E731
    rows, avg_rows, sw_all, st_all, mem = [], [], [], [], []
    raw = {"in_memory": [], "streaming": []}
    for mode in ("in_memory", "streaming"):
        for t in TASKS:
            rb, ra = b.get((t.name, mode)), a.get((t.name, mode))
            if not (rb and ra):
                continue
            sw, st = wrapper(rb) / max(wrapper(ra), 1e-9), rb["t_fit_total"] / max(ra["t_fit_total"], 1e-9)
            sw_all.append(sw)
            st_all.append(st)
            mem.append(rb["rss_peak_mb"] / ra["rss_peak_mb"])
            vals = [wrapper(rb), wrapper(ra), sw, rb["t_embed"], ra["t_embed"], rb["t_fit_total"],
                    ra["t_fit_total"], st, rb["rss_peak_mb"], ra["rss_peak_mb"]]
            raw[mode].append(vals)
            rows.append([esc(t.name), mode.replace("_", " "), ra["n_train"], ra["avg_doc_bytes"],
                         f(wrapper(rb)), f(wrapper(ra)), ratio(wrapper(rb), wrapper(ra)),
                         f(rb["t_embed"]), f(ra["t_embed"]),
                         f(rb["t_fit_total"]), f(ra["t_fit_total"]), ratio(rb["t_fit_total"], ra["t_fit_total"]),
                         f(rb["rss_peak_mb"], 0), f(ra["rss_peak_mb"], 0)])
        if raw[mode]:
            m = [mean_of(col) for col in zip(*raw[mode])]
            avg_rows.append(avg_row(f"average ({mode.replace('_', ' ')}, {len(raw[mode])} tasks)",
                                [mode.replace("_", " "), "", "", f(m[0]), f(m[1]),
                                 f"<span class='pos'>{m[2]:.1f}x</span>", f(m[3]), f(m[4]), f(m[5]), f(m[6]),
                                 f"<span class='pos'>{m[7]:.1f}x</span>", f(m[8], 0), f(m[9], 0)]))
    header = ["task", "mode", "n train", "avg doc bytes", "wrapper s before", "wrapper s after", "wrapper speed-up",
              "hashing embed s before", "hashing embed s after", "fit s before", "fit s after", "fit speed-up",
              "peak MB before", "peak MB after"]
    stats = dict(n=len(rows), sw_med=statistics.median(sw_all) if sw_all else 0, sw_min=min(sw_all, default=0),
                 sw_max=max(sw_all, default=0), st_med=statistics.median(st_all) if st_all else 0,
                 mem_med=statistics.median(mem) if mem else 0)
    stage_rows, stage_avg, stage_raw = [], [], []
    for t in TASKS:
        r = a.get((t.name, "streaming"))
        if r:
            stage_raw.append([r["n_train"], r["avg_doc_bytes"], r["json_chars_per_doc"], r["t_load_split"],
                              r["t_fetch"], r["t_json"], r["t_tabular"], r["t_embed"], r["wrapper_docs_per_s"]])
            stage_rows.append([esc(t.name), r["n_train"], r["avg_doc_bytes"], f(r["json_chars_per_doc"], 0),
                               f(r["t_load_split"]), f(r["t_fetch"]), f(r["t_json"]), f(r["t_tabular"]),
                               f(r["t_embed"]), f"{r['wrapper_docs_per_s']:,.0f}"])
    if stage_raw:
        m = [mean_of(col) for col in zip(*stage_raw)]
        stage_avg.append(avg_row(f"average ({len(stage_raw)} tasks)",
                                  [f"{m[0]:,.0f}", f"{m[1]:,.0f}", f(m[2], 0), f(m[3]), f(m[4]), f(m[5]), f(m[6]),
                                   f(m[7]), f"{m[8]:,.0f}"]))
    stage_header = ["task", "n train", "avg doc bytes", "JSON chars/doc", "load+split s", "fetch s", "JSON s",
                    "tabular s", "hashing embed s", "wrapper docs/s"]
    return (stats, table(header, avg_rows), table(header, rows), table(stage_header, stage_avg),
            table(stage_header, stage_rows))


def main():
    before, after = load("speed_before.json"), load("speed_after.json")
    stats, speed_avg, speed_table, stage_avg, stage_table = speed_section(before, after)
    speed_s = sum(r["t_load_split"] + r["t_fit_total"] + r["t_evaluate_test"] for r in before + after if "t_fit_total" in r)
    n_runs = sum("t_fit_total" in r for r in before + after)
    docs = sum(r["n_train"] for r in after if "t_fit_total" in r)
    fit_after = sum(r["t_fit_total"] for r in after if "t_fit_total" in r)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>embeddings API speed</title><style>{CSS}</style></head><body>
<header class="top"><span><a href="index.html">reports</a> &rsaquo; 00 embeddings api</span>
<span>generated {dt.date.today().isoformat()} by <code>uv run python -m src.embeddings_api_report</code></span></header>

<h1>embeddings API: wrapper speed, before vs after the performance work</h1>
<p class="note">The {len(TASKS)} tasks of <code>src/tasks.py</code> (one per collection of the MongoDB sample
datasets), each run in a fresh process in two modes (in memory and streaming) with the hashing embedder, on the baseline commit
<code>55b3540</code> (the package before it moved to this repository) and on the current code.</p>

<div class="tiles">
{tile(f"{stats['sw_med']:.1f}x", "wrapper stages (fetch + JSON + tabular) faster (median)")}
{tile(f"{stats['st_med']:.1f}x", "end to end fit faster (median)")}
{tile(f"{stats['mem_med']:.2f}x", "peak memory, before / after (median)")}
{tile(f"{stats['n']}", "(task, mode) pairs measured")}
</div>

<div class="callout">
<ul>
<li>Across {stats['n']} (task, mode) pairs the wrapper stages (fetch + JSON + tabular) got <b>{stats['sw_med']:.1f}x faster</b>
(median; {stats['sw_min']:.1f}x to {stats['sw_max']:.1f}x) and the end to end <code>fit</code> <b>{stats['st_med']:.1f}x faster</b>; peak memory changed by a
median factor of {stats['mem_med']:.2f}x.</li>
<li>The code changes behind these numbers (orjson serialisation with a per field budget, tabular columns encoded once per call,
vectorised column coercion, word n-gram hashing) are listed under "What changed" below.</li>
</ul>
</div>

<details><summary>Setup</summary>
<ul>
<li>MongoDB 7.0 standalone on the same machine, sample datasets loaded with <code>python -m demos.load_sample_data</code>.</li>
<li>Every (task, mode) pair runs in a fresh process, hashing embedder (256 features, no cache), linear estimator;
"before" is commit <code>55b3540</code> of the package before it moved to this repository, "after" is the current code. Task definitions are in
<code>src/tasks.py</code>.</li>
</ul>
</details>

<h2>1. Before vs after</h2>
<p>Wrapper seconds = fetching the training documents + serialising them to JSON + extracting and encoding the columns, measured on the
fitted featurizer. The hashing embedder time is listed separately because it stands in for an embedding model and is not part of the
wrapper (its analyzer also changed from character to word n-grams). <code>fit</code> is the end to end call. Speed-ups are blue when
faster and red when slower. The main table averages across tasks (speed-ups: mean of the per task factors); the per dataset rows are
in the expandable below it.</p>
{speed_avg}
<details><summary>Per dataset speed results</summary>{speed_table}</details>

<h2>2. Stage breakdown after the changes (streaming mode)</h2>
{stage_avg}
<details><summary>Per dataset stage breakdown</summary>{stage_table}</details>
<details><summary>What changed</summary>
<ul>
<li><b>JSON serialisation</b> uses <code>orjson</code> with a <code>default</code> hook for BSON types instead of a recursive pure-Python
conversion followed by <code>json.dumps</code>, and the <code>max_json_chars</code> budget is applied field by field so a huge trailing
array (reviews, transactions) is never serialised only to be cut off: 3x to 11x faster per document.</li>
<li><b>Tabular columns are encoded once per call</b>: <code>transform_batches</code> collects the small column frames of all batches and runs
the sklearn preprocessing a single time (embeddings still go batch by batch). Per batch sklearn overhead used to dominate on collections of
small documents (routes: 2.2 s to 0.7 s for 67k documents).</li>
<li><b>Column coercion</b> is vectorised and dotted paths are pre-split once per column.</li>
<li><b>Hashing embedder</b> defaults to hashed word uni/bigrams instead of character 3-5 grams (about 10x faster on long JSON).</li>
<li><b>Prefetching</b>: in streaming mode the next <code>_id</code> batch is fetched in a background thread while the current one is
embedded. Measured separately with voyage-4-nano on 2,048 Airbnb documents: 17.6 / 15.2 s without vs 15.1 / 15.3 s with prefetch, i.e. no
measurable gain with a local MongoDB, where fetching is under 1% of the embedding time.</li>
<li><b>Embedding models</b>: bfloat16 on CUDA (float16 overflows in voyage-4-nano and produced NaN vectors; a non-finite batch now triggers a
float32 fallback), the token limit is capped at each model's own maximum, texts are grouped by length so a batch holds at most
<code>max_batch_tokens</code>, and Matryoshka truncation (<code>"voyage-4-nano@512"</code>) is applied after the cache.</li>
<li>Fixed: removing a nested target such as <code>scores.0.score</code> from the embedded JSON now descends through arrays; stratified
splits pool singleton classes instead of failing.</li>
</ul></details>

<details><summary>Computational cost</summary>
<ul>
<li>{n_runs} (task, mode) runs (before and after), CPU only, {speed_s / 60:.0f} minutes of measured stage time on a shared 176 core box;
no GPU (0 H100 node hours).</li>
<li>Throughput after the changes: {docs / max(fit_after, 1e-9):,.0f} training documents per second of end to end <code>fit</code>
with the hashing embedder, summed over all runs.</li>
</ul></details>

<details><summary>Reproducibility</summary>
<pre># MongoDB with the sample data (any 7.x server; the loader pulls the collections from a public mirror)
uv run python -m demos.load_sample_data --uri mongodb://localhost:27017 \\
    sample_airbnb sample_weatherdata sample_analytics sample_geospatial sample_mflix sample_supplies sample_training
# baseline: a git worktree of the package at 55b3540 (before the changes and the move here) with src/ copied in
uv run python -m src.speed --uri mongodb://localhost:27017 --tag before
# current code
uv run python -m src.speed --uri mongodb://localhost:27017 --tag after
# this document
uv run python -m src.embeddings_api_report</pre>
<p>Prompt for a coding agent: "In the mongo-predict repo (github.com/voyage-ai/mongo-predict), load all MongoDB sample datasets into a local
MongoDB, run <code>python -m src.speed</code> for the baseline (tag <code>before</code>, from a git worktree of the package before it moved here, at
commit 55b3540 with this repo's <code>src/</code> copied in) and for the current code (tag <code>after</code>), then render
<code>reports/00_embeddings_api.html</code> with <code>python -m src.embeddings_api_report</code>. Tasks are defined in
<code>src/tasks.py</code>; each (task, mode) pair runs in a fresh process via <code>src/_speed_one.py</code>."</p>
</details>
<footer>mongo-predict reports. Result files: <code>results/speed_before.json</code>, <code>speed_after.json</code>.</footer>
</body></html>
"""
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text(page)
    print(f"wrote {DOC} ({len(page)} chars)")


if __name__ == "__main__":
    main()
