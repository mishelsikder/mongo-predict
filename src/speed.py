"""Wrapper speed benchmark over the MongoDB sample datasets.

For every task in ``tasks.py`` and every mode (in_memory, streaming) a fresh process runs the
full workflow with the hashing embedder and reports per stage timings and peak RSS. Results
land in ``results/speed_<tag>.json`` and a markdown table is printed.

    uv run python -m src.speed --uri mongodb://localhost:27017 --tag v0.3.0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

from src.tasks import TASKS

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root


def run_one(task, mode, args) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "src._speed_one",
        "--task",
        task.name,
        "--mode",
        mode,
        "--uri",
        args.uri,
        "--batch-size",
        str(args.batch_size),
        "--embed",
        args.embed,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        return {
            "task": task.name,
            "mode": mode,
            "error": proc.stderr.strip().splitlines()[-1][:300],
        }
    return json.loads(proc.stdout.strip().splitlines()[-1])


def table(rows: list[dict]) -> str:
    cols = [
        "task",
        "mode",
        "n_train",
        "avg_doc_bytes",
        "t_load_split",
        "t_fetch",
        "t_json",
        "t_tabular",
        "t_embed",
        "t_fit_total",
        "t_evaluate_test",
        "wrapper_docs_per_s",
        "rss_peak_mb",
    ]
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|"
    lines = [head]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['task']} | {r['mode']} | error: {r['error']} |")
            continue
        vals = [str(r["task"]), r["mode"], str(r["n_train"]), str(r["avg_doc_bytes"])]
        vals += [f"{r[c]:.2f}" for c in cols[4:11]]
        vals += [f"{r['wrapper_docs_per_s']:.0f}", f"{r['rss_peak_mb']:.0f}"]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--uri", default="mongodb://localhost:27017")
    p.add_argument("--tag", required=True, help="label for the results file, e.g. a version")
    p.add_argument("--tasks", default=None, help="comma separated task names (default: all)")
    p.add_argument("--modes", default="in_memory,streaming")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--embed", choices=["full", "none"], default="full")
    args = p.parse_args()
    tasks = [t for t in TASKS if not args.tasks or t.name in args.tasks.split(",")]
    rows = []
    for task in tasks:
        for mode in args.modes.split(","):
            r = run_one(task, mode, args)
            rows.append(r)
            print(json.dumps(r, default=str), flush=True)
    out = ROOT / "results" / f"speed_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=1, default=str))
    print(f"\nwrote {out}\n")
    print(table(rows))


if __name__ == "__main__":
    main()
