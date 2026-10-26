"""Embedding time of voyage-4-nano through the DocumentFeaturizer, on GPU and CPU.

For a few sample tasks of different document sizes, documents are featurized with the
featurizer defaults (the whole document as JSON, up to 8,000 characters, the target excluded)
and voyage-4-nano, without an embedding cache. A first call on 16 documents loads the model
(timed separately); then ``transform`` on ``--n`` further documents is timed.

    uv run python -m src.embedding_timing --device cuda --n 2000
    uv run python -m src.embedding_timing --device cpu --threads 16 --n 200
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from pymongo import MongoClient

from pymongopredict import DocumentFeaturizer
from src.tasks import TASKS

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TASKS = "airbnb_room_type,mflix_imdb_rating,tweets_user_lang"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--uri", default="mongodb://127.0.0.1:27117")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=16, help="torch threads on CPU")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    client = MongoClient(a.uri)
    tasks = {t.name: t for t in TASKS}
    rows = []
    for name in a.tasks.split(","):
        t = tasks[name]
        docs = list(client[t.database][t.collection].find(t.filter or {}).limit(a.n + 16))
        fz = DocumentFeaturizer(
            columns=t.columns,
            exclude=[t.target],
            embedding_model="voyage-4-nano",
            embedding_kwargs={"device": a.device},
            cache_dir=None,
        )
        t0 = time.perf_counter()
        fz.fit_transform(docs[:16])  # loads the model
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_load = time.perf_counter() - t0
        batch = docs[16:]
        texts = fz._texts(batch)  # the JSON each document is embedded as
        t0 = time.perf_counter()
        fz.transform(batch)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_embed = time.perf_counter() - t0
        row = {
            "task": name,
            "device": a.device,
            "threads": a.threads if a.device == "cpu" else None,
            "n": len(batch),
            "mean_json_chars": sum(map(len, texts)) / len(texts),
            "load_s": t_load,
            "embed_s": t_embed,
            "docs_per_s": len(batch) / t_embed,
            "s_per_1000": t_embed / len(batch) * 1000,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    tag = a.device.split(":")[0] + ("" if a.device != "cpu" else f"_{a.threads}t")
    out = pathlib.Path(a.out or ROOT / "results" / f"embedding_timing_{tag}.json")
    gpu = torch.cuda.get_device_name() if a.device.startswith("cuda") else None
    out.write_text(json.dumps({"device": a.device, "gpu": gpu, "tasks": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
