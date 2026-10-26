"""The ``$predict`` aggregation stage, run on the client, modelled on Atlas ``$vectorSearch``.

``$vectorSearch`` reads a search index and hands scored documents to the rest of the pipeline;
``$predict`` reads a prediction model (see :mod:`pymongopredict.predictions_api.models`) and
hands documents with predictions to the rest of the pipeline. Like ``$vectorSearch`` it must be
the first stage, and later stages read its output through ``$meta``:

* ``{"$meta": "prediction"}``: the predicted class (as first stored in the training documents)
  or value.
* ``{"$meta": "predictScore"}``: the probability of the predicted class.
* ``{"$meta": "predictProbabilities"}``: ``{str(class): probability}`` for every class.

The two probability keys are set only when the estimator has ``predict_proba`` (not for
regression or ``"ridge"``).

Main objects:

* :func:`aggregate`: ``collection.aggregate`` that understands a leading ``$predict`` stage.
* :class:`PredictCursor`: the materialised result, with timing ``stats``.
* :func:`enable` / :func:`disable`: patch ``pymongo.collection.Collection`` so that
  ``coll.aggregate(pipeline)`` understands ``$predict`` and the model helpers are methods.

How a pipeline runs:

1. The documents selected by ``filter`` and ``limit`` are fetched, projected to the fields the
   model reads.
2. They are featurized and predicted on the client in batches, and only
   ``{_id, prediction, score, probabilities}`` rows are uploaded to a temporary collection.
3. The rest of the pipeline runs on the server on that temporary collection, joined back to the
   full documents on ``_id``, with ``$meta`` references rewritten to a hidden helper field
   (:data:`META_FIELD`) that is removed before the output (or before a final ``$merge`` /
   ``$out``). So ``$match``, ``$sort``, ``$group``, ``$facet``, ``$lookup``, ``$merge`` and
   ``$out`` run as usual, and ``$merge`` can write the predictions back into the collection.
4. The temporary collection is dropped, even on errors.

Differences from a server stage: the documents travel to the client and the model runs there;
the result is materialised before it is returned; ``$predict`` cannot appear inside ``$lookup``
/ ``$unionWith`` / ``$facet`` sub-pipelines; ``$meta`` is not available inside ``$lookup`` /
``$unionWith`` sub-pipelines or in ``$merge``'s ``let`` / ``whenMatched``; and it cannot run in
a multi-document transaction.

Examples
--------
Usage:

    from pymongopredict import predictions_api

    predictions_api.enable()
    cursor = coll.aggregate([
        {"$predict": {"model": "room_type", "filter": {"address.market": "Porto"}}},
        {"$sort": {"score": {"$meta": "predictScore"}}},
        {"$project": {"name": 1, "predicted": {"$meta": "prediction"}}},
    ])
    cursor.stats   # {"nDocuments": ..., "fetchSeconds": ..., ...}

    # without enable():
    predictions_api.aggregate(coll, [{"$predict": {"model": "room_type"}}])

Requirements
------------
* MongoDB 5.0 or later (``$unsetField``, ``let`` in ``find``), checked before each run
  (``RuntimeError`` otherwise); 5.1 or later when the source collection is sharded (``$lookup``
  from a sharded collection). Tested on MongoDB 5.0, 6.0, 7.0 and 8.0.
* Privileges on the source's database: ``find``, ``listCollections`` (for the source's
  default collation), ``createCollection``, ``insert`` and ``dropCollection`` (the temporary
  collection), plus ``find`` / ``update`` on ``__mongopredict_models`` to train models. A user
  that cannot drop collections still gets the result, with a warning naming what is left.
* Replica sets: the temporary collection and the model records are read on the primary, so a
  ``secondary`` read preference on the client does not miss rows written a moment before; the
  source collection's codec options carry over to the temporary collections.
* A killed process can leave ``__mongopredict_tmp_<time>_<id>`` collections behind; remove them
  with :func:`cleanup_temporary_collections`.
"""

from __future__ import annotations

import datetime as dt
import itertools
import numbers
import time
import uuid
import warnings
from collections.abc import Mapping

import bson
from pymongo import ReadPreference, WriteConcern
from pymongo.errors import CursorNotFound, OperationFailure
from pymongo.read_concern import ReadConcern

from ..embeddings_api.documents import _id_key, _iter_batches, _with_dict_documents
from .models import (
    _check_server_version,
    _class_keys,
    _load_prediction_model,
    create_prediction_model,
    drop_prediction_model,
    list_prediction_models,
    update_prediction_model,
)

STAGE = "$predict"
"""Name of the aggregation stage."""
TMP_PREFIX = "__mongopredict_tmp_"
"""Prefix of the temporary collections that hold the uploaded predictions during a pipeline."""
META_FIELD = "__mongopredict"
"""Helper field that carries the predictions through the pipeline; reserved in documents."""
META_KEYS = {
    "prediction": "prediction",
    "predictScore": "score",
    "predictProbabilities": "probabilities",
}
"""``$meta`` keyword -> subfield of :data:`META_FIELD` that holds it."""
#: a batch of fetched documents is also cut at this many bytes (large documents)
MAX_BATCH_BYTES = 64 * 1024 * 1024
#: up to this many predicted documents, the join on a view or time-series collection starts with
#: an ``_id`` ``$in`` match; beyond it, the whole source is joined (a larger list could exceed
#: the 16 MB command size)
MAX_JOIN_IDS = 100_000
# ... and at most this many bytes of BSON, so the join command stays under the 16 MB limit
MAX_JOIN_ID_BYTES = 8 * 1024 * 1024
STAGE_KEYS = {"model", "filter", "limit", "batchSize"}
"""Keys a ``$predict`` stage may contain; any other key is rejected."""


def _has_stage(node) -> bool:
    """Whether ``$predict`` appears anywhere in ``node``, to refuse it outside the first stage."""
    if isinstance(node, dict):
        return STAGE in node or any(_has_stage(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_has_stage(v) for v in node)
    return False


# operators whose "sortBy" is a sort spec (a field of that name elsewhere is just a field)
SORT_BY_PARENTS = {
    "$setWindowFields",
    "$top",
    "$bottom",
    "$topN",
    "$bottomN",
    "$sortArray",
    "$densify",
    "$fill",
}


# $$ROOT / $$CURRENT without the helper field, so it never reaches nested output ($push, ...)
def _hidden(var: str) -> dict:
    """``var`` without the helper field (``$let`` can rebind CURRENT to a non-document)."""
    hidden = {"$unsetField": {"field": META_FIELD, "input": var}}
    if var == "$$ROOT":
        return hidden
    return {"$cond": [{"$eq": [{"$type": var}, "object"]}, hidden, var]}


MARKER = "__mongopredict_marker"  # inside the helper subdocument, so $redact can keep it


def _is_meta(node) -> bool:
    """Whether ``node`` is exactly a ``{"$meta": ...}`` reference to a ``$predict`` keyword."""
    return isinstance(node, dict) and len(node) == 1 and node.get("$meta") in META_KEYS


def _uses_meta(node) -> bool:
    """Whether a ``$predict`` ``$meta`` reference appears anywhere in ``node``."""
    if _is_meta(node):
        return True
    if isinstance(node, dict):
        return any(_uses_meta(v) for v in node.values())
    if isinstance(node, list):
        return any(_uses_meta(v) for v in node)
    return False


def _excludes(v) -> bool:
    """A projection value that excludes: 0 / False, or a sub-document of exclusions."""
    if isinstance(v, (bool, int, float)) and not isinstance(v, str):
        return v in (0, False)
    if isinstance(v, dict) and v and not any(str(k).startswith("$") for k in v):
        return all(_excludes(x) for x in v.values())
    return False  # 1 / True, a field path, an expression


def _is_inclusion(spec: dict) -> bool:
    """Whether a ``$project`` spec is an inclusion projection (``{"_id": 1}`` alone is one).

    Only an inclusion projection needs the helper field added back; an exclusion keeps it.
    """
    fields = {k: v for k, v in spec.items() if k != "_id"}
    if fields:
        return any(not _excludes(v) for v in fields.values())
    return "_id" in spec and not _excludes(spec["_id"])


def _keep_meta(expr):
    """A new root document that still carries the helper field (metadata survives
    ``$replaceRoot`` / ``$replaceWith`` on the server). A new root that is not a document still
    fails, as it does on the server: ``$mergeObjects`` would turn a missing one into ``{}``."""
    # $concat of a string with a non-string fails: an error for any root that is not a document
    checked = {
        "$cond": [
            {"$eq": [{"$type": expr}, "object"]},
            expr,
            {"$concat": ["newRoot must be a document, not ", {"$type": expr}]},
        ]
    }
    return {"$mergeObjects": [checked, {META_FIELD: f"${META_FIELD}"}]}


def _rewrite_match(node):
    """In ``$match`` only ``$expr`` holds expressions; everything else is query values."""
    if isinstance(node, dict):
        return {
            k: _rewrite(v)
            if k == "$expr"
            else _rewrite_match(v)
            if k in ("$and", "$or", "$nor")
            else v
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_match(v) for v in node]
    return node


def _rewrite_sort(spec: dict) -> dict:
    """A sort spec: a ``$meta`` key sorts by the helper field, descending (as ``textScore``)."""
    return {
        (f"{META_FIELD}.{META_KEYS[s['$meta']]}" if _is_meta(s) else f): (-1 if _is_meta(s) else s)
        for f, s in spec.items()
    }


def _rewrite(node, hide_root: bool = True):
    """Rewrite the stages after ``$predict`` to read the helper field instead of ``$meta``.

    * ``{"$meta": "prediction" | "predictScore" | "predictProbabilities"}`` becomes a field path;
      in a sort spec (``$sort``, ``sortBy``) a ``$meta`` key sorts descending, as
      ``{"$meta": "textScore"}`` does on the server.
    * An inclusion ``$project`` and ``$replaceRoot`` / ``$replaceWith`` keep the helper field
      (server-side metadata survives them too).
    * ``$$ROOT`` / ``$$CURRENT`` in other expressions and each ``$facet`` output leave the helper
      field out; ``$literal`` values and ``$match`` query values are not touched.
    * ``$redact`` always keeps the helper subdocument (recognised by :data:`MARKER`).
    * ``$lookup`` / ``$unionWith`` sub-pipelines and ``$merge``'s ``whenMatched`` pipeline read
      other documents and are left alone (their ``let`` variables are rewritten); ``$meta``
      inside them, or in ``$merge``'s ``let``, raises ``ValueError``.
    """
    if isinstance(node, list):
        return [_rewrite(v, hide_root) for v in node]
    if isinstance(node, str):
        return _hidden(node) if hide_root and node in ("$$ROOT", "$$CURRENT") else node
    if not isinstance(node, dict):
        return node
    if _is_meta(node):
        # from $$ROOT: the document's prediction even where $$CURRENT is a subdocument
        # ($redact with $$DESCEND) or rebound ($let)
        return f"$$ROOT.{META_FIELD}.{META_KEYS[node['$meta']]}"
    out = {}
    for k, v in node.items():
        if k == "$literal":
            out[k] = v
        elif k == "$sort" and isinstance(v, dict):
            out[k] = _rewrite_sort(v)
        elif k in SORT_BY_PARENTS and isinstance(v, dict):
            out[k] = {
                kk: _rewrite_sort(vv)
                if kk == "sortBy" and isinstance(vv, dict)
                else _rewrite(vv, hide_root)
                for kk, vv in v.items()
            }
        elif k == "$match" and isinstance(v, dict):
            out[k] = _rewrite_match(v)
        elif k == "$facet" and isinstance(v, dict):
            # each facet is an output of its own: drop the helper field at its end
            out[k] = {
                name: _rewrite(list(sub)) + [{"$project": {META_FIELD: 0}}]
                for name, sub in v.items()
            }
        elif k == "$redact":
            # $redact also walks into the helper subdocument: always keep it
            keep = {"$eq": [f"$$CURRENT.{MARKER}", True]}
            out[k] = {"$cond": [keep, "$$KEEP", _rewrite(v, hide_root)]}
        elif k == "$merge" and isinstance(v, dict) and _uses_meta(v.get("let", {})):
            raise ValueError(
                f"{STAGE} metadata is not available in $merge's let (the helper field is removed "
                "before $merge); add it as a field with $addFields / $set instead"
            )
        elif k in ("$lookup", "$unionWith", "$merge") and isinstance(v, dict):
            inner = v.get("whenMatched") if k == "$merge" else v.get("pipeline")
            if _uses_meta(inner if isinstance(inner, list) else []):
                raise ValueError(
                    f"{STAGE} metadata is not available inside {k} sub-pipelines, which read "
                    "other documents; add it as a field first ($addFields) and pass it with let"
                )
            out[k] = {**v, **({"let": _rewrite(v["let"])} if "let" in v else {})}
        elif k == "$project" and isinstance(v, dict) and _is_inclusion(v):
            out[k] = {**_rewrite(v), META_FIELD: 1}
        elif k == "$replaceRoot" and isinstance(v, dict) and "newRoot" in v:
            out[k] = {**v, "newRoot": _keep_meta(_rewrite(v["newRoot"]))}
        elif k == "$replaceWith":
            out[k] = _keep_meta(_rewrite(v))
        else:
            out[k] = _rewrite(v, hide_root)
    return out


def _plain(v):
    """A numpy scalar as the Python value BSON can store."""
    return v.item() if hasattr(v, "item") else v


def _meta_used(node) -> set:
    """The ``$meta`` keywords a pipeline reads (``"prediction"`` is always computed)."""
    if _is_meta(node):
        return {node["$meta"]}
    if isinstance(node, dict):
        return set().union(*(_meta_used(v) for v in node.values())) if node else set()
    if isinstance(node, list):
        return set().union(*(_meta_used(v) for v in node)) if node else set()
    return set()


def _predict_rows(
    fz,
    est,
    docs: list[dict],
    wanted=("predictScore", "predictProbabilities"),
    values=None,
    keys=None,
):
    """One ``{"_id", "meta"}`` row per document for the temporary collection; ``meta`` becomes
    the helper field after the join. Scores and probabilities only when ``wanted``: with many
    classes the probabilities make each row large."""
    X = fz.transform(docs)
    pred = est.predict(X)
    need = {"predictScore", "predictProbabilities"} & set(wanted)
    proba = est.predict_proba(X) if need and hasattr(est, "predict_proba") else None
    if proba is not None and (keys is None or len(keys) != proba.shape[1]):
        plain = [_plain(c) for c in est.classes_]
        keys = _class_keys([values.get(c, c) for c in plain] if values else plain)
    rows = []
    for i, d in enumerate(docs):
        p = _plain(pred[i])
        meta = {MARKER: True, "prediction": values.get(p, p) if values else p}
        if proba is not None:
            meta["score"] = float(proba[i].max())
            if "predictProbabilities" in need:
                # string keys: BSON field names must be strings, whatever the class type
                meta["probabilities"] = dict(zip(keys, map(float, proba[i])))
        rows.append({"_id": d["_id"], "meta": meta})
    return rows


class PredictCursor:
    """The materialised result of a ``$predict`` pipeline.

    Behaves like a pymongo ``CommandCursor`` over a list: iterate it, or call ``next()``,
    ``try_next()``, ``to_list()``, ``close()``, check ``alive``, or use it as a context manager.

    Parameters
    ----------
    docs : list of dict
        The pipeline output.
    stats : dict, optional
        Timing statistics (default: empty).

    Attributes
    ----------
    stats : dict
        ``nDocuments``: number of documents predicted (after removing duplicates);
        ``fetchSeconds``: seconds spent fetching them; ``predictSeconds``: featurizing and
        predicting on the client; ``uploadSeconds``: uploading the predictions;
        ``serverSeconds``: running the rest of the pipeline on the server.
    """

    def __init__(self, docs: list[dict], stats: dict | None = None):
        self._docs, self._it, self.stats = docs, iter(docs), stats or {}

    def __iter__(self):
        return self

    def __next__(self):
        """The next document; raises ``StopIteration`` when none are left."""
        return next(self._it)

    next = __next__

    def try_next(self) -> dict | None:
        """The next document, or ``None`` when none are left."""
        return next(self._it, None)

    @property
    def alive(self) -> bool:
        """Whether documents are left (like pymongo's cursors)."""
        try:
            first = next(self._it)
        except StopIteration:
            return False
        # put the peeked document back in front
        self._it = itertools.chain([first], self._it)
        return True

    def to_list(self, length: int | None = None) -> list[dict]:
        """The remaining documents as a list (like pymongo's ``Cursor.to_list``).

        Parameters
        ----------
        length : int, optional
            Return at most this many documents (the rest stay in the cursor). Default: all.

        Raises
        ------
        ValueError
            If ``length`` is less than 1.
        """
        if length is not None and length < 1:
            raise ValueError("to_list() length must be greater than 0")
        return list(self._it) if length is None else [d for _, d in zip(range(length), self._it)]

    def close(self) -> None:
        """Discard the remaining documents."""
        self._it = iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def aggregate(collection, pipeline: list[dict], _aggregate=None, **kwargs):
    """``collection.aggregate`` that understands a leading ``$predict`` stage.

    The stage is ``{"$predict": {"model": <name>, "filter": <query>, "limit": <n>,
    "batchSize": <n>}}``:

    * ``model`` (required): name of a stored prediction model of ``collection`` with a
      queryable version (see :func:`pymongopredict.predictions_api.load_prediction_model`).
    * ``filter``: any query selecting the documents to predict (default: all).
    * ``limit``: positive integer, the most documents to predict (default: every match).
    * ``batchSize``: positive integer, how many documents are featurized and uploaded at a
      time (default 512).

    Later stages read the predictions with ``{"$meta": "prediction"}``,
    ``{"$meta": "predictScore"}`` and ``{"$meta": "predictProbabilities"}``; see the module
    docstring for how the pipeline runs. Pipelines without ``$predict`` go straight to the
    server unchanged.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        Source collection.
    pipeline : list of dict
        Aggregation pipeline; ``$predict``, if present, must be its first stage and nowhere else.
    _aggregate : callable, optional
        Internal: ``(collection, pipeline, **kwargs) -> cursor`` that runs a plain pipeline;
        :func:`enable` passes the unpatched ``Collection.aggregate`` here.
    **kwargs
        ``aggregate`` options, handled as on the server where possible:

        * ``session``: used for every operation (the query, creating, filling and dropping the
          temporary collection, the server pipeline). A session in a transaction, or a snapshot
          session, is refused.
        * ``hint``: applies only to the query on the source collection.
        * ``collation``: applies to the source query and to the user's stages. The ``_id``
          join then runs separately with the source's default collation (into a second
          temporary collection), so a case-insensitive collation cannot join ``"a"`` to
          ``"A"``. Without it, the temporary collection gets the source's default collation,
          so string matching and sorting behave the same as on the source.
        * ``comment``, ``maxTimeMS``: passed to the source query and to the server pipeline
          (``comment`` is dropped from the query on backends that do not support it).
        * ``let``: passed to the server pipeline, and to the source query when the stage has
          a ``filter`` (for ``$$`` variables in its ``$expr``).
        * anything else: passed to the server pipeline.

    Returns
    -------
    PredictCursor or cursor
        A :class:`PredictCursor` for a ``$predict`` pipeline; otherwise whatever
        ``collection.aggregate`` returns.

    Raises
    ------
    ValueError
        If ``$predict`` is not only the first stage, the stage has other fields beside it,
        ``model`` is missing, the spec has unknown keys, ``limit`` / ``batchSize`` are not
        positive integers, the model does not exist or is not queryable, the session is in a
        transaction or is a snapshot session, a document contains the reserved
        :data:`META_FIELD`, a view or time-series source returns an ``_id`` twice, or ``$meta``
        is used where it is not available (``$lookup`` / ``$unionWith`` sub-pipelines,
        ``$merge``'s ``let`` / ``whenMatched``).
    RuntimeError
        If the temporary collection lost rows while the call ran (dropped by
        :func:`cleanup_temporary_collections` with a short ``older_than``).

    Examples
    --------
    Usage:

        aggregate(coll, [
            {"$predict": {"model": "imdb_rating", "filter": {"year": {"$gte": 2010}}}},
            {"$sort": {"p": {"$meta": "prediction"}}},   # a $meta sort key sorts descending
            {"$limit": 10},
        ], maxTimeMS=60_000).to_list()
    """
    run = _aggregate or (lambda coll, p, **kw: coll.aggregate(p, **kw))
    if isinstance(pipeline, (Mapping, str, bytes)):
        raise ValueError(f"a pipeline must be a list of stages, got {pipeline!r}")
    pipeline = list(pipeline)
    if not _has_stage(pipeline):
        return run(collection, pipeline, **kwargs)
    if (
        not pipeline
        or STAGE not in pipeline[0]
        or _has_stage(pipeline[0][STAGE])
        or _has_stage(pipeline[1:])
    ):
        raise ValueError(f"{STAGE} is only valid as the first stage in a pipeline")
    if len(pipeline[0]) != 1:
        raise ValueError(
            f"a pipeline stage specification must have exactly one field: {pipeline[0]}"
        )
    spec = pipeline[0][STAGE]
    if not isinstance(spec, dict) or "model" not in spec:
        raise ValueError(f"{STAGE} requires a 'model'")
    unknown = set(spec) - STAGE_KEYS
    if unknown:
        raise ValueError(f"{STAGE}: unknown fields {sorted(unknown)}")
    limit, batch_size = spec.get("limit"), spec.get("batchSize", 512)
    for key, val in (("limit", limit), ("batchSize", batch_size)):
        if val is not None and (
            isinstance(val, bool) or not isinstance(val, numbers.Integral) or val < 1
        ):
            raise ValueError(f"{STAGE}: {key} must be a positive integer, got {val!r}")
        if val is not None and val > 2**63 - 1:
            raise ValueError(f"{STAGE}: {key} must fit in a 64-bit integer, got {val!r}")
    if spec.get("filter") is not None and not isinstance(spec["filter"], Mapping):
        raise ValueError(f"{STAGE}: filter must be a document, got {spec['filter']!r}")
    _check_server_version(collection)
    fz, est, rec = _load_prediction_model(collection, spec["model"])
    values = None  # classes as first stored (labels are text to the estimator)
    if rec.get("classValues") is not None and hasattr(est, "classes_"):
        values = dict(zip((_plain(c) for c in est.classes_), rec["classValues"]))
    keys = rec.get("classes")  # predictProbabilities keys: str() of each class value

    # hint and collation apply to the source query; the session to every operation
    session = kwargs.get("session")
    if session is not None and getattr(session, "in_transaction", False):
        raise ValueError(
            f"{STAGE} cannot run inside a transaction (it creates and drops a collection)"
        )
    if session is not None and getattr(getattr(session, "options", None), "snapshot", False):
        raise ValueError(f"{STAGE} cannot run in a snapshot session (it writes a collection)")
    on_session = {"session": session} if session is not None else {}
    find_kw = {k: kwargs[k] for k in ("collation", "comment") if kwargs.get(k) is not None}
    if kwargs.get("maxTimeMS") is not None:
        find_kw["max_time_ms"] = kwargs["maxTimeMS"]
    if kwargs.get("let") is not None and spec.get("filter"):  # variables for the filter's $expr
        find_kw["let"] = kwargs["let"]
    if kwargs.get("hint") is not None:
        # popped: the index named by the hint belongs to the source, not the temporary collection
        find_kw["hint"] = kwargs.pop("hint")
    projection = fz._projection(None)  # the fields the model reads (None: whole documents)
    if projection is not None:
        projection = {**projection, META_FIELD: 1}  # to refuse documents with the reserved field

    source = _with_dict_documents(collection)  # the featurizer reads documents as dicts

    def find(**extra):
        """The source query; ``limit=0`` means no limit in pymongo."""
        # one server round trip per prediction batch: the cursor is never idle for longer than
        # one batch takes to predict (the server reaps idle cursors with their session)
        return source.find(
            spec.get("filter") or {},
            projection,
            limit=int(limit or 0),
            batch_size=int(batch_size),
            **extra,
            **on_session,
        )

    try:
        cursor = find(**find_kw)
    except Exception as e:  # noqa: BLE001 (a backend without find comments, e.g. mongomock)
        if "comment" not in find_kw or "comment" not in str(e):
            raise
        cursor = find(**{k: v for k, v in find_kw.items() if k != "comment"})
    db = collection.database
    # the creation time in the name lets cleanup_temporary_collections() find leftovers of a
    # killed process
    name = f"{TMP_PREFIX}{int(time.time())}_{uuid.uuid4().hex[:16]}"
    joined = f"{name}_joined"
    options = _source_options(collection)
    collation = _collation_of(options)
    # views and time-series collections have no _id index to look documents up by: the join
    # then runs from the source into the temporary collection (whose _id is indexed)
    reverse = bool(options.get("viewOn") or options.get("timeseries"))
    created = []  # what to drop afterwards (dropping what was never created is needless work)
    try:
        # the rest of the pipeline runs on the temporary collection: give it the source's
        # default collation, so string comparisons ($match, $sort, $group) behave the same
        if collation and not reverse:  # (reversed, the user's stages run on the source)
            db.create_collection(name, collation=collation, **on_session)
        created.append(name)
        tmp = _work_collection(collection, name)
        steps = ("fetchSeconds", "predictSeconds", "uploadSeconds", "serverSeconds")
        stats = dict.fromkeys(steps, 0.0)
        stats["nDocuments"] = 0
        ids: list | None = []  # predicted _ids, for the reversed join (None: join everything)
        ids_bytes = 0
        seen: set[bytes] = set()  # a scan can return a document twice if it moves meanwhile
        batches = _iter_batches(cursor, int(batch_size), max_bytes=MAX_BATCH_BYTES)
        wanted = _meta_used(
            pipeline[1:]
        )  # scores / probabilities only when the pipeline reads them
        while True:
            t = time.perf_counter()
            try:
                docs = next(batches, None)
            except CursorNotFound as e:
                raise CursorNotFound(
                    f"the server closed the source cursor of {STAGE} while documents were being "
                    f"predicted ({e}); use a smaller batchSize so the cursor is not idle for long",
                    e.code,
                    e.details,
                ) from e
            except OperationFailure as e:
                if e.code == 167 and kwargs.get("collation"):  # OptionNotSupportedOnView
                    raise ValueError(
                        f"the server cannot apply collation= to {collection.name!r}, a view or a "
                        "time-series collection (MongoDB 5.0 treats time-series collections as "
                        "views; 6.0 and later accept a collation there); run without collation= "
                        "or upgrade the server"
                    ) from e
                raise
            stats["fetchSeconds"] += time.perf_counter() - t
            if docs is None:
                break
            if any("_id" not in d for d in docs):
                raise ValueError(
                    f"{STAGE} needs the documents' _id to join the predictions back (is the "
                    "source a view that leaves _id out?)"
                )
            if any(META_FIELD in d for d in docs):
                raise ValueError(f"{META_FIELD!r} is a reserved field name and is in the documents")
            fresh = []
            for d in docs:
                key = _id_key(d["_id"])
                if key in seen:
                    if reverse:  # e.g. a view that $unwinds: predictions would be misjoined
                        raise ValueError(
                            f"{STAGE} needs unique _id values; the source returns {d['_id']!r} "
                            "more than once"
                        )
                    continue  # a collection scan returns a document again if it moved
                seen.add(key)
                fresh.append(d)
                if reverse and ids is not None:
                    ids_bytes += len(bson.encode({"": d["_id"]}))
                    ids.append(d["_id"])
                    if len(ids) > MAX_JOIN_IDS or ids_bytes > MAX_JOIN_ID_BYTES:
                        ids = None  # too many to list: join the whole source instead
            docs = fresh
            if not docs:
                continue
            t = time.perf_counter()
            rows = _predict_rows(fz, est, docs, wanted, values, keys)
            stats["predictSeconds"] += time.perf_counter() - t
            t = time.perf_counter()
            tmp.insert_many(rows, ordered=True, **on_session)
            stats["uploadSeconds"] += time.perf_counter() - t
            stats["nDocuments"] += len(docs)
        if stats["nDocuments"] and tmp.estimated_document_count() != stats["nDocuments"]:
            raise RuntimeError(
                f"the temporary collection {db.name}.{name} lost predictions while {STAGE} ran "
                "(was it dropped by cleanup_temporary_collections with a short older_than?)"
            )
        rest = _rewrite(pipeline[1:])
        # remove the helper field before the output, or before a final $merge / $out writes it
        tail = len(rest) - 1 if rest and ({"$merge", "$out"} & set(rest[-1])) else len(rest)
        rest = rest[:tail] + [{"$project": {META_FIELD: 0}}] + rest[tail:]
        # full source documents, each with its predictions in the helper field; documents
        # deleted since the fetch find no match and drop out at $unwind
        join = [
            {
                "$lookup": {
                    "from": collection.name,
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "__doc",
                }
            },
            {"$unwind": "$__doc"},
            {"$addFields": {f"__doc.{META_FIELD}": "$meta"}},
            {"$replaceRoot": {"newRoot": "$__doc"}},
        ]
        base = tmp
        if reverse:
            helper = f"{META_FIELD}_join"
            # only the predicted documents: the filter is not run again (it may use $rand,
            # $$NOW or a collation the join does not have); the $lookup + $unwind is an inner
            # join either way, the $match only spares looking at the rest of the source
            join = [{"$match": {"_id": {"$in": ids}}}] if ids is not None else []
            join += [
                {
                    "$lookup": {
                        "from": name,
                        "localField": "_id",
                        "foreignField": "_id",
                        "as": helper,
                    }
                },
                {"$unwind": f"${helper}"},
                *(_exact_id_match(helper) if collation else []),
                {"$addFields": {META_FIELD: f"${helper}.meta"}},
                {"$project": {helper: 0}},
            ]
            # on the primary: the temporary collection was just written there
            base = collection.with_options(read_preference=ReadPreference.PRIMARY)
        t = time.perf_counter()
        if kwargs.get("collation") is None:
            out = list(run(base, join + rest, **kwargs))
        else:
            # the _id join must use the source's own collation (a case-insensitive one would join
            # "a" to "A"): join first, then run the user's stages with the user's collation
            created.append(joined)
            try:
                run(
                    base,
                    join + [{"$out": joined}],
                    **{k: v for k, v in kwargs.items() if k != "collation"},
                )
            except OperationFailure as e:
                if "BSONObj size" not in str(e) and e.code not in (10334, 17419):
                    raise
                raise ValueError(
                    f"with collation=, {STAGE} first stores each document with its predictions, "
                    f"and one is over 16 MB ({e}); run without collation= or read fewer $meta "
                    "fields (predictProbabilities with many classes is large)"
                ) from e
            out = list(run(_work_collection(collection, joined), rest, **kwargs))
        stats["serverSeconds"] = time.perf_counter() - t
        return PredictCursor(out, stats)
    finally:
        getattr(cursor, "close", lambda: None)()  # (the query may have stopped early on an error)
        for n in created:
            try:
                db.drop_collection(n, **on_session)
            except Exception as e:  # noqa: BLE001 (no privilege, lost connection: keep the result)
                warnings.warn(
                    f"could not drop the temporary collection {db.name}.{n} ({e}); remove it "
                    "with cleanup_temporary_collections() or grant dropCollection",
                    stacklevel=2,
                )


def _exact_id_match(helper: str) -> list[dict]:
    """Keep joined rows whose string ``_id`` is the same bytes: under a case-insensitive source
    collation ``$lookup`` joins "a" to "A" as well (byte functions ignore the collation)."""
    mine, theirs = "$_id", f"${helper}._id"
    return [
        {
            "$match": {
                "$expr": {
                    "$or": [
                        {"$ne": [{"$type": mine}, "string"]},
                        {
                            "$and": [
                                {"$eq": [{"$type": theirs}, "string"]},
                                {"$eq": [{"$strLenBytes": mine}, {"$strLenBytes": theirs}]},
                                {"$eq": [{"$indexOfBytes": [mine, theirs]}, 0]},
                            ]
                        },
                    ]
                }
            }
        }
    ]


def _work_collection(collection, name: str):
    """A temporary collection read on the primary (it was just written there: a secondary may
    not have it yet), with the source's codec options and an acknowledged write concern."""
    write_concern = collection.write_concern
    if not write_concern.acknowledged:
        write_concern = WriteConcern(w=1)
    return collection.database.get_collection(
        name,
        codec_options=collection.codec_options,
        read_preference=ReadPreference.PRIMARY,
        write_concern=write_concern,
        # $out refuses a linearizable read concern
        read_concern=ReadConcern("majority")
        if collection.read_concern.level == "linearizable"
        else collection.read_concern,
    )


def cleanup_temporary_collections(database, older_than: dt.timedelta = dt.timedelta(hours=24)):
    """Drop the temporary collections that ``$predict`` calls left behind (a killed process,
    or a user without the ``dropCollection`` privilege). Returns the dropped names.

    Collections created less than ``older_than`` ago may belong to a running call and are kept;
    names without a creation time (older versions) count as old. A call whose collection is
    dropped while it runs raises ``RuntimeError`` rather than returning partial results, so keep
    ``older_than`` above the longest ``$predict`` call.
    """
    cutoff = time.time() - older_than.total_seconds()
    dropped = []
    for n in database.list_collection_names(filter={"name": {"$regex": f"^{TMP_PREFIX}"}}):
        head = n[len(TMP_PREFIX) :].split("_", 1)[0]  # unix time of creation (10 digits)
        if head.isdigit() and len(head) == 10 and int(head) > cutoff:
            continue
        database.drop_collection(n)
        dropped.append(n)
    return dropped


def _default_collation(collection) -> dict | None:
    """The source's default collation, for the temporary collection (``None`` if it has none)."""
    return _collation_of(_source_options(collection))


def _collation_of(options: dict) -> dict | None:
    """``version`` is left out: the server reports it, but it is not a setting to copy."""
    collation = options.get("collation")
    if collation:
        collation = {k: v for k, v in collation.items() if k != "version"}
    return collation or None


def _source_options(collection) -> dict:
    """The source's collection options (collation, ``viewOn``, ``timeseries``); ``{}`` when the
    backend has none."""
    try:
        return collection.options() or {}
    except OperationFailure as e:  # no listCollections privilege: string matching would differ
        raise PermissionError(
            f"{STAGE} reads the source collection's default collation (listCollections "
            f"privilege) to compare strings as the source does: {e}"
        ) from e
    except Exception:  # noqa: BLE001 (e.g. not supported by a mock)
        return {}


# ---------------------------------------------------------------------- pymongo integration
_patched: dict[type, dict] = {}  # class -> its own attributes before enable() (None: inherited)
_METHODS = (
    "create_prediction_model",
    "update_prediction_model",
    "list_prediction_models",
    "drop_prediction_model",
)
_AGGREGATE_ARGS = ("session", "let", "comment")  # pymongo's positional parameters after pipeline


def enable(*classes) -> None:
    """Let ``Collection.aggregate`` run ``$predict`` and add the model helpers as methods.

    After this, ``coll.aggregate(pipeline, ...)`` calls :func:`aggregate` (pymongo's positional
    ``session``, ``let`` and ``comment`` arguments are still accepted), and
    ``coll.create_prediction_model``, ``update_prediction_model``, ``list_prediction_models``
    and ``drop_prediction_model`` call the functions of the same name with ``coll`` first.
    Calling it again for a class already patched does nothing.

    Parameters
    ----------
    *classes : type
        Collection classes to patch (e.g. ``mongomock.Collection``). Default:
        ``pymongo.collection.Collection``.

    Examples
    --------
    Usage:

        predictions_api.enable()
        coll.create_prediction_model(PredictionModel({"target": "room_type"}, "room_type"))
        coll.aggregate([{"$predict": {"model": "room_type"}}]).to_list()
        predictions_api.disable()
    """
    if not classes:
        from pymongo.collection import Collection

        classes = (Collection,)
    for cls in classes:
        if cls in _patched:
            continue
        _patched[cls] = {m: cls.__dict__.get(m) for m in ("aggregate", *_METHODS)}
        original = cls.aggregate

        # _original is bound as a default, so each class keeps its own original method
        def patched(self, pipeline, *args, _original=original, **kwargs):
            if len(args) > len(_AGGREGATE_ARGS):
                raise TypeError(
                    f"aggregate() takes at most {len(_AGGREGATE_ARGS) + 1} positional arguments"
                )
            kwargs.update(zip(_AGGREGATE_ARGS, args))
            return aggregate(
                self, pipeline, _aggregate=lambda c, p, **kw: _original(c, p, **kw), **kwargs
            )

        cls.aggregate = patched
        cls.create_prediction_model = create_prediction_model
        cls.update_prediction_model = update_prediction_model
        cls.list_prediction_models = list_prediction_models
        cls.drop_prediction_model = drop_prediction_model


def disable() -> None:
    """Undo :func:`enable` for every patched class, restoring their original attributes.

    Attributes a class inherited before :func:`enable` are deleted from it again, so the
    inherited ones apply. Does nothing if nothing is patched.
    """
    for cls, own in reversed(list(_patched.items())):
        for attr, value in own.items():
            if value is None:
                if attr in cls.__dict__:
                    delattr(cls, attr)
            else:
                setattr(cls, attr, value)
        del _patched[cls]
