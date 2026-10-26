"""Prediction models: the stored, trained models that the ``$predict`` stage serves.

A prediction model plays the role that a search index plays for Atlas ``$vectorSearch``. It is
defined with a :class:`PredictionModel` (the counterpart of ``SearchIndexModel``), trained once,
stored in the database, and then referenced by name from a ``$predict`` stage (see
:mod:`pymongopredict.predictions_api.stage`).

Main objects:

* :class:`PredictionModel`: a validated ``definition`` plus a ``name``.
* :func:`create_prediction_model`, :func:`update_prediction_model`,
  :func:`list_prediction_models`, :func:`drop_prediction_model`: mirror
  ``Collection.create_search_index`` / ``update_search_index`` / ``list_search_indexes`` /
  ``drop_search_index``. After :func:`pymongopredict.predictions_api.enable` they are also
  methods of ``pymongo.collection.Collection``.
* :func:`_load_prediction_model`: returns the fitted featurizer and estimator, cached per process.

Training fits a :class:`pymongopredict.DocumentFeaturizer` and an sklearn estimator on the
documents that have the target (from the ``embeddings_api``), records holdout metrics, refits on
all of them and stores the pickled pair, with its status and metrics, in the
``__mongopredict_models`` collection (:data:`MODELS_COLLECTION`) of the same database. One record
per ``(collection, name)``, with ``_id`` ``"<collection>.<name>"``.

The stored model is a pickle, so only users who may run code on the client should be able to
write that collection; with :data:`MODEL_KEY_ENV` set, models are signed and unsigned ones are
refused.

Examples
--------
Usage:

    from pymongopredict.predictions_api import (
        PredictionModel, create_prediction_model, list_prediction_models,
    )

    create_prediction_model(coll, PredictionModel(
        {"target": "room_type", "columns": ["price", "bedrooms"], "testSize": 0.2},
        name="room_type",
    ))
    list_prediction_models(coll, "room_type")[0]["metrics"]   # {"accuracy": ..., "auroc": ...}
"""

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import hashlib
import hmac
import inspect
import numbers
import os
import pickle
import platform
import tempfile
import uuid
import warnings

import bson
import numpy as np
import pandas as pd
from bson.binary import Binary
from pymongo import ReadPreference, ReturnDocument, WriteConcern

from ..embeddings_api.dataset import MongoDataset, _label
from ..embeddings_api.documents import _id_key
from ..embeddings_api.features import DocumentFeaturizer, _on_documents

MODELS_COLLECTION = "__mongopredict_models"
"""Name of the collection, in the source collection's database, that stores model records."""
MAX_MODEL_BYTES = 15 * 1024 * 1024
# "auto" predicts numbers with fewer distinct values than this as categories (flags, star ratings)
MIN_REGRESSION_VALUES = 8
"""Largest pickled ``(featurizer, estimator)`` that may be stored (15 MB)."""
#: a build still running after this long is reported as ``stale`` (most likely killed)
STALE_BUILD = dt.timedelta(hours=24)
MAX_RECORD_BYTES = (
    16 * 1024 * 1024 - 16 * 1024
)  # MongoDB's document limit, with room for the command
"""Largest whole model record (16 MB BSON limit, less 16 KB for the update command around it)."""

# definition key -> DocumentFeaturizer argument
FEATURIZER_KEYS = {
    "columns": "columns",
    "embedFields": "embed_fields",
    "excludeFields": "exclude",
    "embeddingModel": "embedding_model",
    "embeddingKwargs": "embedding_kwargs",
    "maskTarget": "mask_target",
    "maxTokens": "max_tokens",
    "maxJsonChars": "max_json_chars",
}
"""Definition keys (camelCase, as in a server definition) mapped to ``DocumentFeaturizer`` args."""
DEFINITION_KEYS = set(FEATURIZER_KEYS) | {
    "target",
    "task",
    "estimator",
    "filter",
    "numTrainingDocuments",
    "testSize",
    "featurizerOptions",
}
"""Every key a :class:`PredictionModel` definition may contain; any other key is rejected."""

#: environment variable holding the key that signs stored models (HMAC-SHA256); when set, models
#: are signed when stored and a model without a valid signature is never unpickled
MODEL_KEY_ENV = "MONGOPREDICT_MODEL_KEY"
#: environment variable naming the embedding cache directory used when predicting (no cache when
#: unset: a serving process would otherwise grow a cache of every document it predicts)
PREDICT_CACHE_ENV = "MONGOPREDICT_PREDICT_CACHE_DIR"

# (database, model _id) -> (build id, featurizer, estimator); the build id is unique per training
# run, so a recreated model (or one with the same names on another server) is never served stale
_loaded: dict[tuple, tuple] = {}


class PredictionModel:
    """A prediction model definition, the counterpart of ``pymongo.operations.SearchIndexModel``.

    Only the arguments are checked here; the values are checked when the model is trained.

    Parameters
    ----------
    definition : dict
        How to train the model. All keys are optional except ``target``:

        * ``target``: dotted path of the field to predict.
        * ``columns``, ``embedFields``, ``excludeFields``, ``embeddingModel``,
          ``embeddingKwargs``, ``maskTarget``, ``maxTokens``, ``maxJsonChars``: passed to
          :class:`pymongopredict.DocumentFeaturizer` as ``columns``, ``embed_fields``,
          ``exclude``, ``embedding_model``, ``embedding_kwargs``, ``mask_target``,
          ``max_tokens`` and ``max_json_chars`` (``embedFields`` omitted means embed the whole
          document). ``featurizerOptions`` is a dict of any other ``DocumentFeaturizer``
          argument by its Python name; it overrides the keys above.
        * ``task``: ``"classification"``, ``"regression"`` or ``"auto"`` (default). ``"auto"``
          picks regression when every target is a number (or numeric text; empty strings are
          dropped) and there are at least ``MIN_REGRESSION_VALUES`` (8) distinct values;
          otherwise classification, so a 0/1 flag, 1-5 stars, booleans and text are categories.
        * ``estimator``: ``"auto"`` (default) or ``"xgboost"``: XGBoost with its default settings
          (:class:`_XGBLabelClassifier` for classification, ``xgboost.XGBRegressor`` for
          regression); ``"logistic"`` (classification only), ``"ridge"``
          (``RidgeClassifierCV`` / ``RidgeCV``), ``"hgb"`` (histogram gradient boosting), or
          an sklearn estimator object (usually passed as the ``estimator`` argument of
          :func:`create_prediction_model`; it is deep-copied before fitting).
        * ``filter``: query selecting the training documents. Documents without the target
          are always left out, as are documents whose target is null (and, for regression,
          not a number).
        * ``numTrainingDocuments``: positive integer cap on training documents (default: all).
        * ``testSize``: holdout fraction in ``[0, 1)``, or integer count of documents, used
          for the stored metrics (default 0.2; 0 skips them). The split is stratified for
          classification when every class has at least 2 documents. The stored model is
          refitted on all training documents.
    name : str, default "default"
        Model name, unique per collection. Must be a non-empty string without ``.`` or ``$``.
    type : str, default "prediction"
        Model type; only ``"prediction"`` is supported.

    Raises
    ------
    ValueError
        If ``name`` is invalid, ``type`` is not ``"prediction"``, ``target`` is missing, or
        ``definition`` has a key outside :data:`DEFINITION_KEYS`.

    Examples
    --------
    Usage:

        PredictionModel({"target": "imdb.rating", "embedFields": ["plot", "genres"]},
                        name="imdb_rating")
    """

    def __init__(self, definition: dict, name: str = "default", type: str = "prediction"):
        _check_name(name)
        if type != "prediction":
            raise ValueError(f"unsupported model type {type!r}")
        if not isinstance(definition, dict):
            raise ValueError(f"a model definition must be a document, got {definition!r}")
        if "target" not in definition:
            raise ValueError("a prediction model definition needs a 'target'")
        unknown = set(definition) - DEFINITION_KEYS
        if unknown:
            raise ValueError(f"unknown definition keys {sorted(unknown)}")
        target = definition["target"]
        parts = target.split(".") if isinstance(target, str) else [""]
        if not target or any(not p or p.startswith("$") for p in parts):
            raise ValueError(
                f"the target must be a dotted field path without '$' parts: {target!r}"
            )
        for key in ("columns", "excludeFields") + (
            ("embedFields",) if not isinstance(definition.get("embedFields"), bool) else ()
        ):
            paths = definition.get(key)
            if paths is not None and (
                isinstance(paths, (str, dict)) or not all(isinstance(c, str) and c for c in paths)
            ):
                raise ValueError(f"{key} must be a list of field paths, got {paths!r}")
        _check_no_api_key(definition)
        options = definition.get("featurizerOptions")
        if options is not None:
            if not isinstance(options, dict):
                raise ValueError(f"featurizerOptions must be a document, got {options!r}")
            known = set(inspect.signature(DocumentFeaturizer).parameters)
            if set(options) - known:
                raise ValueError(
                    f"unknown featurizerOptions {sorted(set(options) - known)} "
                    f"(DocumentFeaturizer arguments: {sorted(known)})"
                )
        if any(c == target or c.startswith(target + ".") for c in definition.get("columns") or []):
            raise ValueError(
                f"columns include the target {target!r}: it would leak into the features"
            )
        self.definition, self.name, self.type = definition, name, type

    @property
    def document(self) -> dict:
        """The model as a ``{"name", "type", "definition"}`` dict (``SearchIndexModel.document``).

        :func:`create_prediction_model` accepts this dict in place of the object.
        """
        return {"name": self.name, "type": self.type, "definition": self.definition}


# ---------------------------------------------------------------------- model management
def _models(collection):
    """The records collection, in the same database as ``collection``: read on the primary
    (a secondary may not have the latest record yet) and written with acknowledged writes."""
    return collection.database.get_collection(
        MODELS_COLLECTION, read_preference=ReadPreference.PRIMARY, write_concern=WriteConcern(w=1)
    )


MIN_SERVER_VERSION = (5, 0)
"""Oldest MongoDB server the predictions API supports: ``$predict`` uses ``$unsetField`` and ``let``
in ``find`` (MongoDB 5.0), and stored models must be served by the same servers."""


def _check_server_version(collection) -> None:
    """Raise ``RuntimeError`` when the server is older than :data:`MIN_SERVER_VERSION`.

    A server that does not report its version (a test double, a proxy) is not checked."""
    try:
        info = collection.database.client.server_info()
    except Exception:  # noqa: BLE001 (connection errors surface in the operation itself)
        return
    version = info.get("versionArray") or [
        int(x) for x in str(info.get("version", "")).split(".")[:2] if x.isdigit()
    ]
    if len(version) >= 2 and tuple(version[:2]) < MIN_SERVER_VERSION:
        raise RuntimeError(
            f"the predictions API needs MongoDB {MIN_SERVER_VERSION[0]}.{MIN_SERVER_VERSION[1]} or "
            f"later, but this server is MongoDB {info.get('version')}; upgrade the server (the "
            "embeddings API works with older servers)"
        )


def _check_no_api_key(definition: dict) -> None:
    """Refuse a Voyage API key anywhere in a definition: the definition (``latestDefinition``)
    and the pickled model are stored in the database, readable by anyone who can read
    ``__mongopredict_models``. The key belongs in ``VOYAGE_API_KEY`` where the model is used."""
    options = definition.get("featurizerOptions")
    options = options if isinstance(options, dict) else {}
    places = {
        "embeddingKwargs": definition.get("embeddingKwargs"),
        "featurizerOptions.embedding_kwargs": options.get("embedding_kwargs"),
    }
    for where, kwargs in places.items():
        if isinstance(kwargs, dict) and kwargs.get("api_key"):
            raise ValueError(
                f"do not put api_key in {where}: the definition is stored in the database. "
                "Set the VOYAGE_API_KEY environment variable where the model is fit and used"
            )
    for where, model in (
        ("embeddingModel", definition.get("embeddingModel")),
        ("featurizerOptions.embedding_model", options.get("embedding_model")),
    ):
        if getattr(model, "api_key", None):
            raise ValueError(
                f"the embedder in {where} holds an api_key, which would be stored with the "
                "model; create it without one and set VOYAGE_API_KEY instead"
            )


def _check_name(name) -> None:
    """Model names: non-empty strings without '.' (the record _id joins collection and name with
    one, so "a.b" on collection "c" would be model "b" of collection "c.a") or '$'."""
    if not isinstance(name, str) or not name or "." in name or "$" in name:
        raise ValueError(f"model names must be non-empty strings without '.' or '$': {name!r}")


def _model_id(collection, name: str) -> str:
    """Record ``_id``: model names are per collection, and the records share one collection."""
    _check_name(name)
    return f"{collection.name}.{name}"


#: settings of the default XGBoost estimator ("auto" / "xgboost"), on top of XGBoost's defaults
XGB_PARAMS: dict = {}
#: threads for the default XGBoost estimator when OMP_NUM_THREADS is not set: XGBoost otherwise
#: starts one per core, which on a large server makes small fits much slower
MAX_XGB_THREADS = 16


def _xgb_params() -> dict:
    """``XGB_PARAMS`` plus a thread count, unless OMP_NUM_THREADS or ``n_jobs`` sets one."""
    params = dict(XGB_PARAMS)
    if "n_jobs" not in params and not os.environ.get("OMP_NUM_THREADS"):
        params["n_jobs"] = max(1, min(MAX_XGB_THREADS, os.cpu_count() or 1))
    return params


class _XGBLabelClassifier:
    """``xgboost.XGBClassifier`` for any labels: XGBoost wants the classes as 0 .. k-1, while
    prediction models keep their labels (text, numbers as text, ...). The labels are encoded for
    the fit, and ``classes_``, ``predict`` and ``predict_proba`` are in terms of the labels.

    Parameters
    ----------
    **params
        Passed to ``xgboost.XGBClassifier`` (its defaults otherwise).
    """

    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier

        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        if len(self.classes_) < 2:
            raise ValueError(f"needs at least 2 classes, got {len(self.classes_)}")
        self.model_ = XGBClassifier(**self.params).fit(X, self.encoder_.transform(y))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        """Accuracy, as for scikit-learn classifiers (``GridSearchCV``, ``cross_val_score``)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


# models stored between commits 4d52722 and 961ff9a pickled the class under this name
XGBLabelClassifier = _XGBLabelClassifier


def _estimator(spec, task: str):
    """A fresh, unfitted estimator for ``spec``: a copy of an sklearn object, or one built from a
    string name. Raises ``ValueError`` for a name that does not fit ``task``."""
    if not isinstance(spec, str):
        if not hasattr(spec, "fit"):
            raise ValueError(f"the estimator needs a fit method, got {spec!r}")
        # a copy: the holdout and final fits must not share (or mutate) the user's object
        return copy.deepcopy(spec)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, RidgeCV

    def xgb_regressor():
        from xgboost import XGBRegressor

        return XGBRegressor(**_xgb_params())

    alphas = np.logspace(-3, 3, 13)
    table = {
        ("auto", "classification"): lambda: _XGBLabelClassifier(**_xgb_params()),
        ("xgboost", "classification"): lambda: _XGBLabelClassifier(**_xgb_params()),
        ("logistic", "classification"): lambda: LogisticRegression(max_iter=2000),
        ("ridge", "classification"): lambda: RidgeClassifierCV(alphas=alphas),
        ("hgb", "classification"): HistGradientBoostingClassifier,
        ("auto", "regression"): xgb_regressor,
        ("xgboost", "regression"): xgb_regressor,
        ("ridge", "regression"): lambda: RidgeCV(alphas=alphas),
        ("hgb", "regression"): HistGradientBoostingRegressor,
    }
    if (spec, task) not in table:
        raise ValueError(f"estimator {spec!r} is not available for {task}")
    return table[spec, task]()


def _metrics(task: str, est, X, y) -> dict:
    """Holdout metrics stored with the model: ``r2`` and ``mae`` for regression; ``accuracy`` and,
    when the estimator has ``predict_proba`` and AUROC is defined, ``auroc`` (one-vs-rest)."""
    from sklearn import metrics

    pred = est.predict(X)
    if task == "regression":
        return {
            "r2": float(metrics.r2_score(y, pred)),
            "mae": float(metrics.mean_absolute_error(y, pred)),
        }
    out = {"accuracy": float(metrics.accuracy_score(y, pred))}
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # (a class missing from the holdout: NaN, left out)
            try:
                if len(est.classes_) == 2:
                    auroc = metrics.roc_auc_score(y == est.classes_[1], proba[:, 1])
                else:
                    auroc = metrics.roc_auc_score(y, proba, multi_class="ovr", labels=est.classes_)
            except ValueError:
                auroc = float("nan")
        if np.isfinite(auroc):
            out["auroc"] = float(auroc)
    return out


def _as_numbers(y) -> np.ndarray | None:
    """The target as floats when every value is a number, or a string holding a number apart from
    empty ones (``""`` for a missing rating, which becomes NaN); ``None`` otherwise (booleans,
    categories, mixed values)."""
    if y.dtype.kind in "fiu":
        return y.astype(float)
    if y.dtype.kind != "O":
        return None
    text = pd.Series(y).astype(str).str.strip()
    # currency signs and thousands separators: "$1,100" is 1100. One sign at either end and the
    # spaces beside it go; plain string operations, as a regex here backtracks quadratically on
    # long runs of spaces in document values
    signs = ("$", "€", "£", "¥")
    lead = text.str.startswith(signs)
    text = text.where(~lead, text.str[1:].str.lstrip())
    trail = text.str.endswith(signs)
    text = text.where(~trail, text.str[:-1].str.rstrip())
    grouped = text.str.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?")
    text = text.where(~grouped, text.str.replace(",", "", regex=False))
    values = pd.to_numeric(text.where(text != ""), errors="coerce")
    if values.notna().sum() == 0 or (values.isna() & (text != "")).any():
        return None
    return values.to_numpy(float)


def _auto_task(y) -> str:
    """``"regression"`` for numbers with at least ``MIN_REGRESSION_VALUES`` distinct values;
    ``"classification"`` otherwise, so a 0 / 1 flag or a 1 to 5 star rating stored as numbers is
    predicted as a category."""
    if y.dtype.kind == "f" and len(np.unique(y)) >= MIN_REGRESSION_VALUES:
        return "regression"
    return "classification"


class _Progress:
    """Training progress: one numbered bar per step. Embedding steps count documents; fitting
    steps show their time when done. ``enabled=None`` lets tqdm decide (shown on a terminal or
    in a notebook, hidden when stderr is redirected)."""

    def __init__(self, n_steps: int, enabled: bool | None):
        self.n_steps, self.i = n_steps, 0
        self.disable = None if enabled is None else not enabled

    @contextlib.contextmanager
    def step(self, label: str, n_documents: int | None = None):
        from tqdm.auto import tqdm

        self.i += 1
        desc = f"[{self.i}/{self.n_steps}] {label}"
        if n_documents is None:
            bar = tqdm(total=1, desc=desc, bar_format="{desc}: {elapsed}", disable=self.disable)
        else:
            bar = tqdm(total=n_documents, desc=desc, unit=" docs", disable=self.disable)
        token = _on_documents.set(bar.update) if n_documents is not None else None
        try:
            yield
            if n_documents is None:
                bar.update(1)
        finally:
            if token is not None:
                _on_documents.reset(token)
            bar.close()


def _fit(collection, definition: dict, estimator=None, progress: bool | None = None):
    """Validate the training settings, then fit with a holdout (for metrics) and on everything.

    Documents of a collection are streamed (only ``_id`` and the target held). Views and
    time-series collections are read once into memory: they have no ``_id`` index to fetch
    batches by. If documents are deleted while a streamed fit runs, it is redone in memory.

    Returns ``(featurizer, estimator, task, n_training_documents, metrics, class_values)``.
    """
    from .stage import _source_options

    options = _source_options(collection)
    streaming = not (options.get("viewOn") or options.get("timeseries"))
    try:
        return _fit_once(collection, definition, estimator, streaming, progress)
    except KeyError as e:
        if not streaming or "disappeared" not in str(e):
            raise
        warnings.warn(
            f"documents were deleted during training ({e}); training again from one read",
            stacklevel=3,
        )
        return _fit_once(collection, definition, estimator, False, progress)


def _fit_once(collection, definition: dict, estimator, streaming: bool, progress: bool | None):
    """One training attempt (see :func:`_fit`)."""
    target = definition["target"]
    # null targets are dropped by MongoDataset; a server-side $ne: null would also drop every
    # document for a path through an array index such as scores.0.score
    has_target = {target: {"$exists": True}}
    filt = {"$and": [definition["filter"], has_target]} if definition.get("filter") else has_target
    n_max = definition.get("numTrainingDocuments")
    if n_max is not None and (
        isinstance(n_max, bool) or not isinstance(n_max, numbers.Integral) or n_max < 1
    ):
        raise ValueError(f"numTrainingDocuments must be a positive integer, got {n_max!r}")
    fz_kwargs = {arg: definition[key] for key, arg in FEATURIZER_KEYS.items() if key in definition}
    fz_kwargs.update(definition.get("featurizerOptions") or {})
    # fetch only the fields the featurizer reads (the whole document when it embeds all of it)
    projection = DocumentFeaturizer(**fz_kwargs)._projection(target)
    # streaming: only _id and the target are held, the documents are fetched batch by batch
    ds = MongoDataset(
        collection,
        target=target,
        filter=filt,
        projection=projection,
        limit=definition.get("numTrainingDocuments"),
        streaming=streaming,
    )
    y = ds.y
    if len({_id_key(i) for i in ds.ids}) < len(ds.ids):  # e.g. a view that $unwinds
        raise ValueError("training needs unique _id values; the source returns some twice")
    if len(y) < 2:
        raise ValueError(f"fewer than 2 training documents with {target!r}")
    task = definition.get("task", "auto")
    if task not in ("auto", "classification", "regression"):
        raise ValueError(f"task must be 'auto', 'classification' or 'regression', got {task!r}")
    as_numbers = _as_numbers(y)
    if as_numbers is not None:  # numbers (or numbers as text): blank ones are missing values
        keep = np.flatnonzero(np.isfinite(as_numbers))  # blanks, NaN and infinities
        if len(keep) < len(as_numbers):
            n_bad = int((~np.isnan(as_numbers) & ~np.isfinite(as_numbers)).sum())
            if n_bad:
                warnings.warn(f"{n_bad} infinite {target!r} values are left out", stacklevel=3)
            ds = ds.subset(keep)
        ds._y = y = as_numbers[keep]
    if task == "auto":
        task = _auto_task(y)
    if task == "regression" and y.dtype.kind not in "fiu":  # drop targets that are not numbers
        vals = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
        keep = np.flatnonzero(np.isfinite(vals))
        if len(keep) < 2:
            raise ValueError(f"fewer than 2 training documents with a numeric {target!r}")
        ds = ds.subset(keep)
        ds._y = y = vals[keep]
    spec = estimator if estimator is not None else definition.get("estimator", "auto")

    metrics, test_size = {}, definition.get("testSize", 0.2)
    count = isinstance(test_size, numbers.Integral) and not isinstance(test_size, bool)
    fraction = isinstance(test_size, float) and 0 <= test_size < 1
    if not (count and test_size >= 0 or fraction):
        raise ValueError(
            f"testSize must be a fraction in [0, 1) or a count of documents, got {test_size!r}"
        )
    with tempfile.TemporaryDirectory(prefix="mongopredict_fit_") as tmp:
        uncached = "cache_dir" in fz_kwargs and fz_kwargs["cache_dir"] is None
        model_spec = fz_kwargs.get("embedding_model")
        if test_size and uncached and not (isinstance(model_spec, str) and model_spec == "hashing"):
            # the holdout fit and the final fit embed the same documents: share a temporary cache
            fz_kwargs = {**fz_kwargs, "cache_dir": tmp}
        steps = _Progress(5 if test_size else 2, progress)
        temporary = uncached and fz_kwargs.get("cache_dir") is not None  # (deleted after the fit)
        fz, est, metrics = _fit_with_holdout(
            ds, y, task, spec, fz_kwargs, test_size, steps, temporary
        )
        if uncached and fz.cache_dir is not None:  # the stored model must not point at tmp
            fz.cache_dir = None
            fz.embedder_ = fz._embedder()
    return fz, est, task, len(ds), metrics, _class_values(ds, est, task)


def _class_values(ds, est, task: str) -> list | None:
    """The first stored value of each class of ``est.classes_`` (labels are compared as text:
    ObjectId, date and int labels are strings or floats to the estimator), or ``None``."""
    if task != "classification" or not hasattr(est, "classes_") or ds._raw_y is None:
        return None
    first = {}  # from the read that gave the labels (a second read may see edited values)
    for label, raw in zip(ds.y.tolist(), ds._raw_y):
        first.setdefault(label, raw)
    values = [first.get(c.item() if hasattr(c, "item") else c) for c in est.classes_]
    try:
        bson.encode({"v": values})
    except Exception:  # noqa: BLE001 (a value BSON cannot store: keep the labels)
        return None
    return values


def _fit_with_holdout(
    ds, y, task, spec, fz_kwargs, test_size, steps: _Progress, temporary: bool = False
):
    """Holdout metrics (when ``test_size``), then the featurizer and estimator fit on ``ds``."""
    metrics = {}
    if test_size:
        # stratifying needs at least 2 documents per class, and a class's worth of documents on
        # each side; otherwise the split is plain random
        n = len(y)
        if isinstance(test_size, numbers.Integral) and test_size >= n:
            raise ValueError(
                f"testSize={test_size} leaves no training documents (there are {n}); use a "
                "smaller count, a fraction such as 0.2, or 0 to skip the holdout"
            )
        n_test = (
            int(test_size)
            if isinstance(test_size, numbers.Integral)
            else int(np.ceil(test_size * n))
        )
        counts = np.unique(y, return_counts=True)[1] if task == "classification" else np.array([])
        stratify = bool(
            len(counts) and counts.min() >= 2 and min(n_test, n - n_test) >= len(counts)
        )
        if len(counts) > 1 and not stratify:
            warnings.warn(
                "cannot stratify the holdout, so it is random: a class has under 2 documents, "
                "or a part has fewer documents than there are classes",
                stacklevel=4,
            )
        try:
            from sklearn.model_selection import train_test_split

            tr, te = train_test_split(
                ds, test_size=test_size, stratify=y if stratify else None, random_state=0
            )
            fz = DocumentFeaturizer(**fz_kwargs)
            fz._temporary_cache = temporary  # (for the message if interrupted)
            with steps.step("embedding the training part", len(tr)):
                X = fz.fit_transform(tr)
            with steps.step("fitting on the training part"):
                est = _estimator(spec, task).fit(X, tr.y)
            with steps.step("embedding and scoring the holdout part", len(te)):
                metrics = _metrics(task, est, fz.transform(te), te.y)
            metrics["numTestDocuments"] = len(te)
        except ValueError as e:  # e.g. one class in the training part: no metrics, still a model
            metrics = {"error": f"holdout not evaluated: {e}"[:300]}
            warnings.warn(f"holdout metrics not computed: {e}", stacklevel=3)
            steps.i = 3  # the holdout steps that did not run
    fz = DocumentFeaturizer(**fz_kwargs)
    fz._temporary_cache = temporary
    with steps.step("embedding all documents", len(ds)):  # mostly from the cache after a holdout
        X = fz.fit_transform(ds)
    with steps.step("fitting the final model"):
        est = _estimator(spec, task).fit(X, y)
    return fz, est, metrics


def _dump(fz: DocumentFeaturizer, est) -> bytes:
    """Pickle ``(featurizer, estimator)`` without the parts that cannot or should not be stored.
    Raises ``ValueError`` over :data:`MAX_MODEL_BYTES`."""
    fz = copy.copy(fz)
    fz.embedder_ = None  # holds the loaded model / sqlite connection; rebuilt on load
    # (an API key is never stored: VoyageEmbedder drops its key and client when pickled)
    fz.__dict__.pop("tokenizer_", None)
    fz.__dict__.pop("_temporary_cache", None)
    blob = pickle.dumps((fz, est), protocol=pickle.HIGHEST_PROTOCOL)
    if len(blob) > MAX_MODEL_BYTES:
        raise ValueError(
            f"fitted model is {len(blob) / 2**20:.1f} MB, over the 15 MB document limit"
        )
    return blob


def _json_safe(value):
    """``value`` with everything BSON cannot store (estimators, embedders, paths) as its repr,
    so a definition holding Python objects can still be recorded as ``latestDefinition``."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        bson.encode({"v": value})
        return value
    except Exception:  # noqa: BLE001
        return repr(value)


def create_prediction_model(
    collection, model: PredictionModel | dict, estimator=None, progress: bool | None = None
) -> str:
    """Train and store a prediction model; the counterpart of ``Collection.create_search_index``.

    Training runs synchronously in this process. If a model with this name already exists on
    the collection, it is rebuilt (as ``update_prediction_model`` would): as with a search index
    rebuild, the previous version stays stored and queryable until the new one is written.

    The record in :data:`MODELS_COLLECTION` (``_id`` ``"<collection>.<name>"``) holds:

    * ``collection``, ``name``, ``type`` and ``latestDefinition`` (the definition of the latest
      build, with values BSON cannot store, such as estimator objects, as their ``repr``).
    * ``status``: ``"BUILDING"`` while a build runs, then ``"READY"``, or ``"FAILED"`` with
      ``message`` set to the error (an interrupted build, e.g. ``KeyboardInterrupt``, is marked
      ``FAILED`` too). Only a killed process (SIGKILL, out of memory, a lost node) leaves a
      record ``BUILDING``; ``buildStartedAt`` records when the build started and
      :func:`list_prediction_models` flags such builds as ``stale`` after ``STALE_BUILD``.
    * ``queryable``: ``False`` while the first build runs; ``True`` once a version is stored.
      A failed rebuild keeps the previous version and its ``queryable`` value.
    * ``version``: increments with each build started (taken atomically, so concurrent builds
      get distinct versions); ``buildId``: unique id of the stored build.
    * After a successful build: ``task``, ``classes`` (empty for regression),
      ``numTrainingDocuments``, ``metrics`` (``r2`` / ``mae``, or ``accuracy`` / ``auroc``, plus
      ``numTestDocuments``; empty when ``testSize`` is 0), ``modelBytes``, ``updatedAt`` (UTC)
      and ``model`` (the pickled featurizer and estimator).
    * Internal bookkeeping: ``nextVersion`` and ``buildingId`` (the build allowed to write).

    Only the latest build started may write its result. If a newer build started meanwhile, or
    the model was dropped, this build's result is discarded with a warning.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        Collection holding the training documents; the model is registered on it.
    model : PredictionModel or dict
        The model, or its ``{"name", "type", "definition"}`` document (``name`` defaults to
        ``"default"``, ``type`` to ``"prediction"``). See :class:`PredictionModel` for the
        definition keys.
    estimator : sklearn estimator, optional
        Overrides the definition's ``estimator`` with any sklearn estimator object.
    progress : bool or None, default None
        Show a progress bar for each training step: embedding the training part, fitting on it,
        embedding and scoring the holdout part, embedding all documents (mostly from the cache)
        and fitting the final model (only these last two when ``testSize`` is 0). Embedding steps
        count documents. ``None`` shows the bars on a terminal or in a notebook and hides them
        when stderr is redirected; ``True`` / ``False`` force them on or off.

    Returns
    -------
    str
        The model name.

    Raises
    ------
    ValueError
        If the definition is invalid (see :class:`PredictionModel`), there are fewer than 2
        training documents, the estimator name does not fit the task, the pickled model is over
        15 MB, or the record is over MongoDB's 16 MB document limit. The record is marked
        ``FAILED`` before any error from training is re-raised.

    Warns
    -----
    UserWarning
        If the build was superseded by a newer one (or the model dropped) and was not stored.

    Examples
    --------
    Usage:

        create_prediction_model(coll, PredictionModel({"target": "room_type"}, "room_type"))
        create_prediction_model(coll, {"name": "price", "definition": {"target": "price"}},
                                estimator=HistGradientBoostingRegressor())
    """
    if isinstance(model, dict):
        if "definition" not in model:
            raise ValueError("a model document needs a 'definition' (and a 'name')")
        model = PredictionModel(
            model["definition"], model.get("name", "default"), model.get("type", "prediction")
        )
    elif not isinstance(model, PredictionModel):
        raise ValueError(f"model must be a PredictionModel or a document, got {model!r}")
    definition = dict(model.definition)
    if estimator is not None:
        definition["estimator"] = estimator
    spec = definition.get("estimator")
    if spec is not None and not isinstance(spec, str) and not hasattr(spec, "fit"):
        raise ValueError(f"the estimator needs a fit method, got {spec!r}")
    _check_server_version(collection)
    models, _id = _models(collection), _model_id(collection, model.name)
    base = {
        "collection": collection.name,
        "name": model.name,
        "type": model.type,
        "latestDefinition": _json_safe(definition),
    }
    build_id = uuid.uuid4().hex
    # records from before version counters: start the counter at their version (a conditional
    # write, so concurrent builds agree)
    old = models.find_one({"_id": _id}, {"version": 1, "nextVersion": 1})
    if old and "version" in old and "nextVersion" not in old:
        models.update_one(
            {"_id": _id, "nextVersion": {"$exists": False}},
            {"$set": {"nextVersion": old["version"]}},
        )
    # as in a search index rebuild, the previous version stays queryable while this one trains;
    # the version is taken atomically, and only the latest build may write the result
    rec = models.find_one_and_update(
        {"_id": _id},
        {
            "$set": {
                **base,
                "status": "BUILDING",
                "buildingId": build_id,
                "buildStartedAt": dt.datetime.now(dt.timezone.utc),
            },
            "$inc": {"nextVersion": 1},
        },
        upsert=True,
        projection={"model": 0},  # (not the stored blob, up to 15 MB)
        return_document=ReturnDocument.AFTER,
    )
    if "modelBytes" not in rec:  # no earlier version to serve meanwhile
        models.update_one({"_id": _id, "buildingId": build_id}, {"$set": {"queryable": False}})
    version = rec["nextVersion"]
    # every later write matches on buildingId: a newer build replaces it, so this one loses
    mine = {"_id": _id, "buildingId": build_id}
    try:
        fz, est, task, n, metrics, values = _fit(collection, definition, estimator, progress)
        blob = _dump(fz, est)
        # numpy scalars -> Python values, which BSON can store
        classes = [c.item() if hasattr(c, "item") else c for c in getattr(est, "classes_", [])]
        # as text, the keys of predictProbabilities: str() of the value $predict returns
        classes = _class_keys(values if values is not None else classes)
        result = {
            **base,
            "version": version,
            "buildId": build_id,
            "status": "READY",
            "queryable": True,
            "task": task,
            "classes": classes,
            # each class as first stored (an ObjectId, a date, an int): $predict returns these
            **({"classValues": values} if values is not None else {}),
            "numTrainingDocuments": n,
            "metrics": metrics,
            "modelBytes": len(blob),
            "versions": _versions(),
            "updatedAt": dt.datetime.now(dt.timezone.utc),
            "model": Binary(blob),
        }
        sig = _signature(collection.database.name, {"_id": _id, **result}, blob)
        if sig:
            result["modelSignature"] = sig
        size = len(bson.encode(result))
        if size > MAX_RECORD_BYTES:
            raise ValueError(
                f"the model record is {size / 2**20:.1f} MB (model {len(blob) / 2**20:.1f} MB), "
                "over MongoDB's 16 MB document limit; use a smaller estimator or definition"
            )
        stale = {k: "" for k in ("message", "classValues", "modelSignature") if k not in result}
        written = models.update_one(mine, {"$set": result, "$unset": stale})
    except BaseException as e:  # (KeyboardInterrupt too: never leave the record BUILDING)
        message = (
            f"{type(e).__name__}: {e}"
            if isinstance(e, Exception)
            else f"interrupted ({type(e).__name__})"
        )
        models.update_one(mine, {"$set": {"status": "FAILED", "message": message}})
        if not isinstance(e, Exception):  # Ctrl-C: say what state the model is left in
            import sys

            kept = (
                "the previous version is still used by $predict"
                if "modelBytes" in rec
                else "there is no earlier version, so $predict cannot use it yet"
            )
            print(
                f"Fitting prediction model {model.name!r} was interrupted: its record is marked "
                f"FAILED and {kept}. Run create_prediction_model again to fit it.",
                file=sys.stderr,
            )
        raise
    if written.matched_count == 0:  # a newer build started meanwhile, or the model was dropped
        warnings.warn(
            f"prediction model {model.name!r}: this build was superseded and not stored",
            stacklevel=2,
        )
        return model.name
    # warm the in-process cache, so the first $predict does not unpickle what we just fitted
    _loaded[(collection.database.name, _id)] = (build_id, _for_serving(fz), est)
    return model.name


def update_prediction_model(
    collection, name: str, definition: dict, estimator=None, progress: bool | None = None
) -> str:
    """Retrain an existing model with a new definition (``Collection.update_search_index``).

    The previous version stays queryable while the new one trains, as described in
    :func:`create_prediction_model`.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        Collection the model is registered on.
    name : str
        Name of the existing model.
    definition : dict
        The new definition; it replaces the old one entirely (see :class:`PredictionModel`).
    estimator : sklearn estimator, optional
        Overrides the definition's ``estimator``.
    progress : bool or None, default None
        Progress bars for the training steps, as in :func:`create_prediction_model`.

    Returns
    -------
    str
        The model name.

    Raises
    ------
    ValueError
        If no model ``name`` exists on the collection, or for any reason
        :func:`create_prediction_model` raises.
    """
    if _models(collection).find_one({"_id": _model_id(collection, name)}, {"_id": 1}) is None:
        raise ValueError(f"prediction model {name!r} not found on collection {collection.name!r}")
    return create_prediction_model(
        collection, PredictionModel(definition, name), estimator, progress
    )


def list_prediction_models(collection, name: str | None = None) -> list[dict]:
    """Status, definition and metrics of the models on a collection (``list_search_indexes``).

    Parameters
    ----------
    collection : pymongo.collection.Collection
        Collection the models are registered on.
    name : str, optional
        Return only this model (an empty list if it does not exist).

    Returns
    -------
    list of dict
        One record per model, as described in :func:`create_prediction_model`, without the
        pickled ``model`` and the internal ``nextVersion`` / ``buildingId`` fields. A record
        still ``BUILDING`` after ``STALE_BUILD`` gets ``stale: True``: its build was most likely
        killed (the process cannot record that); create the model again.
    """
    q = {"collection": collection.name, **({"name": name} if name else {})}
    records = list(_models(collection).find(q, {"model": 0, "nextVersion": 0, "buildingId": 0}))
    now = dt.datetime.now(dt.timezone.utc)
    for rec in records:
        started = rec.get("buildStartedAt")
        if rec.get("status") == "BUILDING" and started is not None:
            started = started if started.tzinfo else started.replace(tzinfo=dt.timezone.utc)
            rec["stale"] = now - started > STALE_BUILD
    return records


def drop_prediction_model(collection, name: str) -> None:
    """Delete a stored model and its cached copy (``Collection.drop_search_index``).

    Dropping a model that does not exist is a no-op. A build still running for it is not
    stored when it finishes.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        Collection the model is registered on.
    name : str
        Model name.
    """
    _models(collection).delete_one({"_id": _model_id(collection, name)})
    _loaded.pop((collection.database.name, _model_id(collection, name)), None)


def _load_prediction_model(collection, name: str):
    """Load the stored version of a model, as the ``$predict`` stage does.

    The unpickled featurizer and estimator are cached per process, keyed by database, model and
    build, so they are unpickled once per build. While a new version trains, or after it failed,
    the previous stored version is served.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        Collection the model is registered on.
    name : str
        Model name.

    Returns
    -------
    tuple
        ``(featurizer, estimator, record)``: the fitted
        :class:`pymongopredict.DocumentFeaturizer` (with its embedder rebuilt), the fitted
        estimator, and the model record without the pickled ``model`` field.

    Raises
    ------
    ValueError
        If the model does not exist, or has no stored version yet (its first build is running
        or failed).
    """
    models, _id = _models(collection), _model_id(collection, name)
    # the record without the pickled model first: it is only downloaded for a build not yet loaded
    rec = models.find_one({"_id": _id}, {"model": 0})
    if rec is None:
        raise ValueError(f"prediction model {name!r} not found on collection {collection.name!r}")
    key = (collection.database.name, rec["_id"])
    # records stored before build ids existed: identify the build by version and time
    build = rec.get("buildId") or f"{rec.get('version')}:{rec.get('updatedAt')}"
    entry = _loaded.get(key)  # read once: other threads may replace or drop it meanwhile
    if entry is None or entry[0] != build:
        full = models.find_one({"_id": _id})
        if full is None:
            raise ValueError(f"prediction model {name!r} was dropped")
        if "model" not in full:
            raise ValueError(
                f"prediction model {name!r} is not queryable (status {full.get('status')})"
            )
        rec = full
        build = rec.get("buildId") or f"{rec.get('version')}:{rec.get('updatedAt')}"
        entry = (build, *_unpickle(name, rec, collection.database.name))
        _loaded[key] = entry
    rec.pop("model", None)
    return entry[1], entry[2], rec


def _version_of(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # noqa: BLE001 (not installed)
        return None


def _versions() -> dict:
    """Library versions a model is stored with (a pickle is only safe to load with the same)."""
    import sklearn

    try:
        from importlib.metadata import version

        own = version("pymongopredict")
    except Exception:  # noqa: BLE001 (not installed as a distribution)
        own = None
    return {
        "pymongopredict": own,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "xgboost": _version_of("xgboost"),
        "python": platform.python_version(),
    }


#: record fields the signature covers besides the blob: what decides the predictions
SIGNED_FIELDS = ("_id", "buildId", "task", "classes", "classValues")


def _signature(database: str, rec: dict, blob: bytes) -> str | None:
    """HMAC-SHA256 under :data:`MODEL_KEY_ENV` of the database name, :data:`SIGNED_FIELDS` of
    the record and the blob, or ``None`` without a key."""
    key = os.environ.get(MODEL_KEY_ENV)
    if not key:
        return None
    signed = bson.encode(
        {
            "db": database,
            **{k: rec.get(k) for k in SIGNED_FIELDS},
            "blob": hashlib.sha256(blob).digest(),
        }
    )
    return hmac.new(key.encode(), signed, hashlib.sha256).hexdigest()


def _class_keys(values: list) -> list[str]:
    """``str()`` of each class value (the predictProbabilities keys); the labels' own text if
    two values print alike."""
    keys = [str(v) for v in values]
    return keys if len(set(keys)) == len(keys) else [_label(v) for v in values]


def _unpickle(name: str, rec: dict, database: str):
    """``(featurizer, estimator)`` of a full record, checked before and after unpickling."""
    blob = bytes(rec["model"])
    if os.environ.get(MODEL_KEY_ENV):
        want, got = _signature(database, rec, blob), rec.get("modelSignature") or ""
        if not hmac.compare_digest(want, got):
            raise PermissionError(
                f"prediction model {name!r} is not signed with {MODEL_KEY_ENV} (or was changed "
                "after it was stored); it is not loaded"
            )
    stored, running = rec.get("versions") or {}, _versions()
    differ = {
        k: (v, running.get(k)) for k, v in stored.items() if k != "python" and v != running.get(k)
    }
    try:
        fz, est = pickle.loads(blob)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"prediction model {name!r} cannot be loaded ({type(e).__name__}: {e}); stored with "
            f"{stored or 'unrecorded versions'}, running {running}: rebuild it with "
            "update_prediction_model"
        ) from e
    if differ:
        warnings.warn(
            f"prediction model {name!r} was stored with other library versions "
            f"({', '.join(f'{k} {a} now {b}' for k, (a, b) in differ.items())}); if predictions "
            "fail or look wrong, rebuild it with update_prediction_model",
            stacklevel=3,
        )
    if not hasattr(fz, "_embedder"):
        raise ValueError(f"prediction model {name!r} does not hold a DocumentFeaturizer")
    return _for_serving(fz), est


def _for_serving(fz):
    """The featurizer with its embedder for predicting: the cache is :data:`PREDICT_CACHE_ENV`
    (none when unset), not the training cache_dir."""
    fz.cache_dir = os.environ.get(PREDICT_CACHE_ENV) or None
    fz.embedder_ = fz._embedder()  # dropped by _dump; rebuilt from the stored settings
    return fz
