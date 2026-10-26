"""``DocumentFeaturizer``: MongoDB documents -> numeric feature matrix.

This module holds the featurizer shared by both APIs of the package. The embeddings API exposes
it directly; the predictions API (``$predict``) builds one inside every stored prediction model.
It is an sklearn transformer whose ``fit`` / ``transform`` accept a
:class:`pymongopredict.embeddings_api.dataset.MongoDataset` (streamed in batches), a cursor, a
generator or a list of documents, and return a ``float32`` matrix that any sklearn estimator can
consume, directly or inside a ``Pipeline``.

The matrix is two blocks side by side, ``[tabular columns | embedding]``:

* **columns**: dotted field paths extracted directly as tabular columns. Their kind is inferred
  from the training documents (numeric, boolean, datetime, categorical, or "size" for lists and
  sub-documents) and they go through the usual sklearn preprocessing (median imputation and
  scaling for numbers, one-hot encoding with an "infrequent" bucket for categories).
* **embeddings**: the whole document (or the selected ``embed_fields``) is serialised to a
  compact JSON string, cut to a character or token budget, and embedded with an embedding model
  from :mod:`pymongopredict.embeddings_api.embed` (voyage-4-nano by default). The prediction
  target is always left out of this JSON, or replaced by a mask token (``mask_target=True``).

Document handling (paths, JSON pieces, budgets, batching) lives in
:mod:`pymongopredict.embeddings_api.documents`; this module only decides what goes where.

Examples
--------
With a pymongo collection ``movies`` (here ``sample_mflix.movies``) as input:

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from pymongopredict.embeddings_api import DocumentFeaturizer, MongoDataset

    ds = MongoDataset(movies, target="imdb.rating")  # documents without a rating are dropped
    train, test = train_test_split(ds, test_size=0.2)
    prep = DocumentFeaturizer(columns=["year", "runtime", "rated"],
                              embed_fields=["title", "plot", "genres"])
    model = Ridge().fit(prep.fit_transform(train), train.y)
    model.score(prep.transform(test), test.y)

``embedding_model="hashing"`` gives a fast, dependency free embedder for tests.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import os
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._time import _as_datetime
from .dataset import MongoDataset
from .documents import (
    _as_documents,
    _assemble_json,
    _compile_path,
    _get_compiled,
    _iter_batches,
    _json_pieces,
    _projection_for,
    _scalar,
)
from .embed import DEFAULT_MODEL, LOCAL_ALIASES, BaseEmbedder, _base_spec, _make_embedder

# largest finite float32: tabular values are clipped to it so the float32 matrix stays finite
_F32_MAX = float(np.finfo(np.float32).max)
# a callable given the size of each featurized batch, set while a progress bar counts documents
# (predictions_api training); None otherwise
_on_documents = contextvars.ContextVar("mongopredict_on_documents", default=None)
# datetimes become seconds since this instant
_EPOCH_NAIVE = dt.datetime(1970, 1, 1)
# spellings read as booleans in a column inferred as "bool" (compared lower-cased and stripped)
_TRUE, _FALSE = {"true", "1", "yes", "y", "t"}, {"false", "0", "no", "n", "f"}


def _not_a_date(v):
    """``v``, or NaN for a date (a numeric column must not read dates as numbers)."""
    if isinstance(v, (dt.date, np.datetime64)) or type(v).__name__ == "DatetimeMS":
        return np.nan
    return v


def _epoch_seconds(v) -> float:
    """Seconds since 1970-01-01 UTC, or NaN when ``v`` is not a time."""
    if type(v).__name__ == "DatetimeMS":  # pymongo's out-of-range dates: milliseconds
        return int(v) / 1000.0
    t = _as_datetime(v)
    return np.nan if t is None else (t - _EPOCH_NAIVE).total_seconds()


def _category(v) -> str:
    """Category label: integral floats as integers, so 1, 1.0 and "1" are one category.

    Mixed int / float / string spellings of the same value are common in real collections; without
    this they would become separate one-hot columns.
    """
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _as_bool(v) -> float:
    """1.0 / 0.0 for booleans, 0 / 1 and their usual spellings; NaN for anything else.

    NaN is then imputed like any missing value, so a stray non-boolean value never breaks a fit.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)) and v in (0, 1):
        return float(v)
    if isinstance(v, str):
        low = v.strip().lower()
        return 1.0 if low in _TRUE else 0.0 if low in _FALSE else np.nan
    return np.nan


class DocumentFeaturizer(BaseEstimator, TransformerMixin):
    """Turn documents into ``[tabular columns | embedding of JSON]`` features.

    A scikit-learn transformer. All constructor arguments are stored unchanged (so ``get_params``,
    ``clone`` and grid search work); everything learned is set by ``fit`` in attributes ending
    with ``_``, and every fit starts from scratch.

    Parameters
    ----------
    columns : list of str or None, default None
        Dotted field paths (``"imdb.rating"``, ``"tags.0"``) used directly as tabular features.
        ``None`` or ``[]`` means no tabular block. The kind of each column is inferred from the
        training documents (see ``_infer_kind``):

        * ``"size"``: every value is a list or sub-document; the feature is its length.
        * ``"bool"``: every value is a boolean; at transform time ``0`` / ``1``, ``"yes"`` /
          ``"no"``, ``"true"`` / ``"false"``, ``"t"`` / ``"f"`` and ``"y"`` / ``"n"`` are read too,
          anything else is missing.
        * ``"datetime"``: datetime values; the feature is seconds since 1970-01-01 UTC.
        * ``"numeric"``: a numeric dtype, or at least 95% of the values parse as numbers
          (numeric strings included). Unparsable values and infinities are missing, values
          beyond the float32 range are clipped. A column with no values at all is numeric.
        * ``"categorical"``: anything else. Values become strings (``1``, ``1.0`` and ``"1"``
          are one category) and missing values the category ``"<NA>"``.

        Non-categorical columns are imputed with the training median (and standardised when
        ``scale_numeric``); categorical columns are one-hot encoded.
    embed_fields : list of str, bool or None, default None
        What is serialised to JSON and embedded. ``None`` or ``True``: the whole document except
        ``_id`` and the excluded paths (the ``columns`` stay in the JSON too). A list of dotted
        paths: only those fields, in document order. ``False`` or ``[]``: no embedding block.
    exclude : sequence of str, default ()
        Dotted paths never embedded. The target of a ``MongoDataset`` passed to ``fit`` is
        excluded automatically; ``exclude_`` holds the effective list after fitting.
    embedding_model : str, BaseEmbedder or None, default ``"voyage-4-nano"``
        Embedding model: ``"voyage-4-nano"`` or any
        ``"org/name"`` Hugging Face id (local), other ``"voyage-*"`` names (Voyage API),
        ``"hashing"`` (tests), a ``"@<dim>"`` suffix for Matryoshka truncation, or an embedder
        instance.
    embedding_kwargs : dict or None, default None
        Forwarded to the embedder constructor (e.g. ``{"dimension": 512}``,
        ``{"batch_size": 64}``).
    cache_dir : str or None, default ``".mongopredict_cache"``
        Directory of the sqlite embedding cache, keyed by the embedder's settings and the
        text; ``None`` disables it.
    batch_size : int, default 512
        Documents per batch for cursors, generators and lists. A ``MongoDataset`` uses its own
        ``batch_size``.
    mask_target : bool, default False
        ``False`` removes the target field from the embedded JSON. ``True`` keeps it as the first
        key, with the value ``mask_token``, and puts ``prefix`` before the JSON, so the model
        sees which field is asked for. The masked key is ``mask_key`` or, failing that, the
        dataset target; with plain documents (no dataset target) only ``mask_key`` can be masked,
        and it is then also excluded from the rest of the JSON.
    mask_token : str, default ``"<MASK>"``
        The value shown for the masked field.
    mask_key : str or None, default None
        The key shown with the mask (default: the dataset target). Useful when the target is a
        derived field (e.g. a collapsed or logged copy) and the model should see the original
        field name.
    prefix : str or None, default None
        Text put before the JSON. ``None`` means ``"Predict the value of <mask_token>: "`` when a
        field is masked and nothing otherwise; any string (``""`` included) is used as is.
    max_json_chars : int or None, default 8000
        Character budget for the whole text (prefix included). Fields are added in order and the
        first one that does not fit is cut (when more than its key fits), the rest dropped.
        ``None`` means unlimited. Ignored when ``max_tokens`` is set.
    max_tokens : int, ``"model"`` or None, default None
        Token budget instead of characters, measured with ``tokenizer``. ``"model"`` uses the
        tokenizer's ``model_max_length``, capped by the model config's
        ``max_position_embeddings`` (32768 for voyage-4-nano). 16 tokens are kept free for the
        document prompt and special tokens. For a local embedder the model's ``max_seq_length``
        is set to this budget too, unless ``embedding_kwargs`` sets it.
    tokenizer : str, tokenizer object or None, default None
        Used only to measure ``max_tokens``. A Hugging Face id, or a Hugging Face style tokenizer
        (callable on a list of texts, returning ``{"input_ids": ...}``, with ``decode``).
        ``None``: the embedding model's tokenizer when it is a local model, else voyage-4-nano's.
    scale_numeric : bool, default True
        Standardise the non-categorical columns after imputation.
    min_frequency : int, default 5
        Categories seen fewer times in training go to the one-hot "infrequent" column.
    max_categories : int, default 100
        Upper bound on one-hot columns per categorical column (infrequent bucket included).
        The feature matrix is dense float32: its memory is documents x (embedding dimension +
        numeric columns + one-hot columns) x 4 bytes, so many categorical columns with a large
        ``max_categories`` cost memory on large training sets.

    Attributes
    ----------
    target_ : str or None
        The dataset target seen by the last fit (``None`` for plain documents).
    exclude_ : list of str
        Paths left out of the embedded JSON: ``exclude``, the target and, for plain documents in
        mask mode, ``mask_key``.
    embedder_ : BaseEmbedder or None
        The embedder built by the last fit (``None`` when embeddings are disabled).
    tokenizer_ : tokenizer or None
        Tokenizer loaded for ``max_tokens``; reset by every fit.
    column_kinds_ : dict of str to str
        Inferred kind of each column (see ``columns``).
    tabular_ : sklearn ColumnTransformer or None
        Fitted preprocessing of the columns (``None`` without columns).
    n_tabular_features_, n_embedding_features_ : int
        Widths of the two blocks of the output matrix.

    Examples
    --------
    >>> docs = [{"_id": i, "price": 10 + i, "kind": "a" if i % 2 else "b",
    ...          "text": f"item {i}"} for i in range(20)]
    >>> prep = DocumentFeaturizer(columns=["price", "kind"], embedding_model="hashing",
    ...                           embedding_kwargs={"n_features": 32}, cache_dir=None)
    >>> X = prep.fit_transform(docs)  # the whole document but ``_id`` is embedded
    >>> X.shape, X.dtype
    ((20, 35), dtype('float32'))
    >>> list(prep.get_feature_names_out()[:4])
    ['num__price', 'cat__kind_a', 'cat__kind_b', 'emb_0']
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        embed_fields: Sequence[str] | None | bool = None,
        exclude: Sequence[str] = (),
        embedding_model: str | BaseEmbedder | None = DEFAULT_MODEL,
        embedding_kwargs: dict | None = None,
        cache_dir: str | None = ".mongopredict_cache",
        batch_size: int = 512,
        mask_target: bool = False,
        mask_token: str = "<MASK>",
        mask_key: str | None = None,
        prefix: str | None = None,
        max_json_chars: int | None = 8000,
        max_tokens: int | str | None = None,
        tokenizer=None,
        scale_numeric: bool = True,
        min_frequency: int = 5,
        max_categories: int = 100,
    ):
        self.columns = columns
        self.embed_fields = embed_fields
        self.exclude = exclude
        self.embedding_model = embedding_model
        self.embedding_kwargs = embedding_kwargs
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.mask_target = mask_target
        self.mask_token = mask_token
        self.mask_key = mask_key
        self.prefix = prefix
        self.max_json_chars = max_json_chars
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.scale_numeric = scale_numeric
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    # ------------------------------------------------------------------ tabular block
    def _frame(self, docs: list[dict]) -> pd.DataFrame:
        """Raw values of ``columns`` (BSON scalars made pandas friendly), one row per document.

        Kept raw so kind inference and coercion can see the original types.
        """
        cols = list(self.columns or [])
        data = {}
        for c in cols:
            steps = _compile_path(c)
            data[c] = [_scalar(_get_compiled(d, steps)) for d in docs]
        return pd.DataFrame(data, columns=cols)

    @staticmethod
    def _infer_kind(s: pd.Series) -> str:
        """Kind of a raw column from its training values: size, bool, datetime, numeric or
        categorical (see the ``columns`` parameter). Checked in that order, as e.g. booleans
        also pass as numbers."""
        nonnull = s.dropna()
        if len(nonnull) == 0:
            return "numeric"
        if nonnull.map(lambda v: isinstance(v, (list, dict))).all():
            return "size"  # lists / sub-documents: use their length
        if (
            pd.api.types.is_bool_dtype(nonnull)
            or nonnull.map(lambda v: isinstance(v, (bool, np.bool_))).all()
        ):
            return "bool"
        if pd.api.types.is_datetime64_any_dtype(nonnull):
            return "datetime"
        if pd.api.types.is_numeric_dtype(nonnull):
            return "numeric"
        if nonnull.map(
            lambda v: (
                (hasattr(v, "year") and hasattr(v, "hour")) or type(v).__name__ == "DatetimeMS"
            )
        ).all():
            return "datetime"
        # numbers stored as strings (or mixed with a few junk values) are still numeric
        coerced = pd.to_numeric(nonnull.astype(str), errors="coerce")
        if coerced.notna().mean() >= 0.95:
            return "numeric"
        return "categorical"

    def _coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw columns -> floats (or category strings) according to ``column_kinds_``.

        Values that do not match the fitted kind become missing instead of raising, so a
        collection with mixed types still transforms.
        """
        out = {}
        for c, kind in self.column_kinds_.items():
            s = df[c]
            if kind == "numeric":
                # a stray date is missing in every batch (pandas would turn a batch of dates
                # alone into nanoseconds)
                x = pd.to_numeric(s.astype(object).map(_not_a_date), errors="coerce").astype(float)
                # inf is missing; values beyond float32 are clipped so X stays finite
                out[c] = x.replace([np.inf, -np.inf], np.nan).clip(-_F32_MAX, _F32_MAX)
            elif kind == "bool":
                out[c] = s.map(_as_bool).astype(float)
            elif kind == "datetime":
                # epoch seconds per value (years 1 to 9999, and numbers as epoch seconds or
                # milliseconds, as in time splits), whatever else is in the batch
                out[c] = s.astype(object).map(_epoch_seconds).astype(float)
            elif kind == "size":
                out[c] = s.map(
                    lambda v: len(v) if isinstance(v, (list, tuple, dict)) else np.nan
                ).astype(float)
            else:
                out[c] = (
                    s.map(_category, na_action="ignore").astype(object).where(s.notna(), "<NA>")
                )
        return pd.DataFrame(out, columns=list(self.column_kinds_))

    # ------------------------------------------------------------------ embedding block
    # ------------------------------------------------------------------ text formatting
    def _prefix(self) -> str:
        """Text put before the JSON: ``prefix`` if given, else the mask prompt in mask mode."""
        if self.prefix is not None:
            return self.prefix
        # the default prefix announces a masked field, so only when there is one
        return f"Predict the value of {self.mask_token}: " if self._first() else ""

    def _first(self) -> dict | None:
        """``{key: mask_token}`` to put first in the JSON in mask mode, else ``None``."""
        key = self.mask_key or getattr(self, "target_", None)
        if self.mask_target and key:
            return {key: self.mask_token}
        return None

    def _max_positions(self) -> int | None:
        """``max_position_embeddings`` of the local embedding model's config, when it has one."""
        m = _base_spec(self.embedding_model) if isinstance(self.embedding_model, str) else None
        m = LOCAL_ALIASES.get(m, m)
        if not m or "/" not in m:
            return None
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(m, trust_remote_code=True)
            return int(getattr(config, "max_position_embeddings", 0) or 0) or None
        except Exception:  # noqa: BLE001
            return None

    def _tokenizer(self):
        """Tokenizer that measures ``max_tokens``, loaded once per fit (cached in ``tokenizer_``).

        Defaults to the local embedding model's own tokenizer so the budget matches what the
        model sees; API models fall back to voyage-4-nano's.
        """
        if getattr(self, "tokenizer_", None) is None:
            tok = self.tokenizer
            if tok is None or isinstance(tok, str):
                from transformers import AutoTokenizer

                name = tok
                if name is None:
                    m = (
                        _base_spec(self.embedding_model)
                        if isinstance(self.embedding_model, str)
                        else None
                    )
                    m = LOCAL_ALIASES.get(m, m)
                    name = m if m and "/" in m else LOCAL_ALIASES["voyage-4-nano"]
                tok = AutoTokenizer.from_pretrained(name)
            self.tokenizer_ = tok
        return self.tokenizer_

    def token_limit(self) -> int | None:
        """Effective token budget per text.

        Returns
        -------
        int or None
            ``max_tokens`` as an int; for ``max_tokens="model"`` the tokenizer's
            ``model_max_length`` capped by the model's ``max_position_embeddings``. ``None`` when
            ``max_tokens`` is ``None`` (the character budget ``max_json_chars`` is used).

        Raises
        ------
        ValueError
            With ``max_tokens="model"`` when the tokenizer reports no usable length.
        """
        if self.max_tokens is None:
            return None
        if self.max_tokens == "model":
            limit = int(getattr(self._tokenizer(), "model_max_length", 0) or 0)
            # the model's own position limit caps a tokenizer that advertises more
            # (Qwen3 tokenizers say 131,072)
            pos = self._max_positions()
            if pos and (limit <= 0 or limit > pos):
                limit = pos
            if limit <= 0 or limit > 10**8:
                raise ValueError("tokenizer has no usable model_max_length; pass max_tokens=int")
            return limit
        return int(self.max_tokens)

    _RESERVED_TOKENS = 16  # document prompt + special tokens + merge slack

    def _texts(self, docs: list[dict]) -> list[str]:
        """The texts to embed: ``prefix`` + compact JSON of each document, within the budget.

        In token mode all pieces of the batch are tokenized in one call (much faster than per
        document), and only the piece that overflows is tokenized again to be cut.
        """
        fields = None if self._embed_whole() else list(self.embed_fields)
        exclude = getattr(self, "exclude_", self.exclude)
        prefix, first = self._prefix(), self._first()
        per_doc = [_json_pieces(d, fields, exclude, first) for d in docs]
        limit = self.token_limit()
        if limit is None:
            budget = (
                None if self.max_json_chars is None else max(self.max_json_chars - len(prefix), 0)
            )
            return [_assemble_json(p, budget, prefix=prefix) for p in per_doc]
        tok = self._tokenizer()
        flat = [piece for pieces in per_doc for piece in pieces]
        ids = tok(flat, add_special_tokens=False)["input_ids"] if flat else []
        sizes = [len(x) for x in ids]
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        budget = max(limit - n_prefix - self._RESERVED_TOKENS, 0)

        def truncate(piece, room):
            # cut a piece to ``room`` tokens (decode may not round-trip exactly; the reserve
            # tokens absorb that)
            return tok.decode(tok(piece, add_special_tokens=False)["input_ids"][:room])

        out, i = [], 0
        for pieces in per_doc:
            out.append(_assemble_json(pieces, budget, sizes[i : i + len(pieces)], truncate, prefix))
            i += len(pieces)
        return out

    def _embed_whole(self) -> bool:
        """``embed_fields`` ``None`` (or ``True``): embed the whole document."""
        return self.embed_fields is None or self.embed_fields is True

    def _embedder(self):
        """Build the embedder for a fit, or ``None`` when ``embed_fields`` is ``False`` / ``[]``.

        In token mode a local model's ``max_seq_length`` follows the token budget.
        """
        if self.embed_fields is False or (
            isinstance(self.embed_fields, (list, tuple)) and not self.embed_fields
        ):
            return None
        kwargs = dict(self.embedding_kwargs or {})
        limit = self.token_limit()
        spec = self.embedding_model
        if limit is not None and isinstance(spec, str):
            base = _base_spec(spec)
            if base in LOCAL_ALIASES or "/" in base:  # local sentence-transformers model
                if "max_seq_length" not in kwargs:
                    # texts are already cut to this budget, so the model never truncates and the
                    # length stays out of the cache key, but only when the budget is measured with
                    # the model's own tokenizer and no custom prompt takes up tokens
                    exact = self.tokenizer is None and kwargs.get("prompt") is None
                    kwargs.update(max_seq_length=limit, max_seq_length_in_key=not exact)
        return _make_embedder(spec, self.cache_dir, **kwargs)

    # ------------------------------------------------------------------ sklearn API
    # ------------------------------------------------------------------ inputs
    def _projection(self, target: str | None) -> dict | None:
        """Fields to fetch for a streaming dataset (``None`` = whole document)."""
        if self._embed_whole():
            return None
        fields = list(self.columns or []) + list(self.embed_fields or [])
        return _projection_for(fields + ([target] if target else []))

    def _batches(self, X):
        """Document batches of any accepted input; datasets fetch only the fields used."""
        if isinstance(X, MongoDataset):
            return X.iter_batches(self._projection(X.target))
        return _iter_batches(X, self.batch_size)

    # ------------------------------------------------------------------ sklearn API
    def fit(self, X, y=None):
        """Fit the featurizer (column kinds, tabular preprocessing, embedder).

        Parameters
        ----------
        X : MongoDataset, cursor, iterable of dict or list of dict
            Training documents. A ``MongoDataset`` is read in its own batches and its ``target``
            is excluded from (or masked in) the embedded JSON. Other inputs are read in batches
            of ``batch_size``.
        y : ignored
            Present for sklearn compatibility; take labels from ``MongoDataset.y``.

        Returns
        -------
        DocumentFeaturizer
            ``self``. The documents are embedded during the fit, so with a cache a following
            ``transform`` of the same documents is cheap.

        Raises
        ------
        TypeError
            If ``columns``, ``embed_fields`` or ``exclude`` is a single string.
        ValueError
            If there are no documents, or neither columns nor embeddings.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit on ``X`` and return its feature matrix, reading the documents once.

        Parameters
        ----------
        X : MongoDataset, cursor, iterable of dict or list of dict
            As in :meth:`fit`.
        y : ignored
        **fit_params : ignored

        Returns
        -------
        numpy.ndarray of float32, shape (n_documents, n_features)
            ``[tabular columns | embedding]``; column names from :meth:`get_feature_names_out`.
        """
        target = X.target if isinstance(X, MongoDataset) else None
        return self._fit_transform_batches(self._batches(X), target)

    def transform(self, X):
        """Feature matrix of ``X`` with the fitted preprocessing and embedder.

        Parameters
        ----------
        X : MongoDataset, cursor, iterable of dict or list of dict
            Documents to featurize. The target is excluded (or masked) as in the fit, so it may
            be present or absent.

        Returns
        -------
        numpy.ndarray of float32, shape (n_documents, n_features)
            An empty ``(0, n_features)`` array when ``X`` has no documents.
        """
        return self.transform_batches(self._batches(X))

    def fit_batches(self, batches):
        """Fit from an iterable of document batches (lists of dicts) without holding them all.

        Batches carry no target, so nothing is excluded automatically: pass the target in
        ``exclude`` (or set ``mask_key`` with ``mask_target=True``). Returns ``self``.
        """
        self.fit_transform_batches(batches)
        return self

    def fit_transform_batches(self, batches) -> np.ndarray:
        """Fit and return the feature matrix, consuming the document batches once.

        Only the values of ``columns`` and the embeddings are kept per batch, so memory stays
        proportional to the features rather than to the raw documents. As in
        :meth:`fit_batches`, no target is known, so it must be in ``exclude``.
        """
        return self._fit_transform_batches(batches, None)

    def _fit_transform_batches(self, batches, target: str | None) -> np.ndarray:
        """Shared body of all fits: one pass that embeds each batch and keeps the raw columns,
        after which column kinds are inferred and the tabular preprocessing fitted."""
        for name in ("columns", "embed_fields", "exclude"):
            if isinstance(getattr(self, name), str):
                raise TypeError(
                    f"{name} must be a list of field paths, not the string {getattr(self, name)!r}"
                )
        for name in ("columns", "embed_fields"):
            paths = getattr(self, name)
            if isinstance(paths, (list, tuple)) and len(set(paths)) < len(paths):
                raise ValueError(f"{name} lists a field more than once: {list(paths)}")
        # every fit (refit included) starts from scratch: no target, tokenizer or embedder
        # left over from a previous fit
        if target and any(c == target or c.startswith(target + ".") for c in self.columns or []):
            raise ValueError(
                f"columns include the target {target!r}: it would leak into the features"
            )
        self.target_ = target
        self.exclude_ = list(self.exclude) + ([target] if target else [])
        if self.mask_target and target is None and self.mask_key:  # plain documents: mask mask_key
            self.exclude_.append(self.mask_key)
        self.tokenizer_ = None
        self.embedder_ = self._embedder()
        frames, embs, n = [], [], 0
        try:
            for docs in batches:
                docs = _as_documents(docs)
                if not docs:
                    continue
                frames.append(self._frame(docs))
                if self.embedder_ is not None:
                    embs.append(self.embedder_.embed(self._texts(docs)))
                n += len(docs)
                if (report := _on_documents.get()) is not None:
                    report(len(docs))
        except KeyboardInterrupt:
            self._interrupted(n)
            raise
        cols = list(self.columns or [])
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        self.column_kinds_ = {c: self._infer_kind(df[c]) for c in df.columns}
        empty = [c for c in df.columns if df[c].notna().sum() == 0]
        if empty and len(df):
            warnings.warn(
                f"columns {empty} have no value in any training document (a dotted path reads "
                "nested fields; field names containing '.' cannot be addressed)",
                stacklevel=3,
            )
        num = [c for c, k in self.column_kinds_.items() if k != "categorical"]
        cat = [c for c, k in self.column_kinds_.items() if k == "categorical"]
        transformers = []
        if num:
            steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
            if self.scale_numeric:
                steps.append(StandardScaler())
            transformers.append(("num", make_pipeline(*steps), num))
        if cat:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=self.min_frequency,
                        max_categories=self.max_categories,
                        sparse_output=False,
                    ),
                    cat,
                )
            )
        self.tabular_ = ColumnTransformer(transformers) if transformers else None
        if self.tabular_ is None and self.embedder_ is None:
            raise ValueError("DocumentFeaturizer has neither columns nor embeddings")
        if n == 0:
            raise ValueError("cannot fit a DocumentFeaturizer on zero documents")
        self.n_tabular_features_ = 0
        if self.tabular_ is not None:
            self.tabular_.fit(self._coerce(df))
            self.n_tabular_features_ = len(self.tabular_.get_feature_names_out())
        self.n_embedding_features_ = 0 if not embs else embs[0].shape[1]
        return self._assemble([df], embs, n)

    def _interrupted(self, n_done: int) -> None:
        """On Ctrl-C (KeyboardInterrupt) while featurizing: say what was kept before re-raising."""
        import sys

        if self.embedder_ is None:
            return
        if getattr(self, "_temporary_cache", False):  # a prediction model fit without a cache
            where = "in a temporary cache that is deleted, so running again embeds them again"
        elif self.cache_dir:
            where = (
                f"cached in {os.path.abspath(self.cache_dir)}, so running the same call again "
                "embeds only the rest"
            )
        else:
            where = "not cached (cache_dir=None), so running again embeds them again"
        print(
            f"\nInterrupted after embedding {n_done:,} documents (and part of the next batch); "
            f"the embeddings computed so far are {where}.",
            file=sys.stderr,
        )

    def _embed(self, docs):
        """Embeddings of one batch, or ``None`` without an embedding block."""
        if self.embedder_ is None:
            return None
        return self.embedder_.embed(self._texts(docs))

    def _check_fitted(self) -> None:
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "tabular_"):
            raise NotFittedError(
                "this DocumentFeaturizer is not fitted yet; call fit or fit_transform first"
            )

    def transform_batches(self, batches) -> np.ndarray:
        """Transform an iterable of document batches.

        Embeddings are computed batch by batch (so a streaming dataset is never held in memory),
        while the column values, which are small, are collected and encoded in a single sklearn
        call at the end: per batch sklearn overhead dominated on collections of small documents.

        Parameters
        ----------
        batches : iterable of list of dict
            Document batches, e.g. from ``MongoDataset.iter_batches``.

        Returns
        -------
        numpy.ndarray of float32, shape (n_documents, n_features)
        """
        self._check_fitted()
        frames, embs, n = [], [], 0
        try:
            for docs in batches:
                docs = _as_documents(docs)
                if not docs:
                    continue
                if self.tabular_ is not None:
                    frames.append(self._frame(docs))
                if self.embedder_ is not None:
                    embs.append(self.embedder_.embed(self._texts(docs)))
                n += len(docs)
                if (report := _on_documents.get()) is not None:
                    report(len(docs))
        except KeyboardInterrupt:
            self._interrupted(n)
            raise
        return self._assemble(frames, embs, n)

    def _assemble(self, frames, embs, n) -> np.ndarray:
        """Encode the collected raw columns and stack them with the embeddings into one float32
        matrix (an empty one with the right width for zero documents)."""
        if n == 0:
            return np.zeros((0, self.n_tabular_features_ + self.n_embedding_features_), np.float32)
        blocks = []
        if self.tabular_ is not None:
            df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            tab = np.asarray(self.tabular_.transform(self._coerce(df)), dtype=np.float64)
            # scaling can push a huge (already clipped) value past float32: keep X finite
            blocks.append(np.clip(tab, -_F32_MAX, _F32_MAX).astype(np.float32))
        if self.embedder_ is not None:
            blocks.append(np.vstack(embs) if len(embs) > 1 else embs[0])
        return np.hstack(blocks) if len(blocks) > 1 else blocks[0]

    def get_feature_names_out(self, input_features=None):
        """Names of the output columns, in order.

        Parameters
        ----------
        input_features : ignored
            Present for sklearn compatibility.

        Returns
        -------
        numpy.ndarray of str (object dtype)
            The tabular names from the ``ColumnTransformer`` (e.g. ``"num__year"``,
            ``"cat__rated_PG"``, ``"cat__rated_infrequent_sklearn"``), then ``"emb_0"`` ...
            ``"emb_<d-1>"`` for the embedding dimensions.
        """
        self._check_fitted()
        names = []
        if self.tabular_ is not None:
            names += list(self.tabular_.get_feature_names_out())
        names += [f"emb_{i}" for i in range(self.n_embedding_features_)]
        return np.asarray(names, dtype=object)
