"""``MongoDataset``: the data side of the sklearn workflow.

A dataset points at a collection (plus ``filter`` / ``projection`` / ``limit`` / ``sort`` and a
``target`` field) or at an explicit list of documents. It knows the target values ``y``, the
``_id`` of each document, and can hand out its documents in batches. Documents whose target is
missing (``None``, NaN, ``pd.NA``, NaT, or an absent field) are dropped when the dataset loads.

Two modes:

* **in memory** (default): :meth:`MongoDataset.load` runs the query once and keeps the
  documents.
* **streaming** (``streaming=True``): only ``_id`` values and targets are kept. Documents are
  fetched again by ``_id`` in batches whenever a consumer iterates over the dataset, optionally
  in a background thread (``prefetch``) so database round trips overlap with featurisation.

How it fits the package: ``DocumentFeaturizer`` and the estimators in ``embeddings_api`` read
documents through :meth:`MongoDataset.iter_batches` and :meth:`MongoDataset.field_values`, and
take labels from :attr:`MongoDataset.y`. scikit-learn's splitters index a dataset through
:meth:`MongoDataset.take` (:meth:`MongoDataset.subset`), so each part is a view on the same source.
The projection helpers at the bottom of this module (``_with_paths``, ``_covers``) make sure the
target and other requested fields are fetched without changing what the user's projection means.

Examples
--------
>>> ds = MongoDataset(db.movies, target="imdb.rating", filter={"year": {"$lt": 2010}})
>>> len(ds), ds.y[:3]
>>> for batch in ds.iter_batches():
...     ...  # lists of at most ``batch_size`` documents
>>> big = MongoDataset(db.logs, target="label", streaming=True, batch_size=256)
>>> small = MongoDataset.from_documents([{"x": 1, "y": 2.0}, {"x": 2, "y": 3.5}], target="y")
"""

from __future__ import annotations

import queue
import threading
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .documents import (
    _fetch_by_ids,
    _fetch_documents,
    _get_path,
    _iter_batches,
    _projection_for,
    _scalar,
    _with_dict_documents,
)


def _is_missing(v) -> bool:
    """Whether a target value counts as missing: None, NaN of any float type, pd.NA and NaT.

    Containers (lists, documents) are never missing; ``pd.isna`` would return an array for them.
    """
    if v is None:
        return True
    if isinstance(v, str):  # blank text (placeholders: see _numeric_placeholders)
        return not v.strip()
    if isinstance(v, (float, np.floating)) and np.isinf(v):
        return True
    try:
        return np.ndim(v) == 0 and bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


#: text that stands for a missing value of a numeric target (compared stripped and lower-cased)
MISSING_TEXT = frozenset({"", "n/a", "na", "null", "none", "nan", "-", "?"})


def _numeric_placeholders(values: list) -> set:
    """The placeholder strings (:data:`MISSING_TEXT`) among ``values`` when every other value is
    a number or numeric text; empty otherwise (the placeholders are then labels)."""
    holes, numeric = set(), 0
    for v in values:
        if isinstance(v, str) and v.strip().lower() in MISSING_TEXT:
            holes.add(v)
        elif isinstance(v, (bool, np.bool_)):
            return set()
        elif isinstance(v, (int, float, np.integer, np.floating)):
            numeric += 1
        elif isinstance(v, str) and _is_number_text(v):
            numeric += 1
        else:
            return set()
    return holes if numeric else set()


def _is_number_text(v: str) -> bool:
    try:
        float(v.strip().strip("$€£¥").replace(",", ""))
        return True
    except ValueError:
        return False


def _as_target_array(values: list) -> np.ndarray:
    """Convert raw target values into an array that sklearn estimators accept.

    Parameters
    ----------
    values : list
        Target values, one per document, with missing values already removed.

    Returns
    -------
    numpy.ndarray
        A ``bool`` array when every value is a boolean, a ``float`` array when every value is a
        (non boolean) int or float, and otherwise an ``object`` array of ``str(value)``. Mixed
        types (e.g. ``1`` and ``"a"``) therefore become strings, a classification target. An
        empty list gives an empty ``object`` array.

    Examples
    --------
    >>> _as_target_array([1, 2.5]).dtype
    dtype('float64')
    >>> _as_target_array([True, False]).dtype
    dtype('bool')
    >>> _as_target_array(["a", 1])
    array(['a', '1'], dtype=object)
    """
    bad = next((v for v in values if isinstance(v, (list, tuple, dict))), None)
    if bad is not None:
        raise ValueError(f"target values must be single values, not arrays or documents: {bad!r}")
    # booleans first: bool is a subclass of int and would otherwise be read as numeric
    if values and all(isinstance(v, (bool, np.bool_)) for v in values):
        return np.asarray(values, dtype=bool)
    if values and all(
        isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
        for v in values
    ):
        exact = all(abs(int(v)) <= 2**53 for v in values if isinstance(v, (int, np.integer)))
        if exact:
            return np.asarray(values, dtype=float)
        # integer codes beyond float precision would merge: keep them as exact labels
    return np.asarray([_label(v) for v in values], dtype=object)


def _label(v) -> str:
    """One spelling per label: 1, 1.0 and "1" are "1"; True and "true" are "true"."""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower()
    return str(v)


class MongoDataset:
    """Documents of a collection (or an explicit list) together with an optional target.

    The query is lazy: nothing is read until the first access to ``len()``, :attr:`ids`,
    :attr:`y`, :meth:`iter_batches` or :meth:`field_values` (or an explicit :meth:`load`), and the
    result is cached.

    Parameters
    ----------
    collection : pymongo ``Collection`` or None
        Anything with ``.find`` (mongomock works). ``None`` for in-memory documents (see
        :meth:`from_documents`).
    target : str, optional
        Dotted path of the field to predict (e.g. ``"imdb.rating"``). Documents without a value
        are dropped. ``None`` (default) for an unlabelled dataset; :attr:`y` is then ``None``.
    filter, projection, limit, sort : optional
        Forwarded to ``collection.find``. The ``projection`` defines what a document looks like
        in both modes; the target field is added to an inclusion projection (or removed from the
        excluded fields of an exclusion projection) so ``y`` can always be read. ``limit`` of
        ``None`` or 0 means no limit.
    streaming : bool, default False
        If ``True`` the documents are never held in memory all at once. :meth:`load` only reads
        ``_id`` and the target; documents are fetched again in ``_id`` batches whenever an
        estimator iterates over the dataset (with a projection limited to the fields it uses).
        Use it when documents are much larger than the features derived from them. The
        collection must not change in between: a document that disappeared raises ``KeyError``.
    batch_size : int, default 512
        Documents per batch in :meth:`iter_batches` (and per ``_id`` query in streaming mode).
    prefetch : int, default 2
        In streaming mode, number of batches fetched ahead in a background thread so the
        database round trip overlaps with featurisation / embedding (0 disables).

    Raises
    ------
    ValueError
        If ``batch_size`` is smaller than 1.

    Notes
    -----
    Each document is identified by its ``_id``. When a document has no ``_id`` (in-memory
    documents, or an in-memory projection with ``"_id": 0``) its position is used instead.

    Examples
    --------
    >>> ds = MongoDataset(db.movies, target="imdb.rating", projection={"title": 1, "plot": 1})
    >>> ds.y.dtype
    dtype('float64')
    >>> ds.field_values("title")[:2]
    """

    def __init__(
        self,
        collection,
        target: str | None = None,
        filter: dict | None = None,
        projection: dict | None = None,
        limit: int | None = None,
        sort=None,
        streaming: bool = False,
        batch_size: int = 512,
        prefetch: int = 2,
    ):
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.collection = None if collection is None else _with_dict_documents(collection)
        self.target = target
        self.filter = filter
        self.projection = projection
        self.limit = limit
        self.sort = sort
        self.streaming = streaming
        self.batch_size = batch_size
        self.prefetch = prefetch
        self._docs: list[dict] | None = None
        self._ids: list | None = None
        self._y: np.ndarray | None = None
        self._light: dict[str, list] = {}  # streaming: field path -> values aligned with ids
        self._view = False  # a subset: knows its ids only, cannot reload its query

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_documents(cls, documents: Iterable[dict], target: str | None = None, y=None):
        """In-memory dataset from a list / cursor of documents.

        Parameters
        ----------
        documents : iterable of dict
            Documents (a list, a pymongo cursor, a generator). It is read once into a list.
        target : str, optional
            Dotted path of the label inside each document. Documents missing it are dropped.
            Ignored for labels when ``y`` is given, but still stored as :attr:`target`.
        y : array-like, optional
            Labels given separately, one per document (no document is dropped). Converted with
            :func:`_as_target_array`.

        Returns
        -------
        MongoDataset
            A dataset with no collection; it cannot be reloaded, and fields left out of the
            documents cannot be fetched later.

        Raises
        ------
        ValueError
            If ``y`` is not one dimensional or its length differs from the number of documents.

        Examples
        --------
        >>> ds = MongoDataset.from_documents([{"a": 1, "t": "x"}, {"a": 2}], target="t")
        >>> len(ds)
        1
        >>> ds = MongoDataset.from_documents([{"a": 1}, {"a": 2}], y=[0.5, 1.5])
        """
        docs = list(documents)
        ds = cls(None, target=target)
        if y is not None:
            if np.ndim(y) != 1:
                raise ValueError("y must be one dimensional")
            y = _as_target_array(list(y))
            if len(y) != len(docs):
                raise ValueError("y must have one value per document")
        elif target is not None:
            docs, y = ds._with_target(docs)
        ds._docs, ds._ids, ds._y = docs, [d.get("_id", i) for i, d in enumerate(docs)], y
        ds._raw_y = None if target is None else [_get_path(d, target) for d in docs]
        return ds

    # ------------------------------------------------------------------ loading
    def _target_values(self, docs):
        """Raw target of each document, with BSON scalars (Decimal128, ObjectId) made plain."""
        return [_scalar(_get_path(d, self.target)) for d in docs]

    def _with_target(self, docs):
        """Drop documents with a missing target; return ``(kept_docs, y)`` aligned."""
        y = self._target_values(docs)
        keep = [i for i, v in enumerate(y) if not _is_missing(v)]
        # "N/A", "null", "-" ... are missing values of a numeric target, but can be real labels
        # of a categorical one ("NA" for North America, "none" for a severity)
        holes = _numeric_placeholders([y[i] for i in keep])
        if holes:
            keep = [i for i in keep if not (isinstance(y[i], str) and y[i] in holes)]
        return [docs[i] for i in keep], _as_target_array([y[i] for i in keep])

    def load(self, force: bool = False):
        """Run the query and cache the result.

        In memory, the documents are fetched with the dataset's projection (extended to the
        target). When streaming, only ``_id`` and the target are fetched. Documents with a
        missing target are dropped. A reload clears the cache of :meth:`field_values`.

        Parameters
        ----------
        force : bool, default False
            Run the query again even if the dataset is already loaded (e.g. after the
            collection changed).

        Returns
        -------
        MongoDataset
            ``self``, so calls can be chained (``ds.load().y``).

        Raises
        ------
        ValueError
            If the dataset is a subset (from :meth:`subset` or a scikit-learn split), which
            only knows its ids, or if it has no collection and was not built with
            :meth:`from_documents`.
        """
        if self._ids is not None and not force:
            return self
        if self._view:
            raise ValueError(
                "a subset (e.g. a train or test part) cannot be reloaded; "
                "reload and split its parent"
            )
        if self.collection is None:
            raise ValueError("this dataset has no collection; build it with from_documents")
        if self.streaming:
            projection = self._light_projection(self.target) if self.target else {"_id": 1}
        else:
            projection = _with_paths(self.projection, [self.target] if self.target else [])
        docs = _fetch_documents(self.collection, self.filter, projection, self.limit, self.sort)
        y = None
        if self.target is not None:
            docs, y = self._with_target(docs)
        # a projection may leave out _id in memory (positions stand in); streaming always
        # fetches it
        self._ids = [d.get("_id", i) for i, d in enumerate(docs)]
        self._light = {}  # field values cached for the previous load are stale now
        self._y = y
        # the stored target values of the same read as y (an ObjectId, a date, an int)
        self._raw_y = None if y is None else [_get_path(d, self.target) for d in docs]
        self._docs = None if self.streaming else docs
        return self

    # ------------------------------------------------------------------ basic access
    def __len__(self) -> int:
        """Number of documents (after dropping those with a missing target). Loads if needed."""
        return len(self.load()._ids)

    @property
    def ids(self) -> list:
        """``_id`` of every document, in dataset order (a copy).

        Documents without an ``_id`` (in-memory documents, or a projection with ``"_id": 0``)
        are identified by their position in the query result instead.
        """
        return list(self.load()._ids)

    @property
    def y(self) -> np.ndarray | None:
        """Target values aligned with :attr:`ids` (``None`` when the dataset has no target).

        Numeric targets come back as a float array, booleans as bool, anything else as strings,
        so the array can go straight into an sklearn estimator.
        """
        return self.load()._y

    @property
    def documents(self) -> list[dict]:
        """All documents as a list, in dataset order (a new list on every access).

        In streaming mode this fetches every document (defeating the purpose), so prefer
        :meth:`iter_batches`.
        """
        return [d for batch in self.iter_batches() for d in batch]

    def iter_batches(self, projection: dict | None = None):
        """Yield lists of at most ``batch_size`` documents, in dataset order.

        Parameters
        ----------
        projection : dict, optional
            Fields the consumer needs, used to limit what is fetched in streaming mode. It is
            ignored for in-memory documents (already loaded) and for datasets that have their
            own ``projection``, since that projection defines the documents in both modes.

        Yields
        ------
        list of dict
            Consecutive batches. In streaming mode each batch is one ``_id`` query; with
            ``prefetch > 0`` and more than one batch, the next batches are fetched in a
            background thread while the current one is processed.

        Raises
        ------
        KeyError
            In streaming mode, if documents were deleted from the collection after loading.

        Examples
        --------
        >>> for batch in ds.iter_batches(projection={"title": 1}):
        ...     texts = [d.get("title") for d in batch]
        """
        self.load()
        if not self.streaming:
            yield from _iter_batches(self._docs, self.batch_size)
            return
        if self.projection is not None or projection is None:
            # the dataset's own projection defines the documents (as in memory); a consumer's
            # narrower projection is only an optimisation for unprojected datasets
            projection = _with_paths(self.projection, [self.target] if self.target else [])
        chunks = [
            self._ids[i : i + self.batch_size] for i in range(0, len(self._ids), self.batch_size)
        ]
        if self.prefetch <= 0 or len(chunks) <= 1:
            for chunk in chunks:
                yield _fetch_by_ids(self.collection, chunk, projection)
            return
        yield from _prefetched(
            # c=c binds each chunk now; a plain closure would see only the last chunk
            (lambda c=c: _fetch_by_ids(self.collection, c, projection) for c in chunks),
            self.prefetch,
        )

    def field_values(self, path: str) -> list:
        """Values of a dotted field for every document, aligned with :attr:`ids`.

        Used for stratify or group values in scikit-learn splits, and by featurizers that
        read single columns. Streaming datasets fetch just that field (once, then cached until
        the next :meth:`load`) with a projection, and so do in-memory datasets whose
        ``projection`` left the field out, so both modes see the same values.

        Parameters
        ----------
        path : str
            Dotted field path, e.g. ``"host.host_id"`` or ``"scores.0"``.

        Returns
        -------
        list
            One value per document (a new list). Missing fields give ``None``; BSON scalars
            (Decimal128, ObjectId) are converted to plain Python values.

        Raises
        ------
        ValueError
            For an in-memory dataset whose projection leaves out both ``path`` and ``_id`` (or
            that has no collection), since the field then cannot be fetched by ``_id``.
        """
        if not isinstance(path, str) or not path or any(not p for p in path.split(".")):
            raise ValueError(f"a field path must be a non-empty dotted string, got {path!r}")
        self.load()
        if not self.streaming and _covers(self.projection, path):
            return [_scalar(_get_path(d, path)) for d in self._docs]
        no_ids = self.collection is None or (self.projection or {}).get("_id", 1) in (0, False)
        if not self.streaming and no_ids:
            raise ValueError(
                f"the projection leaves out {path!r} (and _id, so it cannot be fetched)"
            )
        if path not in self._light:
            # fetched without prefetching: one small field per document is cheap to read
            values = []
            for i in range(0, len(self._ids), self.batch_size):
                chunk = self._ids[i : i + self.batch_size]
                for d in _fetch_by_ids(self.collection, chunk, self._light_projection(path)):
                    values.append(_scalar(_get_path(d, path)))
            self._light[path] = values
        return list(self._light[path])

    def _light_projection(self, path: str) -> dict | None:
        """The projection to read just ``path`` by ``_id``.

        It is the dataset's own projection when that computes the path or an ancestor
        (``$slice``, ``$elemMatch``, an expression), so both modes see the same values;
        otherwise only the path. ``_id`` is never projected out, since documents are matched
        by it.
        """
        # computed = projected keys whose value is not a plain 0/1 flag and that hold ``path``
        flat = _flatten(self.projection) if self.projection else {}
        computed = [
            k
            for k, v in flat.items()
            if not isinstance(v, (bool, int, float)) and (path == k or path.startswith(k + "."))
        ]
        if not computed:
            return _projection_for([path])
        try:
            out = _with_paths(self.projection, [path])
        except ValueError:  # the path is computed by an expression: read it as the projection does
            out = self.projection
        # documents are matched by _id: never project it out here
        out = {k: v for k, v in (out or {}).items() if not (k == "_id" and v in (0, False))}
        return out or None

    # ------------------------------------------------------------------ views
    def subset(self, idx: Sequence[int]) -> "MongoDataset":
        """Dataset restricted to positional indices.

        The subset shares the collection, query settings and mode (in memory or streaming),
        and copies the matching ids, targets, documents and cached :meth:`field_values`. It is a
        view: it cannot :meth:`load` its query again.

        Parameters
        ----------
        idx : sequence of int
            Positions into this dataset (e.g. from a scikit-learn splitter). Order is kept and
            repeats are allowed.

        Returns
        -------
        MongoDataset
            A new dataset with ``len(idx)`` documents.

        Examples
        --------
        >>> first_ten = ds.subset(range(10))
        """
        self.load()
        arr = np.asarray(idx)
        if arr.dtype == bool:  # a boolean mask over the documents
            if arr.shape != (len(self),):
                raise ValueError(f"a boolean mask needs {len(self)} values, got {arr.shape}")
            arr = np.flatnonzero(arr)
        elif arr.size and not np.issubdtype(arr.dtype, np.integer):
            raise IndexError(f"positions must be integers or a boolean mask, got {arr.dtype}")
        idx = [int(i) for i in arr.ravel()]
        if any(i < 0 or i >= len(self) for i in idx):
            raise IndexError(f"positions must be in [0, {len(self)}), got {idx[:5]}...")
        ds = MongoDataset(
            self.collection,
            self.target,
            self.filter,
            self.projection,
            self.limit,
            self.sort,
            self.streaming,
            self.batch_size,
            self.prefetch,
        )
        ds._view = True
        ds._ids = [self._ids[i] for i in idx]
        ds._y = None if self._y is None else self._y[idx]
        raw = getattr(self, "_raw_y", None)
        ds._raw_y = None if raw is None else [raw[i] for i in idx]
        ds._docs = None if self._docs is None else [self._docs[i] for i in idx]
        ds._light = {k: [v[i] for i in idx] for k, v in self._light.items()}
        return ds

    # ------------------------------------------------------------------ scikit-learn indexing
    # scikit-learn's splitters and model_selection functions index their inputs through ``take`` /
    # ``iloc`` (the pandas interface), so train_test_split, GroupShuffleSplit, TimeSeriesSplit,
    # cross_val_score, ... return MongoDataset subsets

    @property
    def shape(self) -> tuple[int]:
        """``(n_documents,)``, as scikit-learn expects of an array-like (loads the dataset)."""
        return (len(self),)

    def take(self, indices, axis: int = 0) -> "MongoDataset":
        """The documents at ``indices`` as a new dataset (:meth:`subset`); ``axis`` must be 0.

        This is what ``sklearn.model_selection.train_test_split`` and the other scikit-learn
        splitters call, so they split a dataset into datasets:

        >>> from sklearn.model_selection import train_test_split
        >>> train, test = train_test_split(ds, test_size=0.2, stratify=ds.y, random_state=0)
        """
        if axis != 0:
            raise ValueError("a MongoDataset is indexed by document (axis=0) only")
        return self.subset(indices)

    @property
    def iloc(self):
        """Positional indexing as in pandas: ``ds.iloc[10:20]`` or ``ds.iloc[[0, 5]]``."""
        return _ILoc(self)

    def __getitem__(self, key) -> "MongoDataset":
        """``ds[indices]``, ``ds[mask]`` or ``ds[start:stop]``: a subset (:meth:`subset`)."""
        return _ILoc(self)[key]

    def __repr__(self) -> str:
        """Short description; does not trigger a load (``n=?`` until the dataset is loaded)."""
        n = len(self._ids) if self._ids is not None else "?"
        src = getattr(self.collection, "full_name", None) or (
            "documents" if self.collection is None else "collection"
        )
        mode = ", streaming" if self.streaming else ""
        return f"MongoDataset({src}, n={n}, target={self.target!r}{mode})"


class _ILoc:
    """``MongoDataset.iloc``: integer positions, a slice, or a boolean mask give a subset."""

    def __init__(self, ds: MongoDataset):
        self.ds = ds

    def __getitem__(self, key) -> MongoDataset:
        if isinstance(key, tuple):  # ds[idx, ...] or ds[idx, :]
            key, *rest = key
            if any(r is not Ellipsis and r != slice(None) for r in rest):
                raise IndexError("a MongoDataset has one dimension (documents)")
        if isinstance(key, slice):
            return self.ds.subset(range(*key.indices(len(self.ds))))
        if isinstance(key, (int, np.integer)):
            raise TypeError(
                "index a MongoDataset with a list, array, slice or mask; for one document use "
                "ds.documents[i] (or ds.iloc[[i]] for a dataset of one)"
            )
        return self.ds.subset(key)


def _is_operator(v) -> bool:
    """Whether a projection value is an operator (``{"$slice": 2}``), not a nested projection."""
    return isinstance(v, dict) and any(str(k).startswith("$") for k in v)


def _covers(projection: dict | None, path: str) -> bool:
    """Whether documents fetched with ``projection`` hold ``path`` (as far as it exists).

    Tells :meth:`MongoDataset.field_values` whether it can read in-memory documents or must
    fetch the field by ``_id``.
    """
    # the projection covers the path when extending it to the path changes nothing
    if not projection:
        return True
    flat = _flatten(projection)
    try:
        extended = _with_paths(projection, [path])
    except ValueError:  # computed by the projection: the fetched value is the one asked for
        return True
    return extended is projection or extended == flat


def _flatten(projection: dict, prefix: str = "") -> dict:
    """Nested projections (``{"a": {"b": 1}}``) as dotted keys (``{"a.b": 1}``)."""
    out = {}
    for k, v in projection.items():
        if isinstance(v, dict) and v and not _is_operator(v):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _with_paths(projection: dict | None, paths: list[str]) -> dict | None:
    """``projection`` changed so that fetched documents hold ``paths`` (the target).

    Needed because the user's projection decides what features see, but ``y`` must still be
    readable. The projection is classified as MongoDB reads it:

    * Inclusion: any field other than ``_id`` with a true value (1, 1.0, True, a field path or
      an expression, ``$elemMatch``; ``$slice`` alone does not include), or ``_id`` alone with a
      true value. The paths are added (as flattened dotted keys) unless they or an ancestor
      are already included.
    * Exclusion: every other field is kept. The projection is returned unchanged unless it
      excludes a path, an ancestor or a descendant of it; those exclusions are removed (and
      ``None`` is returned if nothing else is left).

    Returns ``projection`` itself (the same object) when it or ``paths`` is empty, or when an
    exclusion projection leaves the paths alone. An inclusion projection always comes back as
    a new, flattened dict.

    Raises
    ------
    ValueError
        If an ancestor of a path is computed by an expression, which would replace the stored
        value.
    """
    if not projection or not paths:
        return projection

    def includes(v) -> bool:
        """Whether a projection value selects its field (an inclusion, not an exclusion)."""
        if isinstance(v, (bool, int, float)):
            return bool(v)
        return not (_is_operator(v) and set(v) == {"$slice"})

    flat = _flatten(projection)
    fields = {k: v for k, v in flat.items() if k != "_id"}
    if fields:
        inclusion = any(includes(v) for v in fields.values())
        # an explicit _id: 1 makes a projection of $slice fields an inclusion (as on the server)
        if (
            "_id" in flat
            and includes(flat["_id"])
            and not any(v in (0, False) for v in fields.values())
        ):
            inclusion = True
    else:
        inclusion = includes(flat.get("_id", 0))
    if not inclusion:  # an exclusion projection must not exclude the target (or part of it)
        targets = [next(iter(_projection_for([t])), t) for t in paths]
        related = [  # (only 0 / False excludes: $slice keeps the field)
            k
            for k in flat
            if k != "_id"
            and flat[k] in (0, False)
            and any(k == t or t.startswith(k + ".") or k.startswith(t + ".") for t in targets)
        ]
        if not related:
            return projection
        rest = {k: v for k, v in flat.items() if k not in related}
        if set(rest) <= {"_id"} and rest.get("_id", 0) not in (0, False):
            return None  # {"_id": 1} alone would be an inclusion: no projection at all instead
        return rest or None
    out = dict(flat)
    for target in paths:
        path = next(
            iter(_projection_for([target])), None
        )  # array indices cut: "scores.0.x" -> "scores"
        if path is None:
            continue
        ancestors = [k for k in out if path == k or path.startswith(k + ".")]
        if any(isinstance(out[k], (bool, int, float)) and out[k] for k in ancestors):
            continue  # included through itself or an ancestor
        if any(_is_operator(out[k]) and set(out[k]) <= {"$slice", "$elemMatch"} for k in ancestors):
            continue  # an ancestor array projected with $slice / $elemMatch holds the target
        if ancestors:  # an ancestor computed by an expression would replace the target
            raise ValueError(
                f"the projection computes {ancestors[0]!r}, which holds the target {target!r}"
            )
        for k in [k for k in out if k.startswith(path + ".")]:
            del out[k]  # a descendant would collide with the path
        out[path] = 1
    return out


def _prefetched(jobs, depth: int):
    """Run ``jobs`` (callables) in a background thread, ``depth`` results ahead of the consumer.

    Results are yielded in order. An exception in a job is re-raised in the consumer. When the
    consumer stops early (``break``, garbage collection), ``stop`` lets the worker exit instead
    of blocking forever on a full queue.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()
    stop = threading.Event()  # set when the consumer finishes or stops early

    def put(item) -> bool:
        """Queue ``item``; ``False`` if the consumer stopped first (short timeouts poll it)."""
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def worker():
        """Run the jobs in order, then always queue the ``done`` marker."""
        try:
            for job in jobs:
                if stop.is_set() or not put(job()):
                    return
        except BaseException as e:  # surface in the consumer thread
            put(e)
        finally:
            put(done)

    threading.Thread(target=worker, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
