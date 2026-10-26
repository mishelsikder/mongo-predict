"""Helpers for reading MongoDB documents and turning them into plain Python / JSON.

Every other module of the package goes through these helpers to touch raw documents, so the
path rules below hold everywhere (dataset targets, featurizer columns and embedded fields,
time splits, ``MongoSeries`` and the ``$predict`` stage).

Main pieces
-----------
Reading values
    :func:`_get_path` / :func:`_has_path` read a value at a dotted path, and
    :func:`_compile_path` + :func:`_get_compiled` do the same faster when one path is read from
    many documents (the featurizer's columns). :func:`_scalar` and :func:`_to_plain` turn BSON and
    numpy values into plain Python values.
Removing and selecting fields
    :func:`_without_paths` and :func:`_select_fields` drop the prediction target (and ``_id``)
    before a document is embedded, with MongoDB's array semantics.
Serialising for the embedding model
    :func:`_dumps`, :func:`_json_pieces`, :func:`_assemble_json` and :func:`_document_to_json` build
    the JSON text that the embedding model reads, within a budget in characters or tokens.
Talking to a collection
    :func:`_fetch_documents`, :func:`_fetch_by_ids`, :func:`_projection_for`,
    :func:`_as_documents`, :func:`_iter_batches` and :func:`_get_collection`.

Path semantics
--------------
A path is a dotted string such as ``"address.market"`` or ``"scores.0.score"``.

* A part made of ASCII digits indexes an array only where the document holds a list. On a
  sub-document it is an ordinary key (``"2020"`` in ``{"by_year": {"2020": 5}}``).
* When reading (:func:`_get_path`), a non-numeric part on an array finds nothing.
* When removing (:func:`_without_paths`, :func:`_select_fields`), a non-numeric part on an array
  applies to every element, as in MongoDB queries, so a nested target such as
  ``"patient.reaction.reactionoutcome"`` is removed from each reaction and never leaks into the
  embedded JSON.

Examples
--------
>>> doc = {"_id": 1, "name": "Flat", "address": {"market": "Porto"}, "price": 80}
>>> _get_path(doc, "address.market")
'Porto'
>>> _document_to_json(doc, exclude=["price"])
'{"name":"Flat","address":{"market":"Porto"}}'
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

import bson  # ships with pymongo, a dependency
from bson import Decimal128, ObjectId

# Sentinel for "no value at this path", distinct from a stored ``None`` / BSON null.
MISSING = object()


def _decimal_to_float(d) -> float:
    """``Decimal128`` as a float; NaN for the NaN values (a signalling NaN cannot be converted)."""
    x = d.to_decimal()
    return float("nan") if x.is_nan() else float(x)


def _binary_text(value) -> str:
    """Binary data in the embedded text: UUIDs as their usual form, anything else as a short
    placeholder (base64 of an image or a file would fill the text budget with noise)."""
    subtype = getattr(value, "subtype", None)
    if subtype in (3, 4) and len(value) == 16:
        import uuid

        return str(uuid.UUID(bytes=bytes(value)))
    return f"<binary {len(value)} bytes>"


def _is_index(part: str) -> bool:
    """Whether a path part can index an array.

    Only ASCII digits count: ``str.isdigit`` alone would also accept characters such as ``"²"``,
    which ``int`` cannot parse.

    Parameters
    ----------
    part : str
        One part of a dotted path.

    Returns
    -------
    bool
        ``True`` for parts such as ``"0"`` or ``"12"``.
    """
    return part.isascii() and part.isdigit()


def _get_path(doc: Any, path: str, default: Any = None) -> Any:
    """Return the value at a dotted ``path`` such as ``"address.market"`` or ``"coords.0"``.

    Each part is a key on a dict. A numeric part indexes a list or tuple; on a dict it is looked
    up as a key. A non-numeric part on an array does not map over the elements here: the path
    is treated as missing.

    Parameters
    ----------
    doc : Any
        The document (usually a dict as returned by pymongo).
    path : str
        Dotted path, e.g. ``"imdb.rating"`` or ``"scores.0.score"``.
    default : Any, default None
        Returned when any part of the path is missing or out of range. Pass :data:`MISSING` to
        tell a missing path apart from a stored ``None``.

    Returns
    -------
    Any
        The value at ``path``, or ``default``.

    Examples
    --------
    >>> doc = {"address": {"market": "Porto"}, "coords": [1.5, 2.5]}
    >>> _get_path(doc, "address.market"), _get_path(doc, "coords.1")
    ('Porto', 2.5)
    >>> _get_path(doc, "address.street", "n/a")
    'n/a'
    """
    cur = doc
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and _is_index(part):
            idx = int(part)
            if idx >= len(cur):
                return default
            cur = cur[idx]
        else:
            return default
    return cur


def _has_path(doc: Any, path: str) -> bool:
    """Whether ``doc`` has a value (possibly ``None``) at the dotted ``path``.

    Parameters
    ----------
    doc : Any
        The document.
    path : str
        Dotted path, read as in :func:`_get_path`.

    Returns
    -------
    bool
        ``True`` when the path exists, even if the stored value is ``None``.
    """
    return _get_path(doc, path, MISSING) is not MISSING


def _to_plain(value: Any) -> Any:
    """Recursively convert BSON / numpy values into JSON serialisable Python values.

    Conversions: dict keys become ``str``, tuples become lists, ``Decimal128`` becomes
    ``float``, ``ObjectId`` becomes its hex string, ``datetime`` becomes an ISO 8601 string,
    ``bytes`` (and BSON ``Binary``) become base64 text, numpy scalars and arrays become Python
    scalars and lists. Other values are returned unchanged.

    Parameters
    ----------
    value : Any
        A document, a list, or a single value.

    Returns
    -------
    Any
        The converted value; containers are new objects.
    """
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if Decimal128 and isinstance(value, Decimal128):
        return _decimal_to_float(value)
    if ObjectId and isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return _binary_text(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _scalar(value: Any) -> Any:
    """Convert a single BSON scalar into something pandas understands.

    Only ``Decimal128`` (to ``float``) and ``ObjectId`` (to ``str``) are converted; datetimes and
    other values are returned unchanged, since pandas handles them already.

    Parameters
    ----------
    value : Any
        A value read from a document.

    Returns
    -------
    Any
        The converted value.
    """
    if Decimal128 and isinstance(value, Decimal128):
        return _decimal_to_float(value)
    if ObjectId and isinstance(value, ObjectId):
        return str(value)
    return value


def _without_paths(doc: dict, paths: Iterable[str]) -> dict:
    """Copy of ``doc`` with the given dotted paths removed (only touched branches are copied).

    Paths may go through arrays, so a nested target never leaks into the embedded JSON: a numeric
    part indexes the array (``"scores.0.score"``), any other part applies to every element, as in
    MongoDB queries (``"patient.reaction.reactionoutcome"`` removes the field from each reaction).

    Paths covered by another path are skipped, and array elements are removed from the highest
    index down, so earlier removals never shift the position of later ones. Paths that do not
    exist are ignored. The input document is never modified.

    Parameters
    ----------
    doc : dict
        The document.
    paths : iterable of str
        Dotted paths to remove.

    Returns
    -------
    dict
        A new top-level dict (even when nothing was removed). Untouched sub-documents and arrays
        are shared with ``doc``, not copied.

    Examples
    --------
    >>> doc = {"scores": [{"score": 3, "g": 1}, {"score": 4}], "coords": [1, 2]}
    >>> _without_paths(doc, ["scores.score", "coords.0"])
    {'scores': [{'g': 1}, {}], 'coords': [2]}
    """
    paths = _outermost(paths, doc)
    out = doc
    for path in _removal_order(paths):
        out = _remove_path(out, path.split("."))
    return out if out is not doc else dict(doc)


def _removal_order(paths: Iterable[str]) -> list[str]:
    """Higher array indices first, so removing ``arr.0`` does not shift ``arr.1``."""
    return sorted(
        paths, key=lambda p: [(0, -int(x)) if _is_index(x) else (1, x) for x in p.split(".")]
    )


def _remove_path(node, parts):
    """``node`` without ``parts``; returns ``node`` itself when nothing under it changed.

    Returning the same object signals "unchanged" to the caller, which then avoids copying.
    """
    head, rest = parts[0], parts[1:]
    if isinstance(node, tuple):  # (in-memory documents may hold tuples where BSON has arrays)
        new = _remove_path(list(node), parts)
        return node if new == list(node) else new
    if isinstance(node, dict):
        if head not in node:
            return node
        if not rest:
            new = dict(node)
            del new[head]
            return new
        child = node[head]
        if not isinstance(child, (dict, list, tuple)):
            return node
        new_child = _remove_path(child, rest)
        if new_child is child:
            return node
        new = dict(node)
        new[head] = new_child
        return new
    if isinstance(node, list):
        if _is_index(head):
            i = int(head)
            if i >= len(node):
                return node
            if not rest:
                return node[:i] + node[i + 1 :]
            child = node[i]
            if not isinstance(child, (dict, list, tuple)):
                return node
            new_child = _remove_path(child, rest)
            if new_child is child:
                return node
            new = list(node)
            new[i] = new_child
            return new
        # non-numeric part on an array: apply to every element (MongoDB path semantics)
        changed, new = False, []
        for el in node:
            new_el = _remove_path(el, parts) if isinstance(el, (dict, list, tuple)) else el
            changed |= new_el is not el
            new.append(new_el)
        return new if changed else node
    return node


def _select_fields(doc: dict, fields: Sequence[str] | None, exclude: Iterable[str] = ()) -> dict:
    """Sub-document with either the given dotted ``fields`` or all fields minus ``exclude``.

    With explicit ``fields`` the dotted paths are kept as keys (``{"address.market": "Porto"}``)
    so the JSON stays flat and the field name still carries meaning for the embedding model.
    ``exclude`` entries may be dotted paths (e.g. a nested prediction target); ``_id`` is always
    dropped.

    Parameters
    ----------
    doc : dict
        The document.
    fields : sequence of str or None
        Dotted paths to keep, in this order. ``None`` keeps the whole document (minus
        ``exclude``) with its original nesting.
    exclude : iterable of str, default ()
        Dotted paths to remove, with the array semantics of :func:`_without_paths`. A field that
        lies inside an excluded path is dropped; an excluded path inside a selected field is
        removed from that field's value.

    Returns
    -------
    dict
        The selected sub-document. Fields missing from ``doc`` are left out.

    Examples
    --------
    >>> doc = {"_id": 1, "address": {"market": "Porto"}, "scores": [{"score": 3, "g": 1}]}
    >>> _select_fields(doc, ["address.market", "scores"], exclude=["scores.g"])
    {'address.market': 'Porto', 'scores': [{'score': 3}]}
    """
    exclude = set(exclude) | {"_id"}
    if fields is None:
        return _without_paths(doc, exclude)
    out = {}
    exclude = _outermost(exclude, doc)
    for f in fields:
        relations = [_relation(f.split("."), e.split("."), doc) for e in _removal_order(exclude)]
        if any(r == () for r in relations):  # the field is (inside) an excluded one
            continue
        v = _get_path(doc, f, MISSING)
        if v is MISSING:
            continue
        for rest in relations:  # excluded paths inside the field (e.g. a nested target)
            if rest and isinstance(v, (dict, list)):
                v = _remove_path(v, list(rest))
        out[f] = v
    return out


def _outermost(paths: Iterable[str], doc=None) -> list[str]:
    """Paths not covered by another one (``a.0.b`` is gone with ``a``, ``a.0.b.0`` with ``a.b.0``),
    so removing them one by one never hits an element shifted by an earlier removal."""
    paths = set(paths)
    return [
        p
        for p in paths
        if not any(q != p and _relation(p.split("."), q.split("."), doc) == () for q in paths)
    ]


def _relation(field: list[str], excluded: list[str], doc=None):
    """How an excluded path relates to a selected field, with MongoDB's array semantics (a
    non-numeric part of the excluded path applies to every element, so it matches an index in
    the field): ``()`` when it covers the whole field, the remaining parts when it lies inside
    the field, ``None`` when unrelated. With ``doc``, a numeric field part counts as an array
    index only where ``doc`` holds a list (``"2020"`` may be a key of a sub-document)."""
    i = j = 0
    while i < len(field) and j < len(excluded):
        if field[i] == excluded[j]:
            i, j = i + 1, j + 1
        elif _is_index(field[i]) and not _is_index(excluded[j]) and _is_list_at(doc, field[:i]):
            i += 1  # the field picks one element of an array the excluded path maps over
        else:
            return None
    if j == len(excluded):
        return ()
    return tuple(excluded[j:])


def _is_list_at(doc, parts: list[str]) -> bool:
    """Whether ``doc`` holds an array at ``parts``, following paths as removal does (a field name
    on an array maps over its elements)."""
    if doc is None:  # no document: a numeric part is taken as an array index
        return True
    # Follow every branch an array fans out into; one list among them is enough.
    nodes = [doc]
    for part in parts:
        reached = []
        for n in nodes:
            if isinstance(n, dict):
                if part in n:
                    reached.append(n[part])
            elif isinstance(n, (list, tuple)):
                if _is_index(part):
                    if int(part) < len(n):
                        reached.append(n[int(part)])
                else:
                    reached += [el[part] for el in n if isinstance(el, dict) and part in el]
        nodes = reached
    return any(isinstance(n, (list, tuple)) for n in nodes)


def _json_default(value: Any) -> Any:
    """orjson fallback for the BSON types it does not know.

    Mirrors :func:`_to_plain` so both serialisation paths give the same text; anything else
    falls back to ``str`` rather than failing on an unusual BSON type.
    """
    if Decimal128 and isinstance(value, Decimal128):
        return _decimal_to_float(value)
    if ObjectId and isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return _binary_text(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _dumps(value: Any) -> str:
    """JSON string of a (BSON) value; orjson when available, ``json`` otherwise.

    BSON and numpy types are converted as in :func:`_to_plain`; unknown types become their
    ``str``. Non-ASCII text is kept as is (not ``\\u`` escaped). orjson writes compact JSON
    (``{"a":1}``); the ``json`` fallback keeps its default separators (``{"a": 1}``).

    Parameters
    ----------
    value : Any
        A document or a single value.

    Returns
    -------
    str
        The JSON text.

    Examples
    --------
    >>> dumps({"a": [1, 2], "b": "x"})  # with orjson installed
    '{"a":[1,2],"b":"x"}'
    """
    if orjson is not None:
        try:
            return orjson.dumps(
                value, default=_json_default, option=orjson.OPT_NON_STR_KEYS
            ).decode()
        except TypeError:  # e.g. an int beyond 64 bits (only in documents not read from BSON)
            pass
    return json.dumps(_to_plain(value), ensure_ascii=False, default=str)


def _key_units(piece: str, size: int) -> int:
    """The cost of a piece's ``"key":`` part, in the unit of ``size`` (the key itself may
    contain ':'; in token units it is estimated from the share of characters)."""
    try:
        # Decoding the leading JSON string finds where the key ends, even if it contains ':'.
        _, end = json.JSONDecoder().raw_decode(piece)
        chars = end + 1
    except ValueError:
        chars = len(piece)
    return chars if size == len(piece) else max(1, round(chars * size / max(len(piece), 1)))


def _json_pieces(
    doc: dict,
    fields: Sequence[str] | None = None,
    exclude: Iterable[str] = (),
    first: dict | None = None,
) -> list[str]:
    """The ``"key":value`` JSON pieces of a document, in document order.

    ``first`` entries (e.g. a masked target) are emitted before everything else and their keys
    are excluded from the rest.

    The pieces are kept separate so that a caller can measure each one (in characters or
    tokens) and join them with :func:`_assemble_json` within a budget.

    Parameters
    ----------
    doc : dict
        The document.
    fields : sequence of str or None, default None
        Passed to :func:`_select_fields`: dotted paths to keep, or ``None`` for all fields.
    exclude : iterable of str, default ()
        Dotted paths to drop, passed to :func:`_select_fields` (``_id`` is always dropped).
    first : dict or None, default None
        Entries emitted first, as given (not looked up in ``doc``).

    Returns
    -------
    list of str
        One ``"key":value`` string per top-level entry.

    Examples
    --------
    >>> _json_pieces({"name": "A", "rating": 7}, first={"rating": "[MASK]"})
    ['"rating":"[MASK]"', '"name":"A"']
    """
    exclude = list(exclude) + list(first or {})
    selected = _select_fields(doc, fields, exclude)
    pieces = [_dumps(str(k)) + ":" + _dumps(v) for k, v in (first or {}).items()]
    pieces += [_dumps(str(k)) + ":" + _dumps(v) for k, v in selected.items()]
    return pieces


def _assemble_json(
    pieces: Sequence[str],
    max_units: int | None = None,
    sizes: Sequence[int] | None = None,
    truncate=None,
    prefix: str = "",
) -> str:
    """Join pieces into ``prefix{p1,p2,...}`` within a budget.

    ``sizes[i]`` is the cost of piece ``i`` in some unit (characters by default, tokens when a
    tokenizer measured them); ``max_units`` is the budget for the braces, commas and pieces (not
    the prefix: callers subtract its cost themselves). Pieces are added in order; the first
    piece that does not fit is cut with ``truncate(piece, room)`` if it leaves room for more than
    the key, and the rest are dropped.

    Parameters
    ----------
    pieces : sequence of str
        Output of :func:`_json_pieces`.
    max_units : int or None, default None
        Budget in the unit of ``sizes``. ``None`` joins every piece.
    sizes : sequence of int or None, default None
        Cost of each piece. ``None`` uses the piece lengths in characters.
    truncate : callable or None, default None
        ``truncate(piece, room) -> str`` cuts a piece to ``room`` units. ``None`` cuts by
        characters. A cut piece is usually not valid JSON; the embedding model reads it as text.
    prefix : str, default ""
        Text placed before the opening brace (e.g. an instruction for the embedding model).

    Returns
    -------
    str
        ``prefix`` followed by the braced JSON object. With a budget the braces are always
        present, so the result is at least ``prefix + "{}"``.
    """
    if max_units is None:
        return prefix + "{" + ",".join(pieces) + "}"
    sizes = sizes if sizes is not None else [len(p) for p in pieces]
    truncate = truncate or (lambda piece, room: piece[:room])
    used = 2  # the braces; callers subtract the prefix cost from max_units themselves
    parts: list[str] = []
    for piece, size in zip(pieces, sizes):
        sep = 1 if parts else 0  # a comma before every piece but the first
        if used + sep + size > max_units:
            room = max_units - used - sep
            key_len = _key_units(piece, size)
            # Keep a cut piece only if some of its value survives, not just its key.
            if room > key_len + 4:
                parts.append(truncate(piece, room))
            break
        parts.append(piece)
        used += sep + size
    return prefix + "{" + ",".join(parts) + "}"


def _document_to_json(
    doc: dict,
    fields: Sequence[str] | None = None,
    exclude: Iterable[str] = (),
    max_chars: int | None = None,
    first: dict | None = None,
    prefix: str = "",
) -> str:
    """Serialise the selected fields of a document into the JSON string that gets embedded.

    With ``max_chars`` the fields are serialised one by one, in document order, and the
    remaining fields are skipped once the budget is spent, so a document with a huge trailing
    array (reviews, transactions) does not get fully serialised only to be cut off. The output
    is never shorter than ``prefix + "{}"``, even with a smaller budget.
    ``first`` entries are emitted first (a masked target), ``prefix`` goes before the JSON.

    Parameters
    ----------
    doc : dict
        The document.
    fields : sequence of str or None, default None
        Dotted paths to keep (as flat dotted keys), or ``None`` for the whole document.
    exclude : iterable of str, default ()
        Dotted paths to drop, e.g. the prediction target; ``_id`` is always dropped.
    max_chars : int or None, default None
        Budget in characters for the whole output, ``prefix`` included. ``None`` means no
        limit. Token budgets are handled by the featurizer through :func:`_assemble_json`.
    first : dict or None, default None
        Entries emitted before the document's own fields.
    prefix : str, default ""
        Text placed before the JSON.

    Returns
    -------
    str
        The JSON text.

    Examples
    --------
    >>> doc = {"_id": 1, "name": "A", "address": {"market": "Porto"}}
    >>> _document_to_json(doc, fields=["name", "address.market"])
    '{"name":"A","address.market":"Porto"}'
    >>> _document_to_json(doc, max_chars=30)
    '{"name":"A","address":{"marke}'
    """
    pieces = _json_pieces(doc, fields, exclude, first)
    budget = None if max_chars is None else max(max_chars - len(prefix), 0)
    return _assemble_json(pieces, budget, prefix=prefix)


def _compile_path(path: str):
    """Pre-split a dotted path into ``(key, index or None)`` steps for fast lookups.

    Parameters
    ----------
    path : str
        Dotted path.

    Returns
    -------
    tuple of (str, int or None)
        One step per part; the int is set for numeric parts. Pass it to :func:`_get_compiled`.

    Examples
    --------
    >>> _compile_path("coords.0")
    (('coords', None), ('0', 0))
    """
    return tuple((p, int(p) if _is_index(p) else None) for p in path.split("."))


def _get_compiled(doc: Any, steps, default: Any = None) -> Any:
    """``_get_path`` with a path from :func:`_compile_path` (a digit part is a dict key on dicts).

    Parameters
    ----------
    doc : Any
        The document.
    steps : tuple
        Output of :func:`_compile_path`.
    default : Any, default None
        Returned when the path is missing.

    Returns
    -------
    Any
        The same value as ``_get_path(doc, path, default)``.
    """
    cur = doc
    for key, index in steps:
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        elif index is not None and isinstance(cur, (list, tuple)) and index < len(cur):
            cur = cur[index]
        else:
            return default
    return cur


def _fetch_documents(
    collection,
    filter: dict | None = None,
    projection: dict | None = None,
    limit: int | None = None,
    sort: list[tuple[str, int]] | None = None,
) -> list[dict]:
    """Materialise a pymongo (or mongomock) query as a list of dicts.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        The collection to query (a mongomock collection works too).
    filter : dict or None, default None
        ``find`` filter; ``None`` matches every document.
    projection : dict or None, default None
        ``find`` projection, e.g. from :func:`_projection_for`; ``None`` returns whole documents.
    limit : int or None, default None
        Maximum number of documents; ``None`` or ``0`` means no limit.
    sort : list of (str, int) or None, default None
        Sort specification such as ``[("year", 1)]``; ``None`` keeps natural order.

    Returns
    -------
    list of dict
        The documents.

    Examples
    --------
    >>> docs = _fetch_documents(movies, {"year": {"$gte": 2010}}, limit=100)  # doctest: +SKIP
    """
    cursor = collection.find(filter or {}, projection)
    if sort:
        cursor = cursor.sort(sort)
    if limit:
        cursor = cursor.limit(int(limit))
    return list(cursor)


def _as_documents(X) -> list[dict]:
    """Accept a list of dicts, a pymongo cursor, a single dict, or a DataFrame of records.

    Parameters
    ----------
    X : list, cursor, iterable, dict, pandas.DataFrame or None
        The input documents.

    Returns
    -------
    list of dict
        ``[]`` for ``None``, ``[X]`` for a single dict, one dict per row for a DataFrame, and
        ``list(X)`` otherwise (which consumes a cursor or generator).
    """
    if X is None:
        return []
    if isinstance(X, dict):
        return [X]
    if hasattr(X, "to_dict") and hasattr(X, "columns"):  # pandas DataFrame
        return X.to_dict(orient="records")
    return list(X)


def _with_dict_documents(collection):
    """``collection`` with documents decoded as ``dict``.

    The featurizer and dataset read documents as dicts; with a ``RawBSONDocument`` (or another
    non-dict ``document_class``) codec every field would read as missing.
    """
    co = getattr(collection, "codec_options", None)
    if co is None or issubclass(co.document_class, dict):
        return collection
    return collection.with_options(codec_options=co.with_options(document_class=dict))


def _iter_batches(X, batch_size: int, max_bytes: int | None = None):
    """Yield lists of at most ``batch_size`` documents from a list, cursor, generator or DataFrame
    without materialising the whole input.

    Parameters
    ----------
    X : list, cursor, iterable, dict, pandas.DataFrame or None
        The input documents, accepted as in :func:`_as_documents`.
    batch_size : int
        Maximum number of documents per batch.

    Yields
    ------
    list of dict
        Consecutive batches; only the last one may be shorter. Nothing is yielded for ``None``.

    Raises
    ------
    ValueError
        If ``batch_size`` is below 1. Since this is a generator, the error is raised when
        iteration starts, not at the call.
    """
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if X is None:
        return
    if isinstance(X, dict):
        yield [X]
        return
    if hasattr(X, "to_dict") and hasattr(X, "columns"):  # pandas DataFrame
        for i in range(0, len(X), batch_size):
            yield X.iloc[i : i + batch_size].to_dict(orient="records")
        return
    batch, size = [], 0
    for doc in X:
        batch.append(doc)
        if max_bytes is not None:  # large documents: bound the batch in bytes too
            try:
                size += len(bson.encode(doc))
            except Exception:  # noqa: BLE001 (not BSON-encodable: count by documents only)
                pass
        if len(batch) == batch_size or (max_bytes is not None and size >= max_bytes):
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def _projection_for(paths: Iterable[str]) -> dict:
    """Build a ``find`` projection that covers the given dotted paths.

    Array indices are cut off (``"position.coordinates.0"`` projects ``position.coordinates``)
    and paths whose ancestor is already included are dropped, because MongoDB rejects
    projections with path collisions.

    A numeric first part is kept as a field name, and any path into ``_id`` projects the whole
    ``_id``.

    Parameters
    ----------
    paths : iterable of str
        Dotted paths that the caller will read.

    Returns
    -------
    dict
        An inclusion projection ``{path: 1}``, with keys sorted.

    Examples
    --------
    >>> _projection_for(["position.coordinates.0", "position", "_id.x", "a.b"])
    {'_id': 1, 'a.b': 1, 'position': 1}
    """
    paths = list(paths)
    if any(part.startswith("$") for p in paths for part in p.split(".")):
        return None  # a field name starting with '$' cannot be projected: fetch whole documents
    cleaned = set()
    for p in paths:
        parts = []
        for part in p.split("."):
            if _is_index(part) and parts:  # (a first part is always a field name)
                break
            parts.append(part)
        if parts and parts[0] == "_id":
            parts = ["_id"]  # a partial _id would no longer identify the document
        if parts:
            cleaned.add(".".join(parts))
    keep = {p for p in cleaned if not any(p.startswith(q + ".") for q in cleaned if q != p)}
    return {p: 1 for p in sorted(keep)}


def _fetch_by_ids(collection, ids: Sequence, projection: dict | None = None) -> list[dict]:
    """Fetch documents by ``_id`` and return them in the order of ``ids``.

    Ids are matched on their BSON encoding: embedded-document ids are not hashable, and ids
    that are equal in Python but not in MongoDB (``True`` and ``1``) must not be mixed up.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        The collection to read.
    ids : sequence
        ``_id`` values, in the order wanted. A repeated id returns the same dict object twice.
    projection : dict or None, default None
        ``find`` projection. If it excludes ``_id`` (``{"_id": 0}``), ``_id`` is still fetched
        for matching and then removed from the returned documents.

    Returns
    -------
    list of dict
        One document per id, in the order of ``ids``.

    Raises
    ------
    KeyError
        If any id is no longer in the collection.
    """
    drop_id = bool(projection) and projection.get("_id", 1) in (0, False)
    if drop_id:  # matching needs _id; it is removed again below
        projection = {k: v for k, v in projection.items() if k != "_id"} or None
    by_id = {_id_key(d["_id"]): d for d in collection.find({"_id": {"$in": list(ids)}}, projection)}
    keys = [_id_key(i) for i in ids]
    missing = [i for i, k in zip(ids, keys) if k not in by_id]
    if missing:
        raise KeyError(
            f"{len(missing)} documents disappeared from the collection, e.g. {missing[0]!r}"
        )
    docs = [by_id[k] for k in keys]
    if drop_id:
        docs = [{k: v for k, v in d.items() if k != "_id"} for d in docs]
    return docs


# The standard UUID representation lets ``uuid.UUID`` ids be BSON-encoded (the default refuses).
_ID_CODEC = bson.CodecOptions(uuid_representation=bson.binary.UuidRepresentation.STANDARD)


def _id_key(value) -> bytes:
    """Hashable key for an ``_id`` that compares the way MongoDB does (by BSON type and value)."""
    return bson.encode({"v": value}, codec_options=_ID_CODEC)


def _get_collection(uri: str, database: str, collection: str):
    """Small convenience wrapper around ``MongoClient(uri)[database][collection]``.

    Parameters
    ----------
    uri : str
        MongoDB connection string, e.g. ``"mongodb://localhost:27017"``.
    database : str
        Database name.
    collection : str
        Collection name.

    Returns
    -------
    pymongo.collection.Collection
        The collection, on a new ``MongoClient``.

    Examples
    --------
    >>> uri = "mongodb://localhost:27017"
    >>> movies = _get_collection(uri, "sample_mflix", "movies")  # doctest: +SKIP
    """
    from pymongo import MongoClient

    return MongoClient(uri)[database][collection]
