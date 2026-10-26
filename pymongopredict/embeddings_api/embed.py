"""Embedding backends that turn the JSON text of documents into vectors.

Every backend implements :class:`BaseEmbedder`: ``embed(texts)`` maps a list of strings to a
2-D ``float32`` array of shape ``(n_texts, dim)``. The backends are

* :class:`SentenceTransformerEmbedder`: local Hugging Face models through
  sentence-transformers (voyage-4-nano by default), with length-grouped batching, an
  out-of-memory retry and a float32 fallback when reduced precision yields NaN.
* :class:`VoyageEmbedder`: the Voyage AI API.
* :class:`HashingEmbedder`: hashed word n-grams, dependency free, for tests and baselines.

and two wrappers:

* :class:`_CachedEmbedder`: an on-disk sqlite cache keyed by embedder name and text.
* :class:`_MatryoshkaTruncate`: keeps the first ``dimension`` components and re-normalises.

:func:`_make_embedder` turns an ``embedding_model`` spec string (as accepted by
:class:`pymongopredict.embeddings_api.features.DocumentFeaturizer` and the ``$predict``
prediction models) into a backend wrapped in these, in the order backend -> cache ->
truncation, so one cached full vector serves every truncated dimension.

Examples
--------
>>> from pymongopredict.embeddings_api.embed import _make_embedder
>>> emb = _make_embedder("hashing", n_features=64)
>>> emb.embed(["a short text", "another one"]).shape
(2, 64)
>>> nano = _make_embedder("voyage-4-nano@256", cache_dir=".mongopredict_cache")  # doctest: +SKIP
>>> nano.name  # doctest: +SKIP
'voyageai/voyage-4-nano|len=1024@256'
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pathlib
import sqlite3
import threading
import time
import warnings
from typing import Sequence

import numpy as np

# the embedding model used when none is given
DEFAULT_MODEL = "voyage-4-nano"
# open-weight Voyage models that run locally; any "org/name" string is treated as a HF model id
LOCAL_ALIASES = {"voyage-4-nano": "voyageai/voyage-4-nano"}
# default token limit of local models: long JSON documents are truncated to it
_DEFAULT_SEQ_LENGTH = 1024


# _CachedEmbedder's per-call hooks, by thread and then by embedder (see BaseEmbedder._emit)
_HOOKS = threading.local()


class BaseEmbedder:
    """Interface of all embedders: ``embed(texts) -> np.ndarray`` of shape ``(n_texts, dim)``.

    Subclass it to plug in any model: implement :meth:`embed` and set ``name`` to a string that
    identifies the model and every setting that changes its vectors. ``name`` is the cache key
    prefix of :class:`_CachedEmbedder`; an embedder that keeps the default ``"base"`` is not
    cached by :func:`_make_embedder`.

    Attributes
    ----------
    name : str
        Identifier of the model and its settings, used in cache keys.
    """

    name: str = "base"

    def _emit(self, positions, vectors) -> None:
        """Hand finished rows ``(positions, vectors)`` to the cache while a call runs, so they are
        cached as they are computed and an interrupted call loses only the work in progress
        (no-op outside a cached call). The hook is per thread: threads may share an embedder."""
        hook = getattr(_HOOKS, "by_embedder", {}).get(id(self))
        if hook is not None:
            hook(positions, vectors)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts``.

        Parameters
        ----------
        texts : sequence of str

        Returns
        -------
        numpy.ndarray of float32, shape (len(texts), dim)
        """
        raise NotImplementedError

    def __call__(self, texts):
        """Same as :meth:`embed`."""
        return self.embed(texts)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings through sentence-transformers.

    The model is loaded lazily on first use (see :attr:`model`) and is never pickled, so a fitted
    featurizer can be saved and loaded on another host.

    Parameters
    ----------
    model_name : str, default ``"voyageai/voyage-4-nano"``
        Hugging Face id, or a key of ``LOCAL_ALIASES`` (``"voyage-4-nano"``). voyage-4-nano is
        Apache 2.0, 346M parameters.
    batch_size : int, default 32
        Maximum texts per forward pass.
    max_seq_length : int or None, default 1024
        Token limit per text; longer JSON documents are truncated. Capped at the model's own
        limit (e.g. 256 for all-MiniLM-L6-v2). ``None`` keeps the model's limit.
    device : str or None, default None
        ``"cuda"``, ``"cpu"`` (or e.g. ``"cuda:1"``); ``None`` picks CUDA when available.
    normalize : bool, default True
        Return unit-length vectors.
    trust_remote_code : bool, default True
        Forwarded to ``SentenceTransformer`` (needed by models with custom code).
    show_progress_bar : bool, default False
        Forwarded to ``encode``.
    dtype : str, default ``"auto"``
        ``"auto"`` runs in bfloat16 on CUDA devices that support it and float32 elsewhere; or an
        explicit torch dtype name such as ``"float32"`` / ``"float16"``. float16 overflows in
        voyage-4-nano (Qwen3 backbone) and yields NaN vectors: if any vector of a call comes
        back non-finite, a ``RuntimeWarning`` is issued, the model is switched to float32 for
        good and those rows are recomputed.
    max_batch_tokens : int, default 65536
        Texts are grouped by length so that a batch holds at most about this many tokens
        (estimated as characters / 3, capped by ``max_seq_length``). This keeps long documents
        from exhausting GPU memory while short ones still go through in batches of
        ``batch_size``. On a CUDA out-of-memory error a group is retried with a quarter of the
        batch size, down to one text, before the error is raised.
    prompt : str or None, default None
        Text put before every document instead of the model's document prompt. ``None``: the
        model's own (e.g. voyage-4-nano's ``"Represent the document for retrieval: "``);
        ``""``: none.
    max_seq_length_in_key : bool, default True
        Whether ``|len=<max_seq_length>`` goes into ``name`` (the cache key). The featurizer
        turns it off when it has already cut every text to the model's length with the model's
        own tokenizer, so those vectors are the same as untruncated ones.

    Attributes
    ----------
    name : str
        Cache key prefix: the model id, then ``|prompt=<prompt>`` for a custom prompt, ``|raw``
        when not normalised, and ``|len=<max_seq_length>`` (see ``max_seq_length_in_key``).
    part : tuple of (int, int) or None
        Set only by :class:`_CachedEmbedder` during one call in part mode.
    last_done : numpy.ndarray of bool or None
        Rows the last :meth:`embed` call computed (all of them outside part mode).

    Notes
    -----
    Matryoshka truncation is not done here: pass ``dimension=`` to :func:`_make_embedder`, which
    caches the full vectors once and truncates on the way out (see :class:`_MatryoshkaTruncate`).
    """

    def __init__(
        self,
        model_name: str = LOCAL_ALIASES[DEFAULT_MODEL],
        batch_size: int = 32,
        max_seq_length: int | None = _DEFAULT_SEQ_LENGTH,
        device: str | None = None,
        normalize: bool = True,
        trust_remote_code: bool = True,
        show_progress_bar: bool = False,
        dtype: str = "auto",
        max_batch_tokens: int = 65536,
        prompt: str | None = None,
        max_seq_length_in_key: bool = True,
    ):
        self.model_name = LOCAL_ALIASES.get(model_name, model_name)
        self.prompt = prompt
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_seq_length = max_seq_length
        self.device = device
        self.normalize = normalize
        self.trust_remote_code = trust_remote_code
        self.show_progress_bar = show_progress_bar
        self.dtype = dtype
        # a custom prompt replaces the model's document prompt and goes into the name (cache keys)
        custom = prompt is not None
        self.name = f"{self.model_name}|prompt={prompt}" if custom else self.model_name
        if not normalize:  # unnormalised vectors must not share cache keys with normalised ones
            self.name += "|raw"
        # the plain name means the embedding of the whole text; a truncation length (the default
        # 1024 included) goes into the name, as it changes the vectors of longer texts. The
        # featurizer's token budget leaves it out: it has already cut every text to that length.
        if max_seq_length is not None and max_seq_length_in_key:
            self.name += f"|len={max_seq_length}"
        self._model = None

    @property
    def model(self):
        """The ``SentenceTransformer``, loaded on first access.

        Resolves ``device`` and ``dtype="auto"`` at load time and sets the model's
        ``max_seq_length`` to ``min(max_seq_length, native limit)``.
        """
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = str(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            dtype = self.dtype
            if dtype == "auto":
                bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
                dtype = "bfloat16" if bf16 else "float32"
            self._model = SentenceTransformer(
                self.model_name,
                device=device,
                trust_remote_code=self.trust_remote_code,
                model_kwargs={"dtype": getattr(torch, dtype)},
            )
            native = self._model.max_seq_length
            if self.max_seq_length is not None:
                self._model.max_seq_length = (
                    min(self.max_seq_length, native) if native else self.max_seq_length
                )
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` in length-grouped batches.

        Parameters
        ----------
        texts : sequence of str

        Returns
        -------
        numpy.ndarray of float32, shape (len(texts), dim)
            Rows in the order of ``texts``. In part mode (``part`` set) only this part's rows are
            computed; the others are zero and ``last_done`` marks which rows are real.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), np.float32)
        kwargs = dict(
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        # encode_document applies the model's document prompt (sentence-transformers >= 5); a custom
        # ``prompt`` ("" for none) is passed to encode instead
        if self.prompt is not None:
            encode = self.model.encode
            kwargs["prompt"] = self.prompt
        else:
            encode = getattr(self.model, "encode_document", None) or self.model.encode
        E = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()), np.float32)
        done = np.ones(len(texts), bool)
        groups = self._token_groups(texts)
        if self.part is not None:
            # only this part's groups (see part_of); the other rows stay zero and are not done
            done[:] = False
            groups = [
                g for g, owner in zip(groups, self.part_of(groups, texts)) if owner == self.part[0]
            ]
        for idx in groups:
            kwargs["batch_size"] = min(self.batch_size, len(idx))
            E[idx] = self._encode_group(encode, [texts[i] for i in idx], kwargs)
            done[idx] = True
            self._emit(idx, E[idx])
        self.last_done = done
        bad = ~np.isfinite(E).all(axis=1)
        if bad.any():  # reduced precision overflowed: fall back to float32 for good
            warnings.warn(
                f"{int(bad.sum())} embeddings from {self.model_name} were non-finite; "
                "switching the model to float32",
                RuntimeWarning,
                stacklevel=2,
            )
            self.model.float()
            self.dtype = "float32"
            rows = np.flatnonzero(bad)
            redo = [texts[i] for i in rows]
            for idx in self._token_groups(redo):  # the same token-bounded batches as above
                kwargs["batch_size"] = min(self.batch_size, len(idx))
                E[rows[idx]] = self._encode_group(encode, [redo[i] for i in idx], kwargs)
        return E

    def __getstate__(self):
        """Pickle the settings only; the model is loaded again on first use."""
        # a fitted featurizer is pickled with its embedder: never the loaded model (large, and
        # possibly on a GPU the unpickling host lacks)
        state = dict(self.__dict__)
        state["_model"] = None
        for k in ("_loads", "part", "last_done"):  # per-process state of part mode
            state.pop(k, None)
        return state

    def _encode_group(self, encode, group: list[str], kwargs: dict) -> np.ndarray:
        """Encode one length-sorted group; on a CUDA out-of-memory error (shared GPUs) free the
        cache and retry with a smaller batch, down to one text at a time, before giving up."""
        batch = kwargs["batch_size"]
        while True:
            try:
                out = encode(group, **{**kwargs, "batch_size": batch})
                return np.asarray(out, dtype=np.float32)
            except Exception as e:  # noqa: BLE001 (torch.cuda.OutOfMemoryError is a RuntimeError)
                if "out of memory" not in str(e).lower() or batch <= 1:
                    raise
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                batch = max(1, batch // 4)
                warnings.warn(
                    f"CUDA out of memory in {self.model_name}; retrying with batch_size={batch}",
                    RuntimeWarning,
                    stacklevel=3,
                )

    # (k, K): embed only part k of K of the length groups (_CachedEmbedder, MONGOPREDICT_EMBED_PART)
    part: tuple[int, int] | None = None
    last_done = None  # the rows the last embed() call computed

    def part_of(self, groups, texts) -> list[int]:
        """Assign every length group to one of the ``K`` parts of ``part = (k, K)``.

        Each group goes to the least loaded part, the load being the group's padded tokens
        (estimated as characters / 3), in the groups' order (longest first). The assignment is
        deterministic and the loads carry over between calls, so the ``K`` processes, each
        running the same calls, agree on who computes which group.

        Parameters
        ----------
        groups : list of list of int
            Groups from ``_token_groups``.
        texts : list of str
            The texts the indices refer to.

        Returns
        -------
        list of int
            The part (``0`` .. ``K-1``) of each group.
        """
        k_parts = self.part[1]
        if getattr(self, "_loads", None) is None or len(self._loads) != k_parts:
            self._loads = [0] * k_parts
        cap = self.model.max_seq_length or 10**9
        out = []
        for g in groups:
            j = min(range(k_parts), key=lambda x: (self._loads[x], x))
            self._loads[j] += len(g) * max(min(max(len(texts[i]) // 3, 1), cap) for i in g)
            out.append(j)
        return out

    def _token_groups(self, texts) -> list[list[int]]:
        """Indices grouped so that ``len(group) * longest_estimate <= max_batch_tokens``.

        Sorting by length (longest first) keeps padding low, and bounding the padded tokens
        rather than the count keeps a batch of long documents within GPU memory.
        """
        cap = self.model.max_seq_length or 10**9
        est = [min(max(len(t) // 3, 1), cap) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: -est[i])
        groups, cur, longest = [], [], 0
        for i in order:
            longest = max(longest, est[i]) if cur else est[i]
            too_big = (len(cur) + 1) * longest > self.max_batch_tokens
            if cur and (too_big or len(cur) >= self.batch_size):
                groups.append(cur)
                cur, longest = [], est[i]
            cur.append(i)
        if cur:
            groups.append(cur)
        return groups


# Voyage API limits per request (https://docs.voyageai.com/docs/embeddings): at most 1,000 texts and
# this many tokens in total; models not listed get the smallest limit
VOYAGE_MAX_TEXTS = 1000
VOYAGE_REQUEST_TOKENS = {
    "voyage-4-lite": 1_000_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-4": 320_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
}
# voyage-4-large, voyage-code-*, voyage-finance-2, voyage-law-2, ...
_VOYAGE_DEFAULT_TOKENS = 120_000
# tokens per text the API reads (voyage-law-2: 16,000); longer texts are truncated
_VOYAGE_CONTEXT = 32_000
# models served by other endpoints (contextualized_embed, multimodal_embed), not by embed
_VOYAGE_OTHER_ENDPOINTS = ("voyage-context-", "voyage-multimodal-")


class VoyageEmbedder(BaseEmbedder):
    """Embeddings through the Voyage AI API (the ``voyageai`` client is a dependency of the
    package; needs an API key in ``VOYAGE_API_KEY`` or ``api_key``).

    Any text embedding model the API serves works, e.g. ``"voyage-4-lite"``, ``"voyage-4"``,
    ``"voyage-4-large"``, ``"voyage-code-4"``, ``"voyage-finance-2"`` or ``"voyage-law-2"``.
    voyage-4-nano is an open-weight model and always runs locally (see
    :class:`SentenceTransformerEmbedder`). Texts are sent in requests of at most ``batch_size``
    texts (the API takes up to 1,000) and at most the model's tokens per request (1M for
    voyage-4-lite, 320K for voyage-4, 120K for voyage-4-large and the others), estimated as
    characters / 3. Rate limits and server errors are retried by the client, with exponential
    backoff, up to ``max_retries`` times.

    Parameters
    ----------
    model : str, default ``"voyage-4-lite"``
        Voyage model name.
    output_dimension : int or None, default None
        Dimension requested from the API (256, 512, 1024 or 2048 for the voyage-4 models);
        ``None`` is the model's default (1024).
    input_type : str or None, default ``"document"``
        Voyage ``input_type`` (``"document"``, ``"query"`` or ``None``).
    batch_size : int, default 1000
        Most texts per API request.
    api_key : str or None, default None
        API key; ``None`` reads ``VOYAGE_API_KEY``.
    truncation : bool, default True
        Let the API truncate texts longer than the model's context (32,000 tokens for the voyage-4
        models) instead of failing.
    max_retries : int, default 5
        Retries of a failed request (rate limit, timeout, server error), passed to
        ``voyageai.Client``.
    timeout : float or None, default None
        Seconds before a request times out; ``None`` is the client's default.

    Attributes
    ----------
    name : str
        ``"voyage-api:<model>@<output_dimension or full>"``, plus ``|<input_type>|truncation=...``
        when those differ from the defaults (so default keys stay as they were).

    Raises
    ------
    ValueError
        For a model served by another endpoint (``voyage-context-*``, ``voyage-multimodal-*``),
        and, on the first request, when no API key is set.
    ImportError
        On the first request, when the ``voyageai`` package is not installed.
    """

    def __init__(
        self,
        model: str = "voyage-4-lite",
        output_dimension: int | None = None,
        input_type: str | None = "document",
        batch_size: int = VOYAGE_MAX_TEXTS,
        api_key: str | None = None,
        truncation: bool = True,
        max_retries: int = 5,
        timeout: float | None = None,
    ):
        if model.startswith(_VOYAGE_OTHER_ENDPOINTS):
            raise ValueError(
                f"{model!r} is not a text embedding model of the embed endpoint; use one such as "
                "'voyage-4-lite', 'voyage-4' or 'voyage-4-large'"
            )
        if not 1 <= int(batch_size) <= VOYAGE_MAX_TEXTS:
            raise ValueError(
                f"batch_size must be between 1 and {VOYAGE_MAX_TEXTS}, got {batch_size}"
            )
        self.model = model
        self.output_dimension = output_dimension
        self.input_type = input_type
        self.batch_size = int(batch_size)
        self.api_key = api_key
        self.truncation = truncation
        self.max_retries = max_retries
        self.timeout = timeout
        self.name = f"voyage-api:{model}@{output_dimension or 'full'}"
        if input_type != "document" or not truncation:
            self.name += f"|{input_type}|truncation={truncation}"
        self._client = None

    def __getstate__(self):
        """Pickle without credentials: the API key and the client (which holds the key, also when
        it was read from VOYAGE_API_KEY) are dropped, so a stored model or any other pickle of the
        embedder never carries the key. After unpickling, the key is read from the environment."""
        state = dict(self.__dict__)
        state["api_key"] = None
        state["_client"] = None
        return state

    # copies within the process keep the key (scikit-learn's clone deep-copies an embedder passed
    # to a DocumentFeaturizer); only pickling, which can leave the process, drops it
    def __copy__(self):
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        import copy

        new = object.__new__(type(self))
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = v if k == "_client" else copy.deepcopy(v, memo)
        return new

    @property
    def client(self):
        """The ``voyageai.Client``, created on first use (so importing needs no ``voyageai``)."""
        if self._client is None:
            try:
                import voyageai
            except ImportError as e:
                raise ImportError(
                    f"embedding with {self.model!r} uses the Voyage API, but the voyageai package "
                    "is missing; reinstall pymongopredict (it depends on voyageai)"
                ) from e
            if not (self.api_key or os.environ.get("VOYAGE_API_KEY")):
                raise ValueError(
                    f"embedding with {self.model!r} uses the Voyage API and needs an API key: set "
                    "VOYAGE_API_KEY (from https://dashboard.voyageai.com) or pass api_key=; "
                    "voyage-4-nano runs locally without one"
                )
            self._client = voyageai.Client(
                api_key=self.api_key, max_retries=self.max_retries, timeout=self.timeout
            )
        return self._client

    def _requests(self, texts: list[str]):
        """Index ranges of ``texts``, one per API request, within the text and token limits."""
        limit = 0.8 * VOYAGE_REQUEST_TOKENS.get(self.model, _VOYAGE_DEFAULT_TOKENS)  # estimates
        start, tokens = 0, 0.0
        for i, t in enumerate(texts):
            n = min(len(t) / 3 + 1, _VOYAGE_CONTEXT) if self.truncation else len(t) / 3 + 1
            if i > start and (i - start >= self.batch_size or tokens + n > limit):
                yield start, i
                start, tokens = i, 0.0
            tokens += n
        if start < len(texts):
            yield start, len(texts)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` with as few API requests as the limits allow.

        Returns
        -------
        numpy.ndarray of float32, shape (len(texts), dim)
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.output_dimension or 1024), np.float32)
        kwargs = dict(model=self.model, input_type=self.input_type, truncation=self.truncation)
        if self.output_dimension:
            kwargs["output_dimension"] = self.output_dimension
        out = []
        for a, b in self._requests(texts):
            chunk = self.client.embed(texts[a:b], **kwargs).embeddings
            out.extend(chunk)
            self._emit(range(a, b), np.asarray(chunk, dtype=np.float32))
        return np.asarray(out, dtype=np.float32)


class HashingEmbedder(BaseEmbedder):
    """Hashed word uni- and bigram vectors: fast, deterministic and dependency free.

    Not a semantic embedding; use it for tests, CI, quick pipeline checks and as a cheap
    bag-of-tokens baseline. ``analyzer="char_wb"`` with ``ngram_range=(3, 5)`` gives character
    n-grams instead (about 10x slower on long JSON documents).

    Parameters
    ----------
    n_features : int, default 256
        Output dimension (number of hash buckets).
    ngram_range : tuple of (int, int), default (1, 2)
        Range of n-gram lengths.
    analyzer : str, default ``"word"``
        ``HashingVectorizer`` analyzer (``"word"``, ``"char"`` or ``"char_wb"``).

    Attributes
    ----------
    name : str
        ``"hashing@<n_features>"``, plus ``|<analyzer><ngram_range>`` for non-default
        analyzers, so different settings never share cache entries.

    Examples
    --------
    >>> HashingEmbedder(n_features=8).embed(['{"a": 1}']).shape
    (1, 8)
    """

    def __init__(self, n_features: int = 256, ngram_range=(1, 2), analyzer: str = "word"):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.n_features = n_features
        self._vec = HashingVectorizer(
            n_features=n_features,
            analyzer=analyzer,
            ngram_range=ngram_range,
            norm="l2",
            alternate_sign=False,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.name = f"hashing@{n_features}"
        if analyzer != "word" or tuple(ngram_range) != (1, 2):  # (default keys kept as they were)
            self.name += f"|{analyzer}{tuple(ngram_range)}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised hashed n-gram counts of ``texts``, shape ``(len(texts), n_features)``."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.n_features), np.float32)
        return self._vec.transform(texts).toarray().astype(np.float32)


class _MatryoshkaTruncate(BaseEmbedder):
    """Keep the first ``dimension`` components of another embedder's vectors and re-normalise.

    This is what sentence-transformers' ``truncate_dim`` does for Matryoshka models such as
    voyage-4-nano. Doing it here, after the cache, means one cached full vector serves every
    dimension. :func:`_make_embedder` adds it for ``dimension=`` or a ``"model@<dim>"`` spec.

    Parameters
    ----------
    embedder : BaseEmbedder
        The embedder producing full vectors (usually a :class:`_CachedEmbedder`).
    dimension : int
        Number of leading components kept.
    normalize : bool, default True
        Re-normalise the truncated vectors to unit length (zero vectors are left as they are).

    Attributes
    ----------
    name : str
        ``"<embedder.name>@<dimension>"``.
    """

    def __init__(self, embedder: BaseEmbedder, dimension: int, normalize: bool = True):
        self.embedder = embedder
        self.dimension = int(dimension)
        self.normalize = normalize
        self.name = f"{embedder.name}@{dimension}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed with the wrapped embedder and keep the first ``dimension`` components.

        Raises
        ------
        ValueError
            If the wrapped vectors have fewer than ``dimension`` components.
        """
        E = self.embedder.embed(texts)
        if not len(E):
            return np.zeros((0, self.dimension), np.float32)
        if E.size and E.shape[1] < self.dimension:
            raise ValueError(
                f"cannot truncate {E.shape[1]}-d vectors of {self.embedder.name} "
                f"to {self.dimension}"
            )
        E = E[:, : self.dimension]
        if self.normalize and E.size:
            norms = np.linalg.norm(E, axis=1, keepdims=True)
            E = E / np.where(norms == 0, 1, norms)
        return np.ascontiguousarray(E, dtype=np.float32)


# one row per (embedder name, text): key = sha1 hex, dim = vector length, vec = float32 bytes
_CREATE_TABLE = "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"


class _CachedEmbedder(BaseEmbedder):
    """Wraps another embedder with a sqlite cache keyed by (embedder name, text).

    Re-running a demo or an ablation therefore never embeds the same document twice. The cache is
    ``<cache_dir>/embeddings.sqlite``; a row's key is the SHA-1 of ``name + "\\x00" + text``, so
    it is reused only by an embedder with exactly the same ``name`` (model id plus every setting
    that changes the vectors, e.g. ``|len=1024``, ``|raw``, ``|prompt=...``, the hashing
    analyzer). Matryoshka truncation happens after the cache, so the full vectors are stored once
    for every dimension. Non-finite vectors are never stored. Several processes may share one
    cache (sqlite WAL mode).

    Two opt-in modes, read from environment variables when the embedder is created or
    unpickled, split the embedding of one dataset over several processes (possibly on several
    hosts) while forming the same batches as one process would:

    * ``MONGOPREDICT_EMBED_PART=k/K``: this process embeds only part ``k`` of ``K`` of the length
      groups the wrapped :class:`SentenceTransformerEmbedder` forms (see
      :meth:`SentenceTransformerEmbedder.part_of`) and caches only those rows; the other rows of
      the returned array are zero. Texts of earlier calls count as cached, as they would in one
      process. The cache then uses the ``DELETE`` journal mode so its file can be read from
      another host after the process has ended.
    * ``MONGOPREDICT_EXTRA_CACHES=<sqlite>:<sqlite>...`` (``os.pathsep`` separated): texts missing
      from this cache are looked up in these cache files (e.g. the parts' caches), read-only, and
      copied in. An unreadable file only issues a warning; its rows are embedded here.

    Parameters
    ----------
    embedder : BaseEmbedder
        The embedder to cache; it should have a distinctive ``name``.
    cache_dir : str, default ``".mongopredict_cache"``
        Directory of the sqlite file (created if needed).

    Raises
    ------
    sqlite3.DatabaseError
        If the cache file exists but is not a usable sqlite database.
    ValueError
        If ``MONGOPREDICT_EMBED_PART`` is malformed.
    """

    def __init__(self, embedder: BaseEmbedder, cache_dir: str = ".mongopredict_cache"):
        self.embedder = embedder
        self.name = embedder.name
        # part mode and extra caches (see the class docstring) come from the environment; they
        # were added to split report 03's evaluation of a training checkpoint over processes
        self._read_env()
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "embeddings.sqlite")
        mode = "DELETE" if self.part is not None else "WAL"
        try:
            self._init_db(mode)
        except sqlite3.DatabaseError as e:
            raise sqlite3.DatabaseError(
                f"embedding cache {os.path.abspath(self.path)} is unusable ({e}); "
                "delete it or pass cache_dir=None"
            ) from e

    def _cache_failed(self, e: Exception) -> None:
        """Warn (once per embedder) that the cache failed; the vectors are computed anyway."""
        if not getattr(self, "_warned", False):
            self._warned = True
            warnings.warn(
                f"embedding cache {self.path} failed ({e}); embedding without it", stacklevel=3
            )

    def _init_db(self, mode: str) -> None:
        # several processes may share the cache (WAL); a part's cache (MONGOPREDICT_EMBED_PART)
        # is read from other hosts once its process has ended, so its rows go to the database
        # file itself (WAL works on one host only). Switching the journal mode does not wait
        # for other processes, so it is retried, and skipped when another one already did it.
        for attempt in range(50):
            try:
                with self._connect() as con:
                    if con.execute("PRAGMA journal_mode").fetchone()[0].upper() != mode:
                        con.execute(f"PRAGMA journal_mode={mode}")
                    con.execute(_CREATE_TABLE)
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 49:
                    raise
                time.sleep(0.05 * (1 + attempt % 5))

    def _read_env(self) -> None:
        """Read the part mode and extra caches from the environment.

        Done at creation and on unpickling, as they describe the current process, not the model.
        """
        part = os.environ.get("MONGOPREDICT_EMBED_PART")
        self.part = _parse_part(part) if part else None
        extra = os.environ.get("MONGOPREDICT_EXTRA_CACHES", "")
        self.extra = [x for x in extra.split(os.pathsep) if x]
        self._seen: set[str] = set()

    def __getstate__(self):
        """Pickle without the per-process settings (part mode, extra caches, seen keys)."""
        state = dict(self.__dict__)
        for k in ("part", "extra", "_seen"):  # process settings, read again when unpickled
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        """Restore and re-read the per-process settings from this process's environment."""
        self.__dict__.update(state)
        self._read_env()

    @contextlib.contextmanager
    def _connect(self):
        """A connection that commits (or rolls back) and is closed afterwards."""
        con = sqlite3.connect(self.path, timeout=120)
        try:
            with con:
                yield con
        finally:
            con.close()

    def _key(self, text: str) -> str:
        """Cache key: SHA-1 of the embedder name and the text, so models never share rows."""
        # (surrogatepass: a lone surrogate must not crash the cache)
        return hashlib.sha1(f"{self.name}\x00{text}".encode("utf-8", "surrogatepass")).hexdigest()

    def _embed_inner(self, texts: list[str]) -> np.ndarray:
        """The wrapped embedder, in part mode only for this call (the part never sticks to it)."""
        if self.part is None:
            if getattr(self.embedder, "part", None) is not None:
                self.embedder.part = None  # (left over by an older version's pickle)
            return self.embedder.embed(texts)
        before = getattr(self.embedder, "part", None)
        self.embedder.part = self.part
        try:
            return self.embedder.embed(texts)
        finally:
            self.embedder.part = before

    def _write(self, rows) -> None:
        """Store ``(key, vector)`` rows; a read-only or full disk keeps the vectors uncached."""
        if not rows or getattr(self, "_warned", False):  # (the cache already failed once)
            return
        try:
            with self._connect() as con:
                con.executemany(
                    "INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
                    [(k, int(v.shape[0]), np.asarray(v, np.float32).tobytes()) for k, v in rows],
                )
        except sqlite3.Error as e:
            self._cache_failed(e)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts``, computing only those not cached yet (each distinct text once).

        Parameters
        ----------
        texts : sequence of str

        Returns
        -------
        numpy.ndarray of float32, shape (len(texts), dim)
            In part mode, rows computed by other parts (and not in a cache) are zero.

        Raises
        ------
        ValueError
            If vectors found under this ``name`` have different dimensions (two different
            embedders sharing a name).
        """
        texts = list(texts)
        if not texts:
            return self.embedder.embed([])
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        try:
            with self._connect() as con:
                for i in range(0, len(keys), 900):  # sqlite parameter limit
                    chunk = keys[i : i + 900]
                    marks = ",".join("?" * len(chunk))
                    rows = con.execute(
                        f"SELECT key, dim, vec FROM emb WHERE key IN ({marks})", chunk
                    ).fetchall()
                    for k, dim, blob in rows:
                        found[k] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        except sqlite3.Error as e:  # e.g. the file was removed or became unreadable
            self._cache_failed(e)
            found = {}
        if self.extra:  # texts computed elsewhere (MONGOPREDICT_EXTRA_CACHES), copied in here
            want = [k for k in dict.fromkeys(keys) if k not in found]
            got = []
            for path in self.extra:
                if not want or not os.path.exists(path):
                    continue
                # read-only and immutable, without locks: the parts' processes have ended, and
                # sqlite's locks do not work across hosts on a network file system
                try:
                    uri = (
                        pathlib.Path(path).resolve().as_uri() + "?immutable=1"
                    )  # (quoted: '#', '?')
                    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=120)) as con:
                        for i in range(0, len(want), 900):
                            chunk = want[i : i + 900]
                            got += con.execute(
                                "SELECT key, dim, vec FROM emb WHERE key IN "
                                f"({','.join('?' * len(chunk))})",
                                chunk,
                            ).fetchall()
                except sqlite3.Error as e:  # an unreadable part: its rows are embedded here instead
                    warnings.warn(
                        f"extra cache {path} unreadable ({e}); embedding its rows here",
                        stacklevel=2,
                    )
                want = [k for k in want if k not in {g[0] for g in got}]
            if got:
                with self._connect() as con:
                    con.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", got)
                for k, dim, blob in got:
                    found[k] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        missing, seen_now = [], set()  # each distinct missing text once
        for i, k in enumerate(keys):
            if k not in found and k not in self._seen and k not in seen_now:
                seen_now.add(k)
                missing.append(i)
        if missing:
            saved: set[int] = set()  # rows of ``missing`` already written while embedding

            def save(positions, vectors):
                """Cache finished rows as they come, so an interruption keeps them."""
                js = [int(j) for j in positions]
                ok = np.isfinite(vectors).all(axis=1)  # never cache a broken vector
                self._write([(keys[missing[j]], v) for j, v, o in zip(js, vectors, ok) if o])
                saved.update(j for j, o in zip(js, ok) if o)

            hooks = _HOOKS.__dict__.setdefault("by_embedder", {})
            previous = hooks.get(id(self.embedder))
            hooks[id(self.embedder)] = save
            try:
                new = self._embed_inner([texts[i] for i in missing])
            finally:
                if previous is None:
                    hooks.pop(id(self.embedder), None)
                else:
                    hooks[id(self.embedder)] = previous
            done = getattr(self.embedder, "last_done", None) if self.part is not None else None
            finite = np.isfinite(new).all(axis=1)
            rows = [
                j
                for j in range(len(missing))
                if (done is None or done[j]) and finite[j] and j not in saved
            ]
            self._write([(keys[missing[j]], new[j]) for j in rows])
            for j in range(len(missing)):
                if done is None or done[j]:
                    found[keys[missing[j]]] = new[j]
        if self.part is not None:  # the texts of this call count as cached in the calls that follow
            self._seen.update(keys)
        dim = next((v.shape[0] for v in found.values()), 0)
        if not dim and self.part is not None:  # (a part that computed none of this call's rows)
            model = getattr(self.embedder, "model", None)
            if hasattr(model, "get_sentence_embedding_dimension"):
                dim = model.get_sentence_embedding_dimension()
            else:  # hashing, Voyage, a custom embedder: learn the width from one text
                dim = int(np.asarray(self.embedder.embed([""])).shape[1])
        if any(v.shape[0] != dim for v in found.values()):
            raise ValueError(
                f"cached vectors under {self.name!r} have different dimensions; "
                "give each embedder a distinct name"
            )
        return np.stack([found.get(k, np.zeros(dim, np.float32)) for k in keys]).astype(np.float32)


def _make_embedder(spec, cache_dir: str | None = None, **kwargs) -> BaseEmbedder | None:
    """Resolve an ``embedding_model`` argument into an embedder.

    Parameters
    ----------
    spec : str, BaseEmbedder, None or False
        * ``BaseEmbedder`` instance: used as is (still cached and truncated as below).
        * ``"voyage-4-nano"`` or any ``"org/name"`` Hugging Face id: local
          :class:`SentenceTransformerEmbedder`.
        * other ``"voyage-*"`` names (e.g. ``"voyage-4-lite"``): :class:`VoyageEmbedder`.
        * ``"hashing"``: :class:`HashingEmbedder`.
        * ``None`` / ``False``: no embeddings (returns ``None``).

        A string may end in ``"@<dim>"`` (e.g. ``"voyage-4-nano@512"``), which is the same as
        ``dimension=<dim>`` and takes precedence over it.
    cache_dir : str or None, default None
        Wrap the backend in a :class:`_CachedEmbedder` in this directory. Skipped, with a
        warning, for an embedder without its own ``name`` (unnamed embedders would read each
        other's vectors).
    **kwargs
        Forwarded to the backend constructor (not used for an instance), except
        ``dimension``: Matryoshka truncation with :class:`_MatryoshkaTruncate`, applied after the
        cache so the full vectors are computed and stored only once. The truncated vectors are
        re-normalised unless ``normalize=False``.

    Returns
    -------
    BaseEmbedder or None
        backend, or ``_CachedEmbedder(backend)``, optionally inside ``_MatryoshkaTruncate``.

    Raises
    ------
    ValueError
        For an unknown spec or a ``dimension`` below 1.

    Examples
    --------
    >>> _make_embedder(None) is None
    True
    >>> _make_embedder("hashing@16", n_features=64).embed(["x"]).shape
    (1, 16)
    """
    if spec is None or spec is False:
        return None
    if not isinstance(spec, (str, BaseEmbedder)):
        raise ValueError(f"unknown embedding_model {spec!r}")
    dimension = kwargs.pop("dimension", None)
    if isinstance(spec, str) and "@" in spec and spec.rsplit("@", 1)[1].isdigit():
        spec, dim = spec.rsplit("@", 1)
        dimension = int(dim)
    if isinstance(spec, BaseEmbedder):
        emb = spec
    elif spec == "hashing":
        emb = HashingEmbedder(**kwargs)
    elif spec in LOCAL_ALIASES or "/" in spec:
        emb = SentenceTransformerEmbedder(spec, **kwargs)
    elif spec.startswith("voyage"):
        emb = VoyageEmbedder(spec, **kwargs)
    else:
        raise ValueError(f"unknown embedding_model {spec!r}")
    if cache_dir and emb.name == BaseEmbedder.name:
        # the cache is keyed by name: two unnamed embedders would read each other's vectors
        warnings.warn(
            f"{type(emb).__name__} has no `name`; its embeddings are not cached", stacklevel=2
        )
    elif cache_dir:
        try:
            emb = _CachedEmbedder(emb, cache_dir)
        except (OSError, sqlite3.Error) as e:  # read-only disk, full disk, corrupt file
            warnings.warn(f"embedding cache not used, embedding without it: {e}", stacklevel=2)
    if dimension is not None:
        if int(dimension) < 1:
            raise ValueError(f"dimension must be at least 1, got {dimension}")
        emb = _MatryoshkaTruncate(emb, dimension, normalize=kwargs.get("normalize", True))
    return emb


def _base_spec(spec):
    """``embedding_model`` without a ``@<dimension>`` suffix (non-strings returned as is).

    Examples
    --------
    >>> _base_spec("voyage-4-nano@512")
    'voyage-4-nano'
    """
    if isinstance(spec, str) and "@" in spec and spec.rsplit("@", 1)[1].isdigit():
        return spec.rsplit("@", 1)[0]
    return spec


def _parse_part(value: str) -> tuple[int, int]:
    """Parse ``MONGOPREDICT_EMBED_PART`` (``"k/K"``), failing loudly on a typo rather than
    silently embedding everything (or nothing) in every process."""
    try:
        k, n = (int(x) for x in value.split("/"))
    except ValueError:
        k, n = -1, 0
    if not (n >= 1 and 0 <= k < n):
        raise ValueError(f"MONGOPREDICT_EMBED_PART must be k/K with 0 <= k < K, got {value!r}")
    return k, n
