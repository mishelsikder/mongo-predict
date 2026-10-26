"""The embeddings API: datasets and a featurizer that turn documents into ``X`` for scikit-learn.

This is the scikit-learn workflow underneath ``$predict`` (see
:mod:`pymongopredict.predictions_api`): use it to pick your own estimator, cross-validate,
combine features or forecast.

What you use:

* :class:`MongoDataset`: the documents of a collection (with a filter, limit and target) or a
  list of documents, with the target values ``y``. It indexes like an array, so scikit-learn's
  ``train_test_split``, cross-validation splitters and ``cross_val_score`` work on it directly.
* :class:`DocumentFeaturizer`: an sklearn transformer from documents to chosen columns
  and embeddings of the documents, for any sklearn estimator or ``Pipeline``.
* :class:`MongoSeries`: a time series aggregated from documents, for sktime forecasters.
* :class:`SentenceTransformerEmbedder`, :class:`VoyageEmbedder`, :class:`HashingEmbedder`:
  embedding models with non-default settings, to pass as ``embedding_model`` (a name such as
  ``"voyage-4-nano"`` or ``"voyage-4-lite"`` is enough otherwise).

## Datasets and splits

``MongoDataset(coll, target=...)`` reads a collection (``filter=``, ``projection=``, ``limit=`` and ``sort=`` are passed
to ``find``). ``dataset.y`` is a float array for numeric targets, bool for booleans and strings otherwise; documents
without a target value are left out. Split it with scikit-learn: a dataset indexes like an array (``ds[indices]``,
``ds.iloc[...]``, ``ds.take(...)``), so the splitters return train and test datasets that share the same source:

    from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit, train_test_split

    ds_train, ds_test = train_test_split(ds, test_size=0.2, stratify=ds.y, random_state=0)  # by the target
    ds_train, ds_test = train_test_split(ds, test_size=0.2, stratify=ds.field_values("address.market"))
    tr, te = next(GroupShuffleSplit(test_size=0.3).split(ds, groups=ds.field_values("host.host_id")))
    ds_train, ds_test = ds[tr], ds[te]                                            # a host on one side
    ds = MongoDataset(coll, target="y", sort=[("ts", 1)])                         # in time order, then
    ds_train, ds_test = train_test_split(ds, test_size=0.2, shuffle=False)        # train on the past
    for tr, te in TimeSeriesSplit(n_splits=5, gap=24).split(ds): ...              # rolling, with a gap

``cross_val_score(make_pipeline(DocumentFeaturizer(...), Ridge()), ds, ds.y)`` cross-validates in the same way.

## Featurizing and fitting

``DocumentFeaturizer.fit`` / ``transform`` accept a ``MongoDataset``, a pymongo cursor, a generator, a list of documents
or a DataFrame of records, and return a float32 matrix. When fitting on a ``MongoDataset``, its target field is removed
from the embedded JSON (pass ``exclude=`` for plain documents), so the answer cannot leak into the features. The
featurizer is an ordinary scikit-learn transformer, so it works with ``Pipeline``, ``clone`` and
``get_feature_names_out``:

    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(prep, RidgeClassifierCV())
    pipe.fit(ds_train, ds_train.y)
    pipe.score(ds_test, ds_test.y)
    pipe.predict(coll.find({"address.market": "Porto"}).limit(5))   # predict on any cursor

For cross-validation or model selection, compute ``X = prep.fit_transform(ds)`` once and pass ``X`` and ``ds.y`` to
scikit-learn. Embeddings are cached, so refitting with another estimator does not embed a document again.

## Forecasting with sktime

``MongoSeries`` aggregates a time series from a collection, and ``series.split`` cuts it in time order (sktime's
``temporal_train_test_split(series.series, ...)`` does the same). The parts are ``pd.Series`` with a ``PeriodIndex``
for any sktime forecaster:

    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from pymongopredict import MongoSeries

    series = MongoSeries(coll_weather, time_field="ts", value_field="airTemperature.value",
                         freq="h", agg="mean", valid=lambda v: v < 900)
    y_train, y_test = series.split(test_size=24)
    fc = ExponentialSmoothing(trend="add", seasonal="add", sp=24).fit(y_train)
    fc.predict(fh=range(1, 25))

## Streaming large collections

By default a ``MongoDataset`` fetches its documents once and keeps them in memory. When documents are much larger than
the features taken from them (for example listings that carry all their reviews), pass ``streaming=True``:

    ds = MongoDataset(coll, target="room_type", streaming=True, batch_size=512)

- Loading reads only ``_id`` and the target, and ``field_values`` (for a stratify or group field) fetches only that field.
- ``fit`` / ``transform`` fetch documents in batches of ``_id`` values, with only the fields the featurizer uses when
  ``embed_fields`` is given, and keep only the feature vectors.
- The next batch is fetched in a background thread while the current one is embedded (``prefetch=2``).
- Cursors and generators passed to ``fit`` / ``transform`` are also read batch by batch.

The results are the same as in memory (the tests check this). On the Airbnb demo, peak memory drops from about 300 MB
to about 215 MB, and the difference grows with the size of the documents.

## How documents become features

    MongoDataset(collection, target, filter)  ->  sklearn train_test_split  ->  ds_train / ds_test
                                                                                   |
                         DocumentFeaturizer.fit_transform(ds_train) / .transform(ds_test)
                                                                                   |
        columns (dotted paths): types inferred                 embed_fields (or the whole document):
        (number, bool, date, category, list size),             compact JSON -> embedding model
        missing values filled, scaled, one-hot encoded         (voyage-4-nano, Hugging Face, Voyage API)
                                                                                   |
                                        batch by batch, side by side  ->  X for any scikit-learn estimator

- Embeddings are cached in a sqlite file (``.mongopredict_cache/``), keyed by the model and the text.
- Documents are serialised with ``orjson``. ``max_json_chars`` (8000 by default) is applied field by field, so a
  document with a very long trailing array is not serialised in full only to be cut. ``max_tokens=32768`` or
  ``max_tokens="model"`` measures the budget in tokens with the model's tokenizer instead.
- BSON types are converted: ``Decimal128`` to float, ``ObjectId`` to string, dates to ISO strings.

## Embedding models

``embedding_model=`` (``"embeddingModel"`` in a prediction model definition) accepts:

| value | embedding model |
|---|---|
| ``"voyage-4-nano"`` (default) | runs locally with sentence-transformers, 346M parameters, no API key |
| ``"org/name"`` or a path containing ``/`` | any sentence-transformers model from Hugging Face or disk |
| other ``"voyage-*"`` names: ``"voyage-4-lite"``, ``"voyage-4"``, ``"voyage-4-large"``, ``"voyage-code-4"``, ... | the Voyage API (needs ``VOYAGE_API_KEY``) |
| ``"hashing"`` | a hashed bag of words without dependencies, for tests and quick baselines |
| an embedder object | used as it is (cached under its ``name``; an object without a name is not cached) |

- ``"voyage-4-nano@512"`` (or ``embedding_kwargs={"dimension": 512}``) truncates Matryoshka embeddings after the cache,
  so the full vector is computed once and serves every dimension.
- Local models run in bfloat16 on CUDA (float32 elsewhere), with the token limit capped at the model's maximum.
- Voyage API models (:class:`pymongopredict.embeddings_api.embed.VoyageEmbedder`) take ``output_dimension`` (256, 512,
  1024 or 2048), ``api_key``, ``max_retries`` (default 5, with exponential backoff on rate limits) and ``timeout``
  through ``embedding_kwargs``. Texts go in requests of at most 1,000 texts and within the model's tokens per request.
  Contextualized and multimodal models (``voyage-context-*``, ``voyage-multimodal-*``) are not supported.
- ``mask_target=True`` keeps the target as the first key with the value ``<MASK>`` and puts
  ``Predict the value of <MASK>: `` before the text (``mask_token`` and ``prefix`` change them), so the model knows
  which field is asked for. This is the input format of the voyage-json models, which are trained to predict from
  whole documents.

Examples
--------
Usage:

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from pymongopredict import DocumentFeaturizer, MongoDataset

    ds = MongoDataset(collection, target="room_type")
    train, test = train_test_split(ds, test_size=0.2, stratify=ds.y, random_state=0)
    prep = DocumentFeaturizer(columns=[...], embed_fields=[...])   # docs -> X
    clf = LogisticRegression().fit(prep.fit_transform(train), train.y)
    clf.score(prep.transform(test), test.y)
"""

from .dataset import MongoDataset
from .embed import BaseEmbedder, HashingEmbedder, SentenceTransformerEmbedder, VoyageEmbedder
from .features import DocumentFeaturizer
from .forecast import MongoSeries

# the public API; helpers start with "_" (splitting is scikit-learn's: MongoDataset indexes like an array)
__all__ = [
    "MongoDataset",
    "MongoSeries",
    "DocumentFeaturizer",
    "SentenceTransformerEmbedder",
    "VoyageEmbedder",
    "HashingEmbedder",
    "BaseEmbedder",
]
