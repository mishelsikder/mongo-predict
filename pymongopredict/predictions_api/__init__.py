"""The predictions API: a ``$predict`` aggregation stage backed by prediction models stored in the database.

You define a model once on a collection, as you would a vector search index, then call it by name as the first stage
of an aggregation pipeline; later stages read its output through ``$meta``. No server changes are needed.

| | ``$vectorSearch`` | ``$predict`` |
|---|---|---|
| defined with | ``create_search_index`` | ``create_prediction_model`` |
| managed with | ``list_`` / ``update_`` / ``drop_search_index`` | ``list_`` / ``update_`` / ``drop_prediction_model`` |
| stage options | ``index``, ``path``, ``queryVector``, ``filter``, ``limit`` | ``model``, ``filter``, ``limit``, ``batchSize`` |
| position | first stage only | first stage only |
| output | ``{"$meta": "vectorSearchScore"}`` | ``{"$meta": "prediction"}``, ``"predictScore"``, ``"predictProbabilities"`` |
| runs | inside Atlas | in your Python program, then on the server |

What you use:

* :class:`PredictionModel` and :func:`create_prediction_model`: define and train a model; the
  model is stored in the ``__mongopredict_models`` collection of the same database.
* :func:`list_prediction_models`, :func:`update_prediction_model`,
  :func:`drop_prediction_model`: status, metrics, retraining, deletion.
* :func:`enable` / :func:`disable`: let ``coll.aggregate`` run ``$predict`` and make the model
  functions collection methods.
* :func:`aggregate` and :class:`PredictCursor`: run a pipeline that starts with ``$predict``,
  and its result (with timing ``stats``).

Without :func:`enable`, call ``aggregate(coll, pipeline)`` and the module-level model
functions with the collection as their first argument.

## Model definition

The definition given to :class:`PredictionModel` takes:

- ``target``: the dotted path of the field to predict (required).
- ``columns``, ``embedFields``, ``excludeFields``, ``embeddingModel``, ``embeddingKwargs``, ``maskTarget``,
  ``maxTokens``, ``maxJsonChars``: the settings of the
  :class:`pymongopredict.embeddings_api.DocumentFeaturizer`; ``featurizerOptions`` passes any other of its
  arguments by its Python name. Without ``embedFields`` the whole document is embedded.
- ``embeddingModel``: ``"voyage-4-nano"`` (the default, runs locally), any Hugging Face model id (runs locally), or a
  Voyage API model such as ``"voyage-4-lite"``, ``"voyage-4"`` or ``"voyage-4-large"`` (needs ``VOYAGE_API_KEY``); ``embeddingKwargs`` passes its settings, e.g.
  ``{"output_dimension": 512}`` for an API model (but not ``api_key``, which would be stored with the model: set
  ``VOYAGE_API_KEY`` instead). See the embeddings API documentation for every option.
- ``task``: ``"auto"``, ``"classification"`` or ``"regression"``. ``"auto"`` predicts a number when every value of
  the target is a number (numbers stored as text count; empty strings and text such as ``"NA"``, ``"none"`` or ``"-"``
  are missing values, while in a field of categories they are labels) and there are at least 8 distinct values, and
  a category otherwise, so a 0/1 flag or a 1 to 5 star rating stored as numbers is a category.
- ``estimator``: ``"auto"`` (the default: XGBoost with its default settings, for categories and numbers, on at most
  16 threads unless ``OMP_NUM_THREADS`` is set), ``"xgboost"`` (the same), ``"logistic"``, ``"ridge"`` or ``"hgb"`` (histogram gradient boosting), or any scikit-learn estimator
  passed to :func:`create_prediction_model`.
- ``filter``: which documents to learn from (documents without the target are always left out);
  ``numTrainingDocuments``: a cap on their number.
- ``testSize``: the fraction, or the number, of training documents held out to measure the stored metrics (default
  0.2; 0 skips them). The stored model is then refitted on all training documents.

``{"$meta": "predictScore"}`` is the probability of the predicted category and ``{"$meta": "predictProbabilities"}``
maps each category to its probability; both exist only when the estimator has ``predict_proba`` (not for numbers or
``"ridge"``).

## How a pipeline runs

1. :func:`create_prediction_model` fits the featurizer and the estimator on the documents that have the target,
   records the held-out metrics, refits on all of them and stores the fitted model (a pickle of at most 15 MB) with
   its status and version in the ``__mongopredict_models`` collection of the same database. As with a search index,
   the previous version keeps answering while a new one trains.
2. :func:`aggregate` fetches the documents selected by the stage's ``filter`` and ``limit`` (only the fields the model
   reads), predicts them in batches in your program, and uploads only ``{_id, prediction, score, probabilities}``
   rows to a temporary collection.
3. The rest of the pipeline runs on the server: the temporary collection is joined back to the full documents
   (``$lookup`` on ``_id``), the ``$meta`` references are rewritten to the joined fields, and the helper field is
   removed before the output (or before a final ``$merge`` / ``$out``). The temporary collection is then dropped.

On the 14 MongoDB sample tasks, ``$predict`` returns the same predictions as calling the model directly, at a median
1.39 times the time of that code (``reports/01_predictions_api.html``).

## Details and limits

- Requirements: MongoDB 5.0 or later (5.1 or later when the source collection is sharded); fitting a model or running
  ``$predict`` on an older server raises ``RuntimeError``. The tests run on MongoDB 5.0, 6.0, 7.0 and 8.0, and on Atlas (8.0). On 5.0 a
  time-series collection is a view, so ``collation=`` cannot be passed for it there (6.0 and later accept it). ``$predict`` needs
  ``find`` on the source collection and ``createCollection``, ``insert`` and ``dropCollection`` in its database (for
  the temporary ``__mongopredict_tmp_*`` collection); training needs ``find``, ``insert`` and ``update`` on
  ``__mongopredict_models`` (``remove`` to drop a model). Model records and temporary collections are read and written on the primary, even with
  a secondary read preference.
- A killed process can leave a temporary collection behind: :func:`cleanup_temporary_collections` drops those older
  than 24 hours (``older_than``). A call whose temporary collection is dropped while it runs raises
  ``RuntimeError`` instead of returning partial results. A record still ``BUILDING`` after 24 hours is flagged ``stale: True`` by
  :func:`list_prediction_models`.
- Documents are predicted in batches of at most ``batchSize`` documents and 64 MB of BSON. Scores and probabilities
  are computed only when the pipeline reads ``predictScore`` or ``predictProbabilities``. An embedding cache directory
  that cannot be used gives a warning, and the embeddings are then computed without a cache.
- The source can be a collection, a view or a time-series collection; views and time-series collections are joined
  from the source side, through the temporary collection's ``_id`` index. ``_id`` values must be unique (a view with an
  ``$unwind`` stage is refused).
- A class is returned as first stored in the training documents (an ``ObjectId``, a date, an ``int``), so predictions
  can be compared with the labels or joined to them with ``$lookup``. The ``predictProbabilities`` keys and the
  record's ``classes`` are ``str()`` of those values, so ``str(prediction)`` is always a key.
- Model names may not contain ``.`` or ``$``. ``limit`` and ``batchSize`` must be positive integers; without
  ``limit`` every matching document is predicted.
- ``hint=``, ``collation=``, ``let=`` and ``session=`` passed to the aggregate apply to the query on the source
  collection, and the temporary collection gets the source collection's default collation, so string matching and
  sorting behave the same.
- ``$meta`` works in ``$sort``, ``sortBy`` (``$setWindowFields``, ``$top``, ``$bottom``), expressions, ``$group``,
  ``$facet``, ``$redact``, and after an inclusion ``$project`` or a ``$replaceRoot``. It is not available inside
  ``$lookup`` / ``$unionWith`` sub-pipelines (add it as a field first and pass it with ``let``), nor in ``$merge``'s
  ``let`` or ``whenMatched``. ``__mongopredict`` is reserved as a field name.
- ``$predict`` cannot run inside a multi-document transaction, because it creates and drops a temporary collection.
- The documents travel to your program and the model runs there; the result is read in full before it is returned;
  ``$predict`` works from Python only (not from ``mongosh`` or other drivers).
- The stored model is a pickle, so only users trusted to run code in your program should be able to write
  ``__mongopredict_models``. Set ``MONGOPREDICT_MODEL_KEY`` (the same secret where models are trained and where they
  are used) to sign stored models: a model without a valid signature is then never unpickled.
- The record holds the library versions the model was stored with (``versions``). Loading it under other
  scikit-learn, numpy or pymongopredict versions gives a warning; a model that cannot be loaded raises a
  ``ValueError`` that says to rebuild it with :func:`update_prediction_model`.
- ``$predict`` caches embeddings only when ``MONGOPREDICT_PREDICT_CACHE_DIR`` names a directory (the training
  ``cache_dir`` is not used), so a long-running process does not grow a cache of every document it predicts.
- Training on a collection streams the documents: only ``_id`` and the target of each are held in memory (if
  documents are deleted meanwhile, it trains again from a single read). Views and time-series collections are read
  once into memory, and training on them also needs unique ``_id`` values.

Examples
--------
Usage:

    from pymongopredict import PredictionModel, predictions_api

    predictions_api.enable()                       # coll.aggregate() understands $predict
    coll.create_prediction_model(PredictionModel({"target": "room_type"}, name="room_type"))
    coll.list_prediction_models()                  # status, definition, holdout metrics
    coll.aggregate([{"$predict": {"model": "room_type", "limit": 1000}},
                    {"$project": {"p": {"$meta": "prediction"}}}])
"""

from .models import (
    MODELS_COLLECTION,
    PredictionModel,
    _load_prediction_model,  # noqa: F401 (internal: the fitted featurizer and estimator, for tests)
    create_prediction_model,
    drop_prediction_model,
    list_prediction_models,
    update_prediction_model,
)
from .stage import (
    META_FIELD,
    STAGE,
    TMP_PREFIX,
    PredictCursor,
    aggregate,
    cleanup_temporary_collections,
    disable,
    enable,
)

__all__ = [
    "PredictionModel",
    "create_prediction_model",
    "update_prediction_model",
    "list_prediction_models",
    "drop_prediction_model",
    "aggregate",
    "enable",
    "disable",
    "PredictCursor",
    "cleanup_temporary_collections",
    "STAGE",
    "MODELS_COLLECTION",
    "META_FIELD",
    "TMP_PREFIX",
]
